import os
import random
from pathlib import Path

import albumentations as alb
import cv2
import imageio
import numpy as np
import numpy.typing as npt
import sklearn.model_selection as ms
import torch
from albumentations import Compose
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Sequence

Image = Mask = npt.NDArray[np.generic]
Loader = DataLoader[dict[str, torch.Tensor]]
PathPair = tuple[list[Path], list[Path]]


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegDatasetPaths(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        img_paths: Sequence[Path],
        label_paths: Sequence[Path],
        transform: Compose,
        x_channels: int = 1,
    ) -> None:
        assert len(img_paths) == len(label_paths), f"{len(img_paths)=}!={len(label_paths)=}"

        self.img_paths = [Path(p) for p in img_paths]
        self.label_paths = [Path(p) for p in label_paths]
        self.transform = transform
        self.x_channels = x_channels

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_image(self, path: Path) -> Image:
        return np.float32(read_image_normalized(path))

    def _load_label(self, path: Path) -> Mask:
        return np.float32(read_image_normalized(path))

    def _transform(self, image: Image, label: Mask) -> tuple[Image, Mask]:
        out = self.transform(image=image, mask=label)
        return out["image"], out["mask"]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = self._load_image(self.img_paths[idx])
        label = self._load_label(self.label_paths[idx])
        image_aug, label_aug = self._transform(image, label)

        if self.x_channels not in (1, 3):
            raise ValueError(f"{self.x_channels=} is not supported; use 1 or 3.")

        x_np = np.stack([image_aug] * self.x_channels, axis=0).astype(np.float32)

        if self.x_channels == 3:
            x_np = (x_np - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

        x = torch.from_numpy(x_np).float()

        y_fg = (label_aug > 0).astype(np.float32)
        y = torch.from_numpy(y_fg[None, ...]).float()

        return {"x": x, "y": y}


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_dataloaders_from_paths(
    img_paths: Sequence[Path],
    label_paths: Sequence[Path],
    train_augmentation: Compose,
    val_augmentation: Compose,
    *,
    batch_size: int = 32,
    val_size: float = 0.33,
    seed: int = 123,
    x_channels: int = 3,
) -> tuple[Loader, Loader]:
    img_train, img_val, label_train, label_val = ms.train_test_split(
        list(img_paths), list(label_paths), test_size=val_size, random_state=seed
    )

    dataset_train = SegDatasetPaths(img_train, label_train, train_augmentation, x_channels=x_channels)
    dataset_val = SegDatasetPaths(img_val, label_val, val_augmentation, x_channels=x_channels)

    g_train = torch.Generator()
    g_train.manual_seed(seed)

    g_val = torch.Generator()
    g_val.manual_seed(seed)

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g_train,
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g_val,
    )
    return train_dataloader, val_dataloader


def build_dataloaders(
    img_paths: Sequence[Path],
    mask_paths: Sequence[Path],
    *,
    patch_size: int,
    x_channels: int,
    batch_size: int,
    val_size: float,
    split_seed: int = 123,
) -> tuple[Loader, Loader]:
    train_aug = setup_augmentation(
        patch_size=patch_size,
        blur_sharp_power=0.8,
        noise_value=0.01,
        rotate_deg=20,
    )
    val_aug = setup_augmentation(patch_size=patch_size)

    return prepare_dataloaders_from_paths(
        img_paths,
        mask_paths,
        train_augmentation=train_aug,
        val_augmentation=val_aug,
        batch_size=batch_size,
        val_size=val_size,
        x_channels=x_channels,
        seed=split_seed,
    )


def setup_augmentation(
    patch_size: int,
    *,
    advanced: bool = False,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    blur_sharp_power: float | None = None,
    noise_value: float | None = None,
    rotate_deg: int | None = None,
    interpolation: int = cv2.INTER_CUBIC,
) -> Compose:
    patch_size_padded = int(patch_size * 1.5)
    transform_list = [
        alb.PadIfNeeded(patch_size_padded, patch_size_padded),
        alb.RandomCrop(patch_size_padded, patch_size_padded),
    ]

    if advanced:
        transform_list += [
            alb.Perspective(p=0.2),
            alb.GridDistortion(p=0.2, num_steps=17),
            alb.RandomGridShuffle(p=0.2, grid=(13, 13)),
        ]
    if rotate_deg is not None:
        transform_list += [
            alb.Rotate(limit=rotate_deg, interpolation=interpolation),
        ]

    if noise_value is not None:
        transform_list += [
            alb.augmentations.transforms.GaussNoise(noise_value, p=0.5),
        ]

    if blur_sharp_power is not None:
        transform_list += [
            alb.OneOf(
                [
                    alb.Sharpen(p=1, alpha=(0.2, 0.2 * blur_sharp_power)),
                    alb.AdvancedBlur(p=1),
                ],
                p=0.3,
            ),
        ]

    if flip_horizontal:
        transform_list += [alb.HorizontalFlip(p=0.5)]
    if flip_vertical:
        transform_list += [alb.VerticalFlip(p=0.5)]

    transform_list += [alb.CenterCrop(patch_size, patch_size)]
    return alb.Compose(transform_list)


def normalize_image(image: Image) -> Image:
    image_f = image.astype(np.float32, copy=False)
    max_value = float(image_f.max())
    min_value = float(image_f.min())
    if max_value == min_value:
        tqdm.write("Image is constant.")
        return image_f
    return (image_f - min_value) / (max_value - min_value)


def read_image_normalized(image_path: os.PathLike[str] | str, dtype: npt.DTypeLike = np.float32) -> Image:
    image = imageio.v2.imread(image_path)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (1, 2):
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[:, :, :3]
        gray = rgb @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    image_norm = normalize_image(gray)
    return image_norm.astype(dtype)


def load_paths(root: Path, n: int | None = None) -> PathPair:
    img_paths = sorted(root.rglob("*/img.png"))
    if n is not None:
        img_paths = img_paths[:n]
    label_paths = [p.parent / "mask.png" for p in img_paths]
    return img_paths, label_paths


def load_clustered(synth_root: Path, base_counts: Sequence[int], multiplier: int, prefix: str) -> PathPair:
    img_paths: list[Path] = []
    for i, ratio in enumerate(base_counts):
        cluster = sorted(synth_root.rglob(f"*{prefix}{i}/img.png"))
        k = ratio * multiplier
        if len(cluster) < k:
            raise ValueError(f"Cluster {i}: requested {k}, found {len(cluster)}")
        img_paths += cluster[:k]
    return img_paths, [p.parent / "mask.png" for p in img_paths]


def show_augmented_triple(n_show: int, train_dl: Loader) -> None:
    shown = 0
    for batch in train_dl:
        x, y = batch["x"], batch["y"]

        for i in range(x.shape[0]):
            img = x[i].permute(1, 2, 0).cpu().numpy()
            mask = y[i, 0].cpu().numpy()
            denom = img.max() - img.min()
            img_vis = (img - img.min()) / denom if denom > 0 else np.zeros_like(img)

            plt.figure(figsize=(8, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(img_vis, cmap="gray", vmin=0, vmax=1)
            plt.title("Augmented Image")

            plt.subplot(1, 3, 2)
            plt.imshow(img, cmap="gray")
            plt.title("Normalised Augmented Image")

            plt.subplot(1, 3, 3)
            plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
            plt.title("Mask")

            plt.tight_layout()
            plt.show()

            shown += 1
            if shown >= n_show:
                break

        if shown >= n_show:
            break

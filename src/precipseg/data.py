# Adapted from semseg/dataset.py and semseg/io.py in jaroslavknotek/semantic-segmentation.
# Source: https://github.com/jaroslavknotek/semantic-segmentation/blob/main/src/semseg
# Changes: path-based loading instead of in-memory arrays, returns torch tensors, ImageNet normalization,
# simplified image loading

import os
import imageio
import numpy as np
import albumentations as alb
import cv2
from typing import List, Dict
from albumentations import Compose
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
import sklearn.model_selection as ms
import numpy.typing as npt
from tqdm import tqdm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegDatasetPaths(Dataset):
    def __init__(
            self,
            img_paths: List[Path],
            label_paths: List[Path],
            transform: Compose,
            x_channels: int = 1,
    ):
        assert len(img_paths) == len(label_paths), f"{len(img_paths)=}!={len(label_paths)=}"

        self.img_paths = [Path(p) for p in img_paths]
        self.label_paths = [Path(p) for p in label_paths]

        self.transform = transform
        self.x_channels = x_channels

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_image(self, path: Path) -> npt.NDArray:
        img = read_image_normalized(path)
        return np.float32(img)

    def _load_label(self, path: Path) -> npt.NDArray:
        lab = read_image_normalized(path)
        return np.float32(lab)

    def _transform(self, image: NDArray[np.float32], label: NDArray[np.float32]) -> tuple[
        NDArray[np.float32], NDArray[np.float32]]:
        out = self.transform(image=image, mask=label)
        return out["image"], out["mask"]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        image = self._load_image(self.img_paths[idx])
        label = self._load_label(self.label_paths[idx])

        image_aug, label_aug = self._transform(image, label)

        x_np = np.stack([image_aug] * self.x_channels, axis=0).astype(np.float32)
        x_np = (x_np - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        x = torch.from_numpy(x_np).float()

        y_fg = (label_aug > 0).astype(np.float32)
        y = torch.from_numpy(y_fg[None, ...]).float()

        return {"x": x, "y": y}


def prepare_dataloaders_from_paths(
    img_paths: List[Path],
    label_paths: List[Path],
    train_augmentation: Compose,
    val_augmentation: Compose,
    *,
    batch_size: int = 32,
    val_size: float = 0.33,
    seed: int = 123,
    x_channels: int = 3
) -> tuple[DataLoader, DataLoader]:
    img_train, img_val, label_train, label_val = ms.train_test_split(
        img_paths, label_paths, test_size=val_size, random_state=seed
    )

    dataset_train = SegDatasetPaths(
        img_train,
        label_train,
        train_augmentation,
        x_channels=x_channels,
    )
    dataset_val = SegDatasetPaths(
        img_val,
        label_val,
        val_augmentation,
        x_channels=x_channels,
    )

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader


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
            alb.Perspective(p=.2),
            alb.GridDistortion(p=.2, num_steps=17),
            alb.RandomGridShuffle(p=.2, grid=(13, 13)),
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
        transform_list += [
            alb.HorizontalFlip(p=0.5),
        ]
    if flip_vertical:
        transform_list += [
            alb.VerticalFlip(p=0.5),
        ]

    transform_list += [alb.CenterCrop(patch_size, patch_size)]

    return alb.Compose(transform_list)


def normalize_image(image: npt.NDArray) -> NDArray[np.float32]:
    max_value = image.max()
    min_value = image.min()
    if max_value == min_value:
        tqdm.write("Image is constant.")
        return image
    return (image - min_value) / (max_value - min_value)


def read_image_normalized(
        image_path: os.PathLike,
        dtype=np.float32,
) -> NDArray[np.float32]:
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

    image = normalize_image(gray)
    return image.astype(dtype)


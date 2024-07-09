import os
import pathlib
import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from denoiseg import image_utils as iu
from denoiseg import dataset as ds


DATA_DIR = "C:\\Users\\vitko\\Desktop\\CVR\\Data"


def load_images(root_path):
    imgs = []
    masks = []
    imgs_paths = list(root_path.rglob('img.png'))
    mask_paths = [img_path.parent / "mask.png" for img_path in imgs_paths]

    for img_path in imgs_paths:
        if os.path.isfile(img_path):
            # img = np.squeeze(imageio.imread(img_path))
            # print(img_path)
            img = Image.open(img_path).convert("L")
            img = np.array(img, dtype=np.float32)
            # img = transforms.ToTensor()(img)
            # img = img.squeeze(0)
            # img = img.to(torch.float32)
            imgs.append(img)

    for mask_path in mask_paths:
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask, dtype=np.float32)
            masks.append(mask)

    # imgs = np.array(imgs)
    # masks = np.array(masks)
    return imgs, masks


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        assert len(images) == len(labels), f"{len(images)=}!={len(labels)=}"
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # print(image.shape, label.shape)

        image_aug, label_aug = self._transform(image, label)
        # print(image_aug.shape(), label_aug.shape())

        # return image_aug, iu.label_to_classes(label_aug)
        return image_aug, label_aug

    def _transform(self, image, label):
        label = np.stack([label, label])
        transformed = self.transform(image=image, masks=label)
        return transformed["image"], transformed["masks"]


def prepare_dataset():
    data_root = pathlib.Path(DATA_DIR + "\\labeled")
    images, masks = load_images(data_root)

    augmentation_fn = ds.setup_augumentation(
        patch_size=256,
        elastic=True,
        brightness_contrast=True,
        flip_vertical=False,  # tohle bych si rozmyslel vzhledem k tomu, kde je umisten senzor mikroskopu
        flip_horizontal=False,  # tohle taky
        blur_sharp_power=1,
        noise_value=.01,
        rotate_deg=90,
        # interpolation=cv2.INTER_CUBIC,
    )

    dataset = ImageDataset(
        images,
        masks,
        augmentation_fn
    )

    # for data_dict in dataset:
    #     _, (axl, axr) = plt.subplots(1, 2)
    #
    #     # images in data_dict are in channel-first order
    #     axl.imshow(data_dict['x'][0])
    #     axr.imshow(data_dict['y'][0])
    #     plt.show()

    return dataset


def prepare_data():
    # dataset = ImageDataset(DATA_DIR+"\\labeled")
    dataset = prepare_dataset()
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_dataloader = DataLoader(train_dataset, shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False)

    return train_dataloader, val_dataloader

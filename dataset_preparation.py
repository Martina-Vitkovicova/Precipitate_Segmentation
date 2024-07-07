import os
import pathlib
import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from denoiseg import dataset as ds
from baseline_model import DATA_DIR


def load_images(path, img_or_mask):
    imgs = []
    for subdir in os.listdir(path):
        if subdir.startswith('DELISA'):
            img_path = os.path.join(path, subdir, img_or_mask)
            if os.path.isfile(img_path):
                # img = np.squeeze(imageio.imread(img_path))
                # print(img_path)
                img = Image.open(img_path).convert("L")
                # img = np.array(img, dtype=np.float32)
                img = transforms.ToTensor()(img)
                # img = img.squeeze(0)
                img = img.to(torch.float32)
                imgs.append(img)
    imgs = np.array(imgs)
    return imgs


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
        transformed = self.transform(image=image, masks=label)
        return transformed["image"], transformed["masks"]


def prepare_dataset():
    data_root = pathlib.Path(DATA_DIR + "\\labeled")

    # imgs_paths = list(data_root.rglob('img.png'))
    # images = [iu.load_image(img_path) for img_path in imgs_paths]
    #
    # # labels-paths come from imgs_paths img_paths to ensure order
    # mask_paths = [img_path.parent / "mask.png" for img_path in imgs_paths]
    # masks = [iu.load_image(img_path) for img_path in mask_paths]

    images = load_images(DATA_DIR + "\\labeled", 'img.png')
    masks = load_images(DATA_DIR + "\\labeled", 'mask.png')

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

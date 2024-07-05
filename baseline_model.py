import glob
import os
import pathlib

import cv2
import imageio
import numpy as np
import segmentation_models_pytorch
from sklearn.model_selection import train_test_split
# import segmentation_models as sm
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import torch
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from denoiseg import instance_analysis as ia
from denoiseg import evaluation as ev
import denoiseg.visualization as vis
from denoiseg import image_utils as iu

DATA_DIR = "C:\\Users\\vitko\\Desktop\\CVR\\Data"
BACKBONE = 'resnet34'


# preprocess_input = sm.get_preprocessing(BACKBONE)


def load_images(path, img_or_mask):
    imgs = []
    for subdir in os.listdir(path):
        if subdir.startswith('DELISA'):
            img_path = os.path.join(path, subdir, img_or_mask)
            if os.path.isfile(img_path):
                img = np.squeeze(imageio.imread(img_path))
                # print(img_path)
                # img = Image.open(img_path).convert("L")
                # img = transforms.ToTensor()(img)
                # img = img.to(torch.float32)
                # img = np.squeeze(np.float32(img))
                imgs.append(img)
    # Convert list to array for machine learning processing
    # imgs = np.array(imgs)
    return imgs


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = load_images(img_dir, "img.png")
        self.masks = load_images(img_dir, "mask.png")

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


class LitUnet(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )
        self.learning_rate = learning_rate
        self.loss_function = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_function(outputs, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_function(outputs, masks)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def fit_model(self, train_dataloader, val_dataloader):
        trainer = pl.Trainer(
            max_epochs=5,
        )

        trainer.fit(
            self,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )


def prepare_data():
    dataset = ImageDataset(DATA_DIR+"\\labeled")
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    train_dataloader = DataLoader(train_dataset, shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False)

    return train_dataloader, val_dataloader

# gt = ground truth
# TODO: patches


def evaluate_model():
    root = pathlib.Path('../Data/test/')
    gt_paths = [os.path.join(root, subdir, "mask.png") for subdir in os.listdir(root)]
    img_paths = [os.path.join(root, subdir, "img.png") for subdir in os.listdir(root)]
    # img_paths = [root / f"{p.stem}.png" for p in gt_paths]

    imgs_test = list(map(iu.load_image, img_paths))
    mask_test = list(map(iu.load_mask, gt_paths))

    checkpoint = torch.load('segmentation_model.pth')
    model_s = LitUnet()
    model_s.load_state_dict(checkpoint)
    model_s.eval()

    predictions, metrics = ev.evaluate_images(model_s, imgs_test, mask_test, device='cpu')

    for i, (img, gt, pred, met) in enumerate(zip(imgs_test, mask_test, predictions, metrics)):

        segms = []
        for im in pred[1:]:
            im = im.copy()
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
            im[im < .5] = 0
            im[im >= .5] = 1
            segms.append(im)

        show_imgs = [img, gt, *segms, *pred]
        vis.plot_row(show_imgs, vmin_vmax=(0, 1), figsize=(50, 10))
        plt.suptitle(f'Metric: {met}', y=0.72)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # model = LitUnet()
    # train, val = prepare_data()
    # model.fit_model(train, val)
    # torch.save(model.state_dict(), 'segmentation_model.pth')
    evaluate_model()

# images = load_images('DELISA', 'img.png')
# masks = load_images('DELISA', 'mask.png')
# x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# model = smp.Unet(
#     encoder_name="resnet34",
#     encoder_weights="imagenet",
#     in_channels=1,
#     classes=1
# )

# preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")
# loss = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)

# model = sm.Unet(BACKBONE, input_shape=(1024, 1024, 1), activation='sigmoid', encoder_weights=None)
# model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
#
# # model.summary()
#
# model.fit(
#    x=x_train,
#    y=y_train,
#    batch_size=16,
#    epochs=10,
#    validation_data=(x_val, y_val)
# )

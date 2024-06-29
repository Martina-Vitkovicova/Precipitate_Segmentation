import glob
import os
import cv2
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

DATA_DIR = "C:\\Users\\vitko\\Desktop\\CVR\\Data\\labeled"
BACKBONE = 'resnet34'
# preprocess_input = sm.get_preprocessing(BACKBONE)


def load_images(path, img_or_mask):
    imgs = []
    for subdir in os.listdir(path):
        if subdir.startswith('DELISA'):
            img_path = os.path.join(path, subdir, img_or_mask)
            if os.path.isfile(img_path):
                # print(img_path)
                img = Image.open(img_path).convert("L")
                img = transforms.ToTensor()(img)
                img = img.to(torch.float32)
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


# images = load_images('DELISA', 'img.png')
# masks = load_images('DELISA', 'mask.png')
# x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

dataset = ImageDataset(DATA_DIR)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_dataloader = DataLoader(train_dataset, shuffle=True)
val_dataloader = DataLoader(val_dataset, shuffle=False)

# model = smp.Unet(
#     encoder_name="resnet34",
#     encoder_weights="imagenet",
#     in_channels=1,
#     classes=1
# )

# preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")
# loss = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

model = LitUnet()

trainer = pl.Trainer(
    max_epochs=5,
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

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

torch.save(model.state_dict(), 'segmentation_model.pth')

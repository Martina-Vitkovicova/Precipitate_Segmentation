import os
import pathlib

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint

import dataset_preparation
from dataset_preparation import prepare_dataloaders
from denoiseg import evaluation as ev
import denoiseg.visualization as vis

DATA_DIR = "C:\\Users\\vitko\\Desktop\\CVR\\Data\\labeled"
TEST_DIR = "C:\\Users\\vitko\\Desktop\\CVR\\Data\\test"


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
        """unsqueeze treba pri trenovani, ale pri evaluacii to uz nefunguje"""
        x = x.unsqueeze(1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch

        # _, (axl, axr) = plt.subplots(1, 2)
        # axl.imshow(masks[0].squeeze(0))
        # axr.imshow(masks[1].squeeze(0))
        # plt.show()

        outputs = self(images)
        """z nejakeho dovodu masks je list s dvoma rovnakymi obrazkami"""
        loss = self.loss_function(outputs, masks[0])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        # _, (axl, axr) = plt.subplots(1, 2)
        # axl.imshow(masks[0].squeeze(0))
        # axr.imshow(masks[1].squeeze(0))
        # plt.show()

        outputs = self(images)
        """z nejakeho dovodu masks je list s dvoma rovnakymi obrazkami"""
        loss = self.loss_function(outputs, masks[0])
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def fit_model(self, train_dataloader, val_dataloader):
        checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', filename='{epoch}-{val_loss:.2f}', save_top_k=-1,
                                              save_weights_only=True, verbose=True)
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=10,
        )

        trainer.fit(
            self,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def evaluate_model(self):
        root_path = pathlib.Path(TEST_DIR)
        imgs_test, mask_test = dataset_preparation.load_images(root_path)

        predictions, metrics = ev.evaluate_images(self, imgs_test, mask_test, device='cpu')

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
    train, val = prepare_dataloaders(DATA_DIR)
    model = LitUnet()
    model.fit_model(train, val)

    model = LitUnet.load_from_checkpoint('checkpoints/epoch=9-val_loss=-0.36.ckpt')
    model.eval()
    model.evaluate_model()

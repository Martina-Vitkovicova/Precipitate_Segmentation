import os
import pathlib

import cv2
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms

import dataset_preparation
from dataset_preparation import prepare_dataloaders
from denoiseg import evaluation as ev
import denoiseg.visualization as vis
from denoiseg import instance_analysis as ia

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

    def fit_model(self, train_dataloader, val_dataloader, epoch_num):
        checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', filename='{epoch}-{val_loss:.2f}', save_top_k=-1,
                                              save_weights_only=True, verbose=True, every_n_train_steps=1)
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=epoch_num,
        )

        trainer.fit(
            self,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def get_predictions(self, test_dir, file_count):
        predicted_names = ['predictions\\saved_test_image_{}.png'.format(i) for i in range(file_count)]
        for name in predicted_names:
            if not os.path.exists(name):
                imgs_test, mask_test = dataset_preparation.load_images(test_dir)
                predictions, metrics = ev.evaluate_images(self, imgs_test, mask_test, device='cpu')

                for i, pred in enumerate(predictions):
                    image = Image.fromarray((pred[0] * 255).astype(np.uint8))
                    image.save('predictions\\saved_test_image_{}.png'.format(i))
                break

        return [np.array(Image.open('predictions\\saved_test_image_{}.png'.format(i))) for i in range(file_count)]


def evaluate_predictions(test_dir, predictions, show=True):
    imgs_test, mask_test = dataset_preparation.load_images(test_dir)

    if show:
        for i, (img, mask, pred) in enumerate(zip(imgs_test, mask_test, predictions)):
            segms = []
            for im in pred[1:]:
                im = im.copy()
                im = (im - np.min(im)) / (np.max(im) - np.min(im))
                im[im < .5] = 0
                im[im >= .5] = 1
                segms.append(im)

            vis.plot_row([img, mask, pred, segms], vmin_vmax=(0, 1), figsize=(50, 10))
            plt.show()

    evaluations = ia.evaluate_match(imgs_test, mask_test, predictions)

    for evaluation in evaluations.values():
        print(evaluation['samples'])


if __name__ == '__main__':
    # train, val = prepare_dataloaders(DATA_DIR)
    # model = LitUnet()
    epoch_num = 10
    # model.fit_model(train, val, epoch_num)

    checkpoint = list(pathlib.Path(os.getcwd() + "\\checkpoints").glob('epoch={}*'.format(epoch_num - 1)))[0]
    model = LitUnet()
    model.eval()
    preds = model.get_predictions(pathlib.Path(TEST_DIR), 6)
    evaluate_predictions(pathlib.Path(TEST_DIR), preds)

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from dataset_preparation import prepare_data

DATA_DIR = "C:\\Users\\vitko\\Desktop\\CVR\\Data"


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
        # x = x.permute(3, 0, 1, 2)
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


if __name__ == '__main__':
    # imgs = load_images(DATA_DIR+"\\labeled", 'img.png')
    model = LitUnet()
    train, val = prepare_data()
    model.fit_model(train, val)
    torch.save(model.state_dict(), 'segmentation_model.pth')
    # evaluate_model()

# images = load_images('DELISA', 'img.png')
# masks = load_images('DELISA', 'mask.png')
# x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)


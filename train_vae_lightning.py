import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from preprocess_mnist import RotatedMNIST
from vae import VAE

class VAETrainingModule(LightningModule):
    def __init__(self, in_channels, latent_dim, learning_rate=0.005):
        super().__init__()
        self.model = VAE(in_channels=in_channels, latent_dim=latent_dim)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, _ = batch
        recon_batch, x, mu, log_var = self.model(image)
        loss_dict = self.model.loss_function(recon_batch, x, mu, log_var)
        self.log("train_loss", loss_dict['loss'], prog_bar=True)
        self.log("train_recon_loss", loss_dict['Reconstruction_Loss'], prog_bar=True)
        self.log("train_kl_loss", loss_dict['KLD'], prog_bar=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        image, _ = batch
        recon_batch, x, mu, log_var = self.model(image)
        loss_dict = self.model.loss_function(recon_batch, x, mu, log_var)
        self.log("val_loss", loss_dict['loss'], prog_bar=True)
        return loss_dict['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]

# Load datasets
train_dataset = RotatedMNIST(file_path='mnist_rotated_train.h5')
val_dataset = RotatedMNIST(file_path='mnist_rotated_val.h5')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor="val_loss",  # Metric to monitor
    patience=3,          # Number of epochs to wait for improvement
    mode="min"           # Stop when the monitored metric stops decreasing
)

# Initialize the Lightning Trainer
trainer = Trainer(
    max_epochs=10,
    callbacks=[early_stopping],
    accelerator="gpu" if torch.cuda.is_available() else "cpu"
)

# Train the model
vae_module = VAETrainingModule(in_channels=1, latent_dim=20)
trainer.fit(vae_module, train_loader, val_loader)
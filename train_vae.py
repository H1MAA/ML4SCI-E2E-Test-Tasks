import torch
from preprocess_mnist import RotatedMNIST
from vae import VAE
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

train_dataset = RotatedMNIST(file_path='mnist_rotated_train.h5')
val_dataset = RotatedMNIST(file_path='mnist_rotated_val.h5')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(in_channels=1, latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

num_epochs = 10
min_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        image, labels = batch
        image = image.to(device)
        optimizer.zero_grad()
        recon_batch, x, mu, log_var = model(image)
        loss_dict = model.loss_function(recon_batch, x, mu, log_var)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        recon_loss_total += loss_dict['Reconstruction_Loss'].item()
        kl_loss_total += loss_dict['KLD'].item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = recon_loss_total / len(train_loader)
    avg_kl_loss = kl_loss_total / len(train_loader)

    print(f"Train Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    for batch in tqdm(val_loader, desc="Validation", unit="batch"):
        image, labels = batch
        image = image.to(device)
        with torch.no_grad():
            recon_batch, x, mu, log_var = model(image)
            
            loss_dict = model.loss_function(recon_batch, x, mu, log_var)
            loss = loss_dict['loss']
            total_val_loss += loss.item()

    print(f"Validation Loss: {total_val_loss / len(val_loader):.4f}")

    if total_val_loss < min_loss:
        min_loss = total_val_loss
        torch.save(model.state_dict(), 'best_vae_model.pth')
        print("Model saved. Epoch:", epoch + 1)

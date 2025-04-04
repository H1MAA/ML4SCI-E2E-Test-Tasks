import torch
from torch import nn
from torch.nn import functional as F
import cv2

class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_var = nn.Linear(64 * 7 * 7, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.LeakyReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )   
        
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return [recon, x, mu, log_var]
    
    def loss_function(self, recon, x, mu, log_var):
        reconstruction_loss = F.mse_loss(recon, x)
        
        kl_divergence_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kl_weight = 0.0025
        
        loss = reconstruction_loss + kl_weight * kl_divergence_loss

        return {'loss': loss, 'Reconstruction_Loss':reconstruction_loss.detach(), 'KLD':kl_divergence_loss.detach()}
    
    def generate(self, x):
        return self.forward(x)[0]
    
    def sample(self, batch_size, current_device):
        z = torch.randn(batch_size, self.latent_dim).to(current_device)
        return self.decode(z)
    
if __name__ == "__main__":
    vae = VAE(in_channels=1, latent_dim=20)
    print(vae)  # This will print the model architecture

    # You can also test the forward pass with a random input
    test_input = torch.randn(16, 1, 28, 28)  # Batch size of 16, 1 channel, 28x28 image
    output = vae(test_input)
    print(output[0].shape)  # Should print the shape of the reconstructed images
    print(output[1].shape)  # Should print the shape of the input images
    print(output[2].shape)  # Should print the shape of the mu vector
    print(output[3].shape)  # Should print the shape of the log_var vector
    
    output_img = output[0][0].detach().numpy().squeeze()
    output_img = cv2.resize(output_img, (280, 280))
    
    cv2.imshow("Reconstructed Image", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

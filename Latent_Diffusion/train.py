import os
import numpy as np

from tqdm import tqdm
from diffusers import DDPMScheduler

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch.nn.functional as F

import torchio as tio
import matplotlib.pyplot as plt

from latent_diffusion import TimeEmbedding, EmbeddingDataset, UNet1D

def plot_losses(losses, num_epochs, title):
    # Epochs array
    epochs = np.arange(1, num_epochs + 1)  # Epochs from 1 to 100

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue', label='Training Loss')
    plt.title(title, fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(title + ".png")


# Training loop
def train_unet(model, noise_scheduler, optimizer, train_loader, num_epochs, lr, device, time_embed):

    print('Entering training loop')
    training_losses = []
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for latents in train_loader:

            latents = latents.to(device)  # Move to GPU/CPU
            latents = latents.permute(0, 2, 1)  # (batch_size, 16, 953)

            # Add noise to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            timesteps = timesteps.to(device)
            noisy_latents = noisy_latents.to(device)

            # Embed timestamp
            t_emb = time_embed(timesteps)  # Shape: (1, 128)

            # Reshape and add the embedding to the input tensor
            t_emb = t_emb.unsqueeze(-1)  # Shape: (1, 128, 1)

            noise_pred = model(noisy_latents, t_emb)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Log progress
        print(f"UNet Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")
        training_losses.append(epoch_loss / len(train_loader))
    return model, training_losses

# Main Function
def main():
    # Configurations
    directory = os.path.join(os.path.dirname(__file__), 'saved_melspectrogram_files')
    batch_size = 3
    num_epochs = 20
    learning_rate = 1e-4

    # Determine device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare dataset and dataloader
    transform = Compose([torch.tensor])  # Convert numpy arrays to torch tensors

    dataset = EmbeddingDataset(directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    model = UNet1D(in_channels=128, out_channels=128)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    embedding_dim = 128  # Match the channel dimension of the input tensor
    time_embed = TimeEmbedding(embedding_dim, device)

    # Call the training function
    model, training_losses = train_unet(model, noise_scheduler, optimizer, dataloader, num_epochs, learning_rate, device, time_embed)
    
    plot_losses(losses=training_losses, num_epochs=num_epochs, title='Latent Model Loss Over 20 Epochs')
    model_save_path = os.path.join(os.path.dirname(__file__), 'trained_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
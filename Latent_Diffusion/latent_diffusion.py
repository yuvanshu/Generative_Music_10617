import os
import numpy as np

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchio as tio


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters=256):
        super(UNet1D, self).__init__()

        # Encoder: Downsampling
        self.enc1 = self.conv_block(in_channels, num_filters)
        self.enc2 = self.conv_block(num_filters, num_filters * 2)
        self.enc3 = self.conv_block(num_filters * 2, num_filters * 4)
        self.enc4 = self.conv_block(num_filters * 4, num_filters * 8)

        # Bottleneck
        self.bottleneck = self.conv_block(num_filters * 8, num_filters * 16)

        # Decoder: Upsampling
        self.up4 = self.upconv_block(num_filters * 16, num_filters * 8, kernel_size=3, stride=2, padding=1)
        self.up3 = self.upconv_block(num_filters * 8, num_filters * 4, kernel_size=4, stride=2, padding=1)
        self.up2 = self.upconv_block(num_filters * 4, num_filters * 2, kernel_size=3, stride=2, padding=1)
        self.up1 = self.upconv_block(num_filters * 2, num_filters, kernel_size=3, stride=2, padding=1)
        self.up0 = self.upconv_block(num_filters, num_filters//2, kernel_size=4, stride=2, padding=1)

        # Final layer
        self.final_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        """Standard 1D convolutional block"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

    def upconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Upsampling block using transposed convolution"""
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x, time_emb):

        # Fusing Time Embedding With Noisy Latent Embedding
        x = x + time_emb

        # Encoder pass
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck pass
        bottleneck = self.bottleneck(enc4)

        # Decoder pass with skip connections
        up4 = self.up4(bottleneck)
        up3 = self.up3(up4 + enc4)  # Ensure same length before addition
        up2 = self.up2(up3 + enc3)
        up1 = self.up1(up2 + enc2)
        up0 = self.up0(up1 + enc1)

        # Final output
        return self.final_conv(up0)
    
class EmbeddingDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        embedding = np.load(self.files[idx])
        embedding = embedding.T
        if embedding.shape != (938, 128):
            raise ValueError(f"Expected embedding shape (938, 128), but got {embedding.shape}")
        if self.transform:
            embedding = self.transform(embedding)
        return embedding

class TimeEmbedding(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),  # Activation function
            nn.Linear(dim, dim)
        )
        self.linear.to(device)

    def forward(self, t):
        # Convert timesteps to sinusoidal embeddings
        half_dim = self.linear[0].in_features // 2
        emb = torch.arange(half_dim, device=t.device).float() / half_dim
        emb = t[:, None] * 10000**(-emb)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        res = self.linear(emb)
        return res
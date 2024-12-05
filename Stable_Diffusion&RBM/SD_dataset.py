import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


class CustomNormalization:
    def __call__(self, tensor):
        # transform to [-1, 1]
        normalized = tensor * 2 - 1
        rounded = normalized.round()

        # -0 to 0
        return rounded + (rounded == 0).to(torch.float32) * 0.0


class PianoRollDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            CustomNormalization()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        rgb_image = Image.open(image_path).convert('RGB')
        image = self.transform(rgb_image)
        return image
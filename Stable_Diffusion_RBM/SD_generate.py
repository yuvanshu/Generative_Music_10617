import torch
import torchvision

from diffusers import UNet2DModel, DDPMScheduler

import os


def get_device():
    """
    Get the appropriate device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_images(images, path, index):
    """Save a batch of images during training for monitoring."""
    images = images.mean(dim=1, keepdim=True)
    # Convert to binary
    images = (images >= -0.8).float() # Threshold for binary images -0.8
    grid = torchvision.utils.make_grid(images)
    # Convert to PIL image
    grid_image = torchvision.transforms.ToPILImage()(grid)
    os.makedirs(path, exist_ok=True)
    grid_image.save(f"{path}/sample_{index}.png")


def generate_images(
        checkpoint_path,
        image_height=768,
        image_width=512,
        output_dir="generated_images"
):

    config = {
        "image_height": image_height,
        "image_width": image_width,
        "sample_dir": output_dir
    }

    device = get_device()
    print(f"Using device: {device}")

    # Initialize model with 3 input/output channels for RGB
    model = UNet2DModel(
        sample_size=[image_height, image_width],
        in_channels=3,
        out_channels=3,
        layers_per_block=3,
        block_out_channels=(32, 64, 128),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )

    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        for i in range(100):
            # Generate sample images
            sample = torch.randn(1, 3, config["image_height"], config["image_width"]).to(device)
            timesteps = torch.linspace(999, 0, 50).long().to(device)
            for t in timesteps:
                residual = model(sample, t.repeat(1), return_dict=False)[0]
                sample = noise_scheduler.step(residual, t, sample).prev_sample
            save_images(sample, config["sample_dir"], i)


if __name__ == '__main__':
    generate_images(
        checkpoint_path="model_epoch_4.pt",
        image_height=768,
        image_width=512,
        output_dir="../generated_images"
    )
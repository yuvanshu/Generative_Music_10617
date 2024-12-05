import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm

from SD_dataset import PianoRollDataset


def get_device():
    """
    Get the appropriate device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, device):
    def calculate_weights(batch):
        unique, counts = torch.unique(batch, return_counts=True)
        total_pixels = batch.numel()
        weights = total_pixels / (len(unique) * counts)
        weights = weights / weights.sum()
        return {val.item(): weight.item() for val, weight in zip(unique, weights)}

    progress_bar = tqdm(total=config["num_epochs"] * len(train_dataloader))
    global_step = 0

    loss_record = []
    for epoch in range(config["num_epochs"]):
        model.train()
        for batch in train_dataloader:
            clean_images = batch.to(device)
            batch_size = clean_images.shape[0]
            class_weights = calculate_weights(clean_images)

            # Sample noise and add to images
            noise = torch.randn(clean_images.shape).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,),
                device=device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise, reduction='none')
            # print(f"loss:{loss}")
            pixel_weights = torch.ones_like(loss)
            for val, weight in class_weights.items():
                mask = (clean_images == val)
                pixel_weights[mask] = weight

            # Weighted loss
            weighted_loss = (loss * pixel_weights).mean()
            loss_record.append(weighted_loss)

            print(f"loss:{weighted_loss}")

            # Backpropagation
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            progress_bar.update(1)
            global_step += 1

            if global_step % config["save_interval"] == 0:
                # Save checkpoint
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f"checkpoint_{global_step}.pt")

        # Save model after each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
        with open("loss.txt", "w") as file:
            for item in loss:
                file.write(f"{item}\n")


def main():
    # Configuration
    config = {
        "image_height": 768,
        "image_width": 512,
        "batch_size": 2,
        "num_epochs": 5,
        "learning_rate": 1e-4,
        "save_interval": 500,
        "sample_interval": 1000,  # Interval for generating sample images
        "data_dir": "piano_roll_images",  # Your image directory
        "sample_dir": "samples"  # Directory to save generated samples
    }

    # Initialize device
    device = get_device()
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = PianoRollDataset(config["data_dir"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )

    # Initialize model with 3 input/output channels for RGB
    model = UNet2DModel(
        sample_size=(config["image_height"], config["image_width"]),
        in_channels=3,  # RGB input
        out_channels=3,  # RGB output
        layers_per_block=3,
        block_out_channels=(32, 64, 128),  # Further reduced channels
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

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Train model
    train_loop(config, model, noise_scheduler, optimizer, dataloader, device)


if __name__ == "__main__":
    main()
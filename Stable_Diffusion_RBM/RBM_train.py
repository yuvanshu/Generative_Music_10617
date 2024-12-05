import torch
import numpy as np
import os
from PIL import Image

from RBM_dataset import MusicRBM


def get_device():
    """
    Get the appropriate device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_data(noisy_data, clean_data, device=None):
    """
    Convert and move data to specified device

    :param noisy_data: noisy input data
    :param clean_data: clean corresponding data
    :param device: device to move data
    :return: noisy_tensor, clean_tensor
    """
    if device is None:
        device = get_device()

    # Convert to tensors and move to device
    noisy_tensor = torch.tensor(noisy_data, dtype=torch.float32).to(device)
    clean_tensor = torch.tensor(clean_data, dtype=torch.float32).to(device)

    return noisy_tensor, clean_tensor


def train_music_rbm(noisy_data, clean_data,
                    input_shape=(88, 64),
                    num_hidden=128,
                    epochs=100,
                    learning_rate=0.01,
                    batch_size=32,
                    model_save_path='music_rbm_model.pth',
                    device=None):
    """
    Train a Restricted Boltzmann Machine (RBM) for music data denoising

    :param noisy_data: noisy input data
    :param clean_data: clean corresponding data
    :param input_shape: the shape of input image
    :param num_hidden: the number of hidden nodes
    :param epochs: number of training epochs
    :param learning_rate: learning rate
    :param batch_size: batch size
    :param model_save_path: path to save trained model
    :param device: device to run the model

    :return: reconstructed, rbm
    """
    # Automatically select device
    if device is None:
        device = get_device()

    # Prepare data
    noisy_tensor, clean_tensor = prepare_data(noisy_data, clean_data, device)

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(noisy_tensor, clean_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # Drop last batch if size is less than batch_size
    )

    # Initialize RBM model
    rbm = MusicRBM(
        input_shape=input_shape,  # 88 keys, 64 time steps
        num_hidden=num_hidden,
        device=device
    )

    total_batches = len(dataloader)
    for epoch in range(epochs):
        epoch_error = 0.0

        for batch_idx, (noisy_batch, clean_batch) in enumerate(dataloader):
            # Train RBM using contrastive divergence
            error = rbm.contrastive_divergence(
                noisy_batch,
                clean_batch,
                learning_rate=learning_rate
            )

            # Accumulate error
            epoch_error += error.item()

        # Print average error for each epoch
        if epoch % 10 == 0:
            avg_error = epoch_error / total_batches
            print(f'Epoch {epoch}, Average Reconstruction Error: {avg_error}')

    # Reconstruct clean data from noisy data
    reconstructed = rbm.reconstruct(noisy_tensor)

    # Move data back to CPU
    rbm = rbm.cpu()
    reconstructed = reconstructed.cpu()

    # Save model
    torch.save({
        'model_state_dict': rbm.state_dict(),
        'input_shape': rbm.input_shape,
        'num_hidden': rbm.num_hidden
    }, model_save_path)

    print(f'Model saved to {model_save_path}')

    return reconstructed, rbm


def add_random_noise(ground_truth, noise_prob=0.1, noise_type='random'):
    """
    Add random noise to the input data

    :param ground_truth: clean input data
    :param noise_prob: noise probability

    :return: noisy_roll
    """
    # Copy the ground truth data
    noisy_roll = ground_truth.copy()

    if noise_type == 'random':
        # Random noise
        noise_mask = np.random.random(ground_truth.shape) < noise_prob
        noisy_roll[noise_mask] = 1 - ground_truth[noise_mask]

    elif noise_type == 'dropout':
        # Dropout noise
        dropout_mask = np.random.random(ground_truth.shape) < noise_prob
        noisy_roll[dropout_mask] = 0

    elif noise_type == 'sparse':
        # Sparse noise
        for _ in range(int(noise_prob * ground_truth.size)):
            x = np.random.randint(ground_truth.shape[0])
            y = np.random.randint(ground_truth.shape[1])
            noisy_roll[x, y] = 1 - ground_truth[x, y]

    else:
        raise ValueError("Unsupported noise type. Choose 'random', 'dropout', or 'sparse'.")

    return noisy_roll


def generate_noisy_dataset(ground_truth_rolls,
                           noise_prob=0.1,
                           noise_type='random',
                           num_augmentations=5):
    """
    Generate noisy dataset from clean ground truth data

    :param ground_truth_rolls: list of clean ground truth data
    :param noise_prob: noise probability
    :param noise_type: type of noise to add
    :param num_augmentations: number of noisy versions to generate for each ground truth

    :return: noisy_rolls, clean_rolls
    """
    noisy_rolls = []
    clean_rolls = []

    for ground_truth in ground_truth_rolls:
        # Generate noisy versions
        for _ in range(num_augmentations):
            noisy_roll = add_random_noise(
                ground_truth,
                noise_prob=noise_prob,
                noise_type=noise_type
            )

            noisy_rolls.append(noisy_roll)
            clean_rolls.append(ground_truth)

    return noisy_rolls, clean_rolls


def main():
    image_dir = "../piano_roll_images_RBM"
    ground_truth_rolls = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(image_dir, file_name)
            # Load image and convert to binary matrix
            image = Image.open(file_path).convert('L')  # Convert to grayscale
            binary_matrix = np.array(image) < 128  # Convert to binary
            binary_matrix = binary_matrix.astype(int)  # Convert to integer
            ground_truth_rolls.append(binary_matrix)

    # Generate noisy dataset
    noisy_rolls, clean_rolls = generate_noisy_dataset(
        ground_truth_rolls,
        noise_prob=0.3,
        noise_type='random',
        num_augmentations=5  # Generate 5 noisy versions for each ground truth
    )

    # Train RBM model
    reconstructed_data, trained_model = train_music_rbm(
        noisy_rolls,
        clean_rolls,
        device='mps'
    )


def load_rbm_model(model_path, device=None):
    """
    Load a pre-trained RBM model from checkpoint
    """
    # Automatically select device
    if device is None:
        device = get_device()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Initialize RBM model
    rbm = MusicRBM(
        input_shape=checkpoint['input_shape'],
        num_hidden=checkpoint['num_hidden'],
        device=device
    )

    # Load model state
    rbm.load_state_dict(checkpoint['model_state_dict'])

    # Move model to device
    rbm = rbm.to(device)

    return rbm


if __name__ == '__main__':
    main()
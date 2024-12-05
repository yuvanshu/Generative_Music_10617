from diffusers import DDPMScheduler

import torch

from latent_diffusion import UNet1D

import librosa
import librosa.display
import soundfile as sf  # Import soundfile for saving wav files

def main():

    model = UNet1D(in_channels=128, out_channels=128)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    # Configurations
    model.eval()

    # Define the scheduler
    num_timesteps = 1000  # Number of timesteps used during training
    scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
    scheduler.set_timesteps(num_timesteps)

    # Initialize random noise
    batch_size = 1
    num_channels = 128
    sequence_length = 938
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise = torch.randn(batch_size, num_channels, sequence_length, device=device)

    # Perform the reverse diffusion process
    with torch.no_grad():
        for t in scheduler.timesteps:
            # Predict noise residual
            noise_pred = model(noise, t)

            # Perform denoising step
            noise = scheduler.step(noise_pred, t, noise).prev_sample

    # The final `noise` is your generated latent embedding
    generated_latent = noise
    generated_latent = generated_latent.squeeze(0)
    print(f"Generated Latent Shape: {generated_latent.shape}")

    sr = 16000

    # Inverse Griffin-Lim to reconstruct the audio
    generated_audio = librosa.feature.inverse.mel_to_audio(generated_latent)

    # Save the reconstructed audio to a .wav file
    sf.write('generated_audio.wav', generated_audio, sr)

if __name__ == "__main__":
    main()
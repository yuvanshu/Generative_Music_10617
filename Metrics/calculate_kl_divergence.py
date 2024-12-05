import librosa
import numpy as np
import os
from scipy.special import kl_div

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def compute_histogram(features, bins=100):
    hist, edges = np.histogram(features, bins=bins, density=True)
    hist = hist / np.sum(hist)
    return hist

def process_directory(directory):
    all_features = []
    for audio_file in os.listdir(directory):
        if audio_file.endswith('.wav'):
            features = extract_features(os.path.join(directory, audio_file))
            all_features.append(features)
    return np.array(all_features)

def compute_kl_divergence(real_features, gen_features, num_bins=100, epsilon=1e-10):
    """
    Compute KL divergence between real and generated audio features histograms.

    Args:
    - real_features (array-like): The real audio features (e.g., spectrogram).
    - gen_features (array-like): The generated audio features (e.g., spectrogram).
    - num_bins (int): The number of bins for the histogram.
    - epsilon (float): A small value added to histograms to prevent zero values.

    Returns:
    - kl_divergence (float): The computed KL divergence.
    """
    # Compute histograms for real and generated features
    feature_range = (min(np.min(real_features), np.min(gen_features)),
                     max(np.max(real_features), np.max(gen_features)))

    real_hist = np.histogram(real_features, bins=num_bins, range=feature_range, density=True)[0]
    gen_hist = np.histogram(gen_features, bins=num_bins, range=feature_range, density=True)[0]

    # Apply smoothing to both histograms
    real_hist = real_hist + epsilon
    gen_hist = gen_hist + epsilon

    # Normalize histograms
    real_hist = real_hist / np.sum(real_hist)
    gen_hist = gen_hist / np.sum(gen_hist)

    # Compute the KL Divergence
    kl_divergence = np.sum(real_hist * np.log(real_hist / gen_hist))

    return kl_divergence

def main():

    # Directories of real and generated audio
    # Directories for real and generated audio
    real_wav_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Saved_Files', 'saved_real_wav_files')

    # Modify according to generated wav file directory for evaluation
    generated_wav_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Saved_Files', 'saved_generated_latent_wav_files')

    # Process the directories to get features
    real_features = process_directory(real_wav_dir)
    gen_features = process_directory(generated_wav_dir)

    # Compute histograms for the features
    real_hist = compute_histogram(real_features)
    gen_hist = compute_histogram(gen_features)


    # # Compute KL Divergence
    kl_divergence = compute_kl_divergence(real_hist, gen_hist)
    print(f"KL Divergence: {kl_divergence}")

if __name__ == "__main__":
    main()
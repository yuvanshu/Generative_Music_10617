import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import librosa
from transformers import AutoProcessor, ClapAudioModel
import torch

# Function to convert WAV file to Mel Spectrogram embedding
def wav_to_melspectrogram(file_path):
    """
    Converts a WAV file to a Mel Spectrogram.

    Args:
        file_path (str): Path to the WAV file.

    Returns:
        np.ndarray: Mel spectrogram.
    """
    y, sr = librosa.load(file_path, sr=None)  # `sr=None` to preserve original sampling rate
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibel scale
    return mel_spec_db.flatten()

# Function to compute CLAP embedding from a Mel Spectrogram
def compute_clap_embedding(model, processor, mel_spec):
    """
    Computes the CLAP embedding from a Mel Spectrogram.

    Args:
        mel_spec (np.ndarray): Mel spectrogram.

    Returns:
        np.ndarray: CLAP embedding.
    """
    inputs = processor(audios=mel_spec, return_tensors="pt")
    outputs = model(**inputs)
    clap_embedding = outputs.last_hidden_state.detach().numpy()  # Convert to NumPy array
    return clap_embedding.flatten()

# Function to process a directory of WAV files and save CLAP embeddings
def process_directory(input_dir, output_dir, model, processor):
    """
    Processes a directory of WAV files, computes CLAP embeddings, and saves them.

    Args:
        input_dir (str): Path to the directory containing WAV files.
        output_dir (str): Path to save the CLAP embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    if not wav_files:
        print(f"No WAV files found in directory {input_dir}.")
        return

    for wav_file in wav_files:
        input_path = os.path.join(input_dir, wav_file)
        output_path = os.path.join(output_dir, os.path.splitext(wav_file)[0] + '.npy')

        # Convert WAV to Mel Spectrogram
        mel_spec = wav_to_melspectrogram(input_path)

        # Compute CLAP embedding
        clap_embedding = compute_clap_embedding(model, processor, mel_spec)

        # Save the CLAP embedding as a .npy file
        np.save(output_path, clap_embedding)
        print(f"Saved CLAP embedding for {wav_file} to {output_path}")

# Function to load CLAP embeddings from a directory
def load_embeddings(embedding_dir):
    """
    Loads the CLAP embeddings from a directory of .npy files.

    Args:
        embedding_dir (str): Directory containing .npy CLAP embeddings.

    Returns:
        list of np.ndarray: List of loaded CLAP embeddings.
    """
    embeddings = []
    for filename in os.listdir(embedding_dir):
        if filename.endswith('.npy'):
            file_path = os.path.join(embedding_dir, filename)
            embedding = np.load(file_path)
            embeddings.append(embedding)
    return embeddings

# Function to compute the CLAP score (average cosine similarity)
def compute_clap_score(real_embeddings, generated_embeddings):
    """
    Computes the average CLAP score by comparing real and generated embeddings using cosine similarity.

    Args:
        real_embeddings (list of np.ndarray): List of CLAP embeddings for real audio.
        generated_embeddings (list of np.ndarray): List of CLAP embeddings for generated audio.

    Returns:
        float: Average cosine similarity score.
    """
    print('Here is len of real embeddings: ', len(real_embeddings))
    print('Here is len of generated embeddings: ', len(generated_embeddings))
    if len(real_embeddings) != len(generated_embeddings):
        raise ValueError("The number of real and generated embeddings must be the same.")

    similarities = []

    for real, generated in zip(real_embeddings, generated_embeddings):
        # Reshape the embeddings to be 2D (required by cosine_similarity)
        real = real.reshape(1, -1)
        generated = generated.reshape(1, -1)

        # Compute cosine similarity
        similarity = cosine_similarity(real, generated)[0][0]
        similarities.append(similarity)

    # Return the average cosine similarity
    return np.mean(similarities)

def main():

    # Load the pretrained CLAP model and processor
    model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

    # Directories for real and generated audio
    real_wav_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Saved_Files', 'saved_real_wav_files')

    # Modify according to generated wav file directory for evaluation
    generated_wav_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Saved_Files', 'saved_generated_latent_wav_files')

    # Output directories for embeddings
    real_embedding_dir = os.path.join(os.path.dirname(__file__), 'real_clap_embeddings')
    generated_embedding_dir = os.path.join(os.path.dirname(__file__), 'generated_clap_embeddings')

    # Process both directories
    print("Processing real audio directory...")
    process_directory(real_wav_dir, real_embedding_dir, model, processor)

    print("Processing generated audio directory...")
    process_directory(generated_wav_dir, generated_embedding_dir, model, processor)

    print("All files processed and CLAP embeddings saved.")

    # Load embeddings from both directories
    real_embeddings = load_embeddings(real_embedding_dir)
    generated_embeddings = load_embeddings(generated_embedding_dir)

    # Compute the CLAP score
    clap_score = compute_clap_score(real_embeddings, generated_embeddings)

    # Print the CLAP score
    print(f"CLAP Score: {clap_score}")

if __name__ == "__main__":
    main()
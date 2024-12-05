import os
import zipfile

def expand_zip_files():
    # Get the directory of this script
    current_dir = os.path.dirname(__file__)
    saved_files_dir = os.path.join(current_dir)  # Points to "Saved Files"

    # Map of zip file names to corresponding folder names
    zip_to_folder = {
        "generated_latent_wavs.zip": "saved_generated_latent_wav_files",
        "generated_rnn_wavs.zip": "saved_rnn_wav_files",
        "generated_stable_diffusion_rbm_wavs.zip": "saved_stable_diffusion_rbm_files",
        "melspectrograms.zip": "saved_melspectrogram_files",
        "real_wavs.zip": "saved_real_wav_files",
    }

    # Iterate through the zip files and expand them into their corresponding folders
    for zip_file, folder in zip_to_folder.items():
        zip_file_path = os.path.join(saved_files_dir, zip_file)
        target_folder_path = os.path.join(saved_files_dir, folder)

        # Check if the zip file exists
        if os.path.exists(zip_file_path):
            print(f"Expanding {zip_file} into {folder}...")
            
            # Ensure the target folder exists
            os.makedirs(target_folder_path, exist_ok=True)

            # Expand the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(target_folder_path)

            print(f"Finished expanding {zip_file} into {folder}.")
        else:
            print(f"Zip file {zip_file} not found. Skipping...")

if __name__ == "__main__":
    expand_zip_files()

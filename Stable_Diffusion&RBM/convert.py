import numpy as np
from PIL import Image
import pretty_midi
import pandas as pd
import soundfile as sf
import os
import torch

from RBM_train import load_rbm_model


def image_to_pitch(image_path, min_pitch=21, time_step=0.1):
    """
    Convert a binary image to a list of MIDI notes

    Args:
        image_path (str): Path to the input image file
        min_pitch (int): Minimum MIDI pitch (default: 21 = A0)
        time_step (float): Time step in seconds (default: 0.1)

    Returns:
        pd.DataFrame: DataFrame containing MIDI notes with columns:
            'pitch', 'start_time', 'end_time', 'duration', 'step'
    """
    # Load the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    img_array = np.array(image)
    rbm = load_rbm_model('music_rbm_model.pth', device='mps')
    img = img_array
    img = np.where(img == 255, 0, 1)

    picture_raw = np.zeros((88, 64*8))
    count = 0
    for j in range(8):
        noisy_roll = img[:, j * 64:(j + 1) * 64].astype(int)
        noisy_roll = noisy_roll.reshape(96, 8, 64).max(axis=1)
        noisy_roll = noisy_roll[4:-4, :]
        noisy_roll = torch.tensor(noisy_roll, dtype=torch.float32).unsqueeze(0).to(rbm.device)
        noisy_roll = rbm.reconstruct(noisy_roll)
        noisy_roll = noisy_roll.squeeze(0).cpu().detach().numpy()
        picture_raw[:, count * 64:(count + 1) * 64] = noisy_roll
        count += 1

    piano_roll = picture_raw

    notes = []
    num_pitches, num_time_steps = piano_roll.shape

    for pitch_idx in range(num_pitches):
        midi_pitch = pitch_idx + min_pitch

        # get the pitch sequence
        pitch_sequence = piano_roll[pitch_idx]

        # find note onsets and offsets
        note_starts = np.where(np.diff(np.concatenate(([0], pitch_sequence))) == 1)[0]
        note_ends = np.where(np.diff(np.concatenate((pitch_sequence, [0]))) == -1)[0]

        # convert note indices to time
        for start, end in zip(note_starts, note_ends):
            note = {
                'pitch': int(midi_pitch),
                'start_time': float(start * time_step),
                'end_time': float(end * time_step),
                'duration': float((end - start) * time_step),
                'step': 0.0  # 初始化step，稍后更新
            }
            notes.append(note)

        # sort notes by start time
        notes.sort(key=lambda x: x['start_time'])

        # update step times
        for i in range(len(notes)):
            if i == 0:
                notes[i]['step'] = 0.0  # 第一个音符的step为0
            else:
                notes[i]['step'] = round(notes[i]['start_time'] - notes[i - 1]['start_time'], 6)

    return pd.DataFrame(notes)


def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100):  # note loudness
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    prev_start = 0

    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)

    return pm


def midi_to_wav(input_folder, output_folder, duration=30, sample_rate=16000):
    """
    Convert MIDI files in the input folder to 30-second WAV files

    Args:
        input_folder (str): Path to folder containing MIDI files
        output_folder (str): Path to folder for saving WAV files
        duration (float): Desired duration in seconds
        sample_rate (int): Audio sample rate
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Supported MIDI file extensions
    midi_extensions = ['.mid', '.midi']

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is a MIDI file
        if os.path.splitext(filename)[1].lower() in midi_extensions:
            try:
                # Full path to the input MIDI file
                input_path = os.path.join(input_folder, filename)

                # Full path to the output WAV file
                output_filename = os.path.splitext(filename)[0] + '.wav'
                output_path = os.path.join(output_folder, output_filename)

                # Load the MIDI file
                midi_data = pretty_midi.PrettyMIDI(input_path)

                # Synthesize the MIDI file
                audio_data = midi_data.synthesize()

                # Trim or pad the audio to exactly 30 seconds
                if len(audio_data) / sample_rate > duration:
                    # Trim to 30 seconds
                    audio_data = audio_data[:int(duration * sample_rate)]
                else:
                    # Pad with zeros if shorter than 30 seconds
                    pad_length = int(duration * sample_rate) - len(audio_data)
                    audio_data = np.pad(audio_data, (0, pad_length), mode='constant')

                # Normalize audio to prevent clipping
                audio_data = audio_data / np.max(np.abs(audio_data))

                # Save as WAV file
                sf.write(output_path, audio_data, sample_rate)

                print(f"Converted {filename} to {output_filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    image_path = "../generated_images"
    output_dir = "../midi_files"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(image_path):
        if os.path.splitext(file)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            input_path = os.path.join(image_path, file)
            output_midi_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.midi")

            raw_notes = image_to_pitch(input_path)
            midi = notes_to_midi(raw_notes, output_midi_path, instrument_name="Acoustic Grand Piano")

    midi_to_wav("midi_files", "wav_files")




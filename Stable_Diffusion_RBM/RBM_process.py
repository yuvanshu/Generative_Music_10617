import glob
import pathlib
import collections
import pretty_midi
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm
from PIL import Image
import random
import matplotlib.pyplot as plt


def midi_to_notes(midi_file: str) -> pd.DataFrame:

    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def create_piano_roll_matrix(
        notes: pd.DataFrame,
        count: Optional[int] = None,
        time_step: float = 0.01,
        min_pitch: int = 21,
        max_pitch: int = 108  # The range from the dataset is 21 to 108
) -> Tuple[np.ndarray, float]:
    """
    Convert note events to a binary piano roll matrix.

    Args:
        notes: DataFrame with columns 'pitch', 'start', 'end'
        count: Optional number of notes to process
        time_step: Time resolution in seconds
        min_pitch: Minimum MIDI pitch to include (default is 21 for standard piano)
        max_pitch: Maximum MIDI pitch to include (default is 108 for standard piano)

    Returns:
        Tuple containing:
        - Binary matrix of shape (num_pitches, num_time_steps)
        - Time step size used for the matrix
    """

    if count is None:
        count = len(notes)
    else:
        notes = notes.iloc[:count]

    # Calculate time boundaries
    start_time = notes['start'].min()
    end_time = notes['end'].max()

    # Calculate matrix dimensions
    num_time_steps = int(np.ceil((end_time - start_time) / time_step))
    num_pitches = max_pitch - min_pitch + 1

    # Initialize empty matrix
    piano_roll = np.zeros((num_pitches, num_time_steps), dtype=np.int8)

    # Fill in the matrix
    for _, note in notes.iterrows():
        # Convert time to matrix indices
        start_idx = int((note['start'] - start_time) / time_step)
        end_idx = int((note['end'] - start_time) / time_step)
        pitch_idx = int(note['pitch'] - min_pitch)

        # Ensure indices are within bounds
        if 0 <= pitch_idx < num_pitches:
          piano_roll[pitch_idx, start_idx:end_idx] = 1

    return piano_roll, time_step


if __name__ == '__main__':
    data_dir = pathlib.Path('../maestro-v2.0.0')
    filenames = glob.glob(str(data_dir/'**/*.mid*'))

    output_dir = pathlib.Path('../piano_roll_images_RBM')
    output_dir.mkdir(exist_ok=True)
    all_piano_rolls = []  # Your piano roll matrices here
    jpg_index = 0
    for filename in tqdm(filenames):
        raw_notes = midi_to_notes(filename)
        piano_roll, _ = create_piano_roll_matrix(raw_notes)
        num_jpg = piano_roll.shape[1] // 64
        selected_indices = sorted(random.sample(range(num_jpg), int(num_jpg * 0.01)))
        for i in selected_indices:
            picture_raw = np.zeros((88, 64), dtype=np.uint8)
            picture_raw[:,:] = piano_roll[:, i*64:(i+1)*64]
            picture_raw = (1 - picture_raw) * 255
            img = Image.fromarray(picture_raw.astype('uint8'), mode='L')
            img.save(f"{output_dir}/piano_roll_{jpg_index}.jpg")
            jpg_index += 1





import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import multiprocessing
from typing import List
from scripts.config import SOUNDS_DIR, SPECTOGRAM_DIR

def create_spectrogram_from_audio(audio_file: str, image_file: str) -> None:
    """
    Create a spectrogram from an audio file and save it as an image.

    Args:
        audio_file (str): Path to the input audio file (e.g., a .wav file).
        image_file (str): Path to save the output spectrogram image (e.g., a .png file).

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


    y, sr = librosa.load(audio_file)
    y = librosa.util.normalize(y)


    st = abs(librosa.stft(y))
    st = librosa.util.normalize(st)

    # Display the spectrogram and save it as an image
    librosa.display.specshow(st, sr=sr, y_axis='log')
    fig.savefig(image_file)
    plt.close(fig)

def create_pngs_from_wavs(input_path: str, output_path: str) -> None:
    """
    Convert all .wav files in a directory to spectrogram images.

    Args:
        input_path (str): Path to the directory containing .wav files.
        output_path (str): Path to save the spectrogram images.

    Returns:
        None
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    dir = os.listdir(input_path)
    for i, file in enumerate(dir):
        
        if (i + 1) % 100 == 0:
            print(input_path, i + 1, "/", len(dir))

        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))


        create_spectrogram_from_audio(input_file, output_file)

def create_spectrograms(labels: List[str]) -> None:
    """
    Create spectrograms for all audio files in the specified directories using multiprocessing.

    Args:
        labels (List[str]): A list of labels corresponding to the directories containing audio files.
                           Each label should have a corresponding directory under `Sounds/{label}/target`.

    Returns:
        None
    """
    # Create a process for each label to generate spectrograms in parallel
    all_threads = [
        multiprocessing.Process(target=create_pngs_from_wavs, args=(os.path.join(SOUNDS_DIR,label,'target'), os.path.join(SPECTOGRAM_DIR,label)))
        for label in labels
    ]

    [t.start() for t in all_threads]

    [t.join() for t in all_threads]

if __name__ == "__main__":

    labels = ['kai_yuan', 'noise', 'speedboat', 'uuv']
    create_spectrograms(labels=labels)
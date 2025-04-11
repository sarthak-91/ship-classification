import os
import matplotlib.pyplot as plt
import keras.utils as image
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from typing import List
from scripts.config import SPECTOGRAM_DIR,DATA_DIR

def load_images_from_path(path:os.PathLike, label:List[str]):
    """
    Load images from a specified path and assign labels to them.
    
    Args:
        path (str): Path to the directory containing images.
        label (int): Label to assign to the images.
    
    Returns:
        list: List of images.
        list: List of labels.
    """
    images = []
    labels = []
    all_files = os.listdir(path)
    for idx, file in enumerate(all_files):
        if idx % 100 == 0:
            print(idx, "/", len(all_files), "Done")
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(75, 75, 3))))
        labels.append(label)
    return images, labels

def create_image_label(labels:List[str]):
    """
    Create image and label arrays from spectrogram images.
    
    Args:
        labels (list): List of labels corresponding to the classes.
    
    Returns:
        np.array: Array of images.
        np.array: Array of labels.
    """
    x = []
    y = []
    for idx, label in enumerate(labels):
        print("Working on", label)
        images, labels = load_images_from_path(os.path.join(SPECTOGRAM_DIR,label),idx)
        x += images
        y += labels
    image_array = np.array(x, dtype=np.float16).reshape((len(x), -1))
    return image_array, np.array(y, dtype='int')

def store_train_test(x_train:np.ndarray, x_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray):
    """
    Store the preprocessed training and testing data.
    
    Args:
        x_train (np.array): Training images.
        x_test (np.array): Testing images.
        y_train (np.array): Training labels.
        y_test (np.array): Testing labels.
    """
    x_train_norm = x_train / 255
    np.savez_compressed(os.path.join(DATA_DIR,"x_train.npz"), data_array=x_train_norm.reshape((x_train_norm.shape[0], -1)))
    x_test_norm = x_test / 255
    np.savez_compressed(os.path.join(DATA_DIR,"x_test.npz"), data_array=x_test_norm.reshape((x_test_norm.shape[0], -1)))
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)
    np.savetxt(os.path.join(DATA_DIR,"y_train.txt"), y_train_encoded)
    np.savetxt(os.path.join(DATA_DIR,"y_test.txt"), y_test_encoded)

def preprocess_data(labels, array_file_name, target_file_name="targets.txt"):
    """
    Preprocess the data by creating image-label arrays and splitting into train/test sets.
    
    Args:
        labels (list): List of labels corresponding to the classes.
        array_file_name (str): Name of the file to save the image array.
        target_file_name (str): Name of the file to save the target labels.
    """
    if not os.path.exists(array_file_name) or not os.path.exists(target_file_name):
        image_array, targets = create_image_label(labels)
        np.savez_compressed(array_file_name, data_array=image_array.reshape((image_array.shape[0], -1)))
        np.savetxt(target_file_name, targets)

    x = np.load(array_file_name)['data_array']
    x = x.reshape((x.shape[0], 75, 75, 3))
    y = np.loadtxt(target_file_name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3)
    store_train_test(x_train, x_test, y_train, y_test)

if __name__ == "__main__":

    labels = ['kai_yuan', 'noise', 'speedboat', 'uuv']
    preprocess_data(labels, os.path.join(DATA_DIR,"images.npz"), 
        os.path.join(DATA_DIR,"targets.txt"))
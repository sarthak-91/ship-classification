import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.mixed_precision import set_global_policy
from scripts.config import DATA_DIR,MODEL_DIR

tf.random.set_seed(200)
set_global_policy('mixed_float16')

def load_set(set="train"):
    """
    Load the dataset from .npz and .txt files.
    
    Args:
        set (str): Dataset to load ('train' or 'test').
    
    Returns:
        np.array: Image data.
        np.array: Labels.
    """
    x_path = os.path.join(DATA_DIR,f'x_{set}.npz')
    y_path = os.path.join(DATA_DIR,f'y_{set}.txt')
    x = np.load(x_path)['data_array'].astype(np.float16)
    x = x.reshape((x.shape[0], 75, 75, 3))
    y = np.loadtxt(y_path)
    return x, y

def train_cnn():
    """
    Train a CNN model on the spectrogram data.
    
    Returns:
        model: Trained CNN model.
        history: Training history.
    """
    x, y = load_set(set="train")
    N_CLASSES = 4
    batch_size = 32

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(75, 75, 3)))
    model.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Activation('linear', dtype='float16'))
    model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(), metrics=['accuracy'])
    history = model.fit(x, y, epochs=20, validation_split=0.2, batch_size=batch_size)

    model.save(os.path.join(MODEL_DIR,"cnn_model.keras"))
    np.save(os.path.join(MODEL_DIR,'cnn_history.npy'), history.history)
    return model, history

if __name__ == "__main__":
    train_cnn()
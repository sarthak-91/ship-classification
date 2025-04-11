import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import set_global_policy
from scripts.transformer import Patches, PatchEncoder
from scripts.config import DATA_DIR, MODEL_DIR, FIG_DIR
os.environ["XDG_SESSION_TYPE"] = "xcb"
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

def training_performance(history,save_to:str):
    """
    Visualize training and validation accuracy across epochs.
    Args:
        history (dict): Dictionary containing training history 
            with 'accuracy' and 'val_accuracy' keys.
        save_to (str): File path to save the accuracy plot.
    """
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '-', label='Training Accuracy')
    plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(save_to)
    plt.close()


def evaluate_model(confusion_figure:str,load_from_file=True,model_name:str=None,model:tf.keras.models.Sequential=None,transformer_model=False):
    """
    Evaluate a model on the test set and generate a confusion matrix.
    
    Args:
        model_name (str): Name of the model file to load (e.g., "cnn_model.keras").
        confusion_figure (str): Name of the file to save the confusion matrix (e.g., "cnn_confusion.png").
    """
    if load_from_file and model_name is None:
        raise ValueError("model_name must be provided when load_from_file is True.")
    
    if load_from_file and not os.path.exists(model_name):
        raise FileNotFoundError(f"Model file '{model_name}' not found.")
    if load_from_file and model is not None:
        raise ValueError("model should not be provided when load_from_file is True. Use model_name instead.")
    if not load_from_file and model is None:
        raise ValueError("model must be provided when load_from_file is False.")
    # Load the model
    if load_from_file: 
        if transformer_model:
            custom_objects = {
                "Patches": Patches,  
                "PatchEncoder": PatchEncoder, 
            }
            model = load_model(model_name, custom_objects=custom_objects)
        else:
            model = load_model(model_name)

    x_test, y_test = load_set(set="test")
    sns.set()

    # Evaluate the model
    y_predicted = model.predict(x_test)
    if y_predicted.shape[-1] == 4:
        y_predicted = y_predicted.argmax(axis=1)
    if y_test.shape[-1] == 4:
        y_test = y_test.argmax(axis=1)

    # Calculate metrics
    p, r, f, s = precision_recall_fscore_support(y_test, y_predicted)
    print(f"Model Metrics for {model_name}:")
    print("Precision:", p)
    print("Recall:", r)
    print("F1-Score:", f)
    print("Support:", s)

    # Generate and save confusion matrix
    mat = confusion_matrix(y_test, y_predicted)
    class_labels = ['kai_yuan', 'noise', 'speedboat', 'uuv']
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.savefig(confusion_figure)
    plt.close()

if __name__ == "__main__":
    cnn_path = os.path.join(MODEL_DIR,'cnn_model.keras')
    evaluate_model(confusion_figure=os.path.join(FIG_DIR,"cnn_confusion.png"), 
                   load_from_file=True, model_name=cnn_path)
    transformer_path = os.path.join(MODEL_DIR,'transformer_model.keras')
    #evaluate_model(confusion_figure=os.path.join(FIG_DIR,"transformer_confusion.png"), 
                   #load_from_file=True, model_name=transformer_path,
                   #transformer_model=True)


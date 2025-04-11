from scripts.pre_process import preprocess_data
from scripts.spectrogram import create_spectrograms
from scripts.cnn import train_cnn
from scripts.transformer import train_transformer
from scripts.evaluate import training_performance, evaluate_model
from scripts.config import *
import numpy as np


def main():
    labels = ['kai_yuan', 'noise', 'speedboat', 'uuv']
    # Step 1: Create spectrograms from sound files
    print("Creating spectrograms...")

    create_spectrograms(labels=labels)

    # Preprocess the spectrogram data
    print("Preprocessing data...")
    
    imgs_data = os.path.join(DATA_DIR,"images.npz")
    target_data = os.path.join(DATA_DIR,"targets.txt")
    preprocess_data(labels, imgs_data, target_data)

    #Train the CNN model
    print("Training CNN model...")
    cnn_model, cnn_history = train_cnn()

    # Train the Transformer model
    print("Training Transformer model...")
    transformer_model, transformer_history = train_transformer()

    # Evaluate the models
    print("Evaluating models...")
    #Plotting loss
    cnn_loss_img = os.path.join(FIG_DIR,"cnn_training_loss.png")
    transformer_loss_img = os.path.join(FIG_DIR,"transformer_training_loss.png")
    training_performance(cnn_history,save_to=cnn_loss_img)
    training_performance(transformer_history,save_to=transformer_loss_img)
    
    #Evaluate model on test set
    cnn_conf_img = os.path.join(FIG_DIR,"cnn_confusion.png")
    transformer_conf_img = os.path.join(FIG_DIR,"transformer_confusion.pngg")
    evaluate_model(confusion_figure=cnn_conf_img,load_from_file=False,model=cnn_model)
    evaluate_model(confusion_figure=transformer_conf_img,load_from_file=False,model=transformer_model,transformer_model=True)

if __name__ == "__main__":
    main()

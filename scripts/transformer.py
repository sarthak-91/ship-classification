import os
from scripts.cnn import load_set
os.environ["KERAS_BACKEND"] = "tensorflow"
from tensorflow.keras.mixed_precision import set_global_policy
import keras
from keras import layers
from keras import ops
import numpy as np
import tensorflow as tf
from scripts.config import DATA_DIR, MODEL_DIR, FIG_DIR
set_global_policy('float16')


tf.random.set_seed(200)

def mlp(x, hidden_units, dropout_rate):
    """
    Create a multi-layer perceptron (MLP) block.
    
    Args:
        x: Input tensor.
        hidden_units (list): List of hidden units for each layer.
        dropout_rate (float): Dropout rate.
    
    Returns:
        tensor: Output tensor.
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        """
        Initialize the Patches layer.

        Args:
            patch_size (int): Size of the patches to extract.
            **kwargs: Additional keyword arguments (e.g., `name`).
        """
        super().__init__(**kwargs)  # Pass additional arguments to the parent class
        self.patch_size = patch_size

    def call(self, images):
        """
        Extract patches from the input images.

        Args:
            images: Input tensor of shape (batch_size, height, width, channels).

        Returns:
            Tensor of patches with shape (batch_size, num_patches, patch_size * patch_size * channels).
        """
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        """
        Get the configuration of the layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a layer instance from its configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Patches: Layer instance.
        """
        return cls(**config)

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim=64, **kwargs):
        """
        Initialize the PatchEncoder layer.

        Args:
            num_patches (int): Number of patches.
            projection_dim (int, optional): Dimension of the projection. Defaults to 64.
            **kwargs: Additional keyword arguments (e.g., `name`).
        """
        super().__init__(**kwargs)  # Pass additional arguments to the parent class
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        """
        Encode patches with position embeddings.

        Args:
            patch: Input tensor of shape (batch_size, num_patches, patch_size * patch_size * channels).

        Returns:
            Tensor of encoded patches with shape (batch_size, num_patches, projection_dim).
        """
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        """
        Get the configuration of the layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a layer instance from its configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            PatchEncoder: Layer instance.
        """
        return cls(**config)

def create_vit_classifier():
    """
    Create a Vision Transformer (ViT) classifier model.
    
    Returns:
        model: ViT model.
    """
    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    logits = ops.cast(logits, dtype=tf.float32)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model, x_train, y_train):
    """
    Train the ViT model.
    
    Args:
        model: ViT model.
        x_train (np.array): Training images.
        y_train (np.array): Training labels.
    
    Returns:
        history: Training history.
    """
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
    )
    model.save(os.path.join(MODEL_DIR,"transformer_model.keras"))
    return history

def train_transformer():
    """
    Train the Vision Transformer model.
    
    Returns:
        model: Trained Transformer model.
        history: Training history.
    """
    global image_size, learning_rate, weight_decay, batch_size, num_epochs, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units, num_classes, input_shape, data_augmentation

    image_size = 75
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 64
    num_epochs = 20
    patch_size = 12
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 8
    mlp_head_units = [2048, 1024]
    num_classes = 4
    input_shape = (75, 75, 3)

    x_train, y_train = load_set(set="train")
    y_train = np.argmax(y_train, axis=-1)
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
        ],
        name="data_augmentation",
    )
    data_augmentation.layers[0].adapt(x_train)

    vit_classifier = create_vit_classifier()
    history = run_experiment(vit_classifier, x_train, y_train)
    np.save(os.path.join(MODEL_DIR,'transformer_history.npy'), history.history)
    return vit_classifier, history

if __name__ == "__main__":
    train_transformer()
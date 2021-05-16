"""
Title: Image Classification with MLP-Mixer
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/05/30
Last modified: 2021/05/30
Description: Implementing the MLP-Mixer model for image classification.
"""

"""
## Introduction

This example implements the [MLP-Mixer](https://arxiv.org/abs/2105.01601)
model, by Ilya Tolstikhin et al., for image classification, and demonstrates it
on the CIFAR-100 dataset. The MLP-Mixer is an architecture based exclusively on
multi-layer perceptrons (MLPs), and contains two types of MLP layers:

1. One applied independently to image patches, which mixes the per-location features.
2. The other applied across patches (across channels), which mixes spatial information.

This example requires TensorFlow 2.4 or higher, as well as
[TensorFlow Addons](https://www.tensorflow.org/addons/overview),
which can be installed using the following command:

```shell
pip install -U tensorflow-addons
```
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

"""
## Prepare the data
"""

num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

"""
## Configure the hyperparameters
"""

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 64
num_epochs = 50
dropout_rate = 0.1
image_size = 64  # We'll resize input images to this size.
patch_size = 8  # Size of the patches to be extracted from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
hidden_units = 256  # Number of hidden units.
num_mixers = 2  # Number of mixer blocks.

print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")
print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")

"""
## Use data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.Resizing(image_size, image_size),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

"""
## Implement the Multi-layer Perceptron (MLP) module
"""


def create_mlp(hidden_units, dropout_rate):
    mlp = keras.Sequential(
        [
            layers.Dense(units=hidden_units, activation=keras.activations.gelu),
            layers.Dense(units=hidden_units),
            layers.Dropout(rate=dropout_rate),
        ]
    )
    return mlp


"""
## Implement patch creation as a layer
"""


class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])
        return patches


"""
## Implement the patch encoding layer

The `PatchEncoder` layer will linearly transform a patch by projecting it into
a vector of size `hidden_units`. Note that no positional encoding is used.
"""


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, hidden_units):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=hidden_units)

    def call(self, patches):
        encoded = self.projection(patches)
        return encoded


"""
## Implement the MLP-Mixer module
"""


class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate=0.2, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)

        self.mlp1 = create_mlp(num_patches, dropout_rate)
        self.mlp2 = create_mlp(hidden_units, dropout_rate)
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_transposed = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        x_channels = tf.unstack(x_transposed, axis=1)
        mlp1_outputs = []
        for channel in x_channels:
            mlp1_outputs.append(self.mlp1(channel))
        mlp1_outputs = tf.stack(mlp1_outputs, axis=1)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        patches = tf.unstack(x, axis=1)
        mlp2_outputs = []
        for patch in patches:
            mlp2_outputs.append(self.mlp2(patch))
        mlp2_outputs = tf.stack(mlp2_outputs, axis=1)
        # Add skip connection.
        x = x + mlp2_outputs
        return x


"""
## Build the classification model
"""


def create_mlpmixer_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size, num_patches)(augmented)
    # Encode patches to generate a [batch_size, num_patches, hidden_units] tensor.
    x = PatchEncoder(num_patches, hidden_units)(patches)
    # Create multiple blocks of the MLP-Mixer module.
    for _ in range(num_mixers):
        x = MLPMixerLayer(num_patches, hidden_units, dropout_rate)(x)
    # Apply global average pooling to generate a [batch_size, hidden_units] representation tensor.
    representation = layers.GlobalAveragePooling1D()(x)
    # Compute logits outputs.
    logits = layers.Dense(num_classes)(representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


"""
## Compile, train, and evaluate the mode
"""


def run_experiment(model):

    # Create Adam optimizer with weight decay.
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay,
    )
    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )
    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[early_stopping],
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Return history to plot learning curves.
    return history


"""
Note that training the perceiver model with the current settings on a V100 GPUs
takes around 120 seconds.
"""

classifier = create_mlpmixer_classifier()

history = run_experiment(classifier)

"""
As mentioned in the [MLP-Mixer](https://arxiv.org/abs/2105.01601) paper,
when pre-trained on large datasets, or with modern regularization schemes,
the MLP-Mixer attains competitive scores to state-of-the-art models.
You can obtain better results by increasing the hidden units,
increasing, increasing the number of mixer blocks, and training the model for longer.
You may also try to increase the size of the input images and use different patch sizes.
"""

"""
Title: Image Classification with Modern MLP models
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/05/30
Last modified: 2021/05/30
Description: Implementing the MLP-Mixer gMLP models for CIFAR-100 image classification.
"""

"""
## Introduction

This example implements the two modern attention-free, all-multi-layer perceptrons (MLP) models for image
classification, demonstrated on CIFAR-100 dataset:

1. [MLP-Mixer](https://arxiv.org/abs/2105.01601) model, by Ilya Tolstikhin et al., based on two types of MLPs.
2. [gMLP](https://arxiv.org/abs/2105.08050) model, by Hanxiao Liu et al., based on MLP with gating.

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

learning_rate = 0.005
weight_decay = 0.0001
batch_size = 128
num_epochs = 50
dropout_rate = 0.2
image_size = 64  # We'll resize input images to this size.
patch_size = 8  # Size of the patches to be extracted from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
hidden_units = 256  # Number of hidden units.
num_mixers = 4  # Number of mixer blocks.

print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")
print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")

"""
## Define the experiment

We implement a method to compile, train, and evaluate a given model.
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
    # Create a learning rate scheduler callback.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
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
        callbacks=[early_stopping, reduce_lr],
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Return history to plot learning curves.
    return history


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
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches


"""
## Implement the MLP-Mixer model

The MLP-Mixer is an architecture based exclusively on
multi-layer perceptrons (MLPs), and contains two types of MLP layers:

1. One applied independently to image patches, which mixes the per-location features.
2. The other applied across patches (along channels), which mixes spatial information.

This is similar to a [depthwise separable convolution layer](https://arxiv.org/pdf/1610.02357.pdf)
of the Xception model, but with two chained dense transforms, no max pooling, and layer normalization
instead of batch normalization.
"""

"""
### Implement the MLP-Mixer module
"""


class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate=0.2, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=embedding_dims),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x


"""
### Build the MLP-Mixer classification model
"""


def create_mlpmixer_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size, num_patches)(augmented)
    # Encode patches to generate a [batch_size, num_patches, hidden_units] tensor.
    x = layers.Dense(units=hidden_units)(patches)
    # Create multiple blocks of the MLP-Mixer module.
    for _ in range(num_mixers):
        x = MLPMixerLayer(num_patches, hidden_units, dropout_rate)(x)
    # Apply global average pooling to generate a [batch_size, hidden_units] representation tensor.
    representation = layers.GlobalAveragePooling1D()(x)
    # Apply dropout.
    representation = layers.Dropout(rate=dropout_rate)(representation)
    # Compute logits outputs.
    logits = layers.Dense(num_classes)(representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


"""
### Train and evaluate the MLP-Mixer model

Note that training the model with the current settings on a V100 GPUs
takes around 10 seconds per epoch.
"""

classifier = create_mlpmixer_classifier()
history = run_experiment(classifier)

"""
The MLP-Mixer model achieves around 52% accuracy and 81% top-5 accuracy on the test data.
However, the MLP-Mixer model tends to have much less number of parameters compared
to convolutional and transformer-based models, which leads to less training and
serving computational cost.

As mentioned in the [MLP-Mixer](https://arxiv.org/abs/2105.01601) paper,
when pre-trained on large datasets, or with modern regularization schemes,
the MLP-Mixer attains competitive scores to state-of-the-art models.
You can obtain better results by increasing the embedding dimensions,
increasing, increasing the number of mixer blocks, and training the model for longer.
You may also try to increase the size of the input images and use different patch sizes.
"""

"""
## Implement the MLP-Mixer model

The gMLP as an MLP architecture with Spatial Gating Unit (SGU).
The SGU  enables cross-patch interactions across the spatial (channel) dimension by:

1. Transform the input spatially by applying linear projection across patches (along channels).
2. Applying element-wise multiplication of the input and its spatial transformation.
"""

"""
### Implement the gMLP module
"""


class gMLPLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dims, dropout_rate=0.2, *args, **kwargs):
        super(gMLPLayer, self).__init__(*args, **kwargs)

        self.channel_projection1 = keras.Sequential(
            [
                layers.Dense(units=embedding_dims * 2),
                tfa.layers.GELU(),
                layers.Dropout(rate=dropout_rate),
            ]
        )

        self.channel_projection2 = layers.Dense(units=embedding_dims)

        self.spatial_projection = layers.Dense(
            units=num_patches, bias_initializer="Ones"
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def spatial_gating_unit(self, x):
        # Split x along the channel dimensions.
        # Tensors u and v will in th shape of [batch_size, num_patchs, embedding_dim].
        u, v = tf.split(x, num_or_size_splits=2, axis=2)
        # Apply layer normalization.
        v = self.normalize2(v)
        # Apply spatial projection.
        v_channels = tf.linalg.matrix_transpose(v)
        v_projected = self.spatial_projection(v_channels)
        v_projected = tf.linalg.matrix_transpose(v_projected)
        # Apply element-wise multiplication.
        return u * v_projected

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize1(inputs)
        # Apply the first channel projection. x_projected shape: [batch_size, num_patches, embedding_dim * 2].
        x_projected = self.channel_projection1(x)
        # Apply the spatial gating unit. x_spatial shape: [batch_size, num_patches, embedding_dim].
        x_spatial = self.spatial_gating_unit(x_projected)
        # Apply the second channel projection. x_projected shape: [batch_size, num_patches, embedding_dim].
        x_projected = self.channel_projection2(x_spatial)
        # Add skip connection.
        return x + x_projected


"""
### Build the gMLP classification model
"""


def create_gmlp_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size, num_patches)(augmented)
    # Encode patches to generate a [batch_size, num_patches, embedding_dim] tensor.
    x = layers.Dense(units=embedding_dim)(patches)
    # Create multiple blocks of the gMLP module.
    for _ in range(num_blocks):
        x = gMLPLayer(embedding_dim, dropout_rate)(x)
    # Apply global average pooling to generate a [batch_size, embedding_dim] representation tensor.
    representation = layers.GlobalAveragePooling1D()(x)
    # Apply dropout.
    representation = layers.Dropout(rate=dropout_rate)(representation)
    # Compute logits outputs.
    logits = layers.Dense(num_classes)(representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


"""
### Train and evaluate the gMLP model

Note that training the model with the current settings on a V100 GPUs
takes around 20 seconds per epoch.
"""

classifier = create_gmlp_classifier()
history = run_experiment(classifier)

"""
The gMLP model achieves around 51% accuracy and 80% top-5 accuracy on the test data.

As shown in the [gMLP](https://arxiv.org/abs/2105.08050) paper,
better results can be achieved by increasing the embedding dimensions,
increasing, increasing the number of gMLP blocks, and training the model for longer.
You may also try to increase the size of the input images and use different patch sizes.
Note that, the paper used advanced regularization strategies, such as Mixup and CutMix,
as well as AutoAugment.
"""

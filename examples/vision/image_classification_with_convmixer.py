"""
Title: Image classification with ConvMixer
Author: [ZhiYong Chang](https://github.com/czy00000)
Date created: 2021/10/14
Last modified: 2021/10/14
Description: Implementing the ConvMixer model for image classification.
"""

"""
## Introduction

This example implements the [ConvMixer](https://openreview.net/pdf?id=TVHS5Y4dNvM)
model for image classification, and demonstrates it on the CIFAR-10 dataset.
The ConvMixer model is an extremely simple model
that is similar in spirit to the [ViT](https://arxiv.org/abs/2010.11929)
and the even-more-basic [MLP-Mixer]( https://arxiv.org/abs/2105.01601) in that it
operates directly on patches as input, separates the mixing of spatial and channel
dimensions, and maintains equal size and resolution throughout the network.
It consists of a patch embedding layer followed by
repeated applications of a simple fully-convolutional block.

This example requires TensorFlow 2.5 or higher, as well as
[TensorFlow Addons](https://www.tensorflow.org/addons/overview),
which can be installed using the following command:

```python
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
import matplotlib.pyplot as plt

"""
## Prepare the data
"""

num_classes = 10
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

"""
## Configure the hyperparameters
"""

weight_decay = 0.0001
learning_rate = 0.1
label_smoothing = 0.1
validation_split = 0.2
batch_size = 256
num_epochs = 5
patch_size = 2  # Size of the patches to be extracted from the input images.
num_patches = (input_shape[0] // patch_size) ** 2  # Number of patch
embedding_dim = 32  # Number of hidden units.
depth = 3  # Number of repetitions of the ConvMixer layer
kernel_size = 9  # Kernel size of the depthwise convolutional layer

print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")


"""
## Use data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.1),
        layers.RandomContrast(factor=0.1),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

"""
## the ConvMixer model

The ConvMixer model is based on the idea of mixing, In particular, the authors
chose depthwise convolution to mix spatial locations and pointwise convolution
to mix channel locations. A key idea from previous work is that MLPs
and self-attention can mix distant spatial locations, i.e., they can have an
arbitrarily large receptive field. Consequently, the authors used convolutions
with an unusually large kernel size to mix distant spatial locations.

It is worth noting that in the original paper, the author did not use
the position embedding, We think that it may be the ConvMixer uses the
standard convolutional layer and learn the position information.The authors
demonstrate that `patch embeddings` allow all the downsampling to happen
at once, immediately decreasing the internal resolution and thus increasing
the effective receptive field size, making it easier to mix distant spatial
information.We apply `layers.DepthwiseConv2D` to implenment depthwise
convolution and `layers.Conv2D` to implements pointwise convolution.


"""


def get_ConvMixer(dim, depth, kernel_size, patch_size, num_classes):
    inputs = layers.Input(shape=input_shape)
    # Image daugmentation
    x = data_augmentation(inputs)
    # Create patch embedding layer
    x = layers.Conv2D(dim, kernel_size=patch_size, strides=patch_size)(inputs)
    x = layers.Activation("gelu")(x)
    x = layers.BatchNormalization()(x)
    # Set aside residual
    previous_block_activaiton = x
    # Create multiple layers of the ConvMixer block
    for i in range(depth):
        # Apply depthwise convolution
        x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
        # Apply GeLU activation
        x = layers.Activation("gelu")(x)
        # Apply Batch Normalization
        x = layers.BatchNormalization()(x)
        # Add skip connection
        x = layers.add([x, previous_block_activaiton])
        # Apply pointwise convolution
        x = layers.Conv2D(dim, kernel_size=1, padding="same")(x)
        # Apply GeLU activation
        x = layers.Activation("gelu")(x)
        # Apply Batch Normalization
        x = layers.BatchNormalization()(x)
        # Set aside next residual
        previous_block_activaiton = x
    # Apply GlobalAveragePool
    x = layers.GlobalAveragePooling2D()(x)
    # Get the outputs
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    # Create the model
    model = keras.Model(inputs, outputs)
    return model


"""
## Train on CIFAR-10

"""

model = get_ConvMixer(embedding_dim, depth, kernel_size, patch_size, num_classes)

model.compile(
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_split=validation_split,
)

"""
Let's visualize the training progress of the model.


"""

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

"""
Let's display the final results of the training on CIFAR-10.



"""

loss, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {round(loss, 2)}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

"""
The ConvMixer model we just trained has just 12K parameters, and it gets us to
~90% test top-5 accuracy within just 5 epochs.This performance can further be
improved by additional techniques like cosine decay learning rate schedule,
other data augmentation techniques like RandAugment, MixUp or Cutmix.In fact,
we train the ConvMixer model in CPU. We do not design our experiments to maximize
the accuracy.In the [paper](https://openreview.net/pdf?id=TVHS5Y4dNvM), the authors
provided evidence that the increasingly common "isotropic" architecture with a
simple patch embedding stem is itself a powerful template for deep learning. The
authors also present a number of experiments to study how the depth of model,
the size of patch, the kernel size of depthwise convolution , the width of model etc.
affect the final performance of ConvMixer.

This example takes inspiration from the official
[PyTorch](https://github.com/tmp-iclr/convmixer)
implementations.
"""

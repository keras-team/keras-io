"""
Title: Convolutional Autoencoder For Image Denoising
Author: Santiago L. Valdarrama
Date created: 2021/03/01
Last modified: 2021/03/01
Description: How to train a deep convolutional autoencoder for image denoising.
"""

"""
## Introduction

This example demonstrates how to implementat of a deep convolutional autoencoder
for image denoising—mapping noisy digits images from the MNIST dataset to clean digits
images. This implementation is based on an original blog post titled
[Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
by François Chollet.
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.datasets import mnist


def preprocess(dataset):
    """
    Normalizes the MNIST dataset and reshapes it into the appropriate format.
    """

    dataset = dataset.astype("float32") / 255.0
    dataset = np.reshape(dataset, (len(dataset), 28, 28, 1))
    return dataset


def noise(dataset):
    """
    Adds random noise to each image in the supplied dataset.
    """

    noise_factor = 0.4
    noisy_dataset = dataset + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=dataset.shape
    )

    return np.clip(noisy_dataset, 0.0, 1.0)


def display(dataset1, dataset2):
    """
    Displays ten random images from each one of the supplied datasets.
    """

    n = 10

    indices = np.random.randint(len(dataset1), size=n)
    images1 = dataset1[indices, :]
    images2 = dataset2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


"""
## Prepare de data
"""

# We only need images from the dataset. We will never use the target values so
# we aren't going to load them.
(train_dataset, _), (test_dataset, _) = mnist.load_data()

# Normalize and reshape the data
train_dataset = preprocess(train_dataset)
test_dataset = preprocess(test_dataset)

# Create a copy of the datasets with added noise
noisy_train_dataset = noise(train_dataset)
noisy_test_dataset = noise(test_dataset)

# Display the train dataset and the version with added noise
display(train_dataset, noisy_train_dataset)

"""
## Build the autoencoder

We are going to use the functional API to build our convolutional autoencoder.
"""

input = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

"""
Now we can train our autoencoder using the `train_dataset` as our `X` and the same
dataset as our `y`. Noticed we are setting up the validation data using the same format.
"""

autoencoder.fit(
    x=train_dataset,
    y=train_dataset,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(test_dataset, test_dataset),
)

"""
Let's predict on our test dataset and display the original image together with the
prediction from our autoencoder.

Notice how the predictions are pretty close to the original images, although not quite
the same.
"""

predictions = autoencoder.predict(test_dataset)
display(test_dataset, predictions)

"""
Now that we know that our autoencoder works, let's retrain it using the noisy dataset as
our `X` and the clean dataset as our `y`. We want our autoencoder to learn how to denoise
the images.
"""

autoencoder.fit(
    x=noisy_train_dataset,
    y=train_dataset,
    epochs=100,
    batch_size=128,
    shuffle=True,
    validation_data=(noisy_test_dataset, test_dataset),
)

"""
Let's now predict on the noisy dataset and display the results of our autoencoder.

Notice how the autoencoder does an amazing job at removing the noise from the input
images.
"""

predictions = autoencoder.predict(noisy_test_dataset)
display(noisy_test_dataset, predictions)

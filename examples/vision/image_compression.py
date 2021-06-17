"""
Title: Image Compression using Autoencoders 
Author: [Sayan Nath](https://twitter.com/sayannath2350) 
Date created: 2021/06/17 
Last modified: 2021/06/17 
Description: How to compress images using autoencoders. 
"""


"""
## Introduction
Autoencoders are a deep learning model for transforming data from a high-dimensional space to a lower-dimensional space.  The more accurate the autoencoder, the closer the generated data is to the original. This example will demonstrate how to do image compression using Autoencoders with the help of MNIST dataset.
"""

"""
## Setup
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

"""
## Prepare the dataset

In this example, we will be using the MNIST dataset. But this same recipe can be used for other classification datasets as well.
"""

(x_train_orig, y_train), (x_test_orig, y_test) = mnist.load_data()

x_train_orig = x_train_orig.astype("float32") / 255.0
x_test_orig = x_test_orig.astype("float32") / 255.0

x_train = np.reshape(
    x_train_orig, newshape=(x_train_orig.shape[0], np.prod(x_train_orig.shape[1:]))
)
x_test = np.reshape(
    x_test_orig, newshape=(x_test_orig.shape[0], np.prod(x_test_orig.shape[1:]))
)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

"""
## Define hyperparameters
"""

INPUT_SHAPE = 784
BATCH_SIZE = 256
EPOCHS = 50

"""
## Build the Encoder
"""


def get_encoder_model():
    x = layers.Input(shape=(INPUT_SHAPE), name="encoder_input")

    encoder_dense_layer1 = layers.Dense(units=300, name="encoder_dense_1")(x)
    encoder_activ_layer1 = layers.LeakyReLU(name="encoder_leakyrelu_1")(
        encoder_dense_layer1
    )

    encoder_dense_layer2 = layers.Dense(units=2, name="encoder_dense_2")(
        encoder_activ_layer1
    )
    encoder_output = layers.LeakyReLU(name="encoder_output")(encoder_dense_layer2)

    encoder = Model(x, encoder_output, name="encoder_model")
    return encoder


encoder = get_encoder_model()
encoder.summary()

"""
## Build the Decoder
"""


def get_decoder_model():
    decoder_input = layers.Input(shape=(2), name="decoder_input")

    decoder_dense_layer1 = layers.Dense(units=300, name="decoder_dense_1")(
        decoder_input
    )
    decoder_activ_layer1 = layers.LeakyReLU(name="decoder_leakyrelu_1")(
        decoder_dense_layer1
    )

    decoder_dense_layer2 = layers.Dense(units=INPUT_SHAPE, name="decoder_dense_2")(
        decoder_activ_layer1
    )
    decoder_output = layers.LeakyReLU(name="decoder_output")(decoder_dense_layer2)

    decoder = Model(decoder_input, decoder_output, name="decoder_model")
    return decoder


decoder = get_decoder_model()
decoder.summary()

"""
## Build the Autoencoder
"""


def get_autoencoder_model():
    ae_input = layers.Input(shape=(INPUT_SHAPE), name="autoencoder_input")
    ae_encoder_output = encoder(ae_input)
    ae_decoder_output = decoder(ae_encoder_output)

    ae_model = Model(ae_input, ae_decoder_output, name="autoencoder")
    return ae_model


"""
## Summary of the Autoencoder Model
"""

autoencoder_model = get_autoencoder_model()
autoencoder_model.summary()

"""
## Compile Model
"""

autoencoder_model.compile(loss="mse", optimizer=Adam(learning_rate=0.0005))

"""
## Train Model
"""

autoencoder_model.fit(
    x_train,
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
)

"""
## Making Predictions
"""

encoded_images = encoder.predict(x_train)  # Representing the vectors of training images
decoded_images = decoder.predict(encoded_images)

decoded_images_orig = np.reshape(
    decoded_images, newshape=(decoded_images.shape[0], 28, 28)
)  # The output of the decoder is reshaped as 28x28

"""
## View generated samples
"""

num_images_to_show = 5  # Number of samples to be shown

# Let's preview five original image and compressed image
for im_ind in range(num_images_to_show):
    plt.figure(figsize=(10, 10))
    plot_ind = im_ind * 2 + 1
    rand_ind = np.random.randint(low=0, high=x_train.shape[0])
    plt.subplot(num_images_to_show, 2, plot_ind)
    plt.imshow(x_train_orig[rand_ind, :, :], cmap="gray")
    plt.subplot(num_images_to_show, 2, plot_ind + 1)
    plt.imshow(decoded_images_orig[rand_ind, :, :], cmap="gray")

"""
## Notes

In this example, you can notice that how efficiently our autoencoder compress the images.
"""

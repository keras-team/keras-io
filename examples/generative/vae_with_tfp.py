"""
Title: Variational AutoEncoder with Tensorflow Probability
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/01/15
Last modified: 2021/01/15
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits using [TensorFlow Probability](https://www.tensorflow.org/probability).
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

"""
## Create a prior distribution
"""

# The dimensions of the latent variable z.
latent_dim = 2
# The shape of the input image.
input_shape = (28, 28, 1)
# Create a prior distribution for the latent variable z.
# This will be used in the KL divergence: KL(q(x|z) || p(z)).
prior = tfp.distributions.MultivariateNormalDiag(
    loc=tf.zeros(latent_dim), name="p_of_z"
)

"""
## Define a KL-divergence regularizer
"""

# Create a KL divergence regularizer to be added to the losss.
kl_regularizer = tfp.layers.KLDivergenceRegularizer(prior, use_exact_kl=True, weight=1)

"""
## Build the encoder
"""

encoder_inputs = keras.Input(shape=input_shape)

x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

# The encoder output is the posterior: q(z|x).
num_params = 2 * latent_dim
distribution_params = layers.Dense(num_params)(x)
prob_z_given_x = tfp.layers.DistributionLambda(
    lambda params: tfp.distributions.MultivariateNormalDiag(
        loc=params[..., :latent_dim],  # mean
        scale_diag=tf.math.exp(params[..., latent_dim:]),
    ),  # variance
    activity_regularizer=kl_regularizer,  # KL divergence regualizer is added.
    name="q_z_given_x",
)(distribution_params)

encoder = keras.Model(inputs=encoder_inputs, outputs=prob_z_given_x, name="encoder")
encoder.summary()

"""
## Build the decoder
"""

latent_inputs = keras.Input(shape=(latent_dim,))

x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(1, 3, padding="same")(x)

# The decoder output is the approximated p(x|z).
distribution_params = layers.Flatten()(x)
prob_x_given_z = tfp.layers.IndependentBernoulli(
    event_shape=input_shape, name="p_of_x_given_z"
)(distribution_params)

decoder = keras.Model(inputs=latent_inputs, outputs=prob_x_given_z, name="decoder")
decoder.summary()

"""
## Build and train a VAE model
"""

vae = keras.Model(inputs=encoder.input, outputs=decoder(encoder.output))
# Compute the reconstruction loss, as the KL divergence is added as a regularizer.
# The reconstruction loss is the negative log likelihood of x given the approximated p(x|z).
loss = lambda x, prob_x_given_z: -prob_x_given_z.log_prob(x)
vae.compile(loss=loss, optimizer=keras.optimizers.Adam())

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
# The input x and the target y are the same.
vae.fit(x=mnist_digits, y=mnist_digits, epochs=30, batch_size=128)

"""
## Display a grid of sampled digits
"""

import matplotlib.pyplot as plt


def plot_latent_space(n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent_space()

"""
## Display how the latent space clusters different digit classes
"""


def plot_label_clusters(data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(x_train, y_train)

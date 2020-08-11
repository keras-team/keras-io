"""
Title: Density estimation using Real NVP
Authors: [Mandolini Giorgio Maria](https://www.linkedin.com/in/giorgio-maria-mandolini-a2a1b71b4/), [Sanna Daniele](https://www.linkedin.com/in/daniele-sanna-338629bb/), [Zannini Quirini Giorgio](https://www.linkedin.com/in/giorgio-zannini-quirini-16ab181a0/)
Date created: 2020/08/10
Last modified: 2020/08/10
Description: Estimating the density distribution of multi-cluster datasets.
"""

"""
## Introduction

The aim of this work is to map a simple distribution - which is easy to sample
and whose density is simple to estimate - to a more complex one learned from the data.
This kind of generative model is also known as "normalizing flow".

In order to do this, the model is trained via the maximum
likelihood principle, using the "change of variable formula".

We will use an affine coupling function. We create it such that its inverse, as well as
the determinant of the Jacobian, are easy to obtain (more details in the referenced paper).

**Requirements:**

* Tensorflow 2.3 or higher
* Tensorflow probability 0.11.0 or higher

**Reference:**

[Density estimation using Real NVP](https://arxiv.org/pdf/1605.08803.pdf)
"""

"""
## Setup

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

"""
## Load the data
"""

# Generating the examples

url = "http://cs.joensuu.fi/sipu/datasets/s1.txt"
path = tf.keras.utils.get_file("dataset", url)
data = np.loadtxt(path)
scaler = StandardScaler()
data = scaler.fit_transform(data)

"""
## Affine coupling layer
"""

# Creating a custom layer with keras API


nweights = 256
reg = 0.01


def Coupling(input_shape):
    input = keras.layers.Input(shape=input_shape)

    t_layer_1 = keras.layers.Dense(
        nweights, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    t_layer_2 = keras.layers.Dense(
        nweights, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = keras.layers.Dense(
        nweights, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = keras.layers.Dense(
        nweights, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = keras.layers.Dense(
        2, activation="linear", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = keras.layers.Dense(
        nweights, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    s_layer_2 = keras.layers.Dense(
        nweights, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = keras.layers.Dense(
        nweights, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = keras.layers.Dense(
        nweights, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = keras.layers.Dense(
        2, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_4)
    return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])


Coupling(2).summary()
"""
## Real NVP
"""
# class containing the whole set of operations
class Realnvp(keras.Model):
    def __init__(self, layers_list, masks, distr):
        super(Realnvp, self).__init__()

        # number of coupling layers
        self.num_cl = len(masks)
        # distribution of the latent space
        self.distr = distr
        # masks to divide into 1:d and d+1:D
        self.masks = masks
        # s and t custom layers of before
        self.layers_list = layers_list

    # custom function defining the forward operation
    def forward(self, y):
        # log determinant of the forward pass
        log_det_for = tf.zeros(y.shape[0])
        x = y
        for i in range(self.num_cl):
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            x = x_masked + reversed_mask * (x * tf.exp(s) + t)
            log_det_for += tf.reduce_sum(s, [1])

        return x, log_det_for

    # custom function defining the forward operation
    def inverse(self, x):
        # log determinant of the forward pass
        log_det_inv = tf.zeros(x.shape[0])
        y = x
        for i in reversed(range(self.num_cl)):
            y_masked = y * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](y_masked)
            s *= reversed_mask
            t *= reversed_mask
            y = reversed_mask * (y - t) * tf.exp(-s) + y_masked
            log_det_inv -= tf.reduce_sum(s, [1])

        return y, log_det_inv

    # log likelihood of the normal distribution + the log determinant of the jacobian
    def log_likelihood(self, x):
        y, logdet = self.inverse(x)
        return self.distr.log_prob(y) + logdet

    def log_loss(self, x):
        return -tf.reduce_mean(self.log_likelihood(x))

    def train_step(self, data):

        with tf.GradientTape() as tape:
            loss = self.log_loss(data)

        # updating the variables
        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}


"""
## Initialization
"""

# initializing the masks, the layers and the prior distribution for the real nvp
num_cl = 6
masks = np.array([[0, 1], [1, 0]] * (num_cl // 2), dtype="float32")

layers_list = [Coupling(2) for i in range(num_cl)]

tfd = tfp.distributions
distr = tfd.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[1.0, 1.0])

"""
## Model training
"""
loss_tracker = keras.metrics.Mean(name="loss")

# Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model = Realnvp(layers_list, masks, distr)

model.compile(optimizer=optimizer)


class Loss_every_100_epochs(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 500 == 0:
            print(epoch, "loss", round(logs.get("loss"), 4))


# Training model
history = model.fit(
    data,
    batch_size=len(data),
    epochs=7500,
    callbacks=[Loss_every_100_epochs()],
    verbose=0,
)
"""
## Performance evaluation
"""

plt.figure(figsize=(15, 10))
plt.plot(history.history["loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")

# from data to latent space
z, _ = model.inverse(data)

# from latent space to data
samples = distr.sample(3000)
x, _ = model.forward(samples)

f, axes = plt.subplots(2, 2)
f.set_size_inches(20, 15)

axes[0, 0].scatter(data[:, 0], data[:, 1], color="r")
axes[0, 0].set(title="Inference data space X", xlabel="x", ylabel="y")
axes[0, 0].set_xlim([-2.3, 2])
axes[0, 0].set_ylim([-2, 2.3])
axes[0, 1].scatter(z[:, 0], z[:, 1], color="r")
axes[0, 1].set(title="Inference latent space Z", xlabel="x", ylabel="y")
axes[0, 1].set_xlim([-3.5, 4])
axes[0, 1].set_ylim([-4, 4])
axes[1, 0].scatter(samples[:, 0], samples[:, 1], color="g")
axes[1, 0].set(title="Generated latent space Z", xlabel="x", ylabel="y")
axes[1, 0].set_xlim([-3.5, 4])
axes[1, 0].set_ylim([-4, 4])
axes[1, 1].scatter(x[:, 0], x[:, 1], color="g")
axes[1, 1].set(title="Generated data space X", label="x", ylabel="y")
axes[1, 1].set_xlim([-2.3, 2])
axes[1, 1].set_ylim([-2, 2.3])

"""
Title: A Quasi-SVM in Keras
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/17
Last modified: 2020/04/17
Description: Demonstration of how to train a Keras model that approximates a SVM.
"""
"""
## Introduction

This example demonstrates how to train a Keras model that approximates a Support Vector
 Machine (SVM).

The key idea is to stack a `RandomFourierFeatures` layer with a linear layer.

The `RandomFourierFeatures` layer can be used to "kernelize" linear models by applying
 a non-linear transformation to the input
features and then training a linear model on top of the transformed features. Depending
on the loss function of the linear model, the composition of this layer and the linear
model results to models that are equivalent (up to approximation) to kernel SVMs (for
hinge loss), kernel logistic regression (for logistic loss), kernel linear regression
 (for MSE loss), etc.

In our case, we approximate SVM using a hinge loss.
"""

"""
## Setup
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=10),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

"""
## Prepare the data
"""

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data by flattening & scaling it
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255

# Categorical (one hot) encoding of the labels
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

"""
## Train the model
"""

model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

"""
I can't say that it works well or that it is indeed a good idea, but you can probably
 get decent results by tuning your hyperparameters.

You can use this setup to add a "SVM layer" on top of a deep learning model, and train
 the whole model end-to-end.
"""

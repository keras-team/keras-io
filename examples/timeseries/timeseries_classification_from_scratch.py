"""
Title: Time series classification from scratch
Author: [hfawaz](https://github.com/hfawaz/)
Date created: 2020/07/21
Last modified: 2020/08/21
Description: Training a time series classifier from scratch on the FordA dataset from
the UCR/UEA archive.
"""
"""
## Introduction

This example shows how to do time series classification from scratch, starting from raw
CSV time series files on disk. We demonstrate the workflow on the FordA dataset from the 
[UCR/UEA archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

"""

"""
## Setup

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

"""
## Load the data: the FordA dataset

### Description
The dataset we are using here is called FordA.
The data comes from the UCR archive.
The dataset contains 3601 training instances and another 1320 testing instances.
Each time series corresponds to a measurement of engine noise captured by a motor sensor.
For this task, the goal is to automatically determine whether a certain symptom
exists or not in an automotive system. The problem is a balanced binary classification
task. The full description of
this dataset can be found [here](http://www.j-wichard.de/publications/FordPaper.pdf).

### Read the TSV data online
We will use the `FordA_TRAIN` file for training and the
`FordA_TEST` file for testing. The simplicity of this dataset
allows us to demonstrate effectively how to use ConvNets for time series classification.
In this file, the first column corresponds to the label.
"""


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

"""
## Visualize the data
Here we visualize one time series example for each class in the dataset.

"""

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()

"""
## Standardizing the data

Our time series are already in a single length (176). However, their values are
usually in various ranges. This is not ideal for a neural network;
in general we should seek to make the input values normalized. Here, we will
standardize values to have a mean equal to zero and a standard deviation equal to one.
We perform this step for each train and test set.
"""

std_ = x_train.std(axis=1, keepdims=True)
std_[std_ == 0] = 1.0
x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

std_ = x_test.std(axis=1, keepdims=True)
std_[std_ == 0] = 1.0
x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

"""
Note that the time series data used here are univariate, meaning we only have one channel
per time series example.
We will therefore transform the time series into a multivariate one with one channel
using a simple reshaping via numpy.
This will allow us to construct a model that is easily applicable to multivariate time
series.
"""

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

"""
Finally, in order to use `sparse_categorical_crossentropy`, we will have to count
the number of classes beforehand.
"""

nb_classes = len(np.unique(y_train))

"""
Now we shuffle the training set because we will be using the `validation_split` option
later when training.
"""

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

"""
Standardize the labels to positive integers.
"""

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

"""
## Build a model

We build a Fully Convolutional Neural Network originally proposed in
[this paper](https://arxiv.org/abs/1611.06455).
The implementation is based on the TF2.0 version provided
[here](https://github.com/hfawaz/dl-4-tsc/).

"""


def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    # This is the first 1D convolutional block.
    # First a linear convolution is applied by sliding a certain number of filters
    # on the input time series to transform it into a multivariate time series,
    # whose number of dimensions is equal to the number of filters used here.
    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding="same")(input_layer)
    # Following the convolution, we apply a batchnormalization in order to
    # help the network converge quickly while adding an implicit regularization.
    conv1 = keras.layers.BatchNormalization()(conv1)
    # Finally we add the relu activation function to inject some non-linearity
    # to the convolutions' output.
    conv1 = keras.layers.Activation(activation="relu")(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation("relu")(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation("relu")(conv3)

    # A global average pooling is used here.
    # We average the time series over the whole time dimension.
    # This would reduce drastically the number of parameters,
    # while enabling the use of the class activation map method for interpretability.
    # In fact the input time series with m dimensions is averaged
    # resulting in a vector of m dimensions.
    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    # The final softmax traditional classifier is used,
    # with a number of neurons equal to the number of classes.
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    # We link the input and output layer by constructing the keras model.
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:], num_classes=nb_classes)
keras.utils.plot_model(model, show_shapes=True)

"""
## Train the model

"""

epochs = 100
batch_size = 128

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
]
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

"""
## Evaluate model on test data
"""

model = keras.models.load_model("best_model.h5")

y_pred = model.predict(x_test).argmax(axis=1)

print("Test accuracy", accuracy_score(y_pred, y_test))

"""
## Plot the model's training and validation loss
"""

metric = "accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()

"""
We can see how the training accuracy increases and reaches almost 0.9 after 10 epochs. 
Nevertheless, by observing the validation accuracy we can see how the network still needs
training until it reaches almost 0.9 for both the validation and the training accuracy 
after 100 epochs. 
"""

"""
Title: Image Classification with AlexNet
Author: [Sayan Nath](https://twitter.com/sayannath2350)
Date created: 2021/09/30
Last modified: 2021/09/30
Description: Image Classification with AlexNet using Keras on CIFAR-10 dataset
"""
"""
## Introduction

In 2012, Alex Krizhevesky and others proposed a deeper and wider CNN model compared to
LeNet and won the most difficult ImageNet challenge for visual object recognition called
the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. AlexNet achieved
state-of-the-art recognition accuracy against all the traditional machine learning and
computer vision approaches. It was a significant breakthrough in the field of machine
learning and computer vision for visual recognition and classification tasks and is the
point in history where interest in deep learning increased rapidly.
"""

"""
## Setup
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

SEEDS = 42

np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)

"""
## Load the CIFAR-10 dataset
In this example, we will use the [CIFAR-10 image classification
dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Class name
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

"""
## Define hyperparameters
"""

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 64
IMAGE_SIZE = 227  # image size expected by alexnet model
NUM_CLASSES = 10
EPOCHS = 50
LEARNING_RATE = 0.001

"""
## Define the image preprocessing function
"""


def preprocess_image(image, label):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image, label


"""
## Convert the data into TensorFlow `Dataset` objects
"""

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

"""
## Define the data pipeline
"""

# Training pipeline
pipeline_train = (
    train_ds.map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Validation pipeline
pipeline_validation = (
    validation_ds.map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

"""
## Visualise the training samples
"""

# Let's preview 9 samples from the dataset
image_batch, label_batch = next(iter(pipeline_train))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i])
    plt.axis("off")

"""
## Model building
"""

"""
The first convolutional layer performs convolution and max pooling with Local Response
Normalization (LRN) where 96 different receptive filters are used that are 11×11 in size.


Two new concepts, Local Response Normalization (LRN) and dropout, are introduced in this
network. LRN can be applied in two different ways: first applying on single channel or
feature maps, where an N×N patch is selected from same feature map and normalized based
one the neighborhood values. Second, LRN can be applied across the channels or feature
maps (neighborhood along the third dimension but a single pixel or location).

![](https://i.imgur.com/EQI3DQn.png)
"""


def get_training_model():
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                filters=96,
                kernel_size=(11, 11),
                strides=(4, 4),
                activation="relu",
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(
                filters=256,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    return model


model = get_training_model()
model.summary()

"""
## Compile the model
"""

optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

"""
## Train the model
"""

history = model.fit(
    pipeline_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=pipeline_validation,
)

"""
## Plot the training and validation metrics
"""


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Training Progress")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["train_acc", "val_acc"], loc="lower right")
    plt.show()


plot_hist(history)

"""
## Evaluate the model
"""

accuracy = model.evaluate(pipeline_validation)[1] * 100
print("Accuracy: {:.2f}%".format(accuracy))

"""
## Conclusion:

In this experiment, we used CIFAR-10 but you can use any other dataset to get the
state-of-the-art results. You also fine-tune the model to maximise the level of the
accuracy. You can read the [original
paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Pap
er.pdf) to get the reference.
"""

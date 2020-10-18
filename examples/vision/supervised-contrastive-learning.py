"""
Title: Supervised Contrastive Learning
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2020/11/01
Last modified: 2020/11/01
Description: Using supervised contrastive learning for image classification.
"""

"""
## Introduction

[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
(Prannay Khosla et al.) is a training methodology that outperforms cross-entropy
on supervised learning tasks.

Essentially, training an image classification model with Supervised Contrastive Learning
is peformed in two phases:

  1. Pre-training an encoder to generate feature vectors for input images such that feature
    vectors of images in the same class will be more similar compared feature vectors of
    images in other classes.
  2. Training a classifier on top of the freezed encoder.

"""

"""
## Setup
"""

"""shell
!pip install tensorflow-addons
"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

"""
## Prepare the data
"""

num_classes = 10
input_shape = (32, 32, 3)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train, y_test = tf.squeeze(y_train), tf.squeeze(y_test)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

"""
## Build the encoder model

The encoder model takes the image as an input and produce a 128-dimension feature vector.
"""


def create_encoder():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
        ]
    )


encoder = create_encoder()
encoder.summary()

BATCH_SIZE = 256
NUM_EPOCHS = 50
DROPOUT = 0.5
TEMPERATURE = 0.05

"""
## Build the classification model

The classification model adds a fully-connected layer on top of the encoder, plus a
softmax layer with the target classes.
"""


def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(DROPOUT)(features)
    features = tf.keras.layers.Dense(64)(features)
    features = tf.keras.layers.Dropout(DROPOUT)(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
    )
    return model


"""
## Experiment 1: Train the baseline classification model

In this experiment, a baseline classifier is trained normally, i.e., the encoder and the
classifier parts are trained together as a single model to minimize cross-entropy loss.
"""

encoder = create_encoder()
classifier = create_classifier(encoder)
classifier.summary()

history = classifier.fit(
    x=x_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(x_test, y_test),
    verbose=0,
)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy*100, 2)}%")

"""
We get to ~70.1% validation accuracy.
"""

"""
## Experiment 2: Use supervised contrastive learning
"""

"""
### 1. Supervised contrastive learning loss function
"""


def make_supervised_contrastive_loss_fn(temperature=1):
    def supervised_contrastive_loss(labels, feature_vectors):

        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)

        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            temperature,
        )

        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

    return supervised_contrastive_loss


"""
### 2. Pretrain the encoder
"""

encoder = create_encoder()
encoder.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=make_supervised_contrastive_loss_fn(temperature=TEMPERATURE),
)

history = encoder.fit(
    x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=0
)

"""
### 3. Train the classifier with the freezed encoder
"""

classifier = create_classifier(encoder, trainable=False)
history = classifier.fit(
    x=x_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(x_test, y_test),
    verbose=0,
)

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy*100, 2)}%")

"""
We get to ~72.6% validation accuracy.
"""

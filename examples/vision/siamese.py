"""
Title: Siamese Network Using Contrastive Loss
Author: [Santiago L. Valdarrama](https://twitter.com/svpino)
Date created: 2021/03/17
Last modified: 2021/03/17
Description: How to train a Siamese network with contrastive loss.
"""

"""
## Introduction

This example demonstrates how to implement a Siamese network using contrastive loss
to determine whether a pair of images is similar. Contrastive Loss is a metric-learning
loss function introduced by Yann Le Cunn et al. in the paper "Dimensionality Reduction
by Learning an Invariant Mapping," 2005.
"""

"""
## Setup
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array


def display_pairs(images, labels, correct=None):
    """
    Displays the first ten pairs from the supplied array.

    Args:
        - images: An array containing the pair of images.
        - labels: An array containing the corresponding label (0 if both
            pairs are different, and 1 if both pairs are the same.)
        - correct (optional): An array of boolean values indicating whether
            the supplied labels correctly represent the image pairs.
    """

    n = 10

    plt.figure(figsize=(20, 6))
    for i, (image1, image2) in enumerate(zip(images[:n, 0], images[:n, 1])):
        label = int(labels[:n][i][0])

        text = "Label"
        color = "silver"

        # If we know whether the supplied labels are correct, let's change the
        # text and the face color of the annotation on the chart.
        if correct is not None:
            text = "Prediction"
            color = "mediumseagreen" if correct[:n][i][0] else "indianred"

        ax = plt.subplot(3, n, i + 1)
        ax.text(
            1,
            -3,
            f"{text}: {label}",
            style="italic",
            bbox={"facecolor": color, "pad": 4},
        )

        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def plot_history(history):
    """
    Plots the training and validation loss.
    """

    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Training and Validation Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()


def generate_pairs(images, labels):
    """
    Creates a collection of positive and negative image pairs from the supplied
    array of images.
    """

    x_pairs = []
    y_pairs = []

    for i in range(len(images)):
        label = labels[i]

        j = np.random.choice(np.where(labels == label)[0])
        x_pairs.append([images[i], images[j]])
        y_pairs.append([1])

        k = np.random.choice(np.where(labels != label)[0])
        x_pairs.append([images[i], images[k]])
        y_pairs.append([0])

    indices = np.arange(len(x_pairs))
    np.random.shuffle(indices)

    return np.array(x_pairs)[indices], np.array(y_pairs)[indices]


"""
## Prepare the data

We are going to split the data into three sets: train, validation, and test. Since
`mnist.load_data()` already gives us train and test sets, we will get 20% of the
train data and use it for validation.
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

VALIDATION_SIZE = int(len(x_train) * 0.2)

x_val = x_train[:VALIDATION_SIZE]
y_val = y_train[:VALIDATION_SIZE]

x_train = x_train[VALIDATION_SIZE:]
y_train = y_train[VALIDATION_SIZE:]

x_train = preprocess(x_train)
x_val = preprocess(x_val)
x_test = preprocess(x_test)

print(f"Train: {len(x_train)}")
print(f"Validation: {len(x_val)}")
print(f"Test: {len(x_test)}")

"""
Let's now generate positive and negative pairs to train our siamese network.

A positive pair consists of two images representing the same digit, while a negative pair
consists of two images representing different digits.
"""

x_pairs_train, y_pairs_train = generate_pairs(x_train, y_train)
x_pairs_val, y_pairs_val = generate_pairs(x_val, y_val)
x_pairs_test, y_pairs_test = generate_pairs(x_test, y_test)

"""
Let's display the first 10 pairs from our train set and the label assigned to each of
them.
"""

display_pairs(x_pairs_train, y_pairs_train)


def norm(features):
    """
    Computes the euclidean norm of the two feature vectors generated
    by the twins of the Siamese network.
    """
    return tf.norm(features[0] - features[1], axis=1, keepdims=True)


def accuracy(y_true, y_pred):
    """
    Computes the accuracy of the predictions.
    """

    # Notice that `y_true` is 0 whenever two images are not the same and 1
    # otherwise, but `y_pred` is the opposite. The closer `y_pred` is to 0,
    # the shorter the distance between both images, therefore the more likely
    # it is that they are the same image. To correctly compute the accuracy we
    # need to substract `y_pred` from 1 so both vectors are comparable.
    return metrics.binary_accuracy(y_true, 1 - y_pred)


def contrastive_loss(y_true, y_pred):
    """
    Computes the contrastive loss introduced by Yann LeCun et al. in the paper
    "Dimensionality Reduction by Learning an Invariant Mapping," 2005.
    """

    margin = 1
    y_true = tf.cast(y_true, y_pred.dtype)

    # The original formula proposed by Yann LeCunn et al. assumes that Y is 0
    # if both images are similar and 1 otherwise. Our implementation (where Y is
    # `y_true`) is the opposite, hence the modification to the formula below.
    loss = y_true / 2 * K.square(y_pred) + (1 - y_true) / 2 * K.square(
        K.maximum(0.0, margin - y_pred)
    )

    return loss


"""
## Implementation of the Siamese Network

To implement the Siamese Network, we first need to define the architecture of the
subnetwork twin model. This subnetwork will turn the input image into an embedding that
will later be used to determine how similar two images are.

The main model will use this twin subnetwork twice, connected to the two inputs, and cap
it off with a layer to compute the euclidean distance between both embeddings.
"""


def siamese_twin():
    """
    Creates the subnetwork that represents each one of the twins of the
    Siamese network.
    """

    inputs = layers.Input((28, 28, 1))

    x = layers.Conv2D(128, (2, 2), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(128, (2, 2), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(64, (2, 2), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.GlobalAveragePooling2D()(x)

    # We don't want any activation function on the final layer. This layer
    # will contain the embedding for the input image.
    outputs = layers.Dense(128, activation=None)(x)

    return Model(inputs, outputs)


def siamese_network():
    """
    Creates the Siamese Network model.
    """

    input1 = layers.Input(shape=(28, 28, 1))
    input2 = layers.Input(shape=(28, 28, 1))

    twin = siamese_twin()

    # We can use a Lambda layer to compute the euclidean distance of the
    # embeddings from each image.
    distance = layers.Lambda(norm)([twin(input1), twin(input2)])

    # Our model has two inputs: the first input is for the anchor image and
    # the second input is for the second image of the pair. The output of the
    # model is the euclidean distance between the embeddings from each image.
    return Model(inputs=[input1, input2], outputs=distance)


"""
## Build the Siamese Network model

We can build our Siamese network and compile it using our implementation of contrastive
loss and our custom accuracy metric.
"""

model = siamese_network()
model.compile(loss=contrastive_loss, optimizer="adam", metrics=[accuracy])

model.summary()

"""
Now we can train our model using the pairs we generated from the train and validation
sets.
"""

history = model.fit(
    x=[x_pairs_train[:, 0], x_pairs_train[:, 1]],
    y=y_pairs_train[:],
    validation_data=([x_pairs_val[:, 0], x_pairs_val[:, 1]], y_pairs_val[:]),
    batch_size=64,
    epochs=15,
)

plot_history(history.history)

"""
Finally, we can use the pairs we generated from the test set to predict them with our
model and display some of the results and the overall accuracy of the model.
"""

predictions = np.round(1 - model.predict([x_pairs_test[:, 0], x_pairs_test[:, 1]]))
display_pairs(x_pairs_test, predictions, predictions == y_pairs_test)

accuracy = metrics.BinaryAccuracy()
accuracy.update_state(y_pairs_test, predictions)
print(f"\nAccuracy: {accuracy.result().numpy()}")

"""
Title: Pneumonia Classification on TPU
Author: Amy MiHyun Jang
Date created: 2020/07/28
Last modified: 2020/08/04
Description: Medical image classification on TPU
"""
"""
## Introduction + Set-up

This tutorial will explain how to build an X-ray image classification model
to predict whether an X-ray scan shows presence of pneumonia.
"""

import re
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

"""
We need a Google Cloud link to our data to load the data using a TPU.
Below, we define key configuration parameters we'll use in this example.
"""

AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "gs://kds-7c9306925365b635aa934a70a0d94688c717d8c2eda0e47466736307"
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [180, 180]

"""
## Load the data

The Chest X-ray data we are using from
[*Cell*](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) divides the data
into training, validation, and test files. There are only 16 files in the validation folder,
and we would prefer to have a less extreme division between the training and the validation set.
We will append the validation files and create a new split that resembles the standard
80:20 division instead.
"""

filenames = tf.io.gfile.glob(str(GCS_PATH + "/chest_xray/train/*/*"))
filenames.extend(tf.io.gfile.glob(str(GCS_PATH + "/chest_xray/val/*/*")))

random.shuffle(filenames)
split_ind = int(0.8 * len(filenames))

train_filenames, val_filenames = filenames[:split_ind], filenames[split_ind:]

"""
Let's count how many healthy/normal chest X-rays we have and how many
pneumonia chest X-rays we have:
"""

COUNT_NORMAL = len([filename for filename in train_filenames if "NORMAL" in filename])
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len(
    [filename for filename in train_filenames if "PNEUMONIA" in filename]
)
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))

"""
Notice that there are way more images that are classified as pneumonia than normal. This
shows that we have a imbalance in our data. We will correct for this imbalance later on
in our notebook.
"""

train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)

for f in train_list_ds.take(5):
    print(f.numpy())

"""
Run the following cell to see how many images we have in our training dataset and how
many images we have in our validation set. Verify that the ratio of images is 80:20.
"""

TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))

VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))

"""
As expected, we have two labels for our images.
"""

CLASS_NAMES = [
    str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
    for item in tf.io.gfile.glob(str(GCS_PATH + "/chest_xray/train/*"))
]
print("Class names: %s" % (CLASS_NAMES,))

"""
Currently, our dataset is just a list of filenames. We want to map each filename to the
corresponding (image, label) pair. The following methods will help us do that.

As we only have two labels, we will encode the label so that `1` or `True` indicates
pneumonia and `0` or `False` indicates normal.
"""


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == "PNEUMONIA"


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size.
    return tf.image.resize(img, IMAGE_SIZE)


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

"""
Let's visualize the shape of an (image, label) pair.
"""

for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

"""
Load and format the test data as well.
"""

test_list_ds = tf.data.Dataset.list_files(str(GCS_PATH + "/chest_xray/test/*/*"))
TEST_IMG_COUNT = test_list_ds.cardinality().numpy()
test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)

print("Testing images count: " + str(TEST_IMG_COUNT))

"""
## Visualize the dataset

First, let's use buffered prefetching so we can yield data from disk without having I/O
become blocking.

Please note that large image datasets should not be cached in memory. We do it here
because the dataset is not very large and we want to train on TPU.
"""


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


"""
Call the next batch iteration of the training data.
"""

train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)

image_batch, label_batch = next(iter(train_ds))

"""
Define the method to show the images in the batch.
"""


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255)
        if label_batch[n]:
            plt.title("PNEUMONIA")
        else:
            plt.title("NORMAL")
        plt.axis("off")


"""
As the method takes in numpy arrays as its parameters, call the numpy function on the
batches to return the tensor in NumPy array form.
"""

show_batch(image_batch.numpy(), label_batch.numpy())

"""
## Build the CNN

To make our model more modular and easier to understand, let's define some blocks. As
we're building a convolution neural network, we'll create a convolution block and a dense
layer block.

The architecture for this CNN has been inspired by this
[article](https://towardsdatascience.com/deep-learning-for-detecting-pneumonia-from-x-ray-images-fc9a3d9fdba8).
"""

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def conv_block(filters):
    block = tf.keras.Sequential(
        [
            layers.SeparableConv2D(filters, 3, activation="relu", padding="same"),
            layers.SeparableConv2D(filters, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(),
        ]
    )

    return block


def dense_block(units, dropout_rate):
    block = tf.keras.Sequential(
        [
            layers.Dense(units, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
        ]
    )

    return block


"""
The following method will define the function to build our model for us.

The images originally have values that range from [0, 255]. CNNs work better with smaller
numbers so we will scale this down for our input.

The Dropout layers are important, as they reduce
reduce the likelikhood of the model overfitting. We want to end the model with a `Dense`
layer with one node, as this will be the binary output that determines if an X-ray shows
presence of pneumonia.
"""


def build_model():
    inputs = (tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),)
    x = preprocessing.Rescaling(1.0 / 255)(inputs)[0]
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)

    x = conv_block(32)(x)
    x = conv_block(64)(x)

    x = conv_block(128)(x)
    x = layers.Dropout(0.2)(x)

    x = conv_block(256)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = dense_block(512, 0.7)(x)
    x = dense_block(128, 0.5)(x)
    x = dense_block(64, 0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


"""
## Correct for data imbalance

We saw earlier in this example that the data was imbalanced, with more images classified
as pneumonia than normal. We will correct for that by using class weighting:
"""

initial_bias = np.log([COUNT_PNEUMONIA / COUNT_NORMAL])
print("Initial bias: {:.5f}".format(initial_bias[0]))

weight_for_0 = (1 / COUNT_NORMAL) * (TRAIN_IMG_COUNT) / 2.0
weight_for_1 = (1 / COUNT_PNEUMONIA) * (TRAIN_IMG_COUNT) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Weight for class 0: {:.2f}".format(weight_for_0))
print("Weight for class 1: {:.2f}".format(weight_for_1))

"""
The weight for class `0` (Normal) is a lot higher than the weight for class `1`
(Pneumonia). Because there are less normal images, each normal image will be weighted
more to balance the data as the CNN works best when the training data is balanced.
"""

"""
## Train the model

For our metrics, we want to include precision and recall as they will provide use with a
more informed picture of how good our model is. Accuracy tells us what fraction of the
labels is correct. Since our data is not balanced, accuracy might give a skewed sense of
a good model (i.e. a model that always predicts PNEUMONIA will be 74% accurate but is not
a good model).

Precision is the number of true positives (TP) over the sum of TP and false positives
(FP). It shows what fraction of labeled positives are actually correct.

Recall is the number of TP over the sum of TP and false negatves (FN). It shows what
fraction of actual positives are correct.
"""

with strategy.scope():
    model = build_model()

    METRICS = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)

"""
### Adjusting hyperparameters

Tuning is an art when it comes to Machine Learning, and there are ways to adjust the
hyperparameters of a model in efforts to improve it. Tuning is beyond the scope of this
notebook, but check out this
[article](https://medium.com/@jorgesleonel/hyperparameters-in-machine-deep-learning-ca69ad10b981) for more information.

For our purposes, we'll use Keras callbacks to adjust our hyperparameters. The checkpoint
callback saves the best weights of the model, so next time we want to use the model, we
do not have to spend time training it. The early stopping callback stops the training
process when the model starts becoming stagnant, or when the model starts
overfitting.
"""

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

"""
We also want to tune our learning rate. Too high of a learning rate will cause the model
to diverge. Too small of a learning rate will cause the model to be too slow. We
implement the exponential learning rate scheduling method below.
"""

initial_learning_rate = 0.015
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

"""
### Fit the model

Since there are only two possible labels for the image, we will be using the
binary crossentropy loss. When we fit the model, identify the class weights. Because we
are using a TPU, training will be relatively quick.
"""

with strategy.scope():
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=METRICS,
    )

history = model.fit(
    train_ds,
    epochs=100,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

"""
## Visualizing model performance

Let's plot the model accuracy and loss for the training and the validating set. Note that
no random seed is specified for this notebook. For your notebook, there might be slight
variance.
"""

fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(["precision", "recall", "accuracy", "loss"]):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history["val_" + met])
    ax[i].set_title("Model {}".format(met))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(met)
    ax[i].legend(["train", "val"])

"""
We see that the accuracy for our model is around 95%. Tune the model further to see if we
can achieve a higher score.
"""

"""
## Predict and evaluate results

Let's evaluate the model on our test data!
"""

model.evaluate(test_ds, return_dict=True)

"""
We see that our accuracy on our test data is ~87%, which is lower than the accuracy for
our validating set. This may indicate overfitting.

Our recall is greater than our precision, indicating that almost all pneumonia images are
correctly identified but some normal images are falsely identified. We should aim to
increase our precision.
"""

img = tf.io.read_file(
    str(GCS_PATH + "/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg")
)
img = decode_img(img)
plt.imshow(img / 255)

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

prediction = model.predict(img_array)[0]
scores = [1 - prediction, prediction]

for score, name in zip(scores, CLASS_NAMES):
    print("This image is %.2f percent %s" % ((100 * score), name))

"""
Our model could accurately classify this image.
"""

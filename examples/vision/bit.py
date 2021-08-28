"""
Title: Image Classification using BigTransfer(BiT)
Author: [Sayan Nath](https://twitter.com/sayannath2350)
Date created: 2021/08/29 
Last modified: 2021/08/29
Description: BigTransfer (BiT) State-of-the-art transfer learning for image classification.
"""

"""
## Introduction


BigTransfer known as BiT is a State-of-the-art transfer learning method for Image Classification Tasks. a set of pre-trained image models that can be transferred to obtain excellent performance on new datasets, even with only a few examples per class. BiT performs well across a surprisingly wide range of data regimes — from 1 example per class to 1M total examples. A detailed analysis of the main components are described [here](https://arxiv.org/pdf/1912.11370.pdf) which leads to high transfer performance.
"""

"""
## Setup
"""

import re
import numpy as np
import pandas as pd
import requests, zipfile, io
import matplotlib.pyplot as plt
from imutils import paths
from pprint import pprint
from collections import Counter
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical

SEEDS = 42

np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)

"""
## Download the dataset

We will be using Image-Scene-Classification dataset which comprises of 30 classes.
"""


def download_dataset(url):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("./")  # Extracting the Data from the zip file


download_dataset(
    "http://data.vision.ee.ethz.ch/ihnatova/camera_scene_detection_train.zip"
)

"""
## Data Parsing
"""

image_paths = list(paths.list_images("training"))
np.random.shuffle(image_paths)
image_paths[:5]

"""
## Counting number of images per class
"""

labels = []
for image_path in image_paths:
    label = image_path.split("/")[1]
    labels.append(label)
class_count = Counter(labels)
pprint(class_count)

"""
As there is an unequal distribution of classes in the dataset. It is classified as imbalance dataset
"""

"""
## Define hyperparameters
"""

RESIZE_TO = 260
CROP_TO = 224
TRAIN_SPLIT = 0.9
BATCH_SIZE = 128
AUTO = tf.data.AUTOTUNE
NUM_CLASSES = 30
STEPS_PER_EPOCH = 140

"""
## Splitting the Dataset
"""

TRAIN_LENGTH = int(len(image_paths) * TRAIN_SPLIT)

train_paths = image_paths[:TRAIN_LENGTH]
train_labels = labels[:TRAIN_LENGTH]
validation_paths = image_paths[TRAIN_LENGTH:]
validation_labels = labels[TRAIN_LENGTH:]

print(len(train_paths), len(validation_paths))

"""
## Encding the Labels
"""

label_encoder = LabelEncoder()
train_labels_le = label_encoder.fit_transform(train_labels)
validation_labels_le = label_encoder.transform(validation_labels)
print(train_labels_le[:5])

"""
Since the dataset has class imbalance issue, it's good to supply class weights while training the model.
"""

train_labels = to_categorical(train_labels_le)
class_totals = train_labels.sum(axis=0)
class_weight = dict()
# loop over all classes and calculate the class weight
for i in range(0, len(class_totals)):
    class_weight[i] = class_totals.max() / class_totals[i]

"""
## Set dataset-dependent hyperparameters
"""

DATASET_SIZE = "\u003C20k examples"  # @param ["<20k examples", "20k-500k examples", ">500k examples"]

if DATASET_SIZE == "<20k examples":
    SCHEDULE_LENGTH = 500
    SCHEDULE_BOUNDARIES = [200, 300, 400]
elif DATASET_SIZE == "20k-500k examples":
    SCHEDULE_LENGTH = 10000
    SCHEDULE_BOUNDARIES = [3000, 6000, 9000]
else:
    SCHEDULE_LENGTH = 20000
    SCHEDULE_BOUNDARIES = [6000, 12000, 18000]

"""
## Convert the data into TensorFlow Dataset objects
"""

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels_le))
val_ds = tf.data.Dataset.from_tensor_slices((validation_paths, validation_labels_le))

"""
## Preprocessing helper functions
"""

SCHEDULE_LENGTH = SCHEDULE_LENGTH * 512 / BATCH_SIZE


@tf.function
def preprocess_train(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    image = tf.cast(image, tf.float32) / 255.0
    return (image, label)


@tf.function
def preprocess_test(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    image = tf.cast(image, tf.float32) / 255.0
    return (image, label)


DATASET_NUM_TRAIN_EXAMPLES = len(train_paths)

"""
## Create Data Pipeline for training
"""

# Training Pipeline
pipeline_train = (
    train_ds.shuffle(10000)
    .repeat(
        int(SCHEDULE_LENGTH * BATCH_SIZE / DATASET_NUM_TRAIN_EXAMPLES * STEPS_PER_EPOCH)
        + 1
        + 50
    )  # Repeat dataset_size / num_steps
    .map(preprocess_train, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Testing Pipeline
pipeline_test = (
    val_ds.map(preprocess_test, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

"""
## Visualise the Dataset
"""

image_batch, label_batch = next(iter(pipeline_train))

plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(image_batch[n])
    plt.axis("off")

"""
## Load model into KerasLayer
"""

model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
module = hub.KerasLayer(model_url, trainable=True)

"""
## BiT Model
"""

class MyBiTModel(keras.Model):
    def __init__(self, num_classes, module):
        super().__init__()

        self.num_classes = num_classes
        self.head = keras.layers.Dense(num_classes, kernel_initializer="zeros")
        self.bit_model = module

    def call(self, images):
        # No need to cut head off since we are using feature extractor model
        bit_embedding = self.bit_model(images)
        return self.head(bit_embedding)


model = MyBiTModel(num_classes=NUM_CLASSES, module=module)

"""
## Define optimiser and loss
"""

learning_rate = 0.003 * BATCH_SIZE / 512

# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=SCHEDULE_BOUNDARIES,
    values=[
        learning_rate,
        learning_rate * 0.1,
        learning_rate * 0.01,
        learning_rate * 0.001,
    ],
)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

"""
## Compile the Model
"""

model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

"""## Setting the Callback"""

train_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True
    )
]

"""
## Train the Model
"""

history = model.fit(
    pipeline_train,
    batch_size=BATCH_SIZE,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=int(SCHEDULE_LENGTH / STEPS_PER_EPOCH),
    class_weight=class_weight,
    validation_data=pipeline_test,
    callbacks=train_callbacks,
)

"""
## Plot the Model
"""

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Training Progress")
    plt.ylabel("Accuracy/Loss")
    plt.xlabel("Epochs")
    plt.legend(["train_acc", "val_acc", "train_loss", "val_loss"], loc="upper left")
    plt.show()


plot_hist(history)

"""
## Evaluate the Model
"""

accuracy = model.evaluate(pipeline_test)[1] * 100
print("Accuracy: {:.2f}%".format(accuracy))

"""
## Note:
"""

"""
In this example, we trained our model for 5 epochs. In our experiment, the BigTransfer(BiT) Model performs amazing by giving us a good validation accuracy. BiT performs well across a surprisingly wide range of data regimes -- from 1 example per class to 1M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB). On small datasets, BiT attains 76.8% on ILSVRC-2012 with 10 examples per class, and 97.0% on CIFAR-10 with 10 examples per class.

You can experiment further with the BigTransfer Method by following the [original paper](https://arxiv.org/pdf/1912.11370.pdf).
"""

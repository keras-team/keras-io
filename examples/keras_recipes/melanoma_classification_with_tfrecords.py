"""
Title: How to train a Keras model on TFRecord files
Author: Amy MiHyun Jang
Date created: 2020/07/29
Last modified: 2020/08/06
Description: Loading TFRecords for computer vision models
"""
"""
## Introduction + Set Up

TFRecords store a sequence of binary records, read linearly. They are useful format for
storing data because they can be read efficiently. We'll explore how we can easily load
in TFRecords for our melanoma classifier.
"""

"""shell
! pip install gcsfs -q
"""

import re
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
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
Running the following cell will save time in loading the data as it allows for parallel
processing. TPU can only work with Google Cloud files and cannot be run with local files.
We want a bigger batch size as our data is not balanced.

"""

AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "gs://kds-f809fea39df606e89617cd8d8e4cacc083fb6e176982f0e5d69215a8"
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]

"""
## Load the data

Now we will load in our data. For this notebook, we will be importing the TFRecord files.
It is good practice to divide the training set data into two. The smaller dataset will be
the validation set. Having a validation set is useful to slow down overfitting.
"""

FILENAMES = tf.io.gfile.glob(GCS_PATH + "/tfrecords/train*.tfrec")
split_ind = int(0.9 * len(FILENAMES))
TRAINING_FILENAMES, VALID_FILENAMES = FILENAMES[:split_ind], FILENAMES[split_ind:]

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/tfrecords/test*.tfrec")
print("Train TFRecord Files:", len(TRAINING_FILENAMES))
print("Validation TFRecord Files:", len(VALID_FILENAMES))
print("Test TFRecord Files:", len(TEST_FILENAMES))

"""
### Decoding the data

The images have to be converted to tensors so that it will be a valid input in our model.
As images utilize an RBG scale, we specify 3 channels.

It is also best practice to normalize data before it is is fed into the model. For our
image data, we will scale it down so that the value of each pixel will range from [0, 1]
instead of [0, 255].

We also reshape our data so that all of the images will be the same shape. Although the
TFRecord files have already been reshaped for us, it is best practice to reshape the
input so that we know exactly what's going in to our model.
"""


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


"""
As we load in our data, we need both our ```X``` and our ```Y```. The X is our image; the
model will find features and patterns in our image dataset. We want to predict Y, the
probability that the lesion in the image is malignant. When we input our training dataset
into the model, it is necessary to know what the labels of our images are. We will to
through our TFRecords and parse out the image and the target values.
"""


def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"image": tf.io.FixedLenFeature([], tf.string),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image


"""
### Define loading methods

Our dataset is not ordered in any meaningful way, so the order can be ignored when
loading our dataset. By ignoring the order and reading files as soon as they come in, it
will take a shorter time to load the data.
"""


def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=AUTOTUNE
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


"""
We define the following function to get our different datasets.
"""


def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


"""
### Visualize input images
"""

train_dataset = get_dataset(TRAINING_FILENAMES)
valid_dataset = get_dataset(VALID_FILENAMES)
test_dataset = get_dataset(TEST_FILENAMES, labeled=False)

image_batch, label_batch = next(iter(train_dataset))


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title("MALIGNANT")
        else:
            plt.title("BENIGN")
        plt.axis("off")


show_batch(image_batch.numpy(), label_batch.numpy())

"""
## Explore our data

Our data is imbalanced. When we look at our data, we see that only 1.76% of the images
are images of malignant lesions while 98.24% of the images are benign.
"""

train_csv = pd.read_csv(GCS_PATH + "/train.csv")

total_img = train_csv["target"].size

malignant = np.count_nonzero(train_csv["target"])
benign = total_img - malignant

print(
    "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
        total_img, malignant, 100 * malignant / total_img
    )
)

"""
### Correcting for data inbalance
"""

"""
#### Set initial bias

We want to set the correct initial bias for our model so that it will not waste time
figuring out that there are not many malignant images in our dataset. We want our output
layer to reflect the inbalance that we have in our data.
"""

initial_bias = np.log([malignant / benign])
print("Initial bias: %.4f" % initial_bias)

"""
#### Set class weights

Since there are not enough malignant images, we want these malignant images to have more
weight in our model. By increasing the weight of these malignant images, the model will
pay more attention to them, and this will help balance out the difference in quantity.
"""

weight_for_0 = (1 / benign) * (total_img) / 2.0
weight_for_1 = (1 / malignant) * (total_img) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Weight for class 0: {:.2f}".format(weight_for_0))
print("Weight for class 1: {:.2f}".format(weight_for_1))

"""
## Building our model
"""

"""
### Define callbacks

The following function allows for the model to change the learning rate as it runs each
epoch.

We can use callbacks to stop training when there are no improvements in the model. At the
end of the training process, the model will restore the weights of its best iteration.
"""

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "melanoma_model.h5", save_best_only=True
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

"""
### Build our base model

Transfer learning is a great way to reap the benefits of a well-trained model without
having the train the model ourselves. For this notebook, we want to import the Xception
model. A more in-depth analysis of transfer learning can be found
[here](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/).

We do not want our metric to be ```accuracy``` because our data is imbalanced. For our
example, we will be looking at the area under a ROC curve.
"""


def make_model():
    output_bias = tf.keras.initializers.Constant(initial_bias)

    base_model = tf.keras.applications.Xception(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                8, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dropout(0.7),
            tf.keras.layers.Dense(
                1, activation="sigmoid", bias_initializer=output_bias
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=tf.keras.metrics.AUC(name="auc"),
    )

    return model


with strategy.scope():
    model = make_model()

"""
## Train the model
"""

history = model.fit(
    train_dataset,
    epochs=2,
    validation_data=valid_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb],
    class_weight=class_weight,
)

"""
## Predict results

We'll use our model to predict results for our test dataset images. Values closer to `0`
are more likely to be benign and values closer to `1` are more likely to be malignant.
"""


def show_batch_predictions(image_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        img_array = tf.expand_dims(image_batch[n], axis=0)
        plt.title(model.predict(img_array)[0])
        plt.axis("off")


image_batch = next(iter(test_dataset))

show_batch_predictions(image_batch)

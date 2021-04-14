"""
Title: CutMix data augmentation for image classification<br>
Author: [Sayan Nath](https://twitter.com/SayanNa20204009)<br>
Date created: 2021/04/14<br>
Last modified: 2021/04/14<br>
Description: Data augmentation with CutMix for image classification on CIFAR-10.
"""

"""
## Introduction
"""

"""
_CutMix_ is a data augmentation technique that addresses the issue of
information loss and inefficiency present in regional dropout strategies. 
Instead of removing pixels and filling them with black or grey pixels or
Gaussian noise, you replace the removed regions with a patch from another image,
while the ground truth labels are mixed proportionally to the number of pixels
of combined images. The CutMix technique can make efficient use of training
pixels, while retaining the regularization effect of regional dropout.
This data augmentation technique was proposed in
[CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
Features](https://arxiv.org/pdf/1905.04899.pdf)
(Yun et al., 2019).

It's implemented with the following formulas:

![](https://i.imgur.com/cGvd13V.png)

where M is the binary mask which indicates the cutout and the fill-in regions from the
two randomly drawn images and λ is drawn from [Beta(α,α)
distribution](https://en.wikipedia.org/wiki/Beta_distribution) and `λ ∈ [0, 1]`


The coordinates of bounding boxes are ![](https://i.imgur.com/eNisep4.png) which
indicates the cutout and fill-in regions in case of the images.

The bounding box sampling is represented by:

![](https://i.imgur.com/Snph9aj.png)

where `rx,ry` are randomly drawn from a uniform distribution with upper bound

This example requires TensorFlow 2.4 or higher.
"""

"""
## Import the necessary packages
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import resnet50

np.random.seed(42)
tf.random.set_seed(42)


"""
## Load the CIFAR-10 dataset

In this example, we will use the
[CIFAR- 10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

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

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_SHAPE = 32

"""
## Define the image preprocessing function
"""


def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image, label


"""
## Convert the data into TensorFlow `Dataset` objects
"""

train_ds_one = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1024)
    .map(preprocess_image, num_parallel_calls=AUTO)
)
train_ds_two = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1024)
    .map(preprocess_image, num_parallel_calls=AUTO)
)

train_ds_simple = tf.data.Dataset.from_tensor_slices((x_train, y_train))

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_ds_simple = (
    train_ds_simple.map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Combine two shuffled datasets from the same training data.
train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

test_ds = (
    test_ds.map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

"""
## Define the CutMix data augmentation function

The CutMix function takes two `image` and `label` pairs to perform the
augmentation. It samples `λ(l)` from the
[Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) and returns
a bounding box from `get_box` function. We then crop the second image (`image2`)
and pad this image in the final padded image at the same location.
"""


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


@tf.function
def get_box(l):
    cut_rat = tf.math.sqrt(1.0 - l)

    cut_w = IMG_SHAPE * cut_rat  # rw
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = IMG_SHAPE * cut_rat  # rh
    cut_h = tf.cast(cut_h, tf.int32)

    cx = tf.random.uniform((1,), minval=0, maxval=IMG_SHAPE, dtype=tf.int32)  # rx
    cy = tf.random.uniform((1,), minval=0, maxval=IMG_SHAPE, dtype=tf.int32)  # ry

    boundaryx1 = tf.clip_by_value(cx[0] - cut_w // 2, 0, IMG_SHAPE)
    boundaryy1 = tf.clip_by_value(cy[0] - cut_h // 2, 0, IMG_SHAPE)
    bbx2 = tf.clip_by_value(cx[0] + cut_w // 2, 0, IMG_SHAPE)
    bby2 = tf.clip_by_value(cy[0] + cut_h // 2, 0, IMG_SHAPE)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w


@tf.function
def cutmix(a, b):

    (image1, label1), (image2, label2) = a, b

    alpha = [1.0]
    beta = [1.0]

    # Get a sample from the Beta distribution
    l = sample_beta_distribution(1, alpha, beta)

    # Define Lambda
    l = l[0][0]

    # Get the bounding box offsets, heights and widths
    bbx1, bby1, target_h, target_w = get_box(l)

    # Get a patch from the second image (`image2`)
    crop2 = tf.image.crop_to_bounding_box(image2, boundaryy1, boundaryx1, target_h, target_w)
    # Pad the `image2` patch (`crop2`) with the same offset
    image2 = tf.image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SHAPE, IMG_SHAPE)
    # Get a patch from the first image (`image1`)
    crop1 = tf.image.crop_to_bounding_box(image1, boundaryy1, boundaryx1, target_h, target_w)
    # Pad the `image1` patch (`crop1`) with the same offset
    img1 = tf.image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SHAPE, IMG_SHAPE)

    # Modify the first image by subtracting the patch from `image1`
    # (before applying the `image2` patch)
    image1 = image1 - img1
    # Add the modified `image1` and `image2`  together to get the CutMix image
    image = image1 + image2

    # Adjust Lambda in accordance to the pixel ration
    l = 1 - (target_w * target_h) / (IMG_SHAPE * IMG_SHAPE)
    l = tf.cast(l, tf.float32)

    # Combine the labels of both images
    label = l * label1 + (1 - l) * label2

    return image, label


"""
**Note**: we are combining two images to create a single one.
"""

"""
## Visualize the new dataset after applying the CutMix augmentation
"""

# Create the new dataset using our `cutmix` utility
train_ds_cmu = (
    train_ds.shuffle(1024)
    .map(cutmix, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Let's preview 9 samples from the dataset
image_batch, label_batch = next(iter(train_ds_cmu))
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.title(class_names[np.argmax(label_batch[i])])
    plt.imshow(image_batch[i])
    plt.axis("off")

"""
## Define the model using ResNet-20
"""


def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):

    conv = keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v20(input_shape, depth, num_classes=10):

    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = keras.layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )
            x = keras.layers.add([x, y])
            x = keras.layers.Activation("relu")(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = keras.layers.AveragePooling2D(pool_size=8)(x)
    y = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", kernel_initializer="he_normal"
    )(y)

    # Instantiate model.
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def training_model():
    return resnet_v20((32, 32, 3), 20)


initial_model = training_model()
initial_model.save_weights("initial_weights.h5")

"""
## 1. Train the model with the dataset augmented by CutMix
"""

model = training_model()
model.load_weights("initial_weights.h5")

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_ds_cmu, validation_data=test_ds, epochs=10)

test_loss, test_accuracy = model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

"""
## 2. Train the model using the original non-augmented dataset
"""

model = training_model()
model.load_weights("initial_weights.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_ds_simple, validation_data=test_ds, epochs=10)

test_loss, test_accuracy = model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

"""
## Notes
In this example, we trained our model for 15 epochs. In our experiment, the
model with CutMix achieves a better accuracy on the CIFAR-10 dataset (80.02% in
our experiment) compared to the model that doesn't use the augmentation
(74.20%).

You may notice it takes less time to train the model with the CutMix augmentation.

You can experiment further with the CutMix technique by following the
[original paper](https://arxiv.org/pdf/1905.04899.pdf).
"""

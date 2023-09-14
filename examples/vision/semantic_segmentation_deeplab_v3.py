"""
Title: Semantic segmentation using DeepLabV3+
Author: [Alexia Audevart](https://github.com/aaudevart)
Date created: 2023/13/04
Last modified: 2023/13/04
Description: Use the DeepLabV3+ pre-trained models via the Keras CV API for Semantic Segmentation.
Accelerator: GPU
"""

"""
## Introduction

Semantic segmentation, with the goal to assign semantic labels to every pixel in an
image, is an essential computer vision task.

In this example, we implement
the **DeepLabV3+** model for semantic segmentation in healthcare. This model is a
fully-convolutional
architecture that performs well on semantic segmentation benchmarks.

The goal is to detect blood vessels on 2D PAS-stained histology images from healthy human
kidney tissue slides.


### References:

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
Segmentation](https://arxiv.org/abs/1802.02611)
- [Rethinking Atrous Convolution for Semantic Image
Segmentation](https://arxiv.org/abs/1706.05587)
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution,
and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)

### Dataset:

We will use a simplified extract of the [HuBMAP - Hacking the Human Vasculature
Dataset](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature)
for training our model. This dataset is obtained from Periodic acid-Schiff (PAS) stained
tissue slides from human kidneys.
Each image is labeled with pixel-wise annotations for 2 categories: background and blood
vessel.
This dataset focuses on microvasculature in the kidney, including arterioles, capillaries
and venules. Larger vessels structures, i.e. arteries and veins, were not segmented in
this dataset.
"""

"""shell
!pip install keras-core --upgrade
!pip uninstall -y keras-cv
!pip install git+https://github.com/keras-team/keras-cv.git

!pip install gdown

# This sample uses Keras Core, the multi-backend version of Keras.
# The selected backend is TensorFlow (other supported backends are 'jax' and 'torch')
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
"""

import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf


import keras_core as keras

import keras_cv

from keras_cv.models import ResNet18V2Backbone, DeepLabV3Plus

print("TensorFlow version:", tf.__version__)
print("KerasCV version:", keras_cv.__version__)

"""
## Accelerator

Detect hardware, return appropriate distribution strategy
"""

try:
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    # instantiate a distribution strategy
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("TPU not activated")
    strategy = (
        tf.distribute.MirroredStrategy()
    )  # Works on CPU, single GPU and multiple GPUs in a single VM.

print("replicas:", strategy.num_replicas_in_sync)

"""
## Loading the data into TensorFlow Datasets

Training on the entire HuBMAP - Hacking the Human Vasculature dataset with more than 7000
images takes a lot of time, hence we will be using
a smaller subset of about 400 images for training our model in this example.
"""

"""
The dataset is an extract of a dataset used in a Kaggle competition.

You can find more information
[here](https://www.kaggle.com/datasets/alexia/hubmap-kidney).



"""

"""shell
!gdown 1HGYBgRP362lW5Xtt-RW3mYNibjd45r8P
!unzip hubmap-kidney.zip -d hubmap-kidney
"""

DATA_DIR = "hubmap-kidney/"
IMG_DIR = DATA_DIR + "img/"
MASK_DIR = DATA_DIR + "mask/"

import os

images_list = []
for dir_path, dir_names, file_names in os.walk(IMG_DIR):
    for file in file_names:
        if file.endswith("png"):
            images_list.append(file)


NUM_IMAGES = 416
IMG_HEIGHT = 512
IMG_WIDTH = 512
# Define the split percentages
TRAIN_SPLIT = 0.8
NUM_TRAIN_IMAGES = int(NUM_IMAGES * TRAIN_SPLIT)

BATCH_SIZE = 4

train_img_list = images_list[:NUM_TRAIN_IMAGES]
val_img_list = images_list[NUM_TRAIN_IMAGES:]


def gen_pairs(img_list):
    def gen():
        for img_id in img_list:
            image = tf.io.read_file(IMG_DIR + "/" + img_id)
            image = tf.image.decode_png(image, channels=3)
            image.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
            mask = tf.io.read_file(MASK_DIR + "/" + img_id)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.one_hot(tf.squeeze(mask), 2)
            mask.set_shape([IMG_HEIGHT, IMG_WIDTH, 2])
            yield (image, mask)

    return gen


# Prepare the training dataset.
train_gen = tf.data.Dataset.from_generator(
    generator=gen_pairs(train_img_list),
    output_signature=(
        tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, 2), dtype=tf.uint8),
    ),
)
train_gen = train_gen.batch(BATCH_SIZE, drop_remainder=True)

# Prepare the validation dataset.
val_gen = tf.data.Dataset.from_generator(
    generator=gen_pairs(val_img_list),
    output_signature=(
        tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, 2), dtype=tf.uint8),
    ),
)
val_gen = val_gen.batch(BATCH_SIZE, drop_remainder=True)

print("Train Dataset:", train_gen)
print("Val Dataset:", val_gen)

"""
## Displaying some images and mask

For each image, we display two images:
- the original image
- the original image as background and the mask as overlay
"""

# Define the Colormap: white & red
colormap = [[255, 255, 255], [255, 0, 0]]


def decode_segmentation_masks(mask, colormap, n_classes=2):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l][0]
        g[idx] = colormap[l][1]
        b[idx] = colormap[l][2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.utils.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(4, 2)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    title = ["Input Image", "True Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.axis("off")
        if display_list[i].shape[-1] == 2:
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        else:
            plt.imshow(display_list[i])
    plt.show()


def plot_samples(images_dataset, colormap):
    for img_iterator in images_dataset:
        image = img_iterator[0][0]
        true_mask = img_iterator[1][0]
        true_mask = tf.argmax(true_mask, axis=-1)
        true_mask = tf.expand_dims(true_mask, axis=-1)
        overlay_true_mask = decode_segmentation_masks(
            np.squeeze(true_mask, 2), colormap
        )
        overlay_true_mask = get_overlay(image, overlay_true_mask)
        plot_samples_matplotlib([image, overlay_true_mask], figsize=(9, 6))


plot_samples(train_gen.take(2), colormap)

"""
## Training

We train the model using sparse categorical crossentropy as the loss function, and
Adam as the optimizer.
"""

EPOCHS = 16

from keras_cv.models import ResNetV2Backbone, DeepLabV3Plus

with strategy.scope():
    backbone = ResNetV2Backbone.from_preset(
        "resnet50_v2_imagenet", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )

    model = DeepLabV3Plus(backbone=backbone, num_classes=2)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.CategoricalCrossentropy(),
    )

model.summary()

history = model.fit(train_gen, epochs=EPOCHS, verbose=True, validation_data=val_gen)

# summarize history for loss
fig, ax = plt.subplots(1)
ax.set_ylim(bottom=0, top=0.5)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")

"""
## Inference using Colormap Overlay

The raw predictions from the model represent a one-hot encoded tensor of shape `(N, 256,
256, 2)`
where each one of the 2 channels is a binary mask corresponding to a predicted label
(background or blood vessel).
In order to visualize the results, we plot them as RGB segmentation masks where each pixel
is represented by a unique color corresponding to the particular label predicted.
We would also plot an overlay of the RGB segmentation mask on the input image as
this further helps us to identify the different categories present in the image more
intuitively.
"""


def infer(model, image, true_mask):
    predictions = model.predict(np.expand_dims((image), axis=0))

    predictions = np.squeeze(predictions)
    predictions = tf.argmax(predictions, axis=-1)
    return predictions


def plot_samples_and_predict_matplotlib(display_list, iou_metric, figsize=(4, 2)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    title = ["Input Image", "Predicted Mask", "True Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.axis("off")
        if display_list[i].shape[-1] == 3:
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        else:
            plt.imshow(display_list[i])
    plt.show()


def plot_predictions(images_dataset, colormap, model):
    for img_iterator in images_dataset:
        image = img_iterator[0][0]
        true_mask = img_iterator[1][0]
        true_mask = tf.argmax(true_mask, axis=-1)
        true_mask = tf.expand_dims(true_mask, axis=-1)
        prediction_mask = infer(model, image, true_mask)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap)
        overlay_predicted = get_overlay(image, prediction_colormap)
        overlay_true_mask = decode_segmentation_masks(
            np.squeeze(true_mask, 2), colormap
        )
        overlay_true_mask = get_overlay(image, overlay_true_mask)
        plot_samples_and_predict_matplotlib(
            [image, overlay_predicted, overlay_true_mask], 10, figsize=(18, 14)
        )


"""
### Inference on Validation Images
"""

plot_predictions(val_gen.take(3), colormap, model=model)

"""
Title: End-to-end Object detection with transformers
Author: Karan V. Dave
Date created: 2022/03/27
Last modified: 2022/03/27
Description: A simple keras implementation for object detection using transformers.
"""


"""
## Introduction

 This example is a an implementation of the [Vision Transformer (ViT)]
 (https://arxiv.org/abs/2010.11929)
 model by Alexey Dosovitskiy et al. for object detection,
 it is demonstrated on the Caltech 101 dataset for detecting airplane in an image.

 This example requires TensorFlow 2.4 or higher, and
 [TensorFlow Addons](https://www.tensorflow.org/addons/overview),
 which can be installed using the following command:
```
 pip install -U tensorflow-addons
```
"""

"""
## Imports and setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import scipy.io

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import tensorflow as tf
import datetime

"""
 To run the TensorBoard from a Jupyter notebook,
 load the TensorBoard notebook extension inside notebook
```
%load_ext tensorboard
```
"""

"""
## Prepare dataset
"""

# Dataset
# http://www.vision.caltech.edu/Image_Datasets/Caltech101/
# Download the images '101_ObjectCategories.tar.gz' and
# annotations 'Annotations.tar' available seperately
# Extract images (using airplanes here) to images folder and
# all annotations to another folder.

# Path to images and annotations
PATH_IMAGES = "C:/Users/DAVE/Desktop/Object detection with transformers/airplanes/"
PATH_ANNOT = (
    "C:/Users/DAVE/Desktop/Object detection with transformers/Airplanes_Side_2/"
)

# list of paths to images and annotations
image_paths = [
    f for f in os.listdir(PATH_IMAGES) if os.path.isfile(os.path.join(PATH_IMAGES, f))
]
annot_paths = [
    f for f in os.listdir(PATH_ANNOT) if os.path.isfile(os.path.join(PATH_ANNOT, f))
]

annot_paths.sort()
image_paths.sort()

images = []
targets = []

# loop over the annotation rows, preprocess and store in list
for i in range(0, len(annot_paths)):
    #   Access bounding box coordinates
    annot = scipy.io.loadmat(PATH_ANNOT + annot_paths[i])["box_coord"][0]

    topLeft_X = annot[2]
    topLeft_Y = annot[0]
    bottomRight_X = annot[3]
    bottomRight_Y = annot[1]

    image = cv2.imread(
        "C:/Users/DAVE/Desktop/Object detection with transformers/airplanes/"
        + image_paths[i]
    )
    (h, w) = image.shape[:2]

    # apply relative scaling to bounding boxes as per given image
    topLeft_X = float(topLeft_X) / w
    topLeft_Y = float(topLeft_Y) / h
    bottomRight_X = float(bottomRight_X) / w
    bottomRight_Y = float(bottomRight_Y) / h

    # load the image and convert to array. Store image to images list
    # and bounding box in targets list
    image = load_img(
        "C:/Users/DAVE/Desktop/Object detection with transformers/airplanes/"
        + image_paths[i],
        target_size=(224, 224),
    )
    image = img_to_array(image)
    images.append(image)
    targets.append((topLeft_X, topLeft_Y, bottomRight_X, bottomRight_Y))

# Print Shape
# print(np.shape(images))
# print(np.shape(targets))

# Convert to numpy and split train and test dataset
(x_train) = np.asarray(images[: int(len(images) * 0.8)])
(y_train) = np.asarray(targets[: int(len(targets) * 0.8)])
# (x_test) = np.asarray(images[int(len(images) * 0.8) :])
# (y_test) = np.asarray(targets[int(len(targets) * 0.8) :])

# print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
# print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

"""
## Data augmentation
"""

# Some Code referred from
# https://keras.io/examples/vision/image_classification_with_vision_transformer/

image_size = 64  # resize input images to this size
patch_size = 8  # Size of the patches to be extracted from the input images

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

"""
## Implement multilayer-perceptron (MLP)
"""


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


"""
## Implement patch creation layer
"""


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    #     Override function to avoid error while saving model
    #    def get_config(self):
    #        config = super().get_config().copy()
    #        config.update(
    #            {
    #                "input_shape": input_shape,
    #                "patch_size": patch_size,
    #                "num_patches": num_patches,
    #                "projection_dim": projection_dim,
    #                "num_heads": num_heads,
    #                "transformer_units": transformer_units,
    #                "transformer_layers": transformer_layers,
    #                "mlp_head_units": mlp_head_units,
    #            }
    #        )
    #        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


"""
Display patches for a an input image
"""

import matplotlib.pyplot as plt

# patch_size = 16
plt.figure(figsize=(4, 4))
image = x_train[0]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
# print(f"Image size: {image_size} X {image_size}")
# print(f"Patch size: {patch_size} X {patch_size}")
# print(f"Patches per image: {patches.shape[1]}")
# print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")

"""
## Implement the patch encoding layer
 The `PatchEncoder` layer linearly transforms a patch by projecting it into a
 vector of size `projection_dim`. It also adds a learnable position
 embedding to the projected vector.
"""


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    #     Override function to avoid error while saving model
    #    def get_config(self):
    #        config = super().get_config().copy()
    #        config.update(
    #            {
    #                "input_shape": input_shape,
    #                "patch_size": patch_size,
    #                "num_patches": num_patches,
    #                "projection_dim": projection_dim,
    #                "num_heads": num_heads,
    #                "transformer_units": transformer_units,
    #                "transformer_layers": transformer_layers,
    #                "mlp_head_units": mlp_head_units,
    #            }
    #        )
    #        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


"""
## Build the ViT model
 The ViT model have multiple Transformer blocks, after normalizing,
 they use `layers.MultiHeadAttention`
 layer for self attention and fed into multi-layer perceptron which outputs
 four neurons representing bounding box of the object.

 In the [paper](https://arxiv.org/abs/2010.11929), learnable
 embedding is prepended to the sequence of encoded patches.
 Here, `layers.Flatten()` layer is used as image representation to
 MLP followed by final four neurons.
"""


def create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
):
    inputs = layers.Input(shape=input_shape)
    # Data augumentation
    augmented = data_augmentation(inputs)
    # Create patches
    patches = Patches(patch_size)(augmented)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    bounding_box = tf.keras.layers.Dense(4)(
        features
    )  # Final four neurons that output bounding box

    # return Keras model.
    return keras.Model(inputs=inputs, outputs=bounding_box)


"""
## Run experiment
"""

import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model

# import pandas as pd

tf.config.run_functions_eagerly(True)


def run_experiment(model, learning_rate, weight_decay, batch_size, num_epochs):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    #     Comiple model and switch loss function as needed.
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    checkpoint_filepath = "logs/"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )
    #    log_dir = (
    #        "logs/tensorboard/"
    #        + "input_shape-{}-patch_size-{}-learning_rate-{}-weight_decay-{}-batch_size-{}-projection_dim-{}-num_heads-{}-transformer_layers-{}-mlp_head_units-{}".format(
    #            input_shape,
    #            patch_size,
    #            learning_rate,
    #            weight_decay,
    #            batch_size,
    #            projection_dim,
    #            num_heads,
    #            transformer_layers,
    #            mlp_head_units,
    #        )
    #    )
    #    tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #        log_dir=log_dir, histogram_freq=1
    #    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.15,
        callbacks=[
            checkpoint_callback,
            # tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )

    return history


input_shape = (224, 224, 3)  # input image shape
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 50
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers


history = []
num_patches = (image_size // patch_size) ** 2

vit_object_detector = create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
)

# Gridsearch can be use to loop over different hyper params and store histories
# history.append(
#    [
#        'storeHyperparams',
run_experiment(vit_object_detector, learning_rate, weight_decay, batch_size, num_epochs)
# .history,
#    ]
# )
# dff = pd.DataFrame(history)
# dff.to_csv("logs/list.csv", index=False)

# Training loss over epochs, can loop over if history list as multiple history logs
# history[0][1]


"""
View loss for trained models inside jupyter notebook
```python
%tensorboard --logdir logs/tensorboard
```
"""

"""
## Model Evaluation
"""  # Some code referred from
# https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/

# To save the model in current path
# vit_object_detector.save("vit_object_detector.h5", save_format="h5")

# plot the model training history
# N = len(history[0][1]["loss"])
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), history[0][1]["loss"], label="train_loss")
# plt.plot(np.arange(0, N), history[0][1]["val_loss"], label="val_loss")
# plt.title("Plot regression loss on training set for bounding box predictions")
# plt.xlabel("Epoch number")
# plt.ylabel("Loss")
# plt.legend(loc="lower right")


# Testing predictions
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def bounding_box_intersection_over_union(boxA, boxB):
    # get (x, y) coordinates of intersection of bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # calculate area of the intersection bounding box
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # calculate area of the prediction bb and ground-truth bb
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # calculate intersection over union by taking intersection
    # area and dividing it by the sum of predicted bb + ground-truth
    # bb areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return ioU
    return iou


def load_images_from_folder(image_paths):
    images = []
    for filename in image_paths:
        img = cv2.imread(os.path.join(PATH_IMAGES, filename))
        if img is not None:
            images.append(img)
    return images


# load original images before resizing as predictions are scaled for original images
images_original = load_images_from_folder(image_paths)

showImagesAfter = 790
i = showImagesAfter
mIoU = 0

# Compare results for last 10 images which were added to test set
for inputImage in images_original[showImagesAfter:]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    im = inputImage

    # Display the image
    ax1.imshow(im)
    ax2.imshow(im)

    inputImage = cv2.resize(inputImage, (224, 224), interpolation=cv2.INTER_AREA)
    inputImage = inputImage / 255.0
    inputImage = np.expand_dims(inputImage, axis=0)
    preds = vit_object_detector.predict(inputImage)[0]
    topLeft_X, topLeft_Y, endX, endY = preds

    (h, w) = (im).shape[0:2]
    topLeft_X = int(topLeft_X * w)
    topLeft_Y = int(topLeft_Y * h)
    bottomRight_X = int(endX * w)
    bottomRight_Y = int(endY * h)

    boxA = [topLeft_X, topLeft_Y, bottomRight_X, bottomRight_Y]
    # Create bounding box
    rect = patches.Rectangle(
        (topLeft_X, topLeft_Y),
        bottomRight_X - topLeft_X,
        bottomRight_Y - topLeft_Y,
        facecolor="none",
        edgecolor="red",
        linewidth=1,
    )
    # Add bounding box to the image
    ax1.add_patch(rect)
    ax1.set_xlabel(
        "Predicted: "
        + str(topLeft_X)
        + ", "
        + str(topLeft_Y)
        + ", "
        + str(bottomRight_X)
        + ", "
        + str(bottomRight_Y)
    )

    topLeft_X = int(targets[i][0] * w)
    topLeft_Y = int(targets[i][1] * h)
    bottomRight_X = int(targets[i][2] * w)
    bottomRight_Y = int(targets[i][3] * h)

    boxB = topLeft_X, topLeft_Y, bottomRight_X, bottomRight_Y

    mIoU += bounding_box_intersection_over_union(boxA, boxB)
    # Create bounding box
    rect = patches.Rectangle(
        (topLeft_X, topLeft_Y),
        bottomRight_X - topLeft_X,
        bottomRight_Y - topLeft_Y,
        facecolor="none",
        edgecolor="red",
        linewidth=1,
    )
    # Add bounding box to the image
    ax2.add_patch(rect)
    ax2.set_xlabel(
        "Target: "
        + str(topLeft_X)
        + ", "
        + str(topLeft_Y)
        + ", "
        + str(bottomRight_X)
        + ", "
        + str(bottomRight_Y)
        + "\n"
        + "IoU"
        + str(bounding_box_intersection_over_union(boxA, boxB))
    )
    i = i + 1

mIoU = mIoU / len(images_original[790:])
print("mIoU" + str(mIoU))
plt.show()

"""
Model can be improved further by tuning hyper-parameters and pre-training.
"""

"""
Title: End-to-end Object detection with transformers
Author: Karan V. Dave
Date created: 2022/03/27
Last modified: 2022/03/27
Description: A simple keras implementation for object detection using transformers.
"""

"""
## Introduction

 This example is a an implementation of the [Vision Transformer (ViT)] (https://arxiv.org/abs/2010.11929) model by Alexey Dosovitskiy et al. for object detection,
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
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import scipy.io
import shutil

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
PATH_IMAGES = "/101_ObjectCategories/airplanes/"
PATH_ANNOT = "/Annotations/Airplanes_Side_2/"

path_to_downloaded_file = tf.keras.utils.get_file(
    fname="caltech_101_zipped",
    origin="https://data.caltech.edu/tindfiles/serve/e41f5188-0b32-41fa-801b-d1e840915e80/",
    extract=True,
    archive_format="zip",  # downloaded file format
    cache_dir="/",  # cache and extract in current directory
)

# `keras.utils.get_file()` has default subdirectory 'datasets' for downloading and extracting files
# Extracting tar files found inside main zip file
shutil.unpack_archive("/datasets/caltech-101/101_ObjectCategories.tar.gz", "/")
shutil.unpack_archive("/datasets/caltech-101/Annotations.tar", "/")

# list of paths to images and annotations
image_paths = [
    f for f in os.listdir(PATH_IMAGES) if os.path.isfile(os.path.join(PATH_IMAGES, f))
]
annot_paths = [
    f for f in os.listdir(PATH_ANNOT) if os.path.isfile(os.path.join(PATH_ANNOT, f))
]

image_paths.sort()
annot_paths.sort()

image_size = 224  # resize input images to this size

# empty lists
images, targets = [], []

# loop over the annotation rows, preprocess and store in list
for i in range(0, len(annot_paths)):
    #   Access bounding box coordinates
    annot = scipy.io.loadmat(PATH_ANNOT + annot_paths[i])["box_coord"][0]

    topLeft_X = annot[2]
    topLeft_Y = annot[0]
    bottomRight_X = annot[3]
    bottomRight_Y = annot[1]

    image = tf.keras.utils.load_img(PATH_IMAGES + image_paths[i],)
    (w, h) = image.size[:2]

    # convert image to array. Store image to images list
    # and bounding box in targets list
    if i < int(len(annot_paths) * 0.8):
        # resize image if it is for training dataset
        image = image.resize((image_size, image_size))

    # append image to list
    images.append(tf.keras.utils.img_to_array(image))

    # apply relative scaling to bounding boxes as per given image and append to list
    targets.append(
        (
            float(topLeft_X) / w,
            float(topLeft_Y) / h,
            float(bottomRight_X) / w,
            float(bottomRight_Y) / h,
        )
    )

# Convert to numpy and split train and test dataset
(x_train), (y_train) = (
    np.asarray(images[: int(len(images) * 0.8)]),
    np.asarray(targets[: int(len(targets) * 0.8)]),
)
(x_test), (y_test) = (
    np.asarray(images[int(len(images) * 0.8) :]),
    np.asarray(targets[int(len(targets) * 0.8) :]),
)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

"""
## Implement multilayer-perceptron (MLP)
"""

# Some Code referred from https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


"""
## Implement patch creation layer
"""


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    #     Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
        return patches


"""
Display patches for a an input image
"""

patch_size = 32  # Size of the patches to be extracted from the input images

plt.figure(figsize=(4, 4))
plt.imshow(x_train[0].astype("uint8"))
plt.axis("off")

patches = Patches(patch_size)(tf.convert_to_tensor([x_train[0]]))
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

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


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    #     Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(
            positions
        )  # return encoded


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
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)

    bounding_box = tf.keras.layers.Dense(4)(
        features
    )  # Final four neurons that output bounding box

    # return Keras model.
    return tf.keras.Model(inputs=inputs, outputs=bounding_box)


"""
## Run experiment
"""


def run_experiment(model, learning_rate, weight_decay, batch_size, num_epochs):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    #     Comiple model.
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    checkpoint_filepath = "logs/"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback,
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )

    return history


input_shape = (image_size, image_size, 3)  # input image shape
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 100
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers


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

# Train model
history = run_experiment(
    vit_object_detector, learning_rate, weight_decay, batch_size, num_epochs
)


"""
## Model Evaluation
"""

import matplotlib.patches as patches

# To save the model in current path
vit_object_detector.save("vit_object_detector.h5", save_format="h5")

# Testing predictions


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


i, mIoU = 0, 0

# Compare results for last 10 images which were added to test set
for inputImage in x_test[:10]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    im = inputImage

    # Display the image
    ax1.imshow(im.astype("uint8"))
    ax2.imshow(im.astype("uint8"))

    inputImage = cv2.resize(
        inputImage, (image_size, image_size), interpolation=cv2.INTER_AREA
    )
    inputImage = np.expand_dims(inputImage, axis=0)
    preds = vit_object_detector.predict(inputImage)[0]

    (h, w) = (im).shape[0:2]
    topLeft_X = int(preds[0] * w)
    topLeft_Y = int(preds[1] * h)
    bottomRight_X = int(preds[2] * w)
    bottomRight_Y = int(preds[3] * h)

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

    topLeft_X = int(y_test[i][0] * w)
    topLeft_Y = int(y_test[i][1] * h)
    bottomRight_X = int(y_test[i][2] * w)
    bottomRight_Y = int(y_test[i][3] * h)

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

print("mIoU" + str(mIoU / len(x_test[:10])))
plt.show()

"""
Model can be improved further by tuning hyper-parameters and pre-training.
"""

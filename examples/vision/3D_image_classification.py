"""
Title: 3D Image Classification from CT Scans
Author: [Hasib Zunair](https://hasibzunair.github.io/)
Date created: 2020/09/10
Last modified: 2020/09/17
Description: Train a 3D convolutional neural network to classify location of cancer.
"""
"""
## Introduction

This example will show the steps needed to build a 3D image classification
model using a 3D convolutional neural network (CNN) to predict the location
of cancerous regions in CT scans.

## Why 3D CNNs at all?

Traditional use-cases of CNNs are RGB images (3 channels). The goal of
a 3D CNN is to take as input a 3D volume or a sequence of frames
(e.g. slices in a CT scan) and extract features from it. A traditional CNN
extract representations of a single image and puts them in a vector state
(latent space), whereas a 3D CNN extracts representations of a set of images
which is required to make predictions from volumetric data (e.g. CT scans).
A 3D CNN takes the temporal dimension (e.g. 3D context) into account. This way
of learning representations from a volumetric data is useful to find the right
label. This is achieved by using 3D convolutions.
"""

import os
import zipfile
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

"""
## Downloading the  NSCLC-Radiomics-Genomics dataset

Since training 3D convolutional neural network are time consuming, a subset of the
NSCLC-Radiomics-Genomics dataset is used which consists of CT scans
with gene expression and relevant clinical data. In this example, we will be
using the "location" attribtute among the available clinical data to build a
classifier to predict cancerious regions (left or right). Hence, the task is
a binary classification problem.
"""

url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.1/NSCLC-Radiomics-Genomics.zip"
filename = os.path.join(os.getcwd(), "NSCLC-Radiomics-Genomics.zip")
keras.utils.get_file(filename, url)

with zipfile.ZipFile("NSCLC-Radiomics-Genomics.zip", "r") as z_fp:
    z_fp.extractall("./")

"""
## Load data

The files are provided in Nifti format with the extension .nii. To read the
scans, the nibabel package is used. To process the data, the following
is done:

* Volumes are originally rotated by 90 degress, so the orientation is fixed
* Resize width, height and depth

Here several helper functions are defined to process the data. These functions
will be used when building training and test datasets.
"""

import numpy as np
import nibabel as nib
import cv2

from scipy.ndimage import zoom


def read_nifti_file(filepath):
    # read file
    scan = nib.load(filepath)
    # get raw data
    scan = scan.get_fdata()
    # rotate
    scan = np.rot90(np.array(scan))
    return scan


def resize_slices(img):
    # resize all slices
    flatten = [
        cv2.resize(img[:, :, i], (128, 128), interpolation=cv2.INTER_CUBIC)
        for i in range(img.shape[-1])
    ]
    # stack along the z-axis
    img = np.array(np.dstack(flatten))
    return img


def resize_depth(img):
    # set the desired depth
    desired_depth = 128
    # get current depth
    current_depth = img.shape[-1]
    # compute depth factor
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    # resize across z-axis
    img_new = zoom(img, (1, 1, depth_factor), mode="nearest")
    return img_new


def process_scan(path):
    # read scan
    volume = read_nifti_file(path)
    # resize width and height
    volume = resize_slices(volume)
    # resize across z-axis
    volume = resize_depth(volume)
    return volume


"""
Let's read the paths of the CT scans from the class directories
"""

# folder "1" consist of right CT scans
right_scan_paths = [
    os.path.join(os.getcwd(), "NSCLC-Radiomics-Genomics/1", x)
    for x in os.listdir("NSCLC-Radiomics-Genomics/1")
]
# folder "2" consist of left CT scans
left_scan_paths = [
    os.path.join(os.getcwd(), "NSCLC-Radiomics-Genomics/2", x)
    for x in os.listdir("NSCLC-Radiomics-Genomics/2")
]

print("CT scans with cancerous regions in left side: " + str(len(left_scan_paths)))
print("CT scans with cancerous regions in right side: " + str(len(right_scan_paths)))

"""
Let's visualize a CT scan and also look at it's dimensions.
"""

import matplotlib.pyplot as plt

img = read_nifti_file(right_scan_paths[2])
print("Dimension of the CT scan is:", img.shape)
plt.imshow(img[:, :, 5], cmap="gray")

"""
Since a CT scan has many slices, let's visualize a montage of the slices.
"""


def plot_slices(data):
    # plot a montage of 20 CT slices
    a, b = 2, 10
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (a, b, 512, 512))
    test_data = data
    r, c = test_data.shape[0], test_data.shape[1]

    heights = [a[0].shape[0] for a in test_data]
    widths = [a.shape[1] for a in test_data[0]]

    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)

    f, axarr = plt.subplots(
        r, c, figsize=(fig_width, fig_height), gridspec_kw={"height_ratios": heights}
    )

    for i in range(r):
        for j in range(c):
            axarr[i, j].imshow(test_data[i][j], cmap="gray")
            axarr[i, j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


plot_slices(img[:, :, :20])

"""
## Build train and test datasets

Read the scans from the class directories and assign labels.
Lastly, split the dataset into train and test subsets.
"""

# read and process the scans
right_scans = np.array([process_scan(path) for path in right_scan_paths])
left_scans = np.array([process_scan(path) for path in left_scan_paths])

# for the CT scans having cancerous regions in the right side
# assign 1, similarly for left assign 0.
right_labels = np.array([1 for _ in range(len(right_scans))])
left_labels = np.array([0 for _ in range(len(left_scans))])

# merge classes
inputs = np.concatenate((right_scans, left_scans), axis=0)
labels = np.concatenate((right_labels, left_labels), axis=0)
labels = keras.utils.to_categorical(labels, 2)
print("Size of raw data and labels:", inputs.shape, labels.shape)

# split data in the ratio 70-30 for training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    inputs, labels, stratify=labels, test_size=0.3, random_state=1
)
print(
    "After train and test split:",
    x_train.shape,
    y_train.shape,
    x_test.shape,
    y_test.shape,
)

"""
## Preprocessing and data augmentation

CT scans store raw voxel intensity in Hounsfield units (HU). They range from
-1024 to above 2000 in this dataset. Above 400 are bones with different
radiointensity, so this is used as a higher bound. A threshold between
-1000 and 400 are commonly used to normalize CT scans. The CT scans are
also augmented by rotating and blurring. There are different kinds of
preprocessing and augmentation techniques, this example shows a few
simple ones.
"""

import random

from scipy import ndimage
from scipy.ndimage import gaussian_filter


def normalize(array):
    min = -1000
    max = 400
    array = (array - min) / (max - min)
    array[array > 1] = 1.0
    array[array < 0] = 0.0
    return array


def rotate(volume):
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    volume = ndimage.rotate(volume, angle, reshape=False)
    return volume


def blur(volume):
    # gaussian blur
    volume = gaussian_filter(volume, sigma=1)
    return volume


def augment_data(volume, label, func_name):
    # augment the volume
    volume = func_name(volume)
    return volume, label


def get_augmented_data(inputs, labels):
    # augment all the input volumes and keep track of the labels
    # since the number of samples are small, load it once and keep it in memory
    x_train_aug = []
    y_train_aug = []

    # rotate all scans
    for volume, label in zip(inputs, labels):
        volume, label = augment_data(volume, label, rotate)
        volume = normalize(volume)
        x_train_aug.append(volume)
        y_train_aug.append(label)

    # gaussian Blur all scans
    for volume, label in zip(inputs, labels):
        volume, label = augment_data(volume, label, blur)
        volume = normalize(volume)
        x_train_aug.append(volume)
        y_train_aug.append(label)

    x_train_aug = np.array(x_train_aug)
    y_train_aug = np.array(y_train_aug)
    return x_train_aug, y_train_aug


"""
Augmented the training data
"""

x_train_aug, y_train_aug = get_augmented_data(x_train, y_train)
print(x_train_aug.shape, y_train_aug.shape)

"""
Normalize the training and test data subsets
"""

# Normalize data
x_train = np.array([normalize(x) for x in x_train])
x_test = np.array([normalize(x) for x in x_test])

"""
Visualize an augmented CT scan
"""

import matplotlib.pyplot as plt

img = x_train_aug[20]
print("Dimension of the CT scan is:", img.shape)
print(np.max(img), np.min(img))
plt.imshow(img[:, :, 80], cmap="gray")

"""
Merge the augmented and training data. Note that the augmented data is not normalize
here, as it was already normalized right after the augmentation.
"""

# Merge raw data and augmented data
x_all = np.concatenate((x_train_aug, x_train), axis=0)
x_all = np.expand_dims(x_all, axis=4)
y_all = np.concatenate((y_train_aug, y_train), axis=0)
print("Shape of upsampled training data inputs and labels: ", x_all.shape, y_all.shape)

"""
## Define 3D convolutional neural network

To make the model easier to understand, blocks are defined. Since this is a
3D CNN, 3D convolutions are used. The architecture of the 3D CNN used in this example
is based on this [paper](https://arxiv.org/abs/2007.13224).
"""


def get_model(img_size, num_classes):
    # build a 3D convolutional neural network model
    inputs = keras.Input(img_size)

    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(units=2, activation="softmax")(x)

    # define the model
    model = keras.Model(inputs, outputs)
    return model


# free up RAM
keras.backend.clear_session()

# build model
model = get_model((128, 128, 128, 1), 2)
model.summary()

"""
## Train model
"""

# compile model
# use the "binary_crossentropy"
# for the binary classification problem
model.compile(
    loss="binary_crossentropy", optimizer=keras.optimizers.Adam(1e-6), metrics=["acc"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint("3d_image_classification.h5", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_acc", patience=8),
]

# train the model, doing validation at the end of each epoch.
epochs = 100
model.fit(
    x_all,
    y_all,
    validation_data=(x_test, y_test),
    batch_size=2,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=callbacks,
)

"""
It is important to note that the number of sample are really small (only 40) and no
random seed is specified. You can expect variances in your results. It is also a good
exersize to try other parameters and see what works!
"""

"""
## Visualizing model performance

Here the model accuracy and loss for the training and the validation sets are plotted. Since
the test set is class-balanced, accuracy provides an unbiased representation of the
errors.
"""

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[met])
    ax[i].plot(model.history.history["val_" + met])
    ax[i].set_title("Model {}".format(met))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(met)
    ax[i].legend(["train", "val"])


"""
## Predict and evaluate results
"""

# load best weights
model.load_weights("3d_image_classification.h5")
# evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
# prin loss and accuracy on the test set
print("Test loss: %.2f" % (score[0]))
print("Test accuracy: %.2f" % (100 * score[1]))

"""
Make predictions on a single CT scan
"""

prediction = model.predict(np.expand_dims(x_test[0], axis=0))[0]
scores = [prediction[0], prediction[1]]

CLASS_NAMES = ["left", "right"]
for score, name in zip(scores, CLASS_NAMES):
    print(
        "This model is %.2f percent confident that cancer in the %s side"
        % ((100 * score), name)
    )

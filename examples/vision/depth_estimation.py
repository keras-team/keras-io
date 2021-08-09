"""
Title: An approach towards Depth Estimation with CNN
Author: [Victor Basu](https://www.linkedin.com/in/victor-basu-520958147)
Date created: 2021/08/09
Last modified: 2021/08/10
Description: Implement an depth estimation model with CNN.
"""
"""
Title: An approach towards Depth Estimation with CNN.

Author: [Victor Basu](https://www.linkedin.com/in/victor-basu-520958147)

Date created: 2021/08/08

Last modified: 2021/08/09

Description: Implement an depth estimation model with CNN.
"""

"""
## Introduction

**Depth Estimation** is a crucial step towards inferring scene geometry from 2D images.
The goal in monocular Depth Estimation is to predict the depth value of each pixel, given
only a single RGB image as input.

This is an approach to build a depth estimation model with CNN and basic loss functions.
While I was working on this topic I came across various research paper that explains
solving this problem and to be honest my solution is not as good as in those papers. I
would suggest you to go through those, just search depth estimation on
"paperwithcode.com".

![depth](https://paperswithcode.com/media/thumbnails/task/task-0000000605-d9849a91.jpg)


"""

"""
##Setup
"""

import os
import cv2
import csv
import logging
import sys

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import shuffle

tf.compat.v1.set_random_seed(123)
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
logging.disable(sys.maxsize)

"""
## Download the dataset

We will be using the **DIODE: A Dense Indoor and Outdoor Depth Dataset**  for this
tutorial. We have used the validation set for training and validating our model. The
reason we have used validation set and not training set of the orginal dataset because
the training set consist of 81GB data which was a bit difficult to download compared to
validation set which is only 2.6GB.
"""

annotation_folder = "/dataset/"
if not os.path.exists(os.path.abspath(".") + annotation_folder):
    annotation_zip = tf.keras.utils.get_file(
        "val.tar.gz",
        cache_subdir=os.path.abspath("."),
        origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
        extract=True,
    )

"""
##  Preparing the dataset
"""

path = "val"
# we shall store all the file names in this list
filelist = []

for root, dirs, files in os.walk(path):
    for file in files:
        # append the file name to the list
        filelist.append(os.path.join(root, file))

filelist.sort()
data = {
    "image": [x for x in filelist if x.endswith(".png")],
    "depth": [x for x in filelist if x.endswith("_depth.npy")],
    "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
}
df = pd.DataFrame(data)

df = shuffle(df)


class config:
    HEIGHT = 256
    WIDTH = 256
    LR = 1e-4
    EPOCHS = 30
    BATCH_SIZE = 32


"""
## Building Dataset Loader Pipeline

1. The pipeline takes dataframe mainintaing the path for RGB, depth and depth mask files
as input.
2. Reads and resize the RGB images
3. Reads the depth and depth_mask files, process them to generate the depth-map image and
resize it.
4. Returns the RGB images and the depth-map images for a batch.
"""


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=6, dim=(768, 1024), n_channels=3, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # Generate one batch of data
        # Generate indices of the batch
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        X, y = self.__data_generation(batch)

        return X, y

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __load__(self, image_path, depth_map, mask):
        "Load Target image"

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = (image_) / 255.0

        depth_map = np.load(depth_map).squeeze()

        mask = np.load(mask)
        mask = mask > 0

        MIN_DEPTH = 0.1

        MAX_DEPTH = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, MIN_DEPTH, MAX_DEPTH)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(MAX_DEPTH))
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis=2)

        # depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    def __data_generation(self, batch):

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, id_ in enumerate(batch):
            X[i,], y[i,] = self.__load__(
                self.data["image"][id_],
                self.data["depth"][id_],
                self.data["mask"][id_],
            )

        return X, y


"""
## Visualizing Samples
"""

visualize_samples = next(
    iter(DataGenerator(data=df, batch_size=6, dim=(config.HEIGHT, config.WIDTH)))
)
input, target = visualize_samples
cmap = plt.cm.jet
cmap.set_bad(color="black")
fig, ax = plt.subplots(6, 2, figsize=(50, 50))
for i in range(6):
    ax[i, 0].imshow((input[i].squeeze()))
    ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)

d = np.flipud(target[2].squeeze())
img = np.flipud(input[2].squeeze())

"""
## 3D Point-Cloud Visualization
"""

fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection="3d")

STEP = 2
for x in range(0, img.shape[0], STEP):
    for y in range(0, img.shape[1], STEP):
        ax.scatter([d[x, y]] * 3, [y] * 3, [x] * 3, c=tuple(img[x, y, :3] / 255), s=3)
    ax.view_init(15, 165)

"""
### angle-1
"""

ax.view_init(30, 135)
fig

"""
### angle-2
"""

ax.view_init(5, 100)
fig

"""
### angle-3
"""

ax.view_init(45, 220)
fig

"""
## Building the model
1. The basic model architecture has been taken from U-NET.
2. Residual-blocks has been used in the down-scale blocks of the U-NET architecture.
"""


class DownscaleBlock(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1):
        super().__init__()
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.pool = layers.MaxPool2D((2, 2), (2, 2))

    def call(self, input_tensor):
        d = self.convA(input_tensor)
        x = self.bn2a(d)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        x += d
        p = self.pool(x)
        return x, p


class UpscaleBlock(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1):
        super().__init__()
        self.us = layers.UpSampling2D((2, 2))
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conc = layers.Concatenate()

    def call(self, x, skip):
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convA(concat)
        x = self.bn2a(x)
        x = self.reluA(x)

        x = self.convB(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        return x


class BottleNeckBlock(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding="same", strides=1):
        super().__init__()
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x = self.convA(x)
        x = self.reluA(x)
        x = self.convB(x)
        x = self.reluB(x)
        return x


def UnetModel(height, width):

    f = [16, 32, 64, 128, 256]

    inputs = layers.Input((height, width, 3))

    c1, p1 = DownscaleBlock(f[0])(inputs)
    c2, p2 = DownscaleBlock(f[1])(p1)
    c3, p3 = DownscaleBlock(f[2])(p2)
    c4, p4 = DownscaleBlock(f[3])(p3)

    bn = BottleNeckBlock(f[4])(p4)

    u1 = UpscaleBlock(f[3])(bn, c4)
    u2 = UpscaleBlock(f[2])(u1, c3)
    u3 = UpscaleBlock(f[1])(u2, c2)
    u4 = UpscaleBlock(f[0])(u3, c1)

    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="tanh")(u4)

    model = tf.keras.Model(inputs, outputs)
    return model


"""
## Optimizing Loss
We have tried to optimize 3 losses in our model.
1. Structural similarity index(SSIM).
2. L1-loss, or Point-wise depth in our case.
3. Edge wide depth with depth smoothness.

Out of the three loss functions SSIM contributed the most in improving model performance.
"""


class DepthEstimationModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compile(self, optimizer):
        super(DepthEstimationModel, self).compile()
        self.optimizer = optimizer
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch_data):
        input, target = batch_data
        with tf.GradientTape() as tape:
            pred = self.model(input, training=True)

            # Edges
            dy_true, dx_true = tf.image.image_gradients(target)
            dy_pred, dx_pred = tf.image.image_gradients(pred)
            weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
            weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

            # depth_smoothness
            smoothness_x = dx_pred * weights_x
            smoothness_y = dy_pred * weights_y

            l_edges = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
                abs(smoothness_y)
            )

            # Structural similarity (SSIM) index
            l_ssim = tf.reduce_mean(
                1
                - tf.image.ssim(
                    target, pred, max_val=256, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2
                )
            )
            # Point-wise depth
            l1_loss = tf.reduce_mean(tf.abs(target - pred))

            # Weights
            w1 = 0.85
            w2 = 0.1
            w3 = 1.0

            loss = (w1 * l_ssim) + w2 * l1_loss + (w3 * l_edges)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.loss_metric.update_state(loss)
        return {
            "Loss": self.loss_metric.result(),
        }

    def call(self, x):
        pass


"""
## Model training
"""

optimizer = tf.keras.optimizers.Adam(
    learning_rate=config.LR,
    amsgrad=True,
)
model = UnetModel(config.HEIGHT, config.WIDTH)
DEM = DepthEstimationModel(model=model)
DEM.compile(optimizer)

train_loader = DataGenerator(
    data=df[:600].reset_index(drop="true"),
    batch_size=config.BATCH_SIZE,
    dim=(config.HEIGHT, config.WIDTH),
)
DEM.fit(train_loader, epochs=config.EPOCHS)

"""
## Visualizing Model output.
Visualizing model output over validation set.
The first image is the RGB image, the second image is the ground truth depth-map image
and the third one is the predicted depth-map image.
"""

test_loader = next(
    iter(
        DataGenerator(
            data=df[601:].reset_index(drop="true"),
            batch_size=6,
            dim=(config.HEIGHT, config.WIDTH),
        )
    )
)
input, target = test_loader
pred = model.predict(input)
cmap = plt.cm.jet
cmap.set_bad(color="black")
fig, ax = plt.subplots(6, 3, figsize=(50, 50))
for i in range(6):
    ax[i, 0].imshow((input[i].squeeze()))
    ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)
    ax[i, 2].imshow((pred[i].squeeze()), cmap=cmap)

test_loader = next(
    iter(
        DataGenerator(
            data=df[701:].reset_index(drop="true"),
            batch_size=6,
            dim=(config.HEIGHT, config.WIDTH),
        )
    )
)
input, target = test_loader
pred = model.predict(input)
cmap = plt.cm.jet
cmap.set_bad(color="black")
fig, ax = plt.subplots(6, 3, figsize=(50, 50))
for i in range(6):
    ax[i, 0].imshow((input[i].squeeze()))
    ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)
    ax[i, 2].imshow((pred[i].squeeze()), cmap=cmap)

"""
## Scopes of Improvement

1. From the research papers that I read while I was working on this topic the encode part
of the unet was replaced with DenseNet, ResNet or other pre-trained model, which could be
applied for better model predictions.

2. Loss functions plays an immense role in solving this problem, and different paper
explained different ways of developing the loss function out of which SSIM was common in
all. so playing with the loss fuctions gives a huge scope of improvement in this case.
"""

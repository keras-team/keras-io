"""
Title: Enhanced Deep Residual Networks for Single Image Super-Resolution
Author: Gitesh Chawda
Date created: 26-02-2022
Last modified: 26-02-2022
Description: Implementation of Enhanced Deep Residual Networks for Single Image Super-Resolution(EDSR).
"""
"""
This example demonstrates how to implement EDSR paper using keras and tensorflow. 
[EDSR Paper](https://arxiv.org/abs/1707.02921)
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

AUTOTUNE = tf.data.AUTOTUNE

import warnings

warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os

"""
# Data Agumentation
"""

# Fliping left and right
def flip_left_right(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(
        rn < 0.5,
        lambda: (lr_img, hr_img),
        lambda: (tf.image.flip_left_right(lr_img), tf.image.flip_left_right(hr_img)),
    )


# Rotating both images by 90 degree
def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


# Cropping high resolution image to 96x96 and low resolution to 24x24
def random_crop(lr_img, hr_img, hr_crop_size=96, scale=4):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_width = tf.random.uniform(
        shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32
    )
    lr_height = tf.random.uniform(
        shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32
    )

    hr_width = lr_width * scale
    hr_height = lr_height * scale

    lr_img_cropped = lr_img[
        lr_height : lr_height + lr_crop_size, lr_width : lr_width + lr_crop_size
    ]
    hr_img_cropped = hr_img[
        hr_height : hr_height + hr_crop_size, hr_width : hr_width + hr_crop_size
    ]

    return lr_img_cropped, hr_img_cropped


"""
# Loading Data using tensorflow datasets
"""

div2k_data = tfds.image.Div2k(config="bicubic_x4")
div2k_data.download_and_prepare()

train = div2k_data.as_dataset(split="train", as_supervised=True)
train_cache = train.cache()

ds = train_cache

ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=4), num_parallel_calls=AUTOTUNE)
ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)

ds = ds.batch(16)
ds = ds.repeat(None)
ds = ds.prefetch(buffer_size=AUTOTUNE)

lr, hr = next(iter(ds))

"""
# High Resolution Images
"""

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(hr[i].numpy().astype("uint8"))
    plt.title(hr[i].shape)
    plt.axis("off")

"""
# Low Resolution Images
"""

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(lr[i].numpy().astype("uint8"))
    plt.title(lr[i].shape)
    plt.axis("off")

"""
# Normalization and usefull functions
As, the paper mentioned we will subtract image by mean rgb of div2k dataset
"""

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x * 127.5) + rgb_mean


def shuffle_pixels(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


# computes the peak signal-to-noise ratio, measures quality of image
def PSNR(sr, hr):
    psnr_value = tf.image.psnr(hr, sr, max_val=255)[0]
    return psnr_value


"""
# Model Building(Baseline Model)
1. 16 residual blocks
2. filters=64
3. No batchNormalization
"""

"""
<img src = "https://miro.medium.com/max/1400/1*a1tQ4grku_ugTTe39IJzjA.png" width="500">
"""


def ResBlock(inputs):
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.Add()([inputs, x])
    return x


def Upsampling(inputs, factor=2, **kwargs):
    x = tf.keras.layers.Conv2D(64 * (factor**2), 3, padding="same", **kwargs)(inputs)
    x = tf.keras.layers.Lambda(shuffle_pixels(scale=factor))(x)
    x = tf.keras.layers.Conv2D(64 * (factor**2), 3, padding="same", **kwargs)(x)
    x = tf.keras.layers.Lambda(shuffle_pixels(scale=factor))(x)
    return x


def EDSR():
    input_layer = tf.keras.layers.Input(shape=(None, None, 3))
    x = tf.keras.layers.Lambda(normalize)(input_layer)
    x = x_new = tf.keras.layers.Conv2D(64, 3, padding="same")(x)

    # 16 residual blocks
    for _ in range(16):
        x_new = ResBlock(x_new)

    x_new = tf.keras.layers.Conv2D(64, 3, padding="same")(x_new)
    x = tf.keras.layers.Add()([x, x_new])

    x = Upsampling(x)
    x = tf.keras.layers.Conv2D(3, 3, padding="same")(x)
    output_layer = tf.keras.layers.Lambda(denormalize)(x)
    return tf.keras.models.Model(input_layer, output_layer)


edsr = EDSR()
edsr.summary()

"""
# Training
EDSR can take a long time to train, in the code below, but we will train for 5 epochs and
20 steps for checking if the code is working alright. In practice, a larger steps value
(over 200000) is required to get decent results.


paper Suggest that learning rate of adam optimizer will be halfed after 2 × 10^5 steps.
But as we are training for only 5 epochs we will change it to 50 steps.

"""

loss_fn = tf.keras.losses.MeanAbsoluteError()
optim_edsr = tf.keras.optimizers.Adam(
    learning_rate=PiecewiseConstantDecay(boundaries=[50], values=[1e-4, 5e-5])
)


@tf.function
def train_step(ds_low, ds_high):
    with tf.GradientTape() as EDSR:

        ds_low = tf.cast(ds_low, tf.float32)
        ds_high = tf.cast(ds_high, tf.float32)

        sr = edsr(ds_low, training=True)
        loss_value = loss_fn(ds_high, sr)

        # Calculating PSNR value
        psnr_value = PSNR(ds_high, sr)

    gradients = EDSR.gradient(loss_value, edsr.trainable_variables)
    optim_edsr.apply_gradients(zip(gradients, edsr.trainable_variables))

    return loss_value, psnr_value


"""
##### Training for 5*20 = 100 steps
"""

for epoch in range(5):

    for lr, hr in ds.take(20):
        loss_value, psnr_value = train_step(lr, hr)

    print(
        f"Epochs : {epoch}   ||   Loss : {loss_value:.5f}   ||   PSNR : {psnr_value:.5f}"
    )

"""
# Saving Model
"""

"""!mkdir model
edsr.save('saved_model/',save_format='tf')
edsr.save("saved_model/model.h5")"""

"""
# Testing Model
"""


def Predict(model, img):
    sr = model(tf.cast(tf.expand_dims(img, axis=0), tf.float32))
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.squeeze(tf.cast(sr, tf.uint8), axis=0)
    return sr


test = div2k_data.as_dataset(split="validation", as_supervised=True)
test_cache = test.cache()


def plot_results(x):
    plt.figure(figsize=(24, 14))
    plt.subplot(132), plt.imshow(x), plt.title("LR")
    plt.subplot(133), plt.imshow(Predict(edsr, x)), plt.title("Prediction")
    plt.show()


for lr, hr in test.take(5):
    lr = tf.image.random_crop(lr, (150, 150, 3))
    plot_results(lr)

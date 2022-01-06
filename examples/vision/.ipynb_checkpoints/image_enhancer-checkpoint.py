"""
Title: Image Enhancer/Denoiser
Author: [Anish B](https://twitter.com/anishhacko)
Date created: 2021/12/31
Last modified: 2022/01/05
Description: Implementation of Simple Image Enhancer/Denoiser for Document Image Enhancements.
"""
"""
## Introduction

This example signifies a "Image Enhancer/Denoiser" using UNet architecture with 
pretrained ResNet50 as encoder and Sub-Pixel CNN as a decoder, we have followed up on 
seminal papers as reference and used U-Net architecture with pretrained network(ResNet) 
as encoder and Efficient Sub-Pixel CNN as decoder. 
The main difference between Image Denoiser and Super-Resolution is Super-Resolution tries to 
upscale the image than the size of its input dimension enhancing feature representation 
but Image Denoiser tries to restore useful information with reasonable perceptual quality 
eliminating the observed noises. The problem we focus here is denoising the document images 
which are deteriorated through external degradations like blur,corrupt text blocks etc...

**References:**
- [Enhancing OCR Accuracy with Super
Resolution](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2018/ocr_Ankit_Lat_ICPR_2018.pdf)
Resolution](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2018/ocr_Ankit_Lat_ICPR_2018.pdf)
- [Improving the Perceptual Quality of Document
Images Using Deep Neural
Network](http://mile.ee.iisc.ac.in/publications/softCopy/DocumentAnalysis/ISNN_11page_65.pdf)
Network](http://mile.ee.iisc.ac.in/publications/softCopy/DocumentAnalysis/ISNN_11page_65.pdf)
"""

import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input

"""
## Dataset
we have created a synthetic document dataset that depicts several real world document
noises, we have used blurs and morphological filters to create real world noise
mainly.callbacks
"""

"""shell
!gdown https://drive.google.com/uc?id=18u1IlpaMuBtMgKMEYHLTZWxVYMA5RQE7
!unzip -q data.zip
"""

"""
## Dataset Preprocessing
"""

train_source = sorted(glob("data/source/*.jpg"))[:-60]
train_targets = sorted(glob("data/target/*.jpg"))[:-60]
val_source = sorted(glob("data/source/*.jpg"))[-60:]
val_targets = sorted(glob("data/target/*.jpg"))[-60:]

height, width = 352, 608
batch_size = 4


def resize_with_pad(image, height, width):
    return tf.image.resize_with_pad(
        image=image, target_height=height, target_width=width
    )


def data_preprocess(source, target):
    source_image = tf.io.decode_png(tf.io.read_file(source), channels=3)
    target_image = tf.io.decode_png(tf.io.read_file(target), channels=3)
    source_image = resize_with_pad(source_image, height=height, width=width)
    target_image = resize_with_pad(target_image, height=height, width=width)
    source_image = preprocess_input(source_image)
    target_image = preprocess_input(target_image)
    return source_image, target_image


buffer_size = int(len(train_source) * 0.2)
train_pipe = tf.data.Dataset.from_tensor_slices((train_source, train_targets))
train_pipe = train_pipe.map(data_preprocess, tf.data.AUTOTUNE).cache()
train_pipe = train_pipe.shuffle(buffer_size).batch(batch_size).repeat()

val_pipe = tf.data.Dataset.from_tensor_slices((val_source, val_targets))
val_pipe = val_pipe.map(data_preprocess, tf.data.AUTOTUNE).cache()
val_pipe = val_pipe.shuffle(buffer_size).batch(batch_size).repeat()


train_step = len(train_source) // batch_size
val_step = len(val_source) // batch_size

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax = ax.flatten()
for i, f in enumerate(val_pipe.take(1)):
    noise, clean = f
    noise = noise[i].numpy() + 1.0
    noise = noise * 127.5
    clean = clean[i].numpy() + 1.0
    clean = clean * 127.5
    ax[0].imshow(noise.astype(np.int))
    ax[0].set_xlabel("source")
    ax[1].imshow(clean.astype(np.int))
    ax[1].set_xlabel("target")
fig.text(x=0.45, y=0.78, s="Dataset Visualization", fontweight="bold")
fig.set_tight_layout(None)

"""
## Model
"""


def conv_unit(x, filters=64, padding="same"):
    x = keras.layers.Conv2D(filters=filters, kernel_size=3, padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def subpixel_upsample(x, target_channels, upscale_ratio):
    estimated_filters = target_channels * (upscale_ratio ** 2)
    x = keras.layers.Conv2D(estimated_filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    x = tf.nn.depth_to_space(x, block_size=upscale_ratio)
    return x


def upscale_unit(previous_input, current_input, target_channels=3, upscale_ratio=2):
    previous_low_scale = subpixel_upsample(
        previous_input, target_channels=target_channels, upscale_ratio=upscale_ratio
    )
    current_high_scale = conv_unit(current_input, filters=target_channels)
    concat_scales = tf.concat((previous_low_scale, current_high_scale), axis=-1)
    feature_manifold = conv_unit(concat_scales, filters=target_channels)
    return feature_manifold


def denoiser():
    base_model = ResNet50V2(
        include_top=False, weights="imagenet", input_shape=(352, 608, 3)
    )
    d0 = base_model.get_layer("input_1").output
    d1 = base_model.get_layer("conv1_conv").output
    d2 = base_model.get_layer("conv2_block3_1_relu").output
    d3 = base_model.get_layer("conv3_block4_1_relu").output

    decode1 = upscale_unit(previous_input=d3, current_input=d2, target_channels=128)
    decode2 = upscale_unit(previous_input=decode1, current_input=d1, target_channels=64)
    decode3 = upscale_unit(previous_input=decode2, current_input=d0, target_channels=32)

    out_conv = keras.layers.Conv2D(3, 5, padding="same", activation="sigmoid")(decode3)
    denoiser_model = keras.Model(base_model.input, out_conv, name="denoiser")
    return denoiser_model


def infer_results(file):
    source_image = tf.io.decode_png(tf.io.read_file(file), channels=3)
    source_image = resize_with_pad(source_image, height=height, width=width)
    source_image = preprocess_input(source_image)
    ground_truth = source_image + 1.0
    ground_truth *= 127.5
    ground_truth = tf.keras.preprocessing.image.array_to_img(ground_truth)
    source_image = tf.expand_dims(source_image, axis=0)
    prediction = denoiser_net.predict(source_image)
    prediction = prediction[0] + 1.0
    prediction *= 127.5
    prediction = tf.keras.preprocessing.image.array_to_img(prediction)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    axes[0].imshow(ground_truth)
    axes[0].set_xlabel("input")
    axes[1].imshow(prediction)
    axes[1].set_xlabel("prediction")


"""
## Training
"""

denoiser_net = denoiser()
optimizer = tf.keras.optimizers.Nadam(1e-4)
denoiser_net.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

early_stop = callbacks.EarlyStopping(patience=7)
ckpt = callbacks.ModelCheckpoint(
    "weight_epoch{epoch}.h5", save_best_only=True, save_weights_only=True, verbose=1
)

denoiser_net.fit(
    train_pipe,
    validation_data=val_pipe,
    batch_size=batch_size,
    epochs=20,
    steps_per_epoch=train_step,
    validation_steps=val_step,
    workers=-1,
    callbacks=[early_stop, ckpt],
)

"""
## Testing
"""

test_images = [np.random.choice(val_source) for i in range(4)]
for f in test_images:
    infer_results(f)

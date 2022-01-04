"""
Title: Image Enhancer/Denoiser using ESPCN
Author: Anish B
Date created: 31/12/2021
Last modified: 05/01/2022
Description: Implementation of Simple Image Enhancer/Denoiser using Sub-Pixel CNN using U-Net Architecture.
"""
import numpy as np
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

"""shell
!gdown https://drive.google.com/uc?id=18u1IlpaMuBtMgKMEYHLTZWxVYMA5RQE7
!unzip -q data.zip
"""

train_source = sorted(glob("data/source/*.jpg"))[:-60]
train_targets = sorted(glob("data/target/*.jpg"))[:-60]

val_source = sorted(glob("data/source/*.jpg"))[-60:]
val_targets = sorted(glob("data/target/*.jpg"))[-60:]

height, width = 352, 608


def resize_with_pad(image, h, w):
    return tf.image.resize_with_pad(image=image, target_height=h, target_width=w)


def data_preprocess(source, target):
    source_image = tf.io.decode_png(tf.io.read_file(source), channels=3)
    target_image = tf.io.decode_png(tf.io.read_file(target), channels=3)
    source_image = resize_with_pad(source_image, h=height, w=width) / 255.0
    target_image = resize_with_pad(target_image, h=height, w=width) / 255.0
    return source_image, target_image


batch_size = 4
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
    noise = noise[i].numpy() * 255
    clean = clean[i].numpy() * 255
    ax[0].imshow(noise.astype(np.int))
    ax[0].set_xlabel("source")
    ax[1].imshow(clean.astype(np.int))
    ax[1].set_xlabel("target")
fig.text(x=0.45, y=0.78, s="Dataset Visualization", fontweight="bold")
fig.set_tight_layout(None)


def conv_unit(x, filters=64, padding="same"):
    x = layers.Conv2D(filters=filters, kernel_size=3, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def subpixel_upsample(x, target_channels, upscale_ratio):
    estimated_filters = target_channels * (upscale_ratio ** 2)
    x = layers.Conv2D(estimated_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
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
    base_model = ResNet50V2(include_top=False, input_shape=(352, 608, 3))
    d0 = base_model.get_layer("input_1").output
    d1 = base_model.get_layer("conv1_conv").output
    d2 = base_model.get_layer("conv2_block3_1_relu").output
    d3 = base_model.get_layer("conv3_block4_1_relu").output

    decode1 = upscale_unit(previous_input=d3, current_input=d2, target_channels=128)
    decode2 = upscale_unit(previous_input=decode1, current_input=d1, target_channels=64)
    decode3 = upscale_unit(previous_input=decode2, current_input=d0, target_channels=32)

    out_conv = layers.Conv2D(3, 5, padding="same", activation="sigmoid")(decode3)
    denoiser_model = Model(base_model.input, out_conv, name="denoiser")
    return denoiser_model


def infer_results(file):
    source_image = tf.io.decode_png(tf.io.read_file(file), channels=3)
    source_image = resize_with_pad(source_image, h=height, w=width) / 255.0
    gt = tf.keras.preprocessing.image.array_to_img(source_image * 255.0)
    source_image = tf.expand_dims(source_image, axis=0)
    prediction = denoiser_net.predict(source_image)
    prediction = tf.keras.preprocessing.image.array_to_img(prediction[0] * 255.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    axes[0].imshow(gt)
    axes[0].set_xlabel("input")
    axes[1].imshow(prediction)
    axes[1].set_xlabel("prediction")


denoiser_net = denoiser()
optimizer = tf.keras.optimizers.Nadam(1e-4)
denoiser_net.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

early_stop = EarlyStopping(patience=10)
ckpt = ModelCheckpoint(
    "best_weight.h5", save_best_only=True, save_weights_only=True, verbose=1
)

denoiser_net.fit(
    train_pipe,
    validation_data=val_pipe,
    batch_size=batch_size,
    epochs=18,
    steps_per_epoch=train_step,
    validation_steps=val_step,
    workers=-1,
    callbacks=[early_stop, ckpt],
)

test_images = [np.random.choice(val_source) for i in range(4)]
for f in test_images:
    infer_results(f)

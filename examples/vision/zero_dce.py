"""
Title: Zero-DCE for low-light image enhancement
Author: [Soumik Rakshit](http://github.com/soumik12345)
Date created: 2021/09/18
Last modified: 2021/09/18
Description: Implementing Zero-Reference Deep Curve Estimation for low-light image enhancement
"""
"""
## Introduction

**Zero-Reference Deep Curve Estimation** or **Zero-DCE** formulates light
enhancement as a task of image-specific curve estimation with a deep neural network. In
this example, we would train a lightweight deep network, **DCE-Net**, to estimate
pixel-wise and high-order curves for dynamic range adjustment of a given image. The curve
estimation is specially designed, considering pixel value range, monotonicity, and
differentiability. Zero-DCE
is appealing in its relaxed assumption on reference images, i.e., it does not require any
paired or unpaired data during training. This is achieved through a set of carefully
formulated non-reference loss functions, which implicitly
measure the enhancement quality and drive the learning of the network.

### References:

- [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://arxiv.org/abs/2001.06826)
- [A spatial processor model for object colour perception](https://www.sciencedirect.com/science/article/abs/pii/0016003280900587)
"""

"""
## Downloading LOLDataset

The **LoL Dataset** has been created for low-light image enhancement. It provides 485
images for training and 15 for testing. Each image pair in the dataset consists of a
low-light input image and its corresponding well-exposed reference image.
"""

import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""shell
!gdown https://drive.google.com/uc?id=1DdGIJ4PZPlF2ikl8mNM9V-PdVxVLbQi6
!unzip -q lol_dataset.zip
"""

"""
## Creating a TensorFlow Dataset

We use 300 low-light images from the LoL Dataset's training set for training, and we use
the remaining 185 low-light images for validation. We resize the images to size 512 x 512
to be used for both training and validation. Note that in order to train the DCE-Net, we
will not require the corresponding enhanced images.
"""

IMAGE_SIZE = 512
BATCH_SIZE = 8
MAX_TRAIN_IMAGES = 300


def load_data(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255.0
    return image


def data_generator(low_light_images):
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


train_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[:MAX_TRAIN_IMAGES]
val_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[MAX_TRAIN_IMAGES:]
test_low_light_images = sorted(glob("./lol_dataset/eval15/low/*"))


train_dataset = data_generator(train_low_light_images)
val_dataset = data_generator(val_low_light_images)

print("Train Dataset:", train_dataset)
print("Validation Dataset:", val_dataset)

"""
## The Zero-DCE Framework

The DCE-Net is devised to estimate a set of best-fitting Light-Enhancement curves
(LE-curves) given an input image. The framework then maps all pixels of the input’s RGB
channels by applying the curves iteratively for obtaining the final enhanced image.

### Light-Enhancement Curve

It is a kind of curve that can map a low-light image to its enhanced version
automatically, where the self-adaptive curve parameters are solely dependent on the input
image. There are three objectives
in the design of such a curve:
- Each pixel value of the enhanced image should be in the normalized range of `[0,1]` to
avoid information loss induced by overflow truncation.
- It should be monotonous to preserve the differences (contrast) of neighboring pixels.
- The form of this curve should be as simple as possible and differentiable in the
process of gradient backpropagation.

The Light-Enhancement Curve is seperately applied to three RGB channels
instead of solely on the illumination channel. The three-channel adjustment can better
preserve the inherent color and reduce the risk of over-saturation. The Light-Enhancement
Curve can be applied iteratively to enable more versatile adjustment to cope with
challenging low-light conditions.

![](https://li-chongyi.github.io/Zero-DCE_files/framework.png)

### DCE-Net

The DCE-Net is a lightweight deep neural network that learns the mapping between an input
image and its best-fitting curve parameter maps. The input to the DCE-Net is a low-light
image while the outputs are a set of pixel-wise curve parameter maps for corresponding
higher-order curves. It is a plain CNN of seven convolutional layers with symmetrical
concatenation. Each layer consists of 32 convolutional kernels of size 3×3 and stride 1
followed by the ReLU activation function. The last convolutional layer is followed by the
Tanh activation function, which produces 24 parameter maps for 8 iterations, where each
iteration requires three curve parameter maps for the three channels.

![](https://i.imgur.com/HtIg34W.png)
"""


def build_dce_net():
    input_img = keras.Input(shape=[None, None, 3])
    conv1 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(input_img)
    conv2 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv1)
    conv3 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv2)
    conv4 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(
        32, (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(
        int_con3
    )
    return keras.Model(inputs=input_img, outputs=x_r)


"""
## Loss Functions

To enable zero-reference learning in DCE-Net, we would use a set of differentiable
non-reference losses that allow us to evaluate the quality of enhanced images.
"""

"""
### Color Constancy Loss

The Color Constancy Loss is used to correct the potential color deviations in the
enhanced image and also build the relations among the three adjusted channels.
"""


def color_constancy_loss(x):
    mean_rgb = tf.reduce_mean(x, [1, 2], keepdims=True)
    mr, mg, mb = mean_rgb[:, :, :, 0], mean_rgb[:, :, :, 1], mean_rgb[:, :, :, 2]
    d_rg = tf.pow(mr - mg, 2)
    d_rb = tf.pow(mr - mb, 2)
    d_gb = tf.pow(mb - mg, 2)
    return tf.pow(tf.pow(d_rg, 2) + tf.pow(d_rb, 2) + tf.pow(d_gb, 2), 0.5)


"""
### Exposure loss

To restrain under-/over-exposed regions, the Exposure Control Loss is used. The exposure
control loss measures the distance between the average intensity value of a local region
to the well-exposedness level which is set to `0.6`.
"""


def exposure_loss(x, mean_val=0.6):
    x = tf.reduce_mean(x, 3, keepdims=True)
    mean = tf.keras.layers.AveragePooling2D(pool_size=16, strides=16)(x)
    return tf.reduce_mean(tf.pow(mean - mean_val, 2))


"""
### Illumination Smoothness Loss

To preserve the monotonicity relations between neighboring pixels, the Illumination
Smoothness Loss is added to each curve parameter map.
"""


def illumination_smoothness_loss(x):
    batch_size = tf.shape(x)[0]
    h_x = tf.shape(x)[1]
    w_x = tf.shape(x)[2]
    count_h = (tf.shape(x)[2] - 1) * tf.shape(x)[3]
    count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)
    h_tv = tf.reduce_sum(tf.pow((x[:, 1:, :, :] - x[:, : h_x - 1, :, :]), 2))
    w_tv = tf.reduce_sum(tf.pow((x[:, :, 1:, :] - x[:, :, : w_x - 1, :]), 2))
    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


"""
### Spatial Consistancy Loss

The spatial consistency loss encourages spatial coherence of the enhanced image
through preserving the difference of neighboring regions between the input image and its
enhanced version.
"""


class SpatialConsistancyLoss:
    def __init__(self):

        self.left_kernel = tf.constant(
            [[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.right_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.up_kernel = tf.constant(
            [[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.down_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32
        )

        self.pool = tf.keras.layers.AveragePooling2D(pool_size=4)

    def __call__(self, original, enhanced):

        original_mean = tf.reduce_mean(original, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(enhanced, 3, keepdims=True)
        original_pool = self.pool(original_mean)
        enhanced_pool = self.pool(enhanced_mean)

        d_original_left = tf.nn.conv2d(
            original_pool, self.left_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_original_right = tf.nn.conv2d(
            original_pool, self.right_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_original_up = tf.nn.conv2d(
            original_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_original_down = tf.nn.conv2d(
            original_pool, self.down_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )

        d_enhanced_left = tf.nn.conv2d(
            enhanced_pool, self.left_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_enhanced_right = tf.nn.conv2d(
            enhanced_pool, self.right_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_enhanced_up = tf.nn.conv2d(
            enhanced_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_enhanced_down = tf.nn.conv2d(
            enhanced_pool, self.down_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )

        d_left = tf.pow(d_original_left - d_enhanced_left, 2)
        d_right = tf.pow(d_original_right - d_enhanced_right, 2)
        d_up = tf.pow(d_original_up - d_enhanced_up, 2)
        d_down = tf.pow(d_original_down - d_enhanced_down, 2)
        return d_left + d_right + d_up + d_down


"""
### Deep Curve Estimation Model

We implement the Zero-DCE framework as a Keras subclassed model.
"""


class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super(ZeroDCE, self).__init__(**kwargs)
        self.dce_model = build_dce_net()

    def compile(self, learning_rate, **kwargs):
        super(ZeroDCE, self).compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistancyLoss()

    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.pow(data, 2) - data)
        x = x + r2 * (tf.pow(x, 2) - x)
        x = x + r3 * (tf.pow(x, 2) - x)
        enhanced_image = x + r4 * (tf.pow(x, 2) - x)
        x = enhanced_image + r5 * (tf.pow(enhanced_image, 2) - enhanced_image)
        x = x + r6 * (tf.pow(x, 2) - x)
        x = x + r7 * (tf.pow(x, 2) - x)
        enhanced_image = x + r8 * (tf.pow(x, 2) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def _calculate_losses(self, data, output):
        output = self.dce_model(data)
        enhanced_image = self.get_enhanced_image(data, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(
            self.spatial_constancy_loss(enhanced_image, data)
        )
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        total_loss = (
            loss_illumination
            + loss_spatial_constancy
            + loss_color_constancy
            + loss_exposure
        )
        return {
            "total_loss": total_loss,
            "illumination_smoothness_loss": loss_illumination,
            "spatial_constancy_loss": loss_spatial_constancy,
            "color_constancy_loss": loss_color_constancy,
            "exposure_loss": loss_exposure,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self._calculate_losses(data, output)
        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))

        return losses

    def test_step(self, data):
        output = self.dce_model(data)
        return self._calculate_losses(data, output)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.dce_model.save_weights(
            filepath, overwrite=overwrite, save_format=save_format, options=options
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.dce_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )


"""
## Training
"""

zero_dce_model = ZeroDCE()
zero_dce_model.compile(learning_rate=1e-4)
history = zero_dce_model.fit(train_dataset, validation_data=val_dataset, epochs=100)


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("total_loss")
plot_result("illumination_smoothness_loss")
plot_result("spatial_constancy_loss")
plot_result("color_constancy_loss")
plot_result("exposure_loss")

"""
## Inference
"""


def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()


def infer(original_image):
    image = tf.keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_dce_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image


"""
### Inference on Test Images

We compare the test images from LOLDataset enhanced by MIRNet with images enhanced via
the `PIL.ImageOps.autocontrast()` function.
"""

for val_image_file in random.sample(test_low_light_images, 6):
    original_image = Image.open(val_image_file)
    enhanced_image = infer(original_image)
    plot_results(
        [original_image, ImageOps.autocontrast(original_image), enhanced_image],
        ["Original", "PIL Autocontrast", "Enhanced"],
        (20, 12),
    )

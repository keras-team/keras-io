# Zero-DCE for low-light image enhancement

**Author:** [Soumik Rakshit](http://github.com/soumik12345)<br>
**Date created:** 2021/09/18<br>
**Last modified:** 2021/09/19<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/zero_dce.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/zero_dce.py)


**Description:** Implementing Zero-Reference Deep Curve Estimation for low-light image enhancement.

---
## Introduction

**Zero-Reference Deep Curve Estimation** or **Zero-DCE** formulates low-light image
enhancement as the task of estimating an image-specific
[*tonal curve*](https://en.wikipedia.org/wiki/Curve_(tonality)) with a deep neural network.
In this example, we train a lightweight deep network, **DCE-Net**, to estimate
pixel-wise and high-order tonal curves for dynamic range adjustment of a given image.

Zero-DCE takes a low-light image as input and produces high-order tonal curves as its output.
These curves are then used for pixel-wise adjustment on the dynamic range of the input to
obtain an enhanced image. The curve estimation process is done in such a way that it maintains
the range of the enhanced image and preserves the contrast of neighboring pixels. This
curve estimation is inspired by curves adjustment used in photo editing software such as
Adobe Photoshop where users can adjust points throughout an image’s tonal range.

Zero-DCE is appealing because of its relaxed assumptions with regard to reference images:
it does not require any input/output image pairs during training.
This is achieved through a set of carefully formulated non-reference loss functions,
which implicitly measure the enhancement quality and guide the training of the network.

### References

- [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://arxiv.org/pdf/2001.06826.pdf)
- [Curves adjustment in Adobe Photoshop](https://helpx.adobe.com/photoshop/using/curves-adjustment.html)

---
## Downloading LOLDataset

The **LoL Dataset** has been created for low-light image enhancement. It provides 485
images for training and 15 for testing. Each image pair in the dataset consists of a
low-light input image and its corresponding well-exposed reference image.


```python
import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```


```python
!gdown https://drive.google.com/uc?id=1DdGIJ4PZPlF2ikl8mNM9V-PdVxVLbQi6
!unzip -q lol_dataset.zip
```

<div class="k-default-codeblock">
```
Downloading...
From: https://drive.google.com/uc?id=1DdGIJ4PZPlF2ikl8mNM9V-PdVxVLbQi6
To: /content/keras-io/scripts/tmp_4644685/lol_dataset.zip
347MB [00:03, 93.3MB/s]

```
</div>
---
## Creating a TensorFlow Dataset

We use 300 low-light images from the LoL Dataset training set for training, and we use
the remaining 185 low-light images for validation. We resize the images to size `256 x
256` to be used for both training and validation. Note that in order to train the DCE-Net,
we will not require the corresponding enhanced images.


```python
IMAGE_SIZE = 256
BATCH_SIZE = 16
MAX_TRAIN_IMAGES = 400


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
```

<div class="k-default-codeblock">
```
Train Dataset: <BatchDataset shapes: (16, 256, 256, 3), types: tf.float32>
Validation Dataset: <BatchDataset shapes: (16, 256, 256, 3), types: tf.float32>

```
</div>
---
## The Zero-DCE Framework

The goal of DCE-Net is to estimate a set of best-fitting light-enhancement curves
(LE-curves) given an input image. The framework then maps all pixels of the input’s RGB
channels by applying the curves iteratively to obtain the final enhanced image.

### Understanding light-enhancement curves

A ligh-enhancement curve is a kind of curve that can map a low-light image
to its enhanced version automatically,
where the self-adaptive curve parameters are solely dependent on the input image.
When designing such a curve, three objectives should be taken into account:

- Each pixel value of the enhanced image should be in the normalized range `[0,1]`, in order to
avoid information loss induced by overflow truncation.
- It should be monotonous, to preserve the contrast between neighboring pixels.
- The shape of this curve should be as simple as possible,
and the curve should be differentiable to allow backpropagation.

The light-enhancement curve is separately applied to three RGB channels instead of solely on the
illumination channel. The three-channel adjustment can better preserve the inherent color and reduce
the risk of over-saturation.

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


```python

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

```

---
## Loss functions

To enable zero-reference learning in DCE-Net, we use a set of differentiable
zero-reference losses that allow us to evaluate the quality of enhanced images.

### Color constancy loss

The *color constancy loss* is used to correct the potential color deviations in the
enhanced image.


```python

def color_constancy_loss(x):
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    mr, mg, mb = mean_rgb[:, :, :, 0], mean_rgb[:, :, :, 1], mean_rgb[:, :, :, 2]
    d_rg = tf.square(mr - mg)
    d_rb = tf.square(mr - mb)
    d_gb = tf.square(mb - mg)
    return tf.sqrt(tf.square(d_rg) + tf.square(d_rb) + tf.square(d_gb))

```

### Exposure loss

To restrain under-/over-exposed regions, we use the *exposure control loss*.
It measures the distance between the average intensity value of a local region
and a preset well-exposedness level (set to `0.6`).


```python

def exposure_loss(x, mean_val=0.6):
    x = tf.reduce_mean(x, axis=3, keepdims=True)
    mean = tf.nn.avg_pool2d(x, ksize=16, strides=16, padding="VALID")
    return tf.reduce_mean(tf.square(mean - mean_val))

```

### Illumination smoothness loss

To preserve the monotonicity relations between neighboring pixels, the
*illumination smoothness loss* is added to each curve parameter map.


```python

def illumination_smoothness_loss(x):
    batch_size = tf.shape(x)[0]
    h_x = tf.shape(x)[1]
    w_x = tf.shape(x)[2]
    count_h = (tf.shape(x)[2] - 1) * tf.shape(x)[3]
    count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)
    h_tv = tf.reduce_sum(tf.square((x[:, 1:, :, :] - x[:, : h_x - 1, :, :])))
    w_tv = tf.reduce_sum(tf.square((x[:, :, 1:, :] - x[:, :, : w_x - 1, :])))
    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

```

### Spatial consistency loss

The *spatial consistency loss* encourages spatial coherence of the enhanced image by
preserving the contrast between neighboring regions across the input image and its enhanced version.


```python

class SpatialConsistencyLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SpatialConsistencyLoss, self).__init__(reduction="none")

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

    def call(self, y_true, y_pred):

        original_mean = tf.reduce_mean(y_true, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(y_pred, 3, keepdims=True)
        original_pool = tf.nn.avg_pool2d(
            original_mean, ksize=4, strides=4, padding="VALID"
        )
        enhanced_pool = tf.nn.avg_pool2d(
            enhanced_mean, ksize=4, strides=4, padding="VALID"
        )

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

        d_left = tf.square(d_original_left - d_enhanced_left)
        d_right = tf.square(d_original_right - d_enhanced_right)
        d_up = tf.square(d_original_up - d_enhanced_up)
        d_down = tf.square(d_original_down - d_enhanced_down)
        return d_left + d_right + d_up + d_down

```

### Deep curve estimation model

We implement the Zero-DCE framework as a Keras subclassed model.


```python

class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super(ZeroDCE, self).__init__(**kwargs)
        self.dce_model = build_dce_net()

    def compile(self, learning_rate, **kwargs):
        super(ZeroDCE, self).compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction="none")

    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = data + r1 * (tf.square(data) - data)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, data):
        dce_net_output = self.dce_model(data)
        return self.get_enhanced_image(data, dce_net_output)

    def compute_losses(self, data, output):
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
            losses = self.compute_losses(data, output)
        gradients = tape.gradient(
            losses["total_loss"], self.dce_model.trainable_weights
        )
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))
        return losses

    def test_step(self, data):
        output = self.dce_model(data)
        return self.compute_losses(data, output)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """While saving the weights, we simply save the weights of the DCE-Net"""
        self.dce_model.save_weights(
            filepath, overwrite=overwrite, save_format=save_format, options=options
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """While loading the weights, we simply load the weights of the DCE-Net"""
        self.dce_model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )

```

---
## Training


```python
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
```

<div class="k-default-codeblock">
```
Epoch 1/100
25/25 [==============================] - 13s 271ms/step - total_loss: 4.8773 - illumination_smoothness_loss: 1.9298 - spatial_constancy_loss: 4.2610e-06 - color_constancy_loss: 0.0027 - exposure_loss: 2.9448 - val_total_loss: 4.3163 - val_illumination_smoothness_loss: 1.3040 - val_spatial_constancy_loss: 1.4072e-06 - val_color_constancy_loss: 5.3277e-04 - val_exposure_loss: 3.0117
Epoch 2/100
25/25 [==============================] - 7s 270ms/step - total_loss: 4.1537 - illumination_smoothness_loss: 1.2237 - spatial_constancy_loss: 6.5297e-06 - color_constancy_loss: 0.0027 - exposure_loss: 2.9273 - val_total_loss: 3.8239 - val_illumination_smoothness_loss: 0.8263 - val_spatial_constancy_loss: 1.3503e-05 - val_color_constancy_loss: 4.8064e-04 - val_exposure_loss: 2.9971
Epoch 3/100
25/25 [==============================] - 7s 270ms/step - total_loss: 3.7458 - illumination_smoothness_loss: 0.8320 - spatial_constancy_loss: 2.9476e-05 - color_constancy_loss: 0.0028 - exposure_loss: 2.9110 - val_total_loss: 3.5389 - val_illumination_smoothness_loss: 0.5565 - val_spatial_constancy_loss: 4.3614e-05 - val_color_constancy_loss: 4.4507e-04 - val_exposure_loss: 2.9818
Epoch 4/100
25/25 [==============================] - 7s 271ms/step - total_loss: 3.4913 - illumination_smoothness_loss: 0.5945 - spatial_constancy_loss: 7.2733e-05 - color_constancy_loss: 0.0029 - exposure_loss: 2.8939 - val_total_loss: 3.3690 - val_illumination_smoothness_loss: 0.4014 - val_spatial_constancy_loss: 8.7945e-05 - val_color_constancy_loss: 4.3541e-04 - val_exposure_loss: 2.9671
Epoch 5/100
25/25 [==============================] - 7s 271ms/step - total_loss: 3.3210 - illumination_smoothness_loss: 0.4399 - spatial_constancy_loss: 1.2652e-04 - color_constancy_loss: 0.0030 - exposure_loss: 2.8781 - val_total_loss: 3.2557 - val_illumination_smoothness_loss: 0.3019 - val_spatial_constancy_loss: 1.3960e-04 - val_color_constancy_loss: 4.4128e-04 - val_exposure_loss: 2.9533
Epoch 6/100
25/25 [==============================] - 7s 272ms/step - total_loss: 3.1971 - illumination_smoothness_loss: 0.3310 - spatial_constancy_loss: 1.8674e-04 - color_constancy_loss: 0.0031 - exposure_loss: 2.8628 - val_total_loss: 3.1741 - val_illumination_smoothness_loss: 0.2338 - val_spatial_constancy_loss: 1.9747e-04 - val_color_constancy_loss: 4.5618e-04 - val_exposure_loss: 2.9397
Epoch 7/100
25/25 [==============================] - 7s 263ms/step - total_loss: 3.1008 - illumination_smoothness_loss: 0.2506 - spatial_constancy_loss: 2.5713e-04 - color_constancy_loss: 0.0032 - exposure_loss: 2.8468 - val_total_loss: 3.1062 - val_illumination_smoothness_loss: 0.1804 - val_spatial_constancy_loss: 2.6610e-04 - val_color_constancy_loss: 4.7632e-04 - val_exposure_loss: 2.9251
Epoch 8/100
25/25 [==============================] - 7s 272ms/step - total_loss: 3.0244 - illumination_smoothness_loss: 0.1915 - spatial_constancy_loss: 3.4287e-04 - color_constancy_loss: 0.0033 - exposure_loss: 2.8293 - val_total_loss: 3.0512 - val_illumination_smoothness_loss: 0.1415 - val_spatial_constancy_loss: 3.5449e-04 - val_color_constancy_loss: 5.0079e-04 - val_exposure_loss: 2.9088
Epoch 9/100
25/25 [==============================] - 7s 272ms/step - total_loss: 2.9666 - illumination_smoothness_loss: 0.1531 - spatial_constancy_loss: 4.5557e-04 - color_constancy_loss: 0.0035 - exposure_loss: 2.8096 - val_total_loss: 3.0084 - val_illumination_smoothness_loss: 0.1172 - val_spatial_constancy_loss: 4.7605e-04 - val_color_constancy_loss: 5.3119e-04 - val_exposure_loss: 2.8902
Epoch 10/100
25/25 [==============================] - 7s 263ms/step - total_loss: 2.9216 - illumination_smoothness_loss: 0.1294 - spatial_constancy_loss: 6.0396e-04 - color_constancy_loss: 0.0037 - exposure_loss: 2.7879 - val_total_loss: 2.9737 - val_illumination_smoothness_loss: 0.1028 - val_spatial_constancy_loss: 6.3615e-04 - val_color_constancy_loss: 5.6798e-04 - val_exposure_loss: 2.8697
Epoch 11/100
25/25 [==============================] - 7s 264ms/step - total_loss: 2.8823 - illumination_smoothness_loss: 0.1141 - spatial_constancy_loss: 8.0172e-04 - color_constancy_loss: 0.0039 - exposure_loss: 2.7635 - val_total_loss: 2.9422 - val_illumination_smoothness_loss: 0.0951 - val_spatial_constancy_loss: 8.5813e-04 - val_color_constancy_loss: 6.1538e-04 - val_exposure_loss: 2.8456
Epoch 12/100
25/25 [==============================] - 7s 273ms/step - total_loss: 2.8443 - illumination_smoothness_loss: 0.1049 - spatial_constancy_loss: 0.0011 - color_constancy_loss: 0.0043 - exposure_loss: 2.7341 - val_total_loss: 2.9096 - val_illumination_smoothness_loss: 0.0936 - val_spatial_constancy_loss: 0.0012 - val_color_constancy_loss: 6.7707e-04 - val_exposure_loss: 2.8142
Epoch 13/100
25/25 [==============================] - 7s 274ms/step - total_loss: 2.7997 - illumination_smoothness_loss: 0.1031 - spatial_constancy_loss: 0.0016 - color_constancy_loss: 0.0047 - exposure_loss: 2.6903 - val_total_loss: 2.8666 - val_illumination_smoothness_loss: 0.1034 - val_spatial_constancy_loss: 0.0019 - val_color_constancy_loss: 8.0413e-04 - val_exposure_loss: 2.7604
Epoch 14/100
25/25 [==============================] - 7s 275ms/step - total_loss: 2.7249 - illumination_smoothness_loss: 0.1149 - spatial_constancy_loss: 0.0030 - color_constancy_loss: 0.0057 - exposure_loss: 2.6013 - val_total_loss: 2.7764 - val_illumination_smoothness_loss: 0.1291 - val_spatial_constancy_loss: 0.0042 - val_color_constancy_loss: 0.0011 - val_exposure_loss: 2.6419
Epoch 15/100
25/25 [==============================] - 7s 265ms/step - total_loss: 2.5184 - illumination_smoothness_loss: 0.1584 - spatial_constancy_loss: 0.0103 - color_constancy_loss: 0.0093 - exposure_loss: 2.3403 - val_total_loss: 2.4698 - val_illumination_smoothness_loss: 0.1949 - val_spatial_constancy_loss: 0.0194 - val_color_constancy_loss: 0.0031 - val_exposure_loss: 2.2524
Epoch 16/100
25/25 [==============================] - 7s 275ms/step - total_loss: 1.8216 - illumination_smoothness_loss: 0.2401 - spatial_constancy_loss: 0.0934 - color_constancy_loss: 0.0348 - exposure_loss: 1.4532 - val_total_loss: 1.6855 - val_illumination_smoothness_loss: 0.2599 - val_spatial_constancy_loss: 0.1776 - val_color_constancy_loss: 0.0229 - val_exposure_loss: 1.2250
Epoch 17/100
25/25 [==============================] - 7s 267ms/step - total_loss: 1.3387 - illumination_smoothness_loss: 0.2350 - spatial_constancy_loss: 0.2752 - color_constancy_loss: 0.0814 - exposure_loss: 0.7471 - val_total_loss: 1.5451 - val_illumination_smoothness_loss: 0.1862 - val_spatial_constancy_loss: 0.2320 - val_color_constancy_loss: 0.0331 - val_exposure_loss: 1.0938
Epoch 18/100
25/25 [==============================] - 7s 267ms/step - total_loss: 1.2646 - illumination_smoothness_loss: 0.1724 - spatial_constancy_loss: 0.2605 - color_constancy_loss: 0.0720 - exposure_loss: 0.7597 - val_total_loss: 1.5153 - val_illumination_smoothness_loss: 0.1533 - val_spatial_constancy_loss: 0.2295 - val_color_constancy_loss: 0.0343 - val_exposure_loss: 1.0981
Epoch 19/100
25/25 [==============================] - 7s 267ms/step - total_loss: 1.2439 - illumination_smoothness_loss: 0.1559 - spatial_constancy_loss: 0.2706 - color_constancy_loss: 0.0730 - exposure_loss: 0.7443 - val_total_loss: 1.4994 - val_illumination_smoothness_loss: 0.1423 - val_spatial_constancy_loss: 0.2359 - val_color_constancy_loss: 0.0363 - val_exposure_loss: 1.0850
Epoch 20/100
25/25 [==============================] - 7s 276ms/step - total_loss: 1.2311 - illumination_smoothness_loss: 0.1449 - spatial_constancy_loss: 0.2720 - color_constancy_loss: 0.0731 - exposure_loss: 0.7411 - val_total_loss: 1.4889 - val_illumination_smoothness_loss: 0.1299 - val_spatial_constancy_loss: 0.2331 - val_color_constancy_loss: 0.0358 - val_exposure_loss: 1.0901
Epoch 21/100
25/25 [==============================] - 7s 266ms/step - total_loss: 1.2262 - illumination_smoothness_loss: 0.1400 - spatial_constancy_loss: 0.2726 - color_constancy_loss: 0.0734 - exposure_loss: 0.7402 - val_total_loss: 1.4806 - val_illumination_smoothness_loss: 0.1233 - val_spatial_constancy_loss: 0.2356 - val_color_constancy_loss: 0.0371 - val_exposure_loss: 1.0847
Epoch 22/100
25/25 [==============================] - 7s 266ms/step - total_loss: 1.2202 - illumination_smoothness_loss: 0.1325 - spatial_constancy_loss: 0.2739 - color_constancy_loss: 0.0734 - exposure_loss: 0.7404 - val_total_loss: 1.4765 - val_illumination_smoothness_loss: 0.1231 - val_spatial_constancy_loss: 0.2408 - val_color_constancy_loss: 0.0381 - val_exposure_loss: 1.0745
Epoch 23/100
25/25 [==============================] - 7s 277ms/step - total_loss: 1.2122 - illumination_smoothness_loss: 0.1247 - spatial_constancy_loss: 0.2752 - color_constancy_loss: 0.0739 - exposure_loss: 0.7384 - val_total_loss: 1.4757 - val_illumination_smoothness_loss: 0.1253 - val_spatial_constancy_loss: 0.2453 - val_color_constancy_loss: 0.0393 - val_exposure_loss: 1.0658
Epoch 24/100
25/25 [==============================] - 7s 276ms/step - total_loss: 1.2015 - illumination_smoothness_loss: 0.1149 - spatial_constancy_loss: 0.2766 - color_constancy_loss: 0.0740 - exposure_loss: 0.7360 - val_total_loss: 1.4667 - val_illumination_smoothness_loss: 0.1168 - val_spatial_constancy_loss: 0.2456 - val_color_constancy_loss: 0.0390 - val_exposure_loss: 1.0652
Epoch 25/100
25/25 [==============================] - 7s 267ms/step - total_loss: 1.1940 - illumination_smoothness_loss: 0.1087 - spatial_constancy_loss: 0.2783 - color_constancy_loss: 0.0746 - exposure_loss: 0.7324 - val_total_loss: 1.4597 - val_illumination_smoothness_loss: 0.1109 - val_spatial_constancy_loss: 0.2476 - val_color_constancy_loss: 0.0399 - val_exposure_loss: 1.0613
Epoch 26/100
25/25 [==============================] - 7s 277ms/step - total_loss: 1.1878 - illumination_smoothness_loss: 0.1028 - spatial_constancy_loss: 0.2800 - color_constancy_loss: 0.0748 - exposure_loss: 0.7302 - val_total_loss: 1.4537 - val_illumination_smoothness_loss: 0.1054 - val_spatial_constancy_loss: 0.2479 - val_color_constancy_loss: 0.0398 - val_exposure_loss: 1.0606
Epoch 27/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.1827 - illumination_smoothness_loss: 0.0979 - spatial_constancy_loss: 0.2802 - color_constancy_loss: 0.0750 - exposure_loss: 0.7296 - val_total_loss: 1.4488 - val_illumination_smoothness_loss: 0.1015 - val_spatial_constancy_loss: 0.2496 - val_color_constancy_loss: 0.0404 - val_exposure_loss: 1.0573
Epoch 28/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.1774 - illumination_smoothness_loss: 0.0928 - spatial_constancy_loss: 0.2814 - color_constancy_loss: 0.0749 - exposure_loss: 0.7283 - val_total_loss: 1.4439 - val_illumination_smoothness_loss: 0.0968 - val_spatial_constancy_loss: 0.2491 - val_color_constancy_loss: 0.0397 - val_exposure_loss: 1.0583
Epoch 29/100
25/25 [==============================] - 7s 277ms/step - total_loss: 1.1720 - illumination_smoothness_loss: 0.0882 - spatial_constancy_loss: 0.2821 - color_constancy_loss: 0.0754 - exposure_loss: 0.7264 - val_total_loss: 1.4372 - val_illumination_smoothness_loss: 0.0907 - val_spatial_constancy_loss: 0.2504 - val_color_constancy_loss: 0.0405 - val_exposure_loss: 1.0557
Epoch 30/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.1660 - illumination_smoothness_loss: 0.0825 - spatial_constancy_loss: 0.2841 - color_constancy_loss: 0.0757 - exposure_loss: 0.7238 - val_total_loss: 1.4307 - val_illumination_smoothness_loss: 0.0840 - val_spatial_constancy_loss: 0.2500 - val_color_constancy_loss: 0.0406 - val_exposure_loss: 1.0561
Epoch 31/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.1626 - illumination_smoothness_loss: 0.0790 - spatial_constancy_loss: 0.2834 - color_constancy_loss: 0.0753 - exposure_loss: 0.7248 - val_total_loss: 1.4285 - val_illumination_smoothness_loss: 0.0829 - val_spatial_constancy_loss: 0.2508 - val_color_constancy_loss: 0.0399 - val_exposure_loss: 1.0549
Epoch 32/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.1576 - illumination_smoothness_loss: 0.0744 - spatial_constancy_loss: 0.2851 - color_constancy_loss: 0.0759 - exposure_loss: 0.7222 - val_total_loss: 1.4213 - val_illumination_smoothness_loss: 0.0756 - val_spatial_constancy_loss: 0.2509 - val_color_constancy_loss: 0.0403 - val_exposure_loss: 1.0545
Epoch 33/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.1529 - illumination_smoothness_loss: 0.0702 - spatial_constancy_loss: 0.2856 - color_constancy_loss: 0.0757 - exposure_loss: 0.7215 - val_total_loss: 1.4164 - val_illumination_smoothness_loss: 0.0720 - val_spatial_constancy_loss: 0.2525 - val_color_constancy_loss: 0.0403 - val_exposure_loss: 1.0515
Epoch 34/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.1486 - illumination_smoothness_loss: 0.0659 - spatial_constancy_loss: 0.2871 - color_constancy_loss: 0.0762 - exposure_loss: 0.7195 - val_total_loss: 1.4120 - val_illumination_smoothness_loss: 0.0675 - val_spatial_constancy_loss: 0.2528 - val_color_constancy_loss: 0.0410 - val_exposure_loss: 1.0507
Epoch 35/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.1439 - illumination_smoothness_loss: 0.0617 - spatial_constancy_loss: 0.2876 - color_constancy_loss: 0.0761 - exposure_loss: 0.7184 - val_total_loss: 1.4064 - val_illumination_smoothness_loss: 0.0628 - val_spatial_constancy_loss: 0.2538 - val_color_constancy_loss: 0.0408 - val_exposure_loss: 1.0490
Epoch 36/100
25/25 [==============================] - 7s 279ms/step - total_loss: 1.1393 - illumination_smoothness_loss: 0.0575 - spatial_constancy_loss: 0.2891 - color_constancy_loss: 0.0766 - exposure_loss: 0.7161 - val_total_loss: 1.4016 - val_illumination_smoothness_loss: 0.0574 - val_spatial_constancy_loss: 0.2529 - val_color_constancy_loss: 0.0408 - val_exposure_loss: 1.0505
Epoch 37/100
25/25 [==============================] - 7s 270ms/step - total_loss: 1.1360 - illumination_smoothness_loss: 0.0539 - spatial_constancy_loss: 0.2891 - color_constancy_loss: 0.0763 - exposure_loss: 0.7166 - val_total_loss: 1.3975 - val_illumination_smoothness_loss: 0.0545 - val_spatial_constancy_loss: 0.2547 - val_color_constancy_loss: 0.0410 - val_exposure_loss: 1.0473
Epoch 38/100
25/25 [==============================] - 7s 279ms/step - total_loss: 1.1327 - illumination_smoothness_loss: 0.0512 - spatial_constancy_loss: 0.2907 - color_constancy_loss: 0.0770 - exposure_loss: 0.7138 - val_total_loss: 1.3946 - val_illumination_smoothness_loss: 0.0515 - val_spatial_constancy_loss: 0.2546 - val_color_constancy_loss: 0.0414 - val_exposure_loss: 1.0471
Epoch 39/100
25/25 [==============================] - 7s 279ms/step - total_loss: 1.1283 - illumination_smoothness_loss: 0.0465 - spatial_constancy_loss: 0.2916 - color_constancy_loss: 0.0768 - exposure_loss: 0.7133 - val_total_loss: 1.3906 - val_illumination_smoothness_loss: 0.0473 - val_spatial_constancy_loss: 0.2538 - val_color_constancy_loss: 0.0411 - val_exposure_loss: 1.0485
Epoch 40/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.1257 - illumination_smoothness_loss: 0.0441 - spatial_constancy_loss: 0.2907 - color_constancy_loss: 0.0768 - exposure_loss: 0.7141 - val_total_loss: 1.3889 - val_illumination_smoothness_loss: 0.0477 - val_spatial_constancy_loss: 0.2577 - val_color_constancy_loss: 0.0419 - val_exposure_loss: 1.0416
Epoch 41/100
25/25 [==============================] - 7s 271ms/step - total_loss: 1.1225 - illumination_smoothness_loss: 0.0412 - spatial_constancy_loss: 0.2928 - color_constancy_loss: 0.0772 - exposure_loss: 0.7114 - val_total_loss: 1.3848 - val_illumination_smoothness_loss: 0.0433 - val_spatial_constancy_loss: 0.2569 - val_color_constancy_loss: 0.0417 - val_exposure_loss: 1.0428
Epoch 42/100
25/25 [==============================] - 7s 270ms/step - total_loss: 1.1202 - illumination_smoothness_loss: 0.0391 - spatial_constancy_loss: 0.2929 - color_constancy_loss: 0.0771 - exposure_loss: 0.7110 - val_total_loss: 1.3831 - val_illumination_smoothness_loss: 0.0425 - val_spatial_constancy_loss: 0.2583 - val_color_constancy_loss: 0.0420 - val_exposure_loss: 1.0403
Epoch 43/100
25/25 [==============================] - 7s 270ms/step - total_loss: 1.1177 - illumination_smoothness_loss: 0.0365 - spatial_constancy_loss: 0.2932 - color_constancy_loss: 0.0772 - exposure_loss: 0.7107 - val_total_loss: 1.3784 - val_illumination_smoothness_loss: 0.0376 - val_spatial_constancy_loss: 0.2578 - val_color_constancy_loss: 0.0418 - val_exposure_loss: 1.0412
Epoch 44/100
25/25 [==============================] - 7s 279ms/step - total_loss: 1.1155 - illumination_smoothness_loss: 0.0349 - spatial_constancy_loss: 0.2953 - color_constancy_loss: 0.0777 - exposure_loss: 0.7077 - val_total_loss: 1.3767 - val_illumination_smoothness_loss: 0.0341 - val_spatial_constancy_loss: 0.2545 - val_color_constancy_loss: 0.0413 - val_exposure_loss: 1.0467
Epoch 45/100
25/25 [==============================] - 7s 279ms/step - total_loss: 1.1133 - illumination_smoothness_loss: 0.0321 - spatial_constancy_loss: 0.2931 - color_constancy_loss: 0.0770 - exposure_loss: 0.7110 - val_total_loss: 1.3755 - val_illumination_smoothness_loss: 0.0353 - val_spatial_constancy_loss: 0.2590 - val_color_constancy_loss: 0.0424 - val_exposure_loss: 1.0387
Epoch 46/100
25/25 [==============================] - 7s 280ms/step - total_loss: 1.1112 - illumination_smoothness_loss: 0.0304 - spatial_constancy_loss: 0.2952 - color_constancy_loss: 0.0776 - exposure_loss: 0.7080 - val_total_loss: 1.3728 - val_illumination_smoothness_loss: 0.0328 - val_spatial_constancy_loss: 0.2591 - val_color_constancy_loss: 0.0424 - val_exposure_loss: 1.0385
Epoch 47/100
25/25 [==============================] - 7s 279ms/step - total_loss: 1.1094 - illumination_smoothness_loss: 0.0287 - spatial_constancy_loss: 0.2955 - color_constancy_loss: 0.0775 - exposure_loss: 0.7076 - val_total_loss: 1.3720 - val_illumination_smoothness_loss: 0.0329 - val_spatial_constancy_loss: 0.2605 - val_color_constancy_loss: 0.0425 - val_exposure_loss: 1.0361
Epoch 48/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.1079 - illumination_smoothness_loss: 0.0276 - spatial_constancy_loss: 0.2955 - color_constancy_loss: 0.0777 - exposure_loss: 0.7072 - val_total_loss: 1.3707 - val_illumination_smoothness_loss: 0.0316 - val_spatial_constancy_loss: 0.2606 - val_color_constancy_loss: 0.0426 - val_exposure_loss: 1.0359
Epoch 49/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.1056 - illumination_smoothness_loss: 0.0252 - spatial_constancy_loss: 0.2967 - color_constancy_loss: 0.0777 - exposure_loss: 0.7061 - val_total_loss: 1.3672 - val_illumination_smoothness_loss: 0.0277 - val_spatial_constancy_loss: 0.2597 - val_color_constancy_loss: 0.0426 - val_exposure_loss: 1.0372
Epoch 50/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.1047 - illumination_smoothness_loss: 0.0243 - spatial_constancy_loss: 0.2962 - color_constancy_loss: 0.0776 - exposure_loss: 0.7066 - val_total_loss: 1.3653 - val_illumination_smoothness_loss: 0.0256 - val_spatial_constancy_loss: 0.2590 - val_color_constancy_loss: 0.0423 - val_exposure_loss: 1.0383
Epoch 51/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.1038 - illumination_smoothness_loss: 0.0237 - spatial_constancy_loss: 0.2968 - color_constancy_loss: 0.0778 - exposure_loss: 0.7054 - val_total_loss: 1.3657 - val_illumination_smoothness_loss: 0.0273 - val_spatial_constancy_loss: 0.2617 - val_color_constancy_loss: 0.0431 - val_exposure_loss: 1.0335
Epoch 52/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.1020 - illumination_smoothness_loss: 0.0220 - spatial_constancy_loss: 0.2979 - color_constancy_loss: 0.0779 - exposure_loss: 0.7042 - val_total_loss: 1.3635 - val_illumination_smoothness_loss: 0.0234 - val_spatial_constancy_loss: 0.2579 - val_color_constancy_loss: 0.0422 - val_exposure_loss: 1.0400
Epoch 53/100
25/25 [==============================] - 7s 270ms/step - total_loss: 1.1012 - illumination_smoothness_loss: 0.0208 - spatial_constancy_loss: 0.2967 - color_constancy_loss: 0.0775 - exposure_loss: 0.7064 - val_total_loss: 1.3636 - val_illumination_smoothness_loss: 0.0250 - val_spatial_constancy_loss: 0.2607 - val_color_constancy_loss: 0.0428 - val_exposure_loss: 1.0352
Epoch 54/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.1002 - illumination_smoothness_loss: 0.0205 - spatial_constancy_loss: 0.2970 - color_constancy_loss: 0.0777 - exposure_loss: 0.7049 - val_total_loss: 1.3615 - val_illumination_smoothness_loss: 0.0233 - val_spatial_constancy_loss: 0.2611 - val_color_constancy_loss: 0.0427 - val_exposure_loss: 1.0345
Epoch 55/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0989 - illumination_smoothness_loss: 0.0193 - spatial_constancy_loss: 0.2985 - color_constancy_loss: 0.0780 - exposure_loss: 0.7032 - val_total_loss: 1.3608 - val_illumination_smoothness_loss: 0.0225 - val_spatial_constancy_loss: 0.2609 - val_color_constancy_loss: 0.0428 - val_exposure_loss: 1.0346
Epoch 56/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0986 - illumination_smoothness_loss: 0.0190 - spatial_constancy_loss: 0.2971 - color_constancy_loss: 0.0777 - exposure_loss: 0.7048 - val_total_loss: 1.3615 - val_illumination_smoothness_loss: 0.0238 - val_spatial_constancy_loss: 0.2621 - val_color_constancy_loss: 0.0430 - val_exposure_loss: 1.0327
Epoch 57/100
25/25 [==============================] - 7s 279ms/step - total_loss: 1.0977 - illumination_smoothness_loss: 0.0182 - spatial_constancy_loss: 0.2987 - color_constancy_loss: 0.0780 - exposure_loss: 0.7028 - val_total_loss: 1.3601 - val_illumination_smoothness_loss: 0.0226 - val_spatial_constancy_loss: 0.2623 - val_color_constancy_loss: 0.0431 - val_exposure_loss: 1.0321
Epoch 58/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.0971 - illumination_smoothness_loss: 0.0174 - spatial_constancy_loss: 0.2979 - color_constancy_loss: 0.0778 - exposure_loss: 0.7040 - val_total_loss: 1.3596 - val_illumination_smoothness_loss: 0.0218 - val_spatial_constancy_loss: 0.2615 - val_color_constancy_loss: 0.0428 - val_exposure_loss: 1.0334
Epoch 59/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.0974 - illumination_smoothness_loss: 0.0180 - spatial_constancy_loss: 0.2985 - color_constancy_loss: 0.0780 - exposure_loss: 0.7029 - val_total_loss: 1.3611 - val_illumination_smoothness_loss: 0.0246 - val_spatial_constancy_loss: 0.2645 - val_color_constancy_loss: 0.0437 - val_exposure_loss: 1.0282
Epoch 60/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0956 - illumination_smoothness_loss: 0.0165 - spatial_constancy_loss: 0.2985 - color_constancy_loss: 0.0780 - exposure_loss: 0.7026 - val_total_loss: 1.3581 - val_illumination_smoothness_loss: 0.0209 - val_spatial_constancy_loss: 0.2623 - val_color_constancy_loss: 0.0430 - val_exposure_loss: 1.0320
Epoch 61/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.0953 - illumination_smoothness_loss: 0.0159 - spatial_constancy_loss: 0.2992 - color_constancy_loss: 0.0782 - exposure_loss: 0.7020 - val_total_loss: 1.3579 - val_illumination_smoothness_loss: 0.0213 - val_spatial_constancy_loss: 0.2637 - val_color_constancy_loss: 0.0436 - val_exposure_loss: 1.0293
Epoch 62/100
25/25 [==============================] - 7s 277ms/step - total_loss: 1.0945 - illumination_smoothness_loss: 0.0154 - spatial_constancy_loss: 0.2982 - color_constancy_loss: 0.0780 - exposure_loss: 0.7029 - val_total_loss: 1.3571 - val_illumination_smoothness_loss: 0.0199 - val_spatial_constancy_loss: 0.2620 - val_color_constancy_loss: 0.0429 - val_exposure_loss: 1.0323
Epoch 63/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.0948 - illumination_smoothness_loss: 0.0156 - spatial_constancy_loss: 0.2989 - color_constancy_loss: 0.0781 - exposure_loss: 0.7021 - val_total_loss: 1.3577 - val_illumination_smoothness_loss: 0.0215 - val_spatial_constancy_loss: 0.2641 - val_color_constancy_loss: 0.0435 - val_exposure_loss: 1.0287
Epoch 64/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.0935 - illumination_smoothness_loss: 0.0146 - spatial_constancy_loss: 0.2994 - color_constancy_loss: 0.0782 - exposure_loss: 0.7014 - val_total_loss: 1.3565 - val_illumination_smoothness_loss: 0.0200 - val_spatial_constancy_loss: 0.2632 - val_color_constancy_loss: 0.0433 - val_exposure_loss: 1.0300
Epoch 65/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0933 - illumination_smoothness_loss: 0.0144 - spatial_constancy_loss: 0.2992 - color_constancy_loss: 0.0781 - exposure_loss: 0.7015 - val_total_loss: 1.3570 - val_illumination_smoothness_loss: 0.0211 - val_spatial_constancy_loss: 0.2648 - val_color_constancy_loss: 0.0439 - val_exposure_loss: 1.0273
Epoch 66/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.0927 - illumination_smoothness_loss: 0.0141 - spatial_constancy_loss: 0.2993 - color_constancy_loss: 0.0781 - exposure_loss: 0.7012 - val_total_loss: 1.3549 - val_illumination_smoothness_loss: 0.0179 - val_spatial_constancy_loss: 0.2618 - val_color_constancy_loss: 0.0429 - val_exposure_loss: 1.0323
Epoch 67/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0930 - illumination_smoothness_loss: 0.0141 - spatial_constancy_loss: 0.2992 - color_constancy_loss: 0.0781 - exposure_loss: 0.7016 - val_total_loss: 1.3565 - val_illumination_smoothness_loss: 0.0208 - val_spatial_constancy_loss: 0.2652 - val_color_constancy_loss: 0.0441 - val_exposure_loss: 1.0265
Epoch 68/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0919 - illumination_smoothness_loss: 0.0135 - spatial_constancy_loss: 0.3001 - color_constancy_loss: 0.0782 - exposure_loss: 0.7002 - val_total_loss: 1.3543 - val_illumination_smoothness_loss: 0.0173 - val_spatial_constancy_loss: 0.2617 - val_color_constancy_loss: 0.0429 - val_exposure_loss: 1.0323
Epoch 69/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0925 - illumination_smoothness_loss: 0.0136 - spatial_constancy_loss: 0.2989 - color_constancy_loss: 0.0780 - exposure_loss: 0.7019 - val_total_loss: 1.3562 - val_illumination_smoothness_loss: 0.0203 - val_spatial_constancy_loss: 0.2646 - val_color_constancy_loss: 0.0440 - val_exposure_loss: 1.0272
Epoch 70/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.0916 - illumination_smoothness_loss: 0.0130 - spatial_constancy_loss: 0.3005 - color_constancy_loss: 0.0782 - exposure_loss: 0.7000 - val_total_loss: 1.3530 - val_illumination_smoothness_loss: 0.0156 - val_spatial_constancy_loss: 0.2606 - val_color_constancy_loss: 0.0428 - val_exposure_loss: 1.0341
Epoch 71/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0918 - illumination_smoothness_loss: 0.0128 - spatial_constancy_loss: 0.2985 - color_constancy_loss: 0.0778 - exposure_loss: 0.7028 - val_total_loss: 1.3550 - val_illumination_smoothness_loss: 0.0194 - val_spatial_constancy_loss: 0.2645 - val_color_constancy_loss: 0.0439 - val_exposure_loss: 1.0273
Epoch 72/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.0911 - illumination_smoothness_loss: 0.0127 - spatial_constancy_loss: 0.3001 - color_constancy_loss: 0.0782 - exposure_loss: 0.7001 - val_total_loss: 1.3535 - val_illumination_smoothness_loss: 0.0175 - val_spatial_constancy_loss: 0.2638 - val_color_constancy_loss: 0.0438 - val_exposure_loss: 1.0284
Epoch 73/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0906 - illumination_smoothness_loss: 0.0121 - spatial_constancy_loss: 0.2998 - color_constancy_loss: 0.0780 - exposure_loss: 0.7006 - val_total_loss: 1.3521 - val_illumination_smoothness_loss: 0.0153 - val_spatial_constancy_loss: 0.2615 - val_color_constancy_loss: 0.0430 - val_exposure_loss: 1.0323
Epoch 74/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.0914 - illumination_smoothness_loss: 0.0127 - spatial_constancy_loss: 0.2993 - color_constancy_loss: 0.0780 - exposure_loss: 0.7014 - val_total_loss: 1.3547 - val_illumination_smoothness_loss: 0.0189 - val_spatial_constancy_loss: 0.2642 - val_color_constancy_loss: 0.0441 - val_exposure_loss: 1.0275
Epoch 75/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0908 - illumination_smoothness_loss: 0.0125 - spatial_constancy_loss: 0.2994 - color_constancy_loss: 0.0781 - exposure_loss: 0.7008 - val_total_loss: 1.3533 - val_illumination_smoothness_loss: 0.0174 - val_spatial_constancy_loss: 0.2636 - val_color_constancy_loss: 0.0436 - val_exposure_loss: 1.0286
Epoch 76/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.0909 - illumination_smoothness_loss: 0.0126 - spatial_constancy_loss: 0.2998 - color_constancy_loss: 0.0782 - exposure_loss: 0.7004 - val_total_loss: 1.3544 - val_illumination_smoothness_loss: 0.0194 - val_spatial_constancy_loss: 0.2655 - val_color_constancy_loss: 0.0442 - val_exposure_loss: 1.0253
Epoch 77/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0897 - illumination_smoothness_loss: 0.0116 - spatial_constancy_loss: 0.3002 - color_constancy_loss: 0.0783 - exposure_loss: 0.6996 - val_total_loss: 1.3516 - val_illumination_smoothness_loss: 0.0159 - val_spatial_constancy_loss: 0.2635 - val_color_constancy_loss: 0.0436 - val_exposure_loss: 1.0286
Epoch 78/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.0900 - illumination_smoothness_loss: 0.0120 - spatial_constancy_loss: 0.2998 - color_constancy_loss: 0.0781 - exposure_loss: 0.7001 - val_total_loss: 1.3528 - val_illumination_smoothness_loss: 0.0174 - val_spatial_constancy_loss: 0.2641 - val_color_constancy_loss: 0.0437 - val_exposure_loss: 1.0277
Epoch 79/100
25/25 [==============================] - 7s 279ms/step - total_loss: 1.0904 - illumination_smoothness_loss: 0.0122 - spatial_constancy_loss: 0.2999 - color_constancy_loss: 0.0782 - exposure_loss: 0.7001 - val_total_loss: 1.3528 - val_illumination_smoothness_loss: 0.0178 - val_spatial_constancy_loss: 0.2647 - val_color_constancy_loss: 0.0439 - val_exposure_loss: 1.0264
Epoch 80/100
25/25 [==============================] - 7s 279ms/step - total_loss: 1.0895 - illumination_smoothness_loss: 0.0114 - spatial_constancy_loss: 0.2995 - color_constancy_loss: 0.0782 - exposure_loss: 0.7003 - val_total_loss: 1.3520 - val_illumination_smoothness_loss: 0.0168 - val_spatial_constancy_loss: 0.2643 - val_color_constancy_loss: 0.0438 - val_exposure_loss: 1.0270
Epoch 81/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.0895 - illumination_smoothness_loss: 0.0116 - spatial_constancy_loss: 0.3002 - color_constancy_loss: 0.0783 - exposure_loss: 0.6995 - val_total_loss: 1.3520 - val_illumination_smoothness_loss: 0.0170 - val_spatial_constancy_loss: 0.2645 - val_color_constancy_loss: 0.0439 - val_exposure_loss: 1.0267
Epoch 82/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0898 - illumination_smoothness_loss: 0.0116 - spatial_constancy_loss: 0.3001 - color_constancy_loss: 0.0782 - exposure_loss: 0.6999 - val_total_loss: 1.3532 - val_illumination_smoothness_loss: 0.0185 - val_spatial_constancy_loss: 0.2655 - val_color_constancy_loss: 0.0443 - val_exposure_loss: 1.0249
Epoch 83/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.0888 - illumination_smoothness_loss: 0.0112 - spatial_constancy_loss: 0.3002 - color_constancy_loss: 0.0782 - exposure_loss: 0.6992 - val_total_loss: 1.3517 - val_illumination_smoothness_loss: 0.0166 - val_spatial_constancy_loss: 0.2642 - val_color_constancy_loss: 0.0438 - val_exposure_loss: 1.0271
Epoch 84/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0887 - illumination_smoothness_loss: 0.0106 - spatial_constancy_loss: 0.3004 - color_constancy_loss: 0.0781 - exposure_loss: 0.6996 - val_total_loss: 1.3500 - val_illumination_smoothness_loss: 0.0148 - val_spatial_constancy_loss: 0.2639 - val_color_constancy_loss: 0.0439 - val_exposure_loss: 1.0275
Epoch 85/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.0886 - illumination_smoothness_loss: 0.0110 - spatial_constancy_loss: 0.3000 - color_constancy_loss: 0.0781 - exposure_loss: 0.6994 - val_total_loss: 1.3511 - val_illumination_smoothness_loss: 0.0163 - val_spatial_constancy_loss: 0.2644 - val_color_constancy_loss: 0.0438 - val_exposure_loss: 1.0266
Epoch 86/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0889 - illumination_smoothness_loss: 0.0110 - spatial_constancy_loss: 0.3004 - color_constancy_loss: 0.0782 - exposure_loss: 0.6993 - val_total_loss: 1.3513 - val_illumination_smoothness_loss: 0.0166 - val_spatial_constancy_loss: 0.2649 - val_color_constancy_loss: 0.0442 - val_exposure_loss: 1.0257
Epoch 87/100
25/25 [==============================] - 7s 269ms/step - total_loss: 1.0885 - illumination_smoothness_loss: 0.0111 - spatial_constancy_loss: 0.3001 - color_constancy_loss: 0.0781 - exposure_loss: 0.6992 - val_total_loss: 1.3504 - val_illumination_smoothness_loss: 0.0154 - val_spatial_constancy_loss: 0.2639 - val_color_constancy_loss: 0.0437 - val_exposure_loss: 1.0274
Epoch 88/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.0889 - illumination_smoothness_loss: 0.0111 - spatial_constancy_loss: 0.3000 - color_constancy_loss: 0.0781 - exposure_loss: 0.6997 - val_total_loss: 1.3512 - val_illumination_smoothness_loss: 0.0165 - val_spatial_constancy_loss: 0.2650 - val_color_constancy_loss: 0.0443 - val_exposure_loss: 1.0254
Epoch 89/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.0883 - illumination_smoothness_loss: 0.0109 - spatial_constancy_loss: 0.3003 - color_constancy_loss: 0.0781 - exposure_loss: 0.6990 - val_total_loss: 1.3506 - val_illumination_smoothness_loss: 0.0160 - val_spatial_constancy_loss: 0.2645 - val_color_constancy_loss: 0.0439 - val_exposure_loss: 1.0262
Epoch 90/100
25/25 [==============================] - 7s 268ms/step - total_loss: 1.0883 - illumination_smoothness_loss: 0.0106 - spatial_constancy_loss: 0.3003 - color_constancy_loss: 0.0781 - exposure_loss: 0.6993 - val_total_loss: 1.3498 - val_illumination_smoothness_loss: 0.0149 - val_spatial_constancy_loss: 0.2640 - val_color_constancy_loss: 0.0440 - val_exposure_loss: 1.0270
Epoch 91/100
25/25 [==============================] - 7s 277ms/step - total_loss: 1.0883 - illumination_smoothness_loss: 0.0107 - spatial_constancy_loss: 0.3000 - color_constancy_loss: 0.0780 - exposure_loss: 0.6995 - val_total_loss: 1.3492 - val_illumination_smoothness_loss: 0.0146 - val_spatial_constancy_loss: 0.2644 - val_color_constancy_loss: 0.0440 - val_exposure_loss: 1.0262
Epoch 92/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0884 - illumination_smoothness_loss: 0.0108 - spatial_constancy_loss: 0.3007 - color_constancy_loss: 0.0782 - exposure_loss: 0.6987 - val_total_loss: 1.3496 - val_illumination_smoothness_loss: 0.0148 - val_spatial_constancy_loss: 0.2642 - val_color_constancy_loss: 0.0441 - val_exposure_loss: 1.0265
Epoch 93/100
25/25 [==============================] - 7s 277ms/step - total_loss: 1.0878 - illumination_smoothness_loss: 0.0105 - spatial_constancy_loss: 0.2994 - color_constancy_loss: 0.0780 - exposure_loss: 0.6999 - val_total_loss: 1.3497 - val_illumination_smoothness_loss: 0.0150 - val_spatial_constancy_loss: 0.2643 - val_color_constancy_loss: 0.0440 - val_exposure_loss: 1.0263
Epoch 94/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0876 - illumination_smoothness_loss: 0.0098 - spatial_constancy_loss: 0.3005 - color_constancy_loss: 0.0781 - exposure_loss: 0.6992 - val_total_loss: 1.3471 - val_illumination_smoothness_loss: 0.0120 - val_spatial_constancy_loss: 0.2633 - val_color_constancy_loss: 0.0439 - val_exposure_loss: 1.0279
Epoch 95/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0876 - illumination_smoothness_loss: 0.0103 - spatial_constancy_loss: 0.3002 - color_constancy_loss: 0.0782 - exposure_loss: 0.6989 - val_total_loss: 1.3493 - val_illumination_smoothness_loss: 0.0147 - val_spatial_constancy_loss: 0.2642 - val_color_constancy_loss: 0.0441 - val_exposure_loss: 1.0263
Epoch 96/100
25/25 [==============================] - 7s 277ms/step - total_loss: 1.0880 - illumination_smoothness_loss: 0.0105 - spatial_constancy_loss: 0.3001 - color_constancy_loss: 0.0781 - exposure_loss: 0.6994 - val_total_loss: 1.3485 - val_illumination_smoothness_loss: 0.0140 - val_spatial_constancy_loss: 0.2644 - val_color_constancy_loss: 0.0440 - val_exposure_loss: 1.0261
Epoch 97/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0878 - illumination_smoothness_loss: 0.0102 - spatial_constancy_loss: 0.3005 - color_constancy_loss: 0.0782 - exposure_loss: 0.6990 - val_total_loss: 1.3485 - val_illumination_smoothness_loss: 0.0140 - val_spatial_constancy_loss: 0.2645 - val_color_constancy_loss: 0.0443 - val_exposure_loss: 1.0257
Epoch 98/100
25/25 [==============================] - 7s 278ms/step - total_loss: 1.0875 - illumination_smoothness_loss: 0.0104 - spatial_constancy_loss: 0.3003 - color_constancy_loss: 0.0781 - exposure_loss: 0.6987 - val_total_loss: 1.3485 - val_illumination_smoothness_loss: 0.0140 - val_spatial_constancy_loss: 0.2641 - val_color_constancy_loss: 0.0440 - val_exposure_loss: 1.0264
Epoch 99/100
25/25 [==============================] - 7s 277ms/step - total_loss: 1.0879 - illumination_smoothness_loss: 0.0104 - spatial_constancy_loss: 0.3005 - color_constancy_loss: 0.0782 - exposure_loss: 0.6988 - val_total_loss: 1.3486 - val_illumination_smoothness_loss: 0.0140 - val_spatial_constancy_loss: 0.2642 - val_color_constancy_loss: 0.0443 - val_exposure_loss: 1.0260
Epoch 100/100
25/25 [==============================] - 7s 277ms/step - total_loss: 1.0873 - illumination_smoothness_loss: 0.0102 - spatial_constancy_loss: 0.3001 - color_constancy_loss: 0.0780 - exposure_loss: 0.6991 - val_total_loss: 1.3481 - val_illumination_smoothness_loss: 0.0134 - val_spatial_constancy_loss: 0.2635 - val_color_constancy_loss: 0.0439 - val_exposure_loss: 1.0273

```
</div>
![png](/img/examples/vision/zero_dce/zero_dce_21_1.png)



![png](/img/examples/vision/zero_dce/zero_dce_21_2.png)



![png](/img/examples/vision/zero_dce/zero_dce_21_3.png)



![png](/img/examples/vision/zero_dce/zero_dce_21_4.png)



![png](/img/examples/vision/zero_dce/zero_dce_21_5.png)


---
## Inference


```python

def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()


def infer(original_image):
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_dce_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image

```

### Inference on test images

We compare the test images from LOLDataset enhanced by MIRNet with images enhanced via
the `PIL.ImageOps.autocontrast()` function.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/low-light-image-enhancement) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/low-light-image-enhancement).


```python
for val_image_file in test_low_light_images:
    original_image = Image.open(val_image_file)
    enhanced_image = infer(original_image)
    plot_results(
        [original_image, ImageOps.autocontrast(original_image), enhanced_image],
        ["Original", "PIL Autocontrast", "Enhanced"],
        (20, 12),
    )
```


![png](/img/examples/vision/zero_dce/zero_dce_25_0.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_1.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_2.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_3.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_4.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_5.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_6.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_7.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_8.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_9.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_10.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_11.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_12.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_13.png)



![png](/img/examples/vision/zero_dce/zero_dce_25_14.png)
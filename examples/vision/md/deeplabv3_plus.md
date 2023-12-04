# Multiclass semantic segmentation using DeepLabV3+

**Author:** [Soumik Rakshit](http://github.com/soumik12345)<br>
**Date created:** 2021/08/31<br>
**Last modified:** 2023/07/19<br>
**Description:** Implement DeepLabV3+ architecture for Multi-class Semantic Segmentation.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/deeplabv3_plus.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/deeplabv3_plus.py)



---
## Introduction

Semantic segmentation, with the goal to assign semantic labels to every pixel in an image,
is an essential computer vision task. In this example, we implement
the **DeepLabV3+** model for multi-class semantic segmentation, a fully-convolutional
architecture that performs well on semantic segmentation benchmarks.

### References:

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)

---
## Downloading the data

We will use the [Crowd Instance-level Human Parsing Dataset](https://arxiv.org/abs/1811.12596)
for training our model. The Crowd Instance-level Human Parsing (CIHP) dataset has 38,280 diverse human images.
Each image in CIHP is labeled with pixel-wise annotations for 20 categories, as well as instance-level identification.
This dataset can be used for the "human part segmentation" task.


```python

import keras
from keras import layers
from keras import ops

import os
import numpy as np
from glob import glob
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt

# For data preprocessing
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
```


```python
!gdown "1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz&confirm=t"
!unzip -q instance-level-human-parsing.zip
```

<div class="k-default-codeblock">
```
Downloading...
From: https://drive.google.com/uc?id=1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz&confirm=t
To: /content/keras-io/scripts/tmp_7009966/instance-level-human-parsing.zip
100% 2.91G/2.91G [00:22<00:00, 129MB/s]

```
</div>
---
## Creating a TensorFlow Dataset

Training on the entire CIHP dataset with 38,280 images takes a lot of time, hence we will be using
a smaller subset of 200 images for training our model in this example.


```python
IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 20
DATA_DIR = "./instance-level_human_parsing/instance-level_human_parsing/Training"
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50

train_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]


def read_image(image_path, mask=False):
    image = tf_io.read_file(image_path)
    if mask:
        image = tf_image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf_image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf_data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)
```

<div class="k-default-codeblock">
```
Train Dataset: <_BatchDataset element_spec=(TensorSpec(shape=(4, 512, 512, 3), dtype=tf.float32, name=None), TensorSpec(shape=(4, 512, 512, 1), dtype=tf.float32, name=None))>
Val Dataset: <_BatchDataset element_spec=(TensorSpec(shape=(4, 512, 512, 3), dtype=tf.float32, name=None), TensorSpec(shape=(4, 512, 512, 1), dtype=tf.float32, name=None))>

```
</div>
---
## Building the DeepLabV3+ model

DeepLabv3+ extends DeepLabv3 by adding an encoder-decoder structure. The encoder module
processes multiscale contextual information by applying dilated convolution at multiple
scales, while the decoder module refines the segmentation results along object boundaries.

![](https://github.com/lattice-ai/DeepLabV3-Plus/raw/master/assets/deeplabv3_plus_diagram.png)

**Dilated convolution:** With dilated convolution, as we go deeper in the network, we can keep the
stride constant but with larger field-of-view without increasing the number of parameters
or the amount of computation. Besides, it enables larger output feature maps, which is
useful for semantic segmentation.

The reason for using **Dilated Spatial Pyramid Pooling** is that it was shown that as the
sampling rate becomes larger, the number of valid filter weights (i.e., weights that
are applied to the valid feature region, instead of padded zeros) becomes smaller.


```python

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return ops.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

```

The encoder features are first bilinearly upsampled by a factor 4, and then
concatenated with the corresponding low-level features from the network backbone that
have the same spatial resolution. For this example, we
use a ResNet50 pretrained on ImageNet as the backbone model, and we use
the low-level features from the `conv4_block6_2_relu` block of the backbone.


```python

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.summary()
```

<div class="k-default-codeblock">
```
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
 94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step

```
</div>
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)               </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">   Param # </span>┃<span style="font-weight: bold"> Connected to                </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                           │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ get_item (<span style="color: #0087ff; text-decoration-color: #0087ff">GetItem</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]           │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ get_item_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">GetItem</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]           │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ get_item_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">GetItem</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]           │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ stack (<span style="color: #0087ff; text-decoration-color: #0087ff">Stack</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ get_item[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],             │
│                            │                        │           │ get_item_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],           │
│                            │                        │           │ get_item_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ stack[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                 │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv1_pad (<span style="color: #0087ff; text-decoration-color: #0087ff">ZeroPadding2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">518</span>, <span style="color: #00af00; text-decoration-color: #00af00">518</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                   │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv1_conv (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">9,472</span> │ conv1_pad[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv1_bn                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv1_relu (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ pool1_pad (<span style="color: #0087ff; text-decoration-color: #0087ff">ZeroPadding2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">258</span>, <span style="color: #00af00; text-decoration-color: #00af00">258</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ pool1_pool (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ pool1_pad[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">4,160</span> │ pool1_pool[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2_block1_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block1_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │    <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ conv2_block1_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2_block1_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block1_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_0_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │    <span style="color: #00af00; text-decoration-color: #00af00">16,640</span> │ pool1_pool[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │    <span style="color: #00af00; text-decoration-color: #00af00">16,640</span> │ conv2_block1_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_0_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2_block1_0_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2_block1_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block1_0_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],    │
│                            │                        │           │ conv2_block1_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block1_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block1_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block2_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │    <span style="color: #00af00; text-decoration-color: #00af00">16,448</span> │ conv2_block1_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block2_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2_block2_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block2_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block2_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block2_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │    <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ conv2_block2_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block2_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2_block2_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block2_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block2_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block2_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │    <span style="color: #00af00; text-decoration-color: #00af00">16,640</span> │ conv2_block2_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block2_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2_block2_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block2_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block1_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],     │
│                            │                        │           │ conv2_block2_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block2_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block2_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block3_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │    <span style="color: #00af00; text-decoration-color: #00af00">16,448</span> │ conv2_block2_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block3_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2_block3_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block3_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block3_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block3_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │    <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ conv2_block3_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block3_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv2_block3_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block3_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block3_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block3_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │    <span style="color: #00af00; text-decoration-color: #00af00">16,640</span> │ conv2_block3_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block3_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2_block3_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block3_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block2_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],     │
│                            │                        │           │ conv2_block3_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2_block3_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv2_block3_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │    <span style="color: #00af00; text-decoration-color: #00af00">32,896</span> │ conv2_block3_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv3_block1_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block1_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ conv3_block1_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv3_block1_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block1_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_0_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">131,584</span> │ conv2_block3_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │    <span style="color: #00af00; text-decoration-color: #00af00">66,048</span> │ conv3_block1_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_0_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv3_block1_0_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv3_block1_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block1_0_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],    │
│                            │                        │           │ conv3_block1_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block1_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block1_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block2_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │    <span style="color: #00af00; text-decoration-color: #00af00">65,664</span> │ conv3_block1_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block2_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv3_block2_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block2_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block2_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block2_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ conv3_block2_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block2_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv3_block2_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block2_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block2_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block2_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │    <span style="color: #00af00; text-decoration-color: #00af00">66,048</span> │ conv3_block2_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block2_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv3_block2_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block2_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block1_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],     │
│                            │                        │           │ conv3_block2_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block2_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block2_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block3_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │    <span style="color: #00af00; text-decoration-color: #00af00">65,664</span> │ conv3_block2_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block3_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv3_block3_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block3_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block3_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block3_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ conv3_block3_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block3_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv3_block3_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block3_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block3_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block3_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │    <span style="color: #00af00; text-decoration-color: #00af00">66,048</span> │ conv3_block3_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block3_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv3_block3_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block3_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block2_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],     │
│                            │                        │           │ conv3_block3_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block3_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block3_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block4_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │    <span style="color: #00af00; text-decoration-color: #00af00">65,664</span> │ conv3_block3_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block4_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv3_block4_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block4_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block4_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block4_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ conv3_block4_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block4_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv3_block4_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block4_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block4_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block4_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │    <span style="color: #00af00; text-decoration-color: #00af00">66,048</span> │ conv3_block4_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block4_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ conv3_block4_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block4_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block3_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],     │
│                            │                        │           │ conv3_block4_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv3_block4_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv3_block4_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">131,328</span> │ conv3_block4_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block1_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block1_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ conv4_block1_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block1_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block1_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_0_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │   <span style="color: #00af00; text-decoration-color: #00af00">525,312</span> │ conv3_block4_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │   <span style="color: #00af00; text-decoration-color: #00af00">263,168</span> │ conv4_block1_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_0_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">4,096</span> │ conv4_block1_0_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">4,096</span> │ conv4_block1_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block1_0_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],    │
│                            │                        │           │ conv4_block1_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block1_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block1_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block2_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">262,400</span> │ conv4_block1_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block2_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block2_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block2_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block2_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block2_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ conv4_block2_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block2_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block2_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block2_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block2_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block2_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │   <span style="color: #00af00; text-decoration-color: #00af00">263,168</span> │ conv4_block2_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block2_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">4,096</span> │ conv4_block2_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block2_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block1_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],     │
│                            │                        │           │ conv4_block2_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block2_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block2_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block3_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">262,400</span> │ conv4_block2_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block3_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block3_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block3_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block3_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block3_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ conv4_block3_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block3_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block3_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block3_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block3_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block3_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │   <span style="color: #00af00; text-decoration-color: #00af00">263,168</span> │ conv4_block3_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block3_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">4,096</span> │ conv4_block3_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block3_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block2_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],     │
│                            │                        │           │ conv4_block3_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block3_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block3_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block4_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">262,400</span> │ conv4_block3_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block4_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block4_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block4_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block4_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block4_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ conv4_block4_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block4_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block4_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block4_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block4_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block4_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │   <span style="color: #00af00; text-decoration-color: #00af00">263,168</span> │ conv4_block4_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block4_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">4,096</span> │ conv4_block4_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block4_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block3_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],     │
│                            │                        │           │ conv4_block4_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block4_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block4_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block5_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">262,400</span> │ conv4_block4_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block5_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block5_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block5_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block5_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block5_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ conv4_block5_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block5_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block5_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block5_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block5_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block5_3_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │   <span style="color: #00af00; text-decoration-color: #00af00">263,168</span> │ conv4_block5_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block5_3_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">4,096</span> │ conv4_block5_3_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block5_add (<span style="color: #0087ff; text-decoration-color: #0087ff">Add</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block4_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],     │
│                            │                        │           │ conv4_block5_3_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block5_out           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block5_add[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block6_1_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">262,400</span> │ conv4_block5_out[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block6_1_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block6_1_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block6_1_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block6_1_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block6_2_conv        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ conv4_block6_1_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                   │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block6_2_bn          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv4_block6_2_conv[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv4_block6_2_relu        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block6_2_bn[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Activation</span>)               │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ average_pooling2d          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ conv4_block6_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">AveragePooling2D</span>)         │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │    <span style="color: #00af00; text-decoration-color: #00af00">65,792</span> │ average_pooling2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ batch_normalization        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │    <span style="color: #00af00; text-decoration-color: #00af00">65,536</span> │ conv4_block6_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">589,824</span> │ conv4_block6_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">589,824</span> │ conv4_block6_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">589,824</span> │ conv4_block6_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ relu (<span style="color: #0087ff; text-decoration-color: #0087ff">Relu</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ batch_normalization_1      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ batch_normalization_2      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ batch_normalization_3      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ batch_normalization_4      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ up_sampling2d              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">UpSampling2D</span>)             │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ relu_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Relu</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ relu_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Relu</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ relu_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Relu</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ relu_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Relu</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ concatenate (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">1280</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ up_sampling2d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],        │
│                            │                        │           │ relu_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>], relu_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>], │
│                            │                        │           │ relu_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>], relu_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │   <span style="color: #00af00; text-decoration-color: #00af00">327,680</span> │ concatenate[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]           │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ batch_normalization_5      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">3,072</span> │ conv2_block3_2_relu[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ relu_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Relu</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ batch_normalization_6      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">192</span> │ conv2d_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ up_sampling2d_1            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ relu_5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">UpSampling2D</span>)             │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ relu_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Relu</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ concatenate_1              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">304</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ up_sampling2d_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)              │                        │           │ relu_6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2d_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │   <span style="color: #00af00; text-decoration-color: #00af00">700,416</span> │ concatenate_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]         │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ batch_normalization_7      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ relu_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Relu</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2d_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │   <span style="color: #00af00; text-decoration-color: #00af00">589,824</span> │ relu_7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ batch_normalization_8      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ conv2d_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)       │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ relu_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Relu</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalization_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ up_sampling2d_2            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │         <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ relu_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">UpSampling2D</span>)             │                        │           │                             │
├────────────────────────────┼────────────────────────┼───────────┼─────────────────────────────┤
│ conv2d_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">20</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">5,140</span> │ up_sampling2d_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
└────────────────────────────┴────────────────────────┴───────────┴─────────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">11,857,236</span> (45.23 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">11,824,500</span> (45.11 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">32,736</span> (127.88 KB)
</pre>



---
## Training

We train the model using sparse categorical crossentropy as the loss function, and
Adam as the optimizer.


```python
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=25)

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()
```

<div class="k-default-codeblock">
```
Epoch 1/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 89s 90ms/step - accuracy: 0.5872 - loss: 1.4368 - val_accuracy: 0.5146 - val_loss: 25012.6270
Epoch 2/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 72ms/step - accuracy: 0.6607 - loss: 1.1006 - val_accuracy: 0.5959 - val_loss: 935.8270
Epoch 3/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 72ms/step - accuracy: 0.6713 - loss: 1.0537 - val_accuracy: 0.4431 - val_loss: 5.9314
Epoch 4/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.6970 - loss: 0.9693 - val_accuracy: 0.6611 - val_loss: 1.2598
Epoch 5/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7165 - loss: 0.9037 - val_accuracy: 0.6041 - val_loss: 1.3692
Epoch 6/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 72ms/step - accuracy: 0.7293 - loss: 0.8612 - val_accuracy: 0.6718 - val_loss: 1.0125
Epoch 7/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7422 - loss: 0.8207 - val_accuracy: 0.1095 - val_loss: 15888.4795
Epoch 8/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7163 - loss: 0.9071 - val_accuracy: 0.6336 - val_loss: 1.1677
Epoch 9/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7452 - loss: 0.8054 - val_accuracy: 0.4326 - val_loss: 1.9414
Epoch 10/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7239 - loss: 0.8829 - val_accuracy: 0.5505 - val_loss: 2.0953
Epoch 11/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7049 - loss: 0.9413 - val_accuracy: 0.6559 - val_loss: 1.1037
Epoch 12/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7442 - loss: 0.8106 - val_accuracy: 0.6978 - val_loss: 0.9768
Epoch 13/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7643 - loss: 0.7458 - val_accuracy: 0.6363 - val_loss: 1.1337
Epoch 14/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7702 - loss: 0.7268 - val_accuracy: 0.5960 - val_loss: 719.3384
Epoch 15/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.6652 - loss: 1.0875 - val_accuracy: 0.5826 - val_loss: 14.4363
Epoch 16/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.6999 - loss: 0.9540 - val_accuracy: 0.4971 - val_loss: 3.1220
Epoch 17/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7272 - loss: 0.8654 - val_accuracy: 0.5095 - val_loss: 4.9449
Epoch 18/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7410 - loss: 0.8200 - val_accuracy: 0.6976 - val_loss: 0.9777
Epoch 19/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7623 - loss: 0.7529 - val_accuracy: 0.2299 - val_loss: 45.1476
Epoch 20/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7574 - loss: 0.7647 - val_accuracy: 0.0116 - val_loss: 5.0363
Epoch 21/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7537 - loss: 0.7782 - val_accuracy: 0.5814 - val_loss: 1.2912
Epoch 22/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7664 - loss: 0.7357 - val_accuracy: 0.3763 - val_loss: 1.7886
Epoch 23/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 71ms/step - accuracy: 0.7332 - loss: 0.8406 - val_accuracy: 0.3551 - val_loss: 11.2559
Epoch 24/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 72ms/step - accuracy: 0.6643 - loss: 1.0869 - val_accuracy: 0.5804 - val_loss: 1.8700
Epoch 25/25
 250/250 ━━━━━━━━━━━━━━━━━━━━ 18s 72ms/step - accuracy: 0.6302 - loss: 1.1875 - val_accuracy: 0.6420 - val_loss: 1.1440

```
</div>
    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_12_1.png)
    



    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_12_2.png)
    



    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_12_3.png)
    



    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_12_4.png)
    


---
## Inference using Colormap Overlay

The raw predictions from the model represent a one-hot encoded tensor of shape `(N, 512, 512, 20)`
where each one of the 20 channels is a binary mask corresponding to a predicted label.
In order to visualize the results, we plot them as RGB segmentation masks where each pixel
is represented by a unique color corresponding to the particular label predicted. We can easily
find the color corresponding to each label from the `human_colormap.mat` file provided as part
of the dataset. We would also plot an overlay of the RGB segmentation mask on the input image as
this further helps us to identify the different categories present in the image more intuitively.


```python
# Loading the Colormap
colormap = loadmat(
    "./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat"
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = keras.utils.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(keras.utils.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()


def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], figsize=(18, 14)
        )

```

### Inference on Train Images


```python
plot_predictions(train_images[:4], colormap, model=model)
```

<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 7s 7s/step

```
</div>
    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_16_1.png)
    


<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step

```
</div>
    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_16_3.png)
    


<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step

```
</div>
    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_16_5.png)
    


<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step

```
</div>
    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_16_7.png)
    


### Inference on Validation Images

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/deeplabv3p-resnet50)
and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/Human-Part-Segmentation).


```python
plot_predictions(val_images[:4], colormap, model=model)
```

<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step

```
</div>
    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_18_1.png)
    


<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step

```
</div>
    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_18_3.png)
    


<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step

```
</div>
    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_18_5.png)
    


<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step

```
</div>
    
![png](/img/examples/vision/deeplabv3_plus/deeplabv3_plus_18_7.png)
    


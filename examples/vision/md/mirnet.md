# Low-light image enhancement using MIRNet

**Author:** [Soumik Rakshit](http://github.com/soumik12345)<br>
**Date created:** 2021/09/11<br>
**Last modified:** 2021/09/15<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/mirnet.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/mirnet.py)


**Description:** Implementing the MIRNet architecture for low-light image enhancement.

---
## Introduction

With the goal of recovering high-quality image content from its degraded version, image
restoration enjoys numerous applications, such as in
photography, security, medical imaging, and remote sensing. In this example, we implement the
**MIRNet** model for low-light image enhancement, a fully-convolutional architecture that
learns an enriched set of
features that combines contextual information from multiple scales, while
simultaneously preserving the high-resolution spatial details.

### References:

- [Learning Enriched Features for Real Image Restoration and Enhancement](https://arxiv.org/abs/2003.06792)
- [The Retinex Theory of Color Vision](http://www.cnbc.cmu.edu/~tai/cp_papers/E.Land_Retinex_Theory_ScientifcAmerican.pdf)
- [Two deterministic half-quadratic regularization algorithms for computed imaging](https://ieeexplore.ieee.org/document/413553)

---
## Downloading LOLDataset

The **LoL Dataset** has been created for low-light image enhancement.
It provides 485 images for training and 15 for testing. Each image pair in the dataset
consists of a low-light input image and its corresponding well-exposed reference image.


```python
import os
import cv2
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
To: /content/keras-io/scripts/tmp_2614641/lol_dataset.zip
347MB [00:03, 108MB/s]

```
</div>
---
## Creating a TensorFlow Dataset

We use 300 image pairs from the LoL Dataset's training set for training,
and we use the remaining 185 image pairs for validation.
We generate random crops of size `128 x 128` from the image pairs to be
used for both training and validation.


```python
random.seed(10)

IMAGE_SIZE = 128
BATCH_SIZE = 4
MAX_TRAIN_IMAGES = 300


def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.cast(image, dtype=tf.float32) / 255.0
    return image


def random_crop(low_image, enhanced_image):
    low_image_shape = tf.shape(low_image)[:2]
    low_w = tf.random.uniform(
        shape=(), maxval=low_image_shape[1] - IMAGE_SIZE + 1, dtype=tf.int32
    )
    low_h = tf.random.uniform(
        shape=(), maxval=low_image_shape[0] - IMAGE_SIZE + 1, dtype=tf.int32
    )
    enhanced_w = low_w
    enhanced_h = low_h
    low_image_cropped = low_image[
        low_h : low_h + IMAGE_SIZE, low_w : low_w + IMAGE_SIZE
    ]
    enhanced_image_cropped = enhanced_image[
        enhanced_h : enhanced_h + IMAGE_SIZE, enhanced_w : enhanced_w + IMAGE_SIZE
    ]
    return low_image_cropped, enhanced_image_cropped


def load_data(low_light_image_path, enhanced_image_path):
    low_light_image = read_image(low_light_image_path)
    enhanced_image = read_image(enhanced_image_path)
    low_light_image, enhanced_image = random_crop(low_light_image, enhanced_image)
    return low_light_image, enhanced_image


def get_dataset(low_light_images, enhanced_images):
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images, enhanced_images))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


train_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[:MAX_TRAIN_IMAGES]
train_enhanced_images = sorted(glob("./lol_dataset/our485/high/*"))[:MAX_TRAIN_IMAGES]

val_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[MAX_TRAIN_IMAGES:]
val_enhanced_images = sorted(glob("./lol_dataset/our485/high/*"))[MAX_TRAIN_IMAGES:]

test_low_light_images = sorted(glob("./lol_dataset/eval15/low/*"))
test_enhanced_images = sorted(glob("./lol_dataset/eval15/high/*"))


train_dataset = get_dataset(train_low_light_images, train_enhanced_images)
val_dataset = get_dataset(val_low_light_images, val_enhanced_images)


print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)
```

<div class="k-default-codeblock">
```
Train Dataset: <BatchDataset shapes: ((4, None, None, 3), (4, None, None, 3)), types: (tf.float32, tf.float32)>
Val Dataset: <BatchDataset shapes: ((4, None, None, 3), (4, None, None, 3)), types: (tf.float32, tf.float32)>

```
</div>
---
## MIRNet Model

Here are the main features of the MIRNet model:

- A feature extraction model that computes a complementary set of features across multiple
spatial scales, while maintaining the original high-resolution features to preserve
precise spatial details.
- A regularly repeated mechanism for information exchange, where the features across
multi-resolution branches are progressively fused together for improved representation
learning.
- A new approach to fuse multi-scale features using a selective kernel network
that dynamically combines variable receptive fields and faithfully preserves
the original feature information at each spatial resolution.
- A recursive residual design that progressively breaks down the input signal
in order to simplify the overall learning process, and allows the construction
of very deep networks.


![](https://raw.githubusercontent.com/soumik12345/MIRNet/master/assets/mirnet_architecture.png)

### Selective Kernel Feature Fusion

The Selective Kernel Feature Fusion or SKFF module performs dynamic adjustment of
receptive fields via two operations: **Fuse** and **Select**. The Fuse operator generates
global feature descriptors by combining the information from multi-resolution streams.
The Select operator uses these descriptors to recalibrate the feature maps (of different
streams) followed by their aggregation.

**Fuse**: The SKFF receives inputs from three parallel convolution streams carrying
different scales of information. We first combine these multi-scale features using an
element-wise sum, on which we apply Global Average Pooling (GAP) across the spatial
dimension. Next, we apply a channel- downscaling convolution layer to generate a compact
feature representation which passes through three parallel channel-upscaling convolution
layers (one for each resolution stream) and provides us with three feature descriptors.

**Select**: This operator applies the softmax function to the feature descriptors to
obtain the corresponding activations that are used to adaptively recalibrate multi-scale
feature maps. The aggregated features are defined as the sum of product of the corresponding
multi-scale feature and the feature descriptor.

![](https://i.imgur.com/7U6ixF6.png)


```python

def selective_kernel_feature_fusion(
    multi_scale_feature_1, multi_scale_feature_2, multi_scale_feature_3
):
    channels = list(multi_scale_feature_1.shape)[-1]
    combined_feature = layers.Add()(
        [multi_scale_feature_1, multi_scale_feature_2, multi_scale_feature_3]
    )
    gap = layers.GlobalAveragePooling2D()(combined_feature)
    channel_wise_statistics = tf.reshape(gap, shape=(-1, 1, 1, channels))
    compact_feature_representation = layers.Conv2D(
        filters=channels // 8, kernel_size=(1, 1), activation="relu"
    )(channel_wise_statistics)
    feature_descriptor_1 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_descriptor_2 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_descriptor_3 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_1 = multi_scale_feature_1 * feature_descriptor_1
    feature_2 = multi_scale_feature_2 * feature_descriptor_2
    feature_3 = multi_scale_feature_3 * feature_descriptor_3
    aggregated_feature = layers.Add()([feature_1, feature_2, feature_3])
    return aggregated_feature

```

### Dual Attention Unit

The Dual Attention Unit or DAU is used to extract features in the convolutional streams.
While the SKFF block fuses information across multi-resolution branches, we also need a
mechanism to share information within a feature tensor, both along the spatial and the
channel dimensions which is done by the DAU block. The DAU suppresses less useful
features and only allows more informative ones to pass further. This feature
recalibration is achieved by using **Channel Attention** and **Spatial Attention**
mechanisms.

The **Channel Attention** branch exploits the inter-channel relationships of the
convolutional feature maps by applying squeeze and excitation operations. Given a feature
map, the squeeze operation applies Global Average Pooling across spatial dimensions to
encode global context, thus yielding a feature descriptor. The excitation operator passes
this feature descriptor through two convolutional layers followed by the sigmoid gating
and generates activations. Finally, the output of Channel Attention branch is obtained by
rescaling the input feature map with the output activations.

The **Spatial Attention** branch is designed to exploit the inter-spatial dependencies of
convolutional features. The goal of Spatial Attention is to generate a spatial attention
map and use it to recalibrate the incoming features. To generate the spatial attention
map, the Spatial Attention branch first independently applies Global Average Pooling and
Max Pooling operations on input features along the channel dimensions and concatenates
the outputs to form a resultant feature map which is then passed through a convolution
and sigmoid activation to obtain the spatial attention map. This spatial attention map is
then used to rescale the input feature map.

![](https://i.imgur.com/Dl0IwQs.png)


```python

def spatial_attention_block(input_tensor):
    average_pooling = tf.reduce_max(input_tensor, axis=-1)
    average_pooling = tf.expand_dims(average_pooling, axis=-1)
    max_pooling = tf.reduce_mean(input_tensor, axis=-1)
    max_pooling = tf.expand_dims(max_pooling, axis=-1)
    concatenated = layers.Concatenate(axis=-1)([average_pooling, max_pooling])
    feature_map = layers.Conv2D(1, kernel_size=(1, 1))(concatenated)
    feature_map = tf.nn.sigmoid(feature_map)
    return input_tensor * feature_map


def channel_attention_block(input_tensor):
    channels = list(input_tensor.shape)[-1]
    average_pooling = layers.GlobalAveragePooling2D()(input_tensor)
    feature_descriptor = tf.reshape(average_pooling, shape=(-1, 1, 1, channels))
    feature_activations = layers.Conv2D(
        filters=channels // 8, kernel_size=(1, 1), activation="relu"
    )(feature_descriptor)
    feature_activations = layers.Conv2D(
        filters=channels, kernel_size=(1, 1), activation="sigmoid"
    )(feature_activations)
    return input_tensor * feature_activations


def dual_attention_unit_block(input_tensor):
    channels = list(input_tensor.shape)[-1]
    feature_map = layers.Conv2D(
        channels, kernel_size=(3, 3), padding="same", activation="relu"
    )(input_tensor)
    feature_map = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(
        feature_map
    )
    channel_attention = channel_attention_block(feature_map)
    spatial_attention = spatial_attention_block(feature_map)
    concatenation = layers.Concatenate(axis=-1)([channel_attention, spatial_attention])
    concatenation = layers.Conv2D(channels, kernel_size=(1, 1))(concatenation)
    return layers.Add()([input_tensor, concatenation])

```

### Multi-Scale Residual Block

The Multi-Scale Residual Block is capable of generating a spatially-precise output by
maintaining high-resolution representations, while receiving rich contextual information
from low-resolutions. The MRB consists of multiple (three in this paper)
fully-convolutional streams connected in parallel. It allows information exchange across
parallel streams in order to consolidate the high-resolution features with the help of
low-resolution features, and vice versa. The MIRNet employs a recursive residual design
(with skip connections) to ease the flow of information during the learning process. In
order to maintain the residual nature of our architecture, residual resizing modules are
used to perform downsampling and upsampling operations that are used in the Multi-scale
Residual Block.

![](https://i.imgur.com/wzZKV57.png)


```python
# Recursive Residual Modules


def down_sampling_module(input_tensor):
    channels = list(input_tensor.shape)[-1]
    main_branch = layers.Conv2D(channels, kernel_size=(1, 1), activation="relu")(
        input_tensor
    )
    main_branch = layers.Conv2D(
        channels, kernel_size=(3, 3), padding="same", activation="relu"
    )(main_branch)
    main_branch = layers.MaxPooling2D()(main_branch)
    main_branch = layers.Conv2D(channels * 2, kernel_size=(1, 1))(main_branch)
    skip_branch = layers.MaxPooling2D()(input_tensor)
    skip_branch = layers.Conv2D(channels * 2, kernel_size=(1, 1))(skip_branch)
    return layers.Add()([skip_branch, main_branch])


def up_sampling_module(input_tensor):
    channels = list(input_tensor.shape)[-1]
    main_branch = layers.Conv2D(channels, kernel_size=(1, 1), activation="relu")(
        input_tensor
    )
    main_branch = layers.Conv2D(
        channels, kernel_size=(3, 3), padding="same", activation="relu"
    )(main_branch)
    main_branch = layers.UpSampling2D()(main_branch)
    main_branch = layers.Conv2D(channels // 2, kernel_size=(1, 1))(main_branch)
    skip_branch = layers.UpSampling2D()(input_tensor)
    skip_branch = layers.Conv2D(channels // 2, kernel_size=(1, 1))(skip_branch)
    return layers.Add()([skip_branch, main_branch])


# MRB Block
def multi_scale_residual_block(input_tensor, channels):
    # features
    level1 = input_tensor
    level2 = down_sampling_module(input_tensor)
    level3 = down_sampling_module(level2)
    # DAU
    level1_dau = dual_attention_unit_block(level1)
    level2_dau = dual_attention_unit_block(level2)
    level3_dau = dual_attention_unit_block(level3)
    # SKFF
    level1_skff = selective_kernel_feature_fusion(
        level1_dau,
        up_sampling_module(level2_dau),
        up_sampling_module(up_sampling_module(level3_dau)),
    )
    level2_skff = selective_kernel_feature_fusion(
        down_sampling_module(level1_dau), level2_dau, up_sampling_module(level3_dau)
    )
    level3_skff = selective_kernel_feature_fusion(
        down_sampling_module(down_sampling_module(level1_dau)),
        down_sampling_module(level2_dau),
        level3_dau,
    )
    # DAU 2
    level1_dau_2 = dual_attention_unit_block(level1_skff)
    level2_dau_2 = up_sampling_module((dual_attention_unit_block(level2_skff)))
    level3_dau_2 = up_sampling_module(
        up_sampling_module(dual_attention_unit_block(level3_skff))
    )
    # SKFF 2
    skff_ = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)
    conv = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(skff_)
    return layers.Add()([input_tensor, conv])

```

### MIRNet Model


```python

def recursive_residual_group(input_tensor, num_mrb, channels):
    conv1 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(input_tensor)
    for _ in range(num_mrb):
        conv1 = multi_scale_residual_block(conv1, channels)
    conv2 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(conv1)
    return layers.Add()([conv2, input_tensor])


def mirnet_model(num_rrg, num_mrb, channels):
    input_tensor = keras.Input(shape=[None, None, 3])
    x1 = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(input_tensor)
    for _ in range(num_rrg):
        x1 = recursive_residual_group(x1, num_mrb, channels)
    conv = layers.Conv2D(3, kernel_size=(3, 3), padding="same")(x1)
    output_tensor = layers.Add()([input_tensor, conv])
    return keras.Model(input_tensor, output_tensor)


model = mirnet_model(num_rrg=3, num_mrb=2, channels=64)
```

---
## Training

- We train MIRNet using **Charbonnier Loss** as the loss function and **Adam
Optimizer** with a learning rate of `1e-4`.
- We use **Peak Signal Noise Ratio** or PSNR as a metric which is an expression for the
ratio between the maximum possible value (power) of a signal and the power of distorting
noise that affects the quality of its representation.


```python

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))


def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


optimizer = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer, loss=charbonnier_loss, metrics=[peak_signal_noise_ratio]
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_peak_signal_noise_ratio",
            factor=0.5,
            patience=5,
            verbose=1,
            min_delta=1e-7,
            mode="max",
        )
    ],
)

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()


plt.plot(history.history["peak_signal_noise_ratio"], label="train_psnr")
plt.plot(history.history["val_peak_signal_noise_ratio"], label="val_psnr")
plt.xlabel("Epochs")
plt.ylabel("PSNR")
plt.title("Train and Validation PSNR Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()
```

<div class="k-default-codeblock">
```
Epoch 1/50
75/75 [==============================] - 109s 731ms/step - loss: 0.2125 - peak_signal_noise_ratio: 62.0458 - val_loss: 0.1592 - val_peak_signal_noise_ratio: 64.1833
Epoch 2/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1764 - peak_signal_noise_ratio: 63.1356 - val_loss: 0.1257 - val_peak_signal_noise_ratio: 65.6498
Epoch 3/50
75/75 [==============================] - 49s 652ms/step - loss: 0.1724 - peak_signal_noise_ratio: 63.3172 - val_loss: 0.1245 - val_peak_signal_noise_ratio: 65.6902
Epoch 4/50
75/75 [==============================] - 49s 653ms/step - loss: 0.1670 - peak_signal_noise_ratio: 63.4917 - val_loss: 0.1206 - val_peak_signal_noise_ratio: 65.8893
Epoch 5/50
75/75 [==============================] - 49s 653ms/step - loss: 0.1651 - peak_signal_noise_ratio: 63.6555 - val_loss: 0.1333 - val_peak_signal_noise_ratio: 65.6338
Epoch 6/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1572 - peak_signal_noise_ratio: 64.1984 - val_loss: 0.1142 - val_peak_signal_noise_ratio: 66.7711
Epoch 7/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1592 - peak_signal_noise_ratio: 64.0062 - val_loss: 0.1205 - val_peak_signal_noise_ratio: 66.1075
Epoch 8/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1493 - peak_signal_noise_ratio: 64.4675 - val_loss: 0.1170 - val_peak_signal_noise_ratio: 66.1355
Epoch 9/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1446 - peak_signal_noise_ratio: 64.7416 - val_loss: 0.1301 - val_peak_signal_noise_ratio: 66.0207
Epoch 10/50
75/75 [==============================] - 49s 655ms/step - loss: 0.1539 - peak_signal_noise_ratio: 64.3999 - val_loss: 0.1220 - val_peak_signal_noise_ratio: 66.7203
Epoch 11/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1451 - peak_signal_noise_ratio: 64.7352 - val_loss: 0.1219 - val_peak_signal_noise_ratio: 66.3140
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 00011: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
Epoch 12/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1492 - peak_signal_noise_ratio: 64.7238 - val_loss: 0.1204 - val_peak_signal_noise_ratio: 66.4726
Epoch 13/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1456 - peak_signal_noise_ratio: 64.9666 - val_loss: 0.1109 - val_peak_signal_noise_ratio: 67.1270
Epoch 14/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1372 - peak_signal_noise_ratio: 65.3932 - val_loss: 0.1150 - val_peak_signal_noise_ratio: 66.9255
Epoch 15/50
75/75 [==============================] - 49s 650ms/step - loss: 0.1340 - peak_signal_noise_ratio: 65.5611 - val_loss: 0.1111 - val_peak_signal_noise_ratio: 67.2009
Epoch 16/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1377 - peak_signal_noise_ratio: 65.3355 - val_loss: 0.1140 - val_peak_signal_noise_ratio: 67.0495
Epoch 17/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1340 - peak_signal_noise_ratio: 65.6484 - val_loss: 0.1132 - val_peak_signal_noise_ratio: 67.0257
Epoch 18/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1360 - peak_signal_noise_ratio: 65.4871 - val_loss: 0.1070 - val_peak_signal_noise_ratio: 67.4185
Epoch 19/50
75/75 [==============================] - 49s 649ms/step - loss: 0.1349 - peak_signal_noise_ratio: 65.4856 - val_loss: 0.1112 - val_peak_signal_noise_ratio: 67.2248
Epoch 20/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1273 - peak_signal_noise_ratio: 66.0817 - val_loss: 0.1185 - val_peak_signal_noise_ratio: 67.0208
Epoch 21/50
75/75 [==============================] - 49s 656ms/step - loss: 0.1393 - peak_signal_noise_ratio: 65.3710 - val_loss: 0.1102 - val_peak_signal_noise_ratio: 67.0362
Epoch 22/50
75/75 [==============================] - 49s 653ms/step - loss: 0.1326 - peak_signal_noise_ratio: 65.8781 - val_loss: 0.1059 - val_peak_signal_noise_ratio: 67.4949
Epoch 23/50
75/75 [==============================] - 49s 653ms/step - loss: 0.1260 - peak_signal_noise_ratio: 66.1770 - val_loss: 0.1187 - val_peak_signal_noise_ratio: 66.6312
Epoch 24/50
75/75 [==============================] - 49s 650ms/step - loss: 0.1331 - peak_signal_noise_ratio: 65.8160 - val_loss: 0.1075 - val_peak_signal_noise_ratio: 67.2668
Epoch 25/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1288 - peak_signal_noise_ratio: 66.0734 - val_loss: 0.1027 - val_peak_signal_noise_ratio: 67.9508
Epoch 26/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1306 - peak_signal_noise_ratio: 66.0349 - val_loss: 0.1076 - val_peak_signal_noise_ratio: 67.3821
Epoch 27/50
75/75 [==============================] - 49s 655ms/step - loss: 0.1356 - peak_signal_noise_ratio: 65.7978 - val_loss: 0.1079 - val_peak_signal_noise_ratio: 67.4785
Epoch 28/50
75/75 [==============================] - 49s 655ms/step - loss: 0.1270 - peak_signal_noise_ratio: 66.2681 - val_loss: 0.1116 - val_peak_signal_noise_ratio: 67.3327
Epoch 29/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1297 - peak_signal_noise_ratio: 66.0506 - val_loss: 0.1057 - val_peak_signal_noise_ratio: 67.5432
Epoch 30/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1275 - peak_signal_noise_ratio: 66.3542 - val_loss: 0.1034 - val_peak_signal_noise_ratio: 67.4624
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 00030: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05.
Epoch 31/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1258 - peak_signal_noise_ratio: 66.2724 - val_loss: 0.1066 - val_peak_signal_noise_ratio: 67.5729
Epoch 32/50
75/75 [==============================] - 49s 653ms/step - loss: 0.1153 - peak_signal_noise_ratio: 67.0384 - val_loss: 0.1064 - val_peak_signal_noise_ratio: 67.4336
Epoch 33/50
75/75 [==============================] - 49s 653ms/step - loss: 0.1189 - peak_signal_noise_ratio: 66.7662 - val_loss: 0.1062 - val_peak_signal_noise_ratio: 67.5128
Epoch 34/50
75/75 [==============================] - 49s 654ms/step - loss: 0.1159 - peak_signal_noise_ratio: 66.9257 - val_loss: 0.1003 - val_peak_signal_noise_ratio: 67.8672
Epoch 35/50
75/75 [==============================] - 49s 653ms/step - loss: 0.1191 - peak_signal_noise_ratio: 66.7690 - val_loss: 0.1043 - val_peak_signal_noise_ratio: 67.4840
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 00035: ReduceLROnPlateau reducing learning rate to 1.249999968422344e-05.
Epoch 36/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1158 - peak_signal_noise_ratio: 67.0264 - val_loss: 0.1057 - val_peak_signal_noise_ratio: 67.6526
Epoch 37/50
75/75 [==============================] - 49s 652ms/step - loss: 0.1128 - peak_signal_noise_ratio: 67.1950 - val_loss: 0.1104 - val_peak_signal_noise_ratio: 67.1770
Epoch 38/50
75/75 [==============================] - 49s 652ms/step - loss: 0.1200 - peak_signal_noise_ratio: 66.7623 - val_loss: 0.1048 - val_peak_signal_noise_ratio: 67.7003
Epoch 39/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1112 - peak_signal_noise_ratio: 67.3895 - val_loss: 0.1031 - val_peak_signal_noise_ratio: 67.6530
Epoch 40/50
75/75 [==============================] - 49s 650ms/step - loss: 0.1125 - peak_signal_noise_ratio: 67.1694 - val_loss: 0.1034 - val_peak_signal_noise_ratio: 67.6437
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 00040: ReduceLROnPlateau reducing learning rate to 6.24999984211172e-06.
Epoch 41/50
75/75 [==============================] - 49s 650ms/step - loss: 0.1131 - peak_signal_noise_ratio: 67.2471 - val_loss: 0.1152 - val_peak_signal_noise_ratio: 66.8625
Epoch 42/50
75/75 [==============================] - 49s 650ms/step - loss: 0.1069 - peak_signal_noise_ratio: 67.5794 - val_loss: 0.1119 - val_peak_signal_noise_ratio: 67.1944
Epoch 43/50
75/75 [==============================] - 49s 651ms/step - loss: 0.1118 - peak_signal_noise_ratio: 67.2779 - val_loss: 0.1147 - val_peak_signal_noise_ratio: 66.9731
Epoch 44/50
75/75 [==============================] - 48s 647ms/step - loss: 0.1101 - peak_signal_noise_ratio: 67.2777 - val_loss: 0.1107 - val_peak_signal_noise_ratio: 67.2580
Epoch 45/50
75/75 [==============================] - 49s 649ms/step - loss: 0.1076 - peak_signal_noise_ratio: 67.6359 - val_loss: 0.1103 - val_peak_signal_noise_ratio: 67.2720
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 00045: ReduceLROnPlateau reducing learning rate to 3.12499992105586e-06.
Epoch 46/50
75/75 [==============================] - 49s 648ms/step - loss: 0.1066 - peak_signal_noise_ratio: 67.4869 - val_loss: 0.1077 - val_peak_signal_noise_ratio: 67.4986
Epoch 47/50
75/75 [==============================] - 49s 649ms/step - loss: 0.1072 - peak_signal_noise_ratio: 67.4890 - val_loss: 0.1140 - val_peak_signal_noise_ratio: 67.1755
Epoch 48/50
75/75 [==============================] - 49s 649ms/step - loss: 0.1065 - peak_signal_noise_ratio: 67.6796 - val_loss: 0.1091 - val_peak_signal_noise_ratio: 67.3442
Epoch 49/50
75/75 [==============================] - 49s 648ms/step - loss: 0.1098 - peak_signal_noise_ratio: 67.3909 - val_loss: 0.1082 - val_peak_signal_noise_ratio: 67.4616
Epoch 50/50
75/75 [==============================] - 49s 648ms/step - loss: 0.1090 - peak_signal_noise_ratio: 67.5139 - val_loss: 0.1124 - val_peak_signal_noise_ratio: 67.1488
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 00050: ReduceLROnPlateau reducing learning rate to 1.56249996052793e-06.

```
</div>
![png](/img/examples/vision/mirnet/mirnet_17_1.png)



![png](/img/examples/vision/mirnet/mirnet_17_2.png)


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
    output = model.predict(image)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    output_image = Image.fromarray(np.uint8(output_image))
    original_image = Image.fromarray(np.uint8(original_image))
    return output_image

```

### Inference on Test Images

We compare the test images from LOLDataset enhanced by MIRNet with images
enhanced via the `PIL.ImageOps.autocontrast()` function.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/lowlight-enhance-mirnet) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/Enhance_Low_Light_Image).

```python

for low_light_image in random.sample(test_low_light_images, 6):
    original_image = Image.open(low_light_image)
    enhanced_image = infer(original_image)
    plot_results(
        [original_image, ImageOps.autocontrast(original_image), enhanced_image],
        ["Original", "PIL Autocontrast", "MIRNet Enhanced"],
        (20, 12),
    )
```


![png](/img/examples/vision/mirnet/mirnet_21_0.png)



![png](/img/examples/vision/mirnet/mirnet_21_1.png)



![png](/img/examples/vision/mirnet/mirnet_21_2.png)



![png](/img/examples/vision/mirnet/mirnet_21_3.png)



![png](/img/examples/vision/mirnet/mirnet_21_4.png)



![png](/img/examples/vision/mirnet/mirnet_21_5.png)
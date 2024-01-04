"""
Title: Image Segmentation using Composable Fully-Convolutional Networks
Author: [Suvaditya Mukherjee](https://twitter.com/halcyonrayes)
Date created: 2023/06/16
Last modified: 2023/12/25
Description: Using the Fully-Convolutional Network for Image Segmentation.
Accelerator: GPU
"""

"""
## Introduction

The following example walks through the steps to implement Fully-Convolutional Networks
for Image Segmentation on the Oxford-IIIT Pets dataset.
The model was proposed in the paper,
[Fully Convolutional Networks for Semantic Segmentation by Long et. al.(2014)](https://arxiv.org/abs/1411.4038).
Image segmentation is one of the most common and introductory tasks when it comes to
Computer Vision, where we extend the problem of Image Classification from
one-label-per-image to a pixel-wise classification problem.
In this example, we will assemble the aforementioned Fully-Convolutional Segmentation architecture,
capable of performing Image Segmentation.
The network extends the pooling layer outputs from the VGG in order to perform upsampling
and get a final result. The intermediate outputs coming from the 3rd, 4th and 5th Max-Pooling layers from VGG19 are
extracted out and upsampled at different levels and factors to get a final output with the same shape as that
of the output, but with the class of each pixel present at each location, instead of pixel intensity values.
Different intermediate pool layers are extracted and processed upon for different versions of the network.
The FCN architecture has 3 versions of differing quality.

- FCN-32S
- FCN-16S
- FCN-8S

All versions of the model derive their outputs through an iterative processing of
successive intermediate pool layers of the main backbone used.
A better idea can be gained from the figure below.

| ![FCN Architecture](https://i.imgur.com/Ttros06.png) |
| :--: |
| **Diagram 1**: Combined Architecture Versions (Source: Paper) |

To get a better idea on Image Segmentation or find more pre-trained models, feel free to
navigate to the [Hugging Face Image Segmentation Models](https://huggingface.co/models?pipeline_tag=image-segmentation) page,
or a [PyImageSearch Blog on Semantic Segmentation](https://pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/)

"""

"""
## Setup Imports
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import ops
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

"""
## Set configurations for notebook variables

We set the required parameters for the experiment.
The chosen dataset has a total of 4 classes per image, with regards to the segmentation mask.
We also set our hyperparameters in this cell.

Mixed Precision as an option is also available in systems which support it, to reduce
load.
This would make most tensors use `16-bit float` values instead of `32-bit float`
values, in places where it will not adversely affect computation.
This means, during computation, TensorFlow will use `16-bit float` Tensors to increase speed at the cost of precision,
while storing the values in their original default `32-bit float` form.
"""

NUM_CLASSES = 4
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
BATCH_SIZE = 32
MIXED_PRECISION = True
SHUFFLE = True

# Mixed-precision setting
if MIXED_PRECISION:
    policy = keras.mixed_precision.Policy("mixed_float16")
    keras.mixed_precision.set_global_policy(policy)

"""
## Load dataset

We make use of the [Oxford-IIIT Pets dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)
which contains a total of 7,349 samples and their segmentation masks.
We have 37 classes, with roughly 200 samples per class.
Our training and validation dataset has 3,128 and 552 samples respectively.
Aside from this, our test split has a total of 3,669 samples.

We set a `batch_size` parameter that will batch our samples together, use a `shuffle`
parameter to mix our samples together.
"""

(train_ds, valid_ds, test_ds) = tfds.load(
    "oxford_iiit_pet",
    split=["train[:85%]", "train[85%:]", "test"],
    batch_size=BATCH_SIZE,
    shuffle_files=SHUFFLE,
)

"""
## Unpack and preprocess dataset

We define a simple function that includes performs Resizing over our
training, validation and test datasets.
We do the same process on the masks as well, to make sure both are aligned in terms of shape and size.
"""


# Image and Mask Pre-processing
def unpack_resize_data(section):
    image = section["image"]
    segmentation_mask = section["segmentation_mask"]

    resize_layer = keras.layers.Resizing(INPUT_HEIGHT, INPUT_WIDTH)

    image = resize_layer(image)
    segmentation_mask = resize_layer(segmentation_mask)

    return image, segmentation_mask


train_ds = train_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
"""
## Visualize one random sample from the pre-processed dataset

We visualize what a random sample in our test split of the dataset looks like, and plot
the segmentation mask on top to see the effective mask areas.
Note that we have performed pre-processing on this dataset too,
which makes the image and mask size same.
"""

# Select random image and mask. Cast to NumPy array
# for Matplotlib visualization.

images, masks = next(iter(test_ds))
random_idx = keras.random.uniform([], minval=0, maxval=BATCH_SIZE, seed=10)

test_image = images[int(random_idx)].numpy().astype("float")
test_mask = masks[int(random_idx)].numpy().astype("float")

# Overlay segmentation mask on top of image.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ax[0].set_title("Image")
ax[0].imshow(test_image / 255.0)

ax[1].set_title("Image with segmentation mask overlay")
ax[1].imshow(test_image / 255.0)
ax[1].imshow(
    test_mask,
    cmap="inferno",
    alpha=0.6,
)
plt.show()

"""
## Perform VGG-specific pre-processing

`keras.applications.VGG19` requires the use of a `preprocess_input` function that will
pro-actively perform Image-net style Standard Deviation Normalization scheme.
"""


def preprocess_data(image, segmentation_mask):
    image = keras.applications.vgg19.preprocess_input(image)

    return image, segmentation_mask


train_ds = (
    train_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=1024)
    .prefetch(buffer_size=1024)
)
valid_ds = (
    valid_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=1024)
    .prefetch(buffer_size=1024)
)
test_ds = (
    test_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=1024)
    .prefetch(buffer_size=1024)
)
"""
## Model Definition

The Fully-Convolutional Network boasts a simple architecture composed of only
`keras.layers.Conv2D` Layers, `keras.layers.Dense` layers and `keras.layers.Dropout`
layers.

| ![FCN Architecture](https://i.imgur.com/PerTKjf.png) |
| :--: |
| **Diagram 2**: Generic FCN Forward Pass (Source: Paper)|

Pixel-wise prediction is performed by having a Softmax Convolutional layer with the same
size of the image, such that we can perform direct comparison
We can find several important metrics such as Accuracy and Mean-Intersection-over-Union on the network.
"""

"""
### Backbone (VGG-19)

We use the [VGG-19 network](https://keras.io/api/applications/vgg/) as the backbone, as
the paper suggests it to be one of the most effective backbones for this network.
We extract different outputs from the network by making use of `keras.models.Model`.
Following this, we add layers on top to make a network perfectly simulating that of
Diagram 1.
The backbone's `keras.layers.Dense` layers will be converted to `keras.layers.Conv2D`
layers based on the [original Caffe code present here.](https://github.com/linxi159/FCN-caffe/blob/master/pascalcontext-fcn16s/net.py)
All 3 networks will share the same backbone weights, but will have differing results
based on their extensions.
We make the backbone non-trainable to improve training time requirements.
It is also noted in the paper that making the network trainable does not yield major benefits.
"""

input_layer = keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))

# VGG Model backbone with pre-trained ImageNet weights.
vgg_model = keras.applications.vgg19.VGG19(include_top=True, weights="imagenet")

# Extracting different outputs from same model
fcn_backbone = keras.models.Model(
    inputs=vgg_model.layers[1].input,
    outputs=[
        vgg_model.get_layer(block_name).output
        for block_name in ["block3_pool", "block4_pool", "block5_pool"]
    ],
)

# Setting backbone to be non-trainable
fcn_backbone.trainable = False

x = fcn_backbone(input_layer)

# Converting Dense layers to Conv2D layers
units = [4096, 4096]
dense_convs = []

for filter_idx in range(len(units)):
    dense_conv = keras.layers.Conv2D(
        filters=units[filter_idx],
        kernel_size=(7, 7) if filter_idx == 0 else (1, 1),
        strides=(1, 1),
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_initializer=keras.initializers.Constant(1.0),
    )
    dense_convs.append(dense_conv)
    dropout_layer = keras.layers.Dropout(0.5)
    dense_convs.append(dropout_layer)

dense_convs = keras.Sequential(dense_convs)
dense_convs.trainable = False

x[-1] = dense_convs(x[-1])

pool3_output, pool4_output, pool5_output = x

"""
### FCN-32S

We extend the last output, perform a `1x1 Convolution` and perform 2D Bilinear Upsampling
by a factor of 32 to get an image of the same size as that of our input.
We use a simple `keras.layers.UpSampling2D` layer over a `keras.layers.Conv2DTranspose`
since it yields performance benefits from being a deterministic mathematical operation
over a Convolutional operation
It is also noted in the paper that making the Up-sampling parameters trainable does not yield benefits.
Original experiments of the paper used Upsampling as well.
"""

# 1x1 convolution to set channels = number of classes
pool5 = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    padding="same",
    strides=(1, 1),
    activation="relu",
)

# Get Softmax outputs for all classes
fcn32s_conv_layer = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    activation="softmax",
    padding="same",
    strides=(1, 1),
)

# Up-sample to original image size
fcn32s_upsampling = keras.layers.UpSampling2D(
    size=(32, 32),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)

final_fcn32s_pool = pool5(pool5_output)
final_fcn32s_output = fcn32s_conv_layer(final_fcn32s_pool)
final_fcn32s_output = fcn32s_upsampling(final_fcn32s_output)

fcn32s_model = keras.Model(inputs=input_layer, outputs=final_fcn32s_output)

"""
### FCN-16S

The pooling output from the FCN-32S is extended and added to the 4th-level Pooling output
of our backbone.
Following this, we upsample by a factor of 16 to get image of the same
size as that of our input.
"""

# 1x1 convolution to set channels = number of classes
# Followed from the original Caffe implementation
pool4 = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    padding="same",
    strides=(1, 1),
    activation="linear",
    kernel_initializer=keras.initializers.Zeros(),
)(pool4_output)

# Intermediate up-sample
pool5 = keras.layers.UpSampling2D(
    size=(2, 2),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)(final_fcn32s_pool)

# Get Softmax outputs for all classes
fcn16s_conv_layer = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    activation="softmax",
    padding="same",
    strides=(1, 1),
)

# Up-sample to original image size
fcn16s_upsample_layer = keras.layers.UpSampling2D(
    size=(16, 16),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)

# Add intermediate outputs
final_fcn16s_pool = keras.layers.Add()([pool4, pool5])
final_fcn16s_output = fcn16s_conv_layer(final_fcn16s_pool)
final_fcn16s_output = fcn16s_upsample_layer(final_fcn16s_output)

fcn16s_model = keras.models.Model(inputs=input_layer, outputs=final_fcn16s_output)

"""
### FCN-8S

The pooling output from the FCN-16S is extended once more, and added from the 3rd-level
Pooling output of our backbone.
This result is upsampled by a factor of 8 to get an image of the same size as that of our input.
"""

# 1x1 convolution to set channels = number of classes
# Followed from the original Caffe implementation
pool3 = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    padding="same",
    strides=(1, 1),
    activation="linear",
    kernel_initializer=keras.initializers.Zeros(),
)(pool3_output)

# Intermediate up-sample
intermediate_pool_output = keras.layers.UpSampling2D(
    size=(2, 2),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)(final_fcn16s_pool)

# Get Softmax outputs for all classes
fcn8s_conv_layer = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    activation="softmax",
    padding="same",
    strides=(1, 1),
)

# Up-sample to original image size
fcn8s_upsample_layer = keras.layers.UpSampling2D(
    size=(8, 8),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)

# Add intermediate outputs
final_fcn8s_pool = keras.layers.Add()([pool3, intermediate_pool_output])
final_fcn8s_output = fcn8s_conv_layer(final_fcn8s_pool)
final_fcn8s_output = fcn8s_upsample_layer(final_fcn8s_output)

fcn8s_model = keras.models.Model(inputs=input_layer, outputs=final_fcn8s_output)

"""
### Load weights into backbone

It was noted in the paper, as well as through experimentation that extracting the weights
of the last 2 Fully-connected Dense layers from the backbone, reshaping the weights to
fit that of the `keras.layers.Dense` layers we had previously converted into
`keras.layers.Conv2D`, and setting them to it yields far better results and a significant
increase in mIOU performance.
"""

# VGG's last 2 layers
weights1 = vgg_model.get_layer("fc1").get_weights()[0]
weights2 = vgg_model.get_layer("fc2").get_weights()[0]

weights1 = weights1.reshape(7, 7, 512, 4096)
weights2 = weights2.reshape(1, 1, 4096, 4096)

dense_convs.layers[0].set_weights([weights1])
dense_convs.layers[2].set_weights([weights2])

"""
## Training

The original paper talks about making use of [SGD with Momentum](https://keras.io/api/optimizers/sgd/) as the optimizer of choice.
But it was noticed during experimentation that
[AdamW](https://keras.io/api/optimizers/adamw/)
yielded better results in terms of mIOU and Pixel-wise Accuracy.
"""

"""
### FCN-32S
"""

fcn32s_optimizer = keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

fcn32s_loss = keras.losses.SparseCategoricalCrossentropy()

# Maintain mIOU and Pixel-wise Accuracy as metrics
fcn32s_model.compile(
    optimizer=fcn32s_optimizer,
    loss=fcn32s_loss,
    metrics=[
        keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False),
        keras.metrics.SparseCategoricalAccuracy(),
    ],
)

fcn32s_history = fcn32s_model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds)

"""
### FCN-16S
"""

fcn16s_optimizer = keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

fcn16s_loss = keras.losses.SparseCategoricalCrossentropy()

# Maintain mIOU and Pixel-wise Accuracy as metrics
fcn16s_model.compile(
    optimizer=fcn16s_optimizer,
    loss=fcn16s_loss,
    metrics=[
        keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False),
        keras.metrics.SparseCategoricalAccuracy(),
    ],
)

fcn16s_history = fcn16s_model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds)

"""
### FCN-8S
"""

fcn8s_optimizer = keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

fcn8s_loss = keras.losses.SparseCategoricalCrossentropy()

# Maintain mIOU and Pixel-wise Accuracy as metrics
fcn8s_model.compile(
    optimizer=fcn8s_optimizer,
    loss=fcn8s_loss,
    metrics=[
        keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False),
        keras.metrics.SparseCategoricalAccuracy(),
    ],
)

fcn8s_history = fcn8s_model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds)
"""
## Visualizations
"""

"""
### Plotting metrics for training run

We perform a comparative study between all 3 versions of the model by tracking training
and validation metrics of Accuracy, Loss and Mean IoU.
"""

total_plots = len(fcn32s_history.history)
cols = total_plots // 2

rows = total_plots // cols

if total_plots % cols != 0:
    rows += 1

# Set all history dictionary objects
fcn32s_dict = fcn32s_history.history
fcn16s_dict = fcn16s_history.history
fcn8s_dict = fcn8s_history.history

pos = range(1, total_plots + 1)
plt.figure(figsize=(15, 10))

for i, ((key_32s, value_32s), (key_16s, value_16s), (key_8s, value_8s)) in enumerate(
    zip(fcn32s_dict.items(), fcn16s_dict.items(), fcn8s_dict.items())
):
    plt.subplot(rows, cols, pos[i])
    plt.plot(range(len(value_32s)), value_32s)
    plt.plot(range(len(value_16s)), value_16s)
    plt.plot(range(len(value_8s)), value_8s)
    plt.title(str(key_32s) + " (combined)")
    plt.legend(["FCN-32S", "FCN-16S", "FCN-8S"])

plt.show()

"""
### Visualizing predicted segmentation masks

To understand the results and see them better, we pick a random image from the test
dataset and perform inference on it to see the masks generated by each model.
Note: For better results, the model must be trained for a higher number of epochs.
"""

images, masks = next(iter(test_ds))
random_idx = keras.random.uniform([], minval=0, maxval=BATCH_SIZE, seed=10)

# Get random test image and mask
test_image = images[int(random_idx)].numpy().astype("float")
test_mask = masks[int(random_idx)].numpy().astype("float")

pred_image = ops.expand_dims(test_image, axis=0)
pred_image = keras.applications.vgg19.preprocess_input(pred_image)

# Perform inference on FCN-32S
pred_mask_32s = fcn32s_model.predict(pred_image, verbose=0).astype("float")
pred_mask_32s = np.argmax(pred_mask_32s, axis=-1)
pred_mask_32s = pred_mask_32s[0, ...]

# Perform inference on FCN-16S
pred_mask_16s = fcn16s_model.predict(pred_image, verbose=0).astype("float")
pred_mask_16s = np.argmax(pred_mask_16s, axis=-1)
pred_mask_16s = pred_mask_16s[0, ...]

# Perform inference on FCN-8S
pred_mask_8s = fcn8s_model.predict(pred_image, verbose=0).astype("float")
pred_mask_8s = np.argmax(pred_mask_8s, axis=-1)
pred_mask_8s = pred_mask_8s[0, ...]

# Plot all results
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

fig.delaxes(ax[0, 2])

ax[0, 0].set_title("Image")
ax[0, 0].imshow(test_image / 255.0)

ax[0, 1].set_title("Image with ground truth overlay")
ax[0, 1].imshow(test_image / 255.0)
ax[0, 1].imshow(
    test_mask,
    cmap="inferno",
    alpha=0.6,
)

ax[1, 0].set_title("Image with FCN-32S mask overlay")
ax[1, 0].imshow(test_image / 255.0)
ax[1, 0].imshow(pred_mask_32s, cmap="inferno", alpha=0.6)

ax[1, 1].set_title("Image with FCN-16S mask overlay")
ax[1, 1].imshow(test_image / 255.0)
ax[1, 1].imshow(pred_mask_16s, cmap="inferno", alpha=0.6)

ax[1, 2].set_title("Image with FCN-8S mask overlay")
ax[1, 2].imshow(test_image / 255.0)
ax[1, 2].imshow(pred_mask_8s, cmap="inferno", alpha=0.6)

plt.show()

"""
## Conclusion

The Fully-Convolutional Network is an exceptionally simple network that has yielded
strong results in Image Segmentation tasks across different benchmarks.
With the advent of better mechanisms like [Attention](https://arxiv.org/abs/1706.03762) as used in
[SegFormer](https://arxiv.org/abs/2105.15203) and
[DeTR](https://arxiv.org/abs/2005.12872), this model serves as a quick way to iterate and
find baselines for this task on unknown data.
"""

"""
## Acknowledgements

I thank [Aritra Roy Gosthipaty](https://twitter.com/ariG23498), [Ayush
Thakur](https://twitter.com/ayushthakur0) and [Ritwik
Raha](https://twitter.com/ritwik_raha) for giving a preliminary review of the example.
I also thank the [Google Developer
Experts](https://developers.google.com/community/experts) program.

"""

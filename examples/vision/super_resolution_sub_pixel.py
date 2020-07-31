"""
Title: Image Super-Resolution Using an Efficient Sub-Pixel CNN
Author: [Xingyu Long](https://github.com/xingyu-long)
Date created: 2020/07/28
Last modified: 2020/07/30
Description: Implementing Super-Resoluion using Efficient sub-pixel model on BSDS500.
"""

"""
## Introduction

ESPCN (Efficient Sub-Pixel CNN), proposed by [Shi, W.2016](https://arxiv.org/abs/1609.05158)
is used to reconstruct the image from low resolution (lowres) to high resolution (highres).
In this paper, they introduce an efficient sub-pixel convolution layers which learns an array of
upscaling filter to upscale the final lowres feature maps into the highres output. This post will
implement the model in this paper and train it with small dataset
[BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz).
"""

"""
## Setup
"""

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

"""
## Load data: BSDS500 dataset

### Download dataset

We use Keras built-in util `keras.utils.get_file` to get the dataset.
"""

dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
data_dir = keras.utils.get_file(origin=dataset_url, fname="BSR", untar=True)

"""
Check the `data_dir` and build data path.
"""

import os

root_dir = os.path.join(data_dir, "BSDS500/data")
print(root_dir)

"""
Prepare train and test dataset path and check the number of train, test and validation
samples.
"""

dataset = os.path.join(root_dir, "images")
train_path = os.path.join(dataset, "train")
test_path = os.path.join(dataset, "test")
valid_path = os.path.join(dataset, "val")

input_img_paths = sorted(
    [
        os.path.join(train_path, fname)
        for fname in os.listdir(train_path)
        if fname.endswith(".jpg")
    ]
)

test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".jpg")
    ]
)

valid_img_paths = sorted(
    [
        os.path.join(valid_path, fname)
        for fname in os.listdir(valid_path)
        if fname.endswith(".jpg")
    ]
)

print("Number of train samples: ", len(input_img_paths))
print("Number of test samples: ", len(test_img_paths))
print("Number of validation samples: ", len(valid_img_paths))

"""
## Visualize the data
"""

from IPython.display import display
from keras.preprocessing.image import load_img

display(load_img(input_img_paths[0]))

"""
## Crop and resize images.

Let's process image data. For the input data, we crop the image and blur it with
`bicubic` method; For the target data, we only do the crop operation.
"""

import PIL


def calculate_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def process_input(input, crop_size, upscale_factor):
    input = input.crop((0, 0, crop_size, crop_size))
    crop_size = crop_size // upscale_factor
    return input.resize((crop_size, crop_size), PIL.Image.BICUBIC)


def process_target(input, crop_size):
    input = input.crop((0, 0, crop_size, crop_size))
    return input


"""
## Visualize input and target data in YCbCr color space.
"""

"""
We luminance in `YCbCr` color space for images because the changes in that space are easy
to see.
"""

from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
import numpy as np

img = load_img(input_img_paths[0])
ycbcr = img.convert("YCbCr")
# Split into y, cr and cr channel and we will put y into model.
# And we can restore image by other two channels at the end of this tutorial.
y, _, _ = ycbcr.split()

upscale_factor = 3
crop_size = 300

crop_size = calculate_crop_size(crop_size, upscale_factor)
print("crop_size is ", crop_size)
input = img_to_array(y)
print("Actual image size is ", input.shape)

input_transform = process_input(y, crop_size, upscale_factor)
print("Input image size after transformation is ", input_transform.size)
display(input_transform)

target_transform = process_target(y, crop_size)
print("Target image size after transformation is ", target_transform.size)
display(target_transform)

"""
## Use `Sequence` class to load and vectorize batches of data.
"""


class BSDS500(tf.keras.utils.Sequence):
    def __init__(
        self,
        input_img_size,
        target_img_size,
        crop_size,
        batch_size,
        input_img_paths,
        upscale_factor=3,
        process_target=None,
        process_input=None,
        channels=1,
    ):
        super(BSDS500).__init__()
        self.batch_size = batch_size
        self.input_img_size = input_img_size
        self.target_img_size = target_img_size
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.input_img_paths = input_img_paths
        self.process_input = process_input
        self.process_target = process_target
        self.channels = channels

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Return tuple (input, target) with corresponding #idx"""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.input_img_paths[i : i + self.batch_size]
        # Build the dataset as `batch_size, img_size[0], img_size[1], channels`
        x = np.zeros(
            (self.batch_size,) + self.input_img_size + (self.channels,), dtype="float32"
        )
        for j, path in enumerate(batch_input_img_paths):
            img, _, _ = load_img(path).convert("YCbCr").split()
            img = self.process_input(img, self.crop_size, self.upscale_factor)
            img = img_to_array(img)
            # Scale to [0, 1]
            img /= 255.0
            x[j] = img

        y = np.zeros(
            (self.batch_size,) + self.target_img_size + (self.channels,),
            dtype="float32",
        )
        for j, path in enumerate(batch_target_img_paths):
            img, _, _ = load_img(path).convert("YCbCr").split()
            img = self.process_target(img, self.crop_size)
            img = img_to_array(img)
            # Scale to [0, 1]
            img /= 255.0
            y[j] = img

        return x, y


"""
## Build a model
"""

crop_size = calculate_crop_size(crop_size, upscale_factor)

input_img_size = (crop_size // upscale_factor, crop_size // upscale_factor)
target_img_size = (crop_size, crop_size)

channels = 1

batch_size = 4


def get_model(upscale_factor=3, channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=input_img_size + (channels,))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    model = keras.Model(inputs, outputs)

    return model


"""
### Prepare train and test data sequences.
"""

train_gen = BSDS500(
    input_img_size=input_img_size,
    target_img_size=target_img_size,
    crop_size=crop_size,
    input_img_paths=input_img_paths,
    process_input=process_input,
    process_target=process_target,
    batch_size=batch_size,
    upscale_factor=upscale_factor,
    channels=channels,
)

valid_gen = BSDS500(
    input_img_size=input_img_size,
    target_img_size=target_img_size,
    crop_size=crop_size,
    input_img_paths=valid_img_paths,
    process_input=process_input,
    process_target=process_target,
    batch_size=batch_size,
    upscale_factor=upscale_factor,
    channels=channels,
)

"""
## Train the model with custom callback and perform evaluation
"""

"""
### Define callback
"""
import math


class ESPCNCallback(keras.callbacks.Callback):
    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print(
            "End epoch {} of training and loss is {:7.4f} "
            "and the average PSNR is {:7.4f}".format(
                epoch, logs["loss"], sum(self.psnr) / len(self.psnr)
            )
        )

    def on_train_batch_end(self, batch, logs=None):
        print(
            "...Training: end of batch {} and loss is {:7.4f}".format(
                batch, logs["loss"]
            )
        )

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))
        print(
            "...Evaluating: end of batch {} and loss is {:7.4f}".format(
                batch, logs["loss"]
            )
        )


model = get_model(upscale_factor=upscale_factor, channels=channels)
model.summary()

callbacks = [ESPCNCallback()]
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

"""
### Train model
"""

epochs = 100

model.compile(
    optimizer=optimizer, loss=loss_fn,
)

model.fit(
    train_gen, epochs=epochs, validation_data=valid_gen, callbacks=callbacks, verbose=0
)

"""
Save model by `model.save()`.
"""

model_path = os.path.join(os.getcwd(), "models")

model.save(model_path)

"""
## Run model prediction and plot the results.
"""

"""
Import `mpl_tookit` to help us zoom in the image and compared with specific area.
"""

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def plot_results(img, prefix, title):
    """Plot the result with zoom-in area."""
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    # Create a new figure with a default 111 subplot.
    fig, ax = plt.subplots()
    im = ax.imshow(img_array[::-1], origin="lower")

    plt.title(title)
    # zoom-factor: 2.0, location: upper-left
    axins = zoomed_inset_axes(ax, 2, loc=2)
    axins.imshow(img_array[::-1], origin="lower")

    # Specify the limits.
    x1, x2, y1, y2 = 200, 300, 100, 200
    # Apply the x-limits.
    axins.set_xlim(x1, x2)
    # Apply the y-limits.
    axins.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Make the line.
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
    plt.savefig(str(prefix) + "-" + title + ".png")
    plt.show()


def predict_result(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )

    return out_img


def get_lr_hr_predict(model, upscale_factor, img_path):
    """Build low-resolution input and high-resolution original
    image and prediction result to compare."""
    test_img = load_img(img_path)

    lowres_img = test_img.resize(
        (test_img.size[0] // upscale_factor, test_img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )

    highres_img_size_h = lowres_img.size[0] * upscale_factor
    highres_img_size_w = lowres_img.size[1] * upscale_factor

    lowres_img_bic = lowres_img.resize((highres_img_size_h, highres_img_size_w))

    out_img = predict_result(model, lowres_img)

    highres_img = test_img.resize((highres_img_size_h, highres_img_size_w))

    return lowres_img_bic, highres_img, out_img


"""
Let's predict a few images and save the results. The dataset they use to train in
paper is ImageNet, you can try it and it should give better performance.
"""

total_bicubic_psnr = 0.0
total_test_psnr = 0.0

for index, test_img_path in enumerate(test_img_paths[40:45]):
    lowres_img, highres_img, predict = get_lr_hr_predict(
        model, upscale_factor, test_img_path
    )
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(predict)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

    print(
        "PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr
    )
    print("PSNR of predict and high resolution is %.4f" % test_psnr)
    plot_results(lowres_img, index, "lowres")
    plot_results(highres_img, index, "highres")
    plot_results(predict, index, "prediction")

print("Avg. PSNR of lr is %.4f" % (total_bicubic_psnr / 5))
print("Avg. PSNR of predict is %.4f" % (total_test_psnr / 5))

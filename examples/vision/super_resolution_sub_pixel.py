"""
Title: Image Super-Resolution Using an Efficient Sub-Pixel CNN
Author: [Xingyu Long](https://github.com/xingyu-long)
Date created: 2020/07/28
Last modified: 2020/07/30
Description: Implementing Super-Resolution using Efficient sub-pixel model on BSDS500.
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

import os
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import load_img
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image_dataset_from_directory

from IPython.display import display

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

root_dir = os.path.join(data_dir, "BSDS500/data")
print(root_dir)

"""
Load train and validation dataset by using `image_dataset_from_directory`.
"""

crop_size = 300
upscale_factor = 3
batch_size = 4

train_ds = image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)

valid_ds = image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
)

"""
Normalization
"""


def normalize(input_image):
    input_image = input_image / 255.0
    return input_image


# Normalize from (0, 255) to (0, 1)
train_ds = train_ds.map(normalize)
valid_ds = valid_ds.map(normalize)

"""
Data visualization
"""

for ds_batch in train_ds.take(1):
    for img in ds_batch:
        display(array_to_img(img))

"""
Prepare test dataset.
"""

dataset = os.path.join(root_dir, "images")
test_path = os.path.join(dataset, "test")

test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".jpg")
    ]
)

"""
## Crop and resize images.

Let's process image data. For the input data, we crop the image and blur it with `area`
method (use `BICUBIC` if you use PIL); For the target data, we only do the crop
operation.
"""

from tensorflow.image import rgb_to_yuv

# Use TF Ops to process.
def process_input(input, crop_size, upscale_factor):
    input = rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    crop_size = crop_size // upscale_factor
    return tf.image.resize(y, [crop_size, crop_size], method="area")


def process_target(input):
    input = rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y


"""
Apply map function on `train_ds` and `vali_ds`.
"""

train_ds = train_ds.map(
    lambda x: (process_input(x, crop_size, upscale_factor), process_target(x))
)

valid_ds = valid_ds.map(
    lambda x: (process_input(x, crop_size, upscale_factor), process_target(x))
)

"""
Let's take a look for input and target data.
"""

for ds_batch in train_ds.take(1):
    for img in ds_batch[0]:
        display(array_to_img(img))
    for img in ds_batch[1]:
        display(array_to_img(img))

"""
## Build a model
"""

input_img_size = (crop_size // upscale_factor, crop_size // upscale_factor)
target_img_size = (crop_size, crop_size)

channels = 1

"""
For the model, we add one more layer and use `relu` activation function instead of `tanh`
in paper, it gives us better performance even though we train model in small epochs.
"""


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
## Train the model with custom callback
"""

"""
Import `mpl_tookit` to help us zoom in the image and compared with specific area.
"""

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

"""
Plot image and save it.
"""

import PIL


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
    img = load_img(img_path)
    img_arr = img_to_array(img)
    lowres_input = img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor

    lowres_img = lowres_input.resize((w, h))
    highres_img = img.resize((w, h))
    out_img = predict_result(model, lowres_input)
    return lowres_img, highres_img, out_img


"""
### Define callbacks to monitor training

PSNR: We use PSNR to evaluate our model and please check it from
[here](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) for more details.
"""

import math


class ESPCNCallback(keras.callbacks.Callback):
    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []
        self.test_img = test_img_paths[0]

    def on_epoch_end(self, epoch, logs=None):
        print(
            "End epoch {} of training and loss is {:7.4f} "
            "and the average PSNR is {:7.4f}".format(
                epoch + 1, logs["loss"], sum(self.psnr) / len(self.psnr)
            )
        )
        _, _, predict = get_lr_hr_predict(self.model, upscale_factor, self.test_img)
        plot_results(predict, "epoch-" + str(epoch), "prediction")

    def on_train_batch_end(self, batch, logs=None):
        if batch % 50 == 0:
            print(
                "...Training: end of batch {} and loss is {:7.4f}".format(
                    batch, logs["loss"]
                )
            )

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))
        if batch % 20 == 0:
            print(
                "...Evaluating: end of batch {} and loss is {:7.4f}".format(
                    batch, logs["loss"]
                )
            )


"""
Define `ModelCheckpoint` and `EarlyStopping` callbacks.
"""

early_stop_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

checkpoint_filepath = "/tmp/checkpoint"

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

model = get_model(upscale_factor=upscale_factor, channels=channels)
model.summary()

callbacks = [ESPCNCallback(), early_stop_callback, model_checkpoint_callback]
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

"""
### Train model
"""

epochs = 50

model.compile(
    optimizer=optimizer, loss=loss_fn,
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=0
)

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)

"""
## Run model prediction and plot the results.
"""

"""
Let's predict a few images and save the results. <br> The dataset they use to train in
paper is ImageNet, you can try it and it should give better performance.
"""

total_bicubic_psnr = 0.0
total_test_psnr = 0.0

for index, test_img_path in enumerate(test_img_paths[50:60]):
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

print("Avg. PSNR of lr is %.4f" % (total_bicubic_psnr / 10))
print("Avg. PSNR of predict is %.4f" % (total_test_psnr / 10))

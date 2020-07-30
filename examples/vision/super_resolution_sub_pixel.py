"""
Title: Super-Resolution Using an Efficient Sub-Pixel CNN
Author: [Xingyu Long](https://github.com/xingyu-long)
Date created: 2020/07/28
Last modified: 2020/07/30
Description: Implementing Super-Resoluion using Efficient sub-pixel model on BSDS500.
"""

"""
## Introduction

ESPCN (Efficient Sub-Pixel CNN), proposed by [Shi, W.
2016](https://arxiv.org/abs/1609.05158) is used to reconstruct the image from low
resolution (LR) to high resolution (HR). In this paper, they introduce an efficient
sub-pixel convolution layers which learns an array of upscaluing filter to upscale the
final LR feature maps into the HR output. This post will implement the model in this
paper and trained with relatively small dataset
[BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz).
"""

"""
## Setup
"""

import tensorflow as tf

"""
## Load data: BSDS500 dataset

### Download data

We use Keras built-in util `keras.utils.get_file` to get the dataset.
"""

dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname="BSR", untar=True)

"""
Check the `data_dir` and build data path.
"""

import os

BSDS500 = os.path.join(data_dir, "BSDS500/data")
print(BSDS500)

"""
Prepare train and test dataset path and check the number of train, test and validation
samples.
"""

dataset = os.path.join(BSDS500, "images")
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
## Visualiza the data
"""

from IPython.display import display
from tensorflow.keras.preprocessing.image import load_img

display(load_img(input_img_paths[0]))

"""
## Crop and resize images.

Since we need to upscale images and we have to process the input data format. First, you
need to define crop size and calculate the acutal crop size by following functions. For
input data, after we get the crop image and blur it with `bicubic` method. For target
data, we only do the crop operation.
"""

import PIL


def calculate_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def transform_input(input, crop_size, upscale_factor):
    input = input.crop((0, 0, crop_size, crop_size))
    crop_size = crop_size // upscale_factor
    return input.resize((crop_size, crop_size), PIL.Image.BICUBIC)


def transform_target(input, crop_size):
    input = input.crop((0, 0, crop_size, crop_size))
    return input


"""
## Visualize input and target data in YCbCr color space.
"""

"""
We luminance in `YCbCr` colour space for images because the changes in that space are
easy to see.
"""

from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
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

input_transform = transform_input(y, crop_size, upscale_factor)
print("Input image size after transformation is ", input_transform.size)
display(input_transform)

target_transform = transform_target(y, crop_size)
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
        transform_target=None,
        transform_input=None,
        channels=1,
    ):
        super(BSDS500).__init__()
        self.batch_size = batch_size
        self.input_img_size = input_img_size
        self.target_img_size = target_img_size
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.input_img_paths = input_img_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
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
            img = self.transform_input(img, self.crop_size, self.upscale_factor)
            img = tf.keras.preprocessing.image.img_to_array(img)
            # Scale to [0, 1]
            img /= 255.0
            x[j] = img

        y = np.zeros(
            (self.batch_size,) + self.target_img_size + (self.channels,),
            dtype="float32",
        )
        for j, path in enumerate(batch_target_img_paths):
            img, _, _ = load_img(path).convert("YCbCr").split()
            img = self.transform_target(img, self.crop_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            # Scale to [0, 1]
            img /= 255.0
            y[j] = img

        return x, y


"""
## Define the model
"""

crop_size = calculate_crop_size(crop_size, upscale_factor)

input_img_size = (crop_size // upscale_factor, crop_size // upscale_factor)
target_img_size = (crop_size, crop_size)

channels = 1

batch_size = 32

# input_shape = (input_img_size[0], input_img_size[1], channels)

from tensorflow.keras.layers import Conv2D


class ESPCN(tf.keras.Model):
    def __init__(self, channels=1, upscale_factor=3):
        super(ESPCN, self).__init__()
        self.upscale_factor = upscale_factor
        self.channels = channels
        self.conv1 = Conv2D(
            64, 5, activation="relu", kernel_initializer="Orthogonal", padding="same"
        )
        self.conv2 = Conv2D(
            64, 3, activation="relu", kernel_initializer="Orthogonal", padding="same"
        )
        self.conv3 = Conv2D(
            32, 3, activation="relu", kernel_initializer="Orthogonal", padding="same"
        )
        self.conv4 = Conv2D(
            channels * upscale_factor ** 2,
            3,
            kernel_initializer="Orthogonal",
            padding="same",
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return tf.nn.depth_to_space(x, self.upscale_factor)


def get_model(upscale_factor=3, channels=1):

    model = ESPCN(channels, upscale_factor)
    return model


"""
ESPCN class create same model with following code.

```
inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.Conv2D(64, 5, activation='relu', kernel_initializer = 'Orthogonal',
padding='same', name="conv1")(inputs)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer = 'Orthogonal',
padding='same', name="conv2")(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer = 'Orthogonal',
padding='same', name="conv3")(x)
x = tf.keras.layers.Conv2D(channels * upscale_factor**2, 3, kernel_initializer =
'Orthogonal', padding='same', name="conv4")(x)
outputs = tf.nn.depth_to_space(x, upscale_factor)

model = tf.keras.Model(inputs, outputs)
```

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 100, 100, 1)]     0
_________________________________________________________________
conv1 (Conv2D)               (None, 100, 100, 64)      1664
_________________________________________________________________
conv2 (Conv2D)               (None, 100, 100, 64)      36928
_________________________________________________________________
conv3 (Conv2D)               (None, 100, 100, 32)      18464
_________________________________________________________________
conv4 (Conv2D)               (None, 100, 100, 9)       2601
_________________________________________________________________
tf_op_layer_DepthToSpace (Te [(None, 300, 300, 1)]     0
=================================================================
Total params: 59,657
Trainable params: 59,657
Non-trainable params: 0
```

"""

"""
### Prepare train and test data sequences.
"""

train_gen = BSDS500(
    input_img_size=input_img_size,
    target_img_size=target_img_size,
    crop_size=crop_size,
    input_img_paths=input_img_paths,
    transform_input=transform_input,
    transform_target=transform_target,
    batch_size=batch_size,
    upscale_factor=upscale_factor,
    channels=channels,
)

valid_gen = BSDS500(
    input_img_size=input_img_size,
    target_img_size=target_img_size,
    crop_size=crop_size,
    input_img_paths=valid_img_paths,
    transform_input=transform_input,
    transform_target=transform_target,
    batch_size=batch_size,
    upscale_factor=upscale_factor,
    channels=channels,
)

"""
## Train the model with custom training loop.
"""

model = get_model(upscale_factor=upscale_factor, channels=channels)
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

import math
import numpy as np

epochs = 50

best_psnr = 0
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch + 1,))
    total_psnr = 0
    total_loss = 0
    validation_loss = 0
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_gen):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        total_loss += loss_value
        print(
            "Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value))
        )
    print(
        "Avg. Loss for each epoch %d: %.4f" % (epoch + 1, total_loss / len(train_gen))
    )
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in valid_gen:
        val_logits = model(x_batch_val, training=False)
        val_loss_val = loss_fn(y_batch_val, val_logits)
        # Calculate PSNR value.
        psnr = 10 * math.log10(1 / val_loss_val)
        total_psnr += psnr
        validation_loss += val_loss_val
    print("Avg. PSNR for each epoch: %.4f" % (total_psnr / len(valid_gen)))
    print(
        "Avg. Validation Loss for each epoch: %.4f" % (validation_loss / len(valid_gen))
    )
    best_psnr = max(best_psnr, float(total_psnr / len(valid_gen)))

print("Best PSNR: %.4f" % best_psnr)

"""
Save model by `model.save()`
"""

model_path = os.path.join(os.getcwd(), "models")

model.save(model_path)

"""
## Predict and plot the results.
"""

"""
Import `mpl_tookit` to help us zoom in the image and compared with specific area.
"""

import matplotlib.pyplot as plt
import numpy as np
import math

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def plot_results(img, title):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    fig, ax = plt.subplots()  # Create a new figure with a default 111 subplot.
    im = ax.imshow(img_array[::-1], origin="lower")

    plt.title(title)

    axins = zoomed_inset_axes(ax, 2, loc=2)  # zoom-factor: 2.0, location: upper-left
    axins.imshow(img_array[::-1], origin="lower")

    x1, x2, y1, y2 = 200, 300, 100, 200  # Specify the limits.
    axins.set_xlim(x1, x2)  # Apply the x-limits.
    axins.set_ylim(y1, y2)  # Apply the y-limits.

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Make the line.
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")

    plt.show()


def predict_result(model, lr_img):
    ycbcr = lr_img.convert("YCbCr")
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
    test_img = load_img(img_path)

    lr_img = test_img.resize(
        (test_img.size[0] // upscale_factor, test_img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )

    hr_img_size_h = lr_img.size[0] * upscale_factor
    hr_img_size_w = lr_img.size[1] * upscale_factor

    lr_img_bic = lr_img.resize((hr_img_size_h, hr_img_size_w))

    out_img = predict_result(model, lr_img)

    hr_img = test_img.resize((hr_img_size_h, hr_img_size_w))

    return lr_img_bic, hr_img, out_img


import matplotlib.pyplot as plt
import PIL

bicubic_psnr = 0.0
test_psnr = 0.0

for test_img_path in test_img_paths[:2]:
    lr, hr, predict = get_lr_hr_predict(model, upscale_factor, test_img_path)
    lr_img = img_to_array(lr)
    hr_img = img_to_array(hr)
    predict_img = img_to_array(predict)
    bicubic_psnr += tf.image.psnr(lr_img, hr_img, max_val=255)
    test_psnr += tf.image.psnr(predict_img, hr_img, max_val=255)
    lr_img = img_to_array(lr)
    hr_img = img_to_array(hr)
    predict_img = img_to_array(predict)

    print("PSNR of lr and hr is %.4f" % tf.image.psnr(lr_img, hr_img, max_val=255))
    print(
        "PSNR of predict and hr is %.4f"
        % tf.image.psnr(predict_img, hr_img, max_val=255)
    )
    plot_results(lr, "low resolution with bicubic")
    plot_results(hr, "Hight resolution")
    plot_results(predict, "Prediction")

print("Avg. PSNR of lr is %.4f" % (bicubic_psnr / 10))
print("Avg. PSNR of predict is %.4f" % (test_psnr / 10))

# Image Super-Resolution Using an Efficient Sub-Pixel CNN

**Author:** [Xingyu Long](https://github.com/xingyu-long)<br>
**Date created:** 2020/07/28<br>
**Last modified:** 2020/08/27<br>
**Description:** Implementing Super-Resolution using Efficient sub-pixel model on BSDS500.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/visionipynb/super_resolution_sub_pixel.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/visionsuper_resolution_sub_pixel.py)



---
## Introduction

ESPCN (Efficient Sub-Pixel CNN), proposed by [Shi, 2016](https://arxiv.org/abs/1609.05158)
is a model that reconstructs a high-resolution version of an image given a low-resolution version.
It leverages efficient "sub-pixel convolution" layers, which learns an array of
image upscaling filters.

In this code example, we will implement the model from the paper and train it on a small dataset,
[BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).

---
## Setup


```python
import tensorflow as tf

import os
import math
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import load_img
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image_dataset_from_directory

from IPython.display import display
```

---
## Load data: BSDS500 dataset

### Download dataset

We use the built-in `keras.utils.get_file` utility to retrieve the dataset.


```python
dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
data_dir = keras.utils.get_file(origin=dataset_url, fname="BSR", untar=True)
root_dir = os.path.join(data_dir, "BSDS500/data")
```

We create training and validation datasets via `image_dataset_from_directory`.


```python
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
```

<div class="k-default-codeblock">
```
Found 500 files belonging to 2 classes.
Using 400 files for training.
Found 500 files belonging to 2 classes.
Using 100 files for validation.

```
</div>
We rescale the images to take values in the range [0, 1].


```python

def scaling(input_image):
    input_image = input_image / 255.0
    return input_image


# Scale from (0, 255) to (0, 1)
train_ds = train_ds.map(scaling)
valid_ds = valid_ds.map(scaling)
```

Let's visualize a few sample images:


```python
for batch in train_ds.take(1):
    for img in batch:
        display(array_to_img(img))
```


![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_11_0.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_11_1.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_11_2.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_11_3.png)


We prepare a dataset of test image paths that we will use for
visual evaluation at the end of this example.


```python
dataset = os.path.join(root_dir, "images")
test_path = os.path.join(dataset, "test")

test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".jpg")
    ]
)
```

---
## Crop and resize images

Let's process image data.
First, we convert our images from the RGB color space to the
[YUV colour space](https://en.wikipedia.org/wiki/YUV).

For the input data (low-resolution images),
we crop the image, retrieve the `y` channel (luninance),
and resize it with the `area` method (use `BICUBIC` if you use PIL).
We only consider the luminance channel
in the YUV color space because humans are more sensitive to
luminance change.

For the target data (high-resolution images), we just crop the image
and retrieve the `y` channel.


```python

# Use TF Ops to process.
def process_input(input, crop_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    crop_size = crop_size // upscale_factor
    return tf.image.resize(y, [crop_size, crop_size], method="area")


def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y


train_ds = train_ds.map(
    lambda x: (process_input(x, crop_size, upscale_factor), process_target(x))
)

valid_ds = valid_ds.map(
    lambda x: (process_input(x, crop_size, upscale_factor), process_target(x))
)
```

Let's take a look at the input and target data.


```python
for batch in train_ds.take(1):
    for img in batch[0]:
        display(array_to_img(img))
    for img in batch[1]:
        display(array_to_img(img))
```


![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_17_0.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_17_1.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_17_2.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_17_3.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_17_4.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_17_5.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_17_6.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_17_7.png)


---
## Build a model

Compared to the paper, we add one more layer and we use the `relu` activation function
instead of `tanh`.
It achieves better performance even though we train the model for fewer epochs.


```python
input_img_size = (crop_size // upscale_factor, crop_size // upscale_factor)
target_img_size = (crop_size, crop_size)
channels = 1


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

    return keras.Model(inputs, outputs)

```

---
## Define utility functions

We need to define several utility functions to monitor our results:

- `plot_results` to plot an save an image.
- `get_lowres_image` to convert an image to its low-resolution version.
- `upscale_image` to turn a low-resolution image to
a high-resolution version reconstructed by the model.
In this function, we use the `y` channel from the YUV color space
as input to the model and then combine the output with the
other channels to obtain an RGB image.


```python
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
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


def get_lowres_image(img, upscale_factor):
    """Return low-resolution image and will use it as input."""
    lowres_img = img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )
    return lowres_img


def upscale_image(model, img):
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

```

---
## Define callbacks to monitor training

The `ESPCNCallback` object will compute and display
the [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) metric.
This is the main metric we use to evaluate super-resolution performance.


```python

class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self):
        super(ESPCNCallback, self).__init__()
        self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        prediction = upscale_image(self.model, self.test_img)
        plot_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))

```

Define `ModelCheckpoint` and `EarlyStopping` callbacks.


```python
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

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

callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
```

<div class="k-default-codeblock">
```
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 100, 100, 1)]     0         
_________________________________________________________________
conv2d (Conv2D)              (None, 100, 100, 64)      1664      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 100, 100, 64)      36928     
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 100, 100, 32)      18464     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 100, 100, 9)       2601      
_________________________________________________________________
tf_op_layer_DepthToSpace (Te [(None, 300, 300, 1)]     0         
=================================================================
Total params: 59,657
Trainable params: 59,657
Non-trainable params: 0
_________________________________________________________________

```
</div>
---
## Train the model


```python
epochs = 50

model.compile(
    optimizer=optimizer, loss=loss_fn,
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2
)

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)
```

<div class="k-default-codeblock">
```
Epoch 1/50
Mean PSNR for epoch: 24.26
WARNING:tensorflow:Model was constructed with shape (None, 100, 100, 1) for input Tensor("input_1:0", shape=(None, 100, 100, 1), dtype=float32), but it was called on an input with incompatible shape (None, 107, 160, 1).

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_1.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0276 - val_loss: 0.0038
Epoch 2/50
Mean PSNR for epoch: 26.05

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_3.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0035 - val_loss: 0.0027
Epoch 3/50
Mean PSNR for epoch: 25.96

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_5.png)


<div class="k-default-codeblock">
```
100/100 - 17s - loss: 0.0032 - val_loss: 0.0026
Epoch 4/50
Mean PSNR for epoch: 26.27

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_7.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0029 - val_loss: 0.0025
Epoch 5/50
Mean PSNR for epoch: 26.67

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_9.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0028 - val_loss: 0.0025
Epoch 6/50
Mean PSNR for epoch: 26.26

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_11.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0027 - val_loss: 0.0024
Epoch 7/50
Mean PSNR for epoch: 26.75

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_13.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0028 - val_loss: 0.0024
Epoch 8/50
Mean PSNR for epoch: 25.46

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_15.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0030 - val_loss: 0.0029
Epoch 9/50
Mean PSNR for epoch: 26.35

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_17.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0028 - val_loss: 0.0024
Epoch 10/50
Mean PSNR for epoch: 26.73

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_19.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0027 - val_loss: 0.0024
Epoch 11/50
Mean PSNR for epoch: 26.62

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_21.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0027 - val_loss: 0.0024
Epoch 12/50
Mean PSNR for epoch: 26.33

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_23.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0026 - val_loss: 0.0024
Epoch 13/50
Mean PSNR for epoch: 26.58

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_25.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0026 - val_loss: 0.0023
Epoch 14/50
Mean PSNR for epoch: 26.42

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_27.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0026 - val_loss: 0.0023
Epoch 15/50
Mean PSNR for epoch: 26.67

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_29.png)


<div class="k-default-codeblock">
```
100/100 - 20s - loss: 0.0026 - val_loss: 0.0023
Epoch 16/50
Mean PSNR for epoch: 26.60

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_31.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0026 - val_loss: 0.0023
Epoch 17/50
Mean PSNR for epoch: 26.26

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_33.png)


<div class="k-default-codeblock">
```
100/100 - 20s - loss: 0.0028 - val_loss: 0.0024
Epoch 18/50
Mean PSNR for epoch: 26.63

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_35.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0026 - val_loss: 0.0023
Epoch 19/50
Mean PSNR for epoch: 26.73

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_37.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0026 - val_loss: 0.0023
Epoch 20/50
Mean PSNR for epoch: 26.77

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_39.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0026 - val_loss: 0.0023
Epoch 21/50
Mean PSNR for epoch: 26.90

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_41.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0026 - val_loss: 0.0023
Epoch 22/50
Mean PSNR for epoch: 26.77

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_43.png)


<div class="k-default-codeblock">
```
100/100 - 17s - loss: 0.0026 - val_loss: 0.0023
Epoch 23/50
Mean PSNR for epoch: 25.86

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_45.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0028 - val_loss: 0.0027
Epoch 24/50
Mean PSNR for epoch: 26.89

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_47.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0026 - val_loss: 0.0023
Epoch 25/50
Mean PSNR for epoch: 26.84

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_49.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0026 - val_loss: 0.0023
Epoch 26/50
Mean PSNR for epoch: 26.67

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_51.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0025 - val_loss: 0.0023
Epoch 27/50
Mean PSNR for epoch: 26.73

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_53.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0026 - val_loss: 0.0023
Epoch 28/50
Mean PSNR for epoch: 26.76

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_55.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0026 - val_loss: 0.0023
Epoch 29/50
Mean PSNR for epoch: 26.57

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_57.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0025 - val_loss: 0.0023
Epoch 30/50
Mean PSNR for epoch: 24.39

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_59.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0026 - val_loss: 0.0036
Epoch 31/50
Mean PSNR for epoch: 26.42

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_61.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0033 - val_loss: 0.0023
Epoch 32/50
Mean PSNR for epoch: 26.61

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_63.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0023
Epoch 33/50
Mean PSNR for epoch: 26.85

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_65.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0025 - val_loss: 0.0023
Epoch 34/50
Mean PSNR for epoch: 26.81

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_67.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0025 - val_loss: 0.0023
Epoch 35/50
Mean PSNR for epoch: 26.74

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_69.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0023
Epoch 36/50
Mean PSNR for epoch: 26.80

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_71.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0022
Epoch 37/50
Mean PSNR for epoch: 26.67

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_73.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0022
Epoch 38/50
Mean PSNR for epoch: 26.68

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_75.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0022
Epoch 39/50
Mean PSNR for epoch: 26.61

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_77.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0022
Epoch 40/50
Mean PSNR for epoch: 26.76

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_79.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0022
Epoch 41/50
Mean PSNR for epoch: 26.98

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_81.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0022
Epoch 42/50
Mean PSNR for epoch: 26.91

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_83.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0022
Epoch 43/50
Mean PSNR for epoch: 26.84

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_85.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0022
Epoch 44/50
Mean PSNR for epoch: 26.87

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_87.png)


<div class="k-default-codeblock">
```
100/100 - 20s - loss: 0.0025 - val_loss: 0.0022
Epoch 45/50
Mean PSNR for epoch: 26.59

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_89.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0023
Epoch 46/50
Mean PSNR for epoch: 26.62

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_91.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0023
Epoch 47/50
Mean PSNR for epoch: 26.63

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_93.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0025 - val_loss: 0.0022
Epoch 48/50
Mean PSNR for epoch: 26.60

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_95.png)


<div class="k-default-codeblock">
```
100/100 - 19s - loss: 0.0026 - val_loss: 0.0022
Epoch 49/50
Mean PSNR for epoch: 26.80

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_97.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0025 - val_loss: 0.0022
Epoch 50/50
Mean PSNR for epoch: 26.55

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_27_99.png)


<div class="k-default-codeblock">
```
100/100 - 18s - loss: 0.0025 - val_loss: 0.0022

<tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7fec2047cd90>

```
</div>
---
## Run model prediction and plot the results

Let's compute the reconstructed version of a few images and save the results.


```python
total_bicubic_psnr = 0.0
total_test_psnr = 0.0

for index, test_img_path in enumerate(test_img_paths[50:60]):
    img = load_img(test_img_path)
    lowres_input = get_lowres_image(img, upscale_factor)
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    prediction = upscale_image(model, lowres_input)
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
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
    plot_results(prediction, index, "prediction")

print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))
```

<div class="k-default-codeblock">
```
PSNR of low resolution image and high resolution image is 29.8502
PSNR of predict and high resolution is 30.3193

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_1.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_2.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_3.png)


<div class="k-default-codeblock">
```
PSNR of low resolution image and high resolution image is 24.9783
PSNR of predict and high resolution is 25.8983

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_5.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_6.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_7.png)


<div class="k-default-codeblock">
```
WARNING:tensorflow:Model was constructed with shape (None, 100, 100, 1) for input Tensor("input_1:0", shape=(None, 100, 100, 1), dtype=float32), but it was called on an input with incompatible shape (None, 160, 107, 1).
PSNR of low resolution image and high resolution image is 27.7724
PSNR of predict and high resolution is 28.3171

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_9.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_10.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_11.png)


<div class="k-default-codeblock">
```
PSNR of low resolution image and high resolution image is 28.0314
PSNR of predict and high resolution is 28.2335

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_13.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_14.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_15.png)


<div class="k-default-codeblock">
```
PSNR of low resolution image and high resolution image is 25.7630
PSNR of predict and high resolution is 26.3181

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_17.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_18.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_19.png)


<div class="k-default-codeblock">
```
PSNR of low resolution image and high resolution image is 25.7874
PSNR of predict and high resolution is 26.5331

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_21.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_22.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_23.png)


<div class="k-default-codeblock">
```
PSNR of low resolution image and high resolution image is 26.2512
PSNR of predict and high resolution is 27.1049

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_25.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_26.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_27.png)


<div class="k-default-codeblock">
```
PSNR of low resolution image and high resolution image is 23.3820
PSNR of predict and high resolution is 24.6607

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_29.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_30.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_31.png)


<div class="k-default-codeblock">
```
PSNR of low resolution image and high resolution image is 29.8914
PSNR of predict and high resolution is 30.0392

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_33.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_34.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_35.png)


<div class="k-default-codeblock">
```
PSNR of low resolution image and high resolution image is 25.1712
PSNR of predict and high resolution is 25.6792

```
</div>
![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_37.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_38.png)



![png](/img/examples/vision/super_resolution_sub_pixel/super_resolution_sub_pixel_29_39.png)


<div class="k-default-codeblock">
```
Avg. PSNR of lowres images is 26.6879
Avg. PSNR of reconstructions is 27.3103

```
</div>
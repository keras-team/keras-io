
# PixelCNN

**Author:** [ADMoreau](https://github.com/ADMoreau)<br>
**Date created:** 2020/05/17<br>
**Last modified:** 2020/05/23<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/pixelcnn.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/pixelcnn.py)


**Description:** PixelCNN implemented in Keras.

---
## Introduction
PixelCNN is a generative model proposed in 2016 by van den Oord et al.
(https://arxiv.org/abs/1606.05328). It is designed to generate images or other data types
from an input vector where the probability distribution of prior elements dictate the
probability distribution of later elements. In the following example images are generated
in this fashion through a masked convolutional kernel that is only capable of using data
from earlier pixels (origin at the top left) to generate later pixels. During inference,
the output of the network is used as a probability ditribution from which pixel values
for the desired generated image are sampled (here, with MNIST, the pixels values are
either black or white) from the network output.



```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

```

##Getting the Data



```python
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
n_residual_blocks = 5
# the data, split between train and test sets
(x, _), (y, _) = keras.datasets.mnist.load_data()
# Concatenate all of the images together
data = np.concatenate((x, y), axis=0)
# round all pixel values less than 33% of the max 256 value to 0
# anything above this value gets rounded up to 1 so that all values are either
# 0 or 1
data = np.where(data < (0.33 * 256), 0, 1)
data = data.astype(np.float32)

```

<div class="k-default-codeblock">
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step

```
</div>
##Create two classes for the requisite Layers for the model



```python
# the first layer to create will be the PixelCNN layer, this layer simply
# builds on the 2D convolutional layer but with the requisite masking included
class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # use said initialized kernel to develop the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# Next we build our residual block layer,
# this is just a normal residual block but with the PixelConvLayer built in
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.a = keras.layers.ReLU()
        self.b = keras.layers.Conv2D(filters=filters, kernel_size=1, activation="relu")
        self.c = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.d = keras.layers.Conv2D(filters=filters, kernel_size=1, activation="relu")

    def call(self, inputs):
        x = self.a(inputs)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        return keras.layers.add([inputs, x])


```

---
## Build the model based on the original paper



```python
inputs = keras.Input(shape=input_shape)
x = PixelConvLayer(
    mask_type="A", filters=128, kernel_size=7, padding="same", activation="relu"
)(inputs)

for _ in range(n_residual_blocks):
    x = ResidualBlock(filters=128)(x)

for _ in range(2):
    x = PixelConvLayer(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )(x)

out = keras.layers.Conv2D(
    filters=1, kernel_size=1, strides=1, activation="sigmoid", padding="valid"
)(x)

pixel_cnn = keras.Model(inputs, out)
adam = keras.optimizers.Adam(learning_rate=0.0001)
pixel_cnn.compile(optimizer=adam, loss="binary_crossentropy")

pixel_cnn.summary()
pixel_cnn.fit(x=data, y=data, batch_size=64, epochs=50, validation_split=0.1)

```

<div class="k-default-codeblock">
```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
pixel_conv_layer (PixelConvL (None, 28, 28, 128)       6400      
_________________________________________________________________
residual_block (ResidualBloc (None, 28, 28, 128)       98624     
_________________________________________________________________
residual_block_1 (ResidualBl (None, 28, 28, 128)       98624     
_________________________________________________________________
residual_block_2 (ResidualBl (None, 28, 28, 128)       98624     
_________________________________________________________________
residual_block_3 (ResidualBl (None, 28, 28, 128)       98624     
_________________________________________________________________
residual_block_4 (ResidualBl (None, 28, 28, 128)       98624     
_________________________________________________________________
pixel_conv_layer_6 (PixelCon (None, 28, 28, 128)       16512     
_________________________________________________________________
pixel_conv_layer_7 (PixelCon (None, 28, 28, 128)       16512     
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 28, 28, 1)         129       
=================================================================
Total params: 532,673
Trainable params: 532,673
Non-trainable params: 0
_________________________________________________________________
Epoch 1/50
  2/985 [..............................] - ETA: 41s - loss: 0.6949WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time. Check your callbacks.
985/985 [==============================] - 83s 84ms/step - loss: 0.1293 - val_loss: 0.0949
Epoch 2/50
985/985 [==============================] - 83s 84ms/step - loss: 0.0930 - val_loss: 0.0920
Epoch 3/50
985/985 [==============================] - 83s 84ms/step - loss: 0.0912 - val_loss: 0.0909
Epoch 4/50
985/985 [==============================] - 83s 84ms/step - loss: 0.0902 - val_loss: 0.0897
Epoch 5/50
985/985 [==============================] - 82s 84ms/step - loss: 0.0896 - val_loss: 0.0892
Epoch 6/50
554/985 [===============>..............] - ETA: 34s - loss: 0.0890

```
</div>
---
## Demonstration

The PixelCNN cannot create the full image at once and must instead create each pixel in
order, append the next created pixel to current image, and feed the image back into the
model to repeat the process.



```python
from IPython.display import Image, display
from tqdm import tqdm
import tensorflow_probability as tfp

# Create an empty array of pixels.
batch = 4
pixels = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
batch, rows, cols, channels = pixels.shape

# Iterate the pixels because generation has to be done sequentially pixel by pixel.
for row in tqdm(range(rows)):
    for col in range(cols):
        for channel in range(channels):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = pixel_cnn.predict(pixels)[:, row, col, channel]
            # Use the probabilities to pick pixel values and append the values to the image
            # frame.
            pixels[:, row, col, channel] = tfp.distributions.Bernoulli(
                probs=probs
            ).sample()


def deprocess_image(x):
    # stack the single channeled black and white image to rgb values.
    x = np.stack((x, x, x), 2)
    # Undo preprocessing
    x *= 255.0
    # Convert to uint8 and clip to the valid range [0, 255]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


# Iterate the generated images and plot them with matplotlib.
for i, pic in enumerate(pixels):
    keras.preprocessing.image.save_img(
        "generated_image_{}.png".format(i), deprocess_image(np.squeeze(pic, -1))
    )

display(Image("generated_image_0.png"))
display(Image("generated_image_1.png"))
display(Image("generated_image_2.png"))
display(Image("generated_image_3.png"))

```

<div class="k-default-codeblock">
```
100%|██████████| 28/28 [00:19<00:00,  1.47it/s]

```
</div>
![png](/img/examples/generative/pixelcnn/pixelcnn_10_1.png)



![png](/img/examples/generative/pixelcnn/pixelcnn_10_2.png)



![png](/img/examples/generative/pixelcnn/pixelcnn_10_3.png)



![png](/img/examples/generative/pixelcnn/pixelcnn_10_4.png)


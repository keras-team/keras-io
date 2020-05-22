
# PixelCNN

**Author:** ADMoreau<br>
**Date created:** 2020/05/17<br>
**Last modified:** 2020/05/20<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/pixelcnn.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/pixelcnn.py)


**Description:** PixelCNN implemented in Keras.


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras, nn
from tensorflow.keras import layers
from tensorflow.python.ops import nn_ops

```

#Getting the Data



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

```

<div class="k-default-codeblock">
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step

```
</div>
#Create two classes for the requisite Layers for the model



```python
# the first layer to create will be the PixelCNN layer, this layer is simply
# the 2D convolutional layer with the masking included
class PixelConvLayer(keras.layers.Conv2D):
    def __init__(self, mask_type, *args, **kwargs):
        super(PixelConvLayer, self).__init__(*args, **kwargs)
        self.mask_type = mask_type
        self.mask = None

    def call(self, inputs):
        if self._recreate_conv_op(inputs):
            self._convolution_op = nn_ops.Convolution(
                inputs.get_shape(),
                filter_shape=self.kernel.shape,
                dilation_rate=self.dilation_rate,
                strides=self.strides,
                padding=self._padding_op,
                data_format=self._conv_op_data_format,
            )

        # Apply causal padding to inputs for Conv1D.
        if self.padding == "causal" and self.__class__.__name__ == "Conv1D":
            inputs = array_ops.pad(inputs, self._compute_causal_padding())

        # The divergence from the original 2D conv. layer is here, where we
        # create the mask depending on which layer we are using
        if self.mask is None:
            kernel_shape = self.kernel.get_shape()
            self.mask = np.zeros(shape=kernel_shape)
            self.mask[: kernel_shape[0] // 2, ...] = 1.0
            self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
            if self.mask_type == "B":
                self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1

        # and here we apply the mask to convoltional kernal
        outputs = self._convolution_op(inputs, self.kernel * self.mask)

        if self.use_bias:
            if self.data_format == "channels_first":
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format="NCHW")
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format="NHWC")

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# Next we build our residual block layer,
# this is just a normal res. block but with the PixelConvLayer built in
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

# Build the model based on the original paper



```python
PixelCNN_input = keras.Input(shape=input_shape)
x = PixelConvLayer(
    mask_type="A", filters=128, kernel_size=7, padding="same", activation="relu"
)(PixelCNN_input)

for _ in range(n_residual_blocks):
    x = ResidualBlock(filters=128)(x)
x = layers.ReLU()(x)

for _ in range(2):
    x = PixelConvLayer(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )(x)

x = keras.layers.Conv2D(
    filters=1, kernel_size=1, strides=1, activation="sigmoid", padding="valid"
)(x)

PixelCNN = keras.Model(PixelCNN_input, x)
adam = keras.optimizers.Adam(learning_rate=0.0001)
PixelCNN.compile(optimizer=adam, loss="binary_crossentropy")

PixelCNN.summary()
PixelCNN.fit(
    x=data.astype("float32"),
    y=data.astype("float32"),
    batch_size=64,
    epochs=50,
    validation_split=0.1,
)

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
re_lu_5 (ReLU)               (None, 28, 28, 128)       0         
_________________________________________________________________
pixel_conv_layer_6 (PixelCon (None, 28, 28, 128)       16512     
_________________________________________________________________
pixel_conv_layer_7 (PixelCon (None, 28, 28, 128)       16512     
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 28, 28, 1)         129       
=================================================================
Total params: 532,673
Trainable params: 532,673
Non-trainable params: 0
_________________________________________________________________
Epoch 1/50
  2/985 [..............................] - ETA: 1:05 - loss: 0.6946WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time. Check your callbacks.
985/985 [==============================] - 131s 133ms/step - loss: 0.1257 - val_loss: 0.0940
Epoch 2/50
985/985 [==============================] - 130s 132ms/step - loss: 0.0927 - val_loss: 0.0922
Epoch 3/50
985/985 [==============================] - 130s 132ms/step - loss: 0.0908 - val_loss: 0.0903
Epoch 4/50
985/985 [==============================] - 130s 132ms/step - loss: 0.0898 - val_loss: 0.0893
Epoch 5/50
985/985 [==============================] - 130s 132ms/step - loss: 0.0893 - val_loss: 0.0888
Epoch 6/50
547/985 [===============>..............] - ETA: 55s - loss: 0.0887

```
</div>
# Demo

The PixelCNN cannot create the full image at once and must instead create each pixel in
order, append the next created pixel to current image, and feed the image back into the
model to repeat the process.



```python
from IPython.display import Image, display
from tqdm import tqdm
from scipy.stats import bernoulli

# Create an empty array of pixels.
batch = 4
pixels = np.zeros(shape=(batch,) + (PixelCNN.input_shape)[1:])
batch, rows, cols, channels = pixels.shape

# Iterate the pixels because generation has to be done sequentially pixel by pixel.
for row in tqdm(range(rows)):
    for col in range(cols):
        for channel in range(channels):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            p = PixelCNN.predict_on_batch(pixels)[:, row, col, channel]
            # Use the probabilities to pick pixel values and append the values to the image
            # frame.
            pixels[:, row, col, channel] = bernoulli.rvs(size=batch, p=p)


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
100%|██████████| 28/28 [00:07<00:00,  3.69it/s]

```
</div>
![png](/img/examples/generative/pixelcnn/pixelcnn_9_1.png)



![png](/img/examples/generative/pixelcnn/pixelcnn_9_2.png)



![png](/img/examples/generative/pixelcnn/pixelcnn_9_3.png)



![png](/img/examples/generative/pixelcnn/pixelcnn_9_4.png)


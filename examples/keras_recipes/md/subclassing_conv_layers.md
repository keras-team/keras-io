# Customizing the convolution operation of a Conv2D layer

**Author:** [lukewood](https://lukewood.xyz)<br>
**Date created:** 11/03/2021<br>
**Last modified:** 11/03/2021<br>
**Description:** This example shows how to implement custom convolution layers using the `Conv.convolution_op()` API.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/subclassing_conv_layers.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/subclassing_conv_layers.py)



---
## Introduction

You may sometimes need to implement custom versions of convolution layers like `Conv1D` and `Conv2D`.
Keras enables you do this without implementing the entire layer from scratch: you can reuse
most of the base convolution layer and just customize the convolution op itself via the
`convolution_op()` method.

This method was introduced in Keras 2.7. So before using the
`convolution_op()` API, ensure that you are running Keras version 2.7.0 or greater.


```python
import tensorflow.keras as keras

print(keras.__version__)
```

<div class="k-default-codeblock">
```
2.7.0

```
</div>
---
## A Simple `StandardizedConv2D` implementation

There are two ways to use the `Conv.convolution_op()` API. The first way
is to override the `convolution_op()` method on a convolution layer subclass.
Using this approach, we can quickly implement a
[StandardizedConv2D](https://arxiv.org/abs/1903.10520) as shown below.


```python
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers
import numpy as np


class StandardizedConv2DWithOverride(layers.Conv2D):
    def convolution_op(self, inputs, kernel):
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        return tf.nn.conv2d(
            inputs,
            (kernel - mean) / tf.sqrt(var + 1e-10),
            padding="VALID",
            strides=list(self.strides),
            name=self.__class__.__name__,
        )

```

The other way to use the `Conv.convolution_op()` API is to directly call the
`convolution_op()` method from the `call()` method of a convolution layer subclass.
A comparable class implemented using this approach is shown below.


```python

class StandardizedConv2DWithCall(layers.Conv2D):
    def call(self, inputs):
        mean, var = tf.nn.moments(self.kernel, axes=[0, 1, 2], keepdims=True)
        result = self.convolution_op(
            inputs, (self.kernel - mean) / tf.sqrt(var + 1e-10)
        )
        if self.use_bias:
            result = result + self.bias
        return result

```

---
## Example Usage

Both of these layers work as drop-in replacements for `Conv2D`. The following
demonstration performs classification on the MNIST dataset.


```python
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=input_shape),
        StandardizedConv2DWithCall(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        StandardizedConv2DWithOverride(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
```

```python
batch_size = 128
epochs = 5

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_split=0.1)
```
<div class="k-default-codeblock">
```
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 standardized_conv2d_with_ca  (None, 26, 26, 32)       320       
 ll (StandardizedConv2DWithC                                     
 all)                                                            
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
 )                                                               
                                                                 
 standardized_conv2d_with_ov  (None, 11, 11, 64)       18496     
 erride (StandardizedConv2DW                                     
 ithOverride)                                                    
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 1600)              0         
                                                                 
 dropout (Dropout)           (None, 1600)              0         
                                                                 
 dense (Dense)               (None, 10)                16010     
                                                                 
=================================================================
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0
_________________________________________________________________

Epoch 1/5
422/422 [==============================] - 7s 15ms/step - loss: 1.8435 - accuracy: 0.8415 - val_loss: 0.1177 - val_accuracy: 0.9660
Epoch 2/5
422/422 [==============================] - 6s 14ms/step - loss: 0.2460 - accuracy: 0.9338 - val_loss: 0.0727 - val_accuracy: 0.9772
Epoch 3/5
422/422 [==============================] - 6s 14ms/step - loss: 0.1600 - accuracy: 0.9541 - val_loss: 0.0537 - val_accuracy: 0.9862
Epoch 4/5
422/422 [==============================] - 6s 14ms/step - loss: 0.1264 - accuracy: 0.9633 - val_loss: 0.0509 - val_accuracy: 0.9845
Epoch 5/5
422/422 [==============================] - 6s 14ms/step - loss: 0.1090 - accuracy: 0.9679 - val_loss: 0.0457 - val_accuracy: 0.9872

<keras.callbacks.History at 0x7f2ff6ed2f10>

```
</div>
---
## Conclusion

The `Conv.convolution_op()` API provides an easy and readable way to implement custom
convolution layers. A `StandardizedConvolution` implementation using the API is quite
terse, consisting of only four lines of code.

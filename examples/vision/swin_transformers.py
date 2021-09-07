"""
Title: CIFAR-10 Image Classificatiion with Swin Transformers
Author: [Rishit Dagli](https://twitter.com/rishit_dagli)
Date created: 2021/09/08
Last modified: 2021/09/08
Description: (one-line text description)
"""
"""
This example implements [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
paper by Liu et al. for image classification, and demonstrates it on the 
[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

Swin Transformer (**S**hifted **Wi**ndow) capably serves as a general-purpose backbone 
for computer vision. Swin Transformer is a hierarchical Transformer whose 
representation is computed with shifted windows. The shifted windowing scheme 
brings greater efficiency by limiting self-attention computation to 
non-overlapping local windows while also allowing for cross-window connection. 
This architecture also has the flexibility to model at various scales and has 
linear computational complexity with respect to image size.

This example requires TensorFlow 2.5 or higher, as well as 
[Matplotlib](https://matplotlib.org/), which can be installed using the 
following command:
"""

"""shell
pip install -U matplotlib
"""

"""
## Setup
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
## Prepare the data

We will now load the CIFAR-10 dataset through 
[tf.keras.datasets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets)
, normalize the images and convert label integers to matrices.
"""

num_classes = 10
input_shape = (32, 32, 3)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[int(y_train[i][0])])
plt.show()

"""
## Configure the hyperparameters

In order to use each pixel as an individual input, you can set `patch_size` to (1, 1).
"""

patch_size = (2, 2) # 2-by-2 sized patches
drop_rate = 0.01 # Dropout rate
num_heads = 8 # Attention heads
embed_dim = 64 # Embedded dimensions
num_mlp = 256 # MLP nodes
qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2 # Size of attention window
shift_size = 1 # Size of shifting

num_patch_x = input_shape[0]//patch_size[0]
num_patch_y = input_shape[1]//patch_size[1]

learning_rate=1e-4
clipvalue=0.5
batch_size=32
num_epochs=20
validation_split=0.1

"""
## Helper Functions

We will now create two helper functions which can help us get a sequence of 
patches from the image, allow us to merge patches to spatial frames and dropout.
"""

def window_partition(x, window_size):
    _, H, W, C = x.get_shape().as_list()
    patch_num_H = H//window_size
    patch_num_W = W//window_size
    x = tf.reshape(x, shape=(-1, patch_num_H, window_size, patch_num_W, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, C))
    return windows

def window_reverse(windows, window_size, H, W, C):
    patch_num_H = H//window_size
    patch_num_W = W//window_size
    x = tf.reshape(windows, shape=(-1, patch_num_H, patch_num_W, window_size, window_size, C))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, H, W, C))    
    return x

class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_num = input_shape[0]
        rank = len(input_shape)
        shape = (batch_num,) + (1,) * (rank - 1)
        random_tensor = (1-self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1-self.drop_prob) * path_mask
        return output

"""
## MLP layer

We will now create a simple multi-layered perceptron with 2 Dense and 2 Dropout layers.
"""

class Mlp(layers.Layer):
    def __init__(self, filter_num, drop=0., **kwargs):
        super(Mlp, self).__init__(**kwargs)
        self.filter_num = filter_num
        self.drop = drop
        
    def call(self, x):
        x = layers.Dense(self.filter_num[0])(x)
        x= layers.Activation(keras.activations.gelu)(x)
        x = layers.Dropout(self.drop)(x)
        x = layers.Dense(self.filter_num[1])(x)
        x = layers.Dropout(self.drop)(x)        
        return x
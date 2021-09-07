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
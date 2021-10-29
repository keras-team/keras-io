# -*- coding: utf-8 -*-
"""
Title: Image classification with CSWin Transformer
Author: [ZhiYong Chang](https://github.com/czy00000)
Date created: 2021/10/29
Last modified: 2021/10/29
Description: Image classification with CSWin Transformer:A General Vision Transformer.
Backbone with Cross-Shaped Windows
"""

"""
## Introduction
This example implements [CSWin Transformer](https://arxiv.org/pdf/2107.00652v2.pdf)
model for image classification, and demonstrates it on the CIFAR-100 dataset.
CSWin Transformer (**C**ross-**S**haped **Win**dow Transformer) is an efficient and
effective Transformer-based backbone for general-purpose vision tasks. CSWin Transformer
introduces a **C**ross-**S**haped **Win**dow self-attention mechanism for
computing self-attention in the horizontal and vertical stripes in parallel that form
a _cross-shaped_ window, with each stripe obtained by splitting the input feature
into stripes of equal width. With CSWin self-attention, CSWin Transformer can achieve
large receptive filed efficiently while keeping the computation cost low.

This example requires TensorFlow 2.5 or higher, as well as
[TensorFlow Addons](https://www.tensorflow.org/addons/overview) package,
which can be installed using the following command:
```python
pip install -U tensorflow-addons
```
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

"""
## Prepare the data
"""

num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

"""
## Configure the hyperparameters
"""

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 40
image_size = 96  # We'll resize input images to this size
label_smoothing = 0.1
validation_split = 0.2


"""
## Use data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.1),
        layers.RandomContrast(factor=0.1),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

"""
## Implement the MLP block
"""


class MLP(layers.Layer):
    def __init__(self, hidden_dim=None, out_dim=None, act_layer=tf.nn.gelu, drop=0):
        super().__init__()
        self.fc1 = layers.Dense(hidden_dim)
        self.act = act_layer
        self.fc2 = layers.Dense(out_dim)
        self.drop = layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


"""
## Implementing helper functions

"""


def image_to_window(image, height_split, width_split):
    _, channel, height, width = image.shape
    image_reshape = tf.reshape(
        image,
        shape=(
            -1,
            channel,
            height // height_split,
            height_split,
            width // width_split,
            width_split,
        ),
    )
    image_reshape = tf.transpose(image_reshape, perm=(0, 2, 4, 3, 5, 1))
    image_permutation = tf.reshape(
        image_reshape, shape=(-1, height_split * width_split, channel)
    )
    return image_permutation


def window_to_image(image_split, height_split, width_split, height, width):

    channel = image_split.shape[2]
    image = tf.reshape(
        image_split,
        shape=(
            -1,
            height // height_split,
            width // width_split,
            height_split,
            width_split,
            channel,
        ),
    )
    image = tf.transpose(image, perm=(0, 1, 3, 2, 4, 5))
    image = tf.reshape(image, shape=(-1, height, width, channel))
    return image


"""
## Implement the cross-shaped window attention

To enlarge the attention area and achieve global self-attention more efficiently,
the authors present the _cross-shaped_ window self-attention mechanism,
which is achieved by performing self-attention in horizontal and vertical
stripes in parallel that form a cross-shaped window. the input
feature will be first linearly projected to `K` heads, and then each head will perform
local self-attention within either the horizontal or vertical stripes.
In other words, the attention area of each token within
one Transformer block is enlarged via multi-head grouping.
"""


class CSWinAttention(layers.Layer):
    def __init__(
        self,
        in_dim,
        resolution,
        idx,
        split_size=7,
        out_dim=None,
        num_heads=8,
        attention_dropout=0,
        projection_dropout=0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = head_dim ** -0.5
        if idx == -1:
            height_split, width_split = self.resolution, self.resolution
        elif idx == 0:
            height_split, width_split = self.resolution, self.split_size
        elif idx == 1:
            width_split, height_split = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.height_split = height_split
        self.width_split = width_split
        self.get_value = layers.Conv2D(
            in_dim, kernel_size=3, strides=1, padding="same", groups=in_dim
        )

        self.attention_dropout = layers.Dropout(attention_dropout)

    def image_to_cswindow(self, x):
        _, num_patches, channel = x.shape
        height = width = int(np.sqrt(num_patches))
        x = tf.transpose(x, perm=(0, 2, 1))
        x = tf.reshape(x, shape=(-1, channel, height, width))
        x = image_to_window(x, self.height_split, self.width_split)
        x = tf.reshape(
            x,
            shape=(
                -1,
                self.height_split * self.width_split,
                self.num_heads,
                channel // self.num_heads,
            ),
        )
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        return x

    def get_local_enhanced_position_embedding(self, x, function):
        _, num_patches, channel = x.shape
        height = width = int(np.sqrt(num_patches))
        x = tf.transpose(x, perm=(0, 2, 1))
        x = tf.reshape(x, shape=(-1, channel, height, width))

        height_split, width_split = self.height_split, self.width_split
        x = tf.reshape(
            x,
            shape=(
                -1,
                channel,
                height // height_split,
                height_split,
                width // width_split,
                width_split,
            ),
        )
        x = tf.transpose(x, perm=(0, 2, 4, 1, 3, 5))
        x = tf.reshape(x, shape=(-1, height_split, width_split, channel))
        local_enhanced_position_embedding = function(x)  ### B', C, H', W'
        local_enhanced_position_embedding = tf.reshape(
            local_enhanced_position_embedding,
            shape=(
                -1,
                self.num_heads,
                channel // self.num_heads,
                height_split * width_split,
            ),
        )
        local_enhanced_position_embedding = tf.transpose(
            local_enhanced_position_embedding, perm=(0, 1, 3, 2)
        )
        x = tf.reshape(
            x,
            shape=(
                -1,
                self.num_heads,
                channel // self.num_heads,
                self.height_split * self.width_split,
            ),
        )
        x = tf.transpose(x, perm=(0, 1, 3, 2))
        return x, local_enhanced_position_embedding

    def call(self, qkv):
        query, key, value = qkv[0], qkv[1], qkv[2]
        height = width = self.resolution
        _, num_patches, channel = query.shape
        assert num_patches == height * width

        query = self.image_to_cswindow(query)
        key = self.image_to_cswindow(key)
        (
            value,
            local_enhanced_position_embedding,
        ) = self.get_local_enhanced_position_embedding(value, self.get_value)

        query = query * self.scale
        key = tf.transpose(key, perm=(0, 1, 3, 2))
        attention = query @ key
        attention = tf.nn.softmax(attention, axis=-1)
        attention = self.attention_dropout(attention)

        x = (attention @ value) + local_enhanced_position_embedding
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        x = tf.reshape(x, shape=(-1, self.height_split * self.width_split, channel))

        x = window_to_image(x, self.height_split, self.width_split, height, width)
        x = tf.reshape(x, shape=(-1, height * width, channel))

        return x


"""
## Implement the cross-shaped window transformer block

"""


class CSWinBlock(layers.Layer):
    def __init__(
        self,
        embedding_dim,
        resolution,
        num_heads,
        split_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        attention_dropout=0.0,
        act_layer=tf.nn.gelu,
        norm_layer=layers.LayerNormalization,
        last_stage=False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patches_resolution = resolution
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = layers.Dense(embedding_dim * 3)
        self.norm1 = norm_layer()

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.projection = layers.Dense(embedding_dim)
        self.projection_dropout = layers.Dropout(drop)

        if last_stage:
            self.attentions = [
                CSWinAttention(
                    embedding_dim,
                    resolution=self.patches_resolution,
                    idx=-1,
                    split_size=split_size,
                    num_heads=num_heads,
                    out_dim=embedding_dim,
                    attention_dropout=attention_dropout,
                    projection_dropout=drop,
                )
                for i in range(self.branch_num)
            ]
        else:
            self.attentions = [
                CSWinAttention(
                    embedding_dim // 2,
                    resolution=self.patches_resolution,
                    idx=i,
                    split_size=split_size,
                    num_heads=num_heads // 2,
                    out_dim=embedding_dim // 2,
                    attention_dropout=attention_dropout,
                    projection_dropout=drop,
                )
                for i in range(self.branch_num)
            ]

        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = MLP(
            hidden_dim=mlp_hidden_dim,
            out_dim=embedding_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer()

    def call(self, x):
        height = width = self.patches_resolution
        _, num_patches, channel = x.shape
        assert num_patches == height * width
        image = self.norm1(x)
        qkv = self.qkv(image)
        qkv = tf.reshape(qkv, shape=(-1, num_patches, 3, channel))
        qkv = tf.transpose(qkv, perm=(2, 0, 1, 3))
        if self.branch_num == 2:
            x1 = self.attentions[0](qkv[:, :, :, : channel // 2])
            x2 = self.attentions[1](qkv[:, :, :, channel // 2 :])
            attention_x = tf.concat([x1, x2], 2)
        else:
            attention_x = self.attentions[0](qkv)
        attention_x = self.projection(attention_x)
        x = x + attention_x
        x = x + self.mlp(self.norm2(x))
        return x


"""
## Implement the merge block

"""


class MergeBlock(layers.Layer):
    def __init__(self, dim, norm_layer=layers.LayerNormalization):
        super().__init__()
        self.conv = layers.Conv2D(dim, 3, 2, padding="same")
        self.norm = norm_layer()

    def call(self, x):
        _, new_num_patches, channel = x.shape
        height = width = int(np.sqrt(new_num_patches))
        x = tf.reshape(x, shape=(-1, height, width, channel))
        x = self.conv(x)
        channel = x.shape[3]
        x = tf.reshape(x, shape=(-1, channel, height // 2 * width // 2))
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.norm(x)

        return x


"""
## Impelemnts the complete CSWin Transformer model

we use the `layers.Conv2D`(7 x 7 convolution layer with stride 4) to extract patch
embedding.To produce a hierarchical representation, the whole network consists of four stages. A
convolution layer (3 × 3, stride 2) is used between two adjacent stages to reduce the number of
tokens and double the channel dimension.
"""


def get_model(
    image_size=image_size,
    num_classes=100,
    embedding_dim=64,
    num_transformer_layers=[2, 2, 2, 2],
    split_sizes=[1, 2, 3, 3],
    num_heads=[2, 4, 4, 2],
    mlp_ratio=4.0,
    drop_rate=0.7,
    attention_dropout=0.7,
    norm_layer=layers.LayerNormalization,
):

    inputs = layers.Input(shape=input_shape)
    # Image augment
    x = data_augmentation(inputs)
    # create patch embedding
    x = layers.Conv2D(embedding_dim, 7, 4, padding="same")(x)
    _, height, width, channel = x.shape
    x = tf.reshape(x, shape=(-1, (image_size // 4) * (image_size // 4), channel))

    heads = num_heads
    # first stage
    for i in range(num_transformer_layers[0]):
        x = CSWinBlock(
            embedding_dim=embedding_dim,
            num_heads=heads[0],
            resolution=image_size // 4,
            mlp_ratio=mlp_ratio,
            split_size=split_sizes[0],
            drop=drop_rate,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )(x)
    merge1 = MergeBlock(embedding_dim * 2)(x)

    embedding_dim = embedding_dim * 2
    # second stage
    for i in range(num_transformer_layers[1]):
        x = CSWinBlock(
            embedding_dim=embedding_dim,
            num_heads=heads[1],
            resolution=image_size // 8,
            mlp_ratio=mlp_ratio,
            split_size=split_sizes[1],
            drop=drop_rate,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )(merge1)
    merge2 = MergeBlock(embedding_dim * 2)(x)
    embedding_dim = embedding_dim * 2
    # third stage
    for i in range(num_transformer_layers[2]):
        x = CSWinBlock(
            embedding_dim=embedding_dim,
            num_heads=heads[2],
            resolution=image_size // 16,
            mlp_ratio=mlp_ratio,
            split_size=split_sizes[2],
            drop=drop_rate,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )(merge2)
    merge3 = MergeBlock(embedding_dim * 2)(x)
    embedding_dim = embedding_dim * 2
    # fourth stage
    for i in range(num_transformer_layers[3]):
        x = CSWinBlock(
            embedding_dim=embedding_dim,
            num_heads=heads[3],
            resolution=image_size // 32,
            mlp_ratio=mlp_ratio,
            split_size=split_sizes[-1],
            drop=drop_rate,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            last_stage=True,
        )(merge3)
    x = layers.GlobalAvgPool1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


"""
## Train on CIFAR-100
"""

model = get_model()
model.compile(
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_split=validation_split,
)

"""
### Let's visualize the training progress of the model
"""

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracys Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

loss, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {round(loss, 2)}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

"""
## Visualizing the CSWin block

"""

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

image = tf.convert_to_tensor([image])
block_index = []
block_outputs = []
for i, layer in enumerate(model.layers):
    if isinstance(layer, CSWinBlock):
        block_index.append(i)
for i, index in enumerate(block_index):
    block_outputs.append(model.layers[index].output)
    i += 1

activation_model = keras.Model(inputs=model.inputs, outputs=block_outputs)
feature_maps = activation_model.predict(image)

for feature_map in feature_maps:
    _, num_sequence, channel = feature_map.shape
    height = width = int(np.sqrt(num_sequence))
    feature_map = tf.reshape(feature_map, shape=(-1, height, width, channel))
    ix = 1
    plt.figure(figsize=(4, 4))
    for _ in range(64):
        ax = pyplot.subplot(8, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_map[0, :, :, ix - 1], cmap="viridis")
        ix += 1
    plt.show()

"""
In the above figure, we only show the first 64 feature maps of blocks in different stages.
We can see that the local information is
well preserved in the shallow layers, and the representations become more abstract
gradually as the network goes deeper.
"""

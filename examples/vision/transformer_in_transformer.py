"""
Title: Image classification with TNT(Transformer in Transformer)
Author: [ZhiYong Chang](https://github.com/czy00000)
Date created: 2021/10/25
Last modified: 2021/11/06
Description: Implementing the Transformer in Transformer (TNT) model for image classification.
"""

"""
## Introduction
This example implements the [TNT](https://arxiv.org/abs/2103.00112)
model for image classification, and demonstrates it on the CIFAR-100 dataset.
TNT is a novel model for modeling both patch-level and pixel-level
representation. In each TNT block, an ***outer*** transformer block is utilized to
process
patch embeddings, and an ***inner***
transformer block extracts local features from pixel embeddings. The pixel-level
feature is projected to the space of patch embedding by a linear transformation layer
and then added into the patch.
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

import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from itertools import repeat


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

weight_decay = 0.0001
learning_rate = 0.001
label_smoothing = 0.1
validation_split = 0.2
batch_size = 128
image_size = 96  # resize images to this size
num_epochs = 50

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
## Implement the pixel embedding layer
"""


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def pixel_embed(x, image_size=96, patch_size=8, in_dim=48, stride=4):
    _, channel, height, width = x.shape
    image_size = tuple(repeat(image_size, 2))
    patch_size = tuple(repeat(patch_size, 2))
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    new_patch_size = [math.ceil(ps / stride) for ps in patch_size]
    x = layers.Conv2D(in_dim, kernel_size=7, strides=stride, padding="same")(x)
    # pixel extraction
    x = tf.image.extract_patches(
        images=x,
        sizes=(1, new_patch_size[0], new_patch_size[1], 1),
        strides=(1, new_patch_size[0], new_patch_size[1], 1),
        rates=(1, 1, 1, 1),
        padding="VALID",
    )
    x = tf.reshape(x, shape=(-1, new_patch_size[0] * new_patch_size[1], in_dim))
    x = PatchEncoder(new_patch_size[0] * new_patch_size[1], in_dim)(x)
    return x, num_patches, new_patch_size


"""
## Implement the MLP block
"""


def mlp(x, mlp_dim, embedding_dim, drop_rate=0.2):
    x = layers.Dense(mlp_dim, activation=tf.nn.gelu)(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Dropout(drop_rate)(x)
    return x


"""
## Implement the TNT block
"""


def transformer_in_transformer_block(
    pixel_embedding,
    patch_embedding,
    out_embedding_dim,
    in_embedding_dim,
    num_pixel,
    out_num_heads=8,
    in_num_heads=4,
    mlp_ratio=4,
    attention_dropout=0,
    projection_dropout=0,
):
    # inner transformer block
    residual_in_1 = pixel_embedding
    pixel_embedding = layers.LayerNormalization(epsilon=1e-5)(pixel_embedding)
    pixel_embedding = layers.MultiHeadAttention(
        num_heads=in_num_heads, key_dim=in_embedding_dim, dropout=attention_dropout
    )(pixel_embedding, pixel_embedding)
    pixel_embedding = layers.add([pixel_embedding, residual_in_1])
    residual_in_2 = pixel_embedding
    pixel_embedding = layers.LayerNormalization(epsilon=1e-5)(pixel_embedding)
    pixel_embedding = mlp(
        pixel_embedding, in_embedding_dim * mlp_ratio, in_embedding_dim
    )
    pixel_embedding = layers.add([pixel_embedding, residual_in_2])

    # outer transformer block
    _, num_patches, channel = patch_embedding.shape
    # fusion local and global information
    fusion_embedding = tf.reshape(
        pixel_embedding, shape=(-1, num_patches, in_embedding_dim * num_pixel)
    )
    fusion_embedding = layers.LayerNormalization(epsilon=1e-5)(fusion_embedding)
    fusion_embedding = layers.Dense(out_embedding_dim)(fusion_embedding)
    patch_embedding = layers.add([patch_embedding, fusion_embedding])
    residual_out_1 = patch_embedding
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    patch_embedding = layers.MultiHeadAttention(
        num_heads=out_num_heads, key_dim=out_embedding_dim, dropout=attention_dropout
    )(patch_embedding, patch_embedding)
    patch_embedding = layers.add([patch_embedding, residual_out_1])
    residual_out_2 = patch_embedding
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    patch_embedding = mlp(
        patch_embedding, out_embedding_dim * mlp_ratio, out_embedding_dim
    )
    patch_embedding = layers.add([patch_embedding, residual_out_2])
    return pixel_embedding, patch_embedding


"""
## Implement the TNT model
The TNT model consists of multiple TNT blocks.
In the TNT block, there are two transformer blocks where
the outer transformer block models the global relation among patch embeddings,
and the inner one extracts local structure information of pixel embeddings.
The local information is added on the patch
embedding by linearly projecting the pixel embeddings into the space of patch embedding.
Patch-level and pixel-level position embeddings are introduced in order to
retain spatial information.In orginal paper, the authors use the class token for
classification.
We use the `layers.GlobalAvgPool1D` to fusion patch information.
"""


def get_model(
    image_size=96,
    patch_size=8,
    outer_block_embedding_dim=64,  # outer transformer block embedding dim
    inner_block_embedding_dim=16,  # inner transformer block embedding dim
    num_transformer_layer=5,
    outer_block_num_heads=4,
    inner_block_num_heads=2,
    mlp_ratio=4,
    attention_dropout=0.2,
    projection_dropout=0.2,
    first_stride=4,
):
    inputs = layers.Input(shape=input_shape)
    # Image augment
    x = data_augmentation(inputs)
    # extract pixel embedding
    pixel_embedding, num_patches, new_patch_size = pixel_embed(
        x, image_size, patch_size, inner_block_embedding_dim, first_stride
    )
    num_pixel = new_patch_size[0] * new_patch_size[1]
    # extract patch embedding
    patch_embedding = tf.reshape(
        pixel_embedding, shape=(-1, num_patches, inner_block_embedding_dim * num_pixel)
    )
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    patch_embedding = layers.Dense(outer_block_embedding_dim)(patch_embedding)
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    patch_embedding = PatchEncoder(num_patches, outer_block_embedding_dim)(
        patch_embedding
    )
    patch_embedding = layers.Dropout(projection_dropout)(patch_embedding)
    # create multiple layers of the TNT block.
    for _ in range(num_transformer_layer):
        pixel_embedding, patch_embedding = transformer_in_transformer_block(
            pixel_embedding,
            patch_embedding,
            outer_block_embedding_dim,
            inner_block_embedding_dim,
            num_pixel,
            outer_block_num_heads,
            inner_block_num_heads,
            mlp_ratio,
            attention_dropout,
            projection_dropout,
        )
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    x = layers.GlobalAvgPool1D()(patch_embedding)
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
### Visualize the training progress of the model.
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
plt.title("Train and Validation Accuracies Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

"""
### Let's display the final results of the test on CIFAR-100.
"""

loss, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {round(loss, 2)}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

"""
After 50 epochs, the TNT model achieves around 48% accuracy and
78% top-5 accuracy on the test data. It only has 0.8M parameters.
From the above loss curve, we can find that the model gradually converges,
but it never achieves state of the art performance. We can apply some data augmentation
to
obtain better performance, like RandAugment, MixUp etc. We also can adjust the depth
of model, learning rate or increase the size of embedding.Compared to the conventional
vision transformers [(ViT)](https://arxiv.org/abs/2010.11929) which corrupts the local
structure
of the patch, the TNT can better preserve and model the local information
for visual recognition.
"""

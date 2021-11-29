"""
Title: Image classification with TNT(Transformer in Transformer)
Author: [ZhiYong Chang](https://github.com/czy00000)
Date created: 2021/10/25
Last modified: 2021/11/29
Description: Implementing the Transformer in Transformer (TNT) model for image classification.
"""


"""
## Introduction
This example implements the [TNT](https://arxiv.org/pdf/2103.00112v2.pdf)
model for image classification, and demonstrates it's performance on the CIFAR-100
dataset.
To keep training time reasonable, We will train and test a smaller model than is in the
paper(0.66M params vs 23.8M params).
TNT is a novel model for modeling both patch-level and pixel-level
representation. In each TNT block, an ***outer*** transformer block is utilized to process
patch embeddings, and an ***inner***
transformer block extracts local features from pixel embeddings. The pixel-level
feature is projected to the space of patch embedding by a linear transformation layer
and then added into the patch.
This example requires TensorFlow 2.5 or higher, as well as
[TensorFlow Addons](https://www.tensorflow.org/addons/overview) package, it is for the
AdamW optimizer,
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

weight_decay = 0.0002
learning_rate = 0.001
label_smoothing = 0.1
validation_split = 0.2
batch_size = 128
image_size = (96, 96)  # resize images to this size
patch_size = (8, 8)
num_epochs = 50
outer_block_embedding_dim = 64
inner_block_embedding_dim = 32
num_transformer_layer = 5
outer_block_num_heads = 4
inner_block_num_heads = 2
mlp_ratio = 4
attention_dropout = 0.5
projection_dropout = 0.5
first_stride = 4

"""
## Use data augmentation
"""


def data_augmentation(inputs):
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = layers.Resizing(image_size[0], image_size[1])(x)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(factor=0.1)(x)
    x = layers.RandomContrast(factor=0.1)(x)
    x = layers.RandomZoom(height_factor=0.2, width_factor=0.2)(x)
    return x


"""
## Implement the pixel embedding and patch embedding layer
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
        positions = tf.range(start=0, limit=self.num_patches)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def pixel_embed(x, image_size=image_size, patch_size=patch_size, in_dim=48, stride=4):
    _, channel, height, width = x.shape
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    inner_patch_size = [math.ceil(ps / stride) for ps in patch_size]
    x = layers.Conv2D(in_dim, kernel_size=7, strides=stride, padding="same")(x)
    # pixel extraction
    x = tf.image.extract_patches(
        images=x,
        sizes=(1, inner_patch_size[0], inner_patch_size[1], 1),
        strides=(1, inner_patch_size[0], inner_patch_size[1], 1),
        rates=(1, 1, 1, 1),
        padding="VALID",
    )
    x = tf.reshape(x, shape=(-1, inner_patch_size[0] * inner_patch_size[1], in_dim))
    x = PatchEncoder(inner_patch_size[0] * inner_patch_size[1], in_dim)(x)
    return x, num_patches, inner_patch_size


def patch_embed(
    pixel_embedding,
    num_patches,
    outer_block_embedding_dim,
    inner_block_embedding_dim,
    num_pixels,
):
    patch_embedding = tf.reshape(
        pixel_embedding, shape=(-1, num_patches, inner_block_embedding_dim * num_pixels)
    )
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    patch_embedding = layers.Dense(outer_block_embedding_dim)(patch_embedding)
    patch_embedding = layers.LayerNormalization(epsilon=1e-5)(patch_embedding)
    patch_embedding = PatchEncoder(num_patches, outer_block_embedding_dim)(
        patch_embedding
    )
    patch_embedding = layers.Dropout(projection_dropout)(patch_embedding)
    return patch_embedding


"""
## Implement the MLP block
"""


def mlp(x, hidden_dim, output_dim, drop_rate=0.2):
    x = layers.Dense(hidden_dim, activation=tf.nn.gelu)(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(output_dim)(x)
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
    num_pixels,
    out_num_heads,
    in_num_heads,
    mlp_ratio,
    attention_dropout,
    projection_dropout,
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
    # fuse local and global information
    fused_embedding = tf.reshape(
        pixel_embedding, shape=(-1, num_patches, in_embedding_dim * num_pixels)
    )
    fused_embedding = layers.LayerNormalization(epsilon=1e-5)(fused_embedding)
    fused_embedding = layers.Dense(out_embedding_dim)(fused_embedding)
    patch_embedding = layers.add([patch_embedding, fused_embedding])
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
retain spatial information. In orginal paper, the authors use the class token for
classification.
We use the `layers.GlobalAvgPool1D` to fuse patch information.
"""


def get_model(
    image_size=image_size,
    patch_size=patch_size,
    outer_block_embedding_dim=outer_block_embedding_dim,
    inner_block_embedding_dim=inner_block_embedding_dim,
    num_transformer_layer=num_transformer_layer,
    outer_block_num_heads=outer_block_num_heads,
    inner_block_num_heads=inner_block_num_heads,
    mlp_ratio=mlp_ratio,
    attention_dropout=attention_dropout,
    projection_dropout=projection_dropout,
    first_stride=first_stride,
):
    inputs = layers.Input(shape=input_shape)
    # Image augment
    x = data_augmentation(inputs)
    # extract pixel embedding
    pixel_embedding, num_patches, inner_patch_size = pixel_embed(
        x, image_size, patch_size, inner_block_embedding_dim, first_stride
    )
    num_pixels = inner_patch_size[0] * inner_patch_size[1]
    # extract patch embedding
    patch_embedding = patch_embed(
        pixel_embedding,
        num_patches,
        outer_block_embedding_dim,
        inner_block_embedding_dim,
        num_pixels,
    )
    # create multiple layers of the TNT block.
    for _ in range(num_transformer_layer):
        pixel_embedding, patch_embedding = transformer_in_transformer_block(
            pixel_embedding,
            patch_embedding,
            outer_block_embedding_dim,
            inner_block_embedding_dim,
            num_pixels,
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
model.summary()
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
After 50 epochs, the TNT model achieves around 42% accuracy and
73% top-5 accuracy on the test data. It only has 0.6M parameters.
From the above loss curve, we can find that the model gradually converges,
but it never achieves state of the art performance. We could apply further data
augmentation to
obtain better performance, like [RandAugment](https://arxiv.org/abs/1909.13719),
[MixUp](https://arxiv.org/abs/1710.09412)
etc. We also can adjust the depth of model, learning rate or increase the size of
embedding. Compared to the conventional
vision transformers [(ViT)](https://arxiv.org/abs/2010.11929) which corrupts the local
structure
of the patch, the TNT can better preserve and model the local information
for visual recognition.
"""

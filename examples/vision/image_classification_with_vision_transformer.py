"""
Title: Image Classification with Vision Transformer
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/01/30
Last modified: 2021/01/30
Description: Implementing Vision Transformer (ViT) model for image classification.
"""

"""
## Introduction

This example implements the [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
model, by Alexey Dosovitskiy et al., for CIFAR-100 image classification.
The ViT applies the transformer architecture with self-attentions to sequences of
image patches, without using convolutional networks.

The example requires TensorFlow 2.4 or higher, and
[TensorFlow Addons](https://www.tensorflow.org/addons/overview),
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

"""
## Prepare the data
"""

num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

"""
## Define hyperparameters
"""

learning_rate = 0.003
weight_decay = 0.001
batch_size = 512
hidden_units = [128]
num_epochs = 100
dropout_rate = 0.5

"""
## Compile, train, and evaluate the mode
"""


def run_experiment(model):
    k = 5
    model.compile(
        optimizer=tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseTopKCategoricalAccuracy(k)],
    )

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)

    accuracy = model.evaluate(x_test, y_test)[1]
    print(f"Test tok {k} accuracy: {round(accuracy * 100, 2)}%")

    return history


"""
## Using data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.02),
    ],
    name="data_augmentation",
)

"""
## Implement Multilayer perceptron (MLP) as a layer
"""


class MLP(layers.Layer):
    def __init__(self, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        mlp_layers = []
        for units in hidden_units:
            mlp_layers.append(layers.Dense(units, activation=tf.nn.gelu))
            mlp_layers.append(layers.Dropout(dropout_rate))
        self.mlp = keras.Sequential(mlp_layers)

    def call(self, inputs):
        return self.mlp(inputs)


"""
## Experiment 1: Train the baseline classification model

We use an untrained ResNet50 architecture as our baseline model.
"""


def create_resnet_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Generate features using ResNet.
    representation = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )(augmented)
    representation = layers.Dropout(dropout_rate)(features)
    # Create MLP.
    features = MLP(hidden_units, dropout_rate)(representation)
    # Create softmax output.
    outputs = layers.Dense(num_classes, activation="softmax")(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


resnet_classifier = create_resnet_classifier()
resnet_classifier.summary()

history = run_experiment(resnet_classifier)

import matplotlib.pyplot as plt

plt.plot(history.history["sparse_top_k_categorical_accuracy"])
plt.plot(history.history["val_sparse_top_k_categorical_accuracy"])
plt.legend(["Train", "Eval"], loc="upper left")

"""
After 100 epochs, the RestNet50 classification model achieves around 66% top 5
accuracy on the test data.
"""

"""
## Experiment 2: Train Vision Transformer model
"""

patch_size = 4
image_size = input_shape[0]
num_patches = (image_size // patch_size) ** 2
projection_dims = 64
num_heads = 4
transformer_hidden_units = [projection_dims * 2, projection_dims]
transfomer_layers = 4
dropout_rate = 0.1

"""
### Implement patch creation as a layer
"""


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


"""
Let's display patches for a sample image
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

patches = Patches(patch_size)([image])
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")

"""
### Implement the patch encoding layer

The `PatchEncoder` will linearly transform the patch by projecting it into a
vector of size `projection_dims`. In addition, it adds a learnable position
embedding to the projected vector.
"""


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dims):
        super(PatchEncoder, self).__init__()
        self.projection = layers.Dense(units=projection_dims)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dims
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


"""
### Build the ViT model

The ViT model consists of multiple layers of the Transformer block,
which uses the `layers.MultiHeadAttention` layer as a self-attention mechanism
for the sequence of patches. The Transformer blocks produces a
`[batch_size, num_patches, projection_dim]` tensor, which is processed via an
MLP head with softmax to produce the final class probabilities output.

Unlike the technique described in the [paper](https://arxiv.org/abs/2010.11929),
which prepends a learnable embedding to the sequence of encoded patches to serve
as the image representation, the outputs of the final Transformer block are
aggregated, using `layers.GlobalAveragePooling1D()`, and used as the image
representation input to the MLP head.
"""


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dims)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transfomer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-headed attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dims, dropout=dropout_rate
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = MLP(transformer_hidden_units, dropout_rate)(x3)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dims] tensor.
    representation = layers.GlobalAveragePooling1D()(encoded_patches)
    representation = layers.LayerNormalization(epsilon=1e-6)(representation)
    # Create MLP.
    features = MLP(hidden_units, dropout_rate)(representation)
    # Create softmax output.
    outputs = layers.Dense(num_classes, activation="softmax")(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


vit_classifier = create_vit_classifier()
keras.utils.plot_model(vit_classifier, show_shapes=True)

history = run_experiment(vit_classifier)

"""
After 100 epochs, the ViT classification model achieves more than 72% top 5
accuracy on the test data. You can try to train the model for more epochs,
use larger number of Transformer layer, or increase the projection dimensions
to achieve better results.
"""

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
model by Alexey Dosovitskiy et al. for image classification on the CIFAR-100 dataset.
The ViT applies the Transformer architecture with self-attentions on sequences of
image patches without using convolutional networks.

The example requires TensorFlow 2.4 or higher and
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
## Compile, train, and evaluate the mode
"""

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100


def run_experiment(model):
    model.compile(
        optimizer=tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseTopKCategoricalAccuracy(5)],
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.15,
    )

    _, accuracy, top_k_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_k_accuracy * 100, 2)}%")

    return history


"""
## Using data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.02),
        layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
    ],
    name="data_augmentation",
)

data_augmentation.layers[0].adapt(x_train)

"""
## Experiment 1: Train the baseline classification model

We use an untrained ResNet-50 architecture as our baseline model.
"""


def create_resnet_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Generate features using ResNet.
    representation = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )(augmented)
    representation = layers.Dropout(0.5)(representation)
    # Create softmax output.
    outputs = layers.Dense(num_classes, activation="softmax")(representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


resnet_classifier = create_resnet_classifier()
history = run_experiment(resnet_classifier)

"""
After 100 epochs the RestNet-50 classification model achieves around 48% accuracy
and 74% top 5 accuracy, and  on the test data.
"""

"""
## Experiment 2: Train the Vision Transformer model
"""

patch_size = 4
image_size = input_shape[0]
num_patches = (image_size // patch_size) ** 2
projection_dims = 64
num_heads = 4
transformer_units = [projection_dims * 2, projection_dims]
transfomer_layers = 8
mlp_head_units = [512, 128]
dropout_rate = 0.1

"""
### Implement multilayer perceptron (MLP)
"""


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


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
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dims)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dims
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
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
aggregated with `layers.Flatten()` and then used as the image
representation input to the MLP head. The `layers.GlobalAveragePooling1D`
could be used instead.
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
        # Create a mult-headead attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dims, dropout=dropout_rate
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, transformer_units, dropout_rate)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dims] tensor.
    representation = layers.Flatten()(encoded_patches)
    representation = layers.Dropout(0.5)(representation)

    # Create MLP.
    features = mlp(representation, mlp_head_units, dropout_rate)
    # Create softmax output.
    outputs = layers.Dense(num_classes, activation="softmax")(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)

"""
After 100 epochs, the ViT classification model achieves around 52% accuracy and 
80% top 5 accuracy on the test data. You can try to train the model 
for more epochs, use larger number of Transformer layers, or increase 
the projection dimensions to achieve better results. Also note that, as mentioned in 
the [paper](https://arxiv.org/abs/2010.11929), the quality of the model is affected
not only by the architecture choice, but also other parameters, such as training 
schedule, optimizer, weight decay, etc.
"""

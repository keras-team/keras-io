# Image classification with modern MLP models

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/05/30<br>
**Last modified:** 2021/05/30<br>
**Description:** Implementing the MLP-Mixer, FNet, and gMLP models for CIFAR-100 image classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/mlp_image_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/mlp_image_classification.py)



---
## Introduction

This example implements three modern attention-free, multi-layer perceptron (MLP) based models for image
classification, demonstrated on the CIFAR-100 dataset:

1. The [MLP-Mixer](https://arxiv.org/abs/2105.01601) model, by Ilya Tolstikhin et al., based on two types of MLPs.
3. The [FNet](https://arxiv.org/abs/2105.03824) model, by James Lee-Thorp et al., based on unparameterized
Fourier Transform.
2. The [gMLP](https://arxiv.org/abs/2105.08050) model, by Hanxiao Liu et al., based on MLP with gating.

The purpose of the example is not to compare between these models, as they might perform differently on
different datasets with well-tuned hyperparameters. Rather, it is to show simple implementations of their
main building blocks.

This example requires TensorFlow 2.4 or higher, as well as
[TensorFlow Addons](https://www.tensorflow.org/addons/overview),
which can be installed using the following command:

```
pip install -U tensorflow-addons
```

---
## Setup


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
```

---
## Prepare the data


```python
num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
```

<div class="k-default-codeblock">
```
x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 1)
x_test shape: (10000, 32, 32, 3) - y_test shape: (10000, 1)

```
</div>
---
## Configure the hyperparameters


```python
weight_decay = 0.0001
batch_size = 128
num_epochs = 50
dropout_rate = 0.2
image_size = 64  # We'll resize input images to this size.
patch_size = 8  # Size of the patches to be extracted from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
embedding_dim = 256  # Number of hidden units.
num_blocks = 4  # Number of blocks.

print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")
print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")
```

<div class="k-default-codeblock">
```
Image size: 64 X 64 = 4096
Patch size: 8 X 8 = 64 
Patches per image: 64
Elements per patch (3 channels): 192

```
</div>
---
## Build a classification model

We implement a method that builds a classifier given the processing blocks.


```python

def build_classifier(blocks, positional_encoding=False):
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size, num_patches)(augmented)
    # Encode patches to generate a [batch_size, num_patches, embedding_dim] tensor.
    x = layers.Dense(units=embedding_dim)(patches)
    if positional_encoding:
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embedding_dim
        )(positions)
        x = x + position_embedding
    # Process x using the module blocks.
    x = blocks(x)
    # Apply global average pooling to generate a [batch_size, embedding_dim] representation tensor.
    representation = layers.GlobalAveragePooling1D()(x)
    # Apply dropout.
    representation = layers.Dropout(rate=dropout_rate)(representation)
    # Compute logits outputs.
    logits = layers.Dense(num_classes)(representation)
    # Create the Keras model.
    return keras.Model(inputs=inputs, outputs=logits)

```

---
## Define an experiment

We implement a utility function to compile, train, and evaluate a given model.


```python

def run_experiment(model):
    # Create Adam optimizer with weight decay.
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay,
    )
    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )
    # Create a learning rate scheduler callback.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    )
    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Return history to plot learning curves.
    return history

```

---
## Use data augmentation


```python
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

```

---
## Implement patch extraction as a layer


```python

class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

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
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches

```

---
## The MLP-Mixer model

The MLP-Mixer is an architecture based exclusively on
multi-layer perceptrons (MLPs), that contains two types of MLP layers:

1. One applied independently to image patches, which mixes the per-location features.
2. The other applied across patches (along channels), which mixes spatial information.

This is similar to a [depthwise separable convolution based model](https://arxiv.org/pdf/1610.02357.pdf)
such as the Xception model, but with two chained dense transforms, no max pooling, and layer normalization
instead of batch normalization.

### Implement the MLP-Mixer module


```python

class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches),
                tfa.layers.GELU(),
                layers.Dense(units=embedding_dim),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x

```

### Build, train, and evaluate the MLP-Mixer model

Note that training the model with the current settings on a V100 GPUs
takes around 8 seconds per epoch.


```python
mlpmixer_blocks = keras.Sequential(
    [MLPMixerLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
)
learning_rate = 0.005
mlpmixer_classifier = build_classifier(mlpmixer_blocks)
history = run_experiment(mlpmixer_classifier)
```

<div class="k-default-codeblock">
```
/opt/conda/lib/python3.7/site-packages/tensorflow/python/autograph/impl/api.py:390: UserWarning: Default value of `approximate` is changed from `True` to `False`
  return py_builtins.overload_of(f)(*args)

Epoch 1/50
352/352 [==============================] - 13s 25ms/step - loss: 4.1703 - acc: 0.0756 - top5-acc: 0.2322 - val_loss: 3.6202 - val_acc: 0.1532 - val_top5-acc: 0.4140
Epoch 2/50
352/352 [==============================] - 8s 23ms/step - loss: 3.4165 - acc: 0.1789 - top5-acc: 0.4459 - val_loss: 3.1599 - val_acc: 0.2334 - val_top5-acc: 0.5160
Epoch 3/50
352/352 [==============================] - 8s 23ms/step - loss: 3.1367 - acc: 0.2328 - top5-acc: 0.5230 - val_loss: 3.0539 - val_acc: 0.2560 - val_top5-acc: 0.5664
Epoch 4/50
352/352 [==============================] - 8s 23ms/step - loss: 2.9985 - acc: 0.2624 - top5-acc: 0.5600 - val_loss: 2.9498 - val_acc: 0.2798 - val_top5-acc: 0.5856
Epoch 5/50
352/352 [==============================] - 8s 23ms/step - loss: 2.8806 - acc: 0.2809 - top5-acc: 0.5879 - val_loss: 2.8593 - val_acc: 0.2904 - val_top5-acc: 0.6050
Epoch 6/50
352/352 [==============================] - 8s 23ms/step - loss: 2.7860 - acc: 0.3024 - top5-acc: 0.6124 - val_loss: 2.7405 - val_acc: 0.3256 - val_top5-acc: 0.6364
Epoch 7/50
352/352 [==============================] - 8s 23ms/step - loss: 2.7065 - acc: 0.3152 - top5-acc: 0.6280 - val_loss: 2.7548 - val_acc: 0.3328 - val_top5-acc: 0.6450
Epoch 8/50
352/352 [==============================] - 8s 22ms/step - loss: 2.6443 - acc: 0.3263 - top5-acc: 0.6446 - val_loss: 2.6618 - val_acc: 0.3460 - val_top5-acc: 0.6578
Epoch 9/50
352/352 [==============================] - 8s 23ms/step - loss: 2.5886 - acc: 0.3406 - top5-acc: 0.6573 - val_loss: 2.6065 - val_acc: 0.3492 - val_top5-acc: 0.6650
Epoch 10/50
352/352 [==============================] - 8s 23ms/step - loss: 2.5798 - acc: 0.3404 - top5-acc: 0.6591 - val_loss: 2.6546 - val_acc: 0.3502 - val_top5-acc: 0.6630
Epoch 11/50
352/352 [==============================] - 8s 23ms/step - loss: 2.5269 - acc: 0.3498 - top5-acc: 0.6714 - val_loss: 2.6201 - val_acc: 0.3570 - val_top5-acc: 0.6710
Epoch 12/50
352/352 [==============================] - 8s 23ms/step - loss: 2.5003 - acc: 0.3569 - top5-acc: 0.6745 - val_loss: 2.5936 - val_acc: 0.3564 - val_top5-acc: 0.6662
Epoch 13/50
352/352 [==============================] - 8s 22ms/step - loss: 2.4801 - acc: 0.3619 - top5-acc: 0.6792 - val_loss: 2.5236 - val_acc: 0.3700 - val_top5-acc: 0.6786
Epoch 14/50
352/352 [==============================] - 8s 23ms/step - loss: 2.4392 - acc: 0.3676 - top5-acc: 0.6879 - val_loss: 2.4971 - val_acc: 0.3808 - val_top5-acc: 0.6926
Epoch 15/50
352/352 [==============================] - 8s 23ms/step - loss: 2.4073 - acc: 0.3790 - top5-acc: 0.6940 - val_loss: 2.5972 - val_acc: 0.3682 - val_top5-acc: 0.6750
Epoch 16/50
352/352 [==============================] - 8s 23ms/step - loss: 2.3922 - acc: 0.3754 - top5-acc: 0.6980 - val_loss: 2.4317 - val_acc: 0.3964 - val_top5-acc: 0.6992
Epoch 17/50
352/352 [==============================] - 8s 22ms/step - loss: 2.3603 - acc: 0.3891 - top5-acc: 0.7038 - val_loss: 2.4844 - val_acc: 0.3766 - val_top5-acc: 0.6964
Epoch 18/50
352/352 [==============================] - 8s 23ms/step - loss: 2.3560 - acc: 0.3849 - top5-acc: 0.7056 - val_loss: 2.4564 - val_acc: 0.3910 - val_top5-acc: 0.6990
Epoch 19/50
352/352 [==============================] - 8s 23ms/step - loss: 2.3367 - acc: 0.3900 - top5-acc: 0.7069 - val_loss: 2.4282 - val_acc: 0.3906 - val_top5-acc: 0.7058
Epoch 20/50
352/352 [==============================] - 8s 22ms/step - loss: 2.3096 - acc: 0.3945 - top5-acc: 0.7180 - val_loss: 2.4297 - val_acc: 0.3930 - val_top5-acc: 0.7082
Epoch 21/50
352/352 [==============================] - 8s 22ms/step - loss: 2.2935 - acc: 0.3996 - top5-acc: 0.7211 - val_loss: 2.4053 - val_acc: 0.3974 - val_top5-acc: 0.7076
Epoch 22/50
352/352 [==============================] - 8s 22ms/step - loss: 2.2823 - acc: 0.3991 - top5-acc: 0.7248 - val_loss: 2.4756 - val_acc: 0.3920 - val_top5-acc: 0.6988
Epoch 23/50
352/352 [==============================] - 8s 22ms/step - loss: 2.2371 - acc: 0.4126 - top5-acc: 0.7294 - val_loss: 2.3802 - val_acc: 0.3972 - val_top5-acc: 0.7100
Epoch 24/50
352/352 [==============================] - 8s 23ms/step - loss: 2.2234 - acc: 0.4140 - top5-acc: 0.7336 - val_loss: 2.4402 - val_acc: 0.3994 - val_top5-acc: 0.7096
Epoch 25/50
352/352 [==============================] - 8s 23ms/step - loss: 2.2320 - acc: 0.4088 - top5-acc: 0.7333 - val_loss: 2.4343 - val_acc: 0.3936 - val_top5-acc: 0.7052
Epoch 26/50
352/352 [==============================] - 8s 22ms/step - loss: 2.2094 - acc: 0.4193 - top5-acc: 0.7347 - val_loss: 2.4154 - val_acc: 0.4058 - val_top5-acc: 0.7192
Epoch 27/50
352/352 [==============================] - 8s 23ms/step - loss: 2.2029 - acc: 0.4180 - top5-acc: 0.7370 - val_loss: 2.3116 - val_acc: 0.4226 - val_top5-acc: 0.7268
Epoch 28/50
352/352 [==============================] - 8s 23ms/step - loss: 2.1959 - acc: 0.4234 - top5-acc: 0.7380 - val_loss: 2.4053 - val_acc: 0.4064 - val_top5-acc: 0.7168
Epoch 29/50
352/352 [==============================] - 8s 23ms/step - loss: 2.1815 - acc: 0.4227 - top5-acc: 0.7415 - val_loss: 2.4020 - val_acc: 0.4078 - val_top5-acc: 0.7192
Epoch 30/50
352/352 [==============================] - 8s 23ms/step - loss: 2.1783 - acc: 0.4245 - top5-acc: 0.7407 - val_loss: 2.4206 - val_acc: 0.3996 - val_top5-acc: 0.7234
Epoch 31/50
352/352 [==============================] - 8s 22ms/step - loss: 2.1686 - acc: 0.4248 - top5-acc: 0.7442 - val_loss: 2.3743 - val_acc: 0.4100 - val_top5-acc: 0.7162
Epoch 32/50
352/352 [==============================] - 8s 23ms/step - loss: 2.1487 - acc: 0.4317 - top5-acc: 0.7472 - val_loss: 2.3882 - val_acc: 0.4018 - val_top5-acc: 0.7266
Epoch 33/50
352/352 [==============================] - 8s 22ms/step - loss: 1.9836 - acc: 0.4644 - top5-acc: 0.7782 - val_loss: 2.1742 - val_acc: 0.4536 - val_top5-acc: 0.7506
Epoch 34/50
352/352 [==============================] - 8s 23ms/step - loss: 1.8723 - acc: 0.4950 - top5-acc: 0.7985 - val_loss: 2.1716 - val_acc: 0.4506 - val_top5-acc: 0.7546
Epoch 35/50
352/352 [==============================] - 8s 23ms/step - loss: 1.8461 - acc: 0.5009 - top5-acc: 0.8003 - val_loss: 2.1661 - val_acc: 0.4480 - val_top5-acc: 0.7542
Epoch 36/50
352/352 [==============================] - 8s 23ms/step - loss: 1.8499 - acc: 0.4944 - top5-acc: 0.8044 - val_loss: 2.1523 - val_acc: 0.4566 - val_top5-acc: 0.7628
Epoch 37/50
352/352 [==============================] - 8s 22ms/step - loss: 1.8322 - acc: 0.5000 - top5-acc: 0.8059 - val_loss: 2.1334 - val_acc: 0.4570 - val_top5-acc: 0.7560
Epoch 38/50
352/352 [==============================] - 8s 23ms/step - loss: 1.8269 - acc: 0.5027 - top5-acc: 0.8085 - val_loss: 2.1024 - val_acc: 0.4614 - val_top5-acc: 0.7674
Epoch 39/50
352/352 [==============================] - 8s 23ms/step - loss: 1.8242 - acc: 0.4990 - top5-acc: 0.8098 - val_loss: 2.0789 - val_acc: 0.4610 - val_top5-acc: 0.7792
Epoch 40/50
352/352 [==============================] - 8s 23ms/step - loss: 1.7983 - acc: 0.5067 - top5-acc: 0.8122 - val_loss: 2.1514 - val_acc: 0.4546 - val_top5-acc: 0.7628
Epoch 41/50
352/352 [==============================] - 8s 23ms/step - loss: 1.7974 - acc: 0.5112 - top5-acc: 0.8132 - val_loss: 2.1425 - val_acc: 0.4542 - val_top5-acc: 0.7630
Epoch 42/50
352/352 [==============================] - 8s 23ms/step - loss: 1.7972 - acc: 0.5128 - top5-acc: 0.8127 - val_loss: 2.0980 - val_acc: 0.4580 - val_top5-acc: 0.7724
Epoch 43/50
352/352 [==============================] - 8s 23ms/step - loss: 1.8026 - acc: 0.5066 - top5-acc: 0.8115 - val_loss: 2.0922 - val_acc: 0.4684 - val_top5-acc: 0.7678
Epoch 44/50
352/352 [==============================] - 8s 23ms/step - loss: 1.7924 - acc: 0.5092 - top5-acc: 0.8129 - val_loss: 2.0511 - val_acc: 0.4750 - val_top5-acc: 0.7726
Epoch 45/50
352/352 [==============================] - 8s 22ms/step - loss: 1.7695 - acc: 0.5106 - top5-acc: 0.8193 - val_loss: 2.0949 - val_acc: 0.4678 - val_top5-acc: 0.7708
Epoch 46/50
352/352 [==============================] - 8s 23ms/step - loss: 1.7784 - acc: 0.5106 - top5-acc: 0.8141 - val_loss: 2.1094 - val_acc: 0.4656 - val_top5-acc: 0.7704
Epoch 47/50
352/352 [==============================] - 8s 23ms/step - loss: 1.7625 - acc: 0.5155 - top5-acc: 0.8190 - val_loss: 2.0492 - val_acc: 0.4774 - val_top5-acc: 0.7744
Epoch 48/50
352/352 [==============================] - 8s 23ms/step - loss: 1.7441 - acc: 0.5217 - top5-acc: 0.8190 - val_loss: 2.0562 - val_acc: 0.4698 - val_top5-acc: 0.7828
Epoch 49/50
352/352 [==============================] - 8s 23ms/step - loss: 1.7665 - acc: 0.5113 - top5-acc: 0.8196 - val_loss: 2.0348 - val_acc: 0.4708 - val_top5-acc: 0.7730
Epoch 50/50
352/352 [==============================] - 8s 23ms/step - loss: 1.7392 - acc: 0.5201 - top5-acc: 0.8226 - val_loss: 2.0787 - val_acc: 0.4710 - val_top5-acc: 0.7734
313/313 [==============================] - 2s 8ms/step - loss: 2.0571 - acc: 0.4758 - top5-acc: 0.7718
Test accuracy: 47.58%
Test top 5 accuracy: 77.18%

```
</div>
The MLP-Mixer model tends to have much less number of parameters compared
to convolutional and transformer-based models, which leads to less training and
serving computational cost.

As mentioned in the [MLP-Mixer](https://arxiv.org/abs/2105.01601) paper,
when pre-trained on large datasets, or with modern regularization schemes,
the MLP-Mixer attains competitive scores to state-of-the-art models.
You can obtain better results by increasing the embedding dimensions,
increasing, increasing the number of mixer blocks, and training the model for longer.
You may also try to increase the size of the input images and use different patch sizes.

---
## The FNet model

The FNet uses a similar block to the Transformer block. However, FNet replaces the self-attention layer
in the Transformer block with a parameter-free 2D Fourier transformation layer:

1. One 1D Fourier Transform is applied along the patches.
2. One 1D Fourier Transform is applied along the channels.

### Implement the FNet module


```python

class FNetLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super(FNetLayer, self).__init__(*args, **kwargs)

        self.ffn = keras.Sequential(
            [
                layers.Dense(units=embedding_dim),
                tfa.layers.GELU(),
                layers.Dropout(rate=dropout_rate),
                layers.Dense(units=embedding_dim),
            ]
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply fourier transformations.
        x = tf.cast(
            tf.signal.fft2d(tf.cast(inputs, dtype=tf.dtypes.complex64)),
            dtype=tf.dtypes.float32,
        )
        # Add skip connection.
        x = x + inputs
        # Apply layer normalization.
        x = self.normalize1(x)
        # Apply Feedfowrad network.
        x_ffn = self.ffn(x)
        # Add skip connection.
        x = x + x_ffn
        # Apply layer normalization.
        return self.normalize2(x)

```

### Build, train, and evaluate the FNet model

Note that training the model with the current settings on a V100 GPUs
takes around 8 seconds per epoch.


```python
fnet_blocks = keras.Sequential(
    [FNetLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
)
learning_rate = 0.001
fnet_classifier = build_classifier(fnet_blocks, positional_encoding=True)
history = run_experiment(fnet_classifier)
```

<div class="k-default-codeblock">
```
Epoch 1/50
352/352 [==============================] - 11s 23ms/step - loss: 4.3419 - acc: 0.0470 - top5-acc: 0.1652 - val_loss: 3.8279 - val_acc: 0.1178 - val_top5-acc: 0.3268
Epoch 2/50
352/352 [==============================] - 8s 22ms/step - loss: 3.7814 - acc: 0.1202 - top5-acc: 0.3341 - val_loss: 3.5981 - val_acc: 0.1540 - val_top5-acc: 0.3914
Epoch 3/50
352/352 [==============================] - 8s 22ms/step - loss: 3.5319 - acc: 0.1603 - top5-acc: 0.4086 - val_loss: 3.3309 - val_acc: 0.1956 - val_top5-acc: 0.4656
Epoch 4/50
352/352 [==============================] - 8s 22ms/step - loss: 3.3025 - acc: 0.2001 - top5-acc: 0.4730 - val_loss: 3.1215 - val_acc: 0.2334 - val_top5-acc: 0.5234
Epoch 5/50
352/352 [==============================] - 8s 22ms/step - loss: 3.1621 - acc: 0.2224 - top5-acc: 0.5084 - val_loss: 3.0492 - val_acc: 0.2456 - val_top5-acc: 0.5322
Epoch 6/50
352/352 [==============================] - 8s 22ms/step - loss: 3.0506 - acc: 0.2469 - top5-acc: 0.5400 - val_loss: 2.9519 - val_acc: 0.2684 - val_top5-acc: 0.5652
Epoch 7/50
352/352 [==============================] - 8s 22ms/step - loss: 2.9520 - acc: 0.2618 - top5-acc: 0.5677 - val_loss: 2.8936 - val_acc: 0.2688 - val_top5-acc: 0.5864
Epoch 8/50
352/352 [==============================] - 8s 22ms/step - loss: 2.8377 - acc: 0.2828 - top5-acc: 0.5938 - val_loss: 2.7633 - val_acc: 0.2996 - val_top5-acc: 0.6068
Epoch 9/50
352/352 [==============================] - 8s 22ms/step - loss: 2.7670 - acc: 0.2969 - top5-acc: 0.6107 - val_loss: 2.7309 - val_acc: 0.3112 - val_top5-acc: 0.6136
Epoch 10/50
352/352 [==============================] - 8s 22ms/step - loss: 2.7027 - acc: 0.3148 - top5-acc: 0.6231 - val_loss: 2.6552 - val_acc: 0.3214 - val_top5-acc: 0.6436
Epoch 11/50
352/352 [==============================] - 8s 22ms/step - loss: 2.6375 - acc: 0.3256 - top5-acc: 0.6427 - val_loss: 2.6078 - val_acc: 0.3278 - val_top5-acc: 0.6434
Epoch 12/50
352/352 [==============================] - 8s 22ms/step - loss: 2.5573 - acc: 0.3424 - top5-acc: 0.6576 - val_loss: 2.5617 - val_acc: 0.3438 - val_top5-acc: 0.6534
Epoch 13/50
352/352 [==============================] - 8s 22ms/step - loss: 2.5259 - acc: 0.3488 - top5-acc: 0.6640 - val_loss: 2.5177 - val_acc: 0.3550 - val_top5-acc: 0.6652
Epoch 14/50
352/352 [==============================] - 8s 22ms/step - loss: 2.4782 - acc: 0.3586 - top5-acc: 0.6739 - val_loss: 2.5113 - val_acc: 0.3558 - val_top5-acc: 0.6718
Epoch 15/50
352/352 [==============================] - 8s 22ms/step - loss: 2.4242 - acc: 0.3712 - top5-acc: 0.6897 - val_loss: 2.4280 - val_acc: 0.3724 - val_top5-acc: 0.6880
Epoch 16/50
352/352 [==============================] - 8s 22ms/step - loss: 2.3884 - acc: 0.3741 - top5-acc: 0.6967 - val_loss: 2.4670 - val_acc: 0.3654 - val_top5-acc: 0.6794
Epoch 17/50
352/352 [==============================] - 8s 22ms/step - loss: 2.3619 - acc: 0.3797 - top5-acc: 0.7001 - val_loss: 2.3941 - val_acc: 0.3752 - val_top5-acc: 0.6922
Epoch 18/50
352/352 [==============================] - 8s 22ms/step - loss: 2.3183 - acc: 0.3931 - top5-acc: 0.7137 - val_loss: 2.4028 - val_acc: 0.3814 - val_top5-acc: 0.6954
Epoch 19/50
352/352 [==============================] - 8s 22ms/step - loss: 2.2919 - acc: 0.3955 - top5-acc: 0.7209 - val_loss: 2.3672 - val_acc: 0.3878 - val_top5-acc: 0.7022
Epoch 20/50
352/352 [==============================] - 8s 22ms/step - loss: 2.2612 - acc: 0.4038 - top5-acc: 0.7224 - val_loss: 2.3529 - val_acc: 0.3954 - val_top5-acc: 0.6934
Epoch 21/50
352/352 [==============================] - 8s 22ms/step - loss: 2.2416 - acc: 0.4068 - top5-acc: 0.7262 - val_loss: 2.3014 - val_acc: 0.3980 - val_top5-acc: 0.7158
Epoch 22/50
352/352 [==============================] - 8s 22ms/step - loss: 2.2087 - acc: 0.4162 - top5-acc: 0.7359 - val_loss: 2.2904 - val_acc: 0.4062 - val_top5-acc: 0.7120
Epoch 23/50
352/352 [==============================] - 8s 22ms/step - loss: 2.1803 - acc: 0.4200 - top5-acc: 0.7442 - val_loss: 2.3181 - val_acc: 0.4096 - val_top5-acc: 0.7120
Epoch 24/50
352/352 [==============================] - 8s 22ms/step - loss: 2.1718 - acc: 0.4246 - top5-acc: 0.7403 - val_loss: 2.2687 - val_acc: 0.4094 - val_top5-acc: 0.7234
Epoch 25/50
352/352 [==============================] - 8s 22ms/step - loss: 2.1559 - acc: 0.4198 - top5-acc: 0.7458 - val_loss: 2.2730 - val_acc: 0.4060 - val_top5-acc: 0.7190
Epoch 26/50
352/352 [==============================] - 8s 22ms/step - loss: 2.1285 - acc: 0.4300 - top5-acc: 0.7495 - val_loss: 2.2566 - val_acc: 0.4082 - val_top5-acc: 0.7306
Epoch 27/50
352/352 [==============================] - 8s 22ms/step - loss: 2.1118 - acc: 0.4386 - top5-acc: 0.7538 - val_loss: 2.2544 - val_acc: 0.4178 - val_top5-acc: 0.7218
Epoch 28/50
352/352 [==============================] - 8s 22ms/step - loss: 2.1007 - acc: 0.4408 - top5-acc: 0.7562 - val_loss: 2.2703 - val_acc: 0.4136 - val_top5-acc: 0.7172
Epoch 29/50
352/352 [==============================] - 8s 22ms/step - loss: 2.0707 - acc: 0.4446 - top5-acc: 0.7634 - val_loss: 2.2244 - val_acc: 0.4168 - val_top5-acc: 0.7332
Epoch 30/50
352/352 [==============================] - 8s 22ms/step - loss: 2.0694 - acc: 0.4428 - top5-acc: 0.7611 - val_loss: 2.2557 - val_acc: 0.4060 - val_top5-acc: 0.7270
Epoch 31/50
352/352 [==============================] - 8s 22ms/step - loss: 2.0485 - acc: 0.4502 - top5-acc: 0.7672 - val_loss: 2.2192 - val_acc: 0.4214 - val_top5-acc: 0.7308
Epoch 32/50
352/352 [==============================] - 8s 22ms/step - loss: 2.0105 - acc: 0.4617 - top5-acc: 0.7718 - val_loss: 2.2065 - val_acc: 0.4222 - val_top5-acc: 0.7286
Epoch 33/50
352/352 [==============================] - 8s 22ms/step - loss: 2.0238 - acc: 0.4556 - top5-acc: 0.7734 - val_loss: 2.1736 - val_acc: 0.4270 - val_top5-acc: 0.7368
Epoch 34/50
352/352 [==============================] - 8s 22ms/step - loss: 2.0253 - acc: 0.4547 - top5-acc: 0.7712 - val_loss: 2.2231 - val_acc: 0.4280 - val_top5-acc: 0.7308
Epoch 35/50
352/352 [==============================] - 8s 22ms/step - loss: 1.9992 - acc: 0.4593 - top5-acc: 0.7765 - val_loss: 2.1994 - val_acc: 0.4212 - val_top5-acc: 0.7358
Epoch 36/50
352/352 [==============================] - 8s 22ms/step - loss: 1.9849 - acc: 0.4636 - top5-acc: 0.7754 - val_loss: 2.2167 - val_acc: 0.4276 - val_top5-acc: 0.7308
Epoch 37/50
352/352 [==============================] - 8s 22ms/step - loss: 1.9880 - acc: 0.4677 - top5-acc: 0.7783 - val_loss: 2.1746 - val_acc: 0.4270 - val_top5-acc: 0.7416
Epoch 38/50
352/352 [==============================] - 8s 22ms/step - loss: 1.9562 - acc: 0.4720 - top5-acc: 0.7845 - val_loss: 2.1976 - val_acc: 0.4312 - val_top5-acc: 0.7356
Epoch 39/50
352/352 [==============================] - 8s 22ms/step - loss: 1.8736 - acc: 0.4924 - top5-acc: 0.8004 - val_loss: 2.0755 - val_acc: 0.4578 - val_top5-acc: 0.7586
Epoch 40/50
352/352 [==============================] - 8s 22ms/step - loss: 1.8189 - acc: 0.5042 - top5-acc: 0.8076 - val_loss: 2.0804 - val_acc: 0.4508 - val_top5-acc: 0.7600
Epoch 41/50
352/352 [==============================] - 8s 22ms/step - loss: 1.8069 - acc: 0.5062 - top5-acc: 0.8132 - val_loss: 2.0784 - val_acc: 0.4456 - val_top5-acc: 0.7578
Epoch 42/50
352/352 [==============================] - 8s 22ms/step - loss: 1.8156 - acc: 0.5052 - top5-acc: 0.8110 - val_loss: 2.0910 - val_acc: 0.4544 - val_top5-acc: 0.7542
Epoch 43/50
352/352 [==============================] - 8s 22ms/step - loss: 1.8143 - acc: 0.5046 - top5-acc: 0.8105 - val_loss: 2.1037 - val_acc: 0.4466 - val_top5-acc: 0.7562
Epoch 44/50
352/352 [==============================] - 8s 22ms/step - loss: 1.8119 - acc: 0.5032 - top5-acc: 0.8141 - val_loss: 2.0794 - val_acc: 0.4622 - val_top5-acc: 0.7532
Epoch 45/50
352/352 [==============================] - 8s 22ms/step - loss: 1.7611 - acc: 0.5188 - top5-acc: 0.8224 - val_loss: 2.0371 - val_acc: 0.4650 - val_top5-acc: 0.7628
Epoch 46/50
352/352 [==============================] - 8s 22ms/step - loss: 1.7713 - acc: 0.5189 - top5-acc: 0.8226 - val_loss: 2.0245 - val_acc: 0.4630 - val_top5-acc: 0.7644
Epoch 47/50
352/352 [==============================] - 8s 22ms/step - loss: 1.7809 - acc: 0.5130 - top5-acc: 0.8215 - val_loss: 2.0471 - val_acc: 0.4618 - val_top5-acc: 0.7618
Epoch 48/50
352/352 [==============================] - 8s 22ms/step - loss: 1.8052 - acc: 0.5112 - top5-acc: 0.8165 - val_loss: 2.0441 - val_acc: 0.4596 - val_top5-acc: 0.7658
Epoch 49/50
352/352 [==============================] - 8s 22ms/step - loss: 1.8128 - acc: 0.5039 - top5-acc: 0.8178 - val_loss: 2.0569 - val_acc: 0.4600 - val_top5-acc: 0.7614
Epoch 50/50
352/352 [==============================] - 8s 22ms/step - loss: 1.8179 - acc: 0.5089 - top5-acc: 0.8155 - val_loss: 2.0514 - val_acc: 0.4576 - val_top5-acc: 0.7566
313/313 [==============================] - 2s 6ms/step - loss: 2.0142 - acc: 0.4663 - top5-acc: 0.7647
Test accuracy: 46.63%
Test top 5 accuracy: 76.47%

```
</div>
As shown in the [FNet](https://arxiv.org/abs/2105.03824) paper,
better results can be achieved by increasing the embedding dimensions,
increasing the number of FNet blocks, and training the model for longer.
You may also try to increase the size of the input images and use different patch sizes.
The FNet scales very efficiently to long inputs, runs much faster than attention-based
Transformer models, and produces competitive accuracy results.

---
## The gMLP model

The gMLP is a MLP architecture that features a Spatial Gating Unit (SGU).
The SGU enables cross-patch interactions across the spatial (channel) dimension, by:

1. Transforming the input spatially by applying linear projection across patches (along channels).
2. Applying element-wise multiplication of the input and its spatial transformation.

### Implement the gMLP module


```python

class gMLPLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super(gMLPLayer, self).__init__(*args, **kwargs)

        self.channel_projection1 = keras.Sequential(
            [
                layers.Dense(units=embedding_dim * 2),
                tfa.layers.GELU(),
                layers.Dropout(rate=dropout_rate),
            ]
        )

        self.channel_projection2 = layers.Dense(units=embedding_dim)

        self.spatial_projection = layers.Dense(
            units=num_patches, bias_initializer="Ones"
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def spatial_gating_unit(self, x):
        # Split x along the channel dimensions.
        # Tensors u and v will in th shape of [batch_size, num_patchs, embedding_dim].
        u, v = tf.split(x, num_or_size_splits=2, axis=2)
        # Apply layer normalization.
        v = self.normalize2(v)
        # Apply spatial projection.
        v_channels = tf.linalg.matrix_transpose(v)
        v_projected = self.spatial_projection(v_channels)
        v_projected = tf.linalg.matrix_transpose(v_projected)
        # Apply element-wise multiplication.
        return u * v_projected

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize1(inputs)
        # Apply the first channel projection. x_projected shape: [batch_size, num_patches, embedding_dim * 2].
        x_projected = self.channel_projection1(x)
        # Apply the spatial gating unit. x_spatial shape: [batch_size, num_patches, embedding_dim].
        x_spatial = self.spatial_gating_unit(x_projected)
        # Apply the second channel projection. x_projected shape: [batch_size, num_patches, embedding_dim].
        x_projected = self.channel_projection2(x_spatial)
        # Add skip connection.
        return x + x_projected

```

### Build, train, and evaluate the gMLP model

Note that training the model with the current settings on a V100 GPUs
takes around 9 seconds per epoch.


```python
gmlp_blocks = keras.Sequential(
    [gMLPLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
)
learning_rate = 0.003
gmlp_classifier = build_classifier(gmlp_blocks)
history = run_experiment(gmlp_classifier)
```

<div class="k-default-codeblock">
```
Epoch 1/50
352/352 [==============================] - 13s 28ms/step - loss: 4.1713 - acc: 0.0704 - top5-acc: 0.2206 - val_loss: 3.5629 - val_acc: 0.1548 - val_top5-acc: 0.4086
Epoch 2/50
352/352 [==============================] - 9s 27ms/step - loss: 3.5146 - acc: 0.1633 - top5-acc: 0.4172 - val_loss: 3.2899 - val_acc: 0.2066 - val_top5-acc: 0.4900
Epoch 3/50
352/352 [==============================] - 9s 26ms/step - loss: 3.2588 - acc: 0.2017 - top5-acc: 0.4895 - val_loss: 3.1152 - val_acc: 0.2362 - val_top5-acc: 0.5278
Epoch 4/50
352/352 [==============================] - 9s 26ms/step - loss: 3.1037 - acc: 0.2331 - top5-acc: 0.5288 - val_loss: 2.9771 - val_acc: 0.2624 - val_top5-acc: 0.5646
Epoch 5/50
352/352 [==============================] - 9s 26ms/step - loss: 2.9483 - acc: 0.2637 - top5-acc: 0.5680 - val_loss: 2.8807 - val_acc: 0.2784 - val_top5-acc: 0.5840
Epoch 6/50
352/352 [==============================] - 9s 26ms/step - loss: 2.8411 - acc: 0.2821 - top5-acc: 0.5930 - val_loss: 2.7246 - val_acc: 0.3146 - val_top5-acc: 0.6256
Epoch 7/50
352/352 [==============================] - 9s 26ms/step - loss: 2.7221 - acc: 0.3085 - top5-acc: 0.6193 - val_loss: 2.7022 - val_acc: 0.3108 - val_top5-acc: 0.6270
Epoch 8/50
352/352 [==============================] - 9s 26ms/step - loss: 2.6296 - acc: 0.3334 - top5-acc: 0.6420 - val_loss: 2.6289 - val_acc: 0.3324 - val_top5-acc: 0.6494
Epoch 9/50
352/352 [==============================] - 9s 26ms/step - loss: 2.5691 - acc: 0.3413 - top5-acc: 0.6563 - val_loss: 2.5353 - val_acc: 0.3586 - val_top5-acc: 0.6746
Epoch 10/50
352/352 [==============================] - 9s 26ms/step - loss: 2.4854 - acc: 0.3575 - top5-acc: 0.6760 - val_loss: 2.5271 - val_acc: 0.3578 - val_top5-acc: 0.6720
Epoch 11/50
352/352 [==============================] - 9s 26ms/step - loss: 2.4252 - acc: 0.3722 - top5-acc: 0.6870 - val_loss: 2.4553 - val_acc: 0.3684 - val_top5-acc: 0.6850
Epoch 12/50
352/352 [==============================] - 9s 26ms/step - loss: 2.3814 - acc: 0.3822 - top5-acc: 0.6985 - val_loss: 2.3841 - val_acc: 0.3888 - val_top5-acc: 0.6966
Epoch 13/50
352/352 [==============================] - 9s 26ms/step - loss: 2.3119 - acc: 0.3950 - top5-acc: 0.7135 - val_loss: 2.4306 - val_acc: 0.3780 - val_top5-acc: 0.6894
Epoch 14/50
352/352 [==============================] - 9s 26ms/step - loss: 2.2886 - acc: 0.4033 - top5-acc: 0.7168 - val_loss: 2.4053 - val_acc: 0.3932 - val_top5-acc: 0.7010
Epoch 15/50
352/352 [==============================] - 9s 26ms/step - loss: 2.2455 - acc: 0.4080 - top5-acc: 0.7233 - val_loss: 2.3443 - val_acc: 0.4004 - val_top5-acc: 0.7128
Epoch 16/50
352/352 [==============================] - 9s 26ms/step - loss: 2.2128 - acc: 0.4152 - top5-acc: 0.7317 - val_loss: 2.3150 - val_acc: 0.4018 - val_top5-acc: 0.7174
Epoch 17/50
352/352 [==============================] - 9s 26ms/step - loss: 2.1990 - acc: 0.4206 - top5-acc: 0.7357 - val_loss: 2.3590 - val_acc: 0.3978 - val_top5-acc: 0.7086
Epoch 18/50
352/352 [==============================] - 9s 26ms/step - loss: 2.1574 - acc: 0.4258 - top5-acc: 0.7451 - val_loss: 2.3140 - val_acc: 0.4052 - val_top5-acc: 0.7256
Epoch 19/50
352/352 [==============================] - 9s 26ms/step - loss: 2.1369 - acc: 0.4309 - top5-acc: 0.7487 - val_loss: 2.3012 - val_acc: 0.4124 - val_top5-acc: 0.7190
Epoch 20/50
352/352 [==============================] - 9s 26ms/step - loss: 2.1222 - acc: 0.4350 - top5-acc: 0.7494 - val_loss: 2.3294 - val_acc: 0.4076 - val_top5-acc: 0.7186
Epoch 21/50
352/352 [==============================] - 9s 26ms/step - loss: 2.0822 - acc: 0.4436 - top5-acc: 0.7576 - val_loss: 2.2498 - val_acc: 0.4302 - val_top5-acc: 0.7276
Epoch 22/50
352/352 [==============================] - 9s 26ms/step - loss: 2.0609 - acc: 0.4518 - top5-acc: 0.7610 - val_loss: 2.2915 - val_acc: 0.4232 - val_top5-acc: 0.7280
Epoch 23/50
352/352 [==============================] - 9s 26ms/step - loss: 2.0482 - acc: 0.4590 - top5-acc: 0.7648 - val_loss: 2.2448 - val_acc: 0.4242 - val_top5-acc: 0.7296
Epoch 24/50
352/352 [==============================] - 9s 26ms/step - loss: 2.0292 - acc: 0.4560 - top5-acc: 0.7705 - val_loss: 2.2526 - val_acc: 0.4334 - val_top5-acc: 0.7324
Epoch 25/50
352/352 [==============================] - 9s 26ms/step - loss: 2.0316 - acc: 0.4544 - top5-acc: 0.7687 - val_loss: 2.2430 - val_acc: 0.4318 - val_top5-acc: 0.7338
Epoch 26/50
352/352 [==============================] - 9s 26ms/step - loss: 1.9988 - acc: 0.4616 - top5-acc: 0.7748 - val_loss: 2.2053 - val_acc: 0.4470 - val_top5-acc: 0.7366
Epoch 27/50
352/352 [==============================] - 9s 26ms/step - loss: 1.9788 - acc: 0.4646 - top5-acc: 0.7806 - val_loss: 2.2313 - val_acc: 0.4378 - val_top5-acc: 0.7420
Epoch 28/50
352/352 [==============================] - 9s 26ms/step - loss: 1.9702 - acc: 0.4688 - top5-acc: 0.7829 - val_loss: 2.2392 - val_acc: 0.4344 - val_top5-acc: 0.7338
Epoch 29/50
352/352 [==============================] - 9s 26ms/step - loss: 1.9488 - acc: 0.4699 - top5-acc: 0.7866 - val_loss: 2.1600 - val_acc: 0.4490 - val_top5-acc: 0.7446
Epoch 30/50
352/352 [==============================] - 9s 26ms/step - loss: 1.9302 - acc: 0.4803 - top5-acc: 0.7878 - val_loss: 2.2069 - val_acc: 0.4410 - val_top5-acc: 0.7486
Epoch 31/50
352/352 [==============================] - 9s 26ms/step - loss: 1.9135 - acc: 0.4806 - top5-acc: 0.7916 - val_loss: 2.1929 - val_acc: 0.4486 - val_top5-acc: 0.7514
Epoch 32/50
352/352 [==============================] - 9s 26ms/step - loss: 1.8890 - acc: 0.4844 - top5-acc: 0.7961 - val_loss: 2.2176 - val_acc: 0.4404 - val_top5-acc: 0.7494
Epoch 33/50
352/352 [==============================] - 9s 26ms/step - loss: 1.8844 - acc: 0.4872 - top5-acc: 0.7980 - val_loss: 2.2321 - val_acc: 0.4444 - val_top5-acc: 0.7460
Epoch 34/50
352/352 [==============================] - 9s 26ms/step - loss: 1.8588 - acc: 0.4912 - top5-acc: 0.8005 - val_loss: 2.1895 - val_acc: 0.4532 - val_top5-acc: 0.7510
Epoch 35/50
352/352 [==============================] - 9s 26ms/step - loss: 1.7259 - acc: 0.5232 - top5-acc: 0.8266 - val_loss: 2.1024 - val_acc: 0.4800 - val_top5-acc: 0.7726
Epoch 36/50
352/352 [==============================] - 9s 26ms/step - loss: 1.6262 - acc: 0.5488 - top5-acc: 0.8437 - val_loss: 2.0712 - val_acc: 0.4830 - val_top5-acc: 0.7754
Epoch 37/50
352/352 [==============================] - 9s 26ms/step - loss: 1.6164 - acc: 0.5481 - top5-acc: 0.8390 - val_loss: 2.1219 - val_acc: 0.4772 - val_top5-acc: 0.7678
Epoch 38/50
352/352 [==============================] - 9s 26ms/step - loss: 1.5850 - acc: 0.5568 - top5-acc: 0.8510 - val_loss: 2.0931 - val_acc: 0.4892 - val_top5-acc: 0.7732
Epoch 39/50
352/352 [==============================] - 9s 26ms/step - loss: 1.5741 - acc: 0.5589 - top5-acc: 0.8507 - val_loss: 2.0910 - val_acc: 0.4910 - val_top5-acc: 0.7700
Epoch 40/50
352/352 [==============================] - 9s 26ms/step - loss: 1.5546 - acc: 0.5675 - top5-acc: 0.8519 - val_loss: 2.1388 - val_acc: 0.4790 - val_top5-acc: 0.7742
Epoch 41/50
352/352 [==============================] - 9s 26ms/step - loss: 1.5464 - acc: 0.5684 - top5-acc: 0.8561 - val_loss: 2.1121 - val_acc: 0.4786 - val_top5-acc: 0.7718
Epoch 42/50
352/352 [==============================] - 9s 26ms/step - loss: 1.4494 - acc: 0.5890 - top5-acc: 0.8702 - val_loss: 2.1157 - val_acc: 0.4944 - val_top5-acc: 0.7802
Epoch 43/50
352/352 [==============================] - 9s 26ms/step - loss: 1.3847 - acc: 0.6069 - top5-acc: 0.8825 - val_loss: 2.1048 - val_acc: 0.4884 - val_top5-acc: 0.7752
Epoch 44/50
352/352 [==============================] - 9s 26ms/step - loss: 1.3724 - acc: 0.6087 - top5-acc: 0.8832 - val_loss: 2.0681 - val_acc: 0.4924 - val_top5-acc: 0.7868
Epoch 45/50
352/352 [==============================] - 9s 26ms/step - loss: 1.3643 - acc: 0.6116 - top5-acc: 0.8840 - val_loss: 2.0965 - val_acc: 0.4932 - val_top5-acc: 0.7752
Epoch 46/50
352/352 [==============================] - 9s 26ms/step - loss: 1.3517 - acc: 0.6184 - top5-acc: 0.8849 - val_loss: 2.0869 - val_acc: 0.4956 - val_top5-acc: 0.7778
Epoch 47/50
352/352 [==============================] - 9s 26ms/step - loss: 1.3377 - acc: 0.6211 - top5-acc: 0.8891 - val_loss: 2.1120 - val_acc: 0.4882 - val_top5-acc: 0.7764
Epoch 48/50
352/352 [==============================] - 9s 26ms/step - loss: 1.3369 - acc: 0.6186 - top5-acc: 0.8888 - val_loss: 2.1257 - val_acc: 0.4912 - val_top5-acc: 0.7752
Epoch 49/50
352/352 [==============================] - 9s 26ms/step - loss: 1.3266 - acc: 0.6190 - top5-acc: 0.8893 - val_loss: 2.0961 - val_acc: 0.4958 - val_top5-acc: 0.7828
Epoch 50/50
352/352 [==============================] - 9s 26ms/step - loss: 1.2731 - acc: 0.6352 - top5-acc: 0.8976 - val_loss: 2.0897 - val_acc: 0.4982 - val_top5-acc: 0.7788
313/313 [==============================] - 2s 7ms/step - loss: 2.0743 - acc: 0.5064 - top5-acc: 0.7828
Test accuracy: 50.64%
Test top 5 accuracy: 78.28%

```
</div>
As shown in the [gMLP](https://arxiv.org/abs/2105.08050) paper,
better results can be achieved by increasing the embedding dimensions,
increasing the number of gMLP blocks, and training the model for longer.
You may also try to increase the size of the input images and use different patch sizes.
Note that, the paper used advanced regularization strategies, such as MixUp and CutMix,
as well as AutoAugment.

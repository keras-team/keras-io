# Image classification with Perceiver

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/04/30<br>
**Last modified:** 2021/01/30<br>
**Description:** Implementing the Perceiver model for image classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/perceiver_image_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/perceiver_image_classification.py)



---
## Introduction

This example implements the
[Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)
model by Andrew Jaegle et al. for image classification,
and demonstrates it on the CIFAR-100 dataset.

The Perceiver model leverages an asymmetric attention mechanism to iteratively
distill inputs into a tight latent bottleneck,
allowing it to scale to handle very large inputs.

In other words: let's assume that your input data array (e.g. image) has `M` elements (i.e. patches), where `M` is large.
In a standard Transformer model, a self-attention operation is performed for the `M` elements.
The complexity of this operation is `O(M^2)`.
However, the Perceiver model creates a latent array of size `N` elements, where `N << M`,
and performs two operations iteratively:

1. Cross-attention Transformer between the latent array and the data array - The complexity of this operation is `O(M.N)`.
2. Self-attention Transformer on the latent array -  The complexity of this operation is `O(N^2)`.

This example requires TensorFlow 2.4 or higher, as well as
[TensorFlow Addons](https://www.tensorflow.org/addons/overview),
which can be installed using the following command:

```python
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
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 64
num_epochs = 50
dropout_rate = 0.2
image_size = 64  # We'll resize input images to this size.
patch_size = 2  # Size of the patches to be extract from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
latent_dim = 256  # Size of the latent array.
projection_dim = 256  # Embedding size of each element in the data and latent arrays.
num_heads = 8  # Number of Transformer heads.
ffn_units = [
    projection_dim,
    projection_dim,
]  # Size of the Transformer Feedforward network.
num_transformer_blocks = 4
num_iterations = 2  # Repetitions of the cross-attention and Transformer modules.
classifier_units = [
    projection_dim,
    num_classes,
]  # Size of the Feedforward network of the final classifier.

print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")
print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")
print(f"Latent array shape: {latent_dim} X {projection_dim}")
print(f"Data array shape: {num_patches} X {projection_dim}")
```

<div class="k-default-codeblock">
```
Image size: 64 X 64 = 4096
Patch size: 2 X 2 = 4 
Patches per image: 1024
Elements per patch (3 channels): 12
Latent array shape: 256 X 256
Data array shape: 1024 X 256

```
</div>
Note that, in order to use each pixel as an individual input in the data array,
set `patch_size` to 1.

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
## Implement Feedforward network (FFN)


```python

def create_ffn(hidden_units, dropout_rate):
    ffn_layers = []
    for units in hidden_units[:-1]:
        ffn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    ffn_layers.append(layers.Dense(units=hidden_units[-1]))
    ffn_layers.append(layers.Dropout(dropout_rate))

    ffn = keras.Sequential(ffn_layers)
    return ffn

```

---
## Implement patch creation as a layer


```python

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

```

---
## Implement the patch encoding layer

The `PatchEncoder` layer will linearly transform a patch by projecting it into
a vector of size `latent_dim`. In addition, it adds a learnable position embedding
to the projected vector.

Note that the orginal Perceiver paper uses the Fourier feature positional encodings.


```python

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

```

---
## Build the Perceiver model

The Perceiver consists of two modules: a cross-attention
module and a standard Transformer with self-attention.

### Cross-attention module

The cross-attention expects a `(latent_dim, projection_dim)` latent array,
and the `(data_dim,  projection_dim)` data array as inputs,
to produce a `(latent_dim, projection_dim)` latent array as an output.
To apply cross-attention, the `query` vectors are generated from the latent array,
while the `key` and `value` vectors are generated from the encoded image.

Note that the data array in this example is the image,
where the `data_dim` is set to the `num_patches`.


```python

def create_cross_attention_module(
    latent_dim, data_dim, projection_dim, ffn_units, dropout_rate
):

    inputs = {
        # Recieve the latent array as an input of shape [1, latent_dim, projection_dim].
        "latent_array": layers.Input(shape=(latent_dim, projection_dim)),
        # Recieve the data_array (encoded image) as an input of shape [batch_size, data_dim, projection_dim].
        "data_array": layers.Input(shape=(data_dim, projection_dim)),
    }

    # Apply layer norm to the inputs
    latent_array = layers.LayerNormalization(epsilon=1e-6)(inputs["latent_array"])
    data_array = layers.LayerNormalization(epsilon=1e-6)(inputs["data_array"])

    # Create query tensor: [1, latent_dim, projection_dim].
    query = layers.Dense(units=projection_dim)(latent_array)
    # Create key tensor: [batch_size, data_dim, projection_dim].
    key = layers.Dense(units=projection_dim)(data_array)
    # Create value tensor: [batch_size, data_dim, projection_dim].
    value = layers.Dense(units=projection_dim)(data_array)

    # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
    attention_output = layers.Attention(use_scale=True, dropout=0.1)(
        [query, key, value], return_attention_scores=False
    )
    # Skip connection 1.
    attention_output = layers.Add()([attention_output, latent_array])

    # Apply layer norm.
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    # Apply Feedforward network.
    ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
    outputs = ffn(attention_output)
    # Skip connection 2.
    outputs = layers.Add()([outputs, attention_output])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

### Transformer module

The Transformer expects the output latent vector from the cross-attention module
as an input, applies multi-head self-attention to its `latent_dim` elements,
followed by feedforward network, to produce another `(latent_dim, projection_dim)` latent array.


```python

def create_transformer_module(
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    ffn_units,
    dropout_rate,
):

    # input_shape: [1, latent_dim, projection_dim]
    inputs = layers.Input(shape=(latent_dim, projection_dim))

    x0 = inputs
    # Create multiple layers of the Transformer block.
    for _ in range(num_transformer_blocks):
        # Apply layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x0)
        # Create a multi-head self-attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x0])
        # Apply layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # Apply Feedforward network.
        ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
        x3 = ffn(x3)
        # Skip connection 2.
        x0 = layers.Add()([x3, x2])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=x0)
    return model

```

### Perceiver model

The Perceiver model repeats the cross-attention and Transformer modules
`num_iterations` times—with shared weights and skip connections—to allow
the latent array to iteratively extract information from the input image as it is needed.


```python

class Perceiver(keras.Model):
    def __init__(
        self,
        patch_size,
        data_dim,
        latent_dim,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        ffn_units,
        dropout_rate,
        num_iterations,
        classifier_units,
    ):
        super(Perceiver, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.num_iterations = num_iterations
        self.classifier_units = classifier_units

    def build(self, input_shape):
        # Create latent array.
        self.latent_array = self.add_weight(
            shape=(self.latent_dim, self.projection_dim),
            initializer="random_normal",
            trainable=True,
        )

        # Create patching module.
        self.patcher = Patches(self.patch_size)

        # Create patch encoder.
        self.patch_encoder = PatchEncoder(self.data_dim, self.projection_dim)

        # Create cross-attenion module.
        self.cross_attention = create_cross_attention_module(
            self.latent_dim,
            self.data_dim,
            self.projection_dim,
            self.ffn_units,
            self.dropout_rate,
        )

        # Create Transformer module.
        self.transformer = create_transformer_module(
            self.latent_dim,
            self.projection_dim,
            self.num_heads,
            self.num_transformer_blocks,
            self.ffn_units,
            self.dropout_rate,
        )

        # Create global average pooling layer.
        self.global_average_pooling = layers.GlobalAveragePooling1D()

        # Create a classification head.
        self.classification_head = create_ffn(
            hidden_units=self.classifier_units, dropout_rate=self.dropout_rate
        )

        super(Perceiver, self).build(input_shape)

    def call(self, inputs):
        # Augment data.
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = self.patcher(augmented)
        # Encode patches.
        encoded_patches = self.patch_encoder(patches)
        # Prepare cross-attention inputs.
        cross_attention_inputs = {
            "latent_array": tf.expand_dims(self.latent_array, 0),
            "data_array": encoded_patches,
        }
        # Apply the cross-attention and the Transformer modules iteratively.
        for _ in range(self.num_iterations):
            # Apply cross-attention from the latent array to the data array.
            latent_array = self.cross_attention(cross_attention_inputs)
            # Apply self-attention Transformer to the latent array.
            latent_array = self.transformer(latent_array)
            # Set the latent array of the next iteration.
            cross_attention_inputs["latent_array"] = latent_array

        # Apply global average pooling to generate a [batch_size, projection_dim] repesentation tensor.
        representation = self.global_average_pooling(latent_array)
        # Generate logits.
        logits = self.classification_head(representation)
        return logits

```

---
## Compile, train, and evaluate the mode


```python

def run_experiment(model):

    # Create LAMB optimizer with weight decay.
    optimizer = tfa.optimizers.LAMB(
        learning_rate=learning_rate, weight_decay_rate=weight_decay,
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
        monitor="val_loss", factor=0.2, patience=3
    )

    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
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

Note that training the perceiver model with the current settings on a V100 GPUs takes
around 200 seconds.


```python
perceiver_classifier = Perceiver(
    patch_size,
    num_patches,
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    ffn_units,
    dropout_rate,
    num_iterations,
    classifier_units,
)


history = run_experiment(perceiver_classifier)
```

<div class="k-default-codeblock">
```
Epoch 1/100
704/704 [==============================] - 305s 405ms/step - loss: 4.4550 - acc: 0.0389 - top5-acc: 0.1407 - val_loss: 4.0544 - val_acc: 0.0802 - val_top5-acc: 0.2516
Epoch 2/100
704/704 [==============================] - 284s 403ms/step - loss: 4.0639 - acc: 0.0889 - top5-acc: 0.2576 - val_loss: 3.7379 - val_acc: 0.1272 - val_top5-acc: 0.3556
Epoch 3/100
704/704 [==============================] - 283s 402ms/step - loss: 3.8400 - acc: 0.1226 - top5-acc: 0.3326 - val_loss: 3.4527 - val_acc: 0.1750 - val_top5-acc: 0.4350
Epoch 4/100
704/704 [==============================] - 283s 402ms/step - loss: 3.5917 - acc: 0.1657 - top5-acc: 0.4063 - val_loss: 3.2160 - val_acc: 0.2176 - val_top5-acc: 0.5048
Epoch 5/100
704/704 [==============================] - 283s 403ms/step - loss: 3.3820 - acc: 0.2082 - top5-acc: 0.4638 - val_loss: 2.9947 - val_acc: 0.2584 - val_top5-acc: 0.5732
Epoch 6/100
704/704 [==============================] - 284s 403ms/step - loss: 3.2487 - acc: 0.2338 - top5-acc: 0.4991 - val_loss: 2.9179 - val_acc: 0.2770 - val_top5-acc: 0.5744
Epoch 7/100
704/704 [==============================] - 283s 402ms/step - loss: 3.1228 - acc: 0.2605 - top5-acc: 0.5295 - val_loss: 2.7958 - val_acc: 0.2994 - val_top5-acc: 0.6100
Epoch 8/100
704/704 [==============================] - 283s 402ms/step - loss: 2.9989 - acc: 0.2862 - top5-acc: 0.5588 - val_loss: 2.7117 - val_acc: 0.3208 - val_top5-acc: 0.6340
Epoch 9/100
704/704 [==============================] - 283s 402ms/step - loss: 2.9294 - acc: 0.3018 - top5-acc: 0.5763 - val_loss: 2.5933 - val_acc: 0.3390 - val_top5-acc: 0.6636
Epoch 10/100
704/704 [==============================] - 283s 402ms/step - loss: 2.8687 - acc: 0.3139 - top5-acc: 0.5934 - val_loss: 2.5030 - val_acc: 0.3614 - val_top5-acc: 0.6764
Epoch 11/100
704/704 [==============================] - 283s 402ms/step - loss: 2.7771 - acc: 0.3341 - top5-acc: 0.6098 - val_loss: 2.4657 - val_acc: 0.3704 - val_top5-acc: 0.6928
Epoch 12/100
704/704 [==============================] - 283s 402ms/step - loss: 2.7306 - acc: 0.3436 - top5-acc: 0.6229 - val_loss: 2.4441 - val_acc: 0.3738 - val_top5-acc: 0.6878
Epoch 13/100
704/704 [==============================] - 283s 402ms/step - loss: 2.6863 - acc: 0.3546 - top5-acc: 0.6346 - val_loss: 2.3508 - val_acc: 0.3892 - val_top5-acc: 0.7050
Epoch 14/100
704/704 [==============================] - 283s 402ms/step - loss: 2.6107 - acc: 0.3708 - top5-acc: 0.6537 - val_loss: 2.3219 - val_acc: 0.3996 - val_top5-acc: 0.7108
Epoch 15/100
704/704 [==============================] - 283s 402ms/step - loss: 2.5559 - acc: 0.3836 - top5-acc: 0.6664 - val_loss: 2.2748 - val_acc: 0.4140 - val_top5-acc: 0.7242
Epoch 16/100
704/704 [==============================] - 283s 402ms/step - loss: 2.5016 - acc: 0.3942 - top5-acc: 0.6761 - val_loss: 2.2364 - val_acc: 0.4238 - val_top5-acc: 0.7264
Epoch 17/100
704/704 [==============================] - 283s 402ms/step - loss: 2.4554 - acc: 0.4056 - top5-acc: 0.6897 - val_loss: 2.1684 - val_acc: 0.4418 - val_top5-acc: 0.7452
Epoch 18/100
704/704 [==============================] - 283s 402ms/step - loss: 2.3926 - acc: 0.4209 - top5-acc: 0.7024 - val_loss: 2.1614 - val_acc: 0.4372 - val_top5-acc: 0.7428
Epoch 19/100
704/704 [==============================] - 283s 402ms/step - loss: 2.3617 - acc: 0.4264 - top5-acc: 0.7119 - val_loss: 2.1595 - val_acc: 0.4382 - val_top5-acc: 0.7408
Epoch 20/100
704/704 [==============================] - 283s 402ms/step - loss: 2.3355 - acc: 0.4324 - top5-acc: 0.7133 - val_loss: 2.1187 - val_acc: 0.4462 - val_top5-acc: 0.7490
Epoch 21/100
704/704 [==============================] - 283s 402ms/step - loss: 2.2571 - acc: 0.4512 - top5-acc: 0.7299 - val_loss: 2.1095 - val_acc: 0.4424 - val_top5-acc: 0.7534
Epoch 22/100
704/704 [==============================] - 283s 402ms/step - loss: 2.2374 - acc: 0.4559 - top5-acc: 0.7357 - val_loss: 2.0997 - val_acc: 0.4398 - val_top5-acc: 0.7554
Epoch 23/100
704/704 [==============================] - 283s 402ms/step - loss: 2.2108 - acc: 0.4628 - top5-acc: 0.7452 - val_loss: 2.0662 - val_acc: 0.4574 - val_top5-acc: 0.7598
Epoch 24/100
704/704 [==============================] - 283s 402ms/step - loss: 2.1628 - acc: 0.4728 - top5-acc: 0.7555 - val_loss: 2.0564 - val_acc: 0.4564 - val_top5-acc: 0.7584
Epoch 25/100
704/704 [==============================] - 283s 402ms/step - loss: 2.1169 - acc: 0.4834 - top5-acc: 0.7616 - val_loss: 2.0793 - val_acc: 0.4600 - val_top5-acc: 0.7538
Epoch 26/100
704/704 [==============================] - 283s 402ms/step - loss: 2.0938 - acc: 0.4867 - top5-acc: 0.7743 - val_loss: 2.0835 - val_acc: 0.4566 - val_top5-acc: 0.7506
Epoch 27/100
704/704 [==============================] - 283s 402ms/step - loss: 2.0479 - acc: 0.4993 - top5-acc: 0.7816 - val_loss: 2.0790 - val_acc: 0.4610 - val_top5-acc: 0.7556
Epoch 28/100
704/704 [==============================] - 283s 402ms/step - loss: 1.8480 - acc: 0.5493 - top5-acc: 0.8159 - val_loss: 1.8846 - val_acc: 0.5046 - val_top5-acc: 0.7890
Epoch 29/100
704/704 [==============================] - 283s 402ms/step - loss: 1.7532 - acc: 0.5731 - top5-acc: 0.8362 - val_loss: 1.8844 - val_acc: 0.5106 - val_top5-acc: 0.7976
Epoch 30/100
704/704 [==============================] - 283s 402ms/step - loss: 1.7113 - acc: 0.5827 - top5-acc: 0.8434 - val_loss: 1.8792 - val_acc: 0.5096 - val_top5-acc: 0.7928
Epoch 31/100
704/704 [==============================] - 283s 403ms/step - loss: 1.6831 - acc: 0.5891 - top5-acc: 0.8511 - val_loss: 1.8938 - val_acc: 0.5044 - val_top5-acc: 0.7914
Epoch 32/100
704/704 [==============================] - 284s 403ms/step - loss: 1.6480 - acc: 0.5977 - top5-acc: 0.8562 - val_loss: 1.9055 - val_acc: 0.5034 - val_top5-acc: 0.7922
Epoch 33/100
704/704 [==============================] - 284s 403ms/step - loss: 1.6320 - acc: 0.6015 - top5-acc: 0.8627 - val_loss: 1.9064 - val_acc: 0.5056 - val_top5-acc: 0.7896
Epoch 34/100
704/704 [==============================] - 283s 403ms/step - loss: 1.5821 - acc: 0.6145 - top5-acc: 0.8673 - val_loss: 1.8912 - val_acc: 0.5138 - val_top5-acc: 0.7936
Epoch 35/100
704/704 [==============================] - 283s 403ms/step - loss: 1.5791 - acc: 0.6163 - top5-acc: 0.8719 - val_loss: 1.8963 - val_acc: 0.5090 - val_top5-acc: 0.7982
Epoch 36/100
704/704 [==============================] - 283s 402ms/step - loss: 1.5680 - acc: 0.6178 - top5-acc: 0.8741 - val_loss: 1.8998 - val_acc: 0.5142 - val_top5-acc: 0.7936
Epoch 37/100
704/704 [==============================] - 284s 403ms/step - loss: 1.5506 - acc: 0.6218 - top5-acc: 0.8743 - val_loss: 1.8941 - val_acc: 0.5142 - val_top5-acc: 0.7952
Epoch 38/100
704/704 [==============================] - 283s 402ms/step - loss: 1.5611 - acc: 0.6216 - top5-acc: 0.8722 - val_loss: 1.8946 - val_acc: 0.5183 - val_top5-acc: 0.7956
Epoch 39/100
704/704 [==============================] - 284s 403ms/step - loss: 1.5541 - acc: 0.6215 - top5-acc: 0.8764 - val_loss: 1.8923 - val_acc: 0.5180 - val_top5-acc: 0.7962
Epoch 40/100
704/704 [==============================] - 283s 403ms/step - loss: 1.5505 - acc: 0.6228 - top5-acc: 0.8773 - val_loss: 1.8934 - val_acc: 0.5232 - val_top5-acc: 0.7962
Epoch 41/100
704/704 [==============================] - 283s 402ms/step - loss: 1.5604 - acc: 0.6224 - top5-acc: 0.8747 - val_loss: 1.8938 - val_acc: 0.5230 - val_top5-acc: 0.7958
Epoch 42/100
704/704 [==============================] - 283s 402ms/step - loss: 1.5545 - acc: 0.6194 - top5-acc: 0.8784 - val_loss: 1.8938 - val_acc: 0.5240 - val_top5-acc: 0.7966
Epoch 43/100
704/704 [==============================] - 283s 402ms/step - loss: 1.5630 - acc: 0.6210 - top5-acc: 0.8758 - val_loss: 1.8939 - val_acc: 0.5240 - val_top5-acc: 0.7958
Epoch 44/100
704/704 [==============================] - 283s 402ms/step - loss: 1.5569 - acc: 0.6198 - top5-acc: 0.8756 - val_loss: 1.8938 - val_acc: 0.5240 - val_top5-acc: 0.7060
Epoch 45/100
704/704 [==============================] - 283s 402ms/step - loss: 1.5569 - acc: 0.6197 - top5-acc: 0.8770 - val_loss: 1.8940 - val_acc: 0.5140 - val_top5-acc: 0.7962
313/313 [==============================] - 22s 69ms/step - loss: 1.8630 - acc: 0.5264 - top5-acc: 0.8087
Test accuracy: 52.64%
Test top 5 accuracy: 80.87%

```
</div>
After 45 epochs, the Perceiver model achieves around 53% accuracy and 81% top-5 accuracy on the test data.

As mentioned in the ablations of the [Perceiver](https://arxiv.org/abs/2103.03206) paper,
you can obtain better results by increasing the latent array size,
increasing the (projection) dimensions of the latent array and data array elements,
increasing the number of blocks in the Transformer module, and increasing the number of iterations of applying
the cross-attention and the latent Transformer modules. You may also try to increase the size the input images
and use different patch sizes.

The Perceiver benefits from inceasing the model size. However, larger models needs bigger accelerators
to fit in and train efficiently. This is why in the Perceiver paper they used 32 TPU cores to run the experiments.

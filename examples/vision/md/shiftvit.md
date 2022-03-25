# A Vision Transformer without Attention

**Author:** [Aritra Roy Gosthipaty](https://twitter.com/ariG23498), [Ritwik Raha](https://twitter.com/ritwik_raha)<br>
**Date created:** 2022/02/24<br>
**Last modified:** 2022/03/01<br>
**Description:** A minimal implementation of ShiftViT.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/shiftvit.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/shiftvit.py)



---
## Introduction

[Vision Transformers](https://arxiv.org/abs/2010.11929) (ViTs) have sparked a wave of
research at the intersection of Transformers and Computer Vision (CV).

ViTs can simultaneously model long- and short-range dependencies, thanks to
the Multi-Head Self-Attention mechanism in the Transformer block. Many researchers believe
that the success of ViTs are purely due to the attention layer, and they seldom
think about other parts of the ViT model.

In the academic paper
[When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism](https://arxiv.org/abs/2201.10801)
the authors propose to demystify the success of ViTs with the introduction of a **NO
PARAMETER** operation in place of the attention operation. They swap the attention
operation with a shifting operation.

In this example, we minimally implement the paper with close alignement to the author's
[official implementation](https://github.com/microsoft/SPACH/blob/main/models/shiftvit.py).

This example requires TensorFlow 2.6 or higher, as well as TensorFlow Addons, which can
be installed using the following command:

```shell
pip install -qq -U tensorflow-addons
```

---
## Setup and imports


```python
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa

# Setting seed for reproducibiltiy
SEED = 42
keras.utils.set_random_seed(SEED)
```

---
## Hyperparameters

These are the hyperparameters that we have chosen for the experiment.
Please feel free to tune them.


```python

class Config(object):
    # DATA
    batch_size = 256
    buffer_size = batch_size * 2
    input_shape = (32, 32, 3)
    num_classes = 10

    # AUGMENTATION
    image_size = 48

    # ARCHITECTURE
    patch_size = 4
    projected_dim = 96
    num_shift_blocks_per_stages = [2, 4, 8, 2]
    epsilon = 1e-5
    stochastic_depth_rate = 0.2
    mlp_dropout_rate = 0.2
    num_div = 12
    shift_pixel = 1
    mlp_expand_ratio = 2

    # OPTIMIZER
    lr_start = 1e-5
    lr_max = 1e-3
    weight_decay = 1e-4

    # TRAINING
    epochs = 100


config = Config()
```

---
## Load the CIFAR-10 dataset

We use the CIFAR-10 dataset for our experiments.


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
(x_train, y_train), (x_val, y_val) = (
    (x_train[:40000], y_train[:40000]),
    (x_train[40000:], y_train[40000:]),
)
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Testing samples: {len(x_test)}")

AUTO = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(config.buffer_size).batch(config.batch_size).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(config.batch_size).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(config.batch_size).prefetch(AUTO)
```

<div class="k-default-codeblock">
```
Training samples: 40000
Validation samples: 10000
Testing samples: 10000

2022-03-01 03:10:21.342684: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-01 03:10:21.850844: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38420 MB memory:  -> device: 0, name: NVIDIA A100-PCIE-40GB, pci bus id: 0000:61:00.0, compute capability: 8.0

```
</div>
---
## Data Augmentation

The augmentation pipeline consists of:

- Rescaling
- Resizing
- Random cropping
- Random horizontal flipping

_Note_: The image data augmentation layers do not apply
data transformations at inference time. This means that
when these layers are called with `training=False` they
behave differently. Refer to the
[documentation](https://keras.io/api/layers/preprocessing_layers/image_augmentation/)
for more details.


```python

def get_augmentation_model():
    """Build the data augmentation model."""
    data_augmentation = keras.Sequential(
        [
            layers.Resizing(config.input_shape[0] + 20, config.input_shape[0] + 20),
            layers.RandomCrop(config.image_size, config.image_size),
            layers.RandomFlip("horizontal"),
            layers.Rescaling(1 / 255.0),
        ]
    )
    return data_augmentation

```

---
## The ShiftViT architecture

In this section, we build the architecture proposed in
[the ShiftViT paper](https://arxiv.org/abs/2201.10801).

| ![ShiftViT Architecture](https://i.imgur.com/CHU40HX.png) |
| :--: |
| Figure 1: The entire architecutre of ShiftViT.
[Source](https://arxiv.org/abs/2201.10801) |

The architecture as shown in Fig. 1, is inspired by
[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030).
Here the authors propose a modular architecture with 4 stages. Each stage works on its
own spatial size, creating a hierarchical architecture.

An input image of size `HxWx3` is split into non-overlapping patches of size `4x4`.
This is done via the patchify layer which results in individual tokens of feature size `48`
(`4x4x3`). Each stage comprises two parts:

1. Embedding Generation
2. Stacked Shift Blocks

We discuss the stages and the modules in detail in what follows.

_Note_: Compared to the [official implementation](https://github.com/microsoft/SPACH/blob/main/models/shiftvit.py)
we restructure some key components to better fit the Keras API.

### The ShiftViT Block

| ![ShiftViT block](https://i.imgur.com/IDe35vo.gif) |
| :--: |
| Figure 2: From the Model to a Shift Block. |

Each stage in the ShiftViT architecture comprises of a Shift Block as shown in Fig 2.

| ![Shift Vit Block](https://i.imgur.com/0q13pLu.png) |
| :--: |
| Figure 3: The Shift ViT Block. [Source](https://arxiv.org/abs/2201.10801) |

The Shift Block as shown in Fig. 3, comprises of the following:

1. Shift Operation
2. Linear Normalization
3. MLP Layer

#### The MLP block

The MLP block is intended to be a stack of densely-connected layers.s


```python

class MLP(layers.Layer):
    """Get the MLP layer for each shift block.

    Args:
        mlp_expand_ratio (int): The ratio with which the first feature map is expanded.
        mlp_dropout_rate (float): The rate for dropout.
    """

    def __init__(self, mlp_expand_ratio, mlp_dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.mlp_expand_ratio = mlp_expand_ratio
        self.mlp_dropout_rate = mlp_dropout_rate

    def build(self, input_shape):
        input_channels = input_shape[-1]
        initial_filters = int(self.mlp_expand_ratio * input_channels)

        self.mlp = keras.Sequential(
            [
                layers.Dense(units=initial_filters, activation=tf.nn.gelu,),
                layers.Dropout(rate=self.mlp_dropout_rate),
                layers.Dense(units=input_channels),
                layers.Dropout(rate=self.mlp_dropout_rate),
            ]
        )

    def call(self, x):
        x = self.mlp(x)
        return x

```

#### The DropPath layer

Stochastic depth is a regularization technique that randomly drops a set of
layers. During inference, the layers are kept as they are. It is very
similar to Dropout, but it operates on a block of layers rather
than on individual nodes present inside a layer.


```python

class DropPath(layers.Layer):
    """Drop Path also known as the Stochastic Depth layer.

    Refernece:
        - https://keras.io/examples/vision/cct/#stochastic-depth-for-regularization
        - github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_prob = drop_path_prob

    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_path_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

```

#### Block

The most important operation in this paper is the **shift opperation**. In this section,
we describe the shift operation and compare it with its original implementation provided
by the authors.

A generic feature map is assumed to have the shape `[N, H, W, C]`. Here we choose a
`num_div` parameter that decides the division size of the channels. The first 4 divisions
are shifted (1 pixel) in the left, right, up, and down direction. The remaining splits
are kept as is. After partial shifting the shifted channels are padded and the overflown
pixels are chopped off. This completes the partial shifting operation.

In the original implementation, the code is approximately:

```python
out[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  # shift left
out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right
out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up
out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down

out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift
```

In TensorFlow it would be infeasible for us to assign shifted channels to a tensor in the
middle of the training process. This is why we have resorted to the following procedure:

1. Split the channels with the `num_div` parameter.
2. Select each of the first four spilts and shift and pad them in the respective
directions.
3. After shifting and padding, we concatenate the channel back.

| ![Manim rendered animation for shift operation](https://i.imgur.com/PReeULP.gif) |
| :--: |
| Figure 4: The TensorFlow style shifting |

The entire procedure is explained in the Fig. 4.


```python

class ShiftViTBlock(layers.Layer):
    """A unit ShiftViT Block

    Args:
        shift_pixel (int): The number of pixels to shift. Default to 1.
        mlp_expand_ratio (int): The ratio with which MLP features are
            expanded. Default to 2.
        mlp_dropout_rate (float): The dropout rate used in MLP.
        num_div (int): The number of divisions of the feature map's channel.
            Totally, 4/num_div of channels will be shifted. Defaults to 12.
        epsilon (float): Epsilon constant.
        drop_path_prob (float): The drop probability for drop path.
    """

    def __init__(
        self,
        epsilon,
        drop_path_prob,
        mlp_dropout_rate,
        num_div=12,
        shift_pixel=1,
        mlp_expand_ratio=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shift_pixel = shift_pixel
        self.mlp_expand_ratio = mlp_expand_ratio
        self.mlp_dropout_rate = mlp_dropout_rate
        self.num_div = num_div
        self.epsilon = epsilon
        self.drop_path_prob = drop_path_prob

    def build(self, input_shape):
        self.H = input_shape[1]
        self.W = input_shape[2]
        self.C = input_shape[3]
        self.layer_norm = layers.LayerNormalization(epsilon=self.epsilon)
        self.drop_path = (
            DropPath(drop_path_prob=self.drop_path_prob)
            if self.drop_path_prob > 0.0
            else layers.Activation("linear")
        )
        self.mlp = MLP(
            mlp_expand_ratio=self.mlp_expand_ratio,
            mlp_dropout_rate=self.mlp_dropout_rate,
        )

    def get_shift_pad(self, x, mode):
        """Shifts the channels according to the mode chosen."""
        if mode == "left":
            offset_height = 0
            offset_width = 0
            target_height = 0
            target_width = self.shift_pixel
        elif mode == "right":
            offset_height = 0
            offset_width = self.shift_pixel
            target_height = 0
            target_width = self.shift_pixel
        elif mode == "up":
            offset_height = 0
            offset_width = 0
            target_height = self.shift_pixel
            target_width = 0
        else:
            offset_height = self.shift_pixel
            offset_width = 0
            target_height = self.shift_pixel
            target_width = 0
        crop = tf.image.crop_to_bounding_box(
            x,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=self.H - target_height,
            target_width=self.W - target_width,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=self.H,
            target_width=self.W,
        )
        return shift_pad

    def call(self, x, training=False):
        # Split the feature maps
        x_splits = tf.split(x, num_or_size_splits=self.C // self.num_div, axis=-1)

        # Shift the feature maps
        x_splits[0] = self.get_shift_pad(x_splits[0], mode="left")
        x_splits[1] = self.get_shift_pad(x_splits[1], mode="right")
        x_splits[2] = self.get_shift_pad(x_splits[2], mode="up")
        x_splits[3] = self.get_shift_pad(x_splits[3], mode="down")

        # Concatenate the shifted and unshifted feature maps
        x = tf.concat(x_splits, axis=-1)

        # Add the residual connection
        shortcut = x
        x = shortcut + self.drop_path(self.mlp(self.layer_norm(x)), training=training)
        return x

```

### The ShiftViT blocks

| ![Shift Blokcs](https://i.imgur.com/FKy5NnD.png) |
| :--: |
| Figure 5: Shift Blocks in the architecture. [Source](https://arxiv.org/abs/2201.10801) |

Each stage of the architecture has shift blocks as shown in Fig.5. Each of these blocks
contain a variable number of stacked ShiftViT block (as built in the earlier section).

Shift blocks are followed by a PatchMerging layer that scales down feature inputs. The
PatchMerging layer helps in the pyramidal structure of the model.

#### The PatchMerging layer

This layer merges the two adjacent tokens. This layer helps in scaling the features down
spatially and increasing the features up channel wise. We use a Conv2D layer to merge the
patches.


```python

class PatchMerging(layers.Layer):
    """The Patch Merging layer.

    Args:
        epsilon (float): The epsilon constant.
    """

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        filters = 2 * input_shape[-1]
        self.reduction = layers.Conv2D(
            filters=filters, kernel_size=2, strides=2, padding="same", use_bias=False
        )
        self.layer_norm = layers.LayerNormalization(epsilon=self.epsilon)

    def call(self, x):
        # Apply the patch merging algorithm on the feature maps
        x = self.layer_norm(x)
        x = self.reduction(x)
        return x

```

#### Stacked Shift Blocks

Each stage will have a variable number of stacked ShiftViT Blocks, as suggested in
the paper. This is a generic layer that will contain the stacked shift vit blocks
with the patch merging layer as well. Combining the two operations (shift ViT
block and patch merging) is a design choice we picked for better code reusability.


```python
# Note: This layer will have a different depth of stacking
# for different stages on the model.
class StackedShiftBlocks(layers.Layer):
    """The layer containing stacked ShiftViTBlocks.

    Args:
        epsilon (float): The epsilon constant.
        mlp_dropout_rate (float): The dropout rate used in the MLP block.
        num_shift_blocks (int): The number of shift vit blocks for this stage.
        stochastic_depth_rate (float): The maximum drop path rate chosen.
        is_merge (boolean): A flag that determines the use of the Patch Merge
            layer after the shift vit blocks.
        num_div (int): The division of channels of the feature map. Defaults to 12.
        shift_pixel (int): The number of pixels to shift. Defaults to 1.
        mlp_expand_ratio (int): The ratio with which the initial dense layer of
            the MLP is expanded Defaults to 2.
    """

    def __init__(
        self,
        epsilon,
        mlp_dropout_rate,
        num_shift_blocks,
        stochastic_depth_rate,
        is_merge,
        num_div=12,
        shift_pixel=1,
        mlp_expand_ratio=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.mlp_dropout_rate = mlp_dropout_rate
        self.num_shift_blocks = num_shift_blocks
        self.stochastic_depth_rate = stochastic_depth_rate
        self.is_merge = is_merge
        self.num_div = num_div
        self.shift_pixel = shift_pixel
        self.mlp_expand_ratio = mlp_expand_ratio

    def build(self, input_shapes):
        # Calculate stochastic depth probabilities.
        # Reference: https://keras.io/examples/vision/cct/#the-final-cct-model
        dpr = [
            x
            for x in np.linspace(
                start=0, stop=self.stochastic_depth_rate, num=self.num_shift_blocks
            )
        ]

        # Build the shift blocks as a list of ShiftViT Blocks
        self.shift_blocks = list()
        for num in range(self.num_shift_blocks):
            self.shift_blocks.append(
                ShiftViTBlock(
                    num_div=self.num_div,
                    epsilon=self.epsilon,
                    drop_path_prob=dpr[num],
                    mlp_dropout_rate=self.mlp_dropout_rate,
                    shift_pixel=self.shift_pixel,
                    mlp_expand_ratio=self.mlp_expand_ratio,
                )
            )
        if self.is_merge:
            self.patch_merge = PatchMerging(epsilon=self.epsilon)

    def call(self, x, training=False):
        for shift_block in self.shift_blocks:
            x = shift_block(x, training=training)
        if self.is_merge:
            x = self.patch_merge(x)
        return x

```

---
## The ShiftViT model

Build the ShiftViT custom model.


```python

class ShiftViTModel(keras.Model):
    """The ShiftViT Model.

    Args:
        data_augmentation (keras.Model): A data augmentation model.
        projected_dim (int): The dimension to which the patches of the image are
            projected.
        patch_size (int): The patch size of the images.
        num_shift_blocks_per_stages (list[int]): A list of all the number of shit
            blocks per stage.
        epsilon (float): The epsilon constant.
        mlp_dropout_rate (float): The dropout rate used in the MLP block.
        stochastic_depth_rate (float): The maximum drop rate probability.
        num_div (int): The number of divisions of the channesl of the feature
            map. Defaults to 12.
        shift_pixel (int): The number of pixel to shift. Default to 1.
        mlp_expand_ratio (int): The ratio with which the initial mlp dense layer
            is expanded to. Defaults to 2.
    """

    def __init__(
        self,
        data_augmentation,
        projected_dim,
        patch_size,
        num_shift_blocks_per_stages,
        epsilon,
        mlp_dropout_rate,
        stochastic_depth_rate,
        num_div=12,
        shift_pixel=1,
        mlp_expand_ratio=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_augmentation = data_augmentation
        self.patch_projection = layers.Conv2D(
            filters=projected_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="same",
        )
        self.stages = list()
        for index, num_shift_blocks in enumerate(num_shift_blocks_per_stages):
            if index == len(num_shift_blocks_per_stages) - 1:
                # This is the last stage, do not use the patch merge here.
                is_merge = False
            else:
                is_merge = True
            # Build the stages.
            self.stages.append(
                StackedShiftBlocks(
                    epsilon=epsilon,
                    mlp_dropout_rate=mlp_dropout_rate,
                    num_shift_blocks=num_shift_blocks,
                    stochastic_depth_rate=stochastic_depth_rate,
                    is_merge=is_merge,
                    num_div=num_div,
                    shift_pixel=shift_pixel,
                    mlp_expand_ratio=mlp_expand_ratio,
                )
            )
        self.global_avg_pool = layers.GlobalAveragePooling2D()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "data_augmentation": self.data_augmentation,
                "patch_projection": self.patch_projection,
                "stages": self.stages,
                "global_avg_pool": self.global_avg_pool,
            }
        )
        return config

    def _calculate_loss(self, data, training=False):
        (images, labels) = data

        # Augment the images
        augmented_images = self.data_augmentation(images, training=training)

        # Create patches and project the pathces.
        projected_patches = self.patch_projection(augmented_images)

        # Pass through the stages
        x = projected_patches
        for stage in self.stages:
            x = stage(x, training=training)

        # Get the logits.
        logits = self.global_avg_pool(x)

        # Calculate the loss and return it.
        total_loss = self.compiled_loss(labels, logits)
        return total_loss, labels, logits

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            total_loss, labels, logits = self._calculate_loss(
                data=inputs, training=True
            )

        # Apply gradients.
        train_vars = [
            self.data_augmentation.trainable_variables,
            self.patch_projection.trainable_variables,
            self.global_avg_pool.trainable_variables,
        ]
        train_vars = train_vars + [stage.trainable_variables for stage in self.stages]

        # Optimize the gradients.
        grads = tape.gradient(total_loss, train_vars)
        trainable_variable_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                trainable_variable_list.append((g, v))
        self.optimizer.apply_gradients(trainable_variable_list)

        # Update the metrics
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        _, labels, logits = self._calculate_loss(data=data, training=False)

        # Update the metrics
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

```

---
## Instantiate the model


```python
model = ShiftViTModel(
    data_augmentation=get_augmentation_model(),
    projected_dim=config.projected_dim,
    patch_size=config.patch_size,
    num_shift_blocks_per_stages=config.num_shift_blocks_per_stages,
    epsilon=config.epsilon,
    mlp_dropout_rate=config.mlp_dropout_rate,
    stochastic_depth_rate=config.stochastic_depth_rate,
    num_div=config.num_div,
    shift_pixel=config.shift_pixel,
    mlp_expand_ratio=config.mlp_expand_ratio,
)
```

---
## Learning rate schedule

In many experiments, we want to warm up the model with a slowly increasing learning rate
and then cool down the model with a slowly decaying learning rate. In the warmup cosine
decay, the learning rate linearly increases for the warmup steps and then decays with a
cosine decay.


```python
# Some code is taken from:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a warmup cosine decay schedule."""

    def __init__(self, lr_start, lr_max, warmup_steps, total_steps):
        """
        Args:
            lr_start: The initial learning rate
            lr_max: The maximum learning rate to which lr should increase to in
                the warmup steps
            warmup_steps: The number of steps for which the model warms up
            total_steps: The total number of steps for the model training
        """
        super().__init__()
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        # Check whether the total number of steps is larger than the warmup
        # steps. If not, then throw a value error.
        if self.total_steps < self.warmup_steps:
            raise ValueError(
                f"Total number of steps {self.total_steps} must be"
                + f"larger or equal to warmup steps {self.warmup_steps}."
            )

        # `cos_annealed_lr` is a graph that increases to 1 from the initial
        # step to the warmup step. After that this graph decays to -1 at the
        # final step mark.
        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        )

        # Shift the mean of the `cos_annealed_lr` graph to 1. Now the grpah goes
        # from 0 to 2. Normalize the graph with 0.5 so that now it goes from 0
        # to 1. With the normalized graph we scale it with `lr_max` such that
        # it goes from 0 to `lr_max`
        learning_rate = 0.5 * self.lr_max * (1 + cos_annealed_lr)

        # Check whether warmup_steps is more than 0.
        if self.warmup_steps > 0:
            # Check whether lr_max is larger that lr_start. If not, throw a value
            # error.
            if self.lr_max < self.lr_start:
                raise ValueError(
                    f"lr_start {self.lr_start} must be smaller or"
                    + f"equal to lr_max {self.lr_max}."
                )

            # Calculate the slope with which the learning rate should increase
            # in the warumup schedule. The formula for slope is m = ((b-a)/steps)
            slope = (self.lr_max - self.lr_start) / self.warmup_steps

            # With the formula for a straight line (y = mx+c) build the warmup
            # schedule
            warmup_rate = slope * tf.cast(step, tf.float32) + self.lr_start

            # When the current step is lesser that warmup steps, get the line
            # graph. When the current step is greater than the warmup steps, get
            # the scaled cos graph.
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )

        # When the current step is more that the total steps, return 0 else return
        # the calculated graph.
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

```

---
## Compile and train the model


```python
# Get the total number of steps for training.
total_steps = int((len(x_train) / config.batch_size) * config.epochs)

# Calculate the number of steps for warmup.
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps * warmup_epoch_percentage)

# Initialize the warmupcosine schedule.
scheduled_lrs = WarmUpCosine(
    lr_start=1e-5, lr_max=1e-3, warmup_steps=warmup_steps, total_steps=total_steps,
)

# Get the optimizer.
optimizer = tfa.optimizers.AdamW(
    learning_rate=scheduled_lrs, weight_decay=config.weight_decay
)

# Compile and pretrain the model.
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

# Train the model
history = model.fit(
    train_ds,
    epochs=config.epochs,
    validation_data=val_ds,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, mode="auto",)
    ],
)

# Evaluate the model with the test dataset.
print("TESTING")
loss, acc_top1, acc_top5 = model.evaluate(test_ds)
print(f"Loss: {loss:0.2f}")
print(f"Top 1 test accuracy: {acc_top1*100:0.2f}%")
print(f"Top 5 test accuracy: {acc_top5*100:0.2f}%")
```

<div class="k-default-codeblock">
```
Epoch 1/100

2022-03-01 03:10:41.373231: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8202
2022-03-01 03:10:43.145958: I tensorflow/stream_executor/cuda/cuda_blas.cc:1786] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.

157/157 [==============================] - 34s 84ms/step - loss: 3.2975 - accuracy: 0.1084 - top-5-accuracy: 0.4806 - val_loss: 2.1575 - val_accuracy: 0.2017 - val_top-5-accuracy: 0.7184
Epoch 2/100
157/157 [==============================] - 11s 67ms/step - loss: 2.1727 - accuracy: 0.2289 - top-5-accuracy: 0.7516 - val_loss: 1.8819 - val_accuracy: 0.3182 - val_top-5-accuracy: 0.8386
Epoch 3/100
157/157 [==============================] - 10s 67ms/step - loss: 1.8169 - accuracy: 0.3426 - top-5-accuracy: 0.8592 - val_loss: 1.6174 - val_accuracy: 0.4053 - val_top-5-accuracy: 0.8934
Epoch 4/100
157/157 [==============================] - 10s 67ms/step - loss: 1.6215 - accuracy: 0.4092 - top-5-accuracy: 0.8983 - val_loss: 1.4239 - val_accuracy: 0.4903 - val_top-5-accuracy: 0.9216
Epoch 5/100
157/157 [==============================] - 10s 66ms/step - loss: 1.5081 - accuracy: 0.4571 - top-5-accuracy: 0.9148 - val_loss: 1.3359 - val_accuracy: 0.5161 - val_top-5-accuracy: 0.9369
Epoch 6/100
157/157 [==============================] - 11s 68ms/step - loss: 1.4282 - accuracy: 0.4868 - top-5-accuracy: 0.9249 - val_loss: 1.2929 - val_accuracy: 0.5347 - val_top-5-accuracy: 0.9404
Epoch 7/100
157/157 [==============================] - 10s 66ms/step - loss: 1.3465 - accuracy: 0.5181 - top-5-accuracy: 0.9362 - val_loss: 1.2653 - val_accuracy: 0.5497 - val_top-5-accuracy: 0.9449
Epoch 8/100
157/157 [==============================] - 10s 67ms/step - loss: 1.2907 - accuracy: 0.5400 - top-5-accuracy: 0.9416 - val_loss: 1.1919 - val_accuracy: 0.5753 - val_top-5-accuracy: 0.9515
Epoch 9/100
157/157 [==============================] - 11s 67ms/step - loss: 1.2247 - accuracy: 0.5644 - top-5-accuracy: 0.9480 - val_loss: 1.1741 - val_accuracy: 0.5742 - val_top-5-accuracy: 0.9563
Epoch 10/100
157/157 [==============================] - 11s 67ms/step - loss: 1.1983 - accuracy: 0.5760 - top-5-accuracy: 0.9505 - val_loss: 1.4545 - val_accuracy: 0.4804 - val_top-5-accuracy: 0.9198
Epoch 11/100
157/157 [==============================] - 10s 66ms/step - loss: 1.2002 - accuracy: 0.5766 - top-5-accuracy: 0.9510 - val_loss: 1.1129 - val_accuracy: 0.6055 - val_top-5-accuracy: 0.9593
Epoch 12/100
157/157 [==============================] - 10s 66ms/step - loss: 1.1309 - accuracy: 0.5990 - top-5-accuracy: 0.9575 - val_loss: 1.0369 - val_accuracy: 0.6341 - val_top-5-accuracy: 0.9638
Epoch 13/100
157/157 [==============================] - 10s 66ms/step - loss: 1.0786 - accuracy: 0.6204 - top-5-accuracy: 0.9613 - val_loss: 1.0802 - val_accuracy: 0.6193 - val_top-5-accuracy: 0.9594
Epoch 14/100
157/157 [==============================] - 10s 65ms/step - loss: 1.0438 - accuracy: 0.6330 - top-5-accuracy: 0.9640 - val_loss: 0.9584 - val_accuracy: 0.6596 - val_top-5-accuracy: 0.9713
Epoch 15/100
157/157 [==============================] - 10s 66ms/step - loss: 0.9957 - accuracy: 0.6496 - top-5-accuracy: 0.9684 - val_loss: 0.9530 - val_accuracy: 0.6636 - val_top-5-accuracy: 0.9712
Epoch 16/100
157/157 [==============================] - 10s 66ms/step - loss: 0.9710 - accuracy: 0.6599 - top-5-accuracy: 0.9696 - val_loss: 0.8856 - val_accuracy: 0.6863 - val_top-5-accuracy: 0.9756
Epoch 17/100
157/157 [==============================] - 10s 66ms/step - loss: 0.9316 - accuracy: 0.6706 - top-5-accuracy: 0.9721 - val_loss: 0.9919 - val_accuracy: 0.6480 - val_top-5-accuracy: 0.9671
Epoch 18/100
157/157 [==============================] - 10s 66ms/step - loss: 0.8899 - accuracy: 0.6884 - top-5-accuracy: 0.9763 - val_loss: 0.8753 - val_accuracy: 0.6949 - val_top-5-accuracy: 0.9752
Epoch 19/100
157/157 [==============================] - 10s 64ms/step - loss: 0.8529 - accuracy: 0.6979 - top-5-accuracy: 0.9772 - val_loss: 0.8793 - val_accuracy: 0.6943 - val_top-5-accuracy: 0.9754
Epoch 20/100
157/157 [==============================] - 10s 66ms/step - loss: 0.8509 - accuracy: 0.7009 - top-5-accuracy: 0.9783 - val_loss: 0.8183 - val_accuracy: 0.7174 - val_top-5-accuracy: 0.9763
Epoch 21/100
157/157 [==============================] - 10s 66ms/step - loss: 0.8087 - accuracy: 0.7143 - top-5-accuracy: 0.9809 - val_loss: 0.7885 - val_accuracy: 0.7276 - val_top-5-accuracy: 0.9769
Epoch 22/100
157/157 [==============================] - 10s 66ms/step - loss: 0.8004 - accuracy: 0.7192 - top-5-accuracy: 0.9811 - val_loss: 0.7601 - val_accuracy: 0.7371 - val_top-5-accuracy: 0.9805
Epoch 23/100
157/157 [==============================] - 10s 66ms/step - loss: 0.7665 - accuracy: 0.7304 - top-5-accuracy: 0.9816 - val_loss: 0.7564 - val_accuracy: 0.7412 - val_top-5-accuracy: 0.9808
Epoch 24/100
157/157 [==============================] - 10s 66ms/step - loss: 0.7599 - accuracy: 0.7344 - top-5-accuracy: 0.9832 - val_loss: 0.7475 - val_accuracy: 0.7389 - val_top-5-accuracy: 0.9822
Epoch 25/100
157/157 [==============================] - 10s 66ms/step - loss: 0.7398 - accuracy: 0.7427 - top-5-accuracy: 0.9833 - val_loss: 0.7211 - val_accuracy: 0.7504 - val_top-5-accuracy: 0.9829
Epoch 26/100
157/157 [==============================] - 10s 66ms/step - loss: 0.7114 - accuracy: 0.7500 - top-5-accuracy: 0.9857 - val_loss: 0.7385 - val_accuracy: 0.7462 - val_top-5-accuracy: 0.9822
Epoch 27/100
157/157 [==============================] - 10s 66ms/step - loss: 0.6954 - accuracy: 0.7577 - top-5-accuracy: 0.9851 - val_loss: 0.7477 - val_accuracy: 0.7402 - val_top-5-accuracy: 0.9802
Epoch 28/100
157/157 [==============================] - 10s 66ms/step - loss: 0.6807 - accuracy: 0.7588 - top-5-accuracy: 0.9871 - val_loss: 0.7275 - val_accuracy: 0.7536 - val_top-5-accuracy: 0.9822
Epoch 29/100
157/157 [==============================] - 10s 66ms/step - loss: 0.6719 - accuracy: 0.7648 - top-5-accuracy: 0.9876 - val_loss: 0.7261 - val_accuracy: 0.7487 - val_top-5-accuracy: 0.9815
Epoch 30/100
157/157 [==============================] - 10s 65ms/step - loss: 0.6578 - accuracy: 0.7696 - top-5-accuracy: 0.9871 - val_loss: 0.6932 - val_accuracy: 0.7641 - val_top-5-accuracy: 0.9833
Epoch 31/100
157/157 [==============================] - 10s 66ms/step - loss: 0.6489 - accuracy: 0.7740 - top-5-accuracy: 0.9877 - val_loss: 0.7400 - val_accuracy: 0.7486 - val_top-5-accuracy: 0.9820
Epoch 32/100
157/157 [==============================] - 10s 65ms/step - loss: 0.6290 - accuracy: 0.7812 - top-5-accuracy: 0.9895 - val_loss: 0.6954 - val_accuracy: 0.7628 - val_top-5-accuracy: 0.9847
Epoch 33/100
157/157 [==============================] - 10s 67ms/step - loss: 0.6194 - accuracy: 0.7826 - top-5-accuracy: 0.9894 - val_loss: 0.6913 - val_accuracy: 0.7619 - val_top-5-accuracy: 0.9842
Epoch 34/100
157/157 [==============================] - 10s 65ms/step - loss: 0.5917 - accuracy: 0.7930 - top-5-accuracy: 0.9902 - val_loss: 0.6879 - val_accuracy: 0.7715 - val_top-5-accuracy: 0.9831
Epoch 35/100
157/157 [==============================] - 10s 66ms/step - loss: 0.5878 - accuracy: 0.7916 - top-5-accuracy: 0.9907 - val_loss: 0.6759 - val_accuracy: 0.7720 - val_top-5-accuracy: 0.9849
Epoch 36/100
157/157 [==============================] - 10s 66ms/step - loss: 0.5713 - accuracy: 0.8004 - top-5-accuracy: 0.9913 - val_loss: 0.6920 - val_accuracy: 0.7657 - val_top-5-accuracy: 0.9841
Epoch 37/100
157/157 [==============================] - 10s 66ms/step - loss: 0.5590 - accuracy: 0.8040 - top-5-accuracy: 0.9913 - val_loss: 0.6790 - val_accuracy: 0.7718 - val_top-5-accuracy: 0.9831
Epoch 38/100
157/157 [==============================] - 11s 67ms/step - loss: 0.5445 - accuracy: 0.8114 - top-5-accuracy: 0.9926 - val_loss: 0.6756 - val_accuracy: 0.7720 - val_top-5-accuracy: 0.9852
Epoch 39/100
157/157 [==============================] - 11s 67ms/step - loss: 0.5292 - accuracy: 0.8155 - top-5-accuracy: 0.9930 - val_loss: 0.6578 - val_accuracy: 0.7807 - val_top-5-accuracy: 0.9845
Epoch 40/100
157/157 [==============================] - 11s 68ms/step - loss: 0.5169 - accuracy: 0.8181 - top-5-accuracy: 0.9926 - val_loss: 0.6582 - val_accuracy: 0.7795 - val_top-5-accuracy: 0.9849
Epoch 41/100
157/157 [==============================] - 10s 66ms/step - loss: 0.5108 - accuracy: 0.8217 - top-5-accuracy: 0.9937 - val_loss: 0.6344 - val_accuracy: 0.7846 - val_top-5-accuracy: 0.9855
Epoch 42/100
157/157 [==============================] - 10s 65ms/step - loss: 0.5056 - accuracy: 0.8220 - top-5-accuracy: 0.9936 - val_loss: 0.6723 - val_accuracy: 0.7744 - val_top-5-accuracy: 0.9851
Epoch 43/100
157/157 [==============================] - 10s 66ms/step - loss: 0.4824 - accuracy: 0.8317 - top-5-accuracy: 0.9943 - val_loss: 0.6800 - val_accuracy: 0.7771 - val_top-5-accuracy: 0.9834
Epoch 44/100
157/157 [==============================] - 10s 67ms/step - loss: 0.4719 - accuracy: 0.8339 - top-5-accuracy: 0.9938 - val_loss: 0.6742 - val_accuracy: 0.7785 - val_top-5-accuracy: 0.9840
Epoch 45/100
157/157 [==============================] - 10s 65ms/step - loss: 0.4605 - accuracy: 0.8379 - top-5-accuracy: 0.9953 - val_loss: 0.6732 - val_accuracy: 0.7781 - val_top-5-accuracy: 0.9841
Epoch 46/100
157/157 [==============================] - 10s 66ms/step - loss: 0.4608 - accuracy: 0.8390 - top-5-accuracy: 0.9947 - val_loss: 0.6547 - val_accuracy: 0.7846 - val_top-5-accuracy: 0.9852
TESTING
40/40 [==============================] - 1s 22ms/step - loss: 0.6801 - accuracy: 0.7720 - top-5-accuracy: 0.9864
Loss: 0.68
Top 1 test accuracy: 77.20%
Top 5 test accuracy: 98.64%

```
</div>
---
## Conclusion

The most impactful contribution of the paper is not the novel architecture, but
the idea that hierarchical ViTs trained with no attention can perform quite well. This
opens up the question of how essential attention is to the performance of ViTs.

For curious minds, we would suggest reading the
[ConvNexT](https://arxiv.org/abs/2201.03545) paper which attends more to the training
paradigms and architectural details of ViTs rather than providing a novel architecture
based on attention.

Acknowledgements:

- We would like to thank [PyImageSearch](https://pyimagesearch.com) for providing us with
resources that helped in the completion of this project.
- We would like to thank [JarvisLabs.ai](https://jarvislabs.ai/) for providing with the
GPU credits.
- We would like to thank [Manim Community](https://www.manim.community/) for the manim
library.
- A personal note of thanks to [Puja Roychowdhury](https://twitter.com/pleb_talks) for
helping us with the Learning Rate Schedule.

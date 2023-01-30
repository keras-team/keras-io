"""
Title: Focal Modulation Networks
Author: [Aritra Roy Gosthipaty](https://twitter.com/ariG23498), [Ritwik Raha](https://twitter.com/ritwik_raha)
Date created: 2023/01/25
Last modified: 2023/01/25
Description: Image classification with Focal Modulation Networks.
"""
"""
## Introduction

This tutorial aims to provide a comprehensive guide to the implementation of
[Focal Modulation Networks](https://arxiv.org/abs/2203.11926), as
presented in the academic paper by Yang et. al. 

This tutorial will provide a formal, minimalistic approach to implementing Focal
Modulation Networks and explore its potential applications in the field of Deep Learning.

**The Problem Statement**

The architecture of Transformer ([Vaswani et. al](https://arxiv.org/abs/1706.03762)),
which has become the de facto standard in most Natural Language Processing tasks, has
also been applied to the field of computer vision through the seminal work of Vision
Transformers ([Dosovitskiy et. al](https://arxiv.org/abs/2010.11929v2)).

> In Transformers, the self-attention (SA) is arguably the key to its success which
enables input-dependent global interactions, in contrast to convolution operation which
constraints interactions in a local region with a shared kernel.

The **attention** module is mathematically written as

$$\text{Attention} = \text{softmax}(\frac{QK^{T}}{\sqrt{d_k}})V$$

where:
- $Q$ is the query
- $K$ is the key
- $V$ is the value
- $d_k$ is the dimension of the key

With **self-attention** the query, key, and value are all sourced from the input
sequence. Let's rewrite the attention equation for self-attention.

$$\text{Self Attention} = \text{softmax}(\frac{QQ^{T}}{\sqrt{d_q}})Q$$

Upon looking at the equation of self-attention we see that it is a quadratic equation.
This means that as number of tokens increase, so does the computation time (cost too). To
mitigate this problem, and also to make the Transformer more interpretable Yang et. al
have tried to replace the Self Attention module with better componenets.

**The Solution**

Yang et. al introduce the revolutionary Focal Modulation _Layer_, poised to serve as a
seamless replacement for the Self Attention Layer. The layer boasts high
interpretability, making it a valuable tool for Deep Learning practitioners.

In this tutorial, we will delve into the practical application of this layer by training
the entire model on the CIFAR-10 dataset and visually interpreting the layer's
performance.

Note: We have tried aligning our implementation with the
[official implementation](https://github.com/microsoft/FocalNet) as
much as we could.
"""

"""
## Setup and Imports

We use tensorflow version `2.11.0` for this tutorial.
"""

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.experimental import AdamW

from typing import Optional, Tuple, List
from matplotlib import pyplot as plt
from random import randint

# Set seed for reproducibility.
tf.keras.utils.set_random_seed(42)

"""
## Global Configuration

We do not have any strong ratioanle behind choosing these hyperparameters. Please feel
free to change the configuration and train the model.
"""

# DATA
TRAIN_SLICE = 40000
BUFFER_SIZE = 2048
BATCH_SIZE = 1024
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (32, 32, 3)
IMAGE_SIZE = 48
NUM_CLASSES = 10

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# TRAINING
EPOCHS = 50

"""
## Load and process the CIFAR-10 dataset
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
(x_train, y_train), (x_val, y_val) = (
    (x_train[:TRAIN_SLICE], y_train[:TRAIN_SLICE]),
    (x_train[TRAIN_SLICE:], y_train[TRAIN_SLICE:]),
)

"""
### Build the augmentations

We have used the `keras.Sequential` API to compose all the individual augmentation steps
into one API. 
"""

# Build the `train` augmentation pipeline.
train_aug = keras.Sequential(
    [
        layers.Rescaling(1 / 255.0),
        layers.Resizing(INPUT_SHAPE[0] + 20, INPUT_SHAPE[0] + 20),
        layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
        layers.RandomFlip("horizontal"),
    ],
    name="train_data_augmentation",
)

# Build the `val` and `test` data pipeline.
test_aug = keras.Sequential(
    [
        layers.Rescaling(1 / 255.0),
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    ],
    name="test_data_augmentation",
)

"""
### Build `tf.data` pipeline
"""


def train_map(image: tf.Tensor, label: tf.Tensor):
    """Applies `train_aug` transformation to an image.

    Args:
        image (tf.Tensor): Input image
        label (tf.Tensor): Input label

    Returns:
        Tuple of transformed image and original label
    """
    image = train_aug(image)
    return image, label


def test_map(image: tf.Tensor, label: tf.Tensor):
    """Applies `test_aug` transformation to an image.

    Args:
        image (tf.Tensor): Input image
        label (tf.Tensor): Input label

    Returns:
        Tuple of transformed image and original label
    """
    image = test_aug(image)
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = (
    train_ds.map(train_map, num_parallel_calls=AUTO)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.map(test_map, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = (
    test_ds.map(test_map, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
)

"""
## Architecture

We pause here to take a quick look at the Architecture of the Focal Modulation Network.
**Figure 1** shows how every individual layer is compiled into a single model. This gives
us a bird's eye view of the entire architecture.

| ![Diagram of the model](https://i.imgur.com/v5HYV5R.png) |
| :--: |
| Figure 1: A diagram of the Focal Modulation _Model_ (Source: Authors) | 

We dive deep into each of these layers in the following sections. This is the order we
will follow:


- Patch Embedding Layer
- Focal Modulation Block
  - Multi Layer Perceptron
  - Focal Modulation Layer
    - Hierarchical Contextualization
    - Gated Aggregation
  - Building Focal Modulation Block
- Building the Basic Layer

To better understand the architecture in a format we are well versed with, let us see how
the Focal Modulation Network would look when drawn like a Transformer architecture.

**Figure 2** shows the encoder layer of a traditional Transformer architecture where Self
Attention is replaced with the Focal Modulation _Layer_.

The <font color="blue">blue</font> blocks represent the Focal Modulation _Block_. A stack
of these blocks builds a single Basic Layer. The <font color="green">green</font> blocks
represent the Focal Modulation _Layer_.

| ![The Entire Architecture](https://i.imgur.com/PduYD6m.png) |
| :--: |
| Figure 2: The Entire Architecture (Source: Authors) |
"""

"""
## Patch Embedding Layer

The patch embedding layer is used to patchify the input images and project them into a
latent space. This layer is also used as the down-sampling layer in the architecture.
"""


class PatchEmbed(layers.Layer):
    """Image patch embedding layer, and also acts as the down-sampling layer.

    Args:
        image_size (Tuple[int]): Input image resolution.
        patch_size (Tuple[int]): Patch spatial resolution.
        embed_dim (int): Embedding dimnesion.
        norm_layer (tf.keras.layers.Layer): Normalization layer.
    """

    def __init__(
        self,
        image_size: Tuple[int] = (224, 224),
        patch_size: Tuple[int] = (4, 4),
        embed_dim: int = 96,
        norm_layer: tf.keras.layers.Layer = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        patch_resolution = [
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        ]
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution[0] * patch_resolution[1]
        self.proj = layers.Conv2D(
            filters=embed_dim, kernel_size=patch_size, strides=patch_size
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))
        self.norm = norm_layer(epsilon=1e-7) if norm_layer is not None else None

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, int, int, int]:
        """
        Args:
            x: Tensor of shape (B, H, W, C)

        Returns:
            A tuple of the processed tensor, height of the projected
            feature map, width of the projected feature map, number
            of channels of the projected feature map.
        """
        # Project the inputs.
        x = self.proj(x)

        # Obtain the shape from the projected tensor.
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]

        # B, H, W, C -> B, H*W, C
        x = self.flatten(x)
        if self.norm is not None:
            x = self.norm(x)

        return x, height, width, channels


"""
## Focal Modulation _Block_

A Focal Modulation _Block_ can be considered as a single Transformer Block with the Self
Attention (SA) module being replaced with Focal Modulation module as we saw in **Figure
2**.

Let us recall how a focal modulation block is supposed to look like with the aid of the
**Figure 3**.


| ![Focal Modulation Block](https://i.imgur.com/bPYTSiB.png) |
| :--: |
| Figure 3: The isolated view of the Focal Modulation Block (Source: Authors) |

The Focal Modulation Block consists of:
- Multilayer Perceptron
- Focal Modulation _Layer_
"""

"""
### Multilayer Perceptron
"""


def MLP(
    in_features: int,
    hidden_features: Optional[int] = None,
    out_features: Optional[int] = None,
    act_layer=keras.activations.gelu,
    mlp_drop_rate: float = 0.0,
):
    hidden_features = hidden_features or in_features
    out_features = out_features or in_features
    act_layer = act_layer
    mlp_drop_rate = mlp_drop_rate

    return keras.Sequential(
        [
            layers.Dense(units=hidden_features, activation=act_layer),
            layers.Dense(units=out_features),
            layers.Dropout(rate=mlp_drop_rate),
        ]
    )


"""
### Focal Modulation _Layer_

In a typical Transformer architecture, for each visual token (**query**) $x_{i} \in
\mathbb{R}^{C}$ in an input feature map $X \in \mathbb{R}^{H \times W \times C}$ a
**generic encoding process** produces a feature representation $y_{i} \in
\mathbb{R}^{C}$. 

The encoding process consists of **interaction** (with its surroundings for e.g a dot
product), and **aggregation** (over the contexts for e.g weighted mean).

We will talk about two types of encoding here:
- Interaction and then Aggregation in **Self Attention**
- Aggregation and then Interaction in **Focal Modulation**

**Self Attention**

| ![Self Attention Expression](https://i.imgur.com/heBYp0F.png) |
| :--: |
| **Figure 4**: Self Attention module. (Source: Authors) |

$$ y_{i} = \text{Aggregation} (\text{Interaction} (x_{i}, X), X) $$
$$ \text{Self Attention}  = \text{softmax} (\frac{QK}{\sqrt{d_{k}}}) V $$

As shown in **Figure 4** the query and the key interact (in the interaction step) with
each other to output the attention scores. The weighted aggregation of the value comes
next, known as the aggregation step.

**Focal Modulation**

| ![Focal Modulation module](https://i.imgur.com/tmbLgQl.png) |
| :--: |
| **Figure 5**: Focal Modulation module. (Source: Authors) | 

$$ y_{i} = \text{Interaction} (\text{Aggregation} (i, X), x_{i}) $$
$$ y_{i} = q(x_i) \odot m(i, X)$$

$$ y_{i} = q(x_i) \odot m(i, X)$$

**Figure 5** depicts the Focal Modulation _Layer_. $q()$ is the query projection
function. It is a **linear layer** that projects the query into a latent space. $m ()$ is
the context aggregation function. In focal modulation unlike self-attention the
aggregation step takes place before the interaction step.
"""

"""
While $q()$ is fairly straight forward to understand, the context aggregation function
$m()$ is a bit more complex. We will focus on $m()$ in this section.

| ![Context Aggregation](https://i.imgur.com/uqIRXI7.png)|
| :--: |
| **Figure 6**: Context Aggregation function $m()$. (Source: Authors) |

The context aggregation function $m()$ consists of two parts as shown in **Figure 6**:
- Hierarchical Contextualization
- Gated Aggregation
"""

"""
#### Hierarchical Contextualization

| ![Hierarchical Contextualization](https://i.imgur.com/q875c83.png)|
| :--: |
| **Figure 7**: Hierarchical Contextualization (Source: Authors) |

In **Figure 7** we see that the input is first projected linearly. This linear projection
produces $Z^{0}$. Where $Z^{0}$ can be expressed as follows:

$$Z^{0}=f_{z}(X) \in \mathbb{R}^{H \times W \times C}$$

$Z^{0}$ is then passed on to a series of Depth-Wise (DWConv) Conv and
[GeLU](https://www.tensorflow.org/api_docs/python/tf/keras/activations/gelu) layers. The
authors term each block of DWConv and GeLU as levels denoted by $l$. In **Figure 6** we
have 2 levels. Mathematically this is represented as:

$$Z^{l}=f_{a}^{l}(Z^{l-1}) \in \mathbb{R}^{H \times W \times C}$$

where $l \in \{1, \dots , L\}$

The final feature map goes through a Global Average Pooling Layer. This can be expressed
as follows:

$$Z^{L+1}=\text{AvgPool}(Z^{L}) \in \mathbb{R}^{C}$$
"""

"""
#### Gated Aggregation

| ![Gated Aggregation](https://i.imgur.com/LwrdDKo.png[/img)|
| :--: |
| **Figure 8**: Gated Aggregation (Source: Authors) |

Now that we have $L+1$ intermediate feature maps by virtue of the Hierarchical
Contextualization step, we need a gating mechanism that lets some features pass and
prohibits the others. This can be very simply implemented with the attention module.
Later in the tutorial, we will visualize these gates, to better understand their
usefulness.

First we build the weights for aggregation. Here we apply a **linear layer** on the input
feature map that projects it into $L+1$ dimensions.

$$G=f_{g}(X) \in \mathbb{R}^{H \times W \times (L+1)}$$ 

Next we perform the weighted aggregation over the contexts.

$$Z^{\text{out}}=\sum_{l=1}^{L+1}G^{l} \odot Z^{l} \in \mathbb{R}^{H \times W \times C}$$

To enable the communication across different channels we use another linear layer $h()$
to obtain the modulator

$$M = h(Z^{\text{out}}) \in \mathbb{R}^{H \times W \times C}$$

To sum up the Focal Modulation _Layer_ we have:
$$ y_{i} = q(x_i) \odot h(\sum_{l=1}^{L+1}g^{l}_{i} \odot z^{l}_{i})$$
"""


class FocalModulationLayer(layers.Layer):
    """The Focal Modulation module includes query projection and context aggregation.

    Args:
        dim (int): Projection dimension.
        focal_window (int): Window size for focal modulation.
        focal_level (int): The current focal level.
        focal_factor (int): Factor of focal modulation.
        proj_drop_rate (float): Rate of dropout.
    """

    def __init__(
        self,
        dim: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        proj_drop_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.proj_drop_rate = proj_drop_rate

        # Project the input feature into a new feature space using a
        # linear layer. Note the `units` used. We will be projecting the input
        # feature all at once and split the projection into query, context,
        # and gates.
        self.initial_proj = layers.Dense(
            units=(2 * self.dim) + (self.focal_level + 1),
            use_bias=True,
        )
        self.focal_layers = list()
        self.kernel_sizes = list()
        for idx in range(self.focal_level):
            kernel_size = (self.focal_factor * idx) + self.focal_window
            depth_gelu_block = keras.Sequential(
                [
                    layers.ZeroPadding2D(padding=(kernel_size // 2, kernel_size // 2)),
                    layers.Conv2D(
                        filters=self.dim,
                        kernel_size=kernel_size,
                        activation=keras.activations.gelu,
                        groups=self.dim,
                        use_bias=False,
                    ),
                ]
            )
            self.focal_layers.append(depth_gelu_block)
            self.kernel_sizes.append(kernel_size)
        self.activation = keras.activations.gelu
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.modulator_proj = layers.Conv2D(
            filters=self.dim,
            kernel_size=(1, 1),
            use_bias=True,
        )
        self.proj = layers.Dense(units=self.dim)
        self.proj_drop = layers.Dropout(self.proj_drop_rate)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Args:
            x: Tensor of shape (B, H, W, C)
        """
        # Apply the linear projecion to the input feature map
        x_proj = self.initial_proj(x)

        # Split the projected x into query, context and gates
        query, context, self.gates = tf.split(
            value=x_proj,
            num_or_size_splits=[self.dim, self.dim, self.focal_level + 1],
            axis=-1,
        )

        # Context aggregation
        context = self.focal_layers[0](context)
        context_all = context * self.gates[..., 0:1]
        for idx in range(1, self.focal_level):
            context = self.focal_layers[idx](context)
            context_all += context * self.gates[..., idx : idx + 1]

        # Build the global context
        context_global = self.activation(self.gap(context))
        context_all += context_global * self.gates[..., self.focal_level :]

        # Focal Modulation
        self.modulator = self.modulator_proj(context_all)
        x_output = query * self.modulator

        # Project the output and apply dropout
        x_output = self.proj(x_output)
        x_output = self.proj_drop(x_output)

        return x_output


"""
### Building Focal Modulation _Block_

Finally we have all the components we need to build the Focal Modulation block. Here we
take the MLP and Focal Modulaion _Layer_ together and build the Focal Modulation _Block_.
"""


class DropPath(layers.Layer):
    """Drop Path also known as the Stochastic Depth layer.

    Refernece:
        - https://keras.io/examples/vision/cct/#stochastic-depth-for-regularization
        - github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_prob = drop_path_prob

    def call(self, x: tf.Tensor, training: bool = False):
        if training:
            keep_prob = 1 - self.drop_path_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class FocalModulationBlock(layers.Layer):
    """A block of FFN and Focal Modulation Layer.

    Args:
        dim (int): Number of input channels.
        input_resolution (Tuple[int]): Input resulotion.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float): Dropout rate.
        drop_path (float): Stochastic depth rate.
        act_layer (tf.keras.activations.Activation): Activation layer.
        norm_layer (tf.keras.layers.Layer): Normalization layer.
        focal_level (int): Number of focal levels.
        focal_window (int): Focal window size at first focal level
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int],
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=keras.activations.gelu,
        norm_layer=layers.LayerNormalization,
        focal_level: int = 1,
        focal_window: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.norm1 = norm_layer(epsilon=1e-5)
        self.modulation = FocalModulationLayer(
            dim=self.dim,
            focal_window=self.focal_window,
            focal_level=self.focal_level,
            proj_drop_rate=drop,
        )
        self.drop_path = (
            DropPath(drop_path_prob=drop_path)
            if drop_path > 0.0
            else layers.Activation("linear")
        )
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            mlp_drop_rate=drop,
        )

    def call(self, x: tf.Tensor, height: int, width: int, channels: int) -> tf.Tensor:
        """
        Args:
            x (tf.Tensor): Inputs of the shape (B, L, C)
            height (int): The height of the feature map
            width (int): The width of the feature map
            channels (int): The number of channels of the feature map

        Returns:
            The processed tensor.
        """
        shortcut = x

        # Focal Modulation
        x = tf.reshape(x, shape=(-1, height, width, channels))
        x = self.modulation(x)
        x = tf.reshape(x, shape=(-1, height * width, channels))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm1(x)))
        return x


"""
## Building the Basic Layer

The basic layer consists of a collection of Focal Modulation _Blocks_. This is
illustrated in **Figure 9**.

| ![Basic Layer](https://i.imgur.com/UcZV0K6.png) |
| :--: |
| **Figure 9**: Basic Layer, a collection of focal modulation blocks. (Source: Authors) | 

Notice how in **Fig. 9** there are more than one focal modulation blocks denoted by
$N\times$. This shows how the Basic Layer is a collection of Focal Modulation _Blocks_.
"""


class BasicLayer(layers.Layer):
    """A collection of Focal Modulation Blocks.

    Args:
        dim (int): Dimensions of the model.
        out_dim (int): Dimension used by the Patch Embedding Layer.
        input_resolution (Tuple[int]): Input image resolution.
        depth (int): The number of Focal Modulation Blocks.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float): Dropout rate.
        drop_path (float): Droppath rate.
        norm_layer (tf.keras.layers.Layer): The normalizaiotn used.
        downsample (tf.keras.layers.Layer): Downsampling layer at the end of the layer.
        focal_level (int): The current focal level.
        focal_window (int): Focal window used.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        input_resolution: Tuple[int],
        depth: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=layers.LayerNormalization,
        downsample=None,
        focal_level: int = 1,
        focal_window: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = [
            FocalModulationBlock(
                dim=dim,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                focal_level=focal_level,
                focal_window=focal_window,
            )
            for i in range(self.depth)
        ]

        # Downsample layer at the end of the layer
        if downsample is not None:
            self.downsample = downsample(
                image_size=input_resolution,
                patch_size=(2, 2),
                embed_dim=out_dim,
            )
        else:
            self.downsample = None

    def call(
        self, x: tf.Tensor, height: int, width: int, channels: int
    ) -> Tuple[tf.Tensor, int, int, int]:
        """
        Args:
            x (tf.Tensor): Tensor of shape (B, L, C)
            height (int): Height of feature map
            width (int): Width of feature map
            channels (int): Embed Dim of feature map

        Returns:
            A tuple of the processed tensor, changed height, width, and
            dim of the tensor.
        """
        # print(f"Basic Layer {x.shape}")
        # Apply Focal Modulation Blocks
        for block in self.blocks:
            x = block(x, height, width, channels)

        # Except the last Basic Layer, all the layers have
        # downsample at the end of it.
        if self.downsample is not None:
            x = tf.reshape(x, shape=(-1, height, width, channels))
            x, height_o, width_o, channels_o = self.downsample(x)
        else:
            height_o, width_o, channels_o = height, width, channels

        return x, height_o, width_o, channels_o


"""
## Focal Modulation Network _Model_

This is where everything is tied together. The Focal Modulation _Model_ consists of a
collection of Basic Layers with a classification head. For a recap of how this is
structured refer to **Figure 1**.


"""


class FocalModulationNetwork(keras.Model):
    """The Focal Modulation Network.

    Parameters:
        image_size (Tuple[int]): Spatial size of images used.
        patch_size (Tuple[int]): Patch size of each patches.
        num_classes (int): Number of classes used for classification.
        embed_dim (int): Patch embedding dimension.
        depths (List[int]): Depth of each Focal Transformer block.
        mlp_ratio (float): Ratio of expansion for the intermediate layer of MLP.
        drop_rate (float): The dropout rate for FM and MLP layers.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (tf.keras.layers.Layer): Normalization layer.
        patch_norm (bool): If True, add normalization after patch embedding.
        focal_levels (list): How many focal levels at all stages.
            Note that this excludes the finest-grain level.
        focal_windows (list): The focal window size at all stages.
    """

    def __init__(
        self,
        image_size: Tuple[int] = (48, 48),
        patch_size: Tuple[int] = (4, 4),
        num_classes: int = 10,
        embed_dim: int = 256,
        depths: List[int] = [2, 3, 2],
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_layer=layers.LayerNormalization,
        focal_levels=[2, 2, 2],
        focal_windows=[3, 3, 3],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = len(depths)

        # Embedding dimension doubles every stage of the model.
        embed_dim = [embed_dim * (2**i) for i in range(self.num_layers)]

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patch_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = layers.Dropout(drop_rate)

        # Calculate stochastic depth probabilities.
        dpr = [x for x in np.linspace(start=0, stop=drop_path_rate, num=sum(depths))]

        self.basic_layers = list()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim[i_layer],
                out_dim=embed_dim[i_layer + 1]
                if (i_layer < self.num_layers - 1)
                else None,
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
            )
            self.basic_layers.append(layer)

        self.norm = norm_layer(epsilon=1e-7)
        self.avgpool = layers.GlobalAveragePooling1D()
        self.flatten = layers.Flatten()
        self.head = layers.Dense(self.num_classes, activation="softmax")

    def forward_features(self, x: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x: Tensor of shape (B, H, W, C)

        Returns:
            The processed tensor.
        """
        # Patch Embed the input images.
        x, height, width, channels = self.patch_embed(x)
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.basic_layers):
            x, height, width, channels = layer(x, height, width, channels)

        x = self.norm(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        return x

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x: Tensor of shape (B, H, W, C)

        Returns:
            The logits.
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x


"""
## Train the model

Now with all the components in place and the architecture actually built, we are ready to
put it to good use.

In this section, we train our Focal Modulation _Model_ on the CIFAR-10 dataset.
"""

"""
### Visualization Callback
A key feature of the Focal Modulation Network is Explicit input-dependency. This means
the modulator is calculated by looking at the local features around the target location,
so it depends on the input. In very simple terms, this makes interpretaion easy. We can
simply lay down the gating values and the original image, next to each other to see how
the gating mechanism works.

The authors of the paper visualize the gates and the modulator in order to focus on the
interpretability of the Focal Modulation _Layer_. We have devised a visualization
callback that shows the gates and modulator of a specific layer in the model while the
model trains.

We will notice later that as the model trains, the visualizations get better. 

The gates appear to selectively permit certain aspects of the input image to pass
through, while gently disregarding others, ultimately leading to improved classification
accuracy.
"""


def display_grid(
    test_images: tf.Tensor,
    gates: tf.Tensor,
    modulator: tf.Tensor,
):
    """
    Args:
        test_images (tf.Tensor): A batch of test images.
        gates (tf.Tensor): The gates of the Focal Modualtion Layer.
        modulator (tf.Tensor): The modulator of the Focal Modulation Layer.
    """
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))

    # Radomly sample an image from the batch.
    index = randint(0, BATCH_SIZE - 1)
    orig_image = test_images[index]
    gate_image = gates[index]
    modulator_image = modulator[index]

    # Original Image
    ax[0].imshow(orig_image)
    ax[0].set_title("Original:")
    ax[0].axis("off")

    # Gate 1
    img = ax[1].imshow(orig_image)
    ax[1].imshow(gate_image[..., 0], cmap="inferno", alpha=0.6, extent=img.get_extent())
    ax[1].set_title("G 1:")
    ax[1].axis("off")

    # Gate 2
    img = ax[2].imshow(orig_image)
    ax[2].imshow(gate_image[..., 1], cmap="inferno", alpha=0.6, extent=img.get_extent())
    ax[2].set_title("G 2:")
    ax[2].axis("off")

    # Gate 3
    img = ax[3].imshow(orig_image)
    ax[3].imshow(gate_image[..., 2], cmap="inferno", alpha=0.6, extent=img.get_extent())
    ax[3].set_title("G 3:")
    ax[3].axis("off")

    # Gate 4
    img = ax[4].imshow(orig_image)
    ax[4].imshow(
        tf.math.reduce_mean(modulator_image, axis=-1),
        cmap="inferno",
        alpha=0.6,
        extent=img.get_extent(),
    )
    ax[4].set_title("MOD:")
    ax[4].axis("off")

    plt.axis("off")
    plt.show()
    plt.close()


"""
### TrainMonitor
"""

# Taking a batch of test inputs to measure model's progress.
test_images, test_labels = next(iter(test_ds))
upsampler = tf.keras.layers.UpSampling2D(
    size=(4, 4),
    interpolation="bilinear",
)


class TrainMonitor(keras.callbacks.Callback):
    def __init__(self, epoch_interval=None):
        self.epoch_interval = epoch_interval

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 0:
            _ = self.model(test_images)

            # Take the mid layer for visualization
            gates = self.model.basic_layers[1].blocks[-1].modulation.gates
            gates = upsampler(gates)
            modulator = self.model.basic_layers[1].blocks[-1].modulation.modulator
            modulator = upsampler(modulator)

            # Display the grid of gates and modulator.
            display_grid(test_images=test_images, gates=gates, modulator=modulator)


callbacks = [
    TrainMonitor(epoch_interval=10),
]

"""
### Learning Rate scheduler
"""

# Some code is taken from:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)
        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


total_steps = int((len(x_train) / BATCH_SIZE) * EPOCHS)
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=warmup_steps,
)

"""
### Initialize, compile and train the model
"""

focal_mod_net = FocalModulationNetwork()
optimizer = AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)

# Compile and train the model.
focal_mod_net.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = focal_mod_net.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)

"""
## Plot loss and accuracy
"""

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.show()

"""
## Test Visulaizations

Let's test our model on some test images and see how the gates look like.
"""

test_images, test_labels = next(iter(test_ds))
_ = focal_mod_net(test_images)

# Take the mid layer for visualization
gates = focal_mod_net.basic_layers[1].blocks[-1].modulation.gates
gates = upsampler(gates)
modulator = focal_mod_net.basic_layers[1].blocks[-1].modulation.modulator
modulator = upsampler(modulator)

# Plot the test images with the gates and modulator overlayed.
for row in range(5):
    display_grid(
        test_images=test_images,
        gates=gates,
        modulator=modulator,
    )

"""
## Conclusion

The proposed architecture, Focal Modulation Network is a mechanism that allows different
parts of an image to interact with each other in a way that depends on the image itself.
It works by first gathering different levels of context information around each part of
the image (the "query token"), then using a gate to decide which context information is
most relevant, and finally combining the chosen information in a simple but effective
way.

This is meant as a replacement of Self Attention mechanism from the Transformer
architecture. The key feature that makes this research notable is not the conception of
attention-less networks, but rather the introduction of a equally powerful architecture
that is interpretable.

The authors also mention that they created a series of Focal Modulation Networks
(FocalNets) that significantly outperform Self Attention counterparts and with a fraction
of parameters and pretraining data.

The FocalNets architecture has the potential to deliver impressive results and offers a
simple implementation. Its promising performance and ease of use make it an attractive
alternative to Self Attention for researchers to explore in their own projects. It could
potentially become widely adopted by the Deep Learning community in the near future.


"""

"""
## Acknowledgement
We would like to thank [PyImageSearch](https://pyimagesearch.com/) for providing with a
Colab Pro account and also Microsoft Research for providing an [official
implementation](https://github.com/microsoft/FocalNet) of their paper.
"""

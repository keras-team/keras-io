# Video Vision Transformer

**Author:** [Aritra Roy Gosthipaty](https://twitter.com/ariG23498), [Ayush Thakur](https://twitter.com/ayushthakur0) (equal contribution)<br>
**Date created:** 2022/01/12<br>
**Last modified:**  2024/01/15<br>
**Description:** A Transformer-based architecture for video classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/vivit.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/vivit.py)



---
## Introduction

Videos are sequences of images. Let's assume you have an image
representation model (CNN, ViT, etc.) and a sequence model
(RNN, LSTM, etc.) at hand. We ask you to tweak the model for video
classification. The simplest approach would be to apply the image
model to individual frames, use the sequence model to learn
sequences of image features, then apply a classification head on
the learned sequence representation.
The Keras example
[Video Classification with a CNN-RNN Architecture](https://keras.io/examples/vision/video_classification/)
explains this approach in detail. Alernatively, you can also
build a hybrid Transformer-based model for video classification as shown in the Keras example
[Video Classification with Transformers](https://keras.io/examples/vision/video_transformers/).

In this example, we minimally implement
[ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
by Arnab et al., a **pure Transformer-based** model
for video classification. The authors propose a novel embedding scheme
and a number of Transformer variants to model video clips. We implement
the embedding scheme and one of the variants of the Transformer
architecture, for simplicity.

This example requires  `medmnist` package, which can be installed
by running the code cell below.


```python
!pip install -qq medmnist
```

---
## Imports


```python
import os
import io
import imageio
import medmnist
import ipywidgets
import numpy as np
import tensorflow as tf  # for data preprocessing only
import keras
from keras import layers, ops

# Setting seed for reproducibility
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)
```

---
## Hyperparameters

The hyperparameters are chosen via hyperparameter
search. You can learn more about the process in the "conclusion" section.


```python
# DATA
DATASET_NAME = "organmnist3d"
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (28, 28, 28, 1)
NUM_CLASSES = 11

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 60

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8
```

---
## Dataset

For our example we use the
[MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification](https://medmnist.com/)
dataset. The videos are lightweight and easy to train on.


```python

def download_and_prepare_dataset(data_info: dict):
    """Utility function to download the dataset.

    Arguments:
        data_info (dict): Dataset metadata.
    """
    data_path = keras.utils.get_file(origin=data_info["url"], md5_hash=data_info["MD5"])

    with np.load(data_path) as data:
        # Get videos
        train_videos = data["train_images"]
        valid_videos = data["val_images"]
        test_videos = data["test_images"]

        # Get labels
        train_labels = data["train_labels"].flatten()
        valid_labels = data["val_labels"].flatten()
        test_labels = data["test_labels"].flatten()

    return (
        (train_videos, train_labels),
        (valid_videos, valid_labels),
        (test_videos, test_labels),
    )


# Get the metadata of the dataset
info = medmnist.INFO[DATASET_NAME]

# Get the dataset
prepared_dataset = download_and_prepare_dataset(info)
(train_videos, train_labels) = prepared_dataset[0]
(valid_videos, valid_labels) = prepared_dataset[1]
(test_videos, test_labels) = prepared_dataset[2]
```

### `tf.data` pipeline


```python

def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader


trainloader = prepare_dataloader(train_videos, train_labels, "train")
validloader = prepare_dataloader(valid_videos, valid_labels, "valid")
testloader = prepare_dataloader(test_videos, test_labels, "test")
```

---
## Tubelet Embedding

In ViTs, an image is divided into patches, which are then spatially
flattened, a process known as tokenization. For a video, one can
repeat this process for individual frames. **Uniform frame sampling**
as suggested by the authors is a tokenization scheme in which we
sample frames from the video clip and perform simple ViT tokenization.

| ![uniform frame sampling](https://i.imgur.com/aaPyLPX.png) |
| :--: |
| Uniform Frame Sampling [Source](https://arxiv.org/abs/2103.15691) |

**Tubelet Embedding** is different in terms of capturing temporal
information from the video.
First, we extract volumes from the video -- these volumes contain
patches of the frame and the temporal information as well. The volumes
are then flattened to build video tokens.

| ![tubelet embedding](https://i.imgur.com/9G7QTfV.png) |
| :--: |
| Tubelet Embedding [Source](https://arxiv.org/abs/2103.15691) |


```python

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

```

---
## Positional Embedding

This layer adds positional information to the encoded video tokens.


```python

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = ops.arange(0, num_tokens, 1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

```

---
## Video Vision Transformer

The authors suggest 4 variants of Vision Transformer:

- Spatio-temporal attention
- Factorized encoder
- Factorized self-attention
- Factorized dot-product attention

In this example, we will implement the **Spatio-temporal attention**
model for simplicity. The following code snippet is heavily inspired from
[Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/).
One can also refer to the
[official repository of ViViT](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit)
which contains all the variants, implemented in JAX.


```python

def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=ops.gelu),
                layers.Dense(units=embed_dim, activation=ops.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

---
## Train


```python

def run_experiment():
    # Initialize model
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    # Train the model.
    _ = model.fit(trainloader, epochs=EPOCHS, validation_data=validloader)

    _, accuracy, top_5_accuracy = model.evaluate(testloader)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return model


model = run_experiment()
```

<div class="k-default-codeblock">
```
Epoch 1/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  28:17 57s/step - accuracy: 0.0312 - loss: 2.6952 - top-5-accuracy: 0.5938

<div class="k-default-codeblock">
```

```
</div>
  3/31 ━[37m━━━━━━━━━━━━━━━━━━━  1s 36ms/step - accuracy: 0.0694 - loss: 2.8059 - top-5-accuracy: 0.5521  

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 0.0760 - loss: 2.7842 - top-5-accuracy: 0.5506

<div class="k-default-codeblock">
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1705656639.661106    3979 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
W0000 00:00:1705656639.709734    3979 graph_launch.cc:671] Fallback to op-by-op mode because memset node breaks graph update


```
</div>
  7/31 ━━━━[37m━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.0823 - loss: 2.7616 - top-5-accuracy: 0.5409

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 29ms/step - accuracy: 0.0885 - loss: 2.7333 - top-5-accuracy: 0.5338

<div class="k-default-codeblock">
```

```
</div>
 12/31 ━━━━━━━[37m━━━━━━━━━━━━━  0s 27ms/step - accuracy: 0.0924 - loss: 2.6932 - top-5-accuracy: 0.5361

<div class="k-default-codeblock">
```

```
</div>
 15/31 ━━━━━━━━━[37m━━━━━━━━━━━  0s 25ms/step - accuracy: 0.0946 - loss: 2.6574 - top-5-accuracy: 0.5422

<div class="k-default-codeblock">
```

```
</div>
 19/31 ━━━━━━━━━━━━[37m━━━━━━━━  0s 23ms/step - accuracy: 0.0975 - loss: 2.6209 - top-5-accuracy: 0.5478

<div class="k-default-codeblock">
```

```
</div>
 23/31 ━━━━━━━━━━━━━━[37m━━━━━━  0s 22ms/step - accuracy: 0.1006 - loss: 2.5938 - top-5-accuracy: 0.5524

<div class="k-default-codeblock">
```

```
</div>
 27/31 ━━━━━━━━━━━━━━━━━[37m━━━  0s 21ms/step - accuracy: 0.1031 - loss: 2.5713 - top-5-accuracy: 0.5575

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 785ms/step - accuracy: 0.1054 - loss: 2.5517 - top-5-accuracy: 0.5626

<div class="k-default-codeblock">
```
W0000 00:00:1705656663.257172    3976 graph_launch.cc:671] Fallback to op-by-op mode because memset node breaks graph update

W0000 00:00:1705656667.427646    3976 graph_launch.cc:671] Fallback to op-by-op mode because memset node breaks graph update


```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 88s 1s/step - accuracy: 0.1059 - loss: 2.5471 - top-5-accuracy: 0.5638 - val_accuracy: 0.1491 - val_loss: 2.1911 - val_top-5-accuracy: 0.6522


<div class="k-default-codeblock">
```
Epoch 2/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  1s 45ms/step - accuracy: 0.2188 - loss: 2.3489 - top-5-accuracy: 0.5938

<div class="k-default-codeblock">
```

```
</div>
  3/31 ━[37m━━━━━━━━━━━━━━━━━━━  0s 28ms/step - accuracy: 0.1944 - loss: 2.3271 - top-5-accuracy: 0.6198

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 28ms/step - accuracy: 0.1889 - loss: 2.2936 - top-5-accuracy: 0.6372

<div class="k-default-codeblock">
```

```
</div>
  7/31 ━━━━[37m━━━━━━━━━━━━━━━━  0s 28ms/step - accuracy: 0.1878 - loss: 2.2699 - top-5-accuracy: 0.6420

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 27ms/step - accuracy: 0.1878 - loss: 2.2520 - top-5-accuracy: 0.6463

<div class="k-default-codeblock">
```

```
</div>
 12/31 ━━━━━━━[37m━━━━━━━━━━━━━  0s 25ms/step - accuracy: 0.1900 - loss: 2.2315 - top-5-accuracy: 0.6547

<div class="k-default-codeblock">
```

```
</div>
 15/31 ━━━━━━━━━[37m━━━━━━━━━━━  0s 24ms/step - accuracy: 0.1904 - loss: 2.2204 - top-5-accuracy: 0.6614

<div class="k-default-codeblock">
```

```
</div>
 18/31 ━━━━━━━━━━━[37m━━━━━━━━━  0s 23ms/step - accuracy: 0.1889 - loss: 2.2105 - top-5-accuracy: 0.6686

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 22ms/step - accuracy: 0.1875 - loss: 2.2013 - top-5-accuracy: 0.6748

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 21ms/step - accuracy: 0.1869 - loss: 2.1923 - top-5-accuracy: 0.6823

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.1871 - loss: 2.1840 - top-5-accuracy: 0.6884

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - accuracy: 0.1875 - loss: 2.1773 - top-5-accuracy: 0.6929 - val_accuracy: 0.3602 - val_loss: 1.9765 - val_top-5-accuracy: 0.7764


<div class="k-default-codeblock">
```
Epoch 3/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.3438 - loss: 2.0973 - top-5-accuracy: 0.7812

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.3096 - loss: 2.1054 - top-5-accuracy: 0.7740

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.2950 - loss: 2.0784 - top-5-accuracy: 0.7702

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.2913 - loss: 2.0583 - top-5-accuracy: 0.7701

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.2873 - loss: 2.0453 - top-5-accuracy: 0.7721

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.2813 - loss: 2.0366 - top-5-accuracy: 0.7736

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.2761 - loss: 2.0296 - top-5-accuracy: 0.7760

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.2734 - loss: 2.0208 - top-5-accuracy: 0.7790

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.2715 - loss: 2.0148 - top-5-accuracy: 0.7811 - val_accuracy: 0.2981 - val_loss: 1.7838 - val_top-5-accuracy: 0.8820


<div class="k-default-codeblock">
```
Epoch 4/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.2500 - loss: 2.2085 - top-5-accuracy: 0.7500

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.2669 - loss: 2.0202 - top-5-accuracy: 0.7763

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.2827 - loss: 1.9539 - top-5-accuracy: 0.7837

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.2888 - loss: 1.9233 - top-5-accuracy: 0.7930

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.2880 - loss: 1.9074 - top-5-accuracy: 0.8011

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.2870 - loss: 1.8980 - top-5-accuracy: 0.8060

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.2862 - loss: 1.8916 - top-5-accuracy: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.2870 - loss: 1.8851 - top-5-accuracy: 0.8151

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.2874 - loss: 1.8802 - top-5-accuracy: 0.8179 - val_accuracy: 0.3168 - val_loss: 1.6237 - val_top-5-accuracy: 0.9503


<div class="k-default-codeblock">
```
Epoch 5/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.2188 - loss: 1.8940 - top-5-accuracy: 0.8125

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.2865 - loss: 1.7909 - top-5-accuracy: 0.8302

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.3148 - loss: 1.7422 - top-5-accuracy: 0.8377

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.3341 - loss: 1.7154 - top-5-accuracy: 0.8479

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.3461 - loss: 1.7055 - top-5-accuracy: 0.8546

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.3515 - loss: 1.6979 - top-5-accuracy: 0.8604

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.3544 - loss: 1.6918 - top-5-accuracy: 0.8655

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.3565 - loss: 1.6849 - top-5-accuracy: 0.8701

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.3582 - loss: 1.6787 - top-5-accuracy: 0.8732 - val_accuracy: 0.4534 - val_loss: 1.4373 - val_top-5-accuracy: 0.9379


<div class="k-default-codeblock">
```
Epoch 6/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.3750 - loss: 1.4807 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.4018 - loss: 1.5246 - top-5-accuracy: 0.9315

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.4141 - loss: 1.5079 - top-5-accuracy: 0.9223

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.4188 - loss: 1.5011 - top-5-accuracy: 0.9180

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.4232 - loss: 1.4989 - top-5-accuracy: 0.9161

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.4246 - loss: 1.4971 - top-5-accuracy: 0.9155

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.4243 - loss: 1.4946 - top-5-accuracy: 0.9162

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.4240 - loss: 1.4903 - top-5-accuracy: 0.9175

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.4244 - loss: 1.4864 - top-5-accuracy: 0.9186 - val_accuracy: 0.5217 - val_loss: 1.2086 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 7/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.4688 - loss: 1.2712 - top-5-accuracy: 0.9688

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5064 - loss: 1.2803 - top-5-accuracy: 0.9631

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5017 - loss: 1.3105 - top-5-accuracy: 0.9534

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.4981 - loss: 1.3292 - top-5-accuracy: 0.9458

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.4965 - loss: 1.3399 - top-5-accuracy: 0.9410

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.4937 - loss: 1.3456 - top-5-accuracy: 0.9393

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.4915 - loss: 1.3485 - top-5-accuracy: 0.9386

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.4904 - loss: 1.3491 - top-5-accuracy: 0.9390

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.4907 - loss: 1.3469 - top-5-accuracy: 0.9396 - val_accuracy: 0.5466 - val_loss: 1.1944 - val_top-5-accuracy: 0.9627


<div class="k-default-codeblock">
```
Epoch 8/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.5312 - loss: 1.5005 - top-5-accuracy: 0.8438

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.5279 - loss: 1.3889 - top-5-accuracy: 0.8952

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5161 - loss: 1.3697 - top-5-accuracy: 0.9118

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5090 - loss: 1.3693 - top-5-accuracy: 0.9159

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.5017 - loss: 1.3743 - top-5-accuracy: 0.9184

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.4960 - loss: 1.3750 - top-5-accuracy: 0.9216

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.4912 - loss: 1.3745 - top-5-accuracy: 0.9241

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.4891 - loss: 1.3712 - top-5-accuracy: 0.9264

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.4878 - loss: 1.3683 - top-5-accuracy: 0.9281 - val_accuracy: 0.5590 - val_loss: 1.0001 - val_top-5-accuracy: 0.9752


<div class="k-default-codeblock">
```
Epoch 9/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.5625 - loss: 1.2456 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5770 - loss: 1.1445 - top-5-accuracy: 0.9697

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5705 - loss: 1.1367 - top-5-accuracy: 0.9582

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.5675 - loss: 1.1390 - top-5-accuracy: 0.9534

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.5657 - loss: 1.1414 - top-5-accuracy: 0.9523

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.5640 - loss: 1.1432 - top-5-accuracy: 0.9523

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.5622 - loss: 1.1442 - top-5-accuracy: 0.9530

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.5612 - loss: 1.1439 - top-5-accuracy: 0.9538

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.5609 - loss: 1.1426 - top-5-accuracy: 0.9543 - val_accuracy: 0.6770 - val_loss: 0.9213 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 10/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.5000 - loss: 1.0556 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.5709 - loss: 0.9560 - top-5-accuracy: 0.9882

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.5784 - loss: 0.9739 - top-5-accuracy: 0.9831

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.5818 - loss: 0.9985 - top-5-accuracy: 0.9780

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.5842 - loss: 1.0106 - top-5-accuracy: 0.9758

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.5822 - loss: 1.0218 - top-5-accuracy: 0.9744

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.5799 - loss: 1.0324 - top-5-accuracy: 0.9726

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.5779 - loss: 1.0419 - top-5-accuracy: 0.9713

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.5774 - loss: 1.0457 - top-5-accuracy: 0.9707 - val_accuracy: 0.5901 - val_loss: 1.1189 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 11/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.6250 - loss: 1.0098 - top-5-accuracy: 0.9688

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6008 - loss: 1.0137 - top-5-accuracy: 0.9736

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6008 - loss: 1.0072 - top-5-accuracy: 0.9720

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5983 - loss: 1.0092 - top-5-accuracy: 0.9701

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.5971 - loss: 1.0107 - top-5-accuracy: 0.9687

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.5960 - loss: 1.0130 - top-5-accuracy: 0.9677

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.5946 - loss: 1.0162 - top-5-accuracy: 0.9673

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.5943 - loss: 1.0184 - top-5-accuracy: 0.9674

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.5949 - loss: 1.0185 - top-5-accuracy: 0.9676 - val_accuracy: 0.5901 - val_loss: 1.0774 - val_top-5-accuracy: 0.9689


<div class="k-default-codeblock">
```
Epoch 12/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.6562 - loss: 0.9226 - top-5-accuracy: 0.9688

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6006 - loss: 0.9123 - top-5-accuracy: 0.9845

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5931 - loss: 0.9213 - top-5-accuracy: 0.9834

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5923 - loss: 0.9378 - top-5-accuracy: 0.9820

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.5943 - loss: 0.9423 - top-5-accuracy: 0.9805

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.5964 - loss: 0.9475 - top-5-accuracy: 0.9785

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.5983 - loss: 0.9519 - top-5-accuracy: 0.9769

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.6014 - loss: 0.9530 - top-5-accuracy: 0.9760

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.6035 - loss: 0.9537 - top-5-accuracy: 0.9756 - val_accuracy: 0.6646 - val_loss: 0.9072 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 13/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.7188 - loss: 0.7718 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6473 - loss: 0.8973 - top-5-accuracy: 0.9710

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6367 - loss: 0.9401 - top-5-accuracy: 0.9706

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6318 - loss: 0.9676 - top-5-accuracy: 0.9694

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.6298 - loss: 0.9770 - top-5-accuracy: 0.9689

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.6280 - loss: 0.9806 - top-5-accuracy: 0.9687

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.6269 - loss: 0.9820 - top-5-accuracy: 0.9687

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.6265 - loss: 0.9816 - top-5-accuracy: 0.9690

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.6267 - loss: 0.9797 - top-5-accuracy: 0.9693 - val_accuracy: 0.7888 - val_loss: 0.6304 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 14/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.6250 - loss: 0.8279 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6673 - loss: 0.8145 - top-5-accuracy: 0.9920

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6740 - loss: 0.8100 - top-5-accuracy: 0.9924

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.6686 - loss: 0.8244 - top-5-accuracy: 0.9906

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.6657 - loss: 0.8367 - top-5-accuracy: 0.9890

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.6616 - loss: 0.8480 - top-5-accuracy: 0.9874

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.6595 - loss: 0.8552 - top-5-accuracy: 0.9867

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.6583 - loss: 0.8615 - top-5-accuracy: 0.9862

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.6581 - loss: 0.8641 - top-5-accuracy: 0.9859 - val_accuracy: 0.7764 - val_loss: 0.6584 - val_top-5-accuracy: 0.9689


<div class="k-default-codeblock">
```
Epoch 15/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.6250 - loss: 0.9154 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6594 - loss: 0.8413 - top-5-accuracy: 0.9907

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.6683 - loss: 0.8076 - top-5-accuracy: 0.9871

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.6707 - loss: 0.8011 - top-5-accuracy: 0.9849

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.6725 - loss: 0.7989 - top-5-accuracy: 0.9836

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.6758 - loss: 0.7959 - top-5-accuracy: 0.9830

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.6794 - loss: 0.7916 - top-5-accuracy: 0.9829

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.6826 - loss: 0.7865 - top-5-accuracy: 0.9831

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.6855 - loss: 0.7822 - top-5-accuracy: 0.9833 - val_accuracy: 0.7516 - val_loss: 0.5608 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 16/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.7500 - loss: 0.6985 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.7688 - loss: 0.6334 - top-5-accuracy: 0.9972

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.7699 - loss: 0.6326 - top-5-accuracy: 0.9962

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.7721 - loss: 0.6301 - top-5-accuracy: 0.9953

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.7734 - loss: 0.6319 - top-5-accuracy: 0.9940

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.7725 - loss: 0.6338 - top-5-accuracy: 0.9933

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.7707 - loss: 0.6347 - top-5-accuracy: 0.9931

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.7689 - loss: 0.6357 - top-5-accuracy: 0.9927

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.7680 - loss: 0.6362 - top-5-accuracy: 0.9925 - val_accuracy: 0.8137 - val_loss: 0.5312 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 17/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.6875 - loss: 0.8046 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.7303 - loss: 0.6940 - top-5-accuracy: 0.9987

<div class="k-default-codeblock">
```

```
</div>
  8/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.7368 - loss: 0.6738 - top-5-accuracy: 0.9975

<div class="k-default-codeblock">
```

```
</div>
 11/31 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.7408 - loss: 0.6624 - top-5-accuracy: 0.9965

<div class="k-default-codeblock">
```

```
</div>
 14/31 ━━━━━━━━━[37m━━━━━━━━━━━  0s 16ms/step - accuracy: 0.7439 - loss: 0.6539 - top-5-accuracy: 0.9952

<div class="k-default-codeblock">
```

```
</div>
 18/31 ━━━━━━━━━━━[37m━━━━━━━━━  0s 16ms/step - accuracy: 0.7480 - loss: 0.6458 - top-5-accuracy: 0.9946

<div class="k-default-codeblock">
```

```
</div>
 22/31 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - accuracy: 0.7510 - loss: 0.6398 - top-5-accuracy: 0.9943

<div class="k-default-codeblock">
```

```
</div>
 26/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.7532 - loss: 0.6353 - top-5-accuracy: 0.9942

<div class="k-default-codeblock">
```

```
</div>
 30/31 ━━━━━━━━━━━━━━━━━━━[37m━  0s 16ms/step - accuracy: 0.7550 - loss: 0.6311 - top-5-accuracy: 0.9942

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.7557 - loss: 0.6290 - top-5-accuracy: 0.9941 - val_accuracy: 0.8261 - val_loss: 0.4922 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 18/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.8125 - loss: 0.6099 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.7993 - loss: 0.5834 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8044 - loss: 0.5673 - top-5-accuracy: 0.9992

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8025 - loss: 0.5659 - top-5-accuracy: 0.9986

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.8021 - loss: 0.5633 - top-5-accuracy: 0.9984

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.8004 - loss: 0.5621 - top-5-accuracy: 0.9984

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.7978 - loss: 0.5623 - top-5-accuracy: 0.9985

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.7963 - loss: 0.5622 - top-5-accuracy: 0.9984

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.7959 - loss: 0.5617 - top-5-accuracy: 0.9983 - val_accuracy: 0.8571 - val_loss: 0.5097 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 19/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.8438 - loss: 0.5885 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8185 - loss: 0.5564 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8084 - loss: 0.5478 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8029 - loss: 0.5482 - top-5-accuracy: 0.9998

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.7993 - loss: 0.5478 - top-5-accuracy: 0.9994

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.7977 - loss: 0.5458 - top-5-accuracy: 0.9992

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.7963 - loss: 0.5451 - top-5-accuracy: 0.9991

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.7953 - loss: 0.5450 - top-5-accuracy: 0.9990

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.7949 - loss: 0.5443 - top-5-accuracy: 0.9988 - val_accuracy: 0.8820 - val_loss: 0.3987 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 20/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.8438 - loss: 0.4178 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8384 - loss: 0.4479 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8284 - loss: 0.4607 - top-5-accuracy: 0.9996

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8196 - loss: 0.4819 - top-5-accuracy: 0.9980

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.8147 - loss: 0.4926 - top-5-accuracy: 0.9976

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.8122 - loss: 0.4993 - top-5-accuracy: 0.9971

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.8096 - loss: 0.5054 - top-5-accuracy: 0.9969

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.8077 - loss: 0.5090 - top-5-accuracy: 0.9969

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8069 - loss: 0.5107 - top-5-accuracy: 0.9969 - val_accuracy: 0.8882 - val_loss: 0.3625 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 21/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.9688 - loss: 0.2426 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8904 - loss: 0.3674 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8780 - loss: 0.3851 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8693 - loss: 0.4033 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.8651 - loss: 0.4110 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.8635 - loss: 0.4147 - top-5-accuracy: 0.9997

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.8622 - loss: 0.4161 - top-5-accuracy: 0.9995

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.8614 - loss: 0.4163 - top-5-accuracy: 0.9994

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8609 - loss: 0.4159 - top-5-accuracy: 0.9994 - val_accuracy: 0.8509 - val_loss: 0.4748 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 22/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 29ms/step - accuracy: 0.9062 - loss: 0.3291 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8729 - loss: 0.3683 - top-5-accuracy: 0.9972

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8624 - loss: 0.3905 - top-5-accuracy: 0.9965

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8627 - loss: 0.3944 - top-5-accuracy: 0.9968

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.8631 - loss: 0.3950 - top-5-accuracy: 0.9969

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.8617 - loss: 0.3977 - top-5-accuracy: 0.9968

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.8600 - loss: 0.4001 - top-5-accuracy: 0.9966

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.8588 - loss: 0.4019 - top-5-accuracy: 0.9966

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8580 - loss: 0.4027 - top-5-accuracy: 0.9967 - val_accuracy: 0.9006 - val_loss: 0.3011 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 23/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.7188 - loss: 0.5273 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8249 - loss: 0.3896 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8389 - loss: 0.3805 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8518 - loss: 0.3679 - top-5-accuracy: 0.9994

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.8602 - loss: 0.3602 - top-5-accuracy: 0.9991

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.8648 - loss: 0.3559 - top-5-accuracy: 0.9987

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.8676 - loss: 0.3523 - top-5-accuracy: 0.9985

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.8696 - loss: 0.3494 - top-5-accuracy: 0.9984

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8712 - loss: 0.3468 - top-5-accuracy: 0.9983 - val_accuracy: 0.9006 - val_loss: 0.3041 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 24/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.8438 - loss: 0.3418 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8599 - loss: 0.3166 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8712 - loss: 0.3148 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8782 - loss: 0.3116 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.8805 - loss: 0.3100 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.8809 - loss: 0.3096 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.8819 - loss: 0.3073 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.8833 - loss: 0.3043 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8846 - loss: 0.3015 - top-5-accuracy: 1.0000 - val_accuracy: 0.8758 - val_loss: 0.3618 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 25/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 29ms/step - accuracy: 0.9062 - loss: 0.2927 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8963 - loss: 0.2718 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8906 - loss: 0.2844 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8853 - loss: 0.2965 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.8821 - loss: 0.3056 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.8788 - loss: 0.3154 - top-5-accuracy: 0.9999

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.8750 - loss: 0.3254 - top-5-accuracy: 0.9997

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.8730 - loss: 0.3311 - top-5-accuracy: 0.9996

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.8719 - loss: 0.3342 - top-5-accuracy: 0.9995 - val_accuracy: 0.8944 - val_loss: 0.3555 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 26/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9062 - loss: 0.2375 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9040 - loss: 0.2683 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8993 - loss: 0.2798 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9012 - loss: 0.2796 - top-5-accuracy: 0.9992

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9047 - loss: 0.2755 - top-5-accuracy: 0.9989

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9062 - loss: 0.2741 - top-5-accuracy: 0.9988

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9076 - loss: 0.2722 - top-5-accuracy: 0.9988

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9085 - loss: 0.2706 - top-5-accuracy: 0.9988

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9093 - loss: 0.2687 - top-5-accuracy: 0.9988 - val_accuracy: 0.8944 - val_loss: 0.3705 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 27/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.7812 - loss: 0.4278 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8183 - loss: 0.3474 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8451 - loss: 0.3107 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8605 - loss: 0.2932 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.8707 - loss: 0.2814 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.8770 - loss: 0.2756 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.8820 - loss: 0.2704 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.8858 - loss: 0.2657 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8882 - loss: 0.2624 - top-5-accuracy: 1.0000 - val_accuracy: 0.8634 - val_loss: 0.4295 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 28/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 0.9062 - loss: 0.2949 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9155 - loss: 0.2620 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9164 - loss: 0.2611 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9187 - loss: 0.2562 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9206 - loss: 0.2503 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9209 - loss: 0.2482 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9209 - loss: 0.2466 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9208 - loss: 0.2455 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9211 - loss: 0.2444 - top-5-accuracy: 1.0000 - val_accuracy: 0.8634 - val_loss: 0.6419 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 29/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.8438 - loss: 0.3444 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8826 - loss: 0.2828 - top-5-accuracy: 0.9920

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8940 - loss: 0.2613 - top-5-accuracy: 0.9936

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9011 - loss: 0.2486 - top-5-accuracy: 0.9948

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9050 - loss: 0.2428 - top-5-accuracy: 0.9955

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9081 - loss: 0.2394 - top-5-accuracy: 0.9961

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9102 - loss: 0.2369 - top-5-accuracy: 0.9965

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9122 - loss: 0.2341 - top-5-accuracy: 0.9968

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9134 - loss: 0.2317 - top-5-accuracy: 0.9970 - val_accuracy: 0.8882 - val_loss: 0.3478 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 30/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.9375 - loss: 0.1918 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9375 - loss: 0.1868 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9361 - loss: 0.1912 - top-5-accuracy: 0.9996

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9366 - loss: 0.1920 - top-5-accuracy: 0.9989

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9358 - loss: 0.1938 - top-5-accuracy: 0.9987

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9352 - loss: 0.1951 - top-5-accuracy: 0.9986

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9349 - loss: 0.1963 - top-5-accuracy: 0.9986

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9351 - loss: 0.1961 - top-5-accuracy: 0.9987

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9351 - loss: 0.1954 - top-5-accuracy: 0.9987 - val_accuracy: 0.9006 - val_loss: 0.3516 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 31/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.9375 - loss: 0.2136 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9101 - loss: 0.2468 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9072 - loss: 0.2465 - top-5-accuracy: 0.9996

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9093 - loss: 0.2413 - top-5-accuracy: 0.9989

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9120 - loss: 0.2366 - top-5-accuracy: 0.9987

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9141 - loss: 0.2327 - top-5-accuracy: 0.9986

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9159 - loss: 0.2297 - top-5-accuracy: 0.9986

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9171 - loss: 0.2288 - top-5-accuracy: 0.9987

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9178 - loss: 0.2275 - top-5-accuracy: 0.9987 - val_accuracy: 0.8820 - val_loss: 0.3468 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 32/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9688 - loss: 0.1235 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9393 - loss: 0.1657 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9431 - loss: 0.1626 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9451 - loss: 0.1592 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9477 - loss: 0.1545 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9487 - loss: 0.1532 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9489 - loss: 0.1524 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9484 - loss: 0.1536 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9481 - loss: 0.1543 - top-5-accuracy: 1.0000 - val_accuracy: 0.8820 - val_loss: 0.4636 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 33/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.9375 - loss: 0.1628 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9525 - loss: 0.1448 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9554 - loss: 0.1353 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9546 - loss: 0.1318 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9545 - loss: 0.1323 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9539 - loss: 0.1345 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9533 - loss: 0.1365 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9533 - loss: 0.1368 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9532 - loss: 0.1368 - top-5-accuracy: 1.0000 - val_accuracy: 0.9255 - val_loss: 0.3003 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 34/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.9688 - loss: 0.0966 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9577 - loss: 0.1062 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9595 - loss: 0.1047 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9627 - loss: 0.1021 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9644 - loss: 0.1016 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9646 - loss: 0.1034 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9649 - loss: 0.1041 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9656 - loss: 0.1035 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9660 - loss: 0.1031 - top-5-accuracy: 1.0000 - val_accuracy: 0.9317 - val_loss: 0.2345 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 35/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.9375 - loss: 0.0861 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9566 - loss: 0.0832 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9572 - loss: 0.0953 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9599 - loss: 0.0968 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9614 - loss: 0.0975 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9620 - loss: 0.0984 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9622 - loss: 0.1000 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9617 - loss: 0.1018 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9612 - loss: 0.1032 - top-5-accuracy: 1.0000 - val_accuracy: 0.9068 - val_loss: 0.2802 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 36/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0527 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9536 - loss: 0.1427 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9553 - loss: 0.1346 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9602 - loss: 0.1254 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9637 - loss: 0.1192 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9653 - loss: 0.1151 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9668 - loss: 0.1115 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9682 - loss: 0.1078 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9690 - loss: 0.1053 - top-5-accuracy: 1.0000 - val_accuracy: 0.9193 - val_loss: 0.2598 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 37/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0225 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9972 - loss: 0.0384 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9965 - loss: 0.0409 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9956 - loss: 0.0421 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9943 - loss: 0.0432 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9934 - loss: 0.0439 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9927 - loss: 0.0445 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9925 - loss: 0.0445 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9924 - loss: 0.0443 - top-5-accuracy: 1.0000 - val_accuracy: 0.9130 - val_loss: 0.2912 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 38/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9688 - loss: 0.1050 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9777 - loss: 0.0783 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9830 - loss: 0.0656 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9855 - loss: 0.0584 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9870 - loss: 0.0547 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9881 - loss: 0.0520 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9890 - loss: 0.0499 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9897 - loss: 0.0480 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9902 - loss: 0.0468 - top-5-accuracy: 1.0000 - val_accuracy: 0.9193 - val_loss: 0.2657 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 39/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0156 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9920 - loss: 0.0308 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9936 - loss: 0.0285 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9948 - loss: 0.0274 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9955 - loss: 0.0265 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9958 - loss: 0.0262 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9960 - loss: 0.0259 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9962 - loss: 0.0257 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9964 - loss: 0.0255 - top-5-accuracy: 1.0000 - val_accuracy: 0.9255 - val_loss: 0.2832 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 40/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0472 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0420 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0357 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9992 - loss: 0.0340 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9989 - loss: 0.0331 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9988 - loss: 0.0325 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9983 - loss: 0.0328 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9980 - loss: 0.0328 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9978 - loss: 0.0330 - top-5-accuracy: 1.0000 - val_accuracy: 0.9255 - val_loss: 0.3104 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 41/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.9688 - loss: 0.0917 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9829 - loss: 0.0642 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9836 - loss: 0.0623 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9836 - loss: 0.0618 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9834 - loss: 0.0612 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9832 - loss: 0.0612 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9832 - loss: 0.0607 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9835 - loss: 0.0599 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9837 - loss: 0.0599 - top-5-accuracy: 1.0000 - val_accuracy: 0.9130 - val_loss: 0.2980 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 42/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0396 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9975 - loss: 0.0380 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9935 - loss: 0.0412 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9921 - loss: 0.0416 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9917 - loss: 0.0417 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9915 - loss: 0.0414 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9916 - loss: 0.0408 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9916 - loss: 0.0402 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9916 - loss: 0.0399 - top-5-accuracy: 1.0000 - val_accuracy: 0.9193 - val_loss: 0.2599 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 43/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0106 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9825 - loss: 0.0475 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9815 - loss: 0.0526 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9804 - loss: 0.0557 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9807 - loss: 0.0560 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9817 - loss: 0.0556 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9827 - loss: 0.0543 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9833 - loss: 0.0536 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.9837 - loss: 0.0529 - top-5-accuracy: 1.0000 - val_accuracy: 0.9317 - val_loss: 0.3046 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 44/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0306 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0309 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0298 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0285 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0275 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9999 - loss: 0.0269 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9997 - loss: 0.0263 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9996 - loss: 0.0257 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9995 - loss: 0.0257 - top-5-accuracy: 1.0000 - val_accuracy: 0.9255 - val_loss: 0.2716 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 45/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9688 - loss: 0.0380 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9740 - loss: 0.0689 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9747 - loss: 0.0714 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9761 - loss: 0.0712 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9764 - loss: 0.0729 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9770 - loss: 0.0726 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9775 - loss: 0.0720 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9780 - loss: 0.0711 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9781 - loss: 0.0707 - top-5-accuracy: 1.0000 - val_accuracy: 0.8758 - val_loss: 0.5065 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 46/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0398 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9453 - loss: 0.1792 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9326 - loss: 0.1848 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9264 - loss: 0.1961 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9236 - loss: 0.1987 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9227 - loss: 0.1986 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9235 - loss: 0.1959 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9245 - loss: 0.1926 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9254 - loss: 0.1899 - top-5-accuracy: 1.0000 - val_accuracy: 0.9130 - val_loss: 0.3201 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 47/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.9688 - loss: 0.0559 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9672 - loss: 0.0680 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9685 - loss: 0.0765 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9719 - loss: 0.0751 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9730 - loss: 0.0753 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9744 - loss: 0.0741 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9756 - loss: 0.0731 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9768 - loss: 0.0716 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.9776 - loss: 0.0704 - top-5-accuracy: 1.0000 - val_accuracy: 0.9255 - val_loss: 0.2490 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 48/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0411 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9920 - loss: 0.0411 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9853 - loss: 0.0554 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9842 - loss: 0.0573 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9837 - loss: 0.0587 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9833 - loss: 0.0601 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9829 - loss: 0.0607 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9827 - loss: 0.0608 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9827 - loss: 0.0609 - top-5-accuracy: 1.0000 - val_accuracy: 0.9068 - val_loss: 0.3289 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 49/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 29ms/step - accuracy: 1.0000 - loss: 0.0647 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9861 - loss: 0.0517 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9822 - loss: 0.0531 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9816 - loss: 0.0529 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9812 - loss: 0.0549 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9813 - loss: 0.0559 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9814 - loss: 0.0569 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9816 - loss: 0.0575 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9818 - loss: 0.0576 - top-5-accuracy: 1.0000 - val_accuracy: 0.9193 - val_loss: 0.3600 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 50/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0344 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9920 - loss: 0.0526 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9936 - loss: 0.0467 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9948 - loss: 0.0424 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9955 - loss: 0.0389 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9958 - loss: 0.0367 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9961 - loss: 0.0349 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9963 - loss: 0.0332 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9965 - loss: 0.0322 - top-5-accuracy: 1.0000 - val_accuracy: 0.9441 - val_loss: 0.2380 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 51/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0240 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0199 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0184 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0173 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0162 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9999 - loss: 0.0155 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9997 - loss: 0.0150 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9996 - loss: 0.0145 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9995 - loss: 0.0142 - top-5-accuracy: 1.0000 - val_accuracy: 0.9441 - val_loss: 0.2648 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 52/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0032 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0080 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0078 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0078 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0075 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0073 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0072 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0070 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0069 - top-5-accuracy: 1.0000 - val_accuracy: 0.9441 - val_loss: 0.2590 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 53/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0078 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0051 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0045 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0042 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0041 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0040 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0040 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0040 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0040 - top-5-accuracy: 1.0000 - val_accuracy: 0.9317 - val_loss: 0.2595 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 54/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0033 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000 - val_accuracy: 0.9379 - val_loss: 0.2512 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 55/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0025 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0025 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0025 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0026 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0026 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0026 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0026 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0026 - top-5-accuracy: 1.0000 - val_accuracy: 0.9441 - val_loss: 0.2566 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 56/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0026 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0022 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0021 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0021 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0021 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0021 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0021 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0021 - top-5-accuracy: 1.0000 - val_accuracy: 0.9441 - val_loss: 0.2594 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 57/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0028 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0025 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000 - val_accuracy: 0.9379 - val_loss: 0.2510 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 58/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 1.0000 - loss: 0.0029 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0028 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0025 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0024 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0024 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0024 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 1.0000 - loss: 0.0023 - top-5-accuracy: 1.0000 - val_accuracy: 0.9379 - val_loss: 0.2562 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 59/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0015 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000 - val_accuracy: 0.9441 - val_loss: 0.2580 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 60/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0021 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000 - val_accuracy: 0.9441 - val_loss: 0.2572 - val_top-5-accuracy: 1.0000


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  0s 35ms/step - accuracy: 0.7812 - loss: 0.9119 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/20 ━━━━━━━━━━━━━[37m━━━━━━━  0s 4ms/step - accuracy: 0.8093 - loss: 0.9078 - top-5-accuracy: 0.9779 

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 101ms/step - accuracy: 0.8051 - loss: 0.9266 - top-5-accuracy: 0.9778

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 2s 101ms/step - accuracy: 0.8047 - loss: 0.9278 - top-5-accuracy: 0.9777


<div class="k-default-codeblock">
```
Test accuracy: 79.51%
Test top 5 accuracy: 97.54%

```
</div>
---
## Inference


```python
NUM_SAMPLES_VIZ = 25
testsamples, labels = next(iter(testloader))
testsamples, labels = testsamples[:NUM_SAMPLES_VIZ], labels[:NUM_SAMPLES_VIZ]

ground_truths = []
preds = []
videos = []

for i, (testsample, label) in enumerate(zip(testsamples, labels)):
    # Generate gif
    testsample = np.reshape(testsample.numpy(), (-1, 28, 28))
    with io.BytesIO() as gif:
        imageio.mimsave(gif, (testsample * 255).astype("uint8"), "GIF", fps=5)
        videos.append(gif.getvalue())

    # Get model prediction
    output = model.predict(ops.expand_dims(testsample, axis=0))[0]
    pred = np.argmax(output, axis=0)

    ground_truths.append(label.numpy().astype("int"))
    preds.append(pred)


def make_box_for_grid(image_widget, fit):
    """Make a VBox to hold caption/image for demonstrating option_fit values.

    Source: https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Styling.html
    """
    # Make the caption
    if fit is not None:
        fit_str = "'{}'".format(fit)
    else:
        fit_str = str(fit)

    h = ipywidgets.HTML(value="" + str(fit_str) + "")

    # Make the green box with the image widget inside it
    boxb = ipywidgets.widgets.Box()
    boxb.children = [image_widget]

    # Compose into a vertical box
    vb = ipywidgets.widgets.VBox()
    vb.layout.align_items = "center"
    vb.children = [h, boxb]
    return vb


boxes = []
for i in range(NUM_SAMPLES_VIZ):
    ib = ipywidgets.widgets.Image(value=videos[i], width=100, height=100)
    true_class = info["label"][str(ground_truths[i])]
    pred_class = info["label"][str(preds[i])]
    caption = f"T: {true_class} | P: {pred_class}"

    boxes.append(make_box_for_grid(ib, caption))

ipywidgets.widgets.GridBox(
    boxes, layout=ipywidgets.widgets.Layout(grid_template_columns="repeat(5, 200px)")
)
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 4s 4s/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 38ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step





<div class="k-default-codeblock">
```
GridBox(children=(VBox(children=(HTML(value="'T: pancreas | P: pancreas'"), Box(children=(Image(value=b'GIF89a…

```
</div>
---
## Final thoughts

With a vanilla implementation, we achieve ~79-80% Top-1 accuracy on the
test dataset.

The hyperparameters used in this tutorial were finalized by running a
hyperparameter search using
[W&B Sweeps](https://docs.wandb.ai/guides/sweeps).
You can find out our sweeps result
[here](https://wandb.ai/minimal-implementations/vivit/sweeps/66fp0lhz)
and our quick analysis of the results
[here](https://wandb.ai/minimal-implementations/vivit/reports/Hyperparameter-Tuning-Analysis--VmlldzoxNDEwNzcx).

For further improvement, you could look into the following:

- Using data augmentation for videos.
- Using a better regularization scheme for training.
- Apply different variants of the transformer model as in the paper.

We would like to thank [Anurag Arnab](https://anuragarnab.github.io/)
(first author of ViViT) for helpful discussion. We are grateful to
[Weights and Biases](https://wandb.ai/site) program for helping with
GPU credits.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/video-vision-transformer)
and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/video-vision-transformer-CT).

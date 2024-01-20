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
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  30:16 61s/step - accuracy: 0.0312 - loss: 2.6952 - top-5-accuracy: 0.5938

<div class="k-default-codeblock">
```

```
</div>
  3/31 ━[37m━━━━━━━━━━━━━━━━━━━  1s 37ms/step - accuracy: 0.0799 - loss: 2.7918 - top-5-accuracy: 0.5608  

<div class="k-default-codeblock">
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1705722589.399030    2067 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
W0000 00:00:1705722589.449689    2067 graph_launch.cc:671] Fallback to op-by-op mode because memset node breaks graph update


```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 33ms/step - accuracy: 0.0895 - loss: 2.7608 - top-5-accuracy: 0.5586

<div class="k-default-codeblock">
```

```
</div>
  7/31 ━━━━[37m━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.0961 - loss: 2.7325 - top-5-accuracy: 0.5550

<div class="k-default-codeblock">
```

```
</div>
 10/31 ━━━━━━[37m━━━━━━━━━━━━━━  0s 27ms/step - accuracy: 0.1017 - loss: 2.6905 - top-5-accuracy: 0.5489

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 26ms/step - accuracy: 0.1048 - loss: 2.6533 - top-5-accuracy: 0.5542

<div class="k-default-codeblock">
```

```
</div>
 16/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 24ms/step - accuracy: 0.1087 - loss: 2.6217 - top-5-accuracy: 0.5603

<div class="k-default-codeblock">
```

```
</div>
 19/31 ━━━━━━━━━━━━[37m━━━━━━━━  0s 23ms/step - accuracy: 0.1124 - loss: 2.5962 - top-5-accuracy: 0.5656

<div class="k-default-codeblock">
```

```
</div>
 23/31 ━━━━━━━━━━━━━━[37m━━━━━━  0s 22ms/step - accuracy: 0.1161 - loss: 2.5708 - top-5-accuracy: 0.5712

<div class="k-default-codeblock">
```

```
</div>
 27/31 ━━━━━━━━━━━━━━━━━[37m━━━  0s 21ms/step - accuracy: 0.1184 - loss: 2.5500 - top-5-accuracy: 0.5760

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 789ms/step - accuracy: 0.1206 - loss: 2.5320 - top-5-accuracy: 0.5799

<div class="k-default-codeblock">
```
W0000 00:00:1705722613.126173    2068 graph_launch.cc:671] Fallback to op-by-op mode because memset node breaks graph update

W0000 00:00:1705722617.161787    2067 graph_launch.cc:671] Fallback to op-by-op mode because memset node breaks graph update


```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 92s 1s/step - accuracy: 0.1211 - loss: 2.5279 - top-5-accuracy: 0.5808 - val_accuracy: 0.1429 - val_loss: 2.2000 - val_top-5-accuracy: 0.6584


<div class="k-default-codeblock">
```
Epoch 2/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  1s 45ms/step - accuracy: 0.1562 - loss: 2.3899 - top-5-accuracy: 0.6250

<div class="k-default-codeblock">
```

```
</div>
  3/31 ━[37m━━━━━━━━━━━━━━━━━━━  0s 27ms/step - accuracy: 0.1962 - loss: 2.3642 - top-5-accuracy: 0.6424

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 28ms/step - accuracy: 0.2127 - loss: 2.3237 - top-5-accuracy: 0.6620

<div class="k-default-codeblock">
```

```
</div>
  7/31 ━━━━[37m━━━━━━━━━━━━━━━━  0s 28ms/step - accuracy: 0.2145 - loss: 2.2978 - top-5-accuracy: 0.6687

<div class="k-default-codeblock">
```

```
</div>
 10/31 ━━━━━━[37m━━━━━━━━━━━━━━  0s 25ms/step - accuracy: 0.2138 - loss: 2.2681 - top-5-accuracy: 0.6752

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 23ms/step - accuracy: 0.2098 - loss: 2.2490 - top-5-accuracy: 0.6817

<div class="k-default-codeblock">
```

```
</div>
 16/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 22ms/step - accuracy: 0.2059 - loss: 2.2383 - top-5-accuracy: 0.6863

<div class="k-default-codeblock">
```

```
</div>
 19/31 ━━━━━━━━━━━━[37m━━━━━━━━  0s 22ms/step - accuracy: 0.2020 - loss: 2.2285 - top-5-accuracy: 0.6907

<div class="k-default-codeblock">
```

```
</div>
 23/31 ━━━━━━━━━━━━━━[37m━━━━━━  0s 20ms/step - accuracy: 0.1986 - loss: 2.2177 - top-5-accuracy: 0.6962

<div class="k-default-codeblock">
```

```
</div>
 27/31 ━━━━━━━━━━━━━━━━━[37m━━━  0s 20ms/step - accuracy: 0.1966 - loss: 2.2092 - top-5-accuracy: 0.7012

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.1950 - loss: 2.2002 - top-5-accuracy: 0.7058

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 21ms/step - accuracy: 0.1948 - loss: 2.1979 - top-5-accuracy: 0.7069 - val_accuracy: 0.3602 - val_loss: 1.9981 - val_top-5-accuracy: 0.7826


<div class="k-default-codeblock">
```
Epoch 3/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 0.2812 - loss: 2.1480 - top-5-accuracy: 0.7812

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.2669 - loss: 2.1390 - top-5-accuracy: 0.7731

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.2622 - loss: 2.1056 - top-5-accuracy: 0.7712

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.2615 - loss: 2.0814 - top-5-accuracy: 0.7709

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.2605 - loss: 2.0652 - top-5-accuracy: 0.7740

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.2577 - loss: 2.0538 - top-5-accuracy: 0.7781

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.2550 - loss: 2.0446 - top-5-accuracy: 0.7817

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.2546 - loss: 2.0343 - top-5-accuracy: 0.7853

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.2544 - loss: 2.0272 - top-5-accuracy: 0.7872 - val_accuracy: 0.2795 - val_loss: 1.8221 - val_top-5-accuracy: 0.8944


<div class="k-default-codeblock">
```
Epoch 4/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 0.1875 - loss: 2.2079 - top-5-accuracy: 0.7500

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.2533 - loss: 2.0053 - top-5-accuracy: 0.7758

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.2757 - loss: 1.9457 - top-5-accuracy: 0.7805

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.2848 - loss: 1.9173 - top-5-accuracy: 0.7894

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.2859 - loss: 1.9032 - top-5-accuracy: 0.7968

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.2861 - loss: 1.8946 - top-5-accuracy: 0.8018

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.2846 - loss: 1.8884 - top-5-accuracy: 0.8067

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.2845 - loss: 1.8809 - top-5-accuracy: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.2848 - loss: 1.8748 - top-5-accuracy: 0.8144 - val_accuracy: 0.3106 - val_loss: 1.6239 - val_top-5-accuracy: 0.9379


<div class="k-default-codeblock">
```
Epoch 5/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 0.2188 - loss: 1.8382 - top-5-accuracy: 0.7188

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.2966 - loss: 1.7542 - top-5-accuracy: 0.7979

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.3206 - loss: 1.7142 - top-5-accuracy: 0.8191

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.3377 - loss: 1.6902 - top-5-accuracy: 0.8316

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.3481 - loss: 1.6809 - top-5-accuracy: 0.8397

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.3529 - loss: 1.6746 - top-5-accuracy: 0.8457

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.3553 - loss: 1.6699 - top-5-accuracy: 0.8505

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.3570 - loss: 1.6646 - top-5-accuracy: 0.8553

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.3583 - loss: 1.6593 - top-5-accuracy: 0.8584 - val_accuracy: 0.4286 - val_loss: 1.3703 - val_top-5-accuracy: 0.9503


<div class="k-default-codeblock">
```
Epoch 6/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.4375 - loss: 1.4682 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.4430 - loss: 1.4949 - top-5-accuracy: 0.9463

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.4372 - loss: 1.4953 - top-5-accuracy: 0.9331

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.4313 - loss: 1.4945 - top-5-accuracy: 0.9288

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.4271 - loss: 1.4947 - top-5-accuracy: 0.9271

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.4227 - loss: 1.4936 - top-5-accuracy: 0.9266

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.4189 - loss: 1.4926 - top-5-accuracy: 0.9266

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.4158 - loss: 1.4908 - top-5-accuracy: 0.9264

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.4146 - loss: 1.4885 - top-5-accuracy: 0.9265 - val_accuracy: 0.4286 - val_loss: 1.4784 - val_top-5-accuracy: 0.9379


<div class="k-default-codeblock">
```
Epoch 7/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.3750 - loss: 1.4142 - top-5-accuracy: 0.9688

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.4579 - loss: 1.3521 - top-5-accuracy: 0.9505

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.4734 - loss: 1.3533 - top-5-accuracy: 0.9440

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.4770 - loss: 1.3642 - top-5-accuracy: 0.9370

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.4785 - loss: 1.3685 - top-5-accuracy: 0.9356

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.4768 - loss: 1.3710 - top-5-accuracy: 0.9363

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.4751 - loss: 1.3731 - top-5-accuracy: 0.9367

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.4742 - loss: 1.3744 - top-5-accuracy: 0.9374

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.4745 - loss: 1.3741 - top-5-accuracy: 0.9376 - val_accuracy: 0.5155 - val_loss: 1.2686 - val_top-5-accuracy: 0.9565


<div class="k-default-codeblock">
```
Epoch 8/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.5625 - loss: 1.5141 - top-5-accuracy: 0.7500

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.5535 - loss: 1.3644 - top-5-accuracy: 0.8571

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.5261 - loss: 1.3526 - top-5-accuracy: 0.8809

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.5174 - loss: 1.3487 - top-5-accuracy: 0.8920

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.5106 - loss: 1.3462 - top-5-accuracy: 0.9006

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.5056 - loss: 1.3422 - top-5-accuracy: 0.9069

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.5030 - loss: 1.3372 - top-5-accuracy: 0.9122

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.5026 - loss: 1.3304 - top-5-accuracy: 0.9166

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.5027 - loss: 1.3252 - top-5-accuracy: 0.9195 - val_accuracy: 0.5839 - val_loss: 0.9639 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 9/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 0.5938 - loss: 1.3178 - top-5-accuracy: 0.9375

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.5911 - loss: 1.1961 - top-5-accuracy: 0.9421

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5792 - loss: 1.1956 - top-5-accuracy: 0.9461

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5711 - loss: 1.1973 - top-5-accuracy: 0.9480

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.5667 - loss: 1.1995 - top-5-accuracy: 0.9492

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.5638 - loss: 1.1996 - top-5-accuracy: 0.9499

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.5605 - loss: 1.1981 - top-5-accuracy: 0.9510

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.5580 - loss: 1.1957 - top-5-accuracy: 0.9521

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.5572 - loss: 1.1930 - top-5-accuracy: 0.9529 - val_accuracy: 0.5652 - val_loss: 1.0840 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 10/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  1s 52ms/step - accuracy: 0.4688 - loss: 1.1495 - top-5-accuracy: 0.9688

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5329 - loss: 1.0429 - top-5-accuracy: 0.9691

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5426 - loss: 1.0514 - top-5-accuracy: 0.9687

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.5519 - loss: 1.0595 - top-5-accuracy: 0.9674

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.5568 - loss: 1.0640 - top-5-accuracy: 0.9680

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.5570 - loss: 1.0721 - top-5-accuracy: 0.9677

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.5564 - loss: 1.0813 - top-5-accuracy: 0.9669

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.5556 - loss: 1.0871 - top-5-accuracy: 0.9664

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.5561 - loss: 1.0878 - top-5-accuracy: 0.9664 - val_accuracy: 0.6273 - val_loss: 0.9658 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 11/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.7188 - loss: 0.9506 - top-5-accuracy: 0.9688

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6717 - loss: 0.9825 - top-5-accuracy: 0.9656

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.6474 - loss: 0.9802 - top-5-accuracy: 0.9706

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6352 - loss: 0.9847 - top-5-accuracy: 0.9714

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.6283 - loss: 0.9864 - top-5-accuracy: 0.9709

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.6236 - loss: 0.9903 - top-5-accuracy: 0.9705

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.6202 - loss: 0.9934 - top-5-accuracy: 0.9705

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.6178 - loss: 0.9955 - top-5-accuracy: 0.9708

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.6171 - loss: 0.9946 - top-5-accuracy: 0.9711 - val_accuracy: 0.6646 - val_loss: 0.8816 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 12/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.6562 - loss: 0.9352 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.6601 - loss: 0.8445 - top-5-accuracy: 0.9987

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.6499 - loss: 0.8798 - top-5-accuracy: 0.9932

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6434 - loss: 0.9120 - top-5-accuracy: 0.9900

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.6393 - loss: 0.9286 - top-5-accuracy: 0.9875

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.6360 - loss: 0.9398 - top-5-accuracy: 0.9857

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.6346 - loss: 0.9471 - top-5-accuracy: 0.9841

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.6343 - loss: 0.9498 - top-5-accuracy: 0.9830

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.6343 - loss: 0.9504 - top-5-accuracy: 0.9823 - val_accuracy: 0.6211 - val_loss: 1.0638 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 13/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.6562 - loss: 0.8997 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.6244 - loss: 0.8836 - top-5-accuracy: 0.9972

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.6221 - loss: 0.9022 - top-5-accuracy: 0.9923

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.6164 - loss: 0.9232 - top-5-accuracy: 0.9866

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.6161 - loss: 0.9331 - top-5-accuracy: 0.9830

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.6164 - loss: 0.9401 - top-5-accuracy: 0.9803

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.6184 - loss: 0.9429 - top-5-accuracy: 0.9787

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.6215 - loss: 0.9420 - top-5-accuracy: 0.9779

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.6240 - loss: 0.9405 - top-5-accuracy: 0.9775 - val_accuracy: 0.7391 - val_loss: 0.7499 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 14/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.6875 - loss: 0.6800 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.7302 - loss: 0.6762 - top-5-accuracy: 0.9951

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.7375 - loss: 0.6973 - top-5-accuracy: 0.9910

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.7326 - loss: 0.7210 - top-5-accuracy: 0.9884

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.7268 - loss: 0.7438 - top-5-accuracy: 0.9861

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.7195 - loss: 0.7652 - top-5-accuracy: 0.9843

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.7131 - loss: 0.7827 - top-5-accuracy: 0.9836

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.7083 - loss: 0.7962 - top-5-accuracy: 0.9835

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.7054 - loss: 0.8030 - top-5-accuracy: 0.9835 - val_accuracy: 0.7391 - val_loss: 0.6997 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 15/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.6250 - loss: 0.9857 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6665 - loss: 0.8929 - top-5-accuracy: 0.9972

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6816 - loss: 0.8591 - top-5-accuracy: 0.9930

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.6845 - loss: 0.8467 - top-5-accuracy: 0.9908

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.6834 - loss: 0.8443 - top-5-accuracy: 0.9887

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.6849 - loss: 0.8399 - top-5-accuracy: 0.9878

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.6869 - loss: 0.8337 - top-5-accuracy: 0.9874

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.6899 - loss: 0.8268 - top-5-accuracy: 0.9873

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.6928 - loss: 0.8209 - top-5-accuracy: 0.9873 - val_accuracy: 0.7453 - val_loss: 0.6796 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 16/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.6875 - loss: 0.7382 - top-5-accuracy: 0.9688

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.7341 - loss: 0.6428 - top-5-accuracy: 0.9857

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.7400 - loss: 0.6518 - top-5-accuracy: 0.9894

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.7435 - loss: 0.6531 - top-5-accuracy: 0.9908

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.7470 - loss: 0.6559 - top-5-accuracy: 0.9914

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.7492 - loss: 0.6572 - top-5-accuracy: 0.9918

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.7504 - loss: 0.6568 - top-5-accuracy: 0.9923

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.7515 - loss: 0.6555 - top-5-accuracy: 0.9927

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.7522 - loss: 0.6545 - top-5-accuracy: 0.9930 - val_accuracy: 0.7764 - val_loss: 0.5790 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 17/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.6562 - loss: 0.7772 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.6953 - loss: 0.7204 - top-5-accuracy: 0.9920

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.7094 - loss: 0.6992 - top-5-accuracy: 0.9901

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.7168 - loss: 0.6894 - top-5-accuracy: 0.9892

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.7215 - loss: 0.6830 - top-5-accuracy: 0.9889

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.7257 - loss: 0.6776 - top-5-accuracy: 0.9890

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.7297 - loss: 0.6716 - top-5-accuracy: 0.9893

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.7330 - loss: 0.6654 - top-5-accuracy: 0.9896

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.7353 - loss: 0.6608 - top-5-accuracy: 0.9899 - val_accuracy: 0.7764 - val_loss: 0.6156 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 18/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.8125 - loss: 0.5664 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.7524 - loss: 0.6232 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.7563 - loss: 0.6155 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.7582 - loss: 0.6168 - top-5-accuracy: 0.9994

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.7597 - loss: 0.6172 - top-5-accuracy: 0.9991

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.7610 - loss: 0.6164 - top-5-accuracy: 0.9989

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.7608 - loss: 0.6160 - top-5-accuracy: 0.9986

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.7615 - loss: 0.6142 - top-5-accuracy: 0.9985

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.7623 - loss: 0.6117 - top-5-accuracy: 0.9984 - val_accuracy: 0.8199 - val_loss: 0.5801 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 19/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.8125 - loss: 0.6998 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8112 - loss: 0.5927 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8107 - loss: 0.5660 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8058 - loss: 0.5597 - top-5-accuracy: 0.9994

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.8031 - loss: 0.5587 - top-5-accuracy: 0.9986

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.8028 - loss: 0.5567 - top-5-accuracy: 0.9982

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.8029 - loss: 0.5536 - top-5-accuracy: 0.9981

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.8031 - loss: 0.5506 - top-5-accuracy: 0.9980

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8039 - loss: 0.5477 - top-5-accuracy: 0.9980 - val_accuracy: 0.8634 - val_loss: 0.3900 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 20/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.8125 - loss: 0.5183 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8170 - loss: 0.4787 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8231 - loss: 0.4663 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8221 - loss: 0.4679 - top-5-accuracy: 0.9998

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.8207 - loss: 0.4684 - top-5-accuracy: 0.9994

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.8202 - loss: 0.4681 - top-5-accuracy: 0.9992

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.8198 - loss: 0.4679 - top-5-accuracy: 0.9991

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.8196 - loss: 0.4677 - top-5-accuracy: 0.9991

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8196 - loss: 0.4676 - top-5-accuracy: 0.9991 - val_accuracy: 0.8820 - val_loss: 0.3949 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 21/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.8438 - loss: 0.3265 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8102 - loss: 0.4441 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8126 - loss: 0.4473 - top-5-accuracy: 0.9988

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8147 - loss: 0.4505 - top-5-accuracy: 0.9975

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.8180 - loss: 0.4484 - top-5-accuracy: 0.9968

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.8212 - loss: 0.4454 - top-5-accuracy: 0.9959

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.8242 - loss: 0.4418 - top-5-accuracy: 0.9955

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.8278 - loss: 0.4377 - top-5-accuracy: 0.9953

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8302 - loss: 0.4347 - top-5-accuracy: 0.9953 - val_accuracy: 0.8199 - val_loss: 0.4074 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 22/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.8438 - loss: 0.4054 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8555 - loss: 0.3918 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.8657 - loss: 0.3897 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8722 - loss: 0.3828 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.8719 - loss: 0.3809 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.8701 - loss: 0.3811 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.8696 - loss: 0.3797 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.8690 - loss: 0.3790 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8688 - loss: 0.3781 - top-5-accuracy: 1.0000 - val_accuracy: 0.8758 - val_loss: 0.3251 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 23/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.7500 - loss: 0.5183 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8441 - loss: 0.3849 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8504 - loss: 0.3760 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8581 - loss: 0.3641 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.8631 - loss: 0.3587 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.8660 - loss: 0.3555 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.8679 - loss: 0.3528 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.8701 - loss: 0.3494 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8718 - loss: 0.3463 - top-5-accuracy: 1.0000 - val_accuracy: 0.9068 - val_loss: 0.3203 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 24/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 0.7812 - loss: 0.4368 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8350 - loss: 0.3862 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8509 - loss: 0.3664 - top-5-accuracy: 0.9992

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8580 - loss: 0.3544 - top-5-accuracy: 0.9986

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.8637 - loss: 0.3442 - top-5-accuracy: 0.9984

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.8678 - loss: 0.3377 - top-5-accuracy: 0.9984

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.8716 - loss: 0.3316 - top-5-accuracy: 0.9985

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.8752 - loss: 0.3259 - top-5-accuracy: 0.9985

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8775 - loss: 0.3217 - top-5-accuracy: 0.9986 - val_accuracy: 0.8261 - val_loss: 0.5358 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 25/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.8438 - loss: 0.3862 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8757 - loss: 0.3136 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8732 - loss: 0.3119 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8742 - loss: 0.3105 - top-5-accuracy: 0.9994

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.8757 - loss: 0.3092 - top-5-accuracy: 0.9991

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.8772 - loss: 0.3077 - top-5-accuracy: 0.9989

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.8786 - loss: 0.3067 - top-5-accuracy: 0.9989

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.8803 - loss: 0.3039 - top-5-accuracy: 0.9989

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8815 - loss: 0.3012 - top-5-accuracy: 0.9989 - val_accuracy: 0.7888 - val_loss: 0.4613 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 26/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9062 - loss: 0.2728 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8956 - loss: 0.2971 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8959 - loss: 0.2913 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8954 - loss: 0.2938 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.8976 - loss: 0.2920 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.8998 - loss: 0.2885 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9015 - loss: 0.2845 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9026 - loss: 0.2821 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9037 - loss: 0.2798 - top-5-accuracy: 1.0000 - val_accuracy: 0.7826 - val_loss: 0.5074 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 27/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.7500 - loss: 0.5686 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8182 - loss: 0.4283 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8452 - loss: 0.3704 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8587 - loss: 0.3434 - top-5-accuracy: 0.9992

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.8684 - loss: 0.3257 - top-5-accuracy: 0.9989

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.8740 - loss: 0.3145 - top-5-accuracy: 0.9988

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.8779 - loss: 0.3062 - top-5-accuracy: 0.9988

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.8811 - loss: 0.2999 - top-5-accuracy: 0.9988

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.8838 - loss: 0.2950 - top-5-accuracy: 0.9988 - val_accuracy: 0.8696 - val_loss: 0.4785 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 28/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9062 - loss: 0.3785 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9001 - loss: 0.3009 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8993 - loss: 0.2806 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9021 - loss: 0.2703 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9061 - loss: 0.2601 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9087 - loss: 0.2544 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9112 - loss: 0.2493 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9135 - loss: 0.2448 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9152 - loss: 0.2415 - top-5-accuracy: 1.0000 - val_accuracy: 0.9193 - val_loss: 0.2811 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 29/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9062 - loss: 0.3467 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9096 - loss: 0.2743 - top-5-accuracy: 0.9972

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9154 - loss: 0.2506 - top-5-accuracy: 0.9965

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9209 - loss: 0.2345 - top-5-accuracy: 0.9968

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9248 - loss: 0.2234 - top-5-accuracy: 0.9970

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9271 - loss: 0.2175 - top-5-accuracy: 0.9973

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9286 - loss: 0.2136 - top-5-accuracy: 0.9975

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9298 - loss: 0.2106 - top-5-accuracy: 0.9977

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9307 - loss: 0.2082 - top-5-accuracy: 0.9978 - val_accuracy: 0.8385 - val_loss: 0.5072 - val_top-5-accuracy: 0.9876


<div class="k-default-codeblock">
```
Epoch 30/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.8750 - loss: 0.3561 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8907 - loss: 0.2798 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.8983 - loss: 0.2626 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9039 - loss: 0.2542 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9062 - loss: 0.2490 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9080 - loss: 0.2450 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9091 - loss: 0.2423 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9096 - loss: 0.2398 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9100 - loss: 0.2377 - top-5-accuracy: 1.0000 - val_accuracy: 0.8509 - val_loss: 0.4361 - val_top-5-accuracy: 0.9814


<div class="k-default-codeblock">
```
Epoch 31/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9688 - loss: 0.1638 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9191 - loss: 0.2347 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9115 - loss: 0.2373 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9133 - loss: 0.2308 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9161 - loss: 0.2252 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9172 - loss: 0.2229 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9179 - loss: 0.2211 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9184 - loss: 0.2197 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9190 - loss: 0.2181 - top-5-accuracy: 1.0000 - val_accuracy: 0.8571 - val_loss: 0.3413 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 32/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9062 - loss: 0.2091 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9384 - loss: 0.1787 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9441 - loss: 0.1666 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9476 - loss: 0.1569 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9500 - loss: 0.1504 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9514 - loss: 0.1464 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9527 - loss: 0.1428 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9534 - loss: 0.1405 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9537 - loss: 0.1393 - top-5-accuracy: 1.0000 - val_accuracy: 0.9130 - val_loss: 0.3189 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 33/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.8750 - loss: 0.2339 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9228 - loss: 0.1786 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9392 - loss: 0.1524 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9481 - loss: 0.1391 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9526 - loss: 0.1328 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9548 - loss: 0.1297 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9565 - loss: 0.1269 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9579 - loss: 0.1246 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9587 - loss: 0.1231 - top-5-accuracy: 1.0000 - val_accuracy: 0.9068 - val_loss: 0.3229 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 34/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0477 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9972 - loss: 0.0547 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9946 - loss: 0.0566 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9940 - loss: 0.0576 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9932 - loss: 0.0577 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9920 - loss: 0.0590 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9911 - loss: 0.0596 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9896 - loss: 0.0615 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9886 - loss: 0.0630 - top-5-accuracy: 1.0000 - val_accuracy: 0.8323 - val_loss: 0.5829 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 35/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9375 - loss: 0.2170 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9303 - loss: 0.2360 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9236 - loss: 0.2315 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9239 - loss: 0.2239 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9260 - loss: 0.2150 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9283 - loss: 0.2070 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9310 - loss: 0.1995 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9337 - loss: 0.1928 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9355 - loss: 0.1885 - top-5-accuracy: 1.0000 - val_accuracy: 0.9130 - val_loss: 0.3360 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 36/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 0.9688 - loss: 0.1454 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9299 - loss: 0.1882 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9321 - loss: 0.1850 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9348 - loss: 0.1802 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9369 - loss: 0.1748 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9380 - loss: 0.1728 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9388 - loss: 0.1704 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9399 - loss: 0.1671 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9407 - loss: 0.1651 - top-5-accuracy: 1.0000 - val_accuracy: 0.8820 - val_loss: 0.3123 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 37/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0788 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9815 - loss: 0.0998 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9752 - loss: 0.0981 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9747 - loss: 0.0931 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9754 - loss: 0.0887 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9760 - loss: 0.0854 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9769 - loss: 0.0824 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9777 - loss: 0.0798 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9781 - loss: 0.0780 - top-5-accuracy: 1.0000 - val_accuracy: 0.9068 - val_loss: 0.2738 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 38/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0428 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9851 - loss: 0.0585 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9833 - loss: 0.0629 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9808 - loss: 0.0665 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9789 - loss: 0.0690 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9778 - loss: 0.0715 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9774 - loss: 0.0727 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9774 - loss: 0.0733 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9775 - loss: 0.0733 - top-5-accuracy: 1.0000 - val_accuracy: 0.9068 - val_loss: 0.2941 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 39/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0610 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9829 - loss: 0.0826 - top-5-accuracy: 0.9987

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9747 - loss: 0.0943 - top-5-accuracy: 0.9974

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9729 - loss: 0.0947 - top-5-accuracy: 0.9974

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9721 - loss: 0.0939 - top-5-accuracy: 0.9975

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9722 - loss: 0.0927 - top-5-accuracy: 0.9977

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9725 - loss: 0.0912 - top-5-accuracy: 0.9978

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9727 - loss: 0.0897 - top-5-accuracy: 0.9980

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9731 - loss: 0.0884 - top-5-accuracy: 0.9981 - val_accuracy: 0.8882 - val_loss: 0.3096 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 40/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0547 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9972 - loss: 0.0548 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9920 - loss: 0.0578 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9896 - loss: 0.0588 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9886 - loss: 0.0581 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9884 - loss: 0.0567 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9883 - loss: 0.0557 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9885 - loss: 0.0546 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9887 - loss: 0.0539 - top-5-accuracy: 1.0000 - val_accuracy: 0.9130 - val_loss: 0.2928 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 41/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0351 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9907 - loss: 0.0423 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9902 - loss: 0.0440 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9907 - loss: 0.0433 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9910 - loss: 0.0427 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9909 - loss: 0.0424 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9908 - loss: 0.0424 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9908 - loss: 0.0423 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9908 - loss: 0.0423 - top-5-accuracy: 1.0000 - val_accuracy: 0.9130 - val_loss: 0.3551 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 42/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 0.9688 - loss: 0.0444 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9777 - loss: 0.0689 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9786 - loss: 0.0645 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9779 - loss: 0.0644 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9786 - loss: 0.0635 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9788 - loss: 0.0637 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9785 - loss: 0.0650 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9784 - loss: 0.0653 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9785 - loss: 0.0654 - top-5-accuracy: 1.0000 - val_accuracy: 0.8758 - val_loss: 0.3778 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 43/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 0.9375 - loss: 0.1588 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  4/31 ━━[37m━━━━━━━━━━━━━━━━━━  0s 17ms/step - accuracy: 0.9434 - loss: 0.1892 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  8/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9429 - loss: 0.1941 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 12/31 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9482 - loss: 0.1760 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 16/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9530 - loss: 0.1603 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 20/31 ━━━━━━━━━━━━[37m━━━━━━━━  0s 16ms/step - accuracy: 0.9567 - loss: 0.1481 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 24/31 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - accuracy: 0.9595 - loss: 0.1389 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 28/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9617 - loss: 0.1312 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9638 - loss: 0.1243 - top-5-accuracy: 1.0000 - val_accuracy: 0.8696 - val_loss: 0.4815 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 44/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9688 - loss: 0.0880 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9555 - loss: 0.0960 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9597 - loss: 0.0948 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9629 - loss: 0.0923 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9641 - loss: 0.0912 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9636 - loss: 0.0917 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9633 - loss: 0.0927 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9627 - loss: 0.0947 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9625 - loss: 0.0956 - top-5-accuracy: 1.0000 - val_accuracy: 0.9006 - val_loss: 0.3673 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 45/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0486 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9944 - loss: 0.0472 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9896 - loss: 0.0527 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9878 - loss: 0.0544 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9870 - loss: 0.0552 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9867 - loss: 0.0550 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9866 - loss: 0.0545 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9867 - loss: 0.0536 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9868 - loss: 0.0529 - top-5-accuracy: 1.0000 - val_accuracy: 0.9068 - val_loss: 0.3871 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 46/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 0.9375 - loss: 0.1102 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9715 - loss: 0.0650 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9795 - loss: 0.0536 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9833 - loss: 0.0472 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9855 - loss: 0.0437 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9870 - loss: 0.0411 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9882 - loss: 0.0391 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9891 - loss: 0.0374 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9896 - loss: 0.0366 - top-5-accuracy: 1.0000 - val_accuracy: 0.9193 - val_loss: 0.2612 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 47/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0238 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9895 - loss: 0.0342 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  8/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9828 - loss: 0.0524 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 12/31 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9804 - loss: 0.0645 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 16/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9769 - loss: 0.0749 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 20/31 ━━━━━━━━━━━━[37m━━━━━━━━  0s 16ms/step - accuracy: 0.9737 - loss: 0.0838 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 24/31 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - accuracy: 0.9705 - loss: 0.0913 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 28/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9681 - loss: 0.0960 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9663 - loss: 0.0994 - top-5-accuracy: 1.0000 - val_accuracy: 0.9068 - val_loss: 0.3378 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 48/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0290 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9786 - loss: 0.0742 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9719 - loss: 0.0799 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 0.9674 - loss: 0.0878 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9652 - loss: 0.0922 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9647 - loss: 0.0932 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9653 - loss: 0.0926 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9662 - loss: 0.0910 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9670 - loss: 0.0896 - top-5-accuracy: 1.0000 - val_accuracy: 0.9255 - val_loss: 0.3142 - val_top-5-accuracy: 0.9938


<div class="k-default-codeblock">
```
Epoch 49/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 1.0000 - loss: 0.0726 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0453 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0388 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0352 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 0.9998 - loss: 0.0328 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 0.9994 - loss: 0.0314 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9991 - loss: 0.0302 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9989 - loss: 0.0291 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9988 - loss: 0.0284 - top-5-accuracy: 1.0000 - val_accuracy: 0.9130 - val_loss: 0.3080 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 50/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0399 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9972 - loss: 0.0275 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9965 - loss: 0.0243 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9960 - loss: 0.0238 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9955 - loss: 0.0236 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9952 - loss: 0.0236 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9949 - loss: 0.0238 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9947 - loss: 0.0240 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9947 - loss: 0.0240 - top-5-accuracy: 1.0000 - val_accuracy: 0.8882 - val_loss: 0.3771 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 51/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0361 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9939 - loss: 0.0341 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9920 - loss: 0.0320 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9919 - loss: 0.0301 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9924 - loss: 0.0283 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9928 - loss: 0.0270 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9930 - loss: 0.0263 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9930 - loss: 0.0260 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9931 - loss: 0.0257 - top-5-accuracy: 1.0000 - val_accuracy: 0.9130 - val_loss: 0.3303 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 52/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0044 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9972 - loss: 0.0108 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9965 - loss: 0.0125 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 0.9968 - loss: 0.0127 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9970 - loss: 0.0125 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9973 - loss: 0.0121 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - accuracy: 0.9975 - loss: 0.0118 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 0.9977 - loss: 0.0115 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9978 - loss: 0.0113 - top-5-accuracy: 1.0000 - val_accuracy: 0.9193 - val_loss: 0.2734 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 53/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0094 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0078 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0070 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0065 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0063 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0064 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0064 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0064 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0063 - top-5-accuracy: 1.0000 - val_accuracy: 0.9068 - val_loss: 0.2992 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 54/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0041 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0060 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0057 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0053 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - accuracy: 0.9999 - loss: 0.0052 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - accuracy: 0.9996 - loss: 0.0053 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 0.9995 - loss: 0.0053 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 0.9992 - loss: 0.0056 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9991 - loss: 0.0057 - top-5-accuracy: 1.0000 - val_accuracy: 0.9006 - val_loss: 0.2946 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 55/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 32ms/step - accuracy: 1.0000 - loss: 0.0120 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0093 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0088 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0082 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0077 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0073 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0071 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0069 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0067 - top-5-accuracy: 1.0000 - val_accuracy: 0.9317 - val_loss: 0.2697 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 56/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0027 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0029 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0029 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0029 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0030 - top-5-accuracy: 1.0000 - val_accuracy: 0.9130 - val_loss: 0.3177 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 57/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 30ms/step - accuracy: 1.0000 - loss: 0.0035 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0045 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0041 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0038 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0036 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0035 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0033 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0033 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0032 - top-5-accuracy: 1.0000 - val_accuracy: 0.9255 - val_loss: 0.2838 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 58/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0024 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0022 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0021 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0020 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0020 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0020 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0020 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0019 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0019 - top-5-accuracy: 1.0000 - val_accuracy: 0.9255 - val_loss: 0.2952 - val_top-5-accuracy: 1.0000


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
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0016 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0018 - top-5-accuracy: 1.0000 - val_accuracy: 0.9193 - val_loss: 0.2972 - val_top-5-accuracy: 1.0000


<div class="k-default-codeblock">
```
Epoch 60/60

```
</div>
    
  1/31 [37m━━━━━━━━━━━━━━━━━━━━  0s 31ms/step - accuracy: 1.0000 - loss: 0.0026 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  5/31 ━━━[37m━━━━━━━━━━━━━━━━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0019 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  9/31 ━━━━━[37m━━━━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/31 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0017 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 17/31 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0016 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 21/31 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0016 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 25/31 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - accuracy: 1.0000 - loss: 0.0016 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 29/31 ━━━━━━━━━━━━━━━━━━[37m━━  0s 15ms/step - accuracy: 1.0000 - loss: 0.0016 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 1.0000 - loss: 0.0016 - top-5-accuracy: 1.0000 - val_accuracy: 0.9193 - val_loss: 0.2977 - val_top-5-accuracy: 1.0000


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  0s 33ms/step - accuracy: 0.7188 - loss: 1.2798 - top-5-accuracy: 1.0000

<div class="k-default-codeblock">
```

```
</div>
 13/20 ━━━━━━━━━━━━━[37m━━━━━━━  0s 4ms/step - accuracy: 0.7656 - loss: 1.0413 - top-5-accuracy: 0.9821 

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 104ms/step - accuracy: 0.7658 - loss: 1.0333 - top-5-accuracy: 0.9802

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 2s 104ms/step - accuracy: 0.7659 - loss: 1.0316 - top-5-accuracy: 0.9800


<div class="k-default-codeblock">
```
Test accuracy: 76.72%
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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 5s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 5s 5s/step


    
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
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step


    
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


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step

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
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step





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

# Distilling Vision Transformers

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2022/04/05<br>
**Last modified:** 2026/03/11<br>
**Description:** Distillation of Vision Transformers through attention.

## Introduction

In the original *Vision Transformers* (ViT) paper
([Dosovitskiy et al.](https://arxiv.org/abs/2010.11929)),
the authors concluded that to perform on par with Convolutional Neural Networks (CNNs),
ViTs need to be pre-trained on larger datasets. The larger the better. This is mainly
due to the lack of inductive biases in the ViT architecture -- unlike CNNs,
they don't have layers that exploit locality. In a follow-up paper
([Steiner et al.](https://arxiv.org/abs/2106.10270)),
the authors show that it is possible to substantially improve the performance of ViTs
with stronger regularization and longer training.

Many groups have proposed different ways to deal with the problem
of data-intensiveness of ViT training.
One such way was shown in the *Data-efficient image Transformers*,
(DeiT) paper ([Touvron et al.](https://arxiv.org/abs/2012.12877)). The
authors introduced a distillation technique that is specific to transformer-based vision
models. DeiT is among the first works to show that it's possible to train ViTs well
without using larger datasets.

In this example, we implement the distillation recipe proposed in DeiT. This
requires us to slightly tweak the original ViT architecture and write a custom training
loop to implement the distillation recipe.

To comfortably navigate through this example, you'll be expected to know how a ViT and
knowledge distillation work. The following are good resources in case you needed a
refresher:

* [ViT on keras.io](https://keras.io/examples/vision/image_classification_with_vision_transformer)
* [Knowledge distillation on keras.io](https://keras.io/examples/vision/knowledge_distillation/)

## Imports


```python
from pathlib import Path
from typing import List

import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(42)
```

    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1778696636.585494    2411 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1778696636.592069    2411 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1778696636.608151    2411 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1778696636.608181    2411 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1778696636.608183    2411 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1778696636.608184    2411 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.


## Constants


```python
# Model
MODEL_TYPE = "deit_distilled_tiny_patch16_224"
RESOLUTION = 224
PATCH_SIZE = 16
NUM_PATCHES = (RESOLUTION // PATCH_SIZE) ** 2
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 192
NUM_HEADS = 3
NUM_LAYERS = 12
MLP_UNITS = [PROJECTION_DIM * 4, PROJECTION_DIM]
DROPOUT_RATE = 0.0
DROP_PATH_RATE = 0.1

# Training
NUM_EPOCHS = 20
BASE_LR = 0.0005
WEIGHT_DECAY = 0.0001

# Data
BATCH_SIZE = 256
NUM_CLASSES = 5
```

You probably noticed that `DROPOUT_RATE` has been set 0.0. Dropout has been used
in the implementation to keep it complete. For smaller models (like the one used in
this example), you don't need it, but for bigger models, using dropout helps.

## Load the flowers dataset and prepare preprocessing utilities

The authors use an array of different augmentation techniques, including MixUp
([Zhang et al.](https://arxiv.org/abs/1710.09412)),
RandAugment ([Cubuk et al.](https://arxiv.org/abs/1909.13719)),
and so on. However, to keep the example simple to work through, we'll discard them.

We use `keras.utils.PyDataset` to build a fully backend-agnostic data pipeline that
works with JAX, PyTorch, and TensorFlow alike.

A couple of practical details are important here:

* `keras.utils.get_file(untar=True)` may return the extraction cache directory, so we
    explicitly resolve the inner `flower_photos/` folder when present.
* Source images have variable spatial sizes, so we decode each image with a fixed
    `target_size` before stacking into a NumPy batch.


```python
FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


class FlowersDataset(keras.utils.PyDataset):
    """Backend-agnostic flowers dataset that loads images from disk each epoch."""

    def __init__(
        self,
        image_paths,
        labels,
        augmenter,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels, dtype="int32")
        self.augmenter = augmenter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_indices = self.indices[start:end]
        images = []
        for i in batch_indices:
            target_size = (
                (RESOLUTION + 20, RESOLUTION + 20)
                if self.shuffle
                else (RESOLUTION, RESOLUTION)
            )
            image = keras.utils.load_img(self.image_paths[i], target_size=target_size)
            images.append(keras.utils.img_to_array(image))
        images = np.array(images, dtype="float32")
        if self.augmenter is not None:
            images = self.augmenter(images, training=self.shuffle)
        labels = keras.ops.one_hot(self.labels[batch_indices], num_classes=NUM_CLASSES)
        return images, labels


def get_augmenter(is_training=True):
    if is_training:
        return keras.Sequential(
            [
                layers.RandomCrop(RESOLUTION, RESOLUTION),
                layers.RandomFlip("horizontal"),
            ],
            name="train_augmentation",
        )
    return None


def load_flower_file_paths(validation_split=0.1):
    extracted = Path(keras.utils.get_file(origin=FLOWERS_URL, untar=True))
    data_dir = (
        extracted / "flower_photos"
        if (extracted / "flower_photos").is_dir()
        else extracted
    )
    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    rng = np.random.default_rng(42)
    for class_name in class_names:
        class_files = sorted((data_dir / class_name).glob("*.jpg"))
        class_files = np.array([str(path) for path in class_files])
        rng.shuffle(class_files)
        num_val = int(len(class_files) * validation_split)
        val_paths.extend(class_files[:num_val])
        val_labels.extend([class_to_index[class_name]] * num_val)
        train_paths.extend(class_files[num_val:])
        train_labels.extend([class_to_index[class_name]] * (len(class_files) - num_val))
    return train_paths, train_labels, val_paths, val_labels


train_paths, train_labels, val_paths, val_labels = load_flower_file_paths()
print(f"Number of training examples: {len(train_paths)}")
print(f"Number of validation examples: {len(val_paths)}")

train_dataset = FlowersDataset(
    train_paths,
    train_labels,
    augmenter=get_augmenter(is_training=True),
    shuffle=True,
    workers=4,
)
val_dataset = FlowersDataset(
    val_paths,
    val_labels,
    augmenter=get_augmenter(is_training=False),
    shuffle=False,
    workers=4,
)
```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz


    [1m        0/228813984[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 0s/step

    [1m  8396800/228813984[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m2s[0m 0us/step

    [1m 16785408/228813984[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m 25157632/228813984[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m 31784960/228813984[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m 39862272/228813984[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m 50053120/228813984[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m 53944320/228813984[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m 58777600/228813984[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m 67117056/228813984[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m 72622080/228813984[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m 80412672/228813984[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m 92102656/228813984[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m100671488/228813984[0m [32m━━━━━━━━[0m[37m━━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m110772224/228813984[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m1s[0m 0us/step

    [1m121856000/228813984[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 0us/step

    [1m130457600/228813984[0m [32m━━━━━━━━━━━[0m[37m━━━━━━━━━[0m [1m0s[0m 0us/step

    [1m134955008/228813984[0m [32m━━━━━━━━━━━[0m[37m━━━━━━━━━[0m [1m0s[0m 0us/step

    [1m139476992/228813984[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 0us/step

    [1m143564800/228813984[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 0us/step

    [1m146833408/228813984[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 0us/step

    [1m148684800/228813984[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 0us/step

    [1m152199168/228813984[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 0us/step

    [1m153026560/228813984[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 0us/step

    [1m155246592/228813984[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 0us/step

    [1m156377088/228813984[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 0us/step

    [1m157384704/228813984[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 0us/step

    [1m157466624/228813984[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 0us/step

    [1m158457856/228813984[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 0us/step

    [1m161406976/228813984[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 0us/step

    [1m166551552/228813984[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 0us/step

    [1m171532288/228813984[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m0s[0m 0us/step

    [1m179773440/228813984[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 0us/step

    [1m184557568/228813984[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 0us/step

    [1m192995328/228813984[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 0us/step

    [1m201334784/228813984[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 0us/step

    [1m214048768/228813984[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 0us/step

    [1m218112000/228813984[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 0us/step

    [1m228343808/228813984[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 0us/step

    [1m228813984/228813984[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 0us/step


    Number of training examples: 3306
    Number of validation examples: 364


    I0000 00:00:1778696644.640571    2411 gpu_device.cc:2019] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38482 MB memory:  -> device: 0, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:00:04.0, compute capability: 8.0


## Implementing the DeiT variants of ViT

Since DeiT is an extension of ViT it'd make sense to first implement ViT and then extend
it to support DeiT's components.

First, we'll implement a layer for Stochastic Depth
([Huang et al.](https://arxiv.org/abs/1603.09382))
which is used in DeiT for regularization.


```python

# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, x, training=True):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(keras.ops.shape(x)) - 1)
            random_tensor = keep_prob + keras.random.uniform(
                shape, 0, 1, seed=self.seed_generator
            )
            random_tensor = keras.ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

```

Now, we'll implement the MLP and Transformer blocks.


```python

def mlp(x, dropout_rate: float, hidden_units: List):
    """FFN for a Transformer block."""
    for idx, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="gelu" if idx == 0 else None)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer(drop_prob: float, name: str) -> keras.Model:
    """Transformer block with pre-norm."""
    num_patches = NUM_PATCHES + 2 if "distilled" in MODEL_TYPE else NUM_PATCHES + 1
    encoded_patches = layers.Input((num_patches, PROJECTION_DIM))
    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=DROPOUT_RATE
    )(x1, x1)
    attention_output = (
        StochasticDepth(drop_prob)(attention_output) if drop_prob else attention_output
    )
    x2 = layers.Add()([attention_output, encoded_patches])
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)
    x4 = mlp(x3, hidden_units=MLP_UNITS, dropout_rate=DROPOUT_RATE)
    x4 = StochasticDepth(drop_prob)(x4) if drop_prob else x4
    outputs = layers.Add()([x2, x4])
    return keras.Model(encoded_patches, outputs, name=name)

```

We'll now implement a `ViTClassifier` class building on top of the components we just
developed. Here we'll be following the original pooling strategy used in the ViT paper --
use a class token and use the feature representations corresponding to it for
classification.


```python

class ViTClassifier(keras.Model):
    """Vision Transformer base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.projection = keras.Sequential(
            [
                layers.Conv2D(
                    filters=PROJECTION_DIM,
                    kernel_size=(PATCH_SIZE, PATCH_SIZE),
                    strides=(PATCH_SIZE, PATCH_SIZE),
                    padding="VALID",
                    name="conv_projection",
                ),
                layers.Reshape(
                    target_shape=(NUM_PATCHES, PROJECTION_DIM),
                    name="flatten_projection",
                ),
            ],
            name="projection",
        )
        dpr = [x for x in keras.ops.linspace(0.0, DROP_PATH_RATE, NUM_LAYERS)]
        self.transformer_blocks = [
            transformer(drop_prob=dpr[i], name=f"transformer_block_{i}")
            for i in range(NUM_LAYERS)
        ]
        self.dropout = layers.Dropout(DROPOUT_RATE)
        self.layer_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)
        self.head = layers.Dense(NUM_CLASSES, name="classification_head")

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            shape=(1, NUM_PATCHES + 1, PROJECTION_DIM),
            initializer=keras.initializers.Zeros(),
            trainable=True,
            name="pos_embedding",
        )
        self.cls_token = self.add_weight(
            shape=(1, 1, PROJECTION_DIM),
            initializer=keras.initializers.Zeros(),
            trainable=True,
            name="cls",
        )
        super().build(input_shape)

    def call(self, inputs, training=True):
        n = keras.ops.shape(inputs)[0]
        projected_patches = self.projection(inputs)
        cls_token = keras.ops.tile(self.cls_token, (n, 1, 1))
        cls_token = keras.ops.cast(cls_token, projected_patches.dtype)
        projected_patches = keras.ops.concatenate(
            [cls_token, projected_patches], axis=1
        )
        encoded_patches = self.positional_embedding + projected_patches
        encoded_patches = self.dropout(encoded_patches)
        for transformer_module in self.transformer_blocks:
            encoded_patches = transformer_module(encoded_patches)
        representation = self.layer_norm(encoded_patches)
        encoded_patches = representation[:, 0]
        output = self.head(encoded_patches)
        return output

```

This class can be used standalone as ViT and is end-to-end trainable. Just remove the
`distilled` phrase in `MODEL_TYPE` and it should work with `vit_tiny = ViTClassifier()`.
Let's now extend it to DeiT. The following figure presents the schematic of DeiT (taken
from the DeiT paper):

![](https://i.imgur.com/5lmg2Xs.png)

Apart from the class token, DeiT has another token for distillation. During distillation,
the logits corresponding to the class token are compared to the true labels, and the
logits corresponding to the distillation token are compared to the teacher's predictions.


```python

class ViTDistilled(ViTClassifier):
    def __init__(self, regular_training=False, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = 2
        self.regular_training = regular_training
        self.head = layers.Dense(NUM_CLASSES, name="classification_head")
        self.head_dist = layers.Dense(NUM_CLASSES, name="distillation_head")

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, PROJECTION_DIM),
            initializer=keras.initializers.Zeros(),
            trainable=True,
            name="cls",
        )
        self.dist_token = self.add_weight(
            shape=(1, 1, PROJECTION_DIM),
            initializer=keras.initializers.Zeros(),
            trainable=True,
            name="dist_token",
        )
        self.positional_embedding = self.add_weight(
            shape=(1, NUM_PATCHES + self.num_tokens, PROJECTION_DIM),
            initializer=keras.initializers.Zeros(),
            trainable=True,
            name="pos_embedding",
        )

    def call(self, inputs, training=True):
        n = keras.ops.shape(inputs)[0]
        projected_patches = self.projection(inputs)
        cls_token = keras.ops.tile(self.cls_token, (n, 1, 1))
        dist_token = keras.ops.tile(self.dist_token, (n, 1, 1))
        cls_token = keras.ops.cast(cls_token, projected_patches.dtype)
        dist_token = keras.ops.cast(dist_token, projected_patches.dtype)
        projected_patches = keras.ops.concatenate(
            [cls_token, dist_token, projected_patches], axis=1
        )
        encoded_patches = self.positional_embedding + projected_patches
        encoded_patches = self.dropout(encoded_patches)
        for transformer_module in self.transformer_blocks:
            encoded_patches = transformer_module(encoded_patches)
        representation = self.layer_norm(encoded_patches)
        x, x_dist = (
            self.head(representation[:, 0]),
            self.head_dist(representation[:, 1]),
        )
        if training and not self.regular_training:
            return x, x_dist
        return (x + x_dist) / 2

```

Let's verify if the `ViTDistilled` class can be initialized and called as expected.


```python
deit_tiny_distilled = ViTDistilled()

dummy_inputs = keras.ops.ones((2, 224, 224, 3))
outputs = deit_tiny_distilled(dummy_inputs, training=False)
print(outputs.shape)
```

    I0000 00:00:1778696648.330183    2411 cuda_dnn.cc:529] Loaded cuDNN version 92000


    (2, 5)


## Implementing the trainer

Unlike what happens in standard knowledge distillation
([Hinton et al.](https://arxiv.org/abs/1503.02531)),
where a temperature-scaled softmax is used as well as KL divergence,
DeiT authors use the following loss function:

![](https://i.imgur.com/bXdxsBq.png)


Here,

* CE is cross-entropy
* `psi` is the softmax function
* Z_s denotes student predictions
* y denotes true labels
* y_t denotes teacher predictions


```python

class DeiT(keras.Model):
    # Reference: https://keras.io/examples/vision/knowledge_distillation/
    def __init__(self, student, teacher, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.dist_loss_tracker = keras.metrics.Mean(name="distillation_loss")
        self.accuracy_metric = keras.metrics.CategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.student_loss_tracker)
        metrics.append(self.dist_loss_tracker)
        metrics.append(self.accuracy_metric)
        return metrics

    def compile(self, optimizer, student_loss_fn, distillation_loss_fn):
        super().compile(optimizer=optimizer)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        x_normalized = keras.ops.cast(x, "float32") / 255.0
        cls_predictions, dist_predictions = self.student(x_normalized, training=True)
        teacher_logits = self.teacher(keras.ops.cast(x, "float32"), training=False)
        teacher_predictions = keras.ops.softmax(teacher_logits, axis=-1)
        student_loss = self.student_loss_fn(y, cls_predictions)
        distillation_loss = self.distillation_loss_fn(
            teacher_predictions, dist_predictions
        )
        self.student_loss_tracker.update_state(student_loss)
        self.dist_loss_tracker.update_state(distillation_loss)
        student_predictions = (cls_predictions + dist_predictions) / 2
        self.accuracy_metric.update_state(y, student_predictions)
        return (student_loss + distillation_loss) / 2

    def test_step(self, data):
        x, y = data
        x_normalized = keras.ops.cast(x, "float32") / 255.0
        y_prediction = self.student(x_normalized, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)
        self.accuracy_metric.update_state(y, y_prediction)
        self.student_loss_tracker.update_state(student_loss)
        return {
            "loss": student_loss,
            "student_loss": self.student_loss_tracker.result(),
            "accuracy": self.accuracy_metric.result(),
        }

    def call(self, inputs, training=False):
        inputs_normalized = keras.ops.cast(inputs, "float32") / 255.0
        return self.student(inputs_normalized, training=False)

```

## Build a teacher model

For full backend portability in Keras 3, we build a teacher with standard Keras layers
instead of using a TensorFlow-only SavedModel loader. We use `EfficientNetV2B0` (pretrained on
ImageNet) as the backbone, freeze it, and fine-tune only a small classification head on
the flowers dataset. In practice you could swap in any compatible Keras model as the
teacher.

`EfficientNetV2B0` includes preprocessing by default (`include_preprocessing=True`),
so it expects raw `[0, 255]` image values. We therefore avoid adding an extra
`Rescaling(1/255)` layer in the teacher path to prevent double normalization.


```python
teacher_backbone = keras.applications.EfficientNetV2B0(
    include_top=False, pooling="avg", weights="imagenet"
)
teacher_backbone.trainable = False
teacher_model = keras.Sequential(
    [teacher_backbone, layers.Dense(NUM_CLASSES)], name="teacher"
)
teacher_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
)

print("Fine-tuning teacher head on flowers dataset...")
teacher_model.fit(train_dataset, validation_data=val_dataset, epochs=5)
teacher_model.trainable = False
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-b0_notop.h5


    [1m       0/24274472[0m [37m━━━━━━━━━━━━━━━━━━━━[0m [1m0s[0m 0s/step

    [1m15777792/24274472[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 0us/step

    [1m24274472/24274472[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step


    Fine-tuning teacher head on flowers dataset...


    Epoch 1/5


    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1778696668.253681    2467 service.cc:152] XLA service 0x7fddf4004780 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    I0000 00:00:1778696668.253740    2467 service.cc:160]   StreamExecutor device (0): NVIDIA A100-SXM4-40GB, Compute Capability 8.0


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m8:44[0m 44s/step - accuracy: 0.2734 - loss: 1.6129

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m1s[0m 97ms/step - accuracy: 0.2812 - loss: 1.5813 

    I0000 00:00:1778696697.640091    2467 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.


    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m0s[0m 97ms/step - accuracy: 0.2982 - loss: 1.5552

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 96ms/step - accuracy: 0.3188 - loss: 1.5324

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m0s[0m 96ms/step - accuracy: 0.3407 - loss: 1.5094

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m0s[0m 96ms/step - accuracy: 0.3605 - loss: 1.4880

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m0s[0m 96ms/step - accuracy: 0.3797 - loss: 1.4668

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 105ms/step - accuracy: 0.3974 - loss: 1.4470

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 104ms/step - accuracy: 0.4137 - loss: 1.4280

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 103ms/step - accuracy: 0.4286 - loss: 1.4101

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 102ms/step - accuracy: 0.4422 - loss: 1.3930

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 101ms/step - accuracy: 0.4546 - loss: 1.3765

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3s/step - accuracy: 0.4661 - loss: 1.3612   

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m100s[0m 5s/step - accuracy: 0.6031 - loss: 1.1777 - val_accuracy: 0.7363 - val_loss: 0.8961


    Epoch 2/5


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 1s/step - accuracy: 0.8086 - loss: 0.8199

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m1s[0m 104ms/step - accuracy: 0.7979 - loss: 0.8258

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m1s[0m 101ms/step - accuracy: 0.7893 - loss: 0.8273

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 100ms/step - accuracy: 0.7841 - loss: 0.8229

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 264ms/step - accuracy: 0.7802 - loss: 0.8179

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m1s[0m 231ms/step - accuracy: 0.7790 - loss: 0.8122

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 209ms/step - accuracy: 0.7780 - loss: 0.8075

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 193ms/step - accuracy: 0.7784 - loss: 0.8022

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 249ms/step - accuracy: 0.7790 - loss: 0.7973

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 242ms/step - accuracy: 0.7798 - loss: 0.7927

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 227ms/step - accuracy: 0.7805 - loss: 0.7882

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 215ms/step - accuracy: 0.7813 - loss: 0.7840

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 233ms/step - accuracy: 0.7822 - loss: 0.7795

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 303ms/step - accuracy: 0.7919 - loss: 0.7255 - val_accuracy: 0.8049 - val_loss: 0.6591


    Epoch 3/5


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m13s[0m 1s/step - accuracy: 0.8438 - loss: 0.5781

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m1s[0m 102ms/step - accuracy: 0.8486 - loss: 0.5634

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m0s[0m 98ms/step - accuracy: 0.8450 - loss: 0.5656 

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 100ms/step - accuracy: 0.8443 - loss: 0.5651

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 262ms/step - accuracy: 0.8450 - loss: 0.5625

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m1s[0m 230ms/step - accuracy: 0.8447 - loss: 0.5619

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 209ms/step - accuracy: 0.8446 - loss: 0.5616

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 193ms/step - accuracy: 0.8449 - loss: 0.5602

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 261ms/step - accuracy: 0.8449 - loss: 0.5594

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 242ms/step - accuracy: 0.8449 - loss: 0.5582

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 227ms/step - accuracy: 0.8446 - loss: 0.5572

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 215ms/step - accuracy: 0.8446 - loss: 0.5560

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 233ms/step - accuracy: 0.8446 - loss: 0.5550

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 308ms/step - accuracy: 0.8436 - loss: 0.5433 - val_accuracy: 0.8462 - val_loss: 0.5433


    Epoch 4/5


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m13s[0m 1s/step - accuracy: 0.8632 - loss: 0.4643

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m1s[0m 100ms/step - accuracy: 0.8541 - loss: 0.4770

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m1s[0m 100ms/step - accuracy: 0.8549 - loss: 0.4799

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 101ms/step - accuracy: 0.8590 - loss: 0.4774

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 265ms/step - accuracy: 0.8603 - loss: 0.4775

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m1s[0m 232ms/step - accuracy: 0.8615 - loss: 0.4765

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 213ms/step - accuracy: 0.8619 - loss: 0.4760

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 198ms/step - accuracy: 0.8623 - loss: 0.4754

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 243ms/step - accuracy: 0.8629 - loss: 0.4743

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 227ms/step - accuracy: 0.8637 - loss: 0.4728

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 235ms/step - accuracy: 0.8648 - loss: 0.4711

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 222ms/step - accuracy: 0.8654 - loss: 0.4698

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 234ms/step - accuracy: 0.8659 - loss: 0.4685

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 309ms/step - accuracy: 0.8724 - loss: 0.4523 - val_accuracy: 0.8626 - val_loss: 0.4762


    Epoch 5/5


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m13s[0m 1s/step - accuracy: 0.8906 - loss: 0.3990

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m1s[0m 103ms/step - accuracy: 0.8906 - loss: 0.4021

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m0s[0m 97ms/step - accuracy: 0.8900 - loss: 0.4043 

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 98ms/step - accuracy: 0.8901 - loss: 0.4053

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m1s[0m 248ms/step - accuracy: 0.8911 - loss: 0.4036

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m1s[0m 218ms/step - accuracy: 0.8913 - loss: 0.4036

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 210ms/step - accuracy: 0.8916 - loss: 0.4033

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m0s[0m 196ms/step - accuracy: 0.8918 - loss: 0.4028

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m0s[0m 249ms/step - accuracy: 0.8918 - loss: 0.4029

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 234ms/step - accuracy: 0.8917 - loss: 0.4030

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 222ms/step - accuracy: 0.8916 - loss: 0.4028

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 221ms/step - accuracy: 0.8915 - loss: 0.4025

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 233ms/step - accuracy: 0.8914 - loss: 0.4022

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 309ms/step - accuracy: 0.8905 - loss: 0.3986 - val_accuracy: 0.8764 - val_loss: 0.4336


## Training through distillation


```python
deit_tiny = ViTDistilled()
deit_distiller = DeiT(student=deit_tiny, teacher=teacher_model)
lr_scaled = (BASE_LR / 512) * BATCH_SIZE
deit_distiller.compile(
    optimizer=keras.optimizers.AdamW(
        weight_decay=WEIGHT_DECAY, learning_rate=lr_scaled
    ),
    student_loss_fn=keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1
    ),
    distillation_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
)
_ = deit_distiller.fit(train_dataset, validation_data=val_dataset, epochs=NUM_EPOCHS)
```

    Epoch 1/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m20:54[0m 105s/step - accuracy: 0.1836 - distillation_loss: 1.9889 - loss: 1.9450 - student_loss: 1.9012

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m9:36[0m 52s/step - accuracy: 0.1979 - distillation_loss: 2.3747 - loss: 2.3051 - student_loss: 2.2693  

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m4:23[0m 26s/step - accuracy: 0.2012 - distillation_loss: 2.6459 - loss: 2.4954 - student_loss: 2.3784

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2:38[0m 18s/step - accuracy: 0.2038 - distillation_loss: 2.7628 - loss: 2.5658 - student_loss: 2.4013

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m1:46[0m 13s/step - accuracy: 0.2056 - distillation_loss: 2.7913 - loss: 2.5721 - student_loss: 2.3848

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m1:14[0m 11s/step - accuracy: 0.2075 - distillation_loss: 2.7797 - loss: 2.5541 - student_loss: 2.3596

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m53s[0m 9s/step - accuracy: 0.2090 - distillation_loss: 2.7528 - loss: 2.5269 - student_loss: 2.3315  

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m38s[0m 8s/step - accuracy: 0.2114 - distillation_loss: 2.7198 - loss: 2.4958 - student_loss: 2.3018

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m27s[0m 7s/step - accuracy: 0.2138 - distillation_loss: 2.6844 - loss: 2.4640 - student_loss: 2.2728

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m18s[0m 6s/step - accuracy: 0.2161 - distillation_loss: 2.6491 - loss: 2.4333 - student_loss: 2.2460

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m10s[0m 5s/step - accuracy: 0.2178 - distillation_loss: 2.6149 - loss: 2.4043 - student_loss: 2.2216

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m4s[0m 5s/step - accuracy: 0.2186 - distillation_loss: 2.5825 - loss: 2.3773 - student_loss: 2.1992 

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5s/step - accuracy: 0.2191 - distillation_loss: 2.5516 - loss: 2.3518 - student_loss: 2.1785

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m189s[0m 7s/step - accuracy: 0.2257 - distillation_loss: 2.1809 - loss: 2.0463 - student_loss: 1.9302 - val_accuracy: 0.1896 - val_loss: 1.4346 - val_student_loss: 1.5736


    Epoch 2/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m16s[0m 1s/step - accuracy: 0.1992 - distillation_loss: 1.6603 - loss: 1.6378 - student_loss: 1.6152

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 248ms/step - accuracy: 0.2100 - distillation_loss: 1.6578 - loss: 1.6335 - student_loss: 1.6091

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m3s[0m 382ms/step - accuracy: 0.2064 - distillation_loss: 1.6565 - loss: 1.6316 - student_loss: 1.6067

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m3s[0m 338ms/step - accuracy: 0.2036 - distillation_loss: 1.6547 - loss: 1.6302 - student_loss: 1.6057

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 318ms/step - accuracy: 0.2024 - distillation_loss: 1.6525 - loss: 1.6289 - student_loss: 1.6054

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 304ms/step - accuracy: 0.2009 - distillation_loss: 1.6502 - loss: 1.6279 - student_loss: 1.6056

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 328ms/step - accuracy: 0.2009 - distillation_loss: 1.6481 - loss: 1.6271 - student_loss: 1.6062

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 323ms/step - accuracy: 0.2019 - distillation_loss: 1.6461 - loss: 1.6264 - student_loss: 1.6067

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 315ms/step - accuracy: 0.2029 - distillation_loss: 1.6445 - loss: 1.6259 - student_loss: 1.6073

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 306ms/step - accuracy: 0.2043 - distillation_loss: 1.6429 - loss: 1.6253 - student_loss: 1.6077

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 304ms/step - accuracy: 0.2057 - distillation_loss: 1.6415 - loss: 1.6248 - student_loss: 1.6080

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 298ms/step - accuracy: 0.2068 - distillation_loss: 1.6406 - loss: 1.6245 - student_loss: 1.6085

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 292ms/step - accuracy: 0.2077 - distillation_loss: 1.6398 - loss: 1.6244 - student_loss: 1.6089

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 367ms/step - accuracy: 0.2184 - distillation_loss: 1.6308 - loss: 1.6227 - student_loss: 1.6147 - val_accuracy: 0.2445 - val_loss: 1.5118 - val_student_loss: 1.5775


    Epoch 3/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m16s[0m 1s/step - accuracy: 0.2070 - distillation_loss: 1.6183 - loss: 1.6180 - student_loss: 1.6176

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 244ms/step - accuracy: 0.2021 - distillation_loss: 1.6149 - loss: 1.6162 - student_loss: 1.6175

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m3s[0m 302ms/step - accuracy: 0.2098 - distillation_loss: 1.6128 - loss: 1.6144 - student_loss: 1.6158

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 299ms/step - accuracy: 0.2128 - distillation_loss: 1.6122 - loss: 1.6133 - student_loss: 1.6143

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 306ms/step - accuracy: 0.2130 - distillation_loss: 1.6123 - loss: 1.6132 - student_loss: 1.6138

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 294ms/step - accuracy: 0.2126 - distillation_loss: 1.6123 - loss: 1.6130 - student_loss: 1.6135

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 287ms/step - accuracy: 0.2113 - distillation_loss: 1.6126 - loss: 1.6131 - student_loss: 1.6134

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 303ms/step - accuracy: 0.2100 - distillation_loss: 1.6126 - loss: 1.6130 - student_loss: 1.6132

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 325ms/step - accuracy: 0.2093 - distillation_loss: 1.6126 - loss: 1.6129 - student_loss: 1.6130

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 316ms/step - accuracy: 0.2093 - distillation_loss: 1.6123 - loss: 1.6126 - student_loss: 1.6126

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 308ms/step - accuracy: 0.2093 - distillation_loss: 1.6121 - loss: 1.6124 - student_loss: 1.6124

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 306ms/step - accuracy: 0.2094 - distillation_loss: 1.6119 - loss: 1.6121 - student_loss: 1.6122

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 301ms/step - accuracy: 0.2098 - distillation_loss: 1.6116 - loss: 1.6119 - student_loss: 1.6119

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 378ms/step - accuracy: 0.2151 - distillation_loss: 1.6084 - loss: 1.6084 - student_loss: 1.6084 - val_accuracy: 0.2445 - val_loss: 1.5851 - val_student_loss: 1.6011


    Epoch 4/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m13s[0m 1s/step - accuracy: 0.2479 - distillation_loss: 1.6048 - loss: 1.6063 - student_loss: 1.6078

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m3s[0m 284ms/step - accuracy: 0.2556 - distillation_loss: 1.6004 - loss: 1.6021 - student_loss: 1.6041

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m2s[0m 265ms/step - accuracy: 0.2544 - distillation_loss: 1.6016 - loss: 1.6031 - student_loss: 1.6048

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 272ms/step - accuracy: 0.2527 - distillation_loss: 1.6029 - loss: 1.6038 - student_loss: 1.6049

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 301ms/step - accuracy: 0.2525 - distillation_loss: 1.6031 - loss: 1.6035 - student_loss: 1.6040

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 299ms/step - accuracy: 0.2526 - distillation_loss: 1.6034 - loss: 1.6034 - student_loss: 1.6036

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 290ms/step - accuracy: 0.2528 - distillation_loss: 1.6037 - loss: 1.6034 - student_loss: 1.6033

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 284ms/step - accuracy: 0.2522 - distillation_loss: 1.6042 - loss: 1.6037 - student_loss: 1.6033

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 311ms/step - accuracy: 0.2516 - distillation_loss: 1.6045 - loss: 1.6039 - student_loss: 1.6033

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 310ms/step - accuracy: 0.2511 - distillation_loss: 1.6048 - loss: 1.6040 - student_loss: 1.6033

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 303ms/step - accuracy: 0.2504 - distillation_loss: 1.6051 - loss: 1.6041 - student_loss: 1.6033

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 302ms/step - accuracy: 0.2498 - distillation_loss: 1.6053 - loss: 1.6042 - student_loss: 1.6033

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 298ms/step - accuracy: 0.2492 - distillation_loss: 1.6054 - loss: 1.6043 - student_loss: 1.6033

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 375ms/step - accuracy: 0.2426 - distillation_loss: 1.6073 - loss: 1.6051 - student_loss: 1.6028 - val_accuracy: 0.2170 - val_loss: 1.5452 - val_student_loss: 1.5852


    Epoch 5/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m15s[0m 1s/step - accuracy: 0.2109 - distillation_loss: 1.5970 - loss: 1.6003 - student_loss: 1.6037

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 250ms/step - accuracy: 0.2266 - distillation_loss: 1.5971 - loss: 1.5991 - student_loss: 1.6012

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m3s[0m 308ms/step - accuracy: 0.2318 - distillation_loss: 1.5975 - loss: 1.5987 - student_loss: 1.5999

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 301ms/step - accuracy: 0.2324 - distillation_loss: 1.5987 - loss: 1.5992 - student_loss: 1.5998

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 300ms/step - accuracy: 0.2334 - distillation_loss: 1.5993 - loss: 1.5994 - student_loss: 1.5996

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m1s[0m 284ms/step - accuracy: 0.2322 - distillation_loss: 1.5998 - loss: 1.5998 - student_loss: 1.5999

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 279ms/step - accuracy: 0.2324 - distillation_loss: 1.5998 - loss: 1.5998 - student_loss: 1.5999

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 304ms/step - accuracy: 0.2329 - distillation_loss: 1.5999 - loss: 1.5998 - student_loss: 1.5998

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 307ms/step - accuracy: 0.2332 - distillation_loss: 1.6001 - loss: 1.5999 - student_loss: 1.5999

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 300ms/step - accuracy: 0.2332 - distillation_loss: 1.6004 - loss: 1.6002 - student_loss: 1.6001

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 294ms/step - accuracy: 0.2333 - distillation_loss: 1.6008 - loss: 1.6005 - student_loss: 1.6003

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 294ms/step - accuracy: 0.2336 - distillation_loss: 1.6011 - loss: 1.6007 - student_loss: 1.6005

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 291ms/step - accuracy: 0.2340 - distillation_loss: 1.6013 - loss: 1.6009 - student_loss: 1.6007

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 368ms/step - accuracy: 0.2387 - distillation_loss: 1.6037 - loss: 1.6033 - student_loss: 1.6029 - val_accuracy: 0.2445 - val_loss: 1.5618 - val_student_loss: 1.5873


    Epoch 6/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m15s[0m 1s/step - accuracy: 0.2305 - distillation_loss: 1.6023 - loss: 1.5994 - student_loss: 1.5964

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 252ms/step - accuracy: 0.2285 - distillation_loss: 1.6027 - loss: 1.5995 - student_loss: 1.5963

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m3s[0m 382ms/step - accuracy: 0.2288 - distillation_loss: 1.6023 - loss: 1.5994 - student_loss: 1.5965

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m3s[0m 336ms/step - accuracy: 0.2304 - distillation_loss: 1.6020 - loss: 1.5994 - student_loss: 1.5969

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 318ms/step - accuracy: 0.2303 - distillation_loss: 1.6021 - loss: 1.5997 - student_loss: 1.5974

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 304ms/step - accuracy: 0.2312 - distillation_loss: 1.6022 - loss: 1.5999 - student_loss: 1.5976

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 329ms/step - accuracy: 0.2314 - distillation_loss: 1.6022 - loss: 1.6001 - student_loss: 1.5979

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 323ms/step - accuracy: 0.2327 - distillation_loss: 1.6022 - loss: 1.6003 - student_loss: 1.5983

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 316ms/step - accuracy: 0.2339 - distillation_loss: 1.6022 - loss: 1.6004 - student_loss: 1.5986

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 308ms/step - accuracy: 0.2354 - distillation_loss: 1.6021 - loss: 1.6004 - student_loss: 1.5988

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 305ms/step - accuracy: 0.2365 - distillation_loss: 1.6019 - loss: 1.6004 - student_loss: 1.5989

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 299ms/step - accuracy: 0.2376 - distillation_loss: 1.6018 - loss: 1.6004 - student_loss: 1.5989

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 295ms/step - accuracy: 0.2385 - distillation_loss: 1.6017 - loss: 1.6004 - student_loss: 1.5991

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 369ms/step - accuracy: 0.2489 - distillation_loss: 1.6007 - loss: 1.6008 - student_loss: 1.6008 - val_accuracy: 0.2445 - val_loss: 1.6880 - val_student_loss: 1.6270


    Epoch 7/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m15s[0m 1s/step - accuracy: 0.2227 - distillation_loss: 1.6085 - loss: 1.6161 - student_loss: 1.6238

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 216ms/step - accuracy: 0.2277 - distillation_loss: 1.6046 - loss: 1.6110 - student_loss: 1.6169

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m2s[0m 250ms/step - accuracy: 0.2286 - distillation_loss: 1.6036 - loss: 1.6087 - student_loss: 1.6134

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 293ms/step - accuracy: 0.2261 - distillation_loss: 1.6027 - loss: 1.6070 - student_loss: 1.6108

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 292ms/step - accuracy: 0.2240 - distillation_loss: 1.6023 - loss: 1.6059 - student_loss: 1.6092

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m1s[0m 282ms/step - accuracy: 0.2215 - distillation_loss: 1.6024 - loss: 1.6056 - student_loss: 1.6084

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 277ms/step - accuracy: 0.2210 - distillation_loss: 1.6027 - loss: 1.6055 - student_loss: 1.6079

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 279ms/step - accuracy: 0.2214 - distillation_loss: 1.6028 - loss: 1.6052 - student_loss: 1.6072

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 291ms/step - accuracy: 0.2220 - distillation_loss: 1.6027 - loss: 1.6048 - student_loss: 1.6066

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 293ms/step - accuracy: 0.2224 - distillation_loss: 1.6027 - loss: 1.6046 - student_loss: 1.6062

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 288ms/step - accuracy: 0.2230 - distillation_loss: 1.6027 - loss: 1.6044 - student_loss: 1.6057

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 287ms/step - accuracy: 0.2236 - distillation_loss: 1.6028 - loss: 1.6042 - student_loss: 1.6053

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 284ms/step - accuracy: 0.2241 - distillation_loss: 1.6027 - loss: 1.6040 - student_loss: 1.6049

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 361ms/step - accuracy: 0.2308 - distillation_loss: 1.6025 - loss: 1.6014 - student_loss: 1.6003 - val_accuracy: 0.2527 - val_loss: 1.5244 - val_student_loss: 1.5746


    Epoch 8/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 1s/step - accuracy: 0.2852 - distillation_loss: 1.5917 - loss: 1.5895 - student_loss: 1.5874

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 247ms/step - accuracy: 0.2764 - distillation_loss: 1.5882 - loss: 1.5848 - student_loss: 1.5813

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m3s[0m 351ms/step - accuracy: 0.2667 - distillation_loss: 1.5908 - loss: 1.5875 - student_loss: 1.5842

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 323ms/step - accuracy: 0.2582 - distillation_loss: 1.5938 - loss: 1.5909 - student_loss: 1.5883

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 304ms/step - accuracy: 0.2531 - distillation_loss: 1.5962 - loss: 1.5934 - student_loss: 1.5911

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 293ms/step - accuracy: 0.2509 - distillation_loss: 1.5971 - loss: 1.5946 - student_loss: 1.5925

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 320ms/step - accuracy: 0.2487 - distillation_loss: 1.5982 - loss: 1.5957 - student_loss: 1.5937

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 316ms/step - accuracy: 0.2480 - distillation_loss: 1.5986 - loss: 1.5962 - student_loss: 1.5943

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 307ms/step - accuracy: 0.2474 - distillation_loss: 1.5988 - loss: 1.5966 - student_loss: 1.5949

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 299ms/step - accuracy: 0.2465 - distillation_loss: 1.5990 - loss: 1.5969 - student_loss: 1.5954

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 298ms/step - accuracy: 0.2460 - distillation_loss: 1.5990 - loss: 1.5971 - student_loss: 1.5957

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 293ms/step - accuracy: 0.2457 - distillation_loss: 1.5990 - loss: 1.5972 - student_loss: 1.5958

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 288ms/step - accuracy: 0.2459 - distillation_loss: 1.5990 - loss: 1.5972 - student_loss: 1.5958

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 363ms/step - accuracy: 0.2480 - distillation_loss: 1.5985 - loss: 1.5973 - student_loss: 1.5964 - val_accuracy: 0.2857 - val_loss: 1.5183 - val_student_loss: 1.5697


    Epoch 9/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m15s[0m 1s/step - accuracy: 0.2695 - distillation_loss: 1.5793 - loss: 1.5822 - student_loss: 1.5850

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 248ms/step - accuracy: 0.2734 - distillation_loss: 1.5809 - loss: 1.5837 - student_loss: 1.5864

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m3s[0m 340ms/step - accuracy: 0.2769 - distillation_loss: 1.5811 - loss: 1.5832 - student_loss: 1.5852

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 324ms/step - accuracy: 0.2765 - distillation_loss: 1.5824 - loss: 1.5834 - student_loss: 1.5844

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 298ms/step - accuracy: 0.2770 - distillation_loss: 1.5822 - loss: 1.5831 - student_loss: 1.5840

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 288ms/step - accuracy: 0.2776 - distillation_loss: 1.5816 - loss: 1.5827 - student_loss: 1.5837

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 305ms/step - accuracy: 0.2783 - distillation_loss: 1.5806 - loss: 1.5818 - student_loss: 1.5829

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 302ms/step - accuracy: 0.2797 - distillation_loss: 1.5799 - loss: 1.5811 - student_loss: 1.5822

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 300ms/step - accuracy: 0.2816 - distillation_loss: 1.5791 - loss: 1.5802 - student_loss: 1.5813

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 294ms/step - accuracy: 0.2832 - distillation_loss: 1.5782 - loss: 1.5792 - student_loss: 1.5802

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 289ms/step - accuracy: 0.2842 - distillation_loss: 1.5773 - loss: 1.5783 - student_loss: 1.5792

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 289ms/step - accuracy: 0.2858 - distillation_loss: 1.5763 - loss: 1.5771 - student_loss: 1.5779

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 286ms/step - accuracy: 0.2875 - distillation_loss: 1.5752 - loss: 1.5760 - student_loss: 1.5767

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 361ms/step - accuracy: 0.3073 - distillation_loss: 1.5623 - loss: 1.5619 - student_loss: 1.5617 - val_accuracy: 0.3104 - val_loss: 1.5522 - val_student_loss: 1.5427


    Epoch 10/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 1s/step - accuracy: 0.3516 - distillation_loss: 1.5118 - loss: 1.5008 - student_loss: 1.4898

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 246ms/step - accuracy: 0.3447 - distillation_loss: 1.5161 - loss: 1.5033 - student_loss: 1.4906

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m2s[0m 244ms/step - accuracy: 0.3531 - distillation_loss: 1.5148 - loss: 1.5026 - student_loss: 1.4903

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 274ms/step - accuracy: 0.3561 - distillation_loss: 1.5116 - loss: 1.5000 - student_loss: 1.4880

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 277ms/step - accuracy: 0.3553 - distillation_loss: 1.5088 - loss: 1.4981 - student_loss: 1.4870

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m1s[0m 282ms/step - accuracy: 0.3559 - distillation_loss: 1.5065 - loss: 1.4963 - student_loss: 1.4857

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 276ms/step - accuracy: 0.3560 - distillation_loss: 1.5042 - loss: 1.4942 - student_loss: 1.4837

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 272ms/step - accuracy: 0.3555 - distillation_loss: 1.5025 - loss: 1.4926 - student_loss: 1.4823

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 300ms/step - accuracy: 0.3561 - distillation_loss: 1.5006 - loss: 1.4908 - student_loss: 1.4805

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 298ms/step - accuracy: 0.3566 - distillation_loss: 1.4997 - loss: 1.4900 - student_loss: 1.4800

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 292ms/step - accuracy: 0.3570 - distillation_loss: 1.4984 - loss: 1.4891 - student_loss: 1.4794

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 291ms/step - accuracy: 0.3574 - distillation_loss: 1.4973 - loss: 1.4884 - student_loss: 1.4791

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 288ms/step - accuracy: 0.3580 - distillation_loss: 1.4960 - loss: 1.4875 - student_loss: 1.4786

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 363ms/step - accuracy: 0.3657 - distillation_loss: 1.4807 - loss: 1.4769 - student_loss: 1.4729 - val_accuracy: 0.3407 - val_loss: 1.5153 - val_student_loss: 1.4909


    Epoch 11/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m16s[0m 1s/step - accuracy: 0.3359 - distillation_loss: 1.5011 - loss: 1.5089 - student_loss: 1.5166

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 246ms/step - accuracy: 0.3545 - distillation_loss: 1.4779 - loss: 1.4832 - student_loss: 1.4885

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m3s[0m 375ms/step - accuracy: 0.3605 - distillation_loss: 1.4666 - loss: 1.4719 - student_loss: 1.4772

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 332ms/step - accuracy: 0.3685 - distillation_loss: 1.4601 - loss: 1.4647 - student_loss: 1.4693

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 316ms/step - accuracy: 0.3728 - distillation_loss: 1.4553 - loss: 1.4592 - student_loss: 1.4631

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 302ms/step - accuracy: 0.3754 - distillation_loss: 1.4517 - loss: 1.4551 - student_loss: 1.4584

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m2s[0m 340ms/step - accuracy: 0.3780 - distillation_loss: 1.4488 - loss: 1.4518 - student_loss: 1.4549

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 322ms/step - accuracy: 0.3792 - distillation_loss: 1.4469 - loss: 1.4495 - student_loss: 1.4520

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 313ms/step - accuracy: 0.3801 - distillation_loss: 1.4450 - loss: 1.4471 - student_loss: 1.4492

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 305ms/step - accuracy: 0.3811 - distillation_loss: 1.4429 - loss: 1.4446 - student_loss: 1.4463

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 303ms/step - accuracy: 0.3819 - distillation_loss: 1.4408 - loss: 1.4422 - student_loss: 1.4437

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 297ms/step - accuracy: 0.3826 - distillation_loss: 1.4388 - loss: 1.4399 - student_loss: 1.4412

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 292ms/step - accuracy: 0.3835 - distillation_loss: 1.4367 - loss: 1.4377 - student_loss: 1.4389

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 369ms/step - accuracy: 0.3941 - distillation_loss: 1.4111 - loss: 1.4111 - student_loss: 1.4115 - val_accuracy: 0.3544 - val_loss: 1.5445 - val_student_loss: 1.4759


    Epoch 12/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m15s[0m 1s/step - accuracy: 0.3984 - distillation_loss: 1.4051 - loss: 1.4067 - student_loss: 1.4084

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 246ms/step - accuracy: 0.3936 - distillation_loss: 1.3899 - loss: 1.3949 - student_loss: 1.3998

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m3s[0m 309ms/step - accuracy: 0.3900 - distillation_loss: 1.3858 - loss: 1.3895 - student_loss: 1.3932

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 308ms/step - accuracy: 0.3892 - distillation_loss: 1.3818 - loss: 1.3854 - student_loss: 1.3889

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 334ms/step - accuracy: 0.3912 - distillation_loss: 1.3792 - loss: 1.3831 - student_loss: 1.3870

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 316ms/step - accuracy: 0.3934 - distillation_loss: 1.3773 - loss: 1.3814 - student_loss: 1.3854

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m2s[0m 340ms/step - accuracy: 0.3955 - distillation_loss: 1.3768 - loss: 1.3807 - student_loss: 1.3846

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 334ms/step - accuracy: 0.3967 - distillation_loss: 1.3764 - loss: 1.3801 - student_loss: 1.3839

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 323ms/step - accuracy: 0.3976 - distillation_loss: 1.3761 - loss: 1.3798 - student_loss: 1.3835

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 314ms/step - accuracy: 0.3987 - distillation_loss: 1.3756 - loss: 1.3792 - student_loss: 1.3828

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 311ms/step - accuracy: 0.4004 - distillation_loss: 1.3752 - loss: 1.3785 - student_loss: 1.3818

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 305ms/step - accuracy: 0.4021 - distillation_loss: 1.3747 - loss: 1.3776 - student_loss: 1.3806

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 299ms/step - accuracy: 0.4039 - distillation_loss: 1.3741 - loss: 1.3769 - student_loss: 1.3796

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 377ms/step - accuracy: 0.4250 - distillation_loss: 1.3679 - loss: 1.3678 - student_loss: 1.3676 - val_accuracy: 0.3929 - val_loss: 1.3185 - val_student_loss: 1.3640


    Epoch 13/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m15s[0m 1s/step - accuracy: 0.5430 - distillation_loss: 1.2986 - loss: 1.2771 - student_loss: 1.2557

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 251ms/step - accuracy: 0.5186 - distillation_loss: 1.3113 - loss: 1.2969 - student_loss: 1.2824

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m3s[0m 308ms/step - accuracy: 0.5080 - distillation_loss: 1.3130 - loss: 1.3022 - student_loss: 1.2915

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 305ms/step - accuracy: 0.4985 - distillation_loss: 1.3164 - loss: 1.3080 - student_loss: 1.2995

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 308ms/step - accuracy: 0.4911 - distillation_loss: 1.3176 - loss: 1.3103 - student_loss: 1.3029

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 296ms/step - accuracy: 0.4864 - distillation_loss: 1.3161 - loss: 1.3101 - student_loss: 1.3040

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 287ms/step - accuracy: 0.4829 - distillation_loss: 1.3149 - loss: 1.3098 - student_loss: 1.3047

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 308ms/step - accuracy: 0.4797 - distillation_loss: 1.3147 - loss: 1.3101 - student_loss: 1.3055

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 309ms/step - accuracy: 0.4777 - distillation_loss: 1.3140 - loss: 1.3095 - student_loss: 1.3050

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 302ms/step - accuracy: 0.4759 - distillation_loss: 1.3133 - loss: 1.3090 - student_loss: 1.3046

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 296ms/step - accuracy: 0.4746 - distillation_loss: 1.3125 - loss: 1.3084 - student_loss: 1.3042

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 297ms/step - accuracy: 0.4731 - distillation_loss: 1.3121 - loss: 1.3083 - student_loss: 1.3044

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 290ms/step - accuracy: 0.4722 - distillation_loss: 1.3118 - loss: 1.3081 - student_loss: 1.3045

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 367ms/step - accuracy: 0.4604 - distillation_loss: 1.3078 - loss: 1.3064 - student_loss: 1.3049 - val_accuracy: 0.4478 - val_loss: 1.5273 - val_student_loss: 1.4022


    Epoch 14/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m15s[0m 1s/step - accuracy: 0.5195 - distillation_loss: 1.3453 - loss: 1.3315 - student_loss: 1.3176

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 243ms/step - accuracy: 0.5107 - distillation_loss: 1.3298 - loss: 1.3167 - student_loss: 1.3036

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m2s[0m 244ms/step - accuracy: 0.5041 - distillation_loss: 1.3220 - loss: 1.3087 - student_loss: 1.2954

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 306ms/step - accuracy: 0.5004 - distillation_loss: 1.3153 - loss: 1.3034 - student_loss: 1.2914

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 309ms/step - accuracy: 0.4999 - distillation_loss: 1.3123 - loss: 1.3009 - student_loss: 1.2896

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 291ms/step - accuracy: 0.4994 - distillation_loss: 1.3101 - loss: 1.2990 - student_loss: 1.2878

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 283ms/step - accuracy: 0.4985 - distillation_loss: 1.3085 - loss: 1.2975 - student_loss: 1.2863

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 284ms/step - accuracy: 0.4974 - distillation_loss: 1.3073 - loss: 1.2967 - student_loss: 1.2860

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 291ms/step - accuracy: 0.4965 - distillation_loss: 1.3058 - loss: 1.2958 - student_loss: 1.2857

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 289ms/step - accuracy: 0.4961 - distillation_loss: 1.3045 - loss: 1.2949 - student_loss: 1.2851

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 290ms/step - accuracy: 0.4961 - distillation_loss: 1.3032 - loss: 1.2938 - student_loss: 1.2843

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 289ms/step - accuracy: 0.4964 - distillation_loss: 1.3019 - loss: 1.2926 - student_loss: 1.2833

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 286ms/step - accuracy: 0.4974 - distillation_loss: 1.3003 - loss: 1.2911 - student_loss: 1.2819

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 364ms/step - accuracy: 0.5094 - distillation_loss: 1.2820 - loss: 1.2736 - student_loss: 1.2653 - val_accuracy: 0.4753 - val_loss: 1.2482 - val_student_loss: 1.3120


    Epoch 15/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 1s/step - accuracy: 0.5039 - distillation_loss: 1.2945 - loss: 1.2947 - student_loss: 1.2949

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 222ms/step - accuracy: 0.5193 - distillation_loss: 1.2789 - loss: 1.2748 - student_loss: 1.2689

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m2s[0m 232ms/step - accuracy: 0.5214 - distillation_loss: 1.2770 - loss: 1.2714 - student_loss: 1.2635

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 282ms/step - accuracy: 0.5198 - distillation_loss: 1.2751 - loss: 1.2685 - student_loss: 1.2596

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 293ms/step - accuracy: 0.5203 - distillation_loss: 1.2739 - loss: 1.2659 - student_loss: 1.2559

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 291ms/step - accuracy: 0.5199 - distillation_loss: 1.2732 - loss: 1.2645 - student_loss: 1.2538

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 283ms/step - accuracy: 0.5182 - distillation_loss: 1.2739 - loss: 1.2644 - student_loss: 1.2530

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 294ms/step - accuracy: 0.5169 - distillation_loss: 1.2740 - loss: 1.2639 - student_loss: 1.2520

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 296ms/step - accuracy: 0.5159 - distillation_loss: 1.2736 - loss: 1.2631 - student_loss: 1.2509

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 299ms/step - accuracy: 0.5149 - distillation_loss: 1.2735 - loss: 1.2627 - student_loss: 1.2504

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 300ms/step - accuracy: 0.5138 - distillation_loss: 1.2736 - loss: 1.2625 - student_loss: 1.2500

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 295ms/step - accuracy: 0.5132 - distillation_loss: 1.2736 - loss: 1.2623 - student_loss: 1.2495

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 290ms/step - accuracy: 0.5129 - distillation_loss: 1.2734 - loss: 1.2617 - student_loss: 1.2487

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 367ms/step - accuracy: 0.5088 - distillation_loss: 1.2707 - loss: 1.2553 - student_loss: 1.2394 - val_accuracy: 0.5302 - val_loss: 1.3324 - val_student_loss: 1.2801


    Epoch 16/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 1s/step - accuracy: 0.5977 - distillation_loss: 1.1590 - loss: 1.1481 - student_loss: 1.1373

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 262ms/step - accuracy: 0.5908 - distillation_loss: 1.1777 - loss: 1.1664 - student_loss: 1.1552

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m2s[0m 255ms/step - accuracy: 0.5836 - distillation_loss: 1.1921 - loss: 1.1778 - student_loss: 1.1636

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 290ms/step - accuracy: 0.5788 - distillation_loss: 1.2009 - loss: 1.1851 - student_loss: 1.1693

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 301ms/step - accuracy: 0.5746 - distillation_loss: 1.2061 - loss: 1.1899 - student_loss: 1.1737

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 298ms/step - accuracy: 0.5707 - distillation_loss: 1.2115 - loss: 1.1953 - student_loss: 1.1792

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 290ms/step - accuracy: 0.5688 - distillation_loss: 1.2145 - loss: 1.1985 - student_loss: 1.1824

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 302ms/step - accuracy: 0.5663 - distillation_loss: 1.2177 - loss: 1.2021 - student_loss: 1.1865

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 307ms/step - accuracy: 0.5644 - distillation_loss: 1.2201 - loss: 1.2048 - student_loss: 1.1895

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 302ms/step - accuracy: 0.5627 - distillation_loss: 1.2218 - loss: 1.2067 - student_loss: 1.1915

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 296ms/step - accuracy: 0.5616 - distillation_loss: 1.2231 - loss: 1.2080 - student_loss: 1.1929

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 295ms/step - accuracy: 0.5608 - distillation_loss: 1.2241 - loss: 1.2089 - student_loss: 1.1937

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 292ms/step - accuracy: 0.5604 - distillation_loss: 1.2247 - loss: 1.2094 - student_loss: 1.1941

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 366ms/step - accuracy: 0.5544 - distillation_loss: 1.2321 - loss: 1.2154 - student_loss: 1.1985 - val_accuracy: 0.5385 - val_loss: 1.4207 - val_student_loss: 1.2887


    Epoch 17/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 1s/step - accuracy: 0.4883 - distillation_loss: 1.2635 - loss: 1.2671 - student_loss: 1.2707

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 245ms/step - accuracy: 0.5039 - distillation_loss: 1.2504 - loss: 1.2504 - student_loss: 1.2504

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m2s[0m 246ms/step - accuracy: 0.5148 - distillation_loss: 1.2420 - loss: 1.2404 - student_loss: 1.2387

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 312ms/step - accuracy: 0.5233 - distillation_loss: 1.2366 - loss: 1.2324 - student_loss: 1.2283

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 344ms/step - accuracy: 0.5302 - distillation_loss: 1.2313 - loss: 1.2252 - student_loss: 1.2191

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 326ms/step - accuracy: 0.5345 - distillation_loss: 1.2284 - loss: 1.2203 - student_loss: 1.2123

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 332ms/step - accuracy: 0.5389 - distillation_loss: 1.2261 - loss: 1.2163 - student_loss: 1.2064

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 335ms/step - accuracy: 0.5427 - distillation_loss: 1.2238 - loss: 1.2125 - student_loss: 1.2011

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 324ms/step - accuracy: 0.5462 - distillation_loss: 1.2212 - loss: 1.2088 - student_loss: 1.1963

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 315ms/step - accuracy: 0.5487 - distillation_loss: 1.2195 - loss: 1.2065 - student_loss: 1.1932

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 312ms/step - accuracy: 0.5505 - distillation_loss: 1.2190 - loss: 1.2052 - student_loss: 1.1910

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 305ms/step - accuracy: 0.5519 - distillation_loss: 1.2183 - loss: 1.2039 - student_loss: 1.1891

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 300ms/step - accuracy: 0.5532 - distillation_loss: 1.2176 - loss: 1.2027 - student_loss: 1.1875

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 377ms/step - accuracy: 0.5693 - distillation_loss: 1.2090 - loss: 1.1889 - student_loss: 1.1681 - val_accuracy: 0.5549 - val_loss: 1.1973 - val_student_loss: 1.2029


    Epoch 18/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m15s[0m 1s/step - accuracy: 0.6016 - distillation_loss: 1.2087 - loss: 1.1811 - student_loss: 1.1535

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 248ms/step - accuracy: 0.5938 - distillation_loss: 1.2074 - loss: 1.1850 - student_loss: 1.1626

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m3s[0m 341ms/step - accuracy: 0.5929 - distillation_loss: 1.2048 - loss: 1.1831 - student_loss: 1.1615

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m3s[0m 340ms/step - accuracy: 0.5919 - distillation_loss: 1.2023 - loss: 1.1819 - student_loss: 1.1615

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 318ms/step - accuracy: 0.5923 - distillation_loss: 1.1991 - loss: 1.1797 - student_loss: 1.1603

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 304ms/step - accuracy: 0.5925 - distillation_loss: 1.1981 - loss: 1.1790 - student_loss: 1.1598

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 329ms/step - accuracy: 0.5936 - distillation_loss: 1.1968 - loss: 1.1772 - student_loss: 1.1577

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 320ms/step - accuracy: 0.5944 - distillation_loss: 1.1952 - loss: 1.1756 - student_loss: 1.1559

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 311ms/step - accuracy: 0.5942 - distillation_loss: 1.1938 - loss: 1.1744 - student_loss: 1.1550

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 303ms/step - accuracy: 0.5940 - distillation_loss: 1.1927 - loss: 1.1734 - student_loss: 1.1541

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 301ms/step - accuracy: 0.5937 - distillation_loss: 1.1920 - loss: 1.1729 - student_loss: 1.1537

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 295ms/step - accuracy: 0.5938 - distillation_loss: 1.1914 - loss: 1.1723 - student_loss: 1.1531

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 291ms/step - accuracy: 0.5938 - distillation_loss: 1.1911 - loss: 1.1721 - student_loss: 1.1529

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 365ms/step - accuracy: 0.5941 - distillation_loss: 1.1869 - loss: 1.1691 - student_loss: 1.1510 - val_accuracy: 0.5797 - val_loss: 1.1819 - val_student_loss: 1.1792


    Epoch 19/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m14s[0m 1s/step - accuracy: 0.6016 - distillation_loss: 1.1650 - loss: 1.1571 - student_loss: 1.1492

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 250ms/step - accuracy: 0.5947 - distillation_loss: 1.1778 - loss: 1.1686 - student_loss: 1.1594

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m2s[0m 248ms/step - accuracy: 0.5948 - distillation_loss: 1.1775 - loss: 1.1666 - student_loss: 1.1557

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 260ms/step - accuracy: 0.5963 - distillation_loss: 1.1766 - loss: 1.1635 - student_loss: 1.1504

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 339ms/step - accuracy: 0.5944 - distillation_loss: 1.1789 - loss: 1.1642 - student_loss: 1.1494

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m2s[0m 314ms/step - accuracy: 0.5947 - distillation_loss: 1.1788 - loss: 1.1631 - student_loss: 1.1470

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 304ms/step - accuracy: 0.5953 - distillation_loss: 1.1789 - loss: 1.1620 - student_loss: 1.1447

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 300ms/step - accuracy: 0.5962 - distillation_loss: 1.1782 - loss: 1.1603 - student_loss: 1.1420

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 311ms/step - accuracy: 0.5964 - distillation_loss: 1.1779 - loss: 1.1593 - student_loss: 1.1403

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 311ms/step - accuracy: 0.5966 - distillation_loss: 1.1778 - loss: 1.1588 - student_loss: 1.1392

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 304ms/step - accuracy: 0.5972 - distillation_loss: 1.1777 - loss: 1.1581 - student_loss: 1.1380

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 302ms/step - accuracy: 0.5980 - distillation_loss: 1.1773 - loss: 1.1573 - student_loss: 1.1367

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 297ms/step - accuracy: 0.5986 - distillation_loss: 1.1768 - loss: 1.1565 - student_loss: 1.1358

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 372ms/step - accuracy: 0.6062 - distillation_loss: 1.1702 - loss: 1.1475 - student_loss: 1.1242 - val_accuracy: 0.5934 - val_loss: 1.1396 - val_student_loss: 1.1463


    Epoch 20/20


    [1m 1/13[0m [32m━[0m[37m━━━━━━━━━━━━━━━━━━━[0m [1m15s[0m 1s/step - accuracy: 0.5742 - distillation_loss: 1.2056 - loss: 1.1856 - student_loss: 1.1656

    [1m 2/13[0m [32m━━━[0m[37m━━━━━━━━━━━━━━━━━[0m [1m2s[0m 225ms/step - accuracy: 0.6004 - distillation_loss: 1.1820 - loss: 1.1584 - student_loss: 1.1322

    [1m 3/13[0m [32m━━━━[0m[37m━━━━━━━━━━━━━━━━[0m [1m2s[0m 259ms/step - accuracy: 0.6058 - distillation_loss: 1.1761 - loss: 1.1547 - student_loss: 1.1302

    [1m 4/13[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m2s[0m 294ms/step - accuracy: 0.6123 - distillation_loss: 1.1683 - loss: 1.1470 - student_loss: 1.1228

    [1m 5/13[0m [32m━━━━━━━[0m[37m━━━━━━━━━━━━━[0m [1m2s[0m 293ms/step - accuracy: 0.6162 - distillation_loss: 1.1644 - loss: 1.1422 - student_loss: 1.1173

    [1m 6/13[0m [32m━━━━━━━━━[0m[37m━━━━━━━━━━━[0m [1m1s[0m 284ms/step - accuracy: 0.6179 - distillation_loss: 1.1636 - loss: 1.1403 - student_loss: 1.1145

    [1m 7/13[0m [32m━━━━━━━━━━[0m[37m━━━━━━━━━━[0m [1m1s[0m 278ms/step - accuracy: 0.6194 - distillation_loss: 1.1628 - loss: 1.1386 - student_loss: 1.1120

    [1m 8/13[0m [32m━━━━━━━━━━━━[0m[37m━━━━━━━━[0m [1m1s[0m 281ms/step - accuracy: 0.6200 - distillation_loss: 1.1631 - loss: 1.1384 - student_loss: 1.1114

    [1m 9/13[0m [32m━━━━━━━━━━━━━[0m[37m━━━━━━━[0m [1m1s[0m 294ms/step - accuracy: 0.6209 - distillation_loss: 1.1627 - loss: 1.1377 - student_loss: 1.1106

    [1m10/13[0m [32m━━━━━━━━━━━━━━━[0m[37m━━━━━[0m [1m0s[0m 296ms/step - accuracy: 0.6218 - distillation_loss: 1.1622 - loss: 1.1371 - student_loss: 1.1099

    [1m11/13[0m [32m━━━━━━━━━━━━━━━━[0m[37m━━━━[0m [1m0s[0m 291ms/step - accuracy: 0.6220 - distillation_loss: 1.1617 - loss: 1.1369 - student_loss: 1.1101

    [1m12/13[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 290ms/step - accuracy: 0.6223 - distillation_loss: 1.1616 - loss: 1.1369 - student_loss: 1.1104

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 286ms/step - accuracy: 0.6225 - distillation_loss: 1.1613 - loss: 1.1367 - student_loss: 1.1102

    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 358ms/step - accuracy: 0.6243 - distillation_loss: 1.1581 - loss: 1.1337 - student_loss: 1.1085 - val_accuracy: 0.5934 - val_loss: 1.2590 - val_student_loss: 1.1801


In this Keras 3 setup, distillation consistently improves over training the same
backbone from scratch under the same budget. In our current run, the distilled model
reaches about **61.5% validation accuracy** after 20 epochs.

You can adapt the following code to reproduce a non-distilled baseline:

```
vit_tiny = ViTClassifier()

inputs = keras.Input((RESOLUTION, RESOLUTION, 3))
x = keras.layers.Rescaling(scale=1./255)(inputs)
outputs = deit_tiny(x)
model = keras.Model(inputs, outputs)

model.compile(...)
model.fit(...)
```

## Notes

* Through the use of distillation, we're effectively transferring the inductive biases of
a CNN-based teacher model.
* In this example, a compact CNN teacher (`EfficientNetV2B0`) provides a strong
signal and stabilizes DeiT training on the flowers dataset.
* The use of regularization to train DeiT models is very important.
* ViT models are initialized with a combination of different initializers including
truncated normal, random normal, Glorot uniform, etc. If you're looking for
end-to-end reproduction of the original results, don't forget to initialize the ViTs well.
* The entire pipeline is backend-agnostic in Keras 3: data loading, augmentation,
and distillation all run without TensorFlow-specific APIs.
* If you want to explore the pre-trained DeiT models in Keras with code
for fine-tuning, [check out these models on TF-Hub](https://tfhub.dev/sayakpaul/collections/deit/1).

## Acknowledgements

* Ross Wightman for keeping
[`timm`](https://github.com/rwightman/pytorch-image-models)
updated with readable implementations. I referred to the implementations of ViT and DeiT
a lot during implementing them in Keras.
* [Aritra Roy Gosthipaty](https://github.com/ariG23498)
who implemented some portions of the `ViTClassifier` in another project.
* [Google Developers Experts](https://developers.google.com/programs/experts/)
program for supporting me with GCP credits which were used to run experiments for this
example.

Example available on HuggingFace:

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/🤗%20Model-DEIT-black.svg)](https://huggingface.co/keras-io/deit) | [![Generic badge](https://img.shields.io/badge/🤗%20Spaces-DEIT-black.svg)](https://huggingface.co/spaces/keras-io/deit/) |

## Relevant Chapters from Deep Learning with Python
- [Chapter 8: Image classification](https://deeplearningwithpython.io/chapters/chapter08_image-classification)
- [Chapter 15: Language models and the Transformer](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer)

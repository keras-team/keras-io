"""
Title: Distilling Vision Transformers
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2022/04/05
Last modified: 2026/03/11
Description: Distillation of Vision Transformers through attention.
Accelerator: GPU
Converted to Keras 3 by: [Harshith K](https://github.com/kharshith-k/)
"""

"""
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
"""

"""
## Imports
"""

from pathlib import Path
from typing import List

import numpy as np
import keras
from keras import layers

keras.utils.set_random_seed(42)

"""
## Constants
"""

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

"""
You probably noticed that `DROPOUT_RATE` has been set 0.0. Dropout has been used
in the implementation to keep it complete. For smaller models (like the one used in
this example), you don't need it, but for bigger models, using dropout helps.
"""

"""
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
"""

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

"""
## Implementing the DeiT variants of ViT

Since DeiT is an extension of ViT it'd make sense to first implement ViT and then extend
it to support DeiT's components.

First, we'll implement a layer for Stochastic Depth
([Huang et al.](https://arxiv.org/abs/1603.09382))
which is used in DeiT for regularization.
"""


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


"""
Now, we'll implement the MLP and Transformer blocks.
"""


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


"""
We'll now implement a `ViTClassifier` class building on top of the components we just
developed. Here we'll be following the original pooling strategy used in the ViT paper --
use a class token and use the feature representations corresponding to it for
classification.
"""


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


"""
This class can be used standalone as ViT and is end-to-end trainable. Just remove the
`distilled` phrase in `MODEL_TYPE` and it should work with `vit_tiny = ViTClassifier()`.
Let's now extend it to DeiT. The following figure presents the schematic of DeiT (taken
from the DeiT paper):

![](https://i.imgur.com/5lmg2Xs.png)

Apart from the class token, DeiT has another token for distillation. During distillation,
the logits corresponding to the class token are compared to the true labels, and the
logits corresponding to the distillation token are compared to the teacher's predictions.
"""


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


"""
Let's verify if the `ViTDistilled` class can be initialized and called as expected.
"""

deit_tiny_distilled = ViTDistilled()

dummy_inputs = keras.ops.ones((2, 224, 224, 3))
outputs = deit_tiny_distilled(dummy_inputs, training=False)
print(outputs.shape)

"""
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
"""


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


"""
## Build a teacher model

For full backend portability in Keras 3, we build a teacher with standard Keras layers
instead of using a TensorFlow-only SavedModel loader. We use `EfficientNetV2B0` (pretrained on
ImageNet) as the backbone, freeze it, and fine-tune only a small classification head on
the flowers dataset. In practice you could swap in any compatible Keras model as the
teacher.

`EfficientNetV2B0` includes preprocessing by default (`include_preprocessing=True`),
so it expects raw `[0, 255]` image values. We therefore avoid adding an extra
`Rescaling(1/255)` layer in the teacher path to prevent double normalization.
"""

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

"""
## Training through distillation
"""

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

"""
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
"""

"""
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

"""

"""
## Relevant Chapters from Deep Learning with Python
- [Chapter 8: Image classification](https://deeplearningwithpython.io/chapters/chapter08_image-classification)
- [Chapter 15: Language models and the Transformer](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer)
"""

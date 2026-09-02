# 2D Multi-Organ Segmentation with TransUNet

**Author:** [Yassien Wasfy](https://www.linkedin.com/in/yassien-wasfy-315ab5349/)<br>
**Date created:** 2026/04/29<br>
**Last modified:** 2026/04/29<br>
**Description:** TransUNet for 2D multi-organ segmentation on the Synapse dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/trans_unet_multi_organ_segmentation.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/trans_unet_multi_organ_segmentation.py)



---
## Introduction

We implement TransUNet (Chen et al., 2021), a hybrid CNN-Transformer architecture
for 2D multi-organ segmentation. A ResNet-50 backbone extracts a feature pyramid,
a Vision Transformer encodes global context, and a cascaded upsampling decoder
recovers spatial resolution via U-Net skip connections.

---
## The Synapse Dataset

The Synapse Multi-Organ Segmentation dataset contains 30 abdominal CT scans from
the MICCAI 2015 Multi-Atlas Abdomen Labeling Challenge.

Each CT scan covers the abdominal region and is stored as a collection of 2D axial
slices. The dataset is distributed in two formats:

- Training: `.npz` files, one per slice, each containing an `image` and `label` array.
- Validation: `.h5` files, one per volume, each containing the full 3D `image` and
`label` arrays stacked along the slice axis.

The dataset can be accessed on
[Kaggle](https://www.kaggle.com/datasets/dogcdt/synapse).

---
## Segmentation Labels

Each pixel is assigned one of 9 class labels corresponding to abdominal organs:

| Label | Organ |
| ----- | ------------ |
| 0 | Background |
| 1 | Aorta |
| 2 | Gallbladder |
| 3 | Left Kidney |
| 4 | Right Kidney |
| 5 | Liver |
| 6 | Pancreas |
| 7 | Spleen |
| 8 | Stomach |

---
## What This Tutorial Covers

We provide an end-to-end workflow for 2D multi-organ segmentation using TransUNet.

1. **Data Pipeline** — Load `.npz` training slices and `.h5` validation volumes,
apply joint spatial augmentation to images and masks, and build efficient
`tf.data` pipelines.

2. **Model Architecture** — Build TransUNet: a ResNet-50 CNN extracts a feature
pyramid, a Vision Transformer encodes global context from the coarsest feature map,
and a cascaded upsampling decoder reconstructs the segmentation map using U-Net
skip connections.

3. **Loss and Metrics** — Combine sparse categorical cross-entropy with a soft
multi-class Dice loss. Evaluate with `MeanIoU` across all 9 organ classes.

4. **Training** — Train with `AdamW`, `ReduceLROnPlateau`, and `EarlyStopping`.

5. **Inference and Visualization** — Run batched slice-level inference on full
3D volumes and visualize predicted masks overlaid on CT slices.

---
## Setup


```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import ops

BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_CLASSES = 9
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1  # Train for 60 epochs for full convergence.
WEIGHT_DECAY = 1e-3
PROJECTION_DIM = 256
NUM_TRANSFORMER_LAYERS = 8
HIDDEN_DIM = 512
SEED = 42
LABEL_COLORS = {
    0: ("Background", (0, 0, 0)),
    1: ("Aorta", (0, 102, 204)),
    2: ("Gallbladder", (0, 255, 0)),
    3: ("Left Kidney", (255, 0, 0)),
    4: ("Right Kidney", (0, 255, 255)),
    5: ("Liver", (255, 0, 255)),
    6: ("Pancreas", (255, 255, 0)),
    7: ("Spleen", (153, 0, 255)),
    8: ("Stomach", (255, 128, 0)),
}
CLASS_NAMES = [name for name, _ in LABEL_COLORS.values()]
PALETTE = np.array([color for _, color in LABEL_COLORS.values()], dtype=np.uint8)
```

---
## Data Pipeline


```python
TRAIN_PATH = "train_npz"
VAL_PATH = "test_vol_h5"

_aug_kw = dict(fill_mode="nearest", interpolation="nearest", seed=SEED)
joint_augment = keras.Sequential(
    [
        keras.layers.RandomRotation(factor=5 / 360.0, **_aug_kw),
        keras.layers.RandomZoom(height_factor=(-0.1, 0.1), **_aug_kw),
    ]
)


def random_augment(image, label):
    combined = ops.concatenate([image, ops.cast(label, "float32")], axis=-1)
    augmented = joint_augment(combined, training=True)
    return augmented[..., :3], ops.cast(augmented[..., 3:], "uint8")


def _resize(arr, interpolation):
    return ops.image.resize(arr, [IMAGE_SIZE, IMAGE_SIZE], interpolation=interpolation)


def load_npz(path):
    data = np.load(path.numpy().decode())
    return data["image"], data["label"]


def load_h5(path):
    with h5py.File(path.numpy().decode(), "r") as f:
        return f["image"][:], f["label"][:]


def preprocess_npz(path):
    image, label = tf.py_function(load_npz, [path], [tf.float32, tf.uint8])
    image.set_shape([None, None])
    label.set_shape([None, None])
    image = ops.repeat(_resize(image[..., None], "bilinear"), 3, axis=-1)
    return image, _resize(label[..., None], "nearest")


def preprocess_h5(path):
    image, label = tf.py_function(load_h5, [path], [tf.float32, tf.uint8])
    image.set_shape([None, None, None])
    label.set_shape([None, None, None])
    image = ops.repeat(_resize(image[..., None], "bilinear"), 3, axis=-1)
    return image, _resize(label[..., None], "nearest")


def has_label(image, label):
    return ops.max(label) > 0


def get_files(path, ext):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)])


RAW_TRAIN_FILES = get_files(TRAIN_PATH, ".npz")
RAW_VAL_FILES = get_files(VAL_PATH, ".h5")
train_dataset = (
    tf.data.Dataset.from_tensor_slices(RAW_TRAIN_FILES)
    .map(preprocess_npz, num_parallel_calls=tf.data.AUTOTUNE)
    .filter(has_label)
    .cache()
    .shuffle(200)
    .map(random_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_dataset = (
    tf.data.Dataset.from_tensor_slices(RAW_VAL_FILES)
    .map(preprocess_h5, num_parallel_calls=tf.data.AUTOTUNE)
    .flat_map(lambda img, lbl: tf.data.Dataset.from_tensor_slices((img, lbl)))
    .filter(has_label)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
```

---
## Model Architecture

TransUNet operates in three stages:

1. **CNN Feature Pyramid:** ResNet-50 produces four feature maps at 112x112, 56x56,
28x28, and 14x14. The finest three serve as U-Net skip connections.

2. **Vision Transformer:** The 14x14 feature map is tokenized via a learnable `Conv2D`
projection, enriched with positional embeddings, and processed by stacked
`TransformerBlock` layers.

3. **CUP Decoder:** The token sequence is reshaped and progressively upsampled
(14->28->56->112->224). At each scale, a CNN skip connection is concatenated before a
`Conv2D + GroupNormalization + ReLU` block. A final 1x1 `Conv2D` with `softmax`
produces the per-pixel class probabilities.

| ![TransUNet Architecture](https://i.postimg.cc/15cppkRJ/transunet-architecture.png) |
| :--: |
| **TransUNet Architecture Overview** |


```python
_Up = keras.layers.UpSampling2D
_C2D = keras.layers.Conv2D
_GN = keras.layers.GroupNormalization


def _build_resnet_encoder():
    """Returns a ResNet-50 feature pyramid as a plain keras.Model."""
    inputs = keras.Input(shape=(224, 224, 3))
    base = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=keras.applications.resnet50.preprocess_input(inputs),
    )
    names = ("conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out")
    outputs = [base.get_layer(n).output for n in names]
    return keras.Model(inputs=inputs, outputs=outputs)


@keras.saving.register_keras_serializable(package="Trans-UNET")
class PatchEmbedding(keras.layers.Layer):
    """Projects CNN feature map patches into position-aware token embeddings."""

    def __init__(self, patch_size, embedding_dim, num_patches, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.projection = keras.layers.Conv2D(
            embedding_dim, patch_size, strides=patch_size, padding="valid"
        )

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.num_patches, self.embedding_dim),
            initializer="random_normal",
        )

    def call(self, inputs):
        patches = self.projection(inputs)
        batch_size = ops.shape(patches)[0]
        return (
            ops.reshape(patches, [batch_size, -1, self.embedding_dim])
            + self.position_embeddings
        )

    def get_config(self):
        keys = ["patch_size", "embedding_dim", "num_patches"]
        return {**super().get_config(), **{k: getattr(self, k) for k in keys}}


@keras.saving.register_keras_serializable(package="Trans-UNET")
class TransformerBlock(keras.layers.Layer):
    """Pre-LN Transformer block: MHSA + MLP with residual connections."""

    def __init__(
        self, embedding_dim, num_heads, hidden_dim, dropout_rate=0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
        )
        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(hidden_dim, activation="gelu"),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Dense(embedding_dim),
                keras.layers.Dropout(dropout_rate),
            ]
        )

    def call(self, inputs, training=False):
        hidden = inputs + self.attn(
            self.norm1(inputs), self.norm1(inputs), training=training
        )
        return hidden + self.mlp(self.norm2(hidden), training=training)

    def get_config(self):
        keys = ["embedding_dim", "num_heads", "hidden_dim", "dropout_rate"]
        return {**super().get_config(), **{k: getattr(self, k) for k in keys}}


@keras.saving.register_keras_serializable(package="Trans-UNET")
class TransUNet(keras.Model):
    """Hybrid CNN-Transformer U-Net for 2D multi-organ segmentation (Chen et al., 2021)."""

    def __init__(
        self,
        image_size=(224, 224, 3),
        patch_size=1,
        embedding_dim=64,
        num_heads=4,
        n_layers=6,
        hidden_dim=512,
        dropout_rate=0.1,
        num_classes=9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        feature_map_size = image_size[0] // 16
        self.cnn_backbone = _build_resnet_encoder()
        self.patch_embedding = PatchEmbedding(
            patch_size, embedding_dim, (feature_map_size // patch_size) ** 2
        )
        self.transformer_blocks = [
            TransformerBlock(embedding_dim, num_heads, hidden_dim, dropout_rate)
            for _ in range(n_layers)
        ]
        self.encoder_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        decoder_filters = [512, 256, 128, 64]
        self.upsamplers = [_Up(2, interpolation="bilinear") for _ in decoder_filters]
        self.concats = [keras.layers.Concatenate() for _ in decoder_filters[:3]]
        self.convs = [
            _C2D(f, 3, padding="same", use_bias=False) for f in decoder_filters
        ]
        self.gnorms = [_GN(groups=32) for _ in decoder_filters]
        self.acts = [keras.layers.Activation("relu") for _ in decoder_filters]
        self.segmentation_head = keras.layers.Conv2D(
            num_classes, 1, padding="same", activation="softmax"
        )

    def call(self, inputs, training=False):
        skip1, skip2, skip3, features = self.cnn_backbone(inputs, training=training)
        skips = [skip3, skip2, skip1]
        tokens = self.patch_embedding(features)
        for block in self.transformer_blocks:
            tokens = block(tokens, training=training)
        tokens = self.encoder_norm(tokens)
        fmap_size = self.image_size[0] // 16 // self.patch_size
        decoded = ops.reshape(tokens, (-1, fmap_size, fmap_size, self.embedding_dim))
        tr = training
        for i in range(4):
            decoded = self.upsamplers[i](decoded)
            if i < 3:
                decoded = self.concats[i]([decoded, skips[i]])
            decoded = self.acts[i](self.gnorms[i](self.convs[i](decoded), training=tr))
        return self.segmentation_head(decoded)

    def get_config(self):
        keys = ["image_size", "patch_size", "embedding_dim", "num_heads", "n_layers"]
        keys += ["hidden_dim", "dropout_rate", "num_classes"]
        return {**super().get_config(), **{k: getattr(self, k) for k in keys}}

```

---
## Loss and Metrics

We combine sparse categorical cross-entropy with a soft multi-class Dice loss,
weighted equally, The `MeanIoUWrapper` adapts
`keras.metrics.MeanIoU` to accept sparse integer labels directly.


```python

@keras.saving.register_keras_serializable(package="Trans-UNET")
def combined_loss(y_true, y_pred, smooth=1e-6):
    y_true = ops.squeeze(y_true, -1)
    ce = ops.mean(keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
    y_true_oh = ops.one_hot(
        ops.cast(y_true, "int32"), num_classes=ops.shape(y_pred)[-1]
    )
    y_pred = ops.clip(y_pred, 1e-6, 1.0)
    intersection = ops.sum(y_true_oh * y_pred, axis=[1, 2])
    union = ops.sum(y_true_oh + y_pred, axis=[1, 2])
    dice_loss = 1 - ops.mean((2.0 * intersection + smooth) / (union + smooth))
    return 0.5 * ce + 0.5 * dice_loss


@keras.saving.register_keras_serializable(package="Trans-UNET")
class MeanIoUWrapper(keras.metrics.MeanIoU):
    """MeanIoU adapted to accept sparse integer ground-truth labels."""

    def __init__(self, num_classes, name="mean_io_u_wrapper", **kwargs):
        for k in ("ignore_class", "sparse_y_true", "sparse_y_pred"):
            kwargs.pop(k, None)
        super().__init__(
            num_classes,
            name=name,
            sparse_y_true=True,
            sparse_y_pred=False,
            **kwargs,
        )
        self._num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(ops.squeeze(y_true, axis=-1), y_pred, sample_weight)

    def get_config(self):
        return {"num_classes": self._num_classes, "name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

```

---
## Training


```python
model = TransUNet(
    image_size=(IMAGE_SIZE, IMAGE_SIZE, 3),
    embedding_dim=PROJECTION_DIM,
    num_heads=4,
    n_layers=NUM_TRANSFORMER_LAYERS,
    hidden_dim=HIDDEN_DIM,
    dropout_rate=0.1,
    num_classes=NUM_CLASSES,
)
model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    ),
    loss=combined_loss,
    metrics=[MeanIoUWrapper(num_classes=NUM_CLASSES)],
)
_rlr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7
)
_es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=NUM_EPOCHS,
    callbacks=[_rlr, _es],
)
```

---
## Loss Curves


```python
_, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, train_key, val_key, ylabel in zip(
    axes,
    ["loss", "mean_io_u_wrapper"],
    ["val_loss", "val_mean_io_u_wrapper"],
    ["Loss", "mIoU"],
):
    ax.plot(history.history[train_key], label="Train")
    ax.plot(history.history[val_key], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
plt.tight_layout()
plt.show()

```

We train for 2 epochs for demonstration. Full convergence requires 60 epochs
(approximately 1 hour on a T4 GPU). The loss and mIoU curves below are
from the full 60-epoch run.

| ![Loss Curves](https://i.postimg.cc/pTChbNGb/training-history-(1).png) |
| :--: |
| **Training History: Loss and mIoU over 60 epochs.** |

---
## Inference


```python

def colorize_mask(mask):
    """Maps an integer label mask to an RGB image using PALETTE."""
    return PALETTE[ops.convert_to_numpy(ops.squeeze(mask)).astype(np.uint8)]


def run_inference(model, h5_paths, num_samples=5, n_slices=5):
    """Plots batched GT and prediction overlays for n_slices per volume."""
    cases = []
    for path in h5_paths[:num_samples]:
        with h5py.File(path, "r") as f:
            image, label = f["image"][:], f["label"][:]
        occupied = np.where(label.any(axis=(1, 2)))[0]
        if not len(occupied):
            continue
        idx_f = np.linspace(0, len(occupied) - 1, min(n_slices, len(occupied))).astype(
            int
        )
        idxs = occupied[idx_f]

        def resize_batch(arr, interp):
            r = ops.image.resize(
                arr[..., np.newaxis], (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp
            )
            return r.numpy().squeeze(-1)

        imgs = resize_batch(image[idxs].astype(np.float32), "bilinear")
        lbls = resize_batch(label[idxs], "nearest").astype(np.int32)
        x = np.stack([imgs] * 3, axis=-1)
        preds = np.argmax(model.predict(x, verbose=0), axis=-1)
        stem = os.path.splitext(os.path.basename(path))[0]
        cases.append((stem, idxs, imgs, lbls, preds))
    total_rows = sum(len(c[1]) for c in cases)
    fig, axes = plt.subplots(total_rows, 3, figsize=(12, total_rows * 4), squeeze=False)
    row = 0
    for name, idxs, imgs, lbls, preds in cases:
        for j, si in enumerate(idxs):
            overlays = [None, colorize_mask(lbls[j]), colorize_mask(preds[j])]
            tags = ["Input", "Ground Truth", "Prediction"]
            for col, (ov, tag) in enumerate(zip(overlays, tags)):
                axes[row, col].imshow(imgs[j], cmap="gray")
                if ov is not None:
                    axes[row, col].imshow(ov, alpha=0.5)
                axes[row, col].set_title(f"{name} | slice {si} - {tag}", fontsize=9)
                axes[row, col].axis("off")
            row += 1
    legend = [
        mpatches.Patch(color=PALETTE[i] / 255, label=CLASS_NAMES[i])
        for i in range(1, NUM_CLASSES)
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.show()


run_inference(model, RAW_VAL_FILES, num_samples=6, n_slices=5)
```

| ![Inference Results](https://i.postimg.cc/jdr93pVR/inference-case0022-npy.png) |
| :--: |
| **TransUNet predictions after 60 epochs. Ground truth (center) vs. prediction (right).** |

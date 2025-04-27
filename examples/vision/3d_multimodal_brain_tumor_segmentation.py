"""
Title: 3D Multimodal Brain Tumor Segmentation
Author: [Mohammed Innat](https://www.linkedin.com/in/innat2k14/)
Date created: 2025/04/27
Last modified: 2025/04/27
Description: How to implement 3D semantic segmentation for medical imaging.
Accelerator: GPU
"""

"""
## Introduction

This tutorial provides a step-by-step guide on how to use [medicai](https://github.com/innat/medic-ai), a [keras](https://keras.io/)-based medical image processing library that supports multiple backends. We will apply it to solve a multimodal Brain Tumor Segmentation task using the ([BraTS](https://www.med.upenn.edu/cbica/brats2020/data.html)) dataset. Throughout the tutorial, we will cover:

1. **Loading the Dataset**
    - Read TFRecord files that contain `image`, `label`, and `affine` matrix information.
    - Build efficient data pipelines using the `tf.data` API for training and evaluation.
2. **Medical Image Preprocessing**
    - Apply image transformations provided by `medicai` to prepare the data for model input.
3. **Model Building**
    - Use the [`SwinUNETR`](https://arxiv.org/abs/2201.01266) architecture from `medicai` for 3D medical image segmentation.
4. **Loss and Metrics Definition**
    - Implement a specialized Dice loss function and segmentation metrics available in `medicai` for better performance on medical data.
5. **Model Evaluation**
    - Use sliding window inference from `medicai` to efficiently perform inference on large 3D medical volumes.
    - Evaluate the model by calculating per-class metric scores after training is complete.
6. **Visualization of Results**
    - Visualize the model's predictions to qualitatively assess segmentation performance.

By the end of the tutorial, you will have a full workflow for brain tumor segmentation using `medicai`, from data loading to model evaluation and visualization.
"""

"""
## Installation
"""

"""shell
!pip install kagglehub -qU
!pip install medicai -qU
!pip install gdown -qU
"""

"""
## Acquiring data

**Understanding the BraTS Dataset and Preparing TFRecord Files**: The `.nii` format BraTS scans can be found here. We have converted the raw format to tfrecord format. The format process can be found here. The converted tfrecord dataset can be found here. Each tfrecord file contains max 10 file in each. As there is no publickly available validation or testing ground truth dataset, we can keep any tfrecord file(s) aside from training for validation.

[**Imaging Data Description**](https://www.med.upenn.edu/cbica/brats2020/data.html): All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe a) native (**T1**) and b) post-contrast T1-weighted (**T1Gd**), c) T2-weighted (**T2**), and d) T2 Fluid Attenuated Inversion Recovery (**T2-FLAIR**) volumes, and were acquired with different clinical protocols and various scanners from multiple (n=19) institutions, mentioned as data contributors here.

All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (**ET — label 4**), the peritumoral edema (**ED — label 2**), and the necrotic and non-enhancing tumor core (**NCR/NET — label 1**), as described both in the [BraTS 2012-2013 TMI paper](https://ieeexplore.ieee.org/document/6975210) and in the [latest BraTS summarizing paper](https://arxiv.org/abs/1811.02629). The provided data are distributed after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm^3) and skull-stripped. In short, the segmentation labels have values of 1 for **NCR**, 2 for **ED**, 4 for **ET**, and 0 for everything else.

**Note on Dataset Size and Experimental Setup**: The original BraTS 2020 dataset includes around 370 samples. However, due to resource limitations in free Colab environments, we will not be able to load the entire dataset. Instead, for this tutorial, we will select a few shared TFRecord files to run experiments. Again, since there is no official validation set with ground truth labels, we will create our own validation split by holding out a few samples from the training set.
"""

import os

import kagglehub

if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    kagglehub.login()

import shutil

dataset_id = "ipythonx/brats2020"
destination_path = "/content/brats2020_subset"
os.makedirs(destination_path, exist_ok=True)

# Download the 3 shards: 0 and 1st for training set, 36th for validation set.
for i in [0, 1, 36]:
    filename = f"training_shard_{i}.tfrec"
    print(f"Downloading {filename}...")
    path = kagglehub.dataset_download(dataset_id, path=filename)
    shutil.move(path, destination_path)

"""
## Imports
"""

os.environ["KERAS_BACKEND"] = "tensorflow"  # tensorflow, torch, jax

import gdown
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import ops
from matplotlib import pyplot as plt
from medicai.callbacks import SlidingWindowInferenceCallback
from medicai.losses import BinaryDiceCELoss
from medicai.metrics import BinaryDiceMetric
from medicai.models import SwinUNETR
from medicai.transforms import (
    Compose,
    CropForeground,
    NormalizeIntensity,
    RandFlip,
    RandShiftIntensity,
    RandSpatialCrop,
    TensorBundle,
)
from medicai.utils.inference import SlidingWindowInference

print(
    "keras backend: ",
    keras.config.backend(),
    "\n" "keras version: ",
    keras.version(),
    "\n" "tensorflow version: ",
    tf.__version__,
    "\n",
)

# enable mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")

# reproducibility
keras.utils.set_random_seed(101)

"""
## Create Multi-label Brain Tumor Labels

The BraTS segmentation task involves multiple tumor sub-regions, and it is formulated as a multi-label segmentation problem. The label combinations are used to define the following clinical regions of interest:

```shell
- Tumor Core (TC): label = 1 or 4
- Whole Tumor (WT): label = 1 or 2 or 4
- Enhancing Tumor (ET): label = 4
```

These region-wise groupings allow for evaluation across different tumor structures relevant for clinical assessment and treatment planning. A sample view is shown below, figure taken from [BraTS-benchmark](https://arxiv.org/pdf/2107.02314v1) paper.

![](https://drive.google.com/uc?export=view&id=1qVdMXAB84oYEDTt0xKLCEIl-XbB_fVWx)

To effectively organize and manage these multi-label segmentation outputs, we will implement a custom transformation utility using the **TensorBundle**. The **TensorBundle** is a simple container class API from `medicai` for holding a dictionary of tensors (and associated metadata if provided). This class is designed to bundle together the actual **data** (tensors) and any relevant **metadata** (e.g., affine transformations, original shapes, spacing) related to that data. It provides convenient access and modification methods for both the data and the metadata. Each `medicai` transformation API expects the input in a key-value pair format, organized as:

```shell
meta = {"affine": affine}
data = {"image": image, "label": label}
```
"""


class ConvertToMultiChannelBasedOnBratsClasses:
    """
    Convert labels to multi channels based on BRATS classes using TensorFlow.

    Label definitions:
    - 1: necrotic and non-enhancing tumor core
    - 2: peritumoral edema
    - 4: GD-enhancing tumor

    Output channels:
    - Channel 0 (TC): Tumor core (labels 1 or 4)
    - Channel 1 (WT): Whole tumor (labels 1, 2, or 4)
    - Channel 2 (ET): Enhancing tumor (label 4)
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, inputs):
        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        for key in self.keys:
            data = inputs.data[key]

            # TC: label == 1 or 4
            tc = tf.logical_or(tf.equal(data, 1), tf.equal(data, 4))

            # WT: label == 1 or 2 or 4
            wt = tf.logical_or(
                tf.logical_or(tf.equal(data, 1), tf.equal(data, 4)), tf.equal(data, 2)
            )

            # ET: label == 4
            et = tf.equal(data, 4)

            stacked = tf.stack(
                [
                    tf.cast(tc, tf.float32),
                    tf.cast(wt, tf.float32),
                    tf.cast(et, tf.float32),
                ],
                axis=-1,
            )

            inputs.data[key] = stacked
        return inputs


"""
## Transformation

Each `medicai` transformation expects the input to have the shape `(depth, height, width, channel)`. The original `.nii` (and converted `.tfrecord`) format contains the input shape of `(height, width, depth)`. To make it compatible with `medicai`, we need to re-arrange the shape axes.
"""


def rearrange_shape(sample):
    image = sample["image"]
    label = sample["label"]
    affine = sample["affine"]

    # special case
    image = tf.transpose(image, perm=[2, 1, 0, 3])  # whdc -> dhwc
    label = tf.transpose(label, perm=[2, 1, 0])  # whd -> dhw
    cols = tf.gather(affine, [2, 1, 0], axis=1)  # (whd) -> (dhw)
    affine = tf.concat([cols, affine[:, 3:]], axis=1)

    # re-update sample with new / updated tensor
    sample["image"] = image
    sample["label"] = label
    sample["affine"] = affine
    return sample


"""
Each transformation class of `medicai` expects input as either a dictionary or a `TensorBundle` object. When a dictionary of input data (along with metadata) is passed, it is automatically wrapped into a `TensorBundle` instance. The examples below demonstrate how transformations are used in this way.
"""


def train_transformation(sample):
    meta = {"affine": sample["affine"]}
    data = {"image": sample["image"], "label": sample["label"]}

    pipeline = Compose(
        [
            ConvertToMultiChannelBasedOnBratsClasses(keys=["label"]),
            CropForeground(
                keys=("image", "label"),
                source_key="image",
                k_divisible=[96, 96, 96],
            ),
            RandSpatialCrop(
                keys=["image", "label"], roi_size=(96, 96, 96), random_size=False
            ),
            RandFlip(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlip(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlip(keys=["image", "label"], spatial_axis=[2], prob=0.5),
            NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=True),
            RandShiftIntensity(keys=["image"], offsets=0.10, prob=1.0),
        ]
    )
    result = pipeline(data, meta)
    return result["image"], result["label"]


def val_transformation(sample):
    meta = {"affine": sample["affine"]}
    data = {"image": sample["image"], "label": sample["label"]}

    pipeline = Compose(
        [
            ConvertToMultiChannelBasedOnBratsClasses(keys=["label"]),
            NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=True),
        ]
    )
    result = pipeline(data, meta)
    return result["image"], result["label"]


"""
## The `tfrecord` parser
"""


def parse_tfrecord_fn(example_proto):
    feature_description = {
        # Image raw data
        "flair_raw": tf.io.FixedLenFeature([], tf.string),
        "t1_raw": tf.io.FixedLenFeature([], tf.string),
        "t1ce_raw": tf.io.FixedLenFeature([], tf.string),
        "t2_raw": tf.io.FixedLenFeature([], tf.string),
        "label_raw": tf.io.FixedLenFeature([], tf.string),
        # Image shape
        "flair_shape": tf.io.FixedLenFeature([3], tf.int64),
        "t1_shape": tf.io.FixedLenFeature([3], tf.int64),
        "t1ce_shape": tf.io.FixedLenFeature([3], tf.int64),
        "t2_shape": tf.io.FixedLenFeature([3], tf.int64),
        "label_shape": tf.io.FixedLenFeature([3], tf.int64),
        # Affine matrices (4x4 = 16 values)
        "flair_affine": tf.io.FixedLenFeature([16], tf.float32),
        "t1_affine": tf.io.FixedLenFeature([16], tf.float32),
        "t1ce_affine": tf.io.FixedLenFeature([16], tf.float32),
        "t2_affine": tf.io.FixedLenFeature([16], tf.float32),
        "label_affine": tf.io.FixedLenFeature([16], tf.float32),
        # Voxel Spacing (pixdim)
        "flair_pixdim": tf.io.FixedLenFeature([8], tf.float32),
        "t1_pixdim": tf.io.FixedLenFeature([8], tf.float32),
        "t1ce_pixdim": tf.io.FixedLenFeature([8], tf.float32),
        "t2_pixdim": tf.io.FixedLenFeature([8], tf.float32),
        "label_pixdim": tf.io.FixedLenFeature([8], tf.float32),
        # Filenames
        "flair_filename": tf.io.FixedLenFeature([], tf.string),
        "t1_filename": tf.io.FixedLenFeature([], tf.string),
        "t1ce_filename": tf.io.FixedLenFeature([], tf.string),
        "t2_filename": tf.io.FixedLenFeature([], tf.string),
        "label_filename": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode image and label data
    flair = tf.io.decode_raw(example["flair_raw"], tf.float32)
    t1 = tf.io.decode_raw(example["t1_raw"], tf.float32)
    t1ce = tf.io.decode_raw(example["t1ce_raw"], tf.float32)
    t2 = tf.io.decode_raw(example["t2_raw"], tf.float32)
    label = tf.io.decode_raw(example["label_raw"], tf.float32)

    # Reshape to original dimensions
    flair = tf.reshape(flair, example["flair_shape"])
    t1 = tf.reshape(t1, example["t1_shape"])
    t1ce = tf.reshape(t1ce, example["t1ce_shape"])
    t2 = tf.reshape(t2, example["t2_shape"])
    label = tf.reshape(label, example["label_shape"])

    # Decode affine matrices
    flair_affine = tf.reshape(example["flair_affine"], (4, 4))
    t1_affine = tf.reshape(example["t1_affine"], (4, 4))
    t1ce_affine = tf.reshape(example["t1ce_affine"], (4, 4))
    t2_affine = tf.reshape(example["t2_affine"], (4, 4))
    label_affine = tf.reshape(example["label_affine"], (4, 4))

    # add channel axis
    flair = flair[..., None]
    t1 = t1[..., None]
    t1ce = t1ce[..., None]
    t2 = t2[..., None]
    image = tf.concat([flair, t1, t1ce, t2], axis=-1)

    return {
        "image": image,
        "label": label,
        "affine": flair_affine,  # Since affine is the same for all
    }


"""
## Dataloader
"""


def load_tfrecord_dataset(tfrecord_datalist, batch_size=1, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_datalist)
    dataset = dataset.shuffle(buffer_size=100) if shuffle else dataset
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(rearrange_shape, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.map(train_transformation, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(val_transformation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


tfrecord_pattern = "/content/brats2020_subset/training_shard_*.tfrec"
datalist = sorted(
    tf.io.gfile.glob(tfrecord_pattern),
    key=lambda x: int(x.split("_")[-1].split(".")[0]),
)

train_datalist = datalist[:-1]
val_datalist = datalist[-1:]
print(len(train_datalist), len(val_datalist))

print(train_datalist)
print(val_datalist)

train_ds = load_tfrecord_dataset(train_datalist, batch_size=1, shuffle=True)
val_ds = load_tfrecord_dataset(val_datalist, batch_size=1, shuffle=False)

"""
**sanity check**
"""

val_x, val_y = next(iter(val_ds))
test_image = val_x.numpy().squeeze()
test_mask = val_y.numpy().squeeze()
print(test_image.shape, test_mask.shape, np.unique(test_mask))
print(test_image.min(), test_image.max())

slice_no = test_image.shape[0] // 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(test_image[slice_no], cmap="gray")
ax1.set_title(f"Image shape: {test_image.shape}")
ax2.imshow(test_mask[slice_no])
ax2.set_title(f"Label shape: {test_mask.shape}")
plt.show()

print(f"image shape: {test_image.shape}")
plt.figure("image", (24, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(test_image[slice_no, :, :, i], cmap="gray")
plt.show()


print(f"label shape: {test_mask.shape}")
plt.figure("label", (18, 6))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(f"label channel {i}")
    plt.imshow(test_mask[slice_no, :, :, i])
plt.show()

"""
## Model

We will be using the 3D model architecture [`SwinUNETR`](https://arxiv.org/abs/2201.01266). The BraTS dataset provides four input modalities: `flair`, `t1`, `t1ce`, and `t2`. Accordingly, we will build the model with **4** input channels and **3** output channels.

![](https://drive.google.com/uc?export=view&id=1cjELs4TXf8J9DY1gDzNzoYId8PN4Yzq2)
"""

num_classes = 3
epochs = 5
input_shape = (96, 96, 96, 4)

model = SwinUNETR(
    input_shape=input_shape,
    num_classes=num_classes,
    classifier_activation=None,
)

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5,
    ),
    loss=BinaryDiceCELoss(
        from_logits=True,
        num_classes=num_classes,
    ),
    metrics=[
        BinaryDiceMetric(
            from_logits=True,
            ignore_empty=True,
            num_classes=num_classes,
            name="dice",
        ),
        BinaryDiceMetric(
            from_logits=True,
            ignore_empty=True,
            class_ids=[0],
            num_classes=num_classes,
            name="dice_tc",
        ),
        BinaryDiceMetric(
            from_logits=True,
            ignore_empty=True,
            class_ids=[1],
            num_classes=num_classes,
            name="dice_wt",
        ),
        BinaryDiceMetric(
            from_logits=True,
            ignore_empty=True,
            class_ids=[2],
            num_classes=num_classes,
            name="dice_et",
        ),
    ],
)

model.summary(line_length=100)

"""
## Callback

We will be using sliding window inference callback to perform validation at certain interval or epoch during training. Based on the number of `epoch` size, we should set `interval` accordingly. For example, if `epoch` is set 15 and we want to evaluate model on validation set every 5 epoch, then we should set `interval` to 5.
"""

swi_callback_metric = BinaryDiceMetric(
    from_logits=True,
    ignore_empty=True,
    num_classes=num_classes,
    name="val_dice",
)

swi_callback = SlidingWindowInferenceCallback(
    model,
    dataset=val_ds,
    metrics=swi_callback_metric,
    num_classes=num_classes,
    interval=5,
    overlap=0.5,
    roi_size=(96, 96, 96),
    sw_batch_size=4,
    save_path="brats.model.weights.h5",
)

"""
## Training

Set more epoch for better optimization.
"""

history = model.fit(train_ds, epochs=epochs, callbacks=[swi_callback])

print(model.history.history.keys())
his_csv = pd.DataFrame(model.history.history)
his_csv.to_csv("brats.history.csv")
his_csv.head()


def plot_training_history(history_df):
    metrics = history_df.columns
    n_metrics = len(metrics)

    n_rows = 2
    n_cols = (n_metrics + 1) // 2  # ceiling division for columns

    plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    for idx, metric in enumerate(metrics):
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.plot(history_df[metric], label=metric, marker="o")
        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_history(his_csv)

"""
## Evaluation

In this [Kaggle notebook](https://www.kaggle.com/code/ipythonx/3d-brats-segmentation-in-keras-multi-gpu/) (version 2), we trained the model on the entire dataset for approximately 25 epochs. The resulting weights will be used for further evaluation. Note that the validation set used in both the Colab and Kaggle notebooks is the same: `training_shard_36.tfrec`, which contains `8` samples. We will be evaluating the model per-class and on average.
"""

model_weight = gdown.download(id="1b-DpZkMsAX6I-niuLu7h8_y3XiaSHFyN", quiet=False)

model.load_weights(model_weight)
swi = SlidingWindowInference(
    model,
    num_classes=num_classes,
    roi_size=(96, 96, 96),
    sw_batch_size=4,
    overlap=0.5,
)

dice = BinaryDiceMetric(
    from_logits=True,
    ignore_empty=True,
    num_classes=num_classes,
    name="dice",
)
dice_tc = BinaryDiceMetric(
    from_logits=True,
    ignore_empty=True,
    class_ids=[0],
    num_classes=num_classes,
    name="dice_tc",
)
dice_wt = BinaryDiceMetric(
    from_logits=True,
    ignore_empty=True,
    class_ids=[1],
    num_classes=num_classes,
    name="dice_wt",
)
dice_et = BinaryDiceMetric(
    from_logits=True,
    ignore_empty=True,
    class_ids=[2],
    num_classes=num_classes,
    name="dice_et",
)

for sample in val_ds:
    x, y = sample
    output = swi(x)

    y = ops.convert_to_tensor(y)
    output = ops.convert_to_tensor(output)

    dice.update_state(y, output)
    dice_tc.update_state(y, output)
    dice_wt.update_state(y, output)
    dice_et.update_state(y, output)

dice_score = float(ops.convert_to_numpy(dice.result()))
dice_score_tc = float(ops.convert_to_numpy(dice_tc.result()))
dice_score_wt = float(ops.convert_to_numpy(dice_wt.result()))
dice_score_et = float(ops.convert_to_numpy(dice_et.result()))

print(f"Dice Score: {dice_score:.4f}")
print(f"Dice Score on tumor core (TC): {dice_score_tc:.4f}")
print(f"Dice Score on whole tumor (WT): {dice_score_wt:.4f}")
print(f"Dice Score on enhancing tumor (ET): {dice_score_et:.4f}")

dice.reset_state()
dice_tc.reset_state()
dice_wt.reset_state()
dice_et.reset_state()

"""
## Analyse and Visualize
"""


def test_transformation(sample):
    data = {"image": sample["image"], "label": sample["label"]}
    pipeline = Compose(
        [
            ConvertToMultiChannelBasedOnBratsClasses(keys=["label"]),
            NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=True),
        ]
    )
    result = pipeline(data)
    return result["image"], result["label"]


index = 0
dataset = tf.data.TFRecordDataset(val_datalist[index])
dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.map(rearrange_shape, num_parallel_calls=tf.data.AUTOTUNE)

sample = next(iter(dataset))
orig_image = sample["image"]
orig_label = sample["label"]
orig_affine = sample["affine"]
print(orig_image.shape, orig_label.shape, orig_affine.shape, np.unique(orig_label))

pre_image, pre_label = test_transformation(sample)
print(pre_image.shape, pre_label.shape)

y_pred = swi(pre_image[None, ...])
y_pred.shape

y_pred_logits = y_pred.squeeze(axis=0)
y_pred_prob = ops.convert_to_numpy(ops.sigmoid(y_pred_logits))
segment = (y_pred_prob > 0.5).astype(int)
segment.shape, np.unique(segment)

label_map = {0: "TC", 1: "WT", 2: "ET"}

plt.figure(figsize=(16, 4))
for i in range(pre_label.shape[-1]):
    plt.subplot(1, 3, i + 1)
    plt.title(f"label channel {label_map[i]}")
    plt.imshow(pre_label[80, :, :, i])
plt.show()

plt.figure(figsize=(16, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(f"pred channel {label_map[i]}")
    plt.imshow(segment[80, :, :, i])
plt.show()

prediction = np.zeros(
    (segment.shape[0], segment.shape[1], segment.shape[2]), dtype="float32"
)
prediction[segment[..., 1] == 1] = 2
prediction[segment[..., 0] == 1] = 1
prediction[segment[..., 2] == 1] = 4

print("label ", orig_label.shape, np.unique(orig_label))
print("predicted ", prediction.shape, np.unique(prediction))

slice_map = {0: "flair", 1: "t1", 2: "t1ce", 3: "t2"}

plt.figure(figsize=(16, 4))
for i in range(orig_image.shape[-1]):
    plt.subplot(1, 4, i + 1)
    plt.title(f"Original channel: {slice_map[i]}")
    plt.imshow(orig_image[60, :, :, i], cmap="gray")

plt.tight_layout()
plt.show()

slice_num = 75
num_channels = orig_image.shape[-1]
plt.figure("image", (15, 15))

plt.subplot(3, num_channels, num_channels + 1)
plt.title("image")
plt.imshow(orig_image[slice_num, :, :, 0], cmap="gray")

# plotting labels and prediction
plt.subplot(3, num_channels, num_channels + 2)
plt.title("label")
plt.imshow(orig_label[slice_num, :, :])

plt.subplot(3, num_channels, num_channels + 3)
plt.title("prediction")
plt.imshow(prediction[slice_num, :, :])

plt.tight_layout()
plt.show()

"""
## Additional Resources

- [Mult-gpu BraTS Segmentation](https://www.kaggle.com/code/ipythonx/3d-brats-segmentation-in-keras-multi-gpu)
- [Conversion process of BraTS .nii to TFRecord](https://www.kaggle.com/code/ipythonx/brats-nii-to-tfrecord)
- [Covid-19 3D Image Segmentation](https://www.kaggle.com/code/ipythonx/medicai-covid-19-3d-image-segmentation)
- [3D Multi-organ Segmentation - BTCV Challenge](https://www.kaggle.com/code/ipythonx/medicai-3d-btcv-segmentation-in-keras)
- [Spleen 3D segmentation](https://www.kaggle.com/code/ipythonx/medicai-spleen-3d-segmentation-in-keras)
"""

"""
Title: Image Semantic Segmentation with DeepLabV3Plus
Author: Ian Stenbit, Divyashree Sreepathihalli
Date created: 2023/08/22
Last modified: 2023/08/23
Description: Train and use DeepLabV3Plus segmentation model with KerasCV.
Accelerator: GPU
"""

"""
Semantic segmentation is a type of computer vision task that involves assigning a
semantic label to each individual pixel of an image, effectively dividing the image into
regions that correspond to different object classes or categories.

This guide demonstrates how to finetune DeepLabV3Plus model with KerasCV.

![](https://storage.googleapis.com/keras-nlp/getting_started_guide/prof_keras_intermediate.png)
"""

"""
## Setup and Imports

First let's set up install and imports of the dependencies.

To run this tutorial, you will need to install keras-cv with the following command:

```
!pip install keras-core
!pip install git+https://github.com/keras-team/keras-cv.git
```
After installing keras-core and keras-cv, set the fbackend as tensorflow.

```
%env KERAS_BACKEND=tensorflow
```
"""

"""shell
pip install keras-core
pip install --upgrade -q git+https://github.com/keras-team/keras-cv
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from keras_cv.backend import keras

import numpy as np

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization
from keras_cv.backend import ops

from keras_cv.datasets.pascal_voc.segmentation import load

"""
## Download the data

We download
[Pascal VOC
dataset](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
with KerasCV datasets and split them into train dataset `train_ds` and `eval_ds`.
"""

train_ds = load(split="sbd_train")
eval_ds = load(split="sbd_eval")

"""
## Preprocess the data

The utility function `unpackage_tfds_inputs` is used processes the  inputs to a
dictionary of "images" and "segmentation_masks". The images and segmentation masks are
resized to 512x512. The resulting dataset is then batched into groups of 4 image and
segmentation mask pair.

A batch of this preprocessed input training data can then be visualized using
`keras_cv.visualization.plot_segmentation_mask_gallery`
"""


def unpackage_tfds_inputs(inputs):
    return {
        "images": inputs["image"],
        "segmentation_masks": inputs["class_segmentation"],
    }


train_ds = train_ds.map(unpackage_tfds_inputs)
train_ds = train_ds.map(keras_cv.layers.Resizing(height=512, width=512))
train_ds = train_ds.batch(4, drop_remainder=True)

batch = next(iter(train_ds.take(1)))

keras_cv.visualization.plot_segmentation_mask_gallery(
    batch["images"],
    value_range=(0, 255),
    num_classes=21,  # The number of classes for the oxford iiit pet dataset
    y_true=batch["segmentation_masks"],
    scale=3,
    rows=2,
    cols=2,
)

"""
The preprocessing is applied to the evaluation dataset `eval_ds`. A batch of `eval_ds`
can be visualized using `keras_cv.visualization.plot_segmentation_mask_gallery`.
"""

eval_ds = eval_ds.map(unpackage_tfds_inputs)
eval_ds = eval_ds.map(keras_cv.layers.Resizing(height=512, width=512))
eval_ds = eval_ds.batch(4, drop_remainder=True)

batch = next(iter(eval_ds.take(1)))

keras_cv.visualization.plot_segmentation_mask_gallery(
    batch["images"],
    value_range=(0, 255),
    num_classes=21,  # The number of classes for the oxford iiit pet dataset
    y_true=batch["segmentation_masks"],
    scale=3,
    rows=2,
    cols=2,
)

"""
## Data Augmentation

KerasCV provides a variety of image augmentation options. In this example, we will use
the `RandomFlip` augmentation to augment the training dataset. The `RandomFlip`
augmentation randomly flips the images in the training dataset horizontally or
vertically. This can help to improve the model's robustness to changes in the orientation
of the objects in the images.
"""

train_ds = train_ds.map(keras_cv.layers.RandomFlip())
batch = next(iter(train_ds.take(1)))

keras_cv.visualization.plot_segmentation_mask_gallery(
    batch["images"],
    value_range=(0, 255),
    num_classes=21,
    y_true=batch["segmentation_masks"],
    scale=3,
    rows=2,
    cols=2,
)

"""
## Model Configuration

Please feel free to modify the configurations for model training and note how the
training results changes. This is an great exercise to get a better understanding of the
training pipeline.

The learning rate schedule is used by the optimizer to calculate the learning rate for
each epoch. The optimizer then uses the learning rate to update the weights of the model.
In this case, the learning rate schedule uses a cosine decay function. A cosine decay
function starts high and then decreases over time, eventually reaching zero. The initial
learning rate is 0.007 and the decay steps are 2124. This means that the learning rate
will start at 0.007 and then decrease to zero over 2124 steps.
"""

BATCH_SIZE = 4
GLOBAL_BATCH_SIZE = BATCH_SIZE * 1
INITIAL_LR = 0.007 * GLOBAL_BATCH_SIZE / 16
EPOCHS = 100
NUM_CLASSES = 21
learning_rate = keras.optimizers.schedules.CosineDecay(
    INITIAL_LR,
    decay_steps=EPOCHS * 2124,
)

"""
Let us create an instance of the `DeepLabV3Plus` model for semantic segmentation. The
model architecture is initialized with `resnet50_v2_imagenet` preset and 21
classes. `resnet50_v2_imagenet` pre-trained weights will be used as the backbone feature
extractor for the DeepLabV3Plus model. `num_classes` parameter specifies the number of
classes that the model will be trained to segment.
"""

model = keras_cv.models.DeepLabV3Plus.from_preset(
    "resnet50_v2_imagenet", num_classes=NUM_CLASSES
)


"""
## Compile the model

The model.compile() function sets up the training process for the model. It defines the
- optimization algorithm - Stochastic Gradient Descent (SGD)
- the loss function - categorical cross-entropy
- the evaluation metrics - Mean IoU and categorical accuracy

that will be used during training and validation.
"""

model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=learning_rate, weight_decay=0.0001, momentum=0.9, clipnorm=10.0
    ),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[
        keras.metrics.MeanIoU(
            num_classes=NUM_CLASSES, sparse_y_true=False, sparse_y_pred=False
        ),
        keras.metrics.CategoricalAccuracy(),
    ],
)

model.summary()

"""
The utility function `dict_to_tuple` effectively transforms the dictionaries of training
and validation datasets into tuples of images and one-hot encoded segmentation masks,
which is used during training and evaluation of the `DeepLabV3Plus` model.
"""


def dict_to_tuple(x):
    return x["images"], tf.one_hot(
        tf.cast(tf.squeeze(x["segmentation_masks"], axis=-1), tf.int32), 21
    )


train_ds = train_ds.map(dict_to_tuple)
eval_ds = eval_ds.map(dict_to_tuple)

model.fit(train_ds, validation_data=eval_ds, epochs=EPOCHS)

"""
## Prediction with trained model
Now that the model training of DeepLabV3Plus has completed, let's test it by making
predications
on a few sample images.
"""

test_ds = load(split="sbd_test")
test_ds = test_ds.map(unpackage_tfds_inputs)
test_ds = test_ds.map(keras_cv.layers.Resizing(height=512, width=512))
test_ds = test_ds.batch(4, drop_remainder=True)

images, masks = next(iter(train_ds.take(1)))
preds = ops.expand_dims(ops.argmax(model(images), axis=-1), axis=-1)
masks = tf.expand_dims(tf.argmax(masks, axis=-1), axis=-1)

keras_cv.visualization.plot_segmentation_mask_gallery(
    images,
    value_range=(0, 255),
    num_classes=21,
    y_true=masks,
    y_pred=preds,
    scale=3,
    rows=1,
    cols=4,
)

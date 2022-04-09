"""
Title: State of the Arts Image Data Augmentation with KerasCV
Author: [lukewood](https://lukewood.xyz)
Date created: 2022/04/08
Last modified: 2022/04/08
Description: Use KerasCV to augment images with CutMix, MixUp, RandAugment, and more!
"""

"""
## Overview

KerasCV makes it easy to assemble state of the art, industry grade data augmentation
pipelines for image classification and object detection tasks.  KerasCV offers a wide
suite of preprocessing layers implementing common data augmentation techniques.

Perhaps three of the most useful layers are `CutMix`, `MixUp`, and `RandAugment`.  These
layers are used in nearly all state of the art image classification pipelines.

This guide will show you how to compose these layers into your own state of the art data
augmentation pipeline for image classification tasks.  This guide will also walk you
through the process of customizing a KerasCV data augmentation pipeline.
"""

"""
## Imports and Setup

This tutorial requires you to have KerasCV installed:

```shell
!pip install keras-cv
```

We will also begin by import all required packages:
"""

import keras_cv
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from keras_cv.layers import preprocessing
from tensorflow.keras import applications, losses, optimizers

"""
## Data Loading

This guide uses the oxford_flowers102 dataset for demonstration purposes.

To get started, we will first load the dataset:
"""

AUTOTUNE = tf.data.AUTOTUNE
data, ds_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)

"""
Next, we resize the images to a constant size, `(224, 224)`, and one hot encode the
labels.  Please note that `keras_cv.layers.CutMix` and `keras_cv.layers.MixUp` expect
`labels` to be one hot encoded.  This is because they modify the values held in `labels`
in a way that is not possible with a sparse label representation.
"""

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
num_classes = ds_info.features["label"].num_classes


def prepare(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, num_classes)
    return {"images": image, "labels": label}


def prepare_dataset(ds, split):
    if split == "train":
        return (
            ds.map(lambda x, y: prepare(x, y), num_parallel_calls=AUTOTUNE)
            .shuffle(10 * BATCH_SIZE)
            .batch(BATCH_SIZE)
        )
    if split == "test":
        return ds.map(lambda x, y: prepare(x, y), num_parallel_calls=AUTOTUNE).batch(
            BATCH_SIZE
        )


def load_dataset(split="train"):
    ds = data[split]
    return prepare_dataset(ds, split)


train_ds = load_dataset()

"""
Let's inspect some samples from our dataset:
"""


def visualize_dataset(ds, title):
    plt.figure(figsize=(6, 6)).suptitle(title, fontsize=18)
    for i, samples in enumerate(iter(ds.take(9))):
        images = samples["images"]
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


visualize_dataset(train_ds, title="Before Augmentation")

"""
Great!  Now we can move onto the augmentation step.
"""

"""
## RandAugment: Your One Stop Shop for Data Augmentation
"""

"""
While `CutMix` and `MixUp` have produced some great interclass examples, we still need to
perturb the images.  This prevents the model from overfitting the training distribution,
and improves the likelihood that the model can generalize to out of distribution
examples.

To remedy this issue KerasCV offers RandAugment, a general purpose image data
augmentation algorithm.  RandAugment has been shown to provide improved image
classification results across numerous datasets, and only has two hyperparameters to
configure.

Let's check out KerasCV's RandAugment implementation.
"""

rand_augment = preprocessing.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=3,
    magnitude=0.3,
    magnitude_stddev=0.2,
    rate=0.5,
)

"""
To use `RandAugment` you need to provide a few values:

- `value_range` describes the range of values covered in your images
- `magnitude` is a value between 0 and 10, describing the strength of the perturbations
applied
- `num_layers` is an integer telling the layer how many augmentations to apply to each
individual image
- (Optional) `magnitude_standard_deviation` allows `magnitude` to be randomly sampled
from a distribution with a standard deviation of `magnitude_standard_deviation`
- (Optional) `probability_to_apply` indicates the probability to apply the augmentation
applied at each layer.

That's all!  Let's apply our RandAugment layer:
"""


@tf.function
def apply_rand_augment(inputs):
    result = inputs.copy()
    result["images"] = rand_augment(inputs["images"])
    return result


train_ds = load_dataset().map(apply_rand_augment, num_parallel_calls=tf.data.AUTOTUNE)

"""
And finally lets inspect some of the results:
"""

visualize_dataset(train_ds, title="After RandAugment")

"""
Try tweaking the magnitude settings to see a wider variety of results.
"""

"""
## CutMix and MixUp: Generate High Quality Inter-class Examples

First, we will independently apply some data augmentation layers.  In this example, we
will use `CutMix` and `MixUp` independently in preprocessing pipeline.  In most state of
the art pipelines images are augmented by either `CutMix`, `MixUp`, or neither.  The
function below implements this in an equal 1/3 split.

Note that our `cut_mix_and_mix_up` function is annotated with a `tf.function` to ensure
optimal performance.
"""
cut_mix = preprocessing.CutMix()
mix_up = preprocessing.MixUp()


@tf.function
def cut_mix_and_mix_up(samples):
    choice = tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32)
    if choice < 1 / 3:
        return cut_mix(samples, training=True)
    elif choice < 2 / 3:
        return mix_up(samples, training=True)
    else:
        return samples


train_ds = load_dataset().map(cut_mix_and_mix_up, num_parallel_calls=tf.data.AUTOTUNE)

visualize_dataset(train_ds, title="After CutMix and MixUp")

"""
Great!  Looks like we have successfully added `CutMix` and `MixUp` to our preprocessing
pipeline.
"""

"""
## Customizing Your Augmentation Pipeline

Perhaps you want to exclude an augmentation from `RandAugment`, or perhaps you want to
include the `GridMask()` as an option alongside the default `RandAugment` augmentations.

KerasCV allows you to construct production grade custom data augmentation pipelines using
the `keras_cv.layers.RandomAugmentationPipeline` layer.  This class operates similarly to
`RandAugment`; selecting a random layer to apply to each image `augmentations_per_image`
times.  `RandAugment` can be thought of as a specific case of
`RandomAugmentationPipeline`.  In fact, our `RandAugment` implementation inherits from
`RandomAugmentationPipeline` internally.

In this example, we will create a custom `RandomAugmentationPipeline` by removing
`RandomRotation` layers from the standard `RandAugment` policy, and substitutex a
`GridMask` layer in its place.
"""

"""
As a first step, let's use the helper method `RandAugment.get_standard_policy()` to
create a base pipeline.
"""

layers = keras_cv.layers.RandAugment.get_standard_policy(
    value_range=(0, 255), magnitude=0.75, magnitude_stddev=0.3
)

"""
First, let's filter out `RandomRotation` layers
"""

layers = [
    layer for layer in layers if not isinstance(layer, keras_cv.layers.RandomRotation)
]

"""
Next, let's add `GridMask` to our layers:
"""

layers = layers + [keras_cv.layers.GridMask()]

"""
Finally, we can put together our pipeline
"""

pipeline = keras_cv.layers.RandomAugmentationPipeline(
    layers=layers, augmentations_per_image=3
)

"""
Let's check out the results!
"""


@tf.function
def apply_pipeline(inputs):
    inputs = inputs.copy()
    inputs["images"] = pipeline(inputs["images"])
    return inputs


train_ds = load_dataset().map(apply_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
visualize_dataset(train_ds, title="After custom pipeline")

"""
Awesome!  As you can see, no images were randomly rotated.  You can customize the
pipeline however you like:
"""

pipeline = keras_cv.layers.RandomAugmentationPipeline(
    layers=[keras_cv.layers.GridMask(), keras_cv.layers.Grayscale(output_channels=3)],
    augmentations_per_image=1,
)

"""
This pipeline will either apply `GrayScale` or GridMask:
"""


@tf.function
def apply_pipeline(inputs):
    inputs = inputs.copy()
    inputs["images"] = pipeline(inputs["images"])
    return inputs


train_ds = load_dataset().map(apply_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
visualize_dataset(train_ds, title="After custom pipeline")

"""
Looks great!  You can use RandomAugmentationPipeline however you want.
"""

"""
# Training a CNN

As a final exercise, let's take some of these layers for a spin.  In this section, we
will use `CutMix`, `MixUp`, and `RandAugment` to train a state of the art `ResNet50`
image classifier on the Oxford flowers dataset.

"""


def preprocess_for_model(inputs):
    images, labels = inputs["images"], inputs["labels"]
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


train_ds = (
    load_dataset()
    .map(apply_rand_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .map(cut_mix_and_mix_up, num_parallel_calls=tf.data.AUTOTUNE)
)

visualize_dataset(train_ds, "CutMix, MixUp and RandAugment")

train_ds = train_ds.map(preprocess_for_model, num_parallel_calls=tf.data.AUTOTUNE)

test_ds = load_dataset(split="test")
test_ds = test_ds.map(preprocess_for_model, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(5)
test_ds = test_ds.prefetch(5)

"""
Next we should create a the model itself.  Notice that we use `label_smoothing=0.1` in
the loss function.  When using MixUp, label smoothing is _highly_ recommended.
"""

input_shape = IMAGE_SIZE + (3,)


def get_model():
    model = applications.ResNet50(
        input_shape=input_shape, classes=num_classes, weights=None
    )
    model.compile(
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=optimizers.SGD(momentum=0.9),
        metrics=["accuracy"],
    )
    return model


"""
Finally we train the model:
"""

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = get_model()
    model.fit(
        train_ds,
        steps_per_epoch=1000,
        epochs=10,
        validation_data=test_ds,
        validation_steps=50,
    )

"""
# Conclusion & Next Steps
"""

"""
That's all it takes to assemble state of the art image augmentation pipeliens with
KerasCV!
"""

"""
As a further exercise for readers, you can:

- Perform a hyper parameter search over the RandAugment parameters to improve the
classifier accuracy
- Substitute the Oxford Flowers dataset with your own dataset
- Experiment with assembling your own augmentation technique instead of relying on
RandAugment

Currently, between Keras core and KerasCV there are [_28 image augmentation
layers_](https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/
__init__.py)!  Each of these can be used independently.  Check it out, and if you find
any augmentation techniques to be missing please file a [GitHub issue on
KerasCV](https://github.com/keras-team/keras-cv/issues).
"""

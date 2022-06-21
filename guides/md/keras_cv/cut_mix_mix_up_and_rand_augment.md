# CutMix, MixUp, and RandAugment image augmentation with KerasCV

**Author:** [lukewood](https://lukewood.xyz)<br>
**Date created:** 2022/04/08<br>
**Last modified:** 2022/04/08<br>
**Description:** Use KerasCV to augment images with CutMix, MixUp, RandAugment, and more.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_cv/cut_mix_mix_up_and_rand_augment.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/cut_mix_mix_up_and_rand_augment.py)



---
## Overview

KerasCV makes it easy to assemble state-of-the-art, industry-grade data augmentation
pipelines for image classification and object detection tasks. KerasCV offers a wide
suite of preprocessing layers implementing common data augmentation techniques.

Perhaps three of the most useful layers are `keras_cv.layers.CutMix`,
`keras_cv.layers.MixUp`, and `keras_cv.layers.RandAugment`. These
layers are used in nearly all state-of-the-art image classification pipelines.

This guide will show you how to compose these layers into your own data
augmentation pipeline for image classification tasks. This guide will also walk you
through the process of customizing a KerasCV data augmentation pipeline.

---
## Imports & setup

This tutorial requires you to have KerasCV installed:

```shell
pip install keras-cv
```

We begin by importing all required packages:


```python
import keras_cv
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import losses
from tensorflow.keras import optimizers
```

---
## Data loading

This guide uses the
[102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
for demonstration purposes.

To get started, we first load the dataset:


```python
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()
data, dataset_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
val_steps_per_epoch = dataset_info.splits["test"].num_examples // BATCH_SIZE
```

Next, we resize the images to a constant size, `(224, 224)`, and one-hot encode the
labels. Please note that `keras_cv.layers.CutMix` and `keras_cv.layers.MixUp` expect
targets to be one-hot encoded. This is because they modify the values of the targets
in a way that is not possible with a sparse label representation.


```python
IMAGE_SIZE = (224, 224)
num_classes = dataset_info.features["label"].num_classes


def to_dict(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, num_classes)
    return {"images": image, "labels": label}


def prepare_dataset(dataset, split):
    if split == "train":
        return (
            dataset.shuffle(10 * BATCH_SIZE)
            .map(to_dict, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
        )
    if split == "test":
        return dataset.map(to_dict, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)


def load_dataset(split="train"):
    dataset = data[split]
    return prepare_dataset(dataset, split)


train_dataset = load_dataset()
```

Let's inspect some samples from our dataset:


```python

def visualize_dataset(dataset, title):
    plt.figure(figsize=(6, 6)).suptitle(title, fontsize=18)
    for i, samples in enumerate(iter(dataset.take(9))):
        images = samples["images"]
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


visualize_dataset(train_dataset, title="Before Augmentation")
```



![png](/img/guides/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_9_0.png)



Great! Now we can move onto the augmentation step.

---
## RandAugment

[RandAugment](https://arxiv.org/abs/1909.13719)
has been shown to provide improved image
classification results across numerous datasets.
It performs a standard set of augmentations on an image.

To use RandAugment in KerasCV, you need to provide a few values:

- `value_range` describes the range of values covered in your images
- `magnitude` is a value between 0 and 1, describing the strength of the perturbations
applied
- `augmentations_per_image` is an integer telling the layer how many augmentations to apply to each
individual image
- (Optional) `magnitude_stddev` allows `magnitude` to be randomly sampled
from a distribution with a standard deviation of `magnitude_stddev`
- (Optional) `rate` indicates the probability to apply the augmentation
applied at each layer.

You can read more about these
parameters in the
[`RandAugment` API documentation](/api/keras_cv/layers/preprocessing/rand_augment/).

Let's use KerasCV's RandAugment implementation.


```python
rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=3,
    magnitude=0.3,
    magnitude_stddev=0.2,
    rate=0.5,
)


def apply_rand_augment(inputs):
    inputs["images"] = rand_augment(inputs["images"])
    return inputs


train_dataset = load_dataset().map(apply_rand_augment, num_parallel_calls=AUTOTUNE)
```

Finally, let's inspect some of the results:


```python
visualize_dataset(train_dataset, title="After RandAugment")
```



![png](/img/guides/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_15_0.png)



Try tweaking the magnitude settings to see a wider variety of results.

---
## CutMix and MixUp: generate high-quality inter-class examples


`CutMix` and `MixUp` allow us to produce inter-class examples. `CutMix` randomly cuts out
portions of one image and places them over another, and `MixUp` interpolates the pixel
values between two images. Both of these prevent the model from overfitting the
training distribution and improve the likelihood that the model can generalize to out of
distribution examples. Additionally, `CutMix` prevents your model from over-relying on
any particular feature to perform its classifications. You can read more about these
techniques in their respective papers:

- [CutMix: Train Strong Classifiers](https://arxiv.org/abs/1905.04899)
- [MixUp: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

In this example, we will use `CutMix` and `MixUp` independently in a manually created
preprocessing pipeline. In most state of the art pipelines images are randomly
augmented by either `CutMix`, `MixUp`, or neither. The function below implements both.


```python
cut_mix = keras_cv.layers.CutMix()
mix_up = keras_cv.layers.MixUp()


def cut_mix_and_mix_up(samples):
    samples = cut_mix(samples, training=True)
    samples = mix_up(samples, training=True)
    return samples


train_dataset = load_dataset().map(cut_mix_and_mix_up, num_parallel_calls=AUTOTUNE)

visualize_dataset(train_dataset, title="After CutMix and MixUp")
```



![png](/img/guides/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_18_0.png)



Great! Looks like we have successfully added `CutMix` and `MixUp` to our preprocessing
pipeline.

---
## Customizing your augmentation pipeline

Perhaps you want to exclude an augmentation from `RandAugment`, or perhaps you want to
include the `keras_cv.layers.GridMask` as an option alongside the default `RandAugment`
augmentations.

KerasCV allows you to construct production grade custom data augmentation pipelines using
the `keras_cv.layers.RandomAugmentationPipeline` layer. This class operates similarly to
`RandAugment`; selecting a random layer to apply to each image `augmentations_per_image`
times. `RandAugment` can be thought of as a specific case of
`RandomAugmentationPipeline`. In fact, our `RandAugment` implementation inherits from
`RandomAugmentationPipeline` internally.

In this example, we will create a custom `RandomAugmentationPipeline` by removing
`RandomRotation` layers from the standard `RandAugment` policy, and substitutex a
`GridMask` layer in its place.

As a first step, let's use the helper method `RandAugment.get_standard_policy()` to
create a base pipeline.


```python
layers = keras_cv.layers.RandAugment.get_standard_policy(
    value_range=(0, 255), magnitude=0.75, magnitude_stddev=0.3
)
```

First, let's filter out `RandomRotation` layers


```python
layers = [
    layer for layer in layers if not isinstance(layer, keras_cv.layers.RandomRotation)
]
```

Next, let's add `keras_cv.layers.GridMask` to our layers:


```python
layers = layers + [keras_cv.layers.GridMask()]
```

Finally, we can put together our pipeline


```python
pipeline = keras_cv.layers.RandomAugmentationPipeline(
    layers=layers, augmentations_per_image=3
)


def apply_pipeline(inputs):
    inputs["images"] = pipeline(inputs["images"])
    return inputs

```

Let's check out the results!


```python
train_dataset = load_dataset().map(apply_pipeline, num_parallel_calls=AUTOTUNE)
visualize_dataset(train_dataset, title="After custom pipeline")
```



![png](/img/guides/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_30_0.png)



Awesome! As you can see, no images were randomly rotated. You can customize the
pipeline however you like:


```python
pipeline = keras_cv.layers.RandomAugmentationPipeline(
    layers=[keras_cv.layers.GridMask(), keras_cv.layers.Grayscale(output_channels=3)],
    augmentations_per_image=1,
)
```

This pipeline will either apply `GrayScale` or GridMask:


```python

train_dataset = load_dataset().map(apply_pipeline, num_parallel_calls=AUTOTUNE)
visualize_dataset(train_dataset, title="After custom pipeline")
```



![png](/img/guides/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_34_0.png)



Looks great! You can use `RandomAugmentationPipeline` however you want.

---
## Training a CNN

As a final exercise, let's take some of these layers for a spin. In this section, we
will use `CutMix`, `MixUp`, and `RandAugment` to train a state of the art `ResNet50`
image classifier on the Oxford flowers dataset.


```python

def preprocess_for_model(inputs):
    images, labels = inputs["images"], inputs["labels"]
    images = tf.cast(images, tf.float32)
    return images, labels


train_dataset = (
    load_dataset()
    .map(apply_rand_augment, num_parallel_calls=AUTOTUNE)
    .map(cut_mix_and_mix_up, num_parallel_calls=AUTOTUNE)
)

visualize_dataset(train_dataset, "CutMix, MixUp and RandAugment")

train_dataset = train_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

test_dataset = load_dataset(split="test")
test_dataset = test_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

train_dataset = train_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

train_dataset = train_dataset
test_dataset = test_dataset
```



![png](/img/guides/cut_mix_mix_up_and_rand_augment/cut_mix_mix_up_and_rand_augment_37_0.png)



Next we should create a the model itself. Notice that we use `label_smoothing=0.1` in
the loss function. When using `MixUp`, label smoothing is _highly_ recommended.


```python
input_shape = IMAGE_SIZE + (3,)


def get_model():
    model = keras_cv.models.DenseNet121(
        include_rescaling=True, include_top=True, classes=num_classes
    )
    model.compile(
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=optimizers.SGD(momentum=0.9),
        metrics=["accuracy"],
    )
    return model

```

Finally we train the model:


```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = get_model()
    model.fit(
        train_dataset,
        epochs=1,
        validation_data=test_dataset,
    )
```

<div class="k-default-codeblock">
```
32/32 [==============================] - 769s 24s/step - loss: 4.7812 - accuracy: 0.0108 - val_loss: 4.6148 - val_accuracy: 0.0241
```
</div>
---
## Conclusion & next steps

That's all it takes to assemble state of the art image augmentation pipeliens with
KerasCV!

As an additional exercise for readers, you can:

- Perform a hyper parameter search over the RandAugment parameters to improve the
classifier accuracy
- Substitute the Oxford Flowers dataset with your own dataset
- Experiment with custom `RandomAugmentationPipeline` objects.

Currently, between Keras core and KerasCV there are
[_28 image augmentation layers_](https://keras.io/api/keras_cv/layers/preprocessing)!
Each of these can be used independently, or in a pipeline. Check them out, and if you
find an augmentation techniques you need is missing please file a
[GitHub issue on KerasCV](https://github.com/keras-team/keras-cv/issues).

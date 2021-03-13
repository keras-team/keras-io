"""
Title: RandAugment for Image Classification for Improved Robustness
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/03/13
Last modified: 2021/03/13
Description: Applying RandAugment augmentation for training an image classification model with improved robustness.
"""
"""
Data augmentation is a very useful technique that helps to improve the translational
invariance of convolutional neural networks (CNN). RandAugment is a stochastic data
augmentation routine for vision data and was proposed in [RandAugment: Practical
automated data augmentation with a reduced search
space](https://arxiv.org/abs/1909.13719). It is composed of strong augmentation
transforms like color jitters, Gaussian blurs, saturations, etc. along with more
traditional augmentation transforms such as random crops.

RandAugment has two parameters:
* `n` that denotes the number of randomly selected augmentation transforms to apply
sequentially
* `m` strength of all the augmentation transforms

These parameters are tuned for a given dataset and a network architecture. The authors of
RandAugment also provide pseudocode of RandAugment in the original paper (Figure 2):

![](https://i.ibb.co/Df6Ynxd/image.png)

Recently, it has been a key component of works like [Noisy Student Training](https://arxiv.org/abs/1911.04252) and
[Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848). It has been also central to the
success of [EfficientNets](https://arxiv.org/abs/1905.11946).

This example requires TensorFlow 2.4 or higher, as well as
[`imgaug`](https://imgaug.readthedocs.io/), which can be installed using the following
command:
"""

"""shell
pip install -U -q imgaug
"""

"""
## Setup
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

from imgaug import augmenters as iaa
import imgaug as ia

"""
## Load the CIFAR10 dataset

For this example, we will be using the [CIFAR10
dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"Total training examples: {len(x_train)}")
print(f"Total test examples: {len(x_test)}")

"""
## Define hyperparameters
"""

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 10

"""
## Initialize `RandAugment` object

Now, we will initialize a `RandAugment` object from the `imgaug.augmenters` module with
the parameters suggested by the RandAugment authors.
"""

rand_aug = iaa.RandAugment(n=3, m=7)


def augment(images):
    # Input to `augment()` is a TensorFlow tensor which
    # is not supported by `imgaug`. This is why we first
    # convert it to its `numpy` variant.
    return rand_aug(images=images.numpy())


"""
## Create TensorFlow `Dataset` objects

There's one problem, though. We cannot map our `augment()` function to a TensorFlow
`Dataset` object. To tackle this problem, we will make use of the
[`tf.py_function`](https://www.tensorflow.org/api_docs/python/tf/py_function) that can
convert a Python function into a TensorFlow op.
"""

train_ds_rand = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
    .map(
        lambda x, y: (tf.py_function(augment, [x], [tf.float32]), y),
        num_parallel_calls=AUTO,
    )
    # The returned output contains an unncessary axis of
    # 1-D and we need to remove it.
    .map(lambda x, y: (tf.squeeze(x), y), num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

"""
**Note about using `tf.py_function`**:

As our `augment()` function is not a native TensorFlow operation chances are likely that
it can turn into an expensive operation. This is why it is much better to apply it
_after_ batching our dataset.
"""

"""
For comparison purposes, let's also define a simple augmentation pipeline consisting of
random flips, random rotations, and random zoomings.
"""

simple_aug = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(factor=0.02),
        layers.experimental.preprocessing.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

# Now, map the augmentation pipeline to our training dataset
train_ds_simple = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (simple_aug(x), y), num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

"""
## Visualize the dataset augmented with RandAugment
"""

sample_images, _ = next(iter(train_ds_rand))
plt.figure(figsize=(8, 8))
for i, image in enumerate(sample_images[:9]):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("int"))
    plt.axis("off")

"""
You are encouraged to run the above code block a couple of times to see different
variations.
"""

"""
## Visualize the dataset augmented with `simple_aug`
"""

sample_images, _ = next(iter(train_ds_simple))
plt.figure(figsize=(8, 8))
for i, image in enumerate(sample_images[:9]):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("int"))
    plt.axis("off")

"""
## Define a model building utility function

Now, we define a shallow CNN. Also, notice that the network already has a rescaling layer
inside it. This eliminates the need to do any separate preprocessing on our dataset and
is specifically very useful for deployment purposes.
"""


def get_training_model():
    model = tf.keras.Sequential(
        [
            layers.Input((32, 32, 3)),
            layers.experimental.preprocessing.Rescaling(1.0 / 255),
            layers.Conv2D(16, (5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, (5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.2),
            layers.GlobalAvgPool2D(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


"""
We will train this network on two different versions of our dataset:
* One augmented with RandAugment.
* Another one augmented with `simple_aug`.

Since RandAugment is known to enhance the robustness of models to common perturbations
and corruptions, we will also evaluate our models on the CIFAR-10-C dataset, proposed in
[Benchmarking Neural Network Robustness to Common Corruptions and
Perturbations](https://arxiv.org/abs/1903.12261) by Hendrycks et al. CIFAR-10-C dataset
consists of 19 different image corruptions and perturbations (for example speckle noise,
fog, Gaussian blur, etc.) that too at varying severity levels. For this example we will
be using the following configuration:
[`cifar10_corrupted/saturate_5`](https://www.tensorflow.org/datasets/catalog/cifar10_corrupted#cifar10_corruptedsaturate_5). The images from this configuration look like so:

![](https://storage.googleapis.com/tfds-data/visualization/fig/cifar10_corrupted-saturate_5-1.0.0.png)

For the sake of reproducibility, we serialize the initial random weights of our shallow
network.
"""

initial_model = get_training_model()
initial_model.save_weights("initial_weights.h5")

"""
#1. Train model with RandAugment
"""

rand_aug_model = get_training_model()
rand_aug_model.load_weights("initial_weights.h5")
rand_aug_model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
rand_aug_model.fit(train_ds_rand, validation_data=test_ds, epochs=EPOCHS)
_, test_acc = rand_aug_model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_acc * 100))

"""
# 2. Train model with `simple_aug`
"""

simple_aug_model = get_training_model()
simple_aug_model.load_weights("initial_weights.h5")
simple_aug_model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
simple_aug_model.fit(train_ds_simple, validation_data=test_ds, epochs=EPOCHS)
_, test_acc = simple_aug_model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_acc * 100))

"""
## Load the CIFAR-10-C dataset and evaluate performance
"""

# Load and prepare the CIFAR-10-C dataset
cifar_10_c = tfds.load("cifar10_corrupted/saturate_5", split="test", as_supervised=True)
cifar_10_c = cifar_10_c.batch(BATCH_SIZE)

# Evaluate `rand_aug_model`
_, test_acc = rand_aug_model.evaluate(cifar_10_c, verbose=0)
print(
    "Accuracy with RandAugment on CIFAR-10-C (saturate_5): {:.2f}%".format(
        test_acc * 100
    )
)

# Evaluate `simple_aug_model`
_, test_acc = simple_aug_model.evaluate(cifar_10_c, verbose=0)
print(
    "Accuracy with simple_aug on CIFAR-10-C (saturate_5): {:.2f}%".format(
        test_acc * 100
    )
)

"""
As we can see at the expense of increased training time with RandAugment, we are able to
carve out slightly better performance on the CIFAR-10-C dataset. With a deeper model and
longer training schedule, this performance will likely improve. You can run the same
experiment on the other corruption and perturbation settings that come with the
CIFAR-10-C dataset and see if RandAugment helps.

Readers are encouraged to experiment with the different values of `n` and `m` in the
`RandAugment` object. In the [original paper](https://arxiv.org/abs/1909.13719), the
authors show the impact of the individual augmentation transforms for a particular task
and a range of ablation studies. You are welcome to check them out.

RandAugment has shown great progress in improving the robustness of deep models for
computer vision as shown in works like [Noisy Student Training](https://arxiv.org/abs/1911.04252) and
[FixMatch](https://arxiv.org/abs/2001.07685). This makes RandAugment quite a useful
recipe for training different vision models.
"""

"""
Title: mixup augmentation for image classification
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2020/03/06
Last modified: 2020/03/06
Description: Data augmentation using the mixup technique for image classification.
"""
"""
## Introduction
"""

"""
mixup is a *domain-agnostic* data augmentation technique proposed in [mixup: Beyond
Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) by Zhang et al. It's
implemented with the following formulas:

* $\tilde{x}=\lambda x_{i}+(1-\lambda) x_{j}$, where $x_{i}$ and $x_{j}$ are input
features
* $\bar{y}=\lambda y_{i}+(1-\lambda) y_{j}$, where $y_{i}$ and $y_{j}$ are one-hot
encoded labels

(Note that $lambda x_{i}$ and $lambda y_{i}$ are values with the [0, 1] range and are
sampled from the [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution).)

The technique is quite holistically named - we are literally mixing up the features and
their corresponding labels. Implementation-wise it's simple. mixup can be extended to a
variety of data modalities such as computer vision, natural language processing, speech,
and so on.

This example requires TensorFlow 2.4 or higher, as well as TensorFlow Probability, which
can be installed using the following command:
"""

"""shell
pip install tensorflow-probability
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as L
from tensorflow.keras.utils import to_categorical

import tensorflow_probability as tfp

tfd = tfp.distributions

"""
## Prepare the dataset

In this example, we will be using the [FashionMNIST](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/) dataset. But this same recipe can
be used for other classification datasets as well.
"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))
y_train = to_categorical(y_train, 10)

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))
y_test = to_categorical(y_test, 10)

"""
## Define hyperparameters
"""

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 64
EPOCHS = 5

"""
## Convert the data into TensorFlow `Dataset` objects
"""

train_ds_one = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
)
train_ds_two = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
)
# Because we will be mixing up the images and their corresponding labels, we will be
# combining two shuffled datasets from the same training data.
train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

"""
## Define the mixup technique function

To perform the mixup routine, we create new virtual datasets using the training data from
the same dataset, and apply a $\lambda$ value within the [0, 1] range sampled from a [Beta
distribution](https://en.wikipedia.org/wiki/Beta_distribution) — such that, for example, `new_x = lambda * x1 + (1 - lambda) * x2` (where
`x1` and `x2` are images) and the same equation is applied to the labels as well.
"""


def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = tfd.Beta(0.2, 0.2).sample(batch_size)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)


"""
**Note** that here , we are combining two images to create a single one. Theoretically,
we can combine as many we want but that comes at an increased computation cost. In
certain cases, it may not help improve the performance as well.
"""

"""
## Visualize the new augmented dataset
"""

# First create the new dataset using our `mix_up` utility
train_ds_mu = train_ds.map(
    lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
)

# Let's preview 9 samples from the dataset
sample_images, sample_labels = next(iter(train_ds_mu))
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().squeeze())
    print(label.numpy().tolist())
    plt.axis("off")

"""
## Model building
"""


def get_training_model():
    model = tf.keras.Sequential()
    model.add(L.Conv2D(16, (5, 5), activation="relu", input_shape=(28, 28, 1)))
    model.add(L.MaxPooling2D(pool_size=(2, 2)))
    model.add(L.Conv2D(32, (5, 5), activation="relu"))
    model.add(L.MaxPooling2D(pool_size=(2, 2)))
    model.add(L.Dropout(0.2))
    model.add(L.GlobalAvgPool2D())
    model.add(L.Dense(128, activation="relu"))
    model.add(L.Dense(10, activation="softmax"))
    return model


"""
For the sake of reproducibility, we serialize the initial random weights of our shallow
network.
"""

initial_model = get_training_model()
initial_model.save_weights("initial_weights.h5")

"""
## 1. Train the model with the mixed up dataset
"""

model = get_training_model()
model.load_weights("initial_weights.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_ds_mu, validation_data=test_ds, epochs=EPOCHS)
_, test_acc = model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_acc * 100))

"""
## 2. Train the model *without* the mixed up dataset
"""

model = get_training_model()
model.load_weights("initial_weights.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Notice that we are NOT using the mixed up dataset here
model.fit(train_ds_one, validation_data=test_ds, epochs=EPOCHS)
_, test_acc = model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_acc * 100))

"""
Readers are encouraged to try out mixup on different datasets from different domains and
experiment with the $\lambda$ parameter. You are strongly advised to check out the
[original paper](https://arxiv.org/abs/1710.09412) as well - the authors present several ablation studies on mixup
showing how it can improve generalization, as well as show their results of combining
more than two images to create a single one.
"""

"""
## Notes

* With mixup, you can create synthetic examples — especially when you lack a large
dataset - without incurring high computational costs.
* [Label smoothing](https://www.pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/) and mixup usually do not work well together because label smoothing
already modifies the hard labels by some factor.
* mixup does not work well when you are using [Supervised Contrastive
Learning](https://arxiv.org/abs/2004.11362) (SCL) since SCL expects the true labels
during its pre-training phase.
* There are a number of data augmentation techniques that extend mixup such as
[CutMix](https://arxiv.org/abs/1905.04899) and [AugMix](https://arxiv.org/abs/1912.02781).
"""

"""
Title: VICReg: Variance-Invariance-Covariance Regularization for SSL
Author: Abhiraam Eranti
Date created: 4/13/2022
Last modified: 4/13/2022
Description: We implement VICReg using Tensorflow Similarity and train on CIFAR-10.
"""
"""
## Introduction
"""

"""
**Problem**

VicReg, created by Adrien Bardes, Jean Ponce, and Yann LeCun, is a self-supervised method
to generate high-quality embeddings that
maximize the amount of dataset-related information inside them.

Previously, the main way to get these kinds of embeddings was to just calculate
the distance between representations of similar and different images.
Ideally, similar images would have similar embeddings and different images
have different ones. However, there was one problem: they
would *collapse*, or try to "cheat the system". Let's look at an example:

Suppose we had an image of a cat and a dog. The embeddings should
primarily store information from the images that differentiate the cat from its canine
counterpart. For example, it could keep the shape of the ears of both images, or maybe
the tail length, and so on. When used in a downstream task,
like a classification model, these embeddings (which have the curated measurements
mentioned above) should assist the model.

However, instead of this occurring, these approaches would produce embeddings that did
not help the downstream model as much as they should have. This is because they would
become redundant, meaning they repeated information more
than once. This led to less information being passed to the downstream model.

The previous solutions were to carefully and precisely tune the weights and
augmentations of the model and data such that collapse does not occur. However,
this was a finicky task, and even then, redundancy was still an issue.

**Solutions**

VicReg was not the first solution to this. Barlow Twins is
another similar method that was designed to reduce redundancy by measuring both
the invariance and covariance of embeddings. It works pretty well at
doing this, and is generally better in performance to contrastive models like
SimCLR.

VicReg is inspired by Barlow Twins and shares a similar performance to it
on tasks like image classification, the example shown here. Instead of just
measuring the invariance and covariance, it measures *similarity*, *variance*,
and *covariance* concerning the embeddings instead. However, they share the
same model composition and training loop, and both are substantially simpler
than other methods to train.

However, VicReg outperforms Barlow Twins on multimodal tasks like
image-to-text and text-to-image translation.
"""

"""
We will also utilize **Tensorflow Similarity**, a library designed to make metric and
self-supervised learning easier for practical use. Using this
library, we do not need to make the augmentations, model architectures,
training loop, and visualization code ourselves.
"""

"""
### References
[VicReg Paper](https://arxiv.org/abs/2105.04906)

[VicReg PyTorch Implementation](https://github.com/facebookresearch/vicreg)

[Barlow Twins Paper](https://arxiv.org/abs/2103.03230)

[Barlow Twins Example](https://keras.io/examples/vision/barlow_twins/)

[Tensorflow Similarity](https://github.com/tensorflow/similarity)
"""

"""
## Installation and Imports
"""

"""
We need `tensorflow-addons` for the LAMB loss function and
`tensorflow-similarity` for our augmenting, model building, and training setup.
"""

"""shell
!pip install tensorflow-addons
!pip install tensorflow-similarity
"""

import os

# slightly faster improvements, on the first epoch 30 second decrease and a 1-2 second
# decrease in epoch time. Overall saves approx. 5 min of training time

# Allocates two threads for a gpu private which allows more operations to be
# done faster
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

import tensorflow as tf  # framework
from tensorflow import keras  # for tf.keras
import tensorflow_addons as tfa  # LAMB optimizer and gaussian_blur_2d function
import numpy as np  # np.random.random
import matplotlib.pyplot as plt  # graphs
import datetime  # tensorboard logs naming
import tensorflow_similarity  # loss function module
from functools import partial

# XLA optimization for faster performance(up to 10-15 minutes total time saved)
tf.config.optimizer.set_jit(True)

"""
## Dataset preparation
"""

"""
### Data loading
"""

"""
We will be using CIFAR-10 as it is a nice baseline for our task. Because it has been used
for several other models, we can compare our results with other methods.

For the sake of time, we will only use 30% of the dataset, or around 18000 images for
this experiment. 15000 will be unlabeled images used during the VicReg process, and only
3000 labeled images will be used to train our linear evaluation model. Because of this,
we will see subpar results from our model. Try running this project in an interactive
notebook while changing that `DATASET_PERCENTAGE` constant to be higher.
"""

# Batch size of dataset
BATCH_SIZE = 512
# Width and height of image
IMAGE_SIZE = 32

[
    (train_features, train_labels),
    (test_features, test_labels),
] = keras.datasets.cifar10.load_data()

DATASET_PERCENTAGE = 0.3
train_features = train_features[: int(len(train_features) * DATASET_PERCENTAGE)]
test_features = test_features[: int(len(test_features) * DATASET_PERCENTAGE)]
train_labels = train_labels[: int(len(train_labels) * DATASET_PERCENTAGE)]
test_labels = test_labels[: int(len(test_labels) * DATASET_PERCENTAGE)]

train_features = train_features / 255.0
test_features = test_features / 255.0

"""
### Augmentation
"""

"""
VicReg uses the same augmentation pipeline as both Barlow Twins and BYOL. These
augmentations occur probabilistically, which allows for even more variation to help the
model learn.

<details>
<summary>Augmentation Pipeline Details</summary>

The pipeline is as follows:
* ***Random Crop***: We crop a random part of the image out. This resulting cropped image
is between 75% and 100% of the image size. Then, the cropped image is resized to the
original image width and height.
* ***Random Horizontal Flip*** (*50%*): There is a *50%* probability that the image will
be flipped horizontally
* ***Random Color Jitter*** (*80%*): There is an *80%* probability that the image will be
discolored. This process includes:
  * Random brightness (additive), ranging from `-0.8` to `+0.8`.
  * Random contrast (multiplicative), ranging from `0.4` to `1.6`
  * Random saturation (multiplicative), ranging from `0.4` to `1.6`
  * Random hue (multiplicative), ranging from `0.8` to `1.2`
* ***Random Greyscale*** (*20%*)
* ***Random Gaussian Blur***(*20%*): The blur amount σ ranges from `0` to
`1`
* ***Random Solarization***(*20%*): Solarization is when very low pixels get
inverted to do to irregularities in the camera. The solarization threshold for this
pipeline is `10`. If a pixel(not normalized) is below `10`, it will be
flipped to `255-pixel`.

Instead of implementing these pipelines ourselves, Tensorflow Similarity has
a collection of augmenters that we can use instead. In this case, we will be
using the pipeline function
`tensorflow_similarity.augmenters.barlow.augment_barlow` that takes in an image
and returns an augmented version using these transforms.
</details>

<details>
<summary> Dataset method </summary>

We'll use this function in the `tf.data.Dataset` API due to its ease of
use when batching and mapping. However, Tensorflow Similarity offers a simpler method
with it's augmenter library.


You can use `tensorflow_similarity.augmenters.BarlowAugmenter()` as a callable.
However, be aware that it *does* load the dataset into RAM, and you may have to
handle extra preprocessing (like batching) separately.
</details>
"""

# Saves a few minutes of performance - disables intra-op parallelism
performance_options = tf.data.Options()
performance_options.threading.max_intra_op_parallelism = 1

# Adding image width and height to augmenter
configed_augmenter = partial(
    tensorflow_similarity.augmenters.barlow.augment_barlow,
    height=IMAGE_SIZE,
    width=IMAGE_SIZE,
)


def make_version():
    augment_version = (
        tf.data.Dataset.from_tensor_slices(train_features)
        .map(configed_augmenter, tf.data.AUTOTUNE)
        .shuffle(1000, seed=1024)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .with_options(performance_options)
    )

    return augment_version


augment_version_a, augment_version_b = make_version(), make_version()
augment_versions = tf.data.Dataset.zip(
    (augment_version_a, augment_version_b)
).with_options(performance_options)

"""
We can use `tensorflow_similarity.visualization.visualize_views` to check out a
few sample images. Let's verify that each pair of images have a different set of
transforms.
"""

sample = next(iter(augment_versions))

print("Augmented Views")
tensorflow_similarity.visualization.visualize_views(
    sample, num_imgs=20, views_per_col=4, max_pixel_value=1.0, fig_size=(10, 10)
)

"""
## Model Training
"""

"""
### Architecture
"""

"""
VicReg - Like Barlow Twins, requires a backbone (encoder) and a projector. The
projector is responsible for creating the embeddings that represent the dataset
as a whole.

We will be using a ResNet-18 with an output of length 512 and attach that to a
projector which will return embeddings of length 5000. The projector takes the
output of the backbone, and applies a series of Dense, Batch Normalization, and
Relu transformations.

![Model Structure](https://i.imgur.com/GuVyyJW.png)


"""

"""
### Loss
"""

"""
VicReg differs from other self-supervised methods like Barlow Twins due to its
unique way of calculating the loss function. It checks the variance, covariance,
and similarity between the embeddings per each image. We will be using
`tensorflow_similarity.losses.Vicreg()` to do this for us.

![Vicreg Loss](https://i.imgur.com/9Xa6tYD.png)
When the image says to "minimize the similarity", it means to minimize the
mean squared error.

<details>
<summary> Details about VicReg Loss </summary>

The VicReg loss aims to:
* ***Maximize*** the **variance** between corresponding elements of *different*
embeddings within a
batch. The notion is that different images should have different representations
from other ones. One way to measure this is by taking the variance, which checks
how varied, or scattered a dataset is. In this case, if the variance is high, we
can assume that the embeddings for different images are going to be different.
* ***Minimize*** the internal **covariance** of each embedding in the batch.
Covariance is when the individual values in the embeddings "correlate" with each
other. For example, if in an embedding there are two different variables that
always have the same value with each other, we say they are covariant. This is
bad because we want our embeddings to carry as much information about the
dataset as possible, so that downstream tasks have a lot more to work with. If
two different values in the embedding share correlations with each other, we
wouldn't need two separate values; we can just have one embedding that carries
both of their information together. Having two embeddings that always carry the
same information is *redundant*, and we want to remove this redundancy to get
the maximum information we can from these embeddings.
* ***Minimize*** the **distance** between embeddings that are for the same image. Two
similar images must have similar embeddings. To check this we can just use the
Mean Squared Error to find the distance between them.

Each of these losses are weighted summed with each other to get one loss number
</details>

<details>
<summary> VicReg pseudocode of variance, covariance, similarity </summary>

Variance Pseudocode:
```
z = mean_center(z)
std_z = sqrt(var(a, axis=0) + SMALL_CONSTANT)
std_z = mean(max(std_z, 0))

*same for z' and std_z'*

std_loss = average(std_z, std_z')
```

Similarity Pseudocode:
```
sim_loss = mse(z, z')
```

Covariance Pseudocode:
```
z = mean_center(z)
cov_loss_z = mm(transpose(z), z).get_off_diagonal()
cov_loss_z = sum(cov_loss_z) / embedding_size

*do same for z' and cov_loss_z'*

cov_loss = cov_loss_z + cov_loss_z'
```
</details>
"""

"""
### Implementation
"""

"""
We will be using Tensorflow Similarity's `ResNet18Sim` as our backbone and will implement
our custom projector. The backbone and projector will be combined via `ContrastiveModel`,
an API that manages our model composition and training
loop.
"""


# Code for defining our projector
def projector_layers(input_size, n_dense_neurons) -> keras.Model:
    """projector_layers method.

    Builds the projector network for VicReg, which are a series of Dense,
    BatchNormalization, and ReLU layers stacked on top of each other.

    Returns:
        returns the projector layers for VicReg
    """

    # number of dense neurons in the projector
    input_layer = tf.keras.layers.Input(input_size)

    # intermediate layers of the projector network
    n_layers = 2
    for i in range(n_layers):
        dense = tf.keras.layers.Dense(n_dense_neurons, name=f"projector_dense_{i}")
        x = dense(input_layer) if i == 0 else dense(x)
        x = tf.keras.layers.BatchNormalization(name=f"projector_bn_{i}")(x)
        x = tf.keras.layers.ReLU(name=f"projector_relu_{i}")(x)

    x = tf.keras.layers.Dense(n_dense_neurons, name=f"projector_dense_{n_layers}")(x)

    model = keras.Model(input_layer, x)
    return model


backbone = tensorflow_similarity.architectures.ResNet18Sim(
    (IMAGE_SIZE, IMAGE_SIZE, 3), embedding_size=512
)
projector = projector_layers(backbone.output.shape[-1], n_dense_neurons=5000)

model = tensorflow_similarity.models.ContrastiveModel(
    backbone=backbone,
    projector=projector,
    algorithm="barlow",  # VicReg uses same architecture + training loop as Barlow Twins
)

# LAMB optimizer converges faster than ADAM or SGD when using large batch sizes.
optimizer = tfa.optimizers.LAMB()
loss = tensorflow_similarity.losses.VicReg()
model.compile(optimizer=optimizer, loss=loss)

# Expected training time: 1 hour
history = model.fit(augment_versions, epochs=75)
plt.plot(history.history["loss"])
plt.show()

"""
## Evaluation
"""

"""
### Evaluation Method
"""

"""
We will use *linear evaluation* to see how well our model learned embeddings.
This is where we freeze our trained backbone and projector, and just add a
single Dense + Softmax layer. Then, we train our model using the test images and
labels. Because we took 30% of CIFAR-10 when sampling, we are only training this
model with 3000 labeled images. However, remember that we trained the backbone
and projector using 12000 images, though they were unlabeled.
"""

"""
### Code
"""


def gen_lin_ds(features, labels):
    ds = (
        tf.data.Dataset.from_tensor_slices((features, labels))
        .shuffle(1000)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


xy_ds = gen_lin_ds(train_features, train_labels)
test_ds = gen_lin_ds(test_features, test_labels)

evaluator = keras.models.Sequential(
    [
        model.backbone,
        model.projector,
        keras.layers.Dense(
            10, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.02)
        ),
    ]
)

# Need to test the backbone
evaluator.layers[0].trainable = False
evaluator.layers[1].trainable = False

linear_optimizer = tfa.optimizers.LAMB()
evaluator.compile(
    optimizer=linear_optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

evaluator.fit(xy_ds, epochs=35, validation_data=test_ds)

"""
Our accuracy should be between 60%-63%. This shows that our VicReg model was
able to learn a lot from the dataset, and can get better results than just the
10% one may get with random guessing.

**Things To try**
* If you change `DATASET_PERCENTAGE` to 1, meaning that it would use all the
dataset, accuracy should increase to about 70%
* If the number of epochs is changed from 75 to 150, accuracy may also increase
by a few points as well.
"""

"""
## Conclusion
"""

"""
* VicReg is a method of self-supervised partially-contrastive learning to
generate high-quality embeddings that contain dataset relationships.
* Using VicReg on 30% of our dataset, out of which 80% is unlabeled, we can get
an accuracy of around 62% when freezing all layers except a small Dense layer
at the end.
"""

"""
* VicReg, and other similar algorithms, have several use cases.
  * Can be used in semi-supervised learning, as shown in this demo. This is
  where you have a lot of unlabeled data and very little labeled data. You can
  use the unlabeled data to generate embeddings to assist the labeled data when
  training.
* VicReg vs Barlow Twins (Predecessor)
  * VicReg performs similarly to Barlow Twins on CIFAR-10 and other
  Image classification datasets
  * However it significantly outperforms Barlow Twins on multi-modal tasks like
  Image-to-Text and Text-to-Image
  ![Table](https://i.imgur.com/GuWIssF.png)
"""

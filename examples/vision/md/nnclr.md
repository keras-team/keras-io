# Self-supervised contrastive learning with NNCLR

**Author:** [Rishit Dagli](https://twitter.com/rishit_dagli)<br>
**Date created:** 2021/09/13<br>
**Last modified:** 2024/01/22<br>
**Description:** Implementation of NNCLR, a self-supervised learning method for computer vision.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/nnclr.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/nnclr.py)


## Introduction

### Self-supervised learning

Self-supervised representation learning aims to obtain robust representations of samples
from raw data without expensive labels or annotations. Early methods in this field
focused on defining pretraining tasks which involved a surrogate task on a domain with ample
weak supervision labels. Encoders trained to solve such tasks are expected to
learn general features that might be useful for other downstream tasks requiring
expensive annotations like image classification.

### Contrastive Learning

A broad category of self-supervised learning techniques are those that use *contrastive
losses*, which have been used in a wide range of computer vision applications like
[image similarity](https://www.jmlr.org/papers/v11/chechik10a.html),
[dimensionality reduction (DrLIM)](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
and [face verification/identification](https://openaccess.thecvf.com/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html).
These methods learn a latent space that clusters positive samples together while
pushing apart negative samples.

### NNCLR

In this example, we implement NNCLR as proposed in the paper
[With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations](https://arxiv.org/abs/2104.14548),
by Google Research and DeepMind.

NNCLR learns self-supervised representations that go beyond single-instance positives, which
allows for learning better features that are invariant to different viewpoints, deformations,
and even intra-class variations.
Clustering based methods offer a great approach to go beyond single instance positives,
but assuming the entire cluster to be positives could hurt performance due to early
over-generalization. Instead, NNCLR uses nearest neighbors in the learned representation
space as positives.
In addition, NNCLR increases the performance of existing contrastive learning methods like
[SimCLR](https://arxiv.org/abs/2002.05709)([Keras Example](https://keras.io/examples/vision/semisupervised_simclr))
and reduces the reliance of self-supervised methods on data augmentation strategies.

Here is a great visualization by the paper authors showing how NNCLR builds on ideas from
SimCLR:

![](https://i.imgur.com/p2DbZJJ.png)

We can see that SimCLR uses two views of the same image as the positive pair. These two
views, which are produced using random data augmentations, are fed through an encoder to
obtain the positive embedding pair, we end up using two augmentations. NNCLR instead
keeps a _support set_ of embeddings representing the full data distribution, and forms
the positive pairs using nearest-neighbours. A support set is used as memory during
training, similar to a queue (i.e. first-in-first-out) as in
[MoCo](https://arxiv.org/abs/1911.05722).

This example requires `tensorflow_datasets`, which can
be installed with this command:


```python
!pip install tensorflow-datasets
```

<div class="k-default-codeblock">
```
Requirement already satisfied: tensorflow-datasets in /usr/local/lib/python3.9/site-packages (4.9.3)

Requirement already satisfied: psutil in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (5.9.7)
Requirement already satisfied: toml in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (0.10.2)
Requirement already satisfied: termcolor in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (2.4.0)
Requirement already satisfied: tensorflow-metadata in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (1.14.0)
Requirement already satisfied: protobuf>=3.20 in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (3.20.3)
Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (2.31.0)
Requirement already satisfied: absl-py in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (1.4.0)
Requirement already satisfied: array-record in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (0.5.0)
Requirement already satisfied: etils[enp,epath,etree]>=0.9.0 in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (1.5.2)
Requirement already satisfied: numpy in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (1.26.3)
Requirement already satisfied: click in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (8.1.7)
Requirement already satisfied: promise in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (2.3)
Requirement already satisfied: dm-tree in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (0.1.8)
Requirement already satisfied: tqdm in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (4.66.1)
Requirement already satisfied: wrapt in /usr/local/lib/python3.9/site-packages (from tensorflow-datasets) (1.14.1)

Requirement already satisfied: zipp in /usr/local/lib/python3.9/site-packages (from etils[enp,epath,etree]>=0.9.0->tensorflow-datasets) (3.17.0)
Requirement already satisfied: importlib_resources in /usr/local/lib/python3.9/site-packages (from etils[enp,epath,etree]>=0.9.0->tensorflow-datasets) (6.1.1)
Requirement already satisfied: typing_extensions in /usr/local/lib/python3.9/site-packages (from etils[enp,epath,etree]>=0.9.0->tensorflow-datasets) (4.9.0)
Requirement already satisfied: fsspec in /usr/local/lib/python3.9/site-packages (from etils[enp,epath,etree]>=0.9.0->tensorflow-datasets) (2023.12.2)

Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.9/site-packages (from requests>=2.19.0->tensorflow-datasets) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/site-packages (from requests>=2.19.0->tensorflow-datasets) (3.6)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/site-packages (from requests>=2.19.0->tensorflow-datasets) (2023.11.17)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.9/site-packages (from requests>=2.19.0->tensorflow-datasets) (1.26.18)
Requirement already satisfied: six in /usr/local/lib/python3.9/site-packages (from promise->tensorflow-datasets) (1.16.0)

Requirement already satisfied: googleapis-common-protos<2,>=1.52.0 in /usr/local/lib/python3.9/site-packages (from tensorflow-metadata->tensorflow-datasets) (1.62.0)

[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[33m

 [[34;49mnotice[1;39;49m][39;49m A new release of pip is available: [31;49m23.0.1[39;49m -> [32;49m23.3.2
 [[34;49mnotice[1;39;49m][39;49m To update, run: [32;49mpip install --upgrade pip

```
</div>
---
## Setup

```python
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import keras_cv
from keras import ops
from keras import layers
```

---
## Hyperparameters

A greater `queue_size` most likely means better performance as shown in the original
paper, but introduces significant computational overhead. The authors show that the best
results of NNCLR are achieved with a queue size of 98,304 (the largest `queue_size` they
experimented on). We here use 10,000 to show a working example.


```python
AUTOTUNE = tf.data.AUTOTUNE
shuffle_buffer = 5000
# The below two values are taken from https://www.tensorflow.org/datasets/catalog/stl10
labelled_train_images = 5000
unlabelled_images = 100000

temperature = 0.1
queue_size = 10000
contrastive_augmenter = {
    "brightness": 0.5,
    "name": "contrastive_augmenter",
    "scale": (0.2, 1.0),
}
classification_augmenter = {
    "brightness": 0.2,
    "name": "classification_augmenter",
    "scale": (0.5, 1.0),
}
input_shape = (96, 96, 3)
width = 128
num_epochs = 5  # Use 25 for better results
steps_per_epoch = 50  # Use 200 for better results
```

---
## Load the Dataset

We load the [STL-10](http://ai.stanford.edu/~acoates/stl10/) dataset from
TensorFlow Datasets, an image recognition dataset for developing unsupervised
feature learning, deep learning, self-taught learning algorithms. It is inspired by the
CIFAR-10 dataset, with some modifications.


```python
dataset_name = "stl10"


def prepare_dataset():
    unlabeled_batch_size = unlabelled_images // steps_per_epoch
    labeled_batch_size = labelled_train_images // steps_per_epoch
    batch_size = unlabeled_batch_size + labeled_batch_size

    unlabeled_train_dataset = (
        tfds.load(
            dataset_name, split="unlabelled", as_supervised=True, shuffle_files=True
        )
        .shuffle(buffer_size=shuffle_buffer)
        .batch(unlabeled_batch_size, drop_remainder=True)
    )
    labeled_train_dataset = (
        tfds.load(dataset_name, split="train", as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=shuffle_buffer)
        .batch(labeled_batch_size, drop_remainder=True)
    )
    test_dataset = (
        tfds.load(dataset_name, split="test", as_supervised=True)
        .batch(batch_size)
        .prefetch(buffer_size=AUTOTUNE)
    )
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=AUTOTUNE)

    return batch_size, train_dataset, labeled_train_dataset, test_dataset


batch_size, train_dataset, labeled_train_dataset, test_dataset = prepare_dataset()
```

---
## Augmentations

Other self-supervised techniques like [SimCLR](https://arxiv.org/abs/2002.05709),
[BYOL](https://arxiv.org/abs/2006.07733), [SwAV](https://arxiv.org/abs/2006.09882) etc.
rely heavily on a well-designed data augmentation pipeline to get the best performance.
However, NNCLR is _less_ dependent on complex augmentations as nearest-neighbors already
provide richness in sample variations. A few common techniques often included
augmentation pipelines are:

- Random resized crops
- Multiple color distortions
- Gaussian blur

Since NNCLR is less dependent on complex augmentations, we will only use random
crops and random brightness for augmenting the input images.



### Prepare augmentation module


```python

def augmenter(brightness, name, scale):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            keras_cv.layers.RandomCropAndResize(
                target_size=(input_shape[0], input_shape[1]),
                crop_area_factor=scale,
                aspect_ratio_factor=(3 / 4, 4 / 3),
            ),
            keras_cv.layers.RandomBrightness(factor=brightness, value_range=(0.0, 1.0)),
        ],
        name=name,
    )

```

### Encoder architecture

Using a ResNet-50 as the encoder architecture
is standard in the literature. In the original paper, the authors use ResNet-50 as
the encoder architecture and spatially average the outputs of ResNet-50. However, keep in
mind that more powerful models will not only increase training time but will also
require more memory and will limit the maximal batch size you can use. For the purpose of
this example, we just use four convolutional layers.


```python

def encoder():
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(width, activation="relu"),
        ],
        name="encoder",
    )

```

---
## The NNCLR model for contrastive pre-training

We train an encoder on unlabeled images with a contrastive loss. A nonlinear projection
head is attached to the top of the encoder, as it improves the quality of representations
of the encoder.


```python

class NNCLR(keras.Model):
    def __init__(
        self, temperature, queue_size,
    ):
        super().__init__()
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_augmenter = augmenter(**contrastive_augmenter)
        self.classification_augmenter = augmenter(**classification_augmenter)
        self.encoder = encoder()
        self.projection_head = keras.Sequential(
            [
                layers.Input(shape=(width,)),
                layers.Dense(width, activation="relu"),
                layers.Dense(width),
            ],
            name="projection_head",
        )
        self.linear_probe = keras.Sequential(
            [layers.Input(shape=(width,)), layers.Dense(10)], name="linear_probe"
        )
        self.temperature = temperature

        feature_dimensions = self.encoder.output_shape[1]
        self.feature_queue = keras.Variable(
            keras.utils.normalize(
                keras.random.normal(shape=(queue_size, feature_dimensions)),
                axis=1,
                order=2,
            ),
            trainable=False,
        )

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

    def nearest_neighbour(self, projections):
        support_similarities = ops.matmul(projections, ops.transpose(self.feature_queue))
        nn_projections = ops.take(
            self.feature_queue, ops.argmax(support_similarities, axis=1), axis=0
        )
        return projections + ops.stop_gradient(nn_projections - projections)

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = keras.utils.normalize(features_1, axis=1, order=2)
        features_2 = keras.utils.normalize(features_2, axis=1, order=2)
        similarities = ops.matmul(features_1, ops.transpose(features_2))
        batch_size = ops.shape(features_1)[0]
        contrastive_labels = ops.arange(batch_size)
        self.contrastive_accuracy.update_state(
            ops.concatenate([contrastive_labels, contrastive_labels], axis=0),
            ops.concatenate([similarities, ops.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        features_1 = (features_1 - ops.mean(features_1, axis=0)) / ops.std(
            features_1, axis=0
        )
        features_2 = (features_2 - ops.mean(features_2, axis=0)) / ops.std(
            features_2, axis=0
        )

        batch_size = ops.shape(features_1)[0]
        cross_correlation = (
            ops.matmul(ops.transpose(features_1), features_2) / batch_size
        )

        feature_dim = ops.shape(features_1)[1]
        correlation_labels = ops.arange(feature_dim)
        self.correlation_accuracy.update_state(
            ops.concatenate([correlation_labels, correlation_labels], axis=0),
            ops.concatenate(
                [cross_correlation, ops.transpose(cross_correlation)], axis=0
            ),
        )

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = keras.utils.normalize(projections_1, axis=1, order=2)
        projections_2 = keras.utils.normalize(projections_2, axis=1, order=2)

        similarities_1_2_1 = (
            ops.matmul(
                self.nearest_neighbour(projections_1), ops.transpose(projections_2)
            )
            / self.temperature
        )
        similarities_1_2_2 = (
             ops.matmul(
                projections_2, ops.transpose(self.nearest_neighbour(projections_1))
            )
            / self.temperature
        )

        similarities_2_1_1 = (
            ops.matmul(
                self.nearest_neighbour(projections_2), ops.transpose(projections_1)
            )
            / self.temperature
        )
        similarities_2_1_2 = (
            ops.matmul(
                projections_1, ops.transpose(self.nearest_neighbour(projections_2))
            )
            / self.temperature
        )

        batch_size = ops.shape(projections_1)[0]
        contrastive_labels = ops.arange(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            ops.concatenate(
                [
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                ],
                axis=0,
            ),
            ops.concatenate(
                [
                    similarities_1_2_1,
                    similarities_1_2_2,
                    similarities_2_1_1,
                    similarities_2_1_2,
                ],
                axis=0,
            ),
            from_logits=True,
        )

        self.feature_queue.assign(
            ops.concatenate([projections_1, self.feature_queue[:-batch_size]], axis=0)
        )
        return loss

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data
        images = ops.concatenate((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)
        preprocessed_images = self.classification_augmenter(labeled_images)

        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_accuracy.update_state(labels, class_logits)

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result(),
        }

    def test_step(self, data):
        labeled_images, labels = data

        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)

        self.probe_accuracy.update_state(labels, class_logits)
        return {"p_loss": probe_loss, "p_acc": self.probe_accuracy.result()}

```

---
## Pre-train NNCLR

We train the network using a `temperature` of 0.1 as suggested in the paper and
a `queue_size` of 10,000 as explained earlier. We use Adam as our contrastive and probe
optimizer. For this example we train the model for only 30 epochs but it should be
trained for more epochs for better performance.

The following two metrics can be used for monitoring the pretraining performance
which we also log (taken from
[this Keras example](https://keras.io/examples/vision/semisupervised_simclr/#selfsupervised-model-for-contrastive-pretraining)):

- Contrastive accuracy: self-supervised metric, the ratio of cases in which the
representation of an image is more similar to its differently augmented version's one,
than to the representation of any other image in the current batch. Self-supervised
metrics can be used for hyperparameter tuning even in the case when there are no labeled
examples.
- Linear probing accuracy: linear probing is a popular metric to evaluate self-supervised
classifiers. It is computed as the accuracy of a logistic regression classifier trained
on top of the encoder's features. In our case, this is done by training a single dense
layer on top of the frozen encoder. Note that contrary to traditional approach where the
classifier is trained after the pretraining phase, in this example we train it during
pretraining. This might slightly decrease its accuracy, but that way we can monitor its
value during training, which helps with experimentation and debugging.


```python
model = NNCLR(temperature=temperature, queue_size=queue_size)
model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
    jit_compile=False,
)
pretrain_history = model.fit(
    train_dataset, epochs=num_epochs, validation_data=test_dataset
)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

Epoch 1/5

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

```
</div>
    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  6:36 8s/step - c_acc: 0.0615 - c_loss: 6.9203 - p_acc: 0.1040 - p_loss: 2.3040 - r_acc: 0.3008

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  1:46 2s/step - c_acc: 0.0635 - c_loss: 6.8810 - p_acc: 0.1053 - p_loss: 2.3044 - r_acc: 0.3037

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  1:39 2s/step - c_acc: 0.0651 - c_loss: 6.8340 - p_acc: 0.1061 - p_loss: 2.3051 - r_acc: 0.3040

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 2s/step - c_acc: 0.0670 - c_loss: 6.7324 - p_acc: 0.1071 - p_loss: 2.3042 - r_acc: 0.3054

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 2s/step - c_acc: 0.0679 - c_loss: 6.7339 - p_acc: 0.1102 - p_loss: 2.3025 - r_acc: 0.3059

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 2s/step - c_acc: 0.0686 - c_loss: 6.6875 - p_acc: 0.1122 - p_loss: 2.3022 - r_acc: 0.3046

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:24 2s/step - c_acc: 0.0691 - c_loss: 6.6557 - p_acc: 0.1138 - p_loss: 2.3017 - r_acc: 0.3037

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:20 2s/step - c_acc: 0.0694 - c_loss: 6.6060 - p_acc: 0.1151 - p_loss: 2.3013 - r_acc: 0.3032

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:17 2s/step - c_acc: 0.0697 - c_loss: 6.5640 - p_acc: 0.1162 - p_loss: 2.3009 - r_acc: 0.3027

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:15 2s/step - c_acc: 0.0700 - c_loss: 6.5590 - p_acc: 0.1171 - p_loss: 2.3002 - r_acc: 0.3023

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:13 2s/step - c_acc: 0.0701 - c_loss: 6.5354 - p_acc: 0.1178 - p_loss: 2.2993 - r_acc: 0.3020

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:11 2s/step - c_acc: 0.0704 - c_loss: 6.4983 - p_acc: 0.1187 - p_loss: 2.2987 - r_acc: 0.3015

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:09 2s/step - c_acc: 0.0706 - c_loss: 6.4816 - p_acc: 0.1196 - p_loss: 2.2979 - r_acc: 0.3008

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:07 2s/step - c_acc: 0.0708 - c_loss: 6.4568 - p_acc: 0.1205 - p_loss: 2.2970 - r_acc: 0.3002

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:05 2s/step - c_acc: 0.0711 - c_loss: 6.4197 - p_acc: 0.1212 - p_loss: 2.2961 - r_acc: 0.3000

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:03 2s/step - c_acc: 0.0713 - c_loss: 6.3997 - p_acc: 0.1218 - p_loss: 2.2955 - r_acc: 0.2999

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:00 2s/step - c_acc: 0.0716 - c_loss: 6.3685 - p_acc: 0.1223 - p_loss: 2.2950 - r_acc: 0.3001

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  58s 2s/step - c_acc: 0.0718 - c_loss: 6.3385 - p_acc: 0.1227 - p_loss: 2.2942 - r_acc: 0.3004 

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  57s 2s/step - c_acc: 0.0721 - c_loss: 6.3154 - p_acc: 0.1231 - p_loss: 2.2937 - r_acc: 0.3008

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  54s 2s/step - c_acc: 0.0723 - c_loss: 6.2895 - p_acc: 0.1234 - p_loss: 2.2930 - r_acc: 0.3014

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  52s 2s/step - c_acc: 0.0726 - c_loss: 6.2618 - p_acc: 0.1237 - p_loss: 2.2921 - r_acc: 0.3023

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  51s 2s/step - c_acc: 0.0728 - c_loss: 6.2451 - p_acc: 0.1239 - p_loss: 2.2915 - r_acc: 0.3032

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  49s 2s/step - c_acc: 0.0731 - c_loss: 6.2257 - p_acc: 0.1241 - p_loss: 2.2909 - r_acc: 0.3041

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  46s 2s/step - c_acc: 0.0734 - c_loss: 6.2034 - p_acc: 0.1243 - p_loss: 2.2902 - r_acc: 0.3051

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  45s 2s/step - c_acc: 0.0737 - c_loss: 6.1801 - p_acc: 0.1244 - p_loss: 2.2896 - r_acc: 0.3062

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  43s 2s/step - c_acc: 0.0740 - c_loss: 6.1594 - p_acc: 0.1247 - p_loss: 2.2888 - r_acc: 0.3075

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  41s 2s/step - c_acc: 0.0743 - c_loss: 6.1402 - p_acc: 0.1249 - p_loss: 2.2879 - r_acc: 0.3089

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  39s 2s/step - c_acc: 0.0747 - c_loss: 6.1239 - p_acc: 0.1252 - p_loss: 2.2871 - r_acc: 0.3102

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  37s 2s/step - c_acc: 0.0750 - c_loss: 6.1028 - p_acc: 0.1254 - p_loss: 2.2866 - r_acc: 0.3117

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  35s 2s/step - c_acc: 0.0753 - c_loss: 6.0863 - p_acc: 0.1257 - p_loss: 2.2861 - r_acc: 0.3133

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  34s 2s/step - c_acc: 0.0757 - c_loss: 6.0687 - p_acc: 0.1259 - p_loss: 2.2853 - r_acc: 0.3150

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  32s 2s/step - c_acc: 0.0761 - c_loss: 6.0495 - p_acc: 0.1262 - p_loss: 2.2846 - r_acc: 0.3167

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  30s 2s/step - c_acc: 0.0764 - c_loss: 6.0289 - p_acc: 0.1265 - p_loss: 2.2838 - r_acc: 0.3186

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  28s 2s/step - c_acc: 0.0768 - c_loss: 6.0138 - p_acc: 0.1268 - p_loss: 2.2833 - r_acc: 0.3204

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  26s 2s/step - c_acc: 0.0772 - c_loss: 5.9972 - p_acc: 0.1271 - p_loss: 2.2828 - r_acc: 0.3223

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  24s 2s/step - c_acc: 0.0776 - c_loss: 5.9807 - p_acc: 0.1274 - p_loss: 2.2821 - r_acc: 0.3241

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  22s 2s/step - c_acc: 0.0780 - c_loss: 5.9624 - p_acc: 0.1278 - p_loss: 2.2816 - r_acc: 0.3260

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  21s 2s/step - c_acc: 0.0784 - c_loss: 5.9455 - p_acc: 0.1281 - p_loss: 2.2810 - r_acc: 0.3279

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  19s 2s/step - c_acc: 0.0788 - c_loss: 5.9322 - p_acc: 0.1284 - p_loss: 2.2801 - r_acc: 0.3298

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  17s 2s/step - c_acc: 0.0792 - c_loss: 5.9179 - p_acc: 0.1287 - p_loss: 2.2794 - r_acc: 0.3317

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  15s 2s/step - c_acc: 0.0796 - c_loss: 5.9057 - p_acc: 0.1291 - p_loss: 2.2789 - r_acc: 0.3336

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  13s 2s/step - c_acc: 0.0800 - c_loss: 5.8923 - p_acc: 0.1294 - p_loss: 2.2785 - r_acc: 0.3354

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  12s 2s/step - c_acc: 0.0804 - c_loss: 5.8796 - p_acc: 0.1298 - p_loss: 2.2779 - r_acc: 0.3373

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  10s 2s/step - c_acc: 0.0808 - c_loss: 5.8674 - p_acc: 0.1301 - p_loss: 2.2772 - r_acc: 0.3392

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  8s 2s/step - c_acc: 0.0813 - c_loss: 5.8531 - p_acc: 0.1305 - p_loss: 2.2767 - r_acc: 0.3411 

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  6s 2s/step - c_acc: 0.0817 - c_loss: 5.8379 - p_acc: 0.1308 - p_loss: 2.2761 - r_acc: 0.3430

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  5s 2s/step - c_acc: 0.0821 - c_loss: 5.8260 - p_acc: 0.1312 - p_loss: 2.2756 - r_acc: 0.3449

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  3s 2s/step - c_acc: 0.0826 - c_loss: 5.8137 - p_acc: 0.1315 - p_loss: 2.2751 - r_acc: 0.3468

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  1s 2s/step - c_acc: 0.0830 - c_loss: 5.7990 - p_acc: 0.1319 - p_loss: 2.2746 - r_acc: 0.3487

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - c_acc: 0.0835 - c_loss: 5.7860 - p_acc: 0.1322 - p_loss: 2.2743 - r_acc: 0.3506

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.


```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 97s 2s/step - c_acc: 0.0835 - c_loss: 5.7860 - p_acc: 0.1322 - p_loss: 2.2743 - r_acc: 0.3506 - loss: 0.0000e+00 - sparse_categorical_accuracy: 0.1055 - val_loss: 0.0000e+00 - val_sparse_categorical_accuracy: 0.0000e+00


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  1:35 2s/step - c_acc: 0.1880 - c_loss: 5.0992 - p_acc: 0.1690 - p_loss: 2.2565 - r_acc: 0.6328

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  1:37 2s/step - c_acc: 0.1829 - c_loss: 5.1009 - p_acc: 0.1688 - p_loss: 2.2539 - r_acc: 0.6377

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  2:12 3s/step - c_acc: 0.1824 - c_loss: 5.1064 - p_acc: 0.1711 - p_loss: 2.2530 - r_acc: 0.6387

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  1:57 3s/step - c_acc: 0.1824 - c_loss: 5.0725 - p_acc: 0.1725 - p_loss: 2.2523 - r_acc: 0.6394

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:54 3s/step - c_acc: 0.1820 - c_loss: 5.0802 - p_acc: 0.1724 - p_loss: 2.2527 - r_acc: 0.6406

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:49 2s/step - c_acc: 0.1815 - c_loss: 5.0946 - p_acc: 0.1723 - p_loss: 2.2540 - r_acc: 0.6415

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:43 2s/step - c_acc: 0.1808 - c_loss: 5.1212 - p_acc: 0.1720 - p_loss: 2.2538 - r_acc: 0.6421

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:49 3s/step - c_acc: 0.1799 - c_loss: 5.1180 - p_acc: 0.1720 - p_loss: 2.2544 - r_acc: 0.6423

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:44 3s/step - c_acc: 0.1793 - c_loss: 5.1138 - p_acc: 0.1721 - p_loss: 2.2546 - r_acc: 0.6430

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:38 2s/step - c_acc: 0.1790 - c_loss: 5.1204 - p_acc: 0.1726 - p_loss: 2.2535 - r_acc: 0.6441

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:32 2s/step - c_acc: 0.1788 - c_loss: 5.1042 - p_acc: 0.1730 - p_loss: 2.2534 - r_acc: 0.6452

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:27 2s/step - c_acc: 0.1789 - c_loss: 5.0974 - p_acc: 0.1733 - p_loss: 2.2533 - r_acc: 0.6461

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:24 2s/step - c_acc: 0.1791 - c_loss: 5.0975 - p_acc: 0.1737 - p_loss: 2.2534 - r_acc: 0.6471

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:23 2s/step - c_acc: 0.1793 - c_loss: 5.0924 - p_acc: 0.1741 - p_loss: 2.2530 - r_acc: 0.6482

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:18 2s/step - c_acc: 0.1796 - c_loss: 5.0866 - p_acc: 0.1745 - p_loss: 2.2530 - r_acc: 0.6492

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:14 2s/step - c_acc: 0.1799 - c_loss: 5.0765 - p_acc: 0.1750 - p_loss: 2.2530 - r_acc: 0.6502

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 2s/step - c_acc: 0.1803 - c_loss: 5.0665 - p_acc: 0.1753 - p_loss: 2.2530 - r_acc: 0.6512

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 2s/step - c_acc: 0.1808 - c_loss: 5.0577 - p_acc: 0.1756 - p_loss: 2.2527 - r_acc: 0.6522

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 2s/step - c_acc: 0.1812 - c_loss: 5.0530 - p_acc: 0.1759 - p_loss: 2.2529 - r_acc: 0.6531

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 2s/step - c_acc: 0.1817 - c_loss: 5.0470 - p_acc: 0.1761 - p_loss: 2.2530 - r_acc: 0.6541

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  59s 2s/step - c_acc: 0.1822 - c_loss: 5.0378 - p_acc: 0.1763 - p_loss: 2.2524 - r_acc: 0.6551 

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  56s 2s/step - c_acc: 0.1827 - c_loss: 5.0392 - p_acc: 0.1766 - p_loss: 2.2521 - r_acc: 0.6561

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  53s 2s/step - c_acc: 0.1832 - c_loss: 5.0500 - p_acc: 0.1768 - p_loss: 2.2517 - r_acc: 0.6570

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  51s 2s/step - c_acc: 0.1837 - c_loss: 5.0503 - p_acc: 0.1769 - p_loss: 2.2515 - r_acc: 0.6578

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  48s 2s/step - c_acc: 0.1842 - c_loss: 5.0460 - p_acc: 0.1771 - p_loss: 2.2512 - r_acc: 0.6586

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  46s 2s/step - c_acc: 0.1847 - c_loss: 5.0482 - p_acc: 0.1772 - p_loss: 2.2509 - r_acc: 0.6595

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  43s 2s/step - c_acc: 0.1853 - c_loss: 5.0450 - p_acc: 0.1774 - p_loss: 2.2506 - r_acc: 0.6603

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  41s 2s/step - c_acc: 0.1858 - c_loss: 5.0413 - p_acc: 0.1776 - p_loss: 2.2503 - r_acc: 0.6611

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  39s 2s/step - c_acc: 0.1864 - c_loss: 5.0393 - p_acc: 0.1778 - p_loss: 2.2500 - r_acc: 0.6620

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  37s 2s/step - c_acc: 0.1870 - c_loss: 5.0370 - p_acc: 0.1780 - p_loss: 2.2495 - r_acc: 0.6628

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  35s 2s/step - c_acc: 0.1875 - c_loss: 5.0327 - p_acc: 0.1782 - p_loss: 2.2493 - r_acc: 0.6637

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  33s 2s/step - c_acc: 0.1881 - c_loss: 5.0326 - p_acc: 0.1784 - p_loss: 2.2489 - r_acc: 0.6646

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  31s 2s/step - c_acc: 0.1887 - c_loss: 5.0306 - p_acc: 0.1786 - p_loss: 2.2486 - r_acc: 0.6655

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  29s 2s/step - c_acc: 0.1893 - c_loss: 5.0318 - p_acc: 0.1789 - p_loss: 2.2482 - r_acc: 0.6663

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  27s 2s/step - c_acc: 0.1899 - c_loss: 5.0222 - p_acc: 0.1791 - p_loss: 2.2480 - r_acc: 0.6672

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  25s 2s/step - c_acc: 0.1906 - c_loss: 5.0211 - p_acc: 0.1793 - p_loss: 2.2476 - r_acc: 0.6680

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  23s 2s/step - c_acc: 0.1912 - c_loss: 5.0147 - p_acc: 0.1795 - p_loss: 2.2471 - r_acc: 0.6689

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  21s 2s/step - c_acc: 0.1918 - c_loss: 5.0092 - p_acc: 0.1798 - p_loss: 2.2465 - r_acc: 0.6697

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  19s 2s/step - c_acc: 0.1924 - c_loss: 5.0060 - p_acc: 0.1800 - p_loss: 2.2462 - r_acc: 0.6704

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  17s 2s/step - c_acc: 0.1930 - c_loss: 5.0031 - p_acc: 0.1802 - p_loss: 2.2459 - r_acc: 0.6712

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  15s 2s/step - c_acc: 0.1936 - c_loss: 4.9976 - p_acc: 0.1804 - p_loss: 2.2453 - r_acc: 0.6719

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  14s 2s/step - c_acc: 0.1943 - c_loss: 4.9936 - p_acc: 0.1806 - p_loss: 2.2447 - r_acc: 0.6727

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  12s 2s/step - c_acc: 0.1949 - c_loss: 4.9899 - p_acc: 0.1809 - p_loss: 2.2441 - r_acc: 0.6734

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  10s 2s/step - c_acc: 0.1955 - c_loss: 4.9833 - p_acc: 0.1811 - p_loss: 2.2436 - r_acc: 0.6741

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  8s 2s/step - c_acc: 0.1961 - c_loss: 4.9745 - p_acc: 0.1813 - p_loss: 2.2431 - r_acc: 0.6749 

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  7s 2s/step - c_acc: 0.1967 - c_loss: 4.9714 - p_acc: 0.1816 - p_loss: 2.2425 - r_acc: 0.6756

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  5s 2s/step - c_acc: 0.1973 - c_loss: 4.9682 - p_acc: 0.1819 - p_loss: 2.2421 - r_acc: 0.6763

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  3s 2s/step - c_acc: 0.1979 - c_loss: 4.9583 - p_acc: 0.1821 - p_loss: 2.2414 - r_acc: 0.6770

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  1s 2s/step - c_acc: 0.1985 - c_loss: 4.9542 - p_acc: 0.1824 - p_loss: 2.2406 - r_acc: 0.6777

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - c_acc: 0.1991 - c_loss: 4.9499 - p_acc: 0.1827 - p_loss: 2.2400 - r_acc: 0.6784

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 96s 2s/step - c_acc: 0.1991 - c_loss: 4.9499 - p_acc: 0.1827 - p_loss: 2.2400 - r_acc: 0.6784 - loss: 0.0000e+00 - sparse_categorical_accuracy: 0.2290 - val_loss: 0.0000e+00 - val_sparse_categorical_accuracy: 0.0000e+00


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  1:43 2s/step - c_acc: 0.2895 - c_loss: 4.6047 - p_acc: 0.2160 - p_loss: 2.2082 - r_acc: 0.7891

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  1:21 2s/step - c_acc: 0.2833 - c_loss: 4.6604 - p_acc: 0.2153 - p_loss: 2.2092 - r_acc: 0.7920

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  1:17 2s/step - c_acc: 0.2809 - c_loss: 4.6976 - p_acc: 0.2188 - p_loss: 2.2126 - r_acc: 0.7888

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  1:15 2s/step - c_acc: 0.2809 - c_loss: 4.6830 - p_acc: 0.2227 - p_loss: 2.2122 - r_acc: 0.7887

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:11 2s/step - c_acc: 0.2823 - c_loss: 4.6435 - p_acc: 0.2246 - p_loss: 2.2110 - r_acc: 0.7876

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:12 2s/step - c_acc: 0.2838 - c_loss: 4.6242 - p_acc: 0.2253 - p_loss: 2.2115 - r_acc: 0.7867

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:11 2s/step - c_acc: 0.2851 - c_loss: 4.6377 - p_acc: 0.2257 - p_loss: 2.2090 - r_acc: 0.7867

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:10 2s/step - c_acc: 0.2860 - c_loss: 4.6625 - p_acc: 0.2261 - p_loss: 2.2083 - r_acc: 0.7864

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:08 2s/step - c_acc: 0.2863 - c_loss: 4.6717 - p_acc: 0.2264 - p_loss: 2.2074 - r_acc: 0.7859

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:07 2s/step - c_acc: 0.2865 - c_loss: 4.6858 - p_acc: 0.2265 - p_loss: 2.2061 - r_acc: 0.7856

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:06 2s/step - c_acc: 0.2868 - c_loss: 4.6781 - p_acc: 0.2263 - p_loss: 2.2054 - r_acc: 0.7856

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:07 2s/step - c_acc: 0.2868 - c_loss: 4.6859 - p_acc: 0.2263 - p_loss: 2.2056 - r_acc: 0.7856

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:05 2s/step - c_acc: 0.2870 - c_loss: 4.6768 - p_acc: 0.2263 - p_loss: 2.2062 - r_acc: 0.7857

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:03 2s/step - c_acc: 0.2873 - c_loss: 4.6614 - p_acc: 0.2264 - p_loss: 2.2063 - r_acc: 0.7858

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:03 2s/step - c_acc: 0.2875 - c_loss: 4.6535 - p_acc: 0.2264 - p_loss: 2.2062 - r_acc: 0.7859

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:03 2s/step - c_acc: 0.2879 - c_loss: 4.6447 - p_acc: 0.2264 - p_loss: 2.2061 - r_acc: 0.7859

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:02 2s/step - c_acc: 0.2882 - c_loss: 4.6501 - p_acc: 0.2264 - p_loss: 2.2055 - r_acc: 0.7859

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  1:00 2s/step - c_acc: 0.2885 - c_loss: 4.6462 - p_acc: 0.2264 - p_loss: 2.2056 - r_acc: 0.7860

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  58s 2s/step - c_acc: 0.2888 - c_loss: 4.6299 - p_acc: 0.2264 - p_loss: 2.2050 - r_acc: 0.7862 

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  56s 2s/step - c_acc: 0.2890 - c_loss: 4.6243 - p_acc: 0.2266 - p_loss: 2.2044 - r_acc: 0.7863

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  54s 2s/step - c_acc: 0.2894 - c_loss: 4.6110 - p_acc: 0.2268 - p_loss: 2.2039 - r_acc: 0.7864

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  52s 2s/step - c_acc: 0.2898 - c_loss: 4.6008 - p_acc: 0.2270 - p_loss: 2.2029 - r_acc: 0.7865

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  51s 2s/step - c_acc: 0.2902 - c_loss: 4.5960 - p_acc: 0.2272 - p_loss: 2.2025 - r_acc: 0.7867

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  50s 2s/step - c_acc: 0.2905 - c_loss: 4.5901 - p_acc: 0.2274 - p_loss: 2.2021 - r_acc: 0.7869

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  47s 2s/step - c_acc: 0.2909 - c_loss: 4.5860 - p_acc: 0.2276 - p_loss: 2.2017 - r_acc: 0.7871

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  45s 2s/step - c_acc: 0.2913 - c_loss: 4.5785 - p_acc: 0.2278 - p_loss: 2.2013 - r_acc: 0.7872

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  43s 2s/step - c_acc: 0.2916 - c_loss: 4.5676 - p_acc: 0.2280 - p_loss: 2.2009 - r_acc: 0.7873

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  41s 2s/step - c_acc: 0.2920 - c_loss: 4.5579 - p_acc: 0.2281 - p_loss: 2.2007 - r_acc: 0.7875

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  39s 2s/step - c_acc: 0.2924 - c_loss: 4.5522 - p_acc: 0.2283 - p_loss: 2.1998 - r_acc: 0.7877

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  37s 2s/step - c_acc: 0.2929 - c_loss: 4.5424 - p_acc: 0.2285 - p_loss: 2.1993 - r_acc: 0.7879

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  35s 2s/step - c_acc: 0.2933 - c_loss: 4.5353 - p_acc: 0.2287 - p_loss: 2.1989 - r_acc: 0.7881

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  34s 2s/step - c_acc: 0.2938 - c_loss: 4.5291 - p_acc: 0.2290 - p_loss: 2.1984 - r_acc: 0.7883

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  32s 2s/step - c_acc: 0.2942 - c_loss: 4.5240 - p_acc: 0.2292 - p_loss: 2.1981 - r_acc: 0.7885

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  30s 2s/step - c_acc: 0.2946 - c_loss: 4.5262 - p_acc: 0.2294 - p_loss: 2.1978 - r_acc: 0.7888

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  28s 2s/step - c_acc: 0.2950 - c_loss: 4.5255 - p_acc: 0.2295 - p_loss: 2.1974 - r_acc: 0.7890

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  26s 2s/step - c_acc: 0.2954 - c_loss: 4.5213 - p_acc: 0.2297 - p_loss: 2.1970 - r_acc: 0.7892

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  24s 2s/step - c_acc: 0.2958 - c_loss: 4.5151 - p_acc: 0.2299 - p_loss: 2.1965 - r_acc: 0.7895

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  22s 2s/step - c_acc: 0.2962 - c_loss: 4.5103 - p_acc: 0.2300 - p_loss: 2.1954 - r_acc: 0.7897

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  20s 2s/step - c_acc: 0.2966 - c_loss: 4.5095 - p_acc: 0.2301 - p_loss: 2.1948 - r_acc: 0.7899

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  18s 2s/step - c_acc: 0.2970 - c_loss: 4.5046 - p_acc: 0.2303 - p_loss: 2.1943 - r_acc: 0.7902

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  16s 2s/step - c_acc: 0.2974 - c_loss: 4.4988 - p_acc: 0.2304 - p_loss: 2.1936 - r_acc: 0.7904

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  14s 2s/step - c_acc: 0.2978 - c_loss: 4.4946 - p_acc: 0.2305 - p_loss: 2.1928 - r_acc: 0.7907

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  12s 2s/step - c_acc: 0.2981 - c_loss: 4.4921 - p_acc: 0.2307 - p_loss: 2.1922 - r_acc: 0.7909

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  10s 2s/step - c_acc: 0.2985 - c_loss: 4.4867 - p_acc: 0.2308 - p_loss: 2.1919 - r_acc: 0.7911

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  9s 2s/step - c_acc: 0.2988 - c_loss: 4.4795 - p_acc: 0.2310 - p_loss: 2.1911 - r_acc: 0.7914 

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  7s 2s/step - c_acc: 0.2992 - c_loss: 4.4733 - p_acc: 0.2312 - p_loss: 2.1904 - r_acc: 0.7916

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  5s 2s/step - c_acc: 0.2996 - c_loss: 4.4704 - p_acc: 0.2313 - p_loss: 2.1898 - r_acc: 0.7918

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  3s 2s/step - c_acc: 0.2999 - c_loss: 4.4682 - p_acc: 0.2315 - p_loss: 2.1894 - r_acc: 0.7920

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  1s 2s/step - c_acc: 0.3002 - c_loss: 4.4646 - p_acc: 0.2317 - p_loss: 2.1891 - r_acc: 0.7923

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - c_acc: 0.3006 - c_loss: 4.4600 - p_acc: 0.2318 - p_loss: 2.1888 - r_acc: 0.7925

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 94s 2s/step - c_acc: 0.3006 - c_loss: 4.4600 - p_acc: 0.2318 - p_loss: 2.1888 - r_acc: 0.7925 - loss: 0.0000e+00 - sparse_categorical_accuracy: 0.3179 - val_loss: 0.0000e+00 - val_sparse_categorical_accuracy: 0.0000e+00


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  1:31 2s/step - c_acc: 0.3555 - c_loss: 4.1858 - p_acc: 0.2570 - p_loss: 2.1626 - r_acc: 0.8320

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  1:28 2s/step - c_acc: 0.3543 - c_loss: 4.1949 - p_acc: 0.2542 - p_loss: 2.1635 - r_acc: 0.8340

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  1:28 2s/step - c_acc: 0.3541 - c_loss: 4.1907 - p_acc: 0.2558 - p_loss: 2.1608 - r_acc: 0.8342

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  1:28 2s/step - c_acc: 0.3538 - c_loss: 4.1820 - p_acc: 0.2580 - p_loss: 2.1585 - r_acc: 0.8339

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:24 2s/step - c_acc: 0.3531 - c_loss: 4.2297 - p_acc: 0.2585 - p_loss: 2.1611 - r_acc: 0.8323

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:19 2s/step - c_acc: 0.3531 - c_loss: 4.2128 - p_acc: 0.2593 - p_loss: 2.1596 - r_acc: 0.8308

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:14 2s/step - c_acc: 0.3536 - c_loss: 4.2101 - p_acc: 0.2598 - p_loss: 2.1604 - r_acc: 0.8292

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:11 2s/step - c_acc: 0.3538 - c_loss: 4.2265 - p_acc: 0.2600 - p_loss: 2.1592 - r_acc: 0.8279

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:09 2s/step - c_acc: 0.3540 - c_loss: 4.2322 - p_acc: 0.2600 - p_loss: 2.1590 - r_acc: 0.8267

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:07 2s/step - c_acc: 0.3544 - c_loss: 4.2178 - p_acc: 0.2602 - p_loss: 2.1583 - r_acc: 0.8258

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:06 2s/step - c_acc: 0.3548 - c_loss: 4.2111 - p_acc: 0.2604 - p_loss: 2.1585 - r_acc: 0.8249

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:03 2s/step - c_acc: 0.3551 - c_loss: 4.2090 - p_acc: 0.2606 - p_loss: 2.1576 - r_acc: 0.8243

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:01 2s/step - c_acc: 0.3552 - c_loss: 4.2035 - p_acc: 0.2608 - p_loss: 2.1560 - r_acc: 0.8238

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  59s 2s/step - c_acc: 0.3554 - c_loss: 4.1998 - p_acc: 0.2611 - p_loss: 2.1555 - r_acc: 0.8234 

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  58s 2s/step - c_acc: 0.3554 - c_loss: 4.2024 - p_acc: 0.2614 - p_loss: 2.1558 - r_acc: 0.8230

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  56s 2s/step - c_acc: 0.3555 - c_loss: 4.1990 - p_acc: 0.2615 - p_loss: 2.1560 - r_acc: 0.8225

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  55s 2s/step - c_acc: 0.3556 - c_loss: 4.1953 - p_acc: 0.2616 - p_loss: 2.1557 - r_acc: 0.8222

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  53s 2s/step - c_acc: 0.3557 - c_loss: 4.1945 - p_acc: 0.2617 - p_loss: 2.1559 - r_acc: 0.8219

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  53s 2s/step - c_acc: 0.3558 - c_loss: 4.1930 - p_acc: 0.2617 - p_loss: 2.1551 - r_acc: 0.8217

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  53s 2s/step - c_acc: 0.3557 - c_loss: 4.2022 - p_acc: 0.2618 - p_loss: 2.1545 - r_acc: 0.8215

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  51s 2s/step - c_acc: 0.3557 - c_loss: 4.1978 - p_acc: 0.2619 - p_loss: 2.1539 - r_acc: 0.8213

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  50s 2s/step - c_acc: 0.3556 - c_loss: 4.2053 - p_acc: 0.2619 - p_loss: 2.1524 - r_acc: 0.8211

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  48s 2s/step - c_acc: 0.3556 - c_loss: 4.2076 - p_acc: 0.2620 - p_loss: 2.1521 - r_acc: 0.8209

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  46s 2s/step - c_acc: 0.3555 - c_loss: 4.2149 - p_acc: 0.2620 - p_loss: 2.1520 - r_acc: 0.8207

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  45s 2s/step - c_acc: 0.3555 - c_loss: 4.2091 - p_acc: 0.2621 - p_loss: 2.1517 - r_acc: 0.8206

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  43s 2s/step - c_acc: 0.3555 - c_loss: 4.2084 - p_acc: 0.2621 - p_loss: 2.1512 - r_acc: 0.8204

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  41s 2s/step - c_acc: 0.3555 - c_loss: 4.2044 - p_acc: 0.2621 - p_loss: 2.1508 - r_acc: 0.8203

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  39s 2s/step - c_acc: 0.3555 - c_loss: 4.2062 - p_acc: 0.2621 - p_loss: 2.1502 - r_acc: 0.8201

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  37s 2s/step - c_acc: 0.3554 - c_loss: 4.2078 - p_acc: 0.2622 - p_loss: 2.1496 - r_acc: 0.8200

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  35s 2s/step - c_acc: 0.3554 - c_loss: 4.2069 - p_acc: 0.2622 - p_loss: 2.1490 - r_acc: 0.8199

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  34s 2s/step - c_acc: 0.3554 - c_loss: 4.1989 - p_acc: 0.2622 - p_loss: 2.1486 - r_acc: 0.8198

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  32s 2s/step - c_acc: 0.3555 - c_loss: 4.1940 - p_acc: 0.2623 - p_loss: 2.1485 - r_acc: 0.8197

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  30s 2s/step - c_acc: 0.3555 - c_loss: 4.1914 - p_acc: 0.2623 - p_loss: 2.1478 - r_acc: 0.8196

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  29s 2s/step - c_acc: 0.3556 - c_loss: 4.1893 - p_acc: 0.2623 - p_loss: 2.1477 - r_acc: 0.8195

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  27s 2s/step - c_acc: 0.3557 - c_loss: 4.1917 - p_acc: 0.2623 - p_loss: 2.1473 - r_acc: 0.8194

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  25s 2s/step - c_acc: 0.3558 - c_loss: 4.1904 - p_acc: 0.2624 - p_loss: 2.1469 - r_acc: 0.8193

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  23s 2s/step - c_acc: 0.3559 - c_loss: 4.1874 - p_acc: 0.2624 - p_loss: 2.1468 - r_acc: 0.8192

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  21s 2s/step - c_acc: 0.3560 - c_loss: 4.1835 - p_acc: 0.2624 - p_loss: 2.1466 - r_acc: 0.8192

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  20s 2s/step - c_acc: 0.3561 - c_loss: 4.1819 - p_acc: 0.2624 - p_loss: 2.1465 - r_acc: 0.8191

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  18s 2s/step - c_acc: 0.3563 - c_loss: 4.1787 - p_acc: 0.2624 - p_loss: 2.1462 - r_acc: 0.8190

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  16s 2s/step - c_acc: 0.3564 - c_loss: 4.1789 - p_acc: 0.2624 - p_loss: 2.1457 - r_acc: 0.8189

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  14s 2s/step - c_acc: 0.3565 - c_loss: 4.1741 - p_acc: 0.2625 - p_loss: 2.1450 - r_acc: 0.8189

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  13s 2s/step - c_acc: 0.3566 - c_loss: 4.1713 - p_acc: 0.2625 - p_loss: 2.1444 - r_acc: 0.8188

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  11s 2s/step - c_acc: 0.3568 - c_loss: 4.1648 - p_acc: 0.2625 - p_loss: 2.1441 - r_acc: 0.8188

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  9s 2s/step - c_acc: 0.3569 - c_loss: 4.1571 - p_acc: 0.2626 - p_loss: 2.1437 - r_acc: 0.8187 

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  7s 2s/step - c_acc: 0.3571 - c_loss: 4.1511 - p_acc: 0.2626 - p_loss: 2.1437 - r_acc: 0.8187

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  5s 2s/step - c_acc: 0.3572 - c_loss: 4.1490 - p_acc: 0.2626 - p_loss: 2.1432 - r_acc: 0.8186

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  3s 2s/step - c_acc: 0.3574 - c_loss: 4.1455 - p_acc: 0.2627 - p_loss: 2.1427 - r_acc: 0.8186

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  1s 2s/step - c_acc: 0.3576 - c_loss: 4.1418 - p_acc: 0.2627 - p_loss: 2.1425 - r_acc: 0.8186

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - c_acc: 0.3577 - c_loss: 4.1392 - p_acc: 0.2628 - p_loss: 2.1421 - r_acc: 0.8185

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 103s 2s/step - c_acc: 0.3577 - c_loss: 4.1392 - p_acc: 0.2628 - p_loss: 2.1421 - r_acc: 0.8185 - loss: 0.0000e+00 - sparse_categorical_accuracy: 0.3660 - val_loss: 0.0000e+00 - val_sparse_categorical_accuracy: 0.0000e+00


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  2:03 3s/step - c_acc: 0.4000 - c_loss: 3.9511 - p_acc: 0.2910 - p_loss: 2.1243 - r_acc: 0.8359

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  2:53 4s/step - c_acc: 0.3943 - c_loss: 3.9695 - p_acc: 0.2900 - p_loss: 2.1199 - r_acc: 0.8389

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  2:09 3s/step - c_acc: 0.3898 - c_loss: 3.9636 - p_acc: 0.2918 - p_loss: 2.1205 - r_acc: 0.8392

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  1:49 2s/step - c_acc: 0.3890 - c_loss: 3.9392 - p_acc: 0.2924 - p_loss: 2.1191 - r_acc: 0.8389

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:40 2s/step - c_acc: 0.3873 - c_loss: 3.9547 - p_acc: 0.2922 - p_loss: 2.1195 - r_acc: 0.8383

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 2s/step - c_acc: 0.3856 - c_loss: 3.9702 - p_acc: 0.2919 - p_loss: 2.1185 - r_acc: 0.8381

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  1:27 2s/step - c_acc: 0.3850 - c_loss: 3.9570 - p_acc: 0.2916 - p_loss: 2.1175 - r_acc: 0.8379

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:22 2s/step - c_acc: 0.3847 - c_loss: 3.9496 - p_acc: 0.2905 - p_loss: 2.1183 - r_acc: 0.8374

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  1:18 2s/step - c_acc: 0.3844 - c_loss: 3.9463 - p_acc: 0.2899 - p_loss: 2.1181 - r_acc: 0.8371

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:16 2s/step - c_acc: 0.3842 - c_loss: 3.9464 - p_acc: 0.2893 - p_loss: 2.1181 - r_acc: 0.8369

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:13 2s/step - c_acc: 0.3841 - c_loss: 3.9538 - p_acc: 0.2888 - p_loss: 2.1188 - r_acc: 0.8367

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  1:10 2s/step - c_acc: 0.3841 - c_loss: 3.9440 - p_acc: 0.2885 - p_loss: 2.1191 - r_acc: 0.8365

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:09 2s/step - c_acc: 0.3842 - c_loss: 3.9232 - p_acc: 0.2883 - p_loss: 2.1173 - r_acc: 0.8364

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  1:06 2s/step - c_acc: 0.3843 - c_loss: 3.9238 - p_acc: 0.2883 - p_loss: 2.1159 - r_acc: 0.8364

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:04 2s/step - c_acc: 0.3845 - c_loss: 3.9238 - p_acc: 0.2884 - p_loss: 2.1147 - r_acc: 0.8362

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  1:01 2s/step - c_acc: 0.3847 - c_loss: 3.9146 - p_acc: 0.2886 - p_loss: 2.1135 - r_acc: 0.8360

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  59s 2s/step - c_acc: 0.3850 - c_loss: 3.9039 - p_acc: 0.2888 - p_loss: 2.1133 - r_acc: 0.8359 

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  57s 2s/step - c_acc: 0.3853 - c_loss: 3.9035 - p_acc: 0.2888 - p_loss: 2.1130 - r_acc: 0.8357

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  57s 2s/step - c_acc: 0.3856 - c_loss: 3.8959 - p_acc: 0.2889 - p_loss: 2.1134 - r_acc: 0.8356

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  55s 2s/step - c_acc: 0.3858 - c_loss: 3.8955 - p_acc: 0.2889 - p_loss: 2.1130 - r_acc: 0.8355

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  55s 2s/step - c_acc: 0.3860 - c_loss: 3.8958 - p_acc: 0.2888 - p_loss: 2.1127 - r_acc: 0.8354

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  53s 2s/step - c_acc: 0.3862 - c_loss: 3.8953 - p_acc: 0.2888 - p_loss: 2.1121 - r_acc: 0.8353

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  51s 2s/step - c_acc: 0.3864 - c_loss: 3.8890 - p_acc: 0.2888 - p_loss: 2.1121 - r_acc: 0.8352

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  49s 2s/step - c_acc: 0.3866 - c_loss: 3.8836 - p_acc: 0.2888 - p_loss: 2.1120 - r_acc: 0.8351

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  46s 2s/step - c_acc: 0.3869 - c_loss: 3.8824 - p_acc: 0.2887 - p_loss: 2.1124 - r_acc: 0.8350

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  45s 2s/step - c_acc: 0.3871 - c_loss: 3.8829 - p_acc: 0.2887 - p_loss: 2.1117 - r_acc: 0.8350

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  43s 2s/step - c_acc: 0.3873 - c_loss: 3.8809 - p_acc: 0.2887 - p_loss: 2.1114 - r_acc: 0.8349

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  41s 2s/step - c_acc: 0.3876 - c_loss: 3.8750 - p_acc: 0.2886 - p_loss: 2.1114 - r_acc: 0.8349

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  39s 2s/step - c_acc: 0.3879 - c_loss: 3.8758 - p_acc: 0.2886 - p_loss: 2.1104 - r_acc: 0.8348

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  37s 2s/step - c_acc: 0.3881 - c_loss: 3.8734 - p_acc: 0.2885 - p_loss: 2.1098 - r_acc: 0.8348

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  35s 2s/step - c_acc: 0.3884 - c_loss: 3.8722 - p_acc: 0.2884 - p_loss: 2.1092 - r_acc: 0.8348

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  33s 2s/step - c_acc: 0.3886 - c_loss: 3.8727 - p_acc: 0.2884 - p_loss: 2.1090 - r_acc: 0.8347

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  31s 2s/step - c_acc: 0.3888 - c_loss: 3.8748 - p_acc: 0.2883 - p_loss: 2.1085 - r_acc: 0.8347

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  29s 2s/step - c_acc: 0.3890 - c_loss: 3.8759 - p_acc: 0.2883 - p_loss: 2.1083 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  27s 2s/step - c_acc: 0.3892 - c_loss: 3.8757 - p_acc: 0.2882 - p_loss: 2.1082 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  25s 2s/step - c_acc: 0.3894 - c_loss: 3.8725 - p_acc: 0.2881 - p_loss: 2.1083 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  24s 2s/step - c_acc: 0.3896 - c_loss: 3.8679 - p_acc: 0.2881 - p_loss: 2.1079 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  22s 2s/step - c_acc: 0.3898 - c_loss: 3.8662 - p_acc: 0.2880 - p_loss: 2.1075 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  20s 2s/step - c_acc: 0.3900 - c_loss: 3.8649 - p_acc: 0.2880 - p_loss: 2.1073 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  18s 2s/step - c_acc: 0.3902 - c_loss: 3.8629 - p_acc: 0.2879 - p_loss: 2.1073 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  17s 2s/step - c_acc: 0.3904 - c_loss: 3.8618 - p_acc: 0.2878 - p_loss: 2.1069 - r_acc: 0.8345

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  15s 2s/step - c_acc: 0.3906 - c_loss: 3.8619 - p_acc: 0.2878 - p_loss: 2.1067 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  13s 2s/step - c_acc: 0.3908 - c_loss: 3.8595 - p_acc: 0.2877 - p_loss: 2.1062 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  11s 2s/step - c_acc: 0.3910 - c_loss: 3.8588 - p_acc: 0.2877 - p_loss: 2.1055 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  9s 2s/step - c_acc: 0.3912 - c_loss: 3.8552 - p_acc: 0.2877 - p_loss: 2.1054 - r_acc: 0.8346 

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  7s 2s/step - c_acc: 0.3914 - c_loss: 3.8543 - p_acc: 0.2876 - p_loss: 2.1051 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  5s 2s/step - c_acc: 0.3916 - c_loss: 3.8530 - p_acc: 0.2876 - p_loss: 2.1045 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  3s 2s/step - c_acc: 0.3918 - c_loss: 3.8523 - p_acc: 0.2876 - p_loss: 2.1039 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  1s 2s/step - c_acc: 0.3920 - c_loss: 3.8533 - p_acc: 0.2876 - p_loss: 2.1040 - r_acc: 0.8346

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - c_acc: 0.3922 - c_loss: 3.8496 - p_acc: 0.2876 - p_loss: 2.1042 - r_acc: 0.8347

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 98s 2s/step - c_acc: 0.3922 - c_loss: 3.8496 - p_acc: 0.2876 - p_loss: 2.1042 - r_acc: 0.8347 - loss: 0.0000e+00 - sparse_categorical_accuracy: 0.4015 - val_loss: 0.0000e+00 - val_sparse_categorical_accuracy: 0.0000e+00


---
## Evaluate our model

A popular way to evaluate a SSL method in computer vision or for that fact any other
pre-training method as such is to learn a linear classifier on the frozen features of the
trained backbone model and evaluate the classifier on unseen images. Other methods often
include fine-tuning on the source dataset or even a target dataset with 5% or 10% labels
present. You can use the backbone we just trained for any downstream task such as image
classification (like we do here) or segmentation or detection, where the backbone models
are usually pre-trained with supervised learning.


```python
finetuning_model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        augmenter(**classification_augmenter),
        model.encoder,
        layers.Dense(10),
    ],
    name="finetuning_model",
)
finetuning_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    jit_compile=False,
)

finetuning_history = finetuning_model.fit(
    train_dataset, epochs=num_epochs, validation_data=test_dataset
)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

Epoch 1/5

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

```
</div>
    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  3:16 4s/step - acc: 0.0890 - loss: 2.3065

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  36s 760ms/step - acc: 0.0975 - loss: 2.2984

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  35s 765ms/step - acc: 0.1034 - loss: 2.2943

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  36s 783ms/step - acc: 0.1092 - loss: 2.2902

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  34s 760ms/step - acc: 0.1158 - loss: 2.2861

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  32s 748ms/step - acc: 0.1218 - loss: 2.2821

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  31s 734ms/step - acc: 0.1271 - loss: 2.2773

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  30s 720ms/step - acc: 0.1318 - loss: 2.2722

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  29s 709ms/step - acc: 0.1361 - loss: 2.2667

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  28s 717ms/step - acc: 0.1399 - loss: 2.2614

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  27s 713ms/step - acc: 0.1432 - loss: 2.2559

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  27s 721ms/step - acc: 0.1463 - loss: 2.2505

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  26s 724ms/step - acc: 0.1495 - loss: 2.2449

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  25s 721ms/step - acc: 0.1526 - loss: 2.2393

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  25s 728ms/step - acc: 0.1556 - loss: 2.2337

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  24s 724ms/step - acc: 0.1584 - loss: 2.2281

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  23s 727ms/step - acc: 0.1612 - loss: 2.2226

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  23s 721ms/step - acc: 0.1638 - loss: 2.2173

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  22s 720ms/step - acc: 0.1664 - loss: 2.2120

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  21s 722ms/step - acc: 0.1688 - loss: 2.2070

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  20s 720ms/step - acc: 0.1711 - loss: 2.2021

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  20s 716ms/step - acc: 0.1733 - loss: 2.1974

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  19s 717ms/step - acc: 0.1755 - loss: 2.1928

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  18s 714ms/step - acc: 0.1776 - loss: 2.1882

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  17s 711ms/step - acc: 0.1796 - loss: 2.1838

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  17s 708ms/step - acc: 0.1816 - loss: 2.1795

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  16s 705ms/step - acc: 0.1836 - loss: 2.1752

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  15s 703ms/step - acc: 0.1855 - loss: 2.1709

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  14s 703ms/step - acc: 0.1874 - loss: 2.1667

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  14s 702ms/step - acc: 0.1893 - loss: 2.1626

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  13s 702ms/step - acc: 0.1912 - loss: 2.1585

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  12s 704ms/step - acc: 0.1930 - loss: 2.1545

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  12s 709ms/step - acc: 0.1947 - loss: 2.1506

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  11s 711ms/step - acc: 0.1965 - loss: 2.1467

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  10s 710ms/step - acc: 0.1982 - loss: 2.1428

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  9s 711ms/step - acc: 0.1998 - loss: 2.1391 

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  9s 717ms/step - acc: 0.2014 - loss: 2.1355

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  8s 718ms/step - acc: 0.2030 - loss: 2.1319

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  7s 719ms/step - acc: 0.2046 - loss: 2.1283

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  7s 720ms/step - acc: 0.2061 - loss: 2.1249

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  6s 720ms/step - acc: 0.2075 - loss: 2.1215

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  5s 721ms/step - acc: 0.2090 - loss: 2.1182

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  5s 720ms/step - acc: 0.2104 - loss: 2.1149

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  4s 718ms/step - acc: 0.2118 - loss: 2.1116

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  3s 733ms/step - acc: 0.2131 - loss: 2.1084

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  3s 764ms/step - acc: 0.2145 - loss: 2.1053

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  2s 767ms/step - acc: 0.2158 - loss: 2.1021

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  1s 772ms/step - acc: 0.2171 - loss: 2.0991

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  0s 772ms/step - acc: 0.2184 - loss: 2.0961

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 770ms/step - acc: 0.2196 - loss: 2.0931

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.


```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 46s 855ms/step - acc: 0.2208 - loss: 2.0903 - val_acc: 0.3674 - val_loss: 1.7378


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  41s 847ms/step - acc: 0.3530 - loss: 1.7313

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  33s 703ms/step - acc: 0.3497 - loss: 1.7442

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  33s 721ms/step - acc: 0.3507 - loss: 1.7543

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  44s 962ms/step - acc: 0.3530 - loss: 1.7575

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  56s 1s/step - acc: 0.3529 - loss: 1.7612   

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  52s 1s/step - acc: 0.3527 - loss: 1.7633

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  48s 1s/step - acc: 0.3528 - loss: 1.7644

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  44s 1s/step - acc: 0.3528 - loss: 1.7649

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  41s 1s/step - acc: 0.3536 - loss: 1.7640

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  39s 984ms/step - acc: 0.3540 - loss: 1.7634

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  37s 967ms/step - acc: 0.3546 - loss: 1.7622

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  35s 944ms/step - acc: 0.3549 - loss: 1.7614

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  34s 934ms/step - acc: 0.3551 - loss: 1.7608

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  32s 915ms/step - acc: 0.3554 - loss: 1.7602

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  31s 899ms/step - acc: 0.3557 - loss: 1.7594

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  30s 892ms/step - acc: 0.3561 - loss: 1.7584

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  29s 892ms/step - acc: 0.3563 - loss: 1.7575

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  28s 882ms/step - acc: 0.3567 - loss: 1.7565

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  28s 904ms/step - acc: 0.3570 - loss: 1.7555

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  29s 975ms/step - acc: 0.3573 - loss: 1.7544

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  28s 968ms/step - acc: 0.3577 - loss: 1.7532

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  26s 955ms/step - acc: 0.3581 - loss: 1.7522

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  25s 943ms/step - acc: 0.3585 - loss: 1.7512

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  24s 932ms/step - acc: 0.3588 - loss: 1.7502

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  23s 929ms/step - acc: 0.3592 - loss: 1.7491

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  22s 919ms/step - acc: 0.3595 - loss: 1.7481

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  20s 910ms/step - acc: 0.3599 - loss: 1.7470

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  19s 902ms/step - acc: 0.3602 - loss: 1.7458

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  18s 894ms/step - acc: 0.3606 - loss: 1.7448

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  17s 888ms/step - acc: 0.3610 - loss: 1.7437

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  16s 883ms/step - acc: 0.3614 - loss: 1.7427

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  15s 879ms/step - acc: 0.3617 - loss: 1.7416

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  14s 874ms/step - acc: 0.3621 - loss: 1.7406

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  13s 867ms/step - acc: 0.3624 - loss: 1.7397

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  12s 862ms/step - acc: 0.3628 - loss: 1.7386

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  12s 858ms/step - acc: 0.3631 - loss: 1.7377

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  11s 854ms/step - acc: 0.3635 - loss: 1.7367

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  10s 849ms/step - acc: 0.3638 - loss: 1.7358

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  9s 845ms/step - acc: 0.3641 - loss: 1.7349 

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  8s 840ms/step - acc: 0.3645 - loss: 1.7340

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  7s 836ms/step - acc: 0.3648 - loss: 1.7331

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  6s 833ms/step - acc: 0.3651 - loss: 1.7322

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  5s 830ms/step - acc: 0.3654 - loss: 1.7313

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  4s 827ms/step - acc: 0.3658 - loss: 1.7304

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  4s 823ms/step - acc: 0.3661 - loss: 1.7295

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  3s 819ms/step - acc: 0.3664 - loss: 1.7286

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  2s 816ms/step - acc: 0.3668 - loss: 1.7277

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  1s 813ms/step - acc: 0.3671 - loss: 1.7268

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  0s 814ms/step - acc: 0.3674 - loss: 1.7259

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 814ms/step - acc: 0.3677 - loss: 1.7250

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 45s 892ms/step - acc: 0.3680 - loss: 1.7242 - val_acc: 0.4083 - val_loss: 1.6188


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  51s 1s/step - acc: 0.4200 - loss: 1.5782

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  36s 760ms/step - acc: 0.4195 - loss: 1.5887

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  38s 826ms/step - acc: 0.4209 - loss: 1.5900

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  37s 809ms/step - acc: 0.4204 - loss: 1.5920

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  35s 785ms/step - acc: 0.4210 - loss: 1.5918

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  34s 777ms/step - acc: 0.4208 - loss: 1.5931

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  32s 767ms/step - acc: 0.4207 - loss: 1.5935

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  32s 769ms/step - acc: 0.4205 - loss: 1.5949

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  31s 759ms/step - acc: 0.4199 - loss: 1.5964

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  29s 749ms/step - acc: 0.4192 - loss: 1.5977

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  28s 739ms/step - acc: 0.4186 - loss: 1.5990

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  29s 780ms/step - acc: 0.4180 - loss: 1.6002

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  32s 872ms/step - acc: 0.4176 - loss: 1.6007

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  31s 867ms/step - acc: 0.4172 - loss: 1.6014

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  29s 855ms/step - acc: 0.4170 - loss: 1.6018

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  28s 845ms/step - acc: 0.4168 - loss: 1.6019

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  27s 833ms/step - acc: 0.4166 - loss: 1.6022

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  26s 824ms/step - acc: 0.4165 - loss: 1.6022

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  25s 816ms/step - acc: 0.4164 - loss: 1.6023

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  24s 810ms/step - acc: 0.4163 - loss: 1.6022

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  23s 803ms/step - acc: 0.4163 - loss: 1.6021

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  22s 797ms/step - acc: 0.4163 - loss: 1.6020

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  21s 792ms/step - acc: 0.4162 - loss: 1.6018

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  20s 793ms/step - acc: 0.4162 - loss: 1.6015

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  19s 790ms/step - acc: 0.4163 - loss: 1.6012

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  18s 785ms/step - acc: 0.4163 - loss: 1.6009

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  17s 781ms/step - acc: 0.4163 - loss: 1.6006

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  17s 777ms/step - acc: 0.4163 - loss: 1.6003

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  16s 773ms/step - acc: 0.4164 - loss: 1.5999

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  15s 777ms/step - acc: 0.4164 - loss: 1.5997

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  15s 805ms/step - acc: 0.4164 - loss: 1.5994

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  15s 851ms/step - acc: 0.4165 - loss: 1.5991

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  14s 851ms/step - acc: 0.4165 - loss: 1.5989

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  13s 844ms/step - acc: 0.4165 - loss: 1.5986

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  12s 845ms/step - acc: 0.4166 - loss: 1.5983

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  12s 861ms/step - acc: 0.4166 - loss: 1.5980

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  11s 857ms/step - acc: 0.4167 - loss: 1.5977

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  10s 856ms/step - acc: 0.4168 - loss: 1.5974

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  9s 858ms/step - acc: 0.4168 - loss: 1.5972 

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  8s 892ms/step - acc: 0.4169 - loss: 1.5970

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  8s 920ms/step - acc: 0.4169 - loss: 1.5968

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  7s 914ms/step - acc: 0.4170 - loss: 1.5966

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  6s 910ms/step - acc: 0.4170 - loss: 1.5964

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  5s 906ms/step - acc: 0.4171 - loss: 1.5961

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  4s 902ms/step - acc: 0.4172 - loss: 1.5959

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  3s 899ms/step - acc: 0.4172 - loss: 1.5957

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  2s 893ms/step - acc: 0.4173 - loss: 1.5954

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  1s 887ms/step - acc: 0.4174 - loss: 1.5952

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  0s 883ms/step - acc: 0.4175 - loss: 1.5949

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 879ms/step - acc: 0.4176 - loss: 1.5947

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 47s 940ms/step - acc: 0.4177 - loss: 1.5944 - val_acc: 0.4437 - val_loss: 1.5449


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  39s 815ms/step - acc: 0.4330 - loss: 1.5302

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  33s 691ms/step - acc: 0.4268 - loss: 1.5403

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  31s 661ms/step - acc: 0.4229 - loss: 1.5457

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  30s 653ms/step - acc: 0.4219 - loss: 1.5520

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  29s 645ms/step - acc: 0.4222 - loss: 1.5541

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  28s 654ms/step - acc: 0.4241 - loss: 1.5541

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  27s 651ms/step - acc: 0.4257 - loss: 1.5535

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  27s 650ms/step - acc: 0.4275 - loss: 1.5521

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  26s 656ms/step - acc: 0.4288 - loss: 1.5522

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  26s 656ms/step - acc: 0.4301 - loss: 1.5522

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  25s 655ms/step - acc: 0.4313 - loss: 1.5521

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  25s 659ms/step - acc: 0.4321 - loss: 1.5522

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  24s 664ms/step - acc: 0.4329 - loss: 1.5521

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  23s 661ms/step - acc: 0.4336 - loss: 1.5522

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  23s 664ms/step - acc: 0.4342 - loss: 1.5520

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  24s 719ms/step - acc: 0.4348 - loss: 1.5519

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  25s 781ms/step - acc: 0.4352 - loss: 1.5519

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  24s 775ms/step - acc: 0.4355 - loss: 1.5519

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  23s 767ms/step - acc: 0.4358 - loss: 1.5518

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  22s 760ms/step - acc: 0.4361 - loss: 1.5517

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  21s 756ms/step - acc: 0.4364 - loss: 1.5515

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  21s 759ms/step - acc: 0.4366 - loss: 1.5514

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  20s 756ms/step - acc: 0.4368 - loss: 1.5513

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  19s 752ms/step - acc: 0.4370 - loss: 1.5511

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  18s 750ms/step - acc: 0.4372 - loss: 1.5508

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  17s 748ms/step - acc: 0.4374 - loss: 1.5505

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  17s 743ms/step - acc: 0.4377 - loss: 1.5502

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  16s 740ms/step - acc: 0.4379 - loss: 1.5498

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  15s 736ms/step - acc: 0.4382 - loss: 1.5494

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  14s 733ms/step - acc: 0.4385 - loss: 1.5489

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  13s 731ms/step - acc: 0.4387 - loss: 1.5484

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  13s 730ms/step - acc: 0.4390 - loss: 1.5479

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  12s 727ms/step - acc: 0.4393 - loss: 1.5474

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  11s 726ms/step - acc: 0.4396 - loss: 1.5469

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  10s 724ms/step - acc: 0.4398 - loss: 1.5464

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  10s 722ms/step - acc: 0.4401 - loss: 1.5460

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  9s 721ms/step - acc: 0.4403 - loss: 1.5456 

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  8s 718ms/step - acc: 0.4405 - loss: 1.5452

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  7s 716ms/step - acc: 0.4407 - loss: 1.5448

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  7s 714ms/step - acc: 0.4410 - loss: 1.5443

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  6s 712ms/step - acc: 0.4412 - loss: 1.5439

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  5s 710ms/step - acc: 0.4414 - loss: 1.5435

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  4s 708ms/step - acc: 0.4416 - loss: 1.5431

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  4s 709ms/step - acc: 0.4418 - loss: 1.5427

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  3s 707ms/step - acc: 0.4420 - loss: 1.5423

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  2s 705ms/step - acc: 0.4421 - loss: 1.5419

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  2s 703ms/step - acc: 0.4423 - loss: 1.5415

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  1s 702ms/step - acc: 0.4424 - loss: 1.5412

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  0s 702ms/step - acc: 0.4426 - loss: 1.5408

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 700ms/step - acc: 0.4427 - loss: 1.5405

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 38s 763ms/step - acc: 0.4429 - loss: 1.5401 - val_acc: 0.4571 - val_loss: 1.4927


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/50 [37m━━━━━━━━━━━━━━━━━━━━  38s 793ms/step - acc: 0.4810 - loss: 1.4582

<div class="k-default-codeblock">
```

```
</div>
  2/50 [37m━━━━━━━━━━━━━━━━━━━━  30s 626ms/step - acc: 0.4762 - loss: 1.4669

<div class="k-default-codeblock">
```

```
</div>
  3/50 ━[37m━━━━━━━━━━━━━━━━━━━  31s 660ms/step - acc: 0.4742 - loss: 1.4726

<div class="k-default-codeblock">
```

```
</div>
  4/50 ━[37m━━━━━━━━━━━━━━━━━━━  30s 657ms/step - acc: 0.4722 - loss: 1.4774

<div class="k-default-codeblock">
```

```
</div>
  5/50 ━━[37m━━━━━━━━━━━━━━━━━━  30s 679ms/step - acc: 0.4700 - loss: 1.4827

<div class="k-default-codeblock">
```

```
</div>
  6/50 ━━[37m━━━━━━━━━━━━━━━━━━  30s 692ms/step - acc: 0.4679 - loss: 1.4875

<div class="k-default-codeblock">
```

```
</div>
  7/50 ━━[37m━━━━━━━━━━━━━━━━━━  29s 683ms/step - acc: 0.4666 - loss: 1.4902

<div class="k-default-codeblock">
```

```
</div>
  8/50 ━━━[37m━━━━━━━━━━━━━━━━━  28s 686ms/step - acc: 0.4654 - loss: 1.4927

<div class="k-default-codeblock">
```

```
</div>
  9/50 ━━━[37m━━━━━━━━━━━━━━━━━  28s 687ms/step - acc: 0.4647 - loss: 1.4942

<div class="k-default-codeblock">
```

```
</div>
 10/50 ━━━━[37m━━━━━━━━━━━━━━━━  27s 684ms/step - acc: 0.4643 - loss: 1.4949

<div class="k-default-codeblock">
```

```
</div>
 11/50 ━━━━[37m━━━━━━━━━━━━━━━━  26s 683ms/step - acc: 0.4639 - loss: 1.4959

<div class="k-default-codeblock">
```

```
</div>
 12/50 ━━━━[37m━━━━━━━━━━━━━━━━  25s 681ms/step - acc: 0.4634 - loss: 1.4969

<div class="k-default-codeblock">
```

```
</div>
 13/50 ━━━━━[37m━━━━━━━━━━━━━━━  25s 678ms/step - acc: 0.4630 - loss: 1.4974

<div class="k-default-codeblock">
```

```
</div>
 14/50 ━━━━━[37m━━━━━━━━━━━━━━━  24s 680ms/step - acc: 0.4626 - loss: 1.4980

<div class="k-default-codeblock">
```

```
</div>
 15/50 ━━━━━━[37m━━━━━━━━━━━━━━  23s 684ms/step - acc: 0.4624 - loss: 1.4983

<div class="k-default-codeblock">
```

```
</div>
 16/50 ━━━━━━[37m━━━━━━━━━━━━━━  23s 685ms/step - acc: 0.4621 - loss: 1.4986

<div class="k-default-codeblock">
```

```
</div>
 17/50 ━━━━━━[37m━━━━━━━━━━━━━━  22s 687ms/step - acc: 0.4619 - loss: 1.4988

<div class="k-default-codeblock">
```

```
</div>
 18/50 ━━━━━━━[37m━━━━━━━━━━━━━  22s 688ms/step - acc: 0.4616 - loss: 1.4991

<div class="k-default-codeblock">
```

```
</div>
 19/50 ━━━━━━━[37m━━━━━━━━━━━━━  21s 687ms/step - acc: 0.4615 - loss: 1.4993

<div class="k-default-codeblock">
```

```
</div>
 20/50 ━━━━━━━━[37m━━━━━━━━━━━━  20s 687ms/step - acc: 0.4613 - loss: 1.4995

<div class="k-default-codeblock">
```

```
</div>
 21/50 ━━━━━━━━[37m━━━━━━━━━━━━  19s 686ms/step - acc: 0.4612 - loss: 1.4996

<div class="k-default-codeblock">
```

```
</div>
 22/50 ━━━━━━━━[37m━━━━━━━━━━━━  19s 683ms/step - acc: 0.4611 - loss: 1.4995

<div class="k-default-codeblock">
```

```
</div>
 23/50 ━━━━━━━━━[37m━━━━━━━━━━━  18s 684ms/step - acc: 0.4612 - loss: 1.4994

<div class="k-default-codeblock">
```

```
</div>
 24/50 ━━━━━━━━━[37m━━━━━━━━━━━  17s 683ms/step - acc: 0.4612 - loss: 1.4993

<div class="k-default-codeblock">
```

```
</div>
 25/50 ━━━━━━━━━━[37m━━━━━━━━━━  17s 682ms/step - acc: 0.4611 - loss: 1.4993

<div class="k-default-codeblock">
```

```
</div>
 26/50 ━━━━━━━━━━[37m━━━━━━━━━━  17s 709ms/step - acc: 0.4611 - loss: 1.4992

<div class="k-default-codeblock">
```

```
</div>
 27/50 ━━━━━━━━━━[37m━━━━━━━━━━  17s 748ms/step - acc: 0.4611 - loss: 1.4989

<div class="k-default-codeblock">
```

```
</div>
 28/50 ━━━━━━━━━━━[37m━━━━━━━━━  16s 745ms/step - acc: 0.4612 - loss: 1.4988

<div class="k-default-codeblock">
```

```
</div>
 29/50 ━━━━━━━━━━━[37m━━━━━━━━━  15s 741ms/step - acc: 0.4612 - loss: 1.4986

<div class="k-default-codeblock">
```

```
</div>
 30/50 ━━━━━━━━━━━━[37m━━━━━━━━  14s 738ms/step - acc: 0.4612 - loss: 1.4983

<div class="k-default-codeblock">
```

```
</div>
 31/50 ━━━━━━━━━━━━[37m━━━━━━━━  14s 738ms/step - acc: 0.4613 - loss: 1.4980

<div class="k-default-codeblock">
```

```
</div>
 32/50 ━━━━━━━━━━━━[37m━━━━━━━━  13s 735ms/step - acc: 0.4615 - loss: 1.4976

<div class="k-default-codeblock">
```

```
</div>
 33/50 ━━━━━━━━━━━━━[37m━━━━━━━  12s 732ms/step - acc: 0.4616 - loss: 1.4972

<div class="k-default-codeblock">
```

```
</div>
 34/50 ━━━━━━━━━━━━━[37m━━━━━━━  11s 730ms/step - acc: 0.4617 - loss: 1.4968

<div class="k-default-codeblock">
```

```
</div>
 35/50 ━━━━━━━━━━━━━━[37m━━━━━━  10s 727ms/step - acc: 0.4618 - loss: 1.4964

<div class="k-default-codeblock">
```

```
</div>
 36/50 ━━━━━━━━━━━━━━[37m━━━━━━  10s 726ms/step - acc: 0.4619 - loss: 1.4961

<div class="k-default-codeblock">
```

```
</div>
 37/50 ━━━━━━━━━━━━━━[37m━━━━━━  9s 726ms/step - acc: 0.4620 - loss: 1.4957 

<div class="k-default-codeblock">
```

```
</div>
 38/50 ━━━━━━━━━━━━━━━[37m━━━━━  8s 724ms/step - acc: 0.4621 - loss: 1.4954

<div class="k-default-codeblock">
```

```
</div>
 39/50 ━━━━━━━━━━━━━━━[37m━━━━━  7s 722ms/step - acc: 0.4622 - loss: 1.4951

<div class="k-default-codeblock">
```

```
</div>
 40/50 ━━━━━━━━━━━━━━━━[37m━━━━  7s 722ms/step - acc: 0.4622 - loss: 1.4948

<div class="k-default-codeblock">
```

```
</div>
 41/50 ━━━━━━━━━━━━━━━━[37m━━━━  6s 720ms/step - acc: 0.4623 - loss: 1.4945

<div class="k-default-codeblock">
```

```
</div>
 42/50 ━━━━━━━━━━━━━━━━[37m━━━━  5s 718ms/step - acc: 0.4624 - loss: 1.4942

<div class="k-default-codeblock">
```

```
</div>
 43/50 ━━━━━━━━━━━━━━━━━[37m━━━  5s 716ms/step - acc: 0.4625 - loss: 1.4939

<div class="k-default-codeblock">
```

```
</div>
 44/50 ━━━━━━━━━━━━━━━━━[37m━━━  4s 714ms/step - acc: 0.4625 - loss: 1.4936

<div class="k-default-codeblock">
```

```
</div>
 45/50 ━━━━━━━━━━━━━━━━━━[37m━━  3s 712ms/step - acc: 0.4626 - loss: 1.4933

<div class="k-default-codeblock">
```

```
</div>
 46/50 ━━━━━━━━━━━━━━━━━━[37m━━  2s 711ms/step - acc: 0.4627 - loss: 1.4930

<div class="k-default-codeblock">
```

```
</div>
 47/50 ━━━━━━━━━━━━━━━━━━[37m━━  2s 710ms/step - acc: 0.4628 - loss: 1.4927

<div class="k-default-codeblock">
```

```
</div>
 48/50 ━━━━━━━━━━━━━━━━━━━[37m━  1s 710ms/step - acc: 0.4629 - loss: 1.4924

<div class="k-default-codeblock">
```

```
</div>
 49/50 ━━━━━━━━━━━━━━━━━━━[37m━  0s 710ms/step - acc: 0.4629 - loss: 1.4921

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 0s 709ms/step - acc: 0.4630 - loss: 1.4917

<div class="k-default-codeblock">
```

```
</div>
 50/50 ━━━━━━━━━━━━━━━━━━━━ 39s 771ms/step - acc: 0.4631 - loss: 1.4914 - val_acc: 0.4692 - val_loss: 1.4436


Self supervised learning is particularly helpful when you do only have access to very
limited labeled training data but you can manage to build a large corpus of unlabeled
data as shown by previous methods like [SEER](https://arxiv.org/abs/2103.01988),
[SimCLR](https://arxiv.org/abs/2002.05709), [SwAV](https://arxiv.org/abs/2006.09882) and
more.

You should also take a look at the blog posts for these papers which neatly show that it is
possible to achieve good results with few class labels by first pretraining on a large
unlabeled dataset and then fine-tuning on a smaller labeled dataset:

- [Advancing Self-Supervised and Semi-Supervised Learning with SimCLR](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html)
- [High-performance self-supervised image classification with contrastive clustering](https://ai.facebook.com/blog/high-performance-self-supervised-image-classification-with-contrastive-clustering/)
- [Self-supervised learning: The dark matter of intelligence](https://ai.facebook.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/)

You are also advised to check out the [original paper](https://arxiv.org/abs/2104.14548).

*Many thanks to [Debidatta Dwibedi](https://twitter.com/debidatta) (Google Research),
primary author of the NNCLR paper for his super-insightful reviews for this example.
This example also takes inspiration from the [SimCLR Keras Example](https://keras.io/examples/vision/semisupervised_simclr/).*

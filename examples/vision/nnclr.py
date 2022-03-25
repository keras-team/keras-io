"""
Title: Self-supervised contrastive learning with NNCLR
Author: [Rishit Dagli](https://twitter.com/rishit_dagli)
Date created: 2021/09/13
Last modified: 2021/09/13
Description: Implementation of NNCLR, a self-supervised learning method for computer vision.
"""
"""
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

This example requires TensorFlow 2.6 or higher, as well as `tensorflow_datasets`, which can
be installed with this command:
"""

"""shell
pip install tensorflow-datasets
"""

"""
## Setup
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

"""
## Hyperparameters

A greater `queue_size` most likely means better performance as shown in the original
paper, but introduces significant computational overhead. The authors show that the best
results of NNCLR are achieved with a queue size of 98,304 (the largest `queue_size` they
experimented on). We here use 10,000 to show a working example.
"""

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
num_epochs = 25
steps_per_epoch = 200

"""
## Load the Dataset

We load the [STL-10](http://ai.stanford.edu/~acoates/stl10/) dataset from
TensorFlow Datasets, an image recognition dataset for developing unsupervised
feature learning, deep learning, self-taught learning algorithms. It is inspired by the
CIFAR-10 dataset, with some modifications.
"""

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

"""
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
"""

"""
### Random Resized Crops
"""


class RandomResizedCrop(layers.Layer):
    def __init__(self, scale, ratio):
        super(RandomResizedCrop, self).__init__()
        self.scale = scale
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]

        random_scales = tf.random.uniform((batch_size,), self.scale[0], self.scale[1])
        random_ratios = tf.exp(
            tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
        )

        new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
        new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
        height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
        width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )
        images = tf.image.crop_and_resize(
            images, bounding_boxes, tf.range(batch_size), (height, width)
        )
        return images


"""
### Random Brightness
"""


class RandomBrightness(layers.Layer):
    def __init__(self, brightness):
        super(RandomBrightness, self).__init__()
        self.brightness = brightness

    def blend(self, images_1, images_2, ratios):
        return tf.clip_by_value(ratios * images_1 + (1.0 - ratios) * images_2, 0, 1)

    def random_brightness(self, images):
        # random interpolation/extrapolation between the image and darkness
        return self.blend(
            images,
            0,
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.brightness, 1 + self.brightness
            ),
        )

    def call(self, images):
        images = self.random_brightness(images)
        return images


"""
### Prepare augmentation module
"""


def augmenter(brightness, name, scale):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            RandomResizedCrop(scale=scale, ratio=(3 / 4, 4 / 3)),
            RandomBrightness(brightness=brightness),
        ],
        name=name,
    )


"""
### Encoder architecture

Using a ResNet-50 as the encoder architecture
is standard in the literature. In the original paper, the authors use ResNet-50 as
the encoder architecture and spatially average the outputs of ResNet-50. However, keep in
mind that more powerful models will not only increase training time but will also
require more memory and will limit the maximal batch size you can use. For the purpose of
this example, we just use four convolutional layers.
"""


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


"""
## The NNCLR model for contrastive pre-training

We train an encoder on unlabeled images with a contrastive loss. A nonlinear projection
head is attached to the top of the encoder, as it improves the quality of representations
of the encoder.
"""


class NNCLR(keras.Model):
    def __init__(
        self,
        temperature,
        queue_size,
    ):
        super(NNCLR, self).__init__()
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
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super(NNCLR, self).compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

    def nearest_neighbour(self, projections):
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )
        nn_projections = tf.gather(
            self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )
        return projections + tf.stop_gradient(nn_projections - projections)

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        batch_size = tf.shape(features_1, out_type=tf.float32)[0]
        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self.temperature
        )
        similarities_1_2_2 = (
            tf.matmul(
                projections_2, self.nearest_neighbour(projections_1), transpose_b=True
            )
            / self.temperature
        )

        similarities_2_1_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self.temperature
        )
        similarities_2_1_2 = (
            tf.matmul(
                projections_1, self.nearest_neighbour(projections_2), transpose_b=True
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat(
                [
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                    contrastive_labels,
                ],
                axis=0,
            ),
            tf.concat(
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
            tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0)
        )
        return loss

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
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


"""
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
"""

model = NNCLR(temperature=temperature, queue_size=queue_size)
model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
)
pretrain_history = model.fit(
    train_dataset, epochs=num_epochs, validation_data=test_dataset
)

"""
## Evaluate our model

A popular way to evaluate a SSL method in computer vision or for that fact any other
pre-training method as such is to learn a linear classifier on the frozen features of the
trained backbone model and evaluate the classifier on unseen images. Other methods often
include fine-tuning on the source dataset or even a target dataset with 5% or 10% labels
present. You can use the backbone we just trained for any downstream task such as image
classification (like we do here) or segmentation or detection, where the backbone models
are usually pre-trained with supervised learning.
"""

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
)

finetuning_history = finetuning_model.fit(
    labeled_train_dataset, epochs=num_epochs, validation_data=test_dataset
)

"""
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
"""

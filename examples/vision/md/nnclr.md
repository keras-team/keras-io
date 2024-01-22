# Self-supervised contrastive learning with NNCLR

**Author:** [Rishit Dagli](https://twitter.com/rishit_dagli)<br>
**Date created:** 2021/09/13<br>
**Last modified:** 2024/01/21<br>
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

This example requires TensorFlow 2.6 or higher, as well as `tensorflow_datasets`, which can
be installed with this command:


```python
!pip install tensorflow-datasets
```

<div class="k-default-codeblock">
```
Requirement already satisfied: tensorflow-datasets in /opt/conda/lib/python3.7/site-packages (4.3.0)
Requirement already satisfied: requests>=2.19.0 in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (2.25.1)
Requirement already satisfied: typing-extensions in /home/jupyter/.local/lib/python3.7/site-packages (from tensorflow-datasets) (3.7.4.3)
Requirement already satisfied: tensorflow-metadata in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (1.2.0)
Requirement already satisfied: absl-py in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (0.13.0)
Requirement already satisfied: promise in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (2.3)
Requirement already satisfied: six in /home/jupyter/.local/lib/python3.7/site-packages (from tensorflow-datasets) (1.15.0)
Requirement already satisfied: termcolor in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (1.1.0)
Requirement already satisfied: protobuf>=3.12.2 in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (3.16.0)
Requirement already satisfied: tqdm in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (4.62.2)
Requirement already satisfied: attrs>=18.1.0 in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (21.2.0)
Requirement already satisfied: future in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (0.18.2)
Requirement already satisfied: dill in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (0.3.4)
Requirement already satisfied: importlib-resources in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (5.2.2)
Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (from tensorflow-datasets) (1.19.5)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.7/site-packages (from requests>=2.19.0->tensorflow-datasets) (2021.5.30)
Requirement already satisfied: chardet<5,>=3.0.2 in /opt/conda/lib/python3.7/site-packages (from requests>=2.19.0->tensorflow-datasets) (4.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.7/site-packages (from requests>=2.19.0->tensorflow-datasets) (2.10)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.7/site-packages (from requests>=2.19.0->tensorflow-datasets) (1.26.6)
Requirement already satisfied: zipp>=3.1.0 in /opt/conda/lib/python3.7/site-packages (from importlib-resources->tensorflow-datasets) (3.5.0)
Requirement already satisfied: googleapis-common-protos<2,>=1.52.0 in /opt/conda/lib/python3.7/site-packages (from tensorflow-metadata->tensorflow-datasets) (1.53.0)
Collecting absl-py
  Downloading absl_py-0.12.0-py3-none-any.whl (129 kB)
[K     |████████████████████████████████| 129 kB 8.1 MB/s 
[?25hInstalling collected packages: absl-py
  Attempting uninstall: absl-py
    Found existing installation: absl-py 0.13.0
    Uninstalling absl-py-0.13.0:
      Successfully uninstalled absl-py-0.13.0
[31mERROR: Could not install packages due to an OSError: [Errno 13] Permission denied: '_flagvalues.cpython-37.pyc'
Consider using the `--user` option or check the permissions.
[0m

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
num_epochs = 25
steps_per_epoch = 200
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

<div class="k-default-codeblock">
```
[1mDownloading and preparing dataset 2.46 GiB (download: 2.46 GiB, generated: 1.86 GiB, total: 4.32 GiB) to /home/jupyter/tensorflow_datasets/stl10/1.0.0...[0m

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/3 [00:00<?, ? splits/s]

Generating train examples...:   0%|          | 0/5000 [00:00<?, ? examples/s]

2024-01-21 17:09:18.680092: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2024-01-21 17:09:19.511196: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-01-21 17:09:19.511485: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-01-21 17:09:19.648782: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-01-21 17:09:19.862449: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2024-01-21 17:09:19.864743: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-01-21 17:09:29.922392: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT

Shuffling stl10-train.tfrecord...:   0%|          | 0/5000 [00:00<?, ? examples/s]

Generating test examples...:   0%|          | 0/8000 [00:00<?, ? examples/s]

Shuffling stl10-test.tfrecord...:   0%|          | 0/8000 [00:00<?, ? examples/s]

Generating unlabelled examples...:   0%|          | 0/100000 [00:00<?, ? examples/s]

Shuffling stl10-unlabelled.tfrecord...:   0%|          | 0/100000 [00:00<?, ? examples/s]

[1mDataset stl10 downloaded and prepared to /home/jupyter/tensorflow_datasets/stl10/1.0.0. Subsequent calls will reuse this data.[0m

```
</div>
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
        support_similarities = ops.matmul(projections, ops.tranpose(self.feature_queue))
        nn_projections = ops.take(
            self.feature_queue, ops.argmax(support_similarities, axis=1), axis=0
        )
        return projections + ops.stop_gradient(nn_projections - projections)

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = keras.utils.normalize(features_1, axis=1, order=2)
        features_2 = keras.utils.normalize(features_2, axis=1, order=2)
        similarities = ops.matmul(features_1, ops.tranpose(features_2))
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
            ops.matmul(ops.tranpose(features_1), features_2) / batch_size
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
                projections_2, ops.tranpose(self.nearest_neighbour(projections_1))
            )
            / self.temperature
        )

        similarities_2_1_1 = (
            ops.matmul(
                self.nearest_neighbour(projections_2), ops.tranpose(projections_1)
            )
            / self.temperature
        )
        similarities_2_1_2 = (
            ops.matmul(
                projections_1, ops.tranpose(self.nearest_neighbour(projections_2))
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
Epoch 1/25
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
200/200 [==============================] - 46s 125ms/step - c_loss: 3.2847 - c_acc: 0.4083 - r_acc: 0.4421 - p_loss: 2.2195 - p_acc: 0.1298 - val_p_loss: 2.1121 - val_p_acc: 0.2483
Epoch 2/25
200/200 [==============================] - 44s 123ms/step - c_loss: 3.0548 - c_acc: 0.4996 - r_acc: 0.4427 - p_loss: 2.0976 - p_acc: 0.1604 - val_p_loss: 1.9856 - val_p_acc: 0.2827
Epoch 3/25
200/200 [==============================] - 43s 121ms/step - c_loss: 2.8195 - c_acc: 0.5898 - r_acc: 0.4433 - p_loss: 1.9762 - p_acc: 0.1910 - val_p_loss: 1.8799 - val_p_acc: 0.3255
Epoch 4/25
200/200 [==============================] - 42s 119ms/step - c_loss: 2.5867 - c_acc: 0.6812 - r_acc: 0.4439 - p_loss: 1.8550 - p_acc: 0.2216 - val_p_loss: 1.7805 - val_p_acc: 0.3410
Epoch 5/25
200/200 [==============================] - 41s 117ms/step - c_loss: 2.3545 - c_acc: 0.7726 - r_acc: 0.4445 - p_loss: 1.7338 - p_acc: 0.2522 - val_p_loss: 1.7508 - val_p_acc: 0.3562
Epoch 6/25
200/200 [==============================] - 40s 115ms/step - c_loss: 2.1242 - c_acc: 0.8630 - r_acc: 0.4451 - p_loss: 1.6126 - p_acc: 0.2828 - val_p_loss: 1.6986 - val_p_acc: 0.3657
Epoch 7/25
200/200 [==============================] - 39s 113ms/step - c_loss: 1.8918 - c_acc: 0.9532 - r_acc: 0.4457 - p_loss: 1.4914 - p_acc: 0.3134 - val_p_loss: 1.6755 - val_p_acc: 0.3807
Epoch 8/25
200/200 [==============================] - 38s 111ms/step - c_loss: 1.6595 - c_acc: 0.9899 - r_acc: 0.4463 - p_loss: 1.3702 - p_acc: 0.3440 - val_p_loss: 1.6962 - val_p_acc: 0.3771
Epoch 9/25
200/200 [==============================] - 37s 109ms/step - c_loss: 1.4272 - c_acc: 0.9955 - r_acc: 0.4469 - p_loss: 1.2490 - p_acc: 0.3746 - val_p_loss: 1.6273 - val_p_acc: 0.3846
Epoch 10/25
200/200 [==============================] - 36s 107ms/step - c_loss: 1.1949 - c_acc: 0.9999 - r_acc: 0.4475 - p_loss: 1.1268 - p_acc: 0.4052 - val_p_loss: 1.5887 - val_p_acc: 0.3911
Epoch 11/25
200/200 [==============================] - 35s 105ms/step - c_loss: 0.9627 - c_acc: 0.9999 - r_acc: 0.4481 - p_loss: 1.0046 - p_acc: 0.4358 - val_p_loss: 1.5561 - val_p_acc: 0.3902
Epoch 12/25
200/200 [==============================] - 34s 103ms/step - c_loss: 0.7304 - c_acc: 0.9999 - r_acc: 0.4487 - p_loss: 0.8824 - p_acc: 0.4664 - val_p_loss: 1.6117 - val_p_acc: 0.4062
Epoch 13/25
200/200 [==============================] - 33s 101ms/step - c_loss: 0.4981 - c_acc: 0.9999 - r_acc: 0.4493 - p_loss: 0.7602 - p_acc: 0.4970 - val_p_loss: 1.5959 - val_p_acc: 0.3965
Epoch 14/25
200/200 [==============================] - 32s 99ms/step - c_loss: 0.2658 - c_acc: 0.9999 - r_acc: 0.4499 - p_loss: 0.6380 - p_acc: 0.5276 - val_p_loss: 1.5237 - val_p_acc: 0.3971
Epoch 15/25
200/200 [==============================] - 31s 97ms/step - c_loss: 0.0335 - c_acc: 0.9999 - r_acc: 0.4505 - p_loss: 0.5158 - p_acc: 0.5582 - val_p_loss: 1.5431 - val_p_acc: 0.4121
Epoch 16/25
200/200 [==============================] - 30s 95ms/step - c_loss: 0.1910 - c_acc: 0.9999 - r_acc: 0.4511 - p_loss: 0.3936 - p_acc: 0.5888 - val_p_loss: 1.5567 - val_p_acc: 0.4108
Epoch 17/25
200/200 [==============================] - 29s 93ms/step - c_loss: 0.3484 - c_acc: 0.9999 - r_acc: 0.4517 - p_loss: 0.2714 - p_acc: 0.6194 - val_p_loss: 1.6413 - val_p_acc: 0.4040
Epoch 18/25
200/200 [==============================] - 28s 91ms/step - c_loss: 0.5058 - c_acc: 0.9999 - r_acc: 0.4523 - p_loss: 0.1492 - p_acc: 0.6500 - val_p_loss: 1.5944 - val_p_acc: 0.4164
Epoch 19/25
200/200 [==============================] - 27s 89ms/step - c_loss: 0.6633 - c_acc: 0.9999 - r_acc: 0.4529 - p_loss: 0.0270 - p_acc: 0.6806 - val_p_loss: 1.5617 - val_p_acc: 0.4136
Epoch 20/25
200/200 [==============================] - 26s 87ms/step - c_loss: 0.8207 - c_acc: 0.9999 - r_acc: 0.4535 - p_loss: 0.2086 - p_acc: 0.7112 - val_p_loss: 1.5843 - val_p_acc: 0.4052
Epoch 21/25
200/200 [==============================] - 25s 85ms/step - c_loss: 0.9781 - c_acc: 0.9999 - r_acc: 0.4541 - p_loss: 0.3902 - p_acc: 0.7418 - val_p_loss: 1.5242 - val_p_acc: 0.4151
Epoch 22/25
200/200 [==============================] - 24s 83ms/step - c_loss: 1.1355 - c_acc: 0.9920 - r_acc: 0.4547 - p_loss: 0.5718 - p_acc: 0.7724 - val_p_loss: 1.6004 - val_p_acc: 0.4182
Epoch 23/25
200/200 [==============================] - 23s 81ms/step - c_loss: 1.2929 - c_acc: 0.9841 - r_acc: 0.4553 - p_loss: 0.7534 - p_acc: 0.8030 - val_p_loss: 1.6552 - val_p_acc: 0.4057
Epoch 24/25
200/200 [==============================] - 28s 126ms/step - c_loss: 1.3291 - c_acc: 0.9260 - r_acc: 0.4599 - p_loss: 1.5494 - p_acc: 0.4518 - val_p_loss: 1.5957 - val_p_acc: 0.4169
Epoch 25/25
200/200 [==============================] - 27s 123ms/step - c_loss: 1.2963 - c_acc: 0.9281 - r_acc: 0.4613 - p_loss: 1.5282 - p_acc: 0.4517 - val_p_loss: 1.6376 - val_p_acc: 0.4181


```
</div>
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
    labeled_train_dataset, epochs=num_epochs, validation_data=test_dataset
)
```

<div class="k-default-codeblock">
```
Epoch 1/25
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
200/200 [==============================] - 4s 15ms/step - loss: 1.9034 - acc: 0.2819 - val_loss: 1.6183 - val_acc: 0.3754
Epoch 2/25
200/200 [==============================] - 4s 14ms/step - loss: 1.5453 - acc: 0.4297 - val_loss: 1.4769 - val_acc: 0.4318
Epoch 3/25
200/200 [==============================] - 4s 14ms/step - loss: 1.4322 - acc: 0.4716 - val_loss: 1.3994 - val_acc: 0.4659
Epoch 4/25
200/200 [==============================] - 4s 14ms/step - loss: 1.3691 - acc: 0.5042 - val_loss: 1.3348 - val_acc: 0.4997
Epoch 5/25
200/200 [==============================] - 4s 14ms/step - loss: 1.2919 - acc: 0.5341 - val_loss: 1.2853 - val_acc: 0.5273
Epoch 6/25
200/200 [==============================] - 4s 14ms/step - loss: 1.2223 - acc: 0.5593 - val_loss: 1.2662 - val_acc: 0.5229
Epoch 7/25
200/200 [==============================] - 4s 14ms/step - loss: 1.1417 - acc: 0.5874 - val_loss: 1.2084 - val_acc: 0.5523
Epoch 8/25
200/200 [==============================] - 4s 15ms/step - loss: 1.0787 - acc: 0.6171 - val_loss: 1.3145 - val_acc: 0.5076
Epoch 9/25
200/200 [==============================] - 4s 14ms/step - loss: 1.0522 - acc: 0.6368 - val_loss: 1.2697 - val_acc: 0.5321
Epoch 10/25
200/200 [==============================] - 4s 14ms/step - loss: 1.0153 - acc: 0.6453 - val_loss: 1.2203 - val_acc: 0.5476
Epoch 11/25
200/200 [==============================] - 4s 16ms/step - loss: 0.9795 - acc: 0.6517 - val_loss: 1.2022 - val_acc: 0.5598
Epoch 12/25
200/200 [==============================] - 4s 15ms/step - loss: 0.9476 - acc: 0.6714 - val_loss: 1.1947 - val_acc: 0.5562
Epoch 13/25
200/200 [==============================] - 4s 14ms/step - loss: 0.9245 - acc: 0.6872 - val_loss: 1.1284 - val_acc: 0.5758
Epoch 14/25
200/200 [==============================] - 4s 15ms/step - loss: 0.9011 - acc: 0.7038 - val_loss: 1.1058 - val_acc: 0.5954
Epoch 15/25
200/200 [==============================] - 5s 15ms/step - loss: 0.8339 - acc: 0.7183 - val_loss: 1.1106 - val_acc: 0.5978
Epoch 16/25
200/200 [==============================] - 4s 15ms/step - loss: 0.8117 - acc: 0.7258 - val_loss: 1.1688 - val_acc: 0.5936
Epoch 17/25
200/200 [==============================] - 4s 15ms/step - loss: 0.7943 - acc: 0.7347 - val_loss: 1.1343 - val_acc: 0.6109
Epoch 18/25
200/200 [==============================] - 4s 14ms/step - loss: 0.7754 - acc: 0.7444 - val_loss: 1.2063 - val_acc: 0.5844
Epoch 19/25
200/200 [==============================] - 4s 14ms/step - loss: 0.7337 - acc: 0.7567 - val_loss: 1.1897 - val_acc: 0.5945
Epoch 20/25
200/200 [==============================] - 5s 15ms/step - loss: 0.7527 - acc: 0.7509 - val_loss: 1.1084 - val_acc: 0.6112
Epoch 21/25
200/200 [==============================] - 4s 15ms/step - loss: 0.7135 - acc: 0.7627 - val_loss: 1.2842 - val_acc: 0.5679
Epoch 22/25
200/200 [==============================] - 4s 15ms/step - loss: 0.7008 - acc: 0.7711 - val_loss: 1.1004 - val_acc: 0.6074
Epoch 23/25
200/200 [==============================] - 4s 14ms/step - loss: 0.6844 - acc: 0.7786 - val_loss: 1.1809 - val_acc: 0.6077
Epoch 24/25
200/200 [==============================] - 4s 14ms/step - loss: 0.6931 - acc: 0.7652 - val_loss: 1.1696 - val_acc: 0.6023
Epoch 25/25
200/200 [==============================] - 4s 14ms/step - loss: 0.6533 - acc: 0.7873 - val_loss: 1.1889 - val_acc: 0.5969


```
</div>
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

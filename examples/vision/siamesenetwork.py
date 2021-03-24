"""
Title: Siamese Network to compare similarity with a triplet loss
Author: [Hazem Essam](https://twitter.com/hazemessamm) and [Santiago L. Valdarrama](https://twitter.com/svpino)
Date created: 2021/03/13
Last modified: 2021/03/22
Description: A Siamese Network for a similarity comparison with `tf.data`, a triplet loss, and a custom training loop.
"""


"""
TODO: Still need to work on this.

# Introduction

You can use a [Siamese Network](https://en.wikipedia.org/wiki/Siamese_neural_network) 
to solve various problems in machine learning, such as detecting question duplicates, face recognition through a comparison of the similarity of the inputs by comparing their feature
vectors.
First, we need to have a dataset that contains 3 images, of which 2 are similar and 1 is
different. These images are referred to as _anchor_, _positive_, and _negative_ images,
respectively. The neural network will need to know that the anchor and the positive images
are similar, while the anchor and the negative images are dissimilar — this can be done with the help of the [triplet loss](https://www.tensorflow.org/addons/tutorials/losses_triplet)
(you can find out more in the FaceNet paper by [Schroff et al., 2015](https://arxiv.org/pdf/1503.03832.pdf)).
The triplet loss function is measured as follows:

L(Anchor, Positive, Negative) = max((distance(f(Anchor), f(Positive)) -
distance(f(Anchor), f(Negative)))**2, 0.0)

Note that the weights of the network are shared. Therefore, we are going to use only one 
model for training and inference.
In this example, we will use the [Totally Looks Like](https://sites.google.com/view/totally-looks-like-dataset)
by [Rosenfeld et al., 2018](https://arxiv.org/pdf/1803.01485v3.pdf).
Image from:
https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
![1_0E9104t29iMBmtvq7G1G6Q.png](attachment:1_0E9104t29iMBmtvq7G1G6Q.png)
"""

"""
# Setup
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses, optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet import preprocess_input


target_shape = (200, 200)


"""
# Load the dataset

We are going to load the Totally Looks Like dataset and unzip it inside the `~/.keras` directory
in the local environment.

The dataset consists on two separate files:
* `left.zip` contains the images that we will use as the anchor.
* `right.zip` contains the images that we will use as the positive sample (an image that looks like the anchor.)
"""

cache_dir = Path(Path.home()) / ".keras"
anchor_images_path = cache_dir / "left"
positive_images_path = cache_dir / "right"

"""shell
gdown --id 1jvkbTr_giSP3Ru8OwGNCg6B4PvVbcO34
gdown --id 1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW
unzip -oq left.zip -d $cache_dir
unzip -oq right.zip -d $cache_dir
"""

"""
# Preparing the data

We are going to use a `tf.data` pipeline to load the data and generate the triplets that we
need to train the Siamese network.

We'll set up the pipeline using a zipped list with anchor, positive, and negative filenames as
the source. The pipeline will load and preprocess the corresponding images.
"""


def preprocess_image(filename):
    """
    Loads the specified file as a JPEG image, preprocess it and 
    resizes it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, it loads and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


"""
Let's setup our data pipeline using a zipped list with an anchor, positive,
and negative image filename as the source. The output of the pipeline
contains the same triplet with every image loaded and preprocessed.
"""

# We need to make sure both the anchor and positive images are loaded in
# sorted order so we can match them together.
anchor_images = sorted(
    [str(anchor_images_path / f) for f in os.listdir(anchor_images_path)]
)

positive_images = sorted(
    [str(positive_images_path / f) for f in os.listdir(positive_images_path)]
)

image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

# To generate the list of negative images, let's randomize the list of
# available images and concatenate them together.
rng = np.random.RandomState(seed=42)
rng.shuffle(anchor_images)
rng.shuffle(positive_images)

negative_images = anchor_images + positive_images
np.random.RandomState(seed=32).shuffle(negative_images)

negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


"""
Let's take a look at a few examples of triplets. Notice how the first two images
look alike while the third is always different.
"""


def visualize(anchor, positive, negative):
    """
    Visualizes a few triplets from the supplied batches.
    """

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])


visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])

"""
# Setting up the Embedding generator model

Our Siamese Network will generate embeddings for each one of the images of the
triplet. To do this, we will use a pre-trained ResNet50 model on ImageNet and
connect a few `Dense` layers to it so we have space to learn to separate these
embeddings.

We will freeze the weights of all the layers of the model up until
`conv5_block1_out`. This is important so we don't mess with the weights that
the model already learned. We are going to leave the bottom few layers open so
we can fine tune those weights during training.
"""

base_cnn = applications.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

"""
# Setting up the Siamese Network model

The Siamese network will receive each one of the triplet images as an input,
generate the embeddings, and output the distance between the anchor and the
positive embedding, and the distance between the anchor and the negative
embedding.

To compute the distance, we can use a custom layer `DistanceLayer` that
returns both values as a tuple.
"""


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self):
        super().__init__()

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(preprocess_input(anchor_input)),
    embedding(preprocess_input(positive_input)),
    embedding(preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances,
)

"""
# Putting everything together

We now need to implement a model with custom training loop so we can compute
the Triplet Loss using the three embeddings produced by the Siamese network.

Here is the definition of Triplet Loss implemented on this example:
`L(A,P,N) = max(||f(A)-f(P)||**2 - ||f(A)-f(N)||**2 + alpha, 0)`
"""

"""
Let's create a `Mean` metric instance to track the loss of the training process.
"""

loss_tracker = metrics.Mean(name="loss")
val_loss_tracker = metrics.Mean(name="val_loss")


class SiameseModel(Model):
    """
    The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
        L(A,P,N) = max(||f(A)-f(P)||**2 - ||f(A)-f(N)||**2 + margin, 0)

    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin

    def call(self, inputs):
        self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the validation loss metric.
        val_loss_tracker.update_state(loss)
        return {"loss": val_loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)

        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [loss_tracker, val_loss_tracker]


"""
# Training

We are now ready to train our model.
"""

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

"""
# Looking at what the network learned

At this point, we can check how the network learned to separate the embeddings
depending on whether they belong to similar images.

We can use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to measure the
similarity between embeddings.
"""

"""
Let's pick a sample from the dataset to check the similarity between the
embeddings generated for each image.
"""
sample = next(iter(train_dataset))
visualize(*sample)

anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(preprocess_input(anchor)),
    embedding(preprocess_input(positive)),
    embedding(preprocess_input(negative)),
)

"""
Finally, we can compute the cosine similarity between the anchor and positive
images and compare it with the similarity between the anchor and the negative
images.

We should expect the similarity between the anchor and positive images to be
larger than the similarity between the anchor and the negative images.
"""

cosine_similarity = metrics.CosineSimilarity()

positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
print("Positive similarity:", positive_similarity.numpy())

negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
print("Negative similarity", negative_similarity.numpy())


"""
TODO: Still need to work on this.

### Summary

1) The `tf.data` API enables you to build efficient input pipelines for your model. It is 
particularly useful if you have large dataset. You can learn more in the
[`tf.data` guide](https://www.tensorflow.org/guide/data).

2) You can use `tf.data.Dataset` to create your dataset — it enables you to make a sequence
of operations like shuffling your data or applying transformations to preprocess the data.

3) In this example, [transfer learning](https://www.tensorflow.org/guide/keras/transfer_learning?hl=en) is used to avoid re-training or re-writing large architectures. [Learn more](https://www.tensorflow.org/guide/keras/transfer_learning?hl=en).

4) In Keras, layers can have names, which can be used to retrieve layers. This can be helpful
during [fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning?hl=en#fine-tuning). 
In our example, we loop over the ResNet50 layers before a specific layer, and then we train 
the model on the last few layers.

5) We can create custom layers by creating a class that inherits from `tf.keras.layers.Layer`,
as we did in the `DistanceLayer` class — we just need to implement the `call()` method.

6) We used cosine similarity metric to measure how to 2 output embeddings are similar to each other.

7) Overriding the `train_step()` method allows you to have a custom training loop. 
`train_step()` uses [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape),
which records every operation that you perform inside it. We use it to access the gradients
that are passed to the optimizer to update the model weights at every step.
For more details, check out the [Intro to Keras for researchers](https://keras.io/getting_started/intro_to_keras_for_researchers/)
and
[Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch?hl=en).

For more info about GradientTape check out
https://keras.io/getting_started/intro_to_keras_for_researchers/ and
https://www.tensorflow.org/api_docs/python/tf/GradientTape
"""

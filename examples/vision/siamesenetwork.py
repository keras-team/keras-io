"""
Title: Siamese Network with Triplet Loss
Author: [Hazem Essam](https://twitter.com/hazemessamm) and [Santiago L. Valdarrama](https://twitter.com/svpino)
Date created: 2021/03/13
Last modified: 2021/03/22
Description: Siamese network with custom data generator and training loop.
"""


"""
TODO: Still need to work on this.

# Introduction

[Siamese Network](https://en.wikipedia.org/wiki/Siamese_neural_network) is used to solve
many problems like detecting question duplicates, face recognition by comparing the
similarity of the inputs by comparing their feature vectors.
First we need to have a dataset that contains 3 Images, 2 are similar and 1 is different,
they are called Anchor image, Positive Image and Negative image respectively, we need to
tell the network that the anchor image and the positive image are similar, we also need
to tell it that the anchor image and the negative image are NOT similar, we can do that
by the Triplet Loss Function.
Triplet Loss function:
L(Anchor, Positive, Negative) = max((distance(f(Anchor), f(Positive)) -
distance(f(Anchor), f(Negative)))**2, 0.0)
Note that the weights are shared which mean that we are only using one model for
prediction and training
You can find the dataset here:
https://drive.google.com/drive/folders/1qQJHA5m-vLMAkBfWEWgGW9n61gC_orHl
Also more info found here: https://sites.google.com/view/totally-looks-like-dataset
Image from:
https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
![1_0E9104t29iMBmtvq7G1G6Q.png](attachment:1_0E9104t29iMBmtvq7G1G6Q.png)
"""

"""
### Setup
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


target_shape = (200, 200)


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


"""
# Load the dataset

We are going to use the [Totally Looks Like dataset](https://drive.google.com/drive/folders/1qQJHA5m-vLMAkBfWEWgGW9n61gC_orHl). We are going to download it and unzip it inside the `~/.keras` directory.

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
## Preparing the data

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
Let's setup our pipeline using a zipped list with anchor and positive
filenames as the source. The output of our pipeline contains a triplet with
the anchor, the positive, and the negative image.
"""

anchor_dataset = tf.data.Dataset.list_files(
    file_pattern=str(anchor_images_path / "*.jpg"), shuffle=False
)
positive_dataset = tf.data.Dataset.list_files(
    file_pattern=str(positive_images_path / "*.jpg"), shuffle=False
)

# The negative sample is a randomly selected image from either the anchor or
# the positive list of images. To ensure that we select a random image, we need
# to shuffle the dataset.
negative_dataset = anchor_dataset.concatenate(positive_dataset)
negative_dataset = negative_dataset.shuffle(buffer_size=20000)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.map(preprocess_triplets)
dataset = dataset.batch(32, drop_remainder=False)
dataset = dataset.prefetch(1)

"""
Let's take a look at a few examples of triplets. Notice how the first two images
look alike while the third is always different.
"""

visualize(*list(dataset.take(1).as_numpy_iterator())[0])

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


anchor_input = layers.Input(shape=target_shape + (3,))
positive_input = layers.Input(shape=target_shape + (3,))
negative_input = layers.Input(shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(anchor_input), embedding(positive_input), embedding(negative_input)
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


class SiameseModel(Model):
    """
    Model implementing a custom training loop to compute the Triplet Loss
    using the three embeddings produced by the Siamese network.

    Here is the definition of Triplet Loss:
        L(A,P,N) = max(||f(A)-f(P)||**2 - ||f(A)-f(N)||**2 + alpha, 0)

    """

    def __init__(self, siamese_network, alpha=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.alpha = alpha

    def call(self, inputs):
        self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that you do inside.
        # We are using it here to compute the loss so we can get the gradients and apply
        # them using the optimizer specified in `compile()`.
        with tf.GradientTape() as tape:
            anchor, positive, negative = data

            # The output of the network is a tuple containing the distances
            # between the anchor and the positive example, and the anchor and
            # the negative example.
            ap_distance, an_distance = self.siamese_network(
                (anchor, positive, negative)
            )

            # Computing the Triplet Loss by subtracting both distances and
            # making sure we don't get a negative value.
            loss = ap_distance - an_distance
            loss = tf.maximum(loss + self.alpha, 0.0)

        # Let's get the gradients (loss with respect to trainable weights)
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the loss metric.
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metric here so the `reset_states()` can be
        # called automatically.
        return [loss_tracker]


"""
# Training

We are now ready to train our model.
"""

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(dataset, epochs=20)

"""
# Looking at what the network learned

At this point, we can check how the network learned to separate the embeddings
depending on whether they belong to similar images.

We can use Cosine Similarity to measure the similarity between embeddings.
"""

"""
Let's pick a sample from the dataset to check the similarity between the
embeddings generated for each image.
"""
sample = next(iter(dataset))
visualize(*sample)

"""
.
"""

anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(anchor),
    embedding(positive),
    embedding(negative),
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

### Key Takeaways

1) You can create your custom data generator by creating a class that inherits from
tf.keras.utils.Sequence, as we saw, this is really helpful if we want to generate data in
different forms like Anchor, Positive and negative in our case, you just need to
implement the __len__() and __getitem__(). Check out the documentation
https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

2) If you don't have the computation power to train large models like ResNet-50 or if you
don't have the time to re-write really big models you can just download it with one line,
e.g. tf.keras.applications.ResNet50().

3) Every layer has a name, this is really helpful for fine tuning, If you want to fine
tune specific layers, in our example we loop over the layers until we find specific layer
by it's name and we made it trainable, this allows the weights of this layer to change
during training

4) In our example we have only one embedding network that we need to train it but we need
3 outputs to compare them with each other, we can do that by creating a model that have 3
input layers and each input will pass through the embedding network and then we will have
3 outputs embeddings, we did that in the "Model for Training section".

5) You can name your output layers like we did in the "Model for Training section", you
just need to create a dictionary with keys as the name of your output layer and the
output layers as values.

6) We used cosine similarity to measure how to 2 output embeddings are similar to each
other.

6) You can create your custom Layers by just creating a class that inherits from
tf.keras.layers.Layer, you just need to implement the call function. check out the
documentation https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

7) If you want to have custom training loop you can create your own model class that
inherits from tf.keras.Model, you just need to override the train_step function and add
you implementation.

8) You can get your model gradients by using tf.GradientTape(), GradientTape records the
operations that you do inside it, so you can get the predictions and write your custom
loss function inside the GradientTape as we did.

9) You can get the gradients using tape.gradient(loss, model.trainable_weights), this
means that we need the gradients of the loss with respect to the model trainable weights,
where "tape" is the name of our tf.GradientTape() in our example.

10) you can just pass the gradients and the model weights to the optimizer to update them.

For more info about GradientTape check out
https://keras.io/getting_started/intro_to_keras_for_researchers/ and
https://www.tensorflow.org/api_docs/python/tf/GradientTape
"""

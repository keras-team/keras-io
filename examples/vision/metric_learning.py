"""
Title: Metric learning using cross entropy.
Author: [Mat Kelcey](https://twitter.com/mat_kelcey)
Date created: 2020/06/05
Last modified: 2020/06/05
Description: Example of using metric learning using cross entropy on synthetic data.
"""
"""
## Overview

Metric learning aims to train models that can embed inputs into a high dimensional space
such that "similar" inputs, as defined by the training scheme, are located close to each
other. These models once trained can produce embeddings for downstream systems where such
similarity is useful; examples include as a ranking signal for search or as a form of
pretrained embedding model for another supervised problem.

For a more detailed overview of metric learning see..

* [What is metric
learning?](http://contrib.scikit-learn.org/metric-learn/introduction.html)
* ["Using cross entropy for metric learning"
tutorial](https://www.youtube.com/watch?v=Jb4Ewl5RzkI)
"""

"""
## Setup
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

"""
## Dataset

For this example we will be using randomly generated coloured boxes. Toy data such as
this has the benefit of being easy to generate in a controlled way while being simple
enough that we can quickly demonstrate concepts.
"""

# Height / Width of all images generated.
HW = 32


def random_colour():
    # Generate a random (red, green, blue) tuple,
    return np.random.random(size=3)


def random_instance(colour):
    # Start with a black background.
    img = np.zeros((HW, HW, 3), dtype=np.float32)
    while True:
        # Pick box extents, ensuring they are ordered.
        x1, x2, y1, y2 = np.random.randint(0, HW, 4)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        # Ensure box isn't too small or too big.
        area = (x2 - x1) * (y2 - y1)
        if area > 50 and area < 300:
            img[x1:x2, y1:y2] = colour
            return img


"""
We can visualise a grid of 25 randomly generated examples to get a sense of these images.


"""


def show_collage(examples):
    HWB = HW + 2  # Box height / width plus a 2 pixel buffer.
    n_rows, n_cols = examples.shape[:2]

    collage = Image.new(
        mode="RGB", size=(n_cols * HWB, n_rows * HWB), color=(250, 250, 250)
    )
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            array = (np.array(examples[row_idx, col_idx]) * 255).astype(np.uint8)
            collage.paste(Image.fromarray(array), (col_idx * HWB, row_idx * HWB))

    # Double size for visualisation.
    collage = collage.resize((2 * n_cols * HWB, 2 * n_rows * HWB))
    return collage


# Show a collage of 5x5 random images.
examples = [random_instance(random_colour()) for _ in range(25)]
examples = np.stack(examples).reshape((5, 5, HW, HW, 3))
show_collage(examples)

"""
Metric learning provides training data not as explicit `(X, y)` pairs but instead uses
multiple instances that are related in the way we want to express similarity.

In our example the similarity we want to capture will be colour; a single training
instance will not one image, but a pair of images of the same colour.
Note that we could use other descriptions of similarity, for example the area of the
boxes, but we use colour for this minimal version since it is something that is very
quick to train with a small model.

When referring to the images in this pair we'll use the common metric learning names of
the `anchor` (a randomly chosen image) and the `positive` (another randomly chosen image
of the same colour).

"""


def build_dataset(batch_size, num_batchs):
    def anchor_positive_pair():
        for _ in range(batch_size * num_batchs):
            colour = random_colour()
            yield random_instance(colour), random_instance(colour)

    # Exclude optimisations such as prefetching for this simple example.
    return tf.data.Dataset.from_generator(
        anchor_positive_pair, output_types=(tf.float32, tf.float32)
    ).batch(batch_size)


"""
We can visualise a batch of this data in another collage. The top row shows six randomly
chosen anchors, the bottom row shows the corresponding six positives.
"""

for anchors, positives in build_dataset(batch_size=6, num_batchs=1):
    pass

examples = np.stack([anchors, positives])
show_collage(examples)

"""
## Embedding model

Next we define an image embedding model. This model is a minimal sequence of 2d
convolutions followed by global pooling with a final linear projection to an embedding
space. As is common in metric learning we normalise the embeddings so that we can use
simple dot products to measure similarity.
"""

E = 4  # output embedding size.


class NormaliseLayer(Layer):
    def call(self, x):
        return tf.nn.l2_normalize(x, axis=-1)


def construct_model():
    input = Input(shape=(HW, HW, 3))
    model = Conv2D(filters=4, kernel_size=3, strides=2, activation="relu")(input)
    model = Conv2D(filters=8, kernel_size=3, strides=2, activation="relu")(model)
    model = Conv2D(filters=16, kernel_size=3, strides=2, activation="relu")(model)
    model = GlobalAveragePooling2D()(model)
    embeddings = Dense(units=E, activation=None, name="embedding")(model)
    embeddings = NormaliseLayer()(embeddings)
    return Model(input, embeddings)


model = construct_model()

"""
## Training loop

We train this model with a custom training loop that first embeds both anchors and
positives and then uses their pairwise dot products as logits for a softmax. On a Google
Colab CPU instance this step takes approximately 10 seconds.
"""

optimiser = Adam(learning_rate=1e-3)

losses = []
for anchors, positives in build_dataset(batch_size=32, num_batchs=300):

    with tf.GradientTape() as tape:
        # Run anchors and positives through model.
        anchor_embeddings = model(anchors)
        positive_embeddings = model(positives)

        # Calculate cosine similarity between anchors and positives. As they have
        # be normalised this is just the pair wise dot products.
        similarities = tf.einsum("ae,pe->ap", anchor_embeddings, positive_embeddings)

        # Since we intend to use these as logits we scale them by a temperature.
        # This value would normally be chosen as a hyper parameter.
        temperature = 0.2
        similarities /= temperature

        # We use these similarities as logits for a softmax. The labels for
        # this call are just the sequence 0, 1, 2, ... since we want the main
        # diagonal values, which correspond to the anchor/positive pairs, to be
        # high. This loss will move embeddings for the anchor/positive pairs
        # together and move all other pairs apart.
        labels = tf.range(similarities.shape[0])
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=similarities
            )
        )
        losses.append(loss)

        # Calculate gradients and apply via optimiser.
        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))

plt.plot(losses)
plt.show()

"""
## Testing

As a final step we can generate 100 random images across 10 random colours.
"""

random_colours = [random_colour() for _ in range(10)]

random_dataset = []
for _ in range(10):
    for colour in random_colours:
        random_dataset.append(random_instance(colour))
random_dataset = np.stack(random_dataset)

"""
When we run these images through the model we can see that for randomly chosen anchors
the near neighbours in the embedding space are the other images with the same colour.

Below shows 5 examples; the first column is a randomly selected image with the following
10 columns showing the nearest neighbours in order of similarity.
"""

EGS = 5  # Show 5 example images ...
NNS_PER_EG = 10  # .. with their 10 nearest neighbours

embeddings = model(random_dataset).numpy()
gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
near_neighbours = np.argsort(gram_matrix.T)[:, -(NNS_PER_EG + 1) :]

examples = np.empty((EGS, NNS_PER_EG + 1, HW, HW, 3), dtype=np.float32)
for r_idx in range(EGS):
    examples[r_idx, 0] = random_dataset[r_idx]
    anchor_near_neighbours = reversed(near_neighbours[r_idx][:-1])
    for c_idx, nn_idx in enumerate(anchor_near_neighbours):
        examples[r_idx, c_idx + 1] = random_dataset[nn_idx]

show_collage(examples)

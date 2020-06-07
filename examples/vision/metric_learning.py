"""
Title: Metric learning using crossentropy.
Author: [Mat Kelcey](https://twitter.com/mat_kelcey)
Date created: 2020/06/05
Last modified: 2020/06/05
Description: Example of using metric learning using crossentropy on synthetic data.
"""
"""
## Overview

Metric learning aims to train models that can embed inputs into a high-dimensional space
such that "similar" inputs, as defined by the training scheme, are located close to each
other. These models once trained can produce embeddings for downstream systems where such
similarity is useful; examples include as a ranking signal for search or as a form of
pretrained embedding model for another supervised problem.

For a more detailed overview of metric learning see:

* [What is metric learning?](http://contrib.scikit-learn.org/metric-learn/introduction.html)
* ["Using crossentropy for metric learning" tutorial](https://www.youtube.com/watch?v=Jb4Ewl5RzkI)
"""

"""
## Setup
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

"""
## Dataset

For this example we will be using randomly generated coloured boxes. Toy data such as
this has the benefit of being easy to generate in a controlled way while being simple
enough that we can quickly demonstrate concepts.
"""

# Height / Width of all images generated.
height_width = 32


def random_colour():
    # Generate a random (red, green, blue) tuple,
    return np.random.random(size=3)


def random_instance(colour):
    # Start with a black background.
    img = np.zeros((height_width, height_width, 3), dtype=np.float32)
    while True:
        # Pick box extents, ensuring they are ordered.
        x1, x2, y1, y2 = np.random.randint(0, height_width, 4)
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
    box_size = height_width + 2
    num_rows, num_cols = examples.shape[:2]

    collage = Image.new(
        mode="RGB",
        size=(num_cols * box_size, num_rows * box_size),
        color=(250, 250, 250),
    )
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            array = (np.array(examples[row_idx, col_idx]) * 255).astype(np.uint8)
            collage.paste(
                Image.fromarray(array), (col_idx * box_size, row_idx * box_size)
            )

    # Double size for visualisation.
    collage = collage.resize((2 * num_cols * box_size, 2 * num_rows * box_size))
    return collage


# Show a collage of 5x5 random images.
examples = [random_instance(random_colour()) for _ in range(25)]
examples = np.stack(examples).reshape((5, 5, height_width, height_width, 3))
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

embedding_size = 4


def construct_model():
    inputs = layers.Input(shape=(height_width, height_width, 3))
    x = layers.Conv2D(filters=4, kernel_size=3, strides=2, activation="relu")(inputs)
    x = layers.Conv2D(filters=8, kernel_size=3, strides=2, activation="relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=2, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    embeddings = layers.Dense(units=embedding_size, activation=None)(x)
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
    return keras.models.Model(inputs, embeddings)


model = construct_model()

"""
## Training loop

We train this model with a custom training loop that first embeds both anchors and
positives and then uses their pairwise dot products as logits for a softmax. On a Google
Colab CPU instance this step takes approximately 10 seconds.
"""

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

batch_losses = []
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
        loss = loss_fn(labels, similarities)
        batch_losses.append(loss)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

plt.plot(batch_losses)
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

num_collage_examples = 5
near_neighbours_per_example = 10

embeddings = model(random_dataset).numpy()
gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]

examples = np.empty(
    (
        num_collage_examples,
        near_neighbours_per_example + 1,
        height_width,
        height_width,
        3,
    ),
    dtype=np.float32,
)
for row_idx in range(num_collage_examples):
    examples[row_idx, 0] = random_dataset[row_idx]
    anchor_near_neighbours = reversed(near_neighbours[row_idx][:-1])
    for col_idx, nn_idx in enumerate(anchor_near_neighbours):
        examples[row_idx, col_idx + 1] = random_dataset[nn_idx]

show_collage(examples)

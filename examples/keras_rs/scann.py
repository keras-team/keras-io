"""
Title: Faster retrieval with Scalable Nearest Neighbours (ScANN)
Author: [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)
Date created: 2025/04/28
Last modified: 2025/04/28
Description: Using ScANN for faster retrieval.
Accelerator: GPU
"""

"""
# Faster retrieval with Scalable Nearest Neighbours (ScANN)

Retrieval models are designed to quickly identify a small set of highly relevant
candidates from vast pools of data, often comprising millions or even hundreds
of millions of items. To effectively respond to the user's context and behavior
in real time, these models must perform this task in just milliseconds.

Approximate nearest neighbor (ANN) search is the key technology that enables
this level of efficiency. In this tutorial, we'll demonstrate how to leverage
ScANN—a cutting-edge nearest neighbor retrieval library—to effortlessly scale
retrieval for millions of items.

[ScANN](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/),
developed by Google Research, is a high-performance library designed for
dense vector similarity search at scale. It efficiently indexes a database of
candidate embeddings, enabling rapid search during inference. By leveraging
advanced vector compression techniques and finely tuned algorithms, ScaNN
strikes an optimal balance between speed and accuracy. As a result, it can
significantly outperform brute-force search methods, delivering fast retrieval
with minimal loss in accuracy.

We will start with the same code as the
[basic retrieval example](https://keras.io/keras_rs/examples/basic_retrieval/).
Data processing, model building, and training remain exactly the same. Feel free
to skip this part if you have gone over the basic retrieval example before.

Note: ScANN does not have its own separate layer in KerasRS because the ScANN
library is TensorFlow-only. Here, in this example, we directly use the ScANN
library and demonstrate its usage with KerasRS.

## Imports

Let's install the `scann` library and import all necessary packages. We will
also set the backend to JAX.
"""

# ruff: noqa: E402

"""shell
pip install -q scann
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # `"tensorflow"`/`"torch"`

import time
import uuid

import keras
import tensorflow as tf  # Needed for the dataset
import tensorflow_datasets as tfds
from scann import scann_ops

import keras_rs

"""
## Preparing the dataset
"""

# Ratings data with user and movie data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

# Get user and movie counts so that we can define embedding layers for both.
users_count = (
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy()
)

movies_count = movies.cardinality().numpy()


# Preprocess the dataset, by selecting only the relevant columns.
def preprocess_rating(x):
    return (
        # Input is the user IDs
        tf.strings.to_number(x["user_id"], out_type=tf.int32),
        # Labels are movie IDs + ratings between 0 and 1.
        {
            "movie_id": tf.strings.to_number(x["movie_id"], out_type=tf.int32),
            "rating": (x["user_rating"] - 1.0) / 4.0,
        },
    )


shuffled_ratings = ratings.map(preprocess_rating).shuffle(
    100_000, seed=42, reshuffle_each_iteration=False
)
# Train-test split.
train_ratings = shuffled_ratings.take(80_000).batch(1000).cache()
test_ratings = shuffled_ratings.skip(80_000).take(20_000).batch(1000).cache()

"""
## Implementing the Model
"""


class RetrievalModel(keras.Model):
    def __init__(
        self,
        num_users,
        num_candidates,
        embedding_dimension=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Our query tower, simply an embedding table.
        self.user_embedding = keras.layers.Embedding(num_users, embedding_dimension)
        # Our candidate tower, simply an embedding table.
        self.candidate_embedding = keras.layers.Embedding(
            num_candidates, embedding_dimension
        )

        self.loss_fn = keras.losses.MeanSquaredError()

    def build(self, input_shape):
        self.user_embedding.build(input_shape)
        self.candidate_embedding.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training=False):
        user_embeddings = self.user_embedding(inputs)
        result = {
            "user_embeddings": user_embeddings,
        }
        return result

    def compute_loss(self, x, y, y_pred, sample_weight, training=True):
        candidate_id, rating = y["movie_id"], y["rating"]
        user_embeddings = y_pred["user_embeddings"]
        candidate_embeddings = self.candidate_embedding(candidate_id)

        labels = keras.ops.expand_dims(rating, -1)
        # Compute the affinity score by multiplying the two embeddings.
        scores = keras.ops.sum(
            keras.ops.multiply(user_embeddings, candidate_embeddings),
            axis=1,
            keepdims=True,
        )
        return self.loss_fn(labels, scores, sample_weight)


"""
## Training the model
"""

model = RetrievalModel(users_count + 1000, movies_count + 1000)
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))

history = model.fit(
    train_ratings, validation_data=test_ratings, validation_freq=5, epochs=50
)

"""
## Making predictions

Before we try out ScANN, let's go with the brute force method, i.e., for a given
user, scores are computed for all movies, sorted and then the top-k
movies are picked. This is, of course, not very scalable when we have a huge
number of movies.
"""

candidate_embeddings = keras.ops.array(model.candidate_embedding.embeddings.numpy())
# Artificially duplicate candidate embeddings to simulate a large number of
# movies.
candidate_embeddings = keras.ops.concatenate(
    [candidate_embeddings]
    + [
        candidate_embeddings
        * keras.random.uniform(keras.ops.shape(candidate_embeddings))
        for _ in range(100)
    ],
    axis=0,
)

user_embedding = model.user_embedding(keras.ops.array([10, 5, 42, 345]))

# Define the brute force retrieval layer.
brute_force_layer = keras_rs.layers.BruteForceRetrieval(
    candidate_embeddings=candidate_embeddings,
    k=10,
    return_scores=False,
)

"""
Now, let's do a forward pass on the layer. Note that in previous tutorials, we
have the above layer as an attribute of the model class, and we then call
`.predict()`. This will obviously be faster (since it's compiled XLA code), but
since we cannot do the same for ScANN, we just do a normal forward pass here
without compilation to ensure a fair comparison.
"""

t0 = time.time()
pred_movie_ids = brute_force_layer(user_embedding)
print("Time taken by brute force layer (sec):", time.time() - t0)

"""
Now, let's retrieve movies using ScANN. We will use the ScANN library from
Google Research to build the layer and then call it. To fully understand all the
arguments, please refer to the
[ScANN README file](https://github.com/google-research/google-research/tree/master/scann#readme).
"""


def build_scann(
    candidates,
    k=10,
    distance_measure="dot_product",
    dimensions_per_block=2,
    num_reordering_candidates=500,
    num_leaves=100,
    num_leaves_to_search=30,
    training_iterations=12,
):
    builder = scann_ops.builder(
        db=candidates,
        num_neighbors=k,
        distance_measure=distance_measure,
    )

    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=num_leaves_to_search,
        training_iterations=training_iterations,
    )
    builder = builder.score_ah(dimensions_per_block=dimensions_per_block)

    if num_reordering_candidates is not None:
        builder = builder.reorder(num_reordering_candidates)

    # Set a unique name to prevent unintentional sharing between
    # ScaNN instances.
    searcher = builder.build(shared_name=str(uuid.uuid4()))
    return searcher


def run_scann(searcher):
    pred_movie_ids = searcher.search_batched_parallel(
        user_embedding,
        final_num_neighbors=10,
    ).indices
    return pred_movie_ids


searcher = build_scann(candidates=candidate_embeddings)

t0 = time.time()
pred_movie_ids = run_scann(searcher)
print("Time taken by ScANN (sec):", time.time() - t0)

"""
You can clearly see the performance improvement in terms of latency. ScANN
(0.003 seconds) takes one-fiftieth the time it takes for the brute force layer
(0.15 seconds) to run!
"""

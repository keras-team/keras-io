"""
Title: DistributedEmbedding using TPU SparseCore and JAX
Author: [Fabien Hertschuh](https://github.com/hertschuh/), [Abheesht Sharma](https://github.com/abheesht17/), [C. Antonio Sánchez](https://github.com/cantonios/)
Date created: 2025/06/03
Last modified: 2025/09/02
Description: Rank movies using a two tower model with embeddings on SparseCore.
Accelerator: TPU
"""

"""
## Introduction

In the [basic ranking](/keras_rs/examples/basic_ranking/) tutorial, we showed
how to build a ranking model for the MovieLens dataset to suggest movies to
users.

This tutorial implements the same model trained on the same dataset but with the
use of `keras_rs.layers.DistributedEmbedding`, which makes use of SparseCore on
TPU. This is the JAX version of the tutorial. It needs to be run on TPU v5p or
v6e.

Let's begin by choosing JAX as the backend and importing all the necessary
libraries.
"""

"""shell
pip install -q -U jax[tpu]>=0.7.0
pip install -q jax-tpu-embedding
pip install -q tensorflow-cpu
pip install -q keras-rs
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import keras_rs
import tensorflow as tf  # Needed for the dataset
import tensorflow_datasets as tfds

"""
## Dataset distribution

While the model is replicated and the embedding tables are sharded across
SparseCores, the dataset is distributed by sharding each batch across the TPUs.
We need to make sure the batch size is a multiple of the number of TPUs.
"""

PER_REPLICA_BATCH_SIZE = 256
BATCH_SIZE = PER_REPLICA_BATCH_SIZE * jax.local_device_count("tpu")

distribution = keras.distribution.DataParallel(devices=jax.devices("tpu"))
keras.distribution.set_distribution(distribution)

"""
## Preparing the dataset

We're going to use the same MovieLens data. The ratings are the objectives we
are trying to predict.
"""

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

"""
We need to know the number of users as we're using the user ID directly as an
index in the user embedding table.
"""

users_count = int(
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy()
)

"""
We also need do know the number of movies as we're using the movie ID directly
as an index in the movie embedding table.
"""

movies_count = int(movies.cardinality().numpy())

"""
The inputs to the model are the user IDs and movie IDs and the labels are the
ratings.
"""


def preprocess_rating(x):
    return (
        # Inputs are user IDs and movie IDs
        {
            "user_id": tf.strings.to_number(x["user_id"], out_type=tf.int32),
            "movie_id": tf.strings.to_number(x["movie_id"], out_type=tf.int32),
        },
        # Labels are ratings between 0 and 1.
        (x["user_rating"] - 1.0) / 4.0,
    )


"""
We'll split the data by putting 80% of the ratings in the train set, and 20% in
the test set.
"""

shuffled_ratings = ratings.map(preprocess_rating).shuffle(
    100_000, seed=42, reshuffle_each_iteration=False
)
train_ratings = (
    shuffled_ratings.take(80_000).batch(BATCH_SIZE, drop_remainder=True).cache()
)
test_ratings = (
    shuffled_ratings.skip(80_000)
    .take(20_000)
    .batch(BATCH_SIZE, drop_remainder=True)
    .cache()
)

"""
## Configuring DistributedEmbedding

The `keras_rs.layers.DistributedEmbedding` handles multiple features and
multiple embedding tables. This is to enable the sharing of tables between
features and allow some optimizations that come from combining multiple
embedding lookups into a single invocation. In this section, we'll describe
how to configure these.

### Configuring tables

Tables are configured using `keras_rs.layers.TableConfig`, which has:

- A name.
- A vocabulary size (input size).
- an embedding dimension (output size).
- A combiner to specify how to reduce multiple embeddings into a single one in
  the case when we embed a sequence. Note that this doesn't apply to our example
  because we're getting a single embedding for each user and each movie.
- A placement to tell whether to put the table on the SparseCore chips or not.
  In this case, we want the `"sparsecore"` placement.
- An optimizer to specify how to apply gradients when training. Each table has
  its own optimizer and the one passed to `model.compile()` is not used for the
  embedding tables.

### Configuring features

Features are configured using `keras_rs.layers.FeatureConfig`, which has:

- A name.
- A table, the embedding table to use.
- An input shape (batch size is for all TPUs).
- An output shape (batch size is for all TPUs).

We can organize features in any structure we want, which can be nested. A dict
is often a good choice to have names for the inputs and outputs.
"""

EMBEDDING_DIMENSION = 32

movie_table = keras_rs.layers.TableConfig(
    name="movie_table",
    vocabulary_size=movies_count + 1,  # +1 for movie ID 0, which is not used
    embedding_dim=EMBEDDING_DIMENSION,
    optimizer="adam",
    placement="sparsecore",
)
user_table = keras_rs.layers.TableConfig(
    name="user_table",
    vocabulary_size=users_count + 1,  # +1 for user ID 0, which is not used
    embedding_dim=EMBEDDING_DIMENSION,
    optimizer="adam",
    placement="sparsecore",
)

FEATURE_CONFIGS = {
    "movie_id": keras_rs.layers.FeatureConfig(
        name="movie",
        table=movie_table,
        input_shape=(BATCH_SIZE,),
        output_shape=(BATCH_SIZE, EMBEDDING_DIMENSION),
    ),
    "user_id": keras_rs.layers.FeatureConfig(
        name="user",
        table=user_table,
        input_shape=(BATCH_SIZE,),
        output_shape=(BATCH_SIZE, EMBEDDING_DIMENSION),
    ),
}

"""
## Defining the Model

We're now ready to create a `DistributedEmbedding` inside a model. Once we have
the configuration, we simply pass it the constructor of `DistributedEmbedding`.
Then, within the model `call` method, `DistributedEmbedding` is the first layer
we call.

The ouputs have the exact same structure as the inputs. In our example, we
concatenate the embeddings we got as outputs and run them through a tower of
dense layers.
"""


class EmbeddingModel(keras.Model):
    """Create the model with the embedding configuration.

    Args:
        feature_configs: the configuration for `DistributedEmbedding`.
    """

    def __init__(self, feature_configs):
        super().__init__()

        self.embedding_layer = keras_rs.layers.DistributedEmbedding(
            feature_configs=feature_configs
        )
        self.ratings = keras.Sequential(
            [
                # Learn multiple dense layers.
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                # Make rating predictions in the final layer.
                keras.layers.Dense(1),
            ]
        )

    def call(self, preprocessed_features):
        # Embedding lookup. Outputs have the same structure as the inputs.
        embedding = self.embedding_layer(preprocessed_features)
        return self.ratings(
            keras.ops.concatenate(
                [embedding["user_id"], embedding["movie_id"]],
                axis=1,
            )
        )


"""
Let's now instantiate the model. We then use `model.compile()` to configure the
loss, metrics and optimizer. Again, this Adagrad optimizer will only apply to
the dense layers and not the embedding tables.
"""

model = EmbeddingModel(FEATURE_CONFIGS)

model.compile(
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError()],
    optimizer="adagrad",
)

"""
With the JAX backend, we need to preprocess the inputs to convert them to a
hardware-dependent format required for use with SparseCores. We'll do this by
wrapping the datasets into generator functions.
"""


def train_dataset_generator():
    for inputs, labels in iter(train_ratings):
        yield model.embedding_layer.preprocess(inputs, training=True), labels


def test_dataset_generator():
    for inputs, labels in iter(test_ratings):
        yield model.embedding_layer.preprocess(inputs, training=False), labels


"""
## Fitting and evaluating

We can use the standard Keras `model.fit()` to train the model. Keras will
automatically use the `TPUStrategy` to distribute the model and the data.
"""

model.fit(train_dataset_generator(), epochs=5)

"""
Same for `model.evaluate()`.
"""

model.evaluate(test_dataset_generator(), return_dict=True)

"""
That's it.

This example shows that after configuring the `DistributedEmbedding` and setting
up the required preprocessing, you can use the standard Keras workflows.
"""

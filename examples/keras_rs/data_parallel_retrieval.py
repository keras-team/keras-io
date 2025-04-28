"""
Title: Retrieval with data parallel training
Author: [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)
Date created: 2025/04/28
Last modified: 2025/04/28
Description: Retrieve movies using a two tower model (data parallel training).
Accelerator: TPU
"""

"""
# Recommending movies: retrieval with data parallel training

In this tutorial, we are going to train the exact same retrieval model as we
did in our
[basic retrieval](https://keras.io/keras_rs/examples/basic_retrieval/)
tutorial, but in a distributed way.

Distributed training is used to train models on multiple devices or machines
simultaneously, thereby reducing training time. Here, we focus on synchronous
data parallel training. Each accelerator (GPU/TPU) holds a complete replica
of the model, and sees a different mini-batch of the input data. Local gradients
are computed on each device, aggregated and used to compute a global gradient
update.

Before we begin, let's note down a few things:

1. The number of accelerators should be greater than 1.
2. The `keras.distribution` API works only with JAX. So, make sure you select
   JAX as your backend!
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import random

import jax
import keras
import tensorflow as tf  # Needed only for the dataset
import tensorflow_datasets as tfds

import keras_rs

"""
## Data Parallel

For the synchronous data parallelism strategy in distributed training,
we will use the `DataParallel` class present in the `keras.distribution`
API.
"""
devices = jax.devices()  # Assume it has >1 local devices.
data_parallel = keras.distribution.DataParallel(devices=devices)

"""
Alternatively, you can choose to create the `DataParallel` object
using a 1D `DeviceMesh` object, like so:

```
mesh_1d = keras.distribution.DeviceMesh(
    shape=(len(devices),), axis_names=["data"], devices=devices
)
data_parallel = keras.distribution.DataParallel(device_mesh=mesh_1d)
```
"""

# Set the global distribution strategy.
keras.distribution.set_distribution(data_parallel)

"""
## Preparing the dataset

Now that we are done defining the global distribution
strategy, the rest of the guide looks exactly the same
as the previous basic retrieval guide.

Let's load and prepare the dataset. Here too, we use the
MovieLens dataset.
"""

# Ratings data with user and movie data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

# User, movie counts for defining vocabularies.
users_count = (
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy()
)
movies_count = movies.cardinality().numpy()


# Preprocess dataset, and split it into train-test datasets.
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
train_ratings = shuffled_ratings.take(80_000).batch(1000).cache()
test_ratings = shuffled_ratings.skip(80_000).take(20_000).batch(1000).cache()

"""
## Implementing the Model

We build a two-tower retrieval model. Therefore, we need to combine a
query tower for users and a candidate tower for movies. Note that we don't
have to change anything here from the previous basic retrieval tutorial.
"""


class RetrievalModel(keras.Model):
    """Create the retrieval model with the provided parameters.

    Args:
      num_users: Number of entries in the user embedding table.
      num_candidates: Number of entries in the candidate embedding table.
      embedding_dimension: Output dimension for user and movie embedding tables.
    """

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
        # The layer that performs the retrieval.
        self.retrieval = keras_rs.layers.BruteForceRetrieval(k=10, return_scores=False)
        self.loss_fn = keras.losses.MeanSquaredError()

    def build(self, input_shape):
        self.user_embedding.build(input_shape)
        self.candidate_embedding.build(input_shape)
        # In this case, the candidates are directly the movie embeddings.
        # We take a shortcut and directly reuse the variable.
        self.retrieval.candidate_embeddings = self.candidate_embedding.embeddings
        self.retrieval.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        user_embeddings = self.user_embedding(inputs)
        result = {
            "user_embeddings": user_embeddings,
        }
        if not training:
            # Skip the retrieval of top movies during training as the
            # predictions are not used.
            result["predictions"] = self.retrieval(user_embeddings)
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
## Fitting and evaluating

After defining the model, we can use the standard Keras `model.fit()` to train
and evaluate the model.
"""

model = RetrievalModel(users_count + 1, movies_count + 1)
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.2))

"""
Let's train the model. Evaluation takes a bit of time, so we only evaluate the
model every 5 epochs.
"""

history = model.fit(
    train_ratings, validation_data=test_ratings, validation_freq=5, epochs=50
)

"""
## Making predictions

Now that we have a model, let's run inference and make predictions.
"""

movie_id_to_movie_title = {
    int(x["movie_id"]): x["movie_title"] for x in movies.as_numpy_iterator()
}
movie_id_to_movie_title[0] = ""  # Because id 0 is not in the dataset.

"""
We then simply use the Keras `model.predict()` method. Under the hood, it calls
the `BruteForceRetrieval` layer to perform the actual retrieval.
"""

user_ids = random.sample(range(1, 101), len(devices))
predictions = model.predict(keras.ops.convert_to_tensor(user_ids))
predictions = keras.ops.convert_to_numpy(predictions["predictions"])

for user_id in user_ids:
    print(f"\n==Recommended movies for user {user_id}==")
    for movie_id in predictions[0]:
        print(movie_id_to_movie_title[movie_id])

"""
And we're done! For data parallel training, all we had to do was add ~3-5 LoC.
The rest is exactly the same.
"""

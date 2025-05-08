"""
Title: Recommending movies: retrieval
Author: [Fabien Hertschuh](https://github.com/hertschuh/), [Abheesht Sharma](https://github.com/abheesht17/)
Date created: 2025/04/28
Last modified: 2025/04/28
Description: Retrieve movies using a two tower model.
Accelerator: GPU
"""

"""
## Introduction

Recommender systems are often composed of two stages:

1. The retrieval stage is responsible for selecting an initial set of hundreds
   of candidates from all possible candidates. The main objective of this model
   is to efficiently weed out all candidates that the user is not interested in.
   Because the retrieval model may be dealing with millions of candidates, it
   has to be computationally efficient.
2. The ranking stage takes the outputs of the retrieval model and fine-tunes
   them to select the best possible handful of recommendations. Its task is to
   narrow down the set of items the user may be interested in to a shortlist of
   likely candidates.

In this tutorial, we're going to focus on the first stage, retrieval. If you are
interested in the ranking stage, have a look at our
[ranking](/keras_rs/examples/basic_ranking/) tutorial.

Retrieval models are often composed of two sub-models:

1. A query tower computing the query representation (normally a
   fixed-dimensionality embedding vector) using query features.
2. A candidate tower computing the candidate representation (an equally-sized
   vector) using the candidate features. The outputs of the two models are then
   multiplied together to give a query-candidate affinity score, with higher
   scores expressing a better match between the candidate and the query.

In this tutorial, we're going to build and train such a two-tower model using
the Movielens dataset.

We're going to:

1. Get our data and split it into a training and test set.
2. Implement a retrieval model.
3. Fit and evaluate it.
4. Test running predictions with the model.

### The dataset

The Movielens dataset is a classic dataset from the
[GroupLens](https://grouplens.org/datasets/movielens/) research group at the
University of Minnesota. It contains a set of ratings given to movies by a set
of users, and is a standard for recommender systems research.

The data can be treated in two ways:

1. It can be interpreted as expressesing which movies the users watched (and
   rated), and which they did not. This is a form of implicit feedback, where
   users' watches tell us which things they prefer to see and which they'd
   rather not see.
2. It can also be seen as expressesing how much the users liked the movies they
   did watch. This is a form of explicit feedback: given that a user watched a
   movie, we can tell how much they liked by looking at the rating they have
   given.

In this tutorial, we are focusing on a retrieval system: a model that predicts a
set of movies from the catalogue that the user is likely to watch. For this, the
model will try to predict the rating users would give to all the movies in the
catalogue. We will therefore use the explicit rating data.

Let's begin by choosing JAX as the backend we want to run on, and import all
the necessary libraries.
"""

"""shell
pip install -q keras-rs
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # `"tensorflow"`/`"torch"`

import keras
import tensorflow as tf  # Needed for the dataset
import tensorflow_datasets as tfds

import keras_rs

"""
## Preparing the dataset

Let's first have a look at the data.

We use the MovieLens dataset from
[Tensorflow Datasets](https://www.tensorflow.org/datasets). Loading
`movielens/100k_ratings` yields a `tf.data.Dataset` object containing the
ratings alongside user and movie data. Loading `movielens/100k_movies` yields a
`tf.data.Dataset` object containing only the movies data.

Note that since the MovieLens dataset does not have predefined splits, all data
are under `train` split.
"""

# Ratings data with user and movie data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

"""
The ratings dataset returns a dictionary of movie id, user id, the assigned
rating, timestamp, movie information, and user information:
"""

for data in ratings.take(1).as_numpy_iterator():
    print(str(data).replace(", '", ",\n '"))

"""
In the Movielens dataset, user IDs are integers (represented as strings)
starting at 1 and with no gap. Normally, you would need to create a lookup table
to map user IDs to integers from 0 to N-1. But as a simplication, we'll use the
user id directly as an index in our model, in particular to lookup the user
embedding from the user embedding table. So we need do know the number of users.
"""

users_count = (
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy()
)

"""
The movies dataset contains the movie id, movie title, and the genres it belongs
to. Note that the genres are encoded with integer labels.
"""

for data in movies.take(1).as_numpy_iterator():
    print(str(data).replace(", '", ",\n '"))

"""
In the Movielens dataset, movie IDs are integers (represented as strings)
starting at 1 and with no gap. Normally, you would need to create a lookup table
to map movie IDs to integers from 0 to N-1. But as a simplication, we'll use the
movie id directly as an index in our model, in particular to lookup the movie
embedding from the movie embedding table. So we need do know the number of
movies.
"""

movies_count = movies.cardinality().numpy()

"""
In this example, we're going to focus on the ratings data. Other tutorials
explore how to use the movie information data as well as the user information to
improve the model quality.

We keep only the `user_id`, `movie_id` and `rating` fields in the dataset. Our
input is the `user_id`. The labels are the `movie_id` alongside the `rating` for
the given movie and user.

The `rating` is a number between 1 and 5, we adapt it to be between 0 and 1.
"""


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


"""
To fit and evaluate the model, we need to split it into a training and
evaluation set. In a real recommender system, this would most likely be done by
time: the data up to time *T* would be used to predict interactions after *T*.

In this simple example, however, let's use a random split, putting 80% of the
ratings in the train set, and 20% in the test set.
"""

shuffled_ratings = ratings.map(preprocess_rating).shuffle(
    100_000, seed=42, reshuffle_each_iteration=False
)
train_ratings = shuffled_ratings.take(80_000).batch(1000).cache()
test_ratings = shuffled_ratings.skip(80_000).take(20_000).batch(1000).cache()

"""
## Implementing the Model

Choosing the architecture of our model is a key part of modelling.

We are building a two-tower retrieval model, therefore we need to combine a
query tower for users and a candidate tower for movies.

The first step is to decide on the dimensionality of the query and candidate
representations. This is the `embedding_dimension` argument in our model
constructor. We'll test with a value of `32`. Higher values will correspond to
models that may be more accurate, but will also be slower to fit and more prone
to overfitting.

### Query and Candidate Towers

The second step is to define the model itself. In this simple example, the query
tower and candidate tower are simply embeddings with nothing else. We'll use
Keras' `Embedding` layer.

We can easily extend the towers to make them arbitrarily complex using standard
Keras components, as long as we return an `embedding_dimension`-wide output at
the end.

### Retrieval

The retrieval itself will be performed by `BruteForceRetrieval` layer from Keras
Recommenders. This layer computes the affinity scores for the given users and
all the candidate movies, then returns the top K in order.

Note that during training, we don't actually need to perform any retrieval since
the only affinity scores we need are the ones for the users and movies in the
batch. As an optimization, we skip the retrieval entirely in the `call` method.

### Loss

The next component is the loss used to train our model. In this case, we use a
mean square error loss to measure the difference between the predicted movie
ratings and the actual ratins from users.

Note that we override `compute_loss` from the `keras.Model` class. This allows
us to compute the query-candidate affinity score, which is obtained by
multiplying the outputs of the two towers together. That affinity score can then
be passed to the loss function.
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

Let's first instantiate the model. Note that we add `+ 1` to the number of users
and movies to account for the fact that id zero is not used for either (IDs
start at 1), but still takes a row in the embedding tables.
"""

model = RetrievalModel(users_count + 1, movies_count + 1)
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))

"""
Then train the model. Evaluation takes a bit of time, so we only evaluate the
model every 5 epochs.
"""

history = model.fit(
    train_ratings, validation_data=test_ratings, validation_freq=5, epochs=50
)

"""
## Making predictions

Now that we have a model, we would like to be able to make predictions.

So far, we have only handled movies by id. Now is the time to create a mapping
keyed by movie IDs to be able to surface the titles.
"""

movie_id_to_movie_title = {
    int(x["movie_id"]): x["movie_title"] for x in movies.as_numpy_iterator()
}
movie_id_to_movie_title[0] = ""  # Because id 0 is not in the dataset.

"""
We then simply use the Keras `model.predict()` method. Under the hood, it calls
the `BruteForceRetrieval` layer to perform the actual retrieval.

Note that this model can retrieve movies already watched by the user. We could
easily add logic to remove them if that is desirable.
"""

user_id = 42
predictions = model.predict(keras.ops.convert_to_tensor([user_id]))
predictions = keras.ops.convert_to_numpy(predictions["predictions"])

print(f"Recommended movies for user {user_id}:")
for movie_id in predictions[0]:
    print(movie_id_to_movie_title[movie_id])

"""
## Item-to-item recommendation

In this model, we created a user-movie model. However, for some applications
(for example, product detail pages) it's common to perform item-to-item (for
example, movie-to-movie or product-to-product) recommendations.

Training models like this would follow the same pattern as shown in this
tutorial, but with different training data. Here, we had a user and a movie
tower, and used (user, movie) pairs to train them. In an item-to-item model, we
would have two item towers (for the query and candidate item), and train the
model using (query item, candidate item) pairs. These could be constructed from
clicks on product detail pages.
"""

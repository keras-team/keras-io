# Recommending movies: retrieval

**Author:** [Fabien Hertschuh](https://github.com/hertschuh/), [Abheesht Sharma](https://github.com/abheesht17/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Retrieve movies using a two tower model.


<div class='example_version_banner keras_2'>ⓘ This example uses Keras 2</div>
<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_rs/ipynb/basic_retrieval.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_rs/basic_retrieval.py)



---
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


```python
!pip install -q keras-rs
```


```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # `"tensorflow"`/`"torch"`

import keras
import tensorflow as tf  # Needed for the dataset
import tensorflow_datasets as tfds

import keras_rs
```

---
## Preparing the dataset

Let's first have a look at the data.

We use the MovieLens dataset from
[Tensorflow Datasets](https://www.tensorflow.org/datasets). Loading
`movielens/100k_ratings` yields a `tf.data.Dataset` object containing the
ratings alongside user and movie data. Loading `movielens/100k_movies` yields a
`tf.data.Dataset` object containing only the movies data.

Note that since the MovieLens dataset does not have predefined splits, all data
are under `train` split.


```python
# Ratings data with user and movie data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")
```

The ratings dataset returns a dictionary of movie id, user id, the assigned
rating, timestamp, movie information, and user information:


```python
for data in ratings.take(1).as_numpy_iterator():
    print(str(data).replace(", '", ",\n '"))
```

<div class="k-default-codeblock">
```
{'bucketized_user_age': np.float32(45.0),
 'movie_genres': array([7]),
 'movie_id': b'357',
 'movie_title': b"One Flew Over the Cuckoo's Nest (1975)",
 'raw_user_age': np.float32(46.0),
 'timestamp': np.int64(879024327),
 'user_gender': np.True_,
 'user_id': b'138',
 'user_occupation_label': np.int64(4),
 'user_occupation_text': b'doctor',
 'user_rating': np.float32(4.0),
 'user_zip_code': b'53211'}
```
</div>

In the Movielens dataset, user IDs are integers (represented as strings)
starting at 1 and with no gap. Normally, you would need to create a lookup table
to map user IDs to integers from 0 to N-1. But as a simplication, we'll use the
user id directly as an index in our model, in particular to lookup the user
embedding from the user embedding table. So we need do know the number of users.


```python
users_count = (
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy()
)
```

The movies dataset contains the movie id, movie title, and the genres it belongs
to. Note that the genres are encoded with integer labels.


```python
for data in movies.take(1).as_numpy_iterator():
    print(str(data).replace(", '", ",\n '"))
```

<div class="k-default-codeblock">
```
{'movie_genres': array([4]),
 'movie_id': b'1681',
 'movie_title': b'You So Crazy (1994)'}
```
</div>

In the Movielens dataset, movie IDs are integers (represented as strings)
starting at 1 and with no gap. Normally, you would need to create a lookup table
to map movie IDs to integers from 0 to N-1. But as a simplication, we'll use the
movie id directly as an index in our model, in particular to lookup the movie
embedding from the movie embedding table. So we need do know the number of
movies.


```python
movies_count = movies.cardinality().numpy()
```

In this example, we're going to focus on the ratings data. Other tutorials
explore how to use the movie information data as well as the user information to
improve the model quality.

We keep only the `user_id`, `movie_id` and `rating` fields in the dataset. Our
input is the `user_id`. The labels are the `movie_id` alongside the `rating` for
the given movie and user.

The `rating` is a number between 1 and 5, we adapt it to be between 0 and 1.


```python

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

```

To fit and evaluate the model, we need to split it into a training and
evaluation set. In a real recommender system, this would most likely be done by
time: the data up to time *T* would be used to predict interactions after *T*.

In this simple example, however, let's use a random split, putting 80% of the
ratings in the train set, and 20% in the test set.


```python
shuffled_ratings = ratings.map(preprocess_rating).shuffle(
    100_000, seed=42, reshuffle_each_iteration=False
)
train_ratings = shuffled_ratings.take(80_000).batch(1000).cache()
test_ratings = shuffled_ratings.skip(80_000).take(20_000).batch(1000).cache()
```

---
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


```python

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

```

---
## Fitting and evaluating

After defining the model, we can use the standard Keras `model.fit()` to train
and evaluate the model.

Let's first instantiate the model. Note that we add `+ 1` to the number of users
and movies to account for the fact that id zero is not used for either (IDs
start at 1), but still takes a row in the embedding tables.


```python
model = RetrievalModel(users_count + 1, movies_count + 1)
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))
```

Then train the model. Evaluation takes a bit of time, so we only evaluate the
model every 5 epochs.


```python
history = model.fit(
    train_ratings, validation_data=test_ratings, validation_freq=5, epochs=50
)
```

<div class="k-default-codeblock">
```
Epoch 1/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - loss: 0.4772
Epoch 2/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4772
Epoch 3/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4772
Epoch 4/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4771
Epoch 5/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 3s 37ms/step - loss: 0.4771 - val_loss: 0.4836
Epoch 6/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4771
Epoch 7/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4770
Epoch 8/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.4770
Epoch 9/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4770
Epoch 10/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4769 - val_loss: 0.4836
Epoch 11/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4769
Epoch 12/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.4768
Epoch 13/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4768
Epoch 14/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4768
Epoch 15/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4767 - val_loss: 0.4836
Epoch 16/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4767
Epoch 17/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4766
Epoch 18/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4766
Epoch 19/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4765
Epoch 20/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4765 - val_loss: 0.4835
Epoch 21/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4764
Epoch 22/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4763
Epoch 23/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4763
Epoch 24/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4762
Epoch 25/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4761 - val_loss: 0.4833
Epoch 26/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4761
Epoch 27/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4759
Epoch 28/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4758
Epoch 29/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4757
Epoch 30/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4756 - val_loss: 0.4829
Epoch 31/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4755
Epoch 32/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4753
Epoch 33/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4752
Epoch 34/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4750
Epoch 35/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4748 - val_loss: 0.4822
Epoch 36/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4745
Epoch 37/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4742
Epoch 38/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.4740
Epoch 39/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4737
Epoch 40/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4734 - val_loss: 0.4809
Epoch 41/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4730
Epoch 42/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4726
Epoch 43/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4721
Epoch 44/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4716
Epoch 45/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4710 - val_loss: 0.4786
Epoch 46/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4703
Epoch 47/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4696
Epoch 48/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4688
Epoch 49/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4679
Epoch 50/50
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4669 - val_loss: 0.4743
```
</div>

---
## Making predictions

Now that we have a model, we would like to be able to make predictions.

So far, we have only handled movies by id. Now is the time to create a mapping
keyed by movie IDs to be able to surface the titles.


```python
movie_id_to_movie_title = {
    int(x["movie_id"]): x["movie_title"] for x in movies.as_numpy_iterator()
}
movie_id_to_movie_title[0] = ""  # Because id 0 is not in the dataset.
```

We then simply use the Keras `model.predict()` method. Under the hood, it calls
the `BruteForceRetrieval` layer to perform the actual retrieval.

Note that this model can retrieve movies already watched by the user. We could
easily add logic to remove them if that is desirable.


```python
user_id = 42
predictions = model.predict(keras.ops.convert_to_tensor([user_id]))
predictions = keras.ops.convert_to_numpy(predictions["predictions"])

print(f"Recommended movies for user {user_id}:")
for movie_id in predictions[0]:
    print(movie_id_to_movie_title[movie_id])
```

<div class="k-default-codeblock">
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step
Recommended movies for user 42:
b'Star Wars (1977)'
b'Godfather, The (1972)'
b'Back to the Future (1985)'
b'Fargo (1996)'
b'Snow White and the Seven Dwarfs (1937)'
b'Twelve Monkeys (1995)'
b'Pulp Fiction (1994)'
b'Raiders of the Lost Ark (1981)'
b'Dances with Wolves (1990)'
b'Courage Under Fire (1996)'
```
</div>

---
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


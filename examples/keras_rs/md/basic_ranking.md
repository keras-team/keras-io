# Recommending movies: ranking

**Author:** [Fabien Hertschuh](https://github.com/hertschuh/), [Abheesht Sharma](https://github.com/abheesht17/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Rank movies using a two tower model.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_rs/ipynb/basic_ranking.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_rs/basic_ranking.py)



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

In this tutorial, we're going to focus on the second stage, ranking. If you are
interested in the retrieval stage, have a look at our
[retrieval](/keras_rs/examples/basic_retrieval/)
tutorial.

In this tutorial, we're going to:

1. Get our data and split it into a training and test set.
2. Implement a ranking model.
3. Fit and evaluate it.
4. Test running predictions with the model.

Let's begin by choosing JAX as the backend we want to run on, and import all
the necessary libraries.


```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # `"tensorflow"`/`"torch"`

import keras
import tensorflow as tf  # Needed for the dataset
import tensorflow_datasets as tfds
```

---
## Preparing the dataset

We're going to use the same data as the
[retrieval](/keras_rs/examples/basic_retrieval/)
tutorial. The ratings are the objectives we are trying to predict.


```python
# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")
```

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

In the Movielens dataset, movie IDs are integers (represented as strings)
starting at 1 and with no gap. Normally, you would need to create a lookup table
to map movie IDs to integers from 0 to N-1. But as a simplication, we'll use the
movie id directly as an index in our model, in particular to lookup the movie
embedding from the movie embedding table. So we need do know the number of
movies.


```python
movies_count = movies.cardinality().numpy()
```

The inputs to the model are the user IDs and movie IDs and the labels are the
ratings.


```python

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

```

We'll split the data by putting 80% of the ratings in the train set, and 20% in
the test set.


```python
shuffled_ratings = ratings.map(preprocess_rating).shuffle(
    100_000, seed=42, reshuffle_each_iteration=False
)
train_ratings = shuffled_ratings.take(80_000).batch(1000).cache()
test_ratings = shuffled_ratings.skip(80_000).take(20_000).batch(1000).cache()
```

---
## Implementing the Model

### Architecture

Ranking models do not face the same efficiency constraints as retrieval models
do, and so we have a little bit more freedom in our choice of architectures.

A model composed of multiple stacked dense layers is a relatively common
architecture for ranking tasks. We can implement it as follows:


```python

class RankingModel(keras.Model):
    """Create the ranking model with the provided parameters.

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
        # Embedding table for users.
        self.user_embedding = keras.layers.Embedding(num_users, embedding_dimension)
        # Embedding table for candidates.
        self.candidate_embedding = keras.layers.Embedding(
            num_candidates, embedding_dimension
        )
        # Predictions.
        self.ratings = keras.Sequential(
            [
                # Learn multiple dense layers.
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                # Make rating predictions in the final layer.
                keras.layers.Dense(1),
            ]
        )

    def call(self, inputs):
        user_id, movie_id = inputs["user_id"], inputs["movie_id"]
        user_embeddings = self.user_embedding(user_id)
        candidate_embeddings = self.candidate_embedding(movie_id)
        return self.ratings(
            keras.ops.concatenate([user_embeddings, candidate_embeddings], axis=1)
        )

```

Let's first instantiate the model. Note that we add `+ 1` to the number of users
and movies to account for the fact that id zero is not used for either (IDs
start at 1), but still takes a row in the embedding tables.


```python
model = RankingModel(users_count + 1, movies_count + 1)
```

### Loss and metrics

The next component is the loss used to train our model. Keras has several losses
to make this easy. In this instance, we'll make use of the `MeanSquaredError`
loss in order to predict the ratings. We'll also look at the
`RootMeanSquaredError` metric.


```python
model.compile(
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError()],
    optimizer=keras.optimizers.Adagrad(learning_rate=0.1),
)
```

---
## Fitting and evaluating

After defining the model, we can use the standard Keras `model.fit()` to train
the model.


```python
model.fit(train_ratings, epochs=5)
```

<div class="k-default-codeblock">
```
Epoch 1/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - loss: 0.1071 - root_mean_squared_error: 0.3218
Epoch 2/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 0.0769 - root_mean_squared_error: 0.2773
Epoch 3/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0745 - root_mean_squared_error: 0.2730
Epoch 4/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0713 - root_mean_squared_error: 0.2670
Epoch 5/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0682 - root_mean_squared_error: 0.2612

<keras.src.callbacks.history.History at 0x7f69980ee5f0>
```
</div>

As the model trains, the loss is falling and the RMSE metric is improving.

Finally, we can evaluate our model on the test set. The lower the RMSE metric,
the more accurate our model is at predicting ratings.


```python
model.evaluate(test_ratings, return_dict=True)
```

<div class="k-default-codeblock">
```
20/20 ━━━━━━━━━━━━━━━━━━━━ 3s 12ms/step - loss: 0.0649 - root_mean_squared_error: 0.2548

{'loss': 0.06562447547912598, 'root_mean_squared_error': 0.2561727464199066}
```
</div>

---
## Testing the ranking model

So far, we have only handled movies by id. Now is the time to create a mapping
keyed by movie IDs to be able to surface the titles.


```python
movie_id_to_movie_title = {
    int(x["movie_id"]): x["movie_title"] for x in movies.as_numpy_iterator()
}
movie_id_to_movie_title[0] = ""  # Because id 0 is not in the dataset.
```

Now we can test the ranking model by computing predictions for a set of movies
and then rank these movies based on the predictions:


```python
user_id = 42
movie_ids = [204, 141, 131]
predictions = model.predict(
    {
        "user_id": keras.ops.array([user_id] * len(movie_ids)),
        "movie_id": keras.ops.array(movie_ids),
    }
)
predictions = keras.ops.convert_to_numpy(keras.ops.squeeze(predictions, axis=1))

for movie_id, prediction in zip(movie_ids, predictions):
    print(f"{movie_id_to_movie_title[movie_id]}: {5.0 * prediction:,.2f}")
```

<div class="k-default-codeblock">
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 153ms/step
b'Back to the Future (1985)': 3.53
b'20,000 Leagues Under the Sea (1954)': 3.26
b"Breakfast at Tiffany's (1961)": 3.43
```
</div>

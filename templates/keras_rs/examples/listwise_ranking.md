# List-wise ranking

**Author:** [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Rank movies using pairwise losses instead of pointwise losses.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_rs/ipynb/listwise_ranking.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_rs/listwise_ranking.py)



---
## Introduction

In our
[basic ranking tutorial](/keras_rs/examples/basic_ranking/), we explored a model
that learned to predict ratings for specific user-movie combinations. This model
took (user, movie) pairs as input and was trained using mean-squared error to
precisely predict the rating a user might give to a movie.

However, solely optimizing a model's accuracy in predicting individual movie
scores isn't always the most effective strategy for developing ranking systems.
For ranking models, pinpoint accuracy in predicting scores is less critical than
the model's capability to generate an ordered list of items that aligns with a
user's preferences. In essence, the relative order of items matters more than
the exact predicted values.

Instead of focusing on the model's predictions for individual query-item pairs
(a pointwise approach), we can optimize the model based on its ability to
correctly order items. One common method for this is pairwise ranking. In this
approach, the model learns by comparing pairs of items (e.g., item A and item B)
and determining which one should be ranked higher for a given user or query. The
goal is to minimize the number of incorrectly ordered pairs.

Let's begin by importing all the necessary libraries.


```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # `"tensorflow"`/`"torch"`

import collections

import keras
import numpy as np
import tensorflow as tf  # Needed only for the dataset
import tensorflow_datasets as tfds
from keras import ops

import keras_rs
```

Let's define some hyperparameters here.


```python
# Data args
TRAIN_NUM_LIST_PER_USER = 50
TEST_NUM_LIST_PER_USER = 1
NUM_EXAMPLES_PER_LIST = 5

# Model args
EMBEDDING_DIM = 32

# Train args
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 0.1
```

---
## Preparing the dataset

We use the MovieLens dataset. The data loading and processing steps are similar
to previous tutorials, so, we will only discuss the differences here.


```python
# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

users_count = (
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy()
)
movies_count = movies.cardinality().numpy()


def preprocess_rating(x):
    return {
        "user_id": tf.strings.to_number(x["user_id"], out_type=tf.int32),
        "movie_id": tf.strings.to_number(x["movie_id"], out_type=tf.int32),
        # Normalise ratings between 0 and 1.
        "user_rating": (x["user_rating"] - 1.0) / 4.0,
    }


shuffled_ratings = ratings.map(preprocess_rating).shuffle(
    100_000, seed=42, reshuffle_each_iteration=False
)
train_ratings = shuffled_ratings.take(70_000)
val_ratings = shuffled_ratings.skip(70_000).take(15_000)
test_ratings = shuffled_ratings.skip(85_000).take(15_000)
```

So far, we've replicated what we have in the basic ranking tutorial.

However, this existing dataset is not directly applicable to list-wise
optimization. List-wise optimization requires, for each user, a list of movies
they have rated, allowing the model to learn from the relative orderings within
that list. The MovieLens 100K dataset, in its original form, provides individual
rating instances (one user, one movie, one rating per example), rather than
these aggregated user-specific lists.

To enable listwise optimization, we need to restructure the dataset. This
involves transforming it so that each data point or example represents a single
user ID accompanied by a list of movies that user has rated. Within these lists,
some movies will naturally be ranked higher by the user (as evidenced by their
ratings) than others. The primary objective for our model will then be to learn
to predict item orderings that correspond to these observed user preferences.

Let's start by getting the entire list of movies and corresponding ratings for
every user. We remove `user_ids` corresponding to users who have rated less than
`NUM_EXAMPLES_PER_LIST` number of movies.


```python

def get_movie_sequence_per_user(ratings, min_examples_per_list):
    """Gets movieID sequences and ratings for every user."""
    sequences = collections.defaultdict(list)

    for sample in ratings:
        user_id = sample["user_id"]
        movie_id = sample["movie_id"]
        user_rating = sample["user_rating"]

        sequences[int(user_id.numpy())].append(
            {
                "movie_id": int(movie_id.numpy()),
                "user_rating": float(user_rating.numpy()),
            }
        )

    # Remove lists with < `min_examples_per_list` number of elements.
    sequences = {
        user_id: sequence
        for user_id, sequence in sequences.items()
        if len(sequence) >= min_examples_per_list
    }

    return sequences

```

We now sample 50 lists for each user for the training data. For each list, we
randomly sample 5 movies from the movies the user rated.


```python

def sample_sublist_from_list(
    lst,
    num_examples_per_list,
):
    """Random selects `num_examples_per_list` number of elements from list."""

    indices = np.random.choice(
        range(len(lst)),
        size=num_examples_per_list,
        replace=False,
    )

    samples = [lst[i] for i in indices]
    return samples


def get_examples(
    sequences,
    num_list_per_user,
    num_examples_per_list,
):
    inputs = {
        "user_id": [],
        "movie_id": [],
    }
    labels = []
    for user_id, user_list in sequences.items():
        sampled_list = sample_sublist_from_list(
            user_list,
            num_examples_per_list,
        )

        inputs["user_id"].append(user_id)
        inputs["movie_id"].append(
            tf.convert_to_tensor([f["movie_id"] for f in sampled_list])
        )
        labels.append(tf.convert_to_tensor([f["user_rating"] for f in sampled_list]))

    return (
        {"user_id": inputs["user_id"], "movie_id": inputs["movie_id"]},
        labels,
    )


train_sequences = get_movie_sequence_per_user(
    ratings=train_ratings, min_examples_per_list=NUM_EXAMPLES_PER_LIST
)
train_examples = get_examples(
    train_sequences,
    num_list_per_user=TRAIN_NUM_LIST_PER_USER,
    num_examples_per_list=NUM_EXAMPLES_PER_LIST,
)
train_ds = tf.data.Dataset.from_tensor_slices(train_examples)

val_sequences = get_movie_sequence_per_user(
    ratings=val_ratings, min_examples_per_list=5
)
val_examples = get_examples(
    val_sequences,
    num_list_per_user=TEST_NUM_LIST_PER_USER,
    num_examples_per_list=NUM_EXAMPLES_PER_LIST,
)
val_ds = tf.data.Dataset.from_tensor_slices(val_examples)

test_sequences = get_movie_sequence_per_user(
    ratings=test_ratings, min_examples_per_list=5
)
test_examples = get_examples(
    test_sequences,
    num_list_per_user=TEST_NUM_LIST_PER_USER,
    num_examples_per_list=NUM_EXAMPLES_PER_LIST,
)
test_ds = tf.data.Dataset.from_tensor_slices(test_examples)
```

Batch up the dataset, and cache it.


```python
train_ds = train_ds.batch(BATCH_SIZE).cache()
val_ds = val_ds.batch(BATCH_SIZE).cache()
test_ds = test_ds.batch(BATCH_SIZE).cache()
```

---
## Building the model

We build a typical two-tower ranking model, similar to the
[basic ranking tutorial](/keras_rs/examples/basic_ranking/).
We have separate embedding layers for user ID and movie IDs. After obtaining
these embeddings, we concatenate them and pass them through a network of dense
layers.

The only point of difference is that for movie IDs, we take a list of IDs
rather than just one movie ID. So, when we concatenate user ID embedding and
movie IDs' embeddings, we "repeat" the user ID 'NUM_EXAMPLES_PER_LIST' times so
as to get the same shape as the movie IDs' embeddings.


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

    def build(self, input_shape):
        self.user_embedding.build(input_shape["user_id"])
        self.candidate_embedding.build(input_shape["movie_id"])

        output_shape = self.candidate_embedding.compute_output_shape(
            input_shape["movie_id"]
        )

        self.ratings.build(list(output_shape[:-1]) + [2 * output_shape[-1]])

    def call(self, inputs):
        user_id, movie_id = inputs["user_id"], inputs["movie_id"]
        user_embeddings = self.user_embedding(user_id)
        candidate_embeddings = self.candidate_embedding(movie_id)

        list_length = ops.shape(movie_id)[-1]
        user_embeddings_repeated = ops.repeat(
            ops.expand_dims(user_embeddings, axis=1),
            repeats=list_length,
            axis=1,
        )
        concatenated_embeddings = ops.concatenate(
            [user_embeddings_repeated, candidate_embeddings], axis=-1
        )

        scores = self.ratings(concatenated_embeddings)
        scores = ops.squeeze(scores, axis=-1)

        return scores

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

```

Let's instantiate, compile and train our model. We will train two models:
one with vanilla mean-squared error, and the other with pairwise hinge loss.
For the latter, we will use `keras_rs.losses.PairwiseHingeLoss`.

Pairwise losses compare pairs of items within each list, penalizing cases where
an item with a higher true label has a lower predicted score than an item with a
lower true label. This is why they are more suited for ranking tasks than
pointwise losses.

To quantify these results, we compute nDCG. nDCG is a measure of ranking quality
that evaluates how well a system orders items based on relevance, giving more
importance to highly relevant items appearing at the top of the list and
normalizing the score against an ideal ranking.
To compute it, we just need to pass `keras_rs.metrics.NDCG()` as a metric to
`model.compile`.


```python
model_mse = RankingModel(
    num_users=users_count + 1,
    num_candidates=movies_count + 1,
    embedding_dimension=EMBEDDING_DIM,
)
model_mse.compile(
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras_rs.metrics.NDCG(k=NUM_EXAMPLES_PER_LIST, name="ndcg")],
    optimizer=keras.optimizers.Adagrad(learning_rate=LEARNING_RATE),
)
model_mse.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
```

<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.4960 - ndcg: 0.8892

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 3s 3s/step - loss: 0.4960 - ndcg: 0.8892 - val_loss: 0.1187 - val_ndcg: 0.8846


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - loss: 0.1150 - ndcg: 0.8898

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step - loss: 0.1150 - ndcg: 0.8898 - val_loss: 0.0893 - val_ndcg: 0.8878


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0876 - ndcg: 0.8884

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.0876 - ndcg: 0.8884 - val_loss: 0.0864 - val_ndcg: 0.8857


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0834 - ndcg: 0.8896

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 0.0834 - ndcg: 0.8896 - val_loss: 0.0815 - val_ndcg: 0.8876


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0794 - ndcg: 0.8887

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 0.0794 - ndcg: 0.8887 - val_loss: 0.0810 - val_ndcg: 0.8868





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7ae9e43ccc50>

```
</div>
And now, the model with pairwise hinge loss.


```python
model_hinge = RankingModel(
    num_users=users_count + 1,
    num_candidates=movies_count + 1,
    embedding_dimension=EMBEDDING_DIM,
)
model_hinge.compile(
    loss=keras_rs.losses.PairwiseHingeLoss(),
    metrics=[keras_rs.metrics.NDCG(k=NUM_EXAMPLES_PER_LIST, name="ndcg")],
    optimizer=keras.optimizers.Adagrad(learning_rate=LEARNING_RATE),
)
model_hinge.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
```

<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - loss: 1.4067 - ndcg: 0.8933

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step - loss: 1.4067 - ndcg: 0.8933 - val_loss: 1.3927 - val_ndcg: 0.8930


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - loss: 1.4061 - ndcg: 0.8953

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step - loss: 1.4061 - ndcg: 0.8953 - val_loss: 1.3925 - val_ndcg: 0.8936


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 1.4054 - ndcg: 0.8977

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 1.4054 - ndcg: 0.8977 - val_loss: 1.3923 - val_ndcg: 0.8941


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 1.4047 - ndcg: 0.8999

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 1.4047 - ndcg: 0.8999 - val_loss: 1.3921 - val_ndcg: 0.8941


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 1.4041 - ndcg: 0.9004

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 1.4041 - ndcg: 0.9004 - val_loss: 1.3919 - val_ndcg: 0.8940





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7ae9e4166590>

```
</div>
---
## Evaluation

Comparing the validation nDCG values, it is clear that the model trained with
the pairwise hinge loss outperforms the other one. Let's make this observation
more concrete by comparing results on the test set.


```python
ndcg_mse = model_mse.evaluate(test_ds, return_dict=True)["ndcg"]
ndcg_hinge = model_hinge.evaluate(test_ds, return_dict=True)["ndcg"]
print(ndcg_mse, ndcg_hinge)
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - loss: 0.0805 - ndcg: 0.8886

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step - loss: 0.0805 - ndcg: 0.8886


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 910ms/step - loss: 1.3878 - ndcg: 0.8924

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 914ms/step - loss: 1.3878 - ndcg: 0.8924


<div class="k-default-codeblock">
```
0.8885537385940552 0.8924424052238464

```
</div>
---
## Prediction

Now, let's rank some lists!

Let's create a mapping from movie ID to title so that we can surface the titles
for the ranked list.


```python
movie_id_to_movie_title = {
    int(x["movie_id"]): x["movie_title"] for x in movies.as_numpy_iterator()
}
movie_id_to_movie_title[0] = ""  # Because id 0 is not in the dataset.

user_id = 42
movie_ids = [409, 237, 131, 941, 543]
predictions = model_hinge.predict(
    {
        "user_id": keras.ops.array([user_id]),
        "movie_id": keras.ops.array([movie_ids]),
    }
)
predictions = keras.ops.convert_to_numpy(keras.ops.squeeze(predictions, axis=0))
sorted_indices = np.argsort(predictions)
sorted_movies = [movie_ids[i] for i in sorted_indices]

for i, movie_id in enumerate(sorted_movies):
    print(f"{i + 1}. ", movie_id_to_movie_title[movie_id])
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 261ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 262ms/step


<div class="k-default-codeblock">
```
1.  b'Jack (1996)'
2.  b'Mis\xc3\xa9rables, Les (1995)'
3.  b'Jerry Maguire (1996)'
4.  b"Breakfast at Tiffany's (1961)"
5.  b'With Honors (1994)'

```
</div>
And we're all done!

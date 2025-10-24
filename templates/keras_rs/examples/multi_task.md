# Multi-task recommenders: retrieval + ranking

**Author:** [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Using one model for both retrieval and ranking.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_rs/ipynb/multi_task.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_rs/multi_task.py)



---
## Introduction

In the
[basic retrieval](/keras_rs/examples/basic_retrieval/)
and
[basic ranking](/keras_rs/examples/basic_ranking/)
tutorials, we created separate models for retrieval and ranking tasks,
respectively. However, in many cases, building a single, joint model for
multiple tasks can lead to better performance than creating distinct models for
each task. This is especially true when dealing with data that is unevenly
distributed — such as abundant data (e.g., clicks) versus sparse data
(e.g., purchases, returns, or manual reviews). In such scenarios, a joint model
can leverage representations learned from the abundant data to improve
predictions on the sparse data, a technique known as transfer learning.
For instance, [research](https://openreview.net/forum?id=SJxPVcSonN) shows that
a model trained to predict user ratings from sparse survey data can be
significantly enhanced by incorporating an auxiliary task using abundant click
log data.

In this example, we develop a multi-objective recommender system using the
MovieLens dataset. We incorporate both implicit feedback (e.g., movie watches)
and explicit feedback (e.g., ratings) to create a more robust and effective
recommendation model. For the former, we predict "movie watches", i.e., whether
a user has watched a movie, and for the latter, we predict the rating given by a
user to a movie.

Let's start by importing the necessary packages.


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
## Prepare the dataset

We use the MovieLens dataset. The data loading and processing steps are similar
to previous tutorials, so we will not discuss them in details here.


```python
# Ratings data with user and movie data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")
```

Get user and movie counts so that we can define embedding layers.


```python
users_count = (
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy()
)

movies_count = movies.cardinality().numpy()
```

Our inputs are `"user_id"` and `"movie_id"`. Our label for the ranking task is
`"user_rating"`. `"user_rating"` is an integer between 0 to 4. We constrain it
to `[0, 1]`.


```python

def preprocess_rating(x):
    return (
        {
            "user_id": tf.strings.to_number(x["user_id"], out_type=tf.int32),
            "movie_id": tf.strings.to_number(x["movie_id"], out_type=tf.int32),
        },
        (x["user_rating"] - 1.0) / 4.0,
    )


shuffled_ratings = ratings.map(preprocess_rating).shuffle(
    100_000, seed=42, reshuffle_each_iteration=False
)

```

Split the dataset into train-test sets.


```python
train_ratings = shuffled_ratings.take(80_000).batch(1000).cache()
test_ratings = shuffled_ratings.skip(80_000).take(20_000).batch(1000).cache()
```

---
## Building the model

We build the model in a similar way to the basic retrieval and basic ranking
guides.

For the retrieval task (i.e., predicting whether a user watched a movie),
we compute the similarity of the corresponding user and movie embeddings, and
use cross entropy loss, where the positive pairs are labelled one, and all other
samples in the batch are considered "negatives". We report top-k accuracy for
this task.

For the ranking task (i.e., given a user-movie pair, predict rating), we
concatenate user and movie embeddings and pass it to a dense module. We use
MSE loss here, and report the Root Mean Squared Error (RMSE).

The final loss is a weighted combination of the two losses mentioned above,
where the weights are `"retrieval_loss_wt"` and `"ranking_loss_wt"`. These
weights decide which task the model will focus on.


```python

class MultiTaskModel(keras.Model):
    def __init__(
        self,
        num_users,
        num_candidates,
        embedding_dimension=32,
        layer_sizes=(256, 128),
        retrieval_loss_wt=1.0,
        ranking_loss_wt=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Our query tower, simply an embedding table.
        self.user_embedding = keras.layers.Embedding(num_users, embedding_dimension)

        # Our candidate tower, simply an embedding table.
        self.candidate_embedding = keras.layers.Embedding(
            num_candidates, embedding_dimension
        )

        # Rating model.
        self.rating_model = keras.Sequential(
            [
                keras.layers.Dense(layer_size, activation="relu")
                for layer_size in layer_sizes
            ]
            + [keras.layers.Dense(1)]
        )

        # The layer that performs the retrieval.
        self.retrieval = keras_rs.layers.BruteForceRetrieval(k=10, return_scores=False)

        self.retrieval_loss_fn = keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction="sum",
        )
        self.ranking_loss_fn = keras.losses.MeanSquaredError()

        # Top-k accuracy for retrieval
        self.top_k_metric = keras.metrics.SparseTopKCategoricalAccuracy(
            k=10, from_sorted_ids=True
        )
        # RMSE for ranking
        self.rmse_metric = keras.metrics.RootMeanSquaredError()

        # Attributes.
        self.num_users = num_users
        self.num_candidates = num_candidates
        self.embedding_dimension = embedding_dimension
        self.layer_sizes = layer_sizes
        self.retrieval_loss_wt = retrieval_loss_wt
        self.ranking_loss_wt = ranking_loss_wt

    def build(self, input_shape):
        self.user_embedding.build(input_shape)
        self.candidate_embedding.build(input_shape)
        # In this case, the candidates are directly the movie embeddings.
        # We take a shortcut and directly reuse the variable.
        self.retrieval.candidate_embeddings = self.candidate_embedding.embeddings
        self.retrieval.build(input_shape)

        self.rating_model.build((None, 2 * self.embedding_dimension))

        super().build(input_shape)

    def call(self, inputs, training=False):
        # Unpack inputs. Note that we have the if condition throughout this
        # `call()` method so that we can do a `.predict()` for the retrieval
        # task.
        user_id = inputs["user_id"]
        if "movie_id" in inputs:
            movie_id = inputs["movie_id"]

        result = {}

        # Get user, movie embeddings.
        user_embeddings = self.user_embedding(user_id)
        result["user_embeddings"] = user_embeddings

        if "movie_id" in inputs:
            candidate_embeddings = self.candidate_embedding(movie_id)
            result["candidate_embeddings"] = candidate_embeddings

            # Pass both embeddings through the rating block of the model.
            rating = self.rating_model(
                keras.ops.concatenate([user_embeddings, candidate_embeddings], axis=1)
            )
            result["rating"] = rating

        if not training:
            # Skip the retrieval of top movies during training as the
            # predictions are not used.
            result["predictions"] = self.retrieval(user_embeddings)

        return result

    def compute_loss(self, x, y, y_pred, sample_weight, training=True):
        user_embeddings = y_pred["user_embeddings"]
        candidate_embeddings = y_pred["candidate_embeddings"]

        # 1. Retrieval

        # Compute the affinity score by multiplying the two embeddings.
        scores = keras.ops.matmul(
            user_embeddings,
            keras.ops.transpose(candidate_embeddings),
        )

        # Retrieval labels: One-hot vectors
        num_users = keras.ops.shape(user_embeddings)[0]
        num_candidates = keras.ops.shape(candidate_embeddings)[0]
        retrieval_labels = keras.ops.eye(num_users, num_candidates)
        # Retrieval loss
        retrieval_loss = self.retrieval_loss_fn(retrieval_labels, scores, sample_weight)

        # 2. Ranking
        ratings = y
        pred_rating = y_pred["rating"]

        # Ranking labels are just ratings.
        ranking_labels = keras.ops.expand_dims(ratings, -1)
        # Ranking loss
        ranking_loss = self.ranking_loss_fn(ranking_labels, pred_rating, sample_weight)

        # Total loss is a weighted combination of the two losses.
        total_loss = (
            self.retrieval_loss_wt * retrieval_loss
            + self.ranking_loss_wt * ranking_loss
        )

        return total_loss

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        # RMSE can be computed irrespective of whether we are
        # training/evaluating.
        self.rmse_metric.update_state(
            y,
            y_pred["rating"],
            sample_weight=sample_weight,
        )

        if "predictions" in y_pred:
            # We are evaluating or predicting. Update `top_k_metric`.
            movie_ids = x["movie_id"]
            predictions = y_pred["predictions"]
            # For `top_k_metric`, which is a `SparseTopKCategoricalAccuracy`, we
            # only take top rated movies, and we put a weight of 0 for the rest.
            rating_weight = keras.ops.cast(keras.ops.greater(y, 0.9), "float32")
            sample_weight = (
                rating_weight
                if sample_weight is None
                else keras.ops.multiply(rating_weight, sample_weight)
            )
            self.top_k_metric.update_state(
                movie_ids, predictions, sample_weight=sample_weight
            )

            return self.get_metrics_result()
        else:
            # We are training. `top_k_metric` is not updated and is zero, so
            # don't report it.
            result = self.get_metrics_result()
            result.pop(self.top_k_metric.name)
            return result

```

---
## Training and evaluating

We will train three different models here. This can be done easily by passing
the correct loss weights:

1. Rating-specialised model
2. Retrieval-specialised model
3. Multi-task model


```python
# Rating-specialised model
model = MultiTaskModel(
    num_users=users_count + 1,
    num_candidates=movies_count + 1,
    ranking_loss_wt=1.0,
    retrieval_loss_wt=0.0,
)
model.compile(optimizer=keras.optimizers.Adagrad(0.1))
model.fit(train_ratings, epochs=5)

model.evaluate(test_ratings)

# Retrieval-specialised model
model = MultiTaskModel(
    num_users=users_count + 1,
    num_candidates=movies_count + 1,
    ranking_loss_wt=0.0,
    retrieval_loss_wt=1.0,
)
model.compile(optimizer=keras.optimizers.Adagrad(0.1))
model.fit(train_ratings, epochs=5)

model.evaluate(test_ratings)

# Multi-task model
model = MultiTaskModel(
    num_users=users_count + 1,
    num_candidates=movies_count + 1,
    ranking_loss_wt=1.0,
    retrieval_loss_wt=1.0,
)
model.compile(optimizer=keras.optimizers.Adagrad(0.1))
model.fit(train_ratings, epochs=5)

model.evaluate(test_ratings)
```

<div class="k-default-codeblock">
```
Epoch 1/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 5s 13ms/step - loss: 0.1089 - root_mean_squared_error: 0.3242
Epoch 2/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 0.0777 - root_mean_squared_error: 0.2788
Epoch 3/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0764 - root_mean_squared_error: 0.2763
Epoch 4/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0742 - root_mean_squared_error: 0.2724
Epoch 5/5
20/20 ━━━━━━━━━━━━━━━━━━━━ 3s 28ms/step - loss: 0.0716 - root_mean_squared_error: 0.2675 - sparse_top_k_categorical_accuracy: 0.0063
Epoch 1/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 11ms/step - loss: 6855.5034 - root_mean_squared_error: 0.6792
Epoch 2/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 6523.5024 - root_mean_squared_error: 0.6524
Epoch 3/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6376.9727 - root_mean_squared_error: 0.6512
Epoch 4/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6288.7183 - root_mean_squared_error: 0.6527
Epoch 5/5
20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 28ms/step - loss: 6551.5796 - root_mean_squared_error: 0.6573 - sparse_top_k_categorical_accuracy: 0.0197
Epoch 1/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 11ms/step - loss: 6860.5400 - root_mean_squared_error: 0.3157
Epoch 2/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 6520.5342 - root_mean_squared_error: 0.2598
Epoch 3/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6376.9668 - root_mean_squared_error: 0.2528
Epoch 4/5
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6291.7310 - root_mean_squared_error: 0.2502
Epoch 5/5
20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 28ms/step - loss: 6552.1499 - root_mean_squared_error: 0.2483 - sparse_top_k_categorical_accuracy: 0.0178

[6554.578125, 0.01855670101940632, 0.25010260939598083]
```
</div>

Let's plot a table of the metrics and pen down our observations:

| Model                 | Top-K Accuracy (↑) | RMSE (↓) |
|-----------------------|--------------------|----------|
| rating-specialised    | 0.005              | 0.26     |
| retrieval-specialised | 0.020              | 0.78     |
| multi-task            | 0.022              | 0.25     |

As expected, the rating-specialised model has good RMSE, but poor top-k
accuracy. For the retrieval-specialised model, it's the opposite.

For the multi-task model, we notice that the model does well (or even slightly
better than the two specialised models) on both tasks. In general, we can expect
multi-task learning to bring about better results, especially when one task has
a data-abundant source, and the other task is trained on sparse data.

Now, let's make a prediction! We will first do a retrieval, and then for the
retrieved list of movies, we will predict the rating using the same model.


```python
movie_id_to_movie_title = {
    int(x["movie_id"]): x["movie_title"] for x in movies.as_numpy_iterator()
}
movie_id_to_movie_title[0] = ""  # Because id 0 is not in the dataset.

user_id = 5
retrieved_movie_ids = model.predict(
    {
        "user_id": keras.ops.array([user_id]),
    }
)
retrieved_movie_ids = keras.ops.convert_to_numpy(retrieved_movie_ids["predictions"][0])
retrieved_movies = [movie_id_to_movie_title[x] for x in retrieved_movie_ids]
```

<div class="k-default-codeblock">
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 84ms/step
```
</div>

For these retrieved movies, we can now get the corresponding ratings.


```python
pred_ratings = model.predict(
    {
        "user_id": keras.ops.array([user_id] * len(retrieved_movie_ids)),
        "movie_id": keras.ops.array(retrieved_movie_ids),
    }
)["rating"]
pred_ratings = keras.ops.convert_to_numpy(keras.ops.squeeze(pred_ratings, axis=1))

for movie_id, prediction in zip(retrieved_movie_ids, pred_ratings):
    print(f"{movie_id_to_movie_title[movie_id]}: {5.0 * prediction:,.2f}")
```

<div class="k-default-codeblock">
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 418ms/step
b'Blob, The (1958)': 1.54
b'Little Rascals, The (1994)': 1.83
b'Jaws 3-D (1983)': 2.01
b'Black Beauty (1994)': 2.23
b'Burnt Offerings (1976)': 2.00
b'Mighty Morphin Power Rangers: The Movie (1995)': 2.11
b'Beverly Hillbillies, The (1993)': 2.12
b'Flintstones, The (1994)': 2.42
b'Heavy Metal (1981)': 2.67
b'Lassie (1994)': 2.02
```
</div>

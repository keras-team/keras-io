# Multi-task recommenders: retrieval + ranking

**Author:** [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Using one model for both retrieval and ranking.


<div class='example_version_banner keras_2'>ⓘ This example uses Keras 2</div>
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
        self.rating_model = tf.keras.Sequential(
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
            k=100, from_sorted_ids=True
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
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(train_ratings, epochs=5)

model.evaluate(test_ratings)

# Retrieval-specialised model
model = MultiTaskModel(
    num_users=users_count + 1,
    num_candidates=movies_count + 1,
    ranking_loss_wt=0.0,
    retrieval_loss_wt=1.0,
)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(train_ratings, epochs=5)

model.evaluate(test_ratings)

# Multi-task model
model = MultiTaskModel(
    num_users=users_count + 1,
    num_candidates=movies_count + 1,
    ranking_loss_wt=1.0,
    retrieval_loss_wt=1.0,
)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(train_ratings, epochs=5)

model.evaluate(test_ratings)
```

<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  7:13 5s/step - loss: 0.4353 - root_mean_squared_error: 0.6598

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  55s 711ms/step - loss: 0.3566 - root_mean_squared_error: 0.5934

<div class="k-default-codeblock">
```

```
</div>
  3/80 [37m━━━━━━━━━━━━━━━━━━━━  27s 358ms/step - loss: 0.3082 - root_mean_squared_error: 0.5489

<div class="k-default-codeblock">
```

```
</div>
  4/80 ━[37m━━━━━━━━━━━━━━━━━━━  18s 240ms/step - loss: 0.2757 - root_mean_squared_error: 0.5172

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  2s 43ms/step - loss: 0.1505 - root_mean_squared_error: 0.3784  

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  1s 25ms/step - loss: 0.1255 - root_mean_squared_error: 0.3465

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 18ms/step - loss: 0.1141 - root_mean_squared_error: 0.3314

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 0.1071 - root_mean_squared_error: 0.3219

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 0.1028 - root_mean_squared_error: 0.3160

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 7s 14ms/step - loss: 0.1026 - root_mean_squared_error: 0.3157


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  57s 724ms/step - loss: 0.0787 - root_mean_squared_error: 0.2805

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.0790 - root_mean_squared_error: 0.2810   

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0774 - root_mean_squared_error: 0.2783

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 0.0774 - root_mean_squared_error: 0.2781

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 3ms/step - loss: 0.0774 - root_mean_squared_error: 0.2781

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 0.0774 - root_mean_squared_error: 0.2782

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 0.0774 - root_mean_squared_error: 0.2782


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.0771 - root_mean_squared_error: 0.2776

<div class="k-default-codeblock">
```

```
</div>
 20/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0758 - root_mean_squared_error: 0.2753

<div class="k-default-codeblock">
```

```
</div>
 21/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0758 - root_mean_squared_error: 0.2753

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 0.0755 - root_mean_squared_error: 0.2748

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 0.0755 - root_mean_squared_error: 0.2748

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 0.0755 - root_mean_squared_error: 0.2748

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0754 - root_mean_squared_error: 0.2747

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0754 - root_mean_squared_error: 0.2747


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 8ms/step - loss: 0.0739 - root_mean_squared_error: 0.2719

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0731 - root_mean_squared_error: 0.2704

<div class="k-default-codeblock">
```

```
</div>
 38/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 3ms/step - loss: 0.0727 - root_mean_squared_error: 0.2696

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 3ms/step - loss: 0.0726 - root_mean_squared_error: 0.2695

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 3ms/step - loss: 0.0727 - root_mean_squared_error: 0.2696

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0727 - root_mean_squared_error: 0.2696


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.0710 - root_mean_squared_error: 0.2664

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0701 - root_mean_squared_error: 0.2648

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 2ms/step - loss: 0.0702 - root_mean_squared_error: 0.2649

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.0702 - root_mean_squared_error: 0.2649

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.0702 - root_mean_squared_error: 0.2649

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0703 - root_mean_squared_error: 0.2651

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0703 - root_mean_squared_error: 0.2651


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  1:16 4s/step - loss: 0.0682 - root_mean_squared_error: 0.2612 - sparse_top_k_categorical_accuracy: 0.0092

<div class="k-default-codeblock">
```

```
</div>
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  9s 511ms/step - loss: 0.0668 - root_mean_squared_error: 0.2585 - sparse_top_k_categorical_accuracy: 0.0077

<div class="k-default-codeblock">
```

```
</div>
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  4s 255ms/step - loss: 0.0668 - root_mean_squared_error: 0.2585 - sparse_top_k_categorical_accuracy: 0.0077

<div class="k-default-codeblock">
```

```
</div>
  8/20 ━━━━━━━━[37m━━━━━━━━━━━━  1s 90ms/step - loss: 0.0656 - root_mean_squared_error: 0.2563 - sparse_top_k_categorical_accuracy: 0.0073 
  9/20 ━━━━━━━━━[37m━━━━━━━━━━━  0s 79ms/step - loss: 0.0656 - root_mean_squared_error: 0.2563 - sparse_top_k_categorical_accuracy: 0.0073 

<div class="k-default-codeblock">
```

```
</div>
  5/20 ━━━━━[37m━━━━━━━━━━━━━━━  2s 157ms/step - loss: 0.0660 - root_mean_squared_error: 0.2563 - sparse_top_k_categorical_accuracy: 0.0073

<div class="k-default-codeblock">
```

```
</div>
  7/20 ━━━━━━━[37m━━━━━━━━━━━━━  1s 105ms/step - loss: 0.0658 - root_mean_squared_error: 0.2563 - sparse_top_k_categorical_accuracy: 0.0073

<div class="k-default-codeblock">
```

```
</div>
 10/20 ━━━━━━━━━━[37m━━━━━━━━━━  0s 85ms/step - loss: 0.0657 - root_mean_squared_error: 0.2563 - sparse_top_k_categorical_accuracy: 0.0073 
  6/20 ━━━━━━[37m━━━━━━━━━━━━━━  1s 126ms/step - loss: 0.0659 - root_mean_squared_error: 0.2562 - sparse_top_k_categorical_accuracy: 0.0073

<div class="k-default-codeblock">
```

```
</div>
 12/20 ━━━━━━━━━━━━[37m━━━━━━━━  0s 72ms/step - loss: 0.0657 - root_mean_squared_error: 0.2564 - sparse_top_k_categorical_accuracy: 0.0073

<div class="k-default-codeblock">
```

```
</div>
 18/20 ━━━━━━━━━━━━━━━━━━[37m━━  0s 48ms/step - loss: 0.0660 - root_mean_squared_error: 0.2569 - sparse_top_k_categorical_accuracy: 0.0073 

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step - loss: 0.0661 - root_mean_squared_error: 0.2570 - sparse_top_k_categorical_accuracy: 0.0072

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 5s 60ms/step - loss: 0.0661 - root_mean_squared_error: 0.2571 - sparse_top_k_categorical_accuracy: 0.0072


<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1:32 1s/step - loss: 6907.6763 - root_mean_squared_error: 0.6632

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  1:00 777ms/step - loss: 6907.7173 - root_mean_squared_error: 0.6681

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  2s 38ms/step - loss: 6907.5015 - root_mean_squared_error: 0.6806   

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 20ms/step - loss: 6895.8970 - root_mean_squared_error: 0.6774

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 6875.6831 - root_mean_squared_error: 0.6739

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 12ms/step - loss: 6857.6890 - root_mean_squared_error: 0.6712


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  55s 708ms/step - loss: 6598.5684 - root_mean_squared_error: 0.6119

<div class="k-default-codeblock">
```

```
</div>
 21/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6576.6855 - root_mean_squared_error: 0.6306   

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6575.6372 - root_mean_squared_error: 0.6307

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 6555.6934 - root_mean_squared_error: 0.6303

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 6539.7500 - root_mean_squared_error: 0.6297

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 6529.7354 - root_mean_squared_error: 0.6291

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 6529.1782 - root_mean_squared_error: 0.6291


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 6426.4985 - root_mean_squared_error: 0.5996

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 6412.3838 - root_mean_squared_error: 0.6172

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 2ms/step - loss: 6399.0010 - root_mean_squared_error: 0.6176

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 3ms/step - loss: 6390.9131 - root_mean_squared_error: 0.6177

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 3ms/step - loss: 6384.1377 - root_mean_squared_error: 0.6176

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6382.9321 - root_mean_squared_error: 0.6176

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6382.6494 - root_mean_squared_error: 0.6176


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 6326.7461 - root_mean_squared_error: 0.5948

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 6319.2397 - root_mean_squared_error: 0.6115

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6318.5737 - root_mean_squared_error: 0.6116

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 3ms/step - loss: 6305.7837 - root_mean_squared_error: 0.6121

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 3ms/step - loss: 6305.3188 - root_mean_squared_error: 0.6121

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 6298.2529 - root_mean_squared_error: 0.6123

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 6297.9868 - root_mean_squared_error: 0.6123

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6294.5269 - root_mean_squared_error: 0.6123


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6255.7402 - root_mean_squared_error: 0.5915

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 6252.3589 - root_mean_squared_error: 0.6078

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 2ms/step - loss: 6240.0132 - root_mean_squared_error: 0.6084

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6232.9590 - root_mean_squared_error: 0.6086

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6230.0630 - root_mean_squared_error: 0.6087

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6229.9438 - root_mean_squared_error: 0.6087


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  9s 512ms/step - loss: 6516.5493 - root_mean_squared_error: 0.6098 - sparse_top_k_categorical_accuracy: 0.0183

<div class="k-default-codeblock">
```

```
</div>
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  6s 360ms/step - loss: 6534.5430 - root_mean_squared_error: 0.6105 - sparse_top_k_categorical_accuracy: 0.0255

<div class="k-default-codeblock">
```

```
</div>
  7/20 ━━━━━━━[37m━━━━━━━━━━━━━  1s 120ms/step - loss: 6544.4048 - root_mean_squared_error: 0.6105 - sparse_top_k_categorical_accuracy: 0.0255

<div class="k-default-codeblock">
```

```
</div>
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  12s 719ms/step - loss: 6531.3584 - root_mean_squared_error: 0.6105 - sparse_top_k_categorical_accuracy: 0.0255

<div class="k-default-codeblock">
```

```
</div>
  4/20 ━━━━[37m━━━━━━━━━━━━━━━━  3s 240ms/step - loss: 6537.4272 - root_mean_squared_error: 0.6105 - sparse_top_k_categorical_accuracy: 0.0255
  6/20 ━━━━━━[37m━━━━━━━━━━━━━━  2s 144ms/step - loss: 6545.1812 - root_mean_squared_error: 0.6105 - sparse_top_k_categorical_accuracy: 0.0255
  5/20 ━━━━━[37m━━━━━━━━━━━━━━━  2s 180ms/step - loss: 6545.1348 - root_mean_squared_error: 0.6105 - sparse_top_k_categorical_accuracy: 0.0255

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 39ms/step - loss: 6549.4980 - root_mean_squared_error: 0.6116 - sparse_top_k_categorical_accuracy: 0.0201 


<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  57s 729ms/step - loss: 6908.3989 - root_mean_squared_error: 0.6723

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  54s 701ms/step - loss: 6908.2402 - root_mean_squared_error: 0.6139

<div class="k-default-codeblock">
```

```
</div>
  3/80 [37m━━━━━━━━━━━━━━━━━━━━  27s 352ms/step - loss: 6908.1592 - root_mean_squared_error: 0.5696

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  2s 36ms/step - loss: 6907.8389 - root_mean_squared_error: 0.3792  

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 19ms/step - loss: 6899.2207 - root_mean_squared_error: 0.3425

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 6880.0688 - root_mean_squared_error: 0.3258

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 6862.6636 - root_mean_squared_error: 0.3175

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 3s 23ms/step - loss: 6861.5864 - root_mean_squared_error: 0.3171


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  55s 704ms/step - loss: 6594.2095 - root_mean_squared_error: 0.2551

<div class="k-default-codeblock">
```

```
</div>
 21/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6572.7637 - root_mean_squared_error: 0.2580   

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 6552.2993 - root_mean_squared_error: 0.2572

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 6536.5630 - root_mean_squared_error: 0.2571

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 6524.3457 - root_mean_squared_error: 0.2572


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 6417.7388 - root_mean_squared_error: 0.2482

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 6402.9614 - root_mean_squared_error: 0.2534

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 2ms/step - loss: 6389.1875 - root_mean_squared_error: 0.2526

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6380.7139 - root_mean_squared_error: 0.2527

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6375.1587 - root_mean_squared_error: 0.2530

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6374.8774 - root_mean_squared_error: 0.2530


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 6325.0674 - root_mean_squared_error: 0.2435

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 6311.3281 - root_mean_squared_error: 0.2504

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6310.6494 - root_mean_squared_error: 0.2503

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 6299.2373 - root_mean_squared_error: 0.2498

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 3ms/step - loss: 6292.0332 - root_mean_squared_error: 0.2499

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 3ms/step - loss: 6292.3276 - root_mean_squared_error: 0.2499
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 6291.4810 - root_mean_squared_error: 0.2499

<div class="k-default-codeblock">
```

```
</div>
 78/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 3ms/step - loss: 6287.3857 - root_mean_squared_error: 0.2502

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6287.0112 - root_mean_squared_error: 0.2503

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6286.8350 - root_mean_squared_error: 0.2503


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 6261.9985 - root_mean_squared_error: 0.2411

<div class="k-default-codeblock">
```

```
</div>
 20/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6248.2485 - root_mean_squared_error: 0.2484

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6249.1104 - root_mean_squared_error: 0.2484

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6246.6167 - root_mean_squared_error: 0.2483

<div class="k-default-codeblock">
```

```
</div>
 21/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6247.4165 - root_mean_squared_error: 0.2483

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 4ms/step - loss: 6237.0596 - root_mean_squared_error: 0.2478

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 3ms/step - loss: 6236.5112 - root_mean_squared_error: 0.2478

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 3ms/step - loss: 6228.8872 - root_mean_squared_error: 0.2478

<div class="k-default-codeblock">
```

```
</div>
 56/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 3ms/step - loss: 6228.6084 - root_mean_squared_error: 0.2478

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 3ms/step - loss: 6224.5903 - root_mean_squared_error: 0.2482

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6224.0234 - root_mean_squared_error: 0.2482

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6223.8984 - root_mean_squared_error: 0.2482


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  9s 525ms/step - loss: 6515.7671 - root_mean_squared_error: 0.2492 - sparse_top_k_categorical_accuracy: 0.0229

<div class="k-default-codeblock">
```

```
</div>
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  6s 372ms/step - loss: 6542.2920 - root_mean_squared_error: 0.2473 - sparse_top_k_categorical_accuracy: 0.0253

<div class="k-default-codeblock">
```

```
</div>
  5/20 ━━━━━[37m━━━━━━━━━━━━━━━  2s 186ms/step - loss: 6541.9136 - root_mean_squared_error: 0.2473 - sparse_top_k_categorical_accuracy: 0.0253
  6/20 ━━━━━━[37m━━━━━━━━━━━━━━  2s 149ms/step - loss: 6541.8491 - root_mean_squared_error: 0.2473 - sparse_top_k_categorical_accuracy: 0.0253

<div class="k-default-codeblock">
```

```
</div>
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  13s 744ms/step - loss: 6542.5410 - root_mean_squared_error: 0.2473 - sparse_top_k_categorical_accuracy: 0.0253

<div class="k-default-codeblock">
```

```
</div>
  4/20 ━━━━[37m━━━━━━━━━━━━━━━━  3s 248ms/step - loss: 6535.7803 - root_mean_squared_error: 0.2473 - sparse_top_k_categorical_accuracy: 0.0253

<div class="k-default-codeblock">
```

```
</div>
  7/20 ━━━━━━━[37m━━━━━━━━━━━━━  1s 124ms/step - loss: 6541.6343 - root_mean_squared_error: 0.2473 - sparse_top_k_categorical_accuracy: 0.0253

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 40ms/step - loss: 6545.6343 - root_mean_squared_error: 0.2485 - sparse_top_k_categorical_accuracy: 0.0209 





<div class="k-default-codeblock">
```
[6550.57568359375, 0.01878579705953598, 0.25044095516204834]

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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 178ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 181ms/step


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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 420ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 422ms/step


<div class="k-default-codeblock">
```
b'Blob, The (1958)': 1.88
b'Mighty Morphin Power Rangers: The Movie (1995)': 1.82
b'Flintstones, The (1994)': 2.17
b'Heavy Metal (1981)': 2.50
b'Burnt Offerings (1976)': 1.81
b'Little Rascals, The (1994)': 2.08
b'Jaws 3-D (1983)': 2.22
b'Star Trek V: The Final Frontier (1989)': 2.32
b'Street Fighter (1994)': 1.90
b'Mystery Science Theater 3000: The Movie (1996)': 2.72

```
</div>

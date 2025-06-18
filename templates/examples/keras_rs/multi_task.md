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
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  3:45 3s/step - loss: 0.4353 - root_mean_squared_error: 0.6598

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  52s 671ms/step - loss: 0.3644 - root_mean_squared_error: 0.6007

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  1s 29ms/step - loss: 0.1393 - root_mean_squared_error: 0.3644  

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  1s 28ms/step - loss: 0.1376 - root_mean_squared_error: 0.3623

<div class="k-default-codeblock">
```

```
</div>
 48/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 16ms/step - loss: 0.1170 - root_mean_squared_error: 0.3353

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 12ms/step - loss: 0.1073 - root_mean_squared_error: 0.3223

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 12ms/step - loss: 0.1070 - root_mean_squared_error: 0.3218

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 4s 13ms/step - loss: 0.1042 - root_mean_squared_error: 0.3180


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  52s 668ms/step - loss: 0.0780 - root_mean_squared_error: 0.2792

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.0785 - root_mean_squared_error: 0.2801   

<div class="k-default-codeblock">
```

```
</div>
 30/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0776 - root_mean_squared_error: 0.2786

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0776 - root_mean_squared_error: 0.2786

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.0776 - root_mean_squared_error: 0.2786

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 0.0777 - root_mean_squared_error: 0.2787

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 0.0777 - root_mean_squared_error: 0.2787

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 0.0777 - root_mean_squared_error: 0.2787

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.0777 - root_mean_squared_error: 0.2787


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.0763 - root_mean_squared_error: 0.2762

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0770 - root_mean_squared_error: 0.2775

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0761 - root_mean_squared_error: 0.2758

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.0761 - root_mean_squared_error: 0.2758

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.0761 - root_mean_squared_error: 0.2758

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 0.0760 - root_mean_squared_error: 0.2756

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 0.0760 - root_mean_squared_error: 0.2756

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0759 - root_mean_squared_error: 0.2755


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.0734 - root_mean_squared_error: 0.2710

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0745 - root_mean_squared_error: 0.2730

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0736 - root_mean_squared_error: 0.2713

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.0736 - root_mean_squared_error: 0.2713

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 0.0734 - root_mean_squared_error: 0.2710

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 0.0734 - root_mean_squared_error: 0.2710

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0735 - root_mean_squared_error: 0.2710


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.0717 - root_mean_squared_error: 0.2678

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0733 - root_mean_squared_error: 0.2713

    
  3/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0736 - root_mean_squared_error: 0.2713

<div class="k-default-codeblock">
```

```
</div>
 29/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0714 - root_mean_squared_error: 0.2671

<div class="k-default-codeblock">
```

```
</div>
 30/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0714 - root_mean_squared_error: 0.2672

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 0.0713 - root_mean_squared_error: 0.2670

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 0.0713 - root_mean_squared_error: 0.2670

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0713 - root_mean_squared_error: 0.2669


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  42s 2s/step - loss: 0.0685 - root_mean_squared_error: 0.2618 - sparse_top_k_categorical_accuracy: 0.0046

<div class="k-default-codeblock">
```

```
</div>
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  5s 349ms/step - loss: 0.0677 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044

<div class="k-default-codeblock">
```

```
</div>
  5/20 ━━━━━[37m━━━━━━━━━━━━━━━  2s 174ms/step - loss: 0.0670 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044
  9/20 ━━━━━━━━━[37m━━━━━━━━━━━  0s 87ms/step - loss: 0.0667 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044 
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  12s 696ms/step - loss: 0.0681 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044
 11/20 ━━━━━━━━━━━[37m━━━━━━━━━  0s 70ms/step - loss: 0.0667 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044 
  6/20 ━━━━━━[37m━━━━━━━━━━━━━━  1s 140ms/step - loss: 0.0667 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044
 10/20 ━━━━━━━━━━[37m━━━━━━━━━━  0s 78ms/step - loss: 0.0667 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044 

<div class="k-default-codeblock">
```

```
</div>
 13/20 ━━━━━━━━━━━━━[37m━━━━━━━  0s 58ms/step - loss: 0.0671 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044 
  8/20 ━━━━━━━━[37m━━━━━━━━━━━━  1s 100ms/step - loss: 0.0668 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044

<div class="k-default-codeblock">
```

```
</div>
 12/20 ━━━━━━━━━━━━[37m━━━━━━━━  0s 64ms/step - loss: 0.0667 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044 

<div class="k-default-codeblock">
```

```
</div>
  7/20 ━━━━━━━[37m━━━━━━━━━━━━━  1s 116ms/step - loss: 0.0669 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044

<div class="k-default-codeblock">
```

```
</div>
  4/20 ━━━━[37m━━━━━━━━━━━━━━━━  3s 233ms/step - loss: 0.0667 - root_mean_squared_error: 0.2582 - sparse_top_k_categorical_accuracy: 0.0044

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 3s 38ms/step - loss: 0.0670 - root_mean_squared_error: 0.2589 - sparse_top_k_categorical_accuracy: 0.0046 


<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  55s 705ms/step - loss: 6907.7500 - root_mean_squared_error: 0.6712

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  53s 681ms/step - loss: 6907.7939 - root_mean_squared_error: 0.6763

<div class="k-default-codeblock">
```

```
</div>
 29/80 ━━━━━━━[37m━━━━━━━━━━━━━  1s 26ms/step - loss: 6906.6592 - root_mean_squared_error: 0.6932  

<div class="k-default-codeblock">
```

```
</div>
 30/80 ━━━━━━━[37m━━━━━━━━━━━━━  1s 25ms/step - loss: 6906.3804 - root_mean_squared_error: 0.6932

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 14ms/step - loss: 6887.2905 - root_mean_squared_error: 0.6935

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 14ms/step - loss: 6886.2769 - root_mean_squared_error: 0.6935

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 11ms/step - loss: 6861.2632 - root_mean_squared_error: 0.6933


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  52s 668ms/step - loss: 6595.3521 - root_mean_squared_error: 0.6702

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6568.2349 - root_mean_squared_error: 0.6925   

<div class="k-default-codeblock">
```

```
</div>
 29/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6567.1797 - root_mean_squared_error: 0.6926

<div class="k-default-codeblock">
```

```
</div>
 30/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6566.1387 - root_mean_squared_error: 0.6926

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 2ms/step - loss: 6544.7070 - root_mean_squared_error: 0.6939

<div class="k-default-codeblock">
```

```
</div>
 56/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 6543.9644 - root_mean_squared_error: 0.6939

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 6527.7217 - root_mean_squared_error: 0.6952


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6421.3364 - root_mean_squared_error: 0.6830

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 6426.4746 - root_mean_squared_error: 0.6891

<div class="k-default-codeblock">
```

```
</div>
 29/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6402.4702 - root_mean_squared_error: 0.7059

<div class="k-default-codeblock">
```

```
</div>
 30/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6401.7056 - root_mean_squared_error: 0.7059

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6400.9751 - root_mean_squared_error: 0.7059

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 6386.6289 - root_mean_squared_error: 0.7069

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6386.2451 - root_mean_squared_error: 0.7070

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6379.3403 - root_mean_squared_error: 0.7077


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6326.5630 - root_mean_squared_error: 0.6919

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6333.5112 - root_mean_squared_error: 0.6981

<div class="k-default-codeblock">
```

```
</div>
 29/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6309.5977 - root_mean_squared_error: 0.7150

<div class="k-default-codeblock">
```

```
</div>
 30/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6308.8608 - root_mean_squared_error: 0.7151

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 6296.6309 - root_mean_squared_error: 0.7158

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 6296.3599 - root_mean_squared_error: 0.7159

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6296.0918 - root_mean_squared_error: 0.7159

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6291.6152 - root_mean_squared_error: 0.7164


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6259.3281 - root_mean_squared_error: 0.6987

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 6267.6138 - root_mean_squared_error: 0.7051

<div class="k-default-codeblock">
```

```
</div>
 29/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6242.9312 - root_mean_squared_error: 0.7220

<div class="k-default-codeblock">
```

```
</div>
 30/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6242.1875 - root_mean_squared_error: 0.7220

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6241.4839 - root_mean_squared_error: 0.7221

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 6231.3540 - root_mean_squared_error: 0.7226

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 6231.1279 - root_mean_squared_error: 0.7226

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6227.6514 - root_mean_squared_error: 0.7231


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  9s 501ms/step - loss: 6525.7983 - root_mean_squared_error: 0.7341 - sparse_top_k_categorical_accuracy: 0.0183

<div class="k-default-codeblock">
```

```
</div>
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  12s 708ms/step - loss: 6545.6025 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156
  9/20 ━━━━━━━━━[37m━━━━━━━━━━━  0s 89ms/step - loss: 6557.3950 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156 
  5/20 ━━━━━[37m━━━━━━━━━━━━━━━  2s 177ms/step - loss: 6556.7119 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156
  6/20 ━━━━━━[37m━━━━━━━━━━━━━━  1s 142ms/step - loss: 6557.6411 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156
  4/20 ━━━━[37m━━━━━━━━━━━━━━━━  3s 237ms/step - loss: 6556.4917 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156
 13/20 ━━━━━━━━━━━━━[37m━━━━━━━  0s 59ms/step - loss: 6558.5605 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156 

<div class="k-default-codeblock">
```

```
</div>
 11/20 ━━━━━━━━━━━[37m━━━━━━━━━  0s 71ms/step - loss: 6557.2266 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156  

<div class="k-default-codeblock">
```

```
</div>
  7/20 ━━━━━━━[37m━━━━━━━━━━━━━  1s 119ms/step - loss: 6558.2988 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156 
 10/20 ━━━━━━━━━━[37m━━━━━━━━━━  0s 79ms/step - loss: 6557.6724 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156

<div class="k-default-codeblock">
```

```
</div>
  8/20 ━━━━━━━━[37m━━━━━━━━━━━━  1s 102ms/step - loss: 6557.9561 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156

<div class="k-default-codeblock">
```

```
</div>
 12/20 ━━━━━━━━━━━━[37m━━━━━━━━  0s 64ms/step - loss: 6556.1787 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156 

<div class="k-default-codeblock">
```

```
</div>
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  6s 356ms/step - loss: 6558.2368 - root_mean_squared_error: 0.7329 - sparse_top_k_categorical_accuracy: 0.0156

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 39ms/step - loss: 6558.5298 - root_mean_squared_error: 0.7323 - sparse_top_k_categorical_accuracy: 0.0156 


<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  56s 716ms/step - loss: 6907.9180 - root_mean_squared_error: 0.6640

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  51s 656ms/step - loss: 6907.9414 - root_mean_squared_error: 0.6054

<div class="k-default-codeblock">
```

```
</div>
  3/80 [37m━━━━━━━━━━━━━━━━━━━━  25s 330ms/step - loss: 6907.9351 - root_mean_squared_error: 0.5618

<div class="k-default-codeblock">
```

```
</div>
 30/80 ━━━━━━━[37m━━━━━━━━━━━━━  1s 25ms/step - loss: 6906.2886 - root_mean_squared_error: 0.3586  

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  1s 24ms/step - loss: 6905.9717 - root_mean_squared_error: 0.3569

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 13ms/step - loss: 6884.6377 - root_mean_squared_error: 0.3280

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 13ms/step - loss: 6883.6255 - root_mean_squared_error: 0.3274

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - loss: 6861.9297 - root_mean_squared_error: 0.3174


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  52s 660ms/step - loss: 6599.1538 - root_mean_squared_error: 0.2549

<div class="k-default-codeblock">
```

```
</div>
 29/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6566.7197 - root_mean_squared_error: 0.2586   

<div class="k-default-codeblock">
```

```
</div>
 30/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6565.6699 - root_mean_squared_error: 0.2586

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6564.6597 - root_mean_squared_error: 0.2586

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 6541.2002 - root_mean_squared_error: 0.2586

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6540.4863 - root_mean_squared_error: 0.2586

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 6526.9360 - root_mean_squared_error: 0.2591


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6427.2715 - root_mean_squared_error: 0.2496

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6430.3330 - root_mean_squared_error: 0.2527

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6401.6621 - root_mean_squared_error: 0.2532

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 6400.9707 - root_mean_squared_error: 0.2532

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 6400.2896 - root_mean_squared_error: 0.2531

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6386.1152 - root_mean_squared_error: 0.2531

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6385.7368 - root_mean_squared_error: 0.2532

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6385.3530 - root_mean_squared_error: 0.2533

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6379.2231 - root_mean_squared_error: 0.2537


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6332.7959 - root_mean_squared_error: 0.2469

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 6337.2896 - root_mean_squared_error: 0.2503

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 2ms/step - loss: 6308.8354 - root_mean_squared_error: 0.2503

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 6308.1694 - root_mean_squared_error: 0.2503

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6295.6636 - root_mean_squared_error: 0.2502

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6295.3931 - root_mean_squared_error: 0.2502

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 6295.1182 - root_mean_squared_error: 0.2502

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6290.9727 - root_mean_squared_error: 0.2506


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6266.3545 - root_mean_squared_error: 0.2446

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 6271.7319 - root_mean_squared_error: 0.2483

    
  3/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6268.4746 - root_mean_squared_error: 0.2497

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 6240.8154 - root_mean_squared_error: 0.2482

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 6240.1978 - root_mean_squared_error: 0.2482

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 6239.6104 - root_mean_squared_error: 0.2481

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 6229.3428 - root_mean_squared_error: 0.2482

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 6229.1450 - root_mean_squared_error: 0.2482

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 6228.9478 - root_mean_squared_error: 0.2482

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6226.5605 - root_mean_squared_error: 0.2485


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  9s 478ms/step - loss: 6510.3120 - root_mean_squared_error: 0.2476 - sparse_top_k_categorical_accuracy: 0.0183

<div class="k-default-codeblock">
```

```
</div>
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  5s 351ms/step - loss: 6552.2383 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158
  8/20 ━━━━━━━━[37m━━━━━━━━━━━━  1s 100ms/step - loss: 6548.0225 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158

<div class="k-default-codeblock">
```

```
</div>
 11/20 ━━━━━━━━━━━[37m━━━━━━━━━  0s 70ms/step - loss: 6552.4331 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158 
 10/20 ━━━━━━━━━━[37m━━━━━━━━━━  0s 78ms/step - loss: 6553.4868 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158 

<div class="k-default-codeblock">
```

```
</div>
  5/20 ━━━━━[37m━━━━━━━━━━━━━━━  2s 175ms/step - loss: 6552.0576 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158
 13/20 ━━━━━━━━━━━━━[37m━━━━━━━  0s 58ms/step - loss: 6553.3755 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158
  7/20 ━━━━━━━[37m━━━━━━━━━━━━━  1s 117ms/step - loss: 6552.1162 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158
  9/20 ━━━━━━━━━[37m━━━━━━━━━━━  0s 88ms/step - loss: 6552.2988 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158
  4/20 ━━━━[37m━━━━━━━━━━━━━━━━  3s 233ms/step - loss: 6552.1694 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158

<div class="k-default-codeblock">
```

```
</div>
  6/20 ━━━━━━[37m━━━━━━━━━━━━━━  1s 140ms/step - loss: 6551.8081 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158

<div class="k-default-codeblock">
```

```
</div>
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  12s 699ms/step - loss: 6548.6211 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158

<div class="k-default-codeblock">
```

```
</div>
 12/20 ━━━━━━━━━━━━[37m━━━━━━━━  0s 64ms/step - loss: 6552.3442 - root_mean_squared_error: 0.2488 - sparse_top_k_categorical_accuracy: 0.0158

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 38ms/step - loss: 6554.1953 - root_mean_squared_error: 0.2492 - sparse_top_k_categorical_accuracy: 0.0158





<div class="k-default-codeblock">
```
[6555.712890625, 0.016953036189079285, 0.2508334815502167]

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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 109ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 110ms/step


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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 273ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 274ms/step


<div class="k-default-codeblock">
```
b'Blob, The (1958)': 2.01
b'Mighty Morphin Power Rangers: The Movie (1995)': 2.03
b'Flintstones, The (1994)': 2.18
b'Beverly Hillbillies, The (1993)': 1.89
b'Lawnmower Man, The (1992)': 2.57
b'Hot Shots! Part Deux (1993)': 2.28
b'Street Fighter (1994)': 1.84
b'Cabin Boy (1994)': 1.94
b'Little Rascals, The (1994)': 2.12
b'Jaws 3-D (1983)': 2.27

```
</div>

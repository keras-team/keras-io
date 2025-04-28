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
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  7:04 5s/step - loss: 0.4471 - root_mean_squared_error: 0.6686

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  54s 693ms/step - loss: 0.3656 - root_mean_squared_error: 0.6009

<div class="k-default-codeblock">
```

```
</div>
 15/80 ━━━[37m━━━━━━━━━━━━━━━━━  3s 53ms/step - loss: 0.1663 - root_mean_squared_error: 0.3973  

<div class="k-default-codeblock">
```

```
</div>
 29/80 ━━━━━━━[37m━━━━━━━━━━━━━  1s 28ms/step - loss: 0.1328 - root_mean_squared_error: 0.3558

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 20ms/step - loss: 0.1188 - root_mean_squared_error: 0.3375

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 15ms/step - loss: 0.1101 - root_mean_squared_error: 0.3258

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 13ms/step - loss: 0.1052 - root_mean_squared_error: 0.3192

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 7s 14ms/step - loss: 0.1031 - root_mean_squared_error: 0.3163


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1:00 763ms/step - loss: 0.0779 - root_mean_squared_error: 0.2792

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.0782 - root_mean_squared_error: 0.2792    

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0771 - root_mean_squared_error: 0.2776

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 0.0771 - root_mean_squared_error: 0.2776

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 0.0772 - root_mean_squared_error: 0.2778

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 0.0772 - root_mean_squared_error: 0.2778


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.0762 - root_mean_squared_error: 0.2761

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0756 - root_mean_squared_error: 0.2749

<div class="k-default-codeblock">
```

```
</div>
 20/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0755 - root_mean_squared_error: 0.2748

<div class="k-default-codeblock">
```

```
</div>
 39/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 3ms/step - loss: 0.0754 - root_mean_squared_error: 0.2746

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 3ms/step - loss: 0.0754 - root_mean_squared_error: 0.2746

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0754 - root_mean_squared_error: 0.2746

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0754 - root_mean_squared_error: 0.2746


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.0736 - root_mean_squared_error: 0.2714

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0742 - root_mean_squared_error: 0.2725

<div class="k-default-codeblock">
```

```
</div>
 21/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0730 - root_mean_squared_error: 0.2702

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0730 - root_mean_squared_error: 0.2702

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 0.0729 - root_mean_squared_error: 0.2700

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 0.0729 - root_mean_squared_error: 0.2701

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0730 - root_mean_squared_error: 0.2702

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0730 - root_mean_squared_error: 0.2702


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.0715 - root_mean_squared_error: 0.2673

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0704 - root_mean_squared_error: 0.2654 

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0704 - root_mean_squared_error: 0.2653

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 0.0703 - root_mean_squared_error: 0.2652

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 0.0703 - root_mean_squared_error: 0.2652

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0705 - root_mean_squared_error: 0.2656

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0705 - root_mean_squared_error: 0.2656


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  1:16 4s/step - loss: 0.0697 - root_mean_squared_error: 0.2641 - sparse_top_k_categorical_accuracy: 0.0046

<div class="k-default-codeblock">
```

```
</div>
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  8s 484ms/step - loss: 0.0692 - root_mean_squared_error: 0.2624 - sparse_top_k_categorical_accuracy: 0.0043

<div class="k-default-codeblock">
```

```
</div>
  5/20 ━━━━━[37m━━━━━━━━━━━━━━━  2s 177ms/step - loss: 0.0680 - root_mean_squared_error: 0.2599 - sparse_top_k_categorical_accuracy: 0.0043
  4/20 ━━━━[37m━━━━━━━━━━━━━━━━  2s 172ms/step - loss: 0.0683 - root_mean_squared_error: 0.2608 - sparse_top_k_categorical_accuracy: 0.0043

<div class="k-default-codeblock">
```

```
</div>
  6/20 ━━━━━━[37m━━━━━━━━━━━━━━  1s 142ms/step - loss: 0.0677 - root_mean_squared_error: 0.2599 - sparse_top_k_categorical_accuracy: 0.0043

<div class="k-default-codeblock">
```

```
</div>
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  4s 244ms/step - loss: 0.0688 - root_mean_squared_error: 0.2613 - sparse_top_k_categorical_accuracy: 0.0043
  7/20 ━━━━━━━[37m━━━━━━━━━━━━━  1s 118ms/step - loss: 0.0675 - root_mean_squared_error: 0.2599 - sparse_top_k_categorical_accuracy: 0.0043

<div class="k-default-codeblock">
```

```
</div>
 11/20 ━━━━━━━━━━━[37m━━━━━━━━━  0s 75ms/step - loss: 0.0674 - root_mean_squared_error: 0.2597 - sparse_top_k_categorical_accuracy: 0.0055 

<div class="k-default-codeblock">
```

```
</div>
  9/20 ━━━━━━━━━[37m━━━━━━━━━━━  1s 93ms/step - loss: 0.0674 - root_mean_squared_error: 0.2597 - sparse_top_k_categorical_accuracy: 0.0055 
 12/20 ━━━━━━━━━━━━[37m━━━━━━━━  0s 68ms/step - loss: 0.0674 - root_mean_squared_error: 0.2597 - sparse_top_k_categorical_accuracy: 0.0055 

<div class="k-default-codeblock">
```

```
</div>
 10/20 ━━━━━━━━━━[37m━━━━━━━━━━  0s 83ms/step - loss: 0.0674 - root_mean_squared_error: 0.2597 - sparse_top_k_categorical_accuracy: 0.0055

<div class="k-default-codeblock">
```

```
</div>
 13/20 ━━━━━━━━━━━━━[37m━━━━━━━  0s 62ms/step - loss: 0.0674 - root_mean_squared_error: 0.2597 - sparse_top_k_categorical_accuracy: 0.0055

<div class="k-default-codeblock">
```

```
</div>
 14/20 ━━━━━━━━━━━━━━[37m━━━━━━  0s 58ms/step - loss: 0.0675 - root_mean_squared_error: 0.2597 - sparse_top_k_categorical_accuracy: 0.0058

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 5s 40ms/step - loss: 0.0677 - root_mean_squared_error: 0.2601 - sparse_top_k_categorical_accuracy: 0.0058


<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  56s 720ms/step - loss: 6907.5718 - root_mean_squared_error: 0.6611

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  54s 693ms/step - loss: 6907.5967 - root_mean_squared_error: 0.6660

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  2s 35ms/step - loss: 6907.6543 - root_mean_squared_error: 0.6793  

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 19ms/step - loss: 6899.1436 - root_mean_squared_error: 0.6793

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 13ms/step - loss: 6879.3330 - root_mean_squared_error: 0.6793

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 13ms/step - loss: 6878.2617 - root_mean_squared_error: 0.6793

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 6861.5137 - root_mean_squared_error: 0.6790

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 3s 23ms/step - loss: 6860.4180 - root_mean_squared_error: 0.6789


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1:14 949ms/step - loss: 6592.8989 - root_mean_squared_error: 0.6466

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 6592.5723 - root_mean_squared_error: 0.6500    

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6568.9629 - root_mean_squared_error: 0.6609

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6567.9653 - root_mean_squared_error: 0.6609

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 6549.0884 - root_mean_squared_error: 0.6598

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 6533.3462 - root_mean_squared_error: 0.6588

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 6522.8042 - root_mean_squared_error: 0.6579


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 6421.3394 - root_mean_squared_error: 0.6267

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 6425.9844 - root_mean_squared_error: 0.6300

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6405.7856 - root_mean_squared_error: 0.6417

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 6392.1797 - root_mean_squared_error: 0.6412

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 6383.0542 - root_mean_squared_error: 0.6408

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 6382.6963 - root_mean_squared_error: 0.6408

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6377.4087 - root_mean_squared_error: 0.6404


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 6326.1396 - root_mean_squared_error: 0.6142

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6332.9492 - root_mean_squared_error: 0.6228

    
  4/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 6327.3535 - root_mean_squared_error: 0.6258

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6313.2085 - root_mean_squared_error: 0.6292

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 3ms/step - loss: 6301.0547 - root_mean_squared_error: 0.6291

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 3ms/step - loss: 6300.6299 - root_mean_squared_error: 0.6291

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 6293.9229 - root_mean_squared_error: 0.6288

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 6294.4316 - root_mean_squared_error: 0.6289

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 6294.1777 - root_mean_squared_error: 0.6288

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6290.5591 - root_mean_squared_error: 0.6286


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 6257.9858 - root_mean_squared_error: 0.6055

<div class="k-default-codeblock">
```

```
</div>
 21/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6248.0718 - root_mean_squared_error: 0.6203

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6247.2905 - root_mean_squared_error: 0.6203

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 6234.9565 - root_mean_squared_error: 0.6203

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 6228.8110 - root_mean_squared_error: 0.6203

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6225.8696 - root_mean_squared_error: 0.6202


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  9s 493ms/step - loss: 6503.9023 - root_mean_squared_error: 0.6273 - sparse_top_k_categorical_accuracy: 0.0229

<div class="k-default-codeblock">
```

```
</div>
  4/20 ━━━━[37m━━━━━━━━━━━━━━━━  3s 245ms/step - loss: 6543.3096 - root_mean_squared_error: 0.6189 - sparse_top_k_categorical_accuracy: 0.0230

<div class="k-default-codeblock">
```

```
</div>
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  13s 734ms/step - loss: 6533.0210 - root_mean_squared_error: 0.6189 - sparse_top_k_categorical_accuracy: 0.0230
  7/20 ━━━━━━━[37m━━━━━━━━━━━━━  1s 122ms/step - loss: 6543.5859 - root_mean_squared_error: 0.6189 - sparse_top_k_categorical_accuracy: 0.0230
  6/20 ━━━━━━[37m━━━━━━━━━━━━━━  2s 147ms/step - loss: 6543.3711 - root_mean_squared_error: 0.6189 - sparse_top_k_categorical_accuracy: 0.0230
  5/20 ━━━━━[37m━━━━━━━━━━━━━━━  2s 184ms/step - loss: 6543.4663 - root_mean_squared_error: 0.6189 - sparse_top_k_categorical_accuracy: 0.0230
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  6s 367ms/step - loss: 6528.7485 - root_mean_squared_error: 0.6189 - sparse_top_k_categorical_accuracy: 0.0230

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 40ms/step - loss: 6549.4800 - root_mean_squared_error: 0.6196 - sparse_top_k_categorical_accuracy: 0.0187 


<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  57s 727ms/step - loss: 6908.3179 - root_mean_squared_error: 0.6688

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  53s 681ms/step - loss: 6908.2168 - root_mean_squared_error: 0.6108

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  1s 33ms/step - loss: 6907.8174 - root_mean_squared_error: 0.3759  

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 19ms/step - loss: 6897.9658 - root_mean_squared_error: 0.3407

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 14ms/step - loss: 6882.1890 - root_mean_squared_error: 0.3269

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 6861.9189 - root_mean_squared_error: 0.3174

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 3s 23ms/step - loss: 6860.8540 - root_mean_squared_error: 0.3169


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  53s 679ms/step - loss: 6596.2349 - root_mean_squared_error: 0.2571

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 6597.0312 - root_mean_squared_error: 0.2599   

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6575.0918 - root_mean_squared_error: 0.2593

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 6555.1104 - root_mean_squared_error: 0.2583

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 3ms/step - loss: 6554.2500 - root_mean_squared_error: 0.2583

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 3ms/step - loss: 6538.1826 - root_mean_squared_error: 0.2583

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 6528.5957 - root_mean_squared_error: 0.2586

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 6528.0205 - root_mean_squared_error: 0.2586


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 6425.8213 - root_mean_squared_error: 0.2489

<div class="k-default-codeblock">
```

```
</div>
 20/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6411.4111 - root_mean_squared_error: 0.2532

<div class="k-default-codeblock">
```

```
</div>
 21/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6410.5977 - root_mean_squared_error: 0.2531

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 6396.6382 - root_mean_squared_error: 0.2522

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 6386.1509 - root_mean_squared_error: 0.2523

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6379.8896 - root_mean_squared_error: 0.2528


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 6330.2397 - root_mean_squared_error: 0.2449
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 6337.2896 - root_mean_squared_error: 0.2492

<div class="k-default-codeblock">
```

```
</div>
 21/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6317.5742 - root_mean_squared_error: 0.2498

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6316.7524 - root_mean_squared_error: 0.2497

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 6303.6172 - root_mean_squared_error: 0.2490

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 6303.1318 - root_mean_squared_error: 0.2490

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 6296.2310 - root_mean_squared_error: 0.2491

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6291.5854 - root_mean_squared_error: 0.2496

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6291.3911 - root_mean_squared_error: 0.2496


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 8ms/step - loss: 6261.8848 - root_mean_squared_error: 0.2431

<div class="k-default-codeblock">
```

```
</div>
 20/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 6254.0996 - root_mean_squared_error: 0.2481

<div class="k-default-codeblock">
```

```
</div>
 39/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 3ms/step - loss: 6240.4780 - root_mean_squared_error: 0.2473

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 3ms/step - loss: 6233.4023 - root_mean_squared_error: 0.2472

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 3ms/step - loss: 6229.4580 - root_mean_squared_error: 0.2476

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6229.1709 - root_mean_squared_error: 0.2476


    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  9s 496ms/step - loss: 6525.8828 - root_mean_squared_error: 0.2453 - sparse_top_k_categorical_accuracy: 0.0183

<div class="k-default-codeblock">
```

```
</div>
  5/20 ━━━━━[37m━━━━━━━━━━━━━━━  2s 178ms/step - loss: 6548.0352 - root_mean_squared_error: 0.2463 - sparse_top_k_categorical_accuracy: 0.0258
  7/20 ━━━━━━━[37m━━━━━━━━━━━━━  1s 119ms/step - loss: 6552.0479 - root_mean_squared_error: 0.2463 - sparse_top_k_categorical_accuracy: 0.0258

<div class="k-default-codeblock">
```

```
</div>
  6/20 ━━━━━━[37m━━━━━━━━━━━━━━  2s 143ms/step - loss: 6551.8271 - root_mean_squared_error: 0.2463 - sparse_top_k_categorical_accuracy: 0.0258
  4/20 ━━━━[37m━━━━━━━━━━━━━━━━  3s 238ms/step - loss: 6548.7090 - root_mean_squared_error: 0.2463 - sparse_top_k_categorical_accuracy: 0.0258

<div class="k-default-codeblock">
```

```
</div>
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  6s 357ms/step - loss: 6545.3252 - root_mean_squared_error: 0.2463 - sparse_top_k_categorical_accuracy: 0.0258

<div class="k-default-codeblock">
```

```
</div>
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  12s 713ms/step - loss: 6543.3877 - root_mean_squared_error: 0.2463 - sparse_top_k_categorical_accuracy: 0.0258

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 1s 39ms/step - loss: 6554.1426 - root_mean_squared_error: 0.2482 - sparse_top_k_categorical_accuracy: 0.0213  





<div class="k-default-codeblock">
```
[6553.96240234375, 0.01993127167224884, 0.25033557415008545]

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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 114ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 115ms/step


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
b'Blob, The (1958)': 2.23
b'Flintstones, The (1994)': 1.91
b'Burnt Offerings (1976)': 1.98
b'Lawnmower Man, The (1992)': 2.39
b'Jaws 3-D (1983)': 1.52
b'Star Trek V: The Final Frontier (1989)': 2.38
b'Mighty Morphin Power Rangers: The Movie (1995)': 2.11
b'Lassie (1994)': 2.45
b'Heavy Metal (1981)': 2.90
b'Herbie Rides Again (1974)': 1.80

```
</div>

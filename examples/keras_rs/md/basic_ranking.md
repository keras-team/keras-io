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

<div class="k-default-codeblock">
```
WARNING:absl:Variant folder /root/tensorflow_datasets/movielens/100k-ratings/0.1.1 has no dataset_info.json

Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to /root/tensorflow_datasets/movielens/100k-ratings/0.1.1...

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/1 [00:00<?, ? splits/s]

Generating train examples...: 0 examples [00:00, ? examples/s]

Shuffling /root/tensorflow_datasets/movielens/100k-ratings/incomplete.NJQO4Z_0.1.1/movielens-train.tfrecord*..…

Dataset movielens downloaded and prepared to /root/tensorflow_datasets/movielens/100k-ratings/0.1.1. Subsequent calls will reuse this data.

WARNING:absl:Variant folder /root/tensorflow_datasets/movielens/100k-movies/0.1.1 has no dataset_info.json

Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to /root/tensorflow_datasets/movielens/100k-movies/0.1.1...

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/1 [00:00<?, ? splits/s]

Generating train examples...: 0 examples [00:00, ? examples/s]

Shuffling /root/tensorflow_datasets/movielens/100k-movies/incomplete.T86HZ6_0.1.1/movielens-train.tfrecord*...…

Dataset movielens downloaded and prepared to /root/tensorflow_datasets/movielens/100k-movies/0.1.1. Subsequent calls will reuse this data.

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

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  5:51 4s/step - loss: 0.4590 - root_mean_squared_error: 0.6775

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  37s 481ms/step - loss: 0.3960 - root_mean_squared_error: 0.6272

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  2s 33ms/step - loss: 0.1738 - root_mean_squared_error: 0.4062  

<div class="k-default-codeblock">
```

```
</div>
 30/80 ━━━━━━━[37m━━━━━━━━━━━━━  1s 20ms/step - loss: 0.1419 - root_mean_squared_error: 0.3673

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 14ms/step - loss: 0.1245 - root_mean_squared_error: 0.3450

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 12ms/step - loss: 0.1166 - root_mean_squared_error: 0.3345

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 12ms/step - loss: 0.1161 - root_mean_squared_error: 0.3339

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 10ms/step - loss: 0.1095 - root_mean_squared_error: 0.3250

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 5s 12ms/step - loss: 0.1083 - root_mean_squared_error: 0.3233


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  41s 529ms/step - loss: 0.0772 - root_mean_squared_error: 0.2779
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.0776 - root_mean_squared_error: 0.2786

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0770 - root_mean_squared_error: 0.2775

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 0.0770 - root_mean_squared_error: 0.2775

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 3ms/step - loss: 0.0770 - root_mean_squared_error: 0.2775

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 0.0771 - root_mean_squared_error: 0.2777

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 0.0771 - root_mean_squared_error: 0.2777


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.0749 - root_mean_squared_error: 0.2737

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0749 - root_mean_squared_error: 0.2738

<div class="k-default-codeblock">
```

```
</div>
 24/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0749 - root_mean_squared_error: 0.2737

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 0.0748 - root_mean_squared_error: 0.2736

<div class="k-default-codeblock">
```

```
</div>
 56/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 3ms/step - loss: 0.0748 - root_mean_squared_error: 0.2736

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 3ms/step - loss: 0.0748 - root_mean_squared_error: 0.2736

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 3ms/step - loss: 0.0748 - root_mean_squared_error: 0.2736

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 3ms/step - loss: 0.0748 - root_mean_squared_error: 0.2736

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0748 - root_mean_squared_error: 0.2735

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0748 - root_mean_squared_error: 0.2735


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.0710 - root_mean_squared_error: 0.2664

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.0721 - root_mean_squared_error: 0.2686 

<div class="k-default-codeblock">
```

```
</div>
 11/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.0722 - root_mean_squared_error: 0.2687

<div class="k-default-codeblock">
```

```
</div>
 21/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.0717 - root_mean_squared_error: 0.2678

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 5ms/step - loss: 0.0716 - root_mean_squared_error: 0.2676

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 5ms/step - loss: 0.0716 - root_mean_squared_error: 0.2676

<div class="k-default-codeblock">
```

```
</div>
 47/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 5ms/step - loss: 0.0716 - root_mean_squared_error: 0.2676

<div class="k-default-codeblock">
```

```
</div>
 48/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 5ms/step - loss: 0.0716 - root_mean_squared_error: 0.2676

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 5ms/step - loss: 0.0716 - root_mean_squared_error: 0.2676

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 5ms/step - loss: 0.0717 - root_mean_squared_error: 0.2677

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 5ms/step - loss: 0.0717 - root_mean_squared_error: 0.2677

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 5ms/step - loss: 0.0717 - root_mean_squared_error: 0.2677
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 5ms/step - loss: 0.0717 - root_mean_squared_error: 0.2677

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 5ms/step - loss: 0.0718 - root_mean_squared_error: 0.2679

<div class="k-default-codeblock">
```

```
</div>
 74/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 5ms/step - loss: 0.0718 - root_mean_squared_error: 0.2679

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 5ms/step - loss: 0.0718 - root_mean_squared_error: 0.2679

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 5ms/step - loss: 0.0718 - root_mean_squared_error: 0.2679

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 5ms/step - loss: 0.0718 - root_mean_squared_error: 0.2679
 78/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 5ms/step - loss: 0.0718 - root_mean_squared_error: 0.2679

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0718 - root_mean_squared_error: 0.2679

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.0718 - root_mean_squared_error: 0.2679


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 10ms/step - loss: 0.0677 - root_mean_squared_error: 0.2602

<div class="k-default-codeblock">
```

```
</div>
 13/80 ━━━[37m━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.0690 - root_mean_squared_error: 0.2626 

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.0684 - root_mean_squared_error: 0.2614

<div class="k-default-codeblock">
```

```
</div>
 38/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 4ms/step - loss: 0.0687 - root_mean_squared_error: 0.2621

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 4ms/step - loss: 0.0689 - root_mean_squared_error: 0.2625

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 4ms/step - loss: 0.0689 - root_mean_squared_error: 0.2625

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 4ms/step - loss: 0.0690 - root_mean_squared_error: 0.2627

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0690 - root_mean_squared_error: 0.2627





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7efcb212f810>

```
</div>
As the model trains, the loss is falling and the RMSE metric is improving.

Finally, we can evaluate our model on the test set. The lower the RMSE metric,
the more accurate our model is at predicting ratings.


```python
model.evaluate(test_ratings, return_dict=True)
```

    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  1:13 4s/step - loss: 0.0680 - root_mean_squared_error: 0.2608

<div class="k-default-codeblock">
```

```
</div>
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  3s 195ms/step - loss: 0.0672 - root_mean_squared_error: 0.2592

<div class="k-default-codeblock">
```

```
</div>
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  1s 100ms/step - loss: 0.0667 - root_mean_squared_error: 0.2566

<div class="k-default-codeblock">
```

```
</div>
 19/20 ━━━━━━━━━━━━━━━━━━━[37m━  0s 14ms/step - loss: 0.0653 - root_mean_squared_error: 0.2556 

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 68ms/step - loss: 0.0654 - root_mean_squared_error: 0.2557

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 5s 68ms/step - loss: 0.0654 - root_mean_squared_error: 0.2557





<div class="k-default-codeblock">
```
{'loss': 0.06593123823404312, 'root_mean_squared_error': 0.2567707896232605}

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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 295ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 297ms/step


<div class="k-default-codeblock">
```
b'Back to the Future (1985)': 3.78
b'20,000 Leagues Under the Sea (1954)': 3.43
b"Breakfast at Tiffany's (1961)": 3.64

```
</div>
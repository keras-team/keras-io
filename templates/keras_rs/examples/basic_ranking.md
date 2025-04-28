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

Shuffling /root/tensorflow_datasets/movielens/100k-ratings/incomplete.664N7W_0.1.1/movielens-train.tfrecord*..…

Dataset movielens downloaded and prepared to /root/tensorflow_datasets/movielens/100k-ratings/0.1.1. Subsequent calls will reuse this data.

WARNING:absl:Variant folder /root/tensorflow_datasets/movielens/100k-movies/0.1.1 has no dataset_info.json

Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to /root/tensorflow_datasets/movielens/100k-movies/0.1.1...

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/1 [00:00<?, ? splits/s]

Generating train examples...: 0 examples [00:00, ? examples/s]

Shuffling /root/tensorflow_datasets/movielens/100k-movies/incomplete.G9WXJD_0.1.1/movielens-train.tfrecord*...…

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
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  6:08 5s/step - loss: 0.4639 - root_mean_squared_error: 0.6811

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  53s 685ms/step - loss: 0.3940 - root_mean_squared_error: 0.6252

<div class="k-default-codeblock">
```

```
</div>
 12/80 ━━━[37m━━━━━━━━━━━━━━━━━  4s 67ms/step - loss: 0.1978 - root_mean_squared_error: 0.4339  

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  2s 44ms/step - loss: 0.1652 - root_mean_squared_error: 0.3958

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  1s 31ms/step - loss: 0.1375 - root_mean_squared_error: 0.3620

<div class="k-default-codeblock">
```

```
</div>
 29/80 ━━━━━━━[37m━━━━━━━━━━━━━  1s 31ms/step - loss: 0.1363 - root_mean_squared_error: 0.3605

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  1s 26ms/step - loss: 0.1299 - root_mean_squared_error: 0.3523

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 23ms/step - loss: 0.1235 - root_mean_squared_error: 0.3439

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 19ms/step - loss: 0.1175 - root_mean_squared_error: 0.3361

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 17ms/step - loss: 0.1131 - root_mean_squared_error: 0.3301

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 16ms/step - loss: 0.1099 - root_mean_squared_error: 0.3258

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 15ms/step - loss: 0.1079 - root_mean_squared_error: 0.3230

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - loss: 0.1076 - root_mean_squared_error: 0.3226

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - loss: 0.1073 - root_mean_squared_error: 0.3223


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  38s 483ms/step - loss: 0.0782 - root_mean_squared_error: 0.2797

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0773 - root_mean_squared_error: 0.2781   

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0773 - root_mean_squared_error: 0.2781

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 2ms/step - loss: 0.0774 - root_mean_squared_error: 0.2783

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.0775 - root_mean_squared_error: 0.2785

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.0776 - root_mean_squared_error: 0.2785

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.0776 - root_mean_squared_error: 0.2785

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 10ms/step - loss: 0.0776 - root_mean_squared_error: 0.2785


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.0772 - root_mean_squared_error: 0.2778

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0762 - root_mean_squared_error: 0.2761

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 2ms/step - loss: 0.0762 - root_mean_squared_error: 0.2761

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.0763 - root_mean_squared_error: 0.2762
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.0763 - root_mean_squared_error: 0.2762

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.0763 - root_mean_squared_error: 0.2762

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0763 - root_mean_squared_error: 0.2763


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.0755 - root_mean_squared_error: 0.2748

<div class="k-default-codeblock">
```

```
</div>
 21/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.0744 - root_mean_squared_error: 0.2728

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 3ms/step - loss: 0.0743 - root_mean_squared_error: 0.2725

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 0.0743 - root_mean_squared_error: 0.2725

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 3ms/step - loss: 0.0743 - root_mean_squared_error: 0.2725

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0742 - root_mean_squared_error: 0.2724

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0742 - root_mean_squared_error: 0.2724


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.0728 - root_mean_squared_error: 0.2698

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.0716 - root_mean_squared_error: 0.2676

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 2ms/step - loss: 0.0714 - root_mean_squared_error: 0.2671

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 0.0713 - root_mean_squared_error: 0.2670

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.0713 - root_mean_squared_error: 0.2670

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0713 - root_mean_squared_error: 0.2670

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.0713 - root_mean_squared_error: 0.2669





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7ae0c1610a10>

```
</div>
As the model trains, the loss is falling and the RMSE metric is improving.

Finally, we can evaluate our model on the test set. The lower the RMSE metric,
the more accurate our model is at predicting ratings.


```python
model.evaluate(test_ratings, return_dict=True)
```

    
  1/20 ━[37m━━━━━━━━━━━━━━━━━━━  1:09 4s/step - loss: 0.0706 - root_mean_squared_error: 0.2656

<div class="k-default-codeblock">
```

```
</div>
  2/20 ━━[37m━━━━━━━━━━━━━━━━━━  3s 205ms/step - loss: 0.0698 - root_mean_squared_error: 0.2634

<div class="k-default-codeblock">
```

```
</div>
  3/20 ━━━[37m━━━━━━━━━━━━━━━━━  1s 103ms/step - loss: 0.0694 - root_mean_squared_error: 0.2634

<div class="k-default-codeblock">
```

```
</div>
  6/20 ━━━━━━[37m━━━━━━━━━━━━━━  0s 55ms/step - loss: 0.0685 - root_mean_squared_error: 0.2616 

<div class="k-default-codeblock">
```

```
</div>
 18/20 ━━━━━━━━━━━━━━━━━━[37m━━  0s 19ms/step - loss: 0.0687 - root_mean_squared_error: 0.2621

<div class="k-default-codeblock">
```

```
</div>
 20/20 ━━━━━━━━━━━━━━━━━━━━ 4s 18ms/step - loss: 0.0689 - root_mean_squared_error: 0.2624





<div class="k-default-codeblock">
```
{'loss': 0.0697978064417839, 'root_mean_squared_error': 0.26419273018836975}

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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 277ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 278ms/step


<div class="k-default-codeblock">
```
b'Back to the Future (1985)': 3.60
b'20,000 Leagues Under the Sea (1954)': 3.50
b"Breakfast at Tiffany's (1961)": 3.55

```
</div>
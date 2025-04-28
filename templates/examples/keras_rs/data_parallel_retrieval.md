# Recommending movies: retrieval with data parallel training

**Author:** [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Retrieve movies using a two tower model (data parallel training).


<div class='example_version_banner keras_2'>ⓘ This example uses Keras 2</div>
<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_rs/ipynb/data_parallel_retrieval.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_rs/data_parallel_retrieval.py)



---
## Introduction

In this tutorial, we are going to train the exact same retrieval model as we
did in our
[basic retrieval](/keras_rs/examples/basic_retrieval/)
tutorial, but in a distributed way.

Distributed training is used to train models on multiple devices or machines
simultaneously, thereby reducing training time. Here, we focus on synchronous
data parallel training. Each accelerator (GPU/TPU) holds a complete replica
of the model, and sees a different mini-batch of the input data. Local gradients
are computed on each device, aggregated and used to compute a global gradient
update.

Before we begin, let's note down a few things:

1. The number of accelerators should be greater than 1.
2. The `keras.distribution` API works only with JAX. So, make sure you select
   JAX as your backend!


```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import random

import jax
import keras
import tensorflow as tf  # Needed only for the dataset
import tensorflow_datasets as tfds

import keras_rs
```

---
## Data Parallel

For the synchronous data parallelism strategy in distributed training,
we will use the `DataParallel` class present in the `keras.distribution`
API.


```python
devices = jax.devices()  # Assume it has >1 local devices.
data_parallel = keras.distribution.DataParallel(devices=devices)
```

Alternatively, you can choose to create the `DataParallel` object
using a 1D `DeviceMesh` object, like so:

```
mesh_1d = keras.distribution.DeviceMesh(
    shape=(len(devices),), axis_names=["data"], devices=devices
)
data_parallel = keras.distribution.DataParallel(device_mesh=mesh_1d)
```


```python
# Set the global distribution strategy.
keras.distribution.set_distribution(data_parallel)
```

---
## Preparing the dataset

Now that we are done defining the global distribution
strategy, the rest of the guide looks exactly the same
as the previous basic retrieval guide.

Let's load and prepare the dataset. Here too, we use the
MovieLens dataset.


```python
# Ratings data with user and movie data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

# User, movie counts for defining vocabularies.
users_count = (
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy()
)
movies_count = movies.cardinality().numpy()


# Preprocess dataset, and split it into train-test datasets.
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


shuffled_ratings = ratings.map(preprocess_rating).shuffle(
    100_000, seed=42, reshuffle_each_iteration=False
)
train_ratings = shuffled_ratings.take(80_000).batch(1000).cache()
test_ratings = shuffled_ratings.skip(80_000).take(20_000).batch(1000).cache()
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

Shuffling /root/tensorflow_datasets/movielens/100k-ratings/incomplete.H1I1XL_0.1.1/movielens-train.tfrecord*..…

WARNING:absl:Variant folder /root/tensorflow_datasets/movielens/100k-movies/0.1.1 has no dataset_info.json

Dataset movielens downloaded and prepared to /root/tensorflow_datasets/movielens/100k-ratings/0.1.1. Subsequent calls will reuse this data.
Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to /root/tensorflow_datasets/movielens/100k-movies/0.1.1...

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/1 [00:00<?, ? splits/s]

Generating train examples...: 0 examples [00:00, ? examples/s]

Shuffling /root/tensorflow_datasets/movielens/100k-movies/incomplete.16DZ87_0.1.1/movielens-train.tfrecord*...…

Dataset movielens downloaded and prepared to /root/tensorflow_datasets/movielens/100k-movies/0.1.1. Subsequent calls will reuse this data.

```
</div>
---
## Implementing the Model

We build a two-tower retrieval model. Therefore, we need to combine a
query tower for users and a candidate tower for movies. Note that we don't
have to change anything here from the previous basic retrieval tutorial.


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


```python
model = RetrievalModel(users_count + 1, movies_count + 1)
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.2))
```

Let's train the model. Evaluation takes a bit of time, so we only evaluate the
model every 5 epochs.


```python
history = model.fit(
    train_ratings, validation_data=test_ratings, validation_freq=5, epochs=50
)
```

<div class="k-default-codeblock">
```
Epoch 1/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  3:19 3s/step - loss: 0.4474

    
  5/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 10ms/step - loss: 0.4661
  3/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4602

    
  7/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 10ms/step - loss: 0.4692

    
  8/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 10ms/step - loss: 0.4705

    
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.4716 
  4/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 10ms/step - loss: 0.4634

    
  6/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4677

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 15ms/step - loss: 0.4543

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4725

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 8ms/step - loss: 0.4754 

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 8ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 8ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4769

<div class="k-default-codeblock">
```

```
</div>
 74/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4771

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - loss: 0.4772


<div class="k-default-codeblock">
```
Epoch 2/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4473

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4715 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4769

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4771

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4771


<div class="k-default-codeblock">
```
Epoch 3/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4472

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4724 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4769

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4770


<div class="k-default-codeblock">
```
Epoch 4/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4472

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4723 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4770


<div class="k-default-codeblock">
```
Epoch 5/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4713 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 3s 35ms/step - loss: 0.4769 - val_loss: 0.4835


<div class="k-default-codeblock">
```
Epoch 6/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4470

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4712 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4746

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 40/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 48/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4768


<div class="k-default-codeblock">
```
Epoch 7/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  2s 26ms/step - loss: 0.4469

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4711 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4747

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4767


<div class="k-default-codeblock">
```
Epoch 8/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4468

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4710 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4746

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4766


<div class="k-default-codeblock">
```
Epoch 9/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4467

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4709 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4743

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 74/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4765


<div class="k-default-codeblock">
```
Epoch 10/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4466

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4708 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4744

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4763 - val_loss: 0.4832


<div class="k-default-codeblock">
```
Epoch 11/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4465

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4706 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4740

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 74/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4762


<div class="k-default-codeblock">
```
Epoch 12/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.4463

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4704 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4738

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4747

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4760


<div class="k-default-codeblock">
```
Epoch 13/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4461

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4702 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4738

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4746

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4748

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4757


<div class="k-default-codeblock">
```
Epoch 14/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4458

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4709 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4737

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4744

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4746

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4748

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4755


<div class="k-default-codeblock">
```
Epoch 15/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4455

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4706 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4734

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4743

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4745

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4747

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4751 - val_loss: 0.4822


<div class="k-default-codeblock">
```
Epoch 16/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4452

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4702 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4730

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4737

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4739

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4743

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4744

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4746

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4747


<div class="k-default-codeblock">
```
Epoch 17/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4447

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4697 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4725

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4732

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4734

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4736

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4738

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4739

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4742


<div class="k-default-codeblock">
```
Epoch 18/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4442

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4691 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4719

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4726

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4728

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4730

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4732

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4733

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4735

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4736


<div class="k-default-codeblock">
```
Epoch 19/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4435

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4684 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4711

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4718

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4720

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4722

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4724

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4725

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4727

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4728


<div class="k-default-codeblock">
```
Epoch 20/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4426

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4674 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4702

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4709

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4711

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4713

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4715

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4716

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4717

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4718 - val_loss: 0.4787


<div class="k-default-codeblock">
```
Epoch 21/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4415

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4663 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4691

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4697

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4699

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4701

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4703

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4704

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4705

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4706


<div class="k-default-codeblock">
```
Epoch 22/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4402

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4640 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4675

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4682

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4684

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4686

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4688

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4689

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4690

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4691

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4691


<div class="k-default-codeblock">
```
Epoch 23/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4385

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4631 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4658

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4665

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4666

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4668

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4669

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4671

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4672

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4672


<div class="k-default-codeblock">
```
Epoch 24/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4364

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4609 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4636

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4642

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4644

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4645

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4647

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4647

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4648

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4649

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4649


<div class="k-default-codeblock">
```
Epoch 25/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4338

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4582 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4608

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4615

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4616

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4617

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4618

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4619

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4620

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4620 - val_loss: 0.4676


<div class="k-default-codeblock">
```
Epoch 26/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4306

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4548 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4574

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4580

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4581

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4582

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4583

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4583

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4584

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4584


<div class="k-default-codeblock">
```
Epoch 27/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4266

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4506 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4532

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4537

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4538

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4539

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4539

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4540

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4540

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4540


<div class="k-default-codeblock">
```
Epoch 28/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4217

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4446 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4477

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4484

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4485

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4486

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4486

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4486

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4486

<div class="k-default-codeblock">
```

```
</div>
 78/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4486

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4485


<div class="k-default-codeblock">
```
Epoch 29/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4157

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4384 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4416

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4421

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4422

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4422

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4421

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4421

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4420

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4419

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4419


<div class="k-default-codeblock">
```
Epoch 30/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4085

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4308 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4338

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4344

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4345

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4344

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4344

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4343

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4342

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4341

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4340 - val_loss: 0.4358


<div class="k-default-codeblock">
```
Epoch 31/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.3998

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4226 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4249

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4253

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4252

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4251

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4250

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4248

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4247

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4245


<div class="k-default-codeblock">
```
Epoch 32/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.3896

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4112 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4141

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4145

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4144

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4142

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4140

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4138

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4136

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4134

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4133


<div class="k-default-codeblock">
```
Epoch 33/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.3776

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3996 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4017

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4019

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4017

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4015

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4013

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4010

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4007

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4004


<div class="k-default-codeblock">
```
Epoch 34/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.3640

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3854 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3874

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3876

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.3874

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3872

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3868

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.3865

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.3862

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.3858

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3858


<div class="k-default-codeblock">
```
Epoch 35/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.3488

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3697 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3715

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3717

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3714

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3711

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3707

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.3704

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.3700

<div class="k-default-codeblock">
```

```
</div>
 78/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.3697

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.3695 - val_loss: 0.3659


<div class="k-default-codeblock">
```
Epoch 36/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.3321

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3518 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3542

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3543

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3540

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3536

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3533

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.3529

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.3524

<div class="k-default-codeblock">
```

```
</div>
 78/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.3520

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3519


<div class="k-default-codeblock">
```
Epoch 37/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.3144

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.3335 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3358

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3359

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.3357

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.3353

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.3348

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.3344

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.3340

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.3335

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3333


<div class="k-default-codeblock">
```
Epoch 38/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.2960

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3153 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3168

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3169

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3166

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3161

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3156

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.3151

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.3146

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.3142

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3142


<div class="k-default-codeblock">
```
Epoch 39/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.2776

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2957 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2977

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2979

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.2976

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.2971

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.2967

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.2962

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.2957

<div class="k-default-codeblock">
```

```
</div>
 78/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.2952

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2951


<div class="k-default-codeblock">
```
Epoch 40/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.2595

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2778 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2792

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2792

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2789

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2784

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2780

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.2775

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2770

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.2766

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.2765 - val_loss: 0.2729


<div class="k-default-codeblock">
```
Epoch 41/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.2423

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.2595 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2614

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2615

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.2613

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.2609

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2604

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.2600

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.2595

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.2592

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2590


<div class="k-default-codeblock">
```
Epoch 42/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.2263

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2430 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2449

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2450

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2447

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2444

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2439

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.2436

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.2432

<div class="k-default-codeblock">
```

```
</div>
 78/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.2428

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2427


<div class="k-default-codeblock">
```
Epoch 43/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.2117

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2280 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2298

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.2299

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.2298

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.2294

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.2291

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.2287

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.2283

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2280

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2278


<div class="k-default-codeblock">
```
Epoch 44/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.1986

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2144 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2160

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2163

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.2161

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.2158

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.2155

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.2152

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.2149

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2145

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2143


<div class="k-default-codeblock">
```
Epoch 45/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.1869

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2026 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2039

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2040

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2038

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2035

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2032

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.2029

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2025

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.2022 - val_loss: 0.2034


<div class="k-default-codeblock">
```
Epoch 46/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.1765

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1913 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1929

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1930

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.1929

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.1926

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.1923

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.1920

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.1918

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.1915

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1915


<div class="k-default-codeblock">
```
Epoch 47/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.1673

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1815 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1831

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1832

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.1831

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.1829

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.1826

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.1824

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.1821

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.1819

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1818


<div class="k-default-codeblock">
```
Epoch 48/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.1592

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1728 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1742

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1744

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.1743

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.1741

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.1739

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.1736

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.1734

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.1732

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1731


<div class="k-default-codeblock">
```
Epoch 49/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.1519

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1649 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1663

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1664

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.1663

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.1661

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.1659

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.1657

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.1655

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.1654

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1653


<div class="k-default-codeblock">
```
Epoch 50/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.1454

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.1577 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1591

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1592

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.1591

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.1589

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.1587

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.1585

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.1583

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.1582 - val_loss: 0.1621


---
## Making predictions

Now that we have a model, let's run inference and make predictions.


```python
movie_id_to_movie_title = {
    int(x["movie_id"]): x["movie_title"] for x in movies.as_numpy_iterator()
}
movie_id_to_movie_title[0] = ""  # Because id 0 is not in the dataset.
```

We then simply use the Keras `model.predict()` method. Under the hood, it calls
the `BruteForceRetrieval` layer to perform the actual retrieval.


```python
user_ids = random.sample(range(1, 101), len(devices))
predictions = model.predict(keras.ops.convert_to_tensor(user_ids))
predictions = keras.ops.convert_to_numpy(predictions["predictions"])

for user_id in user_ids:
    print(f"\n==Recommended movies for user {user_id}==")
    for movie_id in predictions[0]:
        print(movie_id_to_movie_title[movie_id])
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 217ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 218ms/step


    
<div class="k-default-codeblock">
```
==Recommended movies for user 35==
b'Godfather, The (1972)'
b'Contact (1997)'
b'Amadeus (1984)'
b'Usual Suspects, The (1995)'
b'Star Wars (1977)'
b'Alien (1979)'
b'Butch Cassidy and the Sundance Kid (1969)'
b'Aliens (1986)'
b'Princess Bride, The (1987)'
b'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 86==
b'Godfather, The (1972)'
b'Contact (1997)'
b'Amadeus (1984)'
b'Usual Suspects, The (1995)'
b'Star Wars (1977)'
b'Alien (1979)'
b'Butch Cassidy and the Sundance Kid (1969)'
b'Aliens (1986)'
b'Princess Bride, The (1987)'
b'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 51==
b'Godfather, The (1972)'
b'Contact (1997)'
b'Amadeus (1984)'
b'Usual Suspects, The (1995)'
b'Star Wars (1977)'
b'Alien (1979)'
b'Butch Cassidy and the Sundance Kid (1969)'
b'Aliens (1986)'
b'Princess Bride, The (1987)'
b'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 17==
b'Godfather, The (1972)'
b'Contact (1997)'
b'Amadeus (1984)'
b'Usual Suspects, The (1995)'
b'Star Wars (1977)'
b'Alien (1979)'
b'Butch Cassidy and the Sundance Kid (1969)'
b'Aliens (1986)'
b'Princess Bride, The (1987)'
b'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 97==
b'Godfather, The (1972)'
b'Contact (1997)'
b'Amadeus (1984)'
b'Usual Suspects, The (1995)'
b'Star Wars (1977)'
b'Alien (1979)'
b'Butch Cassidy and the Sundance Kid (1969)'
b'Aliens (1986)'
b'Princess Bride, The (1987)'
b'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 45==
b'Godfather, The (1972)'
b'Contact (1997)'
b'Amadeus (1984)'
b'Usual Suspects, The (1995)'
b'Star Wars (1977)'
b'Alien (1979)'
b'Butch Cassidy and the Sundance Kid (1969)'
b'Aliens (1986)'
b'Princess Bride, The (1987)'
b'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 50==
b'Godfather, The (1972)'
b'Contact (1997)'
b'Amadeus (1984)'
b'Usual Suspects, The (1995)'
b'Star Wars (1977)'
b'Alien (1979)'
b'Butch Cassidy and the Sundance Kid (1969)'
b'Aliens (1986)'
b'Princess Bride, The (1987)'
b'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 85==
b'Godfather, The (1972)'
b'Contact (1997)'
b'Amadeus (1984)'
b'Usual Suspects, The (1995)'
b'Star Wars (1977)'
b'Alien (1979)'
b'Butch Cassidy and the Sundance Kid (1969)'
b'Aliens (1986)'
b'Princess Bride, The (1987)'
b'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)'

```
</div>
And we're done! For data parallel training, all we had to do was add ~3-5 LoC.
The rest is exactly the same.


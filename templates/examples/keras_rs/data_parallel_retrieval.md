# Retrieval with data parallel training

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

Shuffling /root/tensorflow_datasets/movielens/100k-ratings/incomplete.HP2NR7_0.1.1/movielens-train.tfrecord*..…

WARNING:absl:Variant folder /root/tensorflow_datasets/movielens/100k-movies/0.1.1 has no dataset_info.json

Dataset movielens downloaded and prepared to /root/tensorflow_datasets/movielens/100k-ratings/0.1.1. Subsequent calls will reuse this data.
Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to /root/tensorflow_datasets/movielens/100k-movies/0.1.1...

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/1 [00:00<?, ? splits/s]

Generating train examples...: 0 examples [00:00, ? examples/s]

Shuffling /root/tensorflow_datasets/movielens/100k-movies/incomplete.NYP15O_0.1.1/movielens-train.tfrecord*...…

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
    
  8/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.4707
  3/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4606
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  2:04 2s/step - loss: 0.4479
  4/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.4637

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 14ms/step - loss: 0.4547

    
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.4718

    
  6/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 10ms/step - loss: 0.4679

    
  5/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 8ms/step - loss: 0.4663 
  7/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.4694 

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.4727

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 8ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4769

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4770

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4772

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.4773


<div class="k-default-codeblock">
```
Epoch 2/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4478

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4717 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4770

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4771

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4772


<div class="k-default-codeblock">
```
Epoch 3/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4478

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4717 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4762

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
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4769

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4771

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4771


<div class="k-default-codeblock">
```
Epoch 4/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4477

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4716 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4764

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
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4767

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
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4770

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
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4476

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
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4753

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
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4769

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 27ms/step - loss: 0.4770 - val_loss: 0.4835


<div class="k-default-codeblock">
```
Epoch 6/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4476

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
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4766

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
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4769


<div class="k-default-codeblock">
```
Epoch 7/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4475

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
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4768


<div class="k-default-codeblock">
```
Epoch 8/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4474

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4722 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4767


<div class="k-default-codeblock">
```
Epoch 9/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 19ms/step - loss: 0.4473

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
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4748

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
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4766


<div class="k-default-codeblock">
```
Epoch 10/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4472

<div class="k-default-codeblock">
```

```
</div>
  8/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 8ms/step - loss: 0.4699 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4744

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4765 - val_loss: 0.4832


<div class="k-default-codeblock">
```
Epoch 11/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4470

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4718 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4746

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4763


<div class="k-default-codeblock">
```
Epoch 12/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4469

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4716 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4745

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4762


<div class="k-default-codeblock">
```
Epoch 13/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4467

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4705 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4760


<div class="k-default-codeblock">
```
Epoch 14/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4465

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4712 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4740

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4747

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4757


<div class="k-default-codeblock">
```
Epoch 15/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4462

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4700 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4736

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4744

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4746

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4748

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4754 - val_loss: 0.4824


<div class="k-default-codeblock">
```
Epoch 16/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 19ms/step - loss: 0.4459

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
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4748

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
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4751


<div class="k-default-codeblock">
```
Epoch 17/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4455

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
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4738

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
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4742

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
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4745

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4746


<div class="k-default-codeblock">
```
Epoch 18/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4450

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
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4731

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4733

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4735

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4737

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4738

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4740

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4741


<div class="k-default-codeblock">
```
Epoch 19/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4444

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4690 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4718

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4725

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4726

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4728

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4730

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4731

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4733

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4734

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4734


<div class="k-default-codeblock">
```
Epoch 20/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 19ms/step - loss: 0.4437

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4673 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4707

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4716

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4718

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4720

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4722

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4723

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4725

<div class="k-default-codeblock">
```

```
</div>
 78/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4726

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4726 - val_loss: 0.4795


<div class="k-default-codeblock">
```
Epoch 21/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 19ms/step - loss: 0.4427

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4673 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4701

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4707

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4709

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4711

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4712

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4714

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4715

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4716


<div class="k-default-codeblock">
```
Epoch 22/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4416

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4652 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4685

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4693

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4696

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4697

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4699

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4700

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4701

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4703

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4703


<div class="k-default-codeblock">
```
Epoch 23/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4401

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4636 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4672

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4679

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4681

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4683

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4684

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4685

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4686

<div class="k-default-codeblock">
```

```
</div>
 78/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4687

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4687


<div class="k-default-codeblock">
```
Epoch 24/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4383

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4618 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4653

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4660

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4661

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4663

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4664

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4665

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4666

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4667

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4667


<div class="k-default-codeblock">
```
Epoch 25/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4361

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4603 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4631

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4637

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4638

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4639

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4640

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4641

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4642

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4642 - val_loss: 0.4701


<div class="k-default-codeblock">
```
Epoch 26/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4333

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4574 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4601

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4607

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4608

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4610

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4610

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4611

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4612

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4612


<div class="k-default-codeblock">
```
Epoch 27/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4299

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4538 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4565

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4571

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4572

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4573

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4573

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4574

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4574

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4574


<div class="k-default-codeblock">
```
Epoch 28/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4256

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4485 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4517

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4525

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4526

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4527

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4527

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4527

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4527

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4527

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4527

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4527


<div class="k-default-codeblock">
```
Epoch 29/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.4204

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4440 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4466

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4472

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4470


<div class="k-default-codeblock">
```
Epoch 30/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 19ms/step - loss: 0.4141

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4374 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4399

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4404

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4404

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4404

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4403

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4402

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4402

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4401 - val_loss: 0.4427


<div class="k-default-codeblock">
```
Epoch 31/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 19ms/step - loss: 0.4064

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4295 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4319

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4323

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4323

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4322

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4321

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4320

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4319

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4317


<div class="k-default-codeblock">
```
Epoch 32/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.3973

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4200 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4223

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4227

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4226

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4225

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4224

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4222

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4220

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4218


<div class="k-default-codeblock">
```
Epoch 33/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.3866

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4089 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4111

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4114

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4113

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4111

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4109

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4107

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4104

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4102


<div class="k-default-codeblock">
```
Epoch 34/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.3742

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3960 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3981

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3984

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3982

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3979

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3977

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.3974

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.3971

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.3968


<div class="k-default-codeblock">
```
Epoch 35/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 19ms/step - loss: 0.3601

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3813 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3834

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3836

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3833

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3830

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3827

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.3823

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.3820

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.3817 - val_loss: 0.3787


<div class="k-default-codeblock">
```
Epoch 36/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.3443

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3651 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3670

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3671

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3668

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3665

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3661

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.3657

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.3653

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.3649


<div class="k-default-codeblock">
```
Epoch 37/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 19ms/step - loss: 0.3273

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3475 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3493

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3494

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3490

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3487

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3482

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.3478

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.3473

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.3469


<div class="k-default-codeblock">
```
Epoch 38/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.3093

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 8ms/step - loss: 0.3282 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.3305

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.3306

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3303

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3299

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3294

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.3289

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.3285

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3280


<div class="k-default-codeblock">
```
Epoch 39/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 19ms/step - loss: 0.2907

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3098 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3114

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3114

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3111

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3106

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3101

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.3096

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.3091

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.3087


<div class="k-default-codeblock">
```
Epoch 40/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.2722

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2907 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2923

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2923

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2919

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2915

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2910

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.2905

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2900

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.2896 - val_loss: 0.2856


<div class="k-default-codeblock">
```
Epoch 41/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.2542

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2722 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2737

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2737

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2734

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2729

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2725

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.2720

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2715

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.2711


<div class="k-default-codeblock">
```
Epoch 42/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.2372

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2547 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2561

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2562

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2558

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2554

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2550

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.2545

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2540

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2537


<div class="k-default-codeblock">
```
Epoch 43/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.2215

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2384 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2399

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2399

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2396

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2392

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2388

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.2384

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2380

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2376


<div class="k-default-codeblock">
```
Epoch 44/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.2072

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2236 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2250

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2251

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2248

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2244

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2240

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.2237

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2233

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.2230


<div class="k-default-codeblock">
```
Epoch 45/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.1944

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2103 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2116

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2117

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2115

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2111

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2108

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.2104

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2101

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.2098 - val_loss: 0.2106


<div class="k-default-codeblock">
```
Epoch 46/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.1831

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1984 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1997

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1998

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.1995

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.1993

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.1990

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.1987

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.1983

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.1981

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1980


<div class="k-default-codeblock">
```
Epoch 47/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.1730

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1877 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1890

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1891

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.1888

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.1886

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.1884

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.1881

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.1878

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.1875

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1875


<div class="k-default-codeblock">
```
Epoch 48/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.1641

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1782 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1794

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1795

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.1793

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.1791

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.1788

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.1786

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.1783

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1781


<div class="k-default-codeblock">
```
Epoch 49/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.1562

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1693 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1707

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1709

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.1708

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.1706

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.1704

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.1702

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.1700

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.1697

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1696


<div class="k-default-codeblock">
```
Epoch 50/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.1492

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1620 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1631

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1631

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.1630

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.1628

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.1626

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.1624

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.1622

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.1620 - val_loss: 0.1660


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
user_ids = random.sample(range(1, 1001), len(devices))
predictions = model.predict(keras.ops.convert_to_tensor(user_ids))
predictions = keras.ops.convert_to_numpy(predictions["predictions"])

for i, user_id in enumerate(user_ids):
    print(f"\n==Recommended movies for user {user_id}==")
    for movie_id in predictions[i]:
        print(movie_id_to_movie_title[movie_id])
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 204ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 205ms/step


    
<div class="k-default-codeblock">
```
==Recommended movies for user 449==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Silence of the Lambs, The (1991)'
b'Shawshank Redemption, The (1994)'
b'Pulp Fiction (1994)'
b'Raiders of the Lost Ark (1981)'
b"Schindler's List (1993)"
b'Blade Runner (1982)'
b"One Flew Over the Cuckoo's Nest (1975)"
b'Casablanca (1942)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 681==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Silence of the Lambs, The (1991)'
b'Raiders of the Lost Ark (1981)'
b'Return of the Jedi (1983)'
b'Pulp Fiction (1994)'
b"Schindler's List (1993)"
b'Empire Strikes Back, The (1980)'
b'Shawshank Redemption, The (1994)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 151==
b'Princess Bride, The (1987)'
b'Pulp Fiction (1994)'
b'English Patient, The (1996)'
b'Alien (1979)'
b'Raiders of the Lost Ark (1981)'
b'Willy Wonka and the Chocolate Factory (1971)'
b'Amadeus (1984)'
b'Liar Liar (1997)'
b'Psycho (1960)'
b"It's a Wonderful Life (1946)"
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 442==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Silence of the Lambs, The (1991)'
b'Raiders of the Lost Ark (1981)'
b'Return of the Jedi (1983)'
b'Pulp Fiction (1994)'
b'Empire Strikes Back, The (1980)'
b"Schindler's List (1993)"
b'Shawshank Redemption, The (1994)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 134==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Silence of the Lambs, The (1991)'
b'Raiders of the Lost Ark (1981)'
b'Pulp Fiction (1994)'
b'Return of the Jedi (1983)'
b'Empire Strikes Back, The (1980)'
b'Twelve Monkeys (1995)'
b'Contact (1997)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 853==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Raiders of the Lost Ark (1981)'
b'Silence of the Lambs, The (1991)'
b'Return of the Jedi (1983)'
b'Pulp Fiction (1994)'
b"Schindler's List (1993)"
b'Empire Strikes Back, The (1980)'
b'Shawshank Redemption, The (1994)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 707==
b'Star Wars (1977)'
b'Raiders of the Lost Ark (1981)'
b'Toy Story (1995)'
b"Schindler's List (1993)"
b'Empire Strikes Back, The (1980)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Return of the Jedi (1983)'
b'Terminator, The (1984)'
b'Princess Bride, The (1987)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 511==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Raiders of the Lost Ark (1981)'
b'Silence of the Lambs, The (1991)'
b'Return of the Jedi (1983)'
b"Schindler's List (1993)"
b'Empire Strikes Back, The (1980)'
b'Pulp Fiction (1994)'
b'Shawshank Redemption, The (1994)'

```
</div>
And we're done! For data parallel training, all we had to do was add ~3-5 LoC.
The rest is exactly the same.


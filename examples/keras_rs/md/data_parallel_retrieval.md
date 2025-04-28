# Recommending movies: retrieval with data parallel training

**Author:** [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Retrieve movies using a two tower model (data parallel training).


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
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  2:23 2s/step - loss: 0.4476
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.4716
  7/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 10ms/step - loss: 0.4692
  3/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4604

    
  8/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.4705
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 14ms/step - loss: 0.4545
  4/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 10ms/step - loss: 0.4635

    
  5/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.4661 
  6/80 ━[37m━━━━━━━━━━━━━━━━━━━  0s 10ms/step - loss: 0.4678

<div class="k-default-codeblock">
```

```
</div>
 13/80 ━━━[37m━━━━━━━━━━━━━━━━━  0s 9ms/step - loss: 0.4741 

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 8ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 40/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4766

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
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4770

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 7ms/step - loss: 0.4771

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.4772


<div class="k-default-codeblock">
```
Epoch 2/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.4476

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4716 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4764

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
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4768

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
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4771

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
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4475

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4715 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4770

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
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4475

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4715 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4769

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4770


<div class="k-default-codeblock">
```
Epoch 5/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4474

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4714 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4748

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 3s 42ms/step - loss: 0.4770 - val_loss: 0.4836


<div class="k-default-codeblock">
```
Epoch 6/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.4473

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
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4748

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4766

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
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4769


<div class="k-default-codeblock">
```
Epoch 7/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  2s 27ms/step - loss: 0.4473

<div class="k-default-codeblock">
```

```
</div>
  8/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4701 

<div class="k-default-codeblock">
```

```
</div>
 15/80 ━━━[37m━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4742

<div class="k-default-codeblock">
```

```
</div>
 22/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 39/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 7ms/step - loss: 0.4760

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
 56/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4768


<div class="k-default-codeblock">
```
Epoch 8/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4472

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
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4746

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 78/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4767


<div class="k-default-codeblock">
```
Epoch 9/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4471

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
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4745

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4767


<div class="k-default-codeblock">
```
Epoch 10/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4470

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
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4744

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 74/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4766 - val_loss: 0.4835


<div class="k-default-codeblock">
```
Epoch 11/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.4469

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4709 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4743

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4764


<div class="k-default-codeblock">
```
Epoch 12/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4468

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4708 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4742

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4763


<div class="k-default-codeblock">
```
Epoch 13/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 24ms/step - loss: 0.4466

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4706 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4762


<div class="k-default-codeblock">
```
Epoch 14/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.4465

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4705 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4739

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4748

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4760


<div class="k-default-codeblock">
```
Epoch 15/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4463

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4703 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4739

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4747

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
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4758 - val_loss: 0.4830


<div class="k-default-codeblock">
```
Epoch 16/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4461

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4701 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4737

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4745

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4747

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4756


<div class="k-default-codeblock">
```
Epoch 17/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.4458

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4698 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4732

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4744

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.4745

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.4747

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4753


<div class="k-default-codeblock">
```
Epoch 18/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4455

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4694 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4729

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4738

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4740

<div class="k-default-codeblock">
```

```
</div>
 38/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 7ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 47/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 7ms/step - loss: 0.4743

<div class="k-default-codeblock">
```

```
</div>
 56/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.4745

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.4747

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4748

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4749


<div class="k-default-codeblock">
```
Epoch 19/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4451

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4690 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4727

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4734

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4736

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4738

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4740

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4742

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4743

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4744

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4745


<div class="k-default-codeblock">
```
Epoch 20/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4446

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4685 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4720

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4728

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4731

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4733

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4735

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4736

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4737

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4739

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4739 - val_loss: 0.4812


<div class="k-default-codeblock">
```
Epoch 21/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4439

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4679 

<div class="k-default-codeblock">
```

```
</div>
 16/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4711

<div class="k-default-codeblock">
```

```
</div>
 24/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4721

<div class="k-default-codeblock">
```

```
</div>
 31/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4724

<div class="k-default-codeblock">
```

```
</div>
 39/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 7ms/step - loss: 0.4726

<div class="k-default-codeblock">
```

```
</div>
 47/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 7ms/step - loss: 0.4727

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 7ms/step - loss: 0.4729

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 7ms/step - loss: 0.4730

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 7ms/step - loss: 0.4731

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.4732

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4733


<div class="k-default-codeblock">
```
Epoch 22/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4432

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4671 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4705

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4714

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4716

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4718

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4720

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4721

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4722

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4724

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4724


<div class="k-default-codeblock">
```
Epoch 23/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.4422

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4661 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4697

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4705

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4706

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4708

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4710

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4711

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4712

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4714

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4714


<div class="k-default-codeblock">
```
Epoch 24/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4411

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4659 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4685

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4693

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4694

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4696

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4697

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4699

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4700

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4701

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4701


<div class="k-default-codeblock">
```
Epoch 25/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4396

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4634 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4668

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4676

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4679

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4680

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4682

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4683

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4684

<div class="k-default-codeblock">
```

```
</div>
 76/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4685

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4685 - val_loss: 0.4752


<div class="k-default-codeblock">
```
Epoch 26/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4378

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4616 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4651

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4658

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4660

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4661

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4662

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4663

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4664

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4665

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4665


<div class="k-default-codeblock">
```
Epoch 27/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4355

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4592 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4628

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4634

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.4636

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4637

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4638

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4639

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4639

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4640

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4640


<div class="k-default-codeblock">
```
Epoch 28/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4327

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4563 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4598

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4605

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4606

<div class="k-default-codeblock">
```

```
</div>
 43/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4607

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4608

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4609

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4609

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4610

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4610


<div class="k-default-codeblock">
```
Epoch 29/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4293

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4528 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4562

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4569

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4570

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4571

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4571

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4571

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4572

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4571


<div class="k-default-codeblock">
```
Epoch 30/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4250

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4483 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4517

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4524

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4524

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4525

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4525

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4525

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4525

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4525

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4525 - val_loss: 0.4569


<div class="k-default-codeblock">
```
Epoch 31/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.4198

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.4429 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4461

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4468

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4469

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4469

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4469

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4469

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4468

<div class="k-default-codeblock">
```

```
</div>
 74/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4468

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4467


<div class="k-default-codeblock">
```
Epoch 32/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4134

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4364 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4395

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4401

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4402

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.4401

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.4401

<div class="k-default-codeblock">
```

```
</div>
 58/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.4400

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.4399

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.4398

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4398


<div class="k-default-codeblock">
```
Epoch 33/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.4057

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4284 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4316

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4321

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4320

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4320

<div class="k-default-codeblock">
```

```
</div>
 52/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4319

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4317

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4316

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4315

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4314


<div class="k-default-codeblock">
```
Epoch 34/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.3965

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4190 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4220

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4224

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.4224

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4223

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4221

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4219

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4217

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.4215

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4214


<div class="k-default-codeblock">
```
Epoch 35/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.3858

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4078 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4108

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.4112

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.4110

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.4108

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4106

<div class="k-default-codeblock">
```

```
</div>
 61/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.4104

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.4101

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.4099

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4098 - val_loss: 0.4092


<div class="k-default-codeblock">
```
Epoch 36/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 24ms/step - loss: 0.3733

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.3949 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3978

<div class="k-default-codeblock">
```

```
</div>
 23/80 ━━━━━[37m━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.3980

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.3980

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.3977

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.3975

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.3972

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.3968

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.3965

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.3964


<div class="k-default-codeblock">
```
Epoch 37/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.3591

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3803 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3829

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3833

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.3831

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.3828

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.3825

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.3821

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.3817

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.3814

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3812


<div class="k-default-codeblock">
```
Epoch 38/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.3433

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.3641 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3667

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3669

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.3666

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3662

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3658

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.3654

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.3649

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.3645

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3644


<div class="k-default-codeblock">
```
Epoch 39/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.3262

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.3465 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3490

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3491

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.3488

<div class="k-default-codeblock">
```

```
</div>
 44/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3484

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3479

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.3474

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.3469

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.3465

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3464


<div class="k-default-codeblock">
```
Epoch 40/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.3081

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3286 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3303

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3303

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3299

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3295

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3290

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.3285

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.3280

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.3275 - val_loss: 0.3230


<div class="k-default-codeblock">
```
Epoch 41/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.2895

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3088 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3111

<div class="k-default-codeblock">
```

```
</div>
 27/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.3112

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.3108

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.3103

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.3098

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.3092

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.3087

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3082


<div class="k-default-codeblock">
```
Epoch 42/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.2710

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2905 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2921

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2920

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2917

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2912

<div class="k-default-codeblock">
```

```
</div>
 53/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2907

<div class="k-default-codeblock">
```

```
</div>
 62/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.2902

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.2897

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.2892

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2891


<div class="k-default-codeblock">
```
Epoch 43/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.2530

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2720 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2735

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2735

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.2731

<div class="k-default-codeblock">
```

```
</div>
 45/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.2727

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.2722

<div class="k-default-codeblock">
```

```
</div>
 63/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.2716

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.2712

<div class="k-default-codeblock">
```

```
</div>
 79/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.2708

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2707


<div class="k-default-codeblock">
```
Epoch 44/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.2360

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.2539 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.2558

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.2560

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.2558

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.2554

<div class="k-default-codeblock">
```

```
</div>
 49/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.2550

<div class="k-default-codeblock">
```

```
</div>
 57/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 7ms/step - loss: 0.2545

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 7ms/step - loss: 0.2541

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 7ms/step - loss: 0.2537

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.2533


<div class="k-default-codeblock">
```
Epoch 45/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.2203

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2377 

<div class="k-default-codeblock">
```

```
</div>
 18/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2397

<div class="k-default-codeblock">
```

```
</div>
 26/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2397

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.2395

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.2391

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.2387

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.2383

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.2379

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2375

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.2373 - val_loss: 0.2361


<div class="k-default-codeblock">
```
Epoch 46/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 22ms/step - loss: 0.2060

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2229 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2248

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.2249

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 6ms/step - loss: 0.2248

<div class="k-default-codeblock">
```

```
</div>
 42/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 6ms/step - loss: 0.2244

<div class="k-default-codeblock">
```

```
</div>
 51/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 6ms/step - loss: 0.2240

<div class="k-default-codeblock">
```

```
</div>
 60/80 ━━━━━━━━━━━━━━━[37m━━━━━  0s 6ms/step - loss: 0.2236

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 6ms/step - loss: 0.2232

<div class="k-default-codeblock">
```

```
</div>
 77/80 ━━━━━━━━━━━━━━━━━━━[37m━  0s 6ms/step - loss: 0.2228

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2227


<div class="k-default-codeblock">
```
Epoch 47/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 23ms/step - loss: 0.1933

<div class="k-default-codeblock">
```

```
</div>
  9/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.2096 

<div class="k-default-codeblock">
```

```
</div>
 17/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.2114

<div class="k-default-codeblock">
```

```
</div>
 25/80 ━━━━━━[37m━━━━━━━━━━━━━━  0s 7ms/step - loss: 0.2116

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 7ms/step - loss: 0.2115

<div class="k-default-codeblock">
```

```
</div>
 41/80 ━━━━━━━━━━[37m━━━━━━━━━━  0s 7ms/step - loss: 0.2112

<div class="k-default-codeblock">
```

```
</div>
 50/80 ━━━━━━━━━━━━[37m━━━━━━━━  0s 7ms/step - loss: 0.2108

<div class="k-default-codeblock">
```

```
</div>
 59/80 ━━━━━━━━━━━━━━[37m━━━━━━  0s 6ms/step - loss: 0.2104

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.2101

<div class="k-default-codeblock">
```

```
</div>
 75/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.2098

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2096


<div class="k-default-codeblock">
```
Epoch 48/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 20ms/step - loss: 0.1820

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1983 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1996

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1997

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 6ms/step - loss: 0.1994

<div class="k-default-codeblock">
```

```
</div>
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.1991

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.1988

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.1984

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.1981

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.1978


<div class="k-default-codeblock">
```
Epoch 49/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.1720

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1876 

<div class="k-default-codeblock">
```

```
</div>
 19/80 ━━━━[37m━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1889

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1890

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
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.1885

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.1882

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.1879

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.1876

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.1874


<div class="k-default-codeblock">
```
Epoch 50/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  1s 21ms/step - loss: 0.1631

<div class="k-default-codeblock">
```

```
</div>
 10/80 ━━[37m━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1781 

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
 46/80 ━━━━━━━━━━━[37m━━━━━━━━━  0s 6ms/step - loss: 0.1790

<div class="k-default-codeblock">
```

```
</div>
 55/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.1787

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 6ms/step - loss: 0.1784

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 6ms/step - loss: 0.1782

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.1780 - val_loss: 0.1810


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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 207ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 208ms/step


    
<div class="k-default-codeblock">
```
==Recommended movies for user 254==
b'Star Wars (1977)'
b'Raiders of the Lost Ark (1981)'
b'Fargo (1996)'
b'Silence of the Lambs, The (1991)'
b'Godfather, The (1972)'
b'Return of the Jedi (1983)'
b"Schindler's List (1993)"
b'Pulp Fiction (1994)'
b'Empire Strikes Back, The (1980)'
b'Toy Story (1995)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 148==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Silence of the Lambs, The (1991)'
b'Godfather, The (1972)'
b'Raiders of the Lost Ark (1981)'
b'Return of the Jedi (1983)'
b'Empire Strikes Back, The (1980)'
b"Schindler's List (1993)"
b'Pulp Fiction (1994)'
b'Shawshank Redemption, The (1994)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 176==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Silence of the Lambs, The (1991)'
b'Raiders of the Lost Ark (1981)'
b"Schindler's List (1993)"
b'Godfather, The (1972)'
b'Return of the Jedi (1983)'
b'Empire Strikes Back, The (1980)'
b'Toy Story (1995)'
b'Pulp Fiction (1994)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 212==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Raiders of the Lost Ark (1981)'
b'Silence of the Lambs, The (1991)'
b"Schindler's List (1993)"
b'Pulp Fiction (1994)'
b'Return of the Jedi (1983)'
b'Godfather, The (1972)'
b'Empire Strikes Back, The (1980)'
b'Toy Story (1995)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 259==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Silence of the Lambs, The (1991)'
b'Raiders of the Lost Ark (1981)'
b'Godfather, The (1972)'
b"Schindler's List (1993)"
b'Return of the Jedi (1983)'
b'Pulp Fiction (1994)'
b'Empire Strikes Back, The (1980)'
b'Shawshank Redemption, The (1994)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 77==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Silence of the Lambs, The (1991)'
b'Godfather, The (1972)'
b'Raiders of the Lost Ark (1981)'
b'Return of the Jedi (1983)'
b"Schindler's List (1993)"
b'Pulp Fiction (1994)'
b'Empire Strikes Back, The (1980)'
b'Shawshank Redemption, The (1994)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 579==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Silence of the Lambs, The (1991)'
b'Raiders of the Lost Ark (1981)'
b'Godfather, The (1972)'
b"Schindler's List (1993)"
b'Return of the Jedi (1983)'
b'Empire Strikes Back, The (1980)'
b'Pulp Fiction (1994)'
b'Princess Bride, The (1987)'
```
</div>
    
<div class="k-default-codeblock">
```
==Recommended movies for user 897==
b'Star Wars (1977)'
b'Raiders of the Lost Ark (1981)'
b'Silence of the Lambs, The (1991)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b"Schindler's List (1993)"
b'Return of the Jedi (1983)'
b'Empire Strikes Back, The (1980)'
b'Pulp Fiction (1994)'
b'Toy Story (1995)'

```
</div>
And we're done! For data parallel training, all we had to do was add ~3-5 LoC.
The rest is exactly the same.

# Faster retrieval with Scalable Nearest Neighbours (ScANN)

**Author:** [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Using ScANN for faster retrieval.


<div class='example_version_banner keras_2'>ⓘ This example uses Keras 2</div>
<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_rs/ipynb/scann.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_rs/scann.py)



---
## Introduction

Retrieval models are designed to quickly identify a small set of highly relevant
candidates from vast pools of data, often comprising millions or even hundreds
of millions of items. To effectively respond to the user's context and behavior
in real time, these models must perform this task in just milliseconds.

Approximate nearest neighbor (ANN) search is the key technology that enables
this level of efficiency. In this tutorial, we'll demonstrate how to leverage
ScANN—a cutting-edge nearest neighbor retrieval library—to effortlessly scale
retrieval for millions of items.

[ScANN](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/),
developed by Google Research, is a high-performance library designed for
dense vector similarity search at scale. It efficiently indexes a database of
candidate embeddings, enabling rapid search during inference. By leveraging
advanced vector compression techniques and finely tuned algorithms, ScaNN
strikes an optimal balance between speed and accuracy. As a result, it can
significantly outperform brute-force search methods, delivering fast retrieval
with minimal loss in accuracy.

We will start with the same code as the
[basic retrieval example](/keras_rs/examples/basic_retrieval/).
Data processing, model building, and training remain exactly the same. Feel free
to skip this part if you have gone over the basic retrieval example before.

Note: ScANN does not have its own separate layer in KerasRS because the ScANN
library is TensorFlow-only. Here, in this example, we directly use the ScANN
library and demonstrate its usage with KerasRS.

---
## Imports

Let's install the `scann` library and import all necessary packages. We will
also set the backend to JAX.


```python
# ruff: noqa: E402
```


```python
!pip install -q scann
```

<div class="k-default-codeblock">
```
[?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/11.8 MB [31m? eta [36m-:--:--
```
</div>
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01

    
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01

    
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01

    
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/11.8 MB [31m126.7 MB/s eta [36m0:00:01
[2K   [91m━━━━━━━[90m╺[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/11.8 MB [31m2.8 MB/s eta [36m0:00:04

    
[2K   [91m━━━━━━━[90m╺[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/11.8 MB [31m2.8 MB/s eta [36m0:00:04
[2K   [91m━━━━━━━[90m╺[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/11.8 MB [31m2.8 MB/s eta [36m0:00:04
[2K   [91m━━━━━━━[90m╺[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/11.8 MB [31m2.8 MB/s eta [36m0:00:04
[2K   [91m━━━━━━━[90m╺[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/11.8 MB [31m2.8 MB/s eta [36m0:00:04
[2K   [91m━━━━━━━[90m╺[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/11.8 MB [31m2.8 MB/s eta [36m0:00:04
[2K   [91m━━━━━━━[90m╺[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/11.8 MB [31m2.8 MB/s eta [36m0:00:04

    
[2K   [91m━━━━━━━━━━━━━━[90m╺[90m━━━━━━━━━━━━━━━━━━━━━━━━━ 4.2/11.8 MB [31m4.2 MB/s eta [36m0:00:02
[2K   [91m━━━━━━━━━━━━━━━━━━[91m╸[90m━━━━━━━━━━━━━━━━━━━━━ 5.6/11.8 MB [31m5.3 MB/s eta [36m0:00:02
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[91m╸[90m━━━━━━━━ 9.4/11.8 MB [31m8.9 MB/s eta [36m0:00:01
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[91m╸[90m━━━━ 10.5/11.8 MB [31m9.3 MB/s eta [36m0:00:01
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[91m╸[90m━━━━ 10.5/11.8 MB [31m9.3 MB/s eta [36m0:00:01

    
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[91m╸[90m━━━━ 10.5/11.8 MB [31m9.3 MB/s eta [36m0:00:01
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[91m╸[90m━━━━ 10.5/11.8 MB [31m9.3 MB/s eta [36m0:00:01
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[91m╸[90m━━━━ 10.5/11.8 MB [31m9.3 MB/s eta [36m0:00:01
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[91m╸[90m━━━━ 10.5/11.8 MB [31m9.3 MB/s eta [36m0:00:01
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[91m╸[90m━━━━ 10.5/11.8 MB [31m9.3 MB/s eta [36m0:00:01
[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[91m╸ 11.8/11.8 MB [31m17.3 MB/s eta [36m0:00:01
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.8/11.8 MB [31m16.4 MB/s eta [36m0:00:00
    [?25h


```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # `"tensorflow"`/`"torch"`

import time
import uuid

import keras
import tensorflow as tf  # Needed for the dataset
import tensorflow_datasets as tfds
from scann import scann_ops

import keras_rs
```

---
## Preparing the dataset


```python
# Ratings data with user and movie data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

# Get user and movie counts so that we can define embedding layers for both.
users_count = (
    ratings.map(lambda x: tf.strings.to_number(x["user_id"], out_type=tf.int32))
    .reduce(tf.constant(0, tf.int32), tf.maximum)
    .numpy()
)

movies_count = movies.cardinality().numpy()


# Preprocess the dataset, by selecting only the relevant columns.
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
# Train-test split.
train_ratings = shuffled_ratings.take(80_000).batch(1000).cache()
test_ratings = shuffled_ratings.skip(80_000).take(20_000).batch(1000).cache()
```

---
## Implementing the Model


```python

class RetrievalModel(keras.Model):
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

        self.loss_fn = keras.losses.MeanSquaredError()

    def build(self, input_shape):
        self.user_embedding.build(input_shape)
        self.candidate_embedding.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training=False):
        user_embeddings = self.user_embedding(inputs)
        result = {
            "user_embeddings": user_embeddings,
        }
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
## Training the model


```python
model = RetrievalModel(users_count + 1000, movies_count + 1000)
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))

history = model.fit(
    train_ratings, validation_data=test_ratings, validation_freq=5, epochs=50
)
```

<div class="k-default-codeblock">
```
Epoch 1/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  2:34 2s/step - loss: 0.4476

<div class="k-default-codeblock">
```

```
</div>
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  17s 223ms/step - loss: 0.4543

<div class="k-default-codeblock">
```

```
</div>
 28/80 ━━━━━━━[37m━━━━━━━━━━━━━  0s 10ms/step - loss: 0.4760  

<div class="k-default-codeblock">
```

```
</div>
 54/80 ━━━━━━━━━━━━━[37m━━━━━━━  0s 6ms/step - loss: 0.4767 

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.4772

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 6ms/step - loss: 0.4772


<div class="k-default-codeblock">
```
Epoch 2/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  17s 222ms/step - loss: 0.4476

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4761   

<div class="k-default-codeblock">
```

```
</div>
 64/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4769

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4771


<div class="k-default-codeblock">
```
Epoch 3/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4475

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.4542

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 2ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4769

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4771


<div class="k-default-codeblock">
```
Epoch 4/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 8ms/step - loss: 0.4475

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 2ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - loss: 0.4769

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4771


<div class="k-default-codeblock">
```
Epoch 5/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4475

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.4541

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 27ms/step - loss: 0.4770 - val_loss: 0.4836


<div class="k-default-codeblock">
```
Epoch 6/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.4474

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 2ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4770


<div class="k-default-codeblock">
```
Epoch 7/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.4474

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4768

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4770


<div class="k-default-codeblock">
```
Epoch 8/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4474

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4769


<div class="k-default-codeblock">
```
Epoch 9/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4474

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4767

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4769


<div class="k-default-codeblock">
```
Epoch 10/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4473

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4769 - val_loss: 0.4836


<div class="k-default-codeblock">
```
Epoch 11/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4473

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4768


<div class="k-default-codeblock">
```
Epoch 12/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4473

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4768


<div class="k-default-codeblock">
```
Epoch 13/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4472

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.4539

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4766

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4768


<div class="k-default-codeblock">
```
Epoch 14/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.4472

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4765

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4767


<div class="k-default-codeblock">
```
Epoch 15/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 65/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4767 - val_loss: 0.4835


<div class="k-default-codeblock">
```
Epoch 16/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4767


<div class="k-default-codeblock">
```
Epoch 17/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4471

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.4537

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4764

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4766


<div class="k-default-codeblock">
```
Epoch 18/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.4470

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4766


<div class="k-default-codeblock">
```
Epoch 19/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.4470

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4763

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4765


<div class="k-default-codeblock">
```
Epoch 20/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4469

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4765 - val_loss: 0.4835


<div class="k-default-codeblock">
```
Epoch 21/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4469

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4764


<div class="k-default-codeblock">
```
Epoch 22/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4468

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.4535

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 2ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4762

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4763


<div class="k-default-codeblock">
```
Epoch 23/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4468

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4761

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4763


<div class="k-default-codeblock">
```
Epoch 24/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4467

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4753

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4760

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4762


<div class="k-default-codeblock">
```
Epoch 25/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4466

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4759

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4761 - val_loss: 0.4833


<div class="k-default-codeblock">
```
Epoch 26/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4466

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4761


<div class="k-default-codeblock">
```
Epoch 27/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4465

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4760


<div class="k-default-codeblock">
```
Epoch 28/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4464

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.4530

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4750

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - loss: 0.4758

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4759


<div class="k-default-codeblock">
```
Epoch 29/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4463

<div class="k-default-codeblock">
```

```
</div>
 38/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - loss: 0.4756

<div class="k-default-codeblock">
```

```
</div>
 73/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - loss: 0.4757

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4758


<div class="k-default-codeblock">
```
Epoch 30/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4462

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4747

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4755

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4757 - val_loss: 0.4830


<div class="k-default-codeblock">
```
Epoch 31/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4461

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - loss: 0.4746

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4754

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4755


<div class="k-default-codeblock">
```
Epoch 32/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4460

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4744

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4752

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4754


<div class="k-default-codeblock">
```
Epoch 33/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.4458

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.4524

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 2ms/step - loss: 0.4743

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 2ms/step - loss: 0.4744

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 2ms/step - loss: 0.4751

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4753


<div class="k-default-codeblock">
```
Epoch 34/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4457

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4749

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4751


<div class="k-default-codeblock">
```
Epoch 35/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4455

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4740

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4747

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4747

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4749 - val_loss: 0.4823


<div class="k-default-codeblock">
```
Epoch 36/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4453

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4738

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4745

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4745

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4747


<div class="k-default-codeblock">
```
Epoch 37/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4451

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - loss: 0.4736

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4743

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4745


<div class="k-default-codeblock">
```
Epoch 38/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4449

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4734

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4741

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4743


<div class="k-default-codeblock">
```
Epoch 39/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4446

<div class="k-default-codeblock">
```

```
</div>
 37/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4731

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4738

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.4740


<div class="k-default-codeblock">
```
Epoch 40/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4443

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - loss: 0.4509

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4727

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4734

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4737 - val_loss: 0.4812


<div class="k-default-codeblock">
```
Epoch 41/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4440

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 1ms/step - loss: 0.4725

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4732

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4733


<div class="k-default-codeblock">
```
Epoch 42/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4437

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - loss: 0.4721

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4728

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4728

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4730


<div class="k-default-codeblock">
```
Epoch 43/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4433

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - loss: 0.4717

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 2ms/step - loss: 0.4717

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4724

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4725


<div class="k-default-codeblock">
```
Epoch 44/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4429

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - loss: 0.4712

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4719

<div class="k-default-codeblock">
```

```
</div>
 70/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4719

<div class="k-default-codeblock">
```

```
</div>
 71/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4719

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4721


<div class="k-default-codeblock">
```
Epoch 45/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4424

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4707

<div class="k-default-codeblock">
```

```
</div>
 68/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 2ms/step - loss: 0.4714

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4716 - val_loss: 0.4791


<div class="k-default-codeblock">
```
Epoch 46/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4418

<div class="k-default-codeblock">
```

```
</div>
 32/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4701

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4708

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4710


<div class="k-default-codeblock">
```
Epoch 47/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4412

    
  2/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 3ms/step - loss: 0.4478

<div class="k-default-codeblock">
```

```
</div>
 36/80 ━━━━━━━━━[37m━━━━━━━━━━━  0s 2ms/step - loss: 0.4695

<div class="k-default-codeblock">
```

```
</div>
 67/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4701

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4703


<div class="k-default-codeblock">
```
Epoch 48/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4406

<div class="k-default-codeblock">
```

```
</div>
 35/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 1ms/step - loss: 0.4688

<div class="k-default-codeblock">
```

```
</div>
 69/80 ━━━━━━━━━━━━━━━━━[37m━━━  0s 1ms/step - loss: 0.4694

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4696


<div class="k-default-codeblock">
```
Epoch 49/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4398

<div class="k-default-codeblock">
```

```
</div>
 33/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4680

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4680

<div class="k-default-codeblock">
```

```
</div>
 66/80 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.4686

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4688


<div class="k-default-codeblock">
```
Epoch 50/50

```
</div>
    
  1/80 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.4390

<div class="k-default-codeblock">
```

```
</div>
 34/80 ━━━━━━━━[37m━━━━━━━━━━━━  0s 2ms/step - loss: 0.4671

<div class="k-default-codeblock">
```

```
</div>
 72/80 ━━━━━━━━━━━━━━━━━━[37m━━  0s 1ms/step - loss: 0.4678

<div class="k-default-codeblock">
```

```
</div>
 80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4679 - val_loss: 0.4753


---
## Making predictions

Before we try out ScANN, let's go with the brute force method, i.e., for a given
user, scores are computed for all movies, sorted and then the top-k
movies are picked. This is, of course, not very scalable when we have a huge
number of movies.


```python
candidate_embeddings = keras.ops.array(model.candidate_embedding.embeddings.numpy())
# Artificially duplicate candidate embeddings to simulate a large number of
# movies.
candidate_embeddings = keras.ops.concatenate(
    [candidate_embeddings]
    + [
        candidate_embeddings
        * keras.random.uniform(keras.ops.shape(candidate_embeddings))
        for _ in range(100)
    ],
    axis=0,
)

user_embedding = model.user_embedding(keras.ops.array([10, 5, 42, 345]))

# Define the brute force retrieval layer.
brute_force_layer = keras_rs.layers.BruteForceRetrieval(
    candidate_embeddings=candidate_embeddings,
    k=10,
    return_scores=False,
)
```

Now, let's do a forward pass on the layer. Note that in previous tutorials, we
have the above layer as an attribute of the model class, and we then call
`.predict()`. This will obviously be faster (since it's compiled XLA code), but
since we cannot do the same for ScANN, we just do a normal forward pass here
without compilation to ensure a fair comparison.


```python
t0 = time.time()
pred_movie_ids = brute_force_layer(user_embedding)
print("Time taken by brute force layer (sec):", time.time() - t0)
```

<div class="k-default-codeblock">
```
Time taken by brute force layer (sec): 0.22817683219909668

```
</div>
Now, let's retrieve movies using ScANN. We will use the ScANN library from
Google Research to build the layer and then call it. To fully understand all the
arguments, please refer to the
[ScANN README file](https://github.com/google-research/google-research/tree/master/scann#readme).


```python

def build_scann(
    candidates,
    k=10,
    distance_measure="dot_product",
    dimensions_per_block=2,
    num_reordering_candidates=500,
    num_leaves=100,
    num_leaves_to_search=30,
    training_iterations=12,
):
    builder = scann_ops.builder(
        db=candidates,
        num_neighbors=k,
        distance_measure=distance_measure,
    )

    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=num_leaves_to_search,
        training_iterations=training_iterations,
    )
    builder = builder.score_ah(dimensions_per_block=dimensions_per_block)

    if num_reordering_candidates is not None:
        builder = builder.reorder(num_reordering_candidates)

    # Set a unique name to prevent unintentional sharing between
    # ScaNN instances.
    searcher = builder.build(shared_name=str(uuid.uuid4()))
    return searcher


def run_scann(searcher):
    pred_movie_ids = searcher.search_batched_parallel(
        user_embedding,
        final_num_neighbors=10,
    ).indices
    return pred_movie_ids


searcher = build_scann(candidates=candidate_embeddings)

t0 = time.time()
pred_movie_ids = run_scann(searcher)
print("Time taken by ScANN (sec):", time.time() - t0)
```

<div class="k-default-codeblock">
```
Time taken by ScANN (sec): 0.0032587051391601562

```
</div>
You can clearly see the performance improvement in terms of latency. ScANN
(0.003 seconds) takes one-fiftieth the time it takes for the brute force layer
(0.15 seconds) to run!


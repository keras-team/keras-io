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
!pip install -q keras-rs
```


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

Shuffling /root/tensorflow_datasets/movielens/100k-ratings/incomplete.2O98FR_0.1.1/movielens-train.tfrecord*..…

WARNING:absl:Variant folder /root/tensorflow_datasets/movielens/100k-movies/0.1.1 has no dataset_info.json

Dataset movielens downloaded and prepared to /root/tensorflow_datasets/movielens/100k-ratings/0.1.1. Subsequent calls will reuse this data.
Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to /root/tensorflow_datasets/movielens/100k-movies/0.1.1...

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/1 [00:00<?, ? splits/s]

Generating train examples...: 0 examples [00:00, ? examples/s]

Shuffling /root/tensorflow_datasets/movielens/100k-movies/incomplete.4QKWMO_0.1.1/movielens-train.tfrecord*...…

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

80/80 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 0.4772

Epoch 2/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4771

Epoch 3/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4770

Epoch 4/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4769

Epoch 5/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 3s 37ms/step - loss: 0.4769 - val_loss: 0.4836

Epoch 6/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4768

Epoch 7/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4767

Epoch 8/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4766

Epoch 9/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4764

Epoch 10/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4763 - val_loss: 0.4833

Epoch 11/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4761

Epoch 12/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4759

Epoch 13/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4757

Epoch 14/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4754

Epoch 15/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4750 - val_loss: 0.4821

Epoch 16/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4746

Epoch 17/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.4740

Epoch 18/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4734

Epoch 19/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4725

Epoch 20/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4715 - val_loss: 0.4784

Epoch 21/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4702

Epoch 22/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4686

Epoch 23/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4666

Epoch 24/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4641

Epoch 25/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4609 - val_loss: 0.4664

Epoch 26/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4571

Epoch 27/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4524

Epoch 28/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4466

Epoch 29/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4395

Epoch 30/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.4311 - val_loss: 0.4326

Epoch 31/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4210

Epoch 32/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.4093

Epoch 33/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3957

Epoch 34/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3805

Epoch 35/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.3636 - val_loss: 0.3597

Epoch 36/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3455

Epoch 37/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3265

Epoch 38/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.3072

Epoch 39/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.2880

Epoch 40/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.2696 - val_loss: 0.2664

Epoch 41/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2523

Epoch 42/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.2363

Epoch 43/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2218

Epoch 44/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.2087

Epoch 45/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.1970 - val_loss: 0.1986

Epoch 46/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1866

Epoch 47/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1773

Epoch 48/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.1689

Epoch 49/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 0.1613

Epoch 50/50

80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.1544 - val_loss: 0.1586
```
</div>

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

    
<div class="k-default-codeblock">
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 211ms/step

==Recommended movies for user 793==
b'Star Wars (1977)'
b'Godfather, The (1972)'
b'Raiders of the Lost Ark (1981)'
b'Fargo (1996)'
b'Silence of the Lambs, The (1991)'
b"Schindler's List (1993)"
b'Shawshank Redemption, The (1994)'
b'Titanic (1997)'
b'Braveheart (1995)'
b'Pulp Fiction (1994)'

==Recommended movies for user 188==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Silence of the Lambs, The (1991)'
b"Schindler's List (1993)"
b'Return of the Jedi (1983)'
b'Raiders of the Lost Ark (1981)'
b'Pulp Fiction (1994)'
b'Toy Story (1995)'
b'Empire Strikes Back, The (1980)'

==Recommended movies for user 865==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Silence of the Lambs, The (1991)'
b'Raiders of the Lost Ark (1981)'
b"Schindler's List (1993)"
b'Return of the Jedi (1983)'
b'Shawshank Redemption, The (1994)'
b'Pulp Fiction (1994)'
b'Empire Strikes Back, The (1980)'

==Recommended movies for user 710==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Silence of the Lambs, The (1991)'
b'Raiders of the Lost Ark (1981)'
b"Schindler's List (1993)"
b'Pulp Fiction (1994)'
b'Return of the Jedi (1983)'
b'Empire Strikes Back, The (1980)'
b'Toy Story (1995)'

==Recommended movies for user 721==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Raiders of the Lost Ark (1981)'
b'Silence of the Lambs, The (1991)'
b"Schindler's List (1993)"
b'Return of the Jedi (1983)'
b'Empire Strikes Back, The (1980)'
b'Pulp Fiction (1994)'
b'Casablanca (1942)'

==Recommended movies for user 451==
b'Star Wars (1977)'
b'Raiders of the Lost Ark (1981)'
b'Godfather, The (1972)'
b'Fargo (1996)'
b'Silence of the Lambs, The (1991)'
b'Return of the Jedi (1983)'
b'Contact (1997)'
b'Casablanca (1942)'
b'Empire Strikes Back, The (1980)'
b'Pulp Fiction (1994)'

==Recommended movies for user 228==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Godfather, The (1972)'
b'Raiders of the Lost Ark (1981)'
b'Silence of the Lambs, The (1991)'
b"Schindler's List (1993)"
b'Return of the Jedi (1983)'
b'Pulp Fiction (1994)'
b'Empire Strikes Back, The (1980)'
b'Shawshank Redemption, The (1994)'

==Recommended movies for user 175==
b'Star Wars (1977)'
b'Fargo (1996)'
b'Silence of the Lambs, The (1991)'
b'Raiders of the Lost Ark (1981)'
b'Return of the Jedi (1983)'
b'Casablanca (1942)'
b"Schindler's List (1993)"
b'Empire Strikes Back, The (1980)'
b'Godfather, The (1972)'
b"One Flew Over the Cuckoo's Nest (1975)"
```
</div>

And we're done! For data parallel training, all we had to do was add ~3-5 LoC.
The rest is exactly the same.


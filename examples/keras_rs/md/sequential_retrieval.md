# Sequential retrieval [GRU4Rec]

**Author:** [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Recommend movies using a GRU-based sequential retrieval model.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_rs/ipynb/sequential_retrieval.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_rs/sequential_retrieval.py)



---
## Introduction

In this example, we are going to build a sequential retrieval model. Sequential
recommendation is a popular model that looks at a sequence of items that users
have interacted with previously and then predicts the next item. Here, the order
of the items within each sequence matters. So, we are going to use a recurrent
neural network to model the sequential relationship. For more details,
please refer to the [GRU4Rec](https://arxiv.org/abs/1511.06939) paper.

Let's begin by choosing JAX as the backend we want to run on, and import all
the necessary libraries.


```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # `"tensorflow"`/`"torch"`

import collections
import os
import random

import keras
import pandas as pd
import tensorflow as tf  # Needed only for the dataset

import keras_rs
```

Let's also define all important variables/hyperparameters below.


```python
DATA_DIR = "./raw/data/"

# MovieLens-specific variables
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_ZIP_HASH = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"

RATINGS_FILE_NAME = "ratings.dat"
MOVIES_FILE_NAME = "movies.dat"

# Data processing args
MAX_CONTEXT_LENGTH = 10
MIN_SEQUENCE_LENGTH = 3
TRAIN_DATA_FRACTION = 0.9

RATINGS_DATA_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
MOVIES_DATA_COLUMNS = ["MovieID", "Title", "Genres"]
MIN_RATING = 2

# Training/model args
BATCH_SIZE = 4096
TEST_BATCH_SIZE = 2048
EMBEDDING_DIM = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.005
```

---
## Dataset

Next, we need to prepare our dataset. Like we did in the
[basic retrieval](/keras_rs/examples/basic_retrieval/)
example, we are going to use the MovieLens dataset.

The dataset preparation step is fairly involved. The original ratings dataset
contains `(user, movie ID, rating, timestamp)` tuples (among other columns,
which are not important for this example). Since we are dealing with sequential
retrieval, we need to create movie sequences for every user, where the sequences
are ordered by timestamp.

Let's start by downloading and reading the dataset.


```python
# Download the MovieLens dataset.
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

path_to_zip = keras.utils.get_file(
    fname="ml-1m.zip",
    origin=MOVIELENS_1M_URL,
    file_hash=MOVIELENS_ZIP_HASH,
    hash_algorithm="sha256",
    extract=True,
    cache_dir=DATA_DIR,
)
movielens_extracted_dir = os.path.join(
    os.path.dirname(path_to_zip),
    "ml-1m_extracted",
    "ml-1m",
)


# Read the dataset.
def read_data(data_directory, min_rating=None):
    """Read movielens ratings.dat and movies.dat file
    into dataframe.
    """

    ratings_df = pd.read_csv(
        os.path.join(data_directory, RATINGS_FILE_NAME),
        sep="::",
        names=RATINGS_DATA_COLUMNS,
        encoding="unicode_escape",
    )
    ratings_df["Timestamp"] = ratings_df["Timestamp"].apply(int)

    # Remove movies with `rating < min_rating`.
    if min_rating is not None:
        ratings_df = ratings_df[ratings_df["Rating"] >= min_rating]

    movies_df = pd.read_csv(
        os.path.join(data_directory, MOVIES_FILE_NAME),
        sep="::",
        names=MOVIES_DATA_COLUMNS,
        encoding="unicode_escape",
    )
    return ratings_df, movies_df


ratings_df, movies_df = read_data(
    data_directory=movielens_extracted_dir, min_rating=MIN_RATING
)

# Need to know #movies so as to define embedding layers.
movies_count = movies_df["MovieID"].max()
```

<div class="k-default-codeblock">
```
Downloading data from https://files.grouplens.org/datasets/movielens/ml-1m.zip
5917549/5917549 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step

/var/tmp/ipykernel_688439/1372663084.py:26: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
  ratings_df = pd.read_csv(
/var/tmp/ipykernel_688439/1372663084.py:38: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
  movies_df = pd.read_csv(
```
</div>

Now that we have read the dataset, let's create sequences of movies
for every user. Here is the function for doing just that.


```python

def get_movie_sequence_per_user(ratings_df):
    """Get movieID sequences for every user."""
    sequences = collections.defaultdict(list)

    for user_id, movie_id, rating, timestamp in ratings_df.values:
        sequences[user_id].append(
            {
                "movie_id": movie_id,
                "timestamp": timestamp,
                "rating": rating,
            }
        )

    # Sort movie sequences by timestamp for every user.
    for user_id, context in sequences.items():
        context.sort(key=lambda x: x["timestamp"])
        sequences[user_id] = context

    return sequences

```

We need to do some filtering and processing before we proceed
with training the model:

1. Form sequences of all lengths up to
   `min(user_sequence_length, MAX_CONTEXT_LENGTH)`. So, every user
   will have multiple sequences corresponding to it.
2. Get labels, i.e., Given a sequence of length `n`, the first
   `n-1` tokens will be fed to the model as input, and the label
   will be the last token.
3. Remove all user sequences with less than `MIN_SEQUENCE_LENGTH`
   movies.
4. Pad all sequences to `MAX_CONTEXT_LENGTH`.


```python

def generate_examples_from_user_sequences(sequences):
    """Generates sequences for all users, with padding, truncation, etc."""

    def generate_examples_from_user_sequence(sequence):
        """Generates examples for a single user sequence."""

        examples = []
        for label_idx in range(1, len(sequence)):
            start_idx = max(0, label_idx - MAX_CONTEXT_LENGTH)
            context = sequence[start_idx:label_idx]

            # Padding
            while len(context) < MAX_CONTEXT_LENGTH:
                context.append(
                    {
                        "movie_id": 0,
                        "timestamp": 0,
                        "rating": 0.0,
                    }
                )

            label_movie_id = int(sequence[label_idx]["movie_id"])
            context_movie_id = [int(movie["movie_id"]) for movie in context]

            examples.append(
                {
                    "context_movie_id": context_movie_id,
                    "label_movie_id": label_movie_id,
                },
            )
        return examples

    all_examples = []
    for sequence in sequences.values():
        if len(sequence) < MIN_SEQUENCE_LENGTH:
            continue

        user_examples = generate_examples_from_user_sequence(sequence)

        all_examples.extend(user_examples)

    return all_examples

```

Let's split the dataset into train and test sets. Also, we need to
change the format of the dataset dictionary so as to enable conversion
to a `tf.data.Dataset` object.


```python
sequences = get_movie_sequence_per_user(ratings_df)
examples = generate_examples_from_user_sequences(sequences)

# Train-test split.
random.shuffle(examples)
split_index = int(TRAIN_DATA_FRACTION * len(examples))
train_examples = examples[:split_index]
test_examples = examples[split_index:]


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """Convert list of dictionaries to dictionary of lists for
    `tf.data` conversion.
    """
    dict_of_lists = collections.defaultdict(list)
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            dict_of_lists[key].append(value)
    return dict_of_lists


train_examples = list_of_dicts_to_dict_of_lists(train_examples)
test_examples = list_of_dicts_to_dict_of_lists(test_examples)

train_ds = tf.data.Dataset.from_tensor_slices(train_examples).map(
    lambda x: (x["context_movie_id"], x["label_movie_id"])
)
test_ds = tf.data.Dataset.from_tensor_slices(test_examples).map(
    lambda x: (x["context_movie_id"], x["label_movie_id"])
)
```

We need to batch our datasets. We also user `cache()` and `prefetch()`
for better performance.


```python
train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(TEST_BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
```

Let's print out one batch.


```python
for sample in train_ds.take(1):
    print(sample)
```

<div class="k-default-codeblock">
```
(<tf.Tensor: shape=(4096, 10), dtype=int32, numpy=
array([[3512, 3617, 3623, ..., 3007, 2858, 1617],
       [1952, 1206, 1233, ..., 1394, 3683,  593],
       [2114, 1274, 2407, ..., 2100, 1257, 2001],
       ...,
       [  81, 2210, 1343, ..., 1625, 1748, 1407],
       [ 494,  832,  543, ...,   23,  432, 1682],
       [2421,    0,    0, ...,    0,    0,    0]], dtype=int32)>, <tf.Tensor: shape=(4096,), dtype=int32, numpy=array([3265, 1203, 2003, ..., 3044,  367,  110], dtype=int32)>)
```
</div>

---
## Model and Training

In the basic retrieval example, we used one query tower for the
user, and the candidate tower for the candidate movie. We are
going to use a two-tower architecture here as well. However,
we use the query tower with a Gated Recurrent Unit (GRU) layer
to encode the sequence of historical movies, and keep the same
candidate tower for the candidate movie.

Note: Take a look at how the labels are defined. The label tensor
(of shape `(batch_size, batch_size)`) contains one-hot vectors. The idea
is: for every sample, consider movie IDs corresponding to other samples in
the batch as negatives.


```python

class SequentialRetrievalModel(keras.Model):
    """Create the sequential retrieval model.

    Args:
      movies_count: Total number of unique movies in the dataset.
      embedding_dimension: Output dimension for movie embedding tables.
    """

    def __init__(
        self,
        movies_count,
        embedding_dimension=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Our query tower, simply an embedding table followed by
        # a GRU unit. This encodes sequence of historical movies.
        self.query_model = keras.Sequential(
            [
                keras.layers.Embedding(movies_count + 1, embedding_dimension),
                keras.layers.GRU(embedding_dimension),
            ]
        )

        # Our candidate tower, simply an embedding table.
        self.candidate_model = keras.layers.Embedding(
            movies_count + 1, embedding_dimension
        )

        # The layer that performs the retrieval.
        self.retrieval = keras_rs.layers.BruteForceRetrieval(k=10, return_scores=False)
        self.loss_fn = keras.losses.CategoricalCrossentropy(
            from_logits=True,
        )

    def build(self, input_shape):
        self.query_model.build(input_shape)
        self.candidate_model.build(input_shape)

        # In this case, the candidates are directly the movie embeddings.
        # We take a shortcut and directly reuse the variable.
        self.retrieval.candidate_embeddings = self.candidate_model.embeddings
        self.retrieval.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        query_embeddings = self.query_model(inputs)
        result = {
            "query_embeddings": query_embeddings,
        }

        if not training:
            # Skip the retrieval of top movies during training as the
            # predictions are not used.
            result["predictions"] = self.retrieval(query_embeddings)
        return result

    def compute_loss(self, x, y, y_pred, sample_weight, training=True):
        candidate_id = y
        query_embeddings = y_pred["query_embeddings"]
        candidate_embeddings = self.candidate_model(candidate_id)

        num_queries = keras.ops.shape(query_embeddings)[0]
        num_candidates = keras.ops.shape(candidate_embeddings)[0]

        # One-hot vectors for labels.
        labels = keras.ops.eye(num_queries, num_candidates)

        # Compute the affinity score by multiplying the two embeddings.
        scores = keras.ops.matmul(
            query_embeddings, keras.ops.transpose(candidate_embeddings)
        )

        return self.loss_fn(labels, scores, sample_weight)

```

Let's instantiate, compile and train our model.


```python
model = SequentialRetrievalModel(
    movies_count=movies_count, embedding_dimension=EMBEDDING_DIM
)

# Compile.
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE))

# Train.
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=NUM_EPOCHS,
)
```

<div class="k-default-codeblock">
```
Epoch 1/5
207/207 ━━━━━━━━━━━━━━━━━━━━ 6s 24ms/step - loss: 7.9460 - val_loss: 6.4827
Epoch 2/5
207/207 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - loss: 7.0764 - val_loss: 6.1424
Epoch 3/5
207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 6.7779 - val_loss: 6.0004
Epoch 4/5
207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 6.6406 - val_loss: 5.9258
Epoch 5/5
207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 6.5584 - val_loss: 5.8826

<keras.src.callbacks.history.History at 0x7fd1506dc670>
```
</div>

---
## Making predictions

Now that we have a model, we would like to be able to make predictions.

So far, we have only handled movies by id. Now is the time to create a mapping
keyed by movie IDs to be able to surface the titles.


```python
movie_id_to_movie_title = dict(zip(movies_df["MovieID"], movies_df["Title"]))
movie_id_to_movie_title[0] = ""  # Because id 0 is not in the dataset.
```

We then simply use the Keras `model.predict()` method. Under the hood, it calls
the `BruteForceRetrieval` layer to perform the actual retrieval.

Note that this model can retrieve movies already watched by the user. We could
easily add logic to remove them if that is desirable.


```python
print("\n==> Movies the user has watched:")
movie_sequence = test_ds.unbatch().take(1)
for element in movie_sequence:
    for movie_id in element[0][:-1]:
        print(movie_id_to_movie_title[movie_id.numpy()], end=", ")
    print(movie_id_to_movie_title[element[0][-1].numpy()])

predictions = model.predict(movie_sequence.batch(1))
predictions = keras.ops.convert_to_numpy(predictions["predictions"])

print("\n==> Recommended movies for the above sequence:")
for movie_id in predictions[0]:
    print(movie_id_to_movie_title[movie_id])
```

    
<div class="k-default-codeblock">
```
==> Movies the user has watched:
Rob Roy (1995), Legends of the Fall (1994), French Kiss (1995), Terminator 2: Judgment Day (1991), Nikita (La Femme Nikita) (1990), Professional, The (a.k.a. Leon: The Professional) (1994), Seven (Se7en) (1995), Fugitive, The (1993), Enemy of the State (1998), Reservoir Dogs (1992)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 225ms/step

==> Recommended movies for the above sequence:
Red Rock West (1992)
Casino (1995)
Cape Fear (1991)
Simple Plan, A (1998)
Seven (Se7en) (1995)
Hard 8 (a.k.a. Sydney, a.k.a. Hard Eight) (1996)
Primal Fear (1996)
Heat (1995)
Scream (1996)
Zero Effect (1998)

/opt/conda/envs/keras-jax/lib/python3.10/site-packages/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
```
</div>

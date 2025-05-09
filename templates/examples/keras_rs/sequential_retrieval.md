# Sequential retrieval [GRU4Rec]

**Author:** [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Recommend movies using a GRU-based sequential retrieval model.


<div class='example_version_banner keras_2'>ⓘ This example uses Keras 2</div>
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
LEARNING_RATE = 0.05
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

```
</div>
    
       0/5917549 [37m━━━━━━━━━━━━━━━━━━━━  0s 0s/step

<div class="k-default-codeblock">
```

```
</div>
   40960/5917549 [37m━━━━━━━━━━━━━━━━━━━━  10s 2us/step

<div class="k-default-codeblock">
```

```
</div>
  155648/5917549 [37m━━━━━━━━━━━━━━━━━━━━  5s 1us/step 

<div class="k-default-codeblock">
```

```
</div>
  647168/5917549 ━━[37m━━━━━━━━━━━━━━━━━━  1s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 2629632/5917549 ━━━━━━━━[37m━━━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 5917549/5917549 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step


<div class="k-default-codeblock">
```
<ipython-input-3-6fc962858754>:26: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
  ratings_df = pd.read_csv(

<ipython-input-3-6fc962858754>:38: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
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
array([[   1,  858, 1270, ...,  969,  902,    0],
       [ 497, 1500, 2321, ...,  440,  471, 1188],
       [2804, 2204, 1294, ..., 1224, 1186, 2132],
       ...,
       [2993,  653, 1287, ...,  474,  145,   10],
       [1377, 1387, 1388, ..., 1089,  497, 2169],
       [ 494, 3505,  521, ..., 2490,  280, 3686]], dtype=int32)>, <tf.Tensor: shape=(4096,), dtype=int32, numpy=array([ 899,  838, 3152, ...,  377, 1271,  861], dtype=int32)>)

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
    movies_count=movies_count + 1, embedding_dimension=EMBEDDING_DIM
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

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  7:12 2s/step - loss: 8.3177

<div class="k-default-codeblock">
```

```
</div>
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  4:07 1s/step - loss: 8.3176

<div class="k-default-codeblock">
```

```
</div>
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  2:03 607ms/step - loss: 8.3169

<div class="k-default-codeblock">
```

```
</div>
   4/207 [37m━━━━━━━━━━━━━━━━━━━━  1:22 407ms/step - loss: 8.3154

<div class="k-default-codeblock">
```

```
</div>
  11/207 ━[37m━━━━━━━━━━━━━━━━━━━  24s 128ms/step - loss: 8.2616 

<div class="k-default-codeblock">
```

```
</div>
  12/207 ━[37m━━━━━━━━━━━━━━━━━━━  22s 117ms/step - loss: 8.2514

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  20s 108ms/step - loss: 8.2410

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  19s 100ms/step - loss: 8.2303

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  17s 93ms/step - loss: 8.2196 

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  16s 88ms/step - loss: 8.2088

<div class="k-default-codeblock">
```

```
</div>
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  11s 62ms/step - loss: 8.1343

<div class="k-default-codeblock">
```

```
</div>
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  10s 60ms/step - loss: 8.1240

<div class="k-default-codeblock">
```

```
</div>
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  10s 57ms/step - loss: 8.1139

<div class="k-default-codeblock">
```

```
</div>
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  10s 55ms/step - loss: 8.1040

<div class="k-default-codeblock">
```

```
</div>
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  9s 53ms/step - loss: 8.0943 

<div class="k-default-codeblock">
```

```
</div>
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  9s 52ms/step - loss: 8.0846

<div class="k-default-codeblock">
```

```
</div>
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  8s 50ms/step - loss: 8.0751

<div class="k-default-codeblock">
```

```
</div>
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  6s 38ms/step - loss: 7.9869

<div class="k-default-codeblock">
```

```
</div>
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  6s 37ms/step - loss: 7.9788

<div class="k-default-codeblock">
```

```
</div>
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  6s 37ms/step - loss: 7.9708

<div class="k-default-codeblock">
```

```
</div>
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 36ms/step - loss: 7.9629

<div class="k-default-codeblock">
```

```
</div>
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 35ms/step - loss: 7.9551

<div class="k-default-codeblock">
```

```
</div>
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 29ms/step - loss: 7.8825

<div class="k-default-codeblock">
```

```
</div>
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 29ms/step - loss: 7.8757

<div class="k-default-codeblock">
```

```
</div>
  55/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 28ms/step - loss: 7.8691

<div class="k-default-codeblock">
```

```
</div>
  56/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 28ms/step - loss: 7.8625

<div class="k-default-codeblock">
```

```
</div>
  66/207 ━━━━━━[37m━━━━━━━━━━━━━━  3s 25ms/step - loss: 7.8011

<div class="k-default-codeblock">
```

```
</div>
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  3s 24ms/step - loss: 7.7954

<div class="k-default-codeblock">
```

```
</div>
  75/207 ━━━━━━━[37m━━━━━━━━━━━━━  2s 22ms/step - loss: 7.7518

<div class="k-default-codeblock">
```

```
</div>
  83/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 21ms/step - loss: 7.7120

<div class="k-default-codeblock">
```

```
</div>
  91/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 20ms/step - loss: 7.6755

<div class="k-default-codeblock">
```

```
</div>
  99/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 18ms/step - loss: 7.6419

<div class="k-default-codeblock">
```

```
</div>
 107/207 ━━━━━━━━━━[37m━━━━━━━━━━  1s 18ms/step - loss: 7.6108

<div class="k-default-codeblock">
```

```
</div>
 115/207 ━━━━━━━━━━━[37m━━━━━━━━━  1s 17ms/step - loss: 7.5821

<div class="k-default-codeblock">
```

```
</div>
 123/207 ━━━━━━━━━━━[37m━━━━━━━━━  1s 16ms/step - loss: 7.5553

<div class="k-default-codeblock">
```

```
</div>
 131/207 ━━━━━━━━━━━━[37m━━━━━━━━  1s 16ms/step - loss: 7.5303

<div class="k-default-codeblock">
```

```
</div>
 139/207 ━━━━━━━━━━━━━[37m━━━━━━━  1s 15ms/step - loss: 7.5069

<div class="k-default-codeblock">
```

```
</div>
 140/207 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - loss: 7.5041

<div class="k-default-codeblock">
```

```
</div>
 148/207 ━━━━━━━━━━━━━━[37m━━━━━━  0s 14ms/step - loss: 7.4823

<div class="k-default-codeblock">
```

```
</div>
 157/207 ━━━━━━━━━━━━━━━[37m━━━━━  0s 14ms/step - loss: 7.4592

<div class="k-default-codeblock">
```

```
</div>
 165/207 ━━━━━━━━━━━━━━━[37m━━━━━  0s 14ms/step - loss: 7.4400

<div class="k-default-codeblock">
```

```
</div>
 173/207 ━━━━━━━━━━━━━━━━[37m━━━━  0s 13ms/step - loss: 7.4218

<div class="k-default-codeblock">
```

```
</div>
 181/207 ━━━━━━━━━━━━━━━━━[37m━━━  0s 13ms/step - loss: 7.4045

<div class="k-default-codeblock">
```

```
</div>
 189/207 ━━━━━━━━━━━━━━━━━━[37m━━  0s 13ms/step - loss: 7.3881

<div class="k-default-codeblock">
```

```
</div>
 197/207 ━━━━━━━━━━━━━━━━━━━[37m━  0s 12ms/step - loss: 7.3725

<div class="k-default-codeblock">
```

```
</div>
 205/207 ━━━━━━━━━━━━━━━━━━━[37m━  0s 12ms/step - loss: 7.3540

<div class="k-default-codeblock">
```

```
</div>
 206/207 ━━━━━━━━━━━━━━━━━━━[37m━  0s 12ms/step - loss: 7.3558

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - loss: 7.3505

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 8s 28ms/step - loss: 7.3487 - val_loss: 5.9852


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  4:08 1s/step - loss: 6.6873

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6892 

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6883

    
   4/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6841

<div class="k-default-codeblock">
```

```
</div>
   9/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6842

<div class="k-default-codeblock">
```

```
</div>
  10/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6834

<div class="k-default-codeblock">
```

```
</div>
  11/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6829

<div class="k-default-codeblock">
```

```
</div>
  12/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6825

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6822

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6819
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6821

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6813
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6816
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6814

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.6811

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.6810

<div class="k-default-codeblock">
```

```
</div>
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.6806
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 11ms/step - loss: 6.6805

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.6808

<div class="k-default-codeblock">
```

```
</div>
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.6804
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6803

<div class="k-default-codeblock">
```

```
</div>
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6804

<div class="k-default-codeblock">
```

```
</div>
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6804

<div class="k-default-codeblock">
```

```
</div>
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.6803

<div class="k-default-codeblock">
```

```
</div>
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6803
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6803
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.6803

<div class="k-default-codeblock">
```

```
</div>
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6802

<div class="k-default-codeblock">
```

```
</div>
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.6796
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6795

<div class="k-default-codeblock">
```

```
</div>
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6799
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6801

<div class="k-default-codeblock">
```

```
</div>
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6797

<div class="k-default-codeblock">
```

```
</div>
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6800

<div class="k-default-codeblock">
```

```
</div>
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6793
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6792

<div class="k-default-codeblock">
```

```
</div>
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6787
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6788
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6791

<div class="k-default-codeblock">
```

```
</div>
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6783

<div class="k-default-codeblock">
```

```
</div>
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6782

<div class="k-default-codeblock">
```

```
</div>
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6780
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6769

<div class="k-default-codeblock">
```

```
</div>
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6772
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6765
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6776
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.6770

<div class="k-default-codeblock">
```

```
</div>
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6760

<div class="k-default-codeblock">
```

```
</div>
  57/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.6750
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6746
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6756
  56/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6752
  55/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6748

<div class="k-default-codeblock">
```

```
</div>
  58/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6737

<div class="k-default-codeblock">
```

```
</div>
  63/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6728
  62/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6730
  61/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6731

<div class="k-default-codeblock">
```

```
</div>
  60/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6733
  64/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.6726
  59/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6735

<div class="k-default-codeblock">
```

```
</div>
  65/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6724

<div class="k-default-codeblock">
```

```
</div>
  70/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6713
  69/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6707
  71/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6703

<div class="k-default-codeblock">
```

```
</div>
  68/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6709
  66/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6721
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6711

<div class="k-default-codeblock">
```

```
</div>
  77/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6687
  76/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6693
  75/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6691
  72/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 15ms/step - loss: 6.6685

<div class="k-default-codeblock">
```

```
</div>
  73/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6699

<div class="k-default-codeblock">
```

```
</div>
  74/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6697
  82/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6662

<div class="k-default-codeblock">
```

```
</div>
  79/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6672
  81/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6660
  83/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 14ms/step - loss: 6.6664

<div class="k-default-codeblock">
```

```
</div>
  80/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6659
  87/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.6641

<div class="k-default-codeblock">
```

```
</div>
  85/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.6640

<div class="k-default-codeblock">
```

```
</div>
  78/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6657
  84/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.6647
  86/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.6643

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 2s 6ms/step - loss: 6.6328 - val_loss: 5.9231


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5509

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5612 

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5651

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5684

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5687

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5688

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5689

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5691

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5692

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5694
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5694

<div class="k-default-codeblock">
```

```
</div>
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5701
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5696
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5699

<div class="k-default-codeblock">
```

```
</div>
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5695

<div class="k-default-codeblock">
```

```
</div>
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5717
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5704
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5723
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5710
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5722
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5720

<div class="k-default-codeblock">
```

```
</div>
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5713

<div class="k-default-codeblock">
```

```
</div>
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5719

<div class="k-default-codeblock">
```

```
</div>
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5726
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5724

<div class="k-default-codeblock">
```

```
</div>
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5727

<div class="k-default-codeblock">
```

```
</div>
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5728

<div class="k-default-codeblock">
```

```
</div>
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5730
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5731

<div class="k-default-codeblock">
```

```
</div>
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5731 
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5731
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5731
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5731

<div class="k-default-codeblock">
```

```
</div>
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5730

<div class="k-default-codeblock">
```

```
</div>
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5730

<div class="k-default-codeblock">
```

```
</div>
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5729

<div class="k-default-codeblock">
```

```
</div>
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5728

<div class="k-default-codeblock">
```

```
</div>
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5727

<div class="k-default-codeblock">
```

```
</div>
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5727

<div class="k-default-codeblock">
```

```
</div>
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5726

<div class="k-default-codeblock">
```

```
</div>
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5725

<div class="k-default-codeblock">
```

```
</div>
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5723

<div class="k-default-codeblock">
```

```
</div>
  56/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5720
  55/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5718

<div class="k-default-codeblock">
```

```
</div>
  57/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5716
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5716
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5719

<div class="k-default-codeblock">
```

```
</div>
  58/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5710

<div class="k-default-codeblock">
```

```
</div>
  59/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5709

<div class="k-default-codeblock">
```

```
</div>
  60/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5709
  63/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5704

<div class="k-default-codeblock">
```

```
</div>
  62/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5707

<div class="k-default-codeblock">
```

```
</div>
  61/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5706

<div class="k-default-codeblock">
```

```
</div>
  64/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5703

<div class="k-default-codeblock">
```

```
</div>
  65/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5694
  66/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5699
  70/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5695
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5700
  69/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5690
  68/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5697

<div class="k-default-codeblock">
```

```
</div>
  71/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5688

<div class="k-default-codeblock">
```

```
</div>
  72/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5687

<div class="k-default-codeblock">
```

```
</div>
  75/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5684
  73/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5686

<div class="k-default-codeblock">
```

```
</div>
  76/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5683

<div class="k-default-codeblock">
```

```
</div>
  74/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5685

<div class="k-default-codeblock">
```

```
</div>
  83/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5672
  84/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5673
  77/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5683
  80/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5680

<div class="k-default-codeblock">
```

```
</div>
  78/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5682
  81/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5676
  85/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5664
  79/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5681

<div class="k-default-codeblock">
```

```
</div>
  82/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5677
  88/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5666

<div class="k-default-codeblock">
```

```
</div>
  86/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5665

<div class="k-default-codeblock">
```

```
</div>
  90/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5659

<div class="k-default-codeblock">
```

```
</div>
  87/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5662
  91/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5658
  92/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5657
  94/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5655

<div class="k-default-codeblock">
```

```
</div>
  93/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5656
  89/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5660

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 6.5498 - val_loss: 5.9322


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5131

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5257 

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5284

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5314

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5316

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5317

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5317

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5320

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5321

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5325

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5324

<div class="k-default-codeblock">
```

```
</div>
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5327
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5325
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5329

<div class="k-default-codeblock">
```

```
</div>
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5332

<div class="k-default-codeblock">
```

```
</div>
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5335

<div class="k-default-codeblock">
```

```
</div>
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5341
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 6ms/step - loss: 6.5354
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5343
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5356
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5355
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5345
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5338
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5350
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5347

<div class="k-default-codeblock">
```

```
</div>
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5357

<div class="k-default-codeblock">
```

```
</div>
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5363
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5363
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5362
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5361
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5362

<div class="k-default-codeblock">
```

```
</div>
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5362
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5358

<div class="k-default-codeblock">
```

```
</div>
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5358
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5356
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5356
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5362

<div class="k-default-codeblock">
```

```
</div>
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5360

<div class="k-default-codeblock">
```

```
</div>
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5354
  55/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5349
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5347

<div class="k-default-codeblock">
```

```
</div>
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5352
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5353 
  56/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5348
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5354

<div class="k-default-codeblock">
```

```
</div>
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5351

<div class="k-default-codeblock">
```

```
</div>
  57/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5343
  62/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5340 
  58/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5344
  61/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5340
  59/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5344

<div class="k-default-codeblock">
```

```
</div>
  60/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5341

<div class="k-default-codeblock">
```

```
</div>
  63/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5339
  64/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5338
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5336
  68/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5335

<div class="k-default-codeblock">
```

```
</div>
  65/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5337

<div class="k-default-codeblock">
```

```
</div>
  66/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5336

<div class="k-default-codeblock">
```

```
</div>
  70/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5328
  72/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5326
  73/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5330

<div class="k-default-codeblock">
```

```
</div>
  69/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5334
  71/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5332
  74/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5329

<div class="k-default-codeblock">
```

```
</div>
  80/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5318
  78/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5320
  79/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5319

<div class="k-default-codeblock">
```

```
</div>
  77/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5317
  75/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5322

<div class="k-default-codeblock">
```

```
</div>
  76/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5323

<div class="k-default-codeblock">
```

```
</div>
  84/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5309
  83/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5306
  86/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 11ms/step - loss: 6.5307
  87/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 11ms/step - loss: 6.5303
  82/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5312

<div class="k-default-codeblock">
```

```
</div>
  85/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 11ms/step - loss: 6.5308

<div class="k-default-codeblock">
```

```
</div>
  90/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5300

<div class="k-default-codeblock">
```

```
</div>
  81/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5312

<div class="k-default-codeblock">
```

```
</div>
  88/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5302
  91/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5299

<div class="k-default-codeblock">
```

```
</div>
  93/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5294
  94/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5293
  89/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5298
  98/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5287
  92/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5298
  95/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5290
  97/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5291

<div class="k-default-codeblock">
```

```
</div>
  96/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5286

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 6.5158 - val_loss: 5.9527


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5082

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5182 

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5179

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5126

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5127

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5126

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5126

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5127

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5127

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5128
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5128

<div class="k-default-codeblock">
```

```
</div>
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5130
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5128
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5132
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5128

<div class="k-default-codeblock">
```

```
</div>
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5151
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5157

<div class="k-default-codeblock">
```

```
</div>
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5138
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5153
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5148

<div class="k-default-codeblock">
```

```
</div>
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5144

<div class="k-default-codeblock">
```

```
</div>
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5158
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5140

<div class="k-default-codeblock">
```

```
</div>
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5155

<div class="k-default-codeblock">
```

```
</div>
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5160

<div class="k-default-codeblock">
```

```
</div>
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5161

<div class="k-default-codeblock">
```

```
</div>
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5159

<div class="k-default-codeblock">
```

```
</div>
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5160

<div class="k-default-codeblock">
```

```
</div>
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5161
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5159
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5161
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5162
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5160

<div class="k-default-codeblock">
```

```
</div>
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5162

<div class="k-default-codeblock">
```

```
</div>
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5162
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5159 

<div class="k-default-codeblock">
```

```
</div>
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5154
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5154
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5150
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5151
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5153

<div class="k-default-codeblock">
```

```
</div>
  55/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5146
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5145
  57/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5140

<div class="k-default-codeblock">
```

```
</div>
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5145
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5149

<div class="k-default-codeblock">
```

```
</div>
  58/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5141
  56/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5140

<div class="k-default-codeblock">
```

```
</div>
  63/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5135
  60/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5137

<div class="k-default-codeblock">
```

```
</div>
  59/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5138
  61/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5134
  62/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5135

<div class="k-default-codeblock">
```

```
</div>
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5124

<div class="k-default-codeblock">
```

```
</div>
  69/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5117
  65/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5130
  66/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5123

<div class="k-default-codeblock">
```

```
</div>
  64/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5131
  71/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5119
  70/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5125

<div class="k-default-codeblock">
```

```
</div>
  68/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5118

<div class="k-default-codeblock">
```

```
</div>
  72/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5113
  73/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5114

<div class="k-default-codeblock">
```

```
</div>
  75/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5110
  77/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5107

<div class="k-default-codeblock">
```

```
</div>
  76/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5108

<div class="k-default-codeblock">
```

```
</div>
  74/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5110

<div class="k-default-codeblock">
```

```
</div>
  84/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 11ms/step - loss: 6.5098
  81/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5095

<div class="k-default-codeblock">
```

```
</div>
  78/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5107
  80/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5105
  83/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 11ms/step - loss: 6.5098
  79/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5097
  82/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5099
  85/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 11ms/step - loss: 6.5100

<div class="k-default-codeblock">
```

```
</div>
  86/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 11ms/step - loss: 6.5091

<div class="k-default-codeblock">
```

```
</div>
  87/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5089
  89/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5087
  88/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5090
  92/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5085

<div class="k-default-codeblock">
```

```
</div>
  95/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5076
  91/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5079

<div class="k-default-codeblock">
```

```
</div>
  97/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5081
  90/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5080

<div class="k-default-codeblock">
```

```
</div>
  93/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 11ms/step - loss: 6.5078
  98/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5074
  94/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5083

<div class="k-default-codeblock">
```

```
</div>
  96/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 11ms/step - loss: 6.5082

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 6.4960 - val_loss: 5.9651





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x78645944c8d0>

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
10 Things I Hate About You (1999), American Beauty (1999), Bachelor, The (1999), Austin Powers: The Spy Who Shagged Me (1999), Arachnophobia (1990), Big Daddy (1999), Bone Collector, The (1999), Bug's Life, A (1998), Bowfinger (1999), Dead Calm (1989)

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  0s 300ms/step


```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 302ms/step


    
<div class="k-default-codeblock">
```
==> Recommended movies for the above sequence:
Creepshow (1982)
Bringing Out the Dead (1999)
Civil Action, A (1998)
Doors, The (1991)
Cruel Intentions (1999)
Brokedown Palace (1999)
Dead Calm (1989)
Condorman (1981)
Clan of the Cave Bear, The (1986)
Clerks (1994)

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()

```
</div>

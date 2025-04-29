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
    8192/5917549 [37m━━━━━━━━━━━━━━━━━━━━  2:25 25us/step

<div class="k-default-codeblock">
```

```
</div>
   40960/5917549 [37m━━━━━━━━━━━━━━━━━━━━  57s 10us/step 

<div class="k-default-codeblock">
```

```
</div>
   73728/5917549 [37m━━━━━━━━━━━━━━━━━━━━  48s 8us/step 

<div class="k-default-codeblock">
```

```
</div>
  106496/5917549 [37m━━━━━━━━━━━━━━━━━━━━  44s 8us/step

<div class="k-default-codeblock">
```

```
</div>
  172032/5917549 [37m━━━━━━━━━━━━━━━━━━━━  33s 6us/step

<div class="k-default-codeblock">
```

```
</div>
  303104/5917549 ━[37m━━━━━━━━━━━━━━━━━━━  22s 4us/step

<div class="k-default-codeblock">
```

```
</div>
  565248/5917549 ━[37m━━━━━━━━━━━━━━━━━━━  13s 3us/step

<div class="k-default-codeblock">
```

```
</div>
 1105920/5917549 ━━━[37m━━━━━━━━━━━━━━━━━  7s 1us/step 

<div class="k-default-codeblock">
```

```
</div>
 2154496/5917549 ━━━━━━━[37m━━━━━━━━━━━━━  3s 1us/step

<div class="k-default-codeblock">
```

```
</div>
 4284416/5917549 ━━━━━━━━━━━━━━[37m━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 5917549/5917549 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step


<div class="k-default-codeblock">
```
<ipython-input-3-6fc962858754>:26: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
  ratings_df = pd.read_csv(

<ipython-input-3-6fc962858754>:38: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
  movies_df = pd.read_csv(

```
</div>
Let's take a look at a few rows.


```python
ratings_df.head()
movies_df.head()
```





  <div id="df-65710087-37dc-422a-9d7d-16a4b89d8b71" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

<div class="k-default-codeblock">
```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```
</div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MovieID</th>
      <th>Title</th>
      <th>Genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children's|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-65710087-37dc-422a-9d7d-16a4b89d8b71')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

<div class="k-default-codeblock">
```
.colab-df-convert {
  background-color: #E8F0FE;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  display: none;
  fill: #1967D2;
  height: 32px;
  padding: 0 0 0 0;
  width: 32px;
}

.colab-df-convert:hover {
  background-color: #E2EBFA;
  box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
  fill: #174EA6;
}

.colab-df-buttons div {
  margin-bottom: 4px;
}

[theme=dark] .colab-df-convert {
  background-color: #3B4455;
  fill: #D2E3FC;
}

[theme=dark] .colab-df-convert:hover {
  background-color: #434B5C;
  box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
  filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
  fill: #FFFFFF;
}
```
</div>
  </style>

<div class="k-default-codeblock">
```
<script>
  const buttonEl =
    document.querySelector('#df-65710087-37dc-422a-9d7d-16a4b89d8b71 button.colab-df-convert');
  buttonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
    const element = document.querySelector('#df-65710087-37dc-422a-9d7d-16a4b89d8b71');
    const dataTable =
      await google.colab.kernel.invokeFunction('convertToInteractive',
                                                [key], {});
    if (!dataTable) return;

    const docLinkHtml = 'Like what you see? Visit the ' +
      '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
      + ' to learn more about interactive tables.';
    element.innerHTML = '';
    dataTable['output_type'] = 'display_data';
    await google.colab.output.renderOutput(dataTable, element);
    const docLink = document.createElement('div');
    docLink.innerHTML = docLinkHtml;
    element.appendChild(docLink);
  }
</script>
```
</div>
  </div>

<div class="k-default-codeblock">
```
</div>
```
</div>
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
array([[2147, 3203, 2541, ..., 3175, 2716, 2676],
       [1784,  588, 3039, ..., 1883,  514, 1566],
       [2335, 2004,  237, ...,  432, 1918,  435],
       ...,
       [2031, 1839, 1011, ..., 1460, 3248, 2828],
       [3418, 3471, 1120, ...,  475, 3168, 1957],
       [ 832, 2145,  852, ..., 1355, 3307, 3730]], dtype=int32)>, <tf.Tensor: shape=(4096,), dtype=int32, numpy=array([2713, 3040, 2170, ..., 2055, 1535, 3741], dtype=int32)>)

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
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  5:41 2s/step - loss: 8.3178

<div class="k-default-codeblock">
```

```
</div>
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  3:32 1s/step - loss: 8.3176

<div class="k-default-codeblock">
```

```
</div>
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1:46 523ms/step - loss: 8.3169

<div class="k-default-codeblock">
```

```
</div>
   4/207 [37m━━━━━━━━━━━━━━━━━━━━  1:11 351ms/step - loss: 8.3147

<div class="k-default-codeblock">
```

```
</div>
  11/207 ━[37m━━━━━━━━━━━━━━━━━━━  21s 111ms/step - loss: 8.2757 

<div class="k-default-codeblock">
```

```
</div>
  12/207 ━[37m━━━━━━━━━━━━━━━━━━━  19s 101ms/step - loss: 8.2671

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  18s 94ms/step - loss: 8.2581 

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  16s 87ms/step - loss: 8.2487

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  15s 81ms/step - loss: 8.2390

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  14s 76ms/step - loss: 8.2292

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  13s 72ms/step - loss: 8.2192

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  12s 68ms/step - loss: 8.2092

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  12s 65ms/step - loss: 8.1993

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  11s 62ms/step - loss: 8.1893

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  11s 59ms/step - loss: 8.1794

<div class="k-default-codeblock">
```

```
</div>
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  10s 57ms/step - loss: 8.1695

<div class="k-default-codeblock">
```

```
</div>
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  10s 56ms/step - loss: 8.1597

<div class="k-default-codeblock">
```

```
</div>
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  9s 54ms/step - loss: 8.1500 

<div class="k-default-codeblock">
```

```
</div>
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  9s 53ms/step - loss: 8.1403

<div class="k-default-codeblock">
```

```
</div>
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  9s 51ms/step - loss: 8.1307

<div class="k-default-codeblock">
```

```
</div>
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  8s 46ms/step - loss: 8.1016
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  8s 50ms/step - loss: 8.1211

<div class="k-default-codeblock">
```

```
</div>
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  8s 48ms/step - loss: 8.0934

<div class="k-default-codeblock">
```

```
</div>
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  8s 46ms/step - loss: 8.0765

<div class="k-default-codeblock">
```

```
</div>
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  7s 43ms/step - loss: 8.0603

<div class="k-default-codeblock">
```

```
</div>
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  7s 44ms/step - loss: 8.0684

<div class="k-default-codeblock">
```

```
</div>
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  7s 42ms/step - loss: 8.0522

<div class="k-default-codeblock">
```

```
</div>
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  7s 41ms/step - loss: 8.0442

<div class="k-default-codeblock">
```

```
</div>
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  7s 42ms/step - loss: 8.0363

<div class="k-default-codeblock">
```

```
</div>
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  6s 41ms/step - loss: 8.0284
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  6s 40ms/step - loss: 8.0206

<div class="k-default-codeblock">
```

```
</div>
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  6s 39ms/step - loss: 8.0128

<div class="k-default-codeblock">
```

```
</div>
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 35ms/step - loss: 7.9688
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  6s 36ms/step - loss: 7.9625
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  6s 38ms/step - loss: 7.9903

<div class="k-default-codeblock">
```

```
</div>
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  6s 37ms/step - loss: 7.9972

<div class="k-default-codeblock">
```

```
</div>
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 36ms/step - loss: 7.9497

<div class="k-default-codeblock">
```

```
</div>
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 34ms/step - loss: 7.9370

<div class="k-default-codeblock">
```

```
</div>
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 35ms/step - loss: 7.9313

<div class="k-default-codeblock">
```

```
</div>
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 34ms/step - loss: 7.9195

<div class="k-default-codeblock">
```

```
</div>
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 34ms/step - loss: 7.9138

<div class="k-default-codeblock">
```

```
</div>
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 34ms/step - loss: 7.8969
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 33ms/step - loss: 7.9022

<div class="k-default-codeblock">
```

```
</div>
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 33ms/step - loss: 7.8861

<div class="k-default-codeblock">
```

```
</div>
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 32ms/step - loss: 7.8603

<div class="k-default-codeblock">
```

```
</div>
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 31ms/step - loss: 7.8703
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 31ms/step - loss: 7.8651
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 32ms/step - loss: 7.8809

<div class="k-default-codeblock">
```

```
</div>
  57/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 30ms/step - loss: 7.8362

<div class="k-default-codeblock">
```

```
</div>
  55/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 31ms/step - loss: 7.8457
  56/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 31ms/step - loss: 7.8410

<div class="k-default-codeblock">
```

```
</div>
  61/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 29ms/step - loss: 7.8168
  60/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 29ms/step - loss: 7.8124
  59/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 30ms/step - loss: 7.8082
  58/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 30ms/step - loss: 7.8042

<div class="k-default-codeblock">
```

```
</div>
  62/207 ━━━━━[37m━━━━━━━━━━━━━━━  4s 29ms/step - loss: 7.7796

<div class="k-default-codeblock">
```

```
</div>
  63/207 ━━━━━━[37m━━━━━━━━━━━━━━  4s 29ms/step - loss: 7.7834

<div class="k-default-codeblock">
```

```
</div>
  66/207 ━━━━━━[37m━━━━━━━━━━━━━━  3s 28ms/step - loss: 7.7464
  65/207 ━━━━━━[37m━━━━━━━━━━━━━━  4s 28ms/step - loss: 7.7680
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  3s 27ms/step - loss: 7.7430

<div class="k-default-codeblock">
```

```
</div>
  68/207 ━━━━━━[37m━━━━━━━━━━━━━━  3s 27ms/step - loss: 7.7565
  64/207 ━━━━━━[37m━━━━━━━━━━━━━━  4s 29ms/step - loss: 7.7532

<div class="k-default-codeblock">
```

```
</div>
  69/207 ━━━━━━[37m━━━━━━━━━━━━━━  3s 27ms/step - loss: 7.7362

<div class="k-default-codeblock">
```

```
</div>
  70/207 ━━━━━━[37m━━━━━━━━━━━━━━  3s 27ms/step - loss: 7.7329
  74/207 ━━━━━━━[37m━━━━━━━━━━━━━  3s 26ms/step - loss: 7.7130

<div class="k-default-codeblock">
```

```
</div>
  73/207 ━━━━━━━[37m━━━━━━━━━━━━━  3s 26ms/step - loss: 7.7099

<div class="k-default-codeblock">
```

```
</div>
  72/207 ━━━━━━[37m━━━━━━━━━━━━━━  3s 27ms/step - loss: 7.7261
  71/207 ━━━━━━[37m━━━━━━━━━━━━━━  3s 27ms/step - loss: 7.7229

<div class="k-default-codeblock">
```

```
</div>
  76/207 ━━━━━━━[37m━━━━━━━━━━━━━  3s 26ms/step - loss: 7.7005

<div class="k-default-codeblock">
```

```
</div>
  75/207 ━━━━━━━[37m━━━━━━━━━━━━━  3s 26ms/step - loss: 7.6975

<div class="k-default-codeblock">
```

```
</div>
  77/207 ━━━━━━━[37m━━━━━━━━━━━━━  3s 26ms/step - loss: 7.6915

<div class="k-default-codeblock">
```

```
</div>
  80/207 ━━━━━━━[37m━━━━━━━━━━━━━  3s 25ms/step - loss: 7.6824

<div class="k-default-codeblock">
```

```
</div>
  78/207 ━━━━━━━[37m━━━━━━━━━━━━━  3s 26ms/step - loss: 7.6767
  81/207 ━━━━━━━[37m━━━━━━━━━━━━━  3s 25ms/step - loss: 7.6795

<div class="k-default-codeblock">
```

```
</div>
  79/207 ━━━━━━━[37m━━━━━━━━━━━━━  3s 25ms/step - loss: 7.6739

<div class="k-default-codeblock">
```

```
</div>
  85/207 ━━━━━━━━[37m━━━━━━━━━━━━  3s 25ms/step - loss: 7.6411

<div class="k-default-codeblock">
```

```
</div>
  82/207 ━━━━━━━[37m━━━━━━━━━━━━━  3s 25ms/step - loss: 7.6655
  87/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 24ms/step - loss: 7.6517

<div class="k-default-codeblock">
```

```
</div>
  84/207 ━━━━━━━━[37m━━━━━━━━━━━━  3s 25ms/step - loss: 7.6600
  86/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 24ms/step - loss: 7.6545
  83/207 ━━━━━━━━[37m━━━━━━━━━━━━  3s 25ms/step - loss: 7.6628
  88/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 24ms/step - loss: 7.6436
  89/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 23ms/step - loss: 7.6462

<div class="k-default-codeblock">
```

```
</div>
  92/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 24ms/step - loss: 7.6229
  91/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 24ms/step - loss: 7.6182
  90/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 24ms/step - loss: 7.6205
  93/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 24ms/step - loss: 7.6134
  94/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 24ms/step - loss: 7.6110

<div class="k-default-codeblock">
```

```
</div>
  97/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 23ms/step - loss: 7.6038

<div class="k-default-codeblock">
```

```
</div>
  99/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 23ms/step - loss: 7.5856
  98/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 23ms/step - loss: 7.5947
  96/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 23ms/step - loss: 7.5836
 101/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 22ms/step - loss: 7.5878

<div class="k-default-codeblock">
```

```
</div>
 100/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 23ms/step - loss: 7.5752
  95/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 24ms/step - loss: 7.6015
 103/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 22ms/step - loss: 7.5628

<div class="k-default-codeblock">
```

```
</div>
 106/207 ━━━━━━━━━━[37m━━━━━━━━━━  2s 22ms/step - loss: 7.5568
 107/207 ━━━━━━━━━━[37m━━━━━━━━━━  2s 22ms/step - loss: 7.5548

<div class="k-default-codeblock">
```

```
</div>
 104/207 ━━━━━━━━━━[37m━━━━━━━━━━  2s 22ms/step - loss: 7.5669
 102/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 22ms/step - loss: 7.5710
 105/207 ━━━━━━━━━━[37m━━━━━━━━━━  2s 22ms/step - loss: 7.5648
 108/207 ━━━━━━━━━━[37m━━━━━━━━━━  2s 22ms/step - loss: 7.5528

<div class="k-default-codeblock">
```

```
</div>
 113/207 ━━━━━━━━━━[37m━━━━━━━━━━  1s 21ms/step - loss: 7.5373
 111/207 ━━━━━━━━━━[37m━━━━━━━━━━  2s 21ms/step - loss: 7.5469

<div class="k-default-codeblock">
```

```
</div>
 109/207 ━━━━━━━━━━[37m━━━━━━━━━━  2s 22ms/step - loss: 7.5450

<div class="k-default-codeblock">
```

```
</div>
 112/207 ━━━━━━━━━━[37m━━━━━━━━━━  2s 21ms/step - loss: 7.5392
 110/207 ━━━━━━━━━━[37m━━━━━━━━━━  2s 21ms/step - loss: 7.5489
 117/207 ━━━━━━━━━━━[37m━━━━━━━━━  1s 20ms/step - loss: 7.5297
 119/207 ━━━━━━━━━━━[37m━━━━━━━━━  1s 20ms/step - loss: 7.5259
 120/207 ━━━━━━━━━━━[37m━━━━━━━━━  1s 20ms/step - loss: 7.5240
 118/207 ━━━━━━━━━━━[37m━━━━━━━━━  1s 20ms/step - loss: 7.5278
 126/207 ━━━━━━━━━━━━[37m━━━━━━━━  1s 19ms/step - loss: 7.5001

<div class="k-default-codeblock">
```

```
</div>
 132/207 ━━━━━━━━━━━━[37m━━━━━━━━  1s 18ms/step - loss: 7.4895
 128/207 ━━━━━━━━━━━━[37m━━━━━━━━  1s 19ms/step - loss: 7.4879
 129/207 ━━━━━━━━━━━━[37m━━━━━━━━  1s 19ms/step - loss: 7.4862
 130/207 ━━━━━━━━━━━━[37m━━━━━━━━  1s 19ms/step - loss: 7.4846
 131/207 ━━━━━━━━━━━━[37m━━━━━━━━  1s 18ms/step - loss: 7.4829

<div class="k-default-codeblock">
```

```
</div>
 149/207 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 7.4179
 151/207 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 7.4192

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - loss: 7.2804

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 7s 25ms/step - loss: 7.2797 - val_loss: 5.9868


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  3:29 1s/step - loss: 6.6656

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6835 

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6871

    
   4/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6866

    
   5/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6867

<div class="k-default-codeblock">
```

```
</div>
   8/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6882

<div class="k-default-codeblock">
```

```
</div>
   9/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6884

<div class="k-default-codeblock">
```

```
</div>
  10/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6885

<div class="k-default-codeblock">
```

```
</div>
  11/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6885

<div class="k-default-codeblock">
```

```
</div>
  12/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6885

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6888

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6890
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.6889

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6888

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6888

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6888

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.6887

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.6886

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6883

<div class="k-default-codeblock">
```

```
</div>
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6878

<div class="k-default-codeblock">
```

```
</div>
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.6881

<div class="k-default-codeblock">
```

```
</div>
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6875

<div class="k-default-codeblock">
```

```
</div>
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6871

<div class="k-default-codeblock">
```

```
</div>
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6864
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6861

<div class="k-default-codeblock">
```

```
</div>
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6855

<div class="k-default-codeblock">
```

```
</div>
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6851

<div class="k-default-codeblock">
```

```
</div>
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6845
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.6848

<div class="k-default-codeblock">
```

```
</div>
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6842

<div class="k-default-codeblock">
```

```
</div>
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6837
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6831
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.6839

<div class="k-default-codeblock">
```

```
</div>
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6834

<div class="k-default-codeblock">
```

```
</div>
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6817
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.6823

<div class="k-default-codeblock">
```

```
</div>
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6825

<div class="k-default-codeblock">
```

```
</div>
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6815
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6812

<div class="k-default-codeblock">
```

```
</div>
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6807
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6804

<div class="k-default-codeblock">
```

```
</div>
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6799

<div class="k-default-codeblock">
```

```
</div>
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6790
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6793

<div class="k-default-codeblock">
```

```
</div>
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6787
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6796

<div class="k-default-codeblock">
```

```
</div>
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6778
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.6784

<div class="k-default-codeblock">
```

```
</div>
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6775
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6772

<div class="k-default-codeblock">
```

```
</div>
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6764
  56/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6751

<div class="k-default-codeblock">
```

```
</div>
  55/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6759

<div class="k-default-codeblock">
```

```
</div>
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6756

<div class="k-default-codeblock">
```

```
</div>
  57/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.6749

<div class="k-default-codeblock">
```

```
</div>
  58/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.6746

<div class="k-default-codeblock">
```

```
</div>
  59/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6743

<div class="k-default-codeblock">
```

```
</div>
  60/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6741

<div class="k-default-codeblock">
```

```
</div>
  61/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6738
  62/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6735
  63/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6733

<div class="k-default-codeblock">
```

```
</div>
  65/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6728
  64/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6731

<div class="k-default-codeblock">
```

```
</div>
  66/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6716
  68/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6721

<div class="k-default-codeblock">
```

```
</div>
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6718

<div class="k-default-codeblock">
```

```
</div>
  69/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6709

<div class="k-default-codeblock">
```

```
</div>
  71/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6704
  70/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.6707
  74/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 15ms/step - loss: 6.6697

<div class="k-default-codeblock">
```

```
</div>
  75/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 15ms/step - loss: 6.6691
  72/207 ━━━━━━[37m━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6702
  76/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6684

<div class="k-default-codeblock">
```

```
</div>
  77/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.6686
  73/207 ━━━━━━━[37m━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6695

<div class="k-default-codeblock">
```

```
</div>
  80/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 15ms/step - loss: 6.6676
  78/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 15ms/step - loss: 6.6674

<div class="k-default-codeblock">
```

```
</div>
  79/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 15ms/step - loss: 6.6666
  82/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 15ms/step - loss: 6.6660
  83/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 14ms/step - loss: 6.6658

<div class="k-default-codeblock">
```

```
</div>
  85/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 14ms/step - loss: 6.6654

<div class="k-default-codeblock">
```

```
</div>
  84/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 14ms/step - loss: 6.6656
  88/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 14ms/step - loss: 6.6645

<div class="k-default-codeblock">
```

```
</div>
  86/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 14ms/step - loss: 6.6649
  81/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 15ms/step - loss: 6.6668
  87/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 14ms/step - loss: 6.6651

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 2s 6ms/step - loss: 6.6303 - val_loss: 5.9325


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5474

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5741

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5778

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5779

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5784

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5786

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5789

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5792

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5795

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5797

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5799

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5799
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5800

<div class="k-default-codeblock">
```

```
</div>
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5799
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5800

<div class="k-default-codeblock">
```

```
</div>
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5794
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5795

<div class="k-default-codeblock">
```

```
</div>
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5788
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5798
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5792
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5790

<div class="k-default-codeblock">
```

```
</div>
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5789
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5791

<div class="k-default-codeblock">
```

```
</div>
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5776
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5779
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5779

<div class="k-default-codeblock">
```

```
</div>
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5776

<div class="k-default-codeblock">
```

```
</div>
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5770
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5768
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5770
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5772
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5767
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5773
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5766

<div class="k-default-codeblock">
```

```
</div>
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5765

<div class="k-default-codeblock">
```

```
</div>
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5758
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5753

<div class="k-default-codeblock">
```

```
</div>
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5754

<div class="k-default-codeblock">
```

```
</div>
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5759

<div class="k-default-codeblock">
```

```
</div>
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5747
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.5750

<div class="k-default-codeblock">
```

```
</div>
  56/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5742
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5748
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.5740
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5741
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5745

<div class="k-default-codeblock">
```

```
</div>
  55/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5742

<div class="k-default-codeblock">
```

```
</div>
  59/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5730
  58/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5731
  60/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5729
  57/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5732

<div class="k-default-codeblock">
```

```
</div>
  63/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5725
  64/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5724
  62/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5728
  68/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5721
  61/207 ━━━━━[37m━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.5727
  65/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5723

<div class="k-default-codeblock">
```

```
</div>
  66/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5722

<div class="k-default-codeblock">
```

```
</div>
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5720

<div class="k-default-codeblock">
```

```
</div>
  72/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5716
  71/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5714

<div class="k-default-codeblock">
```

```
</div>
  70/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5715

<div class="k-default-codeblock">
```

```
</div>
  69/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5718

<div class="k-default-codeblock">
```

```
</div>
  75/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5710
  77/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5702
  78/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5706
  73/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 15ms/step - loss: 6.5712
  74/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5711
  79/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5707
  80/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5704

<div class="k-default-codeblock">
```

```
</div>
  76/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5703

<div class="k-default-codeblock">
```

```
</div>
  82/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5696
  84/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 14ms/step - loss: 6.5697

<div class="k-default-codeblock">
```

```
</div>
  83/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 14ms/step - loss: 6.5695

<div class="k-default-codeblock">
```

```
</div>
  81/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5699
  89/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5689
  85/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5694
  88/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5692

<div class="k-default-codeblock">
```

```
</div>
  87/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5692
  86/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5691
  90/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5688

<div class="k-default-codeblock">
```

```
</div>
  91/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5687

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 6.5514 - val_loss: 5.9509


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5184

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5444

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5469

    
   4/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5457

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5476

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5480

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5481

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5483

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5485

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5487

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5488
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5488

<div class="k-default-codeblock">
```

```
</div>
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5482

<div class="k-default-codeblock">
```

```
</div>
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5486
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5484
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5487

<div class="k-default-codeblock">
```

```
</div>
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5462
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5461

<div class="k-default-codeblock">
```

```
</div>
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5464
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5477
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5458
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5459
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5475

<div class="k-default-codeblock">
```

```
</div>
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5457

<div class="k-default-codeblock">
```

```
</div>
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5445
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5449
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5448

<div class="k-default-codeblock">
```

```
</div>
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5446

<div class="k-default-codeblock">
```

```
</div>
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5444

<div class="k-default-codeblock">
```

```
</div>
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5440
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5433
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5431
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5441
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5430
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5429

<div class="k-default-codeblock">
```

```
</div>
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5428

<div class="k-default-codeblock">
```

```
</div>
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5415
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.5421
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5399
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5397
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5416
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5412

<div class="k-default-codeblock">
```

```
</div>
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5398

<div class="k-default-codeblock">
```

```
</div>
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5395
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5411

<div class="k-default-codeblock">
```

```
</div>
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5394

<div class="k-default-codeblock">
```

```
</div>
  56/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5389
  55/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5388

<div class="k-default-codeblock">
```

```
</div>
  58/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5385
  62/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5381
  63/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5372

<div class="k-default-codeblock">
```

```
</div>
  60/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5383
  57/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5386
  66/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5377

<div class="k-default-codeblock">
```

```
</div>
  65/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5378
  64/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5379

<div class="k-default-codeblock">
```

```
</div>
  59/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5376

<div class="k-default-codeblock">
```

```
</div>
  61/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5374

<div class="k-default-codeblock">
```

```
</div>
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5367
  68/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5368

<div class="k-default-codeblock">
```

```
</div>
  71/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5358

<div class="k-default-codeblock">
```

```
</div>
  74/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5360
  75/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5355
  72/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5353
  70/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5359
  69/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5353

<div class="k-default-codeblock">
```

```
</div>
  73/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5354

<div class="k-default-codeblock">
```

```
</div>
  76/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5347
  78/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5345
  77/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5346

<div class="k-default-codeblock">
```

```
</div>
  80/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5344

<div class="k-default-codeblock">
```

```
</div>
  81/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5342
  79/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5343

<div class="k-default-codeblock">
```

```
</div>
  83/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5334
  89/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5336
  87/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5337
  82/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 14ms/step - loss: 6.5341
  88/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5336
  86/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5338

<div class="k-default-codeblock">
```

```
</div>
  90/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5335
  85/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5333
  84/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5339

<div class="k-default-codeblock">
```

```
</div>
  93/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5327

<div class="k-default-codeblock">
```

```
</div>
  92/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5328
  91/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5329

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 6.5177 - val_loss: 5.9444


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.4880

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5147

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5166

    
   4/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5150

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5162

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5165

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5166

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5166

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5168

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5168

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5169

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5169

<div class="k-default-codeblock">
```

```
</div>
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5162
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5166
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5164

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5167

<div class="k-default-codeblock">
```

```
</div>
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5145

<div class="k-default-codeblock">
```

```
</div>
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5147
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5158
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5140

<div class="k-default-codeblock">
```

```
</div>
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5138
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5139

<div class="k-default-codeblock">
```

```
</div>
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.5142
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.5132

<div class="k-default-codeblock">
```

```
</div>
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5127
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5129

<div class="k-default-codeblock">
```

```
</div>
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5128

<div class="k-default-codeblock">
```

```
</div>
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5131

<div class="k-default-codeblock">
```

```
</div>
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.5124

<div class="k-default-codeblock">
```

```
</div>
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5123
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5118

<div class="k-default-codeblock">
```

```
</div>
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5119
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5122
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5117

<div class="k-default-codeblock">
```

```
</div>
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5111
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5112

<div class="k-default-codeblock">
```

```
</div>
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5106
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.5109
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5104
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5099
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5103

<div class="k-default-codeblock">
```

```
</div>
  56/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5086
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.5098
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5091
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5090
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5095

<div class="k-default-codeblock">
```

```
</div>
  55/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5087
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5094

<div class="k-default-codeblock">
```

```
</div>
  57/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5085

<div class="k-default-codeblock">
```

```
</div>
  58/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5084

<div class="k-default-codeblock">
```

```
</div>
  60/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5079
  66/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5071
  59/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5080
  61/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5081

<div class="k-default-codeblock">
```

```
</div>
  62/207 ━━━━━[37m━━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5074

<div class="k-default-codeblock">
```

```
</div>
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5068
  65/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5075

<div class="k-default-codeblock">
```

```
</div>
  64/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5070
  63/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5068

<div class="k-default-codeblock">
```

```
</div>
  68/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5065
  69/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5064

<div class="k-default-codeblock">
```

```
</div>
  70/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5063

<div class="k-default-codeblock">
```

```
</div>
  76/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5059
  75/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5055
  72/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5062
  78/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5056
  77/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5058

<div class="k-default-codeblock">
```

```
</div>
  74/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5056
  71/207 ━━━━━━[37m━━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5063

<div class="k-default-codeblock">
```

```
</div>
  79/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 12ms/step - loss: 6.5057
  73/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5054

<div class="k-default-codeblock">
```

```
</div>
  82/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5049

<div class="k-default-codeblock">
```

```
</div>
  80/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5051
  81/207 ━━━━━━━[37m━━━━━━━━━━━━━  1s 13ms/step - loss: 6.5049

<div class="k-default-codeblock">
```

```
</div>
  85/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5044
  87/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5043
  83/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 14ms/step - loss: 6.5048

<div class="k-default-codeblock">
```

```
</div>
  88/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5042
  84/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5047
  86/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5040
  90/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5040
  91/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5041

<div class="k-default-codeblock">
```

```
</div>
  89/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 13ms/step - loss: 6.5044
  93/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5037

<div class="k-default-codeblock">
```

```
</div>
  92/207 ━━━━━━━━[37m━━━━━━━━━━━━  1s 12ms/step - loss: 6.5037

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 6.4935 - val_loss: 5.9402





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7d54219eec10>

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

Shaft (1971), Battle for the Planet of the Apes (1973), Death Wish (1974), Star Trek: The Motion Picture (1979), Family Plot (1976), Hustler, The (1961), Midnight Cowboy (1969), Manchurian Candidate, The (1962), They Shoot Horses, Don't They? (1969), Seven Days in May (1964)

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  0s 246ms/step


```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 248ms/step


    
<div class="k-default-codeblock">
```
==> Recommended movies for the above sequence:
Bonheur, Le (1965)
Seven Days in May (1964)
Midnight Cowboy (1969)
Splendor in the Grass (1961)
Man for All Seasons, A (1966)
Pawnbroker, The (1965)
Inherit the Wind (1960)
Death in the Garden (Mort en ce jardin, La) (1956)
In the Heat of the Night (1967)
Hustler, The (1961)

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()

```
</div>
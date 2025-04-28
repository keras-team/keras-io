# Recommending movies: retrieval using a sequential model [GRU4Rec]

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
   40960/5917549 [37m━━━━━━━━━━━━━━━━━━━━  13s 2us/step

<div class="k-default-codeblock">
```

```
</div>
  172032/5917549 [37m━━━━━━━━━━━━━━━━━━━━  6s 1us/step 

<div class="k-default-codeblock">
```

```
</div>
  696320/5917549 ━━[37m━━━━━━━━━━━━━━━━━━  1s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 1433600/5917549 ━━━━[37m━━━━━━━━━━━━━━━━  1s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 3284992/5917549 ━━━━━━━━━━━[37m━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 5668864/5917549 ━━━━━━━━━━━━━━━━━━━[37m━  0s 0us/step

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
Let's take a look at a few rows.


```python
ratings_df.head()
movies_df.head()
```





  <div id="df-fd918689-64ac-4111-8fa4-438d9a92c1d4" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-fd918689-64ac-4111-8fa4-438d9a92c1d4')"
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
    document.querySelector('#df-fd918689-64ac-4111-8fa4-438d9a92c1d4 button.colab-df-convert');
  buttonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
    const element = document.querySelector('#df-fd918689-64ac-4111-8fa4-438d9a92c1d4');
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
array([[ 934, 3516, 2915, ..., 1968, 1307, 2349],
       [2144, 3526, 2795, ..., 2245, 2416, 2111],
       [ 173, 1136, 1148, ..., 1394, 3504, 2716],
       ...,
       [ 260, 1196, 1200, ..., 1206, 1210, 1214],
       [2492, 2858, 1127, ..., 2997, 3285, 2683],
       [2693, 3911, 3916, ..., 2339, 1293,  316]], dtype=int32)>, <tf.Tensor: shape=(4096,), dtype=int32, numpy=array([1580, 2369,   34, ..., 1253, 2908, 3728], dtype=int32)>)

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
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  5:54 2s/step - loss: 8.3178

<div class="k-default-codeblock">
```

```
</div>
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  3:45 1s/step - loss: 8.3177

<div class="k-default-codeblock">
```

```
</div>
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1:52 553ms/step - loss: 8.3173

<div class="k-default-codeblock">
```

```
</div>
   4/207 [37m━━━━━━━━━━━━━━━━━━━━  1:15 372ms/step - loss: 8.3161

<div class="k-default-codeblock">
```

```
</div>
  11/207 ━[37m━━━━━━━━━━━━━━━━━━━  22s 117ms/step - loss: 8.2763 

<div class="k-default-codeblock">
```

```
</div>
  12/207 ━[37m━━━━━━━━━━━━━━━━━━━  20s 107ms/step - loss: 8.2684

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  19s 99ms/step - loss: 8.2602 

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  12s 65ms/step - loss: 8.1971

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  11s 62ms/step - loss: 8.1874

<div class="k-default-codeblock">
```

```
</div>
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  11s 60ms/step - loss: 8.1777

<div class="k-default-codeblock">
```

```
</div>
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  10s 57ms/step - loss: 8.1679

<div class="k-default-codeblock">
```

```
</div>
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  8s 45ms/step - loss: 8.0999 
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  7s 44ms/step - loss: 8.0904 

<div class="k-default-codeblock">
```

```
</div>
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  7s 43ms/step - loss: 8.0810

<div class="k-default-codeblock">
```

```
</div>
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  7s 42ms/step - loss: 8.0717

<div class="k-default-codeblock">
```

```
</div>
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  7s 41ms/step - loss: 8.0625

<div class="k-default-codeblock">
```

```
</div>
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  5s 33ms/step - loss: 7.9652

<div class="k-default-codeblock">
```

```
</div>
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  4s 29ms/step - loss: 7.9153

<div class="k-default-codeblock">
```

```
</div>
  59/207 ━━━━━[37m━━━━━━━━━━━━━━━  3s 26ms/step - loss: 7.8628

<div class="k-default-codeblock">
```

```
</div>
  67/207 ━━━━━━[37m━━━━━━━━━━━━━━  3s 24ms/step - loss: 7.8150

<div class="k-default-codeblock">
```

```
</div>
  74/207 ━━━━━━━[37m━━━━━━━━━━━━━  2s 22ms/step - loss: 7.7767

<div class="k-default-codeblock">
```

```
</div>
  81/207 ━━━━━━━[37m━━━━━━━━━━━━━  2s 21ms/step - loss: 7.7414

<div class="k-default-codeblock">
```

```
</div>
  88/207 ━━━━━━━━[37m━━━━━━━━━━━━  2s 20ms/step - loss: 7.7085

<div class="k-default-codeblock">
```

```
</div>
  95/207 ━━━━━━━━━[37m━━━━━━━━━━━  2s 19ms/step - loss: 7.6780

<div class="k-default-codeblock">
```

```
</div>
 102/207 ━━━━━━━━━[37m━━━━━━━━━━━  1s 18ms/step - loss: 7.6496

<div class="k-default-codeblock">
```

```
</div>
 108/207 ━━━━━━━━━━[37m━━━━━━━━━━  1s 18ms/step - loss: 7.6267

<div class="k-default-codeblock">
```

```
</div>
 115/207 ━━━━━━━━━━━[37m━━━━━━━━━  1s 17ms/step - loss: 7.6015

<div class="k-default-codeblock">
```

```
</div>
 123/207 ━━━━━━━━━━━[37m━━━━━━━━━  1s 16ms/step - loss: 7.5745

<div class="k-default-codeblock">
```

```
</div>
 131/207 ━━━━━━━━━━━━[37m━━━━━━━━  1s 16ms/step - loss: 7.5493

<div class="k-default-codeblock">
```

```
</div>
 139/207 ━━━━━━━━━━━━━[37m━━━━━━━  1s 15ms/step - loss: 7.5257

<div class="k-default-codeblock">
```

```
</div>
 147/207 ━━━━━━━━━━━━━━[37m━━━━━━  0s 15ms/step - loss: 7.5035

<div class="k-default-codeblock">
```

```
</div>
 154/207 ━━━━━━━━━━━━━━[37m━━━━━━  0s 14ms/step - loss: 7.4851

<div class="k-default-codeblock">
```

```
</div>
 161/207 ━━━━━━━━━━━━━━━[37m━━━━━  0s 14ms/step - loss: 7.4677

<div class="k-default-codeblock">
```

```
</div>
 169/207 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 7.4487

<div class="k-default-codeblock">
```

```
</div>
 177/207 ━━━━━━━━━━━━━━━━━[37m━━━  0s 13ms/step - loss: 7.4307

<div class="k-default-codeblock">
```

```
</div>
 184/207 ━━━━━━━━━━━━━━━━━[37m━━━  0s 13ms/step - loss: 7.4158

<div class="k-default-codeblock">
```

```
</div>
 192/207 ━━━━━━━━━━━━━━━━━━[37m━━  0s 13ms/step - loss: 7.3994

<div class="k-default-codeblock">
```

```
</div>
 199/207 ━━━━━━━━━━━━━━━━━━━[37m━  0s 13ms/step - loss: 7.3858

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 7.3709

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 10s 39ms/step - loss: 7.3691 - val_loss: 5.9861


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  3:50 1s/step - loss: 6.6767

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.6815 

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.6797

<div class="k-default-codeblock">
```

```
</div>
  11/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.6875
   9/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.6849
  10/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 8ms/step - loss: 6.6864
  12/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 7ms/step - loss: 6.6886

<div class="k-default-codeblock">
```

```
</div>
   7/207 [37m━━━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.6832
   8/207 [37m━━━━━━━━━━━━━━━━━━━━  2s 10ms/step - loss: 6.6839

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  3s 16ms/step - loss: 6.6916

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.6906
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  3s 17ms/step - loss: 6.6912
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.6925

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  3s 20ms/step - loss: 6.6929

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  3s 18ms/step - loss: 6.6931
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  3s 18ms/step - loss: 6.6933

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.6933

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  4s 22ms/step - loss: 6.6933
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 19ms/step - loss: 6.6933
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 20ms/step - loss: 6.6933

<div class="k-default-codeblock">
```

```
</div>
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  4s 22ms/step - loss: 6.6930

<div class="k-default-codeblock">
```

```
</div>
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.6926
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  4s 23ms/step - loss: 6.6929

<div class="k-default-codeblock">
```

```
</div>
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  4s 23ms/step - loss: 6.6920

<div class="k-default-codeblock">
```

```
</div>
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  4s 24ms/step - loss: 6.6918
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  4s 25ms/step - loss: 6.6919

<div class="k-default-codeblock">
```

```
</div>
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  4s 24ms/step - loss: 6.6910

<div class="k-default-codeblock">
```

```
</div>
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  4s 26ms/step - loss: 6.6909

<div class="k-default-codeblock">
```

```
</div>
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  4s 25ms/step - loss: 6.6912

<div class="k-default-codeblock">
```

```
</div>
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  4s 26ms/step - loss: 6.6903
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  4s 27ms/step - loss: 6.6901
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  4s 25ms/step - loss: 6.6898

<div class="k-default-codeblock">
```

```
</div>
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  4s 26ms/step - loss: 6.6893

<div class="k-default-codeblock">
```

```
</div>
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  4s 27ms/step - loss: 6.6895
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  4s 28ms/step - loss: 6.6896

<div class="k-default-codeblock">
```

```
</div>
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  4s 26ms/step - loss: 6.6877
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  4s 27ms/step - loss: 6.6887
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  4s 28ms/step - loss: 6.6885

<div class="k-default-codeblock">
```

```
</div>
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  4s 28ms/step - loss: 6.6889

<div class="k-default-codeblock">
```

```
</div>
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  4s 26ms/step - loss: 6.6875
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  4s 27ms/step - loss: 6.6879

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 2s 6ms/step - loss: 6.6486 - val_loss: 5.9451


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5821

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5860

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5816

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5882

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5887

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5892

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5896

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5900

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5903

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 11ms/step - loss: 6.5903

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5903
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5903
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5902
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5898
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5903

<div class="k-default-codeblock">
```

```
</div>
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.5897
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 11ms/step - loss: 6.5888

<div class="k-default-codeblock">
```

```
</div>
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 11ms/step - loss: 6.5886
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.5889
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.5893
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 12ms/step - loss: 6.5894

<div class="k-default-codeblock">
```

```
</div>
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.5884

<div class="k-default-codeblock">
```

```
</div>
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.5883

<div class="k-default-codeblock">
```

```
</div>
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.5875

<div class="k-default-codeblock">
```

```
</div>
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.5877

<div class="k-default-codeblock">
```

```
</div>
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.5874
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.5873

<div class="k-default-codeblock">
```

```
</div>
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 17ms/step - loss: 6.5867

<div class="k-default-codeblock">
```

```
</div>
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 18ms/step - loss: 6.5865
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 17ms/step - loss: 6.5859
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.5861

<div class="k-default-codeblock">
```

```
</div>
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  2s 18ms/step - loss: 6.5855

<div class="k-default-codeblock">
```

```
</div>
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 19ms/step - loss: 6.5857

<div class="k-default-codeblock">
```

```
</div>
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 19ms/step - loss: 6.5851
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 20ms/step - loss: 6.5852
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 20ms/step - loss: 6.5850

<div class="k-default-codeblock">
```

```
</div>
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 19ms/step - loss: 6.5846

<div class="k-default-codeblock">
```

```
</div>
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5844
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5843
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5839
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 20ms/step - loss: 6.5841
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5836
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5834

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 6.5605 

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 6.5605 - val_loss: 5.9451


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5440

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 5ms/step - loss: 6.5484

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5427

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5463

<div class="k-default-codeblock">
```

```
</div>
  12/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 6ms/step - loss: 6.5455

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5469

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5476

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5481

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5486

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5495

<div class="k-default-codeblock">
```

```
</div>
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 11ms/step - loss: 6.5497

<div class="k-default-codeblock">
```

```
</div>
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5494 
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 11ms/step - loss: 6.5490

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5492

<div class="k-default-codeblock">
```

```
</div>
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 14ms/step - loss: 6.5499

<div class="k-default-codeblock">
```

```
</div>
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.5498

<div class="k-default-codeblock">
```

```
</div>
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 17ms/step - loss: 6.5492
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.5495

<div class="k-default-codeblock">
```

```
</div>
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.5494
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 17ms/step - loss: 6.5493

<div class="k-default-codeblock">
```

```
</div>
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 18ms/step - loss: 6.5487

<div class="k-default-codeblock">
```

```
</div>
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 17ms/step - loss: 6.5484

<div class="k-default-codeblock">
```

```
</div>
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 19ms/step - loss: 6.5486

<div class="k-default-codeblock">
```

```
</div>
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 20ms/step - loss: 6.5480

<div class="k-default-codeblock">
```

```
</div>
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 20ms/step - loss: 6.5479

<div class="k-default-codeblock">
```

```
</div>
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 18ms/step - loss: 6.5476
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 19ms/step - loss: 6.5481

<div class="k-default-codeblock">
```

```
</div>
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5475

<div class="k-default-codeblock">
```

```
</div>
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 20ms/step - loss: 6.5470
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5474

<div class="k-default-codeblock">
```

```
</div>
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 20ms/step - loss: 6.5471

<div class="k-default-codeblock">
```

```
</div>
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5468

<div class="k-default-codeblock">
```

```
</div>
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5469
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5467

<div class="k-default-codeblock">
```

```
</div>
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 23ms/step - loss: 6.5466
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 23ms/step - loss: 6.5465

<div class="k-default-codeblock">
```

```
</div>
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5463
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5462

<div class="k-default-codeblock">
```

```
</div>
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 23ms/step - loss: 6.5459
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 23ms/step - loss: 6.5457

<div class="k-default-codeblock">
```

```
</div>
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5456
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 23ms/step - loss: 6.5458
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5453
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5454

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 6.5274 - val_loss: 5.9559


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
   1/207 [37m━━━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5254

    
   2/207 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5275

    
   3/207 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5199

<div class="k-default-codeblock">
```

```
</div>
  13/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5244

<div class="k-default-codeblock">
```

```
</div>
  14/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5249

<div class="k-default-codeblock">
```

```
</div>
  15/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5255

<div class="k-default-codeblock">
```

```
</div>
  16/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5260

<div class="k-default-codeblock">
```

```
</div>
  17/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5265

<div class="k-default-codeblock">
```

```
</div>
  18/207 ━[37m━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 6.5269

<div class="k-default-codeblock">
```

```
</div>
  20/207 ━[37m━━━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5270

<div class="k-default-codeblock">
```

```
</div>
  21/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 10ms/step - loss: 6.5271
  23/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5272
  22/207 ━━[37m━━━━━━━━━━━━━━━━━━  1s 9ms/step - loss: 6.5272 
  19/207 ━[37m━━━━━━━━━━━━━━━━━━━  2s 11ms/step - loss: 6.5274

<div class="k-default-codeblock">
```

```
</div>
  24/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.5270
  25/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 13ms/step - loss: 6.5269

<div class="k-default-codeblock">
```

```
</div>
  26/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 17ms/step - loss: 6.5268

<div class="k-default-codeblock">
```

```
</div>
  27/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.5267
  29/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 15ms/step - loss: 6.5264
  28/207 ━━[37m━━━━━━━━━━━━━━━━━━  2s 16ms/step - loss: 6.5262

<div class="k-default-codeblock">
```

```
</div>
  31/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 18ms/step - loss: 6.5258
  30/207 ━━[37m━━━━━━━━━━━━━━━━━━  3s 18ms/step - loss: 6.5256

<div class="k-default-codeblock">
```

```
</div>
  34/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 19ms/step - loss: 6.5252
  32/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5251
  35/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 19ms/step - loss: 6.5248

<div class="k-default-codeblock">
```

```
</div>
  33/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 20ms/step - loss: 6.5251

<div class="k-default-codeblock">
```

```
</div>
  37/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5246

<div class="k-default-codeblock">
```

```
</div>
  36/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5247

<div class="k-default-codeblock">
```

```
</div>
  41/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5241

<div class="k-default-codeblock">
```

```
</div>
  38/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 23ms/step - loss: 6.5238
  40/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5239

<div class="k-default-codeblock">
```

```
</div>
  39/207 ━━━[37m━━━━━━━━━━━━━━━━━  3s 23ms/step - loss: 6.5240

<div class="k-default-codeblock">
```

```
</div>
  42/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5235

<div class="k-default-codeblock">
```

```
</div>
  44/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 23ms/step - loss: 6.5230
  43/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 23ms/step - loss: 6.5234

<div class="k-default-codeblock">
```

```
</div>
  46/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5231
  45/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5229
  47/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5227
  48/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5226

<div class="k-default-codeblock">
```

```
</div>
  49/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 23ms/step - loss: 6.5225

<div class="k-default-codeblock">
```

```
</div>
  51/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5223
  52/207 ━━━━━[37m━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5217

<div class="k-default-codeblock">
```

```
</div>
  50/207 ━━━━[37m━━━━━━━━━━━━━━━━  3s 22ms/step - loss: 6.5222

<div class="k-default-codeblock">
```

```
</div>
  53/207 ━━━━━[37m━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5218
  54/207 ━━━━━[37m━━━━━━━━━━━━━━━  3s 21ms/step - loss: 6.5215

<div class="k-default-codeblock">
```

```
</div>
 207/207 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 6.5049 - val_loss: 5.9625





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7c58fd953450>

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

Jerry Maguire (1996), Driving Miss Daisy (1989), Chasing Amy (1997), Boogie Nights (1997), Apollo 13 (1995), Breakfast Club, The (1985), Papillon (1973), Malcolm X (1992), Buddy Holly Story, The (1978), Dead Poets Society (1989)

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  0s 260ms/step


```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 261ms/step


    
<div class="k-default-codeblock">
```
==> Recommended movies for the above sequence:
Candidate, The (1972)
Great Santini, The (1979)
Buddy Holly Story, The (1978)
Field of Dreams (1989)
Rocky (1976)
Kramer Vs. Kramer (1979)
What Ever Happened to Baby Jane? (1962)
Midnight Express (1978)
Way We Were, The (1973)
Reds (1981)

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()

```
</div>
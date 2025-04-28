# Retrieval using a Transformer-based sequential model [SasRec]

**Author:** [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Recommend movies using a Transformer-based retrieval model (SASRec).


<div class='example_version_banner keras_2'>ⓘ This example uses Keras 2</div>
<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_rs/ipynb/sas_rec.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_rs/sas_rec.py)



---
## Introduction

Sequential recommendation is a popular model that looks at a sequence of items
that users have interacted with previously and then predicts the next item.
Here, the order of the items within each sequence matters. Previously, in the
[Recommending movies: retrieval using a sequential model](/keras_rs/examples/sequential_retrieval/)
example, we built a GRU-based sequential retrieval model. In this example, we
will build a popular Transformer decoder-based model named
[Self-Attentive Sequential Recommendation (SASRec)](https://arxiv.org/abs/1808.09781)
for the same sequential recommendation task.

Let's begin by importing all the necessary libraries.


```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # `"tensorflow"`/`"torch"`

import collections
import os

import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf  # Needed only for the dataset
from keras import ops

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
MAX_CONTEXT_LENGTH = 200
MIN_SEQUENCE_LENGTH = 3
PAD_ITEM_ID = 0

RATINGS_DATA_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
MOVIES_DATA_COLUMNS = ["MovieID", "Title", "Genres"]
MIN_RATING = 2

# Training/model args picked from SASRec paper
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

NUM_LAYERS = 2
NUM_HEADS = 1
HIDDEN_DIM = 50
DROPOUT = 0.2
```

---
## Dataset

Next, we need to prepare our dataset. Like we did in the
[sequential retrieval](/keras_rs/examples/sequential_retrieval/)
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
  204800/5917549 [37m━━━━━━━━━━━━━━━━━━━━  5s 1us/step 

<div class="k-default-codeblock">
```

```
</div>
  843776/5917549 ━━[37m━━━━━━━━━━━━━━━━━━  1s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 2678784/5917549 ━━━━━━━━━[37m━━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 5316608/5917549 ━━━━━━━━━━━━━━━━━[37m━━━  0s 0us/step

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





  <div id="df-392cb288-42dc-4837-bffb-114748d49735" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-392cb288-42dc-4837-bffb-114748d49735')"
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
    document.querySelector('#df-392cb288-42dc-4837-bffb-114748d49735 button.colab-df-convert');
  buttonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
    const element = document.querySelector('#df-392cb288-42dc-4837-bffb-114748d49735');
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


sequences = get_movie_sequence_per_user(ratings_df)
```

So far, we have essentially replicated what we did in the sequential retrieval
example. We have a sequence of movies for every user.

SASRec is trained contrastively, which means the model learns to distinguish
between sequences of movies a user has actually interacted with (positive
examples) and sequences they have not interacted with (negative examples).

The following function, `format_data`, prepares the data in this specific
format. For each user's movie sequence, it generates a corresponding
"negative sequence". This negative sequence consists of randomly
selected movies that the user has *not* interacted with, but are of the same
length as the original sequence.


```python

def format_data(sequences):
    examples = {
        "sequence": [],
        "negative_sequence": [],
    }

    for user_id in sequences:
        sequence = [int(d["movie_id"]) for d in sequences[user_id]]

        # Get negative sequence.
        def random_negative_item_id(low, high, positive_lst):
            sampled = np.random.randint(low=low, high=high)
            while sampled in positive_lst:
                sampled = np.random.randint(low=low, high=high)
            return sampled

        negative_sequence = [
            random_negative_item_id(1, movies_count + 1, sequence)
            for _ in range(len(sequence))
        ]

        examples["sequence"].append(np.array(sequence))
        examples["negative_sequence"].append(np.array(negative_sequence))

    examples["sequence"] = tf.ragged.constant(examples["sequence"])
    examples["negative_sequence"] = tf.ragged.constant(examples["negative_sequence"])

    return examples


examples = format_data(sequences)
ds = tf.data.Dataset.from_tensor_slices(examples).batch(BATCH_SIZE)
```

Now that we have the original movie interaction sequences for each user (from
`format_data`, stored in `examples["sequence"]`) and their corresponding
random negative sequences (in `examples["negative_sequence"]`), the next step is
to prepare this data for input to the model. The primary goals of this
preprocessing are:

1.  Creating Input Features and Target Labels: For sequential
    recommendation, the model learns to predict the next item in a sequence
    given the preceding items. This is achieved by:
    - taking the original `example["sequence"]` and creating the model's
      input features (`item_ids`) from all items *except the last one*
      (`example["sequence"][..., :-1]`);
    - creating the target "positive sequence" (what the model tries to predict
      as the actual next items) by taking the original `example["sequence"]`
      and shifting it, using all items *except the first one*
      (`example["sequence"][..., 1:]`);
    - shifting `example["negative_sequence"]` (from `format_data`) is
      to create the target "negative sequence" for the contrastive loss
      (`example["negative_sequence"][..., 1:]`).

2.  Handling Variable Length Sequences: Neural networks typically require
    fixed-size inputs. Therefore, both the input feature sequences and the
    target sequences are padded (with a special `PAD_ITEM_ID`) or truncated
    to a predefined `MAX_CONTEXT_LENGTH`. A `padding_mask` is also generated
    from the input features to ensure the model ignores these padded tokens
    during attention calculations, i.e, these tokens will be masked.

3.  Differentiating Training and Validation/Testing:
    - During training:
      - Input features (`item_ids`) and context for negative sequences
        are prepared as described above (all but the last item of the
        original sequences).
      - Target positive and negative sequences are the shifted versions of
        the original sequences.
        - `sample_weight` is created based on the input features to ensure
          that loss is calculated only on actual items, not on padding tokens
          in the targets.
    - During validation/testing:
      - Input features are prepared similarly.
      - The model's performance is typically evaluated on its ability to
        predict the actual last item of the original sequence. Thus,
        `sample_weight` is configured to focus the loss calculation
        only on this final prediction in the target sequences.

Note: SASRec does the same thing we've done above, except that they take the
`item_ids[:-2]` for the validation set and `item_ids[:-1]` for the test set.
We skip that here for brevity.


```python

def _preprocess(example, train=False):
    sequence = example["sequence"]
    negative_sequence = example["negative_sequence"]

    if train:
        sequence = example["sequence"][..., :-1]
        negative_sequence = example["negative_sequence"][..., :-1]

    batch_size = tf.shape(sequence)[0]

    if not train:
        # Loss computed only on last token.
        sample_weight = tf.zeros_like(sequence, dtype="float32")[..., :-1]
        sample_weight = tf.concat(
            [sample_weight, tf.ones((batch_size, 1), dtype="float32")], axis=1
        )

    # Truncate/pad sequence. +1 to account for truncation later.
    sequence = sequence.to_tensor(
        shape=[batch_size, MAX_CONTEXT_LENGTH + 1], default_value=PAD_ITEM_ID
    )
    negative_sequence = negative_sequence.to_tensor(
        shape=[batch_size, MAX_CONTEXT_LENGTH + 1], default_value=PAD_ITEM_ID
    )
    if train:
        sample_weight = tf.cast(sequence != PAD_ITEM_ID, dtype="float32")
    else:
        sample_weight = sample_weight.to_tensor(
            shape=[batch_size, MAX_CONTEXT_LENGTH + 1], default_value=0
        )

    example = (
        {
            # last token does not have a next token
            "item_ids": sequence[..., :-1],
            # padding mask for controlling attention mask
            "padding_mask": (sequence != PAD_ITEM_ID)[..., :-1],
        },
        {
            "positive_sequence": sequence[
                ..., 1:
            ],  # 0th token's label will be 1st token, and so on
            "negative_sequence": negative_sequence[..., 1:],
        },
        sample_weight[..., 1:],  # loss will not be computed on pad tokens
    )
    return example


def preprocess_train(examples):
    return _preprocess(examples, train=True)


def preprocess_val(examples):
    return _preprocess(examples, train=False)


train_ds = ds.map(preprocess_train)
val_ds = ds.map(preprocess_val)
```

We can see a batch for each.


```python
for batch in train_ds.take(1):
    print(batch)

for batch in val_ds.take(1):
    print(batch)

```

<div class="k-default-codeblock">
```
({'item_ids': <tf.Tensor: shape=(128, 200), dtype=int32, numpy=
array([[3186, 1270, 1721, ...,    0,    0,    0],
       [1198, 1210, 1217, ...,    0,    0,    0],
       [ 593, 2858, 3534, ...,    0,    0,    0],
       ...,
       [ 902, 1179, 1210, ...,    0,    0,    0],
       [1270, 3252, 1476, ...,    0,    0,    0],
       [2253, 3073, 1968, ...,    0,    0,    0]], dtype=int32)>, 'padding_mask': <tf.Tensor: shape=(128, 200), dtype=bool, numpy=
array([[ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False],
       ...,
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False]])>}, {'positive_sequence': <tf.Tensor: shape=(128, 200), dtype=int32, numpy=
array([[1270, 1721, 1022, ...,    0,    0,    0],
       [1210, 1217, 2717, ...,    0,    0,    0],
       [2858, 3534, 1968, ...,    0,    0,    0],
       ...,
       [1179, 1210, 3868, ...,    0,    0,    0],
       [3252, 1476,  260, ...,    0,    0,    0],
       [3073, 1968,  852, ...,    0,    0,    0]], dtype=int32)>, 'negative_sequence': <tf.Tensor: shape=(128, 200), dtype=int32, numpy=
array([[3772, 1934,  957, ...,    0,    0,    0],
       [1920,  175,  725, ...,    0,    0,    0],
       [3309,  176, 2662, ...,    0,    0,    0],
       ...,
       [1737, 2010, 1098, ...,    0,    0,    0],
       [3460, 2065, 3783, ...,    0,    0,    0],
       [1776, 1354, 2901, ...,    0,    0,    0]], dtype=int32)>}, <tf.Tensor: shape=(128, 200), dtype=float32, numpy=
array([[1., 1., 1., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.],
       ...,
       [1., 1., 1., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.],
       [1., 1., 1., ..., 0., 0., 0.]], dtype=float32)>)
({'item_ids': <tf.Tensor: shape=(128, 200), dtype=int32, numpy=
array([[3186, 1270, 1721, ...,    0,    0,    0],
       [1198, 1210, 1217, ...,    0,    0,    0],
       [ 593, 2858, 3534, ...,    0,    0,    0],
       ...,
       [ 902, 1179, 1210, ...,    0,    0,    0],
       [1270, 3252, 1476, ...,    0,    0,    0],
       [2253, 3073, 1968, ...,    0,    0,    0]], dtype=int32)>, 'padding_mask': <tf.Tensor: shape=(128, 200), dtype=bool, numpy=
array([[ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False],
       ...,
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False],
       [ True,  True,  True, ..., False, False, False]])>}, {'positive_sequence': <tf.Tensor: shape=(128, 200), dtype=int32, numpy=
array([[1270, 1721, 1022, ...,    0,    0,    0],
       [1210, 1217, 2717, ...,    0,    0,    0],
       [2858, 3534, 1968, ...,    0,    0,    0],
       ...,
       [1179, 1210, 3868, ...,    0,    0,    0],
       [3252, 1476,  260, ...,    0,    0,    0],
       [3073, 1968,  852, ...,    0,    0,    0]], dtype=int32)>, 'negative_sequence': <tf.Tensor: shape=(128, 200), dtype=int32, numpy=
array([[3772, 1934,  957, ...,    0,    0,    0],
       [1920,  175,  725, ...,    0,    0,    0],
       [3309,  176, 2662, ...,    0,    0,    0],
       ...,
       [1737, 2010, 1098, ...,    0,    0,    0],
       [3460, 2065, 3783, ...,    0,    0,    0],
       [1776, 1354, 2901, ...,    0,    0,    0]], dtype=int32)>}, <tf.Tensor: shape=(128, 200), dtype=float32, numpy=
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)>)

```
</div>
---
## Model

To encode the input sequence, we use a Transformer decoder-based model. This
part of the model is very similar to the GPT-2 architecture. Refer to the
[GPT text generation from scratch with KerasHub](/examples/generative/text_generation_gpt/#build-the-model)
guide for more details on this part.

One part to note is that when we are "predicting", i.e., `training` is `False`,
we get the embedding corresponding to the last movie in the sequence. This makes
sense, because at inference time, we want to predict the movie the user will
likely watch after watching the last movie.

Also, it's worth discussing the `compute_loss` method. We embed the positive
and negative sequences using the input embedding matrix. We compute the
similarity of (positive sequence, input sequence) and (negative sequence,
input sequence) pair embeddings by computing the dot product. The goal now is
to maximize the similarity of the former and minimize the similarity of
the latter. Let's see this mathematically. Binary Cross Entropy is written
as follows:

```
 loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

Here, we assign the positive pairs a label of 1 and the negative pairs a label
of 0. So, for a positive pair, the loss reduces to:

```
loss = -np.log(positive_logits)
```

Minimising the loss means we want to maximize the log term, which in turn,
implies maximising `positive_logits`. Similarly, we want to minimize
`negative_logits`.


```python

class SasRec(keras.Model):
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        dropout=0.0,
        max_sequence_length=100,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        # ======== Layers ========

        # === Embeddings ===
        self.item_embedding = keras_hub.layers.ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer="glorot_uniform",
            embeddings_regularizer=keras.regularizers.l2(0.001),
            dtype=dtype,
            name="item_embedding",
        )
        self.position_embedding = keras_hub.layers.PositionEmbedding(
            initializer="glorot_uniform",
            sequence_length=max_sequence_length,
            dtype=dtype,
            name="position_embedding",
        )
        self.embeddings_add = keras.layers.Add(
            dtype=dtype,
            name="embeddings_add",
        )
        self.embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )

        # === Decoder layers ===
        self.transformer_layers = []
        for i in range(num_layers):
            self.transformer_layers.append(
                keras_hub.layers.TransformerDecoder(
                    intermediate_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_norm_epsilon=1e-05,
                    # SASRec uses ReLU, although GeLU might be a better option
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    normalize_first=True,
                    dtype=dtype,
                    name=f"transformer_layer_{i}",
                )
            )

        # === Final layer norm ===
        self.layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-8,
            dtype=dtype,
            name="layer_norm",
        )

        # === Retrieval ===
        # The layer that performs the retrieval.
        self.retrieval = keras_rs.layers.BruteForceRetrieval(k=10, return_scores=False)

        # === Loss ===
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True, reduction=None)

        # === Attributes ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length

    def _get_last_non_padding_token(self, tensor, padding_mask):
        valid_token_mask = ops.logical_not(padding_mask)
        seq_lengths = ops.sum(ops.cast(valid_token_mask, "int32"), axis=1)
        last_token_indices = ops.maximum(seq_lengths - 1, 0)

        indices = ops.expand_dims(last_token_indices, axis=(-2, -1))
        gathered_tokens = ops.take_along_axis(tensor, indices, axis=1)
        last_token_embedding = ops.squeeze(gathered_tokens, axis=1)

        return last_token_embedding

    def build(self, input_shape):
        embedding_shape = list(input_shape) + [self.hidden_dim]

        # Model
        self.item_embedding.build(input_shape)
        self.position_embedding.build(embedding_shape)

        self.embeddings_add.build((embedding_shape, embedding_shape))
        self.embeddings_dropout.build(embedding_shape)

        for transformer_layer in self.transformer_layers:
            transformer_layer.build(decoder_sequence_shape=embedding_shape)

        self.layer_norm.build(embedding_shape)

        # Retrieval
        self.retrieval.candidate_embeddings = self.item_embedding.embeddings
        self.retrieval.build(input_shape)

        # Chain to super
        super().build(input_shape)

    def call(self, inputs, training=False):
        item_ids, padding_mask = inputs["item_ids"], inputs["padding_mask"]

        x = self.item_embedding(item_ids)
        position_embedding = self.position_embedding(x)
        x = self.embeddings_add((x, position_embedding))
        x = self.embeddings_dropout(x)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask)

        item_sequence_embedding = self.layer_norm(x)
        result = {"item_sequence_embedding": item_sequence_embedding}

        # At inference, perform top-k retrieval.
        if not training:
            # need to extract last non-padding token.
            last_item_embedding = self._get_last_non_padding_token(
                item_sequence_embedding, padding_mask
            )
            result["predictions"] = self.retrieval(last_item_embedding)

        return result

    def compute_loss(self, x, y, y_pred, sample_weight, training=False):
        item_sequence_embedding = y_pred["item_sequence_embedding"]
        y_positive_sequence = y["positive_sequence"]
        y_negative_sequence = y["negative_sequence"]

        # Embed positive, negative sequences.
        positive_sequence_embedding = self.item_embedding(y_positive_sequence)
        negative_sequence_embedding = self.item_embedding(y_negative_sequence)

        # Logits
        positive_logits = ops.sum(
            ops.multiply(positive_sequence_embedding, item_sequence_embedding),
            axis=-1,
        )
        negative_logits = ops.sum(
            ops.multiply(negative_sequence_embedding, item_sequence_embedding),
            axis=-1,
        )
        logits = ops.concatenate([positive_logits, negative_logits], axis=1)

        # Labels
        labels = ops.concatenate(
            [
                ops.ones_like(positive_logits),
                ops.zeros_like(negative_logits),
            ],
            axis=1,
        )

        # sample weights
        sample_weight = ops.concatenate(
            [sample_weight, sample_weight],
            axis=1,
        )

        loss = self.loss_fn(
            y_true=ops.expand_dims(labels, axis=-1),
            y_pred=ops.expand_dims(logits, axis=-1),
            sample_weight=sample_weight,
        )
        loss = ops.divide_no_nan(ops.sum(loss), ops.sum(sample_weight))

        return loss

    def compute_output_shape(self, inputs_shape):
        return list(inputs_shape) + [self.hidden_dim]

```

Let's instantiate our model and do some sanity checks.


```python
model = SasRec(
    vocabulary_size=movies_count + 1,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT,
    max_sequence_length=MAX_CONTEXT_LENGTH,
)

# Training
output = model(
    inputs={
        "item_ids": ops.ones((2, MAX_CONTEXT_LENGTH), dtype="int32"),
        "padding_mask": ops.ones((2, MAX_CONTEXT_LENGTH), dtype="bool"),
    },
    training=True,
)
print(output["item_sequence_embedding"].shape)

# Inference
output = model(
    inputs={
        "item_ids": ops.ones((2, MAX_CONTEXT_LENGTH), dtype="int32"),
        "padding_mask": ops.ones((2, MAX_CONTEXT_LENGTH), dtype="bool"),
    },
    training=False,
)
print(output["predictions"].shape)
```

<div class="k-default-codeblock">
```
(2, 200, 50)

(2, 10)

```
</div>
Now, let's compile and train our model.


```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_2=0.98),
)
model.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS,
)
```

<div class="k-default-codeblock">
```
Epoch 1/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  3:17 4s/step - loss: 0.6960

<div class="k-default-codeblock">
```

```
</div>
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  3:00 4s/step - loss: 0.6940

<div class="k-default-codeblock">
```

```
</div>
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  1:28 2s/step - loss: 0.6921

<div class="k-default-codeblock">
```

```
</div>
  4/48 ━[37m━━━━━━━━━━━━━━━━━━━  57s 1s/step - loss: 0.6902 

<div class="k-default-codeblock">
```

```
</div>
  5/48 ━━[37m━━━━━━━━━━━━━━━━━━  42s 993ms/step - loss: 0.6882

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  33s 797ms/step - loss: 0.6863

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  27s 667ms/step - loss: 0.6843

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  23s 589ms/step - loss: 0.6823

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  20s 519ms/step - loss: 0.6738

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  15s 415ms/step - loss: 0.6747
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  17s 461ms/step - loss: 0.6727

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  12s 354ms/step - loss: 0.6675
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  13s 386ms/step - loss: 0.6665

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  8s 276ms/step - loss: 0.6576 
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  10s 321ms/step - loss: 0.6551
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  11s 346ms/step - loss: 0.6559

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  9s 294ms/step - loss: 0.6565 
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  7s 266ms/step - loss: 0.6517

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  6s 238ms/step - loss: 0.6470
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  6s 226ms/step - loss: 0.6481

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  7s 251ms/step - loss: 0.6506
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  5s 215ms/step - loss: 0.6447
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  4s 199ms/step - loss: 0.6423
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  4s 184ms/step - loss: 0.6396

<div class="k-default-codeblock">
```

```
</div>
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  4s 191ms/step - loss: 0.6385

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  3s 166ms/step - loss: 0.6287
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  3s 172ms/step - loss: 0.6278
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  2s 160ms/step - loss: 0.6318
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  3s 178ms/step - loss: 0.6308

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  5s 208ms/step - loss: 0.6435

<div class="k-default-codeblock">
```

```
</div>
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  2s 155ms/step - loss: 0.6248

<div class="k-default-codeblock">
```

```
</div>
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  2s 148ms/step - loss: 0.6228
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  2s 153ms/step - loss: 0.6238
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  1s 137ms/step - loss: 0.6196
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  1s 142ms/step - loss: 0.6177
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  1s 135ms/step - loss: 0.6186

<div class="k-default-codeblock">
```

```
</div>
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  2s 147ms/step - loss: 0.6169
 43/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 116ms/step - loss: 0.6027
 47/48 ━━━━━━━━━━━━━━━━━━━[37m━  0s 106ms/step - loss: 0.6035
 45/48 ━━━━━━━━━━━━━━━━━━[37m━━  0s 111ms/step - loss: 0.6012
 39/48 ━━━━━━━━━━━━━━━━[37m━━━━  1s 128ms/step - loss: 0.6006
 46/48 ━━━━━━━━━━━━━━━━━━━[37m━  0s 108ms/step - loss: 0.5958

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 0s 164ms/step - loss: 0.5945

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 15s 226ms/step - loss: 0.5939 - val_loss: 0.5093


<div class="k-default-codeblock">
```
Epoch 2/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2:17 3s/step - loss: 0.4466

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4482

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4480

    
  4/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4472

<div class="k-default-codeblock">
```

```
</div>
  5/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4470

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4468

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  1s 27ms/step - loss: 0.4470

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  1s 32ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 28ms/step - loss: 0.4476
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  1s 30ms/step - loss: 0.4476

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  1s 34ms/step - loss: 0.4475
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  1s 38ms/step - loss: 0.4473

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  1s 30ms/step - loss: 0.4475

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  1s 34ms/step - loss: 0.4476

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  1s 32ms/step - loss: 0.4476
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  1s 36ms/step - loss: 0.4474
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  1s 34ms/step - loss: 0.4475
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 32ms/step - loss: 0.4473
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  1s 35ms/step - loss: 0.4477
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 32ms/step - loss: 0.4472

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step - loss: 0.4435

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 5s 52ms/step - loss: 0.4434 - val_loss: 0.4982


<div class="k-default-codeblock">
```
Epoch 3/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 45ms/step - loss: 0.4283

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4302

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4302

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4294

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4297

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4300

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4302

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4304

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 20ms/step - loss: 0.4306

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 20ms/step - loss: 0.4306

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 23ms/step - loss: 0.4307
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 26ms/step - loss: 0.4309
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 24ms/step - loss: 0.4309

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 24ms/step - loss: 0.4310

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 28ms/step - loss: 0.4310

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 27ms/step - loss: 0.4310
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 28ms/step - loss: 0.4310
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 25ms/step - loss: 0.4307
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 28ms/step - loss: 0.4308
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 27ms/step - loss: 0.4309

<div class="k-default-codeblock">
```

```
</div>
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 21ms/step - loss: 0.4303

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 20ms/step - loss: 0.4302
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 24ms/step - loss: 0.4306
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 22ms/step - loss: 0.4304
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 23ms/step - loss: 0.4304
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 24ms/step - loss: 0.4305

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 18ms/step - loss: 0.4292 - val_loss: 0.4824


<div class="k-default-codeblock">
```
Epoch 4/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 46ms/step - loss: 0.4158

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4177

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4164

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4166
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4168

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4170

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4171

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4172

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 18ms/step - loss: 0.4174

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 21ms/step - loss: 0.4172

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4173
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.4173
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 20ms/step - loss: 0.4173

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 23ms/step - loss: 0.4173

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 29ms/step - loss: 0.4173

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 25ms/step - loss: 0.4169
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 26ms/step - loss: 0.4172
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 24ms/step - loss: 0.4170
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 28ms/step - loss: 0.4172
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 24ms/step - loss: 0.4167

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.4144 - val_loss: 0.4641


<div class="k-default-codeblock">
```
Epoch 5/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 43ms/step - loss: 0.3969

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3990

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3974

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3975

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3977

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3978

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3978

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3978

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 18ms/step - loss: 0.3973
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.3971
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 20ms/step - loss: 0.3974
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.3973
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 21ms/step - loss: 0.3973
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.3974

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 18ms/step - loss: 0.3962
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 20ms/step - loss: 0.3967
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 21ms/step - loss: 0.3970

<div class="k-default-codeblock">
```

```
</div>
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 17ms/step - loss: 0.3963
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 22ms/step - loss: 0.3971
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 19ms/step - loss: 0.3968

<div class="k-default-codeblock">
```

```
</div>
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 22ms/step - loss: 0.3960
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 20ms/step - loss: 0.3958
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 19ms/step - loss: 0.3955
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 19ms/step - loss: 0.3954
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 21ms/step - loss: 0.3957

<div class="k-default-codeblock">
```

```
</div>
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 18ms/step - loss: 0.3953

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 0.3929

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.3929 - val_loss: 0.4380


<div class="k-default-codeblock">
```
Epoch 6/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 44ms/step - loss: 0.3683

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3702

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3701

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3689

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3690

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3691

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3691

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3691

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3690

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 17ms/step - loss: 0.3687
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 18ms/step - loss: 0.3686

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 21ms/step - loss: 0.3690
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 19ms/step - loss: 0.3689
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.3684

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 22ms/step - loss: 0.3684

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 22ms/step - loss: 0.3683

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 25ms/step - loss: 0.3678

<div class="k-default-codeblock">
```

```
</div>
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 23ms/step - loss: 0.3675
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 26ms/step - loss: 0.3682
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 24ms/step - loss: 0.3679
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 23ms/step - loss: 0.3674
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 22ms/step - loss: 0.3673

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.3646 - val_loss: 0.4222


<div class="k-default-codeblock">
```
Epoch 7/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 44ms/step - loss: 0.3459

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 15ms/step - loss: 0.3478

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3479

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3467

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3467

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3468

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3469

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3469

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 14ms/step - loss: 0.3469

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 20ms/step - loss: 0.3469

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 22ms/step - loss: 0.3469

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 21ms/step - loss: 0.3469

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 23ms/step - loss: 0.3467

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 24ms/step - loss: 0.3466

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 25ms/step - loss: 0.3465

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 29ms/step - loss: 0.3463

<div class="k-default-codeblock">
```

```
</div>
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 28ms/step - loss: 0.3463
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 27ms/step - loss: 0.3464
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 26ms/step - loss: 0.3460
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 24ms/step - loss: 0.3459

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 27ms/step - loss: 0.3461

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.3439 - val_loss: 0.4085


<div class="k-default-codeblock">
```
Epoch 8/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 44ms/step - loss: 0.3262

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3286

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3287

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3276

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3277

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3278

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3278

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3279

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 19ms/step - loss: 0.3279

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 20ms/step - loss: 0.3278

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 24ms/step - loss: 0.3278

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 22ms/step - loss: 0.3277
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 21ms/step - loss: 0.3277
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 25ms/step - loss: 0.3278

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 22ms/step - loss: 0.3276

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 23ms/step - loss: 0.3271
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 24ms/step - loss: 0.3271
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 22ms/step - loss: 0.3267
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 25ms/step - loss: 0.3270
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 28ms/step - loss: 0.3275
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 26ms/step - loss: 0.3270

<div class="k-default-codeblock">
```

```
</div>
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 21ms/step - loss: 0.3264
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 23ms/step - loss: 0.3263
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 22ms/step - loss: 0.3266

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 20ms/step - loss: 0.3262
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 23ms/step - loss: 0.3267

<div class="k-default-codeblock">
```

```
</div>
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 19ms/step - loss: 0.3264

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 18ms/step - loss: 0.3261 - val_loss: 0.3990


<div class="k-default-codeblock">
```
Epoch 9/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 43ms/step - loss: 0.3107

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3138

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3128

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 22ms/step - loss: 0.3127

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 21ms/step - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 17ms/step - loss: 0.3121
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 19ms/step - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 18ms/step - loss: 0.3123

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 24ms/step - loss: 0.3121

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 24ms/step - loss: 0.3118
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 28ms/step - loss: 0.3117
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 26ms/step - loss: 0.3119
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 29ms/step - loss: 0.3120

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 25ms/step - loss: 0.3116

<div class="k-default-codeblock">
```

```
</div>
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 24ms/step - loss: 0.3115

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 18ms/step - loss: 0.3112 - val_loss: 0.3898


<div class="k-default-codeblock">
```
Epoch 10/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 45ms/step - loss: 0.2977

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3013

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3018

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3008

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3006

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3005

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3005

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3005

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 15ms/step - loss: 0.3004

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 21ms/step - loss: 0.3004

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 22ms/step - loss: 0.3003
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 19ms/step - loss: 0.3001
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 21ms/step - loss: 0.3002

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 24ms/step - loss: 0.3000

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 25ms/step - loss: 0.2998

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 26ms/step - loss: 0.2996

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 29ms/step - loss: 0.2995
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 26ms/step - loss: 0.2993
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 27ms/step - loss: 0.2997
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 27ms/step - loss: 0.2992
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 25ms/step - loss: 0.2991

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 26ms/step - loss: 0.2991

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 20ms/step - loss: 0.2988 - val_loss: 0.3814





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7e079412c3d0>

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
for ele in val_ds.unbatch().take(1):
    test_sample = ele[0]
    test_sample["item_ids"] = tf.expand_dims(test_sample["item_ids"], axis=0)
    test_sample["padding_mask"] = tf.expand_dims(test_sample["padding_mask"], axis=0)

movie_sequence = np.array(test_sample["item_ids"])[0]
for movie_id in movie_sequence:
    if movie_id == 0:
        continue
    print(movie_id_to_movie_title[movie_id], end="; ")
print()

predictions = model.predict(test_sample)["predictions"]
predictions = keras.ops.convert_to_numpy(predictions)

for movie_id in predictions[0]:
    print(movie_id_to_movie_title[movie_id])
```

<div class="k-default-codeblock">
```
Girl, Interrupted (1999); Back to the Future (1985); Titanic (1997); Cinderella (1950); Meet Joe Black (1998); Last Days of Disco, The (1998); Erin Brockovich (2000); Christmas Story, A (1983); To Kill a Mockingbird (1962); One Flew Over the Cuckoo's Nest (1975); Wallace & Gromit: The Best of Aardman Animation (1996); Star Wars: Episode IV - A New Hope (1977); Wizard of Oz, The (1939); Fargo (1996); Run Lola Run (Lola rennt) (1998); Rain Man (1988); Saving Private Ryan (1998); Awakenings (1990); Gigi (1958); Sound of Music, The (1965); Driving Miss Daisy (1989); Bambi (1942); Apollo 13 (1995); Mary Poppins (1964); E.T. the Extra-Terrestrial (1982); My Fair Lady (1964); Ben-Hur (1959); Big (1988); Sixth Sense, The (1999); Dead Poets Society (1989); James and the Giant Peach (1996); Ferris Bueller's Day Off (1986); Secret Garden, The (1993); Toy Story 2 (1999); Airplane! (1980); Pleasantville (1998); Dumbo (1941); Princess Bride, The (1987); Snow White and the Seven Dwarfs (1937); Miracle on 34th Street (1947); Ponette (1996); Schindler's List (1993); Beauty and the Beast (1991); Tarzan (1999); Close Shave, A (1995); Aladdin (1992); Toy Story (1995); Bug's Life, A (1998); Antz (1998); Hunchback of Notre Dame, The (1996); Hercules (1997); Mulan (1998); Pocahontas (1995); 

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 834ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 836ms/step


<div class="k-default-codeblock">
```
Toy Story (1995)
Groundhog Day (1993)
Forrest Gump (1994)
Aladdin (1992)
Back to the Future (1985)
As Good As It Gets (1997)
Matrix, The (1999)
Star Wars: Episode VI - Return of the Jedi (1983)
Shakespeare in Love (1998)
Braveheart (1995)

```
</div>
And that's all!


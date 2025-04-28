# Retrieval using a Transformer-based sequential model [SasRec]

**Author:** [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)<br>
**Date created:** 2025/04/28<br>
**Last modified:** 2025/04/28<br>
**Description:** Recommend movies using a Transformer-based retrieval model (SASRec).


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
    8192/5917549 [37m━━━━━━━━━━━━━━━━━━━━  1:57 20us/step

<div class="k-default-codeblock">
```

```
</div>
   40960/5917549 [37m━━━━━━━━━━━━━━━━━━━━  46s 8us/step  

<div class="k-default-codeblock">
```

```
</div>
   73728/5917549 [37m━━━━━━━━━━━━━━━━━━━━  38s 7us/step

<div class="k-default-codeblock">
```

```
</div>
  106496/5917549 [37m━━━━━━━━━━━━━━━━━━━━  35s 6us/step

<div class="k-default-codeblock">
```

```
</div>
  172032/5917549 [37m━━━━━━━━━━━━━━━━━━━━  27s 5us/step

<div class="k-default-codeblock">
```

```
</div>
  303104/5917549 ━[37m━━━━━━━━━━━━━━━━━━━  18s 3us/step

<div class="k-default-codeblock">
```

```
</div>
  581632/5917549 ━[37m━━━━━━━━━━━━━━━━━━━  10s 2us/step

<div class="k-default-codeblock">
```

```
</div>
 1122304/5917549 ━━━[37m━━━━━━━━━━━━━━━━━  5s 1us/step 

<div class="k-default-codeblock">
```

```
</div>
 2220032/5917549 ━━━━━━━[37m━━━━━━━━━━━━━  2s 1us/step

<div class="k-default-codeblock">
```

```
</div>
 4399104/5917549 ━━━━━━━━━━━━━━[37m━━━━━━  0s 0us/step

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
array([[1751,  324,  535, ...,    0,    0,    0],
       [2683, 1970, 3356, ...,    0,    0,    0],
       [2806,  537, 3863, ...,    0,    0,    0],
       ...,
       [3847, 2441, 1164, ...,    0,    0,    0],
       [2930, 3466,    5, ...,    0,    0,    0],
       [2195, 1902,   13, ...,    0,    0,    0]], dtype=int32)>}, <tf.Tensor: shape=(128, 200), dtype=float32, numpy=
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
array([[1751,  324,  535, ...,    0,    0,    0],
       [2683, 1970, 3356, ...,    0,    0,    0],
       [2806,  537, 3863, ...,    0,    0,    0],
       ...,
       [3847, 2441, 1164, ...,    0,    0,    0],
       [2930, 3466,    5, ...,    0,    0,    0],
       [2195, 1902,   13, ...,    0,    0,    0]], dtype=int32)>}, <tf.Tensor: shape=(128, 200), dtype=float32, numpy=
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
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  3:15 4s/step - loss: 0.6972

<div class="k-default-codeblock">
```

```
</div>
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  2:17 3s/step - loss: 0.6954

<div class="k-default-codeblock">
```

```
</div>
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  1:07 2s/step - loss: 0.6935

<div class="k-default-codeblock">
```

```
</div>
  4/48 ━[37m━━━━━━━━━━━━━━━━━━━  44s 1s/step - loss: 0.6915 

<div class="k-default-codeblock">
```

```
</div>
  5/48 ━━[37m━━━━━━━━━━━━━━━━━━  32s 759ms/step - loss: 0.6896

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  25s 611ms/step - loss: 0.6875

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  20s 512ms/step - loss: 0.6854

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  17s 441ms/step - loss: 0.6833

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  15s 388ms/step - loss: 0.6812

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  13s 347ms/step - loss: 0.6790

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  11s 313ms/step - loss: 0.6768

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  10s 286ms/step - loss: 0.6746

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  9s 263ms/step - loss: 0.6724 

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  8s 245ms/step - loss: 0.6702

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  7s 231ms/step - loss: 0.6680

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  6s 205ms/step - loss: 0.6633
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  6s 219ms/step - loss: 0.6616

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  5s 184ms/step - loss: 0.6560

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  5s 194ms/step - loss: 0.6578

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  4s 174ms/step - loss: 0.6541

<div class="k-default-codeblock">
```

```
</div>
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  4s 165ms/step - loss: 0.6523

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  3s 153ms/step - loss: 0.6484
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  3s 146ms/step - loss: 0.6465

<div class="k-default-codeblock">
```

```
</div>
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  3s 140ms/step - loss: 0.6401
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  4s 160ms/step - loss: 0.6450

<div class="k-default-codeblock">
```

```
</div>
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  3s 136ms/step - loss: 0.6386

<div class="k-default-codeblock">
```

```
</div>
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  2s 129ms/step - loss: 0.6324
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  2s 118ms/step - loss: 0.6282
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  2s 132ms/step - loss: 0.6370
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  2s 122ms/step - loss: 0.6296
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  1s 114ms/step - loss: 0.6268

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  2s 125ms/step - loss: 0.6338

<div class="k-default-codeblock">
```

```
</div>
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  1s 105ms/step - loss: 0.6225

<div class="k-default-codeblock">
```

```
</div>
 38/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 97ms/step - loss: 0.6110 
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  1s 100ms/step - loss: 0.6174
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  1s 109ms/step - loss: 0.6213
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  1s 103ms/step - loss: 0.6101

<div class="k-default-codeblock">
```

```
</div>
 39/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 96ms/step - loss: 0.6073 

<div class="k-default-codeblock">
```

```
</div>
 41/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 91ms/step - loss: 0.6054 
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  1s 112ms/step - loss: 0.6164
 40/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 94ms/step - loss: 0.6046 

<div class="k-default-codeblock">
```

```
</div>
 44/48 ━━━━━━━━━━━━━━━━━━[37m━━  0s 86ms/step - loss: 0.6009 

<div class="k-default-codeblock">
```

```
</div>
 42/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 91ms/step - loss: 0.5993 
 47/48 ━━━━━━━━━━━━━━━━━━━[37m━  0s 81ms/step - loss: 0.5953

<div class="k-default-codeblock">
```

```
</div>
 45/48 ━━━━━━━━━━━━━━━━━━[37m━━  0s 85ms/step - loss: 0.5969

<div class="k-default-codeblock">
```

```
</div>
 46/48 ━━━━━━━━━━━━━━━━━━━[37m━  0s 83ms/step - loss: 0.5961

<div class="k-default-codeblock">
```

```
</div>
 43/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 89ms/step - loss: 0.6001

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 0s 144ms/step - loss: 0.5946

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 14s 208ms/step - loss: 0.5938 - val_loss: 0.5023


<div class="k-default-codeblock">
```
Epoch 2/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2:16 3s/step - loss: 0.4458

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4472

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4476

    
  4/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4471

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
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4474

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4476

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4478

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4482

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4480

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4483

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 19ms/step - loss: 0.4484

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 21ms/step - loss: 0.4485

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 23ms/step - loss: 0.4486

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 21ms/step - loss: 0.4486

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 19ms/step - loss: 0.4486
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 21ms/step - loss: 0.4486

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 18ms/step - loss: 0.4485

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 20ms/step - loss: 0.4486

<div class="k-default-codeblock">
```

```
</div>
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 20ms/step - loss: 0.4484
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 19ms/step - loss: 0.4479

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 18ms/step - loss: 0.4481
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 17ms/step - loss: 0.4477

<div class="k-default-codeblock">
```

```
</div>
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 19ms/step - loss: 0.4475

<div class="k-default-codeblock">
```

```
</div>
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 19ms/step - loss: 0.4474

<div class="k-default-codeblock">
```

```
</div>
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 19ms/step - loss: 0.4464

<div class="k-default-codeblock">
```

```
</div>
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 20ms/step - loss: 0.4472
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 21ms/step - loss: 0.4469
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 19ms/step - loss: 0.4471

<div class="k-default-codeblock">
```

```
</div>
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 20ms/step - loss: 0.4466

<div class="k-default-codeblock">
```

```
</div>
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 19ms/step - loss: 0.4462
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 20ms/step - loss: 0.4463
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 19ms/step - loss: 0.4461
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 19ms/step - loss: 0.4467
 38/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 17ms/step - loss: 0.4459
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 18ms/step - loss: 0.4461
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 18ms/step - loss: 0.4460

<div class="k-default-codeblock">
```

```
</div>
 39/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 17ms/step - loss: 0.4459
 40/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 16ms/step - loss: 0.4458

<div class="k-default-codeblock">
```

```
</div>
 42/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 16ms/step - loss: 0.4457
 41/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 16ms/step - loss: 0.4457

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 4s 18ms/step - loss: 0.4451 - val_loss: 0.4965


<div class="k-default-codeblock">
```
Epoch 3/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 44ms/step - loss: 0.4293

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4309

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4315

    
  4/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4311

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4314

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4320

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4323

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4327

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4331

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4335

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4337

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4339

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4342

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 13ms/step - loss: 0.4344

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 15ms/step - loss: 0.4345

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 15ms/step - loss: 0.4346

<div class="k-default-codeblock">
```

```
</div>
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 14ms/step - loss: 0.4346
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4347

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.4347

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - loss: 0.4346

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 17ms/step - loss: 0.4345

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 16ms/step - loss: 0.4344
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - loss: 0.4343
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - loss: 0.4340
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 14ms/step - loss: 0.4341

<div class="k-default-codeblock">
```

```
</div>
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 16ms/step - loss: 0.4338

<div class="k-default-codeblock">
```

```
</div>
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 17ms/step - loss: 0.4338
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 19ms/step - loss: 0.4334
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.4333

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 17ms/step - loss: 0.4337
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 18ms/step - loss: 0.4335
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.4333

<div class="k-default-codeblock">
```

```
</div>
 39/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - loss: 0.4331
 38/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.4330
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 17ms/step - loss: 0.4332

<div class="k-default-codeblock">
```

```
</div>
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - loss: 0.4330
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 0.4330

<div class="k-default-codeblock">
```

```
</div>
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.4330

<div class="k-default-codeblock">
```

```
</div>
 40/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 0.4329

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - loss: 0.4325 - val_loss: 0.4763


<div class="k-default-codeblock">
```
Epoch 4/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 44ms/step - loss: 0.4144

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4163

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.4169

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4162

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4166
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4164

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4171
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4168

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4173
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4174

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4175

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4177

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4178

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.4178

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 13ms/step - loss: 0.4178

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 17ms/step - loss: 0.4178

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - loss: 0.4176

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.4176

<div class="k-default-codeblock">
```

```
</div>
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 13ms/step - loss: 0.4166
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - loss: 0.4160
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 13ms/step - loss: 0.4156

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 15ms/step - loss: 0.4165
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 12ms/step - loss: 0.4160
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 14ms/step - loss: 0.4164
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 14ms/step - loss: 0.4159

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.4154

<div class="k-default-codeblock">
```

```
</div>
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 16ms/step - loss: 0.4155

<div class="k-default-codeblock">
```

```
</div>
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.4152
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.4153
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.4152

<div class="k-default-codeblock">
```

```
</div>
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 0.4149
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - loss: 0.4148
 38/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.4147

<div class="k-default-codeblock">
```

```
</div>
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.4150
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 17ms/step - loss: 0.4150

<div class="k-default-codeblock">
```

```
</div>
 40/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 0.4146

<div class="k-default-codeblock">
```

```
</div>
 41/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.4145
 39/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - loss: 0.4146

<div class="k-default-codeblock">
```

```
</div>
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - loss: 0.4144

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - loss: 0.4136 - val_loss: 0.4540


<div class="k-default-codeblock">
```
Epoch 5/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 43ms/step - loss: 0.3880

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3898

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3902

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3893

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3894

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3894

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3895

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3896

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3897

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3898
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3897

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3898

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3898

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3897

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 14ms/step - loss: 0.3896

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 17ms/step - loss: 0.3895
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - loss: 0.3893

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.3894

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 14ms/step - loss: 0.3890
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 14ms/step - loss: 0.3892

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 16ms/step - loss: 0.3888
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - loss: 0.3877
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 13ms/step - loss: 0.3878
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 14ms/step - loss: 0.3882

<div class="k-default-codeblock">
```

```
</div>
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 13ms/step - loss: 0.3880
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - loss: 0.3876

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.3868

<div class="k-default-codeblock">
```

```
</div>
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 17ms/step - loss: 0.3865
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.3863
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.3862
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 0.3861

<div class="k-default-codeblock">
```

```
</div>
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.3859
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 16ms/step - loss: 0.3866

<div class="k-default-codeblock">
```

```
</div>
 38/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.3859
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 0.3861
 41/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.3857
 42/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.3853
 40/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 0.3857

<div class="k-default-codeblock">
```

```
</div>
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - loss: 0.3860

<div class="k-default-codeblock">
```

```
</div>
 43/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 13ms/step - loss: 0.3853

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - loss: 0.3845 - val_loss: 0.4290


<div class="k-default-codeblock">
```
Epoch 6/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 43ms/step - loss: 0.3645

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3655

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3658

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3650
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3651

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3651

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3654
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3652

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3656

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3657

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3658

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3659

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3659

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 13ms/step - loss: 0.3658

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 13ms/step - loss: 0.3658

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.3657
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 15ms/step - loss: 0.3658

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 15ms/step - loss: 0.3646

<div class="k-default-codeblock">
```

```
</div>
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 13ms/step - loss: 0.3643
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 12ms/step - loss: 0.3647
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 13ms/step - loss: 0.3644
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 14ms/step - loss: 0.3645

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 14ms/step - loss: 0.3645
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - loss: 0.3656

<div class="k-default-codeblock">
```

```
</div>
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - loss: 0.3647

<div class="k-default-codeblock">
```

```
</div>
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 15ms/step - loss: 0.3640
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.3641

<div class="k-default-codeblock">
```

```
</div>
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 14ms/step - loss: 0.3639
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.3639

<div class="k-default-codeblock">
```

```
</div>
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 14ms/step - loss: 0.3635
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 0.3635
 39/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - loss: 0.3634
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.3635
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 15ms/step - loss: 0.3636

<div class="k-default-codeblock">
```

```
</div>
 38/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 14ms/step - loss: 0.3635

<div class="k-default-codeblock">
```

```
</div>
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.3637
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 16ms/step - loss: 0.3638

<div class="k-default-codeblock">
```

```
</div>
 43/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 13ms/step - loss: 0.3633
 41/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.3634

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - loss: 0.3630 - val_loss: 0.4142


<div class="k-default-codeblock">
```
Epoch 7/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 43ms/step - loss: 0.3464

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3476

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3480

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3474

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3475
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3475

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3476

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3478

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3480

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3481

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3481

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3482

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3482

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 13ms/step - loss: 0.3481

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 13ms/step - loss: 0.3481

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.3481

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 17ms/step - loss: 0.3479
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 15ms/step - loss: 0.3476
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - loss: 0.3477

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 14ms/step - loss: 0.3475
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 12ms/step - loss: 0.3470
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 13ms/step - loss: 0.3471
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 13ms/step - loss: 0.3472

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - loss: 0.3479
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 14ms/step - loss: 0.3473

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.3468
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 13ms/step - loss: 0.3462

<div class="k-default-codeblock">
```

```
</div>
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 13ms/step - loss: 0.3464
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.3466
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 14ms/step - loss: 0.3464

<div class="k-default-codeblock">
```

```
</div>
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 14ms/step - loss: 0.3461
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.3467

<div class="k-default-codeblock">
```

```
</div>
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 16ms/step - loss: 0.3469

<div class="k-default-codeblock">
```

```
</div>
 39/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - loss: 0.3459
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - loss: 0.3460
 38/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.3459
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.3460
 44/48 ━━━━━━━━━━━━━━━━━━[37m━━  0s 13ms/step - loss: 0.3457
 40/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 0.3458

<div class="k-default-codeblock">
```

```
</div>
 43/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.3456
 42/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.3457

<div class="k-default-codeblock">
```

```
</div>
 41/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.3458

<div class="k-default-codeblock">
```

```
</div>
 45/48 ━━━━━━━━━━━━━━━━━━[37m━━  0s 13ms/step - loss: 0.3457

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - loss: 0.3454 - val_loss: 0.3994


<div class="k-default-codeblock">
```
Epoch 8/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  1s 42ms/step - loss: 0.3279

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3297

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3302

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3298

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3298
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3298

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3299

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3300

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3302

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3303
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3303

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3303

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3303

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3302

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 13ms/step - loss: 0.3302

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 17ms/step - loss: 0.3301
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - loss: 0.3300
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.3300

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 15ms/step - loss: 0.3296
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - loss: 0.3298
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 14ms/step - loss: 0.3293

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 16ms/step - loss: 0.3290
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 13ms/step - loss: 0.3292
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 14ms/step - loss: 0.3295

<div class="k-default-codeblock">
```

```
</div>
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 13ms/step - loss: 0.3291

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.3285

<div class="k-default-codeblock">
```

```
</div>
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 16ms/step - loss: 0.3284

<div class="k-default-codeblock">
```

```
</div>
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 16ms/step - loss: 0.3282
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 15ms/step - loss: 0.3282
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 15ms/step - loss: 0.3283
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.3281
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 0.3281
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 0.3280

<div class="k-default-codeblock">
```

```
</div>
 39/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - loss: 0.3280
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - loss: 0.3280
 41/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.3280

<div class="k-default-codeblock">
```

```
</div>
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - loss: 0.3280
 40/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 0.3280
 38/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.3280

<div class="k-default-codeblock">
```

```
</div>
 42/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.3279

<div class="k-default-codeblock">
```

```
</div>
 44/48 ━━━━━━━━━━━━━━━━━━[37m━━  0s 13ms/step - loss: 0.3279

<div class="k-default-codeblock">
```

```
</div>
 43/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 13ms/step - loss: 0.3279
 46/48 ━━━━━━━━━━━━━━━━━━━[37m━  0s 13ms/step - loss: 0.3279

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - loss: 0.3278 - val_loss: 0.3862


<div class="k-default-codeblock">
```
Epoch 9/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 43ms/step - loss: 0.3134

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3153

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3156

    
  4/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3152

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3149
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3150

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3148

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3149

<div class="k-default-codeblock">
```

```
</div>
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3150

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3150

<div class="k-default-codeblock">
```

```
</div>
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3151
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3150

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3150

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 14ms/step - loss: 0.3149

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.3149

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.3146
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 15ms/step - loss: 0.3147

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - loss: 0.3145
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 14ms/step - loss: 0.3142
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - loss: 0.3140
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 14ms/step - loss: 0.3141

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 15ms/step - loss: 0.3143

<div class="k-default-codeblock">
```

```
</div>
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 16ms/step - loss: 0.3137
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - loss: 0.3137

<div class="k-default-codeblock">
```

```
</div>
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 14ms/step - loss: 0.3136

<div class="k-default-codeblock">
```

```
</div>
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 15ms/step - loss: 0.3135

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 16ms/step - loss: 0.3135
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 17ms/step - loss: 0.3134
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 17ms/step - loss: 0.3134

<div class="k-default-codeblock">
```

```
</div>
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.3133
 38/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.3132
 39/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 17ms/step - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - loss: 0.3132
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 0.3132
 40/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 18ms/step - loss: 0.3133

<div class="k-default-codeblock">
```

```
</div>
 41/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.3132

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - loss: 0.3132 - val_loss: 0.3769


<div class="k-default-codeblock">
```
Epoch 10/10

```
</div>
    
  1/48 [37m━━━━━━━━━━━━━━━━━━━━  2s 45ms/step - loss: 0.3027

    
  2/48 [37m━━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3045

    
  3/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 11ms/step - loss: 0.3047

    
  4/48 ━[37m━━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3043

<div class="k-default-codeblock">
```

```
</div>
  6/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3039

<div class="k-default-codeblock">
```

```
</div>
  7/48 ━━[37m━━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3037

<div class="k-default-codeblock">
```

```
</div>
  8/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3035

<div class="k-default-codeblock">
```

```
</div>
  9/48 ━━━[37m━━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3034

<div class="k-default-codeblock">
```

```
</div>
 10/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3034

<div class="k-default-codeblock">
```

```
</div>
 12/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3035
 11/48 ━━━━[37m━━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3035

<div class="k-default-codeblock">
```

```
</div>
 13/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3035
 14/48 ━━━━━[37m━━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3034

<div class="k-default-codeblock">
```

```
</div>
 15/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 12ms/step - loss: 0.3034

<div class="k-default-codeblock">
```

```
</div>
 16/48 ━━━━━━[37m━━━━━━━━━━━━━━  0s 14ms/step - loss: 0.3033

<div class="k-default-codeblock">
```

```
</div>
 17/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 17ms/step - loss: 0.3032

<div class="k-default-codeblock">
```

```
</div>
 18/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 16ms/step - loss: 0.3029

<div class="k-default-codeblock">
```

```
</div>
 19/48 ━━━━━━━[37m━━━━━━━━━━━━━  0s 15ms/step - loss: 0.3030

<div class="k-default-codeblock">
```

```
</div>
 20/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 15ms/step - loss: 0.3028

<div class="k-default-codeblock">
```

```
</div>
 22/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 15ms/step - loss: 0.3025
 21/48 ━━━━━━━━[37m━━━━━━━━━━━━  0s 16ms/step - loss: 0.3025

<div class="k-default-codeblock">
```

```
</div>
 23/48 ━━━━━━━━━[37m━━━━━━━━━━━  0s 15ms/step - loss: 0.3021
 25/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 14ms/step - loss: 0.3022

<div class="k-default-codeblock">
```

```
</div>
 24/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 14ms/step - loss: 0.3022

<div class="k-default-codeblock">
```

```
</div>
 26/48 ━━━━━━━━━━[37m━━━━━━━━━━  0s 15ms/step - loss: 0.3019

<div class="k-default-codeblock">
```

```
</div>
 27/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 16ms/step - loss: 0.3019

<div class="k-default-codeblock">
```

```
</div>
 28/48 ━━━━━━━━━━━[37m━━━━━━━━━  0s 15ms/step - loss: 0.3018

<div class="k-default-codeblock">
```

```
</div>
 32/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.3017
 30/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 17ms/step - loss: 0.3018

<div class="k-default-codeblock">
```

```
</div>
 29/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 16ms/step - loss: 0.3018
 35/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 16ms/step - loss: 0.3016
 31/48 ━━━━━━━━━━━━[37m━━━━━━━━  0s 16ms/step - loss: 0.3017

<div class="k-default-codeblock">
```

```
</div>
 37/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.3016

<div class="k-default-codeblock">
```

```
</div>
 40/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 14ms/step - loss: 0.3017
 39/48 ━━━━━━━━━━━━━━━━[37m━━━━  0s 15ms/step - loss: 0.3017
 36/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 16ms/step - loss: 0.3016

<div class="k-default-codeblock">
```

```
</div>
 38/48 ━━━━━━━━━━━━━━━[37m━━━━━  0s 15ms/step - loss: 0.3017
 44/48 ━━━━━━━━━━━━━━━━━━[37m━━  0s 13ms/step - loss: 0.3018
 34/48 ━━━━━━━━━━━━━━[37m━━━━━━  0s 17ms/step - loss: 0.3016
 43/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 13ms/step - loss: 0.3018

<div class="k-default-codeblock">
```

```
</div>
 46/48 ━━━━━━━━━━━━━━━━━━━[37m━  0s 13ms/step - loss: 0.3018

<div class="k-default-codeblock">
```

```
</div>
 45/48 ━━━━━━━━━━━━━━━━━━[37m━━  0s 13ms/step - loss: 0.3018
 42/48 ━━━━━━━━━━━━━━━━━[37m━━━  0s 14ms/step - loss: 0.3018
 33/48 ━━━━━━━━━━━━━[37m━━━━━━━  0s 17ms/step - loss: 0.3017

<div class="k-default-codeblock">
```

```
</div>
 48/48 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - loss: 0.3018 - val_loss: 0.3699





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x79b1902a68d0>

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
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 851ms/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 852ms/step


<div class="k-default-codeblock">
```
Forrest Gump (1994)
Titanic (1997)
Groundhog Day (1993)
There's Something About Mary (1998)
Star Wars: Episode IV - A New Hope (1977)
Sixth Sense, The (1999)
Aladdin (1992)
Matrix, The (1999)
Shakespeare in Love (1998)
Back to the Future (1985)

```
</div>
And that's all!

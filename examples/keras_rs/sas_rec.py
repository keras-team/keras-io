"""
Title: Retrieval using a Transformer-based sequential model [SasRec]
Author: [Abheesht Sharma](https://github.com/abheesht17/), [Fabien Hertschuh](https://github.com/hertschuh/)
Date created: 2025/04/28
Last modified: 2025/04/28
Description: Recommend movies using a Transformer-based retrieval model (SASRec).
Accelerator: GPU
"""

"""
## Introduction

Sequential recommendation is a popular model that looks at a sequence of items
that users have interacted with previously and then predicts the next item.
Here, the order of the items within each sequence matters. Previously, in the
[Recommending movies: retrieval using a sequential model](https://keras.io/keras_rs/examples/sequential_retrieval/)
example, we built a GRU-based sequential retrieval model. In this example, we
will build a popular Transformer decoder-based model named
[Self-Attentive Sequential Recommendation (SASRec)](https://arxiv.org/abs/1808.09781)
for the same sequential recommendation task.

Let's begin by importing all the necessary libraries.
"""

import collections
import os

import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf  # Needed only for the dataset
from keras import ops

import keras_rs

"""
Let's also define all important variables/hyperparameters below.
"""

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

"""
## Dataset

Next, we need to prepare our dataset. Like we did in the
[sequential retrieval](https://keras.io/keras_rs/examples/sequential_retrieval/)
example, we are going to use the MovieLens dataset. 

The dataset preparation step is fairly involved. The original ratings dataset
contains `(user, movie ID, rating, timestamp)` tuples (among other columns,
which are not important for this example). Since we are dealing with sequential
retrieval, we need to create movie sequences for every user, where the sequences
are ordered by timestamp.

Let's start by downloading and reading the dataset.
"""

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

"""
Let's take a look at a few rows.
"""
ratings_df.head()
movies_df.head()

"""
Now that we have read the dataset, let's create sequences of movies
for every user. Here is the function for doing just that.
"""


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

"""
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
"""


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

"""
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
"""


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

"""
We can see a batch for each.
"""

for batch in train_ds.take(1):
    print(batch)

for batch in val_ds.take(1):
    print(batch)


"""
## Model

To encode the input sequence, we use a Transformer decoder-based model. This
part of the model is very similar to the GPT-2 architecture. Refer to the
[GPT text generation from scratch with KerasHub](https://keras.io/examples/generative/text_generation_gpt/#build-the-model)
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
"""


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


"""
Let's instantiate our model and do some sanity checks.
"""

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

"""
Now, let's compile and train our model.
"""

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_2=0.98),
)
model.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS,
)

"""
## Making predictions

Now that we have a model, we would like to be able to make predictions.

So far, we have only handled movies by id. Now is the time to create a mapping
keyed by movie IDs to be able to surface the titles.
"""

movie_id_to_movie_title = dict(zip(movies_df["MovieID"], movies_df["Title"]))
movie_id_to_movie_title[0] = ""  # Because id 0 is not in the dataset.

"""
We then simply use the Keras `model.predict()` method. Under the hood, it calls
the `BruteForceRetrieval` layer to perform the actual retrieval.

Note that this model can retrieve movies already watched by the user. We could
easily add logic to remove them if that is desirable.
"""

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

"""
And that's all!
"""

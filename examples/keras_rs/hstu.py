"""
Title: A simplified HSTU Model for Movie Recommendations
Author: Lakshmi Kala Kadali[https://github.com/LakshmiKalaKadali]
Date created: 2025/11/04
Last modified: 2025/11/04
Description: An end-to-end implementation of a simplified HSTU model for movie recommendations.
Accelerator: GPU
"""

"""
## Introduction
This notebook demonstrates an end-to-end implementation of a simplified 
**Hierarchical Sequential Transduction Unit (HSTU)** model, based on the ICML'24 paper 
["Actions Speak Louder than Words"](https://proceedings.mlr.press/v235/zhai24a.html).
Generative Recommenders (GRs) represent a new paradigm for large-scale 
recommendation systems, reformulating traditional ranking and retrieval 
problems as sequential transduction tasks within a generative modeling 
framework. Unlike conventional Deep Learning Recommendation Models (DLRMs), 
which rely heavily on a massive number of handcrafted, heterogeneous 
features (numerical, categorical, embeddings) and struggle to scale with compute,
GRs consolidate and encode features into a single unified time series. 
This architectural shift is key to unlocking superior performance and scalability 
for recommendation engines, particularly those handling the daily actions of billions 
of users

"""

"""shell
!pip install -q keras-rs
"""

"""
First, we set up the necessary imports.
"""
import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # `"tensorflow"`/`"torch"`

import keras
from keras import ops, layers
import pandas as pd
import numpy as np
import tensorflow as tf
from zipfile import ZipFile
from urllib.request import urlretrieve

"""## Data Pipeline
These functions handle downloading the raw MovieLens 1M data, processing it into user sequences with pandas, and creating a `tf.data.Dataset` for efficient training.

"""


def load_and_process_data(dataset_name="ml-1m"):
    # Downloads and processes the raw MovieLens data using only pandas.
    print("--- Starting Data Loading and Processing ---")
    origin_url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = "movielens.zip"
    data_dir = "ml-1m"
    ratings_file = os.path.join(data_dir, "ratings.dat")

    if not os.path.exists(ratings_file):
        print("Downloading and extracting the MovieLens 1M dataset...")
        urlretrieve(origin_url, zip_path)
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()
        print("Download complete.")
    else:
        print("Raw data found.")

    print("Reading ratings data...")
    ratings_df = pd.read_csv(
        ratings_file,
        sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
        encoding="ISO-8859-1",
        engine="python",
    )

    print("Creating user sequences...")
    ratings_group = ratings_df.sort_values(by=["unix_timestamp"]).groupby("user_id")

    seq_ratings_data = pd.DataFrame(
        data={
            "user_id": list(ratings_group.groups.keys()),
            "sequence_item_ids": list(ratings_group.movie_id.apply(list)),
            "sequence_timestamps": list(ratings_group.unix_timestamp.apply(list)),
        }
    )
    print("--- Data Loading and Processing Complete ---")
    return seq_ratings_data


def create_tf_dataset(dataset_name, batch_size, max_seq_len, num_targets=1):
    """
    Creates a tf.data.Dataset for training the Keras model.
    """
    processed_df = load_and_process_data(dataset_name)

    item_id_sequences = processed_df.sequence_item_ids.values
    timestamp_sequences = processed_df.sequence_timestamps.values
    seq_lengths = np.array([len(s) for s in item_id_sequences], dtype=np.int32)

    print("Padding sequences using tf.keras.utils.pad_sequences...")
    padded_item_ids = tf.keras.utils.pad_sequences(
        item_id_sequences,
        maxlen=max_seq_len,
        padding="pre",
        truncating="pre",
        value=0,
        dtype="int32",
    )
    padded_timestamps = tf.keras.utils.pad_sequences(
        timestamp_sequences,
        maxlen=max_seq_len,
        padding="pre",
        truncating="pre",
        value=0,
        dtype="int32",
    )

    labels = padded_item_ids[:, -1]
    model_input_ids = padded_item_ids[:, :-1]
    model_input_timestamps = padded_timestamps[:, :-1]

    seq_lengths = np.clip(seq_lengths - 1, 0, max_seq_len - 1)

    model_inputs = {
        "seq_embeddings": model_input_ids,
        "seq_lengths": seq_lengths,
        "seq_timestamps": model_input_timestamps,
        "num_targets": np.full((len(processed_df),), num_targets, dtype=np.int32),
    }

    dataset = tf.data.Dataset.from_tensor_slices((model_inputs, labels))

    return (
        dataset.shuffle(buffer_size=1024)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


"""## Model Components

### Positional Encoder Layer

The HSTU Positional Encoder is a hybrid embedding mechanism that fuses sequence position and temporal interval information into each item embedding, enabling the model to understand not just what a user interacted with, but also when and in what order.
"""


class HSTUPositionalEncoder(keras.layers.Layer):
    def __init__(
        self,
        num_position_buckets,
        num_time_buckets,
        embedding_dim,
        contextual_seq_len,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_position_buckets = num_position_buckets
        self.num_time_buckets = num_time_buckets
        self.embedding_dim = embedding_dim
        self.contextual_seq_len = contextual_seq_len

    def build(self, input_shape):
        self.position_embeddings_weight = self.add_weight(
            shape=(self.num_position_buckets, self.embedding_dim),
            initializer=keras.initializers.RandomUniform(
                minval=-((1.0 / self.num_position_buckets) ** 0.5),
                maxval=((1.0 / self.num_position_buckets) ** 0.5),
            ),
            name="position_embeddings_weight",
        )
        self.timestamp_embeddings_weight = self.add_weight(
            shape=(self.num_time_buckets + 1, self.embedding_dim),
            initializer=keras.initializers.RandomUniform(
                minval=-((1.0 / self.num_time_buckets) ** 0.5),
                maxval=((1.0 / self.num_time_buckets) ** 0.5),
            ),
            name="timestamp_embeddings_weight",
        )

    def _get_col_indices(self, seq_lengths, max_seq_len, num_targets=None):
        batch_size = ops.shape(seq_lengths)[0]
        col_indices = ops.tile(
            ops.expand_dims(ops.arange(max_seq_len, dtype="int32"), 0), (batch_size, 1)
        )
        high_inds = seq_lengths - (num_targets if num_targets is not None else 0)
        high_inds = ops.expand_dims(high_inds, 1)
        col_indices = ops.clip(col_indices, 0, high_inds)
        col_indices = high_inds - col_indices
        col_indices = col_indices + self.contextual_seq_len
        col_indices = ops.clip(col_indices, 0, self.num_position_buckets - 1)
        return col_indices

    def call(
        self,
        seq_embeddings,
        seq_lengths,
        timestamps,
        max_seq_len,
        num_targets=None,
        time_bucket_fn="sqrt",
    ):
        alpha = self.embedding_dim**0.5
        seq_embeddings = seq_embeddings * alpha
        pos_indices = self._get_col_indices(seq_lengths, max_seq_len, num_targets)
        position_embeddings = ops.take(
            self.position_embeddings_weight, pos_indices, axis=0
        )
        last_indices = ops.expand_dims(
            ops.clip(seq_lengths - 1, 0, max_seq_len - 1), axis=-1
        )
        query_time = ops.take_along_axis(timestamps, last_indices, axis=1)
        ts = query_time - timestamps
        ts = ops.maximum(ts, 1e-6) / 60.0
        if time_bucket_fn == "log":
            ts = ops.log(ts)
        else:
            ts = ops.sqrt(ts)
        ts = ops.cast(ops.clip(ts, 0, self.num_time_buckets), "int32")
        time_embeddings = ops.take(self.timestamp_embeddings_weight, ts, axis=0)
        return seq_embeddings + position_embeddings + time_embeddings

    def compute_output_shape(self, input_shape):
        return input_shape  # Return the input shape directly


"""### Core STU Layer

The STU layer acts like a fast, efficient Transformer block — it learns to transduce a sequence of user interactions into a higher-level understanding of user intent, without the heavy computation of traditional attention.
"""


class STULayer(keras.layers.Layer):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        attention_dim,
        output_dropout_ratio=0.3,
        causal=True,
        target_aware=True,
        use_group_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.output_dropout_ratio = output_dropout_ratio
        self.causal = causal
        self.target_aware = target_aware
        self.use_group_norm = use_group_norm
        self.input_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        if self.use_group_norm:
            self.output_norm = keras.layers.GroupNormalization(
                groups=self.num_heads, epsilon=1e-6
            )
        else:
            self.output_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(self.output_dropout_ratio)

    def build(self, input_shape):
        self.uvqk_weight = self.add_weight(
            shape=(
                self.embedding_dim,
                (self.hidden_dim * 2 + self.attention_dim * 2) * self.num_heads,
            ),
            initializer=keras.initializers.GlorotUniform(),
            name="uvqk_weight",
        )
        self.uvqk_bias = self.add_weight(
            shape=((self.hidden_dim * 2 + self.attention_dim * 2) * self.num_heads,),
            initializer=keras.initializers.Zeros(),
            name="uvqk_bias",
        )
        self.output_weight = self.add_weight(
            shape=(
                (self.hidden_dim * self.num_heads) * 2 + self.embedding_dim,
                self.embedding_dim,
            ),
            initializer=keras.initializers.GlorotUniform(),
            name="output_weight",
        )

    def _get_valid_attn_mask(self, seq_lengths, max_seq_len, num_targets=None):
        batch_size = ops.shape(seq_lengths)[0]
        ids = ops.arange(max_seq_len, dtype="int32")
        ids = ops.expand_dims(ids, 0)
        row_ids = ops.expand_dims(ids, -1)
        col_ids = ops.expand_dims(ids, 1)
        valid_attn_mask = row_ids >= col_ids
        seq_len_mask = ids < ops.expand_dims(seq_lengths, -1)
        valid_attn_mask = valid_attn_mask & ops.expand_dims(seq_len_mask, 1)
        return ops.cast(valid_attn_mask, self.compute_dtype)

    def call(self, inputs, training=False):
        x, seq_lengths = inputs["x"], inputs["seq_lengths"]
        num_targets = inputs.get("num_targets")
        batch_size, max_seq_len, _ = ops.shape(x)
        normed_x = self.input_norm(x)
        normed_x_flat = ops.reshape(normed_x, (-1, self.embedding_dim))
        uvqk = ops.matmul(normed_x_flat, self.uvqk_weight) + self.uvqk_bias

        total_hidden_dim = self.hidden_dim * self.num_heads
        total_attention_dim = self.attention_dim * self.num_heads

        u = ops.slice(uvqk, [0, 0], [-1, total_hidden_dim])
        v = ops.slice(uvqk, [0, total_hidden_dim], [-1, total_hidden_dim])
        q = ops.slice(uvqk, [0, 2 * total_hidden_dim], [-1, total_attention_dim])
        k = ops.slice(
            uvqk,
            [0, 2 * total_hidden_dim + total_attention_dim],
            [-1, total_attention_dim],
        )

        u = ops.silu(u)
        q = ops.reshape(
            q, (batch_size, max_seq_len, self.num_heads, self.attention_dim)
        )
        k = ops.reshape(
            k, (batch_size, max_seq_len, self.num_heads, self.attention_dim)
        )
        v = ops.reshape(v, (batch_size, max_seq_len, self.num_heads, self.hidden_dim))
        q, k, v = (
            ops.transpose(q, (0, 2, 1, 3)),
            ops.transpose(k, (0, 2, 1, 3)),
            ops.transpose(v, (0, 2, 1, 3)),
        )
        alpha = self.attention_dim**-0.5
        qk_attn = ops.einsum("bhnd,bhmd->bhnm", q, k) * alpha
        qk_attn = ops.silu(qk_attn)
        valid_attn_mask = self._get_valid_attn_mask(
            seq_lengths, max_seq_len, num_targets
        )
        qk_attn = qk_attn * ops.expand_dims(valid_attn_mask, 1)
        attn_output = ops.einsum("bhnm,bhmd->bhnd", qk_attn, v)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output_flat = ops.reshape(
            attn_output, (batch_size * max_seq_len, self.num_heads * self.hidden_dim)
        )
        u_flat = ops.reshape(u, (batch_size * max_seq_len, -1))
        x_flat = ops.reshape(x, (batch_size * max_seq_len, -1))
        if self.use_group_norm:
            norm_input = ops.reshape(
                attn_output_flat, (-1, self.num_heads, self.hidden_dim)
            )
            norm_output = self.output_norm(norm_input)
            norm_output = ops.reshape(
                norm_output, (-1, self.num_heads * self.hidden_dim)
            )
        else:
            norm_output = self.output_norm(attn_output_flat)
        y = u_flat * norm_output
        y = ops.concatenate([u_flat, x_flat, y], axis=1)
        y = self.dropout(y, training=training)
        output = ops.matmul(y, self.output_weight) + x_flat
        return ops.reshape(output, (batch_size, max_seq_len, self.embedding_dim))

    def compute_output_shape(self, input_shape):
        return input_shape["x"]


"""### Final Model Assembly"""


class HSTUTransducer(keras.Model):
    def __init__(
        self,
        stu_layer,
        positional_encoder,
        vocab_size,
        embedding_dim,
        input_dropout_ratio=0.0,
        max_seq_len=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.embedding_layer = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, name="item_embedding_layer"
        )
        self.stu_layer = stu_layer
        self.positional_encoder = positional_encoder
        self.input_dropout = keras.layers.Dropout(input_dropout_ratio)
        self.output_head = keras.layers.Dense(vocab_size, name="output_head")

    def build(self, input_shape):
        # Define inputs using keras.Input for model summary and tracing
        seq_embeddings_input = keras.Input(
            shape=(self.max_seq_len,), dtype=tf.int32, name="seq_embeddings"
        )
        seq_lengths_input = keras.Input(shape=(), dtype=tf.int32, name="seq_lengths")
        seq_timestamps_input = keras.Input(
            shape=(self.max_seq_len,), dtype=tf.int32, name="seq_timestamps"
        )
        num_targets_input = keras.Input(shape=(), dtype=tf.int32, name="num_targets")

        inputs = {
            "seq_embeddings": seq_embeddings_input,
            "seq_lengths": seq_lengths_input,
            "seq_timestamps": seq_timestamps_input,
            "num_targets": num_targets_input,
        }

        # Call the model with the defined inputs to build the layers
        _ = self.call(inputs)

    def call(self, inputs, training=False):
        item_ids = inputs["seq_embeddings"]
        seq_embeddings = self.embedding_layer(item_ids)
        batch_size, max_seq_len, _ = ops.shape(
            seq_embeddings
        )  # Use dynamic shape from input tensor
        preprocessed_embeddings = self.positional_encoder(
            seq_embeddings,
            inputs["seq_lengths"],
            inputs["seq_timestamps"],
            max_seq_len=max_seq_len,  # Pass as keyword argument
            num_targets=inputs.get("num_targets"),  # Pass as keyword argument
        )
        preprocessed_embeddings = self.input_dropout(
            preprocessed_embeddings, training=training
        )
        stu_inputs = {
            "x": preprocessed_embeddings,
            "seq_lengths": inputs["seq_lengths"],
            "num_targets": inputs.get("num_targets"),
        }
        sequence_output = self.stu_layer(stu_inputs, training=training)
        indices = ops.expand_dims(
            ops.expand_dims(inputs["seq_lengths"] - 1, axis=-1), axis=-1
        )
        last_item_embedding = ops.take_along_axis(sequence_output, indices, axis=1)
        last_item_embedding = ops.squeeze(last_item_embedding, axis=1)
        logits = self.output_head(last_item_embedding)
        return logits

    def compute_output_shape(self, input_shape):
        return (
            input_shape["seq_embeddings"][0],
            self.vocab_size,
        )  # Return shape based on batch size and vocab size


"""## Training, Evaluation, and Inference

### Configuration
"""

DATASET = "ml-1m"
MAX_SEQ_LEN = 200
EMBEDDING_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 128
ATTENTION_DIM = 128
OUTPUT_DROPOUT_RATIO = 0.3
INPUT_DROPOUT_RATIO = 0.3
NUM_POSITION_BUCKETS = 256
NUM_TIME_BUCKETS = 256
CONTEXTUAL_SEQ_LEN = 0
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

"""### Main Training Loop"""

print("--- Starting HSTU Keras 3 Training (TensorFlow Data Pipeline) ---")

print("\n--- Loading and Preparing Dataset ---")
# Load raw data first to determine VOCAB_SIZE
processed_df = load_and_process_data(DATASET)
VOCAB_SIZE = processed_df["sequence_item_ids"].explode().max() + 1
print(f"Dynamically determined VOCAB_SIZE: {VOCAB_SIZE}")

train_dataset = create_tf_dataset(DATASET, BATCH_SIZE, MAX_SEQ_LEN)
print("Dataset prepared.")

print("\n--- Building Keras HSTU Model ---")
positional_encoder = HSTUPositionalEncoder(
    NUM_POSITION_BUCKETS, NUM_TIME_BUCKETS, EMBEDDING_DIM, CONTEXTUAL_SEQ_LEN
)
stu_layer = STULayer(
    EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM, ATTENTION_DIM, OUTPUT_DROPOUT_RATIO
)
# Pass max_seq_len during model initialization
model = HSTUTransducer(
    stu_layer,
    positional_encoder,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    INPUT_DROPOUT_RATIO,
    max_seq_len=MAX_SEQ_LEN,
)
print("Model built.")

print("\n--- Compiling Model ---")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
print("Model compiled.")

# Build the model explicitly with defined inputs
model.build(
    {
        "seq_embeddings": tf.TensorShape((None, MAX_SEQ_LEN)),
        "seq_lengths": tf.TensorShape((None,)),
        "seq_timestamps": tf.TensorShape((None, MAX_SEQ_LEN)),
        "num_targets": tf.TensorShape((None,)),
    }
)
model.summary()

print("\n--- Starting Training ---")
model.fit(train_dataset, epochs=EPOCHS)
print("\n--- Training Complete ---")

"""### Evaluation"""

print("\n--- Creating Validation Dataset ---")

# For simplicity, we'll use the same dataset loader for validation, but in a real scenario,
# you would have a separate validation set.
# Set num_targets to 1 for validation as we are predicting one next item.
val_dataset = create_tf_dataset(DATASET, BATCH_SIZE, MAX_SEQ_LEN, num_targets=1)
print("Validation Dataset prepared.")

print("\n--- Evaluating Model on Validation Set ---")
# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

"""### Inference"""


def load_movie_titles(dataset_name="ml-1m"):
    """
    Loads movie titles from the MovieLens dataset.
    """
    data_dir = "ml-1m"
    movies_file = os.path.join(data_dir, "movies.dat")

    if not os.path.exists(movies_file):
        print("Movies file not found. Ensure the dataset is extracted.")
        return None

    print("Reading movie titles data...")
    movies_df = pd.read_csv(
        movies_file,
        sep="::",
        names=["movie_id", "title", "genres"],
        encoding="ISO-8859-1",
        engine="python",
    )
    # Create a mapping from movie_id to title
    movie_id_to_title = dict(zip(movies_df["movie_id"], movies_df["title"]))
    print("Movie titles loaded.")
    return movie_id_to_title


def prepare_single_input(item_ids, timestamps, max_seq_len):
    """
    Prepares a single user's sequence for model prediction.
    """
    # Pad item IDs and timestamps
    padded_item_ids = tf.keras.utils.pad_sequences(
        [item_ids],
        maxlen=max_seq_len,
        padding="pre",
        truncating="pre",
        value=0,
        dtype="int32",
    )
    padded_timestamps = tf.keras.utils.pad_sequences(
        [timestamps],
        maxlen=max_seq_len,
        padding="pre",
        truncating="pre",
        value=0,
        dtype="int32",
    )

    # The model expects inputs for the sequence up to the last item for prediction
    model_input_ids = padded_item_ids[:, :-1]
    model_input_timestamps = padded_timestamps[:, :-1]

    seq_length = np.array([len(item_ids) - 1], dtype=np.int32)
    seq_length = np.clip(seq_length, 0, max_seq_len - 1)

    # Prepare the input dictionary for the model
    model_inputs = {
        "seq_embeddings": tf.constant(model_input_ids, dtype=tf.int32),
        "seq_lengths": tf.constant(seq_length, dtype=tf.int32),
        "seq_timestamps": tf.constant(model_input_timestamps, dtype=tf.int32),
        "num_targets": tf.constant([1], dtype=tf.int32),
    }
    return model_inputs


# Load movie titles
movie_id_to_title = load_movie_titles(DATASET)

if movie_id_to_title:
    print("\n--- Demonstrating an Example Prediction ---")

    # Use the same processed_df from the main training block
    example_user_data = processed_df.iloc[0]
    example_item_ids = example_user_data["sequence_item_ids"]
    example_timestamps = example_user_data["sequence_timestamps"]

    if len(example_item_ids) >= 2:
        input_item_ids = example_item_ids[:-1]  # All but the last item
        input_timestamps = example_timestamps[:-1]
        target_item_id = example_item_ids[-1]  # The actual next item

        print(
            f"User's history (last 5 items): {[movie_id_to_title.get(mid, f'ID:{mid}') for mid in input_item_ids[-5:]]}"
        )
        print(
            f"Actual next item: {movie_id_to_title.get(target_item_id, f'ID:{target_item_id}')}"
        )

        model_input = prepare_single_input(
            input_item_ids, input_timestamps, MAX_SEQ_LEN
        )

        logits = model.predict(model_input)
        prediction_logits = logits[0]

        top_k = 5
        top_k_indices = ops.top_k(prediction_logits, k=top_k).indices.numpy()

        print("\nTop 5 Recommended Movies:")
        for i, movie_id in enumerate(top_k_indices):
            title = movie_id_to_title.get(movie_id, f"Unknown Movie (ID: {movie_id})")
            print(f"{i+1}. {title}")

        if target_item_id in top_k_indices:
            print(f"\n(Actual next item was in the top {top_k} recommendations!)")
        else:
            print(f"\n(Actual next item was NOT in the top {top_k} recommendations.)")

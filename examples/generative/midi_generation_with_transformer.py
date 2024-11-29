"""
Title: Music Generation with Transformer Models
Author: [Joaquin Jimenez](https://github.com/johacks/)
Date created: 2024/11/22
Last modified: 2024/11/26
Description: Use a Transformer model to train on MIDI data and generate music sequences.
Accelerator: GPU
"""

"""
## Introduction

In this tutorial, we learn how to build a music generation model using a
Transformer decode-only architecture.
The model is trained on the [Maestro dataset](https://magenta.tensorflow.org/datasets/maestro)
and implemented using keras 3.
In the process, we explore MIDI tokenization, and relative global attention mechanisms.

This example is based on the paper "Music Transformer" by Huang et al. (2018).
Check out the original [paper](https://arxiv.org/abs/1809.04281) and
[code](https://github.com/jason9693/MusicTransformer-tensorflow2.0).
"""

"""
## Setup

Before we start, let's import and install all the libraries we need.
"""

"""shell
pip install -qq midi_neural_processor
pip install -qq keras_hub
pip install -qq "keras>=3.6.0"  # Allows use of keras.utils.Config.
"""

"""
### Optional dependencies

To hear the audio, install the following additional dependencies:
"""

"""shell
sudo apt-get -qq install -y fluidsynth 2> /dev/null
pip install -qq pyfluidsynth scipy
"""

import os
import random
import tempfile

import keras
import midi_neural_processor.processor as midi_tokenizer
import numpy as np
from keras import callbacks, layers, ops, optimizers, utils
from keras_hub import layers as hub_layers
from os import path

"""
## Configuration

Lets define the configuration for the model and the dataset to be used in this example.
"""
event_range = midi_tokenizer.RANGE_NOTE_ON
event_range += midi_tokenizer.RANGE_NOTE_OFF
event_range += midi_tokenizer.RANGE_TIME_SHIFT
event_range += midi_tokenizer.RANGE_VEL
CONFIG = utils.Config(
    max_sequence_len=2048,
    embedding_dim=256,
    num_transformer_blocks=6,
    batch_size=6,
    token_pad=event_range,
    token_start_of_sentence=event_range + 1,
    token_end_of_sentence=event_range + 2,
    vocabulary_size=event_range + 3,
    model_out="tmp/music_transformer.keras",
    seed=42,
)
utils.set_random_seed(CONFIG.seed)


"""
## Maestro dataset

The Maestro dataset contains MIDI files for piano performances.

### Download the dataset

We now download and extract the dataset, then move the MIDI files to a new directory.
"""


def download_maestro(output_dir=None):
    """Download the Maestro MIDI dataset.
    Extracted from: https://magenta.tensorflow.org/datasets/maestro
    """
    # Ensure the output directory exists
    output_dir = tempfile.mkdtemp() if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Download and extract zip file
    dir = utils.get_file(
        origin="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        extract=True,
    )

    # Gather all MIDI files
    midi_files, file_paths = set(), list()
    for root, _, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(".midi") or file.lower().endswith(".mid"):
                midi_files.add(path.join(root, file))

    # Move the files to the output directory
    for file in sorted(midi_files):
        file_paths.append(new_path := path.join(output_dir, path.basename(file)))
        os.rename(file, new_path)
    return file_paths


paths = list(sorted(download_maestro(output_dir="datasets/maestro")))
output_dir = path.dirname(paths[0])


"""
### Split the dataset

We can now split the dataset into training and validation sets.
"""

indices = np.random.permutation(len(paths))
split = int(len(paths) * 0.1)
train_paths = [paths[i] for i in indices[split:]]
val_paths = [paths[i] for i in indices[:split]]

"""
### Hear a MIDI file

We use the pretty_midi library and fluidsynth to convert MIDI files into waveform audio.
This allows us to listen to the data samples before and after processing.

The following dependencies are required to play the audio:
- fluidsynth: `sudo apt install -y fluidsynth`
- pyfluidsynth, scipy: `pip install pyfluidsynth scipy`
"""


def visualize_midi(midi_path, sampling_rate=16000, seconds=15, out_dir=None):
    import pretty_midi
    from scipy.io.wavfile import write as write_wav
    from IPython.display import Audio

    # Create the audio waveform
    pretty_midi_file = pretty_midi.PrettyMIDI(midi_path)
    waveform = pretty_midi_file.fluidsynth(fs=sampling_rate)[: seconds * sampling_rate]

    # Display the audio if no path is provided
    if out_dir is None:
        # IPython display
        return Audio(waveform, rate=sampling_rate)

    # Save the audio to a file
    os.makedirs(out_dir, exist_ok=True)
    audio_path = path.join(out_dir, path.basename(midi_path).split(".")[0] + ".wav")
    write_wav(audio_path, sampling_rate, (waveform * 32767).astype(np.int16))
    return audio_path


print(visualize_midi(train_paths[0], out_dir="tmp/"))  # Saved audio path
visualize_midi(train_paths[0])  # Display the audio if in a Jupyter notebook


"""
### Tokenize the data

We now preprocess the MIDI files into a tokenized format for training.
"""


def encode_midi_task(midi_path):
    """Define a task that tokenizes a MIDI file."""
    import midi_neural_processor.processor as midi_tokenizer

    return midi_tokenizer.encode_midi(midi_path)


def preprocess_midi_files(file_paths, save_dir=None):
    """Preprocess a list of MIDI files and save the notes to a file."""
    from multiprocessing import Pool, cpu_count

    # Assume all files are in the same directory and save to the same directory
    save_dir = path.dirname(file_paths[0]) if save_dir is None else save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Check if the notes have already been preprocessed
    output_file = path.join(save_dir, "notes.npz")
    if path.exists(output_file):
        npz_file = np.load(output_file)
        return [npz_file[key] for key in npz_file.keys()]

    # Preprocess the MIDI files in parallel
    progbar = utils.Progbar(len(file_paths), unit_name="MIDI_file", interval=5)
    pool = Pool(cpu_count() - 1)
    all_notes = []
    for notes in pool.imap_unordered(encode_midi_task, file_paths):
        progbar.add(1)
        all_notes.append(np.array(notes))

    # Save the notes to a file
    np.savez(output_file, *all_notes)
    return all_notes


train_midis = preprocess_midi_files(train_paths, path.join(output_dir, "train"))
val_midis = preprocess_midi_files(val_paths, path.join(output_dir, "val"))


"""
### Dataset objects

We now define a dataset class that yields batches of input sequences and target sequences.
"""


class MidiDataset(utils.PyDataset):
    """A dataset for MIDI files that yields batches of input sequences and target sequences."""

    def __init__(
        self,
        encoded_midis,
        batch_size=CONFIG.batch_size,
        max_sequence_len=CONFIG.max_sequence_len,
    ):
        super(MidiDataset, self).__init__()
        self.batch_size = batch_size
        self.max_sequence_len = max_sequence_len
        self.encoded_midis = encoded_midis
        batches, last_batch_size = divmod(len(encoded_midis), batch_size)
        self._num_batches = batches + int(last_batch_size > 0)

    def __len__(self):
        """Get the number of batches."""
        return self._num_batches

    def __getitem__(self, idx):
        """Generate random inputs and corresponding targets for the model."""
        # Same as in the original paper, we always get a random batch.
        # See: https://github.com/jason9693/MusicTransformer-tensorflow2.0/blob/f7c06c0cb2e9cdddcbf6db779cb39cd650282778/data.py
        batch = random.sample(self.encoded_midis, k=self.batch_size)

        # Convert the batch to sequences
        batch_data = [
            self._get_sequence(midi, self.max_sequence_len + 1) for midi in batch
        ]
        batch_data = np.array(batch_data)

        # Split the data into input and target sequences
        return batch_data[:, :-1], batch_data[:, 1:]

    def _get_sequence(self, data, max_length):
        """Get a random sequence of notes from a file."""
        # Truncate or pad the sequence
        if len(data) > max_length:
            start = random.randrange(0, len(data) - max_length)
            data = data[start : start + max_length]
        elif len(data) < max_length:
            data = np.append(data, CONFIG.token_end_of_sentence)

        # Pad the sequence if necessary
        if len(data) < max_length:
            data = np.concatenate(
                (data, np.full(max_length - len(data), CONFIG.token_pad))
            )
        return np.asanyarray(data, dtype="int32")


train_dataset, val_dataset = MidiDataset(train_midis), MidiDataset(val_midis)


"""
## Model definition

It is time to define the model architecture. We use a Transformer decoder
architecture with a custom attention mechanism, relative global attention.

### Relative Global Attention

The following code implements the Relative Global Attention layer. It is used
in place of the standard multi-head attention layer in the Transformer decoder.
The main difference is that it includes a relative positional encoding that
allows the model to learn relative positional information between tokens.
"""


@keras.utils.register_keras_serializable()
class RelativeGlobalAttention(layers.Layer):
    """
    From Music Transformer (Huang et al., 2018)
    https://arxiv.org/abs/1809.04281
    """

    def __init__(self, num_heads, embedding_dim, max_sequence_len, **kwargs):
        super().__init__(**kwargs)
        self.key_length = None
        self.max_sequence_len = max_sequence_len
        self.relative_embedding = None
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.query_dense = layers.Dense(int(self.embedding_dim))
        self.key_dense = layers.Dense(int(self.embedding_dim))
        self.value_dense = layers.Dense(int(self.embedding_dim))
        self.output_dense = layers.Dense(embedding_dim, name="output")

    def build(self, input_shape):
        self.query_length = input_shape[0][1]
        self.key_length = input_shape[1][1]
        self.relative_embedding = self.add_weight(
            (self.max_sequence_len, int(self.head_dim)), name="relative_embedding"
        )

    def _apply_dense_layer_and_split_heads(self, inputs, dense_layer):
        # Apply linear transformation
        inputs = dense_layer(inputs)
        new_shape = ops.shape(inputs)
        # Reshape to split by attention heads
        reshaped = ops.reshape(inputs, (new_shape[0], new_shape[1], self.num_heads, -1))
        # Transpose for head-first format
        return ops.transpose(reshaped, (0, 2, 1, 3))

    def call(self, inputs, mask=None):
        # Compute Q, K, V: Batch, head, sequence, features
        query = self._apply_dense_layer_and_split_heads(inputs[0], self.query_dense)
        key = self._apply_dense_layer_and_split_heads(inputs[1], self.key_dense)
        value = self._apply_dense_layer_and_split_heads(inputs[2], self.value_dense)

        # Compute scaled dot-product attention scores
        attention_scores = ops.matmul(query, ops.transpose(key, [0, 1, 3, 2]))

        # Compute relative positional encoding and combine with attention scores
        start_idx = max(0, self.max_sequence_len - ops.shape(query)[2])
        relative_embedding = self.relative_embedding[start_idx:, :]
        attention_scores += self._compute_attention_scores(query, relative_embedding)
        logits = attention_scores / ops.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            logits += ops.cast(mask, "float32") * -1e9

        # Compute attention weights
        attention_weights = ops.nn.softmax(logits, axis=-1)
        attention_output = ops.matmul(attention_weights, value)

        # Merge heads and apply final linear transformation
        merged_attention = ops.transpose(attention_output, (0, 2, 1, 3))
        merged_attention = ops.reshape(
            merged_attention, (ops.shape(merged_attention)[0], -1, self.embedding_dim)
        )
        output = self.output_dense(merged_attention)

        return output, attention_weights

    def _compute_attention_scores(self, query, relative_embedding):
        """
        Compute relative attention scores using positional encodings.
        """
        relative_scores = ops.einsum("bhld, md->bhlm", query, relative_embedding)
        relative_scores = self._apply_mask_to_relative_scores(relative_scores)
        return self._skew_attention_scores(relative_scores)

    def _apply_mask_to_relative_scores(self, scores):
        """
        Apply masking to relative positional scores to ignore future positions.
        """
        mask = ops.flip(
            ops.tri(scores.shape[-2], scores.shape[-1], dtype="float32"), axis=1
        )
        return mask * scores

    def _skew_attention_scores(self, scores):
        """
        Perform skewing operation to align relative attention scores with the sequence.
        """
        padded_scores = ops.pad(scores, ((0, 0), (0, 0), (0, 0), (1, 0)))
        padded_shape = ops.shape(padded_scores)
        reshaped_scores = ops.reshape(
            padded_scores, (-1, padded_shape[1], padded_shape[-1], padded_shape[-2])
        )
        skewed_scores = reshaped_scores[:, :, 1:, :]

        if self.key_length > self.query_length:
            size_diff = self.key_length - self.query_length
            return ops.pad(skewed_scores, [[0, 0], [0, 0], [0, 0], [0, size_diff]])
        else:
            return skewed_scores[:, :, :, : self.key_length]


"""
### Decoder Layer

Using the RelativeGlobalAttention layer, we can define the DecoderLayer. It is mostly like
the standard Transformer decoder layer but with the custom attention mechanism.
"""


@keras.utils.register_keras_serializable()
class DecoderLayer(layers.Layer):
    def __init__(self, embedding_dim, num_heads, max_sequence_len, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # Initialize attributes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_sequence_len = max_sequence_len

        # Initialize layers
        self.relative_global_attention_1 = RelativeGlobalAttention(
            num_heads, embedding_dim, max_sequence_len
        )

        self.feed_forward_network_pre = layers.Dense(self.embedding_dim // 2, "relu")
        self.feed_forward_network_pos = layers.Dense(self.embedding_dim)

        self.layer_normalization_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_normalization_2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = layers.Dropout(dropout)
        self.dropout_2 = layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=False):
        # Attention block. Inputs are (query, key, value)
        attention_out, attention_weights = self.relative_global_attention_1(
            (inputs, inputs, inputs), mask=mask
        )
        attention_out = self.dropout_1(attention_out, training=training)
        attention_out_normalized = self.layer_normalization_1(attention_out + inputs)

        ffn_out = self.feed_forward_network_pre(attention_out)
        ffn_out = self.feed_forward_network_pos(ffn_out)
        ffn_out = self.dropout_2(ffn_out, training=training)
        out = self.layer_normalization_2(attention_out_normalized + ffn_out)

        return out, attention_weights


"""
### Decoder

The Decoder layer is composed of multiple DecoderLayer blocks. It also includes
an embedding layer that converts our tokenized input into an embedding representation.
"""


@keras.utils.register_keras_serializable()
class Decoder(layers.Layer):
    def __init__(
        self, embedding_dim, vocabulary_size, max_sequence_len, num_blocks, dropout
    ):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        self.embedding = layers.Embedding(vocabulary_size, self.embedding_dim)
        self.positional_encoding = hub_layers.SinePositionEncoding()

        self.decode_layers = [
            DecoderLayer(
                embedding_dim, embedding_dim // 64, max_sequence_len, dropout=dropout
            )
            for _ in range(num_blocks)
        ]
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=False, return_attention_weights=False):
        weights = []

        # Adding embedding and position encoding.
        x = self.embedding(inputs)
        x = x * ops.sqrt(ops.cast(self.embedding_dim, "float32"))
        x = x + self.positional_encoding(x)
        x = self.dropout(x, training=training)

        # Passing through the transformer blocks.
        for i in range(self.num_blocks):
            x, w = self.decode_layers[i](x, mask=mask, training=training)
            weights.append(w)
        if return_attention_weights:
            return x, weights
        return x


"""
### Music Transformer Decoder

With the above layers defined, we can now define the MusicTransformerDecoder model. It applies
a linear transformation to the output of the decoder to get the logits for each token.
"""


@keras.utils.register_keras_serializable()
class MusicTransformerDecoder(keras.Model):
    def __init__(
        self,
        embedding_dim=CONFIG.embedding_dim,
        vocabulary_size=CONFIG.vocabulary_size,
        num_blocks=CONFIG.num_transformer_blocks,
        max_sequence_len=CONFIG.max_sequence_len,
        dropout=0.2,
    ):
        # Initialize attributes
        super(MusicTransformerDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size
        self.num_blocks = num_blocks
        self.max_sequence_len = max_sequence_len

        # Initialize layers
        # Transformer decoder
        self.decoder = Decoder(
            embedding_dim, vocabulary_size, max_sequence_len, num_blocks, dropout
        )
        # Output layer
        self.fc = layers.Dense(self.vocabulary_size, activation=None, name="output")

    @staticmethod
    def get_look_ahead_mask(max_sequence_len, inputs):
        sequence_length = min(max_sequence_len, inputs.shape[1])
        sequence_mask = ops.logical_not(
            ops.tri(sequence_length, sequence_length, dtype="bool")
        )

        inputs = ops.cast(inputs[:, None, None, :], "int32")
        output_pad_tensor = ops.ones_like(inputs) * CONFIG.token_pad
        decoder_output_mask = ops.equal(inputs, output_pad_tensor)
        return ops.cast(ops.logical_or(decoder_output_mask, sequence_mask), "int32")

    def call(self, inputs, training=False):
        mask = self.get_look_ahead_mask(self.max_sequence_len, inputs)
        decoding = self.decoder(
            inputs, mask=mask, training=training, return_attention_weights=False
        )
        return self.fc(decoding)

    # --- Sequence generation methods

    def generate(self, inputs: list, length=CONFIG.max_sequence_len, top_k=5):
        inputs = ops.convert_to_tensor([inputs])

        # Generate a new token using output distribution at given index
        def generate_token(inputs, end_idx):
            distribution = ops.stop_gradient(self.call(inputs)[0, end_idx])

            # Select the top-k tokens and their probabilities
            top_k_distribution, top_k_indices = ops.top_k(distribution, k=top_k)

            # Sample from the top-k probabilities
            new_token_idx = keras.random.categorical(top_k_distribution[None, :], 1)
            return ops.take(top_k_indices, new_token_idx[0])

        # Compute the number of tokens to add
        added_tokens = min(length, self.max_sequence_len - inputs.shape[1])
        progbar = utils.Progbar(added_tokens, unit_name="token", interval=5)

        # Pad the input sequence that will be filled with generated tokens
        out = ops.pad(inputs, ((0, 0), (0, added_tokens)), "constant", CONFIG.token_pad)

        # Generate tokens using top-k sampling
        for token_idx in range(inputs.shape[1] - 1, inputs.shape[1] - 1 + added_tokens):
            token = ops.cast(generate_token(out, end_idx=token_idx), out.dtype)
            out = ops.scatter_update(out, ((0, token_idx + 1),), token)
            progbar.add(1)

        return ops.convert_to_numpy(out[0])

    # --- Serialization methods

    def get_config(self):
        atts = ["embedding_dim", "vocabulary_size", "num_blocks", "max_sequence_len"]
        return {a: getattr(self, a) for a in atts}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


"""
### Loss function

We define a custom loss function that computes the categorical cross-entropy
loss for the model. It is computed only for non-padding tokens and uses
`from_logits=True` since the model outputs logits.
"""


@keras.utils.register_keras_serializable()
def train_loss(y_true, y_pred):
    mask = ops.cast(ops.logical_not(ops.equal(y_true, CONFIG.token_pad)), "float32")
    y_true = ops.one_hot(ops.cast(y_true, "int32"), CONFIG.vocabulary_size)
    return ops.categorical_crossentropy(y_true, y_pred, from_logits=True) * mask


"""
### Learning rate schedule

Following the Music Transformer paper, we define an adapted exponential decay
learning rate schedule that takes into account the embedding dimension.
"""


@keras.utils.register_keras_serializable()
class CustomSchedule(optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.embedding_dim = embedding_dim
        self.warmup_steps = warmup_steps

        self._embedding_dim = ops.cast(self.embedding_dim, "float32")
        # Numerical stability adjustment on torch, which is less precise
        self._lr_adjust = 0.1 if keras.backend.backend() == "torch" else 1.0

    def get_config(self):
        return {"embedding_dim": self.embedding_dim, "warmup_steps": self.warmup_steps}

    def __call__(self, step):
        step_rsqrt = ops.rsqrt(ops.cast(step, "float32"))
        warmup_adjust = step * (self.warmup_steps**-1.5)
        output = ops.rsqrt(self._embedding_dim) * ops.minimum(step_rsqrt, warmup_adjust)
        return self._lr_adjust * output


"""
## Training the model

We can now train the model on the Maestro dataset. First, we define a training
function. This function compiles the model, trains it, and saves the best model
checkpoint. This way, we can continue training from the best model checkpoint
if needed.
"""


def train_model(model, train_ds, val_ds, epochs=15):
    # Configure optimizer
    learning_rate = CustomSchedule(CONFIG.embedding_dim)
    optimizer = optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Compile the model
    model.compile(optimizer=optimizer, loss=train_loss)

    # Train the model
    save_cb = callbacks.ModelCheckpoint(CONFIG.model_out, save_best_only=True)
    model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=[save_cb], verbose=2
    )
    return model


"""
We can now train the model on the Maestro dataset. If a model checkpoint exists,
we can load it and continue training.
"""
if path.exists(CONFIG.model_out):
    model = keras.models.load_model(CONFIG.model_out)
    # Comment out to continue model training from the checkpoint
    # train_model(model, train_dataset, val_dataset, epochs=10)
else:
    # Train the model
    model = train_model(MusicTransformerDecoder(), train_dataset, val_dataset)


"""
## Generate music

We can now generate music using the trained model. We use an existing MIDI file
as a seed and generate a new sequence.
"""


def generate_music(model, seed_path, length=1024, out_dir=None, top_k=None):
    # Ensure the output directory exists
    out_dir = out_dir if out_dir is not None else tempfile.mkdtemp()
    os.makedirs(out_dir, exist_ok=True)

    # Get some tokens from the MIDI file
    inputs = midi_tokenizer.encode_midi(seed_path)[100:125]
    print(f"Seed tokens: {inputs}")

    # Generate music that follows the input tokens until the maximum length
    result = model.generate(inputs, length=length, top_k=top_k)

    output_path = path.join(out_dir, path.basename(seed_path).split(".")[0] + ".mid")
    midi_tokenizer.decode_midi(result, output_path)
    return output_path


output_file = generate_music(model, val_paths[-1], out_dir="tmp/", top_k=15)
print(visualize_midi(output_file, out_dir="tmp/"))  # Saved audio path
visualize_midi(output_file)  # Display the audio if in a Jupyter notebook

"""
## Conclusion

In this example, we learned how to build a music generation model using a custom
Transformer decoder architecture.

We did it following the Music Transformer paper by Huang et al. (2018).
To do so we had to:

- Define a custom loss function and learning rate schedule.
- Define a custom attention mechanism.
- Preprocess MIDI files into a tokenized format.

After training the model on the Maestro dataset, we generated music sequences
using a seed MIDI file.

### Next steps

We could further improve inference times by caching attention weights during the
forward pass, in a similar way as `keras_hub` `CausalLM` models, which use the
`CachedMultiHeadAttention` layer.
"""

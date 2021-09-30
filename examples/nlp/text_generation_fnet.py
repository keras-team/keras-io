"""
Title: Text Generation using FNet
Author: [Darshan Deshpande](https://twitter.com/getdarshan)
Date created: 2021/09/30
Last modified: 2021/09/30
Description: FNet transformer for text generation in Keras.
"""
"""
## Introduction

The original transformer implementation (Vaswani et al., 2017) was one of the major
breakthroughs in Natural Language Processing, giving rise to eminent architectures as the
ones derived from BERT and GPT. However, the drawback of such complex architectures is
that the self attention mechanism used by them can cause slow processing speeds. The FNet
architecture proposes a Fourier transformation based linear mixer for input tokens that
replaces this self attention.

The FNet model was able to achieve 92-97% of BERT's accuracy while training 80% faster on
GPUs and almost 70% faster on TPUs. This type of design provides an efficient and small
model size, leading to faster inference times.

In this example, we will implement and train this architecture on the Cornell Movie
Dialog corpus to show the applicability of this model to text generation.
"""

"""
## Imports
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence
import os

# Defining hyperparameters

MAX_SAMPLES = 50000
BUFFER_SIZE = 20000
MAX_LENGTH = 40

EMBED_DIM = 256
LATENT_DIM = 2048
NUM_HEADS = 8
BATCH_SIZE = 64

"""
## Loading data

We will be using the Cornell Dialog Corpus. We will parse the movie conversations into
questions and answers sets.
"""

path_to_zip = keras.utils.get_file(
    "cornell_movie_dialogs.zip",
    origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
    extract=True,
)

path_to_dataset = os.path.join(
    os.path.dirname(path_to_zip), "cornell movie-dialogs corpus"
)

path_to_movie_lines = os.path.join(path_to_dataset, "movie_lines.txt")
path_to_movie_conversations = os.path.join(path_to_dataset, "movie_conversations.txt")


def load_conversations():
    # Helper function for loading the conversation splits
    id2line = {}
    with open(path_to_movie_lines, errors="ignore") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(path_to_movie_conversations, "r") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
        for i in range(len(conversation) - 1):
            inputs.append(id2line[conversation[i]])
            outputs.append(id2line[conversation[i + 1]])
            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs


questions, answers = load_conversations()

"""
### Creating and fitting the tokenizer
"""

tokenizer = text.Tokenizer(
    2 ** 13, oov_token="OOV", filters='!"#$%&*+,-/<=>?@[\\]^_`{|}~\t\n'
)
tokenizer.fit_on_texts(questions + answers)

# Defining separate start and end tokens
START_TOKEN, END_TOKEN = [tokenizer.num_words], [tokenizer.num_words + 1]

# The final vocabulary size is the original size plus the start and end tokens
VOCAB_SIZE = tokenizer.num_words + 2

"""
### Tokenizing, filtering and padding sentences
"""


def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    # Converting the loaded text to integer sequences
    sentence1 = tokenizer.texts_to_sequences(inputs)
    sentence2 = tokenizer.texts_to_sequences(outputs)

    for (s1, s2) in zip(sentence1, sentence2):
        # Taking only those sentences that are less than the maximum allowed length
        if len(s1) <= MAX_LENGTH and len(s2) <= MAX_LENGTH:
            # Adding the start and end tokens
            tokenized_inputs.append(START_TOKEN + s1 + END_TOKEN)
            tokenized_outputs.append(START_TOKEN + s2 + END_TOKEN)

    # Padding sequences
    tokenized_inputs = sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding="post"
    )
    tokenized_outputs = sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH + 1, padding="post"
    )

    return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_and_filter(questions, answers)


# Creating a tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"encoder_inputs": questions, "decoder_inputs": answers[:, :-1]},
        {"outputs": answers[:, 1:]},
    )
)

dataset = dataset.cache().shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

"""
## Creating the FNet Encoder

The FNet paper proposes a replacement for the standard attention mechanism used by the
Transformer architecture (Vaswani et al., 2017).

![Architecture](https://i.imgur.com/rLg47qU.png)

The outputs of the FFT layer are complex in nature. To avoid processing complex layers,
only the real part(magnitude) is extracted.

The dense layers that follow the Fourier transformation act as convolutions applied on
the frequency domain.
"""


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs):
        # Casting the inputs to complex64
        inp_complex = tf.cast(inputs, tf.complex64)
        # Projecting the inputs to the frequency domain using FFT2D and
        # extracting the real part of the output
        fft = tf.math.real(tf.signal.fft2d(inp_complex))
        proj_input = self.layernorm_1(inputs + fft)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


"""
## Creating the Decoder
The decoder architecture remains the same as the one proposed by (Vaswani et al., 2017)
in the original transformer architecture, consisting of an embedding, positional
encoding, two masked multiheaded attention layers and finally the dense output layers


"""


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


"""
## Defining the full model
"""


def create_model():
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(encoder_inputs)
    encoder_outputs = TransformerEncoder(EMBED_DIM, LATENT_DIM)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(
        shape=(None, EMBED_DIM), name="decoder_state_inputs"
    )
    x = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(decoder_inputs)
    x = TransformerDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x, encoded_seq_inputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
    decoder = keras.Model(
        [decoder_inputs, encoded_seq_inputs], decoder_outputs, name="outputs"
    )

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )
    return transformer


"""
## Creating and Training the model
"""

transformer = create_model()
transformer.compile(
    "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

"""
The default `epochs` parameter is set to a single epoch but the model will take around
20-30 epochs of training to start outputting comprehensible sentences. Although accuracy
is not a good measure for this task, we will use it just to get a hint of the improvement
of the network.
"""

transformer.fit(dataset, epochs=1)

"""
## Performing inference
"""


def decode_sentence(input_sentence):
    # Mapping the input sentence to tokens and adding start and end tokens
    tokenized_input_sentence = tf.expand_dims(
        START_TOKEN + tokenizer.texts_to_sequences([input_sentence])[0] + END_TOKEN,
        axis=0,
    )
    # Padding the sequence to MAX_LENGTH
    tokenized_input_sentence = sequence.pad_sequences(
        tokenized_input_sentence, MAX_LENGTH, padding="post"
    )

    # Initializing the initial sentence consisting of only the start token.
    tokenized_target_sentence = tf.expand_dims(START_TOKEN, 0)
    decoded_sentence = ""

    for i in range(MAX_LENGTH):
        # Get the predictions
        predictions = transformer(
            [
                tokenized_input_sentence,
                sequence.pad_sequences(
                    tokenized_target_sentence, MAX_LENGTH, padding="post"
                ),
            ]
        )
        # Calculating the token with maximum probability and getting the corresponding word
        sampled_token_index = tf.argmax(predictions[0, i, :])
        sampled_token = tokenizer.index_word.get(sampled_token_index.numpy())

        # If sampled token is the end token then stop generating and return the sentence
        if tf.equal(sampled_token_index, END_TOKEN[0]):
            break

        decoded_sentence += " " + sampled_token
        tokenized_target_sentence = tf.concat(
            [tokenized_target_sentence, [[sampled_token_index]]], 1
        )

    return decoded_sentence


decode_sentence("Where have you been all this time?")

"""
## Conclusion

This example successfully shows how to train and perform inference using the FNet model.
For getting insight into the architecture or for further reading, you can refer to:

1. [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/pdf/2105.03824v3.pdf)
(Lee-Thorp et al., 2021)
2. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762v5.pdf) (Vaswani et al.,
2017)

Thanks to François Chollet for his Keras example on [English-to-Spanish translation with
a sequence-to-sequence
Transformer](https://keras.io/examples/nlp/neural_machine_translation_with_transformer/)
from which a major part of the decoder and transformer implementation was extracted.
"""

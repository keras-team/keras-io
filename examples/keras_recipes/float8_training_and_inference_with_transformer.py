"""
Title: Float8 training and inference with a simple Transformer model
Author: [Hongyu Chiu](https://github.com/james77777778)
Date created: 2024/05/14
Last modified: 2024/05/14
Description: Train a simple Transformer model with the float8 quantization.
Accelerator: GPU
"""

"""
## Introduction

As the number of parameters in Transformer models continues to grow, training
and inference become highly memory and compute-intensive. Therefore, 8-bit
floating point (FP8) was introduced, offering improved performance over 16-bit
floating point with nearly no degradation in accuracy.

In detail, there are two distinct types of FP8: E4M3 and E5M2, useful in
different parts of training.

- E4M3: It consists of 1 sign bit, 4 exponent bits and 3 bits of mantissa. It
can store values up to +/-448 and nan.
- E5M2: It consists of 1 sign bit, 5 exponent bits and 2 bits of mantissa. It
can store values up to +/-57344, +/-inf and nan. The tradeoff of the
increased dynamic range is lower precision of the stored values.

Typically, E4M3 is best used during the forward pass because activations and
weights require more precision. In the backward pass, however, E5M2 is utilized
because gradients are less susceptible to the loss of precision but require
higher dynamic range.

It is worth noting that FP8 inference deployment is greatly simplified, as
inference and training use the same datatype. This is in contrast to INT8
inference with networks trained in 32- or 16-bit floating point, which require
post-training quantization (PTQ) calibration and even quantization-aware
training (QAT) in order to maintain model accuracy.

In this example, we will build a simple Transformer model and train it with
both FP16 and FP8 precision. You will observe that the accuracy doesn't decrease
with lower precision.

Note: You will need a decent GPU with FP8 Tensor Cores support for the expected
performance improvement.
"""

"""
## Setup

We will use KerasHub library to simplify the model implementation. Additionally,
use mixed precision training to reduce the training time.

Note: The dependency on TensorFlow is only required for data processing.
"""

"""shell
pip install -q --upgrade keras-hub
pip install -q --upgrade keras  # Upgrade to Keras 3.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re

import keras
import keras_hub
import tensorflow as tf

keras.config.set_dtype_policy("mixed_bfloat16")

"""
Define some hyperparameters.
"""

EPOCHS = 3
BATCH_SIZE = 32
VOCABULARY_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200
MODEL_KWARGS = dict(
    vocabulary_size=VOCABULARY_SIZE,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    hidden_dim=32,  # Hidden size for each token
    num_heads=2,  # Number of attention heads
    intermediate_dim=32,  # Intermediate size in feedforward network
    dropout=0.1,  # Dropout rate
)

"""
## Dataset

First, let's download the IMDB dataset and extract it.
"""

"""shell
mkdir -p datasets
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -q -O datasets/aclImdb_v1.tar.gz
mkdir -p datasets/aclImdb
tar -xzf datasets/aclImdb_v1.tar.gz -C datasets
rm -rf datasets/aclImdb/train/unsup
"""

"""
We'll use the `keras.utils.text_dataset_from_directory` utility to generate our
labelled `tf.data.Dataset` dataset from text files.
"""

train_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/train",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
)
val_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/train",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
)
test_ds = keras.utils.text_dataset_from_directory(
    "datasets/aclImdb/test", batch_size=BATCH_SIZE
)

"""
We will now convert the text to lowercase.
"""

train_ds = train_ds.map(lambda x, y: (tf.strings.lower(x), y))
val_ds = val_ds.map(lambda x, y: (tf.strings.lower(x), y))
test_ds = test_ds.map(lambda x, y: (tf.strings.lower(x), y))

"""
Let's print a few samples.
"""

for text_batch, label_batch in train_ds.take(1):
    for i in range(3):
        print(f"Text: {text_batch.numpy()[i]}")
        print(f"Label: {label_batch.numpy()[i]}")

"""
### Tokenizing the data

We'll be using the `keras_hub.tokenizers.WordPieceTokenizer` layer to tokenize
the text. `keras_hub.tokenizers.WordPieceTokenizer` takes a WordPiece vocabulary
and has functions for tokenizing the text, and detokenizing sequences of tokens.

Before we define the tokenizer, we first need to train it on the dataset
we have. The WordPiece tokenization algorithm is a subword tokenization
algorithm; training it on a corpus gives us a vocabulary of subwords. A subword
tokenizer is a compromise between word tokenizers (word tokenizers need very
large vocabularies for good coverage of input words), and character tokenizers
(characters don't really encode meaning like words do). Luckily, KerasHub
makes it very simple to train WordPiece on a corpus with the
`keras_hub.tokenizers.compute_word_piece_vocabulary` utility.
"""


def train_word_piece(ds, vocab_size, reserved_tokens):
    word_piece_ds = ds.unbatch().map(lambda x, y: x)
    vocab = keras_hub.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab


"""
Every vocabulary has a few special, reserved tokens. We have two such tokens:

- `"[PAD]"` - Padding token. Padding tokens are appended to the input sequence
length when the input sequence length is shorter than the maximum sequence
length.
- `"[UNK]"` - Unknown token.
"""

reserved_tokens = ["[PAD]", "[UNK]"]
train_sentences = [element[0] for element in train_ds]
vocab = train_word_piece(train_ds, VOCABULARY_SIZE, reserved_tokens)

"""
Let's see some tokens!
"""

print("Tokens: ", vocab[100:110])

"""
Now, let's define the tokenizer. We will configure the tokenizer with the
the vocabularies trained above. We will define a maximum sequence length so that
all sequences are padded to the same length, if the length of the sequence is
less than the specified sequence length. Otherwise, the sequence is truncated.
"""

tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=False,
    sequence_length=MAX_SEQUENCE_LENGTH,
)

"""
Let's try and tokenize a sample from our dataset! To verify whether the text has
been tokenized correctly, we can also detokenize the list of tokens back to the
original text.
"""

input_sentence_ex = train_ds.take(1).get_single_element()[0][0]
input_tokens_ex = tokenizer(input_sentence_ex)

print("Sentence: ", input_sentence_ex)
print("Tokens: ", input_tokens_ex)
print("Recovered text after detokenizing: ", tokenizer.detokenize(input_tokens_ex))

"""
## Formatting the dataset

Next, we'll format our datasets in the form that will be fed to the models. We
need to tokenize the text.
"""


def format_dataset(sentence, label):
    sentence = tokenizer(sentence)
    return ({"input_ids": sentence}, label)


def make_dataset(dataset):
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(512).prefetch(tf.data.AUTOTUNE).cache()


train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)
test_ds = make_dataset(test_ds)


"""
## Model

Let's build a simple Transformer model. We will use `TokenAndPositionEmbedding`
and `TransformerDecoder` from KerasHub library. `TokenAndPositionEmbedding`
represents words and their order in a sentence, while `TransformerDecoder`
outputs one vector for each time step of our input sequence. Here, we take the
mean across all time steps and use a feedforward network on top of it to
classify text.
"""


def build_model(
    vocabulary_size=20000,
    max_sequence_length=200,
    hidden_dim=32,
    num_heads=2,
    intermediate_dim=32,
    dropout=0.1,
):
    token_id_input = keras.layers.Input(shape=(None,), dtype="int32", name="input_ids")
    x = keras_hub.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocabulary_size,
        sequence_length=max_sequence_length,
        embedding_dim=hidden_dim,
    )(token_id_input)
    x = keras.layers.Dropout(rate=dropout)(x)
    x = keras_hub.layers.TransformerDecoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=dropout,
    )(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(intermediate_dim, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs=token_id_input, outputs=outputs)


"""
## Training and evaluating our model

First, we train and evaluate the model with mixed precision
(`"mixed_bfloat16"`). Afterward, we compare the results with FP8
training/inference.
"""

model = build_model(**MODEL_KWARGS)
model.summary()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
result = model.evaluate(test_ds)
print(f"Accuracy (mixed_bfloat16): {result[1]:.2%}")

"""
We can enable FP8 training/inference with a one-line API:
`model.quantize("float8")`.
"""

model = build_model(**MODEL_KWARGS)
model.quantize("float8")

"""
To inspect that FP8 training takes place, we can print out some variables
related to FP8 training:

- `*_scale`: The scaling factor that shift the distribution of inputs, weights
    and gradients into the representable range of FP8. Defaults to `1.0`
- `*_amax_history`: The amax history window used for scaling factor computation.
    Defaults to `0.0` with the length of 1024.
"""

pattern = r"(transformer).+(multi_head).+(query).+(scale|amax_history)"
for v in model.trainable_variables:
    if re.findall(pattern, v.path):
        print(v.path)
        print(keras.ops.convert_to_numpy(v.value))

"""
The dtype policies of FP8 layers have also been modified.
"""

for layer in model._flatten_layers(recursive=True):
    if "float8" in str(layer.dtype_policy):
        print(f"{layer.name}: {layer.dtype_policy}")

"""
Let's train the model and see the results. We can verify that the accuracy
doesn't decrease with FP8 training that the variables containing FP8 information
change after fitting.
"""

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
result = model.evaluate(test_ds)
print(f"Accuracy (float8): {result[1]:.2%}")

for v in model.trainable_variables:
    if re.findall(pattern, v.path):
        print(v.path)
        print(keras.ops.convert_to_numpy(v.value))

"""
## Recipes

- The improvements in training speed are relatively small if the model is not
sufficiently large. The recommendation is to train with a model containing
parameters >5B.
- You will need hardware such as NVIDIA H100 that supports FP8 Tensor Cores to
gain the speedups.

## References
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Flax - fp8_ops.py](https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py)
"""

"""
Title: Float8 training and inference with a simple Transformer model
Author: [Hongyu Chiu](https://github.com/james77777778)
Date created: 2024/05/09
Last modified: 2024/05/09
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

We will use KerasNLP library to simplify the model implementation. Additionally,
use mixed precision training to reduce the training time.
"""

"""shell
pip install -q --upgrade keras-nlp
pip install -q --upgrade keras  # Upgrade to Keras 3.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import re

import keras
import keras_nlp

keras.config.set_dtype_policy("mixed_bfloat16")

"""
## Dataset

Let's load the IMDB dataset from `keras.datasets`. This dataset consists of
25,000 movie reviews labeled by sentiment (positive/negative). To simplify the
task, we only consider the top 20k words and the first 200 words of each review.
"""

vocabulary_size = 20000
max_sequence_length = 200
index_from = 3
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=vocabulary_size, start_char=1, oov_char=2, index_from=3
)
print(f"Number of training sequences: {len(x_train)}")
print(f"Number of validation sequences: {len(x_val)}")
x_train = keras.utils.pad_sequences(x_train, maxlen=max_sequence_length)
x_val = keras.utils.pad_sequences(x_val, maxlen=max_sequence_length)

"""
To better understand the dataset, we can print one encoded data, its decoded
sequence, and the label in the training data.
"""

word_index = keras.datasets.imdb.get_word_index()
inverted_word_index = dict((i + index_from, word) for (word, i) in word_index.items())
inverted_word_index[1] = "[START]"
inverted_word_index[2] = "[OOV]"

decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
print(f"Encoded training data:\n{x_train[0]}")
print(f"Decoded training data:\n{decoded_sequence}")
print(f"Labeld sentiment (0: negative, 1: positive):\n{y_train[0]}")

"""
## Model

Let's build a simple Transformer model. `TransformerBlock` outputs one vector
for each time step of our input sequence. Here, we take the mean across all time
steps and use a feedforward network on top of it to classify text.
"""


class TransformerBlock(keras.layers.Layer):
    def __init__(self, hidden_dim, intermediate_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=intermediate_dim, dropout=dropout
        )
        self.self_attention_norm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.self_attention_dropout = keras.layers.Dropout(dropout)

        self.feedforward_intermediate = keras.layers.Dense(
            intermediate_dim, activation="relu"
        )
        self.feedforward_output = keras.layers.Dense(hidden_dim)
        self.feedforward_norm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.feedforward_dropout = keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        x = inputs

        # Self attention block.
        residual = x
        x = self.self_attention(inputs, inputs, training=training)
        x = self.self_attention_dropout(x, training=training)
        x = x + residual
        x = self.self_attention_norm(x)

        # Feedforward block.
        residual = x
        x = self.feedforward_intermediate(x)
        x = self.feedforward_output(x)
        x = self.feedforward_dropout(x, training=training)
        x = x + residual
        x = self.feedforward_norm(x)
        return x


def build_model(
    vocabulary_size=20000,
    max_sequence_length=200,
    hidden_dim=32,
    num_heads=2,
    intermediate_dim=32,
    dropout=0.1,
):
    token_id_input = keras.layers.Input(shape=(None,), dtype="int32")
    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocabulary_size,
        sequence_length=max_sequence_length,
        embedding_dim=hidden_dim,
    )(token_id_input)
    x = keras.layers.Dropout(rate=dropout)(x)
    x = TransformerBlock(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=dropout,
    )(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(intermediate_dim, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs=token_id_input, outputs=outputs)


model_kwargs = dict(
    vocabulary_size=vocabulary_size,
    max_sequence_length=max_sequence_length,
    hidden_dim=32,  # Hidden size for each token
    num_heads=2,  # Number of attention heads
    intermediate_dim=32,  # Intermediate size in feedforward network
    dropout=0.1,  # Dropout rate
)

"""
## Training and evaluating our model

First, we train and evaluate the model with mixed precision
(`"mixed_bfloat16"`). Afterward, we compare the results with FP8
training/inference.
"""

model = build_model(**model_kwargs)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
result = model.evaluate(x_val, y_val)
print(f"Validation accuracy (float32): {result[1]:.2%}")

"""
We can enable FP8 training/inference with a one-line API:
`model.quantize("float8")`.
"""

model = build_model(**model_kwargs)
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
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
result = model.evaluate(x_val, y_val)
print(f"Validation accuracy (float8): {result[1]:.2%}")

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

"""
Title: End to End Masked Language Modeling & Fine-Tuning with BERT from Scratch
Author: [Ankur Singh](https://twitter.com/ankur310794)
Date created: 2020/09/03
Last modified: 2020/17/03
Description: Implement a Masked Language Modeling with BERT and fine-tune on IMDB Reviews dataset
"""
"""
## Introduction
Masked language modeling is a fill-in-the-blank task, where a model uses the context words surrounding a [MASK] token to try to predict what the [MASK] word should be. We will use IMDB Reviews raw text to pretrain and then use pretrained model weights to fine-tune sentiment classification.

Steps:
    1. Take raw text
    2. Build pretraining BERT model from scratch using masked language model approach
    3. Save pretrained model into .h5 file
    4. Build classification model using BERT pretrained model (from step 3)
    5. Evaluate classification model using end to end workflow
"""

"""
## Setup
Install tf-nightly via pip install tf-nightly.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from dataclasses import dataclass
import pandas as pd
import numpy as np

"""
## Set-up Configuration
"""


@dataclass
class Config:
    MAX_LEN = 128
    BATCH_SIZE = 16
    LR = 1e-5
    VOCAB_SIZE = 20000
    EMBED_DIM = 32
    NUM_HEAD = 2  # used in bert model
    FF_DIM = 32  # used in bert model


flags = Config()

"""
## Load Data
"""

"""shell
wget https://github.com/LawrenceDuan/IMDb-Review-Analysis/raw/master/IMDb_Reviews.csv
"""

data = pd.read_csv("IMDb_Reviews.csv")

"""
## Dataset Preparation
"""


def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
    """Build Text vectorization layer

  Args:
      texts (list): List of String i.e input texts
      vocab_size (int): vocab size
      max_seq (int): maximum sequence len
      special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].

  Returns:
      layers.Layer: Return TextVectorization Keras Layer
  """
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        standardize=None,
        output_sequence_length=max_seq,
    )

    vectorize_layer.adapt(texts)

    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[MASK]"]
    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer


vectorize_layer = get_vectorize_layer(
    data.review.values.tolist(),
    flags.VOCAB_SIZE,
    flags.MAX_LEN,
    special_tokens=["[MASK]"],
)

# Get mask token id for masked language model
mask_token_id = vectorize_layer(["[MASK]"]).numpy()[0][0]


def encode(texts):
    encoded_texts = vectorize_layer(texts)
    return encoded_texts.numpy()


def get_masked_input_and_labels(encoded_texts):
    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 2] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[
        inp_mask_2mask
    ] = mask_token_id  # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights


# Subset first 25000 examples for classification training
x_train = encode(data.review.values[0:25000])  # encode reviews with vectorizer
y_train = data.sentiment.values[0:25000]

train_classifier_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_classifier_ds = train_classifier_ds.shuffle(1000).batch(flags.BATCH_SIZE)

# Subset rest 25000 examples for classification evaluation

x_eval = data.review.values[25000:]  # take raw text
y_eval = data.sentiment.values[25000:]

eval_classifier_ds = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
eval_classifier_ds = eval_classifier_ds.shuffle(1000).batch(flags.BATCH_SIZE)

# Prepare data for masked language model
x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(x_train)

mlm_ds = tf.data.Dataset.from_tensor_slices(
    (x_masked_train, y_masked_labels, sample_weights)
)
mlm_ds = mlm_ds.shuffle(1000).batch(flags.BATCH_SIZE)

"""
## Create Masked Language BERT Model From Scratch
"""


def create_masked_language_bert_model():
    inputs = layers.Input((flags.MAX_LEN,), dtype=tf.int64)
    # Embedding layer
    embedding_layer = layers.Embedding(flags.VOCAB_SIZE, flags.EMBED_DIM)(inputs)

    # Query, Value, Key for attention
    query = layers.Dense(flags.EMBED_DIM)(embedding_layer)
    value = layers.Dense(flags.EMBED_DIM)(embedding_layer)
    key = layers.Dense(flags.EMBED_DIM)(embedding_layer)

    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=flags.NUM_HEAD, key_dim=flags.EMBED_DIM // flags.NUM_HEAD
    )(query, value, key)
    attention_output = layers.Dropout(0.1)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(
        embedding_layer + attention_output
    )

    # Feed-forward layer
    ffn = keras.Sequential(
        [layers.Dense(flags.FF_DIM, activation="relu"), layers.Dense(flags.EMBED_DIM),]
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1)(ffn_output)
    sequence_output = layers.LayerNormalization(epsilon=1e-6)(
        attention_output + ffn_output
    )
    outputs = layers.Dense(flags.VOCAB_SIZE)(sequence_output)

    mlm_model = keras.Model(inputs, outputs, name="mlm_model")

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=flags.LR)
    mlm_model.compile(optimizer=optimizer, loss=loss_fn)
    return mlm_model


mlm_model = create_masked_language_bert_model()
mlm_model.summary()

"""
## Train and Save 
"""

mlm_model.fit(mlm_ds, epochs=3)
mlm_model.save("bert_mlm_model.h5")

"""
## Create Classification Model
"""

# Load pretrained bert model
pretrained_bert_model = keras.models.load_model("bert_mlm_model.h5")


def create_classifier_bert_model():
    inputs = layers.Input((flags.MAX_LEN,), dtype=tf.int64)
    sequence_output = pretrained_bert_model(inputs)
    pooled_output = layers.GlobalMaxPooling1D()(sequence_output)
    outputs = layers.Dense(1, activation="sigmoid")(pooled_output)
    classifer_model = keras.Model(inputs, outputs, name="classification")
    optimizer = keras.optimizers.Adam(learning_rate=flags.LR)
    classifer_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return classifer_model


classifer_model = create_classifier_bert_model()
classifer_model.summary()


"""
## Train and Evaluate
"""

classifer_model.fit(train_classifier_ds, epochs=5)


def get_end_to_end(model):
    inputs_string = keras.Input(shape=(1,), dtype="string")
    indices = vectorize_layer(inputs_string)
    outputs = model(indices)
    end_to_end_model = keras.Model(inputs_string, outputs, name="end_to_end_model")
    optimizer = keras.optimizers.Adam(learning_rate=flags.LR)
    end_to_end_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return end_to_end_model


end_to_end_classification_model = get_end_to_end(classifer_model)
end_to_end_classification_model.evaluate(eval_classifier_ds)

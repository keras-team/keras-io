"""
Title: End-to-end Masked Language Modeling with BERT
Author: [Ankur Singh](https://twitter.com/ankur310794)
Date created: 2020/09/18
Last modified: 2020/09/18
Description: Implement a Masked Language Model (MLM) with BERT and fine-tune it on the IMDB Reviews dataset.
"""
"""
## Introduction

Masked Language Modeling is a fill-in-the-blank task,
where a model uses the context words surrounding a mask token to try to predict what the
masked word should be.

For an input that contains one or more mask tokens,
the model will generate the most likely substitution for each.

Example: 

- Input: "I have watched this [MASK] and it was awesome."
- Output: "I have watched this movie and it was awesome."

Masked language modeling is a great way to train a language
model in a self-supervised setting (without human-annotated labels).
Such a model can then be fine-tuned to accomplish various supervised
NLP tasks.

This example teaches you how to build a BERT model from scratch,
train it with the mask language modeling task,
and then fine-tune this model on a sentiment classification task.

We will use the Keras `TextVectorization` and `MultiHeadAttention` layers
to create a BERT Transformer-Encoder network architecture.

Note: This example should be run with `tf-nightly`.
"""

"""
## Setup

Install `tf-nightly` via `pip install tf-nightly`.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from dataclasses import dataclass
import pandas as pd
import numpy as np
import glob

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

We will first download IMDB data and load into pandas dataframe.
"""

"""shell
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
"""


def get_text_list_from_files(files):
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list


def get_data():
    pos_files = glob.glob("aclImdb/train/pos/*.txt") + glob.glob(
        "aclImdb/test/pos/*.txt"
    )
    pos_texts = get_text_list_from_files(pos_files)

    neg_files = glob.glob("aclImdb/train/neg/*.txt") + glob.glob(
        "aclImdb/test/neg/*.txt"
    )
    neg_texts = get_text_list_from_files(neg_files)

    data = pd.DataFrame(
        {
            "text": pos_texts + neg_texts,
            "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
        }
    )
    data = data.sample(len(data)).reset_index(drop=True)
    return data


data = get_data()

"""
## Dataset Preparation

We will use TextVectorization to vectorize text into token id. This layer gives flexibilty to manage text in Keras model.
It transforms a batch of strings into either a list of token indices (one sample = 1D tensor of integer token indices) or a dense representation (one sample = 1D tensor of float values representing data about the sample’s tokens).

Below, there will be 3 preprocessing functions. 

1.  get_vectorize_layer function will use to build TextVectorization layer.
2.  encode function will use to encode raw text into integer token ids
3.  get_masked_input_and_labels function will use to mask input token ids. It masks 15% of all input tokens in each sequence at random. 

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


# Subset first 20000 examples for training
x_train = encode(data.review.values[0:20000])  # encode reviews with vectorizer
y_train = data.sentiment.values[0:20000]
train_classifier_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1000)
    .batch(flags.BATCH_SIZE)
)

# Subset first 5000 examples for evaluation
x_eval = encode(data.review.values[20000:25000])  # encode reviews with vectorizer
y_eval = data.sentiment.values[20000:25000]
eval_classifier_ds = (
    tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
    .shuffle(1000)
    .batch(flags.BATCH_SIZE)
)


# Subset rest 25000 examples for testing
x_test = data.review.values[25000:]  # take raw text for end to end model evaluation
y_test = data.sentiment.values[25000:]

test_classifier_ds = (
    tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
    .shuffle(1000)
    .batch(flags.BATCH_SIZE)
)

# Prepare data for masked language model
x_all_review = encode(data.review.values)
x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(
    x_all_review
)

mlm_ds = tf.data.Dataset.from_tensor_slices(
    (x_masked_train, y_masked_labels, sample_weights)
)
mlm_ds = mlm_ds.shuffle(1000).batch(flags.BATCH_SIZE)

"""
## Create BERT model for masked language modeling

We will create a BERT-like pretraining model architecture
using the `MultiHeadAttention` layer.
It will take token ids as inputs (including masked tokens)
and it will predict the correct ids for the masked input tokens.
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
## Fine-tune a sentiment classification model

We will fine-tune our self-supervised model on a downstream task of sentiment classification.
To do this, let's create a classifier by adding a pooling layer and a `Dense` layer on top of the
pretrained BERT features.

"""

# Load pretrained BERT model
pretrained_bert_model = keras.models.load_model("bert_mlm_model.h5")
# Freeze it
pretrained_bert_model.trainable = False


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

# Train the classifier with frozen BERT stage
classifer_model.fit(train_classifier_ds, epochs=5, validation_data=eval_classifier_ds)

# Unfreeze the BERT model for fine-tuning
pretrained_bert_model.trainable = True
classifer_model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)
classifer_model.fit(train_classifier_ds, epochs=5, validation_data=eval_classifier_ds)


"""
## Create an end-to-end model and evaluate it

When you want to deploy a model, it's best if it already includes its preprocessing
pipeline, so that you don't have to reimplement the preprocessing logic in your
production environment. Let's create an end-to-end model that incorporates
the `TextVectorization` layer, and let's evaluate. Our model will accept raw strings
as input.
"""

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
end_to_end_classification_model.evaluate(test_classifier_ds)

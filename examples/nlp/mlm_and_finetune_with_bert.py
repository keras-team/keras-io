"""
Title: End to End Masked Language Modeling & Fine-Tuning with BERT from Scratch
Author: [Ankur Singh](https://twitter.com/ankur310794)
Date created: 2020/09/03
Last modified: 2020/16/03
Description: Implement a Masked Language Modeling with BERT and fine-tune on IMDB Reviews dataset
"""
"""
## Introduction
Masked language modeling is a fill-in-the-blank task, where a model uses the context words surrounding a [MASK] token to try to predict what the [MASK] word should be.
"""

"""
## Setup
Install tf-nightly via pip install tf-nightly.
"""

import tensorflow as tf
from dataclasses import dataclass
import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
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
wget https://raw.githubusercontent.com/SrinidhiRaghavan/AI-Sentiment-Analysis-on-IMDB-Dataset/master/imdb_tr.csv
"""

data = pd.read_csv("imdb_tr.csv", encoding="ISO-8859-1")

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
        tf.keras.layers.Layer: Return TextVectorization Keras Layer
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
    data.text.values.tolist(),
    flags.VOCAB_SIZE,
    flags.MAX_LEN,
    special_tokens=["[MASK]"],
)

# get mask token id for mlm
mask_token_id = vectorize_layer(["[MASK]"]).numpy()[0][0]


class IMDBReviewsDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data

    Args:
        texts (list): List of String i.e input texts
        labels (list): List of Labels i.e for classification model only
        batch_size (int): batch size
        vectorizer (TextVectorization): Keras TextVectorization layer
        is_training (bool, optional): Is training dataset generator. Defaults to True.
        model_type (str, optional): Used as a flag for mlm and classifier. Defaults to 'classification'.
    """

    def __init__(
        self,
        texts,
        labels,
        batch_size,
        vectorizer,
        is_training=True,
        model_type="classification",
    ):
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.vectorizer = vectorizer
        self.is_training = is_training
        self.model_type = model_type
        self.indexes = np.arange(len(self.texts))

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.texts) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        texts = self.texts[indexes]

        encoded = self.vectorizer(texts).numpy()

        if self.is_training and self.model_type == "mlm":
            X_mlm, y_labels, sample_weights = self.prepare_mlm_input_and_labels(encoded)
            return (X_mlm, y_labels, sample_weights)

        elif self.is_training and self.model_type == "classification":
            labels = np.array(self.labels[indexes], dtype="int32")
            return (encoded, labels)

        else:
            return encoded

    def prepare_mlm_input_and_labels(self, X):
        # 15% BERT masking
        inp_mask = np.random.rand(*X.shape) < 0.15
        # do not mask special tokens
        inp_mask[X <= 2] = False
        # set targets to -1 by default, it means ignore
        labels = -1 * np.ones(X.shape, dtype=int)
        # set labels for masked tokens
        labels[inp_mask] = X[inp_mask]

        # prepare input
        X_mlm = np.copy(X)
        # set input to [MASK] which is the last token for the 90% of tokens
        # this means leaving 10% unchanged
        inp_mask_2mask = inp_mask & (np.random.rand(*X.shape) < 0.90)
        X_mlm[inp_mask_2mask] = mask_token_id  # mask token is the last in the dict

        # set 10% to a random token
        inp_mask_2random = inp_mask_2mask & (np.random.rand(*X.shape) < 1 / 9)
        X_mlm[inp_mask_2random] = np.random.randint(
            3, mask_token_id, inp_mask_2random.sum()
        )

        # prepare sample_weights to pass to .fit() method
        sample_weights = np.ones(labels.shape)
        sample_weights[labels == -1] = 0

        # y_labels would be same as X i.e input tokens
        y_labels = np.copy(X)

        return X_mlm, y_labels, sample_weights


classifier_train_data = IMDBReviewsDataGenerator(
    data["text"].values.astype("str"),
    data["polarity"].values.astype("int"),
    flags.BATCH_SIZE,
    vectorize_layer,
)
mlm_train_data = IMDBReviewsDataGenerator(
    data["text"].values.astype("str"),
    None,
    flags.BATCH_SIZE,
    vectorize_layer,
    model_type="mlm",
)

"""
## Create BERT Model From Scratch
"""


class BERT:
    def __init__(self, flags):
        self.flags = flags

    def __call__(self, inputs):
        # embedding layer
        embedding_layer = tf.keras.layers.Embedding(flags.VOCAB_SIZE, flags.EMBED_DIM)(
            inputs
        )

        # query, value, key for attention
        query = tf.keras.layers.Dense(flags.EMBED_DIM)(embedding_layer)
        value = tf.keras.layers.Dense(flags.EMBED_DIM)(embedding_layer)
        key = tf.keras.layers.Dense(flags.EMBED_DIM)(embedding_layer)

        # multi headed self-attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=flags.NUM_HEAD, key_dim=flags.EMBED_DIM // flags.NUM_HEAD
        )(query, value, key)
        attention_output = tf.keras.layers.Dropout(0.1)(attention_output)
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
            embedding_layer + attention_output
        )

        # feed-forward leyer
        ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(flags.FF_DIM, activation="relu"),
                tf.keras.layers.Dense(flags.EMBED_DIM),
            ]
        )
        ffn_output = ffn(attention_output)
        ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
        sequence_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
            attention_output + ffn_output
        )

        pooled_output = tf.keras.layers.GlobalMaxPooling1D()(ffn_output)

        return sequence_output, pooled_output


def build_mlm_model():
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input((flags.MAX_LEN,), dtype=tf.int32)
    sequence_output, pooled_output = BERT(flags)(inputs)
    outputs = tf.keras.layers.Dense(flags.VOCAB_SIZE)(sequence_output)
    mlm_model = tf.keras.Model(inputs, outputs)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE, from_logits=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=flags.LR)
    mlm_model.compile(optimizer=optimizer, loss=loss_fn)
    return mlm_model


def build_classifer_model():
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input((flags.MAX_LEN,), dtype=tf.int32)
    sequence_output, pooled_output = BERT(flags)(inputs)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(pooled_output)
    classifer_model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=flags.LR)
    classifer_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return classifer_model


def get_end_to_end(model):
    inputs_string = tf.keras.Input(shape=(1,), dtype="string")
    indices = vectorize_layer(inputs_string)
    outputs = model(indices)
    end_to_end_model = tf.keras.Model(inputs_string, outputs, name="end_to_end")
    return end_to_end_model


"""
## Build, Train, Save MLM
"""

mlm_model = build_mlm_model()
mlm_model.fit(mlm_train_data, epochs=1)
mlm_model.save("mlm_model.h5")

"""
## Build, Train, Save Fine-Tune Model
"""

classifer_model = build_classifer_model()
classifer_model.load_weights("mlm_model.h5", skip_mismatch=True, by_name=True)
classifer_model.fit(classifier_train_data, epochs=1)

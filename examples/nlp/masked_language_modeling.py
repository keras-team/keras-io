"""
# End-to-end Masked Language Modeling with BERT

**Author:** [Ankur Singh](https://twitter.com/ankur310794)<br>
**Date created:** 2020/09/18<br>
**Last modified:** 2024/05/05<br>
**Description:** Implement a Masked Language Model (MLM) with BERT and fine-tune it on
the IMDB Reviews dataset.<br>
**Accelerator:** GPU
**Converted to Keras 3 by:** [Sitam Meur](https://github.com/sitamgithub-MSIT)<br>
**Converted to Keras 3 Backend-Agnostic by:** [Mrutyunjay
Biswal](https://twitter.com/LearnStochastic)<br>
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
train it with the masked language modeling task,
and then fine-tune this model on a sentiment classification task.

We will use the Keras `TextVectorization` and `MultiHeadAttention` layers, and
`PositionEmbedding` from `keras-nlp`
to create a BERT Transformer-Encoder network architecture.

Note: This is backend-agnostic, i.e. update the keras backend to "tensorflow", "torch",
or "jax" as shown in the code, and it should work with no other code change.
"""

"""
## Setup
"""

# install keras 3.x and keras-nlp
# !pip install --upgrade keras keras-nlp

import os

# set backend ["tensorflow", "jax", "torch"]
os.environ["KERAS_BACKEND"] = "jax"

import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass

import keras
from keras import ops
from keras import layers

import keras_nlp
import tensorflow as tf

"""
## Configuration
"""


@dataclass
class Config:
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1
    NUM_EPOCHS = 1
    STEPS_PER_EPOCH = 2


config = Config()

"""
## Download the Data: IMDB Movie Review Sentiment Classification
Download the IMDB data and load into a Pandas DataFrame.
"""

fpath = keras.utils.get_file(
    origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
)
dirpath = Path(fpath).parent.absolute()
_ = os.system(f"tar -xf {fpath} -C {dirpath}")

"""
The `aclImdb` folder contains a `train` and `test` subfolder:
"""

_ = os.system(f"ls {dirpath}/aclImdb")
_ = os.system(f"ls {dirpath}/aclImdb/train")
_ = os.system(f"ls {dirpath}/aclImdb/test")

"""
We are only interested in the `pos` and `neg` subfolders, so let's delete the rest:
"""

_ = os.system(f"rm -r {dirpath}/aclImdb/train/unsup")
_ = os.system(f"rm -r {dirpath}/aclImdb/train/*.feat")
_ = os.system(f"rm -r {dirpath}/aclImdb/train/*.txt")
_ = os.system(f"rm -r {dirpath}/aclImdb/test/*.feat")
_ = os.system(f"rm -r {dirpath}/aclImdb/test/*.txt")

"""
Let's read the dataset from the text files to a DataFrame.
"""


def get_text_list_from_files(files):
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list


def get_data_from_text_files(folder_name):

    pos_files = glob.glob(f"{dirpath}/aclImdb/" + folder_name + "/pos/*.txt")
    pos_texts = get_text_list_from_files(pos_files)
    neg_files = glob.glob(f"{dirpath}/aclImdb/" + folder_name + "/neg/*.txt")
    neg_texts = get_text_list_from_files(neg_files)
    df = pd.DataFrame(
        {
            "review": pos_texts + neg_texts,
            "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
        }
    )
    df = df.sample(len(df)).reset_index(drop=True)
    return df


train_df = get_data_from_text_files("train")
test_df = get_data_from_text_files("test")

all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
assert len(all_data) != 0, f"{all_data} is empty"

"""
## Dataset preparation

We will use the `TextVectorization` layer to vectorize the text into integer token ids.
It transforms a batch of strings into either
a sequence of token indices (one sample = 1D array of integer token indices, in order)
or a dense representation (one sample = 1D array of float values encoding an unordered
set of tokens).

Below, we define 3 preprocessing functions.

1.  The `get_vectorize_layer` function builds the `TextVectorization` layer.
2.  The `encode` function encodes raw text into integer token ids.
3.  The `get_masked_input_and_labels` function will mask input token ids.
It masks 15% of all input tokens in each sequence at random.
"""


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
    """Build Text vectorization layer

    Args:
      texts (list): List of string i.e input texts
      vocab_size (int): vocab size
      max_seq (int): Maximum sequence length.
      special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].

    Returns:
        layers.Layer: Return TextVectorization Keras Layer
    """
    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        standardize=custom_standardization,
        output_sequence_length=max_seq,
    )
    vectorize_layer.adapt(texts)

    # Insert mask token in vocabulary
    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[mask]"]
    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer


vectorize_layer = get_vectorize_layer(
    all_data.review.values.tolist(),
    config.VOCAB_SIZE,
    config.MAX_LEN,
    special_tokens=["[mask]"],
)

# Get mask token id for masked language model
mask_token_id = ops.convert_to_numpy(vectorize_layer(["[mask]"])[0][0])


def encode(texts):
    encoded_texts = vectorize_layer(texts)
    return ops.convert_to_numpy(encoded_texts)


# todo: make this backend agnostic
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
    encoded_texts_masked[inp_mask_2mask] = (
        mask_token_id  # mask token is the last in the dict
    )

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape, dtype="float32")
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights


# We have 25000 examples for training
x_train = encode(train_df.review.values)  # encode reviews with vectorizer
y_train = train_df.sentiment.values
train_classifier_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1000)
    .batch(config.BATCH_SIZE)
)

# We have 25000 examples for testing
x_test = encode(test_df.review.values)
y_test = test_df.sentiment.values
test_classifier_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
    config.BATCH_SIZE
)

# Build dataset for end to end model input (will be used at the end)
test_raw_classifier_ds = tf.data.Dataset.from_tensor_slices(
    (test_df.review.values, y_test)
).batch(config.BATCH_SIZE)

# Prepare data for masked language model
x_all_review = encode(all_data.review.values)
x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(
    x_all_review
)

mlm_ds = tf.data.Dataset.from_tensor_slices(
    (x_masked_train, y_masked_labels, sample_weights)
)
mlm_ds = mlm_ds.shuffle(1000).batch(config.BATCH_SIZE)

"""
## Create BERT model (Pretraining Model) for masked language modeling

We will create a BERT-like pretraining model architecture
using the `MultiHeadAttention` layer.
It will take token ids as inputs (including masked tokens)
and it will predict the correct ids for the masked input tokens.

We will use `keras.Model` and `Layer` to define sub-classes for
BERT Encoder layer, and the MLM Model.
"""

class BertEncoderLayer(layers.Layer):
    def __init__(self, layer_num, **kwargs):
        super().__init__(**kwargs)
        self.layer_num = layer_num

        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=config.NUM_HEAD,
            key_dim=config.EMBED_DIM // config.NUM_HEAD,
            name=f"encoder_{self.layer_num}_multiheadattention"
        )

        self.multi_head_attention_dropout = layers.Dropout(
            0.1, name=f"encoder_{self.layer_num}_attn_dropout",
        )

        self.multi_head_attention_norm = layers.LayerNormalization(
            epsilon=1e-6, name=f"encoder_{self.layer_num}_attn_layernorm"
        )

        self.ffn = keras.Sequential(
            [
                layers.Dense(config.FF_DIM, activation="relu"),
                layers.Dense(config.EMBED_DIM)
            ],
            name=f"encoder_{self.layer_num}_ffn"
        )

        self.ffn_dropout = layers.Dropout(
            0.1, name=f"encoder_{self.layer_num}_ffn_dropout"
        )

        self.ffn_layernorm = layers.LayerNormalization(
            epsilon=1e-6, name=f"encoder_{self.layer_num}_ffn_layernorm"
        )

    def call(self, inputs, training=False):
        query, key, value = inputs
        attn_output = self.multi_head_attention(query, key, value)
        attn_output = self.multi_head_attention_dropout(attn_output)
        attn_output = self.multi_head_attention_norm(query + attn_output)

        ffn_output = self.ffn(attn_output)
        ffn_output = self.ffn_dropout(ffn_output)

        sequence_output = self.ffn_layernorm(attn_output + ffn_output)

        return sequence_output

class MaskedLanguageModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.word_embeddings = layers.Embedding(
            config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
        )
        self.position_embeddings = keras_nlp.layers.PositionEmbedding(
            sequence_length=config.MAX_LEN
        )

        self.bert_encoder_layers = [
            BertEncoderLayer(n_layer) for n_layer in range(config.NUM_LAYERS)
        ]

        self.mlm_output = layers.Dense(config.VOCAB_SIZE, activation="softmax", name="mlm_cls")

    def call(self, inputs, training=False):
        word_embeddings = self.word_embeddings(inputs)
        position_embeddings = self.position_embeddings(word_embeddings)
        encoder_output = word_embeddings + position_embeddings

        for bert_encoder_layer in self.bert_encoder_layers:
            encoder_output = bert_encoder_layer([encoder_output, encoder_output, encoder_output])

        return self.mlm_output(encoder_output)
        
    def get_config(self):
      return super().get_config()

# Reset Keras backend session
keras.backend.clear_session()

# Define model and compile
optimizer = keras.optimizers.Adam(learning_rate=config.LR)
loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction=None)

bert_masked_model = MaskedLanguageModel()
bert_masked_model.compile(optimizer=optimizer, loss=loss_fn)

# Show model summary
bert_masked_model.summary()

"""
## Define a Callback to Generate Masked Token
"""

id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}


class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5):
        self.sample_tokens = ops.convert_to_numpy(sample_tokens)
        self.k = top_k

    def decode(self, tokens):
        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.sample_tokens)

        masked_index = ops.where(self.sample_tokens == mask_token_id)
        masked_index = ops.convert_to_numpy(masked_index[1])[0]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction.argsort()[-self.k :][::-1]
        values = mask_prediction[top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(self.sample_tokens[0])
            tokens[masked_index] = p
            result = {
                "input_text": self.decode(self.sample_tokens[0]),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)


sample_tokens = vectorize_layer(["I have watched this [mask] and it was awesome"])
generator_callback = MaskedTextGenerator(sample_tokens)

"""
## Train and Save
"""

bert_masked_model.fit(
    mlm_ds,
    epochs=config.NUM_EPOCHS,
    steps_per_epoch=config.STEPS_PER_EPOCH,
    callbacks=[generator_callback],
)
bert_masked_model.save("bert_mlm_imdb.keras")

"""
## Fine-tune a sentiment classification model

We will fine-tune our self-supervised model on a downstream task of sentiment
classification.
To do this, let's create a classifier by adding a pooling layer and a `Dense` layer on
top of the
pretrained BERT features.
"""

# Load pretrained bert model
mlm_model = keras.models.load_model(
    "bert_mlm_imdb.keras", custom_objects={"MaskedLanguageModel": MaskedLanguageModel}
)
pretrained_bert_model = keras.Model(
    mlm_model.input, mlm_model.get_layer("encoder_0_ffn_layernormalization").output
)

# Freeze it
pretrained_bert_model.trainable = False


def create_classifier_bert_model():
    inputs = layers.Input((config.MAX_LEN,), dtype="int32")
    sequence_output = pretrained_bert_model(inputs)
    pooled_output = layers.GlobalMaxPooling1D()(sequence_output)
    hidden_layer = layers.Dense(64, activation="relu")(pooled_output)
    outputs = layers.Dense(1, activation="sigmoid")(hidden_layer)
    classifer_model = keras.Model(inputs, outputs, name="classification")
    optimizer = keras.optimizers.Adam()
    classifer_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return classifer_model


classifer_model = create_classifier_bert_model()
classifer_model.summary()

# Train the classifier with frozen BERT stage
classifer_model.fit(
    train_classifier_ds,
    epochs=5,
    validation_data=test_classifier_ds,
)

# Unfreeze the BERT model for fine-tuning
pretrained_bert_model.trainable = True
optimizer = keras.optimizers.Adam()
classifer_model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)
classifer_model.fit(
    train_classifier_ds,
    epochs=config.NUM_EPOCHS,
    validation_data=test_classifier_ds,
)

"""
## Create an end-to-end model and evaluate it

When you want to deploy a model, it's best if it already includes its preprocessing
pipeline, so that you don't have to reimplement the preprocessing logic in your
production environment. Let's create an end-to-end model that incorporates
the `TextVectorization` layer, and let's evaluate. Our model will accept raw strings
as input.
"""


def get_end_to_end(model):
    inputs_string = layers.Input(shape=(1,), dtype="string")
    indices = vectorize_layer(inputs_string)
    outputs = model(indices)
    end_to_end_model = keras.Model(inputs_string, outputs, name="end_to_end_model")
    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    end_to_end_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return end_to_end_model


end_to_end_classification_model = get_end_to_end(classifer_model)
end_to_end_classification_model.evaluate(test_raw_classifier_ds)

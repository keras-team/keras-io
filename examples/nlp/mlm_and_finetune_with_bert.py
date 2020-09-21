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
import re
from pprint import pprint

"""
## Set-up Configuration
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


config = Config()

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


def get_data_from_text_files(folder_name):

    pos_files = glob.glob("aclImdb/" + folder_name + "/pos/*.txt")
    pos_texts = get_text_list_from_files(pos_files)
    neg_files = glob.glob("aclImdb/" + folder_name + "/neg/*.txt")
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

all_data = train_df.append(test_df)

"""
## Dataset Preparation

We will use TextVectorization to vectorize text into token id. This layer gives flexibilty to manage text in Keras model.
It transforms a batch of strings into either a list of token indices (one sample = 1D tensor of integer token indices) or a dense representation (one sample = 1D tensor of float values representing data about the sample’s tokens).

Below, there will be 3 preprocessing functions. 

1.  get_vectorize_layer function will use to build TextVectorization layer.
2.  encode function will use to encode raw text into integer token ids
3.  get_masked_input_and_labels function will use to mask input token ids. It masks 15% of all input tokens in each sequence at random. 

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
        standardize=custom_standardization,
        output_sequence_length=max_seq,
    )

    vectorize_layer.adapt(texts)

    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[mask]"]
    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer


vectorize_layer = get_vectorize_layer(
    data.review.values.tolist(),
    flags.VOCAB_SIZE,
    flags.MAX_LEN,
    special_tokens=["[mask]"],
)

# Get mask token id for masked language model
mask_token_id = vectorize_layer(["[mask]"]).numpy()[0][0]


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


# We have 25000 examples for training
x_train = encode(train_df.review.values)  # encode reviews with vectorizer
y_train = train_df.sentiment.values
train_classifier_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1000)
    .batch(flags.BATCH_SIZE)
)

# We have 25000 examples for testing
x_test = encode(test_df.review.values)
y_test = test_df.sentiment.values
test_classifier_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
    flags.BATCH_SIZE
)

# Build dataset for end to end model input (will be used at the end)
test_raw_classifier_ds = tf.data.Dataset.from_tensor_slices(
    (test_df.review.values, y_test)
).batch(flags.BATCH_SIZE)

# Prepare data for masked language model
x_all_review = encode(all_data.review.values)
x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(
    x_all_review
)

mlm_ds = tf.data.Dataset.from_tensor_slices(
    (x_masked_train, y_masked_labels, sample_weights)
)
mlm_ds = mlm_ds.shuffle(1000).batch(flags.BATCH_SIZE)

"""
## Create BERT model (Pretraining Model) for masked language modeling

We will create a BERT-like pretraining model architecture
using the `MultiHeadAttention` layer.
It will take token ids as inputs (including masked tokens)
and it will predict the correct ids for the masked input tokens.
"""


class BERTLayer(layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # Multi headed self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=config.NUM_HEAD,
            key_dim=config.EMBED_DIM // config.NUM_HEAD,
            name="multiheadattention",
        )
        # Feed-forward layer
        self.ffn = keras.Sequential(
            [
                layers.Dense(config.FF_DIM, activation="relu"),
                layers.Dense(config.EMBED_DIM),
            ],
            name="feedforwardnetwork",
        )
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        query, key, value = inputs, inputs, inputs
        attention_output = self.attention(inputs, inputs, inputs)
        attention_output = self.dropout1(attention_output)
        attention_output = self.layernorm1(inputs + attention_output)

        ffn_output = self.ffn(attention_output)
        fnn_output = self.dropout2(ffn_output)
        sequence_output = self.layernorm2(attention_output + ffn_output)

        return sequence_output


class Encoder(layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Create BERTLayer
        self.layer = [
            BERTLayer(config, name="layer_._{}".format(i))
            for i in range(config.NUM_LAYERS)
        ]

    def call(self, inputs):
        outputs = inputs
        for layer_module in self.layer:
            outputs = layer_module(outputs)

        return outputs


# Loss metric to track masked language model loss
loss_tracker = keras.metrics.Mean(name="mlm_loss")


class BERTPreTraining(keras.Model):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        # Create Word Embedding
        self.word_embeddings = layers.Embedding(
            config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
        )
        # Create Position Embedding
        self.position_embeddings = layers.Embedding(
            input_dim=config.MAX_LEN,
            output_dim=config.EMBED_DIM,
            weights=[self.get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
            name="position_embedding",
        )
        # Create Encoder
        self.encoder = Encoder(config)
        # Create Masked Model output layer
        self.mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls")

    def get_pos_encoding_matrix(self, max_len, d_emb):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
                if pos != 0
                else np.zeros(d_emb)
                for pos in range(max_len)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    def masked_sparse_categorical_crossentropy(self, y_true, y_pred, sample_weight):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            reduction=keras.losses.Reduction.NONE, from_logits=True
        )
        return loss_fn(y_true, y_pred, sample_weight=sample_weight)

    def call(self, inputs):
        word_embeddings_output = self.word_embeddings(inputs)
        position_embeddings_output = self.position_embeddings(
            tf.range(start=0, limit=config.MAX_LEN, delta=1)
        )

        embeddings = word_embeddings_output + position_embeddings_output

        encoder_output = self.encoder(embeddings)

        output_cls = self.mlm_output(encoder_output)

        return output_cls, encoder_output

    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:

            predictions, _ = self(features, training=True)
            loss = self.masked_sparse_categorical_crossentropy(
                labels, predictions, sample_weight=sample_weight
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]


id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}


class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5):
        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens):
        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        prediction, _ = self.model.predict(self.sample_tokens)
        prediction = tf.convert_to_tensor(prediction)
        prediction = tf.nn.softmax(prediction)

        masked_index = np.where(self.sample_tokens == mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction.numpy()[0][masked_index]

        topk = tf.math.top_k(mask_prediction, k=self.k)
        values, predictions = topk.values.numpy(), topk.indices.numpy()

        for i in range(len(predictions[0])):
            p = predictions[0][i]
            v = values[0][i]
            tokens = np.copy(sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(sample_tokens[0].numpy()),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }

            print()
            pprint(result)


sample_tokens = vectorize_layer(["I have watched this [mask] and it was awesome"])
generator_callback = MaskedTextGenerator(sample_tokens.numpy())

bert_pretraining = BERTPreTraining(config)
optimizer = keras.optimizers.Adam(learning_rate=config.LR)
bert_pretraining.compile(optimizer=optimizer)

"""
## Train and Save
"""

bert_pretraining.fit(mlm_ds, epochs=2, callbacks=[generator_callback])
bert_pretraining.save("mlm_imdb_bert.h5py")

"""
## Fine-tune a sentiment classification model

We will fine-tune our self-supervised model on a downstream task of sentiment classification.
To do this, let's create a classifier by adding a pooling layer and a `Dense` layer on top of the
pretrained BERT features.

"""

# Load pretrained bert model
BERT_MODEL = keras.models.load_model("mlm_imdb_bert.h5py")
# Freeze it
BERT_MODEL.trainable = False


def create_classifier_bert_model():
    inputs = layers.Input((flags.MAX_LEN,), dtype=tf.int64)
    _, sequence_output = BERT_MODEL(inputs)
    pooled_output = layers.GlobalMaxPooling1D()(sequence_output)
    hidden_layer = layers.Dense(64, activation="relu")(pooled_output)
    outputs = layers.Dense(1, activation="sigmoid")(hidden_layer)
    classifer_model = keras.Model(inputs, outputs, name="classification")
    optimizer = keras.optimizers.Adam(0.0001)
    classifer_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return classifer_model


classifer_model = create_classifier_bert_model()
classifer_model.summary()

# Train the classifier with frozen BERT stage
classifer_model.fit(train_classifier_ds, epochs=5, validation_data=test_classifier_ds)

# Unfreeze the BERT model for fine-tuning
BERT_MODEL.trainable = True
optimizer = keras.optimizers.Adam(0.0001)
classifer_model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)
print(classifer_model.summary())
classifer_model.fit(train_classifier_ds, epochs=5, validation_data=test_classifier_ds)

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
    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    end_to_end_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return end_to_end_model


end_to_end_classification_model = get_end_to_end(classifer_model)
end_to_end_classification_model.evaluate(test_raw_classifier_ds)

"""
Title: Text Classification using FNet
Author: [Abheesht Sharma](https://github.com/abheesht17/)
Date created: 2021/06/01
Last modified: 2021/06/01
Description: Text Classification on the SST-2 Dataset using KerasNLP's `FNetEncoder` layer
"""

"""
## Introduction

In this example, we will demonstrate the ability of FNet to achieve comparable
results with a vanilla Transformer model on the text classification task.
We will be using the SST-2 dataset (belongs to the GLUE benchmark), which is a
collection of movie reviews labelled either positive or negative (sentiment
analysis).

To build the tokenizer, model, etc., we will use components from
[KerasNLP](https://github.com/keras-team/keras-nlp). KerasNLP makes life easier
for people who want to build NLP pipelines! :)

### Model

In 2017, a paper titled [Attention is All You Need](https://arxiv.org/abs/1706.03762)
introduced the Transformer architecture. The Transformer model uses self-attention
to learn representations of tokens in a piece of text. Succinctly put, to
compute the representation of a token, the self-attention mechanism computes
attention scores between a token and every other token in the sequence.
It then uses the scores to compute a weighted average of the tokens in the sequence.  
Since then, multiple language models such as BERT, RoBERTa, etc. have
been released. All these models have the same Transformer architecture with 
different pretraining tasks in order to "learn" language.

Note: To learn more about the Transformer architecture, please visit [Jay Alammar's
peerless blog](https://jalammar.github.io/illustrated-transformer/). 

Recently, there has been an effort to reduce the time complexity of the self-attention
mechanism and improve performance without sacrificing the quality of results.
Models such as Longformers, Reformers, etc. come to mind. In 2020, a paper titled
[FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)
replaced the self-attention layer in BERT with a Fourier Transform layer. The
magnitude of this was not lost on the research community; replacing a relatively
complicated self-attention layer with a simpler "token mixing" layer resulted in
comparable accuracy and a speed-up during training. This opens up further avenues
of research - can we replace the self-attention layer with any simple "token mixing"
layer and get comparable results?

A couple of points from the paper stood out:
1. The authors claim that FNet is 80% faster than BERT on GPUs and 70% faster on TPUs.
The reason for this speed-up is two-fold:
    a. The Fourier Transform layer is unparametrized, it does not have any parameters!
    b. The authors use Fast Fourier Transform (FFT); this reduces the time complexity
    from O(n^2) (in the case of self-attention) to O(n log n).
2. What's astounding is that FNet still manages to achieve 92-97% of the accuracy
of BERT on the GLUE benchmark.
"""

"""
## Setup

Before we start with the implementation, let's install the KerasNLP library, and
import all the necessary packages.
"""

"""shell
pip install -q keras-nlp
pip install -q tfds-nightly
"""
import keras_nlp
import numpy as np
import pathlib
import random
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

"""
Let's also define our parameters/hyperparameters.
"""
BATCH_SIZE = 64
EPOCHS = 3
MAX_SEQUENCE_LENGTH = 40
VOCAB_SIZE = 15000

EMBED_DIM = 128
INTERMEDIATE_DIM = 512

"""
## Loading the Dataset
Tensorflow Datasets (TFDS) is a library that provides a unified API for working
with datasets that are stored in the TensorFlow format. We will use TFDS to load
the SST-2 dataset.
"""
train_ds, val_ds, test_ds = tfds.load(
    "huggingface:sst", split=["train", "validation", "test"], shuffle_files=True
)

"""
## Preprocessing the Dataset
We need to perform these two steps:
1. Convert the input sentences to lowercase.
2. The label is a sentiment score (float) lying in the range [0, 1]. We will
convert it to either 0 or 1, keeping 0.5 as the threshold. This is a necessary
step because the task we want to solve is a classification task.
"""


def preprocess_dataset(dataset):
    dataset = dataset.map(
        lambda x: {
            "sentence": tf.strings.lower(x["sentence"]),
            "label": tf.cast(x["label"] >= 0.5, dtype=tf.int32),
        }
    )
    return dataset


train_ds = preprocess_dataset(train_ds)
val_ds = preprocess_dataset(val_ds)
test_ds = preprocess_dataset(test_ds)


"""
Let's analyse the train-validation-test split. We'll also print a few samples.
"""
print("Number of Training Examples: ", len(train_ds))
print("Number of Validation Examples: ", len(val_ds))
print("Number of Test Examples: ", len(test_ds))

for element in train_ds.take(5):
    print(element)


"""
### Tokenizing the Data
We'll be using `WordPieceTokenizer` from KerasNLP to tokenize the text.
`WordPieceTokenizer` takes a WordPiece vocabulary and has functions for
tokenizing the text, and detokenizing sequences of tokens.


Before we define the tokenizer, we first need to train it on the dataset
we have. WordPiece Tokenizer is a subword tokenizer; training it on a corpus gives
us a vocabulary of subwords. A subword tokenizer is a compromise between word tokenizers
(word tokenizers have the issue of many OOV tokens), and character tokenizers
(characters don't really encode meaning like words do). Luckily, TensorFlow Text makes it very
simple to train WordPiece on a corpus. Reference: https://www.tensorflow.org/text/guide/subwords_tokenizer

For more details about WordPiece, please visit [this
blog](https://ai.googleblog.com/2021/12/a-fast-wordpiece-tokenization-system.html).

Note: The official implementation of FNet uses the SentencePiece Tokenizer.
"""


def train_word_piece(text_samples, vocab_size, reserved_tokens):
    bert_tokenizer_params = dict(lower_case=True)

    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=vocab_size,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
    vocab = bert_vocab.bert_vocab_from_dataset(
        word_piece_ds.batch(1000).prefetch(2), **bert_vocab_args
    )
    return vocab


"""
Every vocabulary has a few special, reserved tokens. We have four such tokens:
- [PAD] - Padding token. Padding tokens are appended to the input sequence length
when the input sequence length is shorter than the maximum sequence length.
- [UNK] - Unknown token.
- [START] - Token that marks the start of the input sequence.
- [END] - Token that marks the end of the input sequence.
"""
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

train_sentences = [element["sentence"] for element in train_ds]

vocab = train_word_piece(train_sentences, VOCAB_SIZE, reserved_tokens)

"""
Let's see some tokens!
"""
print("Tokens: ", vocab[100:110])

"""
Now, let's define the tokenizer. We will use the vocabulary obtained above as
input to the tokenizers. We will define a maximum sequence length so that
all sequences are padded to the same length, if the length of the sequence is
less than the specified sequence length. Otherwise, the sequence is truncated.
"""
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=False,
    split_pattern=" ",
    sequence_length=MAX_SEQUENCE_LENGTH,
)

"""
Let's try and tokenize a sample from our dataset! To verify whether the text has
been tokenized correctly, we can also detokenize the list of tokens back to the
original text.
"""

for element in train_ds.take(1):
    input_sentence_ex = element["sentence"]
    input_tokens_ex = tokenizer(input_sentence_ex)

    print("Sentence: ", input_sentence_ex)
    print("Tokens: ", input_tokens_ex)
    print("Recovered text after detokenizing: ", tokenizer.detokenize(input_tokens_ex))


"""
## Formatting the Dataset

Next, we'll format our datasets in the form that will be fed to the models.
We need to add [START] and [END] tokens to the input sentences. We also need
to tokenize the text.

"""


def make_dataset(dataset):
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(
        lambda x: ({"input_ids": tokenizer(x["sentence"])}, x["label"])
    )
    return dataset.shuffle(512).prefetch(16).cache()


train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)
test_ds = make_dataset(test_ds)

"""
## Building the Model

Now, let's move on to the exciting part - defining our model!
We first need an Embedding layer, i.e., a vector for every token in our input sequence.
This Embedding layer can be initialised randomly. We also need a Positional
Embedding layer which encodes the word order in the sequence. The convention is
to add these two embeddings. KerasNLP has a `TokenAndPositionEmbedding ` layer
which does all of the above steps for us.

Our FNet classification model consists of an `FNetEncoder` layer with a `Dense`
layer on top. The `FNetEncoder` layer can be used off-the-shelf from KerasNLP!

Note: For FNet, masking the padding tokens has a minimal effect on results. In the
official implementation, the padding tokens are not masked.
"""

input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)

x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)


x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

fnet_classifier = keras.Model(input_ids, outputs, name="fnet_classifier")

"""
## Training our Model

We'll use accuracy to monitor training progress on the validation data.

We will train our model for 3 epochs.
"""
fnet_classifier.summary()
fnet_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
fnet_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

"""
We obtain a train accuracy of around 70% and a validation accuracy of around
71%. Moreover, for 3 epochs, it takes around 9 seconds to train the model (on Colab
with a 16 GB Tesla T4 GPU).

Let's calculate the test accuracy.
"""
fnet_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)

"""
We obtain a test accuracy of around 72%.
"""

"""
## Comparison with Transformer Model

Let's compare our FNet Classifier model with a Transformer Classifier model.
"""
NUM_HEADS = 2
input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")


x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)

x = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
x = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
x = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)


x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

transformer_classifier = keras.Model(input_ids, outputs, name="transformer_classifier")


transformer_classifier.summary()
transformer_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
transformer_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

"""
We obtain a train accuracy of around 78% and a validation accuracy of around
72%. It takes around 14 seconds to train the model.

Let's calculate the test accuracy.
"""
transformer_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)

"""
We obtain a test accuracy of 73.62%.
"""

"""
Let's make a table and compare the two models.

|                         | **FNet Classifier** | **Transformer Classifier** |
|:-----------------------:|:-------------------:|:--------------------------:|
|    **Training Time**    |      9 seconds      |         14 seconds         |
|    **Train Accuracy**   |        70.88%       |           78.20%           |
| **Validation Accuracy** |        70.94%       |           72.39%           |
|    **Test Accuracy**    |        71.99%       |           73.62%           |
"""

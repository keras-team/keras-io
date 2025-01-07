"""
Title: Text Classification using FNet
Author: [Abheesht Sharma](https://github.com/abheesht17/)
Date created: 2022/06/01
Last modified: 2025/01/06
Description: Text Classification on the IMDb Dataset using `keras_hub.layers.FNetEncoder` layer.
Accelerator: GPU
Made Backend agnostic by: [Humbulani Ndou](https://github.com/Humbulani1234)
"""

"""
## Introduction

In this example, we will demonstrate the ability of FNet to achieve comparable
results with a vanilla Transformer model on the text classification task.
We will be using the IMDb dataset, which is a
collection of movie reviews labelled either positive or negative (sentiment
analysis).

To build the tokenizer, model, etc., we will use components from
[KerasHub](https://github.com/keras-team/keras-hub). KerasHub makes life easier
for people who want to build NLP pipelines! :)

### Model

Transformer-based language models (LMs) such as BERT, RoBERTa, XLNet, etc. have
demonstrated the effectiveness of the self-attention mechanism for computing
rich embeddings for input text. However, the self-attention mechanism is an
expensive operation, with a time complexity of `O(n^2)`, where `n` is the number
of tokens in the input. Hence, there has been an effort to reduce the time
complexity of the self-attention mechanism and improve performance without
sacrificing the quality of results.

In 2020, a paper titled
[FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)
replaced the self-attention layer in BERT with a simple Fourier Transform layer
for "token mixing". This resulted in comparable accuracy and a speed-up during
training. In particular, a couple of points from the paper stand out:

* The authors claim that FNet is 80% faster than BERT on GPUs and 70% faster on
TPUs. The reason for this speed-up is two-fold: a) the Fourier Transform layer
is unparametrized, it does not have any parameters, and b) the authors use Fast
Fourier Transform (FFT); this reduces the time complexity from `O(n^2)`
(in the case of self-attention) to `O(n log n)`.
* FNet manages to achieve 92-97% of the accuracy of BERT on the GLUE benchmark.
"""

"""
## Setup

Before we start with the implementation, let's import all the necessary packages.
"""

"""shell
pip install -q --upgrade keras-hub
pip install -q --upgrade keras  # Upgrade to Keras 3.
"""

import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import math

os.environ["KERAS_BACKEND"] = "torch"  # or jax, or tensorflow

import keras_hub
import keras

keras.utils.set_random_seed(42)

"""
Let's also define our hyperparameters.
"""
BATCH_SIZE = 64
EPOCHS = 1  # maybe adjusted as desired
MAX_SEQUENCE_LENGTH = 512
VOCAB_SIZE = 15000

EMBED_DIM = 128
INTERMEDIATE_DIM = 512

"""
## Loading the dataset

First, let's download the IMDB dataset and extract it.
"""

"""shell
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
"""

"""
Samples are present in the form of text files. Let's inspect the structure of
the directory.
"""

print(os.listdir("./aclImdb"))
print(os.listdir("./aclImdb/train"))
print(os.listdir("./aclImdb/test"))

"""
The directory contains two sub-directories: `train` and `test`. Each subdirectory
in turn contains two folders: `pos` and `neg` for positive and negative reviews,
respectively. Before we load the dataset, let's delete the `./aclImdb/train/unsup`
folder since it has unlabelled samples.
"""

"""shell
rm -rf aclImdb/train/unsup
"""

"""
We'll use Pandas to avoid dependency on `keras.utils.text_dataset_from_directory`
since it is merely a wrapper that returns `tf.data.Dataset` and we wish to make
this script backend-agnostic
"""


# Function to read files and label them
def load_data_from_directory(directory, label):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
                data.append((text, label))
    return data


# Function to create Pandas dataframes
def create_df(directory_pos, directory_neg):
    # Load data from both directories
    pos_data = load_data_from_directory(directory_pos, label=1)
    neg_data = load_data_from_directory(directory_neg, label=0)
    # Combine data from both pos and neg
    all_data = pos_data + neg_data
    # Shuffle the data randomly
    random.shuffle(all_data)
    # Create a pandas DataFrame
    df = pd.DataFrame(all_data, columns=["text", "label"])
    return df


# Create the train and test dataframes, iloc maybe adjusted to a desired size
train_df = create_df("./aclImdb/train/pos", "./aclImdb/train/neg").iloc[0:5000]
test_df = create_df("./aclImdb/test/pos", "./aclImdb/test/neg").iloc[0:2500]

# Train and Validation splits, 5% for validation
train_df, val_df = train_test_split(
    train_df, test_size=0.05, stratify=train_df["label"].values, random_state=42
)

# Data statistics
print(f"Total training examples: {len(train_df)}")
print(f"Total validation examples: {len(val_df)}")
print(f"Total test examples: {len(test_df)}")

"""
Let's print a few samples.
"""

for i in range(3):
    print(f"text: {train_df['text'][i]}, label: {train_df['label'][i]}")

"""
### Tokenizing the data

We'll be using the `keras_hub.tokenizers.WordPieceTokenizer` layer to tokenize
the text. `keras_hub.tokenizers.WordPieceTokenizer` takes a WordPiece vocabulary
and has functions for tokenizing the text, and detokenizing sequences of tokens.

Before we define the tokenizer, we first need to train it on the dataset
we have. The WordPiece tokenization algorithm is a subword tokenization algorithm;
training it on a corpus gives us a vocabulary of subwords. A subword tokenizer
is a compromise between word tokenizers (word tokenizers need very large
vocabularies for good coverage of input words), and character tokenizers
(characters don't really encode meaning like words do). Luckily, KerasHub
makes it very simple to train WordPiece on a corpus with the
`keras_hub.tokenizers.compute_word_piece_vocabulary` utility.

Note: The official implementation of FNet uses the SentencePiece Tokenizer.
"""

"""
Every vocabulary has a few special, reserved tokens. We have two such tokens:

- `"[PAD]"` - Padding token. Padding tokens are appended to the input sequence length
when the input sequence length is shorter than the maximum sequence length.
- `"[UNK]"` - Unknown token.
"""

reserved_tokens = ["[PAD]", "[UNK]"]
train_sentences = [element[0] for element in train_df]

vocab = keras_hub.tokenizers.compute_word_piece_vocabulary(
    data=[
        "./aclImdb/train/pos/" + f
        for f in os.listdir("./aclImdb/train/pos")
        if os.path.isfile(os.path.join("./aclImdb/train/pos", f))
    ],  # list of files in the specified directory used to create the vocab
    vocabulary_size=VOCAB_SIZE,
    reserved_tokens=reserved_tokens,
    lowercase=True,
)


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
    lowercase=True,
    sequence_length=MAX_SEQUENCE_LENGTH,
)


"""
Let's try and tokenize a sample from our dataset! To verify whether the text has
been tokenized correctly, we can also detokenize the list of tokens back to the
original text.
"""

input_sentence_ex = train_df["text"][0]
input_tokens_ex = tokenizer(input_sentence_ex)

print("Sentence: ", input_sentence_ex)
print("Tokens: ", input_tokens_ex)
print("Recovered text after detokenizing: ", tokenizer.detokenize(input_tokens_ex))


"""
Create the final datasets, method adapted from `PyDataset` doc string
"""


class UnifiedPyDataset(keras.utils.PyDataset):
    """A Keras-compatible dataset that processes a DataFrame for TensorFlow, JAX, and PyTorch."""

    def __init__(
        self,
        df,
        batch_size=BATCH_SIZE,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=10,
        **kwargs,
    ):
        """
        Args:
            df: pandas DataFrame with data
            batch_size: Batch size for dataset
            workers: Number of workers to use for parallel loading (Keras)
            use_multiprocessing: Whether to use multiprocessing
            max_queue_size: Maximum size of the data queue for parallel loading
        """
        super().__init__(**kwargs)
        self.dataframe = df
        columns = ["text", "label"]

        # text files
        self.text_x = self.dataframe["text"]
        self.text_y = self.dataframe["label"]

        # general
        self.batch_size = batch_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size

    def __getitem__(self, index):
        """
        Fetches a batch of data from the dataset at the given index.

        Args:
            index: position of the batch in the dataset
        """

        # Return x, y for batch idx.
        low = index * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high_text = min(low + self.batch_size, len(self.text_x))
        # text files batches
        batch_text_x = self.text_x[low:high_text]
        batch_text_y = self.text_y[low:high_text]
        # Tokenize the data in dataframe and stack them into an nd.array of axis_0=batch
        text = np.array([[tokenizer(text) for text in batch_text_x]])
        return (
            {
                "input_ids": keras.ops.squeeze(text),
            },
            # Target lables
            np.array(batch_text_y),
        )

    def __len__(self):
        """
        Returns the number of batches in the dataset.
        """
        return math.ceil(len(self.dataframe) / self.batch_size)


def prepare_dataset(dataframe, training=True):
    ds = UnifiedPyDataset(
        dataframe,
        batch_size=32,
        workers=4,
    )
    return ds


train_ds = prepare_dataset(train_df)
val_ds = prepare_dataset(val_df)
test_ds = prepare_dataset(test_df)

"""
## Building the model

Now, let's move on to the exciting part - defining our model!
We first need an embedding layer, i.e., a layer that maps every token in the input
sequence to a vector. This embedding layer can be initialised randomly. We also
need a positional embedding layer which encodes the word order in the sequence.
The convention is to add, i.e., sum, these two embeddings. KerasHub has a
`keras_hub.layers.TokenAndPositionEmbedding ` layer which does all of the above
steps for us.

Our FNet classification model consists of three `keras_hub.layers.FNetEncoder`
layers with a `keras.layers.Dense` layer on top.

Note: For FNet, masking the padding tokens has a minimal effect on results. In the
official implementation, the padding tokens are not masked.
"""

input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

x = keras_hub.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)

x = keras_hub.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_hub.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_hub.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)


x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

fnet_classifier = keras.Model(input_ids, outputs, name="fnet_classifier")

"""
## Training our model

We'll use accuracy to monitor training progress on the validation data. Let's
train our model for 3 epochs.
"""
fnet_classifier.summary()
fnet_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
fnet_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

"""
We obtain a train accuracy of around 92% and a validation accuracy of around
85%. Moreover, for 3 epochs, it takes around 86 seconds to train the model
(on Colab with a 16 GB Tesla T4 GPU).

Let's calculate the test accuracy.
"""
fnet_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)


"""
## Comparison with Transformer model

Let's compare our FNet Classifier model with a Transformer Classifier model. We
keep all the parameters/hyperparameters the same. For example, we use three
`TransformerEncoder` layers.

We set the number of heads to 2.
"""
NUM_HEADS = 2
input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")


x = keras_hub.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)

x = keras_hub.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
x = keras_hub.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
x = keras_hub.layers.TransformerEncoder(
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
We obtain a train accuracy of around 94% and a validation accuracy of around
86.5%. It takes around 146 seconds to train the model (on Colab with a 16 GB Tesla
T4 GPU).

Let's calculate the test accuracy.
"""
transformer_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)

"""
Let's make a table and compare the two models. We can see that FNet
significantly speeds up our run time (1.7x), with only a small sacrifice in
overall accuracy (drop of 0.75%).

|                         | **FNet Classifier** | **Transformer Classifier** |
|:-----------------------:|:-------------------:|:--------------------------:|
|    **Training Time**    |      86 seconds     |         146 seconds        |
|    **Train Accuracy**   |        92.34%       |           93.85%           |
| **Validation Accuracy** |        85.21%       |           86.42%           |
|    **Test Accuracy**    |        83.94%       |           84.69%           |
|       **#Params**       |      2,321,921      |          2,520,065         |
"""

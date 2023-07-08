"""
Title: Abstractive Text Summarization with BART
Author: [Abheesht Sharma](https://github.com/abheesht17/)
Date created: 2023/07/08
Last modified: 2023/07/08
Description: Use KerasNLP to fine-tune BART on the abstractive summarization task.
Accelerator: GPU
"""

"""
## Introduction

In the era of information overload, it has become crucial to extract the crux
of a long document or a conversation and express it in a few sentences. Owing
to the fact that summarization has widespread applications in different domains,
it has become a key, well-studied NLP task in recent years.

[Bidirectional Autoregressive Transformer (BART)](https://arxiv.org/abs/1910.13461)
is a Transformer-based encoder-decoder model, often used for
sequence-to-sequence tasks like summarization and neural machine translation.
BART is pre-trained in a self-supervised fashion on a large text corpus. During
pre-training, the text is corrupted and BART is trained to reconstruct the
original text (hence called a "denoising autoencoder"). Some pre-training tasks
include token masking, token deletion, sentence permutation (shuffle sentences
and train BART to fix the order), etc.

In this example, we will demonstrate how to fine-tune BART on the abstractive
summarization task using KerasNLP, generate summaries using the fine-tuned
model, and evaluate the summaries using ROUGE score.
"""

"""
## Setup

Before we start implementing the pipeline, let's install and import all the
libraries we need. We'll be using the KerasNLP library. We will also need a
couple of utility libraries.
"""

"""shell
pip install keras-nlp -q
pip install py7zr -q
pip install gdown -q
"""

import gdown
import py7zr

import keras_nlp
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras

"""
Let's also define our hyperparameters.
"""

BATCH_SIZE = 16
NUM_BATCHES = 500
EPOCHS = 1  # Can be set to a higher value for better results
MAX_ENCODER_SEQUENCE_LENGTH = 512
MAX_DECODER_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 200

"""
## Dataset

Let's load the [SAMSum dataset](https://arxiv.org/abs/1911.12237). This dataset
contains around 15,000 pairs of conversations/dialogues and summaries.
"""

# Download the dataset.
gdown.download(
    "https://drive.google.com/uc?id=1u4L2tgfbd5RLYPws47LTE9GeIJ6bZA9g",
    "./corpus.7z",
)

# Extract the `.7z` file.
with py7zr.SevenZipFile("./corpus.7z", mode="r") as z:
    z.extractall(path="/root/tensorflow_datasets/downloads/manual")

# Load data using TFDS.
samsum_ds = tfds.load("samsum", split="train", as_supervised=True)

"""
The dataset has two fields: `dialogue` and `summary`. Let's see a sample.
"""
for dialogue, summary in samsum_ds:
    print(dialogue.numpy())
    print(summary.numpy())
    break

"""
We'll now batch the dataset and retain only a subset of the dataset for the
purpose of this example. The dialogue is fed to the encoder, and the
corresponding summary serves as input to the decoder. We will, therefore, change
the format of the dataset to a dictionary having two keys: `"encoder_text"` and
`"decoder_text"`.This is how `keras_nlp.models.BartSeq2SeqLMPreprocessor`
expects the input format to be.
"""

train_ds = (
    samsum_ds.map(
        lambda dialogue, summary: {"encoder_text": dialogue, "decoder_text": summary}
    )
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
train_ds = train_ds.take(NUM_BATCHES)

"""
## Fine-tune BART

Let's load the model and preprocessor first. We use sequence lengths of 512
and 128 for the encoder and decoder, respectively, instead of 1024 (which is the
default sequence length). This will allow us to run this example quickly
on Colab.

If you observe carefully, the preprocessor is attached to the model. What this
means is that we don't have to worry about preprocessing the text inputs;
everything will be done internally. The preprocessor tokenizes the encoder text
and the decoder text, adds special tokens and pads them. To generate labels
for auto-regressive training, the preprocessor shifts the decoder text one
position to the right.
"""

preprocessor = keras_nlp.models.BartSeq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)
bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset(
    "bart_base_en", preprocessor=preprocessor
)

bart_lm.summary()

"""
Define the optimizer and loss. We use the Adam optimizer with a linearly
decaying learning rate. Compile the model.
"""

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
    epsilon=1e-6,
    global_clipnorm=1.0,  # Gradient clipping.
)
# Exclude layernorm and bias terms from weight decay.
optimizer.exclude_from_weight_decay(var_names=["bias"])
optimizer.exclude_from_weight_decay(var_names=["gamma"])
optimizer.exclude_from_weight_decay(var_names=["beta"])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bart_lm.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=["accuracy"],
)

"""
Let's train the model!
"""

bart_lm.fit(train_ds, epochs=EPOCHS)

"""
## Generate summaries and evaluate them!

Now that the model has been trained, let's get to the fun part - actually
generating summaries! Let's pick the first 10 samples from the validation set
and generate summaries for them.
"""

val_ds = tfds.load("samsum", split="validation", as_supervised=True)

dialogues = []
summaries = []
for dialogue, summary in val_ds.take(10):
    dialogues.append(dialogue.numpy().decode("utf-8"))
    summaries.append(summary.numpy().decode("utf-8"))

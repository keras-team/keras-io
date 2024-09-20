"""
Title: Abstractive Text Summarization with BART
Author: [Abheesht Sharma](https://github.com/abheesht17/)
Date created: 2023/07/08
Last modified: 2024/03/20
Description: Use KerasHub to fine-tune BART on the abstractive summarization task.
Accelerator: GPU
Converted to Keras 3 by: [Sitam Meur](https://github.com/sitamgithub-MSIT)
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
summarization task (on conversations!) using KerasHub, and generate summaries
using the fine-tuned model.
"""

"""
## Setup

Before we start implementing the pipeline, let's install and import all the
libraries we need. We'll be using the KerasHub library. We will also need a
couple of utility libraries.
"""

"""shell
pip install git+https://github.com/keras-team/keras-hub.git py7zr -q
"""

"""
This examples uses [Keras 3](https://keras.io/keras_3/) to work in any of
`"tensorflow"`, `"jax"` or `"torch"`. Support for Keras 3 is baked into
KerasHub, simply change the `"KERAS_BACKEND"` environment variable to select
the backend of your choice. We select the JAX backend below.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

"""
Import all necessary libraries.
"""

import py7zr
import time

import keras_hub
import keras
import tensorflow as tf
import tensorflow_datasets as tfds

"""
Let's also define our hyperparameters.
"""

BATCH_SIZE = 8
NUM_BATCHES = 600
EPOCHS = 1  # Can be set to a higher value for better results
MAX_ENCODER_SEQUENCE_LENGTH = 512
MAX_DECODER_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 40

"""
## Dataset

Let's load the [SAMSum dataset](https://arxiv.org/abs/1911.12237). This dataset
contains around 15,000 pairs of conversations/dialogues and summaries.
"""

# Download the dataset.
filename = keras.utils.get_file(
    "corpus.7z",
    origin="https://huggingface.co/datasets/samsum/resolve/main/data/corpus.7z",
)

# Extract the `.7z` file.
with py7zr.SevenZipFile(filename, mode="r") as z:
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
`"decoder_text"`.This is how `keras_hub.models.BartSeq2SeqLMPreprocessor`
expects the input format to be.
"""

train_ds = (
    samsum_ds.map(
        lambda dialogue, summary: {"encoder_text": dialogue, "decoder_text": summary}
    )
    .batch(BATCH_SIZE)
    .cache()
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
position to the right. This is done because at every timestep, the model is
trained to predict the next token.
"""

preprocessor = keras_hub.models.BartSeq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset(
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
generating summaries! Let's pick the first 100 samples from the validation set
and generate summaries for them. We will use the default decoding strategy, i.e.,
greedy search.

Generation in KerasHub is highly optimized. It is backed by the power of XLA.
Secondly, key/value tensors in the self-attention layer and cross-attention layer
in the decoder are cached to avoid recomputation at every timestep.
"""


def generate_text(model, input_text, max_length=200, print_time_taken=False):
    start = time.time()
    output = model.generate(input_text, max_length=max_length)
    end = time.time()
    print(f"Total Time Elapsed: {end - start:.2f}s")
    return output


# Load the dataset.
val_ds = tfds.load("samsum", split="validation", as_supervised=True)
val_ds = val_ds.take(100)

dialogues = []
ground_truth_summaries = []
for dialogue, summary in val_ds:
    dialogues.append(dialogue.numpy())
    ground_truth_summaries.append(summary.numpy())

# Let's make a dummy call - the first call to XLA generally takes a bit longer.
_ = generate_text(bart_lm, "sample text", max_length=MAX_GENERATION_LENGTH)

# Generate summaries.
generated_summaries = generate_text(
    bart_lm,
    val_ds.map(lambda dialogue, _: dialogue).batch(8),
    max_length=MAX_GENERATION_LENGTH,
    print_time_taken=True,
)

"""
Let's see some of the summaries.
"""
for dialogue, generated_summary, ground_truth_summary in zip(
    dialogues[:5], generated_summaries[:5], ground_truth_summaries[:5]
):
    print("Dialogue:", dialogue)
    print("Generated Summary:", generated_summary)
    print("Ground Truth Summary:", ground_truth_summary)
    print("=============================")

"""
The generated summaries look awesome! Not bad for a model trained only for 1
epoch and on 5000 examples :)
"""

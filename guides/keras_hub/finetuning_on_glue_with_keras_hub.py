"""
Title: Fine-tuning KerasHub Models on GLUE
Author: [bitanb1999](https://github.com/bitanb1999)
Date created: 2026/07/24
Last modified: 2026/07/24
Description: Fine-tune a pretrained KerasHub text classifier on a GLUE task.
Accelerator: GPU
"""

"""
The [General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/)
benchmark is a collection of nine sentence- and sentence-pair language
understanding tasks, widely used to evaluate how well a pretrained language
model transfers to a variety of downstream problems.

KerasHub makes it simple to fine-tune any of its pretrained text
classification backbones (BERT, RoBERTa, DeBERTaV3, and more) on a GLUE task
with just a few lines of code. This guide walks through fine-tuning
`keras_hub.models.BertTextClassifier` on **MRPC** (Microsoft Research
Paraphrase Corpus), a GLUE task where the goal is to predict whether two
sentences are semantically equivalent.

KerasHub also ships a standalone benchmark script at
[`benchmarks/glue.py`](https://github.com/keras-team/keras-hub/blob/master/benchmarks/glue.py)
that automates the workflow shown in this guide across models and presets
from the command line. This guide covers the same workflow interactively, so
you understand each step before running the full script.
"""

"""shell
!pip install -q --upgrade keras-hub
!pip install -q --upgrade keras  # Upgrade to Keras 3.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]

import keras
import tensorflow as tf
import tensorflow_datasets as tfds

import keras_hub

"""
## Load the MRPC dataset

GLUE tasks are available through
[`tensorflow_datasets`](https://www.tensorflow.org/datasets/catalog/glue).
Each MRPC example is a pair of sentences (`sentence1`, `sentence2`) along with
a binary `label` indicating whether the sentences are paraphrases of one
another.
"""

train_ds, validation_ds = tfds.load(
    "glue/mrpc",
    split=["train", "validation"],
)

"""
Let's take a look at a single example. KerasHub text classifiers can consume
sentence pairs directly as a tuple of `(sentence1, sentence2)`, so we just
need to reshape the dictionary format that `tfds` gives us.
"""


def split_features(x):
    features = (x["sentence1"], x["sentence2"])
    label = x["label"]
    return (features, label)


train_ds = train_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
validation_ds = validation_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)

for features, label in train_ds.take(1):
    print(features)
    print(label)

"""
## Fine-tune a `BertTextClassifier`

The highest level module in KerasHub for this task is
`keras_hub.models.BertTextClassifier`, a `keras.Model` that bundles a
pretrained BERT backbone with a classification head and its own text
preprocessing layer. Passing `num_classes=2` builds a two-way classification
head suited to MRPC's binary label.

We use `bert_tiny_en_uncased` here so this guide runs quickly end-to-end;
swap in a larger preset such as `bert_base_en_uncased` for better accuracy.
"""

BATCH_SIZE = 32
EPOCHS = 3

classifier = keras_hub.models.BertTextClassifier.from_preset(
    "bert_tiny_en_uncased",
    num_classes=2,
    activation="softmax",
)

"""
Because the classifier carries its own preprocessor, we can pass raw text
straight to `fit()` -- there's no need for a separate tokenization step.
"""

train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

classifier.compile(
    optimizer=keras.optimizers.AdamW(5e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
classifier.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
)

"""
## Evaluate and predict

`evaluate()` reports the fine-tuned model's loss and accuracy on held-out
data, and `predict()` returns class probabilities for new sentence pairs.
"""

classifier.evaluate(validation_ds)

sentence_pairs = (
    ["Sam ate an apple.", "Sam ate a fruit."],
    ["Sam ate an apple.", "The stock market fell today."],
)
probabilities = classifier.predict(sentence_pairs)
print(probabilities)

"""
## Next steps

This guide fine-tuned `BertTextClassifier` on MRPC, but the same workflow
applies to any KerasHub text classifier and any GLUE task available through
`tensorflow_datasets` -- just point `tfds.load()` at a different
`"glue/<task>"` config and adjust `num_classes` if the task isn't binary.

To sweep over models, presets, and hyperparameters from the command line
instead, see the
[`benchmarks/glue.py`](https://github.com/keras-team/keras-hub/blob/master/benchmarks/glue.py)
script in the KerasHub repository, which implements this same fine-tuning
loop as a configurable benchmark.
"""

"""
Title: Data Parallel Training with KerasNLP
Author: Anshuman Mishra
Date created: 2023/06/04
Last modified: 2023/06/05
Description: This tutorial demonstrates how to perform distributed training using KerasNLP. 
It covers the steps for training a BERT-based masked language model on the wikitext-2 
dataset, using multiple GPUs for accelerated training.

Accelerator: GPU
"""

"""
## Introduction

Distributed training is a technique used to train deep learning models on multiple devices 
or machines simultaneously. It helps to reduce training time and improve model performance 
by leveraging the computational power of multiple resources. KerasNLP is a library that 
provides tools and utilities for natural language processing tasks, including distributed 
training.

In this tutorial, we will use KerasNLP to train a BERT-based masked language model (MLM) 
on the wikitext-2 dataset. The MLM task involves predicting the masked words in a sentence, 
which helps the model learn contextual representations of words. 

This guide focuses on data parallelism, in particular synchronous data parallelism, where 
the different replicas of the model stay in sync after each batch they process. 
Synchronicity keeps the model convergence behavior identical to what you would see for 
single-device training.

Specifically, this guide teaches you how to use the `tf.distribute` API to train Keras
models on multiple GPUs, with minimal changes to your code, in the following two setups:

- On multiple GPUs (typically 2 to 8) installed on a single machine (single host,
multi-device training). This is the most common setup for researchers and small-scale
industry workflows.
- On a cluster of many machines, each hosting one or multiple GPUs (multi-worker
distributed training). This is a good setup for large-scale industry workflows, e.g.
training high-resolution image classification models on tens of millions of images using
20-100 GPUs.

## Setup

The tutorial relies on KerasNLP 0.5.2. Additionally, we need
at least TensorFlow 2.11 in order to use AdamW with mixed precision.
"""

"""shell
pip install keras-nlp==0.5.2 -q
pip install -U tensorflow -q
"""

"""
## Imports
"""

import os
import tensorflow as tf
from tensorflow import keras
import keras_nlp

"""
A simple thing we can do right off the bat is to enable
[mixed precision](https://keras.io/api/mixed_precision/), which will speed up training by
running most of our computations with 16 bit (instead of 32 bit) floating point numbers.
"""

policy = keras.mixed_precision.Policy("mixed_float16")
keras.mixed_precision.set_global_policy(policy)


PRETRAINING_BATCH_SIZE = 128
EPOCHS = 5

"""
First, we need to download and preprocess the wikitext-2 dataset. This dataset will be 
used for pretraining the BERT model. We will filter out short lines to ensure that the 
data has enough context for training.
"""

keras.utils.get_file(
    origin="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
    extract=True,
)
wiki_dir = os.path.expanduser("~/.keras/datasets/wikitext-2/")

# Load wikitext-103 and filter out short lines.
wiki_train_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.train.tokens")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)
wiki_val_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.valid.token")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)
wiki_test_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.test.tokens")
    .filter(lambda x: tf.strings.length(x) > 100)
    .batch(PRETRAINING_BATCH_SIZE)
)

"""
In the above code, we download the wikitext-2 dataset and extract it. Then, we define 
three datasets: wiki_train_ds, wiki_val_ds, and wiki_test_ds. These datasets are 
filtered to remove short lines and are batched for efficient training.

Let's define a function for decaying the learning rate. You can define any decay function you need.
We also define a callback for printing the learning rate at the end of each epoch.
"""


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            "\nLearning rate for epoch {} is {}".format(
                epoch + 1, model_dist.optimizer.lr.numpy()
            )
        )


"""
Let's also make a callback to TensorBoard, this will enable visualization of different metrics
while we train the model in later part of this tutorial We put all the callbacks together as follows:
"""
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir="./logs"),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR(),
]


print(tf.config.list_physical_devices("GPU"))

"""
To do single-host, multi-device synchronous training with a Keras model, you would use
the [`tf.distribute.MirroredStrategy` API](
    https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy).
Here's how it works:

- Instantiate a `MirroredStrategy`, optionally configuring which specific devices you
want to use (by default the strategy will use all GPUs available).
- Use the strategy object to open a scope, and within this scope, create all the Keras
objects you need that contain variables. Typically, that means **creating & compiling the
model** inside the distribution scope.
- Train the model via `fit()` as usual.
"""

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

"""
With the datasets prepared, we now initialize and compile our model and optimizer within
the `strategy.scope()`:
"""
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model_dist = keras_nlp.models.BertMaskedLM.from_preset("bert_tiny_en_uncased")
    model_dist.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        weighted_metrics=keras.metrics.SparseCategoricalAccuracy(),
    )

"""
Let's train our model
"""
model_dist.fit(
    wiki_train_ds, validation_data=wiki_val_ds, epochs=EPOCHS, callbacks=callbacks
)

model_dist.evaluate(wiki_test_ds)

"""
For distributed training across multiple machines (as opposed to training that only leverages
multiple devices on a single machine), there are two distribution strategies you
could use: `MultiWorkerMirroredStrategy` and `ParameterServerStrategy`:

- `tf.distribute.MultiWorkerMirroredStrategy` implements a synchronous CPU/GPU
multi-worker solution to work with Keras-style model building and training loop,
using synchronous reduction of gradients across the replicas.
- `tf.distribute.experimental.ParameterServerStrategy` implements an asynchronous CPU/GPU
multi-worker solution, where the parameters are stored on parameter servers, and
workers update the gradients to parameter servers asynchronously.

### Further reading


1. [TensorFlow distributed training guide](
    https://www.tensorflow.org/guide/distributed_training)
2. [Tutorial on multi-worker training with Keras](
    https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)
3. [MirroredStrategy docs](
    https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
4. [MultiWorkerMirroredStrategy docs](
    https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy)
5. [Distributed training in tf.keras with Weights & Biases](
    https://towardsdatascience.com/distributed-training-in-tf-keras-with-w-b-ccf021f9322e)
    
"""

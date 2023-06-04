
"""
Title: Distributed Training with KerasNLP
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
"""

# Define a function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# Define a callback for printing the learning rate at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model_dist.optimizer.lr.numpy()))

# Put all the callbacks together.
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

print(tf.config.list_physical_devices('GPU'))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():    
    model_dist = keras_nlp.models.BertMaskedLM.from_preset("bert_tiny_en_uncased") 
    model_dist.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        weighted_metrics=keras.metrics.SparseCategoricalAccuracy(),
    )

model_dist.fit(wiki_train_ds, validation_data = wiki_val_ds , epochs=EPOCHS, callbacks=callbacks)

model_dist.evaluate(wiki_test_ds)
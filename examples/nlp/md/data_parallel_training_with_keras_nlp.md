# Data Parallel Training with KerasNLP

**Author:** Anshuman Mishra<br>
**Date created:** 2023/07/07<br>
**Last modified:** 2023/07/07<br>
**Description:** Data Parallel training with KerasNLP.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/data_parallel_training_with_keras_nlp.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/data_parallel_training_with_keras_nlp.py)



---
## Introduction

Distributed training is a technique used to train deep learning models on multiple devices
or machines simultaneously. It helps to reduce training time and allows for training larger
models with more data.KerasNLP is a library that provides tools and utilities for natural
language processing tasks, including distributed training.

In this tutorial, we will use KerasNLP to train a BERT-based masked language model (MLM)
on the wikitext-2 dataset (a 2 million word dataset of wikipedia articles). The MLM task
involves predicting the masked words in a sentence, which helps the model learn contextual
representations of words.

This guide focuses on data parallelism, in particular synchronous data parallelism, where
each accelerator (a GPU or TPU) holds a complete replica of the model, and sees a
different partial batch of the input data. Partial gradients are computed on each device,
aggregated, and used to compute a global gradient update.

Specifically, this guide teaches you how to use the `tf.distribute` API to train Keras
models on multiple GPUs, with minimal changes to your code, in the following two setups:

- On multiple GPUs (typically 2 to 8) installed on a single machine (single host,
multi-device training). This is the most common setup for researchers and small-scale
industry workflows.
- On a cluster of many machines, each hosting one or multiple GPUs (multi-worker
distributed training). This is a good setup for large-scale industry workflows, e.g.
training high-resolution text summarization models on billion word datasets on 20-100 GPUs.

---
## Setup

The tutorial relies on KerasNLP 0.5.2. Additionally, we need
at least TensorFlow 2.11 in order to use AdamW with mixed precision.


```python
!pip install -U -q tensorflow keras-nlp tensorflow_datasets datasets
```

---
## Imports


```python
import os
import tensorflow as tf
from tensorflow import keras
import keras_nlp
```

Before we start any training, let's configure our single GPU to show up as two logical
devices.

When you are training with two or more phsyical GPUs, this is totally uncessary. This
is just a trick to show real distributed training on the default colab GPU runtime,
which has only one GPU availabe.


```python
!nvidia-smi --query-gpu=memory.total --format=csv,noheader
```

```python
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_logical_device_configuration(
    physical_devices[0],
    [
        tf.config.LogicalDeviceConfiguration(memory_limit=15360 // 2),
        tf.config.LogicalDeviceConfiguration(memory_limit=15360 // 2),
    ],
)

logical_devices = tf.config.list_logical_devices("GPU")
logical_devices


PRETRAINING_BATCH_SIZE = 128
```
<div class="k-default-codeblock">
```
15360 MiB

```
</div>
First, we need to download and preprocess the wikitext-2 dataset. This dataset will be
used for pretraining the BERT model. We will filter out short lines to ensure that the
data has enough context for training.


```python
keras.utils.get_file(
    origin="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
    extract=True,
)
wiki_dir = os.path.expanduser("~/.keras/datasets/wikitext-2/")

# Load wikitext-103 and filter out short lines.
wiki_train_ds = (
    tf.data.TextLineDataset(
        wiki_dir + "wiki.train.tokens",
    )
    .filter(lambda x: tf.strings.length(x) > 100)
    .shuffle(buffer_size=500)
    .batch(PRETRAINING_BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
wiki_val_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.valid.tokens")
    .filter(lambda x: tf.strings.length(x) > 100)
    .shuffle(buffer_size=500)
    .batch(PRETRAINING_BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
wiki_test_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.test.tokens")
    .filter(lambda x: tf.strings.length(x) > 100)
    .shuffle(buffer_size=500)
    .batch(PRETRAINING_BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
```

In the above code, we download the wikitext-2 dataset and extract it. Then, we define
three datasets: wiki_train_ds, wiki_val_ds, and wiki_test_ds. These datasets are
filtered to remove short lines and are batched for efficient training.


```python
EPOCHS = 3
```

It's a common practice to use a decayed learning rate in NLP training/tuning. We'll
use `PolynomialDecay` schedule here.


```python
total_training_steps = sum(1 for _ in wiki_train_ds.as_numpy_iterator()) * EPOCHS
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=5e-5,
    decay_steps=total_training_steps,
    end_learning_rate=0.0,
)


class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"\nLearning rate for epoch {epoch + 1} is {model_dist.optimizer.lr.numpy()}"
        )

```

Let's also make a callback to TensorBoard, this will enable visualization of different
metrics while we train the model in later part of this tutorial We put all the callbacks
together as follows:


```python
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir="./logs"),
    PrintLR(),
]


print(tf.config.list_physical_devices("GPU"))
```

<div class="k-default-codeblock">
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

```
</div>
To do single-host, multi-device synchronous training with a Keras model, you would use
the `tf.distribute.MirroredStrategy` API. Here's how it works:

- Instantiate a `MirroredStrategy`, optionally configuring which specific devices you
want to use (by default the strategy will use all GPUs available).
- Use the strategy object to open a scope, and within this scope, create all the Keras
objects you need that contain variables. Typically, that means **creating & compiling the
model** inside the distribution scope.
- Train the model via `fit()` as usual.


```python
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
```

<div class="k-default-codeblock">
```
Number of devices: 2

```
</div>
With the datasets prepared, we now initialize and compile our model and optimizer within
the `strategy.scope()`:


```python
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model_dist = keras_nlp.models.BertMaskedLM.from_preset("bert_tiny_en_uncased")
    model_dist.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.AdamW(lr_schedule),
        weighted_metrics=keras.metrics.SparseCategoricalAccuracy(),
    )

    model_dist.fit(
        wiki_train_ds, validation_data=wiki_val_ds, epochs=EPOCHS, callbacks=callbacks
    )
```

<div class="k-default-codeblock">
```
Epoch 1/3

WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['pooled_dense/kernel:0', 'pooled_dense/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?

    120/Unknown - 90s 568ms/step - loss: 2.0205 - sparse_categorical_accuracy: 0.0415
Learning rate for epoch 1 is 3.33333300659433e-05
120/120 [==============================] - 103s 677ms/step - loss: 2.0205 - sparse_categorical_accuracy: 0.0415 - val_loss: 1.8596 - val_sparse_categorical_accuracy: 0.0940
Epoch 2/3
120/120 [==============================] - ETA: 0s - loss: 1.8300 - sparse_categorical_accuracy: 0.0886
Learning rate for epoch 2 is 1.680555214988999e-05
120/120 [==============================] - 77s 643ms/step - loss: 1.8300 - sparse_categorical_accuracy: 0.0886 - val_loss: 1.7412 - val_sparse_categorical_accuracy: 0.1829
Epoch 3/3
120/120 [==============================] - ETA: 0s - loss: 1.7734 - sparse_categorical_accuracy: 0.1241
Learning rate for epoch 3 is 1.388877564068025e-07
120/120 [==============================] - 76s 633ms/step - loss: 1.7734 - sparse_categorical_accuracy: 0.1241 - val_loss: 1.7002 - val_sparse_categorical_accuracy: 0.2075

```
</div>
After fitting our model under the scope, we evaluate it normally!


```python
model_dist.evaluate(wiki_test_ds)
```

<div class="k-default-codeblock">
```
15/15 [==============================] - 8s 292ms/step - loss: 1.7406 - sparse_categorical_accuracy: 0.2115

[1.7405692338943481, 0.21145851910114288]

```
</div>
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

1. [TensorFlow distributed training guide](https://www.tensorflow.org/guide/distributed_training)
2. [Tutorial on multi-worker training with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)
3. [MirroredStrategy docs](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
4. [MultiWorkerMirroredStrategy docs](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy)
5. [Distributed training in tf.keras with Weights & Biases](https://towardsdatascience.com/distributed-training-in-tf-keras-with-w-b-ccf021f9322e)

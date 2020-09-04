# Multi-GPU and distributed training

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/04/28<br>
**Last modified:** 2020/04/29<br>
**Description:** Guide to multi-GPU & distributed training for Keras models.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/distributed_training.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/distributed_training.py)



---
## Introduction

There are generally two ways to distribute computation across multiple devices:

**Data parallelism**, where a single model gets replicated on multiple devices or
multiple machines. Each of them processes different batches of data, then they merge
their results. There exist many variants of this setup, that differ in how the different
model replicas merge results, in whether they stay in sync at every batch or whether they
are more loosely coupled, etc.

**Model parallelism**, where different parts of a single model run on different devices,
processing a single batch of data together. This works best with models that have a
naturally-parallel architecture, such as models that feature multiple branches.

This guide focuses on data parallelism, in particular **synchronous data parallelism**,
where the different replicas of the model stay in sync after each batch they process.
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


---
## Setup


```python
import tensorflow as tf
from tensorflow import keras
```

---
## Single-host, multi-device synchronous training

In this setup, you have one machine with several GPUs on it (typically 2 to 8). Each
device will run a copy of your model (called a **replica**). For simplicity, in what
follows, we'll assume we're dealing with 8 GPUs, at no loss of generality.

**How it works**

At each step of training:

- The current batch of data (called **global batch**) is split into 8 different
sub-batches (called **local batches**). For instance, if the global batch has 512
samples, each of the 8 local batches will have 64 samples.
- Each of the 8 replicas independently processes a local batch: they run a forward pass,
then a backward pass, outputting the gradient of the weights with respect to the loss of
the model on the local batch.
- The weight updates originating from local gradients are efficiently merged across the 8
replicas. Because this is done at the end of every step, the replicas always stay in
sync.

In practice, the process of synchronously updating the weights of the model replicas is
handled at the level of each individual weight variable. This is done through a **mirrored
variable** object.

**How to use it**

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

Importantly, we recommend that you use `tf.data.Dataset` objects to load data
in a multi-device or distributed workflow.

Schematically, it looks like this:

```python
# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.
  model = Model(...)
  model.compile(...)

# Train the model on all available devices.
model.fit(train_dataset, validation_data=val_dataset, ...)

# Test the model on all available devices.
model.evaluate(test_dataset)
```

Here's a simple end-to-end runnable example:



```python

def get_compiled_model():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_dataset():
    batch_size = 32
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = get_compiled_model()

# Train the model on all available devices.
train_dataset, val_dataset, test_dataset = get_dataset()
model.fit(train_dataset, epochs=2, validation_data=val_dataset)

# Test the model on all available devices.
model.evaluate(test_dataset)
```

<div class="k-default-codeblock">
```
WARNING: Logging before flag parsing goes to stderr.
W0829 16:54:57.025418 4592479680 cross_device_ops.py:1115] There are non-GPU devices in `tf.distribute.Strategy`, not using nccl allreduce.

Number of devices: 1
Epoch 1/2
1563/1563 [==============================] - 3s 2ms/step - loss: 0.3767 - sparse_categorical_accuracy: 0.8889 - val_loss: 0.1257 - val_sparse_categorical_accuracy: 0.9623
Epoch 2/2
1563/1563 [==============================] - 2s 2ms/step - loss: 0.1053 - sparse_categorical_accuracy: 0.9678 - val_loss: 0.0944 - val_sparse_categorical_accuracy: 0.9710
313/313 [==============================] - 0s 779us/step - loss: 0.0900 - sparse_categorical_accuracy: 0.9723

[0.08995261788368225, 0.9722999930381775]

```
</div>
---
## Using callbacks to ensure fault tolerance

When using distributed training, you should always make sure you have a strategy to
recover from failure (fault tolerance). The simplest way to handle this is to pass
`ModelCheckpoint` callback to `fit()`, to save your model
at regular intervals (e.g. every 100 batches or every epoch). You can then restart
training from your saved model.

Here's a simple example:


```python
import os
from tensorflow import keras

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


def run_training(epochs=1):
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()

    # Open a strategy scope and create/restore the model
    with strategy.scope():
        model = make_or_restore_model()

    callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        )
    ]
    model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=2,
    )


# Running the first time creates the model
run_training(epochs=1)

# Calling the same function again will resume from where we left off
run_training(epochs=1)
```

<div class="k-default-codeblock">
```
W0829 16:55:03.609519 4592479680 cross_device_ops.py:1115] There are non-GPU devices in `tf.distribute.Strategy`, not using nccl allreduce.

Creating a new model

W0829 16:55:03.708506 4592479680 callbacks.py:1270] Automatic model reloading for interrupted job was removed from the `ModelCheckpoint` callback in multi-worker mode, please use the `keras.callbacks.experimental.BackupAndRestore` callback instead. See this tutorial for details: https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#backupandrestore_callback.

1563/1563 - 4s - loss: 0.2242 - sparse_categorical_accuracy: 0.9321 - val_loss: 0.1243 - val_sparse_categorical_accuracy: 0.9647

W0829 16:55:07.981292 4592479680 cross_device_ops.py:1115] There are non-GPU devices in `tf.distribute.Strategy`, not using nccl allreduce.

Restoring from ./ckpt/ckpt-1

W0829 16:55:08.245935 4592479680 callbacks.py:1270] Automatic model reloading for interrupted job was removed from the `ModelCheckpoint` callback in multi-worker mode, please use the `keras.callbacks.experimental.BackupAndRestore` callback instead. See this tutorial for details: https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#backupandrestore_callback.

1563/1563 - 4s - loss: 0.0948 - sparse_categorical_accuracy: 0.9709 - val_loss: 0.1006 - val_sparse_categorical_accuracy: 0.9699

```
</div>
---
## `tf.data` performance tips

When doing distributed training, the efficiency with which you load data can often become
critical. Here are a few tips to make sure your `tf.data` pipelines
run as fast as possible.

**Note about dataset batching**

When creating your dataset, make sure it is batched with the global batch size.
For instance, if each of your 8 GPUs is capable of running a batch of 64 samples, you
call use a global batch size of 512.

**Calling `dataset.cache()`**

If you call `.cache()` on a dataset, its data will be cached after running through the
first iteration over the data. Every subsequent iteration will use the cached data. The
cache can be in memory (default) or to a local file you specify.

This can improve performance when:

- Your data is not expected to change from iteration to iteration
- You are reading data from a remote distributed filesystem
- You are reading data from local disk, but your data would fit in memory and your
workflow is significantly IO-bound (e.g. reading & decoding image files).

**Calling `dataset.prefetch(buffer_size)`**

You should almost always call `.prefetch(buffer_size)` after creating a dataset. It means
your data pipeline will run asynchronously from your model,
with new samples being preprocessed and stored in a buffer while the current batch
samples are used to train the model. The next batch will be prefetched in GPU memory by
the time the current batch is over.

---
## Multi-worker distributed synchronous training

**How it works**

In this setup, you have multiple machines (called **workers**), each with one or several
GPUs on them. Much like what happens for single-host training,
each available GPU will run one model replica, and the value of the variables of each
replica is kept in sync after each batch.

Importantly, the current implementation assumes that all workers have the same number of
GPUs (homogeneous cluster).

**How to use it**

1. Set up a cluster (we provide pointers below).
2. Set up an appropriate `TF_CONFIG` environment variable on each worker. This tells the
worker what its role is and how to communicate with its peers.
3. On each worker, run your model construction & compilation code within the scope of a
[`MultiWorkerMirroredStrategy` object](
    https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy),
similarly to we did for single-host training.
4. Run evaluation code on a designated evaluator machine.

**Setting up a cluster**

First, set up a cluster (collective of machines). Each machine individually should be
setup so as to be able to run your model (typically, each machine will run the same
Docker image) and to able to access your data source (e.g. GCS).

Cluster management is beyond the scope of this guide.
[Here is a document](
    https://cloud.google.com/ai-platform/training/docs/distributed-training-containers)
to help you get started.
You can also take a look at [Kubeflow](https://www.kubeflow.org/).

**Setting up the `TF_CONFIG` environment variable**

While the code running on each worker is almost the same as the code used in the
single-host workflow (except with a different `tf.distribute` strategy object), one
significant difference between the single-host workflow and the multi-worker workflow is
that you need to set a `TF_CONFIG` environment variable on each machine running in your
cluster.

The `TF_CONFIG` environment variable is a JSON string that specifies:

- The cluster configuration, while the list of addresses & ports of the machines that
make up the cluster
- The worker's "task", which is the role that this specific machine has to play within
the cluster.

One example of TF_CONFIG is:

```
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
```

In the multi-worker synchronous training setup, valid roles (task types) for the machines
are "worker" and "evaluator".

For example, if you have 8 machines with 4 GPUs each, you could have 7 workers and one
evaluator.

- The workers train the model, each one processing sub-batches of a global batch.
- One of the workers (worker 0) will serve as "chief", a particular kind of worker that
is responsible for saving logs and checkpoints for later reuse (typically to a Cloud
storage location).
- The evaluator runs a continuous loop that loads the latest checkpoint saved by the
chief worker, runs evaluation on it (asynchronously from the other workers) and writes
evaluation logs (e.g. TensorBoard logs).


**Running code on each worker**

You would run training code on each worker (including the chief) and evaluation code on
the evaluator.

The training code is basically the same as what you would use in the single-host setup,
except using `MultiWorkerMirroredStrategy` instead of `MirroredStrategy`.

Each worker would run the same code (minus the difference explained in the note below),
including the same callbacks.

**Note:** Callbacks that save model checkpoints or logs should save to a different
directory for each worker. It is standard practice that all workers should save to local
disk (which is typically temporary), **except worker 0**, which would save TensorBoard
logs checkpoints to a Cloud storage location for later access & reuse.

The evaluator would simply use `MirroredStrategy` (since it runs on a single machine and
does not need to communicate with other machines) and call `model.evaluate()`. It would be
loading the latest checkpoint saved by the chief worker to a Cloud storage location, and
would save evaluation logs to the same location as the chief logs.


### Example: code running in a multi-worker setup

On the chief (worker 0):

```python
# Set TF_CONFIG
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})


# Open a strategy scope and create/restore the model.
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():
  model = make_or_restore_model()

callbacks = [
    # This callback saves a SavedModel every 100 batches
    keras.callbacks.ModelCheckpoint(filepath='path/to/cloud/location/ckpt',
                                    save_freq=100),
    keras.callbacks.TensorBoard('path/to/cloud/location/tb/')
]
model.fit(train_dataset,
          callbacks=callbacks,
          ...)
```

On other workers:

```python
# Set TF_CONFIG
worker_index = 1  # For instance
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': worker_index}
})


# Open a strategy scope and create/restore the model.
# You can restore from the checkpoint saved by the chief.
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():
  model = make_or_restore_model()

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath='local/path/ckpt', save_freq=100),
    keras.callbacks.TensorBoard('local/path/tb/')
]
model.fit(train_dataset,
          callbacks=callbacks,
          ...)
```

On the evaluator:

```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  model = make_or_restore_model()  # Restore from the checkpoint saved by the chief.

results = model.evaluate(val_dataset)
# Then, log the results on a shared location, write TensorBoard logs, etc
```


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

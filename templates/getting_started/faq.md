# Keras FAQ

A list of frequently Asked Keras Questions.

## General questions

- [How can I train a Keras model on multiple GPUs (on a single machine)?](#how-can-i-train-a-keras-model-on-multiple-gpus-on-a-single-machine)
- [How can I distribute training across multiple machines?](#how-can-i-distribute-training-across-multiple-machines)
- [How can I train a Keras model on TPU?](#how-can-i-train-a-keras-model-on-tpu)
- [Where is the Keras configuration file stored?](#where-is-the-keras-configuration-file-stored)
- [How to do hyperparameter tuning with Keras?](#how-to-do-hyperparameter-tuning-with-keras)
- [How can I obtain reproducible results using Keras during development?](#how-can-i-obtain-reproducible-results-using-keras-during-development)
- [What are my options for saving models?](#what-are-my-options-for-saving-models)
- [How can I install HDF5 or h5py to save my models?](#how-can-i-install-hdf5-or-h5py-to-save-my-models)
- [How should I cite Keras?](#how-should-i-cite-keras)

## Training-related questions

- [What do "sample", "batch", and "epoch" mean?](#what-do-sample-batch-and-epoch-mean)
- [Why is my training loss much higher than my testing loss?](#why-is-my-training-loss-much-higher-than-my-testing-loss)
- [How can I use Keras with datasets that don't fit in memory?](#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)
- [How can I ensure my training run can recover from program interruptions?](#how-can-i-ensure-my-training-run-can-recover-from-program-interruptions)
- [How can I interrupt training when the validation loss isn't decreasing anymore?](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)
- [How can I freeze layers and do fine-tuning?](#how-can-i-freeze-layers-and-do-finetuning)
- [What's the difference between the `training` argument in `call()` and the `trainable` attribute?](#whats-the-difference-between-the-training-argument-in-call-and-the-trainable-attribute)
- [In `fit()`, how is the validation split computed?](#in-fit-how-is-the-validation-split-computed)
- [In `fit()`, is the data shuffled during training?](#in-fit-is-the-data-shuffled-during-training)
- [What's the recommended way to monitor my metrics when training with `fit()`?](#whats-the-recommended-way-to-monitor-my-metrics-when-training-with-fit)
- [What if I need to customize what `fit()` does?](#what-if-i-need-to-customize-what-fit-does)
- [How can I train models in mixed precision?](#how-can-i-train-models-in-mixed-precision)
- [What's the difference between `Model` methods `predict()` and `__call__()`?](#whats-the-difference-between-model-methods-predict-and-call)

## Modeling-related questions

- [How can I obtain the output of an intermediate layer (feature extraction)?](#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction)
- [How can I use pre-trained models in Keras?](#how-can-i-use-pre-trained-models-in-keras)
- [How can I use stateful RNNs?](#how-can-i-use-stateful-rnns)


---

## General questions


### How can I train a Keras model on multiple GPUs (on a single machine)?

There are two ways to run a single model on multiple GPUs: **data parallelism** and **device parallelism**.
In most cases, what you need is most likely data parallelism.


**1) Data parallelism**

Data parallelism consists in replicating the target model once on each device, and using each replica to process a different fraction of the input data.

The best way to do data parallelism with Keras models is to use the `tf.distribute` API. Make sure to read our [guide about using `tf.distribute` with Keras](/guides/distributed_training/).

The gist of it is the following:

a) instantiate a "distribution strategy" object, e.g. `MirroredStrategy` (which replicates your model on each available device and keeps the state of each model in sync):

```python
strategy = tf.distribute.MirroredStrategy()
```

b) Create your model and compile it under the strategy's scope:

```python
with strategy.scope():
    # This could be any kind of model -- Functional, subclass...
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(10)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
```

Note that it's important that all state variable creation should happen under the scope.
So in case you create any additional variables, do that under the scope.

c) Call `fit()` with a `tf.data.Dataset` object as input. Distribution is broadly compatible with all callbacks, including custom callbacks.
Note that this call does not need to be under the strategy scope, since it doesn't create new variables.

```python
model.fit(train_dataset, epochs=12, callbacks=callbacks)
```


**2) Model parallelism**

Model parallelism consists in running different parts of a same model on different devices. It works best for models that have a parallel architecture, e.g. a model with two branches.

This can be achieved by using TensorFlow device scopes. Here is a quick example:

```python
# Model where a shared LSTM is used to encode two different sequences in parallel
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# Process the first sequence on one GPU
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(input_a)
# Process the next sequence on another GPU
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(input_b)

# Concatenate results on CPU
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate(
        [encoded_a, encoded_b], axis=-1)
```

---

### How can I distribute training across multiple machines?

TensorFlow 2 enables you to write code that is mostly
agnostic to how you will distribute it:
any code that can run locally can be distributed to multiple
workers and accelerators by only adding to it a distribution strategy
(`tf.distribute.Strategy`) corresponding to your hardware of choice,
without any other code changes.

This also applies to any Keras model: just
add a `tf.distribute` distribution strategy scope enclosing the model
building and compiling code, and the training will be distributed according to
the `tf.distribute` distribution strategy.

For distributed training across multiple machines (as opposed to training that only leverages
multiple devices on a single machine), there are two distribution strategies you
could use: `MultiWorkerMirroredStrategy` and `ParameterServerStrategy`:

- `tf.distribute.MultiWorkerMirroredStrategy` implements a synchronous CPU/GPU
multi-worker solution to work with Keras-style model building and training loop,
using synchronous reduction of gradients across the replicas.
- `tf.distribute.experimental.ParameterServerStrategy` implements an asynchronous CPU/GPU
multi-worker solution, where the parameters are stored on parameter servers, and
workers update the gradients to parameter servers asynchronously.

Distributed training is somewhat more involved than single-machine multi-device training. 
With `ParameterServerStrategy`, you will need to launch a remote cluster of machines
consisting "worker" and "ps", each running a `tf.distribute.Server`, then run your 
python program on a "chief" machine that holds a `TF_CONFIG` environment variable
that specifies how to communicate with the other machines in the cluster. With
`MultiWorkerMirroredStrategy`, you will run the same program on each of the
chief and workers, again with a `TF_CONFIG` environment variable that specifies 
how to communicate with the cluster. From there, the workflow is similar to using 
single-machine training, with the main difference being that you will use 
`ParameterServerStrategy` or `MultiWorkerMirroredStrategy` as your distribution strategy.

Importantly, you should:

- Make sure your dataset is so configured that all workers in the cluster are able to
efficiently pull data from it (e.g. if your cluster is running on Google Cloud,
it's a good idea to host your data on Google Cloud Storage).
- Make sure your training is fault-tolerant
(e.g. by configuring a `keras.callbacks.BackupAndRestore` callback).

Below, we provide a couple of code snippets that cover the basic workflow. For more information
about CPU/GPU multi-worker training, see 
[Multi-GPU and distributed training](/guides/distributed_training/); for TPU
training, see [How can I train a Keras model on TPU?](#how-can-i-train-a-keras-model-on-tpu).

With `ParameterServerStrategy`:

```python
cluster_resolver = ...
if cluster_resolver.task_type in ("worker", "ps"):
  # Start a `tf.distribute.Server` and wait.
  ...
elif cluster_resolver.task_type == "evaluator":
  # Run an (optional) side-car evaluation
  ...

# Otherwise, this is the coordinator that controls the training w/ the strategy.
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver=...)
train_dataset = ...

with strategy.scope():
  model = tf.keras.Sequential([
      layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'],
      steps_per_execution=10)

model.fit(x=train_dataset, epochs=3, steps_per_epoch=100)
```

With `MultiWorkerMirroredStrategy`:

```python
# By default `MultiWorkerMirroredStrategy` uses cluster information
# from `TF_CONFIG`, and "AUTO" collective op communication.
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
train_dataset = get_training_dataset()
with strategy.scope():
  # Define and compile the model in the scope of the strategy. Doing so
  # ensures the variables created are distributed and initialized properly
  # according to the strategy.
  model = tf.keras.Sequential([
      layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
model.fit(x=train_dataset, epochs=3, steps_per_epoch=100)
```

---

### How can I train a Keras model on TPU?

TPUs are a fast & efficient hardware accelerator for deep learning that is publicly available on Google Cloud.
You can use TPUs via Colab, AI Platform (ML Engine), and Deep Learning VMs (provided the `TPU_NAME` environment variable is set on the VM).

Make sure to read the [TPU usage guide](https://www.tensorflow.org/guide/tpu) first. Here's a quick summary:

After connecting to a TPU runtime (e.g. by selecting the TPU runtime in Colab), you will need to detect your TPU using a `TPUClusterResolver`, which automatically detects a linked TPU on all supported platforms:

```python
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
print('Running on TPU: ', tpu.cluster_spec().as_dict()['worker'])

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
print('Replicas: ', strategy.num_replicas_in_sync)

with strategy.scope():
    # Create your model here.
    ...
```

After the initial setup, the workflow is similar to using single-machine
multi-GPU training, with the main difference being that you will use `TPUStrategy` as your distribution strategy.

Importantly, you should:

- Make sure your dataset yields batches with a fixed static shape. A TPU graph can only process inputs with a constant shape.
- Make sure you are able to read your data fast enough to keep the TPU utilized. Using the [TFRecord format](https://www.tensorflow.org/tutorials/load_data/tfrecord) to store your data may be a good idea.
- Consider running multiple steps of gradient descent per graph execution in order to keep the TPU utilized. You can do this via the `experimental_steps_per_execution` argument `compile()`. It will yield a significant speed up for small models.

---

### Where is the Keras configuration file stored?

The default directory where all Keras data is stored is:

`$HOME/.keras/`

For instance, for me, on a MacBook Pro, it's `/Users/fchollet/.keras/`.

Note that Windows users should replace `$HOME` with `%USERPROFILE%`.

In case Keras cannot create the above directory (e.g. due to permission issues), `/tmp/.keras/` is used as a backup.

The Keras configuration file is a JSON file stored at `$HOME/.keras/keras.json`. The default configuration file looks like this:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

It contains the following fields:

- The image data format to be used as default by image processing layers and utilities (either `channels_last` or `channels_first`).
- The `epsilon` numerical fuzz factor to be used to prevent division by zero in some operations.
- The default float data type.
- The default backend. This is legacy; nowadays there is only TensorFlow.

Likewise, cached dataset files, such as those downloaded with [`get_file()`](/utils/#get_file), are stored by default in `$HOME/.keras/datasets/`,
and cached model weights files from Keras Applications are stored by default in `$HOME/.keras/models/`.


---

### How to do hyperparameter tuning with Keras?


We recommend using [KerasTuner](https://keras.io/keras_tuner/).

---

### How can I obtain reproducible results using Keras during development?

During development of a model, sometimes it is useful to be able to obtain reproducible results from run to run in order to determine if a change in performance is due to an actual model or data modification, or merely a result of a new random seed.

First, you need to set the `PYTHONHASHSEED` environment variable to `0` before the program starts (not within the program itself). This is necessary in Python 3.2.3 onwards to have reproducible behavior for certain hash-based operations (e.g., the item order in a set or a dict, see [Python's documentation](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED) or [issue #2280](https://github.com/keras-team/keras/issues/2280#issuecomment-306959926) for further details). One way to set the environment variable is when starting python like this:

```shell
$ cat test_hash.py
print(hash("keras"))
$ python3 test_hash.py                  # non-reproducible hash (Python 3.2.3+)
8127205062320133199
$ python3 test_hash.py                  # non-reproducible hash (Python 3.2.3+)
3204480642156461591
$ PYTHONHASHSEED=0 python3 test_hash.py # reproducible hash
4883664951434749476
$ PYTHONHASHSEED=0 python3 test_hash.py # reproducible hash
4883664951434749476
```

Moreover, when running on a GPU, some operations have non-deterministic outputs, in particular `tf.reduce_sum()`. This is due to the fact that GPUs run many operations in parallel, so the order of execution is not always guaranteed. Due to the limited precision of floats, even adding several numbers together may give slightly different results depending on the order in which you add them. You can try to avoid the non-deterministic operations, but some may be created automatically by TensorFlow to compute the gradients, so it is much simpler to just run the code on the CPU. For this, you can set the `CUDA_VISIBLE_DEVICES` environment variable to an empty string, for example:

```shell
$ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py
```

The below snippet of code provides an example of how to obtain reproducible results:

```python
import numpy as np
import tensorflow as tf
import random as python_random

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

# Rest of code follows ...
```

Note that you don't have to set seeds for individual initializers
in your code if you do the steps above, because their seeds are determined
by the combination of the seeds set above.


---

### What are my options for saving models?

*Note: it is not recommended to use pickle or cPickle to save a Keras model.*

**1) Whole-model saving (configuration + weights)**

Whole-model saving means creating a file that will contain:

- the architecture of the model, allowing you to re-create the model
- the weights of the model
- the training configuration (loss, optimizer)
- the state of the optimizer, allowing you to resume training exactly where you left off.

The default and recommended format to use is the TensorFlow [SavedModel format](https://www.tensorflow.org/guide/saved_model).
In TensorFlow 2.0 and higher, you can just do: `model.save(your_file_path)`.

For explicitness, you can also use `model.save(your_file_path, save_format='tf')`.

Keras still supports its original HDF5-based saving format. To save a model in HDF5 format,
use `model.save(your_file_path, save_format='h5')`. Note that this option is automatically used
if `your_file_path` ends in `.h5` or `.keras`.
Please also see [How can I install HDF5 or h5py to save my models?](#how-can-i-install-hdf5-or-h5py-to-save-my-models) for instructions on how to install `h5py`.

After saving a model in either format, you can reinstantiate it via `model = keras.models.load_model(your_file_path)`.

**Example:**

```python
from tensorflow.keras.models import load_model

model.save('my_model')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model')
```


**2) Weights-only saving**


If you need to save the **weights of a model**, you can do so in HDF5 with the code below:

```python
model.save_weights('my_model_weights.h5')
```

Assuming you have code for instantiating your model, you can then load the weights you saved into a model with the *same* architecture:

```python
model.load_weights('my_model_weights.h5')
```

If you need to load the weights into a *different* architecture (with some layers in common), for instance for fine-tuning or transfer-learning, you can load them by *layer name*:

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

Example:

```python
"""
Assuming the original model looks like this:

model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))
model.add(Dense(3, name='dense_2'))
...
model.save_weights(fname)
"""

# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # will be loaded
model.add(Dense(10, name='new_dense'))  # will not be loaded

# load weights from the first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)
```

Please also see [How can I install HDF5 or h5py to save my models?](#how-can-i-install-hdf5-or-h5py-to-save-my-models) for instructions on how to install `h5py`.


**3) Configuration-only saving (serialization)**


If you only need to save the **architecture of a model**, and not its weights or its training configuration, you can do:

```python
# save as JSON
json_string = model.to_json()
```

The generated JSON file is human-readable and can be manually edited if needed.

You can then build a fresh model from this data:

```python
# model reconstruction from JSON:
from tensorflow.keras.models import model_from_json
model = model_from_json(json_string)
```


**4) Handling custom layers (or other custom objects) in saved models**

If the model you want to load includes custom layers or other custom classes or functions, 
you can pass them to the loading mechanism via the `custom_objects` argument: 

```python
from tensorflow.keras.models import load_model
# Assuming your model includes instance of an "AttentionLayer" class
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

Alternatively, you can use a [custom object scope](https://keras.io/utils/#customobjectscope):

```python
from tensorflow.keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

Custom objects handling works the same way for `load_model` & `model_from_json`:

```python
from tensorflow.keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### How can I install HDF5 or h5py to save my models?

In order to save your Keras models as HDF5 files, Keras uses the h5py Python package. It is
a dependency of Keras and should be installed by default. On Debian-based
distributions, you will have to additionally install `libhdf5`:

<div class="k-default-code-block">
```
sudo apt-get install libhdf5-serial-dev
```
</div>

If you are unsure if h5py is installed you can open a Python shell and load the
module via

```
import h5py
```

If it imports without error it is installed, otherwise you can find
[detailed installation instructions here](http://docs.h5py.org/en/latest/build.html).



---

### How should I cite Keras?

Please cite Keras in your publications if it helps your research. Here is an example BibTeX entry:

<code style="color: gray;">
@misc{chollet2015keras,<br>
&nbsp;&nbsp;title={Keras},<br>
&nbsp;&nbsp;author={Chollet, Fran\c{c}ois and others},<br>
&nbsp;&nbsp;year={2015},<br>
&nbsp;&nbsp;howpublished={\url{https://keras.io}},<br>
}
</code>

---

## Training-related questions


### What do "sample", "batch", and "epoch" mean?


Below are some common definitions that are necessary to know and understand to correctly utilize Keras `fit()`:

- **Sample**: one element of a dataset. For instance, one image is a **sample** in a convolutional network. One audio snippet is a **sample** for a speech recognition model.

- **Batch**: a set of *N* samples. The samples in a **batch** are processed independently, in parallel. If training, a batch results in only one update to the model. A **batch** generally approximates the distribution of the input data better than a single input. The larger the batch, the better the approximation; however, it is also true that the batch will take longer to process and will still result in only one update. For inference (evaluate/predict), it is recommended to pick a batch size that is as large as you can afford without going out of memory (since larger batches will usually result in faster evaluation/prediction).

- **Epoch**: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
When using `validation_data` or `validation_split` with the `fit` method of Keras models, evaluation will be run at the end of every **epoch**.
Within Keras, there is the ability to add [callbacks](/api/callbacks/) specifically designed to be run at the end of an **epoch**. Examples of these are learning rate changes and model checkpointing (saving).

---

### Why is my training loss much higher than my testing loss?


A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.
They are reflected in the training time loss but not in the test time loss.

Besides, the training loss that Keras displays is the average of the losses for each batch of training data, **over the current epoch**.
Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches.
This can bring the epoch-wise average down.
On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.


---

### How can I use Keras with datasets that don't fit in memory?

You should use the [`tf.data` API](https://www.tensorflow.org/guide/data) to create `tf.data.Dataset` objects -- an abstraction over a data pipeline
that can pull data from local disk, from a distributed file system, from GCS, etc., as well as efficiently apply various data transformations.

For instance, the utility [`tf.keras.preprocessing.image_dataset_from_directory`](https://keras.io/api/preprocessing/image/#imagedatasetfromdirectory-function)
will create a dataset that reads image data from a local directory.
Likewise, the utility [`tf.keras.preprocessing.text_dataset_from_directory`](https://keras.io/api/preprocessing/text/#textdatasetfromdirectory-function)
will create a dataset that reads text files from a local directory.

Dataset objects can be directly passed to `fit()`, or can be iterated over in a custom low-level training loop.

```python
model.fit(dataset, epochs=10, validation_data=val_dataset)
```

---

### How can I ensure my training run can recover from program interruptions?

To ensure the ability to recover from an interrupted training run at any time (fault tolerance),
you should use a `tf.keras.callbacks.experimental.BackupAndRestore` that regularly saves your training progress,
including the epoch number and weights, to disk, and loads it the next time you call `Model.fit()`.

```python
import tensorflow as tf
from tensorflow import keras

class InterruptingCallback(keras.callbacks.Callback):
  """A callback to intentionally introduce interruption to training."""
  def on_epoch_end(self, epoch, log=None):
    if epoch == 15:
      raise RuntimeError('Interruption')

model = keras.Sequential([keras.layers.Dense(10)])
optimizer = keras.optimizers.SGD()
model.compile(optimizer, loss="mse")

x = tf.random.uniform((24, 10))
y = tf.random.uniform((24,))
dataset = tf.data.Dataset.from_tensor_slices((x, y)).repeat().batch(2)

backup_callback = keras.callbacks.experimental.BackupAndRestore(
    backup_dir='/tmp/backup')
try:
  model.fit(dataset, epochs=20, steps_per_epoch=5, 
            callbacks=[backup_callback, InterruptingCallback()])
except RuntimeError:
  print('***Handling interruption***')
  # This continues at the epoch where it left off.
  model.fit(dataset, epochs=20, steps_per_epoch=5, 
            callbacks=[backup_callback])
```

Find out more in the [callbacks documentation](/api/callbacks/).


---

### How can I interrupt training when the validation loss isn't decreasing anymore?


You can use an `EarlyStopping` callback:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

Find out more in the [callbacks documentation](/api/callbacks/).

---

### How can I freeze layers and do fine-tuning?

**Setting the `trainable` attribute**

All layers & models have a `layer.trainable` boolean attribute:

```shell
>>> layer = Dense(3)
>>> layer.trainable
True
```

On all layers & models, the `trainable` attribute can be set (to True or False).
When set to `False`, the `layer.trainable_weights` attribute is empty:

```python
>>> layer = Dense(3)
>>> layer.build(input_shape=(3, 3)) # Create the weights of the layer
>>> layer.trainable
True
>>> layer.trainable_weights
[<tf.Variable 'kernel:0' shape=(3, 3) dtype=float32, numpy=
array([[...]], dtype=float32)>, <tf.Variable 'bias:0' shape=(3,) dtype=float32, numpy=array([...], dtype=float32)>]
>>> layer.trainable = False
>>> layer.trainable_weights
[]
```

Setting the `trainable` attribute on a layer recursively sets it on all children layers (contents of `self.layers`).


**1) When training with `fit()`:**

To do fine-tuning with `fit()`, you would:

- Instantiate a base model and load pre-trained weights
- Freeze that base model
- Add trainable layers on top
- Call `compile()` and `fit()`

Like this:

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # Freeze ResNet50Base.

assert model.layers[0].trainable_weights == []  # ResNet50Base has no trainable weights.
assert len(model.trainable_weights) == 2  # Just the bias & kernel of the Dense layer.

model.compile(...)
model.fit(...)  # Train Dense while excluding ResNet50Base.
```

You can follow a similar workflow with the Functional API or the model subclassing API.
Make sure to call `compile()` *after* changing the value of `trainable` in order for your
changes to be taken into account. Calling `compile()` will freeze the state of the training step of the model.


**2) When using a custom training loop:**

When writing a training loop, make sure to only update
weights that are part of `model.trainable_weights` (and not all `model.weights`).

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # Freeze ResNet50Base.

# Iterate over the batches of a dataset.
for inputs, targets in dataset:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = model(inputs)
        # Compute the loss value for this batch.
        loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the *trainable* weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```


**Interaction between `trainable` and `compile()`**

Calling `compile()` on a model is meant to "freeze" the behavior of that model. This implies that the `trainable`
attribute values at the time the model is compiled should be preserved throughout the lifetime of that model,
until `compile` is called again. Hence, if you change `trainable`, make sure to call `compile()` again on your
model for your changes to be taken into account.

For instance, if two models A & B share some layers, and:

- Model A gets compiled
- The `trainable` attribute value on the shared layers is changed
- Model B is compiled

Then model A and B are using different `trainable` values for the shared layers. This mechanism is
critical for most existing GAN implementations, which do:

```python
discriminator.compile(...)  # the weights of `discriminator` should be updated when `discriminator` is trained
discriminator.trainable = False
gan.compile(...)  # `discriminator` is a submodel of `gan`, which should not be updated when `gan` is trained
```


---

### What's the difference between the `training` argument in `call()` and the `trainable` attribute?


`training` is a boolean argument in `call` that determines whether the call
should be run in inference mode or training mode. For example, in training mode,
a `Dropout` layer applies random dropout and rescales the output. In inference mode, the same
layer does nothing. Example:

```python
y = Dropout(0.5)(x, training=True)  # Applies dropout at training time *and* inference time
```

`trainable` is a boolean layer attribute that determines the trainable weights
of the layer should be updated to minimize the loss during training. If `layer.trainable` is set to `False`,
then `layer.trainable_weights` will always be an empty list. Example:

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # Freeze ResNet50Base.

assert model.layers[0].trainable_weights == []  # ResNet50Base has no trainable weights.
assert len(model.trainable_weights) == 2  # Just the bias & kernel of the Dense layer.

model.compile(...)
model.fit(...)  # Train Dense while excluding ResNet50Base.
```

As you can see, "inference mode vs training mode" and "layer weight trainability" are two very different concepts.

You could imagine the following: a dropout layer where the scaling factor is learned during training, via
backpropagation. Let's name it `AutoScaleDropout`.
This layer would have simultaneously a trainable state, and a different behavior in inference and training.
Because the `trainable` attribute and the `training` call argument are independent, you can do the following:

```python
layer = AutoScaleDropout(0.5)

# Applies dropout at training time *and* inference time  
# *and* learns the scaling factor during training
y = layer(x, training=True)

assert len(layer.trainable_weights) == 1
```

```python
# Applies dropout at training time *and* inference time  
# with a *frozen* scaling factor

layer = AutoScaleDropout(0.5)
layer.trainable = False
y = layer(x, training=True)
```


***Special case of the `BatchNormalization` layer***

Consider a `BatchNormalization` layer in the frozen part of a model that's used for fine-tuning.

It has long been debated whether the moving statistics of the `BatchNormalization` layer should
stay frozen or adapt to the new data. Historically, `bn.trainable = False`
would only stop backprop but would not prevent the training-time statistics
update. After extensive testing, we have found that it is *usually* better to freeze the moving statistics
in fine-tuning use cases. **Starting in TensorFlow 2.0, setting `bn.trainable = False`
will *also* force the layer to run in inference mode.**

This behavior only applies for `BatchNormalization`. For every other layer, weight trainability and
"inference vs training mode" remain independent.



---

### In `fit()`, how is the validation split computed?


If you set the `validation_split` argument in `model.fit` to e.g. 0.1, then the validation data used will be the *last 10%* of the data. If you set it to 0.25, it will be the last 25% of the data, etc. Note that the data isn't shuffled before extracting the validation split, so the validation is literally just the *last* x% of samples in the input you passed.

The same validation set is used for all epochs (within the same call to `fit`).

Note that the `validation_split` option is only available if your data is passed as Numpy arrays (not `tf.data.Datasets`, which are not indexable).


---

### In `fit()`, is the data shuffled during training?

If you pass your data as NumPy arrays and if the `shuffle` argument in `model.fit()` is set to `True` (which is the default), the training data will be globally randomly shuffled at each epoch.

If you pass your data as a `tf.data.Dataset` object and if the `shuffle` argument in `model.fit()` is set to `True`, the dataset will be locally shuffled (buffered shuffling).

When using `tf.data.Dataset` objects, prefer shuffling your data beforehand (e.g. by calling `dataset = dataset.shuffle(buffer_size)`) so as to be in control of the buffer size.

Validation data is never shuffled.


---

### What's the recommended way to monitor my metrics when training with `fit()`?

Loss values and metric values are reported via the default progress bar displayed by calls to `fit()`.
However, staring at changing ascii numbers in a console is not an optimal metric-monitoring experience.
We recommend the use of [TensorBoard](https://www.tensorflow.org/tensorboard), which will display nice-looking graphs of your training and validation metrics, regularly
updated during training, which you can access from your browser.

You can use TensorBoard with `fit()` via the [`TensorBoard` callback](/api/callbacks/tensorboard/).

---

### What if I need to customize what `fit()` does?

You have two options:

**1) Subclass the `Model` class and override the `train_step` (and `test_step`) methods**

This is a better option if you want to use custom update rules but still want to leverage the functionality provided by `fit()`,
such as callbacks, efficient step fusing, etc.

Note that this pattern does not prevent you from building models with the
Functional API, in which case you will use the class you created to instantiate
the model with the `inputs` and `outputs`. Same goes for Sequential models, in
which case you will subclass `keras.Sequential` and override its `train_step`
instead of `keras.Model`.

The example below shows a Functional model with a custom `train_step`.

```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

class MyCustomModel(keras.Model):

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred,
                                      regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


# Construct and compile an instance of MyCustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = MyCustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Just use `fit` as usual
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=10)
```

You can also easily add support for sample weighting:

```python
class MyCustomModel(keras.Model):

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(y, y_pred,
                                      sample_weight=sample_weight,
                                      regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(
            y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


# Construct and compile an instance of MyCustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = MyCustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# You can now use sample_weight argument
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, epochs=10)
```

Similarly, you can also customize evaluation by overriding `test_step`:

```python
class MyCustomModel(keras.Model):

    def test_step(self, data):
      # Unpack the data
      x, y = data
      # Compute predictions
      y_pred = self(x, training=False)
      # Updates the metrics tracking the loss
      self.compiled_loss(
          y, y_pred, regularization_losses=self.losses)
      # Update the metrics.
      self.compiled_metrics.update_state(y, y_pred)
      # Return a dict mapping metric names to current value.
      # Note that it will include the loss (tracked in self.metrics).
      return {m.name: m.result() for m in self.metrics}
```

**2) Write a low-level custom training loop**

This is a good option if you want to be in control of every last little detail. But it can be somewhat verbose. Example:

```python
# Prepare an optimizer.
optimizer = tf.keras.optimizers.Adam()
# Prepare a loss function.
loss_fn = tf.keras.losses.kl_divergence

# Iterate over the batches of a dataset.
for inputs, targets in dataset:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = model(inputs)
        # Compute the loss value for this batch.
        loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

This example does not include a lot of essential functionality like displaying a progress bar, calling callbacks,
updating metrics, etc. You would have to do this yourself. It's not difficult at all, but it's a bit of work.


---

### How can I train models in mixed precision?

Keras has built-in support for mixed precision training on GPU and TPU.
See [this extensive guide](https://www.tensorflow.org/guide/keras/mixed_precision). 

---

### What's the difference between `Model` methods `predict()` and `__call__()`?

Let's answer with an extract from
[Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras):

> Both `y = model.predict(x)` and `y = model(x)` (where `x` is an array of input data)
> mean "run the model on `x` and retrieve the output `y`." Yet they aren't exactly
> the same thing.

> `predict()` loops over the data in batches
> (in fact, you can specify the batch size via `predict(x, batch_size=64)`),
> and it extracts the NumPy value of the outputs. It's schematically equivalent to this:
```python
def predict(x):
    y_batches = []
    for x_batch in get_batches(x):
        y_batch = model(x).numpy()
        y_batches.append(y_batch)
    return np.concatenate(y_batches)
```
> This means that `predict()` calls can scale to very large arrays. Meanwhile,
> `model(x)` happens in-memory and doesn't scale.
> On the other hand, `predict()` is not differentiable: you cannot retrieve its gradient
> if you call it in a `GradientTape` scope.

> You should use `model(x)` when you need to retrieve the gradients of the model call,
> and you should use `predict()` if you just need the output value. In other words,
> always use `predict()` unless you're in the middle of writing a low-level gradient
> descent loop (as we are now).

---

## Modeling-related questions


### How can I obtain the output of an intermediate layer (feature extraction)?

In the Functional API and Sequential API, if a layer has been called exactly once, you can retrieve its output via `layer.output` and its input via `layer.input`.
This enables you do quickly instantiate feature-extraction models, like this one:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = Sequential([
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.GlobalMaxPooling2D(),
    layers.Dense(10),
])
extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])
features = extractor(data)
```

Naturally, this is not possible with models that are subclasses of `Model` that override `call`.

Here's another example: instantiating a `Model` that returns the output of a specific named layer:

```python
model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model(data)
```

---

### How can I use pre-trained models in Keras?

You could leverage the [models available in `keras.applications`](/api/applications/), or the models available on [TensorFlow Hub](https://www.tensorflow.org/hub).
TensorFlow Hub is well-integrated with Keras.

---

### How can I use stateful RNNs?


Making a RNN stateful means that the states for the samples of each batch will be reused as initial states for the samples in the next batch.

When using stateful RNNs, it is therefore assumed that:

- all batches have the same number of samples
- If `x1` and `x2` are successive batches of samples, then `x2[i]` is the follow-up sequence to `x1[i]`, for every `i`.

To use statefulness in RNNs, you need to:

- explicitly specify the batch size you are using, by passing a `batch_size` argument to the first layer in your model. E.g. `batch_size=32` for a 32-samples batch of sequences of 10 timesteps with 16 features per timestep.
- set `stateful=True` in your RNN layer(s).
- specify `shuffle=False` when calling `fit()`.

To reset the states accumulated:

- use `model.reset_states()` to reset the states of all layers in the model
- use `layer.reset_states()` to reset the states of a specific stateful RNN layer

Example:

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

x = np.random.random((32, 21, 16))  # this is our input data, of shape (32, 21, 16)
# we will feed it to our model in sequences of length 10

model = keras.Sequential()
model.add(layers.LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(layers.Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()
```

Note that the methods `predict`, `fit`, `train_on_batch`, etc. will *all* update the states of the stateful layers in a model. This allows you to do not only stateful training, but also stateful prediction.


---


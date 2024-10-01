# Keras FAQ

A list of frequently Asked Keras Questions.

## General questions

- [How can I train a Keras model on multiple GPUs (on a single machine)?](#how-can-i-train-a-keras-model-on-multiple-gpus-on-a-single-machine)
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
- [How can I ensure my training run can recover from program interruptions?](#how-can-i-ensure-my-training-run-can-recover-from-program-interruptions)
- [How can I interrupt training when the validation loss isn't decreasing anymore?](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)
- [How can I freeze layers and do fine-tuning?](#how-can-i-freeze-layers-and-do-finetuning)
- [What's the difference between the `training` argument in `call()` and the `trainable` attribute?](#whats-the-difference-between-the-training-argument-in-call-and-the-trainable-attribute)
- [In `fit()`, how is the validation split computed?](#in-fit-how-is-the-validation-split-computed)
- [In `fit()`, is the data shuffled during training?](#in-fit-is-the-data-shuffled-during-training)
- [What's the recommended way to monitor my metrics when training with `fit()`?](#whats-the-recommended-way-to-monitor-my-metrics-when-training-with-fit)
- [What if I need to customize what `fit()` does?](#what-if-i-need-to-customize-what-fit-does)
- [What's the difference between `Model` methods `predict()` and `__call__()`?](#whats-the-difference-between-model-methods-predict-and-call)

## Modeling-related questions

- [How can I obtain the output of an intermediate layer (feature extraction)?](#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction)
- [How can I use pre-trained models in Keras?](#how-can-i-use-pre-trained-models-in-keras)
- [How can I use stateful RNNs?](#how-can-i-use-stateful-rnns)


---

## General questions


### How can I train a Keras model on multiple GPUs (on a single machine)?

There are two ways to run a single model on multiple GPUs: **data parallelism** and **device parallelism**.
Keras covers both.

For data parallelism, Keras supports the built-in data parallel distribution APIs of
JAX, TensorFlow, and PyTorch. See the following guides:

- [Multi-GPU distributed training with JAX](/guides/distributed_training_with_jax/)
- [Multi-GPU distributed training with TensorFlow](/guides/distributed_training_with_tensorflow/)
- [Multi-GPU distributed training with PyTorch](/guides/distributed_training_with_torch/)

For model parallelism, Keras has its own distribution API, which is currently only support by the JAX backend.
See [the documentation for the `LayoutMap` API](/api/distribution/).

---

### How can I train a Keras model on TPU?

TPUs are a fast & efficient hardware accelerator for deep learning that is publicly available on Google Cloud.
You can use TPUs via Colab, Kaggle notebooks, and GCP Deep Learning VMs (provided the `TPU_NAME` environment variable is set on the VM).

All Keras backends (JAX, TensorFlow, PyTorch) are supported on TPU, but we recommend JAX or TensorFlow in this case.

**Using JAX:**

When connected to a TPU runtime, just insert this code snippet before model construction:

```python
import jax
distribution = keras.distribution.DataParallel(devices=jax.devices())
keras.distribution.set_distribution(distribution)
```

**Using TensorFlow:**

When connected to a TPU runtime, use `TPUClusterResolver` to detect the TPU.
Then, create `TPUStrategy` and construct your model in the strategy scope:

```python
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

with strategy.scope():
    # Create your model here.
    ...
```

Importantly, you should:

- Make sure you are able to read your data fast enough to keep the TPU utilized.
- Consider running multiple steps of gradient descent per graph execution in order to keep the TPU utilized.
You can do this via the `experimental_steps_per_execution` argument `compile()`. It will yield a significant speed up for small models.

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
- The default backend. It can be one of `"jax"`, `"tensorflow"`, `"torch"`, or `"numpy"`.

Likewise, cached dataset files, such as those downloaded with [`get_file()`](/utils/#get_file), are stored by default in `$HOME/.keras/datasets/`,
and cached model weights files from Keras Applications are stored by default in `$HOME/.keras/models/`.


---

### How to do hyperparameter tuning with Keras?

We recommend using [KerasTuner](https://keras.io/keras_tuner/).

---

### How can I obtain reproducible results using Keras during development?

There are four sources of randomness to consider:

1. Keras itself (e.g. `keras.random` ops or random layers from `keras.layers`).
2. The current Keras backend (e.g. JAX, TensorFlow, or PyTorch).
3. The Python runtime.
4. The CUDA runtime. When running on a GPU, some operations have non-deterministic outputs. This is due to the fact that GPUs run many operations in parallel, so the order of execution is not always guaranteed. Due to the limited precision of floats, even adding several numbers together may give slightly different results depending on the order in which you add them.

To make both Keras and the current backend framework deterministic, use this:

```python
keras.utils.set_random_seed(1337)
```

To make Python deterministic, you need to set the `PYTHONHASHSEED` environment variable to `0` before the program starts (not within the program itself). This is necessary in Python 3.2.3 onwards to have reproducible behavior for certain hash-based operations (e.g., the item order in a set or a dict, see [Python's documentation](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED)).

To make the CUDA runtime deterministic: if using the TensorFlow backend, call `tf.config.experimental.enable_op_determinism`. Note that this will have a performance cost. What to do for other backends may vary -- check the documentation of your backend framework directly.

---

### What are my options for saving models?

*Note: it is not recommended to use pickle or cPickle to save a Keras model.*

**1) Whole-model saving (configuration + weights)**

Whole-model saving means creating a file that will contain:

- the architecture of the model, allowing you to re-create the model
- the weights of the model
- the training configuration (loss, optimizer)
- the state of the optimizer, allowing you to resume training exactly where you left off.

The default and recommended way to save a whole model is to just do: `model.save(your_file_path.keras)`.

After saving a model in either format, you can reinstantiate it via `model = keras.models.load_model(your_file_path.keras)`.

**Example:**

```python
from keras.saving import load_model

model.save('my_model.keras')
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.keras')
```


**2) Weights-only saving**


If you need to save the **weights of a model**, you can do so in HDF5 with the code below, using the file extension `.weights.h5`:

```python
model.save_weights('my_model.weights.h5')
```

Assuming you have code for instantiating your model, you can then load the weights you saved into a model with the *same* architecture:

```python
model.load_weights('my_model.weights.h5')
```

If you need to load the weights into a *different* architecture (with some layers in common), for instance for fine-tuning or transfer-learning, you can load them by *layer name*:

```python
model.load_weights('my_model.weights.h5', by_name=True)
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

See also [How can I install HDF5 or h5py to save my models?](#how-can-i-install-hdf5-or-h5py-to-save-my-models) for instructions on how to install `h5py`.


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
from keras.models import model_from_json
model = model_from_json(json_string)
```


**4) Handling custom layers (or other custom objects) in saved models**

If the model you want to load includes custom layers or other custom classes or functions,
you can pass them to the loading mechanism via the `custom_objects` argument:

```python
from keras.models import load_model
# Assuming your model includes instance of an "AttentionLayer" class
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

Alternatively, you can use a [custom object scope](https://keras.io/utils/#customobjectscope):

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

Custom objects handling works the same way for `load_model` & `model_from_json`:

```python
from keras.models import model_from_json
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

### How can I ensure my training run can recover from program interruptions?

To ensure the ability to recover from an interrupted training run at any time (fault tolerance),
you should use a `keras.callbacks.BackupAndRestore` callback that regularly saves your training progress,
including the epoch number and weights, to disk, and loads it the next time you call `Model.fit()`.

```python
import numpy as np
import keras

class InterruptingCallback(keras.callbacks.Callback):
  """A callback to intentionally introduce interruption to training."""
  def on_epoch_end(self, epoch, log=None):
    if epoch == 15:
      raise RuntimeError('Interruption')

model = keras.Sequential([keras.layers.Dense(10)])
optimizer = keras.optimizers.SGD()
model.compile(optimizer, loss="mse")

x = np.random.random((24, 10))
y = np.random.random((24,))

backup_callback = keras.callbacks.experimental.BackupAndRestore(
    backup_dir='/tmp/backup')
try:
  model.fit(x, y, epochs=20, steps_per_epoch=5,
            callbacks=[backup_callback, InterruptingCallback()])
except RuntimeError:
  print('***Handling interruption***')
  # This continues at the epoch where it left off.
  model.fit(x, y, epochs=20, steps_per_epoch=5,
            callbacks=[backup_callback])
```

Find out more in the [callbacks documentation](/api/callbacks/).


---

### How can I interrupt training when the validation loss isn't decreasing anymore?


You can use an `EarlyStopping` callback:

```python
from keras.callbacks import EarlyStopping

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
>>> layer.build(input_shape=(None, 3)) # Create the weights of the layer
>>> layer.trainable
True
>>> layer.trainable_weights
[<KerasVariable shape=(3, 3), dtype=float32, path=dense/kernel>, <KerasVariable shape=(3,), dtype=float32, path=dense/bias>]
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
Here's a simple TensorFlow example:

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

For a `BatchNormalization` layer, setting `bn.trainable = False` will also make its `training` call argument
default to `False`, meaning that the layer will no update its state during training.

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

See the following guides:

- [Writing a custom train step in JAX](/guides/custom_train_step_in_jax/)
- [Writing a custom train step in TensorFlow](/guides/custom_train_step_in_tensorflow/)
- [Writing a custom train step in PyTorch](/guides/custom_train_step_in_torch/)

**2) Write a low-level custom training loop**

This is a good option if you want to be in control of every last little detail -- though it can be somewhat verbose.

See the following guides:

- [Writing a custom training loop in JAX](/guides/writing_a_custom_training_loop_in_jax/)
- [Writing a custom training loop in TensorFlow](/guides/writing_a_custom_training_loop_in_tensorflow/)
- [Writing a custom training loop in PyTorch](/guides/writing_a_custom_training_loop_in_torch/)


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
        y_batch = model(x_batch).numpy()
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
import keras
from keras import layers

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

You could leverage the [models available in `keras.applications`](/api/applications/),
or the models available in [KerasCV](/keras_cv/) and [KerasHub](/keras_hub/).


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
import keras
from keras import layers
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

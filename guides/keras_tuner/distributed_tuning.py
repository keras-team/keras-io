"""
Title: Distributed hyperparameter tuning
Authors: Tom O'Malley, Haifeng Jin
Date created: 2019/10/24
Last modified: 2021/06/02
Description: Tuning the hyperparameters of the models with multiple GPUs and multiple machines.
"""

"""shell
pip install keras-tuner -q
"""

"""
## Introduction

KerasTuner makes it easy to perform distributed hyperparameter search. No
changes to your code are needed to scale up from running single-threaded
locally to running on dozens or hundreds of workers in parallel. Distributed
KerasTuner uses a chief-worker model. The chief runs a service to which the
workers report results and query for the hyperparameters to try next. The chief
should be run on a single-threaded CPU instance (or alternatively as a separate
process on one of the workers).

### Configuring distributed mode

Configuring distributed mode for KerasTuner only requires setting three
environment variables:

**KERASTUNER_TUNER_ID**: This should be set to "chief" for the chief process.
Other workers should be passed a unique ID (by convention, "tuner0", "tuner1",
etc).

**KERASTUNER_ORACLE_IP**: The IP address or hostname that the chief service
should run on. All workers should be able to resolve and access this address.

**KERASTUNER_ORACLE_PORT**: The port that the chief service should run on. This
can be freely chosen, but must be a port that is accessible to the other
workers. Instances communicate via the [gRPC](https://www.grpc.io) protocol.

The same code can be run on all workers. Additional considerations for
distributed mode are:

- All workers should have access to a centralized file system to which they can
write their results.
- All workers should be able to access the necessary training and validation
data needed for tuning.
- To support fault-tolerance, `overwrite` should be kept as `False` in
`Tuner.__init__` (`False` is the default).

Example bash script for chief service (sample code for `run_tuning.py` at
bottom of page):

```
export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
python run_tuning.py
```

Example bash script for worker:

```
export KERASTUNER_TUNER_ID="tuner0"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
python run_tuning.py
```
"""

"""
### Data parallelism with `tf.distribute`

KerasTuner also supports data parallelism via
[tf.distribute](https://www.tensorflow.org/tutorials/distribute/keras). Data
parallelism and distributed tuning can be combined. For example, if you have 10
workers with 4 GPUs on each worker, you can run 10 parallel trials with each
trial training on 4 GPUs by using
[tf.distribute.MirroredStrategy](
https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy).
You can also run each trial on TPUs via
[tf.distribute.TPUStrategy](
https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy).
Currently
[tf.distribute.MultiWorkerMirroredStrategy](
https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy)
is not supported, but support for this is on the roadmap.


### Example code

When the enviroment variables described above are set, the example below will
run distributed tuning and use data parallelism within each trial via
`tf.distribute`. The example loads MNIST from `tensorflow_datasets` and uses
[Hyperband](https://arxiv.org/abs/1603.06560) for the hyperparameter
search.
"""


import keras_tuner
import tensorflow as tf
import numpy as np


def build_model(hp):
    """Builds a convolutional model."""
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = inputs
    for i in range(hp.Int("conv_layers", 1, 3, default=3)):
        x = tf.keras.layers.Conv2D(
            filters=hp.Int("filters_" + str(i), 4, 32, step=4, default=8),
            kernel_size=hp.Int("kernel_size_" + str(i), 3, 5),
            activation="relu",
            padding="same",
        )(x)

        if hp.Choice("pooling" + str(i), ["max", "avg"]) == "max":
            x = tf.keras.layers.MaxPooling2D()(x)
        else:
            x = tf.keras.layers.AveragePooling2D()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    if hp.Choice("global_pooling", ["max", "avg"]) == "max":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer = hp.Choice("optimizer", ["adam", "sgd"])
    model.compile(
        optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


tuner = keras_tuner.Hyperband(
    hypermodel=build_model,
    objective="val_accuracy",
    max_epochs=2,
    factor=3,
    hyperband_iterations=1,
    distribution_strategy=tf.distribute.MirroredStrategy(),
    directory="results_dir",
    project_name="mnist",
    overwrite=True,
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape the images to have the channel dimension.
x_train = (x_train.reshape(x_train.shape + (1,)) / 255.0)[:1000]
y_train = y_train.astype(np.int64)[:1000]
x_test = (x_test.reshape(x_test.shape + (1,)) / 255.0)[:100]
y_test = y_test.astype(np.int64)[:100]

tuner.search(
    x_train,
    y_train,
    steps_per_epoch=600,
    validation_data=(x_test, y_test),
    validation_steps=100,
    callbacks=[tf.keras.callbacks.EarlyStopping("val_accuracy")],
)

"""
Title: Writing your own Tuner to support a custom training loop
Authors: Tom O'Malley, Haifeng Jin
Date created: 2019/10/28
Last modified: 2021/06/02
Description: Subclassing the Tuner class in KerasTuner for more customization like custom training loops.
"""

"""
## Introduction

The `Tuner` class at `keras_tuner.engine.tuner.Tuner` can be subclassed to
support advanced uses such as:

* Custom training loops (GANs, reinforement learning, etc.)
* Adding hyperparameters outside of the model builing function (preprocessing,
data augmentation, test time augmentation, etc.)

This tutorial will not cover subclassing to support non-Keras models. To
accomplish this, you can subclass the `keras_tuner.engine.base_tuner.BaseTuner`
class (See `keras_tuner.tuners.sklearn.Sklearn` for an example).

## Understanding the search process

`Tuner.search` can be passed any arguments. These arguments will be passed
directly to `Tuner.run_trial`, along with a `Trial` object that contains
information about the current trial, including hyperparameters and the status
of the trial. Typically, `Tuner.run_trial` is the only method that users need
to override when subclassing `Tuner`.

## Overriding `run_trial`

There are two ways to write `run_trial`. One is to leverage `Tuner`'s built-in
callback hooks, which send the value of the `objective` to the `Oracle` and
save the latest state of the Model. These hooks are:

* `self.on_epoch_end`: Must be called. Reports results to the `Oracle` and
saves the Model. The `logs` dictionary passed to this method must contain the
`objective` name.
* `self.on_epoch_begin`, `self.on_batch_begin`, `self.on_batch_end`: Optional.
These methods do nothing in `Tuner`, but are useful to provide as hooks if you
expect users of your subclass to create their own subclasses that override
these parts of the training process.


```python
class MyTuner(kt.Tuner):

    def run_trial(self, trial, ...):
        model = self.hypermodel.build(trial.hyperparameters)
        for epoch in range(10):
              epoch_loss = ...
              self.on_epoch_end(trial, model, epoch, logs={'loss': epoch_loss})
```

Alternatively, you can instead directly call the methods used to report results
to the `Oracle` and save the Model. This can allow more flexibility for use
cases where there is no natural concept of epoch or where you do not want to
report results to the `Oracle` after each epoch. These methods are:

* `self.oracle.update_trial`: Reports current results to the `Oracle`. The
`metrics` dictionary passed to this method must contain the `objective` name.
* `self.save_model`: Saves the trained model.

```python
class MyTuner(kt.Tuner):

    def run_trial(self, trial, ...):
        model = self.hypermodel.build(trial.hyperparameters)
        score = ...
        self.oracle.update_trial(trial.trial_id, {'score': score})
        self.save_model(trial.trial_id, model)
```

### Adding HyperParameters during preprocessing, evaluation, etc.

New `HyperParameter`s can be defined anywhere in `run_trial`, in the same way
that `HyperParameter`s are defined in a `HyperModel`. These hyperparameters
take on their default value the first time they are encountered, and thereafter
are tuned by the `Oracle`.

```python
class MyTuner(kt.Tuner):

    def run_trial(self, trial, ...):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)

        batch_size = hp.Int('batch_size', 32, 128, step=32)
        random_flip = hp.Boolean('random_flip')
        ...
```

### End-to-end Example:

"""
import keras_tuner as kt
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


class MyTuner(kt.Tuner):
    def run_trial(self, trial, train_ds):
        hp = trial.hyperparameters

        # Hyperparameters can be added anywhere inside `run_trial`.
        # When the first trial is run, they will take on their default values.
        # Afterwards, they will be tuned by the `Oracle`.
        train_ds = train_ds.batch(hp.Int("batch_size", 32, 128, step=32, default=64))

        model = self.hypermodel.build(trial.hyperparameters)
        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        optimizer = tf.keras.optimizers.Adam(lr)
        epoch_loss_metric = tf.keras.metrics.Mean()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        # @tf.function
        def run_train_step(data):
            images = tf.dtypes.cast(data[0], "float32") / 255.0
            labels = data[1]
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)
            return loss

        # `self.on_epoch_end` reports results to the `Oracle` and saves the
        # current state of the Model. The other hooks called here only log values
        # for display but can also be overridden. For use cases where there is no
        # natural concept of epoch, you do not have to call any of these hooks. In
        # this case you should instead call `self.oracle.update_trial` and
        # `self.oracle.save_model` manually.
        for epoch in range(2):
            print("Epoch: {}".format(epoch))

            self.on_epoch_begin(trial, model, epoch, logs={})
            for batch, data in enumerate(train_ds):
                self.on_batch_begin(trial, model, batch, logs={})
                batch_loss = float(run_train_step(data))
                self.on_batch_end(trial, model, batch, logs={"loss": batch_loss})

                if batch % 100 == 0:
                    loss = epoch_loss_metric.result().numpy()
                    print("Batch: {}, Average Loss: {}".format(batch, loss))

            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial, model, epoch, logs={"loss": epoch_loss})
            epoch_loss_metric.reset_states()


tuner = MyTuner(
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective("loss", "min"), max_trials=2
    ),
    hypermodel=build_model,
    directory="results",
    project_name="mnist_custom_training",
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape the images to have the channel dimension.
x_train = x_train.reshape(x_train.shape + (1,))[:1000]
y_train = y_train.astype(np.int64)[:1000]

mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

tuner.search(train_ds=mnist_train)

best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)

best_model = tuner.get_best_models()[0]

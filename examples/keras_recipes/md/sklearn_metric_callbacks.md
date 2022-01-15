# Evaluating and exporting scikit-learn metrics in a Keras callback

**Author:** [lukewood](https://lukewood.xyz)<br>
**Date created:** 10/07/2021<br>
**Last modified:** 10/07/2021<br>
**Description:** This example shows how to use Keras callbacks to evaluate and export non-TensorFlow based metrics.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/sklearn_metric_callbacks.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/sklearn_metric_callbacks.py)



---
## Introduction

[Keras callbacks](https://keras.io/api/callbacks/) allow for the execution of arbitrary
code at various stages of the Keras training process.  While Keras offers first-class
support for metric evaluation, [Keras metrics](https://keras.io/api/metrics/) may only
rely on TensorFlow code internally.

While there are TensorFlow implementations of many metrics online, some metrics are
implemented using [NumPy](https://numpy.org/) or another Python-based numerical computation library.
By performing metric evaluation inside of a Keras callback, we can leverage any existing
metric, and ultimately export the result to TensorBoard.

---
## Jaccard score metric

This example makes use of a sklearn metric, `sklearn.metrics.jaccard_score()`, and
writes the result to TensorBoard using the `tf.summary` API.

This template can be modified slightly to make it work with any existing sklearn metric.


```python
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.metrics import jaccard_score
import numpy as np
import os


class JaccardScoreCallback(keras.callbacks.Callback):
    """Computes the Jaccard score and logs the results to TensorBoard."""

    def __init__(self, model, x_test, y_test, log_dir):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.keras_metric = tf.keras.metrics.Mean("jaccard_score")
        self.epoch = 0
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, model.name)
        )

    def on_epoch_end(self, batch, logs=None):
        self.epoch += 1
        self.keras_metric.reset_state()
        predictions = self.model.predict(self.x_test)
        jaccard_value = jaccard_score(
            np.argmax(predictions, axis=-1), self.y_test, average=None
        )
        self.keras_metric.update_state(jaccard_value)
        self._write_metric(
            self.keras_metric.name, self.keras_metric.result().numpy().astype(float)
        )

    def _write_metric(self, name, value):
        with self.summary_writer.as_default():
            tf.summary.scalar(
                name, value, step=self.epoch,
            )
            self.summary_writer.flush()

```

---
## Sample usage

Let's test our `JaccardScoreCallback` class with a Keras model.


```python
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
callbacks = [JaccardScoreCallback(model, x_test, np.argmax(y_test, axis=-1), "logs")]
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    callbacks=callbacks,
)
```

<div class="k-default-codeblock">
```
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                16010     
=================================================================
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0
_________________________________________________________________
Epoch 1/15
422/422 [==============================] - 6s 14ms/step - loss: 0.3661 - accuracy: 0.8895 - val_loss: 0.0823 - val_accuracy: 0.9765
Epoch 2/15
422/422 [==============================] - 6s 14ms/step - loss: 0.1119 - accuracy: 0.9653 - val_loss: 0.0620 - val_accuracy: 0.9823
Epoch 3/15
422/422 [==============================] - 6s 14ms/step - loss: 0.0841 - accuracy: 0.9742 - val_loss: 0.0488 - val_accuracy: 0.9873
Epoch 4/15
422/422 [==============================] - 6s 14ms/step - loss: 0.0696 - accuracy: 0.9787 - val_loss: 0.0404 - val_accuracy: 0.9888
Epoch 5/15
422/422 [==============================] - 6s 14ms/step - loss: 0.0615 - accuracy: 0.9813 - val_loss: 0.0406 - val_accuracy: 0.9897
Epoch 6/15
422/422 [==============================] - 6s 13ms/step - loss: 0.0565 - accuracy: 0.9826 - val_loss: 0.0373 - val_accuracy: 0.9900
Epoch 7/15
422/422 [==============================] - 6s 14ms/step - loss: 0.0520 - accuracy: 0.9833 - val_loss: 0.0369 - val_accuracy: 0.9898
Epoch 8/15
422/422 [==============================] - 6s 14ms/step - loss: 0.0488 - accuracy: 0.9851 - val_loss: 0.0353 - val_accuracy: 0.9905
Epoch 9/15
422/422 [==============================] - 6s 14ms/step - loss: 0.0440 - accuracy: 0.9861 - val_loss: 0.0347 - val_accuracy: 0.9893
Epoch 10/15
422/422 [==============================] - 6s 14ms/step - loss: 0.0424 - accuracy: 0.9871 - val_loss: 0.0294 - val_accuracy: 0.9907
Epoch 11/15
422/422 [==============================] - 6s 14ms/step - loss: 0.0402 - accuracy: 0.9874 - val_loss: 0.0340 - val_accuracy: 0.9903
Epoch 12/15
422/422 [==============================] - 6s 13ms/step - loss: 0.0382 - accuracy: 0.9878 - val_loss: 0.0290 - val_accuracy: 0.9917
Epoch 13/15
422/422 [==============================] - 6s 14ms/step - loss: 0.0358 - accuracy: 0.9886 - val_loss: 0.0286 - val_accuracy: 0.9923
Epoch 14/15
422/422 [==============================] - 6s 13ms/step - loss: 0.0349 - accuracy: 0.9885 - val_loss: 0.0282 - val_accuracy: 0.9918
Epoch 15/15
422/422 [==============================] - 6s 14ms/step - loss: 0.0323 - accuracy: 0.9899 - val_loss: 0.0283 - val_accuracy: 0.9922

<tensorflow.python.keras.callbacks.History at 0x7f62fc5786d0>

```
</div>
If you now launch a TensorBoard instance using `tensorboard --logdir=logs`, you will
see the `jaccard_score` metric alongside any other exported metrics!

![TensorBoard Jaccard Score](https://i.imgur.com/T4qzrdn.png)

---
## Conclusion

Many ML practitioners and researchers rely on metrics that may not yet have a TensorFlow
implementation. Keras users can still leverage the wide variety of existing metric
implementations in other frameworks by using a Keras callback.  These metrics can be
exported, viewed and analyzed in the TensorBoard like any other metric.

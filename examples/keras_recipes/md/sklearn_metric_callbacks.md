# Evaluating and exporting scikit-learn metrics in a Keras callback

**Author:** [lukewood](https://lukewood.xyz)<br>
**Date created:** 10/07/2021<br>
**Last modified:** 11/17/2023<br>
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
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras as keras
from keras import layers
from sklearn.metrics import jaccard_score
import numpy as np
import os


class JaccardScoreCallback(keras.callbacks.Callback):
    """Computes the Jaccard score and logs the results to TensorBoard."""

    def __init__(self, name, x_test, y_test, log_dir):
        self.x_test = x_test
        self.y_test = y_test
        self.keras_metric = keras.metrics.Mean("jaccard_score")
        self.epoch = 0
        self.summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, name))

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
                name,
                value,
                step=self.epoch,
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
callbacks = [
    JaccardScoreCallback(model.name, x_test, np.argmax(y_test, axis=-1), "logs")
]
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

```
</div>
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │        <span style="color: #00af00; text-decoration-color: #00af00">320</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │     <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1600</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1600</span>)              │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)                │     <span style="color: #00af00; text-decoration-color: #00af00">16,010</span> │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,826</span> (136.04 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,826</span> (136.04 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



<div class="k-default-codeblock">
```
Epoch 1/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7706 - loss: 0.7534 - val_accuracy: 0.9768 - val_loss: 0.0842
Epoch 2/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9627 - loss: 0.1228 - val_accuracy: 0.9862 - val_loss: 0.0533
Epoch 3/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9739 - loss: 0.0854 - val_accuracy: 0.9870 - val_loss: 0.0466
Epoch 4/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9787 - loss: 0.0676 - val_accuracy: 0.9892 - val_loss: 0.0416
Epoch 5/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9818 - loss: 0.0590 - val_accuracy: 0.9892 - val_loss: 0.0396
Epoch 6/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9834 - loss: 0.0534 - val_accuracy: 0.9920 - val_loss: 0.0341
Epoch 7/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9837 - loss: 0.0528 - val_accuracy: 0.9907 - val_loss: 0.0358
Epoch 8/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.9847 - loss: 0.0466 - val_accuracy: 0.9908 - val_loss: 0.0327
Epoch 9/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 18ms/step - accuracy: 0.9873 - loss: 0.0397 - val_accuracy: 0.9912 - val_loss: 0.0346
Epoch 10/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.9862 - loss: 0.0419 - val_accuracy: 0.9913 - val_loss: 0.0315
Epoch 11/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9880 - loss: 0.0370 - val_accuracy: 0.9915 - val_loss: 0.0309
Epoch 12/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9880 - loss: 0.0377 - val_accuracy: 0.9912 - val_loss: 0.0318
Epoch 13/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9889 - loss: 0.0347 - val_accuracy: 0.9930 - val_loss: 0.0293
Epoch 14/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.9896 - loss: 0.0333 - val_accuracy: 0.9913 - val_loss: 0.0326
Epoch 15/15
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
 422/422 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.9908 - loss: 0.0282 - val_accuracy: 0.9925 - val_loss: 0.0303

<keras.src.callbacks.history.History at 0x17f0655a0>

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

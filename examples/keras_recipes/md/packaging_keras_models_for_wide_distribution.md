# Packaging Keras models for wide distribution using Functional Subclassing

**Author:** Martin Görner<br>
**Date created:** 2023-12-13<br>
**Last modified:** 2023-12-13<br>
**Description:** When sharing your deep learning models, package them using the Functional Subclassing pattern.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/packaging_keras_models_for_wide_distribution.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/packaging_keras_models_for_wide_distribution.py)



#Introduction

Keras is the ideal framework for sharing your cutting-edge deep learning models, in a
library of pre-trained (or not) models. Millions of ML engineers are fluent in the
familiar Keras API, making your models  accessible to a global community, whatever their
preferred backend (Jax, PyTorch or TensorFlow).

One of the benefits of the Keras API is that it lets users programmatically inspect or
edit a model, a feature that is necessary when creating new architectures or workflows
based on a pre-trained model.

When distributing models, the Keras team recommends packaging them using the **Functional
Subclassing** pattern. Models implemented in this way combine two benefits:

* They can be instantiated in the normal pythonic way:<br/>
`model = model_collection_xyz.AmazingModel()`

* They are Keras functional models which means that they have a programmatically
accessible graph of layers, for introspection or model surgery.

This guide explains [how to use](#functional-subclassing-model) the Functional
Subclassing pattern, and showcases its benefits for [programmatic model
introspection](#model-introspection) and [model surgery](#model-surgery). It also shows
two other best practices for sharable Keras models: [configuring
models](#unconstrained-inputs) for the widest range of supported inputs, for example
images of various sizes, and [using dictionary inputs](#model-with-dictionary-inputs) for
clarity in more complex models.

# Initialization

The backend must be selected before keras 3 is first imported.


```python
# backend selection in Keras 3. Supported values are jax, torch and tensorflow
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

print("Keras version", keras.version())
print("Keras is running on", keras.config.backend())
```

<div class="k-default-codeblock">
```
Keras version 3.0.1
Keras is running on tensorflow

```
</div>
# Dataset

Let's load an MNIST dataset so that we have something to train with.


```python
import tensorflow as tf  # only for tf.data

# tf.data is a great API for putting together a data stream.
# It works wether you use the TensorFlow, PyTorch or Jax backend,
# as long as you use it in the data stream only and not inside of a model.

BATCH_SIZE = 256

(x_train, train_labels), (x_test, test_labels) = tf.keras.datasets.mnist.load_data()

train_data = tf.data.Dataset.from_tensor_slices((x_train, train_labels))
train_data = train_data.map(
    lambda x, y: (tf.expand_dims(x, axis=-1), y)
)  # 1-channel monochrome
train_data = train_data.batch(BATCH_SIZE)
train_data = train_data.cache()
train_data = train_data.shuffle(5000, reshuffle_each_iteration=True)
train_data = train_data.repeat()

test_data = tf.data.Dataset.from_tensor_slices((x_test, test_labels))
test_data = test_data.map(
    lambda x, y: (tf.expand_dims(x, axis=-1), y)
)  # 1-channel monochrome
test_data = test_data.batch(10000)
test_data = test_data.cache()

STEPS_PER_EPOCH = len(train_labels) // BATCH_SIZE
EPOCHS = 5
```

# Functional Subclassing Model

The model is wrapped in a class so that end users can instantiate it normally by calling
the constructor `MnistModel()` rather than calling a factory function. The factory
function `make_functional_model()` exists
but users don't need to care about it. It is called in the constructor.


```python

class MnistModel(keras.Model):
    def __init__(self, **kwargs):
        inputs, outputs = self.make_functional_model()
        # A Keras Functional model is created by calling keras.Model(inputs, outputs)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    @staticmethod
    def make_functional_model():
        # Keras Functional model. This could have used Sequential as well,
        # Sequential is just syntactic sugar for simple functional models.

        # 1-channel monochrome input
        input = keras.layers.Input(shape=(None, None, 1), dtype="int8")
        # pixel format conversion from uint8 to float32
        y = keras.layers.Rescaling(1 / 255.0)(input)

        # 3 convolutional layers
        y = keras.layers.Conv2D(
            filters=16, kernel_size=3, padding="same", activation="relu"
        )(y)
        y = keras.layers.Conv2D(
            filters=32, kernel_size=6, padding="same", activation="relu", strides=2
        )(y)
        y = keras.layers.Conv2D(
            filters=48, kernel_size=6, padding="same", activation="relu", strides=2
        )(y)

        # 2 dense layers
        y = keras.layers.GlobalAveragePooling2D()(y)
        y = keras.layers.Dense(48, activation="relu")(y)
        y = keras.layers.Dropout(0.4)(y)
        output = keras.layers.Dense(
            10, activation="softmax", name="classification_head"  # 10 classes
        )(y)
        return input, output

```

Let's instantiate and train this model.


```python
model = MnistModel()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(
    train_data,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=test_data,
)
```

<div class="k-default-codeblock">
```
Epoch 1/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 8s 27ms/step - loss: 2.1268 - sparse_categorical_accuracy: 0.1960 - val_loss: 1.2316 - val_sparse_categorical_accuracy: 0.6353
Epoch 2/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - loss: 1.2627 - sparse_categorical_accuracy: 0.5526 - val_loss: 0.7594 - val_sparse_categorical_accuracy: 0.7746
Epoch 3/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - loss: 0.9504 - sparse_categorical_accuracy: 0.6759 - val_loss: 0.6268 - val_sparse_categorical_accuracy: 0.8054
Epoch 4/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - loss: 0.8255 - sparse_categorical_accuracy: 0.7250 - val_loss: 0.5817 - val_sparse_categorical_accuracy: 0.8162
Epoch 5/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 26ms/step - loss: 0.7410 - sparse_categorical_accuracy: 0.7568 - val_loss: 0.4887 - val_sparse_categorical_accuracy: 0.8454

```
</div>
# Unconstrained inputs

Notice, in the model definition above, that the input is specified with undefined
dimensions: `Input(shape=(None, None, 1)`

This allows the model to accept any image size as an input. However, this
only works if the loosely defined shape can be propagated through all the layers and
still determine the size of all weights.

* So if you have a model architecture that can handle different input sizes
with the same weights (like here), then your users will be able to instantiate it without
parameters:<br/> `model = MnistModel()`

* If on the other hand, the model must provision different weights for different input
sizes, you will have to ask your users to specify the size in the constructor:<br/>
`model = ModelXYZ(input_size=...)`

# Model introspection

Keras maintains a programmatically accessible graph of layers for every model. It can be
used for introspection and is accessed through the `model.layers` or `layer.layers`
attribute. The utility function `model.summary()` also uses this mechanism internally.


```python
model = MnistModel()

# Model summary works
model.summary()


# Recursively walking the layer graph works as well
def walk_layers(layer):
    if hasattr(layer, "layers"):
        for layer in layer.layers:
            walk_layers(layer)
    else:
        print(layer.name)


print("\nWalking model layers:\n")
walk_layers(model)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "mnist_model_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ input_layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)     │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ rescaling_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Rescaling</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)     │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">160</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">18,464</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">55,344</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ global_average_pooling2d_1      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)                │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>)        │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)                │      <span style="color: #00af00; text-decoration-color: #00af00">2,352</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)                │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ classification_head (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)                │        <span style="color: #00af00; text-decoration-color: #00af00">490</span> │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">76,810</span> (300.04 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">76,810</span> (300.04 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
<div class="k-default-codeblock">
```
Walking model layers:
```
</div>
    
<div class="k-default-codeblock">
```
input_layer_1
rescaling_1
conv2d_3
conv2d_4
conv2d_5
global_average_pooling2d_1
dense_1
dropout_1
classification_head

```
</div>
# Model surgery

End users might want to instantiate the model from your library but modify it before use.
Functional models have a programmatically accessible graph of layers. Edits are possible
by slicing and splicing the graph and creating a new functional model.

The alternative is to fork the model code and make the modifications but that forces
users to then maintain their fork indefinitely.

Example: instantiate the model but change the classification head to do a binary
classification, "0" or "not 0", instead of the original 10-way digits classification.


```python
model = MnistModel()

input = model.input
# cut before the classification head
y = model.get_layer("classification_head").input

# add a new classification head
output = keras.layers.Dense(
    1,  # single class for binary classification
    activation="sigmoid",
    name="binary_classification_head",
)(y)

# create a new functional model
binary_model = keras.Model(input, output)

binary_model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ input_layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)     │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ rescaling_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Rescaling</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)     │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">160</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">18,464</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">55,344</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ global_average_pooling2d_2      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)                │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>)        │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)                │      <span style="color: #00af00; text-decoration-color: #00af00">2,352</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">48</span>)                │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ binary_classification_head      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                 │         <span style="color: #00af00; text-decoration-color: #00af00">49</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                         │                           │            │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">76,369</span> (298.32 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">76,369</span> (298.32 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



We can now train the new model as a binary classifier.


```python
# new dataset with 0 / 1 labels (1 = digit '0', 0 = all other digits)
bin_train_data = train_data.map(
    lambda x, y: (x, tf.cast(tf.math.equal(y, tf.zeros_like(y)), dtype=tf.uint8))
)
bin_test_data = test_data.map(
    lambda x, y: (x, tf.cast(tf.math.equal(y, tf.zeros_like(y)), dtype=tf.uint8))
)

# appropriate loss and metric for binary classification
binary_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"]
)

history = binary_model.fit(
    bin_train_data,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=bin_test_data,
)
```

<div class="k-default-codeblock">
```
Epoch 1/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 7s 27ms/step - binary_accuracy: 0.8860 - loss: 0.3981 - val_binary_accuracy: 0.9020 - val_loss: 0.2744
Epoch 2/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - binary_accuracy: 0.9002 - loss: 0.2745 - val_binary_accuracy: 0.9019 - val_loss: 0.2087
Epoch 3/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - binary_accuracy: 0.9144 - loss: 0.2078 - val_binary_accuracy: 0.9293 - val_loss: 0.2038
Epoch 4/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - binary_accuracy: 0.9337 - loss: 0.1881 - val_binary_accuracy: 0.9433 - val_loss: 0.1656
Epoch 5/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 26ms/step - binary_accuracy: 0.9495 - loss: 0.1524 - val_binary_accuracy: 0.9608 - val_loss: 0.1142

```
</div>
# Model with dictionary inputs

In more complex models, with multiple inputs, structuring the inputs as a dictionary can
improve readability and usability. This is straightforward to do with a functional model:


```python

class MnistDictModel(keras.Model):
    def __init__(self, **kwargs):
        inputs, outputs = self.make_functional_model()
        # A Keras Functional model is created by calling keras.Model(inputs, outputs)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    @staticmethod
    def make_functional_model():
        #
        # The input is a dictionary
        #
        input = {
            "image": keras.layers.Input(
                shape=(None, None, 1),  # 1-channel monochrome
                dtype="int8",
                name="image",
            )
        }

        # pixel format conversion from uint8 to float32
        y = keras.layers.Rescaling(1 / 255.0)(input["image"])

        # 3 conv layers
        y = keras.layers.Conv2D(
            filters=16, kernel_size=3, padding="same", activation="relu"
        )(y)
        y = keras.layers.Conv2D(
            filters=32, kernel_size=6, padding="same", activation="relu", strides=2
        )(y)
        y = keras.layers.Conv2D(
            filters=48, kernel_size=6, padding="same", activation="relu", strides=2
        )(y)

        # 2 dense layers
        y = keras.layers.GlobalAveragePooling2D()(y)
        y = keras.layers.Dense(48, activation="relu")(y)
        y = keras.layers.Dropout(0.4)(y)
        output = keras.layers.Dense(
            10, activation="softmax", name="classification_head"  # 10 classes
        )(y)

        return input, output

```

We can now train the model on inputs structured as a dictionary.


```python
model = MnistDictModel()

# reformat the dataset as a dictionary
dict_train_data = train_data.map(lambda x, y: ({"image": x}, y))
dict_test_data = test_data.map(lambda x, y: ({"image": x}, y))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(
    dict_train_data,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=dict_test_data,
)
```

<div class="k-default-codeblock">
```
Epoch 1/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 7s 27ms/step - loss: 2.1208 - sparse_categorical_accuracy: 0.2111 - val_loss: 1.3459 - val_sparse_categorical_accuracy: 0.5110
Epoch 2/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - loss: 1.3405 - sparse_categorical_accuracy: 0.5147 - val_loss: 0.8680 - val_sparse_categorical_accuracy: 0.7303
Epoch 3/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - loss: 1.0184 - sparse_categorical_accuracy: 0.6462 - val_loss: 0.6867 - val_sparse_categorical_accuracy: 0.7928
Epoch 4/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - loss: 0.8784 - sparse_categorical_accuracy: 0.7036 - val_loss: 0.6201 - val_sparse_categorical_accuracy: 0.8091
Epoch 5/5
 234/234 ━━━━━━━━━━━━━━━━━━━━ 6s 25ms/step - loss: 0.8009 - sparse_categorical_accuracy: 0.7342 - val_loss: 0.5659 - val_sparse_categorical_accuracy: 0.8110

```
</div>
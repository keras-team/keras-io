# Keras layers API

Layers are the basic building blocks of neural networks in Keras.
A layer consists of a tensor-in tensor-out computation function (the layer's `call` method)
and some state, held in TensorFlow variables (the layer's *weights*).

A Layer instance is callable, much like a function:

```python
from tensorflow.keras import layers

layer = layers.Dense(32, activation='relu')
inputs = tf.random.uniform(shape=(10, 20))
outputs = layer(inputs)
```

Unlike a function, though, layers maintain a state, updated when the layer receives data
during training, and stored in `layer.weights`:

```python
>>> layer.weights
[<tf.Variable 'dense/kernel:0' shape=(20, 32) dtype=float32>,
 <tf.Variable 'dense/bias:0' shape=(32,) dtype=float32>]
```

---

## Creating custom layers

While Keras offers a wide range of built-in layers, they don't cover
ever possible use case. Creating custom layers is very common, and very easy.

See the guide
[Making new layers and models via subclassing](/guides/making_new_layers_and_models_via_subclassing)
for an extensive overview, and refer to the documentation for [the base `Layer` class](base_layer).

---

## Layers API overview

{{toc}}

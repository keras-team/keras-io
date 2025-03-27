# About Keras 3

Keras is a deep learning API written in Python and capable of running on top of either [JAX](https://jax.readthedocs.io/),
[TensorFlow](https://github.com/tensorflow/tensorflow),
or [PyTorch](https://pytorch.org/).

Keras is:

- **Simple** -- but not simplistic. Keras reduces developer *cognitive load* to free you to focus on the parts of the problem that really matter.
- **Flexible** -- Keras adopts the principle of *progressive disclosure of complexity*: simple workflows should be quick and easy,
while arbitrarily advanced workflows should be *possible* via a clear path that builds upon what you've already learned.
- **Powerful** -- Keras provides industry-strength performance and scalability: it is used by organizations including NASA, YouTube, or Waymo.

---

## Keras 3 is a multi-framework deep learning API

As a multi-framework API, Keras can be used to develop modular components that are compatible with any framework -- JAX, TensorFlow, or PyTorch.

This approach has several key benefits:

- **Always get the best performance for your models.** In our benchmarks,
we found that JAX typically delivers the best training and inference performance
on GPU, TPU, and CPU -- but results vary from model to model, as non-XLA
TensorFlow is occasionally faster on GPU. The ability to dynamically select
the backend that will deliver the best performance for your model
*without having to change anything to your code* means you're always guaranteed
to train and serve with the highest achievable efficiency.
- **Maximize available ecosystem surface for your models.** Any Keras
model can be instantiated as a PyTorch `Module`, can be exported as a TensorFlow
`SavedModel`, or can be instantiated as a stateless JAX function. That means
that you can use your Keras models with PyTorch ecosystem packages,
with the full range of TensorFlow deployment & production tools, and with
JAX large-scale TPU training infrastructure. Write one `model.py` using
Keras APIs, and get access to everything the ML world has to offer.
- **Maximize distribution for your open-source model releases.** Want to
release a pretrained model? Want as many people as possible
to be able to use it? If you implement it in pure TensorFlow or PyTorch,
it will be usable by roughly half of the market.
If you implement it in Keras, it is instantly usable by anyone regardless
of their framework of choice (even if they're not Keras users).
Twice the impact at no added development cost.
- **Use data pipelines from any source.** The Keras
`fit()`/`evaluate()`/`predict()` routines are compatible with `tf.data.Dataset` objects,
with PyTorch `DataLoader` objects, with NumPy arrays, Pandas dataframes --
regardless of the backend you're using. You can train a Keras + TensorFlow
model on a PyTorch `DataLoader` or train a Keras + PyTorch model on a
`tf.data.Dataset`.


---

## First contact with Keras

The core data structures of Keras are __layers__ and __models__.
The simplest type of model is the [`Sequential` model](/guides/sequential_model/), a linear stack of layers.
For more complex architectures, you should use the [Keras functional API](/guides/functional_api/),
which allows to build arbitrary graphs of layers, or [write models entirely from scratch via subclasssing](/guides/making_new_layers_and_models_via_subclassing/).

Here is the `Sequential` model:

```python
import keras

model = keras.Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras import layers

model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))
```

Once your model looks good, configure its learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. The Keras philosophy is to keep simple things simple,
while allowing the user to be fully in control when they need to
(the ultimate control being the easy extensibility of the source code via subclassing).

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Evaluate your test loss and metrics in one line:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

Or generate predictions on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

What you just saw is the most elementary way to use Keras.

However, Keras is also a highly-flexible framework suitable to iterate on state-of-the-art research ideas.
Keras follows the principle of **progressive disclosure of complexity**: it makes it easy to get started,
yet it makes it possible to handle arbitrarily advanced use cases,
only requiring incremental learning at each step.

In much the same way that you were able to train and evaluate a simple neural network above in a few lines,
you can use Keras to quickly develop new training procedures or state-of-the-art model architectures.

Here's an example of a custom Keras layer -- which can be used in low-level
workflows in JAX, TensorFlow, or PyTorch, interchangeably:

```python
import keras
from keras import ops

class TokenAndPositionEmbedding(keras.Layer):
    def __init__(self, max_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embed = self.add_weight(
            shape=(vocab_size, embed_dim),
            initializer="random_uniform",
            trainable=True,
        )
        self.position_embed = self.add_weight(
            shape=(max_length, embed_dim),
            initializer="random_uniform",
            trainable=True,
        )

    def call(self, token_ids):
        # Embed positions
        length = token_ids.shape[-1]
        positions = ops.arange(0, length, dtype="int32")
        positions_vectors = ops.take(self.position_embed, positions, axis=0)
        # Embed tokens
        token_ids = ops.cast(token_ids, dtype="int32")
        token_vectors = ops.take(self.token_embed, token_ids, axis=0)
        # Sum both
        embed = token_vectors + positions_vectors
        # Normalize embeddings
        power_sum = ops.sum(ops.square(embed), axis=-1, keepdims=True)
        return embed / ops.sqrt(ops.maximum(power_sum, 1e-7))
```

For more in-depth tutorials about Keras, you can check out:

- [Introduction to Keras for engineers](/getting_started/intro_to_keras_for_engineers/)
- [Developer guides](/guides/)

---

## Support

You can ask questions and join the development discussion on the [Keras Google group](https://groups.google.com/forum/#!forum/keras-users).

You can also post **bug reports and feature requests** (only) in [GitHub issues](https://github.com/keras-team/keras/issues).
Make sure to read [our guidelines](https://github.com/keras-team/keras-io/blob/master/templates/contributing.md) first.

---

## Why this name, Keras?

Keras (κέρας) means _horn_ in ancient Greek. It is a reference to a literary image from ancient Greek and Latin literature, first found in the _Odyssey_, where dream spirits (_Oneiroi_, singular _Oneiros_) are divided between those who deceive dreamers with false visions, who arrive to Earth through a gate of ivory, and those who announce a future that will come to pass, who arrive through a gate of horn. It's a play on the words κέρας (horn) / κραίνω (fulfill), and ἐλέφας (ivory) / ἐλεφαίρομαι (deceive).

Keras was initially developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).

>_"Oneiroi are beyond our unravelling - who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; **those that come out through polished horn have truth behind them, to be accomplished for men who see them.**"_ Homer, Odyssey 19. 562 ff (Shewring translation).

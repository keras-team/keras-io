# Keras Recommenders

<a class="github-button" href="https://github.com/keras-team/keras-rs" data-size="large" data-show-count="true" aria-label="Star keras-team/keras-rs on GitHub">Star</a>

![KerasRS](https://i.imgur.com/m1BX7Zd.png)

Keras Recommenders is a library for building recommender systems on top of
Keras 3. Keras Recommenders works natively with TensorFlow, JAX, or PyTorch. It
provides a collection of building blocks which help with the full workflow of
creating a recommender system. As it's built on Keras 3, models can be trained
and serialized in any framework and re-used in another without costly
migrations.

This library is an extension of the core Keras API; all high-level modules
receive that same level of polish as core Keras. If you are familiar with Keras,
congratulations! You already understand most of Keras Recommenders.

## Quick Links

- [Home page](https://keras.io/keras_rs)
- [Examples](https://keras.io/keras_rs/examples)
- [API documentation](https://keras.io/keras_rs/api)

## Quickstart

### Train your own cross network

Choose a backend:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch"!
```

Import KerasRS and other libraries:

```python
import keras
import keras_rs
import numpy as np
```

Define a simple model using the `FeatureCross` layer:

```python
vocabulary_size = 32
embedding_dim = 6

inputs = keras.Input(shape=(), name='indices', dtype="int32")
x0 = keras.layers.Embedding(
    input_dim=vocabulary_size,
    output_dim=embedding_dim
)(inputs)
x1 = keras_rs.layers.FeatureCross()(x0, x0)
x2 = keras_rs.layers.FeatureCross()(x0, x1)
output = keras.layers.Dense(units=10)(x2)
model = keras.Model(inputs, output)
```

Compile the model:

```python
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4)
)
```

Call `model.fit()` on dummy data:

```python
batch_size = 2
x = np.random.randint(0, vocabulary_size, size=(batch_size,))
y = np.random.random(size=(batch_size,))
model.fit(input_data, y=y)
```

### Use ranking losses and metrics

If your task is to rank items in a list, you can make use of the ranking losses
and metrics which KerasRS provides. Below, we use the pairwise hinge loss and
track the nDCG metric:

```python
model.compile(
    loss=keras_rs.losses.PairwiseHingeLoss(),
    metrics=[keras_rs.metrics.NDCG()]
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
)
```

## Installation

Keras Recommenders is available on PyPI as `keras-rs`:

```bash
pip install keras-rs
```

To try out the latest version of Keras Recommenders, you can use our nightly
package:

```bash
pip install keras-rs-nightly
```

Read [Getting started with Keras](https://keras.io/getting_started/) for more
information on installing Keras 3 and compatibility with different frameworks.

## Configuring your backend

If you have Keras 3 installed in your environment (see installation above), you
can use Keras Recommenders with any of JAX, TensorFlow and PyTorch. To do so,
set the `KERAS_BACKEND` environment variable. For example:

```shell
export KERAS_BACKEND=jax
```

Or in Colab, with:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_rs
```

## Compatibility

We follow [Semantic Versioning](https://semver.org/), and plan to provide
backwards compatibility guarantees both for code and saved models built with our
components. While we continue with pre-release `0.y.z` development, we may break
compatibility at any time and APIs should not be considered stable.

## Citing Keras Recommenders

If Keras Recommenders helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{kerasrecommenders2024,
  title={KerasRecommenders},
  author={Hertschuh, Fabien and  Chollet, Fran\c{c}ois and Sharma, Abheesht and others},
  year={2024},
  howpublished={\url{https://github.com/keras-team/keras-rs}},
}
```

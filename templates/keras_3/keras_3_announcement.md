After five months of extensive public beta testing,
we're excited to announce the official release of Keras 3.0.
Keras 3 is a full rewrite of Keras that enables you to
run your Keras workflows on top of either JAX, TensorFlow, or PyTorch, and that
unlocks brand new large-scale model training and deployment capabilities.
You can pick the framework that suits you best,
and switch from one to another based on your current goals.
You can also use Keras as a low-level cross-framework language
to develop custom components such as layers, models, or metrics
that can be used in native workflows in JAX, TensorFlow, or PyTorch — with one codebase.

---

## Welcome to multi-framework machine learning.

You're already familiar with the benefits of using Keras — it enables
high-velocity development via an obsessive focus on great UX, API design,
and debuggability. It's also a battle-tested framework that has been chosen
by over 2.5M developers and that powers some of the most sophisticated,
largest-scale ML systems in the world,
such as the Waymo self-driving fleet and the YouTube recommendation engine.
But what are the additional benefits of using the new multi-backend Keras 3?

- **Always get the best performance for your models.** In our benchmarks,
we found that JAX typically delivers the best training and inference performance
on GPU, TPU, and CPU — but results vary from model to model, as non-XLA
TensorFlow is occasionally faster on GPU. The ability to dynamically select
the backend that will deliver the best performance for your model
*without having to change anything to your code* means you're guaranteed
to train and serve with the highest achievable efficiency.
- **Unlock ecosystem optionality for your models.** Any Keras 3
model can be instantiated as a PyTorch `Module`, can be exported as a TensorFlow
`SavedModel`, or can be instantiated as a stateless JAX function. That means
that you can use your Keras 3 models with PyTorch ecosystem packages,
with the full range of TensorFlow deployment & production tools
(like TF-Serving, TF.js and TFLite), and with JAX large-scale
TPU training infrastructure. Write one `model.py` using
Keras 3 APIs, and get access to everything the ML world has to offer.
- **Leverage large-scale model parallelism & data parallelism with JAX.** Keras 3 includes
a brand new distribution API, the `keras.distribution` namespace,
currently implemented for the JAX backend (coming soon to the TensorFlow and PyTorch backends).
It makes it easy to do model parallelism, data parallelism, and combinations of both —
at arbitrary model scales and cluster scales.
Because it keeps the model definition, training logic,
and sharding configuration all separate from each other,
it makes your distribution workflow easy to develop and easy to maintain.
See our [starter guide](/guides/distribution/).
- **Maximize reach for your open-source model releases.** Want to
release a pretrained model? Want as many people as possible
to be able to use it? If you implement it in pure TensorFlow or PyTorch,
it will be usable by roughly half of the community.
If you implement it in Keras 3, it is instantly usable by anyone regardless
of their framework of choice (even if they're not Keras users themselves).
Twice the impact at no added development cost.
- **Use data pipelines from any source.** The Keras 3
`fit()`/`evaluate()`/`predict()` routines are compatible with `tf.data.Dataset` objects,
with PyTorch `DataLoader` objects, with NumPy arrays, Pandas dataframes —
regardless of the backend you're using. You can train a Keras 3 + TensorFlow
model on a PyTorch `DataLoader` or train a Keras 3 + PyTorch model on a
`tf.data.Dataset`.

---

## The full Keras API, available for JAX, TensorFlow, and PyTorch.

Keras 3 implements the full Keras API and makes it available
with TensorFlow, JAX, and PyTorch — over a hundred layers, dozens of metrics,
loss functions, optimizers, and callbacks, the Keras training and evaluation
loops, and the Keras saving & serialization infrastructure. All the APIs you
know and love are here.

Any Keras model that only uses built-in layers will immediately work with
all supported backends. In fact, your existing `tf.keras` models
that only use built-in layers can start running in JAX and PyTorch *right away*!
That's right, your codebase just gained a whole new set of capabilities.

<img class="irasto" src="https://s3.amazonaws.com/keras.io/img/keras_3/cross_framework_keras_3.jpg" />

---

## Author multi-framework layers, models, metrics...

Keras 3 enables you to create components
(like arbitrary custom layers or pretrained models) that will work the same
in any framework. In particular, Keras 3 gives you access
to the `keras.ops` namespace that works across all backends. It contains:

- **A full implementation of the NumPy API.**
Not something "NumPy-like" — just literally the
NumPy API, with the same functions and the same arguments.
You get `ops.matmul`, `ops.sum`, `ops.stack`, `ops.einsum`, etc.
- **A set of neural network-specific functions** that are absent from NumPy,
such as `ops.softmax`, `ops.binary_crossentropy`, `ops.conv`, etc.

As long as you only use ops from `keras.ops`, your custom layers,
custom losses, custom metrics, and custom optimizers
**will work with JAX, PyTorch, and TensorFlow — with the same code**.
That means that you can maintain only one
component implementation (e.g. a single `model.py`
together with a single checkpoint file), and you can use it in all frameworks,
with the exact same numerics.

<img class="irasto" src="https://s3.amazonaws.com/keras.io/img/keras_3/custom_component_authoring_keras_3.jpg" />

---

## ...that works seamlessly with any JAX, TensorFlow, and PyTorch workflow.

Keras 3 is not just intended for Keras-centric workflows
where you define a Keras model, a Keras optimizer, a Keras loss and metrics,
and you call `fit()`, `evaluate()`, and `predict()`.
It's also meant to work seamlessly with low-level backend-native workflows:
you can take a Keras model (or any other component, such as a loss or metric)
and start using it in a JAX training loop, a TensorFlow training loop,
or a PyTorch training loop, or as part of a JAX or PyTorch model,
with zero friction. Keras 3 provides exactly
the same degree of low-level implementation flexibility in JAX and PyTorch
as `tf.keras` previously did in TensorFlow.

You can:

- Write a low-level JAX training loop to train a Keras model
using an `optax` optimizer, `jax.grad`, `jax.jit`, `jax.pmap`.
- Write a low-level TensorFlow training loop to train a Keras model
using `tf.GradientTape` and `tf.distribute`.
- Write a low-level PyTorch training loop to train a Keras model
using a `torch.optim` optimizer, a `torch` loss function,
and the `torch.nn.parallel.DistributedDataParallel` wrapper.
- Use Keras layers in a PyTorch `Module` (because they are `Module` instances too!)
- Use any PyTorch `Module` in a Keras model as if it were a Keras layer.
- etc.

<img class="irasto" src="https://s3.amazonaws.com/keras.io/img/keras-core/custom_training_loops.jpg" />

---

## A new distribution API for large-scale data parallelism and model parallelism.

The models we've been working with have been getting larger and larger, so we wanted
to provide a Kerasic solution to the multi-device model sharding problem. The API we designed
keeps the model definition, the training logic, and the sharding configuration entirely separate from each
other, meaning that your models can be written as if they were going to run on a single device. You
can then add arbitrary sharding configurations to arbitrary models when it's time to train them.

Data parallelism (replicating a small model identically on multiple devices) can be handled in just two lines:

<img class="irasto" src="https://s3.amazonaws.com/keras.io/img/keras_3/keras_3_data_parallel.jpg" />

Model parallelism lets you specify sharding layouts for model variables and intermediate output tensors,
along multiple named dimensions. In the typical case, you would organize available devices as a 2D grid
(called a *device mesh*), where the first dimension is used for data parallelism and the second dimension
is used for model parallelism. You would then configure your model to be sharded along the model dimension
and replicated along the data dimension.

The API lets you configure the layout of every variable and every output tensor via regular expressions.
This makes it easy to quickly specify the same layout for entire categories of variables.

<img class="irasto" src="https://s3.amazonaws.com/keras.io/img/keras_3/keras_3_model_parallel.jpg" />

The new distribution API is intended to be multi-backend, but is only available for the JAX backend for the time
being. TensorFlow and PyTorch support is coming soon. Get started with [this guide](/guides/distribution/)!


---

## Pretrained models.

There's a wide range of pretrained models that
you can start using today with Keras 3.

All 40 Keras Applications models (the `keras.applications` namespace)
are available in all backends.
The vast array of pretrained models in [KerasCV](https://keras.io/api/keras_cv/)
and [KerasHub](https://keras.io/api/keras_hub/) also work with all backends. This includes:

- BERT
- OPT
- Whisper
- T5
- StableDiffusion
- YOLOv8
- SegmentAnything
- etc.

---

## Support for cross-framework data pipelines with all backends.

Multi-framework ML also means multi-framework data loading and preprocessing.
Keras 3 models can be trained using a wide range of
data pipelines — regardless of whether you're using the JAX, PyTorch, or
TensorFlow backends. It just works.

- `tf.data.Dataset` pipelines: the reference for scalable production ML.
- `torch.utils.data.DataLoader` objects.
- NumPy arrays and Pandas dataframes.
- Keras's own `keras.utils.PyDataset` objects.

---

## Progressive disclosure of complexity.

*Progressive disclosure of complexity* is the design principle at the heart
of the Keras API. Keras doesn't force you to follow
a single "true" way of building and training models. Instead, it enables
a wide range of different workflows, from the very high-level to the very
low-level, corresponding to different user profiles.

That means that you can start out with simple workflows — such as using
`Sequential` and `Functional` models and training them with `fit()` — and when
you need more flexibility, you can easily customize different components while
reusing most of your prior code. As your needs become more specific,
you don't suddenly fall off a complexity cliff and you don't need to switch
to a different set of tools.

We've brought this principle to all of our backends. For instance,
you can customize what happens in your training loop while still
leveraging the power of `fit()`, without having to write your own training loop
from scratch — just by overriding the `train_step` method.

Here's how it works in PyTorch and TensorFlow:

<img class="irasto" src="https://s3.amazonaws.com/keras.io/img/keras-core/customizing_fit.jpg" />

And [here's the link](http://keras.io/guides/custom_train_step_in_jax/) to the JAX version.

---

## A new stateless API for layers, models, metrics, and optimizers.

Do you enjoy [functional programming](https://en.wikipedia.org/wiki/Functional_programming)?
You're in for a treat.

All stateful objects in Keras (i.e. objects that own numerical variables that
get updated during training or evaluation) now have a stateless API, making it
possible to use them in JAX functions (which are required to be fully stateless):

- All layers and models have a `stateless_call()` method which mirrors `__call__()`.
- All optimizers have a `stateless_apply()` method which mirrors `apply()`.
- All metrics have a `stateless_update_state()` method which mirrors `update_state()`
and a `stateless_result()` method which mirrors `result()`.

These methods have no side-effects whatsoever: they take as input the current value
of the state variables of the target object, and return the update values as part
of their outputs, e.g.:

```python
outputs, updated_non_trainable_variables = layer.stateless_call(
    trainable_variables,
    non_trainable_variables,
    inputs,
)
```

You never have to implement these methods yourself — they're automatically available
as long as you've implemented the stateful version (e.g. `call()` or `update_state()`).

---

## Moving from Keras 2 to Keras 3

Keras 3 is highly backwards compatible with Keras 2:
it implements the full public API surface of Keras 2,
with a limited number of exceptions, listed [here](https://github.com/keras-team/keras/issues/18467).
Most users will not have to make any code change
to start running their Keras scripts on Keras 3.

Larger codebases are likely to require some code changes,
since they are more likely to run into one of the exceptions listed above,
and are more likely to have been using private APIs or deprecated APIs
(`tf.compat.v1.keras` namespace, `experimental` namespace, `keras.src` private namespace).
To help you move to Keras 3, we are releasing a complete [migration guide](/guides/migrating_to_keras_3/)
with quick fixes for all issues you might encounter.

You also have the option to ignore the changes in Keras 3 and just keep using Keras 2 with TensorFlow —
this can be a good option for projects that are not actively developed
but need to keep running with updated dependencies.
You have two possibilities:

1. If you were accessing `keras` as a standalone package,
just switch to using the Python package `tf_keras` instead,
which you can install via `pip install tf_keras`.
The code and API are wholly unchanged — it's Keras 2.15 with a different package name.
We will keep fixing bugs in `tf_keras` and we will keep regularly releasing new versions.
However, no new features or performance improvements will be added,
since the package is now in maintenance mode.
2. If you were accessing `keras` via `tf.keras`,
there are no immediate changes until TensorFlow 2.16.
TensorFlow 2.16+ will use Keras 3 by default.
In TensorFlow 2.16+, to keep using Keras 2, you can first install `tf_keras`,
and then export the environment variable `TF_USE_LEGACY_KERAS=1`.
This will direct TensorFlow 2.16+ to resolve tf.keras to the locally-installed `tf_keras` package.
Note that this may affect more than your own code, however:
it will affect any package importing `tf.keras` in your Python process.
To make sure your changes only affect your own code, you should use the `tf_keras` package. 

---

## Enjoy the library!

We're excited for you to try out the new Keras and improve your workflows by leveraging multi-framework ML.
Let us know how it goes: issues, points of friction, feature requests, or success stories —
we're eager to hear from you!

---

## FAQ

#### Q: Is Keras 3 compatible with legacy Keras 2?

Code developed with `tf.keras` can generally be run as-is with Keras 3
(with the TensorFlow backend). There's a limited number of incompatibilities you should be mindful
of, all addressed in [this migration guide](/guides/migrating_to_keras_3/).

When it comes to using APIs from `tf.keras` and Keras 3 side by side,
that is **not** possible — they're different packages, running on entirely separate engines.

### Q: Do pretrained models developed in legacy Keras 2 work with Keras 3?

Generally, yes. Any `tf.keras` model should work out of the box with Keras 3
with the TensorFlow backend (make sure to save it in the `.keras` v3 format).
In addition, if the model only
uses built-in Keras layers, then it will also work out of the box
with Keras 3 with the JAX and PyTorch backends.

If the model contains custom layers written using TensorFlow APIs,
it is usually easy to convert the code to be backend-agnostic.
For instance, it only took us a few hours to convert all 40
legacy `tf.keras` models from Keras Applications to be backend-agnostic.

### Q: Can I save a Keras 3 model in one backend and reload it in another backend?

Yes, you can. There is no backend specialization in saved `.keras` files whatsoever.
Your saved Keras models are framework-agnostic and can be reloaded with any backend.

However, note that reloading a model that contains custom components
with a different backend requires your custom components to be implemented
using backend-agnostic APIs, e.g. `keras.ops`.

### Q: Can I use Keras 3 components inside `tf.data` pipelines?

With the TensorFlow backend, Keras 3 is fully compatible with `tf.data`
(e.g. you can `.map()` a `Sequential` model into a `tf.data` pipeline).

With a different backend, Keras 3 has limited support for `tf.data`.
You won't be able to `.map()` arbitrary layers or models into a `tf.data`
pipeline. However, you will be able to use specific Keras 3
preprocessing layers with `tf.data`, such as `IntegerLookup` or
`CategoryEncoding`.

When it comes to using a `tf.data` pipeline (that does not use Keras)
to feed your call to `.fit()`, `.evaluate()` or `.predict()` —
that works out of the box with all backends.

### Q: Do Keras 3 models behave the same when run with different backends?

Yes, numerics are identical across backends.
However, keep in mind the following caveats:

- RNG behavior is different across different backends
(even after seeding — your results will be deterministic in each backend
but will differ across backends). So random weight initializations
values and dropout values will differ across backends.
- Due to the nature of floating-point implementations,
results are only identical up to `1e-7` precision in float32,
per function execution. So when training a model for a long time,
small numerical differences will accumulate and may end up resulting
in noticeable numerical differences.
- Due to lack of support for average pooling with asymmetric padding
in PyTorch, average pooling layers with `padding="same"`
may result in different numerics on border rows/columns.
This doesn't happen very often in practice —
out of 40 Keras Applications vision models, only one was affected.

### Q: Does Keras 3 support distributed training?

Data-parallel distribution is supported out of the box in JAX, TensorFlow,
and PyTorch. Model parallel distribution is supported out of the box for JAX
with the `keras.distribution` API.

**With TensorFlow:**

Keras 3 is compatible with `tf.distribute` —
just open a Distribution Strategy scope and create / train your model within it.
[Here's an example](http://keras.io/guides/distributed_training_with_tensorflow/).

**With PyTorch:**

Keras 3 is compatible with PyTorch's `DistributedDataParallel` utility.
[Here's an example](http://keras.io/guides/distributed_training_with_torch/).

**With JAX:**

You can do both data parallel and model parallel distribution in JAX using the `keras.distribution` API.
For instance, to do data parallel distribution, you only need the following code snippet:

```python
distribution = keras.distribution.DataParallel(devices=keras.distribution.list_devices())
keras.distribution.set_distribution(distribution)
```

For model parallel distribution, see [the following guide](/guides/distribution/).

You can also distribute training yourself via JAX APIs such as
`jax.sharding`. [Here's an example](http://keras.io/guides/distributed_training_with_jax/).

### Q: Can my custom Keras layers be used in native PyTorch `Modules` or with Flax `Modules`?

If they are only written using Keras APIs (e.g. the `keras.ops` namespace), then yes, your
Keras layers will work out of the box with native PyTorch and JAX code.
In PyTorch, just use your Keras layer like any other PyTorch `Module`.
In JAX, make sure to use the stateless layer API, i.e. `layer.stateless_call()`.

### Q: Will you add more backends in the future? What about framework XYZ?

We're open to adding new backends as long as the target framework has a large user base
or otherwise has some unique technical benefits to bring to the table.
However, adding and maintaining a new backend is a large burden,
so we're going to carefully consider each new backend candidate on a case by case basis,
and we're not likely to add many new backends. We will not add any new frameworks
that aren't yet well-established.
We are now potentially considering adding a backend written in [Mojo](https://www.modular.com/mojo).
If that's something you might find useful, please let the Mojo team know.

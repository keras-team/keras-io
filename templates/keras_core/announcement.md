We're excited to share with you a new library called **Keras Core**,
a preview version of the future of Keras.
In Fall 2023, this library will become Keras 3.0.
Keras Core is a full rewrite of the Keras codebase that rebases it
on top of a **modular backend architecture**.
It makes it possible to run Keras workflows on top of arbitrary frameworks —
starting with TensorFlow, JAX, and PyTorch.

Keras Core is also a drop-in replacement for `tf.keras`,
with near-full backwards compatibility with `tf.keras` code
when using the TensorFlow backend.
In the vast majority of cases you can just start importing it via
`import keras_core as keras` in place of `from tensorflow import keras`
and your existing code will run with no issue —
and generally with slightly improved performance, thanks to XLA compilation.

---

## Why we're making Keras multi-backend again

Keras Core is a big return to our multi-backend roots. Not so long ago,
Keras could run on top of Theano, TensorFlow, and CNTK (even MXNet!).
In 2018, we made the decision to refocus Keras development exclusively
on TensorFlow. At the time, TensorFlow was the only viable option
available: Theano and CNTK had discontinued development.
The added cost of supporting multiple backends was simply no longer worth it.

But in 2023, this is no longer true. According to large-scale developer surveys
such as the [2023 StackOverflow Developer Survey](https://survey.stackoverflow.co/2023/) and the
[2022 Kaggle Machine Learning & Data Science Survey](https://www.kaggle.com/c/kaggle-survey-2022)
(as well as other adoption metrics such as PyPI downloads, Conda downloads,
and Colab import statistics, which all paint the same picture),
TensorFlow has between 55% and 60% market share and is the top choice for
production ML, while PyTorch has between 40% and 45% market share and is
the top choice for ML research.
At the same time, JAX, while having a much smaller market share,
has been embraced by top players in generative AI such as
[Google DeepMind](https://github.com/deepmind),
[Midjourney](https://github.com/midjourney),
[Cohere](https://cloud.google.com/blog/products/ai-machine-learning/accelerating-language-model-training-with-cohere-and-google-cloud-tpus),
and more.

We believe each of these frameworks provides important value for different
use cases — and what we've created lets you tap into all three at once.
With a new multi-backend Keras,
we hope to make the lives of ML developers easier by fostering an inclusive,
cross-framework deep learning ecosystem. Say goodbye to framework silos,
and say hello to the new world of multi-framework ML!

---

## Why use Keras Core?

You're already familiar with the benefits of using Keras — it enables
high-velocity development via an obsessive focus on great UX, API design,
and debuggability. It's also a battle-tested framework that has been chosen
by over 2.5M developers and that powers some of the most sophisticated
and largest-scale ML systems in the world,
such as the Waymo self-driving fleet or the YouTube recommendation engine.
But what are the additional benefits of using the new multi-backend Keras Core?

- **Always get the best performance for your models.** In our benchmarks,
we found that JAX typically delivers the best training and inference performance
on GPU, TPU, and CPU — but results vary from model to model, as non-XLA
TensorFlow is occasionally faster on GPU. The ability to dynamically select
the backend that will deliver the best performance for your model
*without having to change anything to your code* means you're always guaranteed
to train and serve with the highest achievable efficiency.
- **Maximize available ecosystem surface for your models.** Any Keras Core
model can be instantiated as a PyTorch `Module`, can be exported as a TensorFlow
`SavedModel`, or can be instantiated as a stateless JAX function. That means
that you can use your Keras Core models with PyTorch ecosystem packages,
with the full range of TensorFlow deployment & production tools
(like TF-Serving, TF.js and TFLite), and with JAX large-scale
TPU training infrastructure. Write one `model.py` using
Keras Core APIs, and get access to everything the ML world has to offer.
- **Maximize distribution for your open-source model releases.** Want to
release a pretrained model? Want as many people as possible
to be able to use it? If you implement it in pure TensorFlow or PyTorch,
it will be usable by roughly half of the market.
If you implement it in Keras Core, it is instantly usable by anyone regardless
of their framework of choice (even if they're not Keras users themselves).
Twice the impact at no added development cost.
- **Use data pipelines from any source.** The Keras Core
`fit()`/`evaluate()`/`predict()` routines are compatible with `tf.data.Dataset` objects,
with PyTorch `DataLoader` objects, with NumPy arrays, Pandas dataframes —
regardless of the backend you're using. You can train a Keras Core + TensorFlow
model on a PyTorch `DataLoader` or train a Keras Core + PyTorch model on a
`tf.data.Dataset`.

---

## Main features of Keras Core

Let's look at some of what's included in the preview release.

### The full Keras API, available for TensorFlow, JAX, and PyTorch

To start with, Keras Core implements the full Keras API and makes it available
with TensorFlow, JAX, and PyTorch — over a hundred layers, dozens of metrics,
loss functions, optimizers, and callbacks, the Keras training and evaluation
loops, and the Keras saving & serialization infrastructure. All the APIs you
know and love are here.

Any Keras model that only uses built-in layers will immediately work with
all supported backends. In fact, your existing `tf.keras` models
that only use built-in layers can start running in JAX and PyTorch *right away*
when you change your `keras` import to point to `keras_core`!
That's right, your codebase just gained a whole new set of capabilities.

<img class="irasto" src="https://s3.amazonaws.com/keras.io/img/keras-core/multibackend_workflow.jpg" />

### A cross-framework low-level language for deep learning

Keras Core enables you to create components
(like arbitrary custom layers or pretrained models) that will work the same
in any framework. In particular, Keras Core gives you access
to the `keras_core.ops` namespace that works across all backends. It contains:

- **A near-full implementation of the NumPy API.**
Not something "NumPy-like" — just literally the
NumPy API, with the same functions and the same arguments.
You get `ops.matmul`, `ops.sum`, `ops.stack`, `ops.einsum`, etc.
- **A set of neural network-specific functions** that are absent from NumPy,
such as `ops.softmax`, `ops.binary_crossentropy`, `ops.conv`, etc.

As long as you only use ops from `keras_core.ops`, your custom layers,
custom losses, custom metrics, and custom optimizers
**will work with JAX, PyTorch, and TensorFlow — with the same code**.
That means that you can maintain only one
component implementation (e.g. a single `model.py`
together with a single checkpoint file), and you can use it in all frameworks,
with the exact same numerics.

<img class="irasto" src="https://s3.amazonaws.com/keras.io/img/keras-core/custom_components.jpg" />

### Seamless integration with native workflows in JAX, PyTorch, and TensorFlow

Unlike old-school multi-backend Keras 1.0,
the Keras Core is not just intended for Keras-centric workflows
where you define a Keras model, a Keras optimizer, a Keras loss and metrics,
and you call `fit()`/`evaluate()`/`predict()`.
It's also meant to work seamlessly with low-level backend-native workflows:
you can take a Keras model (or any other component, such as a loss or metric)
and start using it in a JAX training loop, a TensorFlow training loop,
or a PyTorch training loop, or as part of a JAX or PyTorch model,
with zero friction. Keras Core provides exactly
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
- Use a Keras layer or model as part of a `torch.nn.Module`. This means
that PyTorch users can start leveraging Keras models whether or not
they use Keras APIs! You can treat a Keras model just like any other
PyTorch `Module`.
- etc.

<img class="irasto" src="https://s3.amazonaws.com/keras.io/img/keras-core/custom_training_loops.jpg" />

### Support for cross-framework data pipelines with all backends

Multi-framework ML also means multi-framework data loading and preprocessing.
Keras Core models can be trained using a wide range of
data pipelines — regardless of whether you're using the JAX, PyTorch, or
TensorFlow backends. It just works.

- `tf.data.Dataset` pipelines: the reference for scalable production ML.
- `torch.utils.data.DataLoader` objects.
- NumPy arrays and Pandas dataframes.
- `keras_core.utils.PyDataset` objects.

### Pretrained models

What would a deep learning framework be without pretrained models?
Right from launch day, there's a wide range of pretrained models that
you can start using with Keras Core.

All 40 Keras Applications models (the `keras_core.applications` namespace)
are available in all backends
(minus one model that is architecturally incompatible with PyTorch
due to lack of support for asymmetric padding in average pooling).
The vast array of pretrained models in [KerasCV](https://keras.io/api/keras_cv/)
and [KerasNLP](https://keras.io/api/keras_nlp/)
(e.g. BERT, T5, YOLOv8, Whisper, etc.) also work with all backends.

### Progressive disclosure of complexity

*Progressive disclosure of complexity* is the design principle at the heart
of the Keras API. Keras doesn't force you to follow
a single "true" way of building and training models. Instead, it enables
a wide range of different workflows, from the very high-level to the very
low-level, corresponding to different user profiles.

That means that you can start out with simple workflows — such as using
`Sequential` and `Functional` models and training them with `fit()` — and when
you need more flexibility, you can easily customize different components while
reusing most of your prior code. As your need become more specific,
you don't suddenly fall off a complexity cliff and you don't need to switch
to a different set of tools.

We've brought this principle to all of our backends. For instance,
you can customize what happens in your training loop while still
leveraging the power of `fit()`, without having to write your own training loop
from scratch — just by overriding the `train_step` method.

Here's how it works in PyTorch and TensorFlow:

<img class="irasto" src="https://s3.amazonaws.com/keras.io/img/keras-core/customizing_fit.jpg" />

And [here's the link](http://keras.io/keras_core/guides/custom_train_step_in_jax/) to the JAX version.

### A new stateless API for layers, models, metrics, and optimizers

Are you a [functional programming](https://en.wikipedia.org/wiki/Functional_programming) enjoyer?
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
updated_non_trainable_variables = layer.stateless_call(
    trainable_variables,
    non_trainable_variables,
    inputs,
)
```

You never have to implement these methods yourself: they're automatically available
as long as you've implemented the stateful version (e.g. `call()` or `update_state()`).

---

## Give us your feedback!

The purpose of this preview release is to let everyone try out the new
capabilities, spot issues, and help us make the software the best it can be
before the stable Keras 3.0 release this fall. So please send us your feedback!
Here are some things you can do:

- Try to run your existing `tf.keras` codebase on top of Keras Core with the
TensorFlow backend, and report any issue you find. This will help us
guarantee full backwards compatibility.
- Try to adapt your existing `tf.keras` models so they can run on top of
JAX and PyTorch in addition to TensorFlow. This involves replacing calls
to the TensorFlow API with calls the NumPy functions from `keras_core.ops`.
We're looking to offer a comprehensive guide to cover this conversion, and
you can help us write it!
- Try to integrate Keras models into your existing JAX or PyTorch training or
serving infrastructure and let us know how it goes.
- If you're a company with multi-framework workflows looking to adopt
Keras Core, and you'd like to chat about your use case,
reach out to fchollet@google.com.

Enjoy the library!

---

## Known issues

Keras Core is a beta release — you should expect to encounter issues.
Please let us know (via GitHub issues) about any issue you find so we can make
the library work better for you!

Here are known gotchas to watch out for:

- **Import order.** Due to a bug in PyTorch, importing `torch`
when `tensorflow` is already imported will cause
either a segfault crash of your Python runtime, or a deadlock.
In reverse, importing `tensorflow`
when `torch` is already imported is fine — so when importing both packages,
you should make sure
to import `torch` first, and then `tensorflow`.
Note that when using the `torch` backend, `keras_core`
imports `torch`, and thus `keras_core` should be imported before `tensorflow`
if you're importing both.
- **Integer dtypes with PyTorch.** The `torch` package has no support
for dtypes `uint16` and `uint32`. To maintain compatibility
with JAX and TensorFlow, using these dtypes with the `torch`
backend will fallback to `int32` and `int64` respectively.
- **Average pooling with PyTorch.** The `torch` package has no support
for asymmetric padding with pooling ops. As a result, when using average
pooling with `padding="same"`, you may see different results
(on the last row/column) compared to other backends.
- **Using Keras layers or models in a `tf.data` pipeline.** As long as you're
using the TensorFlow backend, you can `.map()` Keras layers and models in a
`tf.data` pipeline, but when using other backends, this is generally
not possible. We've special-cased preprocessing layers so that they can be
used in `tf.data` regardless of your choice of backend, but this doesn't
extend to non-preprocessing layers or to a `Sequential` models wrapping a
list of preprocessing layers.
- **Image layout and performance considerations with PyTorch.**
When using convnets, the typical image layout to use
is `"channels_last"` (aka NHWC), which is
the standard in cuDNN, TensorFlow, JAX, and others. However, PyTorch uses
`"channels_first"`. You can use any Keras Core convnet with any image
layout, and you can easily switch from one default layout to the other via
the `keras_core.config.set_image_data_format()` flag. Importantly, when using
PyTorch convnets in the `"channels_last"` format, Keras will have to
convert layouts back and forth at each layer, which is inefficient. For best
performance, remember to set your default layout to `"channels_first"` when
using convnets in PyTorch. In the future, we hope to resolve this issue by
by-passing `torch.nn` ops and going directly to cuDNN.
- **Sparse NN support.** Unlike in `tf.keras`, there is currently no
support for networks that operate on sparse types. We intend to add support
in the future for the TensorFlow backend, where it is feasible.

---

## Frequently asked questions

### Q: What is the relationship between Keras Core and Keras 3.0?

Keras Core is a preview release of Keras 3.0.
Ultimately, the current Keras Core codebase will become Keras 3.0
and will be released as the `keras` pip package.

### Q: When will Keras 3.0 be released?

We're targeting Fall 2023. As you will see when you try out Keras Core,
the library is already feature-complete and fairly mature,
so all we need is a few months of beta-testing
to iron out any possible issue and pilot large-scale production use cases.

We're currently starting production pilots at Google and other
Alphabet companies. If your company has an interesting production use case
and you'd like to work with us to pilot Keras Core, we can take a look at it
— please contact us.

#### Q: Is Keras Core compatible with `tf.keras`?

Code developed with `tf.keras` can generally be run as-is with Keras Core
(with the TensorFlow backend) simply by changing the Keras imports and making
sure your saving your models in the `.keras` format (as opposed to the
legacy Keras SavedModel or `.h5` formats).

When it comes to using APIs from `tf.keras` and Keras Core side by side,
that is **not** possible at this time.

However, when Keras 3.0 is released, Keras Core will *become* `tf.keras`
(in the sense that Keras Core will be distributed as the `keras` package on
PyPI and `tf.keras` will become a pointer to it). There will only
be one stable, production-ready version of Keras — today, that is `tf.keras`,
and soon that will be multi-backend Keras.

### Q: Do pretrained models developed in `tf.keras` work with Keras Core?

Generally, yes. Any `tf.keras` model should work out of the box with Keras Core
with the TensorFlow backend (make sure to save it in the `.keras` v3 format).
In addition, if the model only
uses built-in Keras layers, then it will also work out of the box
with Keras Core with the JAX and PyTorch backends.

If the model contains custom layers written using TensorFlow APIs,
it is usually easy to convert the code to be backend-agnostic.
For instance, it only took us a few hours to convert all 40
`tf.keras` models from Keras Applications to be backend-agnostic.

### Q: Can I save a Keras Core model in one backend and reload it in another backend?

Yes, you can. There is no backend specialization in saved `.keras` files whatsoever.
Your saved Keras models are framework-agnostic and can be reloaded with any backend.

However, note that reloading a model that contains custom components
with a different backend requires your custom components to be implemented
using backend-agnostic APIs, e.g. `keras.ops`.

### Q: Does Keras add extra overhead in eager mode?

In eager mode, yes — a very small amount, quantified below. In compiled
mode, virtually none.

Keras Layers and Keras Functional and Sequential models do more than
just piping data through to cuDNN.
They run a variety of input validation checks, standardization operations,
and so on, which improve
your development and debugging experience, but which add a small time cost
to every training and inference step when running eagerly.

- For a simple model (e.g. a classification model with 3 layers implemented
in the Sequential or Functional API)
Keras eager overhead per step is about 150μs, on a typical CPU.
- For a model with 50 layer blocks with 10 layers per block
(which is roughly the format of current SotA LLMs),
or 500 layers in total, the eager overhead per step of a
Functional Keras model is about 20ms.

So if your training step time is ~500ms (which is on the lower end of
what's needed to keep your device utilized),
then Keras eager overhead would represent 5% of your total step time.
If you care about a 5% difference, then you
should definitely not be running eagerly — you should be compiling your model,
which will typically bring much larger performance benefits than just 5%.

By comparison, a compiled Keras model only has about 10μs of dispatch
overhead — in total, regardless of model size.

Generally speaking, we recommend using eager mode to debug your code,
then switching to compilation for any real training or inference run.
This workflow works the same in TensorFlow, JAX, and PyTorch.

### Q: Why does using Keras Core with PyTorch or JAX still requires TensorFlow?

Right now, we use `tf.nest` (a Python data structure processing utility)
extensively across the codebase, which requires the TensorFlow package.
In the near future, we intend to turn `tf.nest` into a standalone
package, so that you could use Keras Core without installing TensorFlow.

### Q: Can I use Keras Core components inside `tf.data` pipelines?

With the TensorFlow backend, Keras Core is fully compatible with `tf.data`
(e.g. you can `.map()` a `Sequential` model into a `tf.data` pipeline).

With a different backend, Keras Core has limited support for `tf.data`.
You won't be able to `.map()` arbitrary layers or models into a `tf.data`
pipeline. However, you will be able to use specific Keras Core
preprocessing layers with `tf.data`, such as `IntegerLookup` or
`CategoryEncoding`.

When it comes to using a `tf.data` pipeline (that does not use Keras)
to feed your call to `.fit()`, `.evaluate()` or `.predict()` —
that works out of the box with all backends.

### Q: Do Keras Core models behave the same when run with different backends?

Yes, numerics are identical across backends.
However, keep in mind the following caveats:

- RNG behavior is different across different backends
(even after seeding — your results will be deterministic in each backend
but will differ across backends). So random weight initializations
values and dropout values will differ across backends.
- Due to the nature of floating-point implementations,
results are only identical up to 1e-7 precision in float32,
per function execution. So when training a model for a long time,
small numerical differences will accumulate and may end up resulting
in noticeable numerical differences.
- Due to lack of support for average pooling with asymmetric padding
in PyTorch, average pooling layers with `padding="same"`
may result in different numerics on border rows/columns.
This doesn't happen very often in practice —
out of 40 Keras Applications vision models, only one was affected.

### Q: Does Keras Core support distributed training?

Data-parallel distribution is supported out of the box in TensorFlow
and PyTorch.

Keras Core is compatible with `tf.distribute` —
just open a Distribution Strategy scope and create / train your model within it.
[Here's an example](http://keras.io/keras_core/guides/distributed_training_with_tensorflow/).

Keras Core is also compatible with PyTorch's `DistributedDataParallel` utility.
[Here's an example](https://github.com/keras-team/keras-core/blob/main/examples/demo_torch_multi_gpu.py).

In JAX, you should distribute training yourself via JAX APIs such as
`jax.sharding`. [Here's an example](https://github.com/keras-team/keras-core/blob/main/examples/demo_jax_distributed.py).

### Q: Will you add more backends in the future? What about framework XYZ?

We're open to adding new backends as long as the target framework has a large user base
or otherwise has some unique technical benefits to bring to the table.
However, adding and maintaining a new backend is a large burden,
so we're going to carefully consider each new backend candidate on a case by case basis,
and we're not likely to add many new backends. We will not add any new frameworks
that aren't yet well-established.
At this time, we have no immediate plans for additional backends.

---


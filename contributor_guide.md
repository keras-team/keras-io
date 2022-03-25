# Keras.io code examples contributor guide

This guide offers general tips to be followed when writing code examples for [keras.io](https://keras.io).
Make sure to read it before opening a PR.


## Code style

### Variable names

Make sure to use fully-spelled out variable names. Do not use single-letter variable names.
Do not use abbreviations unless they're completely obvious (e.g. `num_layers` is ok).

This is bad:

```python
m = get_model(u=32, d=0.5)
```

This is good:

```python
model = get_model(units=32, dropout_rate=0.5)
```

### Imports

Import modules, not individual objects. In particular, don't import individual layers. Typically
you should import the following:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

Then access objects from these modules:

```python
tf.Variable(...)
tf.reshape(...)
keras.Input(...)
keras.Model(...)
keras.optimizers.Adam(...)
layers.Layer(...)
layers.Conv2D(...)
```

Note: do **not** use `import keras`. Use `from tensorflow import keras` instead.


### Extra dependencies

If your example requires extra dependencies, don't include installation commands as part of the code of your example.
Instead, mention the dependencies in the text, alongside an example of the pip command to install them, e.g.


```md
This example requires XYZ. You can install it via the following command: `pip install XYZ`
```

---

## Model development best practices

### Model types

**Use Functional models wherever possible.**
Only use subclassing if your model cannot straightforwardly be implemented as a Functional model.
If writing a subclassed Model or Layer, do not instantiate any Layer as part of the `call()` method.

Any layer you use in `call()` should have been instantiated beforehand, either in `__init__()` or in `build()`.

This is bad:

```python
class MyLayer(layers.Layer):
    def call(self, x):
        ...
        x = layers.Add()([x, y])
        ...
```

This is good:

```python
class MyLayer(layers.Layer):
    def call(self, inputs):
        ...
        features += other_features
        ...
```

### Training loop types

**Use `model.fit()` whenever possible.** If you cannot use the built-in `fit()`
(e.g. in the case of a GAN, VAE, similarity model, etc.) then use
a custom `train_step()` method to customize `fit()` ([see guide](https://keras.io/guides/customizing_what_happens_in_fit/)).

If you need to customize how the model iterates on the data (e.g. in the case of a RL algorithm or a curriculum learning algorithm),
then write a training loop from scratch using `tf.GradientTape`.


### Demonstrate generalization power

Whenever you call `fit()` (or otherwise run a training loop), make sure to
use a validation dataset (`validation_data` argument) to monitor the model's performance
on data it has not seen during training. Likewise, when showing inference results,
use samples from a validation or test set, not training samples.

The only exception to this rule is in the case of generative models.


### Demonstrate the full power of your model, but keep the run time short

We need to keep the run time of the notebooks short (typically no more than 20 minutes on a V100 GPU).
However, many models need to be trained for much longer in order to achieve good results. In such
cases:

- Keep the run time short by limiting the number of epochs (e.g. train for a single epoch).
- Highlight the fact that the model should actually be trained for `N` epochs to achieve the expected results
(in a text paragraph and in code comments).
- Showcase the results of the full model trained for `N` epochs, by providing accuracy numbers and showing
inference results of the full model. You can simply insert images in the text paragraphs, hosted on
[imgur.com](imgur.com).


### Argument validation

In general, user-provided input argument validation is not required in custom classes / functions in a keras.io code example.
If you want to add input validation, do so with `ValueError`; do not use `assert` statements.


### Data input

Prefer using either NumPy arrays or a `tf.data.Dataset` for data input whenever possible.
If impossible, then use a `keras.utils.Sequence` subclass. Do not use regular Python generators.

When using `.map()` with a `tf.data.Dataset`, make sure to pass a value for `num_parallel_calls`.
Typically, you can set the value to be 4, 8, or `tf.data.AUTOTUNE`.


---

## Text style

### Length

Examples should be clear and detailed, but not overly verbose. You can add as much text content as you want, as
long as each additional sentence / paragraph provides useful information that helps with understanding the example.
Never use any "filler" content.

### Style

- Use present tense ("We present... we implement...")
- Always define abbreviations / acronyms the first time you use them ("We implement a Graph Attention Network (GAT)...")
- All and any sentence should convey a useful idea; avoid filler at all costs.

### Proofreading

Make sure to proofread your text paragraphs to avoid typos.
Every sentence should start with a capital letter and should end with a period. This applies to code comments as well.

### Introduction and conclusion

There should be an introduction that explains what the reader should expect to find in the example,
and why it is useful/interesting.
If the example presents a specific technique,
the introduction should also include an overview of the technique as well as links to external references.
There should be a conclusion section that recapitulates key takeaways from the example, and offers pointers to next steps.

### Code elements

All code keywords should be formatted with backticks, e.g. `like_this` (standard Markdown code formatting).

When refering to a function or method name, it should be followed with parens, like this: `my_function()` or `my_method()`.

### Mathematical notation

Do not use any LaTeX notation. Explain math operations with pseudocode.
If you really must have an equation, then embed it as an image.

### Line length

Keep text lines relatively short (about 80 characters), unless it's a link.

### Markdown links

Each markdown link should fit on a single line, unbroken, like this:

```md
Here's a link:

[This is the link text](https://github.com/keras-team/keras-io/blob/master/contributor_guide.md)
```

Do not break the link like this (or in any other way):

```md
[This is the link text](
    https://github.com/keras-team/keras-io/blob/master/contributor_guide.md)
```

### Markdown lists

There should be a line break before the first item in any list, e.g.

This is good:

```md
Here's a list:

- First item
- Second item
```

This is bad:

```md
Here's a badly formatted list:
- First item
- Second item
```

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


### Extra dependencies

If your example requires extra dependencies, don't include installation commands as part of the code of your example.
Instead, mention the dependencies in the text, alongside an example of the pip command to install them, e.g.


```md
This example requires XYZ. You can install it via the following command: `pip install XYZ`
```

### Model development style

Use Functional models wherever possible.
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


### Input validation

In general, input argument validation is not required.
If you want to add input validation, do so with `ValueError`; do not use `assert` statements.


---

## Text style

### Length

Examples should be clear and detailed, but not overly verbose. You can add as much text content as you want, as
long as each additional sentence / paragraph provides useful information that helps with understanding the example.
Never use any "filler" content.

### Proofreading

Make sure to proofread your text paragraphs to avoid typos.
Every sentence should start with a capital letter and should end with a period. This applies to code comments as well.

### Introduction and conclusion

There should be an introduction that explains what the reader should expect to find in the example,
and why it is useful/interesting.
If the example presents a specific technique,
the introduction should also include an overview of the technique as well as links to external references.
There should be a conclusion section that recapitulates key takeaways from the example, and offers pointers to next steps.

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

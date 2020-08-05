# Losses

The purpose of loss functions is to compute the quantity that a model should seek
to minimize during training.


## Available losses

Note that all losses are available both via a class handle and via a function handle.
The class handles enable you to pass configuration arguments to the constructor
(e.g.
`loss_fn = CategoricalCrossentropy(from_logits=True)`),
and they perform reduction by default when used in a standalone way (see details below).


{{toc}}


---


## Usage of losses with `compile()` & `fit()`

A loss function is one of the two arguments required for compiling a Keras model:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(layers.Activation('softmax'))

loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer='adam')
```

All built-in loss functions may also be passed via their string identifier:

```python
# pass optimizer by name: default parameters will be used
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

Loss functions are typically created by instantiating a loss class (e.g. `keras.losses.SparseCategoricalCrossentropy`).
All losses are also provided as function handles (e.g. `keras.losses.sparse_categorical_crossentropy`).

Using classes enables you to pass configuration arguments at instantiation time, e.g.:

```python
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

---

## Standalone usage of losses

A loss is a callable with arguments `loss_fn(y_true, y_pred, sample_weight=None)`:

- **y_true**: Ground truth values, of shape `(batch_size, d0, ... dN)`. For
    sparse loss functions, such as sparse categorical crossentropy, the shape
    should be `(batch_size, d0, ... dN-1)`
- **y_pred**: The predicted values, of shape `(batch_size, d0, .. dN)`.
- **sample_weight**: Optional `sample_weight` acts as reduction weighting
    coefficient for the per-sample losses. If a scalar is provided, then the loss is
    simply scaled by the given value. If `sample_weight` is a tensor of size
    `[batch_size]`, then the total loss for each sample of the batch is
    rescaled by the corresponding element in the `sample_weight` vector. If
    the shape of `sample_weight` is `(batch_size, d0, ... dN-1)` (or can be
    broadcasted to this shape), then each loss element of `y_pred` is scaled
    by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
    functions reduce by 1 dimension, usually `axis=-1`.)

By default, loss functions return one scalar loss value per input sample, e.g.

```
>>> tf.keras.losses.mean_squared_error(tf.ones((2, 2,)), tf.zeros((2, 2)))
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 1.], dtype=float32)>
```

However, loss class instances feature a `reduction` constructor argument,
which defaults to `"sum_over_batch_size"` (i.e. average). Allowable values are
"sum_over_batch_size", "sum", and "none":

- "sum_over_batch_size" means the loss instance will return the average
    of the per-sample losses in the batch.
- "sum" means the loss instance will return the sum of the per-sample losses in the batch.
- "none" means the loss instance will return the full array of per-sample losses.

```
>>> loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
>>> loss_fn(tf.ones((2, 2,)), tf.zeros((2, 2)))
<tf.Tensor: shape=(), dtype=float32, numpy=1.0>
```
```
>>> loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum')
>>> loss_fn(tf.ones((2, 2,)), tf.zeros((2, 2)))
<tf.Tensor: shape=(), dtype=float32, numpy=2.0>
```
```
>>> loss_fn = tf.keras.losses.MeanSquaredError(reduction='none')
>>> loss_fn(tf.ones((2, 2,)), tf.zeros((2, 2)))
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 1.], dtype=float32)>
```

Note that this is an important difference between loss functions like `tf.keras.losses.mean_squared_error`
and default loss class instances like `tf.keras.losses.MeanSquaredError`: the function version
does not perform reduction, but by default the class instance does.

```
>>> loss_fn = tf.keras.losses.mean_squared_error
>>> loss_fn(tf.ones((2, 2,)), tf.zeros((2, 2)))
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 1.], dtype=float32)>
```
```
>>> loss_fn = tf.keras.losses.MeanSquaredError()
>>> loss_fn(tf.ones((2, 2,)), tf.zeros((2, 2)))
<tf.Tensor: shape=(), dtype=float32, numpy=1.0>
```

When using `fit()`, this difference is irrelevant since reduction is handled by the framework.

Here's how you would use a loss class instance as part of a simple training loop:

```python
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Iterate over the batches of a dataset.
for x, y in dataset:
    with tf.GradientTape() as tape:
        logits = model(x)
        # Compute the loss value for this batch.
        loss_value = loss_fn(y, logits)

    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

---

## Creating custom losses

Any callable with the signature `loss_fn(y_true, y_pred)`
that returns an array of losses (one of sample in the input batch) can be passed to `compile()` as a loss.
Note that sample weighting is automatically supported for any such loss.

Here's a simple example:

```python
def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

model.compile(optimizer='adam', loss=my_loss_fn)
```


---

## The `add_loss()` API

Loss functions applied to the output of a model aren't the only way to
create losses.

When writing the `call` method of a custom layer or a subclassed model,
you may want to compute scalar quantities that you want to minimize during
training (e.g. regularization losses). You can use the `add_loss()` layer method
to keep track of such loss terms.

Here's an example of a layer that adds a sparsity regularization loss based on the L2 norm of the inputs:

```python
from tensorflow.keras.layers import Layer

class MyActivityRegularizer(Layer):
  """Layer that creates an activity sparsity regularization loss."""
  
  def __init__(self, rate=1e-2):
    super(MyActivityRegularizer, self).__init__()
    self.rate = rate
  
  def call(self, inputs):
    # We use `add_loss` to create a regularization loss
    # that depends on the inputs.
    self.add_loss(self.rate * tf.reduce_sum(tf.square(inputs)))
    return inputs
```

Loss values added via `add_loss` can be retrieved in the `.losses` list property of any `Layer` or `Model`
(they are recursively retrieved from every underlying layer):

```python
from tensorflow.keras import layers

class SparseMLP(Layer):
  """Stack of Linear layers with a sparsity regularization loss."""

  def __init__(self, output_dim):
      super(SparseMLP, self).__init__()
      self.dense_1 = layers.Dense(32, activation=tf.nn.relu)
      self.regularization = MyActivityRegularizer(1e-2)
      self.dense_2 = layers.Dense(output_dim)

  def call(self, inputs):
      x = self.dense_1(inputs)
      x = self.regularization(x)
      return self.dense_2(x)
    

mlp = SparseMLP(1)
y = mlp(tf.ones((10, 10)))

print(mlp.losses)  # List containing one float32 scalar
```

These losses are cleared by the top-level layer at the start of each forward pass -- they don't accumulate.
So `layer.losses` always contain only the losses created during the last forward pass.
You would typically use these losses by summing them before computing your gradients when writing a training loop.

```python
# Losses correspond to the *last* forward pass.
mlp = SparseMLP(1)
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1  # No accumulation.
```

When using `model.fit()`, such loss terms are handled automatically.

When writing a custom training loop, you should retrieve these terms
by hand from `model.losses`, like this:

```python
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Iterate over the batches of a dataset.
for x, y in dataset:
    with tf.GradientTape() as tape:
        # Forward pass.
        logits = model(x)
        # Loss value for this batch.
        loss_value = loss_fn(y, logits)
        # Add extra loss terms to the loss value.
        loss_value += sum(model.losses)

    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

See [the `add_loss()` documentation](/api/layers/base_layer/#add_loss-method) for more details.

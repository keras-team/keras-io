# Customizing what happens in `fit()` with PyTorch

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2023/06/27<br>
**Last modified:** 2024/08/01<br>
**Description:** Overriding the training step of the Model class with PyTorch.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/custom_train_step_in_torch.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/custom_train_step_in_torch.py)



---
## Introduction

When you're doing supervised learning, you can use `fit()` and everything works
smoothly.

When you need to take control of every little detail, you can write your own training
loop entirely from scratch.

But what if you need a custom training algorithm, but you still want to benefit from
the convenient features of `fit()`, such as callbacks, built-in distribution support,
or step fusing?

A core principle of Keras is **progressive disclosure of complexity**. You should
always be able to get into lower-level workflows in a gradual way. You shouldn't fall
off a cliff if the high-level functionality doesn't exactly match your use case. You
should be able to gain more control over the small details while retaining a
commensurate amount of high-level convenience.

When you need to customize what `fit()` does, you should **override the training step
function of the `Model` class**. This is the function that is called by `fit()` for
every batch of data. You will then be able to call `fit()` as usual -- and it will be
running your own learning algorithm.

Note that this pattern does not prevent you from building models with the Functional
API. You can do this whether you're building `Sequential` models, Functional API
models, or subclassed models.

Let's see how that works.

---
## Setup


```python
import os

# This guide can only be run with the torch backend.
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from keras import layers
import numpy as np
```

---
## A first simple example

Let's start from a simple example:

- We create a new class that subclasses `keras.Model`.
- We just override the method `train_step(self, data)`.
- We return a dictionary mapping metric names (including the loss) to their current
value.

The input argument `data` is what gets passed to fit as training data:

- If you pass NumPy arrays, by calling `fit(x, y, ...)`, then `data` will be the tuple
`(x, y)`
- If you pass a `torch.utils.data.DataLoader` or a `tf.data.Dataset`,
by calling `fit(dataset, ...)`, then `data` will be what gets yielded
by `dataset` at each batch.

In the body of the `train_step()` method, we implement a regular training update,
similar to what you are already familiar with. Importantly, **we compute the loss via
`self.compute_loss()`**, which wraps the loss(es) function(s) that were passed to
`compile()`.

Similarly, we call `metric.update_state(y, y_pred)` on metrics from `self.metrics`,
to update the state of the metrics that were passed in `compile()`,
and we query results from `self.metrics` at the end to retrieve their current value.


```python

class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        # Compute loss
        y_pred = self(x, training=True)  # Forward pass
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

```

Let's try this out:


```python
# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Just use `fit` as usual
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=3)
```

<div class="k-default-codeblock">
```
Epoch 1/3

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - mae: 0.4456 - loss: 0.2602

<div class="k-default-codeblock">
```

```
</div>
 23/32 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - mae: 0.3453 - loss: 0.1812

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.3410 - loss: 0.1772


<div class="k-default-codeblock">
```
Epoch 2/3

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - mae: 0.3425 - loss: 0.1898

<div class="k-default-codeblock">
```

```
</div>
 21/32 ━━━━━━━━━━━━━[37m━━━━━━━  0s 3ms/step - mae: 0.3368 - loss: 0.1733

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.3336 - loss: 0.1695


<div class="k-default-codeblock">
```
Epoch 3/3

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - mae: 0.2943 - loss: 0.1318

<div class="k-default-codeblock">
```

```
</div>
 22/32 ━━━━━━━━━━━━━[37m━━━━━━━  0s 2ms/step - mae: 0.3159 - loss: 0.1498

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - mae: 0.3170 - loss: 0.1511





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f48a3255710>

```
</div>
---
## Going lower-level

Naturally, you could just skip passing a loss function in `compile()`, and instead do
everything *manually* in `train_step`. Likewise for metrics.

Here's a lower-level example, that only uses `compile()` to configure the optimizer:

- We start by creating `Metric` instances to track our loss and a MAE score (in `__init__()`).
- We implement a custom `train_step()` that updates the state of these metrics
(by calling `update_state()` on them), then query them (via `result()`) to return their current average value,
to be displayed by the progress bar and to be pass to any callback.
- Note that we would need to call `reset_states()` on our metrics between each epoch! Otherwise
calling `result()` would return an average since the start of training, whereas we usually work
with per-epoch averages. Thankfully, the framework can do that for us: just list any metric
you want to reset in the `metrics` property of the model. The model will call `reset_states()`
on any object listed here at the beginning of each `fit()` epoch or at the beginning of a call to
`evaluate()`.


```python

class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.loss_fn = keras.losses.MeanSquaredError()

    def train_step(self, data):
        x, y = data

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        # Compute loss
        y_pred = self(x, training=True)  # Forward pass
        loss = self.loss_fn(y, y_pred)

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_metric.result(),
        }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.loss_tracker, self.mae_metric]


# Construct an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)

# We don't pass a loss or metrics here.
model.compile(optimizer="adam")

# Just use `fit` as usual -- you can use callbacks, etc.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=5)

```

<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 5ms/step - loss: 0.9188 - mae: 0.8184

<div class="k-default-codeblock">
```

```
</div>
 23/32 ━━━━━━━━━━━━━━[37m━━━━━━  0s 2ms/step - loss: 0.6644 - mae: 0.6911

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.6173 - mae: 0.6607


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.2580 - mae: 0.4124

<div class="k-default-codeblock">
```

```
</div>
 24/32 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 0.2394 - mae: 0.3928

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2340 - mae: 0.3883


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1675 - mae: 0.3378

<div class="k-default-codeblock">
```

```
</div>
 22/32 ━━━━━━━━━━━━━[37m━━━━━━━  0s 2ms/step - loss: 0.1927 - mae: 0.3509

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1922 - mae: 0.3517


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 6ms/step - loss: 0.1773 - mae: 0.3300

<div class="k-default-codeblock">
```

```
</div>
 24/32 ━━━━━━━━━━━━━━━[37m━━━━━  0s 2ms/step - loss: 0.1773 - mae: 0.3373

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1802 - mae: 0.3411


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - loss: 0.1711 - mae: 0.3402

<div class="k-default-codeblock">
```

```
</div>
 20/32 ━━━━━━━━━━━━[37m━━━━━━━━  0s 3ms/step - loss: 0.1866 - mae: 0.3514

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.1862 - mae: 0.3505





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f48975ccbd0>

```
</div>
---
## Supporting `sample_weight` & `class_weight`

You may have noticed that our first basic example didn't make any mention of sample
weighting. If you want to support the `fit()` arguments `sample_weight` and
`class_weight`, you'd simply do the following:

- Unpack `sample_weight` from the `data` argument
- Pass it to `compute_loss` & `update_state` (of course, you could also just apply
it manually if you don't rely on `compile()` for losses & metrics)
- That's it.


```python

class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        # Compute loss
        y_pred = self(x, training=True)  # Forward pass
        loss = self.compute_loss(
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# You can now use sample_weight argument
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, epochs=3)
```

<div class="k-default-codeblock">
```
Epoch 1/3

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - mae: 0.2704 - loss: 0.0603

<div class="k-default-codeblock">
```

```
</div>
 22/32 ━━━━━━━━━━━━━[37m━━━━━━━  0s 2ms/step - mae: 0.3200 - loss: 0.0825

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.3216 - loss: 0.0827


<div class="k-default-codeblock">
```
Epoch 2/3

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 7ms/step - mae: 0.3627 - loss: 0.0897

<div class="k-default-codeblock">
```

```
</div>
 22/32 ━━━━━━━━━━━━━[37m━━━━━━━  0s 2ms/step - mae: 0.3157 - loss: 0.0812

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.3156 - loss: 0.0803


<div class="k-default-codeblock">
```
Epoch 3/3

```
</div>
    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 4ms/step - mae: 0.2894 - loss: 0.0908

<div class="k-default-codeblock">
```

```
</div>
 22/32 ━━━━━━━━━━━━━[37m━━━━━━━  0s 2ms/step - mae: 0.3082 - loss: 0.0763

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.3085 - loss: 0.0760





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f48975d7bd0>

```
</div>
---
## Providing your own evaluation step

What if you want to do the same for calls to `model.evaluate()`? Then you would
override `test_step` in exactly the same way. Here's what it looks like:


```python

class CustomModel(keras.Model):
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss = self.compute_loss(y=y, y_pred=y_pred)
        # Update the metrics.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


# Construct an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(loss="mse", metrics=["mae"])

# Evaluate with our custom test_step
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.evaluate(x, y)
```

    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  0s 2ms/step - mae: 0.8706 - loss: 0.9344

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - mae: 0.8959 - loss: 0.9952





<div class="k-default-codeblock">
```
[1.0077838897705078, 0.8984771370887756]

```
</div>
---
## Wrapping up: an end-to-end GAN example

Let's walk through an end-to-end example that leverages everything you just learned.

Let's consider:

- A generator network meant to generate 28x28x1 images.
- A discriminator network meant to classify 28x28x1 images into two classes ("fake" and
"real").
- One optimizer for each.
- A loss function to train the discriminator.


```python
# Create the discriminator
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator
latent_dim = 128
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        # We want to generate 128 coefficients to reshape into a 7x7x128 map
        layers.Dense(7 * 7 * 128),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)
```

Here's a feature-complete GAN class, overriding `compile()` to use its own signature,
and implementing the entire GAN algorithm in 17 lines in `train_step`:


```python

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.seed_generator = keras.random.SeedGenerator(1337)
        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(real_images, tuple) or isinstance(real_images, list):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = real_images.shape[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        real_images = torch.tensor(real_images, device=device)
        combined_images = torch.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = torch.concat(
            [
                torch.ones((batch_size, 1), device=device),
                torch.zeros((batch_size, 1), device=device),
            ],
            axis=0,
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * keras.random.uniform(labels.shape, seed=self.seed_generator)

        # Train the discriminator
        self.zero_grad()
        predictions = self.discriminator(combined_images)
        d_loss = self.loss_fn(labels, predictions)
        d_loss.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # Sample random points in the latent space
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Assemble labels that say "all real images"
        misleading_labels = torch.zeros((batch_size, 1), device=device)

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        self.zero_grad()
        predictions = self.discriminator(self.generator(random_latent_vectors))
        g_loss = self.loss_fn(misleading_labels, predictions)
        grads = g_loss.backward()
        grads = [v.value.grad for v in self.generator.trainable_weights]
        with torch.no_grad():
            self.g_optimizer.apply(grads, self.generator.trainable_weights)

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

```

Let's test-drive it:


```python
# Prepare the dataset. We use both the training & test MNIST digits.
batch_size = 64
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))

# Create a TensorDataset
dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(all_digits), torch.from_numpy(all_digits)
)
# Create a DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

gan.fit(dataloader, epochs=1)
```

<div class="k-default-codeblock">
```
/tmp/ipykernel_102851/363287871.py:36: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  real_images = torch.tensor(real_images, device=device)

```
</div>
    
    1/1094 [37m━━━━━━━━━━━━━━━━━━━━  8:13 451ms/step - d_loss: 0.7296 - g_loss: 0.7001

<div class="k-default-codeblock">
```

```
</div>
    2/1094 [37m━━━━━━━━━━━━━━━━━━━━  5:55 326ms/step - d_loss: 0.7261 - g_loss: 0.7008

<div class="k-default-codeblock">
```

```
</div>
    3/1094 [37m━━━━━━━━━━━━━━━━━━━━  5:56 326ms/step - d_loss: 0.7230 - g_loss: 0.7016

<div class="k-default-codeblock">
```

```
</div>
    4/1094 [37m━━━━━━━━━━━━━━━━━━━━  5:58 329ms/step - d_loss: 0.7201 - g_loss: 0.7025

<div class="k-default-codeblock">
```

```
</div>
    5/1094 [37m━━━━━━━━━━━━━━━━━━━━  5:59 330ms/step - d_loss: 0.7172 - g_loss: 0.7033

<div class="k-default-codeblock">
```

```
</div>
    6/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:01 332ms/step - d_loss: 0.7144 - g_loss: 0.7042

<div class="k-default-codeblock">
```

```
</div>
    7/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:02 333ms/step - d_loss: 0.7116 - g_loss: 0.7050

<div class="k-default-codeblock">
```

```
</div>
    8/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:03 335ms/step - d_loss: 0.7090 - g_loss: 0.7057

<div class="k-default-codeblock">
```

```
</div>
    9/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:02 335ms/step - d_loss: 0.7064 - g_loss: 0.7065

<div class="k-default-codeblock">
```

```
</div>
   10/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:10 342ms/step - d_loss: 0.7039 - g_loss: 0.7072

<div class="k-default-codeblock">
```

```
</div>
   11/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:18 350ms/step - d_loss: 0.7014 - g_loss: 0.7078

<div class="k-default-codeblock">
```

```
</div>
   12/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:30 361ms/step - d_loss: 0.6991 - g_loss: 0.7084

<div class="k-default-codeblock">
```

```
</div>
   13/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:38 368ms/step - d_loss: 0.6968 - g_loss: 0.7089

<div class="k-default-codeblock">
```

```
</div>
   14/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:45 376ms/step - d_loss: 0.6945 - g_loss: 0.7095

<div class="k-default-codeblock">
```

```
</div>
   15/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:59 389ms/step - d_loss: 0.6923 - g_loss: 0.7100

<div class="k-default-codeblock">
```

```
</div>
   16/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:13 402ms/step - d_loss: 0.6902 - g_loss: 0.7105

<div class="k-default-codeblock">
```

```
</div>
   17/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:25 414ms/step - d_loss: 0.6881 - g_loss: 0.7110

<div class="k-default-codeblock">
```

```
</div>
   18/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:35 424ms/step - d_loss: 0.6860 - g_loss: 0.7115

<div class="k-default-codeblock">
```

```
</div>
   19/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:44 432ms/step - d_loss: 0.6839 - g_loss: 0.7119

<div class="k-default-codeblock">
```

```
</div>
   20/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:43 432ms/step - d_loss: 0.6819 - g_loss: 0.7124

<div class="k-default-codeblock">
```

```
</div>
   21/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:43 432ms/step - d_loss: 0.6799 - g_loss: 0.7128

<div class="k-default-codeblock">
```

```
</div>
   22/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:41 430ms/step - d_loss: 0.6779 - g_loss: 0.7132

<div class="k-default-codeblock">
```

```
</div>
   23/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:36 426ms/step - d_loss: 0.6760 - g_loss: 0.7136

<div class="k-default-codeblock">
```

```
</div>
   24/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:31 422ms/step - d_loss: 0.6741 - g_loss: 0.7140

<div class="k-default-codeblock">
```

```
</div>
   25/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:29 420ms/step - d_loss: 0.6723 - g_loss: 0.7143

<div class="k-default-codeblock">
```

```
</div>
   26/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:26 418ms/step - d_loss: 0.6705 - g_loss: 0.7145

<div class="k-default-codeblock">
```

```
</div>
   27/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:23 416ms/step - d_loss: 0.6687 - g_loss: 0.7148

<div class="k-default-codeblock">
```

```
</div>
   28/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:19 412ms/step - d_loss: 0.6670 - g_loss: 0.7150

<div class="k-default-codeblock">
```

```
</div>
   29/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:15 409ms/step - d_loss: 0.6653 - g_loss: 0.7152

<div class="k-default-codeblock">
```

```
</div>
   30/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:12 407ms/step - d_loss: 0.6636 - g_loss: 0.7154

<div class="k-default-codeblock">
```

```
</div>
   31/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:09 404ms/step - d_loss: 0.6620 - g_loss: 0.7156

<div class="k-default-codeblock">
```

```
</div>
   32/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:06 401ms/step - d_loss: 0.6603 - g_loss: 0.7158

<div class="k-default-codeblock">
```

```
</div>
   33/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:03 399ms/step - d_loss: 0.6587 - g_loss: 0.7160

<div class="k-default-codeblock">
```

```
</div>
   34/1094 [37m━━━━━━━━━━━━━━━━━━━━  7:01 397ms/step - d_loss: 0.6571 - g_loss: 0.7163

<div class="k-default-codeblock">
```

```
</div>
   35/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:58 396ms/step - d_loss: 0.6554 - g_loss: 0.7166

<div class="k-default-codeblock">
```

```
</div>
   36/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:56 394ms/step - d_loss: 0.6538 - g_loss: 0.7169

<div class="k-default-codeblock">
```

```
</div>
   37/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:54 392ms/step - d_loss: 0.6522 - g_loss: 0.7173

<div class="k-default-codeblock">
```

```
</div>
   38/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:53 391ms/step - d_loss: 0.6506 - g_loss: 0.7177

<div class="k-default-codeblock">
```

```
</div>
   39/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:51 390ms/step - d_loss: 0.6490 - g_loss: 0.7181

<div class="k-default-codeblock">
```

```
</div>
   40/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:50 389ms/step - d_loss: 0.6474 - g_loss: 0.7185

<div class="k-default-codeblock">
```

```
</div>
   41/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:48 388ms/step - d_loss: 0.6458 - g_loss: 0.7190

<div class="k-default-codeblock">
```

```
</div>
   42/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:47 387ms/step - d_loss: 0.6442 - g_loss: 0.7195

<div class="k-default-codeblock">
```

```
</div>
   43/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:45 386ms/step - d_loss: 0.6426 - g_loss: 0.7200

<div class="k-default-codeblock">
```

```
</div>
   44/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:43 385ms/step - d_loss: 0.6411 - g_loss: 0.7205

<div class="k-default-codeblock">
```

```
</div>
   45/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:42 384ms/step - d_loss: 0.6395 - g_loss: 0.7210

<div class="k-default-codeblock">
```

```
</div>
   46/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:41 383ms/step - d_loss: 0.6380 - g_loss: 0.7216

<div class="k-default-codeblock">
```

```
</div>
   47/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:39 382ms/step - d_loss: 0.6365 - g_loss: 0.7221

<div class="k-default-codeblock">
```

```
</div>
   48/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:38 381ms/step - d_loss: 0.6350 - g_loss: 0.7227

<div class="k-default-codeblock">
```

```
</div>
   49/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:36 380ms/step - d_loss: 0.6335 - g_loss: 0.7232

<div class="k-default-codeblock">
```

```
</div>
   50/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:35 379ms/step - d_loss: 0.6321 - g_loss: 0.7238

<div class="k-default-codeblock">
```

```
</div>
   51/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:34 378ms/step - d_loss: 0.6306 - g_loss: 0.7243

<div class="k-default-codeblock">
```

```
</div>
   52/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:32 377ms/step - d_loss: 0.6292 - g_loss: 0.7249

<div class="k-default-codeblock">
```

```
</div>
   53/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:31 376ms/step - d_loss: 0.6278 - g_loss: 0.7254

<div class="k-default-codeblock">
```

```
</div>
   54/1094 [37m━━━━━━━━━━━━━━━━━━━━  6:30 375ms/step - d_loss: 0.6264 - g_loss: 0.7259

<div class="k-default-codeblock">
```

```
</div>
   55/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:29 375ms/step - d_loss: 0.6250 - g_loss: 0.7264

<div class="k-default-codeblock">
```

```
</div>
   56/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:27 374ms/step - d_loss: 0.6236 - g_loss: 0.7270

<div class="k-default-codeblock">
```

```
</div>
   57/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:27 373ms/step - d_loss: 0.6223 - g_loss: 0.7275

<div class="k-default-codeblock">
```

```
</div>
   58/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:26 373ms/step - d_loss: 0.6209 - g_loss: 0.7281

<div class="k-default-codeblock">
```

```
</div>
   59/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:25 372ms/step - d_loss: 0.6196 - g_loss: 0.7286

<div class="k-default-codeblock">
```

```
</div>
   60/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:26 373ms/step - d_loss: 0.6182 - g_loss: 0.7292

<div class="k-default-codeblock">
```

```
</div>
   61/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:26 374ms/step - d_loss: 0.6169 - g_loss: 0.7298

<div class="k-default-codeblock">
```

```
</div>
   62/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:26 375ms/step - d_loss: 0.6155 - g_loss: 0.7304

<div class="k-default-codeblock">
```

```
</div>
   63/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:27 376ms/step - d_loss: 0.6142 - g_loss: 0.7310

<div class="k-default-codeblock">
```

```
</div>
   64/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:27 377ms/step - d_loss: 0.6128 - g_loss: 0.7315

<div class="k-default-codeblock">
```

```
</div>
   65/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:27 377ms/step - d_loss: 0.6115 - g_loss: 0.7321

<div class="k-default-codeblock">
```

```
</div>
   66/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:28 378ms/step - d_loss: 0.6102 - g_loss: 0.7327

<div class="k-default-codeblock">
```

```
</div>
   67/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:28 378ms/step - d_loss: 0.6089 - g_loss: 0.7332

<div class="k-default-codeblock">
```

```
</div>
   68/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:28 378ms/step - d_loss: 0.6076 - g_loss: 0.7338

<div class="k-default-codeblock">
```

```
</div>
   69/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:28 379ms/step - d_loss: 0.6063 - g_loss: 0.7343

<div class="k-default-codeblock">
```

```
</div>
   70/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:28 379ms/step - d_loss: 0.6050 - g_loss: 0.7349

<div class="k-default-codeblock">
```

```
</div>
   71/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:28 380ms/step - d_loss: 0.6037 - g_loss: 0.7354

<div class="k-default-codeblock">
```

```
</div>
   72/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:29 381ms/step - d_loss: 0.6024 - g_loss: 0.7360

<div class="k-default-codeblock">
```

```
</div>
   73/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:31 384ms/step - d_loss: 0.6012 - g_loss: 0.7365

<div class="k-default-codeblock">
```

```
</div>
   74/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:31 384ms/step - d_loss: 0.5999 - g_loss: 0.7371

<div class="k-default-codeblock">
```

```
</div>
   75/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:31 384ms/step - d_loss: 0.5986 - g_loss: 0.7376

<div class="k-default-codeblock">
```

```
</div>
   76/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:32 385ms/step - d_loss: 0.5974 - g_loss: 0.7381

<div class="k-default-codeblock">
```

```
</div>
   77/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:33 387ms/step - d_loss: 0.5961 - g_loss: 0.7387

<div class="k-default-codeblock">
```

```
</div>
   78/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:36 390ms/step - d_loss: 0.5949 - g_loss: 0.7392

<div class="k-default-codeblock">
```

```
</div>
   79/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:36 391ms/step - d_loss: 0.5936 - g_loss: 0.7398

<div class="k-default-codeblock">
```

```
</div>
   80/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:37 392ms/step - d_loss: 0.5924 - g_loss: 0.7403

<div class="k-default-codeblock">
```

```
</div>
   81/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:39 395ms/step - d_loss: 0.5912 - g_loss: 0.7409

<div class="k-default-codeblock">
```

```
</div>
   82/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:38 394ms/step - d_loss: 0.5899 - g_loss: 0.7415

<div class="k-default-codeblock">
```

```
</div>
   83/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:38 394ms/step - d_loss: 0.5887 - g_loss: 0.7420

<div class="k-default-codeblock">
```

```
</div>
   84/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:37 393ms/step - d_loss: 0.5875 - g_loss: 0.7426

<div class="k-default-codeblock">
```

```
</div>
   85/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:36 393ms/step - d_loss: 0.5863 - g_loss: 0.7432

<div class="k-default-codeblock">
```

```
</div>
   86/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:34 392ms/step - d_loss: 0.5851 - g_loss: 0.7437

<div class="k-default-codeblock">
```

```
</div>
   87/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:33 391ms/step - d_loss: 0.5839 - g_loss: 0.7443

<div class="k-default-codeblock">
```

```
</div>
   88/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:33 391ms/step - d_loss: 0.5827 - g_loss: 0.7449

<div class="k-default-codeblock">
```

```
</div>
   89/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:32 390ms/step - d_loss: 0.5815 - g_loss: 0.7455

<div class="k-default-codeblock">
```

```
</div>
   90/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:31 390ms/step - d_loss: 0.5803 - g_loss: 0.7462

<div class="k-default-codeblock">
```

```
</div>
   91/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:30 389ms/step - d_loss: 0.5791 - g_loss: 0.7468

<div class="k-default-codeblock">
```

```
</div>
   92/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:29 388ms/step - d_loss: 0.5779 - g_loss: 0.7474

<div class="k-default-codeblock">
```

```
</div>
   93/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:28 388ms/step - d_loss: 0.5767 - g_loss: 0.7481

<div class="k-default-codeblock">
```

```
</div>
   94/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:27 388ms/step - d_loss: 0.5755 - g_loss: 0.7487

<div class="k-default-codeblock">
```

```
</div>
   95/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:26 387ms/step - d_loss: 0.5744 - g_loss: 0.7494

<div class="k-default-codeblock">
```

```
</div>
   96/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:25 386ms/step - d_loss: 0.5732 - g_loss: 0.7501

<div class="k-default-codeblock">
```

```
</div>
   97/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:24 386ms/step - d_loss: 0.5720 - g_loss: 0.7508

<div class="k-default-codeblock">
```

```
</div>
   98/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:23 385ms/step - d_loss: 0.5709 - g_loss: 0.7515

<div class="k-default-codeblock">
```

```
</div>
   99/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:23 385ms/step - d_loss: 0.5697 - g_loss: 0.7522

<div class="k-default-codeblock">
```

```
</div>
  100/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:22 385ms/step - d_loss: 0.5686 - g_loss: 0.7530

<div class="k-default-codeblock">
```

```
</div>
  101/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:21 384ms/step - d_loss: 0.5674 - g_loss: 0.7537

<div class="k-default-codeblock">
```

```
</div>
  102/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:20 384ms/step - d_loss: 0.5663 - g_loss: 0.7545

<div class="k-default-codeblock">
```

```
</div>
  103/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:19 383ms/step - d_loss: 0.5651 - g_loss: 0.7553

<div class="k-default-codeblock">
```

```
</div>
  104/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:18 383ms/step - d_loss: 0.5640 - g_loss: 0.7561

<div class="k-default-codeblock">
```

```
</div>
  105/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:18 382ms/step - d_loss: 0.5628 - g_loss: 0.7569

<div class="k-default-codeblock">
```

```
</div>
  106/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:17 382ms/step - d_loss: 0.5617 - g_loss: 0.7578

<div class="k-default-codeblock">
```

```
</div>
  107/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:16 382ms/step - d_loss: 0.5605 - g_loss: 0.7587

<div class="k-default-codeblock">
```

```
</div>
  108/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:16 382ms/step - d_loss: 0.5594 - g_loss: 0.7595

<div class="k-default-codeblock">
```

```
</div>
  109/1094 ━[37m━━━━━━━━━━━━━━━━━━━  6:15 381ms/step - d_loss: 0.5583 - g_loss: 0.7605

<div class="k-default-codeblock">
```

```
</div>
  110/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:14 381ms/step - d_loss: 0.5571 - g_loss: 0.7614

<div class="k-default-codeblock">
```

```
</div>
  111/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:13 380ms/step - d_loss: 0.5560 - g_loss: 0.7623

<div class="k-default-codeblock">
```

```
</div>
  112/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:12 380ms/step - d_loss: 0.5549 - g_loss: 0.7633

<div class="k-default-codeblock">
```

```
</div>
  113/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:12 379ms/step - d_loss: 0.5538 - g_loss: 0.7643

<div class="k-default-codeblock">
```

```
</div>
  114/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:11 379ms/step - d_loss: 0.5526 - g_loss: 0.7653

<div class="k-default-codeblock">
```

```
</div>
  115/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:10 379ms/step - d_loss: 0.5515 - g_loss: 0.7664

<div class="k-default-codeblock">
```

```
</div>
  116/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:09 378ms/step - d_loss: 0.5504 - g_loss: 0.7675

<div class="k-default-codeblock">
```

```
</div>
  117/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:09 378ms/step - d_loss: 0.5493 - g_loss: 0.7686

<div class="k-default-codeblock">
```

```
</div>
  118/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:08 378ms/step - d_loss: 0.5482 - g_loss: 0.7697

<div class="k-default-codeblock">
```

```
</div>
  119/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:07 377ms/step - d_loss: 0.5470 - g_loss: 0.7709

<div class="k-default-codeblock">
```

```
</div>
  120/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:07 377ms/step - d_loss: 0.5459 - g_loss: 0.7721

<div class="k-default-codeblock">
```

```
</div>
  121/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:06 376ms/step - d_loss: 0.5448 - g_loss: 0.7733

<div class="k-default-codeblock">
```

```
</div>
  122/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:05 376ms/step - d_loss: 0.5437 - g_loss: 0.7745

<div class="k-default-codeblock">
```

```
</div>
  123/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:04 376ms/step - d_loss: 0.5426 - g_loss: 0.7758

<div class="k-default-codeblock">
```

```
</div>
  124/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:04 376ms/step - d_loss: 0.5415 - g_loss: 0.7771

<div class="k-default-codeblock">
```

```
</div>
  125/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:03 375ms/step - d_loss: 0.5404 - g_loss: 0.7784

<div class="k-default-codeblock">
```

```
</div>
  126/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:02 375ms/step - d_loss: 0.5393 - g_loss: 0.7798

<div class="k-default-codeblock">
```

```
</div>
  127/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:02 375ms/step - d_loss: 0.5382 - g_loss: 0.7812

<div class="k-default-codeblock">
```

```
</div>
  128/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:01 374ms/step - d_loss: 0.5371 - g_loss: 0.7826

<div class="k-default-codeblock">
```

```
</div>
  129/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:00 374ms/step - d_loss: 0.5360 - g_loss: 0.7841

<div class="k-default-codeblock">
```

```
</div>
  130/1094 ━━[37m━━━━━━━━━━━━━━━━━━  6:00 374ms/step - d_loss: 0.5349 - g_loss: 0.7856

<div class="k-default-codeblock">
```

```
</div>
  131/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:59 373ms/step - d_loss: 0.5338 - g_loss: 0.7871

<div class="k-default-codeblock">
```

```
</div>
  132/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:58 373ms/step - d_loss: 0.5327 - g_loss: 0.7887

<div class="k-default-codeblock">
```

```
</div>
  133/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:58 373ms/step - d_loss: 0.5316 - g_loss: 0.7903

<div class="k-default-codeblock">
```

```
</div>
  134/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:57 372ms/step - d_loss: 0.5305 - g_loss: 0.7920

<div class="k-default-codeblock">
```

```
</div>
  135/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:56 372ms/step - d_loss: 0.5294 - g_loss: 0.7936

<div class="k-default-codeblock">
```

```
</div>
  136/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:56 372ms/step - d_loss: 0.5283 - g_loss: 0.7953

<div class="k-default-codeblock">
```

```
</div>
  137/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:55 371ms/step - d_loss: 0.5272 - g_loss: 0.7971

<div class="k-default-codeblock">
```

```
</div>
  138/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:54 371ms/step - d_loss: 0.5261 - g_loss: 0.7989

<div class="k-default-codeblock">
```

```
</div>
  139/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:54 371ms/step - d_loss: 0.5250 - g_loss: 0.8007

<div class="k-default-codeblock">
```

```
</div>
  140/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:53 371ms/step - d_loss: 0.5239 - g_loss: 0.8025

<div class="k-default-codeblock">
```

```
</div>
  141/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:53 371ms/step - d_loss: 0.5228 - g_loss: 0.8044

<div class="k-default-codeblock">
```

```
</div>
  142/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:52 370ms/step - d_loss: 0.5217 - g_loss: 0.8064

<div class="k-default-codeblock">
```

```
</div>
  143/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:51 370ms/step - d_loss: 0.5206 - g_loss: 0.8083

<div class="k-default-codeblock">
```

```
</div>
  144/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:51 370ms/step - d_loss: 0.5196 - g_loss: 0.8103

<div class="k-default-codeblock">
```

```
</div>
  145/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:50 369ms/step - d_loss: 0.5185 - g_loss: 0.8124

<div class="k-default-codeblock">
```

```
</div>
  146/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:49 369ms/step - d_loss: 0.5174 - g_loss: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  147/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:49 369ms/step - d_loss: 0.5163 - g_loss: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  148/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:48 369ms/step - d_loss: 0.5152 - g_loss: 0.8187

<div class="k-default-codeblock">
```

```
</div>
  149/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:48 368ms/step - d_loss: 0.5142 - g_loss: 0.8209

<div class="k-default-codeblock">
```

```
</div>
  150/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:47 368ms/step - d_loss: 0.5131 - g_loss: 0.8231

<div class="k-default-codeblock">
```

```
</div>
  151/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:47 368ms/step - d_loss: 0.5120 - g_loss: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  152/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:46 368ms/step - d_loss: 0.5109 - g_loss: 0.8277

<div class="k-default-codeblock">
```

```
</div>
  153/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:46 368ms/step - d_loss: 0.5099 - g_loss: 0.8300

<div class="k-default-codeblock">
```

```
</div>
  154/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:45 368ms/step - d_loss: 0.5088 - g_loss: 0.8324

<div class="k-default-codeblock">
```

```
</div>
  155/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:44 367ms/step - d_loss: 0.5077 - g_loss: 0.8348

<div class="k-default-codeblock">
```

```
</div>
  156/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:44 367ms/step - d_loss: 0.5067 - g_loss: 0.8373

<div class="k-default-codeblock">
```

```
</div>
  157/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:43 367ms/step - d_loss: 0.5056 - g_loss: 0.8398

<div class="k-default-codeblock">
```

```
</div>
  158/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:43 367ms/step - d_loss: 0.5046 - g_loss: 0.8423

<div class="k-default-codeblock">
```

```
</div>
  159/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:42 366ms/step - d_loss: 0.5035 - g_loss: 0.8448

<div class="k-default-codeblock">
```

```
</div>
  160/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:42 366ms/step - d_loss: 0.5025 - g_loss: 0.8474

<div class="k-default-codeblock">
```

```
</div>
  161/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:41 366ms/step - d_loss: 0.5014 - g_loss: 0.8501

<div class="k-default-codeblock">
```

```
</div>
  162/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:40 366ms/step - d_loss: 0.5004 - g_loss: 0.8527

<div class="k-default-codeblock">
```

```
</div>
  163/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:40 366ms/step - d_loss: 0.4993 - g_loss: 0.8554

<div class="k-default-codeblock">
```

```
</div>
  164/1094 ━━[37m━━━━━━━━━━━━━━━━━━  5:39 365ms/step - d_loss: 0.4983 - g_loss: 0.8582

<div class="k-default-codeblock">
```

```
</div>
  165/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:39 365ms/step - d_loss: 0.4972 - g_loss: 0.8610

<div class="k-default-codeblock">
```

```
</div>
  166/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:38 365ms/step - d_loss: 0.4962 - g_loss: 0.8638

<div class="k-default-codeblock">
```

```
</div>
  167/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:38 365ms/step - d_loss: 0.4952 - g_loss: 0.8666

<div class="k-default-codeblock">
```

```
</div>
  168/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:37 365ms/step - d_loss: 0.4941 - g_loss: 0.8695

<div class="k-default-codeblock">
```

```
</div>
  169/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:37 365ms/step - d_loss: 0.4931 - g_loss: 0.8724

<div class="k-default-codeblock">
```

```
</div>
  170/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:36 364ms/step - d_loss: 0.4921 - g_loss: 0.8754

<div class="k-default-codeblock">
```

```
</div>
  171/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:36 364ms/step - d_loss: 0.4911 - g_loss: 0.8783

<div class="k-default-codeblock">
```

```
</div>
  172/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:35 364ms/step - d_loss: 0.4900 - g_loss: 0.8814

<div class="k-default-codeblock">
```

```
</div>
  173/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:35 364ms/step - d_loss: 0.4890 - g_loss: 0.8844

<div class="k-default-codeblock">
```

```
</div>
  174/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:34 364ms/step - d_loss: 0.4880 - g_loss: 0.8875

<div class="k-default-codeblock">
```

```
</div>
  175/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:34 364ms/step - d_loss: 0.4870 - g_loss: 0.8906

<div class="k-default-codeblock">
```

```
</div>
  176/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:33 363ms/step - d_loss: 0.4860 - g_loss: 0.8938

<div class="k-default-codeblock">
```

```
</div>
  177/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:32 363ms/step - d_loss: 0.4850 - g_loss: 0.8969

<div class="k-default-codeblock">
```

```
</div>
  178/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:32 363ms/step - d_loss: 0.4840 - g_loss: 0.9001

<div class="k-default-codeblock">
```

```
</div>
  179/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:31 363ms/step - d_loss: 0.4830 - g_loss: 0.9034

<div class="k-default-codeblock">
```

```
</div>
  180/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:31 363ms/step - d_loss: 0.4820 - g_loss: 0.9067

<div class="k-default-codeblock">
```

```
</div>
  181/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:30 362ms/step - d_loss: 0.4810 - g_loss: 0.9100

<div class="k-default-codeblock">
```

```
</div>
  182/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:30 362ms/step - d_loss: 0.4800 - g_loss: 0.9133

<div class="k-default-codeblock">
```

```
</div>
  183/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:29 362ms/step - d_loss: 0.4790 - g_loss: 0.9167

<div class="k-default-codeblock">
```

```
</div>
  184/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:29 362ms/step - d_loss: 0.4780 - g_loss: 0.9201

<div class="k-default-codeblock">
```

```
</div>
  185/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:28 362ms/step - d_loss: 0.4770 - g_loss: 0.9235

<div class="k-default-codeblock">
```

```
</div>
  186/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:28 362ms/step - d_loss: 0.4760 - g_loss: 0.9270

<div class="k-default-codeblock">
```

```
</div>
  187/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:27 362ms/step - d_loss: 0.4751 - g_loss: 0.9305

<div class="k-default-codeblock">
```

```
</div>
  188/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:27 362ms/step - d_loss: 0.4741 - g_loss: 0.9340

<div class="k-default-codeblock">
```

```
</div>
  189/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:27 361ms/step - d_loss: 0.4731 - g_loss: 0.9375

<div class="k-default-codeblock">
```

```
</div>
  190/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:26 361ms/step - d_loss: 0.4721 - g_loss: 0.9411

<div class="k-default-codeblock">
```

```
</div>
  191/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:26 361ms/step - d_loss: 0.4712 - g_loss: 0.9447

<div class="k-default-codeblock">
```

```
</div>
  192/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:25 361ms/step - d_loss: 0.4702 - g_loss: 0.9483

<div class="k-default-codeblock">
```

```
</div>
  193/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:25 361ms/step - d_loss: 0.4693 - g_loss: 0.9520

<div class="k-default-codeblock">
```

```
</div>
  194/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:24 361ms/step - d_loss: 0.4683 - g_loss: 0.9556

<div class="k-default-codeblock">
```

```
</div>
  195/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:24 361ms/step - d_loss: 0.4673 - g_loss: 0.9593

<div class="k-default-codeblock">
```

```
</div>
  196/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:23 360ms/step - d_loss: 0.4664 - g_loss: 0.9631

<div class="k-default-codeblock">
```

```
</div>
  197/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:23 360ms/step - d_loss: 0.4654 - g_loss: 0.9668

<div class="k-default-codeblock">
```

```
</div>
  198/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:22 360ms/step - d_loss: 0.4645 - g_loss: 0.9706

<div class="k-default-codeblock">
```

```
</div>
  199/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:22 360ms/step - d_loss: 0.4636 - g_loss: 0.9744

<div class="k-default-codeblock">
```

```
</div>
  200/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:21 360ms/step - d_loss: 0.4626 - g_loss: 0.9782

<div class="k-default-codeblock">
```

```
</div>
  201/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:21 360ms/step - d_loss: 0.4617 - g_loss: 0.9821

<div class="k-default-codeblock">
```

```
</div>
  202/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:20 360ms/step - d_loss: 0.4608 - g_loss: 0.9860

<div class="k-default-codeblock">
```

```
</div>
  203/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:20 360ms/step - d_loss: 0.4598 - g_loss: 0.9899

<div class="k-default-codeblock">
```

```
</div>
  204/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:19 360ms/step - d_loss: 0.4589 - g_loss: 0.9938

<div class="k-default-codeblock">
```

```
</div>
  205/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:19 359ms/step - d_loss: 0.4580 - g_loss: 0.9977

<div class="k-default-codeblock">
```

```
</div>
  206/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:19 359ms/step - d_loss: 0.4571 - g_loss: 1.0017

<div class="k-default-codeblock">
```

```
</div>
  207/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:18 359ms/step - d_loss: 0.4562 - g_loss: 1.0057

<div class="k-default-codeblock">
```

```
</div>
  208/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:18 359ms/step - d_loss: 0.4552 - g_loss: 1.0097

<div class="k-default-codeblock">
```

```
</div>
  209/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:17 359ms/step - d_loss: 0.4543 - g_loss: 1.0138

<div class="k-default-codeblock">
```

```
</div>
  210/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:17 359ms/step - d_loss: 0.4534 - g_loss: 1.0178

<div class="k-default-codeblock">
```

```
</div>
  211/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:16 359ms/step - d_loss: 0.4525 - g_loss: 1.0219

<div class="k-default-codeblock">
```

```
</div>
  212/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:16 358ms/step - d_loss: 0.4516 - g_loss: 1.0260

<div class="k-default-codeblock">
```

```
</div>
  213/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:15 358ms/step - d_loss: 0.4507 - g_loss: 1.0301

<div class="k-default-codeblock">
```

```
</div>
  214/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:15 358ms/step - d_loss: 0.4498 - g_loss: 1.0343

<div class="k-default-codeblock">
```

```
</div>
  215/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:14 358ms/step - d_loss: 0.4490 - g_loss: 1.0384

<div class="k-default-codeblock">
```

```
</div>
  216/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:14 358ms/step - d_loss: 0.4481 - g_loss: 1.0426

<div class="k-default-codeblock">
```

```
</div>
  217/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:13 358ms/step - d_loss: 0.4472 - g_loss: 1.0468

<div class="k-default-codeblock">
```

```
</div>
  218/1094 ━━━[37m━━━━━━━━━━━━━━━━━  5:13 358ms/step - d_loss: 0.4463 - g_loss: 1.0510

<div class="k-default-codeblock">
```

```
</div>
  219/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:13 358ms/step - d_loss: 0.4454 - g_loss: 1.0553

<div class="k-default-codeblock">
```

```
</div>
  220/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:12 358ms/step - d_loss: 0.4445 - g_loss: 1.0595

<div class="k-default-codeblock">
```

```
</div>
  221/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:12 357ms/step - d_loss: 0.4437 - g_loss: 1.0638

<div class="k-default-codeblock">
```

```
</div>
  222/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:11 357ms/step - d_loss: 0.4428 - g_loss: 1.0681

<div class="k-default-codeblock">
```

```
</div>
  223/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:11 357ms/step - d_loss: 0.4419 - g_loss: 1.0724

<div class="k-default-codeblock">
```

```
</div>
  224/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:10 357ms/step - d_loss: 0.4411 - g_loss: 1.0767

<div class="k-default-codeblock">
```

```
</div>
  225/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:10 357ms/step - d_loss: 0.4402 - g_loss: 1.0811

<div class="k-default-codeblock">
```

```
</div>
  226/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:09 357ms/step - d_loss: 0.4394 - g_loss: 1.0855

<div class="k-default-codeblock">
```

```
</div>
  227/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:09 357ms/step - d_loss: 0.4385 - g_loss: 1.0899

<div class="k-default-codeblock">
```

```
</div>
  228/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 357ms/step - d_loss: 0.4377 - g_loss: 1.0943

<div class="k-default-codeblock">
```

```
</div>
  229/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 356ms/step - d_loss: 0.4368 - g_loss: 1.0987

<div class="k-default-codeblock">
```

```
</div>
  230/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 357ms/step - d_loss: 0.4360 - g_loss: 1.1031

<div class="k-default-codeblock">
```

```
</div>
  231/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 357ms/step - d_loss: 0.4351 - g_loss: 1.1076

<div class="k-default-codeblock">
```

```
</div>
  232/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 357ms/step - d_loss: 0.4343 - g_loss: 1.1120

<div class="k-default-codeblock">
```

```
</div>
  233/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 358ms/step - d_loss: 0.4335 - g_loss: 1.1165

<div class="k-default-codeblock">
```

```
</div>
  234/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 358ms/step - d_loss: 0.4326 - g_loss: 1.1210

<div class="k-default-codeblock">
```

```
</div>
  235/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 358ms/step - d_loss: 0.4318 - g_loss: 1.1255

<div class="k-default-codeblock">
```

```
</div>
  236/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 359ms/step - d_loss: 0.4310 - g_loss: 1.1301

<div class="k-default-codeblock">
```

```
</div>
  237/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 359ms/step - d_loss: 0.4301 - g_loss: 1.1346

<div class="k-default-codeblock">
```

```
</div>
  238/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 359ms/step - d_loss: 0.4293 - g_loss: 1.1392

<div class="k-default-codeblock">
```

```
</div>
  239/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 359ms/step - d_loss: 0.4285 - g_loss: 1.1438

<div class="k-default-codeblock">
```

```
</div>
  240/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 360ms/step - d_loss: 0.4277 - g_loss: 1.1483

<div class="k-default-codeblock">
```

```
</div>
  241/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 361ms/step - d_loss: 0.4269 - g_loss: 1.1529

<div class="k-default-codeblock">
```

```
</div>
  242/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 362ms/step - d_loss: 0.4261 - g_loss: 1.1576

<div class="k-default-codeblock">
```

```
</div>
  243/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 362ms/step - d_loss: 0.4253 - g_loss: 1.1622

<div class="k-default-codeblock">
```

```
</div>
  244/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 363ms/step - d_loss: 0.4245 - g_loss: 1.1668

<div class="k-default-codeblock">
```

```
</div>
  245/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 363ms/step - d_loss: 0.4237 - g_loss: 1.1715

<div class="k-default-codeblock">
```

```
</div>
  246/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 363ms/step - d_loss: 0.4229 - g_loss: 1.1762

<div class="k-default-codeblock">
```

```
</div>
  247/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 363ms/step - d_loss: 0.4221 - g_loss: 1.1809

<div class="k-default-codeblock">
```

```
</div>
  248/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 363ms/step - d_loss: 0.4213 - g_loss: 1.1856

<div class="k-default-codeblock">
```

```
</div>
  249/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 364ms/step - d_loss: 0.4205 - g_loss: 1.1903

<div class="k-default-codeblock">
```

```
</div>
  250/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 365ms/step - d_loss: 0.4197 - g_loss: 1.1950

<div class="k-default-codeblock">
```

```
</div>
  251/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:08 365ms/step - d_loss: 0.4189 - g_loss: 1.1998

<div class="k-default-codeblock">
```

```
</div>
  252/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 365ms/step - d_loss: 0.4181 - g_loss: 1.2045

<div class="k-default-codeblock">
```

```
</div>
  253/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:07 365ms/step - d_loss: 0.4174 - g_loss: 1.2093

<div class="k-default-codeblock">
```

```
</div>
  254/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:06 365ms/step - d_loss: 0.4166 - g_loss: 1.2141

<div class="k-default-codeblock">
```

```
</div>
  255/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:06 365ms/step - d_loss: 0.4158 - g_loss: 1.2189

<div class="k-default-codeblock">
```

```
</div>
  256/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:05 365ms/step - d_loss: 0.4151 - g_loss: 1.2237

<div class="k-default-codeblock">
```

```
</div>
  257/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:05 365ms/step - d_loss: 0.4143 - g_loss: 1.2285

<div class="k-default-codeblock">
```

```
</div>
  258/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:04 365ms/step - d_loss: 0.4135 - g_loss: 1.2333

<div class="k-default-codeblock">
```

```
</div>
  259/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:04 365ms/step - d_loss: 0.4128 - g_loss: 1.2382

<div class="k-default-codeblock">
```

```
</div>
  260/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:03 364ms/step - d_loss: 0.4120 - g_loss: 1.2430

<div class="k-default-codeblock">
```

```
</div>
  261/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:03 364ms/step - d_loss: 0.4112 - g_loss: 1.2479

<div class="k-default-codeblock">
```

```
</div>
  262/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:03 364ms/step - d_loss: 0.4105 - g_loss: 1.2528

<div class="k-default-codeblock">
```

```
</div>
  263/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:02 364ms/step - d_loss: 0.4097 - g_loss: 1.2577

<div class="k-default-codeblock">
```

```
</div>
  264/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:02 364ms/step - d_loss: 0.4090 - g_loss: 1.2626

<div class="k-default-codeblock">
```

```
</div>
  265/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:01 364ms/step - d_loss: 0.4082 - g_loss: 1.2675

<div class="k-default-codeblock">
```

```
</div>
  266/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:01 364ms/step - d_loss: 0.4075 - g_loss: 1.2724

<div class="k-default-codeblock">
```

```
</div>
  267/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:01 364ms/step - d_loss: 0.4068 - g_loss: 1.2774

<div class="k-default-codeblock">
```

```
</div>
  268/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:01 364ms/step - d_loss: 0.4060 - g_loss: 1.2823

<div class="k-default-codeblock">
```

```
</div>
  269/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:00 365ms/step - d_loss: 0.4053 - g_loss: 1.2873

<div class="k-default-codeblock">
```

```
</div>
  270/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:00 365ms/step - d_loss: 0.4046 - g_loss: 1.2923

<div class="k-default-codeblock">
```

```
</div>
  271/1094 ━━━━[37m━━━━━━━━━━━━━━━━  5:00 365ms/step - d_loss: 0.4038 - g_loss: 1.2973

<div class="k-default-codeblock">
```

```
</div>
  272/1094 ━━━━[37m━━━━━━━━━━━━━━━━  4:59 365ms/step - d_loss: 0.4031 - g_loss: 1.3023

<div class="k-default-codeblock">
```

```
</div>
  273/1094 ━━━━[37m━━━━━━━━━━━━━━━━  4:59 364ms/step - d_loss: 0.4024 - g_loss: 1.3073

<div class="k-default-codeblock">
```

```
</div>
  274/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:58 364ms/step - d_loss: 0.4017 - g_loss: 1.3123

<div class="k-default-codeblock">
```

```
</div>
  275/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:58 364ms/step - d_loss: 0.4009 - g_loss: 1.3173

<div class="k-default-codeblock">
```

```
</div>
  276/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:57 364ms/step - d_loss: 0.4002 - g_loss: 1.3224

<div class="k-default-codeblock">
```

```
</div>
  277/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:57 364ms/step - d_loss: 0.3995 - g_loss: 1.3274

<div class="k-default-codeblock">
```

```
</div>
  278/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:57 364ms/step - d_loss: 0.3988 - g_loss: 1.3325

<div class="k-default-codeblock">
```

```
</div>
  279/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:56 364ms/step - d_loss: 0.3981 - g_loss: 1.3376

<div class="k-default-codeblock">
```

```
</div>
  280/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:56 364ms/step - d_loss: 0.3974 - g_loss: 1.3427

<div class="k-default-codeblock">
```

```
</div>
  281/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:55 364ms/step - d_loss: 0.3967 - g_loss: 1.3478

<div class="k-default-codeblock">
```

```
</div>
  282/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:55 364ms/step - d_loss: 0.3960 - g_loss: 1.3529

<div class="k-default-codeblock">
```

```
</div>
  283/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:54 364ms/step - d_loss: 0.3953 - g_loss: 1.3580

<div class="k-default-codeblock">
```

```
</div>
  284/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:54 363ms/step - d_loss: 0.3946 - g_loss: 1.3632

<div class="k-default-codeblock">
```

```
</div>
  285/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:53 363ms/step - d_loss: 0.3939 - g_loss: 1.3683

<div class="k-default-codeblock">
```

```
</div>
  286/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:53 363ms/step - d_loss: 0.3932 - g_loss: 1.3735

<div class="k-default-codeblock">
```

```
</div>
  287/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:53 363ms/step - d_loss: 0.3925 - g_loss: 1.3787

<div class="k-default-codeblock">
```

```
</div>
  288/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:52 363ms/step - d_loss: 0.3918 - g_loss: 1.3838

<div class="k-default-codeblock">
```

```
</div>
  289/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:52 363ms/step - d_loss: 0.3911 - g_loss: 1.3890

<div class="k-default-codeblock">
```

```
</div>
  290/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:51 363ms/step - d_loss: 0.3904 - g_loss: 1.3942

<div class="k-default-codeblock">
```

```
</div>
  291/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:51 363ms/step - d_loss: 0.3897 - g_loss: 1.3995

<div class="k-default-codeblock">
```

```
</div>
  292/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:50 363ms/step - d_loss: 0.3890 - g_loss: 1.4047

<div class="k-default-codeblock">
```

```
</div>
  293/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:50 363ms/step - d_loss: 0.3884 - g_loss: 1.4099

<div class="k-default-codeblock">
```

```
</div>
  294/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:50 363ms/step - d_loss: 0.3877 - g_loss: 1.4152

<div class="k-default-codeblock">
```

```
</div>
  295/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:49 363ms/step - d_loss: 0.3870 - g_loss: 1.4204

<div class="k-default-codeblock">
```

```
</div>
  296/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:49 363ms/step - d_loss: 0.3863 - g_loss: 1.4257

<div class="k-default-codeblock">
```

```
</div>
  297/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:48 363ms/step - d_loss: 0.3857 - g_loss: 1.4310

<div class="k-default-codeblock">
```

```
</div>
  298/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:48 362ms/step - d_loss: 0.3850 - g_loss: 1.4363

<div class="k-default-codeblock">
```

```
</div>
  299/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:48 362ms/step - d_loss: 0.3843 - g_loss: 1.4416

<div class="k-default-codeblock">
```

```
</div>
  300/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:47 362ms/step - d_loss: 0.3837 - g_loss: 1.4469

<div class="k-default-codeblock">
```

```
</div>
  301/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:47 362ms/step - d_loss: 0.3830 - g_loss: 1.4522

<div class="k-default-codeblock">
```

```
</div>
  302/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:46 362ms/step - d_loss: 0.3824 - g_loss: 1.4576

<div class="k-default-codeblock">
```

```
</div>
  303/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:46 362ms/step - d_loss: 0.3817 - g_loss: 1.4629

<div class="k-default-codeblock">
```

```
</div>
  304/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:45 362ms/step - d_loss: 0.3810 - g_loss: 1.4683

<div class="k-default-codeblock">
```

```
</div>
  305/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:45 362ms/step - d_loss: 0.3804 - g_loss: 1.4736

<div class="k-default-codeblock">
```

```
</div>
  306/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:44 362ms/step - d_loss: 0.3797 - g_loss: 1.4790

<div class="k-default-codeblock">
```

```
</div>
  307/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:44 362ms/step - d_loss: 0.3791 - g_loss: 1.4844

<div class="k-default-codeblock">
```

```
</div>
  308/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:44 361ms/step - d_loss: 0.3784 - g_loss: 1.4898

<div class="k-default-codeblock">
```

```
</div>
  309/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:43 361ms/step - d_loss: 0.3778 - g_loss: 1.4952

<div class="k-default-codeblock">
```

```
</div>
  310/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:43 361ms/step - d_loss: 0.3772 - g_loss: 1.5007

<div class="k-default-codeblock">
```

```
</div>
  311/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:42 361ms/step - d_loss: 0.3765 - g_loss: 1.5061

<div class="k-default-codeblock">
```

```
</div>
  312/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:42 361ms/step - d_loss: 0.3759 - g_loss: 1.5116

<div class="k-default-codeblock">
```

```
</div>
  313/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:41 361ms/step - d_loss: 0.3753 - g_loss: 1.5170

<div class="k-default-codeblock">
```

```
</div>
  314/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:41 361ms/step - d_loss: 0.3746 - g_loss: 1.5225

<div class="k-default-codeblock">
```

```
</div>
  315/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:41 361ms/step - d_loss: 0.3740 - g_loss: 1.5280

<div class="k-default-codeblock">
```

```
</div>
  316/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:40 361ms/step - d_loss: 0.3734 - g_loss: 1.5335

<div class="k-default-codeblock">
```

```
</div>
  317/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:40 361ms/step - d_loss: 0.3727 - g_loss: 1.5390

<div class="k-default-codeblock">
```

```
</div>
  318/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:39 360ms/step - d_loss: 0.3721 - g_loss: 1.5445

<div class="k-default-codeblock">
```

```
</div>
  319/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:39 360ms/step - d_loss: 0.3715 - g_loss: 1.5500

<div class="k-default-codeblock">
```

```
</div>
  320/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:38 360ms/step - d_loss: 0.3709 - g_loss: 1.5556

<div class="k-default-codeblock">
```

```
</div>
  321/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:38 360ms/step - d_loss: 0.3702 - g_loss: 1.5611

<div class="k-default-codeblock">
```

```
</div>
  322/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:38 360ms/step - d_loss: 0.3696 - g_loss: 1.5667

<div class="k-default-codeblock">
```

```
</div>
  323/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 360ms/step - d_loss: 0.3690 - g_loss: 1.5722

<div class="k-default-codeblock">
```

```
</div>
  324/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:37 360ms/step - d_loss: 0.3684 - g_loss: 1.5778

<div class="k-default-codeblock">
```

```
</div>
  325/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:36 360ms/step - d_loss: 0.3678 - g_loss: 1.5834

<div class="k-default-codeblock">
```

```
</div>
  326/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:36 360ms/step - d_loss: 0.3672 - g_loss: 1.5890

<div class="k-default-codeblock">
```

```
</div>
  327/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:36 360ms/step - d_loss: 0.3666 - g_loss: 1.5946

<div class="k-default-codeblock">
```

```
</div>
  328/1094 ━━━━━[37m━━━━━━━━━━━━━━━  4:35 360ms/step - d_loss: 0.3660 - g_loss: 1.6002

<div class="k-default-codeblock">
```

```
</div>
  329/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:35 360ms/step - d_loss: 0.3654 - g_loss: 1.6059

<div class="k-default-codeblock">
```

```
</div>
  330/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:34 360ms/step - d_loss: 0.3648 - g_loss: 1.6115

<div class="k-default-codeblock">
```

```
</div>
  331/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:34 360ms/step - d_loss: 0.3642 - g_loss: 1.6172

<div class="k-default-codeblock">
```

```
</div>
  332/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:33 360ms/step - d_loss: 0.3636 - g_loss: 1.6228

<div class="k-default-codeblock">
```

```
</div>
  333/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:33 359ms/step - d_loss: 0.3630 - g_loss: 1.6285

<div class="k-default-codeblock">
```

```
</div>
  334/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:33 359ms/step - d_loss: 0.3624 - g_loss: 1.6342

<div class="k-default-codeblock">
```

```
</div>
  335/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:32 359ms/step - d_loss: 0.3618 - g_loss: 1.6399

<div class="k-default-codeblock">
```

```
</div>
  336/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:32 359ms/step - d_loss: 0.3612 - g_loss: 1.6456

<div class="k-default-codeblock">
```

```
</div>
  337/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:31 359ms/step - d_loss: 0.3606 - g_loss: 1.6513

<div class="k-default-codeblock">
```

```
</div>
  338/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:31 359ms/step - d_loss: 0.3600 - g_loss: 1.6571

<div class="k-default-codeblock">
```

```
</div>
  339/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:31 359ms/step - d_loss: 0.3594 - g_loss: 1.6628

<div class="k-default-codeblock">
```

```
</div>
  340/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:30 359ms/step - d_loss: 0.3588 - g_loss: 1.6686

<div class="k-default-codeblock">
```

```
</div>
  341/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:30 359ms/step - d_loss: 0.3582 - g_loss: 1.6743

<div class="k-default-codeblock">
```

```
</div>
  342/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:29 359ms/step - d_loss: 0.3577 - g_loss: 1.6801

<div class="k-default-codeblock">
```

```
</div>
  343/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:29 359ms/step - d_loss: 0.3571 - g_loss: 1.6859

<div class="k-default-codeblock">
```

```
</div>
  344/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:29 359ms/step - d_loss: 0.3565 - g_loss: 1.6917

<div class="k-default-codeblock">
```

```
</div>
  345/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:28 359ms/step - d_loss: 0.3559 - g_loss: 1.6975

<div class="k-default-codeblock">
```

```
</div>
  346/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:28 359ms/step - d_loss: 0.3554 - g_loss: 1.7033

<div class="k-default-codeblock">
```

```
</div>
  347/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:27 359ms/step - d_loss: 0.3548 - g_loss: 1.7092

<div class="k-default-codeblock">
```

```
</div>
  348/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:27 359ms/step - d_loss: 0.3542 - g_loss: 1.7150

<div class="k-default-codeblock">
```

```
</div>
  349/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:27 358ms/step - d_loss: 0.3536 - g_loss: 1.7209

<div class="k-default-codeblock">
```

```
</div>
  350/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:26 358ms/step - d_loss: 0.3531 - g_loss: 1.7267

<div class="k-default-codeblock">
```

```
</div>
  351/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:26 358ms/step - d_loss: 0.3525 - g_loss: 1.7326

<div class="k-default-codeblock">
```

```
</div>
  352/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:25 358ms/step - d_loss: 0.3519 - g_loss: 1.7385

<div class="k-default-codeblock">
```

```
</div>
  353/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:25 358ms/step - d_loss: 0.3514 - g_loss: 1.7444

<div class="k-default-codeblock">
```

```
</div>
  354/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:25 358ms/step - d_loss: 0.3508 - g_loss: 1.7504

<div class="k-default-codeblock">
```

```
</div>
  355/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:24 358ms/step - d_loss: 0.3503 - g_loss: 1.7563

<div class="k-default-codeblock">
```

```
</div>
  356/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:24 358ms/step - d_loss: 0.3497 - g_loss: 1.7622

<div class="k-default-codeblock">
```

```
</div>
  357/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:23 358ms/step - d_loss: 0.3491 - g_loss: 1.7682

<div class="k-default-codeblock">
```

```
</div>
  358/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:23 358ms/step - d_loss: 0.3486 - g_loss: 1.7742

<div class="k-default-codeblock">
```

```
</div>
  359/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:23 358ms/step - d_loss: 0.3480 - g_loss: 1.7801

<div class="k-default-codeblock">
```

```
</div>
  360/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:22 358ms/step - d_loss: 0.3475 - g_loss: 1.7861

<div class="k-default-codeblock">
```

```
</div>
  361/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:22 358ms/step - d_loss: 0.3469 - g_loss: 1.7921

<div class="k-default-codeblock">
```

```
</div>
  362/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:21 358ms/step - d_loss: 0.3464 - g_loss: 1.7982

<div class="k-default-codeblock">
```

```
</div>
  363/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:21 358ms/step - d_loss: 0.3458 - g_loss: 1.8042

<div class="k-default-codeblock">
```

```
</div>
  364/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:21 358ms/step - d_loss: 0.3453 - g_loss: 1.8102

<div class="k-default-codeblock">
```

```
</div>
  365/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:20 357ms/step - d_loss: 0.3447 - g_loss: 1.8163

<div class="k-default-codeblock">
```

```
</div>
  366/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:20 357ms/step - d_loss: 0.3442 - g_loss: 1.8224

<div class="k-default-codeblock">
```

```
</div>
  367/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:19 357ms/step - d_loss: 0.3437 - g_loss: 1.8285

<div class="k-default-codeblock">
```

```
</div>
  368/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:19 358ms/step - d_loss: 0.3431 - g_loss: 1.8346

<div class="k-default-codeblock">
```

```
</div>
  369/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:19 358ms/step - d_loss: 0.3426 - g_loss: 1.8407

<div class="k-default-codeblock">
```

```
</div>
  370/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:19 358ms/step - d_loss: 0.3420 - g_loss: 1.8468

<div class="k-default-codeblock">
```

```
</div>
  371/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:18 358ms/step - d_loss: 0.3415 - g_loss: 1.8529

<div class="k-default-codeblock">
```

```
</div>
  372/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:18 358ms/step - d_loss: 0.3410 - g_loss: 1.8591

<div class="k-default-codeblock">
```

```
</div>
  373/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:17 358ms/step - d_loss: 0.3404 - g_loss: 1.8652

<div class="k-default-codeblock">
```

```
</div>
  374/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:17 358ms/step - d_loss: 0.3399 - g_loss: 1.8714

<div class="k-default-codeblock">
```

```
</div>
  375/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:17 358ms/step - d_loss: 0.3394 - g_loss: 1.8776

<div class="k-default-codeblock">
```

```
</div>
  376/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:16 357ms/step - d_loss: 0.3389 - g_loss: 1.8838

<div class="k-default-codeblock">
```

```
</div>
  377/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:16 357ms/step - d_loss: 0.3383 - g_loss: 1.8900

<div class="k-default-codeblock">
```

```
</div>
  378/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:15 357ms/step - d_loss: 0.3378 - g_loss: 1.8963

<div class="k-default-codeblock">
```

```
</div>
  379/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:15 357ms/step - d_loss: 0.3373 - g_loss: 1.9025

<div class="k-default-codeblock">
```

```
</div>
  380/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:15 357ms/step - d_loss: 0.3368 - g_loss: 1.9088

<div class="k-default-codeblock">
```

```
</div>
  381/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:14 357ms/step - d_loss: 0.3362 - g_loss: 1.9150

<div class="k-default-codeblock">
```

```
</div>
  382/1094 ━━━━━━[37m━━━━━━━━━━━━━━  4:14 357ms/step - d_loss: 0.3357 - g_loss: 1.9213

<div class="k-default-codeblock">
```

```
</div>
  383/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 357ms/step - d_loss: 0.3352 - g_loss: 1.9276

<div class="k-default-codeblock">
```

```
</div>
  384/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 357ms/step - d_loss: 0.3347 - g_loss: 1.9339

<div class="k-default-codeblock">
```

```
</div>
  385/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:13 357ms/step - d_loss: 0.3342 - g_loss: 1.9403

<div class="k-default-codeblock">
```

```
</div>
  386/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:12 357ms/step - d_loss: 0.3336 - g_loss: 1.9466

<div class="k-default-codeblock">
```

```
</div>
  387/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:12 357ms/step - d_loss: 0.3331 - g_loss: 1.9530

<div class="k-default-codeblock">
```

```
</div>
  388/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:11 357ms/step - d_loss: 0.3326 - g_loss: 1.9593

<div class="k-default-codeblock">
```

```
</div>
  389/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:11 357ms/step - d_loss: 0.3321 - g_loss: 1.9657

<div class="k-default-codeblock">
```

```
</div>
  390/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:11 357ms/step - d_loss: 0.3316 - g_loss: 1.9721

<div class="k-default-codeblock">
```

```
</div>
  391/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:10 356ms/step - d_loss: 0.3311 - g_loss: 1.9785

<div class="k-default-codeblock">
```

```
</div>
  392/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:10 356ms/step - d_loss: 0.3306 - g_loss: 1.9850

<div class="k-default-codeblock">
```

```
</div>
  393/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:09 356ms/step - d_loss: 0.3301 - g_loss: 1.9914

<div class="k-default-codeblock">
```

```
</div>
  394/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:09 356ms/step - d_loss: 0.3296 - g_loss: 1.9979

<div class="k-default-codeblock">
```

```
</div>
  395/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:09 356ms/step - d_loss: 0.3291 - g_loss: 2.0044

<div class="k-default-codeblock">
```

```
</div>
  396/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:08 356ms/step - d_loss: 0.3286 - g_loss: 2.0109

<div class="k-default-codeblock">
```

```
</div>
  397/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:08 356ms/step - d_loss: 0.3281 - g_loss: 2.0174

<div class="k-default-codeblock">
```

```
</div>
  398/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:07 356ms/step - d_loss: 0.3276 - g_loss: 2.0239

<div class="k-default-codeblock">
```

```
</div>
  399/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:07 356ms/step - d_loss: 0.3271 - g_loss: 2.0304

<div class="k-default-codeblock">
```

```
</div>
  400/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:07 356ms/step - d_loss: 0.3266 - g_loss: 2.0370

<div class="k-default-codeblock">
```

```
</div>
  401/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:07 357ms/step - d_loss: 0.3261 - g_loss: 2.0436

<div class="k-default-codeblock">
```

```
</div>
  402/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:06 357ms/step - d_loss: 0.3256 - g_loss: 2.0502

<div class="k-default-codeblock">
```

```
</div>
  403/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:06 357ms/step - d_loss: 0.3251 - g_loss: 2.0568

<div class="k-default-codeblock">
```

```
</div>
  404/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:06 357ms/step - d_loss: 0.3246 - g_loss: 2.0634

<div class="k-default-codeblock">
```

```
</div>
  405/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:06 357ms/step - d_loss: 0.3241 - g_loss: 2.0701

<div class="k-default-codeblock">
```

```
</div>
  406/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:05 358ms/step - d_loss: 0.3236 - g_loss: 2.0767

<div class="k-default-codeblock">
```

```
</div>
  407/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:05 358ms/step - d_loss: 0.3231 - g_loss: 2.0834

<div class="k-default-codeblock">
```

```
</div>
  408/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:05 359ms/step - d_loss: 0.3226 - g_loss: 2.0901

<div class="k-default-codeblock">
```

```
</div>
  409/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:05 359ms/step - d_loss: 0.3221 - g_loss: 2.0968

<div class="k-default-codeblock">
```

```
</div>
  410/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:05 359ms/step - d_loss: 0.3217 - g_loss: 2.1035

<div class="k-default-codeblock">
```

```
</div>
  411/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:05 359ms/step - d_loss: 0.3212 - g_loss: 2.1103

<div class="k-default-codeblock">
```

```
</div>
  412/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 359ms/step - d_loss: 0.3207 - g_loss: 2.1171

<div class="k-default-codeblock">
```

```
</div>
  413/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 359ms/step - d_loss: 0.3202 - g_loss: 2.1238

<div class="k-default-codeblock">
```

```
</div>
  414/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 360ms/step - d_loss: 0.3197 - g_loss: 2.1306

<div class="k-default-codeblock">
```

```
</div>
  415/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 360ms/step - d_loss: 0.3192 - g_loss: 2.1375

<div class="k-default-codeblock">
```

```
</div>
  416/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 360ms/step - d_loss: 0.3188 - g_loss: 2.1443

<div class="k-default-codeblock">
```

```
</div>
  417/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 361ms/step - d_loss: 0.3183 - g_loss: 2.1512

<div class="k-default-codeblock">
```

```
</div>
  418/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 362ms/step - d_loss: 0.3178 - g_loss: 2.1580

<div class="k-default-codeblock">
```

```
</div>
  419/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:04 362ms/step - d_loss: 0.3173 - g_loss: 2.1649

<div class="k-default-codeblock">
```

```
</div>
  420/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:03 362ms/step - d_loss: 0.3168 - g_loss: 2.1718

<div class="k-default-codeblock">
```

```
</div>
  421/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:03 362ms/step - d_loss: 0.3164 - g_loss: 2.1788

<div class="k-default-codeblock">
```

```
</div>
  422/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:03 362ms/step - d_loss: 0.3159 - g_loss: 2.1857

<div class="k-default-codeblock">
```

```
</div>
  423/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:02 362ms/step - d_loss: 0.3154 - g_loss: 2.1927

<div class="k-default-codeblock">
```

```
</div>
  424/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:02 362ms/step - d_loss: 0.3150 - g_loss: 2.1997

<div class="k-default-codeblock">
```

```
</div>
  425/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:02 362ms/step - d_loss: 0.3145 - g_loss: 2.2067

<div class="k-default-codeblock">
```

```
</div>
  426/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:01 362ms/step - d_loss: 0.3140 - g_loss: 2.2137

<div class="k-default-codeblock">
```

```
</div>
  427/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:01 362ms/step - d_loss: 0.3135 - g_loss: 2.2208

<div class="k-default-codeblock">
```

```
</div>
  428/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:00 362ms/step - d_loss: 0.3131 - g_loss: 2.2278

<div class="k-default-codeblock">
```

```
</div>
  429/1094 ━━━━━━━[37m━━━━━━━━━━━━━  4:00 362ms/step - d_loss: 0.3126 - g_loss: 2.2349

<div class="k-default-codeblock">
```

```
</div>
  430/1094 ━━━━━━━[37m━━━━━━━━━━━━━  3:59 361ms/step - d_loss: 0.3121 - g_loss: 2.2420

<div class="k-default-codeblock">
```

```
</div>
  431/1094 ━━━━━━━[37m━━━━━━━━━━━━━  3:59 361ms/step - d_loss: 0.3117 - g_loss: 2.2492

<div class="k-default-codeblock">
```

```
</div>
  432/1094 ━━━━━━━[37m━━━━━━━━━━━━━  3:59 361ms/step - d_loss: 0.3112 - g_loss: 2.2563

<div class="k-default-codeblock">
```

```
</div>
  433/1094 ━━━━━━━[37m━━━━━━━━━━━━━  3:58 361ms/step - d_loss: 0.3108 - g_loss: 2.2635

<div class="k-default-codeblock">
```

```
</div>
  434/1094 ━━━━━━━[37m━━━━━━━━━━━━━  3:58 361ms/step - d_loss: 0.3103 - g_loss: 2.2707

<div class="k-default-codeblock">
```

```
</div>
  435/1094 ━━━━━━━[37m━━━━━━━━━━━━━  3:58 361ms/step - d_loss: 0.3098 - g_loss: 2.2779

<div class="k-default-codeblock">
```

```
</div>
  436/1094 ━━━━━━━[37m━━━━━━━━━━━━━  3:57 361ms/step - d_loss: 0.3094 - g_loss: 2.2851

<div class="k-default-codeblock">
```

```
</div>
  437/1094 ━━━━━━━[37m━━━━━━━━━━━━━  3:57 361ms/step - d_loss: 0.3089 - g_loss: 2.2924

<div class="k-default-codeblock">
```

```
</div>
  438/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:56 361ms/step - d_loss: 0.3085 - g_loss: 2.2996

<div class="k-default-codeblock">
```

```
</div>
  439/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:56 361ms/step - d_loss: 0.3080 - g_loss: 2.3069

<div class="k-default-codeblock">
```

```
</div>
  440/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:55 361ms/step - d_loss: 0.3075 - g_loss: 2.3143

<div class="k-default-codeblock">
```

```
</div>
  441/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:55 361ms/step - d_loss: 0.3071 - g_loss: 2.3216

<div class="k-default-codeblock">
```

```
</div>
  442/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:55 361ms/step - d_loss: 0.3066 - g_loss: 2.3289

<div class="k-default-codeblock">
```

```
</div>
  443/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:54 361ms/step - d_loss: 0.3062 - g_loss: 2.3363

<div class="k-default-codeblock">
```

```
</div>
  444/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:54 361ms/step - d_loss: 0.3057 - g_loss: 2.3437

<div class="k-default-codeblock">
```

```
</div>
  445/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:53 361ms/step - d_loss: 0.3053 - g_loss: 2.3511

<div class="k-default-codeblock">
```

```
</div>
  446/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:53 360ms/step - d_loss: 0.3048 - g_loss: 2.3586

<div class="k-default-codeblock">
```

```
</div>
  447/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:53 360ms/step - d_loss: 0.3044 - g_loss: 2.3661

<div class="k-default-codeblock">
```

```
</div>
  448/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:52 360ms/step - d_loss: 0.3039 - g_loss: 2.3736

<div class="k-default-codeblock">
```

```
</div>
  449/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:52 360ms/step - d_loss: 0.3035 - g_loss: 2.3811

<div class="k-default-codeblock">
```

```
</div>
  450/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:52 360ms/step - d_loss: 0.3030 - g_loss: 2.3886

<div class="k-default-codeblock">
```

```
</div>
  451/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:51 360ms/step - d_loss: 0.3026 - g_loss: 2.3962

<div class="k-default-codeblock">
```

```
</div>
  452/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:51 360ms/step - d_loss: 0.3021 - g_loss: 2.4037

<div class="k-default-codeblock">
```

```
</div>
  453/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:50 360ms/step - d_loss: 0.3017 - g_loss: 2.4113

<div class="k-default-codeblock">
```

```
</div>
  454/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:50 360ms/step - d_loss: 0.3013 - g_loss: 2.4190

<div class="k-default-codeblock">
```

```
</div>
  455/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:50 360ms/step - d_loss: 0.3008 - g_loss: 2.4266

<div class="k-default-codeblock">
```

```
</div>
  456/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:49 360ms/step - d_loss: 0.3004 - g_loss: 2.4343

<div class="k-default-codeblock">
```

```
</div>
  457/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:49 360ms/step - d_loss: 0.2999 - g_loss: 2.4420

<div class="k-default-codeblock">
```

```
</div>
  458/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:48 360ms/step - d_loss: 0.2995 - g_loss: 2.4497

<div class="k-default-codeblock">
```

```
</div>
  459/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:48 360ms/step - d_loss: 0.2990 - g_loss: 2.4574

<div class="k-default-codeblock">
```

```
</div>
  460/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:48 360ms/step - d_loss: 0.2986 - g_loss: 2.4652

<div class="k-default-codeblock">
```

```
</div>
  461/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:47 360ms/step - d_loss: 0.2982 - g_loss: 2.4730

<div class="k-default-codeblock">
```

```
</div>
  462/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:47 360ms/step - d_loss: 0.2977 - g_loss: 2.4808

<div class="k-default-codeblock">
```

```
</div>
  463/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:46 360ms/step - d_loss: 0.2973 - g_loss: 2.4887

<div class="k-default-codeblock">
```

```
</div>
  464/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:46 360ms/step - d_loss: 0.2969 - g_loss: 2.4965

<div class="k-default-codeblock">
```

```
</div>
  465/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:46 360ms/step - d_loss: 0.2964 - g_loss: 2.5044

<div class="k-default-codeblock">
```

```
</div>
  466/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:45 360ms/step - d_loss: 0.2960 - g_loss: 2.5123

<div class="k-default-codeblock">
```

```
</div>
  467/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:45 359ms/step - d_loss: 0.2956 - g_loss: 2.5203

<div class="k-default-codeblock">
```

```
</div>
  468/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:45 359ms/step - d_loss: 0.2951 - g_loss: 2.5282

<div class="k-default-codeblock">
```

```
</div>
  469/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:44 359ms/step - d_loss: 0.2947 - g_loss: 2.5362

<div class="k-default-codeblock">
```

```
</div>
  470/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:44 359ms/step - d_loss: 0.2943 - g_loss: 2.5443

<div class="k-default-codeblock">
```

```
</div>
  471/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:43 359ms/step - d_loss: 0.2938 - g_loss: 2.5523

<div class="k-default-codeblock">
```

```
</div>
  472/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:43 359ms/step - d_loss: 0.2934 - g_loss: 2.5604

<div class="k-default-codeblock">
```

```
</div>
  473/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:43 359ms/step - d_loss: 0.2930 - g_loss: 2.5685

<div class="k-default-codeblock">
```

```
</div>
  474/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:42 359ms/step - d_loss: 0.2925 - g_loss: 2.5766

<div class="k-default-codeblock">
```

```
</div>
  475/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:42 359ms/step - d_loss: 0.2921 - g_loss: 2.5848

<div class="k-default-codeblock">
```

```
</div>
  476/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:41 359ms/step - d_loss: 0.2917 - g_loss: 2.5930

<div class="k-default-codeblock">
```

```
</div>
  477/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:41 359ms/step - d_loss: 0.2912 - g_loss: 2.6012

<div class="k-default-codeblock">
```

```
</div>
  478/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:41 359ms/step - d_loss: 0.2908 - g_loss: 2.6095

<div class="k-default-codeblock">
```

```
</div>
  479/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:40 359ms/step - d_loss: 0.2904 - g_loss: 2.6177

<div class="k-default-codeblock">
```

```
</div>
  480/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:40 359ms/step - d_loss: 0.2900 - g_loss: 2.6260

<div class="k-default-codeblock">
```

```
</div>
  481/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:40 359ms/step - d_loss: 0.2895 - g_loss: 2.6344

<div class="k-default-codeblock">
```

```
</div>
  482/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:40 360ms/step - d_loss: 0.2891 - g_loss: 2.6427

<div class="k-default-codeblock">
```

```
</div>
  483/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:40 360ms/step - d_loss: 0.2887 - g_loss: 2.6511

<div class="k-default-codeblock">
```

```
</div>
  484/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:40 361ms/step - d_loss: 0.2883 - g_loss: 2.6595

<div class="k-default-codeblock">
```

```
</div>
  485/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:39 361ms/step - d_loss: 0.2878 - g_loss: 2.6680

<div class="k-default-codeblock">
```

```
</div>
  486/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:39 361ms/step - d_loss: 0.2874 - g_loss: 2.6765

<div class="k-default-codeblock">
```

```
</div>
  487/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:39 361ms/step - d_loss: 0.2870 - g_loss: 2.6850

<div class="k-default-codeblock">
```

```
</div>
  488/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:38 361ms/step - d_loss: 0.2866 - g_loss: 2.6935

<div class="k-default-codeblock">
```

```
</div>
  489/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:38 361ms/step - d_loss: 0.2862 - g_loss: 2.7021

<div class="k-default-codeblock">
```

```
</div>
  490/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:37 361ms/step - d_loss: 0.2857 - g_loss: 2.7107

<div class="k-default-codeblock">
```

```
</div>
  491/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:37 361ms/step - d_loss: 0.2853 - g_loss: 2.7193

<div class="k-default-codeblock">
```

```
</div>
  492/1094 ━━━━━━━━[37m━━━━━━━━━━━━  3:37 361ms/step - d_loss: 0.2849 - g_loss: 2.7280

<div class="k-default-codeblock">
```

```
</div>
  493/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:36 361ms/step - d_loss: 0.2845 - g_loss: 2.7366

<div class="k-default-codeblock">
```

```
</div>
  494/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:36 361ms/step - d_loss: 0.2841 - g_loss: 2.7454

<div class="k-default-codeblock">
```

```
</div>
  495/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:36 361ms/step - d_loss: 0.2837 - g_loss: 2.7541

<div class="k-default-codeblock">
```

```
</div>
  496/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:35 361ms/step - d_loss: 0.2832 - g_loss: 2.7629

<div class="k-default-codeblock">
```

```
</div>
  497/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:35 361ms/step - d_loss: 0.2828 - g_loss: 2.7717

<div class="k-default-codeblock">
```

```
</div>
  498/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:34 361ms/step - d_loss: 0.2824 - g_loss: 2.7805

<div class="k-default-codeblock">
```

```
</div>
  499/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:34 361ms/step - d_loss: 0.2820 - g_loss: 2.7894

<div class="k-default-codeblock">
```

```
</div>
  500/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:34 360ms/step - d_loss: 0.2816 - g_loss: 2.7983

<div class="k-default-codeblock">
```

```
</div>
  501/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:33 360ms/step - d_loss: 0.2812 - g_loss: 2.8072

<div class="k-default-codeblock">
```

```
</div>
  502/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:33 360ms/step - d_loss: 0.2808 - g_loss: 2.8162

<div class="k-default-codeblock">
```

```
</div>
  503/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:32 360ms/step - d_loss: 0.2803 - g_loss: 2.8252

<div class="k-default-codeblock">
```

```
</div>
  504/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:32 360ms/step - d_loss: 0.2799 - g_loss: 2.8342

<div class="k-default-codeblock">
```

```
</div>
  505/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:32 360ms/step - d_loss: 0.2795 - g_loss: 2.8433

<div class="k-default-codeblock">
```

```
</div>
  506/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:31 360ms/step - d_loss: 0.2791 - g_loss: 2.8524

<div class="k-default-codeblock">
```

```
</div>
  507/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:31 360ms/step - d_loss: 0.2787 - g_loss: 2.8615

<div class="k-default-codeblock">
```

```
</div>
  508/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:31 360ms/step - d_loss: 0.2783 - g_loss: 2.8707

<div class="k-default-codeblock">
```

```
</div>
  509/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:30 360ms/step - d_loss: 0.2779 - g_loss: 2.8799

<div class="k-default-codeblock">
```

```
</div>
  510/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:30 360ms/step - d_loss: 0.2775 - g_loss: 2.8891

<div class="k-default-codeblock">
```

```
</div>
  511/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:29 360ms/step - d_loss: 0.2771 - g_loss: 2.8984

<div class="k-default-codeblock">
```

```
</div>
  512/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:29 360ms/step - d_loss: 0.2766 - g_loss: 2.9077

<div class="k-default-codeblock">
```

```
</div>
  513/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:29 360ms/step - d_loss: 0.2762 - g_loss: 2.9170

<div class="k-default-codeblock">
```

```
</div>
  514/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:28 360ms/step - d_loss: 0.2758 - g_loss: 2.9264

<div class="k-default-codeblock">
```

```
</div>
  515/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:28 360ms/step - d_loss: 0.2754 - g_loss: 2.9358

<div class="k-default-codeblock">
```

```
</div>
  516/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:27 360ms/step - d_loss: 0.2750 - g_loss: 2.9452

<div class="k-default-codeblock">
```

```
</div>
  517/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:27 360ms/step - d_loss: 0.2746 - g_loss: 2.9547

<div class="k-default-codeblock">
```

```
</div>
  518/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:27 360ms/step - d_loss: 0.2742 - g_loss: 2.9642

<div class="k-default-codeblock">
```

```
</div>
  519/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:26 360ms/step - d_loss: 0.2738 - g_loss: 2.9737

<div class="k-default-codeblock">
```

```
</div>
  520/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:26 360ms/step - d_loss: 0.2734 - g_loss: 2.9833

<div class="k-default-codeblock">
```

```
</div>
  521/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:26 360ms/step - d_loss: 0.2730 - g_loss: 2.9929

<div class="k-default-codeblock">
```

```
</div>
  522/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:25 360ms/step - d_loss: 0.2726 - g_loss: 3.0026

<div class="k-default-codeblock">
```

```
</div>
  523/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:25 360ms/step - d_loss: 0.2722 - g_loss: 3.0123

<div class="k-default-codeblock">
```

```
</div>
  524/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:24 359ms/step - d_loss: 0.2718 - g_loss: 3.0220

<div class="k-default-codeblock">
```

```
</div>
  525/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:24 359ms/step - d_loss: 0.2714 - g_loss: 3.0317

<div class="k-default-codeblock">
```

```
</div>
  526/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:24 360ms/step - d_loss: 0.2710 - g_loss: 3.0415

<div class="k-default-codeblock">
```

```
</div>
  527/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:23 359ms/step - d_loss: 0.2706 - g_loss: 3.0512

<div class="k-default-codeblock">
```

```
</div>
  528/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:23 359ms/step - d_loss: 0.2702 - g_loss: 3.0610

<div class="k-default-codeblock">
```

```
</div>
  529/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:23 359ms/step - d_loss: 0.2698 - g_loss: 3.0708

<div class="k-default-codeblock">
```

```
</div>
  530/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:22 359ms/step - d_loss: 0.2694 - g_loss: 3.0805

<div class="k-default-codeblock">
```

```
</div>
  531/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:22 359ms/step - d_loss: 0.2690 - g_loss: 3.0901

<div class="k-default-codeblock">
```

```
</div>
  532/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:21 359ms/step - d_loss: 0.2686 - g_loss: 3.0997

<div class="k-default-codeblock">
```

```
</div>
  533/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:21 359ms/step - d_loss: 0.2682 - g_loss: 3.1092

<div class="k-default-codeblock">
```

```
</div>
  534/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:21 359ms/step - d_loss: 0.2678 - g_loss: 3.1187

<div class="k-default-codeblock">
```

```
</div>
  535/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:20 359ms/step - d_loss: 0.2674 - g_loss: 3.1281

<div class="k-default-codeblock">
```

```
</div>
  536/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:20 359ms/step - d_loss: 0.2671 - g_loss: 3.1375

<div class="k-default-codeblock">
```

```
</div>
  537/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:19 359ms/step - d_loss: 0.2667 - g_loss: 3.1468

<div class="k-default-codeblock">
```

```
</div>
  538/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:19 359ms/step - d_loss: 0.2663 - g_loss: 3.1560

<div class="k-default-codeblock">
```

```
</div>
  539/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:19 359ms/step - d_loss: 0.2659 - g_loss: 3.1652

<div class="k-default-codeblock">
```

```
</div>
  540/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:18 359ms/step - d_loss: 0.2656 - g_loss: 3.1744

<div class="k-default-codeblock">
```

```
</div>
  541/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:18 359ms/step - d_loss: 0.2652 - g_loss: 3.1835

<div class="k-default-codeblock">
```

```
</div>
  542/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:18 359ms/step - d_loss: 0.2648 - g_loss: 3.1926

<div class="k-default-codeblock">
```

```
</div>
  543/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:17 359ms/step - d_loss: 0.2644 - g_loss: 3.2016

<div class="k-default-codeblock">
```

```
</div>
  544/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:17 359ms/step - d_loss: 0.2641 - g_loss: 3.2106

<div class="k-default-codeblock">
```

```
</div>
  545/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:16 359ms/step - d_loss: 0.2637 - g_loss: 3.2195

<div class="k-default-codeblock">
```

```
</div>
  546/1094 ━━━━━━━━━[37m━━━━━━━━━━━  3:16 359ms/step - d_loss: 0.2633 - g_loss: 3.2284

<div class="k-default-codeblock">
```

```
</div>
  547/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:16 359ms/step - d_loss: 0.2630 - g_loss: 3.2372

<div class="k-default-codeblock">
```

```
</div>
  548/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:15 358ms/step - d_loss: 0.2626 - g_loss: 3.2460

<div class="k-default-codeblock">
```

```
</div>
  549/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:15 358ms/step - d_loss: 0.2622 - g_loss: 3.2548

<div class="k-default-codeblock">
```

```
</div>
  550/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:15 359ms/step - d_loss: 0.2619 - g_loss: 3.2635

<div class="k-default-codeblock">
```

```
</div>
  551/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:14 359ms/step - d_loss: 0.2615 - g_loss: 3.2722

<div class="k-default-codeblock">
```

```
</div>
  552/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:14 359ms/step - d_loss: 0.2612 - g_loss: 3.2808

<div class="k-default-codeblock">
```

```
</div>
  553/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:14 359ms/step - d_loss: 0.2608 - g_loss: 3.2894

<div class="k-default-codeblock">
```

```
</div>
  554/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:14 359ms/step - d_loss: 0.2605 - g_loss: 3.2979

<div class="k-default-codeblock">
```

```
</div>
  555/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:13 359ms/step - d_loss: 0.2601 - g_loss: 3.3064

<div class="k-default-codeblock">
```

```
</div>
  556/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:13 360ms/step - d_loss: 0.2598 - g_loss: 3.3149

<div class="k-default-codeblock">
```

```
</div>
  557/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:13 360ms/step - d_loss: 0.2594 - g_loss: 3.3233

<div class="k-default-codeblock">
```

```
</div>
  558/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:12 360ms/step - d_loss: 0.2591 - g_loss: 3.3316

<div class="k-default-codeblock">
```

```
</div>
  559/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:12 360ms/step - d_loss: 0.2587 - g_loss: 3.3399

<div class="k-default-codeblock">
```

```
</div>
  560/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:12 360ms/step - d_loss: 0.2584 - g_loss: 3.3482

<div class="k-default-codeblock">
```

```
</div>
  561/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:11 360ms/step - d_loss: 0.2580 - g_loss: 3.3564

<div class="k-default-codeblock">
```

```
</div>
  562/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:11 360ms/step - d_loss: 0.2577 - g_loss: 3.3646

<div class="k-default-codeblock">
```

```
</div>
  563/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:11 360ms/step - d_loss: 0.2574 - g_loss: 3.3727

<div class="k-default-codeblock">
```

```
</div>
  564/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:10 360ms/step - d_loss: 0.2570 - g_loss: 3.3808

<div class="k-default-codeblock">
```

```
</div>
  565/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:10 360ms/step - d_loss: 0.2567 - g_loss: 3.3888

<div class="k-default-codeblock">
```

```
</div>
  566/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:10 361ms/step - d_loss: 0.2564 - g_loss: 3.3968

<div class="k-default-codeblock">
```

```
</div>
  567/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:10 361ms/step - d_loss: 0.2561 - g_loss: 3.4048

<div class="k-default-codeblock">
```

```
</div>
  568/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:09 361ms/step - d_loss: 0.2558 - g_loss: 3.4127

<div class="k-default-codeblock">
```

```
</div>
  569/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:09 361ms/step - d_loss: 0.2555 - g_loss: 3.4205

<div class="k-default-codeblock">
```

```
</div>
  570/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:09 361ms/step - d_loss: 0.2551 - g_loss: 3.4283

<div class="k-default-codeblock">
```

```
</div>
  571/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:08 361ms/step - d_loss: 0.2548 - g_loss: 3.4361

<div class="k-default-codeblock">
```

```
</div>
  572/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:08 361ms/step - d_loss: 0.2546 - g_loss: 3.4438

<div class="k-default-codeblock">
```

```
</div>
  573/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:08 361ms/step - d_loss: 0.2543 - g_loss: 3.4514

<div class="k-default-codeblock">
```

```
</div>
  574/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:07 361ms/step - d_loss: 0.2540 - g_loss: 3.4590

<div class="k-default-codeblock">
```

```
</div>
  575/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:07 361ms/step - d_loss: 0.2537 - g_loss: 3.4666

<div class="k-default-codeblock">
```

```
</div>
  576/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:07 361ms/step - d_loss: 0.2534 - g_loss: 3.4741

<div class="k-default-codeblock">
```

```
</div>
  577/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:06 362ms/step - d_loss: 0.2531 - g_loss: 3.4816

<div class="k-default-codeblock">
```

```
</div>
  578/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:06 362ms/step - d_loss: 0.2528 - g_loss: 3.4890

<div class="k-default-codeblock">
```

```
</div>
  579/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:06 362ms/step - d_loss: 0.2526 - g_loss: 3.4964

<div class="k-default-codeblock">
```

```
</div>
  580/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:06 363ms/step - d_loss: 0.2523 - g_loss: 3.5037

<div class="k-default-codeblock">
```

```
</div>
  581/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:06 363ms/step - d_loss: 0.2520 - g_loss: 3.5110

<div class="k-default-codeblock">
```

```
</div>
  582/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:06 363ms/step - d_loss: 0.2518 - g_loss: 3.5183

<div class="k-default-codeblock">
```

```
</div>
  583/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:05 364ms/step - d_loss: 0.2515 - g_loss: 3.5255

<div class="k-default-codeblock">
```

```
</div>
  584/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:05 364ms/step - d_loss: 0.2512 - g_loss: 3.5327

<div class="k-default-codeblock">
```

```
</div>
  585/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:05 364ms/step - d_loss: 0.2510 - g_loss: 3.5398

<div class="k-default-codeblock">
```

```
</div>
  586/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:05 364ms/step - d_loss: 0.2507 - g_loss: 3.5469

<div class="k-default-codeblock">
```

```
</div>
  587/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:04 364ms/step - d_loss: 0.2505 - g_loss: 3.5539

<div class="k-default-codeblock">
```

```
</div>
  588/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:04 364ms/step - d_loss: 0.2502 - g_loss: 3.5610

<div class="k-default-codeblock">
```

```
</div>
  589/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:03 364ms/step - d_loss: 0.2499 - g_loss: 3.5679

<div class="k-default-codeblock">
```

```
</div>
  590/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:03 364ms/step - d_loss: 0.2497 - g_loss: 3.5749

<div class="k-default-codeblock">
```

```
</div>
  591/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:03 364ms/step - d_loss: 0.2494 - g_loss: 3.5817

<div class="k-default-codeblock">
```

```
</div>
  592/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:02 364ms/step - d_loss: 0.2492 - g_loss: 3.5886

<div class="k-default-codeblock">
```

```
</div>
  593/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:02 364ms/step - d_loss: 0.2490 - g_loss: 3.5954

<div class="k-default-codeblock">
```

```
</div>
  594/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:02 364ms/step - d_loss: 0.2487 - g_loss: 3.6022

<div class="k-default-codeblock">
```

```
</div>
  595/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:01 364ms/step - d_loss: 0.2485 - g_loss: 3.6089

<div class="k-default-codeblock">
```

```
</div>
  596/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:01 364ms/step - d_loss: 0.2483 - g_loss: 3.6156

<div class="k-default-codeblock">
```

```
</div>
  597/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:00 364ms/step - d_loss: 0.2480 - g_loss: 3.6222

<div class="k-default-codeblock">
```

```
</div>
  598/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:00 364ms/step - d_loss: 0.2478 - g_loss: 3.6288

<div class="k-default-codeblock">
```

```
</div>
  599/1094 ━━━━━━━━━━[37m━━━━━━━━━━  3:00 364ms/step - d_loss: 0.2476 - g_loss: 3.6354

<div class="k-default-codeblock">
```

```
</div>
  600/1094 ━━━━━━━━━━[37m━━━━━━━━━━  2:59 364ms/step - d_loss: 0.2474 - g_loss: 3.6419

<div class="k-default-codeblock">
```

```
</div>
  601/1094 ━━━━━━━━━━[37m━━━━━━━━━━  2:59 364ms/step - d_loss: 0.2471 - g_loss: 3.6484

<div class="k-default-codeblock">
```

```
</div>
  602/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:58 364ms/step - d_loss: 0.2469 - g_loss: 3.6549

<div class="k-default-codeblock">
```

```
</div>
  603/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:58 364ms/step - d_loss: 0.2467 - g_loss: 3.6613

<div class="k-default-codeblock">
```

```
</div>
  604/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:58 364ms/step - d_loss: 0.2465 - g_loss: 3.6677

<div class="k-default-codeblock">
```

```
</div>
  605/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:57 364ms/step - d_loss: 0.2463 - g_loss: 3.6740

<div class="k-default-codeblock">
```

```
</div>
  606/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:57 364ms/step - d_loss: 0.2461 - g_loss: 3.6803

<div class="k-default-codeblock">
```

```
</div>
  607/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:57 364ms/step - d_loss: 0.2459 - g_loss: 3.6866

<div class="k-default-codeblock">
```

```
</div>
  608/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:56 363ms/step - d_loss: 0.2457 - g_loss: 3.6928

<div class="k-default-codeblock">
```

```
</div>
  609/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:56 363ms/step - d_loss: 0.2455 - g_loss: 3.6990

<div class="k-default-codeblock">
```

```
</div>
  610/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:55 363ms/step - d_loss: 0.2453 - g_loss: 3.7052

<div class="k-default-codeblock">
```

```
</div>
  611/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:55 363ms/step - d_loss: 0.2451 - g_loss: 3.7113

<div class="k-default-codeblock">
```

```
</div>
  612/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:55 363ms/step - d_loss: 0.2450 - g_loss: 3.7174

<div class="k-default-codeblock">
```

```
</div>
  613/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:54 363ms/step - d_loss: 0.2448 - g_loss: 3.7235

<div class="k-default-codeblock">
```

```
</div>
  614/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:54 363ms/step - d_loss: 0.2446 - g_loss: 3.7295

<div class="k-default-codeblock">
```

```
</div>
  615/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:53 363ms/step - d_loss: 0.2445 - g_loss: 3.7355

<div class="k-default-codeblock">
```

```
</div>
  616/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:53 363ms/step - d_loss: 0.2443 - g_loss: 3.7414

<div class="k-default-codeblock">
```

```
</div>
  617/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:53 363ms/step - d_loss: 0.2441 - g_loss: 3.7474

<div class="k-default-codeblock">
```

```
</div>
  618/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:52 363ms/step - d_loss: 0.2440 - g_loss: 3.7533

<div class="k-default-codeblock">
```

```
</div>
  619/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:52 363ms/step - d_loss: 0.2438 - g_loss: 3.7591

<div class="k-default-codeblock">
```

```
</div>
  620/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:52 363ms/step - d_loss: 0.2436 - g_loss: 3.7649

<div class="k-default-codeblock">
```

```
</div>
  621/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:51 363ms/step - d_loss: 0.2435 - g_loss: 3.7707

<div class="k-default-codeblock">
```

```
</div>
  622/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:51 363ms/step - d_loss: 0.2433 - g_loss: 3.7765

<div class="k-default-codeblock">
```

```
</div>
  623/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:50 363ms/step - d_loss: 0.2432 - g_loss: 3.7822

<div class="k-default-codeblock">
```

```
</div>
  624/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:50 363ms/step - d_loss: 0.2430 - g_loss: 3.7879

<div class="k-default-codeblock">
```

```
</div>
  625/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:50 363ms/step - d_loss: 0.2429 - g_loss: 3.7936

<div class="k-default-codeblock">
```

```
</div>
  626/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:49 363ms/step - d_loss: 0.2427 - g_loss: 3.7992

<div class="k-default-codeblock">
```

```
</div>
  627/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:49 363ms/step - d_loss: 0.2426 - g_loss: 3.8048

<div class="k-default-codeblock">
```

```
</div>
  628/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:48 363ms/step - d_loss: 0.2424 - g_loss: 3.8104

<div class="k-default-codeblock">
```

```
</div>
  629/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:48 363ms/step - d_loss: 0.2423 - g_loss: 3.8159

<div class="k-default-codeblock">
```

```
</div>
  630/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:48 363ms/step - d_loss: 0.2421 - g_loss: 3.8215

<div class="k-default-codeblock">
```

```
</div>
  631/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:47 363ms/step - d_loss: 0.2420 - g_loss: 3.8270

<div class="k-default-codeblock">
```

```
</div>
  632/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:47 363ms/step - d_loss: 0.2418 - g_loss: 3.8324

<div class="k-default-codeblock">
```

```
</div>
  633/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:47 363ms/step - d_loss: 0.2417 - g_loss: 3.8379

<div class="k-default-codeblock">
```

```
</div>
  634/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:46 362ms/step - d_loss: 0.2415 - g_loss: 3.8433

<div class="k-default-codeblock">
```

```
</div>
  635/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:46 362ms/step - d_loss: 0.2414 - g_loss: 3.8487

<div class="k-default-codeblock">
```

```
</div>
  636/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:45 362ms/step - d_loss: 0.2412 - g_loss: 3.8540

<div class="k-default-codeblock">
```

```
</div>
  637/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:45 362ms/step - d_loss: 0.2411 - g_loss: 3.8594

<div class="k-default-codeblock">
```

```
</div>
  638/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:45 362ms/step - d_loss: 0.2410 - g_loss: 3.8647

<div class="k-default-codeblock">
```

```
</div>
  639/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:44 362ms/step - d_loss: 0.2408 - g_loss: 3.8700

<div class="k-default-codeblock">
```

```
</div>
  640/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:44 362ms/step - d_loss: 0.2407 - g_loss: 3.8752

<div class="k-default-codeblock">
```

```
</div>
  641/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:44 362ms/step - d_loss: 0.2405 - g_loss: 3.8805

<div class="k-default-codeblock">
```

```
</div>
  642/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:43 362ms/step - d_loss: 0.2404 - g_loss: 3.8857

<div class="k-default-codeblock">
```

```
</div>
  643/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:43 362ms/step - d_loss: 0.2402 - g_loss: 3.8909

<div class="k-default-codeblock">
```

```
</div>
  644/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:42 362ms/step - d_loss: 0.2401 - g_loss: 3.8961

<div class="k-default-codeblock">
```

```
</div>
  645/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:42 362ms/step - d_loss: 0.2400 - g_loss: 3.9012

<div class="k-default-codeblock">
```

```
</div>
  646/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:42 362ms/step - d_loss: 0.2398 - g_loss: 3.9063

<div class="k-default-codeblock">
```

```
</div>
  647/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:41 362ms/step - d_loss: 0.2397 - g_loss: 3.9114

<div class="k-default-codeblock">
```

```
</div>
  648/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:41 362ms/step - d_loss: 0.2395 - g_loss: 3.9165

<div class="k-default-codeblock">
```

```
</div>
  649/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:41 362ms/step - d_loss: 0.2394 - g_loss: 3.9215

<div class="k-default-codeblock">
```

```
</div>
  650/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:40 362ms/step - d_loss: 0.2393 - g_loss: 3.9265

<div class="k-default-codeblock">
```

```
</div>
  651/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:40 362ms/step - d_loss: 0.2391 - g_loss: 3.9315

<div class="k-default-codeblock">
```

```
</div>
  652/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:39 362ms/step - d_loss: 0.2390 - g_loss: 3.9365

<div class="k-default-codeblock">
```

```
</div>
  653/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:39 362ms/step - d_loss: 0.2388 - g_loss: 3.9414

<div class="k-default-codeblock">
```

```
</div>
  654/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:39 362ms/step - d_loss: 0.2387 - g_loss: 3.9463

<div class="k-default-codeblock">
```

```
</div>
  655/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:38 362ms/step - d_loss: 0.2386 - g_loss: 3.9512

<div class="k-default-codeblock">
```

```
</div>
  656/1094 ━━━━━━━━━━━[37m━━━━━━━━━  2:38 362ms/step - d_loss: 0.2384 - g_loss: 3.9561

<div class="k-default-codeblock">
```

```
</div>
  657/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:38 362ms/step - d_loss: 0.2383 - g_loss: 3.9609

<div class="k-default-codeblock">
```

```
</div>
  658/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:37 362ms/step - d_loss: 0.2382 - g_loss: 3.9658

<div class="k-default-codeblock">
```

```
</div>
  659/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:37 362ms/step - d_loss: 0.2380 - g_loss: 3.9706

<div class="k-default-codeblock">
```

```
</div>
  660/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:36 361ms/step - d_loss: 0.2379 - g_loss: 3.9753

<div class="k-default-codeblock">
```

```
</div>
  661/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:36 361ms/step - d_loss: 0.2378 - g_loss: 3.9801

<div class="k-default-codeblock">
```

```
</div>
  662/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:36 361ms/step - d_loss: 0.2377 - g_loss: 3.9848

<div class="k-default-codeblock">
```

```
</div>
  663/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:35 361ms/step - d_loss: 0.2375 - g_loss: 3.9895

<div class="k-default-codeblock">
```

```
</div>
  664/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:35 361ms/step - d_loss: 0.2374 - g_loss: 3.9941

<div class="k-default-codeblock">
```

```
</div>
  665/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:35 361ms/step - d_loss: 0.2373 - g_loss: 3.9988

<div class="k-default-codeblock">
```

```
</div>
  666/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:34 361ms/step - d_loss: 0.2372 - g_loss: 4.0034

<div class="k-default-codeblock">
```

```
</div>
  667/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:34 361ms/step - d_loss: 0.2370 - g_loss: 4.0080

<div class="k-default-codeblock">
```

```
</div>
  668/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:33 361ms/step - d_loss: 0.2369 - g_loss: 4.0125

<div class="k-default-codeblock">
```

```
</div>
  669/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:33 361ms/step - d_loss: 0.2368 - g_loss: 4.0171

<div class="k-default-codeblock">
```

```
</div>
  670/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:33 361ms/step - d_loss: 0.2367 - g_loss: 4.0216

<div class="k-default-codeblock">
```

```
</div>
  671/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:32 361ms/step - d_loss: 0.2366 - g_loss: 4.0261

<div class="k-default-codeblock">
```

```
</div>
  672/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:32 361ms/step - d_loss: 0.2365 - g_loss: 4.0305

<div class="k-default-codeblock">
```

```
</div>
  673/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:31 361ms/step - d_loss: 0.2364 - g_loss: 4.0350

<div class="k-default-codeblock">
```

```
</div>
  674/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:31 361ms/step - d_loss: 0.2363 - g_loss: 4.0394

<div class="k-default-codeblock">
```

```
</div>
  675/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:31 361ms/step - d_loss: 0.2362 - g_loss: 4.0437

<div class="k-default-codeblock">
```

```
</div>
  676/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:30 361ms/step - d_loss: 0.2361 - g_loss: 4.0481

<div class="k-default-codeblock">
```

```
</div>
  677/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:30 361ms/step - d_loss: 0.2359 - g_loss: 4.0524

<div class="k-default-codeblock">
```

```
</div>
  678/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:30 361ms/step - d_loss: 0.2358 - g_loss: 4.0567

<div class="k-default-codeblock">
```

```
</div>
  679/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:29 361ms/step - d_loss: 0.2357 - g_loss: 4.0610

<div class="k-default-codeblock">
```

```
</div>
  680/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:29 361ms/step - d_loss: 0.2357 - g_loss: 4.0653

<div class="k-default-codeblock">
```

```
</div>
  681/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:28 361ms/step - d_loss: 0.2356 - g_loss: 4.0695

<div class="k-default-codeblock">
```

```
</div>
  682/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:28 361ms/step - d_loss: 0.2355 - g_loss: 4.0737

<div class="k-default-codeblock">
```

```
</div>
  683/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:28 361ms/step - d_loss: 0.2354 - g_loss: 4.0779

<div class="k-default-codeblock">
```

```
</div>
  684/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:27 361ms/step - d_loss: 0.2353 - g_loss: 4.0821

<div class="k-default-codeblock">
```

```
</div>
  685/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:27 361ms/step - d_loss: 0.2352 - g_loss: 4.0862

<div class="k-default-codeblock">
```

```
</div>
  686/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:27 361ms/step - d_loss: 0.2351 - g_loss: 4.0904

<div class="k-default-codeblock">
```

```
</div>
  687/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:26 360ms/step - d_loss: 0.2350 - g_loss: 4.0945

<div class="k-default-codeblock">
```

```
</div>
  688/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:26 360ms/step - d_loss: 0.2349 - g_loss: 4.0985

<div class="k-default-codeblock">
```

```
</div>
  689/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:25 360ms/step - d_loss: 0.2348 - g_loss: 4.1026

<div class="k-default-codeblock">
```

```
</div>
  690/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:25 360ms/step - d_loss: 0.2347 - g_loss: 4.1066

<div class="k-default-codeblock">
```

```
</div>
  691/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:25 360ms/step - d_loss: 0.2346 - g_loss: 4.1106

<div class="k-default-codeblock">
```

```
</div>
  692/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:24 360ms/step - d_loss: 0.2346 - g_loss: 4.1146

<div class="k-default-codeblock">
```

```
</div>
  693/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:24 360ms/step - d_loss: 0.2345 - g_loss: 4.1186

<div class="k-default-codeblock">
```

```
</div>
  694/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:24 360ms/step - d_loss: 0.2344 - g_loss: 4.1225

<div class="k-default-codeblock">
```

```
</div>
  695/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:23 360ms/step - d_loss: 0.2343 - g_loss: 4.1265

<div class="k-default-codeblock">
```

```
</div>
  696/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:23 360ms/step - d_loss: 0.2342 - g_loss: 4.1304

<div class="k-default-codeblock">
```

```
</div>
  697/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:22 360ms/step - d_loss: 0.2342 - g_loss: 4.1343

<div class="k-default-codeblock">
```

```
</div>
  698/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:22 360ms/step - d_loss: 0.2341 - g_loss: 4.1381

<div class="k-default-codeblock">
```

```
</div>
  699/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:22 360ms/step - d_loss: 0.2340 - g_loss: 4.1420

<div class="k-default-codeblock">
```

```
</div>
  700/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:21 360ms/step - d_loss: 0.2339 - g_loss: 4.1458

<div class="k-default-codeblock">
```

```
</div>
  701/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:21 360ms/step - d_loss: 0.2338 - g_loss: 4.1496

<div class="k-default-codeblock">
```

```
</div>
  702/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:21 360ms/step - d_loss: 0.2338 - g_loss: 4.1534

<div class="k-default-codeblock">
```

```
</div>
  703/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:20 360ms/step - d_loss: 0.2337 - g_loss: 4.1571

<div class="k-default-codeblock">
```

```
</div>
  704/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:20 360ms/step - d_loss: 0.2336 - g_loss: 4.1609

<div class="k-default-codeblock">
```

```
</div>
  705/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:19 360ms/step - d_loss: 0.2335 - g_loss: 4.1646

<div class="k-default-codeblock">
```

```
</div>
  706/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:19 360ms/step - d_loss: 0.2335 - g_loss: 4.1683

<div class="k-default-codeblock">
```

```
</div>
  707/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:19 360ms/step - d_loss: 0.2334 - g_loss: 4.1720

<div class="k-default-codeblock">
```

```
</div>
  708/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:18 360ms/step - d_loss: 0.2333 - g_loss: 4.1756

<div class="k-default-codeblock">
```

```
</div>
  709/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:18 360ms/step - d_loss: 0.2333 - g_loss: 4.1793

<div class="k-default-codeblock">
```

```
</div>
  710/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:18 360ms/step - d_loss: 0.2332 - g_loss: 4.1829

<div class="k-default-codeblock">
```

```
</div>
  711/1094 ━━━━━━━━━━━━[37m━━━━━━━━  2:17 360ms/step - d_loss: 0.2331 - g_loss: 4.1865

<div class="k-default-codeblock">
```

```
</div>
  712/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:17 360ms/step - d_loss: 0.2331 - g_loss: 4.1901

<div class="k-default-codeblock">
```

```
</div>
  713/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:17 360ms/step - d_loss: 0.2330 - g_loss: 4.1936

<div class="k-default-codeblock">
```

```
</div>
  714/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:16 360ms/step - d_loss: 0.2329 - g_loss: 4.1972

<div class="k-default-codeblock">
```

```
</div>
  715/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:16 360ms/step - d_loss: 0.2329 - g_loss: 4.2007

<div class="k-default-codeblock">
```

```
</div>
  716/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:15 360ms/step - d_loss: 0.2328 - g_loss: 4.2042

<div class="k-default-codeblock">
```

```
</div>
  717/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:15 360ms/step - d_loss: 0.2327 - g_loss: 4.2077

<div class="k-default-codeblock">
```

```
</div>
  718/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:15 360ms/step - d_loss: 0.2327 - g_loss: 4.2111

<div class="k-default-codeblock">
```

```
</div>
  719/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:14 359ms/step - d_loss: 0.2326 - g_loss: 4.2146

<div class="k-default-codeblock">
```

```
</div>
  720/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:14 359ms/step - d_loss: 0.2326 - g_loss: 4.2180

<div class="k-default-codeblock">
```

```
</div>
  721/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:14 359ms/step - d_loss: 0.2325 - g_loss: 4.2214

<div class="k-default-codeblock">
```

```
</div>
  722/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:13 359ms/step - d_loss: 0.2324 - g_loss: 4.2248

<div class="k-default-codeblock">
```

```
</div>
  723/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:13 359ms/step - d_loss: 0.2324 - g_loss: 4.2282

<div class="k-default-codeblock">
```

```
</div>
  724/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:12 359ms/step - d_loss: 0.2323 - g_loss: 4.2315

<div class="k-default-codeblock">
```

```
</div>
  725/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:12 359ms/step - d_loss: 0.2323 - g_loss: 4.2349

<div class="k-default-codeblock">
```

```
</div>
  726/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:12 359ms/step - d_loss: 0.2322 - g_loss: 4.2382

<div class="k-default-codeblock">
```

```
</div>
  727/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:11 359ms/step - d_loss: 0.2322 - g_loss: 4.2415

<div class="k-default-codeblock">
```

```
</div>
  728/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:11 359ms/step - d_loss: 0.2321 - g_loss: 4.2448

<div class="k-default-codeblock">
```

```
</div>
  729/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:11 359ms/step - d_loss: 0.2321 - g_loss: 4.2480

<div class="k-default-codeblock">
```

```
</div>
  730/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:10 359ms/step - d_loss: 0.2320 - g_loss: 4.2513

<div class="k-default-codeblock">
```

```
</div>
  731/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:10 359ms/step - d_loss: 0.2320 - g_loss: 4.2545

<div class="k-default-codeblock">
```

```
</div>
  732/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:10 359ms/step - d_loss: 0.2319 - g_loss: 4.2577

<div class="k-default-codeblock">
```

```
</div>
  733/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:09 360ms/step - d_loss: 0.2319 - g_loss: 4.2609

<div class="k-default-codeblock">
```

```
</div>
  734/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:09 360ms/step - d_loss: 0.2318 - g_loss: 4.2641

<div class="k-default-codeblock">
```

```
</div>
  735/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:09 360ms/step - d_loss: 0.2318 - g_loss: 4.2672

<div class="k-default-codeblock">
```

```
</div>
  736/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:08 360ms/step - d_loss: 0.2317 - g_loss: 4.2703

<div class="k-default-codeblock">
```

```
</div>
  737/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:08 360ms/step - d_loss: 0.2317 - g_loss: 4.2735

<div class="k-default-codeblock">
```

```
</div>
  738/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:08 360ms/step - d_loss: 0.2316 - g_loss: 4.2766

<div class="k-default-codeblock">
```

```
</div>
  739/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:07 360ms/step - d_loss: 0.2316 - g_loss: 4.2797

<div class="k-default-codeblock">
```

```
</div>
  740/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:07 360ms/step - d_loss: 0.2316 - g_loss: 4.2827

<div class="k-default-codeblock">
```

```
</div>
  741/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:07 360ms/step - d_loss: 0.2315 - g_loss: 4.2858

<div class="k-default-codeblock">
```

```
</div>
  742/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:06 360ms/step - d_loss: 0.2315 - g_loss: 4.2888

<div class="k-default-codeblock">
```

```
</div>
  743/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:06 361ms/step - d_loss: 0.2314 - g_loss: 4.2918

<div class="k-default-codeblock">
```

```
</div>
  744/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:06 361ms/step - d_loss: 0.2314 - g_loss: 4.2948

<div class="k-default-codeblock">
```

```
</div>
  745/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:05 361ms/step - d_loss: 0.2314 - g_loss: 4.2978

<div class="k-default-codeblock">
```

```
</div>
  746/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:05 361ms/step - d_loss: 0.2313 - g_loss: 4.3008

<div class="k-default-codeblock">
```

```
</div>
  747/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:05 361ms/step - d_loss: 0.2313 - g_loss: 4.3037

<div class="k-default-codeblock">
```

```
</div>
  748/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:04 361ms/step - d_loss: 0.2313 - g_loss: 4.3066

<div class="k-default-codeblock">
```

```
</div>
  749/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:04 361ms/step - d_loss: 0.2312 - g_loss: 4.3096

<div class="k-default-codeblock">
```

```
</div>
  750/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:04 361ms/step - d_loss: 0.2312 - g_loss: 4.3125

<div class="k-default-codeblock">
```

```
</div>
  751/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:03 361ms/step - d_loss: 0.2312 - g_loss: 4.3153

<div class="k-default-codeblock">
```

```
</div>
  752/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:03 362ms/step - d_loss: 0.2311 - g_loss: 4.3182

<div class="k-default-codeblock">
```

```
</div>
  753/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:03 362ms/step - d_loss: 0.2311 - g_loss: 4.3210

<div class="k-default-codeblock">
```

```
</div>
  754/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:02 362ms/step - d_loss: 0.2311 - g_loss: 4.3239

<div class="k-default-codeblock">
```

```
</div>
  755/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:02 361ms/step - d_loss: 0.2310 - g_loss: 4.3267

<div class="k-default-codeblock">
```

```
</div>
  756/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:02 361ms/step - d_loss: 0.2310 - g_loss: 4.3295

<div class="k-default-codeblock">
```

```
</div>
  757/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:01 361ms/step - d_loss: 0.2310 - g_loss: 4.3323

<div class="k-default-codeblock">
```

```
</div>
  758/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:01 361ms/step - d_loss: 0.2310 - g_loss: 4.3350

<div class="k-default-codeblock">
```

```
</div>
  759/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:01 361ms/step - d_loss: 0.2309 - g_loss: 4.3378

<div class="k-default-codeblock">
```

```
</div>
  760/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:00 361ms/step - d_loss: 0.2309 - g_loss: 4.3405

<div class="k-default-codeblock">
```

```
</div>
  761/1094 ━━━━━━━━━━━━━[37m━━━━━━━  2:00 361ms/step - d_loss: 0.2309 - g_loss: 4.3433

<div class="k-default-codeblock">
```

```
</div>
  762/1094 ━━━━━━━━━━━━━[37m━━━━━━━  1:59 361ms/step - d_loss: 0.2309 - g_loss: 4.3460

<div class="k-default-codeblock">
```

```
</div>
  763/1094 ━━━━━━━━━━━━━[37m━━━━━━━  1:59 361ms/step - d_loss: 0.2308 - g_loss: 4.3487

<div class="k-default-codeblock">
```

```
</div>
  764/1094 ━━━━━━━━━━━━━[37m━━━━━━━  1:59 361ms/step - d_loss: 0.2308 - g_loss: 4.3513

<div class="k-default-codeblock">
```

```
</div>
  765/1094 ━━━━━━━━━━━━━[37m━━━━━━━  1:58 361ms/step - d_loss: 0.2308 - g_loss: 4.3540

<div class="k-default-codeblock">
```

```
</div>
  766/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:58 361ms/step - d_loss: 0.2308 - g_loss: 4.3567

<div class="k-default-codeblock">
```

```
</div>
  767/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:58 361ms/step - d_loss: 0.2307 - g_loss: 4.3593

<div class="k-default-codeblock">
```

```
</div>
  768/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:57 361ms/step - d_loss: 0.2307 - g_loss: 4.3619

<div class="k-default-codeblock">
```

```
</div>
  769/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:57 361ms/step - d_loss: 0.2307 - g_loss: 4.3645

<div class="k-default-codeblock">
```

```
</div>
  770/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:56 361ms/step - d_loss: 0.2307 - g_loss: 4.3671

<div class="k-default-codeblock">
```

```
</div>
  771/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:56 361ms/step - d_loss: 0.2307 - g_loss: 4.3697

<div class="k-default-codeblock">
```

```
</div>
  772/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:56 361ms/step - d_loss: 0.2306 - g_loss: 4.3722

<div class="k-default-codeblock">
```

```
</div>
  773/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:55 361ms/step - d_loss: 0.2306 - g_loss: 4.3748

<div class="k-default-codeblock">
```

```
</div>
  774/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:55 361ms/step - d_loss: 0.2306 - g_loss: 4.3773

<div class="k-default-codeblock">
```

```
</div>
  775/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:55 361ms/step - d_loss: 0.2306 - g_loss: 4.3798

<div class="k-default-codeblock">
```

```
</div>
  776/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:54 361ms/step - d_loss: 0.2306 - g_loss: 4.3823

<div class="k-default-codeblock">
```

```
</div>
  777/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:54 361ms/step - d_loss: 0.2305 - g_loss: 4.3848

<div class="k-default-codeblock">
```

```
</div>
  778/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:53 361ms/step - d_loss: 0.2305 - g_loss: 4.3873

<div class="k-default-codeblock">
```

```
</div>
  779/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:53 361ms/step - d_loss: 0.2305 - g_loss: 4.3897

<div class="k-default-codeblock">
```

```
</div>
  780/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:53 361ms/step - d_loss: 0.2305 - g_loss: 4.3922

<div class="k-default-codeblock">
```

```
</div>
  781/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:52 361ms/step - d_loss: 0.2305 - g_loss: 4.3946

<div class="k-default-codeblock">
```

```
</div>
  782/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:52 361ms/step - d_loss: 0.2305 - g_loss: 4.3970

<div class="k-default-codeblock">
```

```
</div>
  783/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:52 361ms/step - d_loss: 0.2305 - g_loss: 4.3994

<div class="k-default-codeblock">
```

```
</div>
  784/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:51 361ms/step - d_loss: 0.2305 - g_loss: 4.4018

<div class="k-default-codeblock">
```

```
</div>
  785/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:51 361ms/step - d_loss: 0.2304 - g_loss: 4.4042

<div class="k-default-codeblock">
```

```
</div>
  786/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:51 360ms/step - d_loss: 0.2304 - g_loss: 4.4065

<div class="k-default-codeblock">
```

```
</div>
  787/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:50 360ms/step - d_loss: 0.2304 - g_loss: 4.4089

<div class="k-default-codeblock">
```

```
</div>
  788/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:50 360ms/step - d_loss: 0.2304 - g_loss: 4.4112

<div class="k-default-codeblock">
```

```
</div>
  789/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:49 360ms/step - d_loss: 0.2304 - g_loss: 4.4135

<div class="k-default-codeblock">
```

```
</div>
  790/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:49 360ms/step - d_loss: 0.2304 - g_loss: 4.4158

<div class="k-default-codeblock">
```

```
</div>
  791/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:49 360ms/step - d_loss: 0.2304 - g_loss: 4.4181

<div class="k-default-codeblock">
```

```
</div>
  792/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:48 360ms/step - d_loss: 0.2304 - g_loss: 4.4204

<div class="k-default-codeblock">
```

```
</div>
  793/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:48 360ms/step - d_loss: 0.2304 - g_loss: 4.4227

<div class="k-default-codeblock">
```

```
</div>
  794/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:48 360ms/step - d_loss: 0.2304 - g_loss: 4.4249

<div class="k-default-codeblock">
```

```
</div>
  795/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:47 360ms/step - d_loss: 0.2304 - g_loss: 4.4272

<div class="k-default-codeblock">
```

```
</div>
  796/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:47 360ms/step - d_loss: 0.2304 - g_loss: 4.4294

<div class="k-default-codeblock">
```

```
</div>
  797/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:46 360ms/step - d_loss: 0.2304 - g_loss: 4.4316

<div class="k-default-codeblock">
```

```
</div>
  798/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:46 360ms/step - d_loss: 0.2304 - g_loss: 4.4338

<div class="k-default-codeblock">
```

```
</div>
  799/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:46 360ms/step - d_loss: 0.2304 - g_loss: 4.4360

<div class="k-default-codeblock">
```

```
</div>
  800/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:45 360ms/step - d_loss: 0.2304 - g_loss: 4.4382

<div class="k-default-codeblock">
```

```
</div>
  801/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:45 360ms/step - d_loss: 0.2304 - g_loss: 4.4403

<div class="k-default-codeblock">
```

```
</div>
  802/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:45 360ms/step - d_loss: 0.2304 - g_loss: 4.4425

<div class="k-default-codeblock">
```

```
</div>
  803/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:44 360ms/step - d_loss: 0.2304 - g_loss: 4.4446

<div class="k-default-codeblock">
```

```
</div>
  804/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:44 360ms/step - d_loss: 0.2304 - g_loss: 4.4468

<div class="k-default-codeblock">
```

```
</div>
  805/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:44 360ms/step - d_loss: 0.2304 - g_loss: 4.4489

<div class="k-default-codeblock">
```

```
</div>
  806/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:43 360ms/step - d_loss: 0.2304 - g_loss: 4.4510

<div class="k-default-codeblock">
```

```
</div>
  807/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:43 360ms/step - d_loss: 0.2304 - g_loss: 4.4531

<div class="k-default-codeblock">
```

```
</div>
  808/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:42 360ms/step - d_loss: 0.2304 - g_loss: 4.4552

<div class="k-default-codeblock">
```

```
</div>
  809/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:42 360ms/step - d_loss: 0.2304 - g_loss: 4.4572

<div class="k-default-codeblock">
```

```
</div>
  810/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:42 360ms/step - d_loss: 0.2304 - g_loss: 4.4593

<div class="k-default-codeblock">
```

```
</div>
  811/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:41 360ms/step - d_loss: 0.2304 - g_loss: 4.4614

<div class="k-default-codeblock">
```

```
</div>
  812/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:41 360ms/step - d_loss: 0.2304 - g_loss: 4.4634

<div class="k-default-codeblock">
```

```
</div>
  813/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:41 360ms/step - d_loss: 0.2304 - g_loss: 4.4654

<div class="k-default-codeblock">
```

```
</div>
  814/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:40 360ms/step - d_loss: 0.2304 - g_loss: 4.4674

<div class="k-default-codeblock">
```

```
</div>
  815/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:40 360ms/step - d_loss: 0.2304 - g_loss: 4.4694

<div class="k-default-codeblock">
```

```
</div>
  816/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:39 360ms/step - d_loss: 0.2304 - g_loss: 4.4714

<div class="k-default-codeblock">
```

```
</div>
  817/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:39 360ms/step - d_loss: 0.2304 - g_loss: 4.4734

<div class="k-default-codeblock">
```

```
</div>
  818/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:39 359ms/step - d_loss: 0.2304 - g_loss: 4.4754

<div class="k-default-codeblock">
```

```
</div>
  819/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:38 359ms/step - d_loss: 0.2304 - g_loss: 4.4773

<div class="k-default-codeblock">
```

```
</div>
  820/1094 ━━━━━━━━━━━━━━[37m━━━━━━  1:38 359ms/step - d_loss: 0.2304 - g_loss: 4.4793

<div class="k-default-codeblock">
```

```
</div>
  821/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:38 359ms/step - d_loss: 0.2305 - g_loss: 4.4812

<div class="k-default-codeblock">
```

```
</div>
  822/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:37 359ms/step - d_loss: 0.2305 - g_loss: 4.4831

<div class="k-default-codeblock">
```

```
</div>
  823/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:37 359ms/step - d_loss: 0.2305 - g_loss: 4.4850

<div class="k-default-codeblock">
```

```
</div>
  824/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:37 359ms/step - d_loss: 0.2305 - g_loss: 4.4869

<div class="k-default-codeblock">
```

```
</div>
  825/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:36 359ms/step - d_loss: 0.2305 - g_loss: 4.4888

<div class="k-default-codeblock">
```

```
</div>
  826/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:36 359ms/step - d_loss: 0.2305 - g_loss: 4.4907

<div class="k-default-codeblock">
```

```
</div>
  827/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:35 359ms/step - d_loss: 0.2305 - g_loss: 4.4925

<div class="k-default-codeblock">
```

```
</div>
  828/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:35 359ms/step - d_loss: 0.2305 - g_loss: 4.4944

<div class="k-default-codeblock">
```

```
</div>
  829/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:35 359ms/step - d_loss: 0.2306 - g_loss: 4.4962

<div class="k-default-codeblock">
```

```
</div>
  830/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:34 359ms/step - d_loss: 0.2306 - g_loss: 4.4981

<div class="k-default-codeblock">
```

```
</div>
  831/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:34 359ms/step - d_loss: 0.2306 - g_loss: 4.4999

<div class="k-default-codeblock">
```

```
</div>
  832/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:34 359ms/step - d_loss: 0.2306 - g_loss: 4.5017

<div class="k-default-codeblock">
```

```
</div>
  833/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:33 359ms/step - d_loss: 0.2306 - g_loss: 4.5035

<div class="k-default-codeblock">
```

```
</div>
  834/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:33 359ms/step - d_loss: 0.2306 - g_loss: 4.5053

<div class="k-default-codeblock">
```

```
</div>
  835/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:32 359ms/step - d_loss: 0.2306 - g_loss: 4.5071

<div class="k-default-codeblock">
```

```
</div>
  836/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:32 359ms/step - d_loss: 0.2307 - g_loss: 4.5088

<div class="k-default-codeblock">
```

```
</div>
  837/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:32 359ms/step - d_loss: 0.2307 - g_loss: 4.5106

<div class="k-default-codeblock">
```

```
</div>
  838/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:31 359ms/step - d_loss: 0.2307 - g_loss: 4.5124

<div class="k-default-codeblock">
```

```
</div>
  839/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:31 359ms/step - d_loss: 0.2307 - g_loss: 4.5141

<div class="k-default-codeblock">
```

```
</div>
  840/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:31 359ms/step - d_loss: 0.2307 - g_loss: 4.5158

<div class="k-default-codeblock">
```

```
</div>
  841/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:30 359ms/step - d_loss: 0.2308 - g_loss: 4.5175

<div class="k-default-codeblock">
```

```
</div>
  842/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:30 359ms/step - d_loss: 0.2308 - g_loss: 4.5192

<div class="k-default-codeblock">
```

```
</div>
  843/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:30 359ms/step - d_loss: 0.2308 - g_loss: 4.5209

<div class="k-default-codeblock">
```

```
</div>
  844/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:29 359ms/step - d_loss: 0.2308 - g_loss: 4.5226

<div class="k-default-codeblock">
```

```
</div>
  845/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:29 359ms/step - d_loss: 0.2308 - g_loss: 4.5243

<div class="k-default-codeblock">
```

```
</div>
  846/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:28 359ms/step - d_loss: 0.2309 - g_loss: 4.5260

<div class="k-default-codeblock">
```

```
</div>
  847/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:28 359ms/step - d_loss: 0.2309 - g_loss: 4.5276

<div class="k-default-codeblock">
```

```
</div>
  848/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:28 359ms/step - d_loss: 0.2309 - g_loss: 4.5293

<div class="k-default-codeblock">
```

```
</div>
  849/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:27 359ms/step - d_loss: 0.2309 - g_loss: 4.5309

<div class="k-default-codeblock">
```

```
</div>
  850/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:27 359ms/step - d_loss: 0.2309 - g_loss: 4.5325

<div class="k-default-codeblock">
```

```
</div>
  851/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:27 359ms/step - d_loss: 0.2310 - g_loss: 4.5342

<div class="k-default-codeblock">
```

```
</div>
  852/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:26 358ms/step - d_loss: 0.2310 - g_loss: 4.5358

<div class="k-default-codeblock">
```

```
</div>
  853/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:26 358ms/step - d_loss: 0.2310 - g_loss: 4.5374

<div class="k-default-codeblock">
```

```
</div>
  854/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:26 358ms/step - d_loss: 0.2310 - g_loss: 4.5390

<div class="k-default-codeblock">
```

```
</div>
  855/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:25 358ms/step - d_loss: 0.2311 - g_loss: 4.5405

<div class="k-default-codeblock">
```

```
</div>
  856/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:25 358ms/step - d_loss: 0.2311 - g_loss: 4.5421

<div class="k-default-codeblock">
```

```
</div>
  857/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:24 358ms/step - d_loss: 0.2311 - g_loss: 4.5437

<div class="k-default-codeblock">
```

```
</div>
  858/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:24 358ms/step - d_loss: 0.2311 - g_loss: 4.5452

<div class="k-default-codeblock">
```

```
</div>
  859/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:24 358ms/step - d_loss: 0.2312 - g_loss: 4.5468

<div class="k-default-codeblock">
```

```
</div>
  860/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:23 358ms/step - d_loss: 0.2312 - g_loss: 4.5483

<div class="k-default-codeblock">
```

```
</div>
  861/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:23 358ms/step - d_loss: 0.2312 - g_loss: 4.5498

<div class="k-default-codeblock">
```

```
</div>
  862/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:23 358ms/step - d_loss: 0.2312 - g_loss: 4.5513

<div class="k-default-codeblock">
```

```
</div>
  863/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:22 358ms/step - d_loss: 0.2313 - g_loss: 4.5528

<div class="k-default-codeblock">
```

```
</div>
  864/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:22 358ms/step - d_loss: 0.2313 - g_loss: 4.5543

<div class="k-default-codeblock">
```

```
</div>
  865/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:22 358ms/step - d_loss: 0.2313 - g_loss: 4.5558

<div class="k-default-codeblock">
```

```
</div>
  866/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:21 358ms/step - d_loss: 0.2314 - g_loss: 4.5573

<div class="k-default-codeblock">
```

```
</div>
  867/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:21 358ms/step - d_loss: 0.2314 - g_loss: 4.5587

<div class="k-default-codeblock">
```

```
</div>
  868/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:20 358ms/step - d_loss: 0.2314 - g_loss: 4.5602

<div class="k-default-codeblock">
```

```
</div>
  869/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:20 358ms/step - d_loss: 0.2315 - g_loss: 4.5617

<div class="k-default-codeblock">
```

```
</div>
  870/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:20 358ms/step - d_loss: 0.2315 - g_loss: 4.5631

<div class="k-default-codeblock">
```

```
</div>
  871/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:19 358ms/step - d_loss: 0.2315 - g_loss: 4.5645

<div class="k-default-codeblock">
```

```
</div>
  872/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:19 358ms/step - d_loss: 0.2316 - g_loss: 4.5659

<div class="k-default-codeblock">
```

```
</div>
  873/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:19 358ms/step - d_loss: 0.2316 - g_loss: 4.5673

<div class="k-default-codeblock">
```

```
</div>
  874/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:18 358ms/step - d_loss: 0.2316 - g_loss: 4.5687

<div class="k-default-codeblock">
```

```
</div>
  875/1094 ━━━━━━━━━━━━━━━[37m━━━━━  1:18 358ms/step - d_loss: 0.2317 - g_loss: 4.5701

<div class="k-default-codeblock">
```

```
</div>
  876/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:17 358ms/step - d_loss: 0.2317 - g_loss: 4.5715

<div class="k-default-codeblock">
```

```
</div>
  877/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:17 358ms/step - d_loss: 0.2317 - g_loss: 4.5729

<div class="k-default-codeblock">
```

```
</div>
  878/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:17 358ms/step - d_loss: 0.2318 - g_loss: 4.5743

<div class="k-default-codeblock">
```

```
</div>
  879/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:16 358ms/step - d_loss: 0.2318 - g_loss: 4.5756

<div class="k-default-codeblock">
```

```
</div>
  880/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:16 358ms/step - d_loss: 0.2318 - g_loss: 4.5770

<div class="k-default-codeblock">
```

```
</div>
  881/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:16 358ms/step - d_loss: 0.2319 - g_loss: 4.5783

<div class="k-default-codeblock">
```

```
</div>
  882/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:15 358ms/step - d_loss: 0.2319 - g_loss: 4.5797

<div class="k-default-codeblock">
```

```
</div>
  883/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:15 358ms/step - d_loss: 0.2319 - g_loss: 4.5810

<div class="k-default-codeblock">
```

```
</div>
  884/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:15 358ms/step - d_loss: 0.2320 - g_loss: 4.5823

<div class="k-default-codeblock">
```

```
</div>
  885/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:14 358ms/step - d_loss: 0.2320 - g_loss: 4.5836

<div class="k-default-codeblock">
```

```
</div>
  886/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:14 358ms/step - d_loss: 0.2320 - g_loss: 4.5849

<div class="k-default-codeblock">
```

```
</div>
  887/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:14 358ms/step - d_loss: 0.2321 - g_loss: 4.5862

<div class="k-default-codeblock">
```

```
</div>
  888/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:13 357ms/step - d_loss: 0.2321 - g_loss: 4.5875

<div class="k-default-codeblock">
```

```
</div>
  889/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:13 357ms/step - d_loss: 0.2322 - g_loss: 4.5888

<div class="k-default-codeblock">
```

```
</div>
  890/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:12 357ms/step - d_loss: 0.2322 - g_loss: 4.5901

<div class="k-default-codeblock">
```

```
</div>
  891/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:12 357ms/step - d_loss: 0.2322 - g_loss: 4.5913

<div class="k-default-codeblock">
```

```
</div>
  892/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:12 357ms/step - d_loss: 0.2323 - g_loss: 4.5926

<div class="k-default-codeblock">
```

```
</div>
  893/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:11 357ms/step - d_loss: 0.2323 - g_loss: 4.5938

<div class="k-default-codeblock">
```

```
</div>
  894/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:11 357ms/step - d_loss: 0.2323 - g_loss: 4.5951

<div class="k-default-codeblock">
```

```
</div>
  895/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:11 357ms/step - d_loss: 0.2324 - g_loss: 4.5963

<div class="k-default-codeblock">
```

```
</div>
  896/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:10 357ms/step - d_loss: 0.2324 - g_loss: 4.5976

<div class="k-default-codeblock">
```

```
</div>
  897/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:10 357ms/step - d_loss: 0.2325 - g_loss: 4.5988

<div class="k-default-codeblock">
```

```
</div>
  898/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:10 357ms/step - d_loss: 0.2325 - g_loss: 4.6000

<div class="k-default-codeblock">
```

```
</div>
  899/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:09 357ms/step - d_loss: 0.2325 - g_loss: 4.6012

<div class="k-default-codeblock">
```

```
</div>
  900/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:09 357ms/step - d_loss: 0.2326 - g_loss: 4.6024

<div class="k-default-codeblock">
```

```
</div>
  901/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:08 357ms/step - d_loss: 0.2326 - g_loss: 4.6036

<div class="k-default-codeblock">
```

```
</div>
  902/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:08 357ms/step - d_loss: 0.2327 - g_loss: 4.6048

<div class="k-default-codeblock">
```

```
</div>
  903/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:08 357ms/step - d_loss: 0.2327 - g_loss: 4.6059

<div class="k-default-codeblock">
```

```
</div>
  904/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:07 358ms/step - d_loss: 0.2327 - g_loss: 4.6071

<div class="k-default-codeblock">
```

```
</div>
  905/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:07 358ms/step - d_loss: 0.2328 - g_loss: 4.6083

<div class="k-default-codeblock">
```

```
</div>
  906/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:07 358ms/step - d_loss: 0.2328 - g_loss: 4.6094

<div class="k-default-codeblock">
```

```
</div>
  907/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:06 358ms/step - d_loss: 0.2329 - g_loss: 4.6106

<div class="k-default-codeblock">
```

```
</div>
  908/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:06 358ms/step - d_loss: 0.2329 - g_loss: 4.6117

<div class="k-default-codeblock">
```

```
</div>
  909/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:06 358ms/step - d_loss: 0.2329 - g_loss: 4.6128

<div class="k-default-codeblock">
```

```
</div>
  910/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:05 358ms/step - d_loss: 0.2330 - g_loss: 4.6140

<div class="k-default-codeblock">
```

```
</div>
  911/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:05 358ms/step - d_loss: 0.2330 - g_loss: 4.6151

<div class="k-default-codeblock">
```

```
</div>
  912/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:05 358ms/step - d_loss: 0.2331 - g_loss: 4.6162

<div class="k-default-codeblock">
```

```
</div>
  913/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:04 358ms/step - d_loss: 0.2331 - g_loss: 4.6173

<div class="k-default-codeblock">
```

```
</div>
  914/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:04 358ms/step - d_loss: 0.2332 - g_loss: 4.6184

<div class="k-default-codeblock">
```

```
</div>
  915/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:04 358ms/step - d_loss: 0.2332 - g_loss: 4.6195

<div class="k-default-codeblock">
```

```
</div>
  916/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:03 359ms/step - d_loss: 0.2332 - g_loss: 4.6205

<div class="k-default-codeblock">
```

```
</div>
  917/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:03 359ms/step - d_loss: 0.2333 - g_loss: 4.6216

<div class="k-default-codeblock">
```

```
</div>
  918/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:03 359ms/step - d_loss: 0.2333 - g_loss: 4.6227

<div class="k-default-codeblock">
```

```
</div>
  919/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:02 359ms/step - d_loss: 0.2334 - g_loss: 4.6237

<div class="k-default-codeblock">
```

```
</div>
  920/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:02 359ms/step - d_loss: 0.2334 - g_loss: 4.6248

<div class="k-default-codeblock">
```

```
</div>
  921/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:02 359ms/step - d_loss: 0.2335 - g_loss: 4.6258

<div class="k-default-codeblock">
```

```
</div>
  922/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:01 359ms/step - d_loss: 0.2335 - g_loss: 4.6269

<div class="k-default-codeblock">
```

```
</div>
  923/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:01 359ms/step - d_loss: 0.2336 - g_loss: 4.6279

<div class="k-default-codeblock">
```

```
</div>
  924/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:01 360ms/step - d_loss: 0.2336 - g_loss: 4.6289

<div class="k-default-codeblock">
```

```
</div>
  925/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:00 360ms/step - d_loss: 0.2337 - g_loss: 4.6299

<div class="k-default-codeblock">
```

```
</div>
  926/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:00 360ms/step - d_loss: 0.2337 - g_loss: 4.6309

<div class="k-default-codeblock">
```

```
</div>
  927/1094 ━━━━━━━━━━━━━━━━[37m━━━━  1:00 360ms/step - d_loss: 0.2337 - g_loss: 4.6319

<div class="k-default-codeblock">
```

```
</div>
  928/1094 ━━━━━━━━━━━━━━━━[37m━━━━  59s 360ms/step - d_loss: 0.2338 - g_loss: 4.6329 

<div class="k-default-codeblock">
```

```
</div>
  929/1094 ━━━━━━━━━━━━━━━━[37m━━━━  59s 360ms/step - d_loss: 0.2338 - g_loss: 4.6339

<div class="k-default-codeblock">
```

```
</div>
  930/1094 ━━━━━━━━━━━━━━━━━[37m━━━  58s 359ms/step - d_loss: 0.2339 - g_loss: 4.6349

<div class="k-default-codeblock">
```

```
</div>
  931/1094 ━━━━━━━━━━━━━━━━━[37m━━━  58s 359ms/step - d_loss: 0.2339 - g_loss: 4.6359

<div class="k-default-codeblock">
```

```
</div>
  932/1094 ━━━━━━━━━━━━━━━━━[37m━━━  58s 359ms/step - d_loss: 0.2340 - g_loss: 4.6368

<div class="k-default-codeblock">
```

```
</div>
  933/1094 ━━━━━━━━━━━━━━━━━[37m━━━  57s 359ms/step - d_loss: 0.2340 - g_loss: 4.6378

<div class="k-default-codeblock">
```

```
</div>
  934/1094 ━━━━━━━━━━━━━━━━━[37m━━━  57s 359ms/step - d_loss: 0.2341 - g_loss: 4.6387

<div class="k-default-codeblock">
```

```
</div>
  935/1094 ━━━━━━━━━━━━━━━━━[37m━━━  57s 359ms/step - d_loss: 0.2341 - g_loss: 4.6397

<div class="k-default-codeblock">
```

```
</div>
  936/1094 ━━━━━━━━━━━━━━━━━[37m━━━  56s 359ms/step - d_loss: 0.2342 - g_loss: 4.6406

<div class="k-default-codeblock">
```

```
</div>
  937/1094 ━━━━━━━━━━━━━━━━━[37m━━━  56s 359ms/step - d_loss: 0.2342 - g_loss: 4.6416

<div class="k-default-codeblock">
```

```
</div>
  938/1094 ━━━━━━━━━━━━━━━━━[37m━━━  56s 359ms/step - d_loss: 0.2343 - g_loss: 4.6425

<div class="k-default-codeblock">
```

```
</div>
  939/1094 ━━━━━━━━━━━━━━━━━[37m━━━  55s 359ms/step - d_loss: 0.2343 - g_loss: 4.6434

<div class="k-default-codeblock">
```

```
</div>
  940/1094 ━━━━━━━━━━━━━━━━━[37m━━━  55s 359ms/step - d_loss: 0.2344 - g_loss: 4.6443

<div class="k-default-codeblock">
```

```
</div>
  941/1094 ━━━━━━━━━━━━━━━━━[37m━━━  54s 359ms/step - d_loss: 0.2344 - g_loss: 4.6452

<div class="k-default-codeblock">
```

```
</div>
  942/1094 ━━━━━━━━━━━━━━━━━[37m━━━  54s 359ms/step - d_loss: 0.2345 - g_loss: 4.6461

<div class="k-default-codeblock">
```

```
</div>
  943/1094 ━━━━━━━━━━━━━━━━━[37m━━━  54s 359ms/step - d_loss: 0.2345 - g_loss: 4.6470

<div class="k-default-codeblock">
```

```
</div>
  944/1094 ━━━━━━━━━━━━━━━━━[37m━━━  53s 359ms/step - d_loss: 0.2346 - g_loss: 4.6479

<div class="k-default-codeblock">
```

```
</div>
  945/1094 ━━━━━━━━━━━━━━━━━[37m━━━  53s 359ms/step - d_loss: 0.2346 - g_loss: 4.6488

<div class="k-default-codeblock">
```

```
</div>
  946/1094 ━━━━━━━━━━━━━━━━━[37m━━━  53s 359ms/step - d_loss: 0.2347 - g_loss: 4.6497

<div class="k-default-codeblock">
```

```
</div>
  947/1094 ━━━━━━━━━━━━━━━━━[37m━━━  52s 359ms/step - d_loss: 0.2347 - g_loss: 4.6506

<div class="k-default-codeblock">
```

```
</div>
  948/1094 ━━━━━━━━━━━━━━━━━[37m━━━  52s 359ms/step - d_loss: 0.2348 - g_loss: 4.6514

<div class="k-default-codeblock">
```

```
</div>
  949/1094 ━━━━━━━━━━━━━━━━━[37m━━━  52s 359ms/step - d_loss: 0.2348 - g_loss: 4.6523

<div class="k-default-codeblock">
```

```
</div>
  950/1094 ━━━━━━━━━━━━━━━━━[37m━━━  51s 359ms/step - d_loss: 0.2349 - g_loss: 4.6532

<div class="k-default-codeblock">
```

```
</div>
  951/1094 ━━━━━━━━━━━━━━━━━[37m━━━  51s 359ms/step - d_loss: 0.2349 - g_loss: 4.6540

<div class="k-default-codeblock">
```

```
</div>
  952/1094 ━━━━━━━━━━━━━━━━━[37m━━━  50s 359ms/step - d_loss: 0.2350 - g_loss: 4.6549

<div class="k-default-codeblock">
```

```
</div>
  953/1094 ━━━━━━━━━━━━━━━━━[37m━━━  50s 359ms/step - d_loss: 0.2350 - g_loss: 4.6557

<div class="k-default-codeblock">
```

```
</div>
  954/1094 ━━━━━━━━━━━━━━━━━[37m━━━  50s 359ms/step - d_loss: 0.2351 - g_loss: 4.6565

<div class="k-default-codeblock">
```

```
</div>
  955/1094 ━━━━━━━━━━━━━━━━━[37m━━━  49s 359ms/step - d_loss: 0.2351 - g_loss: 4.6574

<div class="k-default-codeblock">
```

```
</div>
  956/1094 ━━━━━━━━━━━━━━━━━[37m━━━  49s 359ms/step - d_loss: 0.2352 - g_loss: 4.6582

<div class="k-default-codeblock">
```

```
</div>
  957/1094 ━━━━━━━━━━━━━━━━━[37m━━━  49s 359ms/step - d_loss: 0.2352 - g_loss: 4.6590

<div class="k-default-codeblock">
```

```
</div>
  958/1094 ━━━━━━━━━━━━━━━━━[37m━━━  48s 359ms/step - d_loss: 0.2353 - g_loss: 4.6598

<div class="k-default-codeblock">
```

```
</div>
  959/1094 ━━━━━━━━━━━━━━━━━[37m━━━  48s 359ms/step - d_loss: 0.2353 - g_loss: 4.6606

<div class="k-default-codeblock">
```

```
</div>
  960/1094 ━━━━━━━━━━━━━━━━━[37m━━━  48s 359ms/step - d_loss: 0.2354 - g_loss: 4.6614

<div class="k-default-codeblock">
```

```
</div>
  961/1094 ━━━━━━━━━━━━━━━━━[37m━━━  47s 359ms/step - d_loss: 0.2354 - g_loss: 4.6622

<div class="k-default-codeblock">
```

```
</div>
  962/1094 ━━━━━━━━━━━━━━━━━[37m━━━  47s 359ms/step - d_loss: 0.2355 - g_loss: 4.6630

<div class="k-default-codeblock">
```

```
</div>
  963/1094 ━━━━━━━━━━━━━━━━━[37m━━━  46s 359ms/step - d_loss: 0.2355 - g_loss: 4.6638

<div class="k-default-codeblock">
```

```
</div>
  964/1094 ━━━━━━━━━━━━━━━━━[37m━━━  46s 359ms/step - d_loss: 0.2356 - g_loss: 4.6645

<div class="k-default-codeblock">
```

```
</div>
  965/1094 ━━━━━━━━━━━━━━━━━[37m━━━  46s 359ms/step - d_loss: 0.2356 - g_loss: 4.6653

<div class="k-default-codeblock">
```

```
</div>
  966/1094 ━━━━━━━━━━━━━━━━━[37m━━━  45s 359ms/step - d_loss: 0.2357 - g_loss: 4.6661

<div class="k-default-codeblock">
```

```
</div>
  967/1094 ━━━━━━━━━━━━━━━━━[37m━━━  45s 359ms/step - d_loss: 0.2357 - g_loss: 4.6668

<div class="k-default-codeblock">
```

```
</div>
  968/1094 ━━━━━━━━━━━━━━━━━[37m━━━  45s 359ms/step - d_loss: 0.2358 - g_loss: 4.6676

<div class="k-default-codeblock">
```

```
</div>
  969/1094 ━━━━━━━━━━━━━━━━━[37m━━━  44s 359ms/step - d_loss: 0.2358 - g_loss: 4.6683

<div class="k-default-codeblock">
```

```
</div>
  970/1094 ━━━━━━━━━━━━━━━━━[37m━━━  44s 359ms/step - d_loss: 0.2359 - g_loss: 4.6691

<div class="k-default-codeblock">
```

```
</div>
  971/1094 ━━━━━━━━━━━━━━━━━[37m━━━  44s 359ms/step - d_loss: 0.2360 - g_loss: 4.6698

<div class="k-default-codeblock">
```

```
</div>
  972/1094 ━━━━━━━━━━━━━━━━━[37m━━━  43s 359ms/step - d_loss: 0.2360 - g_loss: 4.6706

<div class="k-default-codeblock">
```

```
</div>
  973/1094 ━━━━━━━━━━━━━━━━━[37m━━━  43s 359ms/step - d_loss: 0.2361 - g_loss: 4.6713

<div class="k-default-codeblock">
```

```
</div>
  974/1094 ━━━━━━━━━━━━━━━━━[37m━━━  43s 359ms/step - d_loss: 0.2361 - g_loss: 4.6720

<div class="k-default-codeblock">
```

```
</div>
  975/1094 ━━━━━━━━━━━━━━━━━[37m━━━  42s 358ms/step - d_loss: 0.2362 - g_loss: 4.6727

<div class="k-default-codeblock">
```

```
</div>
  976/1094 ━━━━━━━━━━━━━━━━━[37m━━━  42s 358ms/step - d_loss: 0.2362 - g_loss: 4.6734

<div class="k-default-codeblock">
```

```
</div>
  977/1094 ━━━━━━━━━━━━━━━━━[37m━━━  41s 358ms/step - d_loss: 0.2363 - g_loss: 4.6741

<div class="k-default-codeblock">
```

```
</div>
  978/1094 ━━━━━━━━━━━━━━━━━[37m━━━  41s 358ms/step - d_loss: 0.2363 - g_loss: 4.6748

<div class="k-default-codeblock">
```

```
</div>
  979/1094 ━━━━━━━━━━━━━━━━━[37m━━━  41s 358ms/step - d_loss: 0.2364 - g_loss: 4.6755

<div class="k-default-codeblock">
```

```
</div>
  980/1094 ━━━━━━━━━━━━━━━━━[37m━━━  40s 358ms/step - d_loss: 0.2365 - g_loss: 4.6762

<div class="k-default-codeblock">
```

```
</div>
  981/1094 ━━━━━━━━━━━━━━━━━[37m━━━  40s 358ms/step - d_loss: 0.2365 - g_loss: 4.6769

<div class="k-default-codeblock">
```

```
</div>
  982/1094 ━━━━━━━━━━━━━━━━━[37m━━━  40s 358ms/step - d_loss: 0.2366 - g_loss: 4.6776

<div class="k-default-codeblock">
```

```
</div>
  983/1094 ━━━━━━━━━━━━━━━━━[37m━━━  39s 358ms/step - d_loss: 0.2366 - g_loss: 4.6782

<div class="k-default-codeblock">
```

```
</div>
  984/1094 ━━━━━━━━━━━━━━━━━[37m━━━  39s 358ms/step - d_loss: 0.2367 - g_loss: 4.6789

<div class="k-default-codeblock">
```

```
</div>
  985/1094 ━━━━━━━━━━━━━━━━━━[37m━━  39s 358ms/step - d_loss: 0.2367 - g_loss: 4.6796

<div class="k-default-codeblock">
```

```
</div>
  986/1094 ━━━━━━━━━━━━━━━━━━[37m━━  38s 359ms/step - d_loss: 0.2368 - g_loss: 4.6802

<div class="k-default-codeblock">
```

```
</div>
  987/1094 ━━━━━━━━━━━━━━━━━━[37m━━  38s 359ms/step - d_loss: 0.2368 - g_loss: 4.6809

<div class="k-default-codeblock">
```

```
</div>
  988/1094 ━━━━━━━━━━━━━━━━━━[37m━━  38s 359ms/step - d_loss: 0.2369 - g_loss: 4.6815

<div class="k-default-codeblock">
```

```
</div>
  989/1094 ━━━━━━━━━━━━━━━━━━[37m━━  37s 359ms/step - d_loss: 0.2370 - g_loss: 4.6822

<div class="k-default-codeblock">
```

```
</div>
  990/1094 ━━━━━━━━━━━━━━━━━━[37m━━  37s 359ms/step - d_loss: 0.2370 - g_loss: 4.6828

<div class="k-default-codeblock">
```

```
</div>
  991/1094 ━━━━━━━━━━━━━━━━━━[37m━━  36s 359ms/step - d_loss: 0.2371 - g_loss: 4.6835

<div class="k-default-codeblock">
```

```
</div>
  992/1094 ━━━━━━━━━━━━━━━━━━[37m━━  36s 359ms/step - d_loss: 0.2371 - g_loss: 4.6841

<div class="k-default-codeblock">
```

```
</div>
  993/1094 ━━━━━━━━━━━━━━━━━━[37m━━  36s 359ms/step - d_loss: 0.2372 - g_loss: 4.6847

<div class="k-default-codeblock">
```

```
</div>
  994/1094 ━━━━━━━━━━━━━━━━━━[37m━━  35s 359ms/step - d_loss: 0.2372 - g_loss: 4.6853

<div class="k-default-codeblock">
```

```
</div>
  995/1094 ━━━━━━━━━━━━━━━━━━[37m━━  35s 359ms/step - d_loss: 0.2373 - g_loss: 4.6860

<div class="k-default-codeblock">
```

```
</div>
  996/1094 ━━━━━━━━━━━━━━━━━━[37m━━  35s 359ms/step - d_loss: 0.2373 - g_loss: 4.6866

<div class="k-default-codeblock">
```

```
</div>
  997/1094 ━━━━━━━━━━━━━━━━━━[37m━━  34s 359ms/step - d_loss: 0.2374 - g_loss: 4.6872

<div class="k-default-codeblock">
```

```
</div>
  998/1094 ━━━━━━━━━━━━━━━━━━[37m━━  34s 360ms/step - d_loss: 0.2374 - g_loss: 4.6878

<div class="k-default-codeblock">
```

```
</div>
  999/1094 ━━━━━━━━━━━━━━━━━━[37m━━  34s 360ms/step - d_loss: 0.2375 - g_loss: 4.6884

<div class="k-default-codeblock">
```

```
</div>
 1000/1094 ━━━━━━━━━━━━━━━━━━[37m━━  33s 360ms/step - d_loss: 0.2376 - g_loss: 4.6890

<div class="k-default-codeblock">
```

```
</div>
 1001/1094 ━━━━━━━━━━━━━━━━━━[37m━━  33s 360ms/step - d_loss: 0.2376 - g_loss: 4.6896

<div class="k-default-codeblock">
```

```
</div>
 1002/1094 ━━━━━━━━━━━━━━━━━━[37m━━  33s 360ms/step - d_loss: 0.2377 - g_loss: 4.6902

<div class="k-default-codeblock">
```

```
</div>
 1003/1094 ━━━━━━━━━━━━━━━━━━[37m━━  32s 360ms/step - d_loss: 0.2377 - g_loss: 4.6907

<div class="k-default-codeblock">
```

```
</div>
 1004/1094 ━━━━━━━━━━━━━━━━━━[37m━━  32s 360ms/step - d_loss: 0.2378 - g_loss: 4.6913

<div class="k-default-codeblock">
```

```
</div>
 1005/1094 ━━━━━━━━━━━━━━━━━━[37m━━  31s 360ms/step - d_loss: 0.2378 - g_loss: 4.6919

<div class="k-default-codeblock">
```

```
</div>
 1006/1094 ━━━━━━━━━━━━━━━━━━[37m━━  31s 359ms/step - d_loss: 0.2379 - g_loss: 4.6925

<div class="k-default-codeblock">
```

```
</div>
 1007/1094 ━━━━━━━━━━━━━━━━━━[37m━━  31s 359ms/step - d_loss: 0.2379 - g_loss: 4.6930

<div class="k-default-codeblock">
```

```
</div>
 1008/1094 ━━━━━━━━━━━━━━━━━━[37m━━  30s 359ms/step - d_loss: 0.2380 - g_loss: 4.6936

<div class="k-default-codeblock">
```

```
</div>
 1009/1094 ━━━━━━━━━━━━━━━━━━[37m━━  30s 359ms/step - d_loss: 0.2381 - g_loss: 4.6941

<div class="k-default-codeblock">
```

```
</div>
 1010/1094 ━━━━━━━━━━━━━━━━━━[37m━━  30s 359ms/step - d_loss: 0.2381 - g_loss: 4.6947

<div class="k-default-codeblock">
```

```
</div>
 1011/1094 ━━━━━━━━━━━━━━━━━━[37m━━  29s 359ms/step - d_loss: 0.2382 - g_loss: 4.6952

<div class="k-default-codeblock">
```

```
</div>
 1012/1094 ━━━━━━━━━━━━━━━━━━[37m━━  29s 359ms/step - d_loss: 0.2382 - g_loss: 4.6958

<div class="k-default-codeblock">
```

```
</div>
 1013/1094 ━━━━━━━━━━━━━━━━━━[37m━━  29s 359ms/step - d_loss: 0.2383 - g_loss: 4.6963

<div class="k-default-codeblock">
```

```
</div>
 1014/1094 ━━━━━━━━━━━━━━━━━━[37m━━  28s 359ms/step - d_loss: 0.2383 - g_loss: 4.6969

<div class="k-default-codeblock">
```

```
</div>
 1015/1094 ━━━━━━━━━━━━━━━━━━[37m━━  28s 359ms/step - d_loss: 0.2384 - g_loss: 4.6974

<div class="k-default-codeblock">
```

```
</div>
 1016/1094 ━━━━━━━━━━━━━━━━━━[37m━━  28s 359ms/step - d_loss: 0.2384 - g_loss: 4.6979

<div class="k-default-codeblock">
```

```
</div>
 1017/1094 ━━━━━━━━━━━━━━━━━━[37m━━  27s 359ms/step - d_loss: 0.2385 - g_loss: 4.6984

<div class="k-default-codeblock">
```

```
</div>
 1018/1094 ━━━━━━━━━━━━━━━━━━[37m━━  27s 359ms/step - d_loss: 0.2385 - g_loss: 4.6990

<div class="k-default-codeblock">
```

```
</div>
 1019/1094 ━━━━━━━━━━━━━━━━━━[37m━━  26s 359ms/step - d_loss: 0.2386 - g_loss: 4.6995

<div class="k-default-codeblock">
```

```
</div>
 1020/1094 ━━━━━━━━━━━━━━━━━━[37m━━  26s 359ms/step - d_loss: 0.2387 - g_loss: 4.7000

<div class="k-default-codeblock">
```

```
</div>
 1021/1094 ━━━━━━━━━━━━━━━━━━[37m━━  26s 359ms/step - d_loss: 0.2387 - g_loss: 4.7005

<div class="k-default-codeblock">
```

```
</div>
 1022/1094 ━━━━━━━━━━━━━━━━━━[37m━━  25s 359ms/step - d_loss: 0.2388 - g_loss: 4.7010

<div class="k-default-codeblock">
```

```
</div>
 1023/1094 ━━━━━━━━━━━━━━━━━━[37m━━  25s 359ms/step - d_loss: 0.2388 - g_loss: 4.7015

<div class="k-default-codeblock">
```

```
</div>
 1024/1094 ━━━━━━━━━━━━━━━━━━[37m━━  25s 359ms/step - d_loss: 0.2389 - g_loss: 4.7020

<div class="k-default-codeblock">
```

```
</div>
 1025/1094 ━━━━━━━━━━━━━━━━━━[37m━━  24s 359ms/step - d_loss: 0.2389 - g_loss: 4.7025

<div class="k-default-codeblock">
```

```
</div>
 1026/1094 ━━━━━━━━━━━━━━━━━━[37m━━  24s 359ms/step - d_loss: 0.2390 - g_loss: 4.7029

<div class="k-default-codeblock">
```

```
</div>
 1027/1094 ━━━━━━━━━━━━━━━━━━[37m━━  24s 359ms/step - d_loss: 0.2390 - g_loss: 4.7034

<div class="k-default-codeblock">
```

```
</div>
 1028/1094 ━━━━━━━━━━━━━━━━━━[37m━━  23s 359ms/step - d_loss: 0.2391 - g_loss: 4.7039

<div class="k-default-codeblock">
```

```
</div>
 1029/1094 ━━━━━━━━━━━━━━━━━━[37m━━  23s 359ms/step - d_loss: 0.2392 - g_loss: 4.7044

<div class="k-default-codeblock">
```

```
</div>
 1030/1094 ━━━━━━━━━━━━━━━━━━[37m━━  22s 359ms/step - d_loss: 0.2392 - g_loss: 4.7048

<div class="k-default-codeblock">
```

```
</div>
 1031/1094 ━━━━━━━━━━━━━━━━━━[37m━━  22s 359ms/step - d_loss: 0.2393 - g_loss: 4.7053

<div class="k-default-codeblock">
```

```
</div>
 1032/1094 ━━━━━━━━━━━━━━━━━━[37m━━  22s 359ms/step - d_loss: 0.2393 - g_loss: 4.7057

<div class="k-default-codeblock">
```

```
</div>
 1033/1094 ━━━━━━━━━━━━━━━━━━[37m━━  21s 359ms/step - d_loss: 0.2394 - g_loss: 4.7062

<div class="k-default-codeblock">
```

```
</div>
 1034/1094 ━━━━━━━━━━━━━━━━━━[37m━━  21s 359ms/step - d_loss: 0.2394 - g_loss: 4.7066

<div class="k-default-codeblock">
```

```
</div>
 1035/1094 ━━━━━━━━━━━━━━━━━━[37m━━  21s 359ms/step - d_loss: 0.2395 - g_loss: 4.7071

<div class="k-default-codeblock">
```

```
</div>
 1036/1094 ━━━━━━━━━━━━━━━━━━[37m━━  20s 359ms/step - d_loss: 0.2396 - g_loss: 4.7075

<div class="k-default-codeblock">
```

```
</div>
 1037/1094 ━━━━━━━━━━━━━━━━━━[37m━━  20s 359ms/step - d_loss: 0.2396 - g_loss: 4.7080

<div class="k-default-codeblock">
```

```
</div>
 1038/1094 ━━━━━━━━━━━━━━━━━━[37m━━  20s 359ms/step - d_loss: 0.2397 - g_loss: 4.7084

<div class="k-default-codeblock">
```

```
</div>
 1039/1094 ━━━━━━━━━━━━━━━━━━[37m━━  19s 359ms/step - d_loss: 0.2397 - g_loss: 4.7088

<div class="k-default-codeblock">
```

```
</div>
 1040/1094 ━━━━━━━━━━━━━━━━━━━[37m━  19s 359ms/step - d_loss: 0.2398 - g_loss: 4.7092

<div class="k-default-codeblock">
```

```
</div>
 1041/1094 ━━━━━━━━━━━━━━━━━━━[37m━  19s 359ms/step - d_loss: 0.2399 - g_loss: 4.7096

<div class="k-default-codeblock">
```

```
</div>
 1042/1094 ━━━━━━━━━━━━━━━━━━━[37m━  18s 359ms/step - d_loss: 0.2399 - g_loss: 4.7101

<div class="k-default-codeblock">
```

```
</div>
 1043/1094 ━━━━━━━━━━━━━━━━━━━[37m━  18s 359ms/step - d_loss: 0.2400 - g_loss: 4.7105

<div class="k-default-codeblock">
```

```
</div>
 1044/1094 ━━━━━━━━━━━━━━━━━━━[37m━  17s 359ms/step - d_loss: 0.2401 - g_loss: 4.7109

<div class="k-default-codeblock">
```

```
</div>
 1045/1094 ━━━━━━━━━━━━━━━━━━━[37m━  17s 359ms/step - d_loss: 0.2401 - g_loss: 4.7113

<div class="k-default-codeblock">
```

```
</div>
 1046/1094 ━━━━━━━━━━━━━━━━━━━[37m━  17s 359ms/step - d_loss: 0.2402 - g_loss: 4.7116

<div class="k-default-codeblock">
```

```
</div>
 1047/1094 ━━━━━━━━━━━━━━━━━━━[37m━  16s 359ms/step - d_loss: 0.2403 - g_loss: 4.7120

<div class="k-default-codeblock">
```

```
</div>
 1048/1094 ━━━━━━━━━━━━━━━━━━━[37m━  16s 359ms/step - d_loss: 0.2403 - g_loss: 4.7124

<div class="k-default-codeblock">
```

```
</div>
 1049/1094 ━━━━━━━━━━━━━━━━━━━[37m━  16s 359ms/step - d_loss: 0.2404 - g_loss: 4.7128

<div class="k-default-codeblock">
```

```
</div>
 1050/1094 ━━━━━━━━━━━━━━━━━━━[37m━  15s 359ms/step - d_loss: 0.2404 - g_loss: 4.7132

<div class="k-default-codeblock">
```

```
</div>
 1051/1094 ━━━━━━━━━━━━━━━━━━━[37m━  15s 359ms/step - d_loss: 0.2405 - g_loss: 4.7135

<div class="k-default-codeblock">
```

```
</div>
 1052/1094 ━━━━━━━━━━━━━━━━━━━[37m━  15s 359ms/step - d_loss: 0.2406 - g_loss: 4.7139

<div class="k-default-codeblock">
```

```
</div>
 1053/1094 ━━━━━━━━━━━━━━━━━━━[37m━  14s 359ms/step - d_loss: 0.2406 - g_loss: 4.7143

<div class="k-default-codeblock">
```

```
</div>
 1054/1094 ━━━━━━━━━━━━━━━━━━━[37m━  14s 359ms/step - d_loss: 0.2407 - g_loss: 4.7146

<div class="k-default-codeblock">
```

```
</div>
 1055/1094 ━━━━━━━━━━━━━━━━━━━[37m━  13s 359ms/step - d_loss: 0.2408 - g_loss: 4.7150

<div class="k-default-codeblock">
```

```
</div>
 1056/1094 ━━━━━━━━━━━━━━━━━━━[37m━  13s 359ms/step - d_loss: 0.2409 - g_loss: 4.7153

<div class="k-default-codeblock">
```

```
</div>
 1057/1094 ━━━━━━━━━━━━━━━━━━━[37m━  13s 359ms/step - d_loss: 0.2409 - g_loss: 4.7157

<div class="k-default-codeblock">
```

```
</div>
 1058/1094 ━━━━━━━━━━━━━━━━━━━[37m━  12s 358ms/step - d_loss: 0.2410 - g_loss: 4.7160

<div class="k-default-codeblock">
```

```
</div>
 1059/1094 ━━━━━━━━━━━━━━━━━━━[37m━  12s 358ms/step - d_loss: 0.2411 - g_loss: 4.7163

<div class="k-default-codeblock">
```

```
</div>
 1060/1094 ━━━━━━━━━━━━━━━━━━━[37m━  12s 358ms/step - d_loss: 0.2411 - g_loss: 4.7167

<div class="k-default-codeblock">
```

```
</div>
 1061/1094 ━━━━━━━━━━━━━━━━━━━[37m━  11s 358ms/step - d_loss: 0.2412 - g_loss: 4.7170

<div class="k-default-codeblock">
```

```
</div>
 1062/1094 ━━━━━━━━━━━━━━━━━━━[37m━  11s 358ms/step - d_loss: 0.2413 - g_loss: 4.7173

<div class="k-default-codeblock">
```

```
</div>
 1063/1094 ━━━━━━━━━━━━━━━━━━━[37m━  11s 358ms/step - d_loss: 0.2413 - g_loss: 4.7177

<div class="k-default-codeblock">
```

```
</div>
 1064/1094 ━━━━━━━━━━━━━━━━━━━[37m━  10s 358ms/step - d_loss: 0.2414 - g_loss: 4.7180

<div class="k-default-codeblock">
```

```
</div>
 1065/1094 ━━━━━━━━━━━━━━━━━━━[37m━  10s 358ms/step - d_loss: 0.2415 - g_loss: 4.7183

<div class="k-default-codeblock">
```

```
</div>
 1066/1094 ━━━━━━━━━━━━━━━━━━━[37m━  10s 358ms/step - d_loss: 0.2415 - g_loss: 4.7186

<div class="k-default-codeblock">
```

```
</div>
 1067/1094 ━━━━━━━━━━━━━━━━━━━[37m━  9s 358ms/step - d_loss: 0.2416 - g_loss: 4.7189 

<div class="k-default-codeblock">
```

```
</div>
 1068/1094 ━━━━━━━━━━━━━━━━━━━[37m━  9s 358ms/step - d_loss: 0.2417 - g_loss: 4.7192

<div class="k-default-codeblock">
```

```
</div>
 1069/1094 ━━━━━━━━━━━━━━━━━━━[37m━  8s 359ms/step - d_loss: 0.2418 - g_loss: 4.7195

<div class="k-default-codeblock">
```

```
</div>
 1070/1094 ━━━━━━━━━━━━━━━━━━━[37m━  8s 359ms/step - d_loss: 0.2418 - g_loss: 4.7198

<div class="k-default-codeblock">
```

```
</div>
 1071/1094 ━━━━━━━━━━━━━━━━━━━[37m━  8s 359ms/step - d_loss: 0.2419 - g_loss: 4.7201

<div class="k-default-codeblock">
```

```
</div>
 1072/1094 ━━━━━━━━━━━━━━━━━━━[37m━  7s 359ms/step - d_loss: 0.2420 - g_loss: 4.7203

<div class="k-default-codeblock">
```

```
</div>
 1073/1094 ━━━━━━━━━━━━━━━━━━━[37m━  7s 359ms/step - d_loss: 0.2420 - g_loss: 4.7206

<div class="k-default-codeblock">
```

```
</div>
 1074/1094 ━━━━━━━━━━━━━━━━━━━[37m━  7s 359ms/step - d_loss: 0.2421 - g_loss: 4.7209

<div class="k-default-codeblock">
```

```
</div>
 1075/1094 ━━━━━━━━━━━━━━━━━━━[37m━  6s 359ms/step - d_loss: 0.2422 - g_loss: 4.7212

<div class="k-default-codeblock">
```

```
</div>
 1076/1094 ━━━━━━━━━━━━━━━━━━━[37m━  6s 359ms/step - d_loss: 0.2422 - g_loss: 4.7214

<div class="k-default-codeblock">
```

```
</div>
 1077/1094 ━━━━━━━━━━━━━━━━━━━[37m━  6s 359ms/step - d_loss: 0.2423 - g_loss: 4.7217

<div class="k-default-codeblock">
```

```
</div>
 1078/1094 ━━━━━━━━━━━━━━━━━━━[37m━  5s 359ms/step - d_loss: 0.2424 - g_loss: 4.7220

<div class="k-default-codeblock">
```

```
</div>
 1079/1094 ━━━━━━━━━━━━━━━━━━━[37m━  5s 359ms/step - d_loss: 0.2425 - g_loss: 4.7222

<div class="k-default-codeblock">
```

```
</div>
 1080/1094 ━━━━━━━━━━━━━━━━━━━[37m━  5s 359ms/step - d_loss: 0.2425 - g_loss: 4.7225

<div class="k-default-codeblock">
```

```
</div>
 1081/1094 ━━━━━━━━━━━━━━━━━━━[37m━  4s 359ms/step - d_loss: 0.2426 - g_loss: 4.7227

<div class="k-default-codeblock">
```

```
</div>
 1082/1094 ━━━━━━━━━━━━━━━━━━━[37m━  4s 360ms/step - d_loss: 0.2427 - g_loss: 4.7230

<div class="k-default-codeblock">
```

```
</div>
 1083/1094 ━━━━━━━━━━━━━━━━━━━[37m━  3s 360ms/step - d_loss: 0.2427 - g_loss: 4.7232

<div class="k-default-codeblock">
```

```
</div>
 1084/1094 ━━━━━━━━━━━━━━━━━━━[37m━  3s 360ms/step - d_loss: 0.2428 - g_loss: 4.7235

<div class="k-default-codeblock">
```

```
</div>
 1085/1094 ━━━━━━━━━━━━━━━━━━━[37m━  3s 360ms/step - d_loss: 0.2429 - g_loss: 4.7237

<div class="k-default-codeblock">
```

```
</div>
 1086/1094 ━━━━━━━━━━━━━━━━━━━[37m━  2s 360ms/step - d_loss: 0.2430 - g_loss: 4.7239

<div class="k-default-codeblock">
```

```
</div>
 1087/1094 ━━━━━━━━━━━━━━━━━━━[37m━  2s 360ms/step - d_loss: 0.2430 - g_loss: 4.7242

<div class="k-default-codeblock">
```

```
</div>
 1088/1094 ━━━━━━━━━━━━━━━━━━━[37m━  2s 360ms/step - d_loss: 0.2431 - g_loss: 4.7244

<div class="k-default-codeblock">
```

```
</div>
 1089/1094 ━━━━━━━━━━━━━━━━━━━[37m━  1s 360ms/step - d_loss: 0.2432 - g_loss: 4.7246

<div class="k-default-codeblock">
```

```
</div>
 1090/1094 ━━━━━━━━━━━━━━━━━━━[37m━  1s 360ms/step - d_loss: 0.2432 - g_loss: 4.7248

<div class="k-default-codeblock">
```

```
</div>
 1091/1094 ━━━━━━━━━━━━━━━━━━━[37m━  1s 360ms/step - d_loss: 0.2433 - g_loss: 4.7251

<div class="k-default-codeblock">
```

```
</div>
 1092/1094 ━━━━━━━━━━━━━━━━━━━[37m━  0s 360ms/step - d_loss: 0.2434 - g_loss: 4.7253

<div class="k-default-codeblock">
```

```
</div>
 1093/1094 ━━━━━━━━━━━━━━━━━━━[37m━  0s 360ms/step - d_loss: 0.2435 - g_loss: 4.7255

<div class="k-default-codeblock">
```

```
</div>
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 0s 360ms/step - d_loss: 0.2435 - g_loss: 4.7257

<div class="k-default-codeblock">
```

```
</div>
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 394s 360ms/step - d_loss: 0.2436 - g_loss: 4.7259





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f489760a490>

```
</div>
The ideas behind deep learning are simple, so why should their implementation be painful?

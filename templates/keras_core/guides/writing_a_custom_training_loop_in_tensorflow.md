# Writing a training loop from scratch in TensorFlow

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2019/03/01<br>
**Last modified:** 2023/06/25<br>
**Description:** Writing low-level training & evaluation loops in TensorFlow.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_core/writing_a_custom_training_loop_in_tensorflow.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_core/writing_a_custom_training_loop_in_tensorflow.py)



---
## Setup


```python
import time
import os

# This guide can only be run with the TensorFlow backend.
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras_core as keras
import numpy as np
```

<div class="k-default-codeblock">
```
Using TensorFlow backend

```
</div>
---
## Introduction

Keras provides default training and evaluation loops, `fit()` and `evaluate()`.
Their usage is covered in the guide
[Training & evaluation with the built-in methods](/keras_core/guides/training_with_built_in_methods/).

If you want to customize the learning algorithm of your model while still leveraging
the convenience of `fit()`
(for instance, to train a GAN using `fit()`), you can subclass the `Model` class and
implement your own `train_step()` method, which
is called repeatedly during `fit()`.

Now, if you want very low-level control over training & evaluation, you should write
your own training & evaluation loops from scratch. This is what this guide is about.

---
## A first end-to-end example

Let's consider a simple MNIST model:


```python

def get_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = get_model()
```

Let's train it using mini-batch gradient with a custom training loop.

First, we're going to need an optimizer, a loss function, and a dataset:


```python
# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the training dataset.
batch_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)
```

Calling a model inside a `GradientTape` scope enables you to retrieve the gradients of
the trainable weights of the layer with respect to a loss value. Using an optimizer
instance, you can use these gradients to update these variables (which you can
retrieve using `model.trainable_weights`).

Here's our training loop, step by step:

- We open a `for` loop that iterates over epochs
- For each epoch, we open a `for` loop that iterates over the dataset, in batches
- For each batch, we open a `GradientTape()` scope
- Inside this scope, we call the model (forward pass) and compute the loss
- Outside the scope, we retrieve the gradients of the weights
of the model with regard to the loss
- Finally, we use the optimizer to update the weights of the model based on the
gradients


```python
epochs = 3
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply(grads, model.trainable_weights)

        # Log every 100 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")
```

    
<div class="k-default-codeblock">
```
Start of epoch 0
Training loss (for 1 batch) at step 0: 106.0587
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 5.1022
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 2.9819
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 1.8298
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.8804
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 1.2183
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.6585
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.7860
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 2.0385
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.3691
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.5903
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.8463
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.3598
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.9601
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.4314
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.5752
Seen so far: 48032 samples
```
</div>
    
<div class="k-default-codeblock">
```
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.4125
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.5126
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.1091
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.2702
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.3086
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.4386
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.3783
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.3026
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.7587
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.5452
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.2533
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.6167
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.3133
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.2733
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.3427
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.2213
Seen so far: 48032 samples
```
</div>
    
<div class="k-default-codeblock">
```
Start of epoch 2
Training loss (for 1 batch) at step 0: 0.3657
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.0396
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.0965
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.3169
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.1414
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3255
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.3497
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1856
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.3831
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.3505
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.1679
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.1079
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.2311
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.0782
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.8396
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.3415
Seen so far: 48032 samples

```
</div>
---
## Low-level handling of metrics

Let's add metrics monitoring to this basic loop.

You can readily reuse the built-in metrics (or custom ones you wrote) in such training
loops written from scratch. Here's the flow:

- Instantiate the metric at the start of the loop
- Call `metric.update_state()` after each batch
- Call `metric.result()` when you need to display the current value of the metric
- Call `metric.reset_state()` when you need to clear the state of the metric
(typically at the end of an epoch)

Let's use this knowledge to compute `SparseCategoricalAccuracy` on training and
validation data at the end of each epoch:


```python
# Get a fresh model
model = get_model()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
```

Here's our training & evaluation loop:


```python
epochs = 2
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply(grads, model.trainable_weights)

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 100 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print(f"Training acc over epoch: {float(train_acc):.4f}")

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_state()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f"Validation acc: {float(val_acc):.4f}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
```

    
<div class="k-default-codeblock">
```
Start of epoch 0
Training loss (for 1 batch) at step 0: 133.2436
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 3.7417
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 1.3833
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 1.5943
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 2.8395
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 1.1180
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.7868
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.7968
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.7529
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.7973
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 1.1071
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.3704
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.8647
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.4324
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.5812
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.1011
Seen so far: 48032 samples
Training acc over epoch: 0.8216
Validation acc: 0.8976
Time taken: 11.40s
```
</div>
    
<div class="k-default-codeblock">
```
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.0387
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.6132
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.1871
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.0445
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.4673
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.0076
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.0518
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.3156
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.6133
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.6535
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.1136
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.4392
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.1285
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.8116
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.3707
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.1317
Seen so far: 48032 samples
Training acc over epoch: 0.9133
Validation acc: 0.9141
Time taken: 11.41s

```
</div>
---
## Speeding-up your training step with `tf.function`

The default runtime in TensorFlow is eager execution.
As such, our training loop above executes eagerly.

This is great for debugging, but graph compilation has a definite performance
advantage. Describing your computation as a static graph enables the framework
to apply global performance optimizations. This is impossible when
the framework is constrained to greedily execute one operation after another,
with no knowledge of what comes next.

You can compile into a static graph any function that takes tensors as input.
Just add a `@tf.function` decorator on it, like this:


```python

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)
    train_acc_metric.update_state(y, logits)
    return loss_value

```

Let's do the same with the evaluation step:


```python

@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

```

Now, let's re-run our training loop with this compiled training step:


```python
epochs = 2
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 100 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print(f"Training acc over epoch: {float(train_acc):.4f}")

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_state()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f"Validation acc: {float(val_acc):.4f}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
```

    
<div class="k-default-codeblock">
```
Start of epoch 0
Training loss (for 1 batch) at step 0: 0.1939
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.2028
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.1964
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.0649
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.7233
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3667
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.0549
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.2851
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.1821
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.3732
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.5364
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.0638
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.1033
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.3501
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.1024
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.2764
Seen so far: 48032 samples
Training acc over epoch: 0.9322
Validation acc: 0.9423
Time taken: 5.11s
```
</div>
    
<div class="k-default-codeblock">
```
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.0448
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.0965
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.5181
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.0812
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.1292
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3675
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.0257
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.2009
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.1045
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.7620
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.0285
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.0081
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.0230
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.1346
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.0264
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.2445
Seen so far: 48032 samples
Training acc over epoch: 0.9433
Validation acc: 0.9449
Time taken: 0.98s

```
</div>
Much faster, isn't it?

---
## Low-level handling of losses tracked by the model

Layers & models recursively track any losses created during the forward pass
by layers that call `self.add_loss(value)`. The resulting list of scalar loss
values are available via the property `model.losses`
at the end of the forward pass.

If you want to be using these loss components, you should sum them
and add them to the main loss in your training step.

Consider this layer, that creates an activity regularization loss:


```python

class ActivityRegularizationLayer(keras.layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs

```

Let's build a really simple model that uses it:


```python
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu")(inputs)
# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

Here's what our training step should look like now:


```python

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        # Add any extra losses created during the forward pass.
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)
    train_acc_metric.update_state(y, logits)
    return loss_value

```

---
## Summary

Now you know everything there is to know about using built-in training loops and
writing your own from scratch.

To conclude, here's a simple end-to-end example that ties together everything
you've learned in this guide: a DCGAN trained on MNIST digits.

---
## End-to-end example: a GAN training loop from scratch

You may be familiar with Generative Adversarial Networks (GANs). GANs can generate new
images that look almost real, by learning the latent distribution of a training
dataset of images (the "latent space" of the images).

A GAN is made of two parts: a "generator" model that maps points in the latent
space to points in image space, a "discriminator" model, a classifier
that can tell the difference between real images (from the training dataset)
and fake images (the output of the generator network).

A GAN training loop looks like this:

1) Train the discriminator.
- Sample a batch of random points in the latent space.
- Turn the points into fake images via the "generator" model.
- Get a batch of real images and combine them with the generated images.
- Train the "discriminator" model to classify generated vs. real images.

2) Train the generator.
- Sample random points in the latent space.
- Turn the points into fake images via the "generator" network.
- Get a batch of real images and combine them with the generated images.
- Train the "generator" model to "fool" the discriminator and classify the fake images
as real.

For a much more detailed overview of how GANs works, see
[Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python).

Let's implement this training loop. First, create the discriminator meant to classify
fake vs real digits:


```python
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dense(1),
    ],
    name="discriminator",
)
discriminator.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "discriminator"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │        <span style="color: #00af00; text-decoration-color: #00af00">640</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)         │     <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ global_max_pooling2d            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)               │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalMaxPooling2D</span>)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                 │        <span style="color: #00af00; text-decoration-color: #00af00">129</span> │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">74,625</span> (2.28 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">74,625</span> (2.28 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



Then let's create a generator network,
that turns latent vectors into outputs of shape `(28, 28, 1)` (representing
MNIST digits):


```python
latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        # We want to generate 128 coefficients to reshape into a 7x7x128 map
        keras.layers.Dense(7 * 7 * 128),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Reshape((7, 7, 128)),
        keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)
```

Here's the key bit: the training loop. As you can see it is quite straightforward. The
training step function only takes 17 lines.


```python
# Instantiate one optimizer for the discriminator and another for the generator.
d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0004)

# Instantiate a loss function.
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step(real_images):
    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    # Decode them to fake images
    generated_images = generator(random_latent_vectors)
    # Combine them with real images
    combined_images = tf.concat([generated_images, real_images], axis=0)

    # Assemble labels discriminating real from fake images
    labels = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((real_images.shape[0], 1))], axis=0
    )
    # Add random noise to the labels - important trick!
    labels += 0.05 * tf.random.uniform(labels.shape)

    # Train the discriminator
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply(grads, discriminator.trainable_weights)

    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    # Assemble labels that say "all real images"
    misleading_labels = tf.zeros((batch_size, 1))

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(random_latent_vectors))
        g_loss = loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply(grads, generator.trainable_weights)
    return d_loss, g_loss, generated_images

```

Let's train our GAN, by repeatedly calling `train_step` on batches of images.

Since our discriminator and generator are convnets, you're going to want to
run this code on a GPU.


```python
# Prepare the dataset. We use both the training & test MNIST digits.
batch_size = 64
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

epochs = 1  # In practice you need at least 20 epochs to generate nice digits.
save_dir = "./"

for epoch in range(epochs):
    print(f"\nStart epoch {epoch}")

    for step, real_images in enumerate(dataset):
        # Train the discriminator & generator on one batch of real images.
        d_loss, g_loss, generated_images = train_step(real_images)

        # Logging.
        if step % 100 == 0:
            # Print metrics
            print(f"discriminator loss at step {step}: {d_loss:.2f}")
            print(f"adversarial loss at step {step}: {g_loss:.2f}")

            # Save one generated image
            img = keras.utils.array_to_img(generated_images[0] * 255.0, scale=False)
            img.save(os.path.join(save_dir, f"generated_img_{step}.png"))

        # To limit execution time we stop after 10 steps.
        # Remove the lines below to actually train the model!
        if step > 10:
            break
```

    
<div class="k-default-codeblock">
```
Start epoch 0
discriminator loss at step 0: 0.70
adversarial loss at step 0: 0.67

```
</div>
That's it! You'll get nice-looking fake MNIST digits after just ~30s of training on the
Colab GPU.

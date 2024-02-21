# Writing a training loop from scratch in PyTorch

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2023/06/25<br>
**Last modified:** 2023/06/25<br>
**Description:** Writing low-level training & evaluation loops in PyTorch.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/writing_a_custom_training_loop_in_torch.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/writing_a_custom_training_loop_in_torch.py)



---
## Setup


```python
import os

# This guide can only be run with the torch backend.
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
import numpy as np
```

---
## Introduction

Keras provides default training and evaluation loops, `fit()` and `evaluate()`.
Their usage is covered in the guide
[Training & evaluation with the built-in methods](/guides/training_with_built_in_methods/).

If you want to customize the learning algorithm of your model while still leveraging
the convenience of `fit()`
(for instance, to train a GAN using `fit()`), you can subclass the `Model` class and
implement your own `train_step()` method, which
is called repeatedly during `fit()`.

Now, if you want very low-level control over training & evaluation, you should write
your own training & evaluation loops from scratch. This is what this guide is about.

---
## A first end-to-end example

To write a custom training loop, we need the following ingredients:

- A model to train, of course.
- An optimizer. You could either use a `keras.optimizers` optimizer,
or a native PyTorch optimizer from `torch.optim`.
- A loss function. You could either use a `keras.losses` loss,
or a native PyTorch loss from `torch.nn`.
- A dataset. You could use any format: a `tf.data.Dataset`,
a PyTorch `DataLoader`, a Python generator, etc.

Let's line them up. We'll use torch-native objects in each case --
except, of course, for the Keras model.

First, let's get the model and the MNIST dataset:


```python

# Let's consider a simple MNIST model
def get_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Create load up the MNIST dataset and put it in a torch DataLoader
# Prepare the training dataset.
batch_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)).astype("float32")
x_test = np.reshape(x_test, (-1, 784)).astype("float32")
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Create torch Datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_train), torch.from_numpy(y_train)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_val), torch.from_numpy(y_val)
)

# Create DataLoaders for the Datasets
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
```

Next, here's our PyTorch optimizer and our PyTorch loss function:


```python
# Instantiate a torch optimizer
model = get_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Instantiate a torch loss function
loss_fn = torch.nn.CrossEntropyLoss()
```

Let's train our model using mini-batch gradient with a custom training loop.

Calling `loss.backward()` on a loss tensor triggers backpropagation.
Once that's done, your optimizer is magically aware of the gradients for each variable
and can update its variables, which is done via `optimizer.step()`.
Tensors, variables, optimizers are all interconnected to one another via hidden global state.
Also, don't forget to call `model.zero_grad()` before `loss.backward()`, or you won't
get the right gradients for your variables.

Here's our training loop, step by step:

- We open a `for` loop that iterates over epochs
- For each epoch, we open a `for` loop that iterates over the dataset, in batches
- For each batch, we call the model on the input data to retrieve the predictions,
then we use them to compute a loss value
- We call `loss.backward()` to
- Outside the scope, we retrieve the gradients of the weights
of the model with regard to the loss
- Finally, we use the optimizer to update the weights of the model based on the
gradients


```python
epochs = 3
for epoch in range(epochs):
    for step, (inputs, targets) in enumerate(train_dataloader):
        # Forward pass
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Optimizer variable updates
        optimizer.step()

        # Log every 100 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")
```

<div class="k-default-codeblock">
```
Training loss (for 1 batch) at step 0: 110.9115
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 2.9493
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 2.7383
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 1.6616
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 1.5927
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 1.0992
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.5425
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.3308
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.8231
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.5570
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.6321
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.4962
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 1.0833
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 1.3607
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 1.1250
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 1.2562
Seen so far: 48032 samples
Training loss (for 1 batch) at step 0: 0.5181
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.3939
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.3406
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.1122
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.2015
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.1184
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 1.0702
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.4062
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.4570
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 1.2490
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.0714
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.3677
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.8291
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.8320
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.1179
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.5390
Seen so far: 48032 samples
Training loss (for 1 batch) at step 0: 0.1309
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.4061
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.2734
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.2972
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.4282
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3504
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.3556
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.7834
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.2522
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.2056
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.3259
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.5215
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.8051
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.4423
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.0473
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.1419
Seen so far: 48032 samples

```
</div>
As an alternative, let's look at what the loop looks like when using a Keras optimizer
and a Keras loss function.

Important differences:

- You retrieve the gradients for the variables via `v.value.grad`,
called on each trainable variable.
- You update your variables via `optimizer.apply()`, which must be
called in a `torch.no_grad()` scope.

**Also, a big gotcha:** while all NumPy/TensorFlow/JAX/Keras APIs
as well as Python `unittest` APIs use the argument order convention
`fn(y_true, y_pred)` (reference values first, predicted values second),
PyTorch actually uses `fn(y_pred, y_true)` for its losses.
So make sure to invert the order of `logits` and `targets`.


```python
model = get_model()
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    for step, (inputs, targets) in enumerate(train_dataloader):
        # Forward pass
        logits = model(inputs)
        loss = loss_fn(targets, logits)

        # Backward pass
        model.zero_grad()
        trainable_weights = [v for v in model.trainable_weights]

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            optimizer.apply(gradients, trainable_weights)

        # Log every 100 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")
```

    
<div class="k-default-codeblock">
```
Start of epoch 0
Training loss (for 1 batch) at step 0: 98.9569
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 5.3304
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.3246
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 1.6745
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 1.0936
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 1.4159
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.2796
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 2.3532
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.7533
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 1.0432
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.3959
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.4722
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.3851
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.8599
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.1237
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.4919
Seen so far: 48032 samples
```
</div>
    
<div class="k-default-codeblock">
```
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.8972
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.5844
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.1285
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.0671
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.4296
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.1483
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.0230
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1368
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.1531
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.0472
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.2343
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.4449
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.3942
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.3236
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.0717
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.9288
Seen so far: 48032 samples
```
</div>
    
<div class="k-default-codeblock">
```
Start of epoch 2
Training loss (for 1 batch) at step 0: 0.9393
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.2383
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.1116
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.6736
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.6713
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3394
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.2385
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.4248
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.0200
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.1259
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.7566
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.0594
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.2821
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.2088
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.5654
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.0512
Seen so far: 48032 samples

```
</div>
---
## Low-level handling of metrics

Let's add metrics monitoring to this basic training loop.

You can readily reuse built-in Keras metrics (or custom ones you wrote) in such training
loops written from scratch. Here's the flow:

- Instantiate the metric at the start of the loop
- Call `metric.update_state()` after each batch
- Call `metric.result()` when you need to display the current value of the metric
- Call `metric.reset_state()` when you need to clear the state of the metric
(typically at the end of an epoch)

Let's use this knowledge to compute `CategoricalAccuracy` on training and
validation data at the end of each epoch:


```python
# Get a fresh model
model = get_model()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()
```

Here's our training & evaluation loop:


```python
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    for step, (inputs, targets) in enumerate(train_dataloader):
        # Forward pass
        logits = model(inputs)
        loss = loss_fn(targets, logits)

        # Backward pass
        model.zero_grad()
        trainable_weights = [v for v in model.trainable_weights]

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            optimizer.apply(gradients, trainable_weights)

        # Update training metric.
        train_acc_metric.update_state(targets, logits)

        # Log every 100 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print(f"Training acc over epoch: {float(train_acc):.4f}")

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_state()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataloader:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f"Validation acc: {float(val_acc):.4f}")

```

    
<div class="k-default-codeblock">
```
Start of epoch 0
Training loss (for 1 batch) at step 0: 59.2206
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 8.9801
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 5.2990
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 3.6978
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 1.9965
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 2.1896
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 1.2416
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.9403
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.1838
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.5884
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.7836
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.7015
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.3335
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.2763
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.4787
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.2562
Seen so far: 48032 samples
Training acc over epoch: 0.8411
Validation acc: 0.8963
```
</div>
    
<div class="k-default-codeblock">
```
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.3417
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 1.1465
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.7274
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.1273
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.6500
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.2008
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.7483
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.5821
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.5696
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.3112
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.1761
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.1811
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.2736
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.3848
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.4627
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.3934
Seen so far: 48032 samples
Training acc over epoch: 0.9053
Validation acc: 0.9221
```
</div>
    
<div class="k-default-codeblock">
```
Start of epoch 2
Training loss (for 1 batch) at step 0: 0.5743
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.4448
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.9880
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.2268
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.5607
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.1178
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.4305
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1712
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.3109
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.1548
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.1090
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.5169
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.3791
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.6963
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.6204
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.1111
Seen so far: 48032 samples
Training acc over epoch: 0.9216
Validation acc: 0.9356

```
</div>
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
        self.add_loss(1e-2 * torch.sum(inputs))
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

Here's what our training loop should look like now:


```python
# Get a fresh model
model = get_model()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    for step, (inputs, targets) in enumerate(train_dataloader):
        # Forward pass
        logits = model(inputs)
        loss = loss_fn(targets, logits)
        if model.losses:
            loss = loss + torch.sum(*model.losses)

        # Backward pass
        model.zero_grad()
        trainable_weights = [v for v in model.trainable_weights]

        # Call torch.Tensor.backward() on the loss to compute gradients
        # for the weights.
        loss.backward()
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            optimizer.apply(gradients, trainable_weights)

        # Update training metric.
        train_acc_metric.update_state(targets, logits)

        # Log every 100 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print(f"Training acc over epoch: {float(train_acc):.4f}")

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_state()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataloader:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f"Validation acc: {float(val_acc):.4f}")
```

    
<div class="k-default-codeblock">
```
Start of epoch 0
Training loss (for 1 batch) at step 0: 138.7979
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 4.4268
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 1.0779
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 1.7229
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.5801
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.4298
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.4717
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 1.3369
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 1.3239
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.5972
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.1983
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.5228
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 1.0025
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.3424
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.5196
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.4287
Seen so far: 48032 samples
Training acc over epoch: 0.8089
Validation acc: 0.8947
```
</div>
    
<div class="k-default-codeblock">
```
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.2903
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.4118
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.6533
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.0402
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.3638
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3313
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.5119
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1628
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.4793
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.2726
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.5721
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.5783
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.2533
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.2218
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.1232
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.6805
Seen so far: 48032 samples
Training acc over epoch: 0.8970
Validation acc: 0.9097
```
</div>
    
<div class="k-default-codeblock">
```
Start of epoch 2
Training loss (for 1 batch) at step 0: 0.4553
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.3975
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 1.2382
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.0927
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.3530
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3842
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.6423
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1751
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.4769
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.1854
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.3130
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.1633
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.1446
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.4661
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.9977
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.3392
Seen so far: 48032 samples
Training acc over epoch: 0.9182
Validation acc: 0.9200

```
</div>
That's it!

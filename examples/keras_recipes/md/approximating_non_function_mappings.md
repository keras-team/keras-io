# Approximating non-Function Mappings with Mixture Density Networks

**Author:** [lukewood](https://twitter.com/luke_wood_ml)<br>
**Date created:** 2023/07/15<br>
**Last modified:** 2023/07/15<br>
**Description:** Approximate non one to one mapping using mixture density networks.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/approximating_non_function_mappings.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/approximating_non_function_mappings.py)



---
## Approximating NonFunctions

Neural networks are universal function approximators. Key word: function!
While powerful function approximators, neural networks are not able to
approximate non-functions.
One important restriction to remember about functions - they have one input, one
output!
Neural networks suffer greatly when the training set has multiple values of Y for a single X.

In this guide I'll show you how to approximate the class of non-functions
consisting of mappings from `x -> y` such that multiple `y` may exist for a
given `x`.  We'll use a class of neural networks called
"Mixture Density Networks".

I'm going to use the new
[multibackend Keras Core project](https://github.com/keras-team/keras-core) to
build my Mixture Density networks.
Great job to the Keras team on the project - it's awesome to be able to swap
frameworks in one line of code.

Some bad news: I use TensorFlow probability in this guide... so it doesn't
actually work with other backends.

Anyways, let's start by installing dependencies and sorting out imports:


```python
!pip install -q --upgrade tensorflow-probability keras-core
```


```python
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import math
import random
from keras_core import callbacks
import keras_core
import tensorflow as tf
from keras_core import layers
from keras_core import optimizers
from tensorflow_probability import distributions as tfd
```

<div class="k-default-codeblock">
```
Using TensorFlow backend

```
</div>
Next, lets generate a noisy spiral that we're going to attempt to approximate.
I've defined a few functions below to do this:


```python

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def create_noisy_spiral(n, max_jitter=0.2, size_range=[5, 10]):
    R_X = random.choice(range(size_range[0], size_range[1], 10))
    R_Y = random.choice(range(size_range[0], size_range[1], 10))
    X_0 = random.random() * 100 - 50
    Y_0 = random.random() * 100 - 50

    randomized_jitter = lambda: random.uniform(-max_jitter / 2, max_jitter / 2)
    x = lambda _t: _t / 360.0 * math.cos(_t / 90.0 * math.pi) * R_X + X_0
    y = lambda _t: _t / 360.0 * math.sin(_t / 90.0 * math.pi) * R_Y + Y_0

    out = np.zeros([n, 2])

    for i in range(n):
        t = 360.0 / n * i
        out[i, 0] = x(t) + randomized_jitter()
        out[i, 1] = y(t) + randomized_jitter()
    out = out.astype("float32")
    return (normalize(out[:, 0]) * 10) - 5, (normalize(out[:, 1]) * 10) - 5

```

Next, lets invoke this function many times to construct a sample dataset:


```python
xs = []
ys = []

for _ in range(10):
    x, y = create_noisy_spiral(1000)
    xs.append(x)
    ys.append(y)

x = np.concatenate(xs, axis=0)
y = np.concatenate(ys, axis=0)
x = np.expand_dims(x, axis=1)

plt.scatter(x, y)
plt.show()
```


    
![png](/img/examples/keras_recipes/approximating_non_function_mappings/approximating_non_function_mappings_7_0.png)
    


As you can see, there's multiple possible values for Y with respect to a given
X.  Normal neural networks will simply learn the mean of these points with
respect to geometric space.

We can quickly show this with a simple linear model:


```python
N_HIDDEN = 128

model = keras_core.Sequential(
    [
        layers.Dense(N_HIDDEN, activation="relu"),
        layers.Dense(N_HIDDEN, activation="relu"),
        layers.Dense(1),
    ]
)
```

Let's use mean squared error as well as the adam optimizer.
These tend to be reasonable prototyping choices:


```python
model.compile(optimizer="adam", loss="mse")
```

We can fit this model quite easy


```python
model.fit(
    x,
    y,
    epochs=300,
    batch_size=128,
    validation_split=0.15,
    callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=5)],
)
```

<div class="k-default-codeblock">
```
Epoch 1/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - loss: 6.2038 - val_loss: 8.2260
Epoch 2/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.0679 - val_loss: 8.2122
Epoch 3/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.2306 - val_loss: 8.1716
Epoch 4/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.1781 - val_loss: 8.1718
Epoch 5/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.1569 - val_loss: 8.1571
Epoch 6/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.1782 - val_loss: 8.1454
Epoch 7/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 5.9934 - val_loss: 8.1431
Epoch 8/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.0931 - val_loss: 8.1498
Epoch 9/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.2061 - val_loss: 8.1457
Epoch 10/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.1596 - val_loss: 8.1160
Epoch 11/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 6.1648 - val_loss: 8.1139
Epoch 12/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.2059 - val_loss: 8.1539
Epoch 13/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.0393 - val_loss: 8.0939
Epoch 14/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.1600 - val_loss: 8.0973
Epoch 15/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.1924 - val_loss: 8.0831
Epoch 16/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.0318 - val_loss: 8.0885
Epoch 17/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.0827 - val_loss: 8.0533
Epoch 18/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.9681 - val_loss: 8.0655
Epoch 19/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.0547 - val_loss: 8.0054
Epoch 20/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.9178 - val_loss: 7.9885
Epoch 21/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 6.0278 - val_loss: 7.9938
Epoch 22/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.9816 - val_loss: 7.9390
Epoch 23/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.9373 - val_loss: 7.9237
Epoch 24/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.7925 - val_loss: 7.8570
Epoch 25/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.9036 - val_loss: 7.8400
Epoch 26/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.7948 - val_loss: 7.8329
Epoch 27/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.8374 - val_loss: 7.7440
Epoch 28/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.8218 - val_loss: 7.7199
Epoch 29/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.9213 - val_loss: 7.6937
Epoch 30/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.7941 - val_loss: 7.7333
Epoch 31/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.7686 - val_loss: 7.6799
Epoch 32/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.9021 - val_loss: 7.6881
Epoch 33/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.7143 - val_loss: 7.6437
Epoch 34/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.8997 - val_loss: 7.6737
Epoch 35/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.8537 - val_loss: 7.6614
Epoch 36/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.9942 - val_loss: 7.7032
Epoch 37/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.8474 - val_loss: 7.6531
Epoch 38/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 5.8344 - val_loss: 7.6350
Epoch 39/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.9014 - val_loss: 7.6356
Epoch 40/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.8538 - val_loss: 7.6340
Epoch 41/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.8875 - val_loss: 7.6895
Epoch 42/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.8900 - val_loss: 7.6233
Epoch 43/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.6214 - val_loss: 7.6156
Epoch 44/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.8410 - val_loss: 7.6537
Epoch 45/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.7603 - val_loss: 7.6313
Epoch 46/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.7771 - val_loss: 7.6212
Epoch 47/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.9544 - val_loss: 7.6183
Epoch 48/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 5.8080 - val_loss: 7.6256

<keras_core.src.callbacks.history.History at 0x13d1a82d0>

```
</div>
And let's check out the result:


```python
y_pred = model.predict(x)
```

<div class="k-default-codeblock">
```
 313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 744us/step

```
</div>
As expected, the model learns the geometric mean of all points in `y` for a
given `x`.


```python
plt.scatter(x, y)
plt.scatter(x, y_pred)
plt.show()
```


    
![png](/img/examples/keras_recipes/approximating_non_function_mappings/approximating_non_function_mappings_17_0.png)
    


---
## Mixture Density Networks

Mixture Density networks can alleviate this problem.
A Mixture density is a class of complicated densities expressible in terms of simpler densities.
They are effectively the sum of a ton of probability distributions.
Mixture Density networks learn to parameterize a mixture density distribution
based on a given training set.

As a practitioner, all you need to know, is that Mixture Density Networks solve
the problem of multiple values of Y for a given X.
I'm hoping to add a tool to your kit- but I'm not going to formally explain the
derivation of Mixture Density networks in this guide.
The most important thing to know is that a Mixture Density network learns to
parameterize a mixture density distribution.
This is done by computing a special loss with respect to both the provided
`y_i` label as well as the predicted distribution for the corresponding `x_i`.
This loss function operates by computing the probability that `y_i` would be
drawn from the predicted mixture distribution.

Let's implement a Mixture density network.
Below, a ton of helper functions are defined based on an old Keras library
[`Keras Mixture Density Network Layer`](https://github.com/cpmpercussion/keras-mdn-layer).

I've adapted the code for use with Keras core.

Lets start writing a Mixture Density Network!
First, we need a special activation function: ELU plus a tiny epsilon.
This helps prevent ELU from outputting 0 which causes NaNs in Mixture Density
Network loss evaluation.


```python

def elu_plus_one_plus_epsilon(x):
    return keras_core.activations.elu(x) + 1 + keras_core.backend.epsilon()

```

Next, lets actually define a MixtureDensity layer that outputs all values needed
to sample from the learned mixture distribution:


```python

class MixtureDensityOutput(layers.Layer):
    def __init__(self, output_dimension, num_mixtures, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.mdn_mus = layers.Dense(
            self.num_mix * self.output_dim, name="mdn_mus"
        )  # mix*output vals, no activation
        self.mdn_sigmas = layers.Dense(
            self.num_mix * self.output_dim,
            activation=elu_plus_one_plus_epsilon,
            name="mdn_sigmas",
        )  # mix*output vals exp activation
        self.mdn_pi = layers.Dense(self.num_mix, name="mdn_pi")  # mix vals, logits

    def build(self, input_shape):
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        super().build(input_shape)

    @property
    def trainable_weights(self):
        return (
            self.mdn_mus.trainable_weights
            + self.mdn_sigmas.trainable_weights
            + self.mdn_pi.trainable_weights
        )

    @property
    def non_trainable_weights(self):
        return (
            self.mdn_mus.non_trainable_weights
            + self.mdn_sigmas.non_trainable_weights
            + self.mdn_pi.non_trainable_weights
        )

    def call(self, x, mask=None):
        return layers.concatenate(
            [self.mdn_mus(x), self.mdn_sigmas(x), self.mdn_pi(x)], name="mdn_outputs"
        )

```

Lets construct an Mixture Density Network using our new layer:


```python
OUTPUT_DIMS = 1
N_MIXES = 20

mdn_network = keras_core.Sequential(
    [
        layers.Dense(N_HIDDEN, activation="relu"),
        layers.Dense(N_HIDDEN, activation="relu"),
        MixtureDensityOutput(OUTPUT_DIMS, N_MIXES),
    ]
)
```

Next, let's implement a custom loss function to train the Mixture Density
Network layer based on the true values and our expected outputs:


```python

def get_mixture_loss_func(output_dim, num_mixes):
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(
            y_pred,
            [-1, (2 * num_mixes * output_dim) + num_mixes],
            name="reshape_ypreds",
        )
        y_true = tf.reshape(y_true, [-1, output_dim], name="reshape_ytrue")
        # Split the inputs into paramaters
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[
                num_mixes * output_dim,
                num_mixes * output_dim,
                num_mixes,
            ],
            axis=-1,
            name="mdn_coef_split",
        )
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [
            tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            for loc, scale in zip(mus, sigs)
        ]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    return mdn_loss_func


mdn_network.compile(loss=get_mixture_loss_func(OUTPUT_DIMS, N_MIXES), optimizer="adam")
```

Finally, we can call `model.fit()` like any other Keras model.


```python
mdn_network.fit(
    x,
    y,
    epochs=300,
    batch_size=128,
    validation_split=0.15,
    callbacks=[
        callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="loss", patience=5),
    ],
)
```

<div class="k-default-codeblock">
```
Epoch 1/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 26s 140ms/step - loss: 3.0517 - val_loss: 2.6356 - learning_rate: 0.0010
Epoch 2/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 2.3725 - val_loss: 2.4995 - learning_rate: 0.0010
Epoch 3/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 2.2466 - val_loss: 2.3983 - learning_rate: 0.0010
Epoch 4/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 2.1122 - val_loss: 2.2618 - learning_rate: 0.0010
Epoch 5/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 2.0055 - val_loss: 2.0238 - learning_rate: 0.0010
Epoch 6/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.8034 - val_loss: 1.8412 - learning_rate: 0.0010
Epoch 7/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.6459 - val_loss: 1.7513 - learning_rate: 0.0010
Epoch 8/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.5698 - val_loss: 1.6769 - learning_rate: 0.0010
Epoch 9/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 1.3283 - val_loss: 1.2596 - learning_rate: 0.0010
Epoch 10/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 1.0068 - val_loss: 0.9509 - learning_rate: 0.0010
Epoch 11/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.8148 - val_loss: 1.0103 - learning_rate: 0.0010
Epoch 12/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.7056 - val_loss: 0.8431 - learning_rate: 0.0010
Epoch 13/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.6862 - val_loss: 0.7096 - learning_rate: 0.0010
Epoch 14/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.6253 - val_loss: 0.7228 - learning_rate: 0.0010
Epoch 15/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.5864 - val_loss: 0.7389 - learning_rate: 0.0010
Epoch 16/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.5722 - val_loss: 0.7374 - learning_rate: 0.0010
Epoch 17/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.5565 - val_loss: 0.6133 - learning_rate: 0.0010
Epoch 18/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.5375 - val_loss: 0.5929 - learning_rate: 0.0010
Epoch 19/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.5303 - val_loss: 0.6271 - learning_rate: 0.0010
Epoch 20/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.5067 - val_loss: 0.5417 - learning_rate: 0.0010
Epoch 21/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4540 - val_loss: 0.5955 - learning_rate: 0.0010
Epoch 22/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.4994 - val_loss: 0.6285 - learning_rate: 0.0010
Epoch 23/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4846 - val_loss: 0.5513 - learning_rate: 0.0010
Epoch 24/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4608 - val_loss: 0.5198 - learning_rate: 0.0010
Epoch 25/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.4388 - val_loss: 0.4908 - learning_rate: 0.0010
Epoch 26/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4272 - val_loss: 0.5387 - learning_rate: 0.0010
Epoch 27/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4339 - val_loss: 0.5094 - learning_rate: 0.0010
Epoch 28/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4413 - val_loss: 0.5499 - learning_rate: 0.0010
Epoch 29/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4422 - val_loss: 0.4884 - learning_rate: 0.0010
Epoch 30/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4075 - val_loss: 0.9174 - learning_rate: 0.0010
Epoch 31/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4841 - val_loss: 0.4696 - learning_rate: 0.0010
Epoch 32/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4174 - val_loss: 0.5822 - learning_rate: 0.0010
Epoch 33/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4008 - val_loss: 0.4692 - learning_rate: 0.0010
Epoch 34/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.4246 - val_loss: 0.5135 - learning_rate: 0.0010
Epoch 35/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3970 - val_loss: 0.5175 - learning_rate: 0.0010
Epoch 36/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4236 - val_loss: 0.4357 - learning_rate: 0.0010
Epoch 37/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.4179 - val_loss: 0.4693 - learning_rate: 0.0010
Epoch 38/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3936 - val_loss: 0.4756 - learning_rate: 0.0010
Epoch 39/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3701 - val_loss: 0.4022 - learning_rate: 1.0000e-04
Epoch 40/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3324 - val_loss: 0.3937 - learning_rate: 1.0000e-04
Epoch 41/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3434 - val_loss: 0.3959 - learning_rate: 1.0000e-04
Epoch 42/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3252 - val_loss: 0.4105 - learning_rate: 1.0000e-04
Epoch 43/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.3187 - val_loss: 0.4037 - learning_rate: 1.0000e-04
Epoch 44/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3333 - val_loss: 0.3950 - learning_rate: 1.0000e-04
Epoch 45/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3276 - val_loss: 0.3898 - learning_rate: 1.0000e-04
Epoch 46/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3342 - val_loss: 0.4029 - learning_rate: 1.0000e-04
Epoch 47/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.3302 - val_loss: 0.3911 - learning_rate: 1.0000e-04
Epoch 48/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3431 - val_loss: 0.4003 - learning_rate: 1.0000e-04
Epoch 49/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3391 - val_loss: 0.4032 - learning_rate: 1.0000e-04
Epoch 50/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.3381 - val_loss: 0.3902 - learning_rate: 1.0000e-04
Epoch 51/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3323 - val_loss: 0.3978 - learning_rate: 1.0000e-04
Epoch 52/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3207 - val_loss: 0.3938 - learning_rate: 1.0000e-04
Epoch 53/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3300 - val_loss: 0.3897 - learning_rate: 1.0000e-04
Epoch 54/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3400 - val_loss: 0.3928 - learning_rate: 1.0000e-04
Epoch 55/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3372 - val_loss: 0.3968 - learning_rate: 1.0000e-04
Epoch 56/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3331 - val_loss: 0.3891 - learning_rate: 1.0000e-04
Epoch 57/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3137 - val_loss: 0.3930 - learning_rate: 1.0000e-04
Epoch 58/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3229 - val_loss: 0.3882 - learning_rate: 1.0000e-04
Epoch 59/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3280 - val_loss: 0.3927 - learning_rate: 1.0000e-04
Epoch 60/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3264 - val_loss: 0.3830 - learning_rate: 1.0000e-05
Epoch 61/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.3108 - val_loss: 0.3838 - learning_rate: 1.0000e-05
Epoch 62/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3074 - val_loss: 0.3855 - learning_rate: 1.0000e-05
Epoch 63/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3208 - val_loss: 0.3834 - learning_rate: 1.0000e-05
Epoch 64/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3158 - val_loss: 0.3825 - learning_rate: 1.0000e-05
Epoch 65/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3214 - val_loss: 0.3862 - learning_rate: 1.0000e-05
Epoch 66/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.3026 - val_loss: 0.3854 - learning_rate: 1.0000e-05
Epoch 67/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3193 - val_loss: 0.3849 - learning_rate: 1.0000e-06
Epoch 68/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3102 - val_loss: 0.3847 - learning_rate: 1.0000e-06
Epoch 69/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3178 - val_loss: 0.3845 - learning_rate: 1.0000e-06
Epoch 70/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3177 - val_loss: 0.3845 - learning_rate: 1.0000e-06
Epoch 71/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.3179 - val_loss: 0.3846 - learning_rate: 1.0000e-06
Epoch 72/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3105 - val_loss: 0.3843 - learning_rate: 1.0000e-06
Epoch 73/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3186 - val_loss: 0.3844 - learning_rate: 1.0000e-06
Epoch 74/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3110 - val_loss: 0.3843 - learning_rate: 1.0000e-06
Epoch 75/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.3146 - val_loss: 0.3842 - learning_rate: 1.0000e-06
Epoch 76/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3241 - val_loss: 0.3843 - learning_rate: 1.0000e-07
Epoch 77/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3172 - val_loss: 0.3843 - learning_rate: 1.0000e-07
Epoch 78/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3256 - val_loss: 0.3843 - learning_rate: 1.0000e-07
Epoch 79/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.3232 - val_loss: 0.3843 - learning_rate: 1.0000e-07
Epoch 80/300
 67/67 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.3235 - val_loss: 0.3843 - learning_rate: 1.0000e-07

<keras_core.src.callbacks.history.History at 0x14141db10>

```
</div>
Let's make some predictions!


```python
y_pred_mixture = mdn_network.predict(x)
print(y_pred_mixture.shape)
```

<div class="k-default-codeblock">
```
 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step
(10000, 60)

```
</div>
The MDN does not output a single value; instead it outputs values to
parameterize a mixture distribution.
To visualize these outputs, lets sample from the distribution.

Note that sampling is a lossy process.
If you want to preserve all information as part of a greater latent
representation (i.e. for downstream processing) I recommend you simply keep the
distribution parameters in place.


```python

def split_mixture_params(params, output_dim, num_mixes):
    mus = params[: num_mixes * output_dim]
    sigs = params[num_mixes * output_dim : 2 * num_mixes * output_dim]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits


def softmax(w, t=1.0):
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist


def sample_from_categorical(dist):
    r = np.random.rand(1)  # uniform random number in [0,1]
    accumulate = 0
    for i in range(0, dist.size):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    tf.logging.info("Error sampling categorical model.")
    return -1


def sample_from_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim : (m + 1) * output_dim]
    sig_vector = sigs[m * output_dim : (m + 1) * output_dim]
    scale_matrix = np.identity(output_dim) * sig_vector  # scale matrix from diag
    cov_matrix = np.matmul(scale_matrix, scale_matrix.T)  # cov is scale squared.
    cov_matrix = cov_matrix * sigma_temp  # adjust for sigma temperature
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample

```

Next lets use our sampling function:


```python
# Sample from the predicted distributions
y_samples = np.apply_along_axis(
    sample_from_output, 1, y_pred_mixture, 1, N_MIXES, temp=1.0
)
```

Finally, we can visualize our network outputs


```python
plt.scatter(x, y, alpha=0.05, color="blue", label="Ground Truth")
plt.scatter(
    x,
    y_samples[:, :, 0],
    color="green",
    alpha=0.05,
    label="Mixture Density Network prediction",
)
plt.show()
```


    
![png](/img/examples/keras_recipes/approximating_non_function_mappings/approximating_non_function_mappings_35_0.png)
    


Beautiful.  Love to see it

# Conclusions

Neural Networks are universal function approximators - but they can only
approximate functions.  Mixture Density networks can approximate arbitrary
x->y mappings using some neat probability tricks.

One more pretty graphic for the road:


```python
fig, axs = plt.subplots(1, 3)
fig.set_figheight(3)
fig.set_figwidth(12)
axs[0].set_title("Ground Truth")
axs[0].scatter(x, y, alpha=0.05, color="blue")
axs[1].set_title("Normal Model prediction")
axs[1].scatter(x, y_pred, alpha=0.05, color="red")
axs[2].scatter(
    x,
    y_samples[:, :, 0],
    color="green",
    alpha=0.05,
    label="Mixture Density Network prediction",
)
axs[2].set_title("Mixture Density Network prediction")
plt.show()
```


    
![png](/img/examples/keras_recipes/approximating_non_function_mappings/approximating_non_function_mappings_37_0.png)
    


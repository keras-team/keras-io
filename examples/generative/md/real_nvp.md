# Density estimation using Real NVP

**Authors:** [Mandolini Giorgio Maria](https://www.linkedin.com/in/giorgio-maria-mandolini-a2a1b71b4/), [Sanna Daniele](https://www.linkedin.com/in/daniele-sanna-338629bb/), [Zannini Quirini Giorgio](https://www.linkedin.com/in/giorgio-zannini-quirini-16ab181a0/)<br>
**Date created:** 2020/08/10<br>
**Last modified:** 2020/08/10<br>
**Description:** Estimating the density distribution of the "double moon" dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/real_nvp.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/real_nvp.py)



---
## Introduction

The aim of this work is to map a simple distribution - which is easy to sample
and whose density is simple to estimate - to a more complex one learned from the data.
This kind of generative model is also known as "normalizing flow".

In order to do this, the model is trained via the maximum
likelihood principle, using the "change of variable" formula.

We will use an affine coupling function. We create it such that its inverse, as well as
the determinant of the Jacobian, are easy to obtain (more details in the referenced paper).

**Requirements:**

* Tensorflow 2.9.1
* Tensorflow probability 0.17.0

**Reference:**

[Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)

---
## Setup


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
```

---
## Load the data


```python
data = make_moons(3000, noise=0.05)[0].astype("float32")
norm = layers.Normalization()
norm.adapt(data)
normalized_data = norm(data)
```

---
## Affine coupling layer


```python
# Creating a custom layer with keras API.
output_dim = 256
reg = 0.01


def Coupling(input_shape):
    input = keras.layers.Input(shape=input_shape)

    t_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    t_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    s_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_4)

    return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])

```

---
## Real NVP


```python

class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers):
        super(RealNVP, self).__init__()

        self.num_coupling_layers = num_coupling_layers

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0, 0.0], scale_diag=[1.0, 1.0]
        )
        self.masks = np.array(
            [[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype="float32"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(2) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.

        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:

            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

```

---
## Model training


```python
model = RealNVP(num_coupling_layers=6)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

history = model.fit(
    normalized_data, batch_size=256, epochs=300, verbose=2, validation_split=0.2
)
```

<div class="k-default-codeblock">
```
Epoch 1/300
10/10 - 3s - loss: 2.7067 - val_loss: 2.6912 - 3s/epoch - 260ms/step
Epoch 2/300
10/10 - 0s - loss: 2.6339 - val_loss: 2.6351 - 180ms/epoch - 18ms/step
Epoch 3/300
10/10 - 0s - loss: 2.5795 - val_loss: 2.5893 - 185ms/epoch - 18ms/step
Epoch 4/300
10/10 - 0s - loss: 2.5362 - val_loss: 2.5389 - 179ms/epoch - 18ms/step
Epoch 5/300
10/10 - 0s - loss: 2.4912 - val_loss: 2.4909 - 187ms/epoch - 19ms/step
Epoch 6/300
10/10 - 0s - loss: 2.4583 - val_loss: 2.4492 - 187ms/epoch - 19ms/step
Epoch 7/300
10/10 - 0s - loss: 2.4249 - val_loss: 2.4218 - 185ms/epoch - 18ms/step
Epoch 8/300
10/10 - 0s - loss: 2.3917 - val_loss: 2.3774 - 178ms/epoch - 18ms/step
Epoch 9/300
10/10 - 0s - loss: 2.3752 - val_loss: 2.3658 - 190ms/epoch - 19ms/step
Epoch 10/300
10/10 - 0s - loss: 2.3428 - val_loss: 2.3352 - 189ms/epoch - 19ms/step
Epoch 11/300
10/10 - 0s - loss: 2.3093 - val_loss: 2.3004 - 197ms/epoch - 20ms/step
Epoch 12/300
10/10 - 0s - loss: 2.2888 - val_loss: 2.2859 - 199ms/epoch - 20ms/step
Epoch 13/300
10/10 - 0s - loss: 2.2764 - val_loss: 2.2778 - 202ms/epoch - 20ms/step
Epoch 14/300
10/10 - 0s - loss: 2.2599 - val_loss: 2.2458 - 190ms/epoch - 19ms/step
Epoch 15/300
10/10 - 0s - loss: 2.2489 - val_loss: 2.2173 - 199ms/epoch - 20ms/step
Epoch 16/300
10/10 - 0s - loss: 2.2098 - val_loss: 2.2111 - 191ms/epoch - 19ms/step
Epoch 17/300
10/10 - 0s - loss: 2.1831 - val_loss: 2.1643 - 199ms/epoch - 20ms/step
Epoch 18/300
10/10 - 0s - loss: 2.1726 - val_loss: 2.1599 - 195ms/epoch - 19ms/step
Epoch 19/300
10/10 - 0s - loss: 2.1568 - val_loss: 2.1759 - 204ms/epoch - 20ms/step
Epoch 20/300
10/10 - 0s - loss: 2.1473 - val_loss: 2.1302 - 184ms/epoch - 18ms/step
Epoch 21/300
10/10 - 0s - loss: 2.1234 - val_loss: 2.0842 - 193ms/epoch - 19ms/step
Epoch 22/300
10/10 - 0s - loss: 2.0959 - val_loss: 2.1207 - 185ms/epoch - 18ms/step
Epoch 23/300
10/10 - 0s - loss: 2.1207 - val_loss: 2.0731 - 192ms/epoch - 19ms/step
Epoch 24/300
10/10 - 0s - loss: 2.0908 - val_loss: 2.0888 - 205ms/epoch - 20ms/step
Epoch 25/300
10/10 - 0s - loss: 2.0686 - val_loss: 2.0331 - 190ms/epoch - 19ms/step
Epoch 26/300
10/10 - 0s - loss: 2.0309 - val_loss: 2.0040 - 199ms/epoch - 20ms/step
Epoch 27/300
10/10 - 0s - loss: 1.9997 - val_loss: 1.9805 - 196ms/epoch - 20ms/step
Epoch 28/300
10/10 - 0s - loss: 1.9758 - val_loss: 1.9350 - 199ms/epoch - 20ms/step
Epoch 29/300
10/10 - 0s - loss: 1.9342 - val_loss: 1.8666 - 201ms/epoch - 20ms/step
Epoch 30/300
10/10 - 0s - loss: 1.8781 - val_loss: 1.8226 - 203ms/epoch - 20ms/step
Epoch 31/300
10/10 - 0s - loss: 1.8295 - val_loss: 1.7998 - 214ms/epoch - 21ms/step
Epoch 32/300
10/10 - 0s - loss: 1.8498 - val_loss: 1.7397 - 198ms/epoch - 20ms/step
Epoch 33/300
10/10 - 0s - loss: 1.8128 - val_loss: 1.7548 - 212ms/epoch - 21ms/step
Epoch 34/300
10/10 - 0s - loss: 1.7902 - val_loss: 1.7398 - 198ms/epoch - 20ms/step
Epoch 35/300
10/10 - 0s - loss: 1.7743 - val_loss: 1.7032 - 205ms/epoch - 20ms/step
Epoch 36/300
10/10 - 0s - loss: 1.7374 - val_loss: 1.6623 - 201ms/epoch - 20ms/step
Epoch 37/300
10/10 - 0s - loss: 1.7548 - val_loss: 1.6954 - 212ms/epoch - 21ms/step
Epoch 38/300
10/10 - 0s - loss: 1.7141 - val_loss: 1.6585 - 200ms/epoch - 20ms/step
Epoch 39/300
10/10 - 0s - loss: 1.6930 - val_loss: 1.6529 - 201ms/epoch - 20ms/step
Epoch 40/300
10/10 - 0s - loss: 1.6457 - val_loss: 1.6055 - 205ms/epoch - 20ms/step
Epoch 41/300
10/10 - 0s - loss: 1.6454 - val_loss: 1.5982 - 212ms/epoch - 21ms/step
Epoch 42/300
10/10 - 0s - loss: 1.6383 - val_loss: 1.6343 - 204ms/epoch - 20ms/step
Epoch 43/300
10/10 - 0s - loss: 1.6311 - val_loss: 1.6123 - 208ms/epoch - 21ms/step
Epoch 44/300
10/10 - 0s - loss: 1.6357 - val_loss: 1.6040 - 203ms/epoch - 20ms/step
Epoch 45/300
10/10 - 0s - loss: 1.6065 - val_loss: 1.6026 - 204ms/epoch - 20ms/step
Epoch 46/300
10/10 - 0s - loss: 1.5918 - val_loss: 1.5872 - 211ms/epoch - 21ms/step
Epoch 47/300
10/10 - 0s - loss: 1.5937 - val_loss: 1.5762 - 218ms/epoch - 22ms/step
Epoch 48/300
10/10 - 0s - loss: 1.5816 - val_loss: 1.5639 - 215ms/epoch - 21ms/step
Epoch 49/300
10/10 - 0s - loss: 1.5873 - val_loss: 1.5721 - 211ms/epoch - 21ms/step
Epoch 50/300
10/10 - 0s - loss: 1.5884 - val_loss: 1.6026 - 226ms/epoch - 23ms/step
Epoch 51/300
10/10 - 0s - loss: 1.5827 - val_loss: 1.5482 - 212ms/epoch - 21ms/step
Epoch 52/300
10/10 - 0s - loss: 1.5588 - val_loss: 1.5938 - 211ms/epoch - 21ms/step
Epoch 53/300
10/10 - 0s - loss: 1.5843 - val_loss: 1.5964 - 220ms/epoch - 22ms/step
Epoch 54/300
10/10 - 0s - loss: 1.6062 - val_loss: 1.5903 - 213ms/epoch - 21ms/step
Epoch 55/300
10/10 - 0s - loss: 1.5988 - val_loss: 1.6025 - 213ms/epoch - 21ms/step
Epoch 56/300
10/10 - 0s - loss: 1.5613 - val_loss: 1.5479 - 207ms/epoch - 21ms/step
Epoch 57/300
10/10 - 0s - loss: 1.5307 - val_loss: 1.5149 - 213ms/epoch - 21ms/step
Epoch 58/300
10/10 - 0s - loss: 1.5532 - val_loss: 1.5365 - 216ms/epoch - 22ms/step
Epoch 59/300
10/10 - 0s - loss: 1.5413 - val_loss: 1.5330 - 219ms/epoch - 22ms/step
Epoch 60/300
10/10 - 0s - loss: 1.5507 - val_loss: 1.5550 - 226ms/epoch - 23ms/step
Epoch 61/300
10/10 - 0s - loss: 1.5512 - val_loss: 1.5034 - 220ms/epoch - 22ms/step
Epoch 62/300
10/10 - 0s - loss: 1.5757 - val_loss: 1.6400 - 207ms/epoch - 21ms/step
Epoch 63/300
10/10 - 0s - loss: 1.5975 - val_loss: 1.5630 - 206ms/epoch - 21ms/step
Epoch 64/300
10/10 - 0s - loss: 1.5430 - val_loss: 1.5092 - 202ms/epoch - 20ms/step
Epoch 65/300
10/10 - 0s - loss: 1.5213 - val_loss: 1.5200 - 212ms/epoch - 21ms/step
Epoch 66/300
10/10 - 0s - loss: 1.5070 - val_loss: 1.4972 - 218ms/epoch - 22ms/step
Epoch 67/300
10/10 - 0s - loss: 1.4972 - val_loss: 1.4830 - 217ms/epoch - 22ms/step
Epoch 68/300
10/10 - 0s - loss: 1.4924 - val_loss: 1.4792 - 216ms/epoch - 22ms/step
Epoch 69/300
10/10 - 0s - loss: 1.4837 - val_loss: 1.4932 - 214ms/epoch - 21ms/step
Epoch 70/300
10/10 - 0s - loss: 1.4721 - val_loss: 1.4740 - 218ms/epoch - 22ms/step
Epoch 71/300
10/10 - 0s - loss: 1.4906 - val_loss: 1.4743 - 218ms/epoch - 22ms/step
Epoch 72/300
10/10 - 0s - loss: 1.4897 - val_loss: 1.4668 - 220ms/epoch - 22ms/step
Epoch 73/300
10/10 - 0s - loss: 1.4627 - val_loss: 1.4738 - 217ms/epoch - 22ms/step
Epoch 74/300
10/10 - 0s - loss: 1.4719 - val_loss: 1.4446 - 215ms/epoch - 21ms/step
Epoch 75/300
10/10 - 0s - loss: 1.4560 - val_loss: 1.4891 - 230ms/epoch - 23ms/step
Epoch 76/300
10/10 - 0s - loss: 1.5028 - val_loss: 1.4755 - 227ms/epoch - 23ms/step
Epoch 77/300
10/10 - 0s - loss: 1.4782 - val_loss: 1.4656 - 224ms/epoch - 22ms/step
Epoch 78/300
10/10 - 0s - loss: 1.4687 - val_loss: 1.4554 - 220ms/epoch - 22ms/step
Epoch 79/300
10/10 - 0s - loss: 1.4738 - val_loss: 1.4252 - 229ms/epoch - 23ms/step
Epoch 80/300
10/10 - 0s - loss: 1.4528 - val_loss: 1.4409 - 214ms/epoch - 21ms/step
Epoch 81/300
10/10 - 0s - loss: 1.4929 - val_loss: 1.4963 - 217ms/epoch - 22ms/step
Epoch 82/300
10/10 - 0s - loss: 1.5288 - val_loss: 1.4505 - 222ms/epoch - 22ms/step
Epoch 83/300
10/10 - 0s - loss: 1.4569 - val_loss: 1.4421 - 231ms/epoch - 23ms/step
Epoch 84/300
10/10 - 0s - loss: 1.4249 - val_loss: 1.3958 - 226ms/epoch - 23ms/step
Epoch 85/300
10/10 - 0s - loss: 1.4069 - val_loss: 1.3989 - 218ms/epoch - 22ms/step
Epoch 86/300
10/10 - 0s - loss: 1.4080 - val_loss: 1.4226 - 216ms/epoch - 22ms/step
Epoch 87/300
10/10 - 0s - loss: 1.4261 - val_loss: 1.3816 - 220ms/epoch - 22ms/step
Epoch 88/300
10/10 - 0s - loss: 1.4508 - val_loss: 1.4641 - 231ms/epoch - 23ms/step
Epoch 89/300
10/10 - 0s - loss: 1.4149 - val_loss: 1.4426 - 213ms/epoch - 21ms/step
Epoch 90/300
10/10 - 0s - loss: 1.4112 - val_loss: 1.3897 - 215ms/epoch - 22ms/step
Epoch 91/300
10/10 - 0s - loss: 1.3829 - val_loss: 1.4165 - 210ms/epoch - 21ms/step
Epoch 92/300
10/10 - 0s - loss: 1.3891 - val_loss: 1.4192 - 208ms/epoch - 21ms/step
Epoch 93/300
10/10 - 0s - loss: 1.4233 - val_loss: 1.4004 - 224ms/epoch - 22ms/step
Epoch 94/300
10/10 - 0s - loss: 1.3959 - val_loss: 1.4059 - 227ms/epoch - 23ms/step
Epoch 95/300
10/10 - 0s - loss: 1.3999 - val_loss: 1.4248 - 225ms/epoch - 22ms/step
Epoch 96/300
10/10 - 0s - loss: 1.3835 - val_loss: 1.3886 - 228ms/epoch - 23ms/step
Epoch 97/300
10/10 - 0s - loss: 1.4151 - val_loss: 1.4211 - 224ms/epoch - 22ms/step
Epoch 98/300
10/10 - 0s - loss: 1.4297 - val_loss: 1.3924 - 221ms/epoch - 22ms/step
Epoch 99/300
10/10 - 0s - loss: 1.4213 - val_loss: 1.4187 - 221ms/epoch - 22ms/step
Epoch 100/300
10/10 - 0s - loss: 1.3953 - val_loss: 1.4066 - 221ms/epoch - 22ms/step
Epoch 101/300
10/10 - 0s - loss: 1.3663 - val_loss: 1.4393 - 220ms/epoch - 22ms/step
Epoch 102/300
10/10 - 0s - loss: 1.3962 - val_loss: 1.3858 - 219ms/epoch - 22ms/step
Epoch 103/300
10/10 - 0s - loss: 1.4126 - val_loss: 1.4219 - 223ms/epoch - 22ms/step
Epoch 104/300
10/10 - 0s - loss: 1.4120 - val_loss: 1.3851 - 220ms/epoch - 22ms/step
Epoch 105/300
10/10 - 0s - loss: 1.3979 - val_loss: 1.4365 - 213ms/epoch - 21ms/step
Epoch 106/300
10/10 - 0s - loss: 1.4337 - val_loss: 1.4254 - 223ms/epoch - 22ms/step
Epoch 107/300
10/10 - 0s - loss: 1.3885 - val_loss: 1.3771 - 216ms/epoch - 22ms/step
Epoch 108/300
10/10 - 0s - loss: 1.3738 - val_loss: 1.4432 - 216ms/epoch - 22ms/step
Epoch 109/300
10/10 - 0s - loss: 1.3812 - val_loss: 1.3506 - 220ms/epoch - 22ms/step
Epoch 110/300
10/10 - 0s - loss: 1.3673 - val_loss: 1.4030 - 205ms/epoch - 20ms/step
Epoch 111/300
10/10 - 0s - loss: 1.4047 - val_loss: 1.3900 - 211ms/epoch - 21ms/step
Epoch 112/300
10/10 - 0s - loss: 1.3698 - val_loss: 1.4333 - 210ms/epoch - 21ms/step
Epoch 113/300
10/10 - 0s - loss: 1.4039 - val_loss: 1.4082 - 218ms/epoch - 22ms/step
Epoch 114/300
10/10 - 0s - loss: 1.3896 - val_loss: 1.3688 - 213ms/epoch - 21ms/step
Epoch 115/300
10/10 - 0s - loss: 1.3864 - val_loss: 1.3791 - 211ms/epoch - 21ms/step
Epoch 116/300
10/10 - 0s - loss: 1.3888 - val_loss: 1.3684 - 215ms/epoch - 22ms/step
Epoch 117/300
10/10 - 0s - loss: 1.3712 - val_loss: 1.3544 - 206ms/epoch - 21ms/step
Epoch 118/300
10/10 - 0s - loss: 1.3664 - val_loss: 1.3793 - 207ms/epoch - 21ms/step
Epoch 119/300
10/10 - 0s - loss: 1.3498 - val_loss: 1.3496 - 209ms/epoch - 21ms/step
Epoch 120/300
10/10 - 0s - loss: 1.3461 - val_loss: 1.4419 - 210ms/epoch - 21ms/step
Epoch 121/300
10/10 - 0s - loss: 1.4577 - val_loss: 1.4269 - 211ms/epoch - 21ms/step
Epoch 122/300
10/10 - 0s - loss: 1.4060 - val_loss: 1.3913 - 218ms/epoch - 22ms/step
Epoch 123/300
10/10 - 0s - loss: 1.4248 - val_loss: 1.6816 - 218ms/epoch - 22ms/step
Epoch 124/300
10/10 - 0s - loss: 1.5045 - val_loss: 1.4364 - 209ms/epoch - 21ms/step
Epoch 125/300
10/10 - 0s - loss: 1.4164 - val_loss: 1.4160 - 202ms/epoch - 20ms/step
Epoch 126/300
10/10 - 0s - loss: 1.3642 - val_loss: 1.4265 - 215ms/epoch - 21ms/step
Epoch 127/300
10/10 - 0s - loss: 1.3606 - val_loss: 1.4255 - 209ms/epoch - 21ms/step
Epoch 128/300
10/10 - 0s - loss: 1.4506 - val_loss: 1.4580 - 202ms/epoch - 20ms/step
Epoch 129/300
10/10 - 0s - loss: 1.4550 - val_loss: 1.4483 - 208ms/epoch - 21ms/step
Epoch 130/300
10/10 - 0s - loss: 1.3984 - val_loss: 1.4076 - 203ms/epoch - 20ms/step
Epoch 131/300
10/10 - 0s - loss: 1.3704 - val_loss: 1.4075 - 205ms/epoch - 20ms/step
Epoch 132/300
10/10 - 0s - loss: 1.3929 - val_loss: 1.3997 - 203ms/epoch - 20ms/step
Epoch 133/300
10/10 - 0s - loss: 1.3982 - val_loss: 1.4138 - 204ms/epoch - 20ms/step
Epoch 134/300
10/10 - 0s - loss: 1.3853 - val_loss: 1.4612 - 207ms/epoch - 21ms/step
Epoch 135/300
10/10 - 0s - loss: 1.3837 - val_loss: 1.3768 - 218ms/epoch - 22ms/step
Epoch 136/300
10/10 - 0s - loss: 1.3849 - val_loss: 1.3880 - 210ms/epoch - 21ms/step
Epoch 137/300
10/10 - 0s - loss: 1.4019 - val_loss: 1.4395 - 216ms/epoch - 22ms/step
Epoch 138/300
10/10 - 0s - loss: 1.4317 - val_loss: 1.4130 - 210ms/epoch - 21ms/step
Epoch 139/300
10/10 - 0s - loss: 1.3901 - val_loss: 1.3807 - 216ms/epoch - 22ms/step
Epoch 140/300
10/10 - 0s - loss: 1.3807 - val_loss: 1.3794 - 208ms/epoch - 21ms/step
Epoch 141/300
10/10 - 0s - loss: 1.3666 - val_loss: 1.3523 - 198ms/epoch - 20ms/step
Epoch 142/300
10/10 - 0s - loss: 1.3507 - val_loss: 1.3601 - 202ms/epoch - 20ms/step
Epoch 143/300
10/10 - 0s - loss: 1.3617 - val_loss: 1.4297 - 203ms/epoch - 20ms/step
Epoch 144/300
10/10 - 0s - loss: 1.3641 - val_loss: 1.3875 - 206ms/epoch - 21ms/step
Epoch 145/300
10/10 - 0s - loss: 1.3666 - val_loss: 1.3471 - 205ms/epoch - 20ms/step
Epoch 146/300
10/10 - 0s - loss: 1.3378 - val_loss: 1.3676 - 202ms/epoch - 20ms/step
Epoch 147/300
10/10 - 0s - loss: 1.3464 - val_loss: 1.3662 - 204ms/epoch - 20ms/step
Epoch 148/300
10/10 - 0s - loss: 1.3486 - val_loss: 1.3426 - 210ms/epoch - 21ms/step
Epoch 149/300
10/10 - 0s - loss: 1.3437 - val_loss: 1.3662 - 202ms/epoch - 20ms/step
Epoch 150/300
10/10 - 0s - loss: 1.3468 - val_loss: 1.3549 - 200ms/epoch - 20ms/step
Epoch 151/300
10/10 - 0s - loss: 1.3332 - val_loss: 1.4187 - 212ms/epoch - 21ms/step
Epoch 152/300
10/10 - 0s - loss: 1.3600 - val_loss: 1.4051 - 221ms/epoch - 22ms/step
Epoch 153/300
10/10 - 0s - loss: 1.4089 - val_loss: 1.4273 - 204ms/epoch - 20ms/step
Epoch 154/300
10/10 - 0s - loss: 1.3970 - val_loss: 1.3772 - 211ms/epoch - 21ms/step
Epoch 155/300
10/10 - 0s - loss: 1.3988 - val_loss: 1.4207 - 218ms/epoch - 22ms/step
Epoch 156/300
10/10 - 0s - loss: 1.3885 - val_loss: 1.3793 - 216ms/epoch - 22ms/step
Epoch 157/300
10/10 - 0s - loss: 1.3630 - val_loss: 1.3938 - 205ms/epoch - 20ms/step
Epoch 158/300
10/10 - 0s - loss: 1.3865 - val_loss: 1.3365 - 204ms/epoch - 20ms/step
Epoch 159/300
10/10 - 0s - loss: 1.3240 - val_loss: 1.3564 - 203ms/epoch - 20ms/step
Epoch 160/300
10/10 - 0s - loss: 1.3424 - val_loss: 1.3769 - 210ms/epoch - 21ms/step
Epoch 161/300
10/10 - 0s - loss: 1.3611 - val_loss: 1.3736 - 214ms/epoch - 21ms/step
Epoch 162/300
10/10 - 0s - loss: 1.3294 - val_loss: 1.3682 - 202ms/epoch - 20ms/step
Epoch 163/300
10/10 - 0s - loss: 1.3394 - val_loss: 1.3614 - 212ms/epoch - 21ms/step
Epoch 164/300
10/10 - 0s - loss: 1.3759 - val_loss: 1.4190 - 213ms/epoch - 21ms/step
Epoch 165/300
10/10 - 0s - loss: 1.3961 - val_loss: 1.4092 - 218ms/epoch - 22ms/step
Epoch 166/300
10/10 - 0s - loss: 1.3981 - val_loss: 1.4790 - 206ms/epoch - 21ms/step
Epoch 167/300
10/10 - 0s - loss: 1.4246 - val_loss: 1.4177 - 218ms/epoch - 22ms/step
Epoch 168/300
10/10 - 0s - loss: 1.3881 - val_loss: 1.3845 - 206ms/epoch - 21ms/step
Epoch 169/300
10/10 - 0s - loss: 1.3726 - val_loss: 1.3466 - 203ms/epoch - 20ms/step
Epoch 170/300
10/10 - 0s - loss: 1.3552 - val_loss: 1.4030 - 202ms/epoch - 20ms/step
Epoch 171/300
10/10 - 0s - loss: 1.3156 - val_loss: 1.3702 - 205ms/epoch - 20ms/step
Epoch 172/300
10/10 - 0s - loss: 1.3237 - val_loss: 1.3065 - 205ms/epoch - 20ms/step
Epoch 173/300
10/10 - 0s - loss: 1.3393 - val_loss: 1.3296 - 207ms/epoch - 21ms/step
Epoch 174/300
10/10 - 0s - loss: 1.3681 - val_loss: 1.3923 - 204ms/epoch - 20ms/step
Epoch 175/300
10/10 - 0s - loss: 1.3875 - val_loss: 1.3561 - 202ms/epoch - 20ms/step
Epoch 176/300
10/10 - 0s - loss: 1.3602 - val_loss: 1.3480 - 209ms/epoch - 21ms/step
Epoch 177/300
10/10 - 0s - loss: 1.3393 - val_loss: 1.3353 - 207ms/epoch - 21ms/step
Epoch 178/300
10/10 - 0s - loss: 1.3215 - val_loss: 1.3172 - 218ms/epoch - 22ms/step
Epoch 179/300
10/10 - 0s - loss: 1.2992 - val_loss: 1.3189 - 217ms/epoch - 22ms/step
Epoch 180/300
10/10 - 0s - loss: 1.3239 - val_loss: 1.4028 - 206ms/epoch - 21ms/step
Epoch 181/300
10/10 - 0s - loss: 1.3519 - val_loss: 1.3475 - 209ms/epoch - 21ms/step
Epoch 182/300
10/10 - 0s - loss: 1.3683 - val_loss: 1.3633 - 207ms/epoch - 21ms/step
Epoch 183/300
10/10 - 0s - loss: 1.3604 - val_loss: 1.3797 - 212ms/epoch - 21ms/step
Epoch 184/300
10/10 - 0s - loss: 1.3438 - val_loss: 1.3655 - 208ms/epoch - 21ms/step
Epoch 185/300
10/10 - 0s - loss: 1.3394 - val_loss: 1.3347 - 211ms/epoch - 21ms/step
Epoch 186/300
10/10 - 0s - loss: 1.3261 - val_loss: 1.3219 - 208ms/epoch - 21ms/step
Epoch 187/300
10/10 - 0s - loss: 1.3143 - val_loss: 1.3249 - 208ms/epoch - 21ms/step
Epoch 188/300
10/10 - 0s - loss: 1.3134 - val_loss: 1.3248 - 209ms/epoch - 21ms/step
Epoch 189/300
10/10 - 0s - loss: 1.2993 - val_loss: 1.3140 - 209ms/epoch - 21ms/step
Epoch 190/300
10/10 - 0s - loss: 1.3054 - val_loss: 1.3256 - 201ms/epoch - 20ms/step
Epoch 191/300
10/10 - 0s - loss: 1.2996 - val_loss: 1.3214 - 208ms/epoch - 21ms/step
Epoch 192/300
10/10 - 0s - loss: 1.3025 - val_loss: 1.3218 - 205ms/epoch - 20ms/step
Epoch 193/300
10/10 - 0s - loss: 1.2993 - val_loss: 1.3093 - 204ms/epoch - 20ms/step
Epoch 194/300
10/10 - 0s - loss: 1.2975 - val_loss: 1.3044 - 213ms/epoch - 21ms/step
Epoch 195/300
10/10 - 0s - loss: 1.3049 - val_loss: 1.3109 - 204ms/epoch - 20ms/step
Epoch 196/300
10/10 - 0s - loss: 1.3112 - val_loss: 1.3166 - 207ms/epoch - 21ms/step
Epoch 197/300
10/10 - 0s - loss: 1.3070 - val_loss: 1.3232 - 219ms/epoch - 22ms/step
Epoch 198/300
10/10 - 0s - loss: 1.2980 - val_loss: 1.3213 - 205ms/epoch - 20ms/step
Epoch 199/300
10/10 - 0s - loss: 1.2924 - val_loss: 1.2979 - 208ms/epoch - 21ms/step
Epoch 200/300
10/10 - 0s - loss: 1.2985 - val_loss: 1.3144 - 209ms/epoch - 21ms/step
Epoch 201/300
10/10 - 0s - loss: 1.3073 - val_loss: 1.3186 - 214ms/epoch - 21ms/step
Epoch 202/300
10/10 - 0s - loss: 1.3092 - val_loss: 1.3206 - 220ms/epoch - 22ms/step
Epoch 203/300
10/10 - 0s - loss: 1.3300 - val_loss: 1.3610 - 205ms/epoch - 21ms/step
Epoch 204/300
10/10 - 0s - loss: 1.3374 - val_loss: 1.3176 - 216ms/epoch - 22ms/step
Epoch 205/300
10/10 - 0s - loss: 1.3240 - val_loss: 1.3531 - 208ms/epoch - 21ms/step
Epoch 206/300
10/10 - 0s - loss: 1.3241 - val_loss: 1.3287 - 225ms/epoch - 22ms/step
Epoch 207/300
10/10 - 0s - loss: 1.3244 - val_loss: 1.3156 - 214ms/epoch - 21ms/step
Epoch 208/300
10/10 - 0s - loss: 1.3226 - val_loss: 1.3273 - 214ms/epoch - 21ms/step
Epoch 209/300
10/10 - 0s - loss: 1.3312 - val_loss: 1.3338 - 211ms/epoch - 21ms/step
Epoch 210/300
10/10 - 0s - loss: 1.3149 - val_loss: 1.3182 - 216ms/epoch - 22ms/step
Epoch 211/300
10/10 - 0s - loss: 1.3102 - val_loss: 1.3363 - 212ms/epoch - 21ms/step
Epoch 212/300
10/10 - 0s - loss: 1.2764 - val_loss: 1.3164 - 215ms/epoch - 21ms/step
Epoch 213/300
10/10 - 0s - loss: 1.2809 - val_loss: 1.3028 - 208ms/epoch - 21ms/step
Epoch 214/300
10/10 - 0s - loss: 1.2738 - val_loss: 1.3061 - 204ms/epoch - 20ms/step
Epoch 215/300
10/10 - 0s - loss: 1.2708 - val_loss: 1.3025 - 215ms/epoch - 21ms/step
Epoch 216/300
10/10 - 0s - loss: 1.2827 - val_loss: 1.3392 - 205ms/epoch - 20ms/step
Epoch 217/300
10/10 - 0s - loss: 1.3306 - val_loss: 1.3572 - 205ms/epoch - 20ms/step
Epoch 218/300
10/10 - 0s - loss: 1.3407 - val_loss: 1.3207 - 212ms/epoch - 21ms/step
Epoch 219/300
10/10 - 0s - loss: 1.3248 - val_loss: 1.3274 - 208ms/epoch - 21ms/step
Epoch 220/300
10/10 - 0s - loss: 1.3038 - val_loss: 1.3188 - 209ms/epoch - 21ms/step
Epoch 221/300
10/10 - 0s - loss: 1.2954 - val_loss: 1.3333 - 210ms/epoch - 21ms/step
Epoch 222/300
10/10 - 0s - loss: 1.2948 - val_loss: 1.3067 - 206ms/epoch - 21ms/step
Epoch 223/300
10/10 - 0s - loss: 1.2917 - val_loss: 1.3050 - 208ms/epoch - 21ms/step
Epoch 224/300
10/10 - 0s - loss: 1.2916 - val_loss: 1.3366 - 205ms/epoch - 21ms/step
Epoch 225/300
10/10 - 0s - loss: 1.2768 - val_loss: 1.3469 - 202ms/epoch - 20ms/step
Epoch 226/300
10/10 - 0s - loss: 1.2800 - val_loss: 1.3177 - 204ms/epoch - 20ms/step
Epoch 227/300
10/10 - 0s - loss: 1.2710 - val_loss: 1.3121 - 206ms/epoch - 21ms/step
Epoch 228/300
10/10 - 0s - loss: 1.3074 - val_loss: 1.3514 - 202ms/epoch - 20ms/step
Epoch 229/300
10/10 - 0s - loss: 1.3043 - val_loss: 1.3227 - 207ms/epoch - 21ms/step
Epoch 230/300
10/10 - 0s - loss: 1.2976 - val_loss: 1.3226 - 208ms/epoch - 21ms/step
Epoch 231/300
10/10 - 0s - loss: 1.2934 - val_loss: 1.3254 - 209ms/epoch - 21ms/step
Epoch 232/300
10/10 - 0s - loss: 1.2946 - val_loss: 1.3570 - 203ms/epoch - 20ms/step
Epoch 233/300
10/10 - 0s - loss: 1.2870 - val_loss: 1.2861 - 203ms/epoch - 20ms/step
Epoch 234/300
10/10 - 0s - loss: 1.2864 - val_loss: 1.3551 - 197ms/epoch - 20ms/step
Epoch 235/300
10/10 - 0s - loss: 1.3355 - val_loss: 1.4227 - 203ms/epoch - 20ms/step
Epoch 236/300
10/10 - 0s - loss: 1.3320 - val_loss: 1.3517 - 201ms/epoch - 20ms/step
Epoch 237/300
10/10 - 0s - loss: 1.2976 - val_loss: 1.3438 - 196ms/epoch - 20ms/step
Epoch 238/300
10/10 - 0s - loss: 1.2849 - val_loss: 1.3064 - 198ms/epoch - 20ms/step
Epoch 239/300
10/10 - 0s - loss: 1.2962 - val_loss: 1.2863 - 196ms/epoch - 20ms/step
Epoch 240/300
10/10 - 0s - loss: 1.2973 - val_loss: 1.3162 - 202ms/epoch - 20ms/step
Epoch 241/300
10/10 - 0s - loss: 1.2844 - val_loss: 1.3100 - 200ms/epoch - 20ms/step
Epoch 242/300
10/10 - 0s - loss: 1.2812 - val_loss: 1.3209 - 208ms/epoch - 21ms/step
Epoch 243/300
10/10 - 0s - loss: 1.2899 - val_loss: 1.3279 - 205ms/epoch - 21ms/step
Epoch 244/300
10/10 - 0s - loss: 1.3213 - val_loss: 1.3086 - 211ms/epoch - 21ms/step
Epoch 245/300
10/10 - 0s - loss: 1.3096 - val_loss: 1.3225 - 211ms/epoch - 21ms/step
Epoch 246/300
10/10 - 0s - loss: 1.3134 - val_loss: 1.3338 - 198ms/epoch - 20ms/step
Epoch 247/300
10/10 - 0s - loss: 1.2945 - val_loss: 1.3027 - 204ms/epoch - 20ms/step
Epoch 248/300
10/10 - 0s - loss: 1.2819 - val_loss: 1.3745 - 202ms/epoch - 20ms/step
Epoch 249/300
10/10 - 0s - loss: 1.2934 - val_loss: 1.3414 - 209ms/epoch - 21ms/step
Epoch 250/300
10/10 - 0s - loss: 1.3134 - val_loss: 1.3394 - 208ms/epoch - 21ms/step
Epoch 251/300
10/10 - 0s - loss: 1.2976 - val_loss: 1.3571 - 209ms/epoch - 21ms/step
Epoch 252/300
10/10 - 0s - loss: 1.2942 - val_loss: 1.3450 - 200ms/epoch - 20ms/step
Epoch 253/300
10/10 - 0s - loss: 1.2874 - val_loss: 1.3249 - 208ms/epoch - 21ms/step
Epoch 254/300
10/10 - 0s - loss: 1.2790 - val_loss: 1.3438 - 212ms/epoch - 21ms/step
Epoch 255/300
10/10 - 0s - loss: 1.2809 - val_loss: 1.3392 - 205ms/epoch - 20ms/step
Epoch 256/300
10/10 - 0s - loss: 1.2880 - val_loss: 1.3311 - 205ms/epoch - 20ms/step
Epoch 257/300
10/10 - 0s - loss: 1.2720 - val_loss: 1.3094 - 212ms/epoch - 21ms/step
Epoch 258/300
10/10 - 0s - loss: 1.2864 - val_loss: 1.3161 - 199ms/epoch - 20ms/step
Epoch 259/300
10/10 - 0s - loss: 1.2702 - val_loss: 1.2991 - 205ms/epoch - 20ms/step
Epoch 260/300
10/10 - 0s - loss: 1.2614 - val_loss: 1.3000 - 199ms/epoch - 20ms/step
Epoch 261/300
10/10 - 0s - loss: 1.2791 - val_loss: 1.3481 - 210ms/epoch - 21ms/step
Epoch 262/300
10/10 - 0s - loss: 1.2779 - val_loss: 1.3130 - 210ms/epoch - 21ms/step
Epoch 263/300
10/10 - 0s - loss: 1.2902 - val_loss: 1.3223 - 206ms/epoch - 21ms/step
Epoch 264/300
10/10 - 0s - loss: 1.2762 - val_loss: 1.2870 - 202ms/epoch - 20ms/step
Epoch 265/300
10/10 - 0s - loss: 1.2832 - val_loss: 1.3069 - 209ms/epoch - 21ms/step
Epoch 266/300
10/10 - 0s - loss: 1.2944 - val_loss: 1.3130 - 201ms/epoch - 20ms/step
Epoch 267/300
10/10 - 0s - loss: 1.3090 - val_loss: 1.3192 - 206ms/epoch - 21ms/step
Epoch 268/300
10/10 - 0s - loss: 1.2931 - val_loss: 1.3111 - 204ms/epoch - 20ms/step
Epoch 269/300
10/10 - 0s - loss: 1.2817 - val_loss: 1.3115 - 203ms/epoch - 20ms/step
Epoch 270/300
10/10 - 0s - loss: 1.2787 - val_loss: 1.2921 - 203ms/epoch - 20ms/step
Epoch 271/300
10/10 - 0s - loss: 1.2714 - val_loss: 1.2897 - 204ms/epoch - 20ms/step
Epoch 272/300
10/10 - 0s - loss: 1.2586 - val_loss: 1.2950 - 203ms/epoch - 20ms/step
Epoch 273/300
10/10 - 0s - loss: 1.2728 - val_loss: 1.3013 - 202ms/epoch - 20ms/step
Epoch 274/300
10/10 - 0s - loss: 1.2632 - val_loss: 1.2794 - 209ms/epoch - 21ms/step
Epoch 275/300
10/10 - 0s - loss: 1.2680 - val_loss: 1.3076 - 206ms/epoch - 21ms/step
Epoch 276/300
10/10 - 0s - loss: 1.2796 - val_loss: 1.3095 - 209ms/epoch - 21ms/step
Epoch 277/300
10/10 - 0s - loss: 1.2829 - val_loss: 1.3034 - 207ms/epoch - 21ms/step
Epoch 278/300
10/10 - 0s - loss: 1.2988 - val_loss: 1.3296 - 201ms/epoch - 20ms/step
Epoch 279/300
10/10 - 0s - loss: 1.2920 - val_loss: 1.3411 - 212ms/epoch - 21ms/step
Epoch 280/300
10/10 - 0s - loss: 1.3283 - val_loss: 1.4149 - 206ms/epoch - 21ms/step
Epoch 281/300
10/10 - 0s - loss: 1.3645 - val_loss: 1.3459 - 210ms/epoch - 21ms/step
Epoch 282/300
10/10 - 0s - loss: 1.3335 - val_loss: 1.3568 - 208ms/epoch - 21ms/step
Epoch 283/300
10/10 - 0s - loss: 1.3276 - val_loss: 1.3500 - 200ms/epoch - 20ms/step
Epoch 284/300
10/10 - 0s - loss: 1.3143 - val_loss: 1.3351 - 207ms/epoch - 21ms/step
Epoch 285/300
10/10 - 0s - loss: 1.3491 - val_loss: 1.3593 - 205ms/epoch - 21ms/step
Epoch 286/300
10/10 - 0s - loss: 1.3495 - val_loss: 1.3396 - 205ms/epoch - 21ms/step
Epoch 287/300
10/10 - 0s - loss: 1.3061 - val_loss: 1.3276 - 199ms/epoch - 20ms/step
Epoch 288/300
10/10 - 0s - loss: 1.2856 - val_loss: 1.3133 - 198ms/epoch - 20ms/step
Epoch 289/300
10/10 - 0s - loss: 1.2898 - val_loss: 1.3382 - 203ms/epoch - 20ms/step
Epoch 290/300
10/10 - 0s - loss: 1.2815 - val_loss: 1.2902 - 201ms/epoch - 20ms/step
Epoch 291/300
10/10 - 0s - loss: 1.2662 - val_loss: 1.3137 - 213ms/epoch - 21ms/step
Epoch 292/300
10/10 - 0s - loss: 1.2676 - val_loss: 1.2988 - 208ms/epoch - 21ms/step
Epoch 293/300
10/10 - 0s - loss: 1.2639 - val_loss: 1.3081 - 207ms/epoch - 21ms/step
Epoch 294/300
10/10 - 0s - loss: 1.3039 - val_loss: 1.3184 - 213ms/epoch - 21ms/step
Epoch 295/300
10/10 - 0s - loss: 1.2870 - val_loss: 1.3231 - 219ms/epoch - 22ms/step
Epoch 296/300
10/10 - 0s - loss: 1.2794 - val_loss: 1.3042 - 226ms/epoch - 23ms/step
Epoch 297/300
10/10 - 0s - loss: 1.2921 - val_loss: 1.3219 - 226ms/epoch - 23ms/step
Epoch 298/300
10/10 - 0s - loss: 1.2840 - val_loss: 1.2890 - 229ms/epoch - 23ms/step
Epoch 299/300
10/10 - 0s - loss: 1.2728 - val_loss: 1.2809 - 222ms/epoch - 22ms/step
Epoch 300/300
10/10 - 0s - loss: 1.2603 - val_loss: 1.2990 - 225ms/epoch - 22ms/step
```
</div>
    

---
## Performance evaluation


```python
plt.figure(figsize=(15, 10))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.legend(["train", "validation"], loc="upper right")
plt.ylabel("loss")
plt.xlabel("epoch")

# From data to latent space.
z, _ = model(normalized_data)

# From latent space to data.
samples = model.distribution.sample(3000)
x, _ = model.predict(samples)

f, axes = plt.subplots(2, 2)
f.set_size_inches(20, 15)

axes[0, 0].scatter(normalized_data[:, 0], normalized_data[:, 1], color="r")
axes[0, 0].set(title="Inference data space X", xlabel="x", ylabel="y")
axes[0, 1].scatter(z[:, 0], z[:, 1], color="r")
axes[0, 1].set(title="Inference latent space Z", xlabel="x", ylabel="y")
axes[0, 1].set_xlim([-3.5, 4])
axes[0, 1].set_ylim([-4, 4])
axes[1, 0].scatter(samples[:, 0], samples[:, 1], color="g")
axes[1, 0].set(title="Generated latent space Z", xlabel="x", ylabel="y")
axes[1, 1].scatter(x[:, 0], x[:, 1], color="g")
axes[1, 1].set(title="Generated data space X", label="x", ylabel="y")
axes[1, 1].set_xlim([-2, 2])
axes[1, 1].set_ylim([-2, 2])
```

<div class="k-default-codeblock">
```
94/94 [==============================] - 0s 2ms/step
```
</div>
    




<div class="k-default-codeblock">
```
(-2.0, 2.0)

```
</div>
    
![png](/img/examples/generative/real_nvp/real_nvp_13_1.png)
    



    
![png](/img/examples/generative/real_nvp/real_nvp_13_2.png)
    


**Example available on HuggingFace**
| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model%3A%20-Real%20NVP-black.svg)](https://huggingface.co/keras-io/real_nvp) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces%3A-Real%20NVP-black.svg)](https://huggingface.co/spaces/keras-io/Real_NVP) |

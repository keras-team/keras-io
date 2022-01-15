# Density estimation using Real NVP

**Authors:** [Mandolini Giorgio Maria](https://www.linkedin.com/in/giorgio-maria-mandolini-a2a1b71b4/), [Sanna Daniele](https://www.linkedin.com/in/daniele-sanna-338629bb/), [Zannini Quirini Giorgio](https://www.linkedin.com/in/giorgio-zannini-quirini-16ab181a0/)<br>
**Date created:** 2020/08/10<br>
**Last modified:** 2020/08/10<br>
**Description:** Estimating the density distribution of the "double moon" dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generativeipynb/real_nvp.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generativereal_nvp.py)



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

* Tensorflow 2.3
* Tensorflow probability 0.11.0

**Reference:**

[Density estimation using Real NVP](https://arxiv.org/pdf/1605.08803.pdf)

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
10/10 - 1s - loss: 2.7178 - val_loss: 2.5872
Epoch 2/300
10/10 - 0s - loss: 2.6151 - val_loss: 2.5421
Epoch 3/300
10/10 - 0s - loss: 2.5702 - val_loss: 2.5001
Epoch 4/300
10/10 - 0s - loss: 2.5241 - val_loss: 2.4650
Epoch 5/300
10/10 - 0s - loss: 2.4934 - val_loss: 2.4377
Epoch 6/300
10/10 - 0s - loss: 2.4684 - val_loss: 2.4236
Epoch 7/300
10/10 - 0s - loss: 2.4420 - val_loss: 2.3976
Epoch 8/300
10/10 - 0s - loss: 2.4185 - val_loss: 2.3722
Epoch 9/300
10/10 - 0s - loss: 2.3857 - val_loss: 2.3591
Epoch 10/300
10/10 - 0s - loss: 2.3611 - val_loss: 2.3341
Epoch 11/300
10/10 - 0s - loss: 2.3323 - val_loss: 2.2999
Epoch 12/300
10/10 - 0s - loss: 2.3035 - val_loss: 2.2688
Epoch 13/300
10/10 - 0s - loss: 2.2694 - val_loss: 2.2435
Epoch 14/300
10/10 - 0s - loss: 2.2359 - val_loss: 2.2137
Epoch 15/300
10/10 - 0s - loss: 2.2053 - val_loss: 2.1877
Epoch 16/300
10/10 - 0s - loss: 2.1775 - val_loss: 2.1626
Epoch 17/300
10/10 - 0s - loss: 2.1546 - val_loss: 2.1257
Epoch 18/300
10/10 - 0s - loss: 2.1310 - val_loss: 2.1022
Epoch 19/300
10/10 - 0s - loss: 2.1258 - val_loss: 2.1022
Epoch 20/300
10/10 - 0s - loss: 2.1097 - val_loss: 2.0670
Epoch 21/300
10/10 - 0s - loss: 2.0811 - val_loss: 2.0502
Epoch 22/300
10/10 - 0s - loss: 2.0407 - val_loss: 2.0235
Epoch 23/300
10/10 - 0s - loss: 2.0169 - val_loss: 1.9946
Epoch 24/300
10/10 - 0s - loss: 2.0011 - val_loss: 1.9843
Epoch 25/300
10/10 - 0s - loss: 2.0151 - val_loss: 1.9728
Epoch 26/300
10/10 - 0s - loss: 1.9427 - val_loss: 1.9473
Epoch 27/300
10/10 - 0s - loss: 1.9266 - val_loss: 1.9245
Epoch 28/300
10/10 - 0s - loss: 1.8574 - val_loss: 1.7811
Epoch 29/300
10/10 - 0s - loss: 1.7765 - val_loss: 1.7016
Epoch 30/300
10/10 - 0s - loss: 1.7020 - val_loss: 1.6801
Epoch 31/300
10/10 - 0s - loss: 1.6935 - val_loss: 1.6644
Epoch 32/300
10/10 - 0s - loss: 1.6643 - val_loss: 1.6998
Epoch 33/300
10/10 - 0s - loss: 1.6733 - val_loss: 1.7054
Epoch 34/300
10/10 - 0s - loss: 1.6405 - val_loss: 1.6217
Epoch 35/300
10/10 - 0s - loss: 1.6035 - val_loss: 1.6094
Epoch 36/300
10/10 - 0s - loss: 1.5700 - val_loss: 1.6086
Epoch 37/300
10/10 - 0s - loss: 1.5750 - val_loss: 1.6160
Epoch 38/300
10/10 - 0s - loss: 1.5512 - val_loss: 1.6023
Epoch 39/300
10/10 - 0s - loss: 1.5664 - val_loss: 1.5859
Epoch 40/300
10/10 - 0s - loss: 1.5949 - val_loss: 1.6684
Epoch 41/300
10/10 - 0s - loss: 1.6125 - val_loss: 1.5688
Epoch 42/300
10/10 - 0s - loss: 1.5855 - val_loss: 1.5783
Epoch 43/300
10/10 - 0s - loss: 1.5394 - val_loss: 1.5332
Epoch 44/300
10/10 - 0s - loss: 1.5093 - val_loss: 1.6073
Epoch 45/300
10/10 - 0s - loss: 1.5417 - val_loss: 1.5910
Epoch 46/300
10/10 - 0s - loss: 1.5095 - val_loss: 1.5061
Epoch 47/300
10/10 - 0s - loss: 1.4626 - val_loss: 1.5143
Epoch 48/300
10/10 - 0s - loss: 1.4588 - val_loss: 1.5005
Epoch 49/300
10/10 - 0s - loss: 1.4683 - val_loss: 1.5071
Epoch 50/300
10/10 - 0s - loss: 1.4285 - val_loss: 1.5894
Epoch 51/300
10/10 - 0s - loss: 1.4110 - val_loss: 1.4964
Epoch 52/300
10/10 - 0s - loss: 1.4510 - val_loss: 1.5608
Epoch 53/300
10/10 - 0s - loss: 1.4584 - val_loss: 1.5640
Epoch 54/300
10/10 - 0s - loss: 1.4393 - val_loss: 1.5073
Epoch 55/300
10/10 - 0s - loss: 1.4248 - val_loss: 1.5284
Epoch 56/300
10/10 - 0s - loss: 1.4659 - val_loss: 1.4654
Epoch 57/300
10/10 - 0s - loss: 1.4572 - val_loss: 1.4633
Epoch 58/300
10/10 - 0s - loss: 1.4254 - val_loss: 1.4536
Epoch 59/300
10/10 - 0s - loss: 1.3927 - val_loss: 1.4672
Epoch 60/300
10/10 - 0s - loss: 1.3782 - val_loss: 1.4166
Epoch 61/300
10/10 - 0s - loss: 1.3674 - val_loss: 1.4340
Epoch 62/300
10/10 - 0s - loss: 1.3521 - val_loss: 1.4302
Epoch 63/300
10/10 - 0s - loss: 1.3656 - val_loss: 1.4610
Epoch 64/300
10/10 - 0s - loss: 1.3916 - val_loss: 1.5597
Epoch 65/300
10/10 - 0s - loss: 1.4478 - val_loss: 1.4781
Epoch 66/300
10/10 - 0s - loss: 1.3987 - val_loss: 1.5077
Epoch 67/300
10/10 - 0s - loss: 1.3553 - val_loss: 1.4511
Epoch 68/300
10/10 - 0s - loss: 1.3901 - val_loss: 1.4013
Epoch 69/300
10/10 - 0s - loss: 1.3682 - val_loss: 1.4378
Epoch 70/300
10/10 - 0s - loss: 1.3688 - val_loss: 1.4445
Epoch 71/300
10/10 - 0s - loss: 1.3341 - val_loss: 1.4139
Epoch 72/300
10/10 - 0s - loss: 1.3621 - val_loss: 1.5097
Epoch 73/300
10/10 - 0s - loss: 1.4158 - val_loss: 1.4735
Epoch 74/300
10/10 - 0s - loss: 1.4013 - val_loss: 1.4390
Epoch 75/300
10/10 - 0s - loss: 1.3637 - val_loss: 1.4306
Epoch 76/300
10/10 - 0s - loss: 1.3278 - val_loss: 1.4007
Epoch 77/300
10/10 - 0s - loss: 1.3153 - val_loss: 1.4226
Epoch 78/300
10/10 - 0s - loss: 1.3687 - val_loss: 1.4315
Epoch 79/300
10/10 - 0s - loss: 1.3377 - val_loss: 1.4520
Epoch 80/300
10/10 - 0s - loss: 1.3214 - val_loss: 1.4643
Epoch 81/300
10/10 - 0s - loss: 1.2906 - val_loss: 1.5738
Epoch 82/300
10/10 - 0s - loss: 1.3231 - val_loss: 1.8303
Epoch 83/300
10/10 - 0s - loss: 1.3099 - val_loss: 1.4406
Epoch 84/300
10/10 - 0s - loss: 1.3427 - val_loss: 1.5539
Epoch 85/300
10/10 - 0s - loss: 1.3270 - val_loss: 1.5454
Epoch 86/300
10/10 - 0s - loss: 1.3959 - val_loss: 1.4328
Epoch 87/300
10/10 - 0s - loss: 1.3469 - val_loss: 1.4087
Epoch 88/300
10/10 - 0s - loss: 1.3383 - val_loss: 1.4003
Epoch 89/300
10/10 - 0s - loss: 1.2968 - val_loss: 1.4284
Epoch 90/300
10/10 - 0s - loss: 1.4229 - val_loss: 1.4831
Epoch 91/300
10/10 - 0s - loss: 1.4664 - val_loss: 1.4332
Epoch 92/300
10/10 - 0s - loss: 1.4076 - val_loss: 1.4708
Epoch 93/300
10/10 - 0s - loss: 1.3508 - val_loss: 1.3865
Epoch 94/300
10/10 - 0s - loss: 1.3170 - val_loss: 1.3794
Epoch 95/300
10/10 - 0s - loss: 1.3266 - val_loss: 1.5315
Epoch 96/300
10/10 - 0s - loss: 1.3247 - val_loss: 1.4001
Epoch 97/300
10/10 - 0s - loss: 1.2963 - val_loss: 1.4036
Epoch 98/300
10/10 - 0s - loss: 1.2839 - val_loss: 1.4195
Epoch 99/300
10/10 - 0s - loss: 1.3517 - val_loss: 1.4023
Epoch 100/300
10/10 - 0s - loss: 1.3468 - val_loss: 1.4460
Epoch 101/300
10/10 - 0s - loss: 1.3938 - val_loss: 1.4292
Epoch 102/300
10/10 - 0s - loss: 1.3313 - val_loss: 1.4288
Epoch 103/300
10/10 - 0s - loss: 1.3267 - val_loss: 1.3968
Epoch 104/300
10/10 - 0s - loss: 1.3321 - val_loss: 1.4145
Epoch 105/300
10/10 - 0s - loss: 1.2973 - val_loss: 1.3500
Epoch 106/300
10/10 - 0s - loss: 1.2455 - val_loss: 1.4672
Epoch 107/300
10/10 - 0s - loss: 1.3255 - val_loss: 1.4633
Epoch 108/300
10/10 - 0s - loss: 1.3379 - val_loss: 1.3717
Epoch 109/300
10/10 - 0s - loss: 1.3243 - val_loss: 1.4118
Epoch 110/300
10/10 - 0s - loss: 1.3184 - val_loss: 1.3922
Epoch 111/300
10/10 - 0s - loss: 1.2779 - val_loss: 1.3783
Epoch 112/300
10/10 - 0s - loss: 1.3495 - val_loss: 1.6651
Epoch 113/300
10/10 - 0s - loss: 1.5595 - val_loss: 1.5984
Epoch 114/300
10/10 - 0s - loss: 1.4541 - val_loss: 1.4844
Epoch 115/300
10/10 - 0s - loss: 1.4001 - val_loss: 1.4477
Epoch 116/300
10/10 - 0s - loss: 1.3305 - val_loss: 1.4097
Epoch 117/300
10/10 - 0s - loss: 1.3084 - val_loss: 1.3643
Epoch 118/300
10/10 - 0s - loss: 1.2993 - val_loss: 1.3726
Epoch 119/300
10/10 - 0s - loss: 1.2624 - val_loss: 1.3927
Epoch 120/300
10/10 - 0s - loss: 1.3288 - val_loss: 1.3912
Epoch 121/300
10/10 - 0s - loss: 1.2925 - val_loss: 1.3809
Epoch 122/300
10/10 - 0s - loss: 1.2756 - val_loss: 1.3434
Epoch 123/300
10/10 - 0s - loss: 1.2540 - val_loss: 1.3699
Epoch 124/300
10/10 - 0s - loss: 1.3008 - val_loss: 1.3272
Epoch 125/300
10/10 - 0s - loss: 1.2932 - val_loss: 1.3365
Epoch 126/300
10/10 - 0s - loss: 1.2844 - val_loss: 1.3824
Epoch 127/300
10/10 - 0s - loss: 1.2688 - val_loss: 1.3413
Epoch 128/300
10/10 - 0s - loss: 1.2636 - val_loss: 1.3659
Epoch 129/300
10/10 - 0s - loss: 1.2590 - val_loss: 1.3724
Epoch 130/300
10/10 - 0s - loss: 1.4471 - val_loss: 1.4119
Epoch 131/300
10/10 - 0s - loss: 1.5125 - val_loss: 1.5486
Epoch 132/300
10/10 - 0s - loss: 1.5826 - val_loss: 1.4578
Epoch 133/300
10/10 - 0s - loss: 1.4168 - val_loss: 1.4405
Epoch 134/300
10/10 - 0s - loss: 1.3739 - val_loss: 1.4728
Epoch 135/300
10/10 - 0s - loss: 1.3304 - val_loss: 1.3734
Epoch 136/300
10/10 - 0s - loss: 1.2987 - val_loss: 1.3769
Epoch 137/300
10/10 - 0s - loss: 1.2883 - val_loss: 1.3542
Epoch 138/300
10/10 - 0s - loss: 1.2805 - val_loss: 1.4974
Epoch 139/300
10/10 - 0s - loss: 1.3558 - val_loss: 1.3958
Epoch 140/300
10/10 - 0s - loss: 1.3244 - val_loss: 1.3705
Epoch 141/300
10/10 - 0s - loss: 1.3043 - val_loss: 1.3563
Epoch 142/300
10/10 - 0s - loss: 1.3302 - val_loss: 1.3611
Epoch 143/300
10/10 - 0s - loss: 1.3188 - val_loss: 1.4500
Epoch 144/300
10/10 - 0s - loss: 1.3100 - val_loss: 1.3893
Epoch 145/300
10/10 - 0s - loss: 1.2864 - val_loss: 1.3436
Epoch 146/300
10/10 - 0s - loss: 1.3013 - val_loss: 1.3548
Epoch 147/300
10/10 - 0s - loss: 1.2672 - val_loss: 1.4179
Epoch 148/300
10/10 - 0s - loss: 1.2650 - val_loss: 1.3705
Epoch 149/300
10/10 - 0s - loss: 1.2931 - val_loss: 1.3274
Epoch 150/300
10/10 - 0s - loss: 1.3365 - val_loss: 1.4164
Epoch 151/300
10/10 - 0s - loss: 1.3562 - val_loss: 1.3815
Epoch 152/300
10/10 - 0s - loss: 1.3067 - val_loss: 1.4100
Epoch 153/300
10/10 - 0s - loss: 1.2752 - val_loss: 1.3928
Epoch 154/300
10/10 - 0s - loss: 1.2659 - val_loss: 1.3512
Epoch 155/300
10/10 - 0s - loss: 1.2696 - val_loss: 1.3715
Epoch 156/300
10/10 - 0s - loss: 1.2719 - val_loss: 1.3366
Epoch 157/300
10/10 - 0s - loss: 1.2718 - val_loss: 1.5284
Epoch 158/300
10/10 - 0s - loss: 1.3099 - val_loss: 1.3342
Epoch 159/300
10/10 - 0s - loss: 1.2655 - val_loss: 1.3692
Epoch 160/300
10/10 - 0s - loss: 1.2694 - val_loss: 1.5034
Epoch 161/300
10/10 - 0s - loss: 1.3370 - val_loss: 1.3611
Epoch 162/300
10/10 - 0s - loss: 1.2799 - val_loss: 1.3745
Epoch 163/300
10/10 - 0s - loss: 1.2714 - val_loss: 1.3639
Epoch 164/300
10/10 - 0s - loss: 1.2711 - val_loss: 1.3178
Epoch 165/300
10/10 - 0s - loss: 1.2754 - val_loss: 1.3722
Epoch 166/300
10/10 - 0s - loss: 1.2515 - val_loss: 1.3407
Epoch 167/300
10/10 - 0s - loss: 1.2431 - val_loss: 1.4075
Epoch 168/300
10/10 - 0s - loss: 1.2534 - val_loss: 1.3128
Epoch 169/300
10/10 - 0s - loss: 1.2159 - val_loss: 1.3614
Epoch 170/300
10/10 - 0s - loss: 1.2591 - val_loss: 1.3247
Epoch 171/300
10/10 - 0s - loss: 1.2424 - val_loss: 1.3186
Epoch 172/300
10/10 - 0s - loss: 1.2218 - val_loss: 1.3259
Epoch 173/300
10/10 - 0s - loss: 1.2328 - val_loss: 1.3401
Epoch 174/300
10/10 - 0s - loss: 1.2168 - val_loss: 1.3092
Epoch 175/300
10/10 - 0s - loss: 1.2779 - val_loss: 1.3349
Epoch 176/300
10/10 - 0s - loss: 1.2560 - val_loss: 1.3331
Epoch 177/300
10/10 - 0s - loss: 1.2445 - val_loss: 1.3119
Epoch 178/300
10/10 - 0s - loss: 1.2250 - val_loss: 1.3168
Epoch 179/300
10/10 - 0s - loss: 1.2139 - val_loss: 1.3217
Epoch 180/300
10/10 - 0s - loss: 1.2020 - val_loss: 1.2753
Epoch 181/300
10/10 - 0s - loss: 1.1906 - val_loss: 1.2765
Epoch 182/300
10/10 - 0s - loss: 1.2045 - val_loss: 1.2821
Epoch 183/300
10/10 - 0s - loss: 1.2229 - val_loss: 1.2810
Epoch 184/300
10/10 - 0s - loss: 1.1967 - val_loss: 1.3295
Epoch 185/300
10/10 - 0s - loss: 1.1852 - val_loss: 1.2866
Epoch 186/300
10/10 - 0s - loss: 1.1941 - val_loss: 1.3126
Epoch 187/300
10/10 - 0s - loss: 1.1783 - val_loss: 1.3282
Epoch 188/300
10/10 - 0s - loss: 1.1758 - val_loss: 1.2702
Epoch 189/300
10/10 - 0s - loss: 1.1763 - val_loss: 1.2694
Epoch 190/300
10/10 - 0s - loss: 1.1802 - val_loss: 1.3377
Epoch 191/300
10/10 - 0s - loss: 1.1989 - val_loss: 1.2996
Epoch 192/300
10/10 - 0s - loss: 1.1998 - val_loss: 1.2948
Epoch 193/300
10/10 - 0s - loss: 1.1977 - val_loss: 1.3324
Epoch 194/300
10/10 - 0s - loss: 1.1756 - val_loss: 1.3388
Epoch 195/300
10/10 - 0s - loss: 1.1738 - val_loss: 1.3121
Epoch 196/300
10/10 - 0s - loss: 1.1752 - val_loss: 1.2886
Epoch 197/300
10/10 - 0s - loss: 1.1894 - val_loss: 1.2996
Epoch 198/300
10/10 - 0s - loss: 1.1771 - val_loss: 1.2697
Epoch 199/300
10/10 - 0s - loss: 1.1741 - val_loss: 1.2830
Epoch 200/300
10/10 - 0s - loss: 1.1775 - val_loss: 1.3095
Epoch 201/300
10/10 - 0s - loss: 1.1814 - val_loss: 1.2873
Epoch 202/300
10/10 - 0s - loss: 1.1782 - val_loss: 1.2748
Epoch 203/300
10/10 - 0s - loss: 1.1623 - val_loss: 1.2861
Epoch 204/300
10/10 - 0s - loss: 1.1691 - val_loss: 1.2960
Epoch 205/300
10/10 - 0s - loss: 1.1722 - val_loss: 1.3015
Epoch 206/300
10/10 - 0s - loss: 1.2002 - val_loss: 1.2970
Epoch 207/300
10/10 - 0s - loss: 1.1916 - val_loss: 1.3317
Epoch 208/300
10/10 - 0s - loss: 1.1938 - val_loss: 1.3479
Epoch 209/300
10/10 - 0s - loss: 1.2207 - val_loss: 1.2718
Epoch 210/300
10/10 - 0s - loss: 1.1927 - val_loss: 1.2947
Epoch 211/300
10/10 - 0s - loss: 1.1799 - val_loss: 1.2910
Epoch 212/300
10/10 - 0s - loss: 1.1877 - val_loss: 1.3001
Epoch 213/300
10/10 - 0s - loss: 1.1671 - val_loss: 1.2740
Epoch 214/300
10/10 - 0s - loss: 1.2021 - val_loss: 1.3010
Epoch 215/300
10/10 - 0s - loss: 1.1937 - val_loss: 1.2906
Epoch 216/300
10/10 - 0s - loss: 1.1659 - val_loss: 1.2879
Epoch 217/300
10/10 - 0s - loss: 1.1914 - val_loss: 1.2839
Epoch 218/300
10/10 - 0s - loss: 1.1787 - val_loss: 1.2966
Epoch 219/300
10/10 - 0s - loss: 1.1651 - val_loss: 1.2927
Epoch 220/300
10/10 - 0s - loss: 1.1803 - val_loss: 1.2818
Epoch 221/300
10/10 - 0s - loss: 1.1701 - val_loss: 1.2787
Epoch 222/300
10/10 - 0s - loss: 1.2009 - val_loss: 1.3056
Epoch 223/300
10/10 - 0s - loss: 1.1741 - val_loss: 1.3055
Epoch 224/300
10/10 - 0s - loss: 1.1955 - val_loss: 1.3187
Epoch 225/300
10/10 - 0s - loss: 1.2137 - val_loss: 1.2908
Epoch 226/300
10/10 - 0s - loss: 1.1723 - val_loss: 1.2808
Epoch 227/300
10/10 - 0s - loss: 1.1682 - val_loss: 1.2974
Epoch 228/300
10/10 - 0s - loss: 1.1569 - val_loss: 1.3180
Epoch 229/300
10/10 - 0s - loss: 1.1848 - val_loss: 1.2840
Epoch 230/300
10/10 - 0s - loss: 1.1912 - val_loss: 1.2940
Epoch 231/300
10/10 - 0s - loss: 1.1633 - val_loss: 1.2905
Epoch 232/300
10/10 - 0s - loss: 1.1539 - val_loss: 1.2985
Epoch 233/300
10/10 - 0s - loss: 1.1574 - val_loss: 1.2750
Epoch 234/300
10/10 - 0s - loss: 1.1555 - val_loss: 1.2690
Epoch 235/300
10/10 - 0s - loss: 1.1519 - val_loss: 1.2961
Epoch 236/300
10/10 - 0s - loss: 1.1763 - val_loss: 1.2750
Epoch 237/300
10/10 - 0s - loss: 1.1670 - val_loss: 1.3295
Epoch 238/300
10/10 - 0s - loss: 1.1574 - val_loss: 1.2904
Epoch 239/300
10/10 - 0s - loss: 1.1588 - val_loss: 1.3034
Epoch 240/300
10/10 - 0s - loss: 1.1630 - val_loss: 1.2803
Epoch 241/300
10/10 - 0s - loss: 1.1688 - val_loss: 1.2860
Epoch 242/300
10/10 - 0s - loss: 1.1730 - val_loss: 1.3309
Epoch 243/300
10/10 - 0s - loss: 1.2057 - val_loss: 1.3330
Epoch 244/300
10/10 - 0s - loss: 1.1706 - val_loss: 1.3037
Epoch 245/300
10/10 - 0s - loss: 1.1526 - val_loss: 1.2910
Epoch 246/300
10/10 - 0s - loss: 1.1625 - val_loss: 1.2869
Epoch 247/300
10/10 - 0s - loss: 1.1555 - val_loss: 1.3253
Epoch 248/300
10/10 - 0s - loss: 1.1527 - val_loss: 1.3349
Epoch 249/300
10/10 - 0s - loss: 1.1544 - val_loss: 1.2894
Epoch 250/300
10/10 - 0s - loss: 1.1434 - val_loss: 1.2844
Epoch 251/300
10/10 - 0s - loss: 1.1479 - val_loss: 1.3500
Epoch 252/300
10/10 - 0s - loss: 1.1594 - val_loss: 1.3206
Epoch 253/300
10/10 - 0s - loss: 1.1975 - val_loss: 1.2897
Epoch 254/300
10/10 - 0s - loss: 1.1800 - val_loss: 1.2983
Epoch 255/300
10/10 - 0s - loss: 1.1656 - val_loss: 1.2979
Epoch 256/300
10/10 - 0s - loss: 1.1658 - val_loss: 1.3044
Epoch 257/300
10/10 - 0s - loss: 1.1665 - val_loss: 1.2955
Epoch 258/300
10/10 - 0s - loss: 1.1577 - val_loss: 1.2998
Epoch 259/300
10/10 - 0s - loss: 1.1625 - val_loss: 1.3247
Epoch 260/300
10/10 - 0s - loss: 1.1652 - val_loss: 1.3172
Epoch 261/300
10/10 - 0s - loss: 1.1551 - val_loss: 1.2899
Epoch 262/300
10/10 - 0s - loss: 1.1433 - val_loss: 1.2832
Epoch 263/300
10/10 - 0s - loss: 1.1498 - val_loss: 1.2781
Epoch 264/300
10/10 - 0s - loss: 1.1599 - val_loss: 1.3124
Epoch 265/300
10/10 - 0s - loss: 1.1693 - val_loss: 1.2873
Epoch 266/300
10/10 - 0s - loss: 1.1663 - val_loss: 1.2625
Epoch 267/300
10/10 - 0s - loss: 1.1706 - val_loss: 1.2935
Epoch 268/300
10/10 - 0s - loss: 1.1641 - val_loss: 1.2688
Epoch 269/300
10/10 - 0s - loss: 1.1564 - val_loss: 1.2748
Epoch 270/300
10/10 - 0s - loss: 1.1558 - val_loss: 1.2903
Epoch 271/300
10/10 - 0s - loss: 1.1699 - val_loss: 1.3047
Epoch 272/300
10/10 - 0s - loss: 1.1511 - val_loss: 1.3155
Epoch 273/300
10/10 - 0s - loss: 1.1574 - val_loss: 1.3227
Epoch 274/300
10/10 - 0s - loss: 1.2026 - val_loss: 1.2986
Epoch 275/300
10/10 - 0s - loss: 1.1880 - val_loss: 1.3880
Epoch 276/300
10/10 - 0s - loss: 1.1912 - val_loss: 1.3257
Epoch 277/300
10/10 - 0s - loss: 1.2500 - val_loss: 1.3678
Epoch 278/300
10/10 - 0s - loss: 1.2577 - val_loss: 1.3459
Epoch 279/300
10/10 - 0s - loss: 1.2060 - val_loss: 1.3124
Epoch 280/300
10/10 - 0s - loss: 1.1785 - val_loss: 1.2839
Epoch 281/300
10/10 - 0s - loss: 1.1617 - val_loss: 1.2958
Epoch 282/300
10/10 - 0s - loss: 1.1535 - val_loss: 1.2837
Epoch 283/300
10/10 - 0s - loss: 1.1544 - val_loss: 1.2685
Epoch 284/300
10/10 - 0s - loss: 1.1444 - val_loss: 1.2963
Epoch 285/300
10/10 - 0s - loss: 1.1540 - val_loss: 1.3266
Epoch 286/300
10/10 - 0s - loss: 1.1817 - val_loss: 1.2867
Epoch 287/300
10/10 - 0s - loss: 1.1504 - val_loss: 1.2798
Epoch 288/300
10/10 - 0s - loss: 1.1495 - val_loss: 1.3050
Epoch 289/300
10/10 - 0s - loss: 1.1667 - val_loss: 1.2821
Epoch 290/300
10/10 - 0s - loss: 1.1761 - val_loss: 1.3154
Epoch 291/300
10/10 - 0s - loss: 1.1608 - val_loss: 1.3160
Epoch 292/300
10/10 - 0s - loss: 1.1688 - val_loss: 1.3394
Epoch 293/300
10/10 - 0s - loss: 1.1595 - val_loss: 1.3182
Epoch 294/300
10/10 - 0s - loss: 1.1630 - val_loss: 1.3249
Epoch 295/300
10/10 - 0s - loss: 1.1427 - val_loss: 1.3061
Epoch 296/300
10/10 - 0s - loss: 1.1473 - val_loss: 1.2985
Epoch 297/300
10/10 - 0s - loss: 1.1393 - val_loss: 1.3054
Epoch 298/300
10/10 - 0s - loss: 1.1641 - val_loss: 1.3133
Epoch 299/300
10/10 - 0s - loss: 1.1740 - val_loss: 1.2902
Epoch 300/300
10/10 - 0s - loss: 1.1717 - val_loss: 1.2780

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
(-2.0, 2.0)

```
</div>
![png](/img/examples/generative/real_nvp/real_nvp_13_1.png)



![png](/img/examples/generative/real_nvp/real_nvp_13_2.png)


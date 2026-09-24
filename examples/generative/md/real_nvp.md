# Density estimation using Real NVP

**Authors:** [Mandolini Giorgio Maria](https://www.linkedin.com/in/giorgio-maria-mandolini-a2a1b71b4/), [Sanna Daniele](https://www.linkedin.com/in/daniele-sanna-338629bb/), [Zannini Quirini Giorgio](https://www.linkedin.com/in/giorgio-zannini-quirini-16ab181a0/)<br>
**Date created:** 2020/08/10<br>
**Last modified:** 2026/03/23<br>
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

**Reference:**

[Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)

---
## Setup


```python
import os

# Set backend to JAX, PyTorch, or TensorFlow
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers
from keras import ops
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
```

---
## Load the data


```python
# make_moons(3000, noise=0.05): 3000 samples with Gaussian noise level 0.05;
# [0] selects feature coordinates (X) and drops labels (y).
data = make_moons(3000, noise=0.05)[0].astype("float32")
norm = layers.Normalization()
norm.adapt(data)
normalized_data = norm(data)
```

---
## Affine coupling layer


```python
COUPLING_HIDDEN_UNITS = 256
COUPLING_MLP_LAYERS = 4
COUPLING_L2_WEIGHT = 0.01


def Coupling(input_shape):
    input_layer = layers.Input(shape=(input_shape,))

    def mlp(x):
        for _ in range(COUPLING_MLP_LAYERS):
            x = layers.Dense(
                COUPLING_HIDDEN_UNITS,
                activation="relu",
                kernel_regularizer=regularizers.l2(COUPLING_L2_WEIGHT),
            )(x)
        return x

    # Scale and translation parameters
    shared = mlp(input_layer)
    scale = layers.Dense(
        input_shape,
        activation="tanh",
        kernel_regularizer=regularizers.l2(COUPLING_L2_WEIGHT),
    )(shared)
    translation = layers.Dense(
        input_shape,
        activation="linear",
        kernel_regularizer=regularizers.l2(COUPLING_L2_WEIGHT),
    )(shared)
    return keras.Model(inputs=input_layer, outputs=[scale, translation])

```

---
## Real NVP

Real NVP stacks invertible affine coupling layers to transform data space (x)
and latent space (z).

In each coupling layer, one subset of features is kept fixed by a mask, while
the other subset is scaled and shifted:
z_part = x_part * exp(scale) + translation

Because each layer is invertible and has a tractable Jacobian determinant,
we can compute exact log-likelihood using the change-of-variables formula.
In this implementation:
- training=True maps data -> latent (x -> z)
- training=False maps latent -> data (z -> x)


```python

class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.masks = ops.convert_to_tensor(
            np.array([[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype="float32")
        )
        self.coupling_layers = [Coupling(2) for _ in range(num_coupling_layers)]

    def log_prob_std_normal(self, z):
        d = ops.cast(ops.shape(z)[-1], "float32")
        log2pi = ops.cast(np.log(2.0 * np.pi), "float32")
        return -0.5 * (d * log2pi + ops.sum(ops.square(z), axis=-1))

    def call(self, x, training=False):
        log_det_inv = 0
        direction = -1.0 if training else 1.0
        layer_indices = range(self.num_coupling_layers)
        if training:
            layer_indices = reversed(layer_indices)

        for i in layer_indices:
            x_masked = x * self.masks[i]
            reversed_mask = 1.0 - self.masks[i]
            scale, translation = self.coupling_layers[i](x_masked)
            scale *= reversed_mask
            translation *= reversed_mask
            gate = (direction - 1.0) / 2.0
            x = (
                reversed_mask
                * (
                    x * ops.exp(direction * scale)
                    + direction * translation * ops.exp(gate * scale)
                )
                + x_masked
            )
            log_det_inv += gate * ops.sum(scale, axis=1)
        return x, log_det_inv

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        z, logdet = y_pred
        log_likelihood = self.log_prob_std_normal(z) + logdet
        main_loss = -ops.mean(log_likelihood)

        # Manually sum the L2 losses from the coupling layers
        # Ensure reg_losses is a Keras tensor, even if self.losses is empty
        if self.losses:
            # Stack the losses into a single tensor and then sum them up
            reg_losses = ops.sum(ops.stack(self.losses))
        else:
            reg_losses = ops.convert_to_tensor(0.0, dtype="float32")

        return main_loss + reg_losses

```

---
## Model training


```python
model = RealNVP(num_coupling_layers=6)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

shuffle_indices = np.random.permutation(len(normalized_data))
normalized_data = normalized_data[shuffle_indices]

# Now fit
history = model.fit(
    normalized_data, batch_size=256, epochs=300, verbose=2, validation_split=0.2
)
```

<div class="k-default-codeblock">
```
Epoch 1/300

10/10 - 2s - 157ms/step - loss: 49.0175 - val_loss: 48.4240

Epoch 2/300

10/10 - 0s - 11ms/step - loss: 47.7146 - val_loss: 47.2222

Epoch 3/300

10/10 - 0s - 11ms/step - loss: 46.4805 - val_loss: 46.0394

Epoch 4/300

10/10 - 0s - 11ms/step - loss: 45.2807 - val_loss: 44.9248

Epoch 5/300

10/10 - 0s - 11ms/step - loss: 44.1114 - val_loss: 43.7890

Epoch 6/300

10/10 - 0s - 11ms/step - loss: 42.9730 - val_loss: 42.6985

Epoch 7/300

10/10 - 0s - 11ms/step - loss: 41.8634 - val_loss: 41.6346

Epoch 8/300

10/10 - 0s - 11ms/step - loss: 40.7819 - val_loss: 40.5966

Epoch 9/300

10/10 - 0s - 11ms/step - loss: 39.7299 - val_loss: 39.5842

Epoch 10/300

10/10 - 0s - 11ms/step - loss: 38.7046 - val_loss: 38.5952

Epoch 11/300

10/10 - 0s - 11ms/step - loss: 37.7057 - val_loss: 37.6284

Epoch 12/300

10/10 - 0s - 11ms/step - loss: 36.7332 - val_loss: 36.6842

Epoch 13/300

10/10 - 0s - 11ms/step - loss: 35.7860 - val_loss: 35.7688

Epoch 14/300

10/10 - 0s - 11ms/step - loss: 34.8616 - val_loss: 34.8609

Epoch 15/300

10/10 - 0s - 11ms/step - loss: 33.9601 - val_loss: 33.9874

Epoch 16/300

10/10 - 0s - 11ms/step - loss: 33.0829 - val_loss: 33.1279

Epoch 17/300

10/10 - 0s - 11ms/step - loss: 32.2277 - val_loss: 32.2967

Epoch 18/300

10/10 - 0s - 11ms/step - loss: 31.3920 - val_loss: 31.4831

Epoch 19/300

10/10 - 0s - 11ms/step - loss: 30.5777 - val_loss: 30.6943

Epoch 20/300

10/10 - 0s - 11ms/step - loss: 29.7853 - val_loss: 29.9239

Epoch 21/300

10/10 - 0s - 11ms/step - loss: 29.0120 - val_loss: 29.1746

Epoch 22/300

10/10 - 0s - 11ms/step - loss: 28.2591 - val_loss: 28.4414

Epoch 23/300

10/10 - 0s - 11ms/step - loss: 27.5255 - val_loss: 27.7340

Epoch 24/300

10/10 - 0s - 11ms/step - loss: 26.8113 - val_loss: 27.0393

Epoch 25/300

10/10 - 0s - 11ms/step - loss: 26.1175 - val_loss: 26.3662

Epoch 26/300

10/10 - 0s - 11ms/step - loss: 25.4441 - val_loss: 25.7099

Epoch 27/300

10/10 - 0s - 11ms/step - loss: 24.7811 - val_loss: 25.0861

Epoch 28/300

10/10 - 0s - 11ms/step - loss: 24.1436 - val_loss: 24.4741

Epoch 29/300

10/10 - 0s - 11ms/step - loss: 23.5179 - val_loss: 23.8532

Epoch 30/300

10/10 - 0s - 11ms/step - loss: 22.9107 - val_loss: 23.2619

Epoch 31/300

10/10 - 0s - 11ms/step - loss: 22.3190 - val_loss: 22.7003

Epoch 32/300

10/10 - 0s - 11ms/step - loss: 21.7379 - val_loss: 22.1518

Epoch 33/300

10/10 - 0s - 11ms/step - loss: 21.1776 - val_loss: 21.6212

Epoch 34/300

10/10 - 0s - 11ms/step - loss: 20.6259 - val_loss: 21.0966

Epoch 35/300

10/10 - 0s - 11ms/step - loss: 20.0921 - val_loss: 20.5893

Epoch 36/300

10/10 - 0s - 11ms/step - loss: 19.5697 - val_loss: 20.0977

Epoch 37/300

10/10 - 0s - 11ms/step - loss: 19.0629 - val_loss: 19.6424

Epoch 38/300

10/10 - 0s - 11ms/step - loss: 18.5709 - val_loss: 19.1837

Epoch 39/300

10/10 - 0s - 11ms/step - loss: 18.0888 - val_loss: 18.6942

Epoch 40/300

10/10 - 0s - 11ms/step - loss: 17.6274 - val_loss: 18.2833

Epoch 41/300

10/10 - 0s - 11ms/step - loss: 17.1687 - val_loss: 17.8493

Epoch 42/300

10/10 - 0s - 11ms/step - loss: 16.7301 - val_loss: 17.4660

Epoch 43/300

10/10 - 0s - 11ms/step - loss: 16.2986 - val_loss: 17.0411

Epoch 44/300

10/10 - 0s - 11ms/step - loss: 15.8840 - val_loss: 16.6611

Epoch 45/300

10/10 - 0s - 11ms/step - loss: 15.4783 - val_loss: 16.2950

Epoch 46/300

10/10 - 0s - 11ms/step - loss: 15.0839 - val_loss: 15.8930

Epoch 47/300

10/10 - 0s - 11ms/step - loss: 14.6967 - val_loss: 15.5284

Epoch 48/300

10/10 - 0s - 11ms/step - loss: 14.3449 - val_loss: 15.1774

Epoch 49/300

10/10 - 0s - 11ms/step - loss: 13.9690 - val_loss: 14.8690

Epoch 50/300

10/10 - 0s - 11ms/step - loss: 13.6098 - val_loss: 14.5874

Epoch 51/300

10/10 - 0s - 11ms/step - loss: 13.2662 - val_loss: 14.2679

Epoch 52/300

10/10 - 0s - 11ms/step - loss: 12.9477 - val_loss: 13.9899

Epoch 53/300

10/10 - 0s - 11ms/step - loss: 12.6383 - val_loss: 13.6155

Epoch 54/300

10/10 - 0s - 11ms/step - loss: 12.3277 - val_loss: 13.3185

Epoch 55/300

10/10 - 0s - 11ms/step - loss: 12.0231 - val_loss: 13.0721

Epoch 56/300

10/10 - 0s - 11ms/step - loss: 11.7343 - val_loss: 12.7888

Epoch 57/300

10/10 - 0s - 11ms/step - loss: 11.4593 - val_loss: 12.5280

Epoch 58/300

10/10 - 0s - 11ms/step - loss: 11.1800 - val_loss: 12.2650

Epoch 59/300

10/10 - 0s - 11ms/step - loss: 10.9463 - val_loss: 11.9898

Epoch 60/300

10/10 - 0s - 11ms/step - loss: 10.6631 - val_loss: 11.7316

Epoch 61/300

10/10 - 0s - 11ms/step - loss: 10.4205 - val_loss: 11.5557

Epoch 62/300

10/10 - 0s - 11ms/step - loss: 10.1809 - val_loss: 11.3534

Epoch 63/300

10/10 - 0s - 12ms/step - loss: 9.9457 - val_loss: 11.1059

Epoch 64/300

10/10 - 0s - 11ms/step - loss: 9.7153 - val_loss: 10.8720

Epoch 65/300

10/10 - 0s - 11ms/step - loss: 9.5069 - val_loss: 10.6568

Epoch 66/300

10/10 - 0s - 11ms/step - loss: 9.2999 - val_loss: 10.5041

Epoch 67/300

10/10 - 0s - 11ms/step - loss: 9.1074 - val_loss: 10.2816

Epoch 68/300

10/10 - 0s - 11ms/step - loss: 8.8768 - val_loss: 10.1193

Epoch 69/300

10/10 - 0s - 11ms/step - loss: 8.6935 - val_loss: 9.9112

Epoch 70/300

10/10 - 0s - 11ms/step - loss: 8.5114 - val_loss: 9.7768

Epoch 71/300

10/10 - 0s - 11ms/step - loss: 8.3168 - val_loss: 9.5499

Epoch 72/300

10/10 - 0s - 11ms/step - loss: 8.1461 - val_loss: 9.3909

Epoch 73/300

10/10 - 0s - 11ms/step - loss: 7.9696 - val_loss: 9.2185

Epoch 74/300

10/10 - 0s - 11ms/step - loss: 7.8009 - val_loss: 9.0605

Epoch 75/300

10/10 - 0s - 11ms/step - loss: 7.6542 - val_loss: 8.8743

Epoch 76/300

10/10 - 0s - 11ms/step - loss: 7.5275 - val_loss: 8.7208

Epoch 77/300

10/10 - 0s - 11ms/step - loss: 7.3482 - val_loss: 8.5794

Epoch 78/300

10/10 - 0s - 11ms/step - loss: 7.2124 - val_loss: 8.4322

Epoch 79/300

10/10 - 0s - 11ms/step - loss: 7.0606 - val_loss: 8.3742

Epoch 80/300

10/10 - 0s - 11ms/step - loss: 6.9202 - val_loss: 8.2034

Epoch 81/300

10/10 - 0s - 11ms/step - loss: 6.8098 - val_loss: 8.0811

Epoch 82/300

10/10 - 0s - 12ms/step - loss: 6.6593 - val_loss: 7.9904

Epoch 83/300

10/10 - 0s - 12ms/step - loss: 6.5282 - val_loss: 7.8095

Epoch 84/300

10/10 - 0s - 12ms/step - loss: 6.4125 - val_loss: 7.7400

Epoch 85/300

10/10 - 0s - 11ms/step - loss: 6.3065 - val_loss: 7.6289

Epoch 86/300

10/10 - 0s - 11ms/step - loss: 6.1958 - val_loss: 7.4956

Epoch 87/300

10/10 - 0s - 11ms/step - loss: 6.0744 - val_loss: 7.4377

Epoch 88/300

10/10 - 0s - 11ms/step - loss: 5.9751 - val_loss: 7.3066

Epoch 89/300

10/10 - 0s - 11ms/step - loss: 5.8538 - val_loss: 7.1792

Epoch 90/300

10/10 - 0s - 11ms/step - loss: 5.7497 - val_loss: 7.1219

Epoch 91/300

10/10 - 0s - 11ms/step - loss: 5.6696 - val_loss: 6.9557

Epoch 92/300

10/10 - 0s - 11ms/step - loss: 5.5611 - val_loss: 6.8829

Epoch 93/300

10/10 - 0s - 11ms/step - loss: 5.4578 - val_loss: 6.8266

Epoch 94/300

10/10 - 0s - 11ms/step - loss: 5.3660 - val_loss: 6.7422

Epoch 95/300

10/10 - 0s - 11ms/step - loss: 5.2799 - val_loss: 6.6761

Epoch 96/300

10/10 - 0s - 11ms/step - loss: 5.1968 - val_loss: 6.6043

Epoch 97/300

10/10 - 0s - 11ms/step - loss: 5.1310 - val_loss: 6.5088

Epoch 98/300

10/10 - 0s - 11ms/step - loss: 5.0496 - val_loss: 6.4587

Epoch 99/300

10/10 - 0s - 11ms/step - loss: 4.9694 - val_loss: 6.3452

Epoch 100/300

10/10 - 0s - 11ms/step - loss: 4.8850 - val_loss: 6.3160

Epoch 101/300

10/10 - 0s - 11ms/step - loss: 4.8368 - val_loss: 6.2213

Epoch 102/300

10/10 - 0s - 11ms/step - loss: 4.7552 - val_loss: 6.0851

Epoch 103/300

10/10 - 0s - 11ms/step - loss: 4.6810 - val_loss: 6.0580

Epoch 104/300

10/10 - 0s - 11ms/step - loss: 4.6138 - val_loss: 6.0430

Epoch 105/300

10/10 - 0s - 11ms/step - loss: 4.5489 - val_loss: 5.9949

Epoch 106/300

10/10 - 0s - 11ms/step - loss: 4.4998 - val_loss: 5.9037

Epoch 107/300

10/10 - 0s - 11ms/step - loss: 4.4424 - val_loss: 5.7772

Epoch 108/300

10/10 - 0s - 11ms/step - loss: 4.4201 - val_loss: 5.7635

Epoch 109/300

10/10 - 0s - 11ms/step - loss: 4.3238 - val_loss: 5.7433

Epoch 110/300

10/10 - 0s - 11ms/step - loss: 4.2522 - val_loss: 5.6838

Epoch 111/300

10/10 - 0s - 11ms/step - loss: 4.2054 - val_loss: 5.6344

Epoch 112/300

10/10 - 0s - 11ms/step - loss: 4.1483 - val_loss: 5.5675

Epoch 113/300

10/10 - 0s - 11ms/step - loss: 4.1000 - val_loss: 5.5242

Epoch 114/300

10/10 - 0s - 11ms/step - loss: 4.0531 - val_loss: 5.5297

Epoch 115/300

10/10 - 0s - 11ms/step - loss: 4.0132 - val_loss: 5.4493

Epoch 116/300

10/10 - 0s - 11ms/step - loss: 3.9585 - val_loss: 5.3879

Epoch 117/300

10/10 - 0s - 11ms/step - loss: 3.9139 - val_loss: 5.3577

Epoch 118/300

10/10 - 0s - 11ms/step - loss: 3.8683 - val_loss: 5.3061

Epoch 119/300

10/10 - 0s - 11ms/step - loss: 3.8280 - val_loss: 5.2842

Epoch 120/300

10/10 - 0s - 11ms/step - loss: 3.8004 - val_loss: 5.2665

Epoch 121/300

10/10 - 0s - 11ms/step - loss: 3.7680 - val_loss: 5.2492

Epoch 122/300

10/10 - 0s - 11ms/step - loss: 3.7431 - val_loss: 5.1860

Epoch 123/300

10/10 - 0s - 11ms/step - loss: 3.6904 - val_loss: 5.0876

Epoch 124/300

10/10 - 0s - 11ms/step - loss: 3.6604 - val_loss: 5.0622

Epoch 125/300

10/10 - 0s - 12ms/step - loss: 3.6317 - val_loss: 5.0914

Epoch 126/300

10/10 - 0s - 11ms/step - loss: 3.5957 - val_loss: 5.0557

Epoch 127/300

10/10 - 0s - 14ms/step - loss: 3.5505 - val_loss: 4.9895

Epoch 128/300

10/10 - 0s - 11ms/step - loss: 3.5228 - val_loss: 4.9688

Epoch 129/300

10/10 - 0s - 11ms/step - loss: 3.4874 - val_loss: 4.9063

Epoch 130/300

10/10 - 0s - 11ms/step - loss: 3.4703 - val_loss: 4.9330

Epoch 131/300

10/10 - 0s - 12ms/step - loss: 3.4419 - val_loss: 4.8896

Epoch 132/300

10/10 - 0s - 11ms/step - loss: 3.4052 - val_loss: 4.8758

Epoch 133/300

10/10 - 0s - 11ms/step - loss: 3.3902 - val_loss: 4.8430

Epoch 134/300

10/10 - 0s - 11ms/step - loss: 3.3655 - val_loss: 4.8364

Epoch 135/300

10/10 - 0s - 11ms/step - loss: 3.3374 - val_loss: 4.7951

Epoch 136/300

10/10 - 0s - 12ms/step - loss: 3.3200 - val_loss: 4.7908

Epoch 137/300

10/10 - 0s - 12ms/step - loss: 3.2763 - val_loss: 4.7286

Epoch 138/300

10/10 - 0s - 11ms/step - loss: 3.2599 - val_loss: 4.7468

Epoch 139/300

10/10 - 0s - 12ms/step - loss: 3.2269 - val_loss: 4.7081

Epoch 140/300

10/10 - 0s - 11ms/step - loss: 3.2102 - val_loss: 4.6803

Epoch 141/300

10/10 - 0s - 11ms/step - loss: 3.1892 - val_loss: 4.6475

Epoch 142/300

10/10 - 0s - 11ms/step - loss: 3.1806 - val_loss: 4.6445

Epoch 143/300

10/10 - 0s - 11ms/step - loss: 3.1510 - val_loss: 4.6643

Epoch 144/300

10/10 - 0s - 11ms/step - loss: 3.1397 - val_loss: 4.6192

Epoch 145/300

10/10 - 0s - 11ms/step - loss: 3.1191 - val_loss: 4.5993

Epoch 146/300

10/10 - 0s - 12ms/step - loss: 3.0980 - val_loss: 4.5785

Epoch 147/300

10/10 - 0s - 12ms/step - loss: 3.0742 - val_loss: 4.5396

Epoch 148/300

10/10 - 0s - 12ms/step - loss: 3.0888 - val_loss: 4.4919

Epoch 149/300

10/10 - 0s - 11ms/step - loss: 3.1123 - val_loss: 4.4981

Epoch 150/300

10/10 - 0s - 11ms/step - loss: 3.0742 - val_loss: 4.5085

Epoch 151/300

10/10 - 0s - 11ms/step - loss: 3.0455 - val_loss: 4.5119

Epoch 152/300

10/10 - 0s - 12ms/step - loss: 3.0032 - val_loss: 4.4645

Epoch 153/300

10/10 - 0s - 12ms/step - loss: 2.9870 - val_loss: 4.4632

Epoch 154/300

10/10 - 0s - 12ms/step - loss: 2.9672 - val_loss: 4.4231

Epoch 155/300

10/10 - 0s - 11ms/step - loss: 2.9630 - val_loss: 4.4289

Epoch 156/300

10/10 - 0s - 12ms/step - loss: 2.9529 - val_loss: 4.4117

Epoch 157/300

10/10 - 0s - 12ms/step - loss: 2.9379 - val_loss: 4.4015

Epoch 158/300

10/10 - 0s - 12ms/step - loss: 2.9390 - val_loss: 4.3516

Epoch 159/300

10/10 - 0s - 12ms/step - loss: 2.9323 - val_loss: 4.3800

Epoch 160/300

10/10 - 0s - 12ms/step - loss: 2.9128 - val_loss: 4.4267

Epoch 161/300

10/10 - 0s - 12ms/step - loss: 2.8947 - val_loss: 4.3629

Epoch 162/300

10/10 - 0s - 11ms/step - loss: 2.8669 - val_loss: 4.3716

Epoch 163/300

10/10 - 0s - 12ms/step - loss: 2.8690 - val_loss: 4.3452

Epoch 164/300

10/10 - 0s - 12ms/step - loss: 2.8622 - val_loss: 4.3003

Epoch 165/300

10/10 - 0s - 11ms/step - loss: 2.8375 - val_loss: 4.3311

Epoch 166/300

10/10 - 0s - 11ms/step - loss: 2.8299 - val_loss: 4.3363

Epoch 167/300

10/10 - 0s - 12ms/step - loss: 2.8205 - val_loss: 4.3158

Epoch 168/300

10/10 - 0s - 12ms/step - loss: 2.8058 - val_loss: 4.3415

Epoch 169/300

10/10 - 0s - 12ms/step - loss: 2.7937 - val_loss: 4.2824

Epoch 170/300

10/10 - 0s - 12ms/step - loss: 2.7869 - val_loss: 4.2729

Epoch 171/300

10/10 - 0s - 12ms/step - loss: 2.7838 - val_loss: 4.2572

Epoch 172/300

10/10 - 0s - 12ms/step - loss: 2.7770 - val_loss: 4.2663

Epoch 173/300

10/10 - 0s - 12ms/step - loss: 2.7614 - val_loss: 4.2912

Epoch 174/300

10/10 - 0s - 12ms/step - loss: 2.7554 - val_loss: 4.2525

Epoch 175/300

10/10 - 0s - 12ms/step - loss: 2.7531 - val_loss: 4.2471

Epoch 176/300

10/10 - 0s - 12ms/step - loss: 2.7482 - val_loss: 4.2644

Epoch 177/300

10/10 - 0s - 12ms/step - loss: 2.7366 - val_loss: 4.2912

Epoch 178/300

10/10 - 0s - 12ms/step - loss: 2.7463 - val_loss: 4.2820

Epoch 179/300

10/10 - 0s - 12ms/step - loss: 2.7402 - val_loss: 4.2234

Epoch 180/300

10/10 - 0s - 12ms/step - loss: 2.7169 - val_loss: 4.1953

Epoch 181/300

10/10 - 0s - 12ms/step - loss: 2.6974 - val_loss: 4.2176

Epoch 182/300

10/10 - 0s - 12ms/step - loss: 2.7045 - val_loss: 4.1877

Epoch 183/300

10/10 - 0s - 12ms/step - loss: 2.6900 - val_loss: 4.1763

Epoch 184/300

10/10 - 0s - 12ms/step - loss: 2.6943 - val_loss: 4.2212

Epoch 185/300

10/10 - 0s - 12ms/step - loss: 2.6842 - val_loss: 4.1738

Epoch 186/300

10/10 - 0s - 12ms/step - loss: 2.6712 - val_loss: 4.1703

Epoch 187/300

10/10 - 0s - 12ms/step - loss: 2.6921 - val_loss: 4.2083

Epoch 188/300

10/10 - 0s - 12ms/step - loss: 2.6740 - val_loss: 4.1922

Epoch 189/300

10/10 - 0s - 12ms/step - loss: 2.6708 - val_loss: 4.1346

Epoch 190/300

10/10 - 0s - 12ms/step - loss: 2.6565 - val_loss: 4.1553

Epoch 191/300

10/10 - 0s - 12ms/step - loss: 2.6476 - val_loss: 4.1539

Epoch 192/300

10/10 - 0s - 12ms/step - loss: 2.6468 - val_loss: 4.1301

Epoch 193/300

10/10 - 0s - 12ms/step - loss: 2.6294 - val_loss: 4.1160

Epoch 194/300

10/10 - 0s - 12ms/step - loss: 2.6217 - val_loss: 4.1498

Epoch 195/300

10/10 - 0s - 12ms/step - loss: 2.6147 - val_loss: 4.1707

Epoch 196/300

10/10 - 0s - 12ms/step - loss: 2.6090 - val_loss: 4.1487

Epoch 197/300

10/10 - 0s - 12ms/step - loss: 2.5961 - val_loss: 4.1108

Epoch 198/300

10/10 - 0s - 12ms/step - loss: 2.5996 - val_loss: 4.1376

Epoch 199/300

10/10 - 0s - 12ms/step - loss: 2.5981 - val_loss: 4.1390

Epoch 200/300

10/10 - 0s - 12ms/step - loss: 2.6126 - val_loss: 4.0746

Epoch 201/300

10/10 - 0s - 12ms/step - loss: 2.5988 - val_loss: 4.1201

Epoch 202/300

10/10 - 0s - 12ms/step - loss: 2.5846 - val_loss: 4.1158

Epoch 203/300

10/10 - 0s - 12ms/step - loss: 2.5774 - val_loss: 4.0837

Epoch 204/300

10/10 - 0s - 12ms/step - loss: 2.5651 - val_loss: 4.1185

Epoch 205/300

10/10 - 0s - 12ms/step - loss: 2.5612 - val_loss: 4.0988

Epoch 206/300

10/10 - 0s - 12ms/step - loss: 2.5612 - val_loss: 4.0963

Epoch 207/300

10/10 - 0s - 12ms/step - loss: 2.5654 - val_loss: 4.1181

Epoch 208/300

10/10 - 0s - 12ms/step - loss: 2.5798 - val_loss: 4.0546

Epoch 209/300

10/10 - 0s - 12ms/step - loss: 2.5853 - val_loss: 4.0827

Epoch 210/300

10/10 - 0s - 12ms/step - loss: 2.5575 - val_loss: 4.1168

Epoch 211/300

10/10 - 0s - 12ms/step - loss: 2.5497 - val_loss: 4.0711

Epoch 212/300

10/10 - 0s - 12ms/step - loss: 2.5322 - val_loss: 4.0557

Epoch 213/300

10/10 - 0s - 12ms/step - loss: 2.5263 - val_loss: 4.0610

Epoch 214/300

10/10 - 0s - 12ms/step - loss: 2.5216 - val_loss: 4.0658

Epoch 215/300

10/10 - 0s - 12ms/step - loss: 2.5305 - val_loss: 4.0499

Epoch 216/300

10/10 - 0s - 12ms/step - loss: 2.5422 - val_loss: 4.0892

Epoch 217/300

10/10 - 0s - 12ms/step - loss: 2.5333 - val_loss: 4.0450

Epoch 218/300

10/10 - 0s - 12ms/step - loss: 2.5104 - val_loss: 4.0514

Epoch 219/300

10/10 - 0s - 12ms/step - loss: 2.5159 - val_loss: 4.0223

Epoch 220/300

10/10 - 0s - 12ms/step - loss: 2.5151 - val_loss: 4.0719

Epoch 221/300

10/10 - 0s - 12ms/step - loss: 2.5246 - val_loss: 4.0368

Epoch 222/300

10/10 - 0s - 12ms/step - loss: 2.4963 - val_loss: 4.0647

Epoch 223/300

10/10 - 0s - 12ms/step - loss: 2.4986 - val_loss: 4.0338

Epoch 224/300

10/10 - 0s - 12ms/step - loss: 2.4883 - val_loss: 4.0142

Epoch 225/300

10/10 - 0s - 12ms/step - loss: 2.4880 - val_loss: 4.0130

Epoch 226/300

10/10 - 0s - 12ms/step - loss: 2.5148 - val_loss: 4.0296

Epoch 227/300

10/10 - 0s - 12ms/step - loss: 2.4961 - val_loss: 4.0418

Epoch 228/300

10/10 - 0s - 12ms/step - loss: 2.4836 - val_loss: 4.0218

Epoch 229/300

10/10 - 0s - 12ms/step - loss: 2.4808 - val_loss: 4.0201

Epoch 230/300

10/10 - 0s - 12ms/step - loss: 2.4908 - val_loss: 4.0124

Epoch 231/300

10/10 - 0s - 12ms/step - loss: 2.4919 - val_loss: 4.0002

Epoch 232/300

10/10 - 0s - 12ms/step - loss: 2.4667 - val_loss: 4.0065

Epoch 233/300

10/10 - 0s - 12ms/step - loss: 2.4671 - val_loss: 4.0027

Epoch 234/300

10/10 - 0s - 12ms/step - loss: 2.4613 - val_loss: 4.0160

Epoch 235/300

10/10 - 0s - 12ms/step - loss: 2.4594 - val_loss: 4.0236

Epoch 236/300

10/10 - 0s - 12ms/step - loss: 2.4569 - val_loss: 3.9953

Epoch 237/300

10/10 - 0s - 12ms/step - loss: 2.4509 - val_loss: 4.0022

Epoch 238/300

10/10 - 0s - 12ms/step - loss: 2.4526 - val_loss: 4.0175

Epoch 239/300

10/10 - 0s - 12ms/step - loss: 2.4523 - val_loss: 3.9841

Epoch 240/300

10/10 - 0s - 12ms/step - loss: 2.4413 - val_loss: 3.9988

Epoch 241/300

10/10 - 0s - 12ms/step - loss: 2.4446 - val_loss: 3.9807

Epoch 242/300

10/10 - 0s - 12ms/step - loss: 2.4531 - val_loss: 4.0107

Epoch 243/300

10/10 - 0s - 12ms/step - loss: 2.4397 - val_loss: 3.9265

Epoch 244/300

10/10 - 0s - 12ms/step - loss: 2.4501 - val_loss: 3.9765

Epoch 245/300

10/10 - 0s - 12ms/step - loss: 2.4337 - val_loss: 3.9988

Epoch 246/300

10/10 - 0s - 12ms/step - loss: 2.4369 - val_loss: 3.9541

Epoch 247/300

10/10 - 0s - 12ms/step - loss: 2.4248 - val_loss: 3.9730

Epoch 248/300

10/10 - 0s - 12ms/step - loss: 2.4233 - val_loss: 3.9498

Epoch 249/300

10/10 - 0s - 12ms/step - loss: 2.4135 - val_loss: 3.9375

Epoch 250/300

10/10 - 0s - 12ms/step - loss: 2.4162 - val_loss: 3.9425

Epoch 251/300

10/10 - 0s - 12ms/step - loss: 2.4081 - val_loss: 3.9738

Epoch 252/300

10/10 - 0s - 12ms/step - loss: 2.4080 - val_loss: 3.9493

Epoch 253/300

10/10 - 0s - 12ms/step - loss: 2.4093 - val_loss: 3.9688

Epoch 254/300

10/10 - 0s - 12ms/step - loss: 2.4064 - val_loss: 3.9436

Epoch 255/300

10/10 - 0s - 12ms/step - loss: 2.4073 - val_loss: 3.9476

Epoch 256/300

10/10 - 0s - 12ms/step - loss: 2.4065 - val_loss: 3.9568

Epoch 257/300

10/10 - 0s - 12ms/step - loss: 2.4031 - val_loss: 3.9325

Epoch 258/300

10/10 - 0s - 12ms/step - loss: 2.4040 - val_loss: 3.9299

Epoch 259/300

10/10 - 0s - 12ms/step - loss: 2.4011 - val_loss: 3.9365

Epoch 260/300

10/10 - 0s - 12ms/step - loss: 2.3872 - val_loss: 3.9359

Epoch 261/300

10/10 - 0s - 12ms/step - loss: 2.3875 - val_loss: 3.9251

Epoch 262/300

10/10 - 0s - 12ms/step - loss: 2.3988 - val_loss: 3.9515

Epoch 263/300

10/10 - 0s - 14ms/step - loss: 2.4102 - val_loss: 3.9392

Epoch 264/300

10/10 - 0s - 11ms/step - loss: 2.4098 - val_loss: 3.9708

Epoch 265/300

10/10 - 0s - 12ms/step - loss: 2.3987 - val_loss: 3.9549

Epoch 266/300

10/10 - 0s - 12ms/step - loss: 2.3870 - val_loss: 3.9501

Epoch 267/300

10/10 - 0s - 12ms/step - loss: 2.3912 - val_loss: 3.9307

Epoch 268/300

10/10 - 0s - 12ms/step - loss: 2.3845 - val_loss: 3.9358

Epoch 269/300

10/10 - 0s - 12ms/step - loss: 2.3780 - val_loss: 3.9111

Epoch 270/300

10/10 - 0s - 12ms/step - loss: 2.3848 - val_loss: 3.9463

Epoch 271/300

10/10 - 0s - 12ms/step - loss: 2.3806 - val_loss: 3.8892

Epoch 272/300

10/10 - 0s - 12ms/step - loss: 2.3792 - val_loss: 3.8823

Epoch 273/300

10/10 - 0s - 12ms/step - loss: 2.3821 - val_loss: 3.8836

Epoch 274/300

10/10 - 0s - 12ms/step - loss: 2.3652 - val_loss: 3.9064

Epoch 275/300

10/10 - 0s - 12ms/step - loss: 2.3795 - val_loss: 3.8820

Epoch 276/300

10/10 - 0s - 12ms/step - loss: 2.3818 - val_loss: 3.9169

Epoch 277/300

10/10 - 0s - 12ms/step - loss: 2.3595 - val_loss: 3.9392

Epoch 278/300

10/10 - 0s - 12ms/step - loss: 2.3602 - val_loss: 3.9120

Epoch 279/300

10/10 - 0s - 12ms/step - loss: 2.3508 - val_loss: 3.9248

Epoch 280/300

10/10 - 0s - 12ms/step - loss: 2.3504 - val_loss: 3.8786

Epoch 281/300

10/10 - 0s - 12ms/step - loss: 2.3546 - val_loss: 3.9174

Epoch 282/300

10/10 - 0s - 12ms/step - loss: 2.3384 - val_loss: 3.9122

Epoch 283/300

10/10 - 0s - 12ms/step - loss: 2.3455 - val_loss: 3.8842

Epoch 284/300

10/10 - 0s - 12ms/step - loss: 2.3489 - val_loss: 3.9046

Epoch 285/300

10/10 - 0s - 12ms/step - loss: 2.3444 - val_loss: 3.9018

Epoch 286/300

10/10 - 0s - 12ms/step - loss: 2.3426 - val_loss: 3.9284

Epoch 287/300

10/10 - 0s - 12ms/step - loss: 2.3370 - val_loss: 3.8992

Epoch 288/300

10/10 - 0s - 12ms/step - loss: 2.3363 - val_loss: 3.8687

Epoch 289/300

10/10 - 0s - 12ms/step - loss: 2.3409 - val_loss: 3.8979

Epoch 290/300

10/10 - 0s - 12ms/step - loss: 2.3356 - val_loss: 3.8955

Epoch 291/300

10/10 - 0s - 12ms/step - loss: 2.3332 - val_loss: 3.8985

Epoch 292/300

10/10 - 0s - 12ms/step - loss: 2.3356 - val_loss: 3.9107

Epoch 293/300

10/10 - 0s - 12ms/step - loss: 2.3228 - val_loss: 3.8890

Epoch 294/300

10/10 - 0s - 12ms/step - loss: 2.3197 - val_loss: 3.8786

Epoch 295/300

10/10 - 0s - 12ms/step - loss: 2.3180 - val_loss: 3.8769

Epoch 296/300

10/10 - 0s - 12ms/step - loss: 2.3128 - val_loss: 3.8831

Epoch 297/300

10/10 - 0s - 12ms/step - loss: 2.3414 - val_loss: 3.8752

Epoch 298/300

10/10 - 0s - 12ms/step - loss: 2.3270 - val_loss: 3.8791

Epoch 299/300

10/10 - 0s - 12ms/step - loss: 2.3164 - val_loss: 3.8413

Epoch 300/300

10/10 - 0s - 12ms/step - loss: 2.3139 - val_loss: 3.8602
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
z, _ = model(normalized_data, training=True)  # Ensure training=True for data->latent
z = ops.convert_to_numpy(z)

# From latent space to data.
samples = keras.random.normal(shape=(3000, 2))  # Correctly sample from standard normal
x, _ = model(
    samples, training=False
)  # Use model's call method for generation (latent->data)
x = ops.convert_to_numpy(x)

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
axes[1, 1].set(title="Generated data space X", xlabel="x", ylabel="y")
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
    


---
## Relevant Chapters from Deep Learning with Python
- [Chapter 17: Image generation](https://deeplearningwithpython.io/chapters/chapter17_image-generation)

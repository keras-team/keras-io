# Variational AutoEncoder

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2020/05/03<br>
**Last modified:** 2020/05/03<br>
**Description:** Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/vae.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py)



---
## Setup



```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

```

---
## Create a sampling layer



```python

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


```

---
## Build the encoder



```python
latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

```

<div class="k-default-codeblock">
```
Model: "encoder"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 28, 28, 1)]  0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 14, 14, 32)   320         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 7, 7, 64)     18496       conv2d[0][0]                     
__________________________________________________________________________________________________
flatten (Flatten)               (None, 3136)         0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
dense (Dense)                   (None, 16)           50192       flatten[0][0]                    
__________________________________________________________________________________________________
z_mean (Dense)                  (None, 2)            34          dense[0][0]                      
__________________________________________________________________________________________________
z_log_var (Dense)               (None, 2)            34          dense[0][0]                      
__________________________________________________________________________________________________
sampling (Sampling)             (None, 2)            0           z_mean[0][0]                     
                                                                 z_log_var[0][0]                  
==================================================================================================
Total params: 69,076
Trainable params: 69,076
Non-trainable params: 0
__________________________________________________________________________________________________

```
</div>
---
## Build the decoder



```python
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

```

<div class="k-default-codeblock">
```
Model: "decoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 2)]               0         
_________________________________________________________________
dense_1 (Dense)              (None, 3136)              9408      
_________________________________________________________________
reshape (Reshape)            (None, 7, 7, 64)          0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 14, 14, 64)        36928     
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 32)        18464     
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 28, 28, 1)         289       
=================================================================
Total params: 65,089
Trainable params: 65,089
Non-trainable params: 0
_________________________________________________________________

```
</div>
---
## Define the VAE as a `Model` with a custom `train_step`



```python

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


```

---
## Train the VAE



```python
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)

```

<div class="k-default-codeblock">
```
Epoch 1/30
547/547 [==============================] - 3s 5ms/step - loss: 202.1639 - reconstruction_loss: 199.6418 - kl_loss: 2.5221
Epoch 2/30
547/547 [==============================] - 2s 4ms/step - loss: 161.9867 - reconstruction_loss: 158.9957 - kl_loss: 2.9910
Epoch 3/30
547/547 [==============================] - 2s 4ms/step - loss: 157.0864 - reconstruction_loss: 153.9148 - kl_loss: 3.1716
Epoch 4/30
547/547 [==============================] - 2s 5ms/step - loss: 154.6892 - reconstruction_loss: 151.4092 - kl_loss: 3.2800
Epoch 5/30
547/547 [==============================] - 2s 4ms/step - loss: 153.1740 - reconstruction_loss: 149.8300 - kl_loss: 3.3441
Epoch 6/30
547/547 [==============================] - 2s 5ms/step - loss: 152.0346 - reconstruction_loss: 148.6590 - kl_loss: 3.3756
Epoch 7/30
547/547 [==============================] - 2s 5ms/step - loss: 151.2110 - reconstruction_loss: 147.7929 - kl_loss: 3.4181
Epoch 8/30
547/547 [==============================] - 2s 5ms/step - loss: 150.5230 - reconstruction_loss: 147.0933 - kl_loss: 3.4297
Epoch 9/30
547/547 [==============================] - 2s 5ms/step - loss: 149.9584 - reconstruction_loss: 146.5069 - kl_loss: 3.4515
Epoch 10/30
547/547 [==============================] - 2s 5ms/step - loss: 149.4152 - reconstruction_loss: 145.9451 - kl_loss: 3.4701
Epoch 11/30
547/547 [==============================] - 2s 5ms/step - loss: 149.0085 - reconstruction_loss: 145.5200 - kl_loss: 3.4885
Epoch 12/30
547/547 [==============================] - 2s 5ms/step - loss: 148.6831 - reconstruction_loss: 145.1854 - kl_loss: 3.4977
Epoch 13/30
547/547 [==============================] - 2s 4ms/step - loss: 148.3130 - reconstruction_loss: 144.7828 - kl_loss: 3.5302
Epoch 14/30
547/547 [==============================] - 2s 5ms/step - loss: 148.0216 - reconstruction_loss: 144.4819 - kl_loss: 3.5397
Epoch 15/30
547/547 [==============================] - 2s 5ms/step - loss: 147.7056 - reconstruction_loss: 144.1588 - kl_loss: 3.5468
Epoch 16/30
547/547 [==============================] - 2s 5ms/step - loss: 147.4493 - reconstruction_loss: 143.8943 - kl_loss: 3.5549
Epoch 17/30
547/547 [==============================] - 2s 5ms/step - loss: 147.1656 - reconstruction_loss: 143.5847 - kl_loss: 3.5809
Epoch 18/30
547/547 [==============================] - 2s 5ms/step - loss: 147.0080 - reconstruction_loss: 143.4251 - kl_loss: 3.5829
Epoch 19/30
547/547 [==============================] - 2s 5ms/step - loss: 146.8182 - reconstruction_loss: 143.2218 - kl_loss: 3.5964
Epoch 20/30
547/547 [==============================] - 2s 5ms/step - loss: 146.5972 - reconstruction_loss: 142.9844 - kl_loss: 3.6128
Epoch 21/30
547/547 [==============================] - 2s 5ms/step - loss: 146.3822 - reconstruction_loss: 142.7513 - kl_loss: 3.6309
Epoch 22/30
547/547 [==============================] - 2s 5ms/step - loss: 146.1550 - reconstruction_loss: 142.5334 - kl_loss: 3.6215
Epoch 23/30
547/547 [==============================] - 2s 4ms/step - loss: 145.9934 - reconstruction_loss: 142.3690 - kl_loss: 3.6245
Epoch 24/30
547/547 [==============================] - 2s 5ms/step - loss: 145.8778 - reconstruction_loss: 142.2351 - kl_loss: 3.6426
Epoch 25/30
547/547 [==============================] - 2s 4ms/step - loss: 145.6936 - reconstruction_loss: 142.0350 - kl_loss: 3.6586
Epoch 26/30
547/547 [==============================] - 2s 4ms/step - loss: 145.5037 - reconstruction_loss: 141.8405 - kl_loss: 3.6633
Epoch 27/30
547/547 [==============================] - 2s 4ms/step - loss: 145.3262 - reconstruction_loss: 141.6582 - kl_loss: 3.6680
Epoch 28/30
547/547 [==============================] - 2s 4ms/step - loss: 145.2551 - reconstruction_loss: 141.5739 - kl_loss: 3.6812
Epoch 29/30
547/547 [==============================] - 2s 5ms/step - loss: 145.1028 - reconstruction_loss: 141.4197 - kl_loss: 3.6831
Epoch 30/30
547/547 [==============================] - 2s 4ms/step - loss: 145.0274 - reconstruction_loss: 141.3409 - kl_loss: 3.6864

<tensorflow.python.keras.callbacks.History at 0x7f74c83e79e8>

```
</div>
---
## Display a grid of sampled digits



```python
import matplotlib.pyplot as plt


def plot_latent(encoder, decoder):
    # display a n*n 2D manifold of digits
    n = 30
    digit_size = 28
    scale = 2.0
    figsize = 15
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent(encoder, decoder)

```


![png](/img/examples/generative/vae/vae_14_0.png)


---
## Display how the latent space clusters different digit classes



```python

def plot_label_clusters(encoder, decoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(encoder, decoder, x_train, y_train)

```


![png](/img/examples/generative/vae/vae_16_0.png)


# GAN overriding `Model.train_step`

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2019/04/29<br>
**Last modified:** 2020/04/29<br>
**Description:** A simple DCGAN trained using `fit()` by overriding `train_step`.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/dcgan_overriding_train_step.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/dcgan_overriding_train_step.py)



---
## Setup



```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

```

---
## Prepare MNIST data



```python
# We use both the training & test MNIST digits.
batch_size = 64
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(32)

```

---
## Create the discriminator

It maps 28x28 digits to a binary classification score.



```python
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

discriminator.summary()

```

<div class="k-default-codeblock">
```
Model: "discriminator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 14, 14, 64)        640       
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 128)         73856     
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 7, 7, 128)         0         
_________________________________________________________________
global_max_pooling2d (Global (None, 128)               0         
_________________________________________________________________
dense (Dense)                (None, 1)                 129       
=================================================================
Total params: 74,625
Trainable params: 74,625
Non-trainable params: 0
_________________________________________________________________

```
</div>
---
## Create the generator

It mirrors the discriminator, replacing `Conv2D` layers with `Conv2DTranspose` layers.



```python
latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        # We want to generate 128 coefficients to reshape into a 7x7x128 map
        layers.Dense(7 * 7 * 128),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

generator.summary()

```

<div class="k-default-codeblock">
```
Model: "generator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 6272)              809088    
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 6272)              0         
_________________________________________________________________
reshape (Reshape)            (None, 7, 7, 128)         0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 14, 14, 128)       262272    
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 14, 14, 128)       0         
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 128)       262272    
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 28, 28, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 1)         6273      
=================================================================
Total params: 1,339,905
Trainable params: 1,339,905
Non-trainable params: 0
_________________________________________________________________

```
</div>
---
## Override `train_step`



```python

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}


```

---
## Create a callback that periodically saves generated images



```python

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))


```

---
## Train the end-to-end model



```python
epochs = 30

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=3, latent_dim=latent_dim)]
)

```

<div class="k-default-codeblock">
```
Epoch 1/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.4052 - g_loss: 1.6674
Epoch 2/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.3953 - g_loss: 1.8534
Epoch 3/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.6629 - g_loss: 0.9196
Epoch 4/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.6655 - g_loss: 0.8764
Epoch 5/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.6333 - g_loss: 0.9402
Epoch 6/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.6182 - g_loss: 0.9575
Epoch 7/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5910 - g_loss: 1.0202
Epoch 8/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5736 - g_loss: 1.0444
Epoch 9/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5545 - g_loss: 1.0886
Epoch 10/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5440 - g_loss: 1.1305
Epoch 11/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5397 - g_loss: 1.1554
Epoch 12/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5350 - g_loss: 1.1622
Epoch 13/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5368 - g_loss: 1.1702
Epoch 14/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5488 - g_loss: 1.1196
Epoch 15/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5551 - g_loss: 1.1108
Epoch 16/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5606 - g_loss: 1.1079
Epoch 17/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5709 - g_loss: 1.0922
Epoch 18/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5794 - g_loss: 1.0737
Epoch 19/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5853 - g_loss: 1.0659
Epoch 20/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5901 - g_loss: 1.0476
Epoch 21/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5919 - g_loss: 1.0464
Epoch 22/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5950 - g_loss: 1.0370
Epoch 23/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5969 - g_loss: 1.0343
Epoch 24/30
1094/1094 [==============================] - 12s 11ms/step - d_loss: 0.5986 - g_loss: 1.0278
Epoch 25/30
 954/1094 [=========================>....] - ETA: 1s - d_loss: 0.5964 - g_loss: 1.0263

```
</div>
Display the last generated images:



```python
from IPython.display import Image, display

display(Image("generated_img_0_29.png"))
display(Image("generated_img_1_29.png"))
display(Image("generated_img_2_29.png"))

```


![png](/img/examples/generative/dcgan_overriding_train_step/dcgan_overriding_train_step_16_0.png)



![png](/img/examples/generative/dcgan_overriding_train_step/dcgan_overriding_train_step_16_1.png)



![png](/img/examples/generative/dcgan_overriding_train_step/dcgan_overriding_train_step_16_2.png)


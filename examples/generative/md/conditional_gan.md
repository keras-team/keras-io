# Conditional GAN

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2021/07/13<br>
**Last modified:** 2021/07/15<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/conditional_gan.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/conditional_gan.py)


**Description:** Training a GAN conditioned on class labels to generate handwritten digits.

Generative Adversarial Networks (GANs) let us generate novel image data, video data,
or audio data from a random input. Typically, the random input is sampled
from a normal distribution, before going through a series of transformations that turn
it into something plausible (image, video, audio, etc.).

However, a simple [DCGAN](https://arxiv.org/abs/1511.06434) doesn't let us control
the appearance (e.g. class) of the samples we're generating. For instance,
with a GAN that generates MNIST handwritten digits, a simple DCGAN wouldn't let us
choose the class of digits we're generating.
To be able to control what we generate, we need to _condition_ the GAN output
on a semantic input, such as the class of an image.

In this example, we'll build a **Conditional GAN** that can generate MNIST handwritten
digits conditioned on a given class. Such a model can have various useful applications:

* let's say you are dealing with an
[imbalanced image dataset](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data),
and you'd like to gather more examples for the skewed class to balance the dataset.
Data collection can be a costly process on its own. You could instead train a Conditional GAN and use
it to generate novel images for the class that needs balancing.
* Since the generator learns to associate the generated samples with the class labels,
its representations can also be used for [other downstream tasks](https://arxiv.org/abs/1809.11096).

Following are the references used for developing this example:

* [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
* [Lecture on Conditional Generation from Coursera](https://www.coursera.org/lecture/build-basic-generative-adversarial-networks-gans/conditional-generation-inputs-2OPrG)

If you need a refresher on GANs, you can refer to the "Generative adversarial networks"
section of
[this resource](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-12/r-3/232).

This example requires TensorFlow 2.5 or higher, as well as TensorFlow Docs, which can be
installed using the following command:


```python
!pip install -q git+https://github.com/tensorflow/docs
```

<div class="k-default-codeblock">
```
  Building wheel for tensorflow-docs (setup.py) ... [?25l[?25hdone

```
</div>
---
## Imports


```python
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio
```

---
## Constants and hyperparameters


```python
batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128
```

---
## Loading the MNIST dataset and preprocessing it


```python
# We'll use all the available examples from both the training and test
# sets.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = keras.utils.to_categorical(all_labels, 10)

# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")
```

<div class="k-default-codeblock">
```
Shape of training images: (70000, 28, 28, 1)
Shape of training labels: (70000, 10)

```
</div>
---
## Calculating the number of input channel for the generator and discriminator

In a regular (unconditional) GAN, we start by sampling noise (of some fixed
dimension) from a normal distribution. In our case, we also need to account
for the class labels. We will have to add the number of classes to
the input channels of the generator (noise input) as well as the discriminator
(generated image input).


```python
generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)
```

<div class="k-default-codeblock">
```
138 11

```
</div>
---
## Creating the discriminator and generator

The model definitions (`discriminator`, `generator`, and `ConditionalGAN`) have been
adapted from [this example](https://keras.io/guides/customizing_what_happens_in_fit/).


```python
# Create the discriminator.
discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((28, 28, discriminator_in_channels)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator.
generator = keras.Sequential(
    [
        keras.layers.InputLayer((generator_in_channels,)),
        # We want to generate 128 + num_classes coefficients to reshape into a
        # 7x7x(128 + num_classes) map.
        layers.Dense(7 * 7 * generator_in_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)
```

---
## Creating a `ConditionalGAN` model


```python

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

```

---
## Training the Conditional GAN


```python
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

cond_gan.fit(dataset, epochs=20)
```

<div class="k-default-codeblock">
```
Epoch 1/20
1094/1094 [==============================] - 34s 16ms/step - g_loss: 1.4316 - d_loss: 0.4501
Epoch 2/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 1.2608 - d_loss: 0.4962
Epoch 3/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 1.4321 - d_loss: 0.4443
Epoch 4/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 1.9275 - d_loss: 0.2990
Epoch 5/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 2.2511 - d_loss: 0.2491
Epoch 6/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.9803 - d_loss: 0.6354
Epoch 7/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.8971 - d_loss: 0.6596
Epoch 8/20
1094/1094 [==============================] - 17s 16ms/step - g_loss: 0.8358 - d_loss: 0.6748
Epoch 9/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.8089 - d_loss: 0.6726
Epoch 10/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.7995 - d_loss: 0.6739
Epoch 11/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.7873 - d_loss: 0.6789
Epoch 12/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.7666 - d_loss: 0.6820
Epoch 13/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.7637 - d_loss: 0.6839
Epoch 14/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.7572 - d_loss: 0.6840
Epoch 15/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.7563 - d_loss: 0.6795
Epoch 16/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.7469 - d_loss: 0.6855
Epoch 17/20
1094/1094 [==============================] - 18s 16ms/step - g_loss: 0.7623 - d_loss: 0.6798
Epoch 18/20
 889/1094 [=======================>......] - ETA: 3s - g_loss: 0.7421 - d_loss: 0.6802

```
</div>
---
## Interpolating between classes with the trained generator


```python
# We first extract the trained generator from our Conditiona GAN.
trained_gen = cond_gan.generator

# Choose the number of intermediate images that would be generated in
# between the interpolation + 2 (start and last images).
num_interpolation = 9  # @param {type:"integer"}

# Sample noise for the interpolation.
interpolation_noise = tf.random.normal(shape=(1, latent_dim))
interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))


def interpolate_class(first_number, second_number):
    # Convert the start and end labels to one-hot encoded vectors.
    first_label = keras.utils.to_categorical([first_number], num_classes)
    second_label = keras.utils.to_categorical([second_number], num_classes)
    first_label = tf.cast(first_label, tf.float32)
    second_label = tf.cast(second_label, tf.float32)

    # Calculate the interpolation vector between the two labels.
    percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = (
        first_label * (1 - percent_second_label) + second_label * percent_second_label
    )

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
    fake = trained_gen.predict(noise_and_labels)
    return fake


start_class = 1  # @param {type:"slider", min:0, max:9, step:1}
end_class = 5  # @param {type:"slider", min:0, max:9, step:1}

fake_images = interpolate_class(start_class, end_class)
```

Here, we first sample noise from a normal distribution and then we repeat that for
`num_interpolation` times and reshape the result accordingly.
We then distribute it uniformly for `num_interpolation`
with the label indentities being present in some proportion.


```python
fake_images *= 255.0
converted_images = fake_images.astype(np.uint8)
converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
imageio.mimsave("animation.gif", converted_images, fps=1)
embed.embed_file("animation.gif")
```




<img src="data:image/gif;base64,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"/>



We can further improve the performance of this model with recipes like
[WGAN-GP](https://keras.io/examples/generative/wgan_gp).
Conditional generation is also widely used in many modern image generation architectures like
[VQ-GANs](https://arxiv.org/abs/2012.09841), [DALL-E](https://openai.com/blog/dall-e/),
etc.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/conditional-gan) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/conditional-GAN).
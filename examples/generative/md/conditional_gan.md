# Conditional GAN

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2021/07/13<br>
**Last modified:** 2024/01/02<br>
**Description:** Training a GAN conditioned on class labels to generate handwritten digits.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/conditional_gan.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/conditional_gan.py)



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

---
## Imports


```python
import keras

from keras import layers
from keras import ops
from tensorflow_docs.vis import embed
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
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
 11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
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
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
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
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
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
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(1337)
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = ops.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = ops.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = ops.shape(real_images)[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        random_vector_labels = ops.concatenate(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = ops.concatenate(
            [generated_images, image_one_hot_labels], -1
        )
        real_image_and_labels = ops.concatenate([real_images, image_one_hot_labels], -1)
        combined_images = ops.concatenate(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = ops.concatenate(
            [ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0
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
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        random_vector_labels = ops.concatenate(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = ops.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = ops.concatenate(
                [fake_images, image_one_hot_labels], -1
            )
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
   18/1094 [37m━━━━━━━━━━━━━━━━━━━━  10s 9ms/step - d_loss: 0.6321 - g_loss: 0.7887 

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1704233262.157522    6737 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 24s 14ms/step - d_loss: 0.4052 - g_loss: 1.5851 - discriminator_loss: 0.4390 - generator_loss: 1.4775
Epoch 2/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.5116 - g_loss: 1.2740 - discriminator_loss: 0.4872 - generator_loss: 1.3330
Epoch 3/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.3626 - g_loss: 1.6775 - discriminator_loss: 0.3252 - generator_loss: 1.8219
Epoch 4/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.2248 - g_loss: 2.2898 - discriminator_loss: 0.3418 - generator_loss: 2.0042
Epoch 5/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6017 - g_loss: 1.0428 - discriminator_loss: 0.6076 - generator_loss: 1.0176
Epoch 6/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6395 - g_loss: 0.9258 - discriminator_loss: 0.6448 - generator_loss: 0.9134
Epoch 7/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6402 - g_loss: 0.8914 - discriminator_loss: 0.6458 - generator_loss: 0.8773
Epoch 8/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6549 - g_loss: 0.8440 - discriminator_loss: 0.6555 - generator_loss: 0.8364
Epoch 9/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6603 - g_loss: 0.8316 - discriminator_loss: 0.6606 - generator_loss: 0.8241
Epoch 10/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6594 - g_loss: 0.8169 - discriminator_loss: 0.6605 - generator_loss: 0.8218
Epoch 11/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6719 - g_loss: 0.7979 - discriminator_loss: 0.6649 - generator_loss: 0.8096
Epoch 12/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6641 - g_loss: 0.7992 - discriminator_loss: 0.6621 - generator_loss: 0.7953
Epoch 13/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6657 - g_loss: 0.7979 - discriminator_loss: 0.6624 - generator_loss: 0.7924
Epoch 14/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6586 - g_loss: 0.8220 - discriminator_loss: 0.6566 - generator_loss: 0.8174
Epoch 15/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6646 - g_loss: 0.7916 - discriminator_loss: 0.6578 - generator_loss: 0.7973
Epoch 16/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6624 - g_loss: 0.7911 - discriminator_loss: 0.6587 - generator_loss: 0.7966
Epoch 17/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6586 - g_loss: 0.8060 - discriminator_loss: 0.6550 - generator_loss: 0.7997
Epoch 18/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6526 - g_loss: 0.7946 - discriminator_loss: 0.6523 - generator_loss: 0.7948
Epoch 19/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6525 - g_loss: 0.8039 - discriminator_loss: 0.6497 - generator_loss: 0.8066
Epoch 20/20
 1094/1094 ━━━━━━━━━━━━━━━━━━━━ 10s 9ms/step - d_loss: 0.6480 - g_loss: 0.8005 - discriminator_loss: 0.6469 - generator_loss: 0.8022

<keras.src.callbacks.history.History at 0x7f541a1b5f90>

```
</div>
---
## Interpolating between classes with the trained generator


```python
# We first extract the trained generator from our Conditional GAN.
trained_gen = cond_gan.generator

# Choose the number of intermediate images that would be generated in
# between the interpolation + 2 (start and last images).
num_interpolation = 9  # @param {type:"integer"}

# Sample noise for the interpolation.
interpolation_noise = keras.random.normal(shape=(1, latent_dim))
interpolation_noise = ops.repeat(interpolation_noise, repeats=num_interpolation)
interpolation_noise = ops.reshape(interpolation_noise, (num_interpolation, latent_dim))


def interpolate_class(first_number, second_number):
    # Convert the start and end labels to one-hot encoded vectors.
    first_label = keras.utils.to_categorical([first_number], num_classes)
    second_label = keras.utils.to_categorical([second_number], num_classes)
    first_label = ops.cast(first_label, "float32")
    second_label = ops.cast(second_label, "float32")

    # Calculate the interpolation vector between the two labels.
    percent_second_label = ops.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = ops.cast(percent_second_label, "float32")
    interpolation_labels = (
        first_label * (1 - percent_second_label) + second_label * percent_second_label
    )

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = ops.concatenate([interpolation_noise, interpolation_labels], 1)
    fake = trained_gen.predict(noise_and_labels)
    return fake


start_class = 2  # @param {type:"slider", min:0, max:9, step:1}
end_class = 6  # @param {type:"slider", min:0, max:9, step:1}

fake_images = interpolate_class(start_class, end_class)
```

<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 427ms/step

```
</div>
Here, we first sample noise from a normal distribution and then we repeat that for
`num_interpolation` times and reshape the result accordingly.
We then distribute it uniformly for `num_interpolation`
with the label identities being present in some proportion.


```python
fake_images *= 255.0
converted_images = fake_images.astype(np.uint8)
converted_images = ops.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
imageio.mimsave("animation.gif", converted_images[:, :, :, 0], fps=1)
embed.embed_file("animation.gif")
```




<img src="data:image/gif;base64,R0lGODlhYABgAIcAAAAAAAEBAQICAgMDAwQEBAUFBQYGBgcHBwgICAkJCQoKCgsLCwwMDA0NDQ4ODg8PDxAQEBERERISEhMTExQUFBUVFRYWFhcXFxgYGBkZGRoaGhsbGxwcHB0dHR4eHh8fHyAgICEhISIiIiMjIyQkJCUlJSYmJicnJygoKCkpKSoqKisrKywsLC0tLS4uLi8vLzAwMDExMTIyMjMzMzQ0NDU1NTY2Njc3Nzg4ODk5OTo6Ojs7Ozw8PD09PT4+Pj8/P0BAQEFBQUJCQkNDQ0REREVFRUZGRkdHR0hISElJSUpKSktLS0xMTE1NTU5OTk9PT1BQUFFRUVJSUlNTU1RUVFVVVVZWVldXV1hYWFlZWVpaWltbW1xcXF1dXV5eXl9fX2BgYGFhYWJiYmNjY2RkZGVlZWZmZmdnZ2hoaGlpaWpqamtra2xsbG1tbW5ubm9vb3BwcHFxcXJycnNzc3R0dHV1dXZ2dnd3d3h4eHl5eXp6ent7e3x8fH19fX5+fn9/f4CAgIGBgYKCgoODg4SEhIWFhYaGhoeHh4iIiImJiYqKiouLi4yMjI2NjY6Ojo+Pj5CQkJGRkZKSkpOTk5SUlJWVlZaWlpeXl5iYmJmZmZqampubm5ycnJ2dnZ6enp+fn6CgoKGhoaKioqOjo6SkpKWlpaampqenp6ioqKmpqaqqqqurq62tra6urq+vr7CwsLGxsbKysrOzs7S0tLW1tba2tre3t7i4uLm5ubq6uru7u7y8vL29vb6+vr+/v8DAwMHBwcLCwsPDw8TExMXFxcbGxsfHx8jIyMnJycrKysvLy8zMzM3Nzc7Ozs/Pz9DQ0NHR0dLS0tPT09TU1NXV1dbW1tfX19jY2NnZ2dra2tvb29zc3N3d3d7e3t/f3+Dg4OHh4eLi4uPj4+Tk5OXl5ebm5ufn5+jo6Onp6erq6uvr6+zs7O3t7e7u7u/v7/Dw8PHx8fLy8vPz8/T09PX19fb29vf39/j4+Pn5+fr6+vv7+/z8/P39/QAAAAAAAAAAACH5BABkAAAALAAAAABgAGAAAAj/AAEIHEiwoMGDCBMqXMiwocOHECNKnEixosWLGDNq3Mixo8ePIEOKHEmypMmTKFOqXMmypcuXMA8GCEAAwYIGDBYkOFBggIAAMUEGGEDAAYYRJ0iE6HABggIDBH4G7SigwIELKXQE4XEjhgkOERYgiAp0asYABRI0IMGDCpcqTo7kWMGBwgMFBaKarRhAwAAFHV4EucImUCJCf/TAOdNlCpIeMUZQMCB1L0SaBijQqGInESdXtmS9YoVqlKdIgN5wEXKCQc+ylhsKMACYyR5Vu5hxGxcOnLff3p7hEoUojI4JCAoIiN0wwAIMJn6w0XTMmrh39ejNm0evnj111YCh/wKE5YWHCQeYK5wpAAMNKG4k1co2bt28e/fs2buHLx88cdQIAwofWfyAAgQzqSeTAASU8EQentwCjTrw0IOPPvlkqI8++9gDTzrd+AKKIGDwkEFlChLkVwEwlLHJMNWcc+E++/Bj4402YihPNsCc8scUJiRgwAApEjQUATGgAQoy2aSTD404RrmPPvWME80vmrgxxAgZOEBWkUfGsAYpznSzjj5RpsnPPvi0I841tCjyhQ8rZHCAXin2NYAMbqQyDTju1KimlPfQA48zpvCBhQ8kLAAVbMyxl0IWisBSjDYV3oPmjTROuWE+9swDTzW1VFIHFjt0QEEDlEFq2Uwc+P8AhiCiBBOOOvHg0+mn+OBnDz3xtJPONcGwMgkdVNBgQgYMEDCAq5ZBQMINU/iRijTfqFPPp/n0+us88sCzzjniZOPMMLA80sYSN5AgAU/LKXhABBrIMAYlxVQzjjze1iOPO+OSE0432EzjjDPPRGPMKYecAYUOI1TgwAHQmlVAAxSooIUiuDDTjTv1iNrOONpAc8wvtryyyimjtMILM87kUsojeIShRA0kTOBTxTENkIADIkDxByvCWKPOv+qUU00xtqCySSOB+KGHHY2oUt0ywdgCCiJwWLEDCFDFaxmDB2RwBB2h5PKMOe+kM043x8TCCSJ2jDEFFE0owcYlwXT/89s3x7hiyR1SsCDks7ENVYAEM1BRxyGXoOKKKqaEEkkgcIhBhRE64GDDe3VoUssw0XwDzS+vRDKGDQyMxTNMfi3ggQxGRLFFGWmYMQYYVzQxhA4yqDBCCCB8UIMUbihiCjDcWPPMMarUAYQDeL0O01AKQHDBBh6EMIIIxHeQgQUTQODAAgsooIAGLgzxBSKwZOP3N8IYkkQEDFBc5EQKQAfEG59oRjbEsQ5pYKILJtjAAwiwP4kcAAIZeEEXGJELZWgjHdlYxR6ccAMQJMB6DTTIAA7AgA8wAQ+l4IU00BGOXmSiDlOAgQMSFEKIPEAHZYBEK5JhDnQ4gxaaaAMQ/yaAoho2ZAEsaMIbKrGLcaQDGrbgxBuEQIGdGdEhCPiADZrQh1Z8Yx3RuEUn4DAEC7zmig0hQAQ+wAIygGIb7JBGLj4hhyJg4FFoXE8AEJCBFfAgDqjwRjukgQtPxIEIdyQACI3YFwI8oAVLSIMkdlGOQd5iE28o4xnzKBMCGOACQVhDJWTxDHawIxq1yEQbglABZy2yhjVpgAiqgAhdOAMc8mAHNGZhiTUAgQJf4uRA9MQAC4QgB2rYBDO2cQ56rMMZsJBEGn5QgU0KEwBVOQAFSlADobWiG+d4x3eW0QpHmMEHF7iT2IQ5QgVwYAZJOEMlguGOedgDH+pIhioUQf8GH2AgOetsYCMR0AAJYCAEK6DBEbgQh0W0Ihr04E8+0GEMUxDiCzqoAB6NOAADJEACHUiBDY5wBTTogRGeiEUywHGPDOHjHMMIRR+yUAMJBDOEaFHAAzjgAiBQgQ2E6ITLlpENcryDV+UAxibsQIUYzPCV1zuSARCgvQ/AwAhcmIMjUjEMaXhDHfbMBz/0Uahv3OIRaEiCChbQQLItAAIW4AAKahAEKqghEJdYxS+m4Y1zwOOeaPJQOp4xCjssgQYeSM/+BoAABkyAAyaAQRCkQAY8PKIUuTiGvihUj3xwiB/yQIc3gPGILrQABDZtYAEWEIENoMAGRNACHAzRCVn/JIMb5LCPPZ6Eo3eEYxqx2AMRIOAoqLZET9u8wRGsQIY5EMISp7gFk87BjnjsdlNr0sc5pNGLTaxBB60rgHFZMoACIIAEQfiCHQpRiVLAYhfHkAY44RHWz2Y3H98YxikIsYUYKOBO41UJWho7gy4cohS0yBc3xJGOdsSjHv2xr4320a1rzCIS60IBT4ikIAEk4AEWIAIdStFVcdDXHhJO0z7uISpmmAIQXgjCCJKDOPU4UgMmoIIhbEGNbNHjnoLi1IY4hA94oAMcvtCEHa7QgxEogMZFtFgFuFmGSyAjt/zyrJT0gY/+5KMeIIoGLBzBhifgQAQNeHJPohyUA3Ag/wZGsAMqtPFjsaoJQ/jp8jzGMY1hlIIQY0iCDUYggQeIhSfOEgCbV9KXBqjACGNgRC2+UY+WDorCXdazObChjFZAQg5baMIPYsACFJCABCdQgQpEkAHiui7AGjnSBG6wBUCIYhjlAOylMdQtfNADHd0IIycA0QYyaAEKSijCD4KAhCdIIQgw+IAFHoAAK6aEbBggQhw6ISF1RPjSnuqWM8slnkckpg5wQEMYtOAFNdQhD2A4wgtCYIFm1RglBFBABEgAhT2cwhfVYMe3cUQjbt0jZG66xjJq4YlD7GEPesjDHNyABjfowRCKgMMVfNCCD0AgbCpBwAQ+MAMtAMIUvf8I+Iw6VfBu5cdfABOHNIQhC08kgg5reMMc7EAHOcDBDoBYhCPwIIYm6OAEFaCxShjAgRYMYQyEKEUvqMEOz7IcQ/0ClzvQ8aZhuGITh5gDGLQwhjXMoQ52wIMfDgGJSQDCDVswAgw08F8Om6QvEjgBD6YAh0akAhjWaIfVC47n7MQDYOTYhjSI0QpL/MENXWBCEqjwBTXoHA9/MIQjLicHMTCBBhxoHQNNUpMFfOAGT0CDIUbxi2iEQx4cujqoDHUObyyNFqRwexu00IQg0GAGOxjCEqJwhS6QoQ12yIMatrAEHaDAAk+x+0j2+IAKsOAIYvADJ3BhHXbYg+WdAtX/PNrhDWf8AhWPsAMYojAEGrCgBB3ggAhO0IIZaEUJU+hCGKZAhBqggAMQAC8mIQALUAEfkANXUAeTAAvP0A73gV0s92XyoA7U4AupoAhpcAQuwCXnIyQetQAOYAEeUAIvoANEoAQ70AIeYAFOsWYmMQARAAIuwARt8AisQAzcAGGbMiX4MH61Nw3IsAuoQAmAYAZL8AIbMAGusTONVAAMIAEZEAIpIAM3sAIfQAHUo06wRhEEcAEr8ANgUAhWcw0y8lkUVijnsA3OwAupQAmBEAdhIAVA8AIdgD8HYG16QkIQUAHyB38VwABCclMkUQAeYANPQAedUAzicDT2tWKG/8INyFALm9AHYMA5LzACScgAUHFvAjET5XUACuAAEmAXeMGEKFEABzgFfJAK1lBnnHIP7mAO3UAMrFAJeIAFN1AXDaAcr+SJVqF0LEEAGdACQuAGoSANESVWmNYO2EAMr4AJgIAGUsADJTABDlBtNMQQekIALkheEiACMQAGmPAMycgP+VAo4yAMojAIazAFPMACIEAB/6VIEcEeiraFGiEAxQQCUgAJzJCMbAIs2MAKgFAFPmACD5AXAYVG1GcBPLAGmoALusALvqALthALn9AHXMADK6ABCLBoaJQWDWACRDAGd5AHfOAHEucGYcAE7ZIBCYmPiUMUEOABLGADOEmgAzxgAzAgPJmYHDKpIATwM6NoARcgAfljbdekELORAAzQAA7wAKIHkktpJAxSAAZgAIhGlVXZlV75lWAZlmI5lmRZlmbJHAEBACH5BABkAAAALA4ACQBGAE4AhwAAAAEBAQICAgMDAwQEBAUFBQYGBgcHBwgICAkJCQoKCgsLCwwMDA0NDQ4ODg8PDxAQEBERERISEhMTExQUFBUVFRYWFhcXFxgYGBkZGRoaGhsbGxwcHB0dHR4eHh8fHyAgICEhISIiIiMjIyQkJCUlJSYmJicnJygoKCkpKSoqKisrKywsLC0tLS4uLi8vLzAwMDExMTIyMjMzMzQ0NDU1NTY2Njc3Nzg4ODk5OTo6Ojs7Ozw8PD09PT4+Pj8/P0BAQEFBQUJCQkNDQ0REREVFRUZGRkdHR0hISElJSUpKSktLS0xMTE1NTU5OTk9PT1BQUFFRUVJSUlNTU1RUVFVVVVZWVldXV1hYWFlZWVpaWltbW1xcXF1dXV5eXl9fX2BgYGFhYWJiYmNjY2RkZGVlZWZmZmdnZ2hoaGlpaWpqamtra2xsbG1tbW5ubm9vb3BwcHFxcXJycnNzc3R0dHV1dXZ2dnd3d3h4eHl5eXp6ent7e3x8fH19fX5+fn9/f4CAgIGBgYKCgoODg4SEhIWFhYaGhoeHh4iIiImJiYqKiouLi4yMjI2NjY6Ojo+Pj5CQkJGRkZKSkpOTk5SUlJWVlZaWlpeXl5iYmJmZmZqampubm52dnZ6enp+fn6CgoKGhoaKioqOjo6SkpKWlpaampqenp6ioqKmpqaqqqqurq6ysrK2tra6urq+vr7CwsLGxsbKysrOzs7S0tLW1tba2tre3t7i4uLm5ubq6uru7u7y8vL29vb6+vr+/v8DAwMHBwcLCwsPDw8TExMXFxcbGxsfHx8jIyMnJycrKysvLy8zMzM3Nzc7Ozs/Pz9DQ0NHR0dLS0tPT09TU1NXV1dbW1tfX19jY2NnZ2dra2tvb29zc3N3d3d7e3t/f3+Dg4OHh4eLi4uPj4+Tk5OXl5ebm5ufn5+jo6Onp6erq6uvr6+zs7O3t7e7u7u/v7/Dw8PHx8fLy8vPz8/T09PX19fb29vf39/j4+Pn5+fr6+vv7+/z8/P39/QAAAAAAAAAAAAj/AAEIHEiwoEGBAQIIQNBAwgYTMXTs2MFjRw4cNFiQ2DCBAYGDIEOKHAlA4YACEDacyAHlTB49e/jowVOnDRclOVJgQECyp8+DAU4iyKBiBxY9nG7l0sUrV61YpRjFueKjRIOfWEkKOKlAQgYRNpR42dMJ2LZu3r5tq/YM2ChDaJa4iCAgQNa7BgsoeLDBRRAqZ/Y4GrVLWrnD5r5Zc+ZrFCI2T2RQIDDALt67CRNEyNBiCRpDmVb5YoaNnLt27diNw8aM1yhFcabUuHCggIDLWRMKcKDBBBA0j2gZy7ZO3jx69ObFg1cO2zJdpBzh0cLDQwMEH3H7DFAggQPfWepE/2qFjBo4duzIdbt2DRs2acqE4WolKtMfMklmjKBQYMBt7SIJoMBmPpTRCCq6LMONOOi4U041xeTCiy/B/MILLrfcgksuojBChxY/jKDAAQRYBuBBA/BWghWL/MLNOOrAA88773xzjCucgFJKKqu08kotwTSjzTGzeBKIFjM4oIABJp5YkAEXoJDDGp9Q4w4766TzTTXM5BKKInjwAQghhyziCCatCIONM8LY0gkeU7wQwgQFOHkQAiHoMMUfq2STHjfT+JJKJYG8wYUTUUxhRRZcfLFGIqUIQ8www9CCCR9fDHHCAnYatAALTbQxCS7epLNNM7504kcXSfDgAgklnP+AggotvOADGYm0oksxzyxzyyiPlKEDBE122oAMV/DRyS/gnFONMK4UogUMHCkwwLUnHaCABkS4gYkrw2TjjTTG4FJIFB0kYFunAimAghFjKBJLNuVYYwwth2SxggUPIIDtAAQYcEAFN2jBhya2mKcNNct4IkcRK2zAALsAHOABDUrcQco041yjzC6JXFECBAsYIMDJWxFQwAMrGBGGIao8I87M3+RCSRtOyGABxQVMIEIMY2DCTDhrBsOIFR+MSADKKA+ggAcyGBFHJ8qQk8478kxDiyVwIPEBxQM0gIEIS9gRiiyqhIJJG0LQ1l9CcCs08AgxXNHHKL40w0061gz/A8siWqRQV6cLPWCBDE+0gQccaXxBRAoOUNYk3AQ0YIEHOlxBxyOoBIPNNM4gI8oaMhSwtJ1BGZAABijgwIMNL5zAQQRMFjsQdwgw4EEMRHwxSCnHPINNOLvwwUMCBgzAbkIHNEABBQ74O7hPClgQwg5iKPIKMdm8s4wiR0jAAJPLd9eAkibbLtIBD1ywwhJrMKKKMuo8QwkWJmjgwH+EF3BAbZXJCgEYwgEaNOEMkthFOaTBCTToAAUWUB7FSqK+nphEdy8YAh5YEY5qmAIPTqhBB+o0QQApRAEZSMEO5pAKcGCjFYPYwg9EYIASmjAACLhACWzwhlOAYxu0cAQa/45wghraUDsBOAAFQACDNZTiG97YBSbqAAUWHOCIADrABD7wAjWQ4hvgCEYoAoEFGPAEi7hRoghkwIYnisMYqlgEGGqQADTiBgEWIEEN3GCKb5BjGbKwxBlyoAA74gWHFzABDuDgw3I8IxedaAMPOGXInwSFAA74QAyKkIdWzMwYqZBjDRRQwUoaRAAFQMAETrCDKAhCFuT4Bi80cQcptCABpTQlQQZwgAVg4AVF6MIicGEObshCEWMgggkQkEtdCgQBEdBAy8Cwh08M4xzYYAUZeyCCAzRTlwGAABeNUAZCiCIX0UgHNVDBhynkAARGdCZIFHKBFhAhDIU4RTKo8f8NdURjFHcI4QjlCZJsicAHW+BDKIyhjoaqoxmfkEMSZLCB7BB0lwyogAd2kIU7VGIW0rhSOsyRDE50LQYVvWhBClCBEtRgCnOYRCuK0Y15uAMd4iCGJtxwBBhowKIqBQACPFADJrDBEa44BjbSgY93nOMbwLDEGorwgp8SdACqa8AEMFACH2BBDgcaxjXG8Y58tEMc17DFIryAAxNUQILgRIADKPCBFdwgCWUIBCdgQYxrlIMd88CHOrjhjFX0wQkn0B//TCkABlgABDEYghXasAhUGGMa3yhOPe6Bj3NYoxifeIMP+MVMQyYEq97ZQApssIQw4AESrEAGOdghD3z/6OMe9IAHN5Axi0d8AQZLgivqdMM0gBWAARcIQQt88AQw0AEReyUGNtgRj3rkwx6qwUYwUjEJOTChBAY4XacUshWAEeC8/ktABU6gAyaEwQ6L4MQrghENbqBjHva4Rz7kIQ5pBMMUjJBDFXCgAf9805InM24BFmwABChAdziQAhsIwQnhWMMc8aAHPvaxD33owx3XAEYqGOGGKOSABBAAkABU5oAMjKAFNNiBEGYshB7IAAUe6AAH+sKENRyCE7JAhjXE4Q572HYf+cDtOIYxCkKwYQo7SIEGKImbAeoOY11YAx4MYYhCFCIPY2DCDWbgAhX4IAzmpMUxsjGOdQQ2/x/62Mc9bHqNaHVhCW3VAASuiJsAGIABEmjBE+DQiFDUQhnJOMa9JhGHKCQhCDl4gh04kTdurKO6+eAwh+vxDnQwQxNnsIEKPBCB4B4SqwqowAdUkAQ3SIKv2UhNOs4xjVYkwgxckEISvFCIVTAjG+fQcJz5oY8kq6Mbz3jFH5rgAQs4IH2XIcACJsCBFwiBCnGARFKrIY53tEMd5qAGLShRBzaMQQtteMQrhIQOe8CZw3N+hzV60WS2Qi8BJTrwQbQIghckgQx+wIQsmPGNclzaHeC2hi4+UYg+0IENe8AELZyxjXTcQ9P5mEc7zCEMTtzhCtxU2mLvsgAOuEAIZP8ohCl6IY1gz2Me8nBHOsgRYlVUwhGG+MMiRKGLZ3RjHZnmcFPLwQ1X+KEJLfDAA1ScygzMoAloKAQoekHxBqEjHNu4xjSggYxexMIUpRhFKFbBi2Z0wxzwiHOH5fENZuAiEmbYQQgsUEjt6OUBJ1ACGxpBCl5UwxvkUEc6ugENYgTDF7vQRS4WzwtgEIMZ1gCHOt5RDw7rAx/tgEYg5+AEFVCgAfG8DALaV4MwMCIWxfCTt9cxjmj8ohWqQMUpVmELYCyDGt44RzvgQQ/96oMfSL7HOYChiTlQ4QYZQMC6+tyADJgACXUQRTKwcQ57zEMdaD2GLUpxilSwYhbAII3/ONZhXbVr2h6droYq/kAFHpTgAQZGIgVMkAMuJKIW2jDNPd4hDmoggxdf1wq28AvIEA3aIA7pAA++Z3nF5g7hMA27AAlkgAMngAEJMD24IQAbMANMEAecgAzrwHv5sA7ZUC6ywAqmQAvEMA3dIA66x3u2FWfFhg/CJw2+EAp1gAQbIAELAFS4MQAggFCCwArXkA9wpg/qQA2/AAuv0Aqr0AvRQA7vEA/zsFkx6GH5kFvscA2+QArS8gIl0ikDEAI/0AWD0ApFeIQ1Ag3GUAzDIAw+52b0kF9Hdnn2EA/9JQyrIAl40AU/EALCZUIDIAJA8AWF4ArYcIT6cH1Ytw3a/5AN4ZAO8nAPnPVu/HBd8lA/tKAJf1AGS0ADIyAB+mZBBCACQQAGhvAK2eBhHoYP1rccMoJfQadp/HCJ9MB/vHAJcOBOHsAA2LE8A7ABNMAEcJAJwjAO5FAO5oAO69AOUygP9FAP9WAP9kAP8gAP62AOLtQMwzAL3WViI4OBWGEADVABGCAB1/E2ACAAEiACMzAFfEAKxXAMybAM0qAN4cCM7wBz8hAPCGcOHZQMvdAKnNAIejAGn0gCF0BKo0gQmQQDNzBq4nMA/6EZGWADX4AIqLAKrhALvJAMkWcOtAUPV3IO4KAN0vALrKAJhNAGVQAEMlACF/AACuCD2xEAFv8QA0tQBULwAhxAAR4hEKiEACBQBGlQJo8wCaEwC8Rwj5FIDt6ADdBwDL5AC5+gCHXgBUXQAhhQakvTkASREBhgA1PwBUpQAyBgAQ2QHUFRABbgAkfABWKABm7wB5SQCrkQDMkgDcWAC6rQCZJgCHrABl3wBD/wAh5AMhQJlgURABFQAjogBDRwAhlQanClEL5UAjGQA0CABF1QB44ACqkgC7twCpPQB2twKD+AAzCQAiCAAf2SPIxpEOoFAibwAV3JABTZmFtxABDwFTpgBXKACJGwCa/RBlBAgRVgOv5RF7MZEit2AAmQAAqgNOqjEAkAAReQAj0gBWKQBnBwB2NK0ARtlQETg0UKoTIFIDDqCBTmgwEkEBE/MARHsAMuEAIzyWfoWV4AE0AhcUm5AwETUAEWcAET8AALUBsjd0Rx85xB9aAQGqECERAAIfkEAGQAAAAsCgAJAEoATgCHAAAAAQEBAgICAwMDBAQEBQUFBgYGBwcHCAgICQkJCgoKCwsLDAwMDQ0NDg4ODw8PEBAQEREREhISExMTFBQUFRUVFhYWFxcXGBgYGRkZGhoaGxsbHBwcHR0dHh4eHx8fICAgISEhIiIiIyMjJCQkJSUlJiYmJycnKCgoKSkpKioqKysrLCwsLS0tLi4uLy8vMDAwMTExMjIyMzMzNDQ0NTU1NjY2Nzc3ODg4OTk5Ojo6Ozs7PDw8PT09Pj4+Pz8/QEBAQUFBQkJCQ0NDRERERUVFRkZGR0dHSEhISUlJSkpKS0tLTExMTU1NTk5OT09PUFBQUVFRUlJSU1NTVFRUVVVVVlZWV1dXWFhYWVlZWlpaW1tbXFxcXV1dXl5eX19fYGBgYWFhYmJiY2NjZGRkZWVlZmZmZ2dnaGhoaWlpampqa2trbGxsbW1tbm5ub29vcHBwcXFxcnJyc3NzdHR0dXV1dnZ2d3d3eHh4eXl5enp6e3t7fHx8fX19fn5+f39/gICAgYGBgoKCg4ODhISEhYWFhoaGh4eHiIiIiYmJioqKi4uLjIyMjY2Njo6Oj4+PkJCQkZGRkpKSk5OTlJSUlZWVlpaWl5eXmJiYmZmZmpqam5ubnJycnZ2dnp6en5+foKCgoaGhoqKio6OjpKSkpaWlpqamp6enqKioqampqqqqq6urrKysra2trq6ur6+vsLCwsbGxsrKys7OztLS0tbW1tra2t7e3uLi4ubm5urq6u7u7vLy8vb29vr6+v7+/wMDAwcHBwsLCw8PDxMTExcXFxsbGx8fHyMjIycnJysrKy8vLzMzMzc3Nzs7Oz8/P0NDQ0dHR0tLS09PT1NTU1dXV1tbW19fX2NjY2dnZ2tra29vb3Nzc3d3d3t7e39/f4ODg4eHh4uLi4+Pj5OTk5eXl5ubm5+fn6Ojo6enp6urq6+vr7Ozs7e3t7u7u7+/v8PDw8fHx8vLy8/Pz9PT09fX19vb29/f3+Pj4+fn5+vr6+/v7/Pz8/f39/v7+AAAACP8AAQgcSLCgwYMIEypcyLChw4cQI0qcSLGixYsYM2rcyLGjx48gQ4ocSbKkyY4CCBhYaeCAAgcSIDRIkIABBAoRGig4UEDAyYUBDCh4AKFohA0kWqQIkeFCCBU0WIzQMKHBgQABfh4MICBBhAwbNGTgsCIHkiA1UJSoQWTKkRwrPlRgMEBAVq0CBRRAsMBCCBYwXriQQaTKGS9PfuxwEmbOGCg9WoCYYICAz59cCTDIYIIGESpl1rBhA2fPoUmM/MRxswdRJUFtsBShIUICg6s/9SKo0KJIFzmIOplCparVLF3AcrkiBQrVq1uiFM3hckSGh7kDfg44wOBDEDKGQO3/shauHLp169ixM+fNmjRq17IFO9WIzhYhKjpEIPDzwIMLL1jhBym6RHNOPPTck8899czjDjridMPNNtkYI0snh8ShxRAvcIDAXSUtgAEJQKxhCTDPfONOPQrmY8888LBzjjjeZFMNNMTcooomh8wBBhEpNFCXSQ+EIIMUgLyyDTnt1JOPPlDaE0876ZgzTnvPHCMML7e00kkjfGRhwwQFDACiSBfAkMQamBTjjjz26LOPPvngE8865YwjToTTJONLL71wyYoolbRRRAcMIJAdSSQoAccktFAjT4J1vgiPOd1MU0023nDzzDC20DKLLLC4soopgGQxgwcTIEBSADGQ/3HJLs+MY8899+BTz5TobPOMMMdAk402zPRiqirEwVILLpO0gcQLHjQwElY++PHLOOvMQyc+98zTDjrgQBNMLLkcsykyt5gyCiiejBLLL8qUEkgXQaQgQUgBEOBSEYQMYw478txjTz30tCPONcrUAsoijVjySSiVJNKHIIk8sgkrvSwDSyV5bLGDBmdyJMABDUywBCLFmKPOO/PIE0885lhjDC2Z+DHGF2KUQYYXWEQBBhyASGKKLszogooldTAxgl0fCbDABBtI0cgx5qTDzjvvuNPONwl3wgcXO8TgwgoonGBCCT1cEYciodzCDDG8zAKJFytYFnJGBETggQpeUP+yjDnnoINOONlEA4wqlPThBRAhXBABAwkowEADKyyxxiGg5AINM8kUE4oaM/B0GUcGZMDCD258As055dCIjCybIGIHGVL4oIIFDyxwQEsIJJCCEmwoQoov1kwDTTOowIFDovx1hIAIO1DhhyrVQMgNNa8sYgYUQtBwggcU7FTmAAMQUIAJSrjRCCrEdKPNNdjf4cPjBtx9kQIpHGEGI7JgYw430jBGJcgwgw1E4AAJ4QoJkuCGR7AiGeQAh/tqwQci4O5DzguBDqRAh0voYhiw8MQi0GAEEkyAAQVACFcGMAIjpGERq1iG9awBizwIoQIOwCDpTOeDLvSBE58ghBr/oMADFFBAd4sySADKJ4IhgCcVzGgHObYhDVbYAQgUsIr9LJI3D6zACGL4wx6wUIMKPEABPdkiAAQwgAKEAAhfCAQqnBGPcmTDGaeQgw+qghuOEOABGihBDpwwBi/0oAMOGVkDJAADKMAhErOoxjzEEY1gbEINObBN/TaCFQNEgAMomIEPkjAEFlTAIX/cwAmQsIZHtKIY3pgHN4yRPTDQgH5qlAgbDzCBvbHABS9IwQak1RADYCAFOwDDIVqBDGycgx7X0EUn9FCFF0RgAZvMiF4O4AANmEAGLTDBBzAAAQQ2BAEguMET7PCJY3zjHO+ghzRi8Qg3NGEFEMBmLiFy/wAIZIAEMxACFJDwgxqogAMO2CdXGpACMB6CFdI4xzrgEQ9pyGIScXDCChyAxo0wYAMr2AETuuAGMmCBCUWkgF22uMS8zYAKdcCELrjhjnfA4x3SoIUl6PAEFTBvIxE4QQ+kUIY8NAIQcRBDE2iwgbqwVCUW4IEYEoEKZJyjQVOKRi00cQcopGAnzbNISwF0hC/YwRGm4IQjAvGGLAihBStIwQlG0AEMVAADG/DACFLwgh+EYRClIJ47dCUPd2g1E3VwwgkQUCaM6EsBHlBnGgohCl90aRSWIIQcwuCFLWDhCUTAAQxu4IMiOCELY5iDI1RRjGqQYx4uogdOaXGJOf8wgQRlGh1FgrIACJTAB1eYAyVqUTxmGIMXryBUJBqRCD6o4QpLsEIY2JAHRFjiFLuABjkA9iRdwUMas6hEHJIQgpVWBCsEcMAF0gkFNRjCFMZoHTeskSVf2GIWsGjFJxiBhzXggRCRAMUrfuEMbqjjSfvghz661Y5mtIIRaRiCBy6SL5J9IAZD4IIdIpGKXmgXgMcAxi5uIQta2OIWsUhrJT6Riln0AhnT6MY54CGnfezDTuoQhy82gYcr4CADF9kOAywAgyWYIRCcuMUyprGNcVjjGLiIBSxe8Ypa7CIYwwjGL35hjGZQYxvhOAc74mEPG8+pHusQBw0VMQYjtGD/AhchwAIk8AEhlAERphCGN9yhDnOIAxq9SEUoSoEKVtgiGMygBjfGkY432QMf+bAxPyY9p3zIwxzZUEYo8PAEGoTAAecNAAM60AIhjKEQp+hFNMrxjnN84xrFmAUoLoEJTGhiFb1whjbEoQ4E3UNOk+6HsPvBj33k4x3dUIYtHoGGIKhgAwvYbUoqUOQzGKIUxaDGN9TRjnBYYxm6SAUmHLEIRCACFLiAhjfM8Y5HA5sfwxZ2sfXBjmnoAhRhqgEIKpAAiqTEAB4YQhoYkQpjjIPb71gHN5rxi1d84hGE8EMe6kCJWURDHOugR4LhHe9hz1sdyljFItSQBBRc4AEG/6AIAiCAgRp0wRCrGMY2WrYOcmhjGbzQLyQEoYc70IEOlbDFNeB5D38Y/ej+kLeC70EPcfximl0AwgiymEJdVuAEOcDCgFpbDjRvA+fJtYTE4OCGOvChEKMIhopgi/SjE5tO9HBHOqohtzdc4QcnyEA5JZKvEPhgC3z4hDDe+Y57kGMZsejEJBLxBzqkAWdzIIQlYsEMdCRoH203ej/0gQ97uIM9yDAFIR7Tg/y0KiK7dIEV+vCJX3wDV7jiRi4u4Qc7vOEwVmiC2hhxil9cg8YJ7ri88UGPS20jGrvQRB60sIQeuCAEFlDAPglCgAb4JQly2AQunmEOz2PKF5qow/8WqvCEJCDhCEWwwh0ykYtmgGMewY/3je0BD3FUAxm/wAUsRPEIPqQBC0dgAyegAQ1gXgyBABiAAjpQBo+QC9IADitSDtRADKGgB1OAAzQAAywQAzfQA1XQB6bADNqADvcQbB6XD7JFDs1gC6CQCZTwCIpACH+QB2UQBZHhARBgNw2xACMQPYDACtgwUQTDDcOgCoSgBTBQATHRTShQA1QwCLPwDegQD/rQcZNmD++ADtiAC5hwB3NAdnGQB4KACHKwBUUwAyNAJpbREA/QAkugBpPAC+YAI+5QDseACopwBhK2OyvRATbwBHCQCcOgDvBQD/tAbDdmMNfgDMbgC67/AAlzYAVTIAVRIF1sUAdi8AQ8wAI4mFsNIQE2oAV+IArG0Gvo4A3PwAqJYAZKkIQF0BIJ0AJUoAecgAvVwDL4UGwL8g7UkAueEAmI8Ady8AVLgAM2UAM1kANBkARPIAQ1kALXsQBO1RAV8ANpIAmxAA029Q3OkAuRkAZBwAIcICQGUBM9IAem0ExX9WsK9iLmsAuQUAZUsARBcAMtMAIZkI8awAEgUAIpIAIcYAH5ZABM0xAYcAR2MDzZ4CDYIAyo4AdTkAIZEAGMxQAToAFT8AjLwA6FiGD7YA9SJA2ekAY1IAIZ8AAIUBkCwEbmcwAIoAAJkJKe6BABkAFLwAcx/8cNsqUNxcAKhyAGm7hvFZACPRAFfaAK2AAP8/Bo3HIP5aAxk+AGSUACjoNGdsMVLNkSBlAABDCNNLkBTiAIs5AM3iBb23AMsQAJcAAFPgADIlACRnAGiHAK8VUPtyIw9CAP1YAKe0AFPpACWZQAuXUXWLGS5EM+Kzl9A4EVHCAFh6ALzhAO9YBsCaMJNgMFP/ACNSAGkOAL0RAO8VAneBkP7kAMhnAEZpQAVwkSNakEe6AKwpAN8QAP4TANw/AKm3AIecAGXjAGifAK1RAOGYdm4pANz0AMulAJYuACMzkSFgAEakAJseAME3UO3UANyMALsYAKoIAJmzALy0AOhP9YD+5wlrLWCHpgSB3glSMBiqIICv4SI+hADuHgPvATDdLQDegwKXaJDsrACo4gB1WQAyaQAQygmBrhACygBGkACbfQDenBDu3wDvxpl3d5D3dCDs/QCiO3BC3gAAV5EgzwW1jwB6fgDN4QDuNgDusQT3YpMDDCDtlADLCACX0gBkewKgmAFVqRAB0QA0bgBpfwC9FgDdsADufQDvRwK52XheEQDJ7AB2HABDhQAhrwAAWAoBxxABVAAjXgBYcAC8OwDNKwaxmXKwvCDt6GCniABCmAUAaIFwUASCUwBGHwB45ACZnwCabACiVWC6HCCqOwCXlQBTKwARLgKnhBEANZkAAPYAEKyARVkAVdEAZlgAZs0AaaygZnIAZdkASrAgEdtaiL2UYkMwEZ0AEfEAIjQAIlYAJnQwIkIAIgME4RgEZmQqpAwUZ1Eae6yndYiRVa+qsqRKwOERAAIfkEAGQAAAAsCgAFAE0AUgCHAAAAAQEBAgICAwMDBAQEBQUFBgYGBwcHCAgICQkJCgoKCwsLDAwMDQ0NDg4ODw8PEBAQEREREhISExMTFBQUFRUVFhYWFxcXGBgYGRkZGhoaGxsbHBwcHR0dHh4eHx8fICAgISEhIiIiIyMjJCQkJSUlJiYmJycnKCgoKSkpKioqKysrLCwsLS0tLi4uLy8vMDAwMTExMjIyMzMzNDQ0NTU1NjY2Nzc3ODg4OTk5Ojo6Ozs7PDw8PT09Pj4+Pz8/QEBAQUFBQkJCQ0NDRERERUVFRkZGR0dHSEhISUlJSkpKS0tLTExMTU1NTk5OT09PUFBQUVFRUlJSU1NTVFRUVVVVVlZWV1dXWFhYWVlZWlpaW1tbXFxcXV1dXl5eX19fYGBgYWFhYmJiY2NjZGRkZWVlZmZmZ2dnaGhoaWlpampqa2trbGxsbW1tbm5ub29vcHBwcXFxcnJyc3NzdHR0dXV1dnZ2d3d3eHh4eXl5enp6e3t7fHx8fX19fn5+f39/gICAgYGBgoKCg4ODhISEhYWFhoaGh4eHiIiIiYmJioqKi4uLjIyMjY2Njo6Oj4+PkJCQkZGRkpKSk5OTlJSUlZWVlpaWl5eXmJiYmZmZmpqam5ubnJycnZ2dnp6en5+foKCgoaGhoqKio6OjpKSkpaWlpqamp6enqKioqampqqqqq6urrKysra2trq6ur6+vsLCwsbGxsrKys7OztLS0tbW1tra2t7e3uLi4ubm5urq6u7u7vLy8vb29vr6+v7+/wMDAwcHBwsLCw8PDxMTExcXFxsbGx8fHyMjIycnJysrKy8vLzMzMzc3Nzs7Oz8/P0NDQ0dHR0tLS09PT1NTU1dXV1tbW19fX2NjY2dnZ2tra29vb3Nzc3d3d3t7e39/f4ODg4eHh4uLi4+Pj5OTk5eXl5ubm5+fn6Ojo6enp6urq6+vr7Ozs7e3t7u7u7+/v8PDw8fHx8vLy8/Pz9PT09fX19vb29/f3+Pj4+fn5+vr6+/v7/Pz8/f39AAAAAAAACP8AAQgcSLCgwYMIDwYIIGDAAAEBEkqcSLFiRYYEDihwsABBgYcRLYocSRIAwwEGGEjIQMFBgo8hS8qcWTDAgAIJJGwY0aFCA48CaAoVuXAAggYTNIhQMePHkitLfMAggYFBzKFYDzY0EIHDiRpDpIB588dRnzRWhLCwADGrW4I2CyjYsIIHlDJ5GoGSVcxVJT5fgIAA+TbrQgMNKHiYcYRLHEOcXgGLJi4Zq0dvlpggMOBqYZk2CTwAEWNIFjiHMKHCZSzaNnPLXEWCwwSFAQJBP9MUUOBAhhlNzgDKJAuYsmrdxJlT5wyWpDhNVCAwMEA3TZwOTCBZw8gUMG3l1MH/mxfvnbtosibJcbJCwQECnq1LjItgg4sgWuxIWuUrGrl178xDD3nvQPOKI20skYJ78MlnUQA4PRCDFHQ4Mgouy1wjTjvxzFOPPfTIEw80rSiCBhIoMBifg1oh4MAFR9xxijDQfAPgPPfgo+M9IMrzjCqFjFGECQlQtyKLBR1AgQgyiDHJMNuU8849VFbJIz3lJQMKHlbsAMIBHyFJUQMl+IBFIKpMYw47ONZDnjweYtlOOr9IUgYRLmTwUW5iIhTABDdo4ccnwYgznj340AMPO+3AI4887aAzTi2ESCFDCBG01adWGXmghB2h7FLNO/mUis88kabTKDzohKNNKnUE/yGCBQpsmpACFYCwgxmR6NIMOPOYeg+k6KTDzjvrdPNMMJSUYYMGESBgK0ISmJADFoKwMo2N9uiTDz72xJPqOu2cI00vqPxBhQoSMFDAtAdpYMMUdXhiDDvx1KOPt/fUI26x6rAzzjGrPKIGER4oQB28BoEgRBmItBKNh/d8C2I87qhjTjnkjHNNLZPEMUUNFtzGJ8MmldBEHZjgcg099HxYjzzwtKPOOeSAww02yISShxQ7lPAAbkf2udAKWBSSyjDd1ANziO+wczM54WxDjTO6NALGDCNYgMBCKAt00wE1mHGJLs6MY4/T88DjzjrpmEO1NtEgEwsgT3gwwQLVhf8NgAAKRIABEnq04gw36uDT74DxuA23Od9UwwwtgTzBQQQK9B02ARB0kMIWjwwTjjry6KP42jC3/fY4dOMyiBMZPJCA5igbcEEKOrQxSjb1VLyPtzoG72Y86HhTzS6ENHHBTyfDG8ACKBQhBiO3iINPPvr8/q2OpfY7zzriYOOLIU74dEDztjIUgQ5iLLKKMupgv++3VF5vMT3tlMONMIg4QQED5wtbQy6wBEDUYhnekMfvgKe4HGEvHzxiFTiMoYgn7M0A6OuTARxgAReQQRM1Wsc9+KG9HV0ve/SLxzrIwYxJbEEEGIDASwpAw4f0CQIl0MEVCBELb6QjHvkgobf/toeP7P1OccSahijkIAQZkOACEIDAAx6gsKK9RQM88AIgSIEMdsjjHvvohxCHiL198MN095gHsrQhi0SEgQk4KMEGNqCBDEhAARn8DAmkEAhUDMMb+TBjP8S4wH1ljx9nBBeW3BEOYHziD2ZoQg1QgIITmEADQ3PQADQigzFUghfRKIc+EIlI7ZWqVEYcljukho5uCMMUi9DDGrYwhShAwQk4GIEDYPKZABzAART4AR1SUaN2mHEfJcxR8OQ3D3bIbRysI0YrLvEIRASiD3vQgx2ucAMMKIx2WRHAAhTDhEL0ghzsqAchgccje9jDgd6Cxzm8wQ1udOMaxqjFKEhh/4pTnKIUo/BEHZYgAggogACFgZAEQOACLkxCGQHChxj51c53wjMf8vxGN7zhDW0soxeumAUufAEMX/DiFosAQw30Ji23CIAACeDAC4LQhk9IYx72COI+IEgldypTfvpoJsfIQY5vSKMYufBFMZjxjGYswxgCZYIMQPAAtwSAAIgpQQ+q4AdWaOOER3SnRe23r9/VAx7rUEc60lGObDhjGMqQxjbouQ1rwEIRZDBCCypgRZEYpQEYoEEU2jAJXYgDmftSnNM+xD1Desse8jCPO+YEjms4oxreOAfc0EGOYHhiD1rYwQY0JZQEYOAE17qDJWThDHVo70qPosc7TwguAv9JbR2MWkc5vpGNb5ijHe9oxzrOoQzZuAEJIuBMXynyABQAYQt86IQvpCEOBUJwePBwVO9OOKw5nWNj5wiYzXCGjnbIY5HqoEYvTiEIK6gATHkUSQVwoAU+gGIY55BHt/aBj5nBI7juiAc9flqPdpgDHAgGBznQwQ53vOPB8qiHjmbmjm40gxeWIEMMigROvxYAASJAQhskIQtpyKOM/O1R4w5FVnvAQx3liHE50AGgRwkIUfxSFDmwwYxU4AEJILAAA+IrEcRQAAZYAEQpfqENfc3vdPVYbI7Kyl9FtaMd7Ghwhz40W/n1dx7qEIc2dvGIMfRABRd4V0lw9YEeoEH/ErdohvVKGLxlHpKUO13cgGTrQPkh9rrxYAc6mmEKP2TBByI4wHJr8oAOsKAJeCBFMrKhjlFCsMt+5scgB0lCRc6DZuetGFCfvLZPw8OuicjrCb626IEEgAIoyKIiapENcrxjX2mEU+9+OkoxktAe70CHOMBBz3Ccwx0C4rI72da4VUKDFHiAAg5AgADSXkQDM2ACHDZhjHS8Q1+JCq52o/xO7JHwd/Iwxzai4YxlIAMa4AFuPB71qMYFV2rpSAYmztCDFWgAATYUiQBA4AMuCKIV2ABquOB2LDgl23faq3AzfrGLW9DiF4c7h7Ee7I7JsiOt6DhHOYTBCCukgKVE/6PIABLwgAvowAuCGIUwDgu8d3gsGtbQBm+n0QxkEIPiucCFLVoRiko0ohKdKIUtlOGNBjtKRPaeLMiX4Qk5NAEINEBBCUgwgg9YoAF7MogBJPABF1wBEKkoBjbYsVNwnYMawhgpMZThi1iQAhOMCEQd5gAHN6AhDFrIQhsEcYlXICMccJJtlAckov8Ktxq0kEQdzpCFJSABCUcAggs4oLAGESSmLyjCHEQRjWPbgx/5sMc8vmGMVnzCFK2gxSgesQc0VEEIMGhBCkwwghB8gARGOAMiTmEMcjSWfssO0X+/kQxZdKIRfnDDGtjABjEcYQUGxWBBGGCCH2ghEbg4R/8RzZj6eXRjGKjIxCdKsQpL/AENUuDBCSYAAY40IAK5MgIbJAGLZqQDmX8GZYwDD+ZwDciQC6vgCZDgCAwoCGDQAxsgAXxTEA/wAk7wBpgQDOtgRDsFIuLADLWACqrACq2ACYGgBrf3Ah3gAR7wASYQAzzQBG8gCbSgDN0AD6SUSCqWXfEQD+rwDdSQDMGQC7HwCq7gCqDwMyvQARJgAAUhATjQBYNwCsvwDgCYZ/ZwDtUwDLZAC7IAC5pACG2QBUhwAyiwAi4gAz8QBWSQB5hwC9ZADu1wD76GeqpXHqvkDo7iDufAW9ZADdEADc/gDL4wCWewAymQAQlwFRUABGv/UAm0MA3ysE70kyzP8HO5MAuegAhyAAZRMAQ2wANCcARZIAeLQAq/sA2nt2mE1F/lIVwbJyIZQ2PAZR7ssA2rsAdLUAMgYBUxQQE8QAaLwArMwA4N1E4FRg7aIA3PwAzJoAursAmOQAh6IAd2oAd+oAic0Aq/IA3lgA+axmlHNDy2+A4C1jbCJW942A2xQAhY4AMm4ACkJQE3wAWBMArFcA7zsI8CAjMvljPg8A3fYA3NQAy9gAuy0AqvIAu0sAvF8Ay1dmubJkQW4yaNp2tQI2/72DjgcAuOUAZI0AIRQBgAEAEzUAV4sAm+EA4O9g7wMG+m1nHaRTOrtFnpoA4B/2OOA6YPrEhC9MMji3djPEIejhIzFjkOvXAJciAFM0ABKQcADYACQ+AFfLAJuYAMz2AN3RAO5WAsehhhQRSOrMhpOSiWpURRa8M2Dgc1UoOTGhMO3NAMqoAIZpAELSABJIkAGVAXUoAGfQAJoRALwRBX2yAO58AhEoZIY+lriEVlV9h2QBkizdZxqyQpOpMNBDkMuYAKibAGTGADIdAApDUACzABG4ACNSAEWCAHjUAKxeEM2BAO6QAP9IAPgjSWZ3RKjeVY/FJqNNmWa1UO4LAN1fAMyTAMtWAKljAIaNAEMtA1LXUQKSEBKZAEaWAIlnAKt0AMz6AN48BgApZTHP94XYxnY8kmM04TWezAVuHQDdywDdpgDdCgDMOwC7KQCplwCHQABkowAxywNx02EASAAAzAATKQBF2QBnUQCIxgCaCQCrGQC8DQDNnwW+aonnEjDt6wDdmADdaQDd6gHOigDuUCDtoADcRwC6ygCqhgCqPwCZtgCY1ACHvwBmFABUVQAyQwAcyTEC9lAA/AASlAAzxABEtABVxQBm6QB4UACabAC9MADuMQXubQntcwn8QADL2QC79wDFm5DeBgDtwgDcdwC6LgCHqAB3sXB2/QBmoABlfQBEXAAzOgAiBQAQvwHn1lEwaQAAzwABJgARwwAipAA0IQBWDwB5/Qjdj/kBzcIJ/I8AtDdwqiwAmkAAu8YAzOYA3eEA3DQAueEAhkkARGMARA8AOo2gMzkAIfoAEXMAEPsAAY1GomgREGQKARYAEeoAI2IARbYAeQAAqlkAqtgAqk8AmYAAmJIAh9gAdzoAeE0AiW0AmjoAqgQAmIYAdeYAS5twKUhAIpcAIggAER0AALkADwVRIL0RC9sQAOIAEaEAIpkANJkAVjcAZq0AZpYAZj8AVaUAVQ0ARKgARMIAVY0AVhUAZqMAZcQAVJoAMr4AEcsAEZULEZgAEUQEUIcAC30RkzsRCh0RsK0AAQIKgjcAIpoAIsoALhWgK99wEe0AEc0AEgIAIkPWACKGsCJBACHGABEbAAC6AACTC0QsuxnCEAEEGrH8sQSes3bwGySuu0Uju1VFu1Vnu1WJu1Wru1XPsWAQEAIfkEAGQAAAAsDAAFAEsATACHAAAAAQEBAgICAwMDBAQEBQUFBgYGBwcHCAgICQkJCgoKCwsLDAwMDQ0NDg4ODw8PEBAQEREREhISExMTFBQUFRUVFhYWFxcXGBgYGRkZGhoaGxsbHBwcHR0dHh4eHx8fICAgISEhIiIiIyMjJCQkJSUlJiYmJycnKCgoKSkpKioqKysrLCwsLS0tLi4uLy8vMDAwMTExMjIyMzMzNDQ0NTU1NjY2Nzc3ODg4OTk5Ojo6Ozs7PDw8PT09Pj4+Pz8/QEBAQUFBQkJCQ0NDRERERUVFRkZGR0dHSEhISUlJSkpKS0tLTExMTU1NTk5OT09PUFBQUVFRUlJSU1NTVFRUVVVVVlZWV1dXWFhYWVlZWlpaW1tbXFxcXV1dXl5eX19fYGBgYWFhYmJiY2NjZGRkZWVlZmZmZ2dnaGhoaWlpampqa2trbGxsbW1tbm5ub29vcHBwcXFxcnJyc3NzdHR0dXV1dnZ2d3d3eHh4eXl5enp6e3t7fHx8fX19fn5+f39/gICAgYGBgoKCg4ODhISEhYWFhoaGh4eHiIiIiYmJioqKi4uLjIyMjY2Njo6Oj4+PkJCQkZGRkpKSk5OTlJSUlZWVlpaWl5eXmJiYmZmZmpqam5ubnJycnZ2dnp6en5+foKCgoaGhoqKio6OjpKSkpaWlpqamp6enqKioqampqqqqq6urrKysra2trq6ur6+vsLCwsbGxsrKys7OztLS0tbW1tra2t7e3uLi4ubm5urq6u7u7vLy8vb29vr6+v7+/wMDAwcHBwsLCw8PDxMTExcXFxsbGx8fHyMjIycnJysrKy8vLzMzMzc3Nzs7Oz8/P0NDQ0dHR0tLS09PT1NTU1dXV1tbW19fX2NjY2dnZ2tra29vb3Nzc3d3d3t7e39/f4ODg4eHh4uLi4+Pj5OTk5eXl5ubm5+fn6Ojo6enp6urq6+vr7Ozs7e3t7u7u7+/v8PDw8fHx8vLy8/Pz9PT09fX19vb29/f3+Pj4+fn5+vr6+/v7AAAAAAAAAAAAAAAACP8AAQgcSLCgwYMIEypcyLChw4cCA0gMALGixYsBBGiUeLGjR4QZB4gUwPGjyYcBBhh4gAHEiRc2dOzg0eOGjBYjLjAYQPKkz4EBCBhwAELGkChf2tTBs6ePHDRdksTIYICAgJ8/BRhIUAHGkjN7Ho2CVUtXr1adGsFZciKBgQFYP6YkoCACBhRI0CDqJMvYNGzdwkH79eqRmR4bJiigGNdigAIJGmAoIcOImUOnbhmr9k1cuXPbnAkzJagLDxUXCDS2KCABhAsncjAhQ6iUsWjbyKlb186duW7VfHXq4wXICAOrUQZ9kIEEDihnBIkaNk6dO3n0ss97t+4cNVyc/Fz/gZEgeUMCCyR0kHGky5tAkkrtkoZOnf10+PHb30aMVaQyOTTQk3kJJWDBCDVUYYclpsTSCzLVhNOOOuaE88033ngzDjrsiPNML6TMIQQEBAzAGIEGNRACDU3kQQo03YhzzjruwCOPOuJkUw01PHZjzjvpbNPMLYEoQUFVJ6IIwAEOVHDCD1bEkUkw6MgzTz30xNNOOt9QkwwxwwxDjDTguMNOONYIw0gVHziQAFxKAhCABCTY0ER0nOAyjTvzyBOPO+eAk00zwdDiyiuwyFJMNr2Vw00ylYTRggYQIBenAB7oYMUckLRyjDXl0FOPPO6g442Xu7TySSabcPIJLtGs//MOOuFE80kbO5RwQXkoSlSACk/QUcks0ajzDj351AMPjtUgs8sqmihCCCKMROIKM+ksW841qOSxhAwfNICiVnXlEAYipwCjjTvx0GNPPOqMkw0zwMziSSJzvLEHIpXAssw57qRTTjaxJAJGES1UUICJyQ2wwAQdICFHJ7o4M84882TXDjnbREMMLqtEckcXWrxBSCayLFMOO+mYw40umNiBhQ4duAVnYwVAwEEKVxAiCzPcsIOPPfTIk4430yDDyyujFFKGEUWI8QcntCxTHTrmeDMMKoysgUQJDSSg2moLgEDDEnNsMgw25LxjjzrfTFOMLadoAokhfZTxLQ1QtP+RSCnCeIPOOeaAYwwrksChRAkMIDA2VhlJIIMUckgiizTVySMPf61gUkgcYmxBhRM/vAACCT1YAccktmRTjjkVHuMKJYqPYHNcIWVAxBuZxKKMOO3AQ887zKhyCBtT6JCCCSJ8kMEEDVSgApSAqDLNOK+Lg8wrlsSRRAgHWBXXAAgwYMIViNySjDayUqiNLIuMcYQLFZR41UQKcODCEHF80gw42BPHMVgBCTYUwQMLS9JJFmABEfygDZu4DTfO8Y1m7MIUTiMCDNrEkxMFIAF30QEaLmEMbXyDHOEAxif80IUdaMAqCjSJBEqQAywIohXVMOE4oAELSMgBCz8oQQb/IHCAkgzkABQIQQzA8AhfUGMb4wiHLibBBifIwAIdjEsGaPAEOGACGOSIIjiCMYky9GAFGTjAgAySMw2g4AqHsEUzsCEOcMziEFr4AQoisEa5EOAAClABE97wiFb8axvPGAYp8uAEFXQAAjc7CBJFMAMxSCIY1+gGOb5xxzzusY8eEQACHEABHpThEa4QxjXQIY1dlKIQX8hBBiKQgKsUSAMs+MEbPuGMcJDDHJw8BBeAkAI+xvAiA2CABT7QBD+8ohnXEEc6kIGKQ5iBCCNAwMIWUrYaMGEPq9jGOtTBpVkgogtBSIEEQFmRjKBHfz9YQyaO8Y1wjGMcu5jEGpYA/4MKLKQAC4DAB2zghDNEYhflgAc7zqGNWBzCC0JQwTqP+RDyNUADOLACHShRi2pschvUQAUfoECDEDgAJBqJAAhgQIQu2AES11oHPNIRDmmsohAQVUEEjHiRoUiPCn5IhS+iQQ5Tya0SZahBBySAAIQIYAAE4IANnpAGQnwiF8zwRjzgUQ5tKIM0XhiCTk+SAAp8QAdvKIU2yLGOeLBjG8mwhSCewAFt2nIiGhEKAgLZhDc4QhXFAMc53nEPeIhjGsEAxR/CmgIIhFIoTtJUIWgxDtiZgz+usAQbhpABBSzgsw14QAQyMAIW0CAHPyCCF/JQCVYEoxocioc91oENYv+44hFveAIORCCujhAAAWXTARXSNox1oEMc3FCGKyRhByvY4AINgMAELKABD4ggBkOoAhnYUAc/RMIUuljGNcgBj3hcyRzOqAUn+vCFILSAAwvoSAAMwIAIsKAJbnhELKQxD3V0Ixoway4QVEABCFhgAyE4QQtmkAQy/IESoFjFLYpBJuvMAx/3wFI4iGGKRLDhCTMQQQWa6phkVsADPhhDIlQxjG7YIx3YQIYsLKGHL0RBCDSoQQ+KwAQpXKELcVCEKGoBDGVQwxvngIc97pEPfeCjHvPwhi888QcxFOEEQ7QURDJiAAqIAAZT2AMpiFGNctTDHIh9xScc4Yc7wAH/DWywQyAS0YhIVEIUshCGM6qxDXGgwx30wEeT9XGPLG1jF5rQwxZ40IEILOBxDnFnAvQXBDRM4hfeMEc76iEOZ+SiFagYhSc60QlOkAIWvDhGMpbRjGloIxznUEfwroQPfdia0NvJBi4sUYcq0EAxBrDlQ8glgRQEAQuAWMU1MJYdcCzDFq6QhS128YthIEMa32DHPfAh6FvnIx8Y3ja3BZ0sUl2jFpOQAxRcsACeWAQBSYxBE9RgCFIMQxxY8tNhiVFtZDTjGdJ4YjncseRagzvDWkLH4M6RDnYILx7vaMc1bFEJOkShBQpgJ0MaIILnxEEStEjGNtqBjyy5oxwd/3sGNUwIDnCEoxxtLbiTiTbTcGQDG9ewRjbA8ed2tIMd17jFJewghRYkgKILmUAMnuCGSuwiHHyqtbK6M45vgMMcbYXHO9zxDvPWY9tOrkc81gGOaiwDGccohpG/kQ77rOMauMgEHqbQAhKjBKocAMIYDpEKZrgD7PiQBzvMIY5vbGMb36hs4bdx82tUAxrLMMYwgOELXuTCFrOwhS58YQxsj9M+1ahFJeoABRYcoCIDOMACTPCEO1DMGvMA95MBBQ5uXGMa0ZDGNKjRDGMAYxe3mIWqIjGIP/jBD4BIRCQyQQpX5KIY0xDHO9ihDlbGIhJwaIIKTg8RAywgAjDwwv8iaKGMcNxj5jcaxzYev4xjGIMYwcCFK0jBCUs8ghBtuAIQaOKDImCBDX8QCaJAC8RADeQwD+5AIc/QCl2TBCjAfQ+RABKwAT3gBp6ADNmQDrZmD4InL9BgDL1gKKpwCqOACY4wCHswB2vwBUbAAhLgABCgHjcgBWxACJvQF9ZgDvUAKOGADKUgCGAgBCSgZQ0RAA/QASwABYAAC9gwDu9AaLMSN8ewC6hgCYOAB3UQB25wBmCABVLQBEgABC/gAQzgWcrkFWUACJxwC8ygDehAD+oADtagC5TwBkxAAxxQAFtWASnAA5YkDH9GD/owD1wyDcRQC6awCG4gBULwAzz/kAM1EAMuoAIoUAIhgAEPYAAFUAAGAAEq0FJ9wAm8IA3ekA7zYA7z0gqEsAU3YAIVEEkMEQAbwCJ1QArT0G36YFjVYAy0MAqR0AZHAAIMkADhUyI8oREkkSQJMAKpkwedEAzYEA7rIA/iIA3B8AlzcAQjgAECAhEDUAJHgAaOUAvcoA8Hlw7XUAyxoAmDwAZQMAMVUBXChlIDEAE1kAV8kAm0AA3gQA4NV1ux8AhlwAOzdHQoUQAvoAWEcArFMA77kA9EIw7LQAuc8AdjYAQzAAINkEUKIQAFgAAakAR4AF7N8A0LVw7OcAuKhQUx4GgGgHQFkREHoANtEArBYA3r/7APJRcP2/ALpGAIZCAEIGABDlAAMDkQ5MMAI9AFlwAN3DBw2uINxJAKj+AGSmACCPAWkaYSDFAEfVAL0RcP+2APy/IMrtAIbMAELLAASOIQCoABJiAEehAL6PAO8iB25+AN0nALm/AHYPADHgBDDiEAB9AAFvAEiTAMpUgP+2A04PALmAAHT3ADHxA+87gQGGADVXAHn2AM12El7fAN0SAMp8AIcVAFN6ABGocQDjMBH6AFlQAN6hAP5+cO4lANrxAIUxADIjABgukQJVAFhcAKxMANotIn6GANxCALl9AHY5AEL1ABR1kQBfAAbkQGoNAN9HB++aAO2rAMn8AGOP9QAQ4AgTGZEU8lEgRAAAVgA3QgC0gGD4RGD/AwDs2AC6KACG5QBT1wAhFQEQTgABdQAmLwCdp5fvhgDtUwDJhQBjMQAQzwkhLxVAYAXBTQASbAAjBAAzjwA0WwBG+wCcigaYL4LuzQDcSwCpKQB2BwBDNgUgAaPSIABp5gnEx2D+PwDLsACV/gAmFTABoBVapnYCiQA0tgBV5QBm6gB4YgCanQYu8wD+dXD0ByDbvwCYOgBlKgAyigAQwwnQTRmh/gBZ1gnPYAkd+QDLKACFiQAsQIpAPAiXWBASGwA1XwBoCwCJYwCrVgDHS0Dhc2aPTADuUQDbIgCXPABUTQAh7/MAFHB6YCMQCusQFWIAnOUJf1YA85yguWsAY/MAIiEAIgUAIq8AI1wANDsARjsAeTIAqtcAvCcD2BamtDQw/ooA1Dkgl6wAVJUAMhQJ5FBKkAQJjRwwSDkAvVIA7wcA/+xUOLYAZO4ARN0ARXMAZuYAd7AAiGcAmZEQzI4AzWII1namv5MDzqUA3AoAqScAdcEAQzoCtuAmlFCBkQEARzQArCYA3qgA/wMg7KsAqMgAd5kAd6oAic8Kq74AvCoAzSsA0AZA7pcCyDBoUUNAymgAhtwJ8m4AEVwAAHAKTtpBILYANi0AitgAzikGFF8w2y8wmgEAqjYAvLAA7s0A5d/3cl4sZt33ZwytJVZqkIZ7AENzACENA4v1kRAvBbgvQGkCAL09Ab71Cf0gAMtoALusALypAN6GAjGVNw4UY0EKcOm2QNx5ALAJuWNXACGLAAdsVTKJG0HnADUEBc5BBr7fAb09AM0CAN1RCfmZqzB0c0QMIx0qA0qoAJhqAGThADH0CUVcEwchEBIiADVjBZ2gAOMFd95VAO6EAj7VJrtrYPt4Zho4IjuPoLshAKjGAHYYAEL2ABNiOsDMFAIfADZuAIr5ALxOAM06BzgWEO5HQO5SAO3qAN1oB7zsAMyFAMwZALsZAKnjAJh5AHaGAFRQAuG7maHcEkfFgEYbAHi/+wCatQC71QDMwgDdjwDdkgDcxADLoAC6YQCpxwCZLgCIlACH1gB29wBl9gBU0gBDagAh8wAQigvR3hkQgwASRQA0bABXKACJcwFrswDMswDcngC7JQCpVgCHfwBmgABlkwBUxgBECgAzTwAitgAiHAARZQtAcAizhjmCGAA09QBnVACJLgCabgCrewCp4QCYIAB1/wBEfgAzbwAiggAhtgARLQAAqQlSUiux8hqQ5gASVQA0TwBFkgBmsAB9g6B2ogBlfABEBQAy+gAiQAAkssAQ8wjB9rP1LsEUGhehOwASOQAi4wAzegAz3wAztgEy1wAiGwARhQARMAAQ6wAAkXAMUj4bZxEhHoKRLqaYwjkYwT8cgGERAAIfkEAGQAAAAsDAAGAEQASwCHAAAAAQEBAgICAwMDBAQEBQUFBgYGBwcHCAgICQkJCgoKCwsLDAwMDQ0NDg4ODw8PEBAQEREREhISExMTFBQUFRUVFhYWFxcXGBgYGRkZGhoaGxsbHBwcHR0dHh4eHx8fICAgISEhIiIiIyMjJCQkJSUlJiYmJycnKCgoKSkpKioqKysrLCwsLS0tLi4uLy8vMDAwMTExMjIyMzMzNDQ0NTU1NjY2Nzc3ODg4OTk5Ojo6Ozs7PDw8PT09Pj4+Pz8/QEBAQUFBQkJCQ0NDRERERUVFRkZGR0dHSEhISUlJSkpKS0tLTExMTU1NTk5OT09PUFBQUVFRUlJSU1NTVFRUVVVVVlZWV1dXWFhYWVlZWlpaW1tbXFxcXV1dXl5eX19fYGBgYWFhYmJiY2NjZGRkZWVlZmZmZ2dnaGhoaWlpampqa2trbGxsbW1tbm5ub29vcHBwcXFxcnJyc3NzdHR0dXV1dnZ2d3d3eHh4eXl5enp6e3t7fHx8fX19fn5+f39/gICAgYGBgoKCg4ODhISEhYWFhoaGh4eHiIiIiYmJioqKi4uLjIyMjY2Njo6Oj4+PkJCQkZGRkpKSk5OTlJSUlZWVlpaWl5eXmJiYmZmZmpqam5ubnJycnZ2dnp6en5+foKCgoaGhoqKio6OjpKSkpaWlpqamp6enqKioqampqqqqq6urrKysra2trq6ur6+vsLCwsbGxsrKys7OztLS0tbW1tra2t7e3uLi4ubm5urq6u7u7vLy8vb29vr6+v7+/wMDAwcHBwsLCw8PDxMTExcXFxsbGx8fHyMjIycnJysrKy8vLzMzMzc3Nzs7Oz8/P0NDQ0dHR0tLS09PT1NTU1dXV1tbW19fX2NjY2dnZ2tra29vb3Nzc3d3d3t7e39/f4ODg4eHh4uLi4+Pj5OTk5eXl5ubm5+fn6Ojo6enp6urq6+vr7Ozs7e3t7u7u7+/v8PDw8fHx8vLy8/Pz9PT09fX19vb29/f3+Pj4+fn5+vr6+/v7/Pz8/f39/v7+AAAACP8AAQgcSLCgwYMIEypcyLChw4cQI0qcSLEixQADEDB4wAABAQEBLIpkGEDAAQgZQmSIoKDAgJAjYxYsSWBBBhMyTGh4kKCAAJkiSw44oMABhQ0iWOxQYiXJDhchKiQAWjEAAQMLKoBQoWMJlzV3BjUCNMfMkxkXYFKNOMCAggkkahgJs8dSKlm8iOFa5alPlBIg1z4MEMAAgwgdajAhc0hVs3Hq5OFb903aKTYxQKoVfBDjggkbcPpwQkbPI1TCtql7R+/eOWvGOJVpQZhzwpIGLJSgYQTLGj+TTu1Sli1dPHr27o1zpitSlxSbbROkqQDEjSZm/mB6FYzaOXbx7OH/G3/PmzFXhqyYkF5QAIEDDChwQBEki5xEoG4t04buXr58+uBTTzzrPCOLJXIkAQJ7BBkmwQcwCFFFG4Z48sovznBTzjv45DMePeyQo00umOBxBQ4YMDjQWx3EwMQZgWgiizLVeGMOePXok8899rwzjjXIjOJHFj2cEIGKAATgAAcpEHGGIqoAc007yOGjz5UB2kPPOtw0w8skaQCBQgYLIClABSjwoMUfowwzjTjyiPefh/fUI8876HADjTCVnKGDCFKpiBEHNTgBhyW8aOiOf3PiY4+d7qxTDjfUIHNJGTRwIMEBDJZUwAhChGGIK9XU498+qOp4Dz3yuKOOOeJw/3ONM5mQ4YIFDhjAHkYFKBDDFX2AEkw4+thzJzvpwLpNNc9Mg003skZzjCRdoBDBAgSwJ4BbEQDhBie8SJNOPvCY08010jBTTC6uiBJLMNJY8wwyvSRihQgNIDAAewMk8AAGTxSyyzXkxINPOtowI8wus6ByiSF2PLLKMdAc88ssgDjBAQI+sVeAAxiU8AUm0qgDTz31hMPMXqBUgogdZEgRByW3ELOLLKbcccQFLkW3VgIZqNBDHKZo4w4887xDjS6fOPIHHGJYocQPXfgBSiungHIJG0BU0LN0ATRAAg9VEDILnPHAgw4ypxwShxdL8HBDDCsggQYilkjiCCJh6P8wwUc+y0RYBC840YYlwKQzYDvi7FLJG1T0UMIFlGNAgxRw/NEHH3hYMUMEgNtGmAQ0WJGHJ8SkI88641TTSiFdFAEDBxBIQIEFNmChhyKJJFIIGDxcwPFPnBFGQQ5fEGLKMem4M042x3hiRxM3mFDBAg5IUAHyhmSCCSaUtFGEB/nuW3wAFvhwxiOtLJOOOtwwB4kZPKCwgQMHLAABBTycIckpACRFHp5QgmsVQHToA0IaJNGKZJCDHNUoBiwKsQUYeIACCjDAv0KgBDuI4hYgnMUhtvCCDECAU+dLnxkacQphcKNLv1CFH6YwJpYoQANcEcMiapEMZBxDGJZIQw//SHABBSCwAjwQAyJCsQtqUMMYuBjFHZgAggk4IAEOMEEPqKCHUBxDG9hAVynuoIQXkA+BFMBBFwKRiRkpoxev2AQcirCTBiyAAjKAQhsiUQttrAMd5gDHLAyBhR2YIAKBi0kAHqCCueDBEaQoxSYgEQgxCGEEHxgBCmgQBTcs4hTECEek0DEOXUgiDUqIAQYI8BLOKIADLxCCFdTgB0DkQQ5ooAIQWAADHRCBCnBgBCp8MQ10xCNS5QiGJ/SghR584AAdE8zHMCAC3mCBC1eYghOM0IMaAAEKYajDI1RRDGqIgzXxaAc6kqGKRagBCSZYwAGyJZh+QQADLAiCFbig/4UsXKEKU4BCF94AiEmoIhjYEMc6WjMPeKgDGrXQRB6m0IIHtCSRFhmAAiKQgRk04QxxkIMc5kCHOthBEJIQhSyGQY1yrCMe9xCQPNpxDWCk4hBeqMEEGnAAjFZEoxLYgA60kAdDJIIRkJjEJTZxiloMAxrbOIc86mElfawKHt5YBi4qkQYeYAACCfApRQjQAK04aRGbAIUpVhGLW/DiGNP4RjreYQ9+2JUfAbJT64wxijocAQQWYADxqHKACoxABlbYwydgYYteDOMYzIAGNsCBjnfM4x6p2hGr3EGObURDFiO0QQkoYD6qKKADMChCGhwhC2Iwgxrb8EY4yIGOdv/Ew1T52EeW6HHMdZgjHNsABibgsIRMHXAtDSiBD67Ah1AYIxsKnUeVAISq6jpqHulUxznOYQ5zOCMVhQjDEEigK5lsC2g3qMIcKEGLlrJjHtS9q111q9neoqMc53jfNXBRIim0AAFAMUADJpACJrxBEqw4hijlcQ997EO+/KBvnXqbLHKcQx3t8EYxWPEIMdhgKjJJAAU8gAMwJAIWxcjGoqwE4fl6yB5JQ5Y5yCEOc6zjHeWQxi9MIQcflEkmDeCACo4wh08kAxvnuId86QsgVdWjoe1QByBp/J14oAMbx3gFHoTw46AIYAIluIEVBOGKhLYDHw+OMD54K495oIz/VfBwBzvUkQ50cFcd7pBHOaDBtDbwwIheJgAGXECEMkSiF+doh2Squ496uAMdGIZH2uTMjnWo49KXbgfSxoEMVzxiDDYAtEXcU4AO3CAKdAAFM5LTIQfrNh7pAEeN19GOOafjfepYBzvacTR50AMcwAjFH7AAAxBXJDcnuIETzNCHTOhCGx1q8oP1YeVvhCPR8HiHO3j9jndIOh5troc9xCGMUAACCzEwNkUYwIImtEEQl2AFMKiBjitVd77xOIdsK0uPefhbuigTN6vzQQ5ijCIQWZCBAsS6EAkIQQ6juEUyokrXu6a5H/vIdzfCobp7xDTaDsaSq81hjFIMYgsK/2f4QTIiNi08ohjXKAc9HtyPmt/V5vNQRzjKQaUmX4lOTz5Og9GhjFUoAgw1WPhEbHICIcyBFNUQBzuUXPOq88PqPUoHeBqMJWO5Ix3lCEc3eG4PdDyjFpdIgw4WoHKDSEAFQQADI3BBjnbMQx9Vz3ve8zGgedgjt9WdxzrIwY1qOCMZ1ihYOqgBjFLI4QcMoEgGdLCFP5SCGfNgsT82z3l/5B2v/3G1qhhnDWYMQxettQY6IFiMVtxBCJGfSAiWQAdN5OIaf99HPzrP+bzv48WsYkc4qpEMXrxCFJiIRCIG4QhNlEIUl1jEHqjgAgBLJAAo2EIjbMEMcaD56rzfvP/VszTTdHjjGLHYBCP64AYyeCELVMBCF8LwBSxIAQky2MBxIUIYGKzBFNHwDe1Ac7sXfjb3e3XyDufwDc/gCo3QBluwBDsAAyyAAiQgAiDgARygARhQAQ+AAG03EIQhACwQBpbQC8hADeAwDvg1ZbFyDdHgDM7QDMywDMlwDMGgC7NACoaABnHjAiDQgRLAERwzAEY4AIERETQhAknwBo2gCadAC7rwC8VgDMHQC7iQCpqwCIZQCIQwCIIQCIDAB3cgB2lgBUMAAybQAR7YAAygANCEhAKgGRMhAANQABgAA0jQBWuQB4gQCZowCqTQCZbgCHbgBdzEAzuwAzqQAzmjcAM0IAMuUAJsGAENkAAGYAAFwEqaEYIH4R4atAEnQANCEAVfoAZ18AeAgAdvUAZJwAIPwImE4YmC4x6fwQEl4AI4IARJEAVYkAVT0ARGIAMeoABzOItI0hnboj8WsAEgUAIp0AIxEAMtoAImsAEQUAC1kYwKwSsH8I0IgAAJMI4IcAAGEDrcSBJzeIzbmI4TMYu06I7yOI/0WI/2eI/4+BABAQAh+QQAZAAAACwXAAkANABIAIcAAAABAQECAgIDAwMEBAQFBQUGBgYHBwcICAgJCQkKCgoLCwsMDAwNDQ0ODg4PDw8QEBARERESEhITExMUFBQVFRUWFhYXFxcYGBgZGRkaGhobGxscHBwdHR0eHh4fHx8gICAhISEiIiIjIyMkJCQlJSUmJiYnJycoKCgpKSkqKiorKyssLCwtLS0uLi4vLy8wMDAxMTEyMjIzMzM0NDQ1NTU2NjY3Nzc4ODg5OTk6Ojo7Ozs8PDw9PT0+Pj4/Pz9AQEBBQUFCQkJDQ0NERERFRUVGRkZHR0dISEhJSUlKSkpLS0tMTExNTU1OTk5PT09QUFBRUVFSUlJTU1NUVFRVVVVWVlZXV1dYWFhZWVlaWlpbW1tcXFxdXV1eXl5fX19gYGBhYWFiYmJjY2NkZGRlZWVmZmZnZ2doaGhpaWlqampra2tsbGxtbW1ubm5vb29wcHBxcXFycnJzc3N0dHR1dXV2dnZ3d3d4eHh5eXl6enp7e3t8fHx9fX1+fn5/f3+AgICBgYGCgoKDg4OEhISFhYWGhoaIiIiJiYmKioqLi4uMjIyNjY2Ojo6Pj4+QkJCRkZGSkpKTk5OUlJSVlZWWlpaXl5eYmJiZmZmampqbm5ucnJydnZ2enp6fn5+goKChoaGioqKjo6OkpKSlpaWmpqanp6eoqKipqamqqqqrq6usrKytra2urq6vr6+wsLCxsbGysrKzs7O0tLS1tbW2tra3t7e4uLi5ubm6urq7u7u8vLy9vb2+vr6/v7/AwMDBwcHCwsLDw8PExMTFxcXGxsbHx8fIyMjJycnKysrLy8vMzMzNzc3Ozs7Pz8/Q0NDR0dHS0tLT09PU1NTV1dXW1tbX19fY2NjZ2dna2trb29vc3Nzd3d3e3t7f39/g4ODh4eHi4uLj4+Pk5OTl5eXm5ubn5+fo6Ojp6enq6urr6+vs7Ozt7e3u7u7v7+/w8PDx8fHy8vLz8/P09PT19fX29vb39/f4+Pj5+fn6+vr7+/v8/Pz9/f3+/v4AAAAAAAAI/wABCBxIsKDBgwgDBEDIsKFDhwEESFT4sKJFgxEHaBRA8aJHggEGGEjAIEIFDRw8gPBg4QGCAgM+eoxIQIGEDSVmBHkyxUqWKDxQVHApQKZFAQQMRPjQwkeVNoQSNYpEaIwQERgcxDSqsGtEARoJFECwwEGHF0Gw4LFkq5ewY7IMZXEBQgIBiyGTJmgAQcIEChU2iEABA0cQJEygUPnCJg8iT7WWOXsWDZehKSMyaK0YcSQDCRk+jChxIkUNIU66uPGj6BGlTKBQwcplbBq4b9209TLUJEMEBUUhIt17s8SLGjdy9Hgiho4iUbqcUcv2TVy5c+rcxasHTx25YYaQOP9QUGBhwwILJGgg4QKHECZWuoARY+bOIUyshGFT945ePn37BCjgO+JUI0sePyBgwFYNMcDBCjs04QUceQiiCCSUXKKJKbHwksw15cRDzz0ACqjPieU8k0slZcxgAAHBMRQABSwQwUUdjZgCyy2/FJMMM9BUsw045rAjzz3/8cNPgCfmg882u3CSBxQoDMBRQwppoIMWeVxSCzXcjMPfPPaUqOSZaO6jj5P20PPMKYBw0YMHeHUQBBmGhOLLNuKg44489eBjJppn7sOmPO6o80skZAjBQgUXfXDEG5Gwckyf7WyHpICEnqkPPvbIw04537zCBxIoaNDARSI4oYcnt0D/k86f9SBZ4j6dKpmPPfO4Y0431HxiRgsOJHCXRRv4EAYgmtDyzDXdkMOff/+dCKA+99QjTzvojKMNM768IkgUILwY40MRoODDFXM4ssotwjRzjTjs2HMPPvnk66So5XhjzTPCtIIJIWHwcAGM5lV0QAQbtLCEGodcYkotxVBDzjz02IvPxvawM042zxSziyqP0IHFDyY40BFeAxQAQQtKnMHHI6Tgsow38MiT8T0803NONuDGMsojclBhgwkWHGAUWAuAUIMSZgTCCS3HbKMdmffYY0884DzjyyqXEPKGFT6UoMEDBXAVgAEUhODCE3JQ8gox2bDzDpn21EPPO9oU/xMLJn+U4cQOQjmAwLkfCXBASTmMwcgqwViTnTxa1zMPO9TsMkohZRjBwgcSwJQwVwSQlQISaBDCSS3MZGMOmfTMs04zr0RSBxY7lKDZACsbBQBSBmTQQhBa2BFJK8Nk806b86hjzCiBlMHEDB9YwMCVvoMkgE0bvOCEG42wwsw6zKcDzCVvVAGEChpIkMDo2Q8EfAY7dPGHKMaoU74vlKwhRQ8pyIAEFkAAK2EvfiEpwAV04IVAkAIZ68hW84bBiTpsoQgu6MAFIrCABBzAXPD7SEYsoIMvDMIUyWgHqOixjmOMAhDSo4EIOLBBBzDAgzCKHwAUQoESFuIUy3BHPv/uwUJlpOIQbJACDkwQAg5goAISeMACDsAg3/EwB144oTKESER1CGMTdNjCEWZggjKaoAQkGIEHKsAAK4UQLzPCQRcCUYotDpEe6dhFI8CgBB64wAQqeMEMamCDG8BABBQ4QA7VRgEccAEQpEiGO7BVj3PMAhBL0EEMUGCCFximCEhYwhBgwAEFKPKNEGmZBoKQhke8AhrwwJbPLsmEIAhhCEeIAhfMoAY2vEFwNwjBBRpQRYsMAAEMIEEU/ICKYGhjHrI8xy0OgYUrjMENeiCEIzChCU1swhF24IIQXrABpX2kAJ95QRgkEYxpkKNMREQHLyBRBqhkYhW2AMYylJH/DGTkghOAIIMRUsAAVDIkARPoAA/iUIp5qVAfvEJHMDZhh0KI4hfWAEc65hGPd7CDG74IBSG8kIMJFNCgBokACW6ABUPQ4hvogEe+6AGPczRjFpkoxS6i0aflxS4e5GiGLTxxhyecgAINSJtFMlCDKMxBE8JARzvmsa92mCMby/CFMagxjkwFClT1+OgycvGIMuxgBBZQwEVCYIQ1NOKV8CDTELtTDnKMQxzncEc9qsUkfdRjHeGgxin24IQZgMABFzEBFPLACV1gI2OCuse20rGOdmhnr2kK0D3gkY5vzOIQXPABCiJwERVcgRCnGIY37pUPQ2nrHTmbR6D0kat8/8yjHeXYRSTQkAQXUOAiLvjCI2axjHGciElg1ViScgVReaxjGGCcQg0ugJcYkMESOz3HkkyUr+MKyEQBWtI9bluMTtjBaNSFSERkUIZL8EIa6Niutbp7rTVtbGP/cS1nefEIMhBhBRN4SEZmYIZM9GIa8WWSvqpl38rVox72EBQ+bkuOWQwiCjQIAWJTSQAanEETvphGOpa0pgVba4XzkIeKRzTEeKgDHKqoww9IcAG1Yql0C+jBHE6hDG60g8T0nS+KMQZZiLZjHNYAxRpqsIEImFNGB2iABIzgh1pggxzxUJKavNsksNIDwqz1Kzq40QxLhAHADFAqQ7YHGigkgv8Y5WhHPbQ8KDWx6cER5is9yFGNYTDiCiGYYjENMgAGVMADVohEM9wxD3xoObxadlLWHrwpAcWDG8eABSCe0IEEGABxhG7ABUKghUpIA1D5KBSaPvVgvdWqtQJiBzRmYYk3HOGwxnLIAByQgRF0IRPWiDBtO2UoXs0DY6+WLzqG8Qk+eAEIIphAmh1CAAhs4ARh8MQ2AMSPfni7H4XCRz3iEQ95yJZEdCZHLhpRBijswAQYeMCTERK8FfhADqj4Rni/De5IE/HLEBYUkM0hDE/sQQxJeMEHKKCA3hUkASLYARUGMYtx4Krb3171EHm2MW6rCR/pWIYrIkEHK+TABBn/aABHULmAFSxhDZPohXaVxO80WctaFzcUPthRDV+cwhBkGEILPAABhCFkAac7AyR2YQ6bD5F5AFcutz91j3ZsIxm2kIQblCCDEETA6AeBOA+qQHGLI9dy8GAHOsyBjsrGtVb4MlS22vGNaEy0Dk+wAQkk4EaEHKB7Q7CDKr6x4HFzSxzbwAY3wlGOyQWcTfRwx8eYQQo+UOHkfD9gQQpAgRHIAA2gyIbeUuyOdPTrGs9YRjSw4Q1ypONukH1tOsKBjWaUog9V0IEJJKD5guw6AyTQQiSUQQ5wbOMa1qDGNJ6hjGIIQxnS0AY4zpEpZOOWG9IwRi5YkYgzEGHoKjOo/wASAIELMIEQu5jGMoaxi178QhjA6IUucjEMZlQjWuuAB7nlUY5rJEMXqrAJiuAGUDADIVABNiYjBHAACuADdoA8t7AKoFAKqwALsQALrxALu2AM0JAN4fB67xCC3YB1Q9MHaRAFOiACFdAABvAQwOMCWVAIlKAIgYAHf2AIjBAJlqAJoKAKtOALyVAN01dX4tAMteAJiGAHY/AEPaACWZFrwjEAHZADi4EFUcAEVMAFZOAGeUAIjWAJn4AKtlAxuWEN06ALn1AIbsAFSoADKtABLlEAoHYQCsEAGEACKFACIgACKSADO6AEW8AGezAIizAJo2ALyTANzpAMxHAKiZWwBlMgBDEAAvFGRXMoIwNAAJqoEYbWAS0ABFMwBmoQB3hwCJrQCrcQC6xgCovwBlGwAywAhwtgACi1ZhshEQjgABTgASuAA0OAGLqUBnXAB3hQB3HgBUpwAyjgAQx3SmrjFQuoAA9gAR0gAqSRAi9gAz0ABD7AAzoAAybAARTwAORhJTqEEQohERKRiWKxiW7kcAgREAAh+QQAZAAAACwMAAkAQQBLAIcAAAABAQECAgIDAwMEBAQFBQUGBgYHBwcICAgJCQkKCgoLCwsMDAwNDQ0ODg4PDw8QEBARERESEhITExMUFBQVFRUWFhYXFxcYGBgZGRkaGhobGxscHBwdHR0eHh4fHx8gICAhISEiIiIjIyMkJCQlJSUmJiYnJycoKCgpKSkqKiorKyssLCwtLS0uLi4vLy8wMDAxMTEyMjIzMzM0NDQ1NTU2NjY3Nzc4ODg5OTk6Ojo7Ozs8PDw9PT0+Pj4/Pz9AQEBBQUFCQkJDQ0NERERFRUVGRkZHR0dISEhJSUlKSkpLS0tMTExNTU1OTk5PT09QUFBRUVFSUlJTU1NUVFRVVVVWVlZXV1dYWFhZWVlaWlpbW1tcXFxdXV1eXl5fX19gYGBhYWFiYmJjY2NkZGRlZWVmZmZnZ2doaGhpaWlqampra2tsbGxtbW1ubm5vb29wcHBxcXFycnJzc3N0dHR1dXV2dnZ3d3d4eHh5eXl6enp7e3t8fHx9fX1+fn5/f3+AgICBgYGCgoKDg4OEhISFhYWGhoaHh4eIiIiJiYmKioqLi4uMjIyNjY2Ojo6Pj4+QkJCRkZGSkpKTk5OUlJSVlZWWlpaXl5eYmJiZmZmampqbm5ucnJydnZ2enp6fn5+goKChoaGioqKjo6OkpKSlpaWmpqanp6eoqKipqamqqqqrq6usrKytra2urq6vr6+wsLCxsbGysrKzs7O0tLS1tbW2tra3t7e4uLi5ubm6urq7u7u8vLy9vb2+vr6/v7/AwMDBwcHCwsLDw8PExMTFxcXGxsbHx8fIyMjJycnKysrLy8vMzMzNzc3Ozs7Pz8/Q0NDR0dHS0tLT09PU1NTV1dXW1tbX19fY2NjZ2dna2trb29vc3Nzd3d3e3t7f39/g4ODh4eHi4uLj4+Pk5OTl5eXm5ubn5+fo6Ojp6enq6urr6+vs7Ozt7e3u7u7v7+/w8PDx8fHy8vLz8/P09PT19fX29vb39/f4+Pj5+fn6+vr7+/v8/Pz9/f3+/v4AAAAI/wABCBxIsKDBgwgTKlzIsKHDhxAjSpxIsSLDAAIIFCAwQICAABZDLhxwgAGEBgsQGCAAUqRLggEKMKCw4YIEBwoMDHgZEqOAAQUMLLAAQgUJDxgmNEDQsSVPhRkNHECQoMEFEClk7BjSxAoYNGTAbKnSpMiOFBcMBHD6tCABBA0eRJCggYUPKGHiBHJk6dMpUqE+XWIkSA4UFww6tjUY4ECDCRYybDghhMudR6d+PbvmrRy5ceKyMful6g4RCQUGsH2KsYCEDytm5PjB5EygTbGOcVMHz94+ffnwvQs3jRehJhYOsFwMIABQBiJwONliJg4gSqh6Lct2Dh69e8Dx2f9j5w1aLkJOMuQUwFxAgQQTZESBQ4gSqVnCoHVmN88evnz6iEePOtw0c8sgT3Cw1E6LEbBABB8UsYYkqfDyjDfdAahPePfYE0876HTzTDCr8MFEBw4kwCBPazHAwQpAlLEILPl1c0478wRnDz0fpkOONcfksoonkfzBBQ4VrMfaTxOwUEQYhajiDDfjrBPPd/rYM8876YizjTS6iKJIHm2IMQUPJDSw0mo9AbVBD2AIUkoy7NRzj4Yb1sNlONc0E8wnfWxxRA8ynMABBAV81NZbDaQgBR+i+JJNPRvmc4887qTTTTTF5PIKKZjo0cUPL5zwwQUQqMimSAg8cAEOaGT/Igw15tx5Tz3wmNPNNMCwgkkif9CRxhQ+pOBBBhTgpBZzQ4WAxB+1iKOOPAHSA0862jDTyyiHsJHFE0Tk0AIISqW0EntPRXUBCjmIcUkz9+DzHz3uhKiMLaIUYgYSNKwgAgYSLJCooosNkICrM0DBxiO1bIPPrfOwM442y8SCyR9mNDHDURdEsBQBHq0qUgEPaHDCE3d8wos06WT54TjXLINLJnoIOsMHE5ykgEodEdzWARaUcMManrAMzz370HsOxbyYEsiRJ6g3wNRU9ywyRQRUtUEMR3yRCC3ewCNPPfSUY40xtITiCB5dCHHCBhGo5dEABlTFQAKp+WxRAhJw/wBDE2ogggoy6MxjeDvU8DLKInR0kQQOKKCqAEseEaBABBhYAIECebvUwItFpOFILMZwE8898+RKjCh/jIFECxkodUBqIGFEgAMZjBACBg8gkJpFtmPgwhBhGKJKM9ugs2M63jiTCiFfDKFCBGsVZPADFXygQg0wjJCkchXFhAADKBxRxiCiANONOe7Ys041vphCiBhAsMDBAmxhpAAGJ9TQQxFNGAIMOtCABBCgIgJAgAMokIMvGMIUvrBGO+Axj3uIAxie6MMXgCCCCjSgADD5CQRQAAQrbCEMZriCD0jwAM5dbSQMsMAHlJCHUxwjG+vYh6XocY1V+MEKOwABCP8NkpEDYCAHWbiDHvjwhzUsgQUSYMCyIrKWBHhgBkhwQyV6YQ1xuAMf6thGM15xiC/wIAUWOKBBGJCBE/SgC33QxCbmOAgu2MACvXshQn7yABhEIQ6SkIU0zMGOsW0DGKYwxBiCMAIMNABdBbGAC4wghkCEohe80AUuNNEGIGxAAgrQI2OAYgEhuEETtXgGOurBSnkwoxR/8AIHc7KigoRgCGY4RCmIEQ5uYIMasfCDEkBgAQZA0iGWe0AJrHAIXDDjG/GoB/OioQpCdCEIKZDAKBOwQBw40BS/wAY81FGOb+QiEEngwARCGREEUOADOVhDJ57RjcKtQxq5+MQfvlD/rA0woCAYQYAFSCADLPzBFMKYBjnm0Y5zgAMXfRCC7ERZkAZ44AVMAMQszNGOK4ljF5NowxRyAIIoDlEgtmvACHQQhTt4whgYesc9rhWOW+RhBwoAH0QCQIEVDIEMljhGcObhjmmIAg5BcMEGDnCQAcBFAzegQhwssQtw0MM/+oDHObwhiznUoCkPcc8BREAEMyTiFdTQRzu4wQxXDAILLwjBBE5KEIu+gAhg4AMmZNEM5cULH+jABjI8YQYX6I0h72kADLiACFccIxz7EIcxUoEIMvygAxRYQC0HcgEaRKENiBiFL57xDZnGyx7gSIYsFJEFFFB0II15zA/qsIpp/4DDHfy4xisUWYQT5HQ5BgmBEuZwiVgsoxzsiIc9gnOra+iCE3ZgQggeghEDvGYFVlCEL8KRjnjgoxmeeMMSZrCBzsE2ARDAgA7K8AjSmY4edtrRh44hij5oQQcZcIhzDMAAENAACXHwBDPU4Y4rDcMRWrBBCSYA1oEMgAIkqAEWADG4a5ijHvawRz22pI5x1AIRW+CBCSLgEPfwzQVKMMMiZqGNecgjHvDIxR+KIIIL4M8gBYDnFOzACWFgCB46slY6xpGNUbgBBx6YAAIcQjIMmMAIZihEKYxhjnzQ4x3qiMUcbKAUA8CEbn2UAh0qYYtruEMeSMtHPeQhMW00g/8SWgiBkhrigBLwoApxaEQqgHGNduBDHusgxyraAIMWqlEgBXDABUzQBDpwwhbMGEd/8sEPfFx5ac7wBSKkwAHlHFMhF+BBGAbBCVog7xzzEI6uSIGGFSzgALVMQAZQ0AM1XGIY1hiHTPOxD35cqh3ikIYwYNEHJVwAZK8FwRQKIQvdKBcfAQL2NTwhBhN4uq4jyEEVDpGLddyp1/wI90zV0Y17hUIOEq3eQmx3Ai5IAhjRGEc9gmPpc2zDGZxAwww04IEQlEAFL6CBEKywBkGcohnUAne4+VGPdpRDGrW4RB6qEAMHUJRuCWhBGTixDG2kw1Z6Io4xNuGGIMAgB1z/uUIY2LCHRnxCFsj4hm8UHu55qAMcx/CEHZ5wAxAkAJlVoYEbTnENL+YjH1paB1t7oYk6QKEIVShDHepDCloMYxrcpdbCty4Pe/MCEVUgQQYecOiFJCACGgiCHmaRdX2MGxzREMYsNiEINJTBDojQhCpykYxr6BpAvf7N0QGfDmsIAxRu4EEBC0DRAEDAAy2ogiJ+oY53UEoe5tCGM4RRi1R8ohKLcEQnWsELY0RjG+RgBz32wXod7kge8L0HNm5hiTk8QQU8068FWBCENGhiGWMDz1qX0YtZoMITo0CFK25BjGh4Yxzo6Oi8+cF6txOVHQWexzEyoYYjwCADyC6x/wd0YIU+qOIaGwIOOqKxC1WIQhOSIEUunFHPefTj/v3wh/77QX190Asd5HAO6bAOtaAHlzUBPwcRApACT2AHmtAL4KBDt8INveAJhwAIeTAHi4BQuRYP+Jd/+yeB5BENxgAMu1ALisAFLkABahIRA0ADZWAJtwAN6bAPlgYP0IAKgEAGX5AFUyAHlGALzuAN7vCB+ucP/eB29DAOz6ALqKAJjOAH/OQBr7ZZDEEAP6AHtkAN4yAP/OAh60AMkyAGQdADOEADWRAIqnBD7GCE+6dm8bANvzAKiUAHXsAvINAAihERBCAEgOAL3IAOqycP6QAOusAIXQAEQSAERHAGjf/AV93QDluHD1uSDt9QDctQC5sQCGlABT+gAh4gAV42EQQABH2QC9dADvKQD+zwDdEQC4tQBlCgBWTQBoqACsewDeYQD9WXD/BADtdwL6MACX6wBlmABDmgAhswAZqFNT+QB7QgDeFwNOZADcOACovgBmEwB4VwCa1ADNuQDpYngfaADtQADKrgCHJgBUfAAzFgAh1QAQzgO6+FYz+AB7IADd/QPuCQiaCwCHYQB4ogCrvADNygevYQLxyGDb9QCoqABkdgAhtQARDwamU3EQFgAELwB7yQDeTgHd1QDK7QCY7wB36ACbGgDH7nDvMQD+/ADiLSK5TAB2MwXsv4ACn/kSjAIwAHYASGUAzcFQ/ykA2/YAqaEAmIwAimAAzaMA7qIJQNBQ7I8AqX8AdpIAU94AIgEAF3cy71eBABpQSM8AzpwJLxUA264AmWQAmRkAm0QJYFxiPo8A3UIAuPIFLF0jFSxBFfiRAB8BYQEAWTQA1xGQ/SQAuWsAiF8AeCcAmr8AvBEAzAwAu0wAqiYAhp0ASQgwELkAA6cVgUcT0ZkAWZgA3x0B/xAA2v4Ah+EAdlEAZtoAeGQAiB4Ad6MAdtYAZUIATdM3az02AiQQANUAEiAAaf0A2xFw/NoAqHAAdesARAQARI0ARHIAQ8cAMy4AIpIAIaoBSeGTJPQTJttjQGoMAN8NAON8cLmFAHXcAvJ1ACJDACHYBHOfkRfSkRt3MBJOAFmCAN3hANxCALkhAHUfADMTACGpABGJAULfSb6sYcAyGcFiACWRAJyNAMtdAJhHAGStACIZABEtAADcAADJBTeXOfWDMTH0AFieALu5AJd2AFPCACCQCeEBpWCnQBPUAGioAIblAFPKACx3ajEiE+DWBnUyAFQrAxGeAAn0akDWE7M8EBG2ABXIkAFwmlDBEQACH5BABkAAAALAkACQBSAEsAhwAAAAEBAQICAgMDAwQEBAUFBQYGBgcHBwgICAkJCQoKCgsLCwwMDA0NDQ4ODg8PDxAQEBERERISEhMTExQUFBUVFRYWFhcXFxgYGBkZGRoaGhsbGxwcHB0dHR4eHh8fHyAgICEhISIiIiMjIyQkJCUlJSYmJicnJygoKCkpKSoqKisrKywsLC0tLS4uLi8vLzAwMDExMTIyMjMzMzQ0NDU1NTY2Njc3Nzg4ODk5OTo6Ojs7Ozw8PD09PT4+Pj8/P0BAQEFBQUJCQkNDQ0REREVFRUZGRkdHR0hISElJSUpKSktLS0xMTE1NTU5OTk9PT1BQUFFRUVJSUlNTU1RUVFVVVVZWVldXV1hYWFlZWVpaWltbW1xcXF1dXV5eXl9fX2BgYGFhYWJiYmNjY2RkZGVlZWZmZmdnZ2hoaGlpaWpqamtra2xsbG1tbW5ubm9vb3BwcHFxcXJycnNzc3R0dHV1dXZ2dnd3d3h4eHl5eXp6ent7e3x8fH19fX5+fn9/f4CAgIGBgYKCgoODg4SEhIWFhYaGhoeHh4iIiImJiYqKiouLi4yMjI2NjY6Ojo+Pj5CQkJGRkZKSkpOTk5SUlJWVlZaWlpeXl5iYmJmZmZqampubm5ycnJ2dnZ6enp+fn6CgoKGhoaKioqOjo6SkpKWlpaampqenp6ioqKmpqaqqqqurq6ysrK2tra6urq+vr7CwsLGxsbKysrOzs7S0tLW1tba2tre3t7i4uLm5ubq6uru7u7y8vL29vb6+vr+/v8DAwMHBwcLCwsPDw8TExMXFxcbGxsfHx8jIyMnJycrKysvLy8zMzM3Nzc7Ozs/Pz9DQ0NHR0dLS0tPT09TU1NXV1dbW1tfX19jY2NnZ2dra2tvb29zc3N3d3d7e3t/f3+Dg4OHh4eLi4uPj4+Tk5OXl5ebm5ufn5+jo6Onp6erq6uvr6+zs7O3t7e7u7u/v7/Dw8PHx8fLy8vPz8/T09PX19fb29vf39/j4+Pn5+fr6+vv7+/z8/P39/f7+/gAAAAj/AAEIHEiwoMGDCBMaDFAAwQIFCA4smKAhhIgRJDhEMKCwo8ePIAsOUBDhAoUIDi6cuFHkiBImO0g4CEmzZs0CEDSQ+KDBAgkeVdKwgUNHC44KNpMqRYhAA4seO3LcOAImTyNIkyzRQcJhqVevDlQcEYOmzRxAk0rNmiVLViIrIgIE+EqX5oQdYg5RCuVKV7Fo2apJc/aJDAoBc+sq7ohByZ5VvqSVg0cvH7967tDNivOCAOLFoAvK5RClEK5j1869k2cPnzx25mjVscEAAYHQuAEIGPDhCqRi0LihcwdvHr136srVyuOjgoMDuUEHGEBghBdNgMOla0ccHjt05Gz5/ynigYKC6IsbaDhxRE+rbuLOsXNHv506c+NuBVpCAgMD9HQFIIAHPXDRRyjGnLOOO/HE844766BTTji3DPIEChvMBOBSAg4AAxiR0JKMN/LMU4899MDTTjr4gYNLIVO04MEDiW1I03QHMBABEoP4Ak468uijTz74HHcfOeKAswsjXNxQwgQ12phQhwQUYIADI+DAxB2jPDMcPfrYM49354jjTTjjmINMKHlYoYMHAkjZkQAEGICAAgxo8MMYiJQizDfvzGNPPvO4k8443mRjTTflsFONLZfU8UQKccqZ0AB2LuBABCZo8cgw1IgDzz334IMPPOmEs0010DijjTnyiP+zDC2UkDHDAJYiNEAEILhwgw9FYEHIK9qQww4999AjjzvcJGOLLLXg4ssz4chDzjO5ZHJGDbflatABJAThRRt5EGJJLMyc00489RgpjjCiCAKII5ywMow28ZDTzC2XmMGtt6IpYIMYjpBCCzHTeJPOifbYI4864ViTyh5ONHEGIJrYMs074zBTSyVl1FBAlDbuVkACEFhQQhWF1MKMN+/osw8//dTDzjjaQGNMLolggcIKS6yxCCvNuNPxx2jowMABBJCcm4AHNDABCTpAkYYiqzAjHD37zMzPOMi0skkkiPTRRQ8elDBEGICQcsw64jBzyyZyKBGCBQ10i16HDVz/IIIPZSzSyi/SGCsPPvzw03U0o9ixhRRKABGDCBN8sMMVdnAiDDrhNJMLKH1oYYMJFyCwoYAETCACDFg8Ygy7+Mzcz+yK6/PLH0BwQAEDVR7QFA1OrEFJL+aA4wwvpRySxhI1gLBAaAJCAAIMOfQQhBFRdIHGIa9gg08+MtOjzjfVLBPMLYpswUIECxQAAAESgeBDF3yAQgx41iBzyyiO5MGFDyBwQAL05pXdVGcIY7iDHwrBCEpwghS2cMY5hCSkdFDjF6qwhCDcMAUbaEABBsBVARTwgBMooQ2PcAUz1JEO8injF7RIBSG6YAMPTCABdRlAAQ5QAzNcgha+QIY0/6yhjW+Ywx32UByRvPGLTwBCDEQwwQYkgIABfMYAUnuBFfwwil1Qox3tKFM3uLGNbLSiD01oQQc0lJQRQqACGvgACahwiF1kYxztaI0++sGPfDgMHeTThSb0sIUefKBSBWEABkYgBDVIwhbJ4MY74uEOFrKjOMfIhBqOAIMMEGAATlOIXB4Aghj8AAlS2AIgTuGMRskDfPvox6nQwY1hqCISezADFHaAAgo4LQAWWAEQwFAIVSjjGuUwjjzeMUkTVSMWjmADE1bAgASMDCRyEUAGahAFNNyhEJNgxTHEEQ96xI5mNTtHNozhCTs4oQcyOAEHJnAegwggBD/ogh8+Ef8MbxzRHvdoFz3qUSpxKEMWkThDDyrwAAQgck51KgET5BCJUdTiGNYgR5C8to8hsUMbx2gFHojwACtOSQAGaMEU9LCJXGCjOIPKh0xlKiR3hIMatAiEEzxQAQbg6qQESMAFSkADKcyhEq/wyzOqkY0zkeMc6cDGMWxxCk00Qg9ViAHvPkmAKhngqxKpgAdU8AQ6ZEJE34CHPOphqraCTx/t+AY0ZEGIKqSgAxEgoD0bEgEWHCEMftgELpgBDWpggxrOSEYynDGNatSCEnMQgxakcAQZeMCaYE0AnhrAWQugQAdNiEMlctGMbJRjNcgiFam+pw92cKMZsDhEF2pQAgv/cORSCGhABoCAhkesYhjbUMc5yBGOaQyDFrC4hS+EEYkwxKADGjDJAxbwVQQkYAENgIAEJjABCoSAB1aIwyRsgQ1zpIMdzWRYw1i7Dm0kwxWKKEMQXKAB6BxEQAqwwAhq4IVDwMIY10gHx6zhLFE8AhGOqAQm2AAECnxSAAJIwAMq0IESuIAGN9iBD34AhCA4wQx9OFcz0BEPeDBTrQM90WrzcbRM4AELOUDBBW4rGh1SgAVD6EIgSqEMbIRDHeh4xi08kQg6fOEKW/hCGIqQggZAOAAEkAAIXgAEKZRBDnbQgx8AIQhCMIITq9jFy+BRjzG9Ax4NMhFAVYuNW1gC/w9cCIIUH6BXgQigIR8IwhgKUQpijGMd62AHOHiBCTp0wQgwMAEKVKACD0jgmgAowAZigIQx/IETsKiFLn4RDGIYYxnSyAY4VFOqejiImZNEFj7ugaJmiAIPVwBCCibQAIceZIQPYMEVBKEKYnjDVPFYh8QoZgMQ1LMgApAIBV6ghDIYAhXKiE871nrOxFm7j6bmDn0ocw8/KksYkOiCDlBQgU+GEgCKHAER4NCJYVTjHPmgRzq8YQxLnOEGJrCAfQsCARQAAQtsAAQlfquNBa21210bEqvLTMl1pCPQDGJrPuqxLF4Y4gkt+AAErBjKAEzABDjgQiJu0Q1zwEMf8P8IhzRsQQgphKCndQ5AB5Rgh02kwhbDgEY31GGibsusa/gQk4oidI5ykAMd61hNPWRqpFrwQQggsMAC5IKQAGhgBk2ggyiiocd9sCMbxTCFHHyggAKAUjQBWMEaVmFEdrAGH/qwNh8TF6Z5vGMd5kgSOL4BjnKoI1AxnUcYXSGHGtCaxgcRwAd4kAVAtCIb+6jHOyyYCydaoQVNEw0EPuCCKziiGOzKx+xGP/qO4mMe6xjHNrTB+m104xvSXmtr6oGObThDE2JYQa3rTBABiGAIZFDELLgBV3BM47FymAIOOvBQO4+gCGZYRCyu0ZpYkn7u3l5HN54hjGMwQxrX4Eb/fC6p4nvAQxvFgIUgohACBBSg+cg2gRLcMIlcfCMf5IAGLy7hhiKYQAMN4DQDcANr8Am/QA3q0DXoNHqJsw/3YHfj8Ay7kAq08AvKMA3bQA7rQBlEwmrq0AyvEAlpEAQYwHEIQScIwDqBQArEQA73oA2+wCZPUAIIEELIRgAKkASIcAx4ZA8M2FEdaA/1AA/rUA7rNAui8Aq8kAzU4A2qISjxlnrSgFBxEAUwAAEKMQAokwFHQAehAAzWwA714Aym4AdcwAMbYACeQRAjEQEckAWVIA1uhzhASHEQAmTFkw3SwAzF4Au5MAzNcA3e4EpqJg7JAAuX4Adm4AQ3EALP/5MQBeCGKtAFkCAMWzMPweAIXCBrEmCCA2EAEuABLoAGo9ANyBJ3+RBQ8KAO5PAN3KAN2AANyAAMvkAMygAN19AN5LAd5UQP8zAP0XAKfLAFS7ADjSYBiGcQCIABKLADc7AK4wB3/AAPtHAHOjACFpAATtMUK/ADeAAL6TB3+4APpgYxevgMzJAMwGALrRALw2ANCsIgapYsywQMhZAEIHABD8A08FcQCbABLjAEegAL56CA8sBcYUAELWBba3gAD2ABJwAEWUAHnWAM70A740hx7FAO3wCL11ANz4AMwVAM0xAOxGEiyrKK2GAMtNAIYDADFCBAnpgQCuABM6AEf/9AC+nQgPQgDbDgCGtwBCRQTSOzeS9wBFfzW9xQDwzoR3Z3H+MQleQADtyQDRl4LCdSJJnBDbqACXWABTvwAbUxMudGEAsQAjgQBYRwC+vQgPYgDs6gC5EgBjIgAbUmABgAA0eABpGwC+KgDvJgfbOjD/T4Du0wH2olDyVWTvdAQfpAD+sQDs/ACWlgAyJwAQvgGWVZEA2QAkUwBpHwC+3ARw7IDhGzCncwBCHwAR2wAcBzNbGQDf4wm/5AevpAjr4oD/KALDOViimyIuUgDtjADL+wCnhgBBOghpt5EBJgA1vwB6WwDPBAmvkQD+owDrqACFaAAz9QBEwwBnswCa3/kAzkQJu1OXq3WQ8CRQ8AlYpl1g7C6QzDgAuuAAqSMAhx8AQuwDufQRMWMARwwAm5YA3zMHd19w7JkAlp8ATjwgeSgAq+EA3g8A7myYC32TANUyrfY2rsIA7RAAyv0AmLoAdogAVHMDq21Z80oQFQYBrNIA4+2IBAWA2ooAdlwAeToAqRASv2oA8VSjsX2hoytWr2gCriQA29YAqQgAdggAQzEAIPQBcegAWSoAzZkA740A+mJ3jogAyisAdsgAij4AvOwA3rcIo/Sp2k0jDzkA7gkD+4kAqZcAh2UAZUMAQyQAIXcGxJIRchAAagsA2TEXf7MHHr8Ka7wAl8MAeT/zAL09AN5wAPg7IPaaqltxlQ9MAO25AMt/AJiAAHXiAFRqADidYBFNAAyVgTAkICadAK03YPdHcP8jAO1EAMrmAJfaAHnjAM6gI7cVepSpQs8WAOzBALl3AHV3ADI8ABDMUACnAAZrecorQbJYAGqyBglXGhs0oNLCkKkJBCzjBtvBl3Wkok2eam1+AMynAMxaALoqAIcVAFOMABC3AAP6UY01EAJSAGoKAN4yCGazoP6NAN0lAMucAKqxAM1rCB49o1D/gO5+AN1JAMtAAKi1AIgwAIeZAGWZAENkACEsCPoEEn4NIFlrAM10AOZHYi9NAO5dAN1hCSyJAw6uAO8//gc0IyD7ChDcswgYiABkfwAz7AAzgAAygAAhgAAfaqonVxZwggAlTQMsMgDaN2Xu2ADuLQDdrwkYa1Dd9QDuvQi+oZmddwDLPwCYcABjkQARFhANF6OjqEATwABoLwCJuACrGQCyP5C7gQC66wCqmACqzwCrPQCzOrDa8XDtEgDLLwCYtwB0uGAgxQAAXwYNIaIAPgAFQDBVpQBm+wB4hQCZ5gCY5gCILwB33AB4AwCIdgCaVQC79QDMrQDLgwCo2AB2TwBMaIAUuLGJerGATAABRmAr4SLGpgB20wBlpgBVMQBVBABUjmBoNwCaXQCpr2CYSABlHAAyfgHLYGMLqbYQAK4AAUUBEssANKQAVNUAQ+EBU3YAM5wAM/4ARf8LmBcAiMYAdgoAQ4IGMgdK8AMx12wgAQQAEbYGE0AAMrYAIkMAIicBElcAIwsAPe+bxXkAQ84AIjgAEOoIb9aCkdYiV34gASQAESEAEP4ACc1QAO4AAPMAEXsAEe8AEWwQEXYJfWxLTgu8M83MM+/MNAHMRCPMREXMS5ERAAOw=="/>



We can further improve the performance of this model with recipes like
[WGAN-GP](https://keras.io/examples/generative/wgan_gp).
Conditional generation is also widely used in many modern image generation architectures like
[VQ-GANs](https://arxiv.org/abs/2012.09841), [DALL-E](https://openai.com/blog/dall-e/),
etc.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/conditional-gan) and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/conditional-GAN).

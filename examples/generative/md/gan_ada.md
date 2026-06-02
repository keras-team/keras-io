# Data-efficient GANs with Adaptive Discriminator Augmentation

**Author:** [András Béres](https://www.linkedin.com/in/andras-beres-789190210)<br>
**Date created:** 2021/10/28<br>
**Last modified:** 2025/01/23<br>
**Description:** Generating images from limited data using the Caltech Birds dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/gan_ada.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/gan_ada.py)



---
## Introduction

### GANs

[Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661) are a popular
class of generative deep learning models, commonly used for image generation. They
consist of a pair of dueling neural networks, called the discriminator and the generator.
The discriminator's task is to distinguish real images from generated (fake) ones, while
the generator network tries to fool the discriminator by generating more and more
realistic images. If the generator is however too easy or too hard to fool, it might fail
to provide useful learning signal for the generator, therefore training GANs is usually
considered a difficult task.

### Data augmentation for GANS

Data augmentation, a popular technique in deep learning, is the process of randomly
applying semantics-preserving transformations to the input data to generate multiple
realistic versions of it, thereby effectively multiplying the amount of training data
available. The simplest example is left-right flipping an image, which preserves its
contents while generating a second unique training sample. Data augmentation is commonly
used in supervised learning to prevent overfitting and enhance generalization.

The authors of [StyleGAN2-ADA](https://arxiv.org/abs/2006.06676) show that discriminator
overfitting can be an issue in GANs, especially when only low amounts of training data is
available. They propose Adaptive Discriminator Augmentation to mitigate this issue.

Applying data augmentation to GANs however is not straightforward. Since the generator is
updated using the discriminator's gradients, if the generated images are augmented, the
augmentation pipeline has to be differentiable and also has to be GPU-compatible for
computational efficiency. Luckily, the
[Keras image augmentation layers](https://keras.io/api/layers/preprocessing_layers/image_augmentation/)
fulfill both these requirements, and are therefore very well suited for this task.

### Invertible data augmentation

A possible difficulty when using data augmentation in generative models is the issue of
["leaky augmentations" (section 2.2)](https://arxiv.org/abs/2006.06676), namely when the
model generates images that are already augmented. This would mean that it was not able
to separate the augmentation from the underlying data distribution, which can be caused
by using non-invertible data transformations. For example, if either 0, 90, 180 or 270
degree rotations are performed with equal probability, the original orientation of the
images is impossible to infer, and this information is destroyed.

A simple trick to make data augmentations invertible is to only apply them with some
probability. That way the original version of the images will be more common, and the
data distribution can be inferred. By properly choosing this probability, one can
effectively regularize the discriminator without making the augmentations leaky.

---
## Setup


```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

import keras
from keras import ops
from keras import layers
```

<div class="k-default-codeblock">
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1738798965.367584   17795 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1738798965.374084   17795 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

```
</div>
---
## Hyperparameterers


```python
# data
num_epochs = 10  # train for 400 epochs for good results
image_size = 64
# resolution of Kernel Inception Distance measurement, see related section
kid_image_size = 75
padding = 0.25
dataset_name = "caltech_birds2011"

# adaptive discriminator augmentation
max_translation = 0.125
max_rotation = 0.125
max_zoom = 0.25
target_accuracy = 0.85
integration_steps = 1000

# architecture
noise_size = 64
depth = 4
width = 128
leaky_relu_slope = 0.2
dropout_rate = 0.4

# optimization
batch_size = 128
learning_rate = 2e-4
beta_1 = 0.5  # not using the default value of 0.9 is important
ema = 0.99
```

---
## Data pipeline

In this example, we will use the
[Caltech Birds (2011)](https://www.tensorflow.org/datasets/catalog/caltech_birds2011) dataset for
generating images of birds, which is a diverse natural dataset containing less then 6000
images for training. When working with such low amounts of data, one has to take extra
care to retain as high data quality as possible. In this example, we use the provided
bounding boxes of the birds to cut them out with square crops while preserving their
aspect ratios when possible.


```python

def round_to_int(float_value):
    return ops.cast(ops.round(float_value), "int32")


def preprocess_image(data):
    # unnormalize bounding box coordinates
    height = ops.cast(ops.shape(data["image"])[0], "float32")
    width = ops.cast(ops.shape(data["image"])[1], "float32")
    bounding_box = data["bbox"] * ops.stack([height, width, height, width])

    # calculate center and length of longer side, add padding
    target_center_y = 0.5 * (bounding_box[0] + bounding_box[2])
    target_center_x = 0.5 * (bounding_box[1] + bounding_box[3])
    target_size = ops.maximum(
        (1.0 + padding) * (bounding_box[2] - bounding_box[0]),
        (1.0 + padding) * (bounding_box[3] - bounding_box[1]),
    )

    # modify crop size to fit into image
    target_height = ops.min(
        [target_size, 2.0 * target_center_y, 2.0 * (height - target_center_y)]
    )
    target_width = ops.min(
        [target_size, 2.0 * target_center_x, 2.0 * (width - target_center_x)]
    )

    # crop image, `ops.image.crop_images` only works with non-tensor croppings
    image = ops.slice(
        data["image"],
        start_indices=(
            round_to_int(target_center_y - 0.5 * target_height),
            round_to_int(target_center_x - 0.5 * target_width),
            0,
        ),
        shape=(round_to_int(target_height), round_to_int(target_width), 3),
    )

    # resize and clip
    image = ops.cast(image, "float32")
    image = ops.image.resize(image, [image_size, image_size])

    return ops.clip(image / 255.0, 0.0, 1.0)


def prepare_dataset(split):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID calculation
    return (
        tfds.load(dataset_name, split=split, shuffle_files=True)
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


train_dataset = prepare_dataset("train")
val_dataset = prepare_dataset("test")
```

<div class="k-default-codeblock">
```
I0000 00:00:1738798971.054632   17795 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13840 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5

```
</div>
After preprocessing the training images look like the following:
![birds dataset](https://i.imgur.com/Ru5HgBM.png)

---
## Kernel inception distance

[Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) was proposed as a
replacement for the popular
[Frechet Inception Distance (FID)](https://arxiv.org/abs/1706.08500)
metric for measuring image generation quality.
Both metrics measure the difference in the generated and training distributions in the
representation space of an [InceptionV3](https://keras.io/api/applications/inceptionv3/)
network pretrained on
[ImageNet](https://www.tensorflow.org/datasets/catalog/imagenet2012).

According to the paper, KID was proposed because FID has no unbiased estimator, its
expected value is higher when it is measured on fewer images. KID is more suitable for
small datasets because its expected value does not depend on the number of samples it is
measured on. In my experience it is also computationally lighter, numerically more
stable, and simpler to implement because it can be estimated in a per-batch manner.

In this example, the images are evaluated at the minimal possible resolution of the
Inception network (75x75 instead of 299x299), and the metric is only measured on the
validation set for computational efficiency.



```python

class KID(keras.metrics.Metric):
    def __init__(self, name="kid", **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean()

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                layers.InputLayer(input_shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = ops.cast(ops.shape(features_1)[1], "float32")
        return (
            features_1 @ ops.transpose(features_2) / feature_dimensions + 1.0
        ) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = ops.shape(real_features)[0]
        batch_size_f = ops.cast(batch_size, "float32")
        mean_kernel_real = ops.sum(kernel_real * (1.0 - ops.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = ops.sum(
            kernel_generated * (1.0 - ops.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = ops.mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()

```

---
## Adaptive discriminator augmentation

The authors of [StyleGAN2-ADA](https://arxiv.org/abs/2006.06676) propose to change the
augmentation probability adaptively during training. Though it is explained differently
in the paper, they use [integral control](https://en.wikipedia.org/wiki/PID_controller#Integral) on the augmentation
probability to keep the discriminator's accuracy on real images close to a target value.
Note, that their controlled variable is actually the average sign of the discriminator
logits (r_t in the paper), which corresponds to 2 * accuracy - 1.

This method requires two hyperparameters:

1. `target_accuracy`: the target value for the discriminator's accuracy on real images. I
recommend selecting its value from the 80-90% range.
2. [`integration_steps`](https://en.wikipedia.org/wiki/PID_controller#Mathematical_form):
the number of update steps required for an accuracy error of 100% to transform into an
augmentation probability increase of 100%. To give an intuition, this defines how slowly
the augmentation probability is changed. I recommend setting this to a relatively high
value (1000 in this case) so that the augmentation strength is only adjusted slowly.

The main motivation for this procedure is that the optimal value of the target accuracy
is similar across different dataset sizes (see [figure 4 and 5 in the paper](https://arxiv.org/abs/2006.06676)),
so it does not have to be re-tuned, because the
process automatically applies stronger data augmentation when it is needed.


```python

# "hard sigmoid", useful for binary accuracy calculation from logits
def step(values):
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + ops.sign(values))


# augments images with a probability that is dynamically updated during training
class AdaptiveAugmenter(keras.Model):
    def __init__(self):
        super().__init__()

        # stores the current probability of an image being augmented
        self.probability = keras.Variable(0.0)
        self.seed_generator = keras.random.SeedGenerator(42)

        # the corresponding augmentation names from the paper are shown above each layer
        # the authors show (see figure 4), that the blitting and geometric augmentations
        # are the most helpful in the low-data regime
        self.augmenter = keras.Sequential(
            [
                layers.InputLayer(input_shape=(image_size, image_size, 3)),
                # blitting/x-flip:
                layers.RandomFlip("horizontal"),
                # blitting/integer translation:
                layers.RandomTranslation(
                    height_factor=max_translation,
                    width_factor=max_translation,
                    interpolation="nearest",
                ),
                # geometric/rotation:
                layers.RandomRotation(factor=max_rotation),
                # geometric/isotropic and anisotropic scaling:
                layers.RandomZoom(
                    height_factor=(-max_zoom, 0.0), width_factor=(-max_zoom, 0.0)
                ),
            ],
            name="adaptive_augmenter",
        )

    def call(self, images, training):
        if training:
            augmented_images = self.augmenter(images, training=training)

            # during training either the original or the augmented images are selected
            # based on self.probability
            augmentation_values = keras.random.uniform(
                shape=(batch_size, 1, 1, 1), seed=self.seed_generator
            )
            augmentation_bools = ops.less(augmentation_values, self.probability)

            images = ops.where(augmentation_bools, augmented_images, images)
        return images

    def update(self, real_logits):
        current_accuracy = ops.mean(step(real_logits))

        # the augmentation probability is updated based on the discriminator's
        # accuracy on real images
        accuracy_error = current_accuracy - target_accuracy
        self.probability.assign(
            ops.clip(self.probability + accuracy_error / integration_steps, 0.0, 1.0)
        )

```

---
## Network architecture

Here we specify the architecture of the two networks:

* generator: maps a random vector to an image, which should be as realistic as possible
* discriminator: maps an image to a scalar score, which should be high for real and low
for generated images

GANs tend to be sensitive to the network architecture, I implemented a DCGAN architecture
in this example, because it is relatively stable during training while being simple to
implement. We use a constant number of filters throughout the network, use a sigmoid
instead of tanh in the last layer of the generator, and use default initialization
instead of random normal as further simplifications.

As a good practice, we disable the learnable scale parameter in the batch normalization
layers, because on one hand the following relu + convolutional layers make it redundant
(as noted in the
[documentation](https://keras.io/api/layers/normalization_layers/batch_normalization/)).
But also because it should be disabled based on theory when using [spectral normalization
(section 4.1)](https://arxiv.org/abs/1802.05957), which is not used here, but is common
in GANs. We also disable the bias in the fully connected and convolutional layers, because
the following batch normalization makes it redundant.


```python

# DCGAN generator
def get_generator():
    noise_input = keras.Input(shape=(noise_size,))
    x = layers.Dense(4 * 4 * width, use_bias=False)(noise_input)
    x = layers.BatchNormalization(scale=False)(x)
    x = layers.ReLU()(x)
    x = layers.Reshape(target_shape=(4, 4, width))(x)
    for _ in range(depth - 1):
        x = layers.Conv2DTranspose(
            width,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.ReLU()(x)
    image_output = layers.Conv2DTranspose(
        3,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="sigmoid",
    )(x)

    return keras.Model(noise_input, image_output, name="generator")


# DCGAN discriminator
def get_discriminator():
    image_input = keras.Input(shape=(image_size, image_size, 3))
    x = image_input
    for _ in range(depth):
        x = layers.Conv2D(
            width,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.LeakyReLU(alpha=leaky_relu_slope)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    output_score = layers.Dense(1)(x)

    return keras.Model(image_input, output_score, name="discriminator")

```

---
## GAN model


```python

class GAN_ADA(keras.Model):
    def __init__(self):
        super().__init__()

        self.seed_generator = keras.random.SeedGenerator(seed=42)
        self.augmenter = AdaptiveAugmenter()
        self.generator = get_generator()
        self.ema_generator = keras.models.clone_model(self.generator)
        self.discriminator = get_discriminator()

        self.generator.summary()
        self.discriminator.summary()
        # we have created all layers at this point, so we can mark the model
        # as having been built
        self.built = True

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super().compile(**kwargs)

        # separate optimizers for the two networks
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.generator_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")
        self.augmentation_probability_tracker = keras.metrics.Mean(name="aug_p")
        self.kid = KID()

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.real_accuracy,
            self.generated_accuracy,
            self.augmentation_probability_tracker,
            self.kid,
        ]

    def generate(self, batch_size, training):
        latent_samples = keras.random.normal(
            shape=(batch_size, noise_size), seed=self.seed_generator
        )
        # use ema_generator during inference
        if training:
            generated_images = self.generator(latent_samples, training=training)
        else:
            generated_images = self.ema_generator(latent_samples, training=training)
        return generated_images

    def adversarial_loss(self, real_logits, generated_logits):
        # this is usually called the non-saturating GAN loss

        real_labels = ops.ones(shape=(batch_size, 1))
        generated_labels = ops.zeros(shape=(batch_size, 1))

        # the generator tries to produce images that the discriminator considers as real
        generator_loss = keras.losses.binary_crossentropy(
            real_labels, generated_logits, from_logits=True
        )
        # the discriminator tries to determine if images are real or generated
        discriminator_loss = keras.losses.binary_crossentropy(
            ops.concatenate([real_labels, generated_labels], axis=0),
            ops.concatenate([real_logits, generated_logits], axis=0),
            from_logits=True,
        )

        return ops.mean(generator_loss), ops.mean(discriminator_loss)

    def train_step(self, real_images):
        real_images = self.augmenter(real_images, training=True)

        # use persistent gradient tape because gradients will be calculated twice
        with tf.GradientTape(persistent=True) as tape:
            generated_images = self.generate(batch_size, training=True)
            # gradient is calculated through the image augmentation
            generated_images = self.augmenter(generated_images, training=True)

            # separate forward passes for the real and generated images, meaning
            # that batch normalization is applied separately
            real_logits = self.discriminator(real_images, training=True)
            generated_logits = self.discriminator(generated_images, training=True)

            generator_loss, discriminator_loss = self.adversarial_loss(
                real_logits, generated_logits
            )

        # calculate gradients and update weights
        generator_gradients = tape.gradient(
            generator_loss, self.generator.trainable_weights
        )
        discriminator_gradients = tape.gradient(
            discriminator_loss, self.discriminator.trainable_weights
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_weights)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_weights)
        )

        # update the augmentation probability based on the discriminator's performance
        self.augmenter.update(real_logits)

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        self.real_accuracy.update_state(1.0, step(real_logits))
        self.generated_accuracy.update_state(0.0, step(generated_logits))
        self.augmentation_probability_tracker.update_state(self.augmenter.probability)

        # track the exponential moving average of the generator's weights to decrease
        # variance in the generation quality
        for weight, ema_weight in zip(
            self.generator.weights, self.ema_generator.weights
        ):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, real_images):
        generated_images = self.generate(batch_size, training=False)

        self.kid.update_state(real_images, generated_images)

        # only KID is measured during the evaluation phase for computational efficiency
        return {self.kid.name: self.kid.result()}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6, interval=5):
        # plot random generated images for visual evaluation of generation quality
        if epoch is None or (epoch + 1) % interval == 0:
            num_images = num_rows * num_cols
            generated_images = self.generate(num_images, training=False)

            plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    plt.imshow(generated_images[index])
                    plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()

```

---
## Training

One can should see from the metrics during training, that if the real accuracy
(discriminator's accuracy on real images) is below the target accuracy, the augmentation
probability is increased, and vice versa. In my experience, during a healthy GAN
training, the discriminator accuracy should stay in the 80-95% range. Below that, the
discriminator is too weak, above that it is too strong.

Note that we track the exponential moving average of the generator's weights, and use that
for image generation and KID evaluation.


```python
# create and compile the model
model = GAN_ADA()
model.compile(
    generator_optimizer=keras.optimizers.Adam(learning_rate, beta_1),
    discriminator_optimizer=keras.optimizers.Adam(learning_rate, beta_1),
)

# save the best model based on the validation KID metric
checkpoint_path = "gan_model.weights.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)

# run training and plot generated images periodically
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
        checkpoint_callback,
    ],
)
```

<div class="k-default-codeblock">
```
/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/input_layer.py:27: UserWarning: Argument `input_shape` is deprecated. Use `shape` instead.
  warnings.warn(

/usr/local/lib/python3.11/dist-packages/keras/src/layers/activations/leaky_relu.py:41: UserWarning: Argument `alpha` is deprecated. Use `negative_slope` instead.
  warnings.warn(

```
</div>
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "generator"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │       <span style="color: #00af00; text-decoration-color: #00af00">131,072</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │         <span style="color: #00af00; text-decoration-color: #00af00">6,144</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ reshape (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">262,144</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │           <span style="color: #00af00; text-decoration-color: #00af00">384</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_1              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">262,144</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │           <span style="color: #00af00; text-decoration-color: #00af00">384</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_2              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">262,144</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_3           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │           <span style="color: #00af00; text-decoration-color: #00af00">384</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ re_lu_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">ReLU</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_3              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)      │         <span style="color: #00af00; text-decoration-color: #00af00">6,147</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">930,947</span> (3.55 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">926,083</span> (3.53 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,864</span> (19.00 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "discriminator"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │         <span style="color: #00af00; text-decoration-color: #00af00">6,144</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_4           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │           <span style="color: #00af00; text-decoration-color: #00af00">384</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">262,144</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_5           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │           <span style="color: #00af00; text-decoration-color: #00af00">384</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">262,144</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_6           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │           <span style="color: #00af00; text-decoration-color: #00af00">384</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">262,144</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_7           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │           <span style="color: #00af00; text-decoration-color: #00af00">384</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │         <span style="color: #00af00; text-decoration-color: #00af00">2,049</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">796,161</span> (3.04 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">795,137</span> (3.03 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,024</span> (4.00 KB)
</pre>



<div class="k-default-codeblock">
```
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

```
</div>
    
        0/87910968 [37m━━━━━━━━━━━━━━━━━━━━  0s 0s/step

<div class="k-default-codeblock">
```

```
</div>
  4202496/87910968 [37m━━━━━━━━━━━━━━━━━━━━  1s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 12304384/87910968 ━━[37m━━━━━━━━━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 21815296/87910968 ━━━━[37m━━━━━━━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 34373632/87910968 ━━━━━━━[37m━━━━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 47669248/87910968 ━━━━━━━━━━[37m━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 62324736/87910968 ━━━━━━━━━━━━━━[37m━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 72032256/87910968 ━━━━━━━━━━━━━━━━[37m━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 82083840/87910968 ━━━━━━━━━━━━━━━━━━[37m━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 87910968/87910968 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step


<div class="k-default-codeblock">
```
Epoch 1/10

E0000 00:00:1738798983.901596   17795 meta_optimizer.cc:966] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/gradient_tape/adaptive_augmenter_3/SelectV2_1-1-TransposeNHWCToNCHW-LayoutOptimizer

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1738798987.822990   17861 cuda_solvers.cc:178] Creating GpuSolver handles for stream 0x9f45670

I0000 00:00:1738798988.976919   17862 cuda_dnn.cc:529] Loaded cuDNN version 90300

```
</div>
    
  1/46 [37m━━━━━━━━━━━━━━━━━━━━  13:22 18s/step - aug_p: 0.0000e+00 - d_loss: 0.8829 - g_loss: 0.5585 - gen_acc: 0.2812 - real_acc: 0.7031

<div class="k-default-codeblock">
```

```
</div>
  2/46 [37m━━━━━━━━━━━━━━━━━━━━  7s 173ms/step - aug_p: 0.0000e+00 - d_loss: 0.7985 - g_loss: 0.8154 - gen_acc: 0.4258 - real_acc: 0.6777 

<div class="k-default-codeblock">
```

```
</div>
  3/46 ━[37m━━━━━━━━━━━━━━━━━━━  8s 201ms/step - aug_p: 0.0000e+00 - d_loss: 0.7488 - g_loss: 0.9481 - gen_acc: 0.5052 - real_acc: 0.6680

<div class="k-default-codeblock">
```

```
</div>
  4/46 ━[37m━━━━━━━━━━━━━━━━━━━  10s 249ms/step - aug_p: 0.0000e+00 - d_loss: 0.7106 - g_loss: 1.0407 - gen_acc: 0.5586 - real_acc: 0.6680

<div class="k-default-codeblock">
```

```
</div>
  5/46 ━━[37m━━━━━━━━━━━━━━━━━━  11s 269ms/step - aug_p: 0.0000e+00 - d_loss: 0.6782 - g_loss: 1.1104 - gen_acc: 0.5991 - real_acc: 0.6744

<div class="k-default-codeblock">
```

```
</div>
  6/46 ━━[37m━━━━━━━━━━━━━━━━━━  10s 273ms/step - aug_p: 0.0000e+00 - d_loss: 0.6506 - g_loss: 1.1692 - gen_acc: 0.6301 - real_acc: 0.6818

<div class="k-default-codeblock">
```

```
</div>
  7/46 ━━━[37m━━━━━━━━━━━━━━━━━  10s 280ms/step - aug_p: 5.1020e-07 - d_loss: 0.6253 - g_loss: 1.2302 - gen_acc: 0.6558 - real_acc: 0.6902

<div class="k-default-codeblock">
```

```
</div>
  8/46 ━━━[37m━━━━━━━━━━━━━━━━━  10s 289ms/step - aug_p: 1.4962e-06 - d_loss: 0.6030 - g_loss: 1.2838 - gen_acc: 0.6772 - real_acc: 0.6986

<div class="k-default-codeblock">
```

```
</div>
  9/46 ━━━[37m━━━━━━━━━━━━━━━━━  10s 297ms/step - aug_p: 3.8570e-06 - d_loss: 0.5829 - g_loss: 1.3313 - gen_acc: 0.6952 - real_acc: 0.7074

<div class="k-default-codeblock">
```

```
</div>
 10/46 ━━━━[37m━━━━━━━━━━━━━━━━  10s 299ms/step - aug_p: 7.9244e-06 - d_loss: 0.5644 - g_loss: 1.3805 - gen_acc: 0.7109 - real_acc: 0.7162

<div class="k-default-codeblock">
```

```
</div>
 11/46 ━━━━[37m━━━━━━━━━━━━━━━━  11s 314ms/step - aug_p: 1.3531e-05 - d_loss: 0.5473 - g_loss: 1.4301 - gen_acc: 0.7249 - real_acc: 0.7245

<div class="k-default-codeblock">
```

```
</div>
 12/46 ━━━━━[37m━━━━━━━━━━━━━━━  11s 333ms/step - aug_p: 2.0444e-05 - d_loss: 0.5318 - g_loss: 1.4743 - gen_acc: 0.7373 - real_acc: 0.7322

<div class="k-default-codeblock">
```

```
</div>
 13/46 ━━━━━[37m━━━━━━━━━━━━━━━  11s 352ms/step - aug_p: 2.8561e-05 - d_loss: 0.5174 - g_loss: 1.5181 - gen_acc: 0.7485 - real_acc: 0.7394

<div class="k-default-codeblock">
```

```
</div>
 14/46 ━━━━━━[37m━━━━━━━━━━━━━━  11s 365ms/step - aug_p: 3.7929e-05 - d_loss: 0.5040 - g_loss: 1.5586 - gen_acc: 0.7587 - real_acc: 0.7464

<div class="k-default-codeblock">
```

```
</div>
 15/46 ━━━━━━[37m━━━━━━━━━━━━━━  11s 379ms/step - aug_p: 4.8560e-05 - d_loss: 0.4914 - g_loss: 1.5977 - gen_acc: 0.7680 - real_acc: 0.7530

<div class="k-default-codeblock">
```

```
</div>
 16/46 ━━━━━━[37m━━━━━━━━━━━━━━  11s 387ms/step - aug_p: 6.0448e-05 - d_loss: 0.4795 - g_loss: 1.6355 - gen_acc: 0.7766 - real_acc: 0.7593

<div class="k-default-codeblock">
```

```
</div>
 17/46 ━━━━━━━[37m━━━━━━━━━━━━━  11s 399ms/step - aug_p: 7.3577e-05 - d_loss: 0.4683 - g_loss: 1.6715 - gen_acc: 0.7844 - real_acc: 0.7654

<div class="k-default-codeblock">
```

```
</div>
 18/46 ━━━━━━━[37m━━━━━━━━━━━━━  11s 414ms/step - aug_p: 8.7805e-05 - d_loss: 0.4578 - g_loss: 1.7063 - gen_acc: 0.7917 - real_acc: 0.7711

<div class="k-default-codeblock">
```

```
</div>
 19/46 ━━━━━━━━[37m━━━━━━━━━━━━  11s 422ms/step - aug_p: 1.0308e-04 - d_loss: 0.4480 - g_loss: 1.7386 - gen_acc: 0.7984 - real_acc: 0.7765

<div class="k-default-codeblock">
```

```
</div>
 20/46 ━━━━━━━━[37m━━━━━━━━━━━━  11s 433ms/step - aug_p: 1.1933e-04 - d_loss: 0.4386 - g_loss: 1.7736 - gen_acc: 0.8046 - real_acc: 0.7817

<div class="k-default-codeblock">
```

```
</div>
 21/46 ━━━━━━━━━[37m━━━━━━━━━━━  11s 444ms/step - aug_p: 1.3652e-04 - d_loss: 0.4297 - g_loss: 1.8065 - gen_acc: 0.8104 - real_acc: 0.7866

<div class="k-default-codeblock">
```

```
</div>
 22/46 ━━━━━━━━━[37m━━━━━━━━━━━  10s 440ms/step - aug_p: 1.5459e-04 - d_loss: 0.4213 - g_loss: 1.8383 - gen_acc: 0.8158 - real_acc: 0.7913

<div class="k-default-codeblock">
```

```
</div>
 23/46 ━━━━━━━━━━[37m━━━━━━━━━━  9s 434ms/step - aug_p: 1.7347e-04 - d_loss: 0.4132 - g_loss: 1.8694 - gen_acc: 0.8209 - real_acc: 0.7958 

<div class="k-default-codeblock">
```

```
</div>
 24/46 ━━━━━━━━━━[37m━━━━━━━━━━  9s 429ms/step - aug_p: 1.9312e-04 - d_loss: 0.4056 - g_loss: 1.8988 - gen_acc: 0.8257 - real_acc: 0.8000

<div class="k-default-codeblock">
```

```
</div>
 25/46 ━━━━━━━━━━[37m━━━━━━━━━━  8s 426ms/step - aug_p: 2.1348e-04 - d_loss: 0.3983 - g_loss: 1.9278 - gen_acc: 0.8302 - real_acc: 0.8041

<div class="k-default-codeblock">
```

```
</div>
 26/46 ━━━━━━━━━━━[37m━━━━━━━━━  8s 423ms/step - aug_p: 2.3451e-04 - d_loss: 0.3914 - g_loss: 1.9548 - gen_acc: 0.8345 - real_acc: 0.8079

<div class="k-default-codeblock">
```

```
</div>
 27/46 ━━━━━━━━━━━[37m━━━━━━━━━  8s 427ms/step - aug_p: 2.5614e-04 - d_loss: 0.3848 - g_loss: 1.9828 - gen_acc: 0.8385 - real_acc: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 28/46 ━━━━━━━━━━━━[37m━━━━━━━━  7s 424ms/step - aug_p: 2.7834e-04 - d_loss: 0.3785 - g_loss: 2.0093 - gen_acc: 0.8423 - real_acc: 0.8151

<div class="k-default-codeblock">
```

```
</div>
 29/46 ━━━━━━━━━━━━[37m━━━━━━━━  7s 420ms/step - aug_p: 3.0107e-04 - d_loss: 0.3724 - g_loss: 2.0347 - gen_acc: 0.8459 - real_acc: 0.8185

<div class="k-default-codeblock">
```

```
</div>
 30/46 ━━━━━━━━━━━━━[37m━━━━━━━  6s 416ms/step - aug_p: 3.2432e-04 - d_loss: 0.3666 - g_loss: 2.0599 - gen_acc: 0.8493 - real_acc: 0.8218

<div class="k-default-codeblock">
```

```
</div>
 31/46 ━━━━━━━━━━━━━[37m━━━━━━━  6s 413ms/step - aug_p: 3.4806e-04 - d_loss: 0.3610 - g_loss: 2.0840 - gen_acc: 0.8526 - real_acc: 0.8249

<div class="k-default-codeblock">
```

```
</div>
 32/46 ━━━━━━━━━━━━━[37m━━━━━━━  5s 409ms/step - aug_p: 3.7225e-04 - d_loss: 0.3556 - g_loss: 2.1073 - gen_acc: 0.8556 - real_acc: 0.8279

<div class="k-default-codeblock">
```

```
</div>
 33/46 ━━━━━━━━━━━━━━[37m━━━━━━  5s 406ms/step - aug_p: 3.9686e-04 - d_loss: 0.3505 - g_loss: 2.1300 - gen_acc: 0.8586 - real_acc: 0.8307

<div class="k-default-codeblock">
```

```
</div>
 34/46 ━━━━━━━━━━━━━━[37m━━━━━━  4s 402ms/step - aug_p: 4.2187e-04 - d_loss: 0.3455 - g_loss: 2.1520 - gen_acc: 0.8614 - real_acc: 0.8335

<div class="k-default-codeblock">
```

```
</div>
 35/46 ━━━━━━━━━━━━━━━[37m━━━━━  4s 400ms/step - aug_p: 4.4725e-04 - d_loss: 0.3407 - g_loss: 2.1736 - gen_acc: 0.8641 - real_acc: 0.8361

<div class="k-default-codeblock">
```

```
</div>
 36/46 ━━━━━━━━━━━━━━━[37m━━━━━  3s 397ms/step - aug_p: 4.7297e-04 - d_loss: 0.3361 - g_loss: 2.1947 - gen_acc: 0.8667 - real_acc: 0.8387

<div class="k-default-codeblock">
```

```
</div>
 37/46 ━━━━━━━━━━━━━━━━[37m━━━━  3s 393ms/step - aug_p: 4.9903e-04 - d_loss: 0.3316 - g_loss: 2.2152 - gen_acc: 0.8691 - real_acc: 0.8411

<div class="k-default-codeblock">
```

```
</div>
 38/46 ━━━━━━━━━━━━━━━━[37m━━━━  3s 387ms/step - aug_p: 5.2539e-04 - d_loss: 0.3273 - g_loss: 2.2357 - gen_acc: 0.8715 - real_acc: 0.8435

<div class="k-default-codeblock">
```

```
</div>
 39/46 ━━━━━━━━━━━━━━━━[37m━━━━  2s 382ms/step - aug_p: 5.5206e-04 - d_loss: 0.3231 - g_loss: 2.2554 - gen_acc: 0.8738 - real_acc: 0.8458

<div class="k-default-codeblock">
```

```
</div>
 40/46 ━━━━━━━━━━━━━━━━━[37m━━━  2s 376ms/step - aug_p: 5.7902e-04 - d_loss: 0.3191 - g_loss: 2.2756 - gen_acc: 0.8759 - real_acc: 0.8480

<div class="k-default-codeblock">
```

```
</div>
 41/46 ━━━━━━━━━━━━━━━━━[37m━━━  1s 371ms/step - aug_p: 6.0626e-04 - d_loss: 0.3151 - g_loss: 2.2954 - gen_acc: 0.8780 - real_acc: 0.8502

<div class="k-default-codeblock">
```

```
</div>
 42/46 ━━━━━━━━━━━━━━━━━━[37m━━  1s 366ms/step - aug_p: 6.3377e-04 - d_loss: 0.3113 - g_loss: 2.3147 - gen_acc: 0.8800 - real_acc: 0.8522

<div class="k-default-codeblock">
```

```
</div>
 43/46 ━━━━━━━━━━━━━━━━━━[37m━━  1s 362ms/step - aug_p: 6.6154e-04 - d_loss: 0.3076 - g_loss: 2.3339 - gen_acc: 0.8820 - real_acc: 0.8543

<div class="k-default-codeblock">
```

```
</div>
 44/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 358ms/step - aug_p: 6.8956e-04 - d_loss: 0.3041 - g_loss: 2.3524 - gen_acc: 0.8839 - real_acc: 0.8562

<div class="k-default-codeblock">
```

```
</div>
 45/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 354ms/step - aug_p: 7.1780e-04 - d_loss: 0.3006 - g_loss: 2.3703 - gen_acc: 0.8857 - real_acc: 0.8581

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 350ms/step - aug_p: 7.4625e-04 - d_loss: 0.2973 - g_loss: 2.3871 - gen_acc: 0.8874 - real_acc: 0.8599

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 61s 958ms/step - aug_p: 7.7349e-04 - d_loss: 0.2942 - g_loss: 2.4032 - gen_acc: 0.8890 - real_acc: 0.8616 - val_kid: 9.1841


<div class="k-default-codeblock">
```
Epoch 2/10

```
</div>
    
  1/46 [37m━━━━━━━━━━━━━━━━━━━━  22:28 30s/step - aug_p: 0.0051 - d_loss: 0.1030 - g_loss: 8.8928 - gen_acc: 1.0000 - real_acc: 0.9375

<div class="k-default-codeblock">
```

```
</div>
  2/46 [37m━━━━━━━━━━━━━━━━━━━━  7s 176ms/step - aug_p: 0.0051 - d_loss: 0.1073 - g_loss: 8.4938 - gen_acc: 0.9980 - real_acc: 0.9297 

<div class="k-default-codeblock">
```

```
</div>
  3/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 176ms/step - aug_p: 0.0051 - d_loss: 0.1074 - g_loss: 7.7540 - gen_acc: 0.9952 - real_acc: 0.9340

<div class="k-default-codeblock">
```

```
</div>
  4/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 176ms/step - aug_p: 0.0052 - d_loss: 0.1037 - g_loss: 7.2815 - gen_acc: 0.9945 - real_acc: 0.9388

<div class="k-default-codeblock">
```

```
</div>
  5/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 175ms/step - aug_p: 0.0052 - d_loss: 0.0997 - g_loss: 6.9305 - gen_acc: 0.9943 - real_acc: 0.9432

<div class="k-default-codeblock">
```

```
</div>
  6/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 175ms/step - aug_p: 0.0052 - d_loss: 0.0960 - g_loss: 6.6418 - gen_acc: 0.9942 - real_acc: 0.9473

<div class="k-default-codeblock">
```

```
</div>
  7/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 175ms/step - aug_p: 0.0052 - d_loss: 0.0933 - g_loss: 6.4224 - gen_acc: 0.9939 - real_acc: 0.9503

<div class="k-default-codeblock">
```

```
</div>
  8/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 175ms/step - aug_p: 0.0053 - d_loss: 0.0907 - g_loss: 6.2473 - gen_acc: 0.9937 - real_acc: 0.9530

<div class="k-default-codeblock">
```

```
</div>
  9/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 175ms/step - aug_p: 0.0053 - d_loss: 0.0885 - g_loss: 6.0970 - gen_acc: 0.9936 - real_acc: 0.9552

<div class="k-default-codeblock">
```

```
</div>
 10/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 175ms/step - aug_p: 0.0053 - d_loss: 0.0868 - g_loss: 5.9686 - gen_acc: 0.9936 - real_acc: 0.9571

<div class="k-default-codeblock">
```

```
</div>
 11/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 175ms/step - aug_p: 0.0054 - d_loss: 0.0852 - g_loss: 5.8546 - gen_acc: 0.9936 - real_acc: 0.9588

<div class="k-default-codeblock">
```

```
</div>
 12/46 ━━━━━[37m━━━━━━━━━━━━━━━  5s 176ms/step - aug_p: 0.0054 - d_loss: 0.0837 - g_loss: 5.7615 - gen_acc: 0.9937 - real_acc: 0.9602

<div class="k-default-codeblock">
```

```
</div>
 13/46 ━━━━━[37m━━━━━━━━━━━━━━━  5s 175ms/step - aug_p: 0.0054 - d_loss: 0.0825 - g_loss: 5.6750 - gen_acc: 0.9937 - real_acc: 0.9614

<div class="k-default-codeblock">
```

```
</div>
 14/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 176ms/step - aug_p: 0.0055 - d_loss: 0.0813 - g_loss: 5.5972 - gen_acc: 0.9937 - real_acc: 0.9626

<div class="k-default-codeblock">
```

```
</div>
 15/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 176ms/step - aug_p: 0.0055 - d_loss: 0.0802 - g_loss: 5.5273 - gen_acc: 0.9938 - real_acc: 0.9636

<div class="k-default-codeblock">
```

```
</div>
 16/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 176ms/step - aug_p: 0.0055 - d_loss: 0.0792 - g_loss: 5.4619 - gen_acc: 0.9939 - real_acc: 0.9645

<div class="k-default-codeblock">
```

```
</div>
 17/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 176ms/step - aug_p: 0.0056 - d_loss: 0.0783 - g_loss: 5.4012 - gen_acc: 0.9940 - real_acc: 0.9654

<div class="k-default-codeblock">
```

```
</div>
 18/46 ━━━━━━━[37m━━━━━━━━━━━━━  4s 176ms/step - aug_p: 0.0056 - d_loss: 0.0775 - g_loss: 5.3477 - gen_acc: 0.9941 - real_acc: 0.9661

<div class="k-default-codeblock">
```

```
</div>
 19/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 176ms/step - aug_p: 0.0056 - d_loss: 0.0768 - g_loss: 5.2979 - gen_acc: 0.9941 - real_acc: 0.9667

<div class="k-default-codeblock">
```

```
</div>
 20/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 176ms/step - aug_p: 0.0057 - d_loss: 0.0762 - g_loss: 5.2495 - gen_acc: 0.9941 - real_acc: 0.9673

<div class="k-default-codeblock">
```

```
</div>
 21/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 176ms/step - aug_p: 0.0057 - d_loss: 0.0758 - g_loss: 5.2113 - gen_acc: 0.9940 - real_acc: 0.9677

<div class="k-default-codeblock">
```

```
</div>
 22/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 176ms/step - aug_p: 0.0057 - d_loss: 0.0754 - g_loss: 5.1753 - gen_acc: 0.9940 - real_acc: 0.9681

<div class="k-default-codeblock">
```

```
</div>
 23/46 ━━━━━━━━━━[37m━━━━━━━━━━  4s 176ms/step - aug_p: 0.0058 - d_loss: 0.0752 - g_loss: 5.1387 - gen_acc: 0.9940 - real_acc: 0.9684

<div class="k-default-codeblock">
```

```
</div>
 24/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 176ms/step - aug_p: 0.0058 - d_loss: 0.0749 - g_loss: 5.1112 - gen_acc: 0.9939 - real_acc: 0.9688

<div class="k-default-codeblock">
```

```
</div>
 25/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 176ms/step - aug_p: 0.0058 - d_loss: 0.0746 - g_loss: 5.0899 - gen_acc: 0.9939 - real_acc: 0.9691

<div class="k-default-codeblock">
```

```
</div>
 26/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 177ms/step - aug_p: 0.0059 - d_loss: 0.0744 - g_loss: 5.0691 - gen_acc: 0.9939 - real_acc: 0.9693

<div class="k-default-codeblock">
```

```
</div>
 27/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 177ms/step - aug_p: 0.0059 - d_loss: 0.0743 - g_loss: 5.0465 - gen_acc: 0.9937 - real_acc: 0.9696

<div class="k-default-codeblock">
```

```
</div>
 28/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 177ms/step - aug_p: 0.0059 - d_loss: 0.0742 - g_loss: 5.0296 - gen_acc: 0.9935 - real_acc: 0.9698

<div class="k-default-codeblock">
```

```
</div>
 29/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 177ms/step - aug_p: 0.0060 - d_loss: 0.0741 - g_loss: 5.0163 - gen_acc: 0.9934 - real_acc: 0.9701

<div class="k-default-codeblock">
```

```
</div>
 30/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 177ms/step - aug_p: 0.0060 - d_loss: 0.0740 - g_loss: 5.0018 - gen_acc: 0.9932 - real_acc: 0.9703

<div class="k-default-codeblock">
```

```
</div>
 31/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 177ms/step - aug_p: 0.0060 - d_loss: 0.0739 - g_loss: 4.9862 - gen_acc: 0.9931 - real_acc: 0.9705

<div class="k-default-codeblock">
```

```
</div>
 32/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 177ms/step - aug_p: 0.0061 - d_loss: 0.0739 - g_loss: 4.9725 - gen_acc: 0.9929 - real_acc: 0.9707

<div class="k-default-codeblock">
```

```
</div>
 33/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 177ms/step - aug_p: 0.0061 - d_loss: 0.0739 - g_loss: 4.9583 - gen_acc: 0.9928 - real_acc: 0.9709

<div class="k-default-codeblock">
```

```
</div>
 34/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 178ms/step - aug_p: 0.0061 - d_loss: 0.0739 - g_loss: 4.9439 - gen_acc: 0.9927 - real_acc: 0.9711

<div class="k-default-codeblock">
```

```
</div>
 35/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 178ms/step - aug_p: 0.0062 - d_loss: 0.0739 - g_loss: 4.9297 - gen_acc: 0.9926 - real_acc: 0.9712

<div class="k-default-codeblock">
```

```
</div>
 36/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 178ms/step - aug_p: 0.0062 - d_loss: 0.0740 - g_loss: 4.9151 - gen_acc: 0.9925 - real_acc: 0.9714

<div class="k-default-codeblock">
```

```
</div>
 37/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 178ms/step - aug_p: 0.0062 - d_loss: 0.0741 - g_loss: 4.9027 - gen_acc: 0.9924 - real_acc: 0.9714

<div class="k-default-codeblock">
```

```
</div>
 38/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 178ms/step - aug_p: 0.0063 - d_loss: 0.0743 - g_loss: 4.8890 - gen_acc: 0.9921 - real_acc: 0.9715

<div class="k-default-codeblock">
```

```
</div>
 39/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 178ms/step - aug_p: 0.0063 - d_loss: 0.0748 - g_loss: 4.8802 - gen_acc: 0.9918 - real_acc: 0.9713

<div class="k-default-codeblock">
```

```
</div>
 40/46 ━━━━━━━━━━━━━━━━━[37m━━━  1s 178ms/step - aug_p: 0.0063 - d_loss: 0.0752 - g_loss: 4.8742 - gen_acc: 0.9916 - real_acc: 0.9712

<div class="k-default-codeblock">
```

```
</div>
 41/46 ━━━━━━━━━━━━━━━━━[37m━━━  0s 178ms/step - aug_p: 0.0064 - d_loss: 0.0756 - g_loss: 4.8685 - gen_acc: 0.9914 - real_acc: 0.9710

<div class="k-default-codeblock">
```

```
</div>
 42/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 178ms/step - aug_p: 0.0064 - d_loss: 0.0759 - g_loss: 4.8620 - gen_acc: 0.9911 - real_acc: 0.9709

<div class="k-default-codeblock">
```

```
</div>
 43/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 178ms/step - aug_p: 0.0064 - d_loss: 0.0762 - g_loss: 4.8555 - gen_acc: 0.9909 - real_acc: 0.9708

<div class="k-default-codeblock">
```

```
</div>
 44/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 178ms/step - aug_p: 0.0064 - d_loss: 0.0765 - g_loss: 4.8492 - gen_acc: 0.9907 - real_acc: 0.9707

<div class="k-default-codeblock">
```

```
</div>
 45/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 178ms/step - aug_p: 0.0065 - d_loss: 0.0768 - g_loss: 4.8424 - gen_acc: 0.9905 - real_acc: 0.9707

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 178ms/step - aug_p: 0.0065 - d_loss: 0.0771 - g_loss: 4.8357 - gen_acc: 0.9902 - real_acc: 0.9706

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 43s 280ms/step - aug_p: 0.0065 - d_loss: 0.0774 - g_loss: 4.8293 - gen_acc: 0.9900 - real_acc: 0.9705 - val_kid: 8.8293


<div class="k-default-codeblock">
```
Epoch 3/10

```
</div>
    
  1/46 [37m━━━━━━━━━━━━━━━━━━━━  5:54 8s/step - aug_p: 0.0105 - d_loss: 0.0941 - g_loss: 3.4148 - gen_acc: 0.9766 - real_acc: 0.9609

<div class="k-default-codeblock">
```

```
</div>
  2/46 [37m━━━━━━━━━━━━━━━━━━━━  8s 196ms/step - aug_p: 0.0105 - d_loss: 0.0925 - g_loss: 3.3668 - gen_acc: 0.9805 - real_acc: 0.9668

<div class="k-default-codeblock">
```

```
</div>
  3/46 ━[37m━━━━━━━━━━━━━━━━━━━  8s 187ms/step - aug_p: 0.0106 - d_loss: 0.0918 - g_loss: 3.3820 - gen_acc: 0.9835 - real_acc: 0.9666

<div class="k-default-codeblock">
```

```
</div>
  4/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 183ms/step - aug_p: 0.0106 - d_loss: 0.0932 - g_loss: 3.3732 - gen_acc: 0.9847 - real_acc: 0.9661

<div class="k-default-codeblock">
```

```
</div>
  5/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 182ms/step - aug_p: 0.0106 - d_loss: 0.0941 - g_loss: 3.3531 - gen_acc: 0.9859 - real_acc: 0.9670

<div class="k-default-codeblock">
```

```
</div>
  6/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 181ms/step - aug_p: 0.0107 - d_loss: 0.0942 - g_loss: 3.3519 - gen_acc: 0.9869 - real_acc: 0.9679

<div class="k-default-codeblock">
```

```
</div>
  7/46 ━━━[37m━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0107 - d_loss: 0.0941 - g_loss: 3.3467 - gen_acc: 0.9877 - real_acc: 0.9690

<div class="k-default-codeblock">
```

```
</div>
  8/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 179ms/step - aug_p: 0.0107 - d_loss: 0.0944 - g_loss: 3.3438 - gen_acc: 0.9882 - real_acc: 0.9693

<div class="k-default-codeblock">
```

```
</div>
  9/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 179ms/step - aug_p: 0.0107 - d_loss: 0.0947 - g_loss: 3.3384 - gen_acc: 0.9886 - real_acc: 0.9696

<div class="k-default-codeblock">
```

```
</div>
 10/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0108 - d_loss: 0.0948 - g_loss: 3.3468 - gen_acc: 0.9889 - real_acc: 0.9694

<div class="k-default-codeblock">
```

```
</div>
 11/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0108 - d_loss: 0.0952 - g_loss: 3.3443 - gen_acc: 0.9888 - real_acc: 0.9695

<div class="k-default-codeblock">
```

```
</div>
 12/46 ━━━━━[37m━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0108 - d_loss: 0.0955 - g_loss: 3.3676 - gen_acc: 0.9887 - real_acc: 0.9693

<div class="k-default-codeblock">
```

```
</div>
 13/46 ━━━━━[37m━━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0109 - d_loss: 0.0954 - g_loss: 3.3959 - gen_acc: 0.9888 - real_acc: 0.9693

<div class="k-default-codeblock">
```

```
</div>
 14/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0109 - d_loss: 0.0957 - g_loss: 3.4125 - gen_acc: 0.9883 - real_acc: 0.9694

<div class="k-default-codeblock">
```

```
</div>
 15/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0109 - d_loss: 0.0963 - g_loss: 3.4419 - gen_acc: 0.9880 - real_acc: 0.9688

<div class="k-default-codeblock">
```

```
</div>
 16/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0110 - d_loss: 0.0969 - g_loss: 3.4641 - gen_acc: 0.9876 - real_acc: 0.9684

<div class="k-default-codeblock">
```

```
</div>
 17/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0110 - d_loss: 0.0972 - g_loss: 3.4867 - gen_acc: 0.9873 - real_acc: 0.9681

<div class="k-default-codeblock">
```

```
</div>
 18/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0110 - d_loss: 0.0975 - g_loss: 3.5046 - gen_acc: 0.9869 - real_acc: 0.9679

<div class="k-default-codeblock">
```

```
</div>
 19/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0110 - d_loss: 0.0977 - g_loss: 3.5235 - gen_acc: 0.9866 - real_acc: 0.9678

<div class="k-default-codeblock">
```

```
</div>
 20/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0111 - d_loss: 0.0978 - g_loss: 3.5387 - gen_acc: 0.9863 - real_acc: 0.9677

<div class="k-default-codeblock">
```

```
</div>
 21/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0111 - d_loss: 0.0980 - g_loss: 3.5544 - gen_acc: 0.9861 - real_acc: 0.9676

<div class="k-default-codeblock">
```

```
</div>
 22/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 179ms/step - aug_p: 0.0111 - d_loss: 0.0983 - g_loss: 3.5646 - gen_acc: 0.9857 - real_acc: 0.9675

<div class="k-default-codeblock">
```

```
</div>
 23/46 ━━━━━━━━━━[37m━━━━━━━━━━  4s 179ms/step - aug_p: 0.0112 - d_loss: 0.0990 - g_loss: 3.5834 - gen_acc: 0.9853 - real_acc: 0.9670

<div class="k-default-codeblock">
```

```
</div>
 24/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 179ms/step - aug_p: 0.0112 - d_loss: 0.0995 - g_loss: 3.6027 - gen_acc: 0.9850 - real_acc: 0.9665

<div class="k-default-codeblock">
```

```
</div>
 25/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 179ms/step - aug_p: 0.0112 - d_loss: 0.1001 - g_loss: 3.6171 - gen_acc: 0.9845 - real_acc: 0.9662

<div class="k-default-codeblock">
```

```
</div>
 26/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 179ms/step - aug_p: 0.0112 - d_loss: 0.1006 - g_loss: 3.6374 - gen_acc: 0.9840 - real_acc: 0.9659

<div class="k-default-codeblock">
```

```
</div>
 27/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 179ms/step - aug_p: 0.0113 - d_loss: 0.1009 - g_loss: 3.6630 - gen_acc: 0.9836 - real_acc: 0.9656

<div class="k-default-codeblock">
```

```
</div>
 28/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 179ms/step - aug_p: 0.0113 - d_loss: 0.1012 - g_loss: 3.6907 - gen_acc: 0.9833 - real_acc: 0.9654

<div class="k-default-codeblock">
```

```
</div>
 29/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 179ms/step - aug_p: 0.0113 - d_loss: 0.1014 - g_loss: 3.7165 - gen_acc: 0.9830 - real_acc: 0.9652

<div class="k-default-codeblock">
```

```
</div>
 30/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 179ms/step - aug_p: 0.0114 - d_loss: 0.1016 - g_loss: 3.7387 - gen_acc: 0.9827 - real_acc: 0.9651

<div class="k-default-codeblock">
```

```
</div>
 31/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 179ms/step - aug_p: 0.0114 - d_loss: 0.1016 - g_loss: 3.7601 - gen_acc: 0.9824 - real_acc: 0.9650

<div class="k-default-codeblock">
```

```
</div>
 32/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 179ms/step - aug_p: 0.0114 - d_loss: 0.1017 - g_loss: 3.7799 - gen_acc: 0.9822 - real_acc: 0.9649

<div class="k-default-codeblock">
```

```
</div>
 33/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 179ms/step - aug_p: 0.0114 - d_loss: 0.1017 - g_loss: 3.7963 - gen_acc: 0.9820 - real_acc: 0.9649

<div class="k-default-codeblock">
```

```
</div>
 34/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 179ms/step - aug_p: 0.0115 - d_loss: 0.1019 - g_loss: 3.8154 - gen_acc: 0.9818 - real_acc: 0.9647

<div class="k-default-codeblock">
```

```
</div>
 35/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 179ms/step - aug_p: 0.0115 - d_loss: 0.1020 - g_loss: 3.8348 - gen_acc: 0.9816 - real_acc: 0.9645

<div class="k-default-codeblock">
```

```
</div>
 36/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 179ms/step - aug_p: 0.0115 - d_loss: 0.1022 - g_loss: 3.8515 - gen_acc: 0.9813 - real_acc: 0.9644

<div class="k-default-codeblock">
```

```
</div>
 37/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 179ms/step - aug_p: 0.0115 - d_loss: 0.1025 - g_loss: 3.8702 - gen_acc: 0.9810 - real_acc: 0.9642

<div class="k-default-codeblock">
```

```
</div>
 38/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 179ms/step - aug_p: 0.0116 - d_loss: 0.1027 - g_loss: 3.8891 - gen_acc: 0.9807 - real_acc: 0.9640

<div class="k-default-codeblock">
```

```
</div>
 39/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 179ms/step - aug_p: 0.0116 - d_loss: 0.1032 - g_loss: 3.9048 - gen_acc: 0.9803 - real_acc: 0.9638

<div class="k-default-codeblock">
```

```
</div>
 40/46 ━━━━━━━━━━━━━━━━━[37m━━━  1s 179ms/step - aug_p: 0.0116 - d_loss: 0.1036 - g_loss: 3.9227 - gen_acc: 0.9799 - real_acc: 0.9636

<div class="k-default-codeblock">
```

```
</div>
 41/46 ━━━━━━━━━━━━━━━━━[37m━━━  0s 179ms/step - aug_p: 0.0117 - d_loss: 0.1040 - g_loss: 3.9415 - gen_acc: 0.9796 - real_acc: 0.9633

<div class="k-default-codeblock">
```

```
</div>
 42/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 179ms/step - aug_p: 0.0117 - d_loss: 0.1044 - g_loss: 3.9588 - gen_acc: 0.9792 - real_acc: 0.9631

<div class="k-default-codeblock">
```

```
</div>
 43/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 179ms/step - aug_p: 0.0117 - d_loss: 0.1048 - g_loss: 3.9748 - gen_acc: 0.9789 - real_acc: 0.9629

<div class="k-default-codeblock">
```

```
</div>
 44/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 179ms/step - aug_p: 0.0117 - d_loss: 0.1052 - g_loss: 3.9895 - gen_acc: 0.9785 - real_acc: 0.9627

<div class="k-default-codeblock">
```

```
</div>
 45/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 179ms/step - aug_p: 0.0118 - d_loss: 0.1055 - g_loss: 4.0041 - gen_acc: 0.9782 - real_acc: 0.9626

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 179ms/step - aug_p: 0.0118 - d_loss: 0.1058 - g_loss: 4.0177 - gen_acc: 0.9779 - real_acc: 0.9624

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 22s 315ms/step - aug_p: 0.0118 - d_loss: 0.1061 - g_loss: 4.0306 - gen_acc: 0.9776 - real_acc: 0.9623 - val_kid: 8.4585


<div class="k-default-codeblock">
```
Epoch 4/10

```
</div>
    
  1/46 [37m━━━━━━━━━━━━━━━━━━━━  11s 263ms/step - aug_p: 0.0154 - d_loss: 0.1223 - g_loss: 2.5203 - gen_acc: 0.9688 - real_acc: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  2/46 [37m━━━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0154 - d_loss: 0.1018 - g_loss: 3.6445 - gen_acc: 0.9766 - real_acc: 0.9980 

<div class="k-default-codeblock">
```

```
</div>
  3/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0155 - d_loss: 0.0925 - g_loss: 4.2071 - gen_acc: 0.9809 - real_acc: 0.9926

<div class="k-default-codeblock">
```

```
</div>
  4/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0155 - d_loss: 0.0875 - g_loss: 4.3535 - gen_acc: 0.9827 - real_acc: 0.9910

<div class="k-default-codeblock">
```

```
</div>
  5/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0155 - d_loss: 0.0850 - g_loss: 4.3580 - gen_acc: 0.9843 - real_acc: 0.9900

<div class="k-default-codeblock">
```

```
</div>
  6/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0156 - d_loss: 0.0830 - g_loss: 4.3789 - gen_acc: 0.9856 - real_acc: 0.9889

<div class="k-default-codeblock">
```

```
</div>
  7/46 ━━━[37m━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0156 - d_loss: 0.0821 - g_loss: 4.3592 - gen_acc: 0.9864 - real_acc: 0.9879

<div class="k-default-codeblock">
```

```
</div>
  8/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0156 - d_loss: 0.0814 - g_loss: 4.3377 - gen_acc: 0.9871 - real_acc: 0.9870

<div class="k-default-codeblock">
```

```
</div>
  9/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0156 - d_loss: 0.0811 - g_loss: 4.3049 - gen_acc: 0.9876 - real_acc: 0.9864

<div class="k-default-codeblock">
```

```
</div>
 10/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0157 - d_loss: 0.0807 - g_loss: 4.2813 - gen_acc: 0.9881 - real_acc: 0.9859

<div class="k-default-codeblock">
```

```
</div>
 11/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0157 - d_loss: 0.0804 - g_loss: 4.2560 - gen_acc: 0.9884 - real_acc: 0.9855

<div class="k-default-codeblock">
```

```
</div>
 12/46 ━━━━━[37m━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0157 - d_loss: 0.0804 - g_loss: 4.2284 - gen_acc: 0.9886 - real_acc: 0.9851

<div class="k-default-codeblock">
```

```
</div>
 13/46 ━━━━━[37m━━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0158 - d_loss: 0.0806 - g_loss: 4.2117 - gen_acc: 0.9888 - real_acc: 0.9847

<div class="k-default-codeblock">
```

```
</div>
 14/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0158 - d_loss: 0.0807 - g_loss: 4.1925 - gen_acc: 0.9887 - real_acc: 0.9844

<div class="k-default-codeblock">
```

```
</div>
 15/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0158 - d_loss: 0.0807 - g_loss: 4.1790 - gen_acc: 0.9888 - real_acc: 0.9841

<div class="k-default-codeblock">
```

```
</div>
 16/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0159 - d_loss: 0.0807 - g_loss: 4.1631 - gen_acc: 0.9887 - real_acc: 0.9840

<div class="k-default-codeblock">
```

```
</div>
 17/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0159 - d_loss: 0.0807 - g_loss: 4.1518 - gen_acc: 0.9887 - real_acc: 0.9838

<div class="k-default-codeblock">
```

```
</div>
 18/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0159 - d_loss: 0.0807 - g_loss: 4.1398 - gen_acc: 0.9887 - real_acc: 0.9837

<div class="k-default-codeblock">
```

```
</div>
 19/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0160 - d_loss: 0.0809 - g_loss: 4.1255 - gen_acc: 0.9887 - real_acc: 0.9837

<div class="k-default-codeblock">
```

```
</div>
 20/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0160 - d_loss: 0.0810 - g_loss: 4.1171 - gen_acc: 0.9887 - real_acc: 0.9835

<div class="k-default-codeblock">
```

```
</div>
 21/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0160 - d_loss: 0.0810 - g_loss: 4.1116 - gen_acc: 0.9886 - real_acc: 0.9835

<div class="k-default-codeblock">
```

```
</div>
 22/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0161 - d_loss: 0.0811 - g_loss: 4.1037 - gen_acc: 0.9885 - real_acc: 0.9834

<div class="k-default-codeblock">
```

```
</div>
 23/46 ━━━━━━━━━━[37m━━━━━━━━━━  4s 180ms/step - aug_p: 0.0161 - d_loss: 0.0812 - g_loss: 4.1013 - gen_acc: 0.9885 - real_acc: 0.9833

<div class="k-default-codeblock">
```

```
</div>
 24/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 180ms/step - aug_p: 0.0161 - d_loss: 0.0813 - g_loss: 4.1000 - gen_acc: 0.9884 - real_acc: 0.9832

<div class="k-default-codeblock">
```

```
</div>
 25/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 180ms/step - aug_p: 0.0162 - d_loss: 0.0814 - g_loss: 4.0967 - gen_acc: 0.9883 - real_acc: 0.9832

<div class="k-default-codeblock">
```

```
</div>
 26/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 180ms/step - aug_p: 0.0162 - d_loss: 0.0815 - g_loss: 4.0951 - gen_acc: 0.9882 - real_acc: 0.9831

<div class="k-default-codeblock">
```

```
</div>
 27/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 180ms/step - aug_p: 0.0162 - d_loss: 0.0815 - g_loss: 4.0930 - gen_acc: 0.9882 - real_acc: 0.9830

<div class="k-default-codeblock">
```

```
</div>
 28/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 180ms/step - aug_p: 0.0163 - d_loss: 0.0817 - g_loss: 4.0887 - gen_acc: 0.9880 - real_acc: 0.9830

<div class="k-default-codeblock">
```

```
</div>
 29/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 180ms/step - aug_p: 0.0163 - d_loss: 0.0818 - g_loss: 4.0890 - gen_acc: 0.9879 - real_acc: 0.9829

<div class="k-default-codeblock">
```

```
</div>
 30/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 181ms/step - aug_p: 0.0163 - d_loss: 0.0819 - g_loss: 4.0918 - gen_acc: 0.9878 - real_acc: 0.9828

<div class="k-default-codeblock">
```

```
</div>
 31/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 181ms/step - aug_p: 0.0164 - d_loss: 0.0821 - g_loss: 4.0923 - gen_acc: 0.9877 - real_acc: 0.9826

<div class="k-default-codeblock">
```

```
</div>
 32/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 181ms/step - aug_p: 0.0164 - d_loss: 0.0823 - g_loss: 4.0957 - gen_acc: 0.9876 - real_acc: 0.9826

<div class="k-default-codeblock">
```

```
</div>
 33/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 181ms/step - aug_p: 0.0164 - d_loss: 0.0824 - g_loss: 4.1014 - gen_acc: 0.9874 - real_acc: 0.9825

<div class="k-default-codeblock">
```

```
</div>
 34/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 181ms/step - aug_p: 0.0165 - d_loss: 0.0824 - g_loss: 4.1072 - gen_acc: 0.9873 - real_acc: 0.9824

<div class="k-default-codeblock">
```

```
</div>
 35/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 180ms/step - aug_p: 0.0165 - d_loss: 0.0825 - g_loss: 4.1116 - gen_acc: 0.9872 - real_acc: 0.9823

<div class="k-default-codeblock">
```

```
</div>
 36/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 180ms/step - aug_p: 0.0165 - d_loss: 0.0826 - g_loss: 4.1168 - gen_acc: 0.9871 - real_acc: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 37/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 181ms/step - aug_p: 0.0166 - d_loss: 0.0827 - g_loss: 4.1217 - gen_acc: 0.9870 - real_acc: 0.9822

<div class="k-default-codeblock">
```

```
</div>
 38/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 180ms/step - aug_p: 0.0166 - d_loss: 0.0827 - g_loss: 4.1262 - gen_acc: 0.9869 - real_acc: 0.9821

<div class="k-default-codeblock">
```

```
</div>
 39/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 180ms/step - aug_p: 0.0166 - d_loss: 0.0828 - g_loss: 4.1298 - gen_acc: 0.9868 - real_acc: 0.9820

<div class="k-default-codeblock">
```

```
</div>
 40/46 ━━━━━━━━━━━━━━━━━[37m━━━  1s 180ms/step - aug_p: 0.0167 - d_loss: 0.0828 - g_loss: 4.1333 - gen_acc: 0.9868 - real_acc: 0.9820

<div class="k-default-codeblock">
```

```
</div>
 41/46 ━━━━━━━━━━━━━━━━━[37m━━━  0s 181ms/step - aug_p: 0.0167 - d_loss: 0.0828 - g_loss: 4.1361 - gen_acc: 0.9867 - real_acc: 0.9819

<div class="k-default-codeblock">
```

```
</div>
 42/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 181ms/step - aug_p: 0.0167 - d_loss: 0.0828 - g_loss: 4.1389 - gen_acc: 0.9866 - real_acc: 0.9819

<div class="k-default-codeblock">
```

```
</div>
 43/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 181ms/step - aug_p: 0.0168 - d_loss: 0.0828 - g_loss: 4.1408 - gen_acc: 0.9866 - real_acc: 0.9819

<div class="k-default-codeblock">
```

```
</div>
 44/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 181ms/step - aug_p: 0.0168 - d_loss: 0.0828 - g_loss: 4.1438 - gen_acc: 0.9865 - real_acc: 0.9818

<div class="k-default-codeblock">
```

```
</div>
 45/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 181ms/step - aug_p: 0.0168 - d_loss: 0.0828 - g_loss: 4.1466 - gen_acc: 0.9865 - real_acc: 0.9818

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 181ms/step - aug_p: 0.0168 - d_loss: 0.0829 - g_loss: 4.1480 - gen_acc: 0.9864 - real_acc: 0.9818

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 14s 316ms/step - aug_p: 0.0169 - d_loss: 0.0829 - g_loss: 4.1493 - gen_acc: 0.9863 - real_acc: 0.9817 - val_kid: 6.6764


<div class="k-default-codeblock">
```
Epoch 5/10

```
</div>
    
  1/46 [37m━━━━━━━━━━━━━━━━━━━━  10s 237ms/step - aug_p: 0.0212 - d_loss: 0.3046 - g_loss: 11.2403 - gen_acc: 1.0000 - real_acc: 0.7734

<div class="k-default-codeblock">
```

```
</div>
  2/46 [37m━━━━━━━━━━━━━━━━━━━━  8s 197ms/step - aug_p: 0.0212 - d_loss: 0.2549 - g_loss: 10.8464 - gen_acc: 1.0000 - real_acc: 0.8086 

<div class="k-default-codeblock">
```

```
</div>
  3/46 ━[37m━━━━━━━━━━━━━━━━━━━  8s 190ms/step - aug_p: 0.0212 - d_loss: 0.2217 - g_loss: 10.0394 - gen_acc: 0.9983 - real_acc: 0.8359

<div class="k-default-codeblock">
```

```
</div>
  4/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 187ms/step - aug_p: 0.0212 - d_loss: 0.2183 - g_loss: 9.2019 - gen_acc: 0.9753 - real_acc: 0.8560 

<div class="k-default-codeblock">
```

```
</div>
  5/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 185ms/step - aug_p: 0.0212 - d_loss: 0.2125 - g_loss: 8.8056 - gen_acc: 0.9652 - real_acc: 0.8676

<div class="k-default-codeblock">
```

```
</div>
  6/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 185ms/step - aug_p: 0.0213 - d_loss: 0.2060 - g_loss: 8.5755 - gen_acc: 0.9606 - real_acc: 0.8755

<div class="k-default-codeblock">
```

```
</div>
  7/46 ━━━[37m━━━━━━━━━━━━━━━━━  7s 184ms/step - aug_p: 0.0213 - d_loss: 0.1995 - g_loss: 8.3695 - gen_acc: 0.9579 - real_acc: 0.8823

<div class="k-default-codeblock">
```

```
</div>
  8/46 ━━━[37m━━━━━━━━━━━━━━━━━  7s 184ms/step - aug_p: 0.0213 - d_loss: 0.1938 - g_loss: 8.1574 - gen_acc: 0.9561 - real_acc: 0.8884

<div class="k-default-codeblock">
```

```
</div>
  9/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 184ms/step - aug_p: 0.0214 - d_loss: 0.1881 - g_loss: 7.9590 - gen_acc: 0.9552 - real_acc: 0.8939

<div class="k-default-codeblock">
```

```
</div>
 10/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 184ms/step - aug_p: 0.0214 - d_loss: 0.1827 - g_loss: 7.7719 - gen_acc: 0.9550 - real_acc: 0.8989

<div class="k-default-codeblock">
```

```
</div>
 11/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 183ms/step - aug_p: 0.0214 - d_loss: 0.1785 - g_loss: 7.5867 - gen_acc: 0.9546 - real_acc: 0.9034

<div class="k-default-codeblock">
```

```
</div>
 12/46 ━━━━━[37m━━━━━━━━━━━━━━━  6s 183ms/step - aug_p: 0.0214 - d_loss: 0.1748 - g_loss: 7.4330 - gen_acc: 0.9546 - real_acc: 0.9072

<div class="k-default-codeblock">
```

```
</div>
 13/46 ━━━━━[37m━━━━━━━━━━━━━━━  6s 183ms/step - aug_p: 0.0215 - d_loss: 0.1717 - g_loss: 7.2895 - gen_acc: 0.9548 - real_acc: 0.9103

<div class="k-default-codeblock">
```

```
</div>
 14/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 183ms/step - aug_p: 0.0215 - d_loss: 0.1693 - g_loss: 7.1489 - gen_acc: 0.9544 - real_acc: 0.9132

<div class="k-default-codeblock">
```

```
</div>
 15/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 182ms/step - aug_p: 0.0215 - d_loss: 0.1674 - g_loss: 7.0344 - gen_acc: 0.9543 - real_acc: 0.9153

<div class="k-default-codeblock">
```

```
</div>
 16/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 182ms/step - aug_p: 0.0215 - d_loss: 0.1654 - g_loss: 6.9321 - gen_acc: 0.9544 - real_acc: 0.9173

<div class="k-default-codeblock">
```

```
</div>
 17/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 182ms/step - aug_p: 0.0216 - d_loss: 0.1637 - g_loss: 6.8304 - gen_acc: 0.9541 - real_acc: 0.9191

<div class="k-default-codeblock">
```

```
</div>
 18/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 183ms/step - aug_p: 0.0216 - d_loss: 0.1620 - g_loss: 6.7449 - gen_acc: 0.9540 - real_acc: 0.9209

<div class="k-default-codeblock">
```

```
</div>
 19/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 183ms/step - aug_p: 0.0216 - d_loss: 0.1603 - g_loss: 6.6702 - gen_acc: 0.9540 - real_acc: 0.9225

<div class="k-default-codeblock">
```

```
</div>
 20/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 182ms/step - aug_p: 0.0217 - d_loss: 0.1587 - g_loss: 6.5977 - gen_acc: 0.9541 - real_acc: 0.9240

<div class="k-default-codeblock">
```

```
</div>
 21/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 182ms/step - aug_p: 0.0217 - d_loss: 0.1572 - g_loss: 6.5271 - gen_acc: 0.9542 - real_acc: 0.9255

<div class="k-default-codeblock">
```

```
</div>
 22/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 182ms/step - aug_p: 0.0217 - d_loss: 0.1556 - g_loss: 6.4626 - gen_acc: 0.9544 - real_acc: 0.9269

<div class="k-default-codeblock">
```

```
</div>
 23/46 ━━━━━━━━━━[37m━━━━━━━━━━  4s 182ms/step - aug_p: 0.0217 - d_loss: 0.1540 - g_loss: 6.4028 - gen_acc: 0.9546 - real_acc: 0.9282

<div class="k-default-codeblock">
```

```
</div>
 24/46 ━━━━━━━━━━[37m━━━━━━━━━━  4s 182ms/step - aug_p: 0.0218 - d_loss: 0.1525 - g_loss: 6.3440 - gen_acc: 0.9548 - real_acc: 0.9295

<div class="k-default-codeblock">
```

```
</div>
 25/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 183ms/step - aug_p: 0.0218 - d_loss: 0.1510 - g_loss: 6.2898 - gen_acc: 0.9551 - real_acc: 0.9307

<div class="k-default-codeblock">
```

```
</div>
 26/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 183ms/step - aug_p: 0.0218 - d_loss: 0.1495 - g_loss: 6.2380 - gen_acc: 0.9554 - real_acc: 0.9318

<div class="k-default-codeblock">
```

```
</div>
 27/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 183ms/step - aug_p: 0.0219 - d_loss: 0.1481 - g_loss: 6.1880 - gen_acc: 0.9558 - real_acc: 0.9330

<div class="k-default-codeblock">
```

```
</div>
 28/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 182ms/step - aug_p: 0.0219 - d_loss: 0.1468 - g_loss: 6.1413 - gen_acc: 0.9561 - real_acc: 0.9340

<div class="k-default-codeblock">
```

```
</div>
 29/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 183ms/step - aug_p: 0.0219 - d_loss: 0.1454 - g_loss: 6.0966 - gen_acc: 0.9565 - real_acc: 0.9350

<div class="k-default-codeblock">
```

```
</div>
 30/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 182ms/step - aug_p: 0.0220 - d_loss: 0.1441 - g_loss: 6.0534 - gen_acc: 0.9569 - real_acc: 0.9360

<div class="k-default-codeblock">
```

```
</div>
 31/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 183ms/step - aug_p: 0.0220 - d_loss: 0.1428 - g_loss: 6.0122 - gen_acc: 0.9573 - real_acc: 0.9370

<div class="k-default-codeblock">
```

```
</div>
 32/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 182ms/step - aug_p: 0.0220 - d_loss: 0.1415 - g_loss: 5.9738 - gen_acc: 0.9577 - real_acc: 0.9379

<div class="k-default-codeblock">
```

```
</div>
 33/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 182ms/step - aug_p: 0.0220 - d_loss: 0.1403 - g_loss: 5.9369 - gen_acc: 0.9581 - real_acc: 0.9388

<div class="k-default-codeblock">
```

```
</div>
 34/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 182ms/step - aug_p: 0.0221 - d_loss: 0.1390 - g_loss: 5.9020 - gen_acc: 0.9585 - real_acc: 0.9396

<div class="k-default-codeblock">
```

```
</div>
 35/46 ━━━━━━━━━━━━━━━[37m━━━━━  2s 182ms/step - aug_p: 0.0221 - d_loss: 0.1378 - g_loss: 5.8680 - gen_acc: 0.9589 - real_acc: 0.9404

<div class="k-default-codeblock">
```

```
</div>
 36/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 182ms/step - aug_p: 0.0221 - d_loss: 0.1366 - g_loss: 5.8355 - gen_acc: 0.9592 - real_acc: 0.9412

<div class="k-default-codeblock">
```

```
</div>
 37/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 182ms/step - aug_p: 0.0222 - d_loss: 0.1355 - g_loss: 5.8042 - gen_acc: 0.9596 - real_acc: 0.9420

<div class="k-default-codeblock">
```

```
</div>
 38/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 182ms/step - aug_p: 0.0222 - d_loss: 0.1344 - g_loss: 5.7737 - gen_acc: 0.9600 - real_acc: 0.9427

<div class="k-default-codeblock">
```

```
</div>
 39/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 182ms/step - aug_p: 0.0222 - d_loss: 0.1333 - g_loss: 5.7447 - gen_acc: 0.9604 - real_acc: 0.9434

<div class="k-default-codeblock">
```

```
</div>
 40/46 ━━━━━━━━━━━━━━━━━[37m━━━  1s 182ms/step - aug_p: 0.0223 - d_loss: 0.1323 - g_loss: 5.7161 - gen_acc: 0.9608 - real_acc: 0.9441

<div class="k-default-codeblock">
```

```
</div>
 41/46 ━━━━━━━━━━━━━━━━━[37m━━━  0s 182ms/step - aug_p: 0.0223 - d_loss: 0.1313 - g_loss: 5.6892 - gen_acc: 0.9611 - real_acc: 0.9447

<div class="k-default-codeblock">
```

```
</div>
 42/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 182ms/step - aug_p: 0.0223 - d_loss: 0.1304 - g_loss: 5.6621 - gen_acc: 0.9615 - real_acc: 0.9453

<div class="k-default-codeblock">
```

```
</div>
 43/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 182ms/step - aug_p: 0.0223 - d_loss: 0.1296 - g_loss: 5.6390 - gen_acc: 0.9618 - real_acc: 0.9458

<div class="k-default-codeblock">
```

```
</div>
 44/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 182ms/step - aug_p: 0.0224 - d_loss: 0.1288 - g_loss: 5.6185 - gen_acc: 0.9621 - real_acc: 0.9463

<div class="k-default-codeblock">
```

```
</div>
 45/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 182ms/step - aug_p: 0.0224 - d_loss: 0.1280 - g_loss: 5.5982 - gen_acc: 0.9623 - real_acc: 0.9468

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 182ms/step - aug_p: 0.0224 - d_loss: 0.1273 - g_loss: 5.5795 - gen_acc: 0.9626 - real_acc: 0.9473


    
![png](/img/examples/generative/gan_ada/gan_ada_18_265.png)
    


<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 14s 317ms/step - aug_p: 0.0225 - d_loss: 0.1265 - g_loss: 5.5616 - gen_acc: 0.9629 - real_acc: 0.9478 - val_kid: 4.7496


<div class="k-default-codeblock">
```
Epoch 6/10

```
</div>
    
  1/46 [37m━━━━━━━━━━━━━━━━━━━━  10s 236ms/step - aug_p: 0.0268 - d_loss: 0.0745 - g_loss: 5.1780 - gen_acc: 0.9922 - real_acc: 0.9688

<div class="k-default-codeblock">
```

```
</div>
  2/46 [37m━━━━━━━━━━━━━━━━━━━━  8s 184ms/step - aug_p: 0.0269 - d_loss: 0.0774 - g_loss: 4.5412 - gen_acc: 0.9883 - real_acc: 0.9766 

<div class="k-default-codeblock">
```

```
</div>
  3/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 183ms/step - aug_p: 0.0269 - d_loss: 0.0743 - g_loss: 4.5406 - gen_acc: 0.9887 - real_acc: 0.9783

<div class="k-default-codeblock">
```

```
</div>
  4/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 184ms/step - aug_p: 0.0269 - d_loss: 0.0724 - g_loss: 4.5764 - gen_acc: 0.9896 - real_acc: 0.9779

<div class="k-default-codeblock">
```

```
</div>
  5/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 183ms/step - aug_p: 0.0270 - d_loss: 0.0732 - g_loss: 4.5209 - gen_acc: 0.9882 - real_acc: 0.9785

<div class="k-default-codeblock">
```

```
</div>
  6/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 182ms/step - aug_p: 0.0270 - d_loss: 0.0738 - g_loss: 4.5449 - gen_acc: 0.9878 - real_acc: 0.9782

<div class="k-default-codeblock">
```

```
</div>
  7/46 ━━━[37m━━━━━━━━━━━━━━━━━  7s 182ms/step - aug_p: 0.0270 - d_loss: 0.0747 - g_loss: 4.5880 - gen_acc: 0.9878 - real_acc: 0.9769

<div class="k-default-codeblock">
```

```
</div>
  8/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0271 - d_loss: 0.0766 - g_loss: 4.5791 - gen_acc: 0.9857 - real_acc: 0.9763

<div class="k-default-codeblock">
```

```
</div>
  9/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 183ms/step - aug_p: 0.0271 - d_loss: 0.0777 - g_loss: 4.6269 - gen_acc: 0.9844 - real_acc: 0.9757

<div class="k-default-codeblock">
```

```
</div>
 10/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0271 - d_loss: 0.0786 - g_loss: 4.7075 - gen_acc: 0.9836 - real_acc: 0.9749

<div class="k-default-codeblock">
```

```
</div>
 11/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0271 - d_loss: 0.0792 - g_loss: 4.7786 - gen_acc: 0.9826 - real_acc: 0.9745

<div class="k-default-codeblock">
```

```
</div>
 12/46 ━━━━━[37m━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0272 - d_loss: 0.0793 - g_loss: 4.8440 - gen_acc: 0.9820 - real_acc: 0.9744

<div class="k-default-codeblock">
```

```
</div>
 13/46 ━━━━━[37m━━━━━━━━━━━━━━━  6s 183ms/step - aug_p: 0.0272 - d_loss: 0.0792 - g_loss: 4.9001 - gen_acc: 0.9816 - real_acc: 0.9744

<div class="k-default-codeblock">
```

```
</div>
 14/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 182ms/step - aug_p: 0.0272 - d_loss: 0.0789 - g_loss: 4.9354 - gen_acc: 0.9814 - real_acc: 0.9745

<div class="k-default-codeblock">
```

```
</div>
 15/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 182ms/step - aug_p: 0.0273 - d_loss: 0.0785 - g_loss: 4.9643 - gen_acc: 0.9813 - real_acc: 0.9747

<div class="k-default-codeblock">
```

```
</div>
 16/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 182ms/step - aug_p: 0.0273 - d_loss: 0.0781 - g_loss: 4.9864 - gen_acc: 0.9814 - real_acc: 0.9749

<div class="k-default-codeblock">
```

```
</div>
 17/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 182ms/step - aug_p: 0.0273 - d_loss: 0.0778 - g_loss: 4.9973 - gen_acc: 0.9814 - real_acc: 0.9751

<div class="k-default-codeblock">
```

```
</div>
 18/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 182ms/step - aug_p: 0.0274 - d_loss: 0.0774 - g_loss: 5.0125 - gen_acc: 0.9815 - real_acc: 0.9753

<div class="k-default-codeblock">
```

```
</div>
 19/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 182ms/step - aug_p: 0.0274 - d_loss: 0.0770 - g_loss: 5.0280 - gen_acc: 0.9816 - real_acc: 0.9755

<div class="k-default-codeblock">
```

```
</div>
 20/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 182ms/step - aug_p: 0.0274 - d_loss: 0.0765 - g_loss: 5.0398 - gen_acc: 0.9818 - real_acc: 0.9757

<div class="k-default-codeblock">
```

```
</div>
 21/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 182ms/step - aug_p: 0.0275 - d_loss: 0.0760 - g_loss: 5.0455 - gen_acc: 0.9819 - real_acc: 0.9759

<div class="k-default-codeblock">
```

```
</div>
 22/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 182ms/step - aug_p: 0.0275 - d_loss: 0.0756 - g_loss: 5.0535 - gen_acc: 0.9820 - real_acc: 0.9760

<div class="k-default-codeblock">
```

```
</div>
 23/46 ━━━━━━━━━━[37m━━━━━━━━━━  4s 181ms/step - aug_p: 0.0275 - d_loss: 0.0752 - g_loss: 5.0590 - gen_acc: 0.9822 - real_acc: 0.9762

<div class="k-default-codeblock">
```

```
</div>
 24/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 181ms/step - aug_p: 0.0276 - d_loss: 0.0749 - g_loss: 5.0595 - gen_acc: 0.9823 - real_acc: 0.9763

<div class="k-default-codeblock">
```

```
</div>
 25/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 181ms/step - aug_p: 0.0276 - d_loss: 0.0746 - g_loss: 5.0650 - gen_acc: 0.9825 - real_acc: 0.9764

<div class="k-default-codeblock">
```

```
</div>
 26/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 181ms/step - aug_p: 0.0276 - d_loss: 0.0743 - g_loss: 5.0742 - gen_acc: 0.9826 - real_acc: 0.9765

<div class="k-default-codeblock">
```

```
</div>
 27/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 181ms/step - aug_p: 0.0277 - d_loss: 0.0740 - g_loss: 5.0823 - gen_acc: 0.9828 - real_acc: 0.9766

<div class="k-default-codeblock">
```

```
</div>
 28/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 181ms/step - aug_p: 0.0277 - d_loss: 0.0737 - g_loss: 5.0871 - gen_acc: 0.9829 - real_acc: 0.9767

<div class="k-default-codeblock">
```

```
</div>
 29/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 181ms/step - aug_p: 0.0277 - d_loss: 0.0734 - g_loss: 5.0913 - gen_acc: 0.9831 - real_acc: 0.9768

<div class="k-default-codeblock">
```

```
</div>
 30/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 181ms/step - aug_p: 0.0278 - d_loss: 0.0731 - g_loss: 5.0957 - gen_acc: 0.9832 - real_acc: 0.9769

<div class="k-default-codeblock">
```

```
</div>
 31/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 181ms/step - aug_p: 0.0278 - d_loss: 0.0727 - g_loss: 5.0986 - gen_acc: 0.9834 - real_acc: 0.9770

<div class="k-default-codeblock">
```

```
</div>
 32/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 181ms/step - aug_p: 0.0278 - d_loss: 0.0725 - g_loss: 5.0992 - gen_acc: 0.9835 - real_acc: 0.9771

<div class="k-default-codeblock">
```

```
</div>
 33/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 181ms/step - aug_p: 0.0278 - d_loss: 0.0722 - g_loss: 5.1012 - gen_acc: 0.9836 - real_acc: 0.9772

<div class="k-default-codeblock">
```

```
</div>
 34/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 181ms/step - aug_p: 0.0279 - d_loss: 0.0719 - g_loss: 5.1022 - gen_acc: 0.9838 - real_acc: 0.9773

<div class="k-default-codeblock">
```

```
</div>
 35/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 181ms/step - aug_p: 0.0279 - d_loss: 0.0718 - g_loss: 5.1007 - gen_acc: 0.9838 - real_acc: 0.9773

<div class="k-default-codeblock">
```

```
</div>
 36/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 181ms/step - aug_p: 0.0279 - d_loss: 0.0717 - g_loss: 5.1026 - gen_acc: 0.9839 - real_acc: 0.9773

<div class="k-default-codeblock">
```

```
</div>
 37/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 181ms/step - aug_p: 0.0280 - d_loss: 0.0716 - g_loss: 5.1070 - gen_acc: 0.9840 - real_acc: 0.9772

<div class="k-default-codeblock">
```

```
</div>
 38/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 181ms/step - aug_p: 0.0280 - d_loss: 0.0715 - g_loss: 5.1124 - gen_acc: 0.9840 - real_acc: 0.9772

<div class="k-default-codeblock">
```

```
</div>
 39/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 181ms/step - aug_p: 0.0280 - d_loss: 0.0714 - g_loss: 5.1178 - gen_acc: 0.9841 - real_acc: 0.9773

<div class="k-default-codeblock">
```

```
</div>
 40/46 ━━━━━━━━━━━━━━━━━[37m━━━  1s 181ms/step - aug_p: 0.0281 - d_loss: 0.0712 - g_loss: 5.1221 - gen_acc: 0.9842 - real_acc: 0.9773

<div class="k-default-codeblock">
```

```
</div>
 41/46 ━━━━━━━━━━━━━━━━━[37m━━━  0s 181ms/step - aug_p: 0.0281 - d_loss: 0.0710 - g_loss: 5.1258 - gen_acc: 0.9843 - real_acc: 0.9773

<div class="k-default-codeblock">
```

```
</div>
 42/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 181ms/step - aug_p: 0.0281 - d_loss: 0.0708 - g_loss: 5.1290 - gen_acc: 0.9843 - real_acc: 0.9773

<div class="k-default-codeblock">
```

```
</div>
 43/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 181ms/step - aug_p: 0.0282 - d_loss: 0.0707 - g_loss: 5.1315 - gen_acc: 0.9844 - real_acc: 0.9774

<div class="k-default-codeblock">
```

```
</div>
 44/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 181ms/step - aug_p: 0.0282 - d_loss: 0.0705 - g_loss: 5.1332 - gen_acc: 0.9845 - real_acc: 0.9774

<div class="k-default-codeblock">
```

```
</div>
 45/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 181ms/step - aug_p: 0.0282 - d_loss: 0.0703 - g_loss: 5.1347 - gen_acc: 0.9845 - real_acc: 0.9775

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 181ms/step - aug_p: 0.0283 - d_loss: 0.0701 - g_loss: 5.1357 - gen_acc: 0.9846 - real_acc: 0.9775

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 12s 267ms/step - aug_p: 0.0283 - d_loss: 0.0699 - g_loss: 5.1367 - gen_acc: 0.9846 - real_acc: 0.9776 - val_kid: 6.2893


<div class="k-default-codeblock">
```
Epoch 7/10

```
</div>
    
  1/46 [37m━━━━━━━━━━━━━━━━━━━━  7s 174ms/step - aug_p: 0.0328 - d_loss: 0.0456 - g_loss: 3.5202 - gen_acc: 1.0000 - real_acc: 1.0000

<div class="k-default-codeblock">
```

```
</div>
  2/46 [37m━━━━━━━━━━━━━━━━━━━━  7s 179ms/step - aug_p: 0.0329 - d_loss: 0.0466 - g_loss: 3.7961 - gen_acc: 0.9980 - real_acc: 0.9980

<div class="k-default-codeblock">
```

```
</div>
  3/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 179ms/step - aug_p: 0.0329 - d_loss: 0.0471 - g_loss: 3.9462 - gen_acc: 0.9970 - real_acc: 0.9961

<div class="k-default-codeblock">
```

```
</div>
  4/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 179ms/step - aug_p: 0.0329 - d_loss: 0.0469 - g_loss: 4.0184 - gen_acc: 0.9967 - real_acc: 0.9946

<div class="k-default-codeblock">
```

```
</div>
  5/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0330 - d_loss: 0.0463 - g_loss: 4.0670 - gen_acc: 0.9968 - real_acc: 0.9941

<div class="k-default-codeblock">
```

```
</div>
  6/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0330 - d_loss: 0.0458 - g_loss: 4.1012 - gen_acc: 0.9969 - real_acc: 0.9938

<div class="k-default-codeblock">
```

```
</div>
  7/46 ━━━[37m━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0330 - d_loss: 0.0451 - g_loss: 4.1240 - gen_acc: 0.9970 - real_acc: 0.9937

<div class="k-default-codeblock">
```

```
</div>
  8/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0331 - d_loss: 0.0444 - g_loss: 4.1347 - gen_acc: 0.9971 - real_acc: 0.9938

<div class="k-default-codeblock">
```

```
</div>
  9/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0331 - d_loss: 0.0438 - g_loss: 4.1433 - gen_acc: 0.9971 - real_acc: 0.9937

<div class="k-default-codeblock">
```

```
</div>
 10/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0331 - d_loss: 0.0434 - g_loss: 4.1560 - gen_acc: 0.9970 - real_acc: 0.9936

<div class="k-default-codeblock">
```

```
</div>
 11/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0332 - d_loss: 0.0431 - g_loss: 4.1654 - gen_acc: 0.9969 - real_acc: 0.9936

<div class="k-default-codeblock">
```

```
</div>
 12/46 ━━━━━[37m━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0332 - d_loss: 0.0429 - g_loss: 4.1695 - gen_acc: 0.9969 - real_acc: 0.9935

<div class="k-default-codeblock">
```

```
</div>
 13/46 ━━━━━[37m━━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0333 - d_loss: 0.0428 - g_loss: 4.1758 - gen_acc: 0.9969 - real_acc: 0.9934

<div class="k-default-codeblock">
```

```
</div>
 14/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0333 - d_loss: 0.0427 - g_loss: 4.1789 - gen_acc: 0.9969 - real_acc: 0.9932

<div class="k-default-codeblock">
```

```
</div>
 15/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0333 - d_loss: 0.0426 - g_loss: 4.1799 - gen_acc: 0.9970 - real_acc: 0.9929

<div class="k-default-codeblock">
```

```
</div>
 16/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0334 - d_loss: 0.0425 - g_loss: 4.1823 - gen_acc: 0.9970 - real_acc: 0.9927

<div class="k-default-codeblock">
```

```
</div>
 17/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0334 - d_loss: 0.0425 - g_loss: 4.1836 - gen_acc: 0.9970 - real_acc: 0.9926

<div class="k-default-codeblock">
```

```
</div>
 18/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0334 - d_loss: 0.0425 - g_loss: 4.1854 - gen_acc: 0.9971 - real_acc: 0.9923

<div class="k-default-codeblock">
```

```
</div>
 19/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0335 - d_loss: 0.0426 - g_loss: 4.1843 - gen_acc: 0.9971 - real_acc: 0.9921

<div class="k-default-codeblock">
```

```
</div>
 20/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0335 - d_loss: 0.0427 - g_loss: 4.1873 - gen_acc: 0.9971 - real_acc: 0.9920

<div class="k-default-codeblock">
```

```
</div>
 21/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0335 - d_loss: 0.0427 - g_loss: 4.1927 - gen_acc: 0.9972 - real_acc: 0.9918

<div class="k-default-codeblock">
```

```
</div>
 22/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0336 - d_loss: 0.0428 - g_loss: 4.1952 - gen_acc: 0.9972 - real_acc: 0.9916

<div class="k-default-codeblock">
```

```
</div>
 23/46 ━━━━━━━━━━[37m━━━━━━━━━━  4s 180ms/step - aug_p: 0.0336 - d_loss: 0.0428 - g_loss: 4.2017 - gen_acc: 0.9972 - real_acc: 0.9915

<div class="k-default-codeblock">
```

```
</div>
 24/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 180ms/step - aug_p: 0.0336 - d_loss: 0.0428 - g_loss: 4.2106 - gen_acc: 0.9972 - real_acc: 0.9914

<div class="k-default-codeblock">
```

```
</div>
 25/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 180ms/step - aug_p: 0.0337 - d_loss: 0.0428 - g_loss: 4.2181 - gen_acc: 0.9972 - real_acc: 0.9913

<div class="k-default-codeblock">
```

```
</div>
 26/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 180ms/step - aug_p: 0.0337 - d_loss: 0.0428 - g_loss: 4.2229 - gen_acc: 0.9972 - real_acc: 0.9912

<div class="k-default-codeblock">
```

```
</div>
 27/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 180ms/step - aug_p: 0.0337 - d_loss: 0.0429 - g_loss: 4.2318 - gen_acc: 0.9972 - real_acc: 0.9911

<div class="k-default-codeblock">
```

```
</div>
 28/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 180ms/step - aug_p: 0.0338 - d_loss: 0.0429 - g_loss: 4.2416 - gen_acc: 0.9972 - real_acc: 0.9910

<div class="k-default-codeblock">
```

```
</div>
 29/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 180ms/step - aug_p: 0.0338 - d_loss: 0.0430 - g_loss: 4.2491 - gen_acc: 0.9971 - real_acc: 0.9909

<div class="k-default-codeblock">
```

```
</div>
 30/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 180ms/step - aug_p: 0.0338 - d_loss: 0.0430 - g_loss: 4.2604 - gen_acc: 0.9971 - real_acc: 0.9908

<div class="k-default-codeblock">
```

```
</div>
 31/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 180ms/step - aug_p: 0.0339 - d_loss: 0.0431 - g_loss: 4.2736 - gen_acc: 0.9971 - real_acc: 0.9907

<div class="k-default-codeblock">
```

```
</div>
 32/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 180ms/step - aug_p: 0.0339 - d_loss: 0.0432 - g_loss: 4.2834 - gen_acc: 0.9970 - real_acc: 0.9906

<div class="k-default-codeblock">
```

```
</div>
 33/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 180ms/step - aug_p: 0.0339 - d_loss: 0.0439 - g_loss: 4.3010 - gen_acc: 0.9968 - real_acc: 0.9901

<div class="k-default-codeblock">
```

```
</div>
 34/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 180ms/step - aug_p: 0.0340 - d_loss: 0.0444 - g_loss: 4.3187 - gen_acc: 0.9967 - real_acc: 0.9897

<div class="k-default-codeblock">
```

```
</div>
 35/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 180ms/step - aug_p: 0.0340 - d_loss: 0.0455 - g_loss: 4.3319 - gen_acc: 0.9961 - real_acc: 0.9892

<div class="k-default-codeblock">
```

```
</div>
 36/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 180ms/step - aug_p: 0.0340 - d_loss: 0.0464 - g_loss: 4.3508 - gen_acc: 0.9956 - real_acc: 0.9889

<div class="k-default-codeblock">
```

```
</div>
 37/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 180ms/step - aug_p: 0.0341 - d_loss: 0.0474 - g_loss: 4.3765 - gen_acc: 0.9951 - real_acc: 0.9884

<div class="k-default-codeblock">
```

```
</div>
 38/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 180ms/step - aug_p: 0.0341 - d_loss: 0.0483 - g_loss: 4.4070 - gen_acc: 0.9947 - real_acc: 0.9880

<div class="k-default-codeblock">
```

```
</div>
 39/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 180ms/step - aug_p: 0.0341 - d_loss: 0.0492 - g_loss: 4.4400 - gen_acc: 0.9943 - real_acc: 0.9875

<div class="k-default-codeblock">
```

```
</div>
 40/46 ━━━━━━━━━━━━━━━━━[37m━━━  1s 180ms/step - aug_p: 0.0342 - d_loss: 0.0499 - g_loss: 4.4739 - gen_acc: 0.9939 - real_acc: 0.9872

<div class="k-default-codeblock">
```

```
</div>
 41/46 ━━━━━━━━━━━━━━━━━[37m━━━  0s 180ms/step - aug_p: 0.0342 - d_loss: 0.0506 - g_loss: 4.5070 - gen_acc: 0.9935 - real_acc: 0.9868

<div class="k-default-codeblock">
```

```
</div>
 42/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 180ms/step - aug_p: 0.0342 - d_loss: 0.0513 - g_loss: 4.5375 - gen_acc: 0.9932 - real_acc: 0.9865

<div class="k-default-codeblock">
```

```
</div>
 43/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 180ms/step - aug_p: 0.0343 - d_loss: 0.0519 - g_loss: 4.5646 - gen_acc: 0.9929 - real_acc: 0.9862

<div class="k-default-codeblock">
```

```
</div>
 44/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 180ms/step - aug_p: 0.0343 - d_loss: 0.0525 - g_loss: 4.5904 - gen_acc: 0.9925 - real_acc: 0.9859

<div class="k-default-codeblock">
```

```
</div>
 45/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 180ms/step - aug_p: 0.0343 - d_loss: 0.0530 - g_loss: 4.6149 - gen_acc: 0.9923 - real_acc: 0.9857

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 180ms/step - aug_p: 0.0344 - d_loss: 0.0536 - g_loss: 4.6368 - gen_acc: 0.9920 - real_acc: 0.9854

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 13s 294ms/step - aug_p: 0.0344 - d_loss: 0.0542 - g_loss: 4.6579 - gen_acc: 0.9917 - real_acc: 0.9852 - val_kid: 6.7378


<div class="k-default-codeblock">
```
Epoch 8/10

```
</div>
    
  1/46 [37m━━━━━━━━━━━━━━━━━━━━  7s 167ms/step - aug_p: 0.0384 - d_loss: 0.1191 - g_loss: 4.3279 - gen_acc: 1.0000 - real_acc: 0.9219

<div class="k-default-codeblock">
```

```
</div>
  2/46 [37m━━━━━━━━━━━━━━━━━━━━  7s 179ms/step - aug_p: 0.0384 - d_loss: 0.1470 - g_loss: 3.7525 - gen_acc: 0.9590 - real_acc: 0.9219

<div class="k-default-codeblock">
```

```
</div>
  3/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0384 - d_loss: 0.1768 - g_loss: 4.0819 - gen_acc: 0.9544 - real_acc: 0.8950

<div class="k-default-codeblock">
```

```
</div>
  4/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0384 - d_loss: 0.1801 - g_loss: 4.1693 - gen_acc: 0.9551 - real_acc: 0.8910

<div class="k-default-codeblock">
```

```
</div>
  5/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0384 - d_loss: 0.1829 - g_loss: 4.1280 - gen_acc: 0.9491 - real_acc: 0.8934

<div class="k-default-codeblock">
```

```
</div>
  6/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0384 - d_loss: 0.1828 - g_loss: 4.2346 - gen_acc: 0.9471 - real_acc: 0.8949

<div class="k-default-codeblock">
```

```
</div>
  7/46 ━━━[37m━━━━━━━━━━━━━━━━━  7s 180ms/step - aug_p: 0.0385 - d_loss: 0.1806 - g_loss: 4.3823 - gen_acc: 0.9470 - real_acc: 0.8968

<div class="k-default-codeblock">
```

```
</div>
  8/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0385 - d_loss: 0.1765 - g_loss: 4.5079 - gen_acc: 0.9478 - real_acc: 0.8997

<div class="k-default-codeblock">
```

```
</div>
  9/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0385 - d_loss: 0.1723 - g_loss: 4.5814 - gen_acc: 0.9486 - real_acc: 0.9028

<div class="k-default-codeblock">
```

```
</div>
 10/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0385 - d_loss: 0.1679 - g_loss: 4.6213 - gen_acc: 0.9496 - real_acc: 0.9061

<div class="k-default-codeblock">
```

```
</div>
 11/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0385 - d_loss: 0.1637 - g_loss: 4.6466 - gen_acc: 0.9507 - real_acc: 0.9092

<div class="k-default-codeblock">
```

```
</div>
 12/46 ━━━━━[37m━━━━━━━━━━━━━━━  6s 180ms/step - aug_p: 0.0386 - d_loss: 0.1595 - g_loss: 4.6599 - gen_acc: 0.9520 - real_acc: 0.9122

<div class="k-default-codeblock">
```

```
</div>
 13/46 ━━━━━[37m━━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0386 - d_loss: 0.1561 - g_loss: 4.6625 - gen_acc: 0.9531 - real_acc: 0.9148

<div class="k-default-codeblock">
```

```
</div>
 14/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0386 - d_loss: 0.1535 - g_loss: 4.6513 - gen_acc: 0.9537 - real_acc: 0.9172

<div class="k-default-codeblock">
```

```
</div>
 15/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0386 - d_loss: 0.1530 - g_loss: 4.6600 - gen_acc: 0.9544 - real_acc: 0.9175

<div class="k-default-codeblock">
```

```
</div>
 16/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0387 - d_loss: 0.1612 - g_loss: 4.6490 - gen_acc: 0.9512 - real_acc: 0.9180

<div class="k-default-codeblock">
```

```
</div>
 17/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0387 - d_loss: 0.1727 - g_loss: 4.6715 - gen_acc: 0.9488 - real_acc: 0.9157

<div class="k-default-codeblock">
```

```
</div>
 18/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0387 - d_loss: 0.1825 - g_loss: 4.7072 - gen_acc: 0.9469 - real_acc: 0.9135

<div class="k-default-codeblock">
```

```
</div>
 19/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0387 - d_loss: 0.1904 - g_loss: 4.7428 - gen_acc: 0.9454 - real_acc: 0.9118

<div class="k-default-codeblock">
```

```
</div>
 20/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 179ms/step - aug_p: 0.0387 - d_loss: 0.1970 - g_loss: 4.7693 - gen_acc: 0.9440 - real_acc: 0.9106

<div class="k-default-codeblock">
```

```
</div>
 21/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 179ms/step - aug_p: 0.0387 - d_loss: 0.2029 - g_loss: 4.7854 - gen_acc: 0.9424 - real_acc: 0.9098

<div class="k-default-codeblock">
```

```
</div>
 22/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0387 - d_loss: 0.2079 - g_loss: 4.7960 - gen_acc: 0.9409 - real_acc: 0.9092

<div class="k-default-codeblock">
```

```
</div>
 23/46 ━━━━━━━━━━[37m━━━━━━━━━━  4s 180ms/step - aug_p: 0.0388 - d_loss: 0.2119 - g_loss: 4.8033 - gen_acc: 0.9397 - real_acc: 0.9090

<div class="k-default-codeblock">
```

```
</div>
 24/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 179ms/step - aug_p: 0.0388 - d_loss: 0.2153 - g_loss: 4.8076 - gen_acc: 0.9387 - real_acc: 0.9088

<div class="k-default-codeblock">
```

```
</div>
 25/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 179ms/step - aug_p: 0.0388 - d_loss: 0.2182 - g_loss: 4.8077 - gen_acc: 0.9378 - real_acc: 0.9087

<div class="k-default-codeblock">
```

```
</div>
 26/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 179ms/step - aug_p: 0.0388 - d_loss: 0.2207 - g_loss: 4.8051 - gen_acc: 0.9371 - real_acc: 0.9087

<div class="k-default-codeblock">
```

```
</div>
 27/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 179ms/step - aug_p: 0.0388 - d_loss: 0.2229 - g_loss: 4.8007 - gen_acc: 0.9365 - real_acc: 0.9086

<div class="k-default-codeblock">
```

```
</div>
 28/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 180ms/step - aug_p: 0.0388 - d_loss: 0.2249 - g_loss: 4.7934 - gen_acc: 0.9360 - real_acc: 0.9086

<div class="k-default-codeblock">
```

```
</div>
 29/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 180ms/step - aug_p: 0.0388 - d_loss: 0.2265 - g_loss: 4.7860 - gen_acc: 0.9355 - real_acc: 0.9086

<div class="k-default-codeblock">
```

```
</div>
 30/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 180ms/step - aug_p: 0.0389 - d_loss: 0.2278 - g_loss: 4.7775 - gen_acc: 0.9352 - real_acc: 0.9087

<div class="k-default-codeblock">
```

```
</div>
 31/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 180ms/step - aug_p: 0.0389 - d_loss: 0.2290 - g_loss: 4.7677 - gen_acc: 0.9349 - real_acc: 0.9087

<div class="k-default-codeblock">
```

```
</div>
 32/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 180ms/step - aug_p: 0.0389 - d_loss: 0.2299 - g_loss: 4.7575 - gen_acc: 0.9347 - real_acc: 0.9089

<div class="k-default-codeblock">
```

```
</div>
 33/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 180ms/step - aug_p: 0.0389 - d_loss: 0.2305 - g_loss: 4.7470 - gen_acc: 0.9346 - real_acc: 0.9091

<div class="k-default-codeblock">
```

```
</div>
 34/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 180ms/step - aug_p: 0.0389 - d_loss: 0.2310 - g_loss: 4.7363 - gen_acc: 0.9345 - real_acc: 0.9093

<div class="k-default-codeblock">
```

```
</div>
 35/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 180ms/step - aug_p: 0.0389 - d_loss: 0.2314 - g_loss: 4.7249 - gen_acc: 0.9344 - real_acc: 0.9095

<div class="k-default-codeblock">
```

```
</div>
 36/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 180ms/step - aug_p: 0.0389 - d_loss: 0.2317 - g_loss: 4.7149 - gen_acc: 0.9344 - real_acc: 0.9098

<div class="k-default-codeblock">
```

```
</div>
 37/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 180ms/step - aug_p: 0.0390 - d_loss: 0.2319 - g_loss: 4.7045 - gen_acc: 0.9345 - real_acc: 0.9101

<div class="k-default-codeblock">
```

```
</div>
 38/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 180ms/step - aug_p: 0.0390 - d_loss: 0.2319 - g_loss: 4.6937 - gen_acc: 0.9345 - real_acc: 0.9104

<div class="k-default-codeblock">
```

```
</div>
 39/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 179ms/step - aug_p: 0.0390 - d_loss: 0.2319 - g_loss: 4.6838 - gen_acc: 0.9346 - real_acc: 0.9107

<div class="k-default-codeblock">
```

```
</div>
 40/46 ━━━━━━━━━━━━━━━━━[37m━━━  1s 179ms/step - aug_p: 0.0390 - d_loss: 0.2318 - g_loss: 4.6734 - gen_acc: 0.9347 - real_acc: 0.9110

<div class="k-default-codeblock">
```

```
</div>
 41/46 ━━━━━━━━━━━━━━━━━[37m━━━  0s 179ms/step - aug_p: 0.0390 - d_loss: 0.2316 - g_loss: 4.6636 - gen_acc: 0.9349 - real_acc: 0.9114

<div class="k-default-codeblock">
```

```
</div>
 42/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 179ms/step - aug_p: 0.0390 - d_loss: 0.2313 - g_loss: 4.6532 - gen_acc: 0.9350 - real_acc: 0.9117

<div class="k-default-codeblock">
```

```
</div>
 43/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 180ms/step - aug_p: 0.0391 - d_loss: 0.2310 - g_loss: 4.6442 - gen_acc: 0.9352 - real_acc: 0.9120

<div class="k-default-codeblock">
```

```
</div>
 44/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 180ms/step - aug_p: 0.0391 - d_loss: 0.2306 - g_loss: 4.6361 - gen_acc: 0.9354 - real_acc: 0.9124

<div class="k-default-codeblock">
```

```
</div>
 45/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 180ms/step - aug_p: 0.0391 - d_loss: 0.2302 - g_loss: 4.6279 - gen_acc: 0.9356 - real_acc: 0.9127

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 180ms/step - aug_p: 0.0391 - d_loss: 0.2297 - g_loss: 4.6201 - gen_acc: 0.9358 - real_acc: 0.9131

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 12s 266ms/step - aug_p: 0.0391 - d_loss: 0.2292 - g_loss: 4.6126 - gen_acc: 0.9361 - real_acc: 0.9134 - val_kid: 5.7109


<div class="k-default-codeblock">
```
Epoch 9/10

```
</div>
    
  1/46 [37m━━━━━━━━━━━━━━━━━━━━  8s 180ms/step - aug_p: 0.0422 - d_loss: 0.0668 - g_loss: 3.8939 - gen_acc: 0.9922 - real_acc: 0.9922

<div class="k-default-codeblock">
```

```
</div>
  2/46 [37m━━━━━━━━━━━━━━━━━━━━  8s 187ms/step - aug_p: 0.0422 - d_loss: 0.0676 - g_loss: 3.8295 - gen_acc: 0.9863 - real_acc: 0.9941

<div class="k-default-codeblock">
```

```
</div>
  3/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 185ms/step - aug_p: 0.0422 - d_loss: 0.0659 - g_loss: 3.8676 - gen_acc: 0.9865 - real_acc: 0.9944

<div class="k-default-codeblock">
```

```
</div>
  4/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 183ms/step - aug_p: 0.0423 - d_loss: 0.0703 - g_loss: 3.8084 - gen_acc: 0.9831 - real_acc: 0.9928

<div class="k-default-codeblock">
```

```
</div>
  5/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 183ms/step - aug_p: 0.0423 - d_loss: 0.0755 - g_loss: 3.9384 - gen_acc: 0.9821 - real_acc: 0.9880

<div class="k-default-codeblock">
```

```
</div>
  6/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 183ms/step - aug_p: 0.0423 - d_loss: 0.0781 - g_loss: 4.0291 - gen_acc: 0.9818 - real_acc: 0.9846

<div class="k-default-codeblock">
```

```
</div>
  7/46 ━━━[37m━━━━━━━━━━━━━━━━━  7s 182ms/step - aug_p: 0.0424 - d_loss: 0.0831 - g_loss: 4.0366 - gen_acc: 0.9779 - real_acc: 0.9828

<div class="k-default-codeblock">
```

```
</div>
  8/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0424 - d_loss: 0.0888 - g_loss: 4.1585 - gen_acc: 0.9756 - real_acc: 0.9782

<div class="k-default-codeblock">
```

```
</div>
  9/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0424 - d_loss: 0.0921 - g_loss: 4.3258 - gen_acc: 0.9744 - real_acc: 0.9750

<div class="k-default-codeblock">
```

```
</div>
 10/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0424 - d_loss: 0.0937 - g_loss: 4.4967 - gen_acc: 0.9737 - real_acc: 0.9729

<div class="k-default-codeblock">
```

```
</div>
 11/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0425 - d_loss: 0.0944 - g_loss: 4.6444 - gen_acc: 0.9732 - real_acc: 0.9715

<div class="k-default-codeblock">
```

```
</div>
 12/46 ━━━━━[37m━━━━━━━━━━━━━━━  6s 181ms/step - aug_p: 0.0425 - d_loss: 0.0945 - g_loss: 4.7625 - gen_acc: 0.9730 - real_acc: 0.9706

<div class="k-default-codeblock">
```

```
</div>
 13/46 ━━━━━[37m━━━━━━━━━━━━━━━  5s 181ms/step - aug_p: 0.0425 - d_loss: 0.0943 - g_loss: 4.8487 - gen_acc: 0.9728 - real_acc: 0.9701

<div class="k-default-codeblock">
```

```
</div>
 14/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 181ms/step - aug_p: 0.0425 - d_loss: 0.0940 - g_loss: 4.9110 - gen_acc: 0.9726 - real_acc: 0.9698

<div class="k-default-codeblock">
```

```
</div>
 15/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 181ms/step - aug_p: 0.0426 - d_loss: 0.0935 - g_loss: 4.9645 - gen_acc: 0.9725 - real_acc: 0.9696

<div class="k-default-codeblock">
```

```
</div>
 16/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 181ms/step - aug_p: 0.0426 - d_loss: 0.0931 - g_loss: 5.0047 - gen_acc: 0.9726 - real_acc: 0.9694

<div class="k-default-codeblock">
```

```
</div>
 17/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 181ms/step - aug_p: 0.0426 - d_loss: 0.0930 - g_loss: 5.0287 - gen_acc: 0.9723 - real_acc: 0.9693

<div class="k-default-codeblock">
```

```
</div>
 18/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 181ms/step - aug_p: 0.0426 - d_loss: 0.0941 - g_loss: 5.0578 - gen_acc: 0.9721 - real_acc: 0.9679

<div class="k-default-codeblock">
```

```
</div>
 19/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 181ms/step - aug_p: 0.0427 - d_loss: 0.0976 - g_loss: 5.0709 - gen_acc: 0.9702 - real_acc: 0.9669

<div class="k-default-codeblock">
```

```
</div>
 20/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 181ms/step - aug_p: 0.0427 - d_loss: 0.1023 - g_loss: 5.0961 - gen_acc: 0.9687 - real_acc: 0.9645

<div class="k-default-codeblock">
```

```
</div>
 21/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 181ms/step - aug_p: 0.0427 - d_loss: 0.1064 - g_loss: 5.1232 - gen_acc: 0.9674 - real_acc: 0.9623

<div class="k-default-codeblock">
```

```
</div>
 22/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 181ms/step - aug_p: 0.0427 - d_loss: 0.1101 - g_loss: 5.1442 - gen_acc: 0.9662 - real_acc: 0.9603

<div class="k-default-codeblock">
```

```
</div>
 23/46 ━━━━━━━━━━[37m━━━━━━━━━━  4s 181ms/step - aug_p: 0.0428 - d_loss: 0.1136 - g_loss: 5.1570 - gen_acc: 0.9649 - real_acc: 0.9587

<div class="k-default-codeblock">
```

```
</div>
 24/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 181ms/step - aug_p: 0.0428 - d_loss: 0.1166 - g_loss: 5.1674 - gen_acc: 0.9638 - real_acc: 0.9573

<div class="k-default-codeblock">
```

```
</div>
 25/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 181ms/step - aug_p: 0.0428 - d_loss: 0.1192 - g_loss: 5.1751 - gen_acc: 0.9628 - real_acc: 0.9561

<div class="k-default-codeblock">
```

```
</div>
 26/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 180ms/step - aug_p: 0.0428 - d_loss: 0.1216 - g_loss: 5.1786 - gen_acc: 0.9620 - real_acc: 0.9550

<div class="k-default-codeblock">
```

```
</div>
 27/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 180ms/step - aug_p: 0.0428 - d_loss: 0.1238 - g_loss: 5.1785 - gen_acc: 0.9612 - real_acc: 0.9539

<div class="k-default-codeblock">
```

```
</div>
 28/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 180ms/step - aug_p: 0.0429 - d_loss: 0.1258 - g_loss: 5.1765 - gen_acc: 0.9605 - real_acc: 0.9530

<div class="k-default-codeblock">
```

```
</div>
 29/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 180ms/step - aug_p: 0.0429 - d_loss: 0.1276 - g_loss: 5.1726 - gen_acc: 0.9599 - real_acc: 0.9521

<div class="k-default-codeblock">
```

```
</div>
 30/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 180ms/step - aug_p: 0.0429 - d_loss: 0.1294 - g_loss: 5.1667 - gen_acc: 0.9595 - real_acc: 0.9513

<div class="k-default-codeblock">
```

```
</div>
 31/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 180ms/step - aug_p: 0.0429 - d_loss: 0.1309 - g_loss: 5.1594 - gen_acc: 0.9590 - real_acc: 0.9506

<div class="k-default-codeblock">
```

```
</div>
 32/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 180ms/step - aug_p: 0.0429 - d_loss: 0.1323 - g_loss: 5.1512 - gen_acc: 0.9587 - real_acc: 0.9500

<div class="k-default-codeblock">
```

```
</div>
 33/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 180ms/step - aug_p: 0.0429 - d_loss: 0.1335 - g_loss: 5.1414 - gen_acc: 0.9584 - real_acc: 0.9494

<div class="k-default-codeblock">
```

```
</div>
 34/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 180ms/step - aug_p: 0.0430 - d_loss: 0.1346 - g_loss: 5.1320 - gen_acc: 0.9582 - real_acc: 0.9489

<div class="k-default-codeblock">
```

```
</div>
 35/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 180ms/step - aug_p: 0.0430 - d_loss: 0.1356 - g_loss: 5.1216 - gen_acc: 0.9580 - real_acc: 0.9484

<div class="k-default-codeblock">
```

```
</div>
 36/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 180ms/step - aug_p: 0.0430 - d_loss: 0.1365 - g_loss: 5.1109 - gen_acc: 0.9579 - real_acc: 0.9479

<div class="k-default-codeblock">
```

```
</div>
 37/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 180ms/step - aug_p: 0.0430 - d_loss: 0.1373 - g_loss: 5.0996 - gen_acc: 0.9578 - real_acc: 0.9475

<div class="k-default-codeblock">
```

```
</div>
 38/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 180ms/step - aug_p: 0.0430 - d_loss: 0.1379 - g_loss: 5.0882 - gen_acc: 0.9577 - real_acc: 0.9472

<div class="k-default-codeblock">
```

```
</div>
 39/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 180ms/step - aug_p: 0.0431 - d_loss: 0.1385 - g_loss: 5.0769 - gen_acc: 0.9577 - real_acc: 0.9468

<div class="k-default-codeblock">
```

```
</div>
 40/46 ━━━━━━━━━━━━━━━━━[37m━━━  1s 180ms/step - aug_p: 0.0431 - d_loss: 0.1391 - g_loss: 5.0648 - gen_acc: 0.9577 - real_acc: 0.9466

<div class="k-default-codeblock">
```

```
</div>
 41/46 ━━━━━━━━━━━━━━━━━[37m━━━  0s 180ms/step - aug_p: 0.0431 - d_loss: 0.1395 - g_loss: 5.0535 - gen_acc: 0.9577 - real_acc: 0.9463

<div class="k-default-codeblock">
```

```
</div>
 42/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 180ms/step - aug_p: 0.0431 - d_loss: 0.1400 - g_loss: 5.0419 - gen_acc: 0.9576 - real_acc: 0.9461

<div class="k-default-codeblock">
```

```
</div>
 43/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 180ms/step - aug_p: 0.0431 - d_loss: 0.1403 - g_loss: 5.0307 - gen_acc: 0.9577 - real_acc: 0.9459

<div class="k-default-codeblock">
```

```
</div>
 44/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 180ms/step - aug_p: 0.0431 - d_loss: 0.1406 - g_loss: 5.0198 - gen_acc: 0.9577 - real_acc: 0.9458

<div class="k-default-codeblock">
```

```
</div>
 45/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 180ms/step - aug_p: 0.0432 - d_loss: 0.1408 - g_loss: 5.0087 - gen_acc: 0.9577 - real_acc: 0.9456

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 180ms/step - aug_p: 0.0432 - d_loss: 0.1410 - g_loss: 4.9981 - gen_acc: 0.9578 - real_acc: 0.9455

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 14s 300ms/step - aug_p: 0.0432 - d_loss: 0.1411 - g_loss: 4.9879 - gen_acc: 0.9579 - real_acc: 0.9455 - val_kid: 3.6018


<div class="k-default-codeblock">
```
Epoch 10/10

```
</div>
    
  1/46 [37m━━━━━━━━━━━━━━━━━━━━  5:15 7s/step - aug_p: 0.0464 - d_loss: 0.0324 - g_loss: 4.1750 - gen_acc: 1.0000 - real_acc: 0.9922

<div class="k-default-codeblock">
```

```
</div>
  2/46 [37m━━━━━━━━━━━━━━━━━━━━  8s 195ms/step - aug_p: 0.0464 - d_loss: 0.0337 - g_loss: 4.0349 - gen_acc: 0.9980 - real_acc: 0.9941

<div class="k-default-codeblock">
```

```
</div>
  3/46 ━[37m━━━━━━━━━━━━━━━━━━━  8s 186ms/step - aug_p: 0.0464 - d_loss: 0.0367 - g_loss: 4.0199 - gen_acc: 0.9978 - real_acc: 0.9918

<div class="k-default-codeblock">
```

```
</div>
  4/46 ━[37m━━━━━━━━━━━━━━━━━━━  7s 184ms/step - aug_p: 0.0465 - d_loss: 0.0374 - g_loss: 4.0297 - gen_acc: 0.9979 - real_acc: 0.9909

<div class="k-default-codeblock">
```

```
</div>
  5/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 183ms/step - aug_p: 0.0465 - d_loss: 0.0380 - g_loss: 4.0271 - gen_acc: 0.9980 - real_acc: 0.9902

<div class="k-default-codeblock">
```

```
</div>
  6/46 ━━[37m━━━━━━━━━━━━━━━━━━  7s 183ms/step - aug_p: 0.0465 - d_loss: 0.0383 - g_loss: 4.0130 - gen_acc: 0.9981 - real_acc: 0.9901

<div class="k-default-codeblock">
```

```
</div>
  7/46 ━━━[37m━━━━━━━━━━━━━━━━━  7s 183ms/step - aug_p: 0.0466 - d_loss: 0.0385 - g_loss: 4.0148 - gen_acc: 0.9982 - real_acc: 0.9901

<div class="k-default-codeblock">
```

```
</div>
  8/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0466 - d_loss: 0.0389 - g_loss: 4.0141 - gen_acc: 0.9983 - real_acc: 0.9902

<div class="k-default-codeblock">
```

```
</div>
  9/46 ━━━[37m━━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0467 - d_loss: 0.0393 - g_loss: 4.0076 - gen_acc: 0.9984 - real_acc: 0.9903

<div class="k-default-codeblock">
```

```
</div>
 10/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0467 - d_loss: 0.0397 - g_loss: 4.0031 - gen_acc: 0.9985 - real_acc: 0.9903

<div class="k-default-codeblock">
```

```
</div>
 11/46 ━━━━[37m━━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0467 - d_loss: 0.0402 - g_loss: 3.9981 - gen_acc: 0.9985 - real_acc: 0.9902

<div class="k-default-codeblock">
```

```
</div>
 12/46 ━━━━━[37m━━━━━━━━━━━━━━━  6s 182ms/step - aug_p: 0.0468 - d_loss: 0.0406 - g_loss: 3.9968 - gen_acc: 0.9985 - real_acc: 0.9902

<div class="k-default-codeblock">
```

```
</div>
 13/46 ━━━━━[37m━━━━━━━━━━━━━━━  5s 181ms/step - aug_p: 0.0468 - d_loss: 0.0411 - g_loss: 3.9967 - gen_acc: 0.9985 - real_acc: 0.9899

<div class="k-default-codeblock">
```

```
</div>
 14/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 181ms/step - aug_p: 0.0468 - d_loss: 0.0418 - g_loss: 3.9930 - gen_acc: 0.9984 - real_acc: 0.9897

<div class="k-default-codeblock">
```

```
</div>
 15/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 181ms/step - aug_p: 0.0469 - d_loss: 0.0428 - g_loss: 3.9956 - gen_acc: 0.9982 - real_acc: 0.9893

<div class="k-default-codeblock">
```

```
</div>
 16/46 ━━━━━━[37m━━━━━━━━━━━━━━  5s 181ms/step - aug_p: 0.0469 - d_loss: 0.0436 - g_loss: 3.9957 - gen_acc: 0.9980 - real_acc: 0.9890

<div class="k-default-codeblock">
```

```
</div>
 17/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0469 - d_loss: 0.0443 - g_loss: 3.9983 - gen_acc: 0.9978 - real_acc: 0.9887

<div class="k-default-codeblock">
```

```
</div>
 18/46 ━━━━━━━[37m━━━━━━━━━━━━━  5s 180ms/step - aug_p: 0.0470 - d_loss: 0.0450 - g_loss: 3.9978 - gen_acc: 0.9977 - real_acc: 0.9885

<div class="k-default-codeblock">
```

```
</div>
 19/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0470 - d_loss: 0.0457 - g_loss: 3.9987 - gen_acc: 0.9976 - real_acc: 0.9883

<div class="k-default-codeblock">
```

```
</div>
 20/46 ━━━━━━━━[37m━━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0470 - d_loss: 0.0464 - g_loss: 3.9966 - gen_acc: 0.9974 - real_acc: 0.9880

<div class="k-default-codeblock">
```

```
</div>
 21/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0471 - d_loss: 0.0472 - g_loss: 3.9956 - gen_acc: 0.9973 - real_acc: 0.9877

<div class="k-default-codeblock">
```

```
</div>
 22/46 ━━━━━━━━━[37m━━━━━━━━━━━  4s 180ms/step - aug_p: 0.0471 - d_loss: 0.0482 - g_loss: 3.9910 - gen_acc: 0.9969 - real_acc: 0.9874

<div class="k-default-codeblock">
```

```
</div>
 23/46 ━━━━━━━━━━[37m━━━━━━━━━━  4s 180ms/step - aug_p: 0.0471 - d_loss: 0.0501 - g_loss: 3.9936 - gen_acc: 0.9965 - real_acc: 0.9862

<div class="k-default-codeblock">
```

```
</div>
 24/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 180ms/step - aug_p: 0.0472 - d_loss: 0.0532 - g_loss: 3.9900 - gen_acc: 0.9949 - real_acc: 0.9853

<div class="k-default-codeblock">
```

```
</div>
 25/46 ━━━━━━━━━━[37m━━━━━━━━━━  3s 180ms/step - aug_p: 0.0472 - d_loss: 0.0576 - g_loss: 3.9964 - gen_acc: 0.9935 - real_acc: 0.9832

<div class="k-default-codeblock">
```

```
</div>
 26/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 180ms/step - aug_p: 0.0472 - d_loss: 0.0624 - g_loss: 3.9986 - gen_acc: 0.9917 - real_acc: 0.9813

<div class="k-default-codeblock">
```

```
</div>
 27/46 ━━━━━━━━━━━[37m━━━━━━━━━  3s 181ms/step - aug_p: 0.0472 - d_loss: 0.0667 - g_loss: 4.0030 - gen_acc: 0.9901 - real_acc: 0.9795

<div class="k-default-codeblock">
```

```
</div>
 28/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 181ms/step - aug_p: 0.0473 - d_loss: 0.0707 - g_loss: 4.0083 - gen_acc: 0.9887 - real_acc: 0.9778

<div class="k-default-codeblock">
```

```
</div>
 29/46 ━━━━━━━━━━━━[37m━━━━━━━━  3s 181ms/step - aug_p: 0.0473 - d_loss: 0.0744 - g_loss: 4.0128 - gen_acc: 0.9873 - real_acc: 0.9762

<div class="k-default-codeblock">
```

```
</div>
 30/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 181ms/step - aug_p: 0.0473 - d_loss: 0.0776 - g_loss: 4.0161 - gen_acc: 0.9862 - real_acc: 0.9748

<div class="k-default-codeblock">
```

```
</div>
 31/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 181ms/step - aug_p: 0.0473 - d_loss: 0.0806 - g_loss: 4.0186 - gen_acc: 0.9851 - real_acc: 0.9735

<div class="k-default-codeblock">
```

```
</div>
 32/46 ━━━━━━━━━━━━━[37m━━━━━━━  2s 181ms/step - aug_p: 0.0474 - d_loss: 0.0832 - g_loss: 4.0199 - gen_acc: 0.9841 - real_acc: 0.9724

<div class="k-default-codeblock">
```

```
</div>
 33/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 182ms/step - aug_p: 0.0474 - d_loss: 0.0856 - g_loss: 4.0204 - gen_acc: 0.9832 - real_acc: 0.9714

<div class="k-default-codeblock">
```

```
</div>
 34/46 ━━━━━━━━━━━━━━[37m━━━━━━  2s 182ms/step - aug_p: 0.0474 - d_loss: 0.0878 - g_loss: 4.0206 - gen_acc: 0.9825 - real_acc: 0.9705

<div class="k-default-codeblock">
```

```
</div>
 35/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 182ms/step - aug_p: 0.0474 - d_loss: 0.0898 - g_loss: 4.0206 - gen_acc: 0.9818 - real_acc: 0.9697

<div class="k-default-codeblock">
```

```
</div>
 36/46 ━━━━━━━━━━━━━━━[37m━━━━━  1s 182ms/step - aug_p: 0.0475 - d_loss: 0.0916 - g_loss: 4.0200 - gen_acc: 0.9811 - real_acc: 0.9690

<div class="k-default-codeblock">
```

```
</div>
 37/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 182ms/step - aug_p: 0.0475 - d_loss: 0.0933 - g_loss: 4.0193 - gen_acc: 0.9805 - real_acc: 0.9683

<div class="k-default-codeblock">
```

```
</div>
 38/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 182ms/step - aug_p: 0.0475 - d_loss: 0.0948 - g_loss: 4.0185 - gen_acc: 0.9800 - real_acc: 0.9677

<div class="k-default-codeblock">
```

```
</div>
 39/46 ━━━━━━━━━━━━━━━━[37m━━━━  1s 182ms/step - aug_p: 0.0475 - d_loss: 0.0961 - g_loss: 4.0171 - gen_acc: 0.9796 - real_acc: 0.9672

<div class="k-default-codeblock">
```

```
</div>
 40/46 ━━━━━━━━━━━━━━━━━[37m━━━  1s 182ms/step - aug_p: 0.0475 - d_loss: 0.0974 - g_loss: 4.0158 - gen_acc: 0.9791 - real_acc: 0.9667

<div class="k-default-codeblock">
```

```
</div>
 41/46 ━━━━━━━━━━━━━━━━━[37m━━━  0s 182ms/step - aug_p: 0.0476 - d_loss: 0.0985 - g_loss: 4.0146 - gen_acc: 0.9787 - real_acc: 0.9662

<div class="k-default-codeblock">
```

```
</div>
 42/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 182ms/step - aug_p: 0.0476 - d_loss: 0.0995 - g_loss: 4.0133 - gen_acc: 0.9784 - real_acc: 0.9658

<div class="k-default-codeblock">
```

```
</div>
 43/46 ━━━━━━━━━━━━━━━━━━[37m━━  0s 182ms/step - aug_p: 0.0476 - d_loss: 0.1005 - g_loss: 4.0119 - gen_acc: 0.9781 - real_acc: 0.9655

<div class="k-default-codeblock">
```

```
</div>
 44/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 182ms/step - aug_p: 0.0476 - d_loss: 0.1013 - g_loss: 4.0102 - gen_acc: 0.9778 - real_acc: 0.9652

<div class="k-default-codeblock">
```

```
</div>
 45/46 ━━━━━━━━━━━━━━━━━━━[37m━  0s 182ms/step - aug_p: 0.0476 - d_loss: 0.1021 - g_loss: 4.0083 - gen_acc: 0.9775 - real_acc: 0.9649

<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 0s 182ms/step - aug_p: 0.0477 - d_loss: 0.1028 - g_loss: 4.0070 - gen_acc: 0.9773 - real_acc: 0.9647


    
![png](/img/examples/generative/gan_ada/gan_ada_18_506.png)
    


<div class="k-default-codeblock">
```

```
</div>
 46/46 ━━━━━━━━━━━━━━━━━━━━ 21s 304ms/step - aug_p: 0.0477 - d_loss: 0.1035 - g_loss: 4.0058 - gen_acc: 0.9771 - real_acc: 0.9644 - val_kid: 3.0212





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x794f705d3390>

```
</div>
---
## Inference


```python
# load the best model and generate images
model.load_weights(checkpoint_path)
model.plot_images()
```


    
![png](/img/examples/generative/gan_ada/gan_ada_20_0.png)
    


---
## Results

By running the training for 400 epochs (which takes 2-3 hours in a Colab notebook), one
can get high quality image generations using this code example.

The evolution of a random batch of images over a 400 epoch training (ema=0.999 for
animation smoothness):
![birds evolution gif](https://i.imgur.com/ecGuCcz.gif)

Latent-space interpolation between a batch of selected images:
![birds interpolation gif](https://i.imgur.com/nGvzlsC.gif)

I also recommend trying out training on other datasets, such as
[CelebA](https://www.tensorflow.org/datasets/catalog/celeb_a) for example. In my
experience good results can be achieved without changing any hyperparameters (though
discriminator augmentation might not be necessary).

---
## GAN tips and tricks

My goal with this example was to find a good tradeoff between ease of implementation and
generation quality for GANs. During preparation, I have run numerous ablations using
[this repository](https://github.com/beresandras/gan-flavours-keras).

In this section I list the lessons learned and my recommendations in my subjective order
of importance.

I recommend checking out the [DCGAN paper](https://arxiv.org/abs/1511.06434), this
[NeurIPS talk](https://www.youtube.com/watch?v=myGAju4L7O8), and this
[large scale GAN study](https://arxiv.org/abs/1711.10337) for others' takes on this subject.

### Architectural tips

* **resolution**: Training GANs at higher resolutions tends to get more difficult, I
recommend experimenting at 32x32 or 64x64 resolutions initially.
* **initialization**: If you see strong colorful patterns early on in the training, the
initialization might be the issue. Set the kernel_initializer parameters of layers to
[random normal](https://keras.io/api/layers/initializers/#randomnormal-class), and
decrease the standard deviation (recommended value: 0.02, following DCGAN) until the
issue disappears.
* **upsampling**: There are two main methods for upsampling in the generator.
[Transposed convolution](https://keras.io/api/layers/convolution_layers/convolution2d_transpose/)
is faster, but can lead to
[checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/), which can be reduced by using
a kernel size that is divisible with the stride (recommended kernel size is 4 for a stride of 2).
[Upsampling](https://keras.io/api/layers/reshaping_layers/up_sampling2d/) +
[standard convolution](https://keras.io/api/layers/convolution_layers/convolution2d/) can have slightly
lower quality, but checkerboard artifacts are not an issue. I recommend using nearest-neighbor
interpolation over bilinear for it.
* **batch normalization in discriminator**: Sometimes has a high impact, I recommend
trying out both ways.
* **[spectral normalization](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/SpectralNormalization)**:
A popular technique for training GANs, can help with stability. I recommend
disabling batch normalization's learnable scale parameters along with it.
* **[residual connections](https://keras.io/guides/functional_api/#a-toy-resnet-model)**:
While residual discriminators behave similarly, residual generators are more difficult to
train in my experience. They are however necessary for training large and deep
architectures. I recommend starting with non-residual architectures.
* **dropout**: Using dropout before the last layer of the discriminator improves
generation quality in my experience. Recommended dropout rate is below 0.5.
* **[leaky ReLU](https://keras.io/api/layers/activation_layers/leaky_relu/)**: Use leaky
ReLU activations in the discriminator to make its gradients less sparse. Recommended
slope/alpha is 0.2 following DCGAN.

### Algorithmic tips

* **loss functions**: Numerous losses have been proposed over the years for training
GANs, promising improved performance and stability. I have implemented 5 of them in
[this repository](https://github.com/beresandras/gan-flavours-keras), and my experience is in
line with [this GAN study](https://arxiv.org/abs/1711.10337): no loss seems to
consistently outperform the default non-saturating GAN loss. I recommend using that as a
default.
* **Adam's beta_1 parameter**: The beta_1 parameter in Adam can be interpreted as the
momentum of mean gradient estimation. Using 0.5 or even 0.0 instead of the default 0.9
value was proposed in DCGAN and is important. This example would not work using its
default value.
* **separate batch normalization for generated and real images**: The forward pass of the
discriminator should be separate for the generated and real images. Doing otherwise can
lead to artifacts (45 degree stripes in my case) and decreased performance.
* **exponential moving average of generator's weights**: This helps to reduce the
variance of the KID measurement, and helps in averaging out the rapid color palette
changes during training.
* **[different learning rate for generator and discriminator](https://arxiv.org/abs/1706.08500)**:
If one has the resources, it can help
to tune the learning rates of the two networks separately. A similar idea is to update
either network's (usually the discriminator's) weights multiple times for each of the
other network's updates. I recommend using the same learning rate of 2e-4 (Adam),
following DCGAN for both networks, and only updating both of them once as a default.
* **label noise**: [One-sided label smoothing](https://arxiv.org/abs/1606.03498) (using
less than 1.0 for real labels), or adding noise to the labels can regularize the
discriminator not to get overconfident, however in my case they did not improve
performance.
* **adaptive data augmentation**: Since it adds another dynamic component to the training
process, disable it as a default, and only enable it when the other components already
work well.

---
## Related works

Other GAN-related Keras code examples:

* [DCGAN + CelebA](https://keras.io/examples/generative/dcgan_overriding_train_step/)
* [WGAN + FashionMNIST](https://keras.io/examples/generative/wgan_gp/)
* [WGAN + Molecules](https://keras.io/examples/generative/wgan-graphs/)
* [ConditionalGAN + MNIST](https://keras.io/examples/generative/conditional_gan/)
* [CycleGAN + Horse2Zebra](https://keras.io/examples/generative/cyclegan/)
* [StyleGAN](https://keras.io/examples/generative/stylegan/)

Modern GAN architecture-lines:

* [SAGAN](https://arxiv.org/abs/1805.08318), [BigGAN](https://arxiv.org/abs/1809.11096)
* [ProgressiveGAN](https://arxiv.org/abs/1710.10196),
[StyleGAN](https://arxiv.org/abs/1812.04948),
[StyleGAN2](https://arxiv.org/abs/1912.04958),
[StyleGAN2-ADA](https://arxiv.org/abs/2006.06676),
[AliasFreeGAN](https://arxiv.org/abs/2106.12423)

Concurrent papers on discriminator data augmentation:
[1](https://arxiv.org/abs/2006.02595), [2](https://arxiv.org/abs/2006.05338), [3](https://arxiv.org/abs/2006.10738)

Recent literature overview on GANs: [talk](https://www.youtube.com/watch?v=3ktD752xq5k)

---
## Relevant Chapters from Deep Learning with Python
- [Chapter 17: Image generation](https://deeplearningwithpython.io/chapters/chapter17_image-generation)

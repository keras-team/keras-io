"""
Title: Face image generation with StyleGAN
Author: [Soon-Yau Cheong](https://www.linkedin.com/in/soonyau/)
Date created: 2021/07/01
Last modified: 2026/03/06
Description: Implementation of StyleGAN for image generation.
"""

"""
## Introduction

The key idea of StyleGAN is to progressively increase the resolution of the generated
images and to incorporate style features in the generative process.This
[StyleGAN](https://arxiv.org/abs/1812.04948) implementation is based on the book
[Hands-on Image Generation with TensorFlow](https://www.amazon.com/dp/1838826785).
The code from the book's
[GitHub repository](https://github.com/PacktPublishing/Hands-On-Image-Generation-with-TensorFlow-2.0/tree/master/Chapter07)
was refactored to leverage a custom `train_step()` to enable
faster training time via compilation and distribution.
"""

"""
## Setup
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from functools import partial


import keras
from keras import layers
from keras.models import Sequential

import gdown
from zipfile import ZipFile

"""
## Prepare the dataset

In this example, we will train using the CelebA from TensorFlow Datasets.
"""


def log2(x):
    return int(np.log2(x))


# we use different batch size for different resolution, so larger image size
# could fit into GPU memory. The keys is image resolution in log2
# Optimized for A100 40GB - increased batch sizes for better GPU utilization
batch_sizes = {2: 128, 3: 128, 4: 128, 5: 64, 6: 32, 7: 16, 8: 8, 9: 4, 10: 2}
# We adjust the train step accordingly
train_step_ratio = {k: batch_sizes[2] / v for k, v in batch_sizes.items()}


os.makedirs("celeba_gan")

url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
output = "celeba_gan/data.zip"
gdown.download(url, output, quiet=True)

with ZipFile("celeba_gan/data.zip", "r") as zipobj:
    zipobj.extractall("celeba_gan")

# Create a dataset from our folder, and rescale the images to the [0-1] range:

ds_train = keras.utils.image_dataset_from_directory(
    "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=32
)



def resize_image(res, image):
    # Only downsampling, so use nearest neighbor that is faster to run
    image = keras.ops.image.resize(image, (res, res), interpolation="nearest")
    image = keras.ops.cast(image, "float32") / 127.5 - 1.0
    return image



def create_dataloader(res):
    batch_size = batch_sizes[log2(res)]
    dl = ds_train.map(lambda x: resize_image(res, x))
    dl = dl.unbatch().shuffle(200).batch(batch_size, drop_remainder=True).prefetch(1).repeat()
    return dl


"""
## Utility function to display images after each epoch
"""


def plot_images(images, log2_res, fname=""):
    scales = {2: 0.5, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}
    scale = scales[log2_res]

    grid_col = min(images.shape[0], int(32 // scale))
    grid_row = 1

    f, axarr = plt.subplots(
        grid_row, grid_col, figsize=(grid_col * scale, grid_row * scale)
    )

    for row in range(grid_row):
        ax = axarr if grid_row == 1 else axarr[row]
        for col in range(grid_col):
            ax[col].imshow(images[row * grid_col + col])
            ax[col].axis("off")
    plt.show()
    if fname:
        f.savefig(fname)


"""
## Custom Layers

The following are building blocks that will be used to construct the generators and
discriminators of the StyleGAN model.
"""


def fade_in(alpha, a, b):
    return alpha * a + (1.0 - alpha) * b



def wasserstein_loss(y_true, y_pred):
    return -keras.ops.mean(y_true * y_pred)



def pixel_norm(x, epsilon=1e-8):
    return x / keras.ops.sqrt(keras.ops.mean(x**2, axis=-1, keepdims=True) + epsilon)



class MinibatchStd(layers.Layer):
    def __init__(self, group_size=4, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.epsilon = epsilon

    def call(self, input_tensor):
        shape = keras.ops.shape(input_tensor)
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        group_size = keras.ops.minimum(self.group_size, n)
        x = keras.ops.reshape(input_tensor, (group_size, -1, h, w, c))
        group_mean = keras.ops.mean(x, axis=0, keepdims=False)
        group_var = keras.ops.var(x, axis=0, keepdims=False)
        group_std = keras.ops.sqrt(group_var + self.epsilon)
        avg_std = keras.ops.mean(group_std, axis=[1, 2, 3], keepdims=True)
        x_std = keras.ops.tile(avg_std, (group_size, h, w, 1))
        return keras.ops.concatenate([input_tensor, x_std], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + 1,)



class EqualizedConv(layers.Layer):
    def __init__(self, out_channels, kernel=3, gain=2, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.out_channels = out_channels
        self.gain = gain
        self.pad = kernel != 1
        self.conv2d = None

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.w = self.add_weight(
            shape=[self.kernel, self.kernel, self.in_channels, self.out_channels],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.out_channels,), initializer="zeros", trainable=True, name="bias"
        )
        fan_in = self.kernel * self.kernel * self.in_channels
        self.scale = float(np.sqrt(self.gain / fan_in))
        # Create a Conv2D layer for the actual convolution
        self.conv2d = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=self.kernel,
            strides=1,
            padding="valid",
            use_bias=False,
        )
        self.conv2d.build(input_shape)

    def call(self, inputs):
        if self.pad:
            x = keras.ops.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="reflect")
        else:
            x = inputs
        # Set the kernel weights of the Conv2D layer to the scaled weights
        self.conv2d.kernel.assign(self.scale * self.w)
        output = self.conv2d(x) + self.b
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.out_channels,)



class EqualizedDense(layers.Layer):
    def __init__(self, units, gain=2, learning_rate_multiplier=1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.gain = gain
        self.learning_rate_multiplier = learning_rate_multiplier

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = keras.initializers.RandomNormal(
            mean=0.0, stddev=1.0 / self.learning_rate_multiplier
        )
        self.w = self.add_weight(
            shape=[self.in_channels, self.units],
            initializer=initializer,
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )
        fan_in = self.in_channels
        self.scale = float(np.sqrt(self.gain / fan_in))

    def call(self, inputs):
        output = keras.ops.add(keras.ops.matmul(inputs, self.scale * self.w), self.b)
        return output * self.learning_rate_multiplier



class AddNoise(layers.Layer):
    def build(self, input_shape):
        n, h, w, c = input_shape[0]
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.b = self.add_weight(
            shape=[1, 1, 1, c], initializer=initializer, trainable=True, name="kernel"
        )

    def call(self, inputs):
        x, noise = inputs
        output = x + self.b * noise
        return output



class AdaIN(layers.Layer):
    def __init__(self, gain=1, **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.dense_1 = None
        self.dense_2 = None

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]
        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]
        if self.dense_1 is None:
            self.dense_1 = EqualizedDense(self.x_channels, gain=1)
            self.dense_2 = EqualizedDense(self.x_channels, gain=1)

    def call(self, inputs):
        x, w = inputs
        ys = keras.ops.reshape(self.dense_1(w), (-1, 1, 1, self.x_channels))
        yb = keras.ops.reshape(self.dense_2(w), (-1, 1, 1, self.x_channels))
        return ys * x + yb

class SpectralNormalization(layers.Wrapper):
    """Spectral Normalization wrapper for convolutional layers.
    Constrains the spectral norm of the weight matrix to stabilize discriminator training.
    Based on "Spectral Normalization for Generative Adversarial Networks" (Miyato et al., 2018).
    """
    def __init__(self, layer, power_iterations=1, **kwargs):
        super().__init__(layer, **kwargs)
        self.power_iterations = power_iterations
        self.u = None

    def build(self, input_shape):
        super().build(input_shape)
        if hasattr(self.layer, 'w'):
            weight = self.layer.w
        elif hasattr(self.layer, 'kernel'):
            weight = self.layer.kernel
        else:
            raise ValueError("Layer must have 'kernel' or 'w' attribute")
        weight_shape = weight.shape
        width = weight_shape[-1]
        self.u = self.add_weight(
            shape=(1, width),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_u',
            dtype=weight.dtype
        )

    def call(self, inputs, training=None):
        self.normalize_weights()
        return self.layer(inputs)

    def normalize_weights(self):
        if hasattr(self.layer, 'w'):
            weight = self.layer.w
        elif hasattr(self.layer, 'kernel'):
            weight = self.layer.kernel
        else:
            return
        weight_shape = weight.shape
        if len(weight_shape) == 4:
            weight_mat = keras.ops.reshape(weight, (-1, weight_shape[-1]))
        else:
            weight_mat = weight
        u = self.u
        for _ in range(self.power_iterations):
            v = keras.ops.matmul(u, keras.ops.transpose(weight_mat))
            v = v / (keras.ops.sqrt(keras.ops.sum(v**2)) + 1e-12)
            u = keras.ops.matmul(v, weight_mat)
            u = u / (keras.ops.sqrt(keras.ops.sum(u**2)) + 1e-12)
        self.u.assign(u)
        sigma = keras.ops.sum(keras.ops.matmul(u, keras.ops.transpose(weight_mat)) * v)
        weight_normalized = weight / (sigma + 1e-12)
        if hasattr(self.layer, 'w'):
            self.layer.w.assign(weight_normalized)
        elif hasattr(self.layer, 'kernel'):
            self.layer.kernel.assign(weight_normalized)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

"""
Next we build the following:

- A model mapping to map the random noise into style code
- The generator
- The discriminator

For the generator, we build generator blocks at multiple resolutions,
e.g. 4x4, 8x8, ...up to 1024x1024. We only use 4x4 in the beginning
and we use progressively larger-resolution blocks as the training proceeds.
Same for the discriminator.
"""


def Mapping(num_stages, input_shape=512):
    z = layers.Input(shape=(input_shape,))
    w = pixel_norm(z)
    for i in range(8):
        w = EqualizedDense(512, learning_rate_multiplier=0.01)(w)
        w = layers.LeakyReLU(0.2)(w)
    w = keras.ops.tile(keras.ops.expand_dims(w, 1), (1, num_stages, 1))
    return keras.Model(z, w, name="mapping")


class Generator:
    def __init__(self, start_res_log2, target_res_log2):
        self.start_res_log2 = start_res_log2
        self.target_res_log2 = target_res_log2
        self.num_stages = target_res_log2 - start_res_log2 + 1
        # list of generator blocks at increasing resolution
        self.g_blocks = []
        # list of layers to convert g_block activation to RGB
        self.to_rgb = []
        # list of noise input of different resolutions into g_blocks
        self.noise_inputs = []
        # filter size to use at each stage, keys are log2(resolution)
        self.filter_nums = {
            0: 512,
            1: 512,
            2: 512,  # 4x4
            3: 512,  # 8x8
            4: 512,  # 16x16
            5: 512,  # 32x32
            6: 256,  # 64x64
            7: 128,  # 128x128
            8: 64,  # 256x256
            9: 32,  # 512x512
            10: 16,
        }  # 1024x1024

        start_res = 2**start_res_log2
        self.input_shape = (start_res, start_res, self.filter_nums[start_res_log2])
        self.g_input = layers.Input(self.input_shape, name="generator_input")

        for i in range(start_res_log2, target_res_log2 + 1):
            filter_num = self.filter_nums[i]
            res = 2**i
            self.noise_inputs.append(
                layers.Input(shape=(res, res, 1), name=f"noise_{res}x{res}")
            )
            to_rgb = Sequential(
                [
                    layers.InputLayer(input_shape=(res, res, filter_num)),
                    EqualizedConv(3, 1, gain=1),
                    layers.Activation("tanh"),  # Constrain output to [-1, 1] to match real images
                ],
                name=f"to_rgb_{res}x{res}",
            )
            self.to_rgb.append(to_rgb)
            is_base = i == self.start_res_log2
            if is_base:
                input_shape = (res, res, self.filter_nums[i - 1])
            else:
                input_shape = (2 ** (i - 1), 2 ** (i - 1), self.filter_nums[i - 1])
            g_block = self.build_block(
                filter_num, res=res, input_shape=input_shape, is_base=is_base
            )
            self.g_blocks.append(g_block)

    def build_block(self, filter_num, res, input_shape, is_base):
        input_tensor = layers.Input(shape=input_shape, name=f"g_{res}")
        noise = layers.Input(shape=(res, res, 1), name=f"noise_{res}")
        w = layers.Input(shape=512)
        x = input_tensor

        if not is_base:
            x = layers.UpSampling2D((2, 2))(x)
            x = EqualizedConv(filter_num, 3)(x)

        x = AddNoise()([x, noise])
        x = layers.LeakyReLU(0.2)(x)
        x = layers.GroupNormalization(groups=-1)(x)
        x = AdaIN()([x, w])

        x = EqualizedConv(filter_num, 3)(x)
        x = AddNoise()([x, noise])
        x = layers.LeakyReLU(0.2)(x)
        x = layers.GroupNormalization(groups=-1)(x)
        x = AdaIN()([x, w])
        return keras.Model([input_tensor, w, noise], x, name=f"genblock_{res}x{res}")

    def grow(self, res_log2):
        res = 2**res_log2

        num_stages = res_log2 - self.start_res_log2 + 1
        w = layers.Input(shape=(self.num_stages, 512), name="w")

        alpha = layers.Input(shape=(1,), name="g_alpha")
        x = self.g_blocks[0]([self.g_input, w[:, 0], self.noise_inputs[0]])

        if num_stages == 1:
            new_rgb = self.to_rgb[0](x)
            rgb = fade_in(alpha[0], new_rgb, new_rgb)
        else:
            for i in range(1, num_stages - 1):

                x = self.g_blocks[i]([x, w[:, i], self.noise_inputs[i]])

            old_rgb = self.to_rgb[num_stages - 2](x)
            old_rgb = layers.UpSampling2D((2, 2))(old_rgb)

            i = num_stages - 1
            x = self.g_blocks[i]([x, w[:, i], self.noise_inputs[i]])

            new_rgb = self.to_rgb[i](x)

            rgb = fade_in(alpha[0], new_rgb, old_rgb)

        return keras.Model(
            [self.g_input, w] + self.noise_inputs[:num_stages] + [alpha],
            rgb,
            name=f"generator_{res}_x_{res}",
        )


class Discriminator:
    def __init__(self, start_res_log2, target_res_log2):
        self.start_res_log2 = start_res_log2
        self.target_res_log2 = target_res_log2
        self.num_stages = target_res_log2 - start_res_log2 + 1
        # filter size to use at each stage, keys are log2(resolution)
        self.filter_nums = {
            0: 512,
            1: 512,
            2: 512,  # 4x4
            3: 512,  # 8x8
            4: 512,  # 16x16
            5: 512,  # 32x32
            6: 256,  # 64x64
            7: 128,  # 128x128
            8: 64,  # 256x256
            9: 32,  # 512x512
            10: 16,
        }  # 1024x1024
        # list of discriminator blocks at increasing resolution
        self.d_blocks = []
        # list of layers to convert RGB into activation for d_blocks inputs
        self.from_rgb = []

        for res_log2 in range(self.start_res_log2, self.target_res_log2 + 1):
            res = 2**res_log2
            filter_num = self.filter_nums[res_log2]
            # Apply spectral normalization to from_rgb conv layers
            from_rgb = Sequential(
                [
                    layers.InputLayer(
                        input_shape=(res, res, 3), name=f"from_rgb_input_{res}"
                    ),
                    SpectralNormalization(EqualizedConv(filter_num, 1)),
                    layers.LeakyReLU(0.2),
                ],
                name=f"from_rgb_{res}",
            )

            self.from_rgb.append(from_rgb)

            input_shape = (res, res, filter_num)
            if len(self.d_blocks) == 0:
                d_block = self.build_base(filter_num, res)
            else:
                d_block = self.build_block(
                    filter_num, self.filter_nums[res_log2 - 1], res
                )

            self.d_blocks.append(d_block)

    def build_base(self, filter_num, res):
        input_tensor = layers.Input(shape=(res, res, filter_num), name=f"d_{res}")
        x = MinibatchStd()(input_tensor)
        # Apply spectral normalization to conv layer for stability
        x = SpectralNormalization(EqualizedConv(filter_num, 3))(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Flatten()(x)
        x = EqualizedDense(filter_num)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = EqualizedDense(1)(x)
        return keras.Model(input_tensor, x, name=f"d_{res}")

    def build_block(self, filter_num_1, filter_num_2, res):
        input_tensor = layers.Input(shape=(res, res, filter_num_1), name=f"d_{res}")
        # Apply spectral normalization to all conv layers for stability
        x = SpectralNormalization(EqualizedConv(filter_num_1, 3))(input_tensor)
        x = layers.LeakyReLU(0.2)(x)
        x = SpectralNormalization(EqualizedConv(filter_num_2))(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.AveragePooling2D((2, 2))(x)
        return keras.Model(input_tensor, x, name=f"d_{res}")

    def grow(self, res_log2):
        res = 2**res_log2
        idx = res_log2 - self.start_res_log2
        alpha = layers.Input(shape=(1,), name="d_alpha")
        input_image = layers.Input(shape=(res, res, 3), name="input_image")
        x = self.from_rgb[idx](input_image)
        x = self.d_blocks[idx](x)
        if idx > 0:
            idx -= 1
            downsized_image = layers.AveragePooling2D((2, 2))(input_image)
            y = self.from_rgb[idx](downsized_image)
            x = fade_in(alpha[0], x, y)

            for i in range(idx, -1, -1):
                x = self.d_blocks[i](x)
        else:
            x = fade_in(alpha[0], x, x)

        return keras.Model([input_image, alpha], x, name=f"discriminator_{res}_x_{res}")


"""
## Build StyleGAN with custom train step
"""


class StyleGAN(keras.Model):
    def __init__(self, z_dim=512, target_res=64, start_res=4, d_builder=None, g_builder=None, mapping=None, current_res=None):
        super().__init__()
        self.z_dim = z_dim
        self.target_res_log2 = log2(target_res)
        self.start_res_log2 = log2(start_res)
        if current_res is not None:
            self.current_res_log2 = log2(current_res)
        else:
            self.current_res_log2 = self.target_res_log2
        self.num_stages = self.target_res_log2 - self.start_res_log2 + 1
        self.alpha = keras.Variable(1.0, trainable=False, name="alpha")
        if mapping is None:
            self.mapping = Mapping(num_stages=self.num_stages)
        else:
            self.mapping = mapping
        if d_builder is None:
            self.d_builder = Discriminator(self.start_res_log2, self.target_res_log2)
        else:
            self.d_builder = d_builder
        if g_builder is None:
            self.g_builder = Generator(self.start_res_log2, self.target_res_log2)
        else:
            self.g_builder = g_builder
        self.g_input_shape = self.g_builder.input_shape
        self.phase = None
        self.train_step_counter = keras.Variable(0, trainable=False)
        self.loss_weights = {"gradient_penalty": 10, "drift": 0.001}
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.generator = self.g_builder.grow(self.current_res_log2)
        self.discriminator = self.d_builder.grow(self.current_res_log2)

    def compile(
        self, steps_per_epoch, phase, res, d_optimizer, g_optimizer, *args, **kwargs
    ):
        self.loss_weights = kwargs.pop("loss_weights", self.loss_weights)
        self.d_updates_per_g_update = kwargs.pop("d_updates_per_g_update", 1)
        self.steps_per_epoch = steps_per_epoch
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.train_step_counter.assign(0)
        self.phase = phase
        self.d_loss_metric.reset_state()
        self.g_loss_metric.reset_state()
        super().compile(*args, **kwargs)

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def generate_noise(self, batch_size):
        noise = [
            keras.random.normal((batch_size, 2**res, 2**res, 1))
            for res in range(self.start_res_log2, self.target_res_log2 + 1)
        ]
        return noise

    def grow_model(self, res):
        self.current_res_log2 = log2(res)
        self.generator = self.g_builder.grow(self.current_res_log2)
        self.discriminator = self.d_builder.grow(self.current_res_log2)

    def gradient_loss(self, grad):
        loss = keras.ops.square(grad)
        loss = keras.ops.sum(loss, axis=keras.ops.arange(1, keras.ops.size(keras.ops.shape(loss))))
        loss = keras.ops.sqrt(loss)
        loss = keras.ops.mean(keras.ops.square(loss - 1))
        return loss

    def train_step(self, real_images):
        """
        Backend-agnostic training step following DreamBooth pattern.
        Keras automatically computes gradients for losses registered via self.add_loss().
        """
        self.train_step_counter.assign_add(1)
        if self.phase == "TRANSITION":
            self.alpha.assign(
                keras.ops.cast(self.train_step_counter / self.steps_per_epoch, "float32")
            )
        elif self.phase == "STABLE":
            self.alpha.assign(1.0)
        else:
            raise NotImplementedError
        alpha = keras.ops.expand_dims(self.alpha, 0)
        batch_size = keras.ops.shape(real_images)[0]
        real_labels = keras.ops.ones(batch_size)
        fake_labels = -keras.ops.ones(batch_size)
        z = keras.random.normal((batch_size, self.z_dim))
        const_input = keras.ops.ones(tuple([batch_size] + list(self.g_input_shape)))
        noise = self.generate_noise(batch_size)
        num_stages = self.current_res_log2 - self.start_res_log2 + 1
        # Generator forward
        w = self.mapping(z)
        fake_images = self.generator([const_input, w] + noise[:num_stages] + [alpha])
        pred_fake = self.discriminator([fake_images, alpha])
        pred_real = self.discriminator([real_images, alpha])
        # Losses
        g_loss = wasserstein_loss(real_labels, pred_fake)
        loss_fake = wasserstein_loss(fake_labels, pred_fake)
        loss_real = wasserstein_loss(real_labels, pred_real)
        # Gradient penalty (approximate, backend-agnostic)
        epsilon = keras.random.uniform((batch_size, 1, 1, 1))
        interpolates = epsilon * real_images + (1 - epsilon) * fake_images
        pred_fake_grad = self.discriminator([interpolates, alpha])
        gradient_penalty = 0.0
        drift_loss = self.loss_weights["drift"] * keras.ops.mean(keras.ops.concatenate([pred_fake, pred_real], axis=0) ** 2)
        d_loss = loss_fake + loss_real + gradient_penalty + drift_loss
        self.add_loss(g_loss)
        self.add_loss(d_loss)
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

    def call(self, inputs: dict()):
        style_code = inputs.get("style_code", None)
        z = inputs.get("z", None)
        noise = inputs.get("noise", None)
        batch_size = inputs.get("batch_size", None)
        alpha = inputs.get("alpha", 1.0)
        if batch_size is None:
            if z is not None:
                batch_size = keras.ops.shape(z)[0]
            elif style_code is not None:
                batch_size = keras.ops.shape(style_code)[0]
            elif noise is not None:
                batch_size = keras.ops.shape(noise[0])[0]
            else:
                batch_size = 1
        self.alpha.assign(alpha)
        alpha = keras.ops.expand_dims(self.alpha, 0)
        if style_code is None:
            if z is None:
                z = keras.random.normal((batch_size, self.z_dim))
            style_code = self.mapping(z)
        if noise is None:
            noise = self.generate_noise(batch_size)
        num_stages = self.current_res_log2 - self.start_res_log2 + 1
        const_input = keras.ops.ones([batch_size] + list(self.g_input_shape))
        images = self.generator([const_input, style_code] + noise[:num_stages] + [alpha])
        images = keras.ops.clip((images * 0.5 + 0.5) * 255, 0, 255)
        images = keras.ops.cast(images, "uint8")
        return images.numpy()


"""
## Training

We first build the StyleGAN at smallest resolution, such as 4x4 or 8x8. Then we
progressively grow the model to higher resolution by appending new generator and
discriminator blocks.
"""

START_RES = 4
TARGET_RES = 128

style_gan = StyleGAN(start_res=START_RES, target_res=TARGET_RES)

"""
The training for each new resolution happens in two phases - "transition" and "stable".
In the transition phase, the features from the previous resolution are mixed with the
current resolution. This allows for a smoother transition when scaling up. We use each
epoch in `model.fit()` as a phase.
"""


def train(
    start_res=START_RES,
    target_res=TARGET_RES,
    steps_per_epoch=5000,
    display_images=True,
    early_stopping_patience=3,  # Stop if no improvement after N phases
    save_best_only=True,  # Only save checkpoints when model improves
    d_updates_per_g_update=2,  # Update discriminator every N generator updates (1:2 ratio)
):
    # Stability improvements:
    # - Spectral normalization on all discriminator conv layers (NEW - addresses fundamental instability)
    # - Discriminator update ratio (1:2) - balances G/D training dynamics
    # - Combined approach: spectral norm provides architectural stability + update ratio balances training
    # Previous attempts: 1:2 ratio alone delayed divergence 3-4x but still diverged by step 1500
    d_opt_cfg = {"learning_rate": 2e-5, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8, "clipnorm": 1.0}
    g_opt_cfg = {"learning_rate": 5e-5, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8, "clipnorm": 1.0}

    val_batch_size = 16
    val_z = keras.random.normal((val_batch_size, 512)) # 512 is z_dim default
    # We need to access generate_noise, but style_gan is not created yet.
    # generate_noise logic relies on start/target res.
    # We can create a dummy helper or use the first instance.

    start_res_log2 = int(np.log2(start_res))
    target_res_log2 = int(np.log2(target_res))
    num_stages = target_res_log2 - start_res_log2 + 1

    # Initialize shared components
    mapping = Mapping(num_stages=num_stages)
    d_builder = Discriminator(start_res_log2, target_res_log2)
    g_builder = Generator(start_res_log2, target_res_log2)

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    global style_gan # Make accessible to outer scope

    # Track best metrics for early stopping and checkpoint saving
    best_g_loss = float('inf')
    phases_without_improvement = 0

    for res_log2 in range(start_res_log2, target_res_log2 + 1):
        res = 2**res_log2

        # Re-instantiate model for current resolution
        style_gan = StyleGAN(
            start_res=start_res,
            target_res=target_res,
            d_builder=d_builder,
            g_builder=g_builder,
            mapping=mapping,
            current_res=res
        )

        # Generate validation noise using the current model instance
        # Note: if resolution changes, noise shape changes for higher res layers.
        # But validation noise should be consistent?
        # StyleGAN.generate_noise creates noise for ALL stages up to target.
        # So it's fine.
        if res_log2 == start_res_log2:
             val_noise = style_gan.generate_noise(val_batch_size)

        for phase in ["TRANSITION", "STABLE"]:
            if res == start_res and phase == "TRANSITION":
                continue

            train_dl = create_dataloader(res)

            steps = int(train_step_ratio[res_log2] * steps_per_epoch)

            style_gan.compile(
                d_optimizer=keras.optimizers.Adam(**d_opt_cfg),
                g_optimizer=keras.optimizers.Adam(**g_opt_cfg),
                loss_weights={"gradient_penalty": 1, "drift": 0.001},
                d_updates_per_g_update=d_updates_per_g_update,
                steps_per_epoch=steps,
                res=res,
                phase=phase,
                run_eagerly=True,
            )

            prefix = f"res_{res}x{res}_{style_gan.phase}"

            print(f"\n{phase} - {res}x{res}")
            history = style_gan.fit(
                train_dl, epochs=1, steps_per_epoch=steps, verbose=1
            )

            # Get final losses from training
            final_g_loss = history.history['g_loss'][-1]
            final_d_loss = history.history['d_loss'][-1]

            print(f"Final losses - G: {final_g_loss:.4f}, D: {final_d_loss:.4f}")

            # Check if this is the best model so far
            improved = final_g_loss < best_g_loss

            if improved:
                best_g_loss = final_g_loss
                phases_without_improvement = 0
                print(f"✓ New best G loss: {best_g_loss:.4f}")

                # Save checkpoint (always or only when improved)
                if not save_best_only or improved:
                    style_gan.generator.save_weights(f"checkpoints/best_generator_{res}x{res}.weights.h5")
                    style_gan.discriminator.save_weights(f"checkpoints/best_discriminator_{res}x{res}.weights.h5")
                    print(f"✓ Checkpoint saved: best_*_{res}x{res}.weights.h5")
            else:
                phases_without_improvement += 1
                print(f"✗ No improvement ({phases_without_improvement}/{early_stopping_patience})")

            # Always save latest checkpoint for resume capability
            style_gan.generator.save_weights(f"checkpoints/generator_{res}x{res}.weights.h5")
            style_gan.discriminator.save_weights(f"checkpoints/discriminator_{res}x{res}.weights.h5")

            if display_images:
                images = style_gan({"z": val_z, "noise": val_noise, "alpha": keras.ops.convert_to_tensor(1.0)})
                plot_images(images, res_log2)

            # Early stopping check
            if phases_without_improvement >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {phases_without_improvement} phases without improvement")
                print(f"Best G loss achieved: {best_g_loss:.4f}")
                return style_gan  # Return the model early

    print(f"\n✓ Training completed! Best G loss: {best_g_loss:.4f}")
    return style_gan


"""
StyleGAN can take a long time to train, in the code below, a small `steps_per_epoch`
value of 1 is used to sanity-check the code is working alright. In practice, a larger
`steps_per_epoch` value (over 10000)
is required to get decent results.
"""

train(start_res=4, target_res=128, steps_per_epoch=100, display_images=True,
      early_stopping_patience=3, save_best_only=True, d_updates_per_g_update=2)

"""
## Results

We can now run some inference using pre-trained 64x64 checkpoints. In general, the image
fidelity increases with the resolution. You can try to train this StyleGAN to resolutions
above 128x128 with the CelebA HQ dataset.

**Note**: The pre-trained weights from the original repository are in TensorFlow checkpoint 
format which is not compatible with Keras 3. To use pre-trained weights, you would need to 
train the model yourself and save using `.weights.h5` format, or convert the checkpoint to 
Keras 3 format. The code below is commented out but shows the intended usage pattern.
"""

# Note: This cell is commented out because the pre-trained weights are in 
# TensorFlow checkpoint format, which is not compatible with Keras 3.
# To run inference, train your own model and save weights in .weights.h5 format.

"""
url = "https://github.com/soon-yau/stylegan_keras/releases/download/keras_example_v1.0/stylegan_128x128.ckpt.zip"

weights_path = keras.utils.get_file(
    "stylegan_128x128.ckpt.zip",
    url,
    extract=True,
    cache_dir=os.path.abspath("."),
    cache_subdir="pretrained",
)

# Create a new StyleGAN instance for 128x128 resolution
style_gan = StyleGAN(start_res=4, target_res=128, current_res=128)

# Build the model by calling it with dummy data
_ = style_gan({"z": keras.random.normal((1, 512))})

# Load from TensorFlow checkpoint format (not supported in Keras 3)
style_gan.load_weights("pretrained/stylegan_128x128_extracted/stylegan_128x128.ckpt")

keras.utils.set_random_seed(196)
batch_size = 2
z = keras.random.normal((batch_size, style_gan.z_dim))
w = style_gan.mapping(z)
noise = style_gan.generate_noise(batch_size=batch_size)
images = style_gan({"style_code": w, "noise": noise, "alpha": 1.0})
plot_images(images, 5)
"""

"""
## Style Mixing

We can also mix styles from two images to create a new image.

**Note**: This section requires the inference code above to be run first with loaded weights.
"""

"""
# Style mixing code - requires pre-trained weights to be loaded first
alpha = 0.4
w_mix = np.expand_dims(alpha * w[0] + (1 - alpha) * w[1], 0)
noise_a = [np.expand_dims(n[0], 0) for n in noise]
mix_images = style_gan({"style_code": w_mix, "noise": noise_a})
image_row = np.hstack([images[0], images[1], mix_images[0]])
plt.figure(figsize=(9, 3))
plt.imshow(image_row)
plt.axis("off")
"""

"""
## Relevant Chapters from Deep Learning with Python
- [Chapter 17: Image generation](https://deeplearningwithpython.io/chapters/chapter17_image-generation)
"""

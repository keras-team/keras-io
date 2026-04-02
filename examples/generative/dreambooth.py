"""
Title: DreamBooth
Author: [Sayak Paul](https://twitter.com/RisingSayak), [Chansung Park](https://twitter.com/algo_diver)
Date created: 2023/02/01
Last modified: 2026/03/06
Description: Implementing DreamBooth.
Accelerator: GPU
Converted to Keras 3 by: [Harshith K](https://github.com/kharshith-k/)
"""

"""
## Introduction

In this example, we implement DreamBooth, a fine-tuning technique to teach new visual
concepts to text-conditioned Diffusion models with just 3 - 5 images. DreamBooth was
proposed in
[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)
by Ruiz et al.

DreamBooth, in a sense, is similar to the
[traditional way of fine-tuning a text-conditioned Diffusion model except](https://keras.io/examples/generative/finetune_stable_diffusion/)
for a few gotchas. This example assumes that you have basic familiarity with
Diffusion models and how to fine-tune them. Here are some reference examples that might
help you to get familiarized quickly:

* [Teach StableDiffusion new concepts via Textual Inversion](https://keras.io/examples/generative/fine_tune_via_textual_inversion/)
* [Fine-tuning Stable Diffusion](https://keras.io/examples/generative/finetune_stable_diffusion/)
"""


"""
This example is resource-intensive. For reliable execution, use a GPU with at least 80 GB of VRAM.
"""


"""
## Initial imports
"""


import math

import keras
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths

"""
## Usage of DreamBooth

... is very versatile. By teaching Stable Diffusion about your favorite visual
concepts, you can

* Recontextualize objects in interesting ways:

  ![](https://i.imgur.com/4Da9ozw.png)

* Generate artistic renderings of the underlying visual concept:

  ![](https://i.imgur.com/nI2N8bI.png)


And many other applications. We welcome you to check out the original
[DreamBooth paper](https://arxiv.org/abs/2208.12242) in this regard.
"""


instance_images_root = keras.utils.get_file(
    origin="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instance-images.tar.gz",
    untar=True,
)
class_images_root = keras.utils.get_file(
    origin="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/class-images.tar.gz",
    untar=True,
)


"""
## Visualize images

First, let's load the image paths.
"""


instance_image_paths = list(paths.list_images(instance_images_root))
class_image_paths = list(paths.list_images(class_images_root))


"""
Then we load the images from the paths.
"""


def load_images(image_paths):
    images = [np.array(keras.utils.load_img(path)) for path in image_paths]
    return images


"""
And then we make use a utility function to plot the loaded images.
"""


def plot_images(images, title=None):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        if title is not None:
            plt.title(title)
        plt.imshow(images[i])
        plt.axis("off")


"""
**Instance images**:
"""


plot_images(load_images(instance_image_paths[:5]))


"""
**Class images**:
"""


plot_images(load_images(class_image_paths[:5]))


"""
## Prepare datasets

Dataset preparation includes two stages: (1): preparing the captions, (2) processing the
images.
"""


"""
### Prepare the captions
"""


new_instance_image_paths = [
    instance_image_paths[index % len(instance_image_paths)]
    for index in range(len(class_image_paths))
]
instance_count = len(new_instance_image_paths)
class_count = len(class_image_paths)

unique_id = "sks"
class_label = "dog"

instance_prompt = f"a photo of {unique_id} {class_label}"
class_prompt = f"a photo of {class_label}"


"""
Next, we embed the prompts to save some compute.
"""


print("Loading Stable Diffusion 3 (will be reused for training)...")
sd3_backbone = keras_hub.models.StableDiffusion3Backbone.from_preset(
    "stable_diffusion_3_medium",
    image_shape=(512, 512, 3),
)
sd3_preprocessor = keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset(
    "stable_diffusion_3_medium"
)

unique_prompts = [instance_prompt, class_prompt]
print(
    f"Encoding {len(unique_prompts)} unique prompts (instead of {instance_count + class_count})..."
)

token_ids = sd3_preprocessor.generate_preprocess(unique_prompts)
negative_token_ids = sd3_preprocessor.generate_preprocess(["", ""])

(
    positive_embeddings,
    _,
    positive_pooled,
    _,
) = sd3_backbone.encode_text_step(token_ids, negative_token_ids)

instance_embedding = positive_embeddings[0:1]
class_embedding = positive_embeddings[1:2]
instance_pooled = positive_pooled[0:1]
class_pooled_single = positive_pooled[1:2]


def repeat_embedding(embedding, count):
    return np.repeat(embedding, count, axis=0)


instance_embedded_texts = repeat_embedding(instance_embedding, instance_count)
class_embedded_texts = repeat_embedding(class_embedding, class_count)
instance_pooled_embeddings = repeat_embedding(instance_pooled, instance_count)
class_pooled_embeddings = repeat_embedding(class_pooled_single, class_count)

print(
    f"Text embedding shapes: {instance_embedded_texts.shape}, {class_embedded_texts.shape}"
)
print(
    f"Pooled embedding shapes: {instance_pooled_embeddings.shape}, {class_pooled_embeddings.shape}"
)


"""
## Prepare the images
"""


resolution = 512

augmenter = keras.Sequential(
    layers=[
        keras.layers.CenterCrop(resolution, resolution),
        keras.layers.RandomFlip(),
        keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)


class DreamBoothDataset(keras.utils.PyDataset):
    """Backend-agnostic dataset for DreamBooth training.

    This dataset handles both instance and class images for prior preservation.
    """

    def __init__(
        self,
        instance_image_paths,
        class_image_paths,
        instance_embedded_texts,
        class_embedded_texts,
        instance_pooled_embeddings,
        class_pooled_embeddings,
        augmenter,
        batch_size=1,
        shuffle=True,
        seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.instance_image_paths = instance_image_paths
        self.class_image_paths = class_image_paths
        self.instance_embedded_texts = instance_embedded_texts
        self.class_embedded_texts = class_embedded_texts
        self.instance_pooled_embeddings = instance_pooled_embeddings
        self.class_pooled_embeddings = class_pooled_embeddings
        self.augmenter = augmenter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.num_samples = len(class_image_paths)

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self):
        """Shuffle indices at end of epoch if shuffle=True."""
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _get_batch_indices(self, batch_indices, num_items=None):
        if num_items is None:
            return batch_indices
        return [index % num_items for index in batch_indices]

    def _load_batch_images(self, image_paths, batch_indices, repeat=False):
        indices = self._get_batch_indices(
            batch_indices, len(image_paths) if repeat else None
        )
        images = [
            keras.utils.img_to_array(
                keras.utils.load_img(
                    image_paths[index], target_size=(resolution, resolution)
                )
            )
            for index in indices
        ]
        return np.array(images)

    def _gather_batch(self, values, batch_indices, repeat=False):
        indices = self._get_batch_indices(
            batch_indices, len(values) if repeat else None
        )
        return np.array([values[index] for index in indices])

    def __getitem__(self, idx):
        """Generate one batch of data."""
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        instance_images = self._load_batch_images(
            self.instance_image_paths, batch_indices, repeat=True
        )
        class_images = self._load_batch_images(self.class_image_paths, batch_indices)

        instance_embeds = self._gather_batch(
            self.instance_embedded_texts, batch_indices, repeat=True
        )
        class_embeds = self._gather_batch(self.class_embedded_texts, batch_indices)

        instance_pooled = self._gather_batch(
            self.instance_pooled_embeddings, batch_indices, repeat=True
        )
        class_pooled = self._gather_batch(self.class_pooled_embeddings, batch_indices)

        instance_images = self.augmenter(instance_images, training=True)
        class_images = self.augmenter(class_images, training=True)

        instance_batch = {
            "instance_images": instance_images,
            "instance_embedded_texts": instance_embeds,
            "instance_pooled_embeddings": instance_pooled,
        }
        class_batch = {
            "class_images": class_images,
            "class_embedded_texts": class_embeds,
            "class_pooled_embeddings": class_pooled,
        }

        return (instance_batch, class_batch)


"""
## Assemble dataset
"""


train_dataset = DreamBoothDataset(
    instance_image_paths=new_instance_image_paths,
    class_image_paths=class_image_paths,
    instance_embedded_texts=instance_embedded_texts,
    class_embedded_texts=class_embedded_texts,
    instance_pooled_embeddings=instance_pooled_embeddings,
    class_pooled_embeddings=class_pooled_embeddings,
    augmenter=augmenter,
    batch_size=1,
    shuffle=True,
    workers=2,
    use_multiprocessing=False,
)


"""
## Check shapes

Now that the dataset has been prepared, let's quickly check what's inside it.
"""


sample_batch = next(iter(train_dataset))
print(sample_batch[0].keys(), sample_batch[1].keys())

for k in sample_batch[0]:
    print(k, sample_batch[0][k].shape)

for k in sample_batch[1]:
    print(k, sample_batch[1][k].shape)


"""
During training, we make use of these keys to gather the images and text embeddings and
concat them accordingly.
"""


"""
## DreamBooth training loop

Our DreamBooth training loop is very much inspired by
[this script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)
provided by the Diffusers team at Hugging Face. However, there is an important
difference to note. We only fine-tune the diffusion model (the component responsible for predicting
noise / velocity) and don't fine-tune the text encoder in this example. If you're looking for an
implementation that also performs the additional fine-tuning of the text encoder, refer
to [this repository](https://github.com/sayakpaul/dreambooth-keras/).
"""


class DreamBoothTrainer(keras.Model):
    def __init__(
        self,
        diffusion_model,
        vae,
        backbone,
        noise_scheduler,
        use_mixed_precision=False,
        prior_loss_weight=1.0,
        max_grad_norm=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.diffusion_model = diffusion_model
        self.vae = vae
        self.backbone = backbone
        self.noise_scheduler = noise_scheduler
        self.prior_loss_weight = prior_loss_weight
        self.max_grad_norm = max_grad_norm

        self.use_mixed_precision = use_mixed_precision
        self.vae.trainable = False
        self.backbone.trainable = False
        self.diffusion_model.trainable = True

    def call(self, inputs):
        return inputs

    def compute_loss(self, x, y, y_pred, sample_weight):
        """Backend-agnostic loss computation override.

        The default train_step calls this method inside a gradient recording scope
        (e.g., GradientTape for TF, Autograd for Torch), so we don't need to manually
        handle gradients.
        """
        instance_batch = x
        class_batch = y

        instance_images = instance_batch["instance_images"]
        instance_embedded_text = instance_batch["instance_embedded_texts"]
        instance_pooled = instance_batch["instance_pooled_embeddings"]
        class_images = class_batch["class_images"]
        class_embedded_text = class_batch["class_embedded_texts"]
        class_pooled = class_batch["class_pooled_embeddings"]

        images = keras.ops.concatenate([instance_images, class_images], axis=0)
        embedded_texts = keras.ops.concatenate(
            [instance_embedded_text, class_embedded_text], axis=0
        )
        pooled_embeddings = keras.ops.concatenate(
            [instance_pooled, class_pooled], axis=0
        )
        batch_size = keras.ops.shape(images)[0]

        return self._compute_dreambooth_loss(
            images, embedded_texts, pooled_embeddings, batch_size
        )

    def _compute_dreambooth_loss(
        self, images, embedded_texts, pooled_embeddings, batch_size
    ):
        """Internal logic for DreamBooth loss (Flow Matching)."""
        latents = self.backbone.encode_image_step(images)

        noise = keras.random.normal(keras.ops.shape(latents))

        timesteps = keras.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=1,
            dtype="float32",
        )

        noisy_latents = keras.ops.add(
            keras.ops.multiply(
                keras.ops.subtract(1.0, keras.ops.reshape(timesteps, (-1, 1, 1, 1))),
                latents,
            ),
            keras.ops.multiply(keras.ops.reshape(timesteps, (-1, 1, 1, 1)), noise),
        )

        target = keras.ops.subtract(noise, latents)

        model_pred = self.diffusion_model(
            {
                "latent": noisy_latents,
                "context": embedded_texts,
                "pooled_projection": pooled_embeddings,
                "timestep": keras.ops.reshape(timesteps, (-1, 1)),
            },
            training=True,
        )

        loss = self._compute_split_loss(target, model_pred)
        return loss

    def _compute_split_loss(self, target, model_pred):
        """Compute split loss for instance and class images."""
        model_pred, model_pred_prior = keras.ops.split(model_pred, 2, axis=0)
        target, target_prior = keras.ops.split(target, 2, axis=0)

        target = keras.ops.cast(target, "float32")
        model_pred = keras.ops.cast(model_pred, "float32")
        target_prior = keras.ops.cast(target_prior, "float32")
        model_pred_prior = keras.ops.cast(model_pred_prior, "float32")

        loss = keras.ops.mean(keras.ops.square(target - model_pred))
        prior_loss = keras.ops.mean(keras.ops.square(target_prior - model_pred_prior))

        return loss + self.prior_loss_weight * prior_loss

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.diffusion_model.save_weights(filepath=filepath)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.diffusion_model.load_weights(filepath=filepath)


"""
## Trainer initialization
"""


use_mp = True

keras.mixed_precision.set_global_policy("mixed_float16")

print("Reusing SD3 backbone from text encoding step...")

diffusion_model = sd3_backbone.diffuser

vae = sd3_backbone.vae


"""
## Train!

We first calculate the number of epochs, we need to train for.
"""


num_update_steps_per_epoch = len(train_dataset)
max_train_steps = 1200
epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
print(f"Training for {epochs} epochs.")


"""
And then we start training!
"""


dreambooth_trainer = DreamBoothTrainer(
    diffusion_model=sd3_backbone.diffuser,
    vae=sd3_backbone.vae,
    backbone=sd3_backbone,
    noise_scheduler=None,
    use_mixed_precision=use_mp,
    prior_loss_weight=1.0,
)

learning_rate = 1e-5
optimizer = keras.optimizers.AdamW(
    learning_rate=learning_rate,
    weight_decay=0.0,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    clipnorm=1.0,
)

dreambooth_trainer.compile(optimizer=optimizer, loss="mse")

print("Starting training (resolution: 512x512)...")
ckpt_path = "dreambooth-unet.weights.h5"
ckpt_callback = keras.callbacks.ModelCheckpoint(
    ckpt_path,
    save_weights_only=True,
    monitor="loss",
    mode="min",
)

dreambooth_trainer.fit(train_dataset, epochs=epochs, callbacks=[ckpt_callback])


"""
## Experiments and inference

We ran various experiments with a slightly modified version of this example. Our
experiments are based on
[this repository](https://github.com/sayakpaul/dreambooth-keras/) and are inspired by
[this blog post](https://huggingface.co/blog/dreambooth) from Hugging Face.

First, let's see how we can use the fine-tuned checkpoint for running inference.
"""


import numpy as np

print("Loading Stable Diffusion 3 with 512x512 resolution (float32)...")
dreambooth_model_512 = keras_hub.models.StableDiffusion3TextToImage.from_preset(
    "stable_diffusion_3_medium",
    image_shape=(512, 512, 3),
    dtype="float32",
)

print(f"Loading fine-tuned weights from {ckpt_path}...")
dreambooth_model_512.backbone.diffuser.load_weights(ckpt_path)


"""
The default number of steps for generating an image with Stable Diffusion 3
is 50. Let's increase it to 100 for potentially better quality.
"""


prompt = f"A photo of {unique_id} {class_label} in a bucket"
print(f"Generating images for prompt: '{prompt}'...")

prompts = [prompt] * 3

images_dreamboothed = dreambooth_model_512.generate(prompts, num_steps=100, seed=42)

images_dreamboothed = np.array(images_dreamboothed)
if images_dreamboothed.ndim == 3:
    images_dreamboothed = np.expand_dims(images_dreamboothed, axis=0)

plot_images(images_dreamboothed, title=prompt)


"""
Feel free to experiment with different prompts (don't forget to add the unique identifier
and the class label!) to see how the results change. We welcome you to check out our
codebase and more experimental results
[here](https://github.com/sayakpaul/dreambooth-keras#results). You can also read
[this blog post](https://huggingface.co/blog/dreambooth) to get more ideas.
"""


"""
## Acknowledgements

* Thanks to the
[DreamBooth example script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)
provided by Hugging Face which helped us a lot in getting the initial implementation
ready quickly.
* Getting DreamBooth to work on human faces can be challenging. We have compiled some
general recommendations
[here](https://github.com/sayakpaul/dreambooth-keras#notes-on-preparing-data-for-dreambooth-training-of-faces).
Thanks to
[Abhishek Thakur](https://no.linkedin.com/in/abhi1thakur)
for helping with these.
"""


"""
## Relevant Chapters from Deep Learning with Python
- [Chapter 17: Image generation](https://deeplearningwithpython.io/chapters/chapter17_image-generation)
"""

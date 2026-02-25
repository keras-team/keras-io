"""
Title: DreamBooth
Author: [Sayak Paul](https://twitter.com/RisingSayak), [Chansung Park](https://twitter.com/algo_diver)
Date created: 2023/02/01
Last modified: 2026/02/17
Description: Implementing DreamBooth.
Accelerator: GPU
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

* [High-performance image generation using Stable Diffusion in KerasCV](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/)
* [Teach StableDiffusion new concepts via Textual Inversion](https://keras.io/examples/generative/fine_tune_via_textual_inversion/)
* [Fine-tuning Stable Diffusion](https://keras.io/examples/generative/finetune_stable_diffusion/)

First, let's install the latest versions of KerasCV and TensorFlow.

"""

"""shell
pip install -q -U keras-hub keras-cv
"""

"""
If you're running the code, please ensure you're using a GPU with at least 24 GBs of
VRAM.
"""

"""
## Initial imports
"""

import math

import keras
import keras_cv
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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

"""
## Download the instance and class images

DreamBooth uses a technique called "prior preservation" to meaningfully guide the
training procedure such that the fine-tuned models can still preserve some of the prior
semantics of the visual concept you're introducing. To know more about the idea of "prior
preservation" refer to [this document](https://dreambooth.github.io/).

Here, we need to introduce a few key terms specific to DreamBooth:

* **Unique class**: Examples include "dog", "person", etc. In this example, we use "dog".
* **Unique identifier**: A unique identifier that is prepended to the unique class while
forming the "instance prompts". In this example, we use "sks" as this unique identifier.
* **Instance prompt**: Denotes a prompt that best describes the "instance images". An
example prompt could be - "f"a photo of {unique_id} {unique_class}". So, for our example,
this becomes - "a photo  of sks dog".
* **Class prompt**: Denotes a prompt without the unique identifier. This prompt is used
for generating "class images" for prior preservation. For our example, this prompt is -
"a photo of dog".
* **Instance images**: Denote the images that represent the visual concept you're trying
to teach aka the "instance prompt". This number is typically just 3 - 5. We typically
gather these images ourselves.
* **Class images**: Denote the images generated using the "class prompt" for using prior
preservation in DreamBooth training. We leverage the pre-trained model before fine-tuning
it to generate these class images. Typically, 200 - 300 class images are enough.

In code, this generation process looks quite simply:

```py
from tqdm import tqdm
import numpy as np
import hashlib
import keras_cv
import PIL
import os

class_images_dir = "class-images"
os.makedirs(class_images_dir, exist_ok=True)

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=True)

class_prompt = "a photo of dog"
num_imgs_to_generate = 200
for i in tqdm(range(num_imgs_to_generate)):
    images = model.text_to_image(
        class_prompt,
        batch_size=3,
    )
    idx = np.random.choice(len(images))
    selected_image = PIL.Image.fromarray(images[idx])
    hash_image = hashlib.sha1(selected_image.tobytes()).hexdigest()
    image_filename = os.path.join(class_images_dir, f"{hash_image}.jpg")
    selected_image.save(image_filename)
```

To keep the runtime of this example short, the authors of this example have gone ahead
and generated some class images using
[this notebook](https://colab.research.google.com/gist/sayakpaul/6b5de345d29cf5860f84b6d04d958692/generate_class_priors.ipynb).

**Note** that prior preservation is an optional technique used in DreamBooth, but it
almost always helps in improving the quality of the generated images.
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

# Since we're using prior preservation, we need to match the number
# of instance images we're using. We just repeat the instance image paths
# to do so.
new_instance_image_paths = []
for index in range(len(class_image_paths)):
    instance_image = instance_image_paths[index % len(instance_image_paths)]
    new_instance_image_paths.append(instance_image)

# We just repeat the prompts / captions per images.
unique_id = "sks"
class_label = "dog"

instance_prompt = f"a photo of {unique_id} {class_label}"
instance_prompts = [instance_prompt] * len(new_instance_image_paths)

class_prompt = f"a photo of {class_label}"
class_prompts = [class_prompt] * len(class_image_paths)

"""
Next, we embed the prompts to save some compute.
"""

import itertools

# OPTIMIZATION: Only encode unique prompts once, then replicate
# Since all instance prompts are identical and all class prompts are identical,
# we only need to encode 2 prompts instead of 200+!

# Load SD3 backbone once and reuse for both text encoding AND training
print("Loading Stable Diffusion 3 (will be reused for training)...")
sd3_backbone = keras_hub.models.StableDiffusion3Backbone.from_preset(
    "stable_diffusion_3_medium",
    image_shape=(1024, 1024, 3),  # Needed for training later
)
sd3_preprocessor = keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset(
    "stable_diffusion_3_medium"
)

# Encode only the unique prompts
unique_prompts = [instance_prompt, class_prompt]  # Just 2 prompts!
print(f"Encoding {len(unique_prompts)} unique prompts (instead of {len(instance_prompts) + len(class_prompts)})...")

# Tokenize both prompts
token_ids = sd3_preprocessor.generate_preprocess(unique_prompts)
negative_token_ids = sd3_preprocessor.generate_preprocess(["", ""])

# Encode both prompts in one call
(
    positive_embeddings,
    _,  # negative_embeddings (not needed for training)
    positive_pooled,
    _,  # negative_pooled (not needed for training)
) = sd3_backbone.encode_text_step(token_ids, negative_token_ids)

# Extract embeddings for instance and class prompts
instance_embedding = positive_embeddings[0:1]  # Keep batch dimension
class_embedding = positive_embeddings[1:2]
instance_pooled = positive_pooled[0:1]
class_pooled_single = positive_pooled[1:2]

# Replicate embeddings for all images
instance_embedded_texts = np.tile(instance_embedding, (len(new_instance_image_paths), 1, 1))
class_embedded_texts = np.tile(class_embedding, (len(class_image_paths), 1, 1))
instance_pooled_embeddings = np.tile(instance_pooled, (len(new_instance_image_paths), 1))
class_pooled_embeddings = np.tile(class_pooled_single, (len(class_image_paths), 1))

# Combine for compatibility with original code
embedded_text = np.concatenate([instance_embedded_texts, class_embedded_texts], axis=0)
pooled_embeddings = np.concatenate([instance_pooled_embeddings, class_pooled_embeddings], axis=0)

print(f"Text embeddings shape: {embedded_text.shape}")
print(f"Pooled embeddings shape: {pooled_embeddings.shape}")

# Keep sd3_backbone and sd3_preprocessor - we'll reuse them for training!

"""
## Dataset preparation using PyDataset (backend-agnostic)
"""

# Stable Diffusion 3 was trained on 1024x1024 images
# The VAE downsamples by 8x: 1024x1024 -> 128x128 latents
resolution = 1024

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

        # Match lengths
        self.num_samples = len(class_image_paths)

        # Build indices
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self):
        """Shuffle indices at end of epoch if shuffle=True."""
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, idx):
        """Generate one batch of data."""
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Load instance images
        instance_images = []
        for i in batch_indices:
            img_idx = i % len(self.instance_image_paths)
            img = keras.utils.load_img(
                self.instance_image_paths[img_idx], target_size=(resolution, resolution)
            )
            instance_images.append(keras.utils.img_to_array(img))
        instance_images = np.array(instance_images)

        # Load class images
        class_images = []
        for i in batch_indices:
            img = keras.utils.load_img(
                self.class_image_paths[i], target_size=(resolution, resolution)
            )
            class_images.append(keras.utils.img_to_array(img))
        class_images = np.array(class_images)

        # Get corresponding embeddings
        instance_embeds = np.array(
            [
                self.instance_embedded_texts[i % len(self.instance_image_paths)]
                for i in batch_indices
            ]
        )
        class_embeds = np.array([self.class_embedded_texts[i] for i in batch_indices])
        
        # Get corresponding pooled embeddings
        instance_pooled = np.array(
            [
                self.instance_pooled_embeddings[i % len(self.instance_image_paths)]
                for i in batch_indices
            ]
        )
        class_pooled = np.array([self.class_pooled_embeddings[i] for i in batch_indices])

        # Apply augmentation
        instance_images = self.augmenter(instance_images, training=True)
        class_images = self.augmenter(class_images, training=True)

        # Return as tuple of dicts (instance_batch, class_batch)
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
# Create the backend-agnostic training dataset
train_dataset = DreamBoothDataset(
    instance_image_paths=new_instance_image_paths,
    class_image_paths=class_image_paths,
    instance_embedded_texts=embedded_text[: len(new_instance_image_paths)],
    class_embedded_texts=embedded_text[len(new_instance_image_paths) :],
    instance_pooled_embeddings=pooled_embeddings[: len(new_instance_image_paths)],
    class_pooled_embeddings=pooled_embeddings[len(new_instance_image_paths) :],
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
difference to note. We only fine-tune the UNet (the model responsible for predicting
noise) and don't fine-tune the text encoder in this example. If you're looking for an
implementation that also performs the additional fine-tuning of the text encoder, refer
to [this repository](https://github.com/sayakpaul/dreambooth-keras/).
"""


class DreamBoothTrainer(keras.Model):
    # Reference:
    # https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

    def __init__(
        self,
        diffusion_model,
        vae,
        backbone,
        noise_scheduler,
        use_mixed_precision=False,
        prior_loss_weight=1.0,
        max_grad_norm=1.0,
        conditioning_dropout_prob=0.1,  # CFG dropout probability
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.diffusion_model = diffusion_model
        self.vae = vae
        self.backbone = backbone
        self.noise_scheduler = noise_scheduler
        self.prior_loss_weight = prior_loss_weight
        self.max_grad_norm = max_grad_norm
        self.conditioning_dropout_prob = conditioning_dropout_prob

        self.use_mixed_precision = use_mixed_precision
        self.vae.trainable = False
        self.backbone.trainable = False  # Freeze entire backbone (VAE + rescaling)
        self.diffusion_model.trainable = True  # Only train the diffusion model

    def train_step(self, inputs):
        """Backend-agnostic training step.
        
        Keras automatically computes gradients for losses registered via self.add_loss().
        This works across TensorFlow, JAX, and PyTorch backends.
        """
        instance_batch = inputs[0]
        class_batch = inputs[1]

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

        # Compute loss using backend-agnostic operations
        loss = self.compute_loss(images, embedded_texts, pooled_embeddings, batch_size)

        # Register the loss - Keras handles gradients automatically
        self.add_loss(loss)

        # Return metrics
        return {"loss": loss}

    def compute_loss(self, images, embedded_texts, pooled_embeddings, batch_size):
        """Compute loss using only backend-agnostic operations."""
        # CRITICAL FIX: Use backbone's encode_image_step() instead of vae.encode()
        # This applies the correct rescaling: (latents - offset) * scale
        # Without this, latents are in the wrong scale and training fails!
        latents = self.backbone.encode_image_step(images)

        # NOTE: Testing WITHOUT scaling - KerasHub's diffuser may handle this internally
        # latents = keras.ops.multiply(latents, 1.5305)
        # SD3 uses normalized timesteps [0, 1] for flow matching
        timesteps = keras.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=1,
            dtype="float32",
        )

        # Add noise to latents using flow matching formula
        # latent_t = (1 - t) * latent + t * noise
        noisy_latents = keras.ops.add(
            keras.ops.multiply(
                keras.ops.subtract(1.0, keras.ops.reshape(timesteps, (-1, 1, 1, 1))),
                latents,
            ),
            keras.ops.multiply(keras.ops.reshape(timesteps, (-1, 1, 1, 1)), noise),
        )

        # Target is the velocity field for flow matching: v = noise - latent
        # This is the derivative of the interpolation path
        target = keras.ops.subtract(noise, latents)
        
        # CRITICAL: Apply CFG dropout for classifier-free guidance training
        # Randomly replace text embeddings with zeros (null conditioning)
        # This enables CFG at inference time
        cfg_dropout_mask = keras.random.uniform(shape=(batch_size,)) < self.conditioning_dropout_prob
        cfg_dropout_mask = keras.ops.cast(cfg_dropout_mask, dtype="float32")
        cfg_dropout_mask = keras.ops.reshape(cfg_dropout_mask, (-1, 1, 1))
        
        # Apply dropout: if mask=1, use zeros (drop conditioning), else use original
        embedded_texts = keras.ops.where(
            cfg_dropout_mask > 0,
            keras.ops.zeros_like(embedded_texts),
            embedded_texts
        )
        pooled_embeddings = keras.ops.where(
            keras.ops.reshape(cfg_dropout_mask, (-1, 1)) > 0,
            keras.ops.zeros_like(pooled_embeddings),
            pooled_embeddings
        )

        # Predict noise using MMDiT
        # MMDiT expects dictionary input with context and timestep
        # CRITICAL: Key is "latent" (singular), not "latents" (plural)
        # CRITICAL: timestep shape must be (batch_size, 1)
        model_pred = self.diffusion_model(
            {
                "latent": noisy_latents,  # Singular "latent"
                "context": embedded_texts,
                "pooled_projection": pooled_embeddings,  # Use actual pooled embeddings
                "timestep": keras.ops.reshape(timesteps, (-1, 1)),  # Shape: (batch_size, 1)
            },
            training=True,
        )

        loss = self.compute_loss_for_dreambooth(target, model_pred)
        return loss

    def compute_loss_for_dreambooth(self, target, model_pred):
        # Chunk the noise and model_pred into two parts and compute the loss
        # on each part separately.
        # Since the first half of the inputs has instance samples and the second half
        # has class samples, we do the chunking accordingly.
        model_pred, model_pred_prior = keras.ops.split(model_pred, 2, axis=0)
        target, target_prior = keras.ops.split(target, 2, axis=0)

        # Cast to float32 to avoid dtype mismatch in mixed precision training
        target = keras.ops.cast(target, "float32")
        model_pred = keras.ops.cast(model_pred, "float32")
        target_prior = keras.ops.cast(target_prior, "float32")
        model_pred_prior = keras.ops.cast(model_pred_prior, "float32")

        # Compute instance loss using MSE.
        loss = keras.ops.mean(keras.ops.square(target - model_pred))

        # Compute prior loss.
        prior_loss = keras.ops.mean(keras.ops.square(target_prior - model_pred_prior))

        # Add the prior loss to the instance loss.
        loss = loss + self.prior_loss_weight * prior_loss
        return loss

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Overriding this method will allow us to use the `ModelCheckpoint`
        # callback directly with this trainer class. In this case, it will
        # only checkpoint the `diffusion_model` since that's what we're training
        # during fine-tuning.
        self.diffusion_model.save_weights(filepath=filepath)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        # Similarly override `load_weights()` so that we can directly call it on
        # the trainer class object.
        self.diffusion_model.load_weights(filepath=filepath)


"""
## Trainer initialization
"""

# Comment it if you are not using a GPU having tensor cores.
keras.mixed_precision.set_global_policy("mixed_float16")

use_mp = True  # Set it to False if you're not using a GPU with tensor cores.

# Reuse the SD3 backbone we loaded earlier for text encoding
# (No need to reload - saves time and memory!)
print("Reusing SD3 backbone from text encoding step...")

# Extract the MMDiT (diffusion transformer) for fine-tuning
diffusion_model = sd3_backbone.diffuser

# Extract VAE for encoding images to latents
vae = sd3_backbone.vae

# Note: noise_scheduler not needed for training (we use manual flow matching formula)
dreambooth_trainer = DreamBoothTrainer(
    diffusion_model=diffusion_model,
    vae=vae,
    backbone=sd3_backbone,  # CRITICAL: Needed for proper latent rescaling
    noise_scheduler=None,  # Not used during training
    use_mixed_precision=use_mp,
)

# These hyperparameters come from this tutorial by Hugging Face:
# https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
# CRITICAL FIX: Previous weight_decay=1e-2 and clipnorm=1.0 were preventing learning!
# For a 2B parameter model, clipnorm=1.0 clips gradients WAY too aggressively.
# High weight decay also prevents fine-tuning from making progress.
learning_rate = 5e-6  # Standard DreamBooth learning rate
beta_1, beta_2 = 0.9, 0.999
weight_decay = 1e-4  # Reduced from 1e-2 (was too aggressive)
epsilon = 1e-08

optimizer = keras.optimizers.AdamW(
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
    clipnorm=None,  # Removed - was clipping entire 2B param gradient to norm=1.0!
)
dreambooth_trainer.compile(optimizer=optimizer, loss="mse")

# DEBUG: Verify training setup
print("\n=== Training Setup Verification ===")
print(f"Diffusion model trainable: {diffusion_model.trainable}")
print(f"Number of trainable weights: {len(diffusion_model.trainable_weights)}")
total_params = sum([keras.ops.size(w) for w in diffusion_model.trainable_weights])
print(f"Total trainable params: {total_params:,}")
print(f"Learning rate: {learning_rate}")
print(f"Mixed precision: {use_mp}")

# Test one training step with diagnostics
print("\nTesting one training step with full diagnostics...")
test_batch = next(iter(train_dataset))

# Extract batch data
instance_batch = test_batch[0]
class_batch = test_batch[1]
instance_images = instance_batch["instance_images"]
class_images = class_batch["class_images"]
images = keras.ops.concatenate([instance_images, class_images], axis=0)

# Check input image range
print(f"Input image range: [{float(keras.ops.min(images)):.3f}, {float(keras.ops.max(images)):.3f}]")

# Encode to latents and check magnitude (using backbone's rescaling)
latents = sd3_backbone.encode_image_step(images)
print(f"Latent shape (properly scaled): {latents.shape}")
print(f"Latent range (properly scaled): [{float(keras.ops.min(latents)):.3f}, {float(keras.ops.max(latents)):.3f}]")
print(f"Latent mean: {float(keras.ops.mean(latents)):.3f}, std: {float(keras.ops.std(latents)):.3f}")

# Sample noise and check velocity magnitude
noise = keras.random.normal(keras.ops.shape(latents))
velocity = keras.ops.subtract(noise, latents)
print(f"Velocity (target) range: [{float(keras.ops.min(velocity)):.3f}, {float(keras.ops.max(velocity)):.3f}]")
print(f"Velocity mean: {float(keras.ops.mean(velocity)):.3f}, std: {float(keras.ops.std(velocity)):.3f}")

# Run training step
test_loss = dreambooth_trainer.train_step(test_batch)
print(f"\nTest loss value: {test_loss['loss']:.4f}")

# Check if CFG dropout is working by running multiple steps
print("\nTesting CFG dropout (10% dropout rate)...")
dropout_count = 0
for i in range(20):
    batch_size = 2  # instance + class
    mask = keras.random.uniform(shape=(batch_size,)) < 0.1
    if keras.ops.any(mask):
        dropout_count += 1
print(f"CFG dropout triggered in {dropout_count}/20 tests (expected ~2)")

print("\nTraining setup verified successfully!\n")

"""
## Train!

We first calculate the number of epochs, we need to train for.
"""

num_update_steps_per_epoch = len(train_dataset)
max_train_steps = 800
epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
print(f"Training for {epochs} epochs.")

"""
And then we start training!
"""

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

# Initialize KerasHub Stable Diffusion 3 model for inference.
print("Loading KerasHub Stable Diffusion 3 for inference...")
dreambooth_model = keras_hub.models.StableDiffusion3TextToImage.from_preset(
    "stable_diffusion_3_medium"
)

# Load the fine-tuned diffusion model weights into the backbone
dreambooth_model.backbone.diffuser.load_weights(ckpt_path)

# Note how the unique identifier and the class have been used in the prompt.
prompt = f"A photo of {unique_id} {class_label} in a bucket"
num_imgs_to_gen = 3

# Generate images using the fine-tuned model
print(f"Generating images for prompt: '{prompt}'...")
images_dreamboothed = dreambooth_model.generate(prompt, num_steps=50, seed=42)
# Note: generate returns a single batch, select first num_imgs_to_gen if needed
if images_dreamboothed.shape[0] > num_imgs_to_gen:
    images_dreamboothed = images_dreamboothed[:num_imgs_to_gen]
plot_images(images_dreamboothed, prompt)

"""
The default number of steps for generating an image with Stable Diffusion 3
is 50. Let's increase it to 100 for potentially better quality.
"""

images_dreamboothed = dreambooth_model.generate(prompt, num_steps=100, seed=42)
if images_dreamboothed.shape[0] > num_imgs_to_gen:
    images_dreamboothed = images_dreamboothed[:num_imgs_to_gen]
plot_images(images_dreamboothed, prompt)

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

"""
Title: Teach StableDiffusion new concepts via Textual Inversion
Authors: Ian Stenbit, [lukewood](https://lukewood.xyz)
Date created: 2022/12/09
Last modified: 2026/03/31
Description: Learning new visual concepts with KerasHub's Stable Diffusion 3 implementation.
"""

"""
## Textual Inversion

Since its release, StableDiffusion has quickly become a favorite amongst
the generative machine learning community.
The high volume of traffic has led to open source contributed improvements,
heavy prompt engineering, and even the invention of novel algorithms.

Perhaps the most impressive new algorithm being used is
[Textual Inversion](https://github.com/rinongal/textual_inversion), presented in
[_An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion_](https://textual-inversion.github.io/).

Textual Inversion is the process of teaching an image generator a specific visual concept
through the use of fine-tuning. In the diagram below, you can see an
example of this process where the authors teach the model new concepts, calling them
"S_*".

![https://i.imgur.com/KqEeBsM.jpg](https://i.imgur.com/KqEeBsM.jpg)

Conceptually, textual inversion works by learning a token embedding for a new text
token, keeping the remaining components of StableDiffusion frozen.

This guide shows you how to fine-tune Stable Diffusion 3 using KerasHub
using a Textual-Inversion-inspired approach. By the end of the guide, you will be able to
generate "Gandalf the Gray as a &lt;my-funny-cat-token&gt;".

![https://i.imgur.com/rcb1Yfx.png](https://i.imgur.com/rcb1Yfx.png)

### Adapting Textual Inversion for Stable Diffusion 3

**Important Note on Implementation:**
The original Textual Inversion algorithm trains only new token embeddings while keeping
the entire diffusion model frozen. However, Stable Diffusion 3's architecture presents
some challenges for this classical approach:

1. **Multi-encoder architecture:** SD3 uses three text encoders (CLIP-L, CLIP-G, and T5-XXL)
   with complex preprocessing pipelines that are tightly coupled to their pretrained vocabularies.

2. **Limited vocabulary extension:** KerasHub's SD3 implementation doesn't expose a simple
   API to extend the tokenizer vocabulary and add custom embeddings like the original
   KerasCV Stable Diffusion did.

3. **Frozen preprocessing:** The text encoding pipeline is established at model creation
   and isn't easily modifiable for custom tokens.

Therefore, this tutorial adapts the Textual Inversion concept for SD3 by:
- Pre-computing text embeddings for prompts containing our placeholder token using SD3's
  existing text encoders
- Fine-tuning the diffusion model (transformer) to associate these embeddings with our
  visual concept
- Keeping the text encoders and VAE frozen

This approach is conceptually similar to **DreamBooth** but maintains the spirit of
Textual Inversion: teaching the model a new visual concept through a textual placeholder.
The end result is the same — you get a personalized model that understands your custom token!

First, let's import the packages we need, and create a
Stable Diffusion 3 instance so we can use some of its subcomponents for fine-tuning.
"""

"""shell
pip install -q keras keras-hub
"""

import os

import keras
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
from keras import ops


class StableDiffusionHubWrapper:
    def __init__(self, preset="stable_diffusion_3_medium", image_shape=(512, 512, 3)):
        self.model = keras_hub.models.StableDiffusion3TextToImage.from_preset(
            preset,
            image_shape=image_shape,
            dtype="float32",
        )
        self.backbone = self.model.backbone
        self.diffusion_model = self.backbone.diffuser
        self.preprocessor = (
            keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset(preset)
        )

    def text_to_image(self, prompt, batch_size=1, num_steps=50, seed=None):
        prompts = [prompt] * batch_size if isinstance(prompt, str) else prompt
        return np.array(self.model.generate(prompts, num_steps=num_steps, seed=seed))


stable_diffusion = StableDiffusionHubWrapper()

"""
Next, let's define a visualization utility to show off the generated images:
"""


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


"""
## Assembling a text-image pair dataset

In order to train the embedding of our new token, we first must assemble a dataset
consisting of text-image pairs.
Each sample from the dataset must contain an image of the concept we are teaching
StableDiffusion, as well as a caption accurately representing the content of the image.
In this tutorial, we will teach StableDiffusion the concept of Luke and Ian's GitHub
avatars:

![gh-avatars](https://i.imgur.com/WyEHDIR.jpg)

First, let's construct an image dataset of cat dolls:
"""


def assemble_image_array(paths):
    """
    Load images from local file paths or remote URLs.

    Args:
        paths: List of local file paths (relative to script) or URLs to images
    """
    files = []

    for i, path in enumerate(paths):
        # Check if it's a URL or local path
        if path.startswith("http://") or path.startswith("https://"):
            print(f"Downloading image {i+1}/{len(paths)}: {path}")
            file_path = keras.utils.get_file(origin=path)
            files.append(file_path)
        else:
            # Local file path - relative to script execution directory
            print(f"Loading local image {i+1}/{len(paths)}: {path}")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Image file not found: {path}\n"
                    f"Current working directory: {os.getcwd()}\n"
                    f"Make sure the path is correct relative to the script location."
                )
            files.append(path)

    # Resize and normalize images to [-1, 1].
    resize = keras.layers.Resizing(height=512, width=512, crop_to_aspect_ratio=True)
    images = [keras.utils.load_img(img) for img in files]
    images = [keras.utils.img_to_array(img) for img in images]
    images = np.array([resize(img) for img in images], dtype="float32")
    images = images / 127.5 - 1.0
    return images


"""
Next, we pre-compute text features using the SD3 text encoders:
"""

placeholder_token = "<my-funny-cat-token>"


def assemble_text_features(prompts):
    prompts = [prompt.format(placeholder_token) for prompt in prompts]
    token_ids = stable_diffusion.preprocessor.generate_preprocess(prompts)
    negative_token_ids = stable_diffusion.preprocessor.generate_preprocess(
        [""] * len(prompts)
    )
    (
        positive_embeddings,
        _,
        positive_pooled,
        _,
    ) = stable_diffusion.backbone.encode_text_step(token_ids, negative_token_ids)
    return np.array(positive_embeddings), np.array(positive_pooled)


augmenter = keras.Sequential(
    [
        keras.layers.RandomFlip(mode="horizontal"),
    ]
)


class TextualInversionDataset(keras.utils.PyDataset):
    def __init__(
        self,
        images,
        embedded_texts,
        pooled_embeddings,
        batch_size=1,
        repeats=5,
        shuffle=True,
        seed=1337,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.images = images
        self.embedded_texts = embedded_texts
        self.pooled_embeddings = pooled_embeddings
        self.batch_size = batch_size
        self.repeats = repeats
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.num_samples = len(embedded_texts) * repeats
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        text_indices = np.array([i % len(self.embedded_texts) for i in batch_indices])
        image_indices = np.array([i % len(self.images) for i in batch_indices])

        # Use numpy fancy indexing for efficient batch extraction
        batch_images = self.images[image_indices]
        batch_embedded_texts = self.embedded_texts[text_indices]
        batch_pooled_embeddings = self.pooled_embeddings[text_indices]

        batch_images = augmenter(batch_images, training=True)
        return {
            "images": batch_images,
            "embedded_texts": batch_embedded_texts,
            "pooled_embeddings": batch_pooled_embeddings,
        }


"""
Finally, we wrap the preprocessed arrays in a `keras.utils.PyDataset` so the pipeline
stays backend-agnostic and works with `model.fit()` directly.
"""


def assemble_dataset(urls, prompts, batch_size=1, repeats=5):
    images = assemble_image_array(urls)
    embedded_texts, pooled_embeddings = assemble_text_features(prompts)
    return TextualInversionDataset(
        images=images,
        embedded_texts=embedded_texts,
        pooled_embeddings=pooled_embeddings,
        batch_size=batch_size,
        repeats=repeats,
        shuffle=True,
    )


"""
In order to ensure our prompts are descriptive, we use extremely generic prompts.

Let's try this out with some sample images and prompts.
"""

# Using local paths to avoid imgur rate limiting
train_ds = assemble_dataset(
    urls=[
        "img/fine_tune_via_textual_inversion/input-data/VIedH1X.jpeg",
        "img/fine_tune_via_textual_inversion/input-data/eBw13hE.png",
        "img/fine_tune_via_textual_inversion/input-data/oJ3rSg7.png",
        "img/fine_tune_via_textual_inversion/input-data/5mCL6Df.jpeg",
        "img/fine_tune_via_textual_inversion/input-data/4Q6WWyI.jpeg",
    ],
    prompts=[
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ],
)

"""
## On the importance of prompt accuracy

During our first attempt at writing this guide we included images of groups of these cat
dolls in our dataset but continued to use the generic prompts listed above.
Our results were anecdotally poor. For example, here's cat doll gandalf using this method:

![mediocre-wizard](https://i.imgur.com/Thq7XOu.jpg)

It's conceptually close, but it isn't as great as it can be.

In order to remedy this, we began experimenting with splitting our images into images of
singular cat dolls and groups of cat dolls.
Following this split, we came up with new prompts for the group shots.

Training on text-image pairs that accurately represent the content boosted the quality
of our results *substantially*.  This speaks to the importance of prompt accuracy.

In addition to separating the images into singular and group images, we also remove some
inaccurate prompts; such as "a dark photo of the {}"

Keeping this in mind, we assemble our final training dataset below:
"""

# Local paths to downloaded images (relative to script location)
single_urls = [
    "img/fine_tune_via_textual_inversion/input-data/VIedH1X.jpeg",
    "img/fine_tune_via_textual_inversion/input-data/eBw13hE.png",
    "img/fine_tune_via_textual_inversion/input-data/oJ3rSg7.png",
    "img/fine_tune_via_textual_inversion/input-data/5mCL6Df.jpeg",
    "img/fine_tune_via_textual_inversion/input-data/4Q6WWyI.jpeg",
]

single_prompts = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

"""
![https://i.imgur.com/gQCRjK6.png](https://i.imgur.com/gQCRjK6.png)

Looks great!

Next, we assemble a dataset of groups of our GitHub avatars:
"""

# Group images - download these manually to input-data/ if you want to use them:
# https://i.imgur.com/yVmZ2Qa.jpg
# https://i.imgur.com/JbyFbZJ.jpg
# https://i.imgur.com/CCubd3q.jpg

group_urls = [
    "https://i.imgur.com/yVmZ2Qa.jpg",
    "https://i.imgur.com/JbyFbZJ.jpg",
    "https://i.imgur.com/CCubd3q.jpg",
]

group_prompts = [
    "a photo of a group of {}",
    "a rendering of a group of {}",
    "a cropped photo of the group of {}",
    "the photo of a group of {}",
    "a photo of a clean group of {}",
    "a photo of my group of {}",
    "a photo of a cool group of {}",
    "a close-up photo of a group of {}",
    "a bright photo of the group of {}",
    "a cropped photo of a group of {}",
    "a photo of the group of {}",
    "a good photo of the group of {}",
    "a photo of one group of {}",
    "a close-up photo of the group of {}",
    "a rendition of the group of {}",
    "a photo of the clean group of {}",
    "a rendition of a group of {}",
    "a photo of a nice group of {}",
    "a good photo of a group of {}",
    "a photo of the nice group of {}",
    "a photo of the small group of {}",
    "a photo of the weird group of {}",
    "a photo of the large group of {}",
    "a photo of a cool group of {}",
    "a photo of a small group of {}",
]

"""
![https://i.imgur.com/GY9Pf3D.png](https://i.imgur.com/GY9Pf3D.png)

Finally, we merge the two URL and prompt lists into a single dataset:
"""

# For this tutorial, we'll use only the single images to avoid rate limiting.
# If you have group images downloaded locally, add them here.
all_urls = single_urls  # + group_urls (if you have them locally)
all_prompts = single_prompts  # + group_prompts (if using group images)

train_ds = assemble_dataset(
    all_urls,
    all_prompts,
    batch_size=1,
    repeats=5,
)

"""
## Preparing the Stable Diffusion 3 model for fine-tuning

Now that we have our dataset ready, let's prepare the model for training.

### Architecture Overview

With SD3 and KerasHub, our training strategy differs from classical Textual Inversion:

**Classical Textual Inversion (original SD):**
- Add new token to vocabulary
- Create trainable embedding for that token
- Freeze everything else (diffusion model, VAE, rest of text encoder)
- Train only the new token embedding

**Our SD3 Approach:**
- Pre-compute text features with SD3's multi-encoder system (in `assemble_text_features`)
- Freeze text encoders and VAE
- Fine-tune the diffusion model (transformer denoiser) to associate the pre-computed
  text features with visual concepts from training images

This adapted approach is necessary because SD3's architecture doesn't allow easy vocabulary
extension, but it achieves the same goal: teaching the model to understand your custom
placeholder token and generate images based on it.
"""

# Confirm the SD3 components are all loaded correctly.
print("Backbone:   ", stable_diffusion.backbone.__class__.__name__)
print("Diffuser:   ", stable_diffusion.diffusion_model.__class__.__name__)
print("Preprocessor:", stable_diffusion.preprocessor.__class__.__name__)

"""
Now we freeze the SD3 backbone (which contains the VAE and text encoders) and mark
only the diffusion model as trainable:
"""

# Prepare trainable/frozen components for SD3 fine-tuning.
stable_diffusion.backbone.trainable = False
stable_diffusion.diffusion_model.trainable = True

"""
## Training

Now we can move on to the exciting part: training!

In this SD3-based Textual Inversion approach, we fine-tune the diffusion model
(the transformer denoiser) while keeping the VAE and text encoders frozen.
The placeholder token `<my-funny-cat-token>` is embedded by the SD3 text encoders
during dataset assembly; the diffusion model learns to associate those embeddings
with the visual concept from the training images.

**Note:** This approach is adapted for Stable Diffusion 3's architecture. Unlike
classical Textual Inversion (which trains only token embeddings while freezing the
diffusion model), we fine-tune the diffusion model itself.
"""

# Explicitly freeze the VAE as an additional safeguard.
# The VAE is used only for image encoding in compute_loss and must not be updated.
stable_diffusion.backbone.vae.trainable = False

"""
Let's confirm the proper weights are set to trainable.
"""

print([w.shape for w in stable_diffusion.diffusion_model.trainable_weights][:10])
print(
    "Total trainable weights:", len(stable_diffusion.diffusion_model.trainable_weights)
)

"""
## Training the diffusion model with SD3 conditioning

In order to train with KerasHub Stable Diffusion 3, we reuse precomputed text
conditioning from the SD3 preprocessor/backbone and optimize only the diffuser.

We use a backend-agnostic `compute_loss` with `keras.ops` and `keras.random`,
following the same migration pattern as the DreamBooth notebook.
"""

noise_seed = keras.random.SeedGenerator(1337)


def sample_noisy_latents(images):
    latents = stable_diffusion.backbone.encode_image_step(images)
    noise = keras.random.normal(shape=ops.shape(latents), seed=noise_seed)
    batch_size = ops.shape(latents)[0]
    timesteps = keras.random.uniform(
        shape=(batch_size,),
        minval=0.0,
        maxval=1.0,
        dtype="float32",
        seed=noise_seed,
    )
    t = ops.reshape(timesteps, (-1, 1, 1, 1))
    noisy_latents = (1.0 - t) * latents + t * noise
    target = noise - latents
    return noisy_latents, target, timesteps


"""
Next, we implement a `StableDiffusionFineTuner`, which is a subclass of `keras.Model`
that overrides `compute_loss` (instead of `train_step`) for backend-agnostic training.

The loss function follows the SD3 Flow Matching-style objective used in the DreamBooth
migration:
- encode images to latents
- sample noise and continuous timesteps
- mix latents/noise according to timestep
- predict the target velocity with the diffuser
- optimize MSE between target and prediction
"""


class StableDiffusionFineTuner(keras.Model):
    def __init__(self, stable_diffusion, **kwargs):
        super().__init__(**kwargs)
        self.stable_diffusion = stable_diffusion
        self.diffusion_model = stable_diffusion.diffusion_model

    def call(self, inputs):
        return inputs

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        _ = (y, y_pred, sample_weight)
        images = x["images"]
        embedded_texts = x["embedded_texts"]
        pooled_embeddings = x["pooled_embeddings"]

        noisy_latents, target, timesteps = sample_noisy_latents(images)
        model_pred = self.diffusion_model(
            {
                "latent": noisy_latents,
                "context": embedded_texts,
                "pooled_projection": pooled_embeddings,
                "timestep": ops.reshape(timesteps, (-1, 1)),
            },
            training=True,
        )
        return ops.mean(ops.square(target - model_pred))


"""
Before we start training, let's take a look at what Stable Diffusion 3 produces for our
placeholder token prompt. This gives us a baseline to compare against after fine-tuning.
"""

generated = stable_diffusion.text_to_image(
    f"an oil painting of {placeholder_token}", seed=1337, batch_size=3
)
plot_images(generated)

"""
Now, to get started with training, we can `compile()` our model like any other
Keras model and configure our training parameters such as learning rate and optimizer.
"""

trainer = StableDiffusionFineTuner(stable_diffusion, name="trainer")
EPOCHS = 50
learning_rate = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-5, decay_steps=len(train_ds) * EPOCHS
)
optimizer = keras.optimizers.AdamW(
    learning_rate=learning_rate,
    weight_decay=0.0,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    clipnorm=1.0,
)

trainer.compile(optimizer=optimizer)

"""
To monitor training, we can produce a `keras.callbacks.Callback` to produce a few images
every epoch using our custom token.

We create three callbacks with different prompts so that we can see how they progress
over the course of training. We use a fixed seed so that we can easily see the
progression of the learned token.
"""


class GenerateImages(keras.callbacks.Callback):
    def __init__(
        self, stable_diffusion, prompt, steps=50, frequency=10, seed=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.stable_diffusion = stable_diffusion
        self.prompt = prompt
        self.seed = seed
        self.frequency = frequency
        self.steps = steps

    def on_epoch_end(self, epoch, logs=None):
        _ = logs
        if epoch % self.frequency == 0:
            images = self.stable_diffusion.text_to_image(
                self.prompt, batch_size=3, num_steps=self.steps, seed=self.seed
            )
            plot_images(
                images,
            )


cbs = [
    GenerateImages(
        stable_diffusion, prompt=f"an oil painting of {placeholder_token}", seed=1337
    ),
    GenerateImages(
        stable_diffusion,
        prompt=f"gandalf the gray as a {placeholder_token}",
        seed=1337,
    ),
    GenerateImages(
        stable_diffusion,
        prompt=f"two {placeholder_token} getting married, photorealistic, high quality",
        seed=1337,
    ),
]

"""
Now, all that is left to do is to call `model.fit()`!
"""

trainer.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=cbs,
)

"""
It's pretty fun to see how the model learns our new token over time. Play around with it
and see how you can tune training parameters and your training dataset to produce the
best images!
"""

"""
## Taking the Fine Tuned Model for a Spin

Now for the really fun part. We've learned a token embedding for our custom token, so
now we can generate images with StableDiffusion the same way we would for any other
token!

Here are some fun example prompts to get you started, with sample outputs from our cat
doll token!
"""

generated = stable_diffusion.text_to_image(
    f"Gandalf as a {placeholder_token} fantasy art drawn by disney concept artists, "
    "golden colour, high quality, highly detailed, elegant, sharp focus, concept art, "
    "character concepts, digital painting, mystery, adventure",
    batch_size=3,
)
plot_images(generated)

"""
"""

generated = stable_diffusion.text_to_image(
    f"A masterpiece of a {placeholder_token} crying out to the heavens. "
    f"Behind the {placeholder_token}, an dark, evil shade looms over it - sucking the "
    "life right out of it.",
    batch_size=3,
)
plot_images(generated)

"""
"""

generated = stable_diffusion.text_to_image(
    f"An evil {placeholder_token}.", batch_size=3
)
plot_images(generated)

"""
"""

generated = stable_diffusion.text_to_image(
    f"A mysterious {placeholder_token} approaches the great pyramids of egypt.",
    batch_size=3,
)
plot_images(generated)

"""
## Conclusions

Using the Textual Inversion algorithm you can teach StableDiffusion new concepts!

Some possible next steps to follow:

- Try out your own prompts
- Teach the model a style
- Gather a dataset of your favorite pet cat or dog and teach the model about it
"""

"""
## Relevant Chapters from Deep Learning with Python
- [Chapter 17: Image generation](https://deeplearningwithpython.io/chapters/chapter17_image-generation)
"""

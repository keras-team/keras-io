# DreamBooth

**Author:** [Sayak Paul](https://twitter.com/RisingSayak), [Chansung Park](https://twitter.com/algo_diver)<br>
**Date created:** 2023/02/01<br>
**Last modified:** 2026/03/06<br>
**Description:** Implementing DreamBooth.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/dreambooth.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/dreambooth.py)



---
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

This example is resource-intensive. For reliable execution, use a GPU with at least 80 GB of VRAM.

---
## Initial imports


```python

import math

import keras
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
```

<div class="k-default-codeblock">
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1775110433.373311    5778 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1775110433.377650    5778 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1775110433.388657    5778 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1775110433.388668    5778 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1775110433.388669    5778 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1775110433.388671    5778 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
```
</div>

---
## Usage of DreamBooth

... is very versatile. By teaching Stable Diffusion about your favorite visual
concepts, you can

* Recontextualize objects in interesting ways:

  ![](https://i.imgur.com/4Da9ozw.png)

* Generate artistic renderings of the underlying visual concept:

  ![](https://i.imgur.com/nI2N8bI.png)


And many other applications. We welcome you to check out the original
[DreamBooth paper](https://arxiv.org/abs/2208.12242) in this regard.


```python

instance_images_root = keras.utils.get_file(
    origin="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instance-images.tar.gz",
    untar=True,
)
class_images_root = keras.utils.get_file(
    origin="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/class-images.tar.gz",
    untar=True,
)

```

<div class="k-default-codeblock">
```
Downloading data from https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instance-images.tar.gz

5556967/5556967 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step

Downloading data from https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/class-images.tar.gz

9093120/9093120 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step
```
</div>

---
## Visualize images

First, let's load the image paths.


```python

instance_image_paths = list(paths.list_images(instance_images_root))
class_image_paths = list(paths.list_images(class_images_root))

```

Then we load the images from the paths.


```python

def load_images(image_paths):
    images = [np.array(keras.utils.load_img(path)) for path in image_paths]
    return images

```

And then we make use a utility function to plot the loaded images.


```python

def plot_images(images, title=None):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        if title is not None:
            plt.title(title)
        plt.imshow(images[i])
        plt.axis("off")

```

**Instance images**:


```python

plot_images(load_images(instance_image_paths[:5]))

```


    
![png](/img/examples/generative/dreambooth/dreambooth_14_0.png)
    


**Class images**:


```python

plot_images(load_images(class_image_paths[:5]))

```


    
![png](/img/examples/generative/dreambooth/dreambooth_16_0.png)
    


---
## Prepare datasets

Dataset preparation includes two stages: (1): preparing the captions, (2) processing the
images.

### Prepare the captions


```python

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

```

Next, we embed the prompts to save some compute.


```python

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

```

<div class="k-default-codeblock">
```
Loading Stable Diffusion 3 (will be reused for training)...

Downloading to /home/kharshith/.cache/kagglehub/models/keras/stablediffusion3/keras/stable_diffusion_3_medium/5/config.json...
```
</div>

  0%|                                                                                                                                                                                                                                                                                                   | 0.00/4.19k [00:00<?, ?B/s]

    
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.19k/4.19k [00:00<00:00, 13.4MB/s]

    


<div class="k-default-codeblock">
```
Downloading to /home/kharshith/.cache/kagglehub/models/keras/stablediffusion3/keras/stable_diffusion_3_medium/5/model.weights.h5...
```
</div>

  0%|                                                                                                                                                                                                                                                                                                   | 0.00/5.57G [00:00<?, ?B/s]

    
  0%|                                                                                                                                                                                                                                                                                          | 1.00M/5.57G [00:00<16:21, 6.09MB/s]

    
  0%|▎                                                                                                                                                                                                                                                                                         | 6.00M/5.57G [00:00<03:57, 25.2MB/s]

    
  0%|▉                                                                                                                                                                                                                                                                                         | 18.0M/5.57G [00:00<01:35, 62.7MB/s]

    
  1%|█▋                                                                                                                                                                                                                                                                                        | 33.0M/5.57G [00:00<01:04, 92.4MB/s]

    
  1%|██▍                                                                                                                                                                                                                                                                                        | 48.0M/5.57G [00:00<00:53, 110MB/s]

    
  1%|███                                                                                                                                                                                                                                                                                        | 62.0M/5.57G [00:00<00:50, 118MB/s]

    
  1%|███▊                                                                                                                                                                                                                                                                                       | 77.0M/5.57G [00:00<00:46, 126MB/s]

    
  2%|████▌                                                                                                                                                                                                                                                                                      | 91.0M/5.57G [00:00<00:45, 130MB/s]

    
  2%|█████▎                                                                                                                                                                                                                                                                                      | 106M/5.57G [00:01<00:43, 134MB/s]

    
  2%|█████▉                                                                                                                                                                                                                                                                                      | 120M/5.57G [00:01<00:43, 135MB/s]

    
  2%|██████▋                                                                                                                                                                                                                                                                                     | 133M/5.57G [00:01<00:50, 115MB/s]

    
  3%|███████▎                                                                                                                                                                                                                                                                                    | 147M/5.57G [00:01<00:47, 121MB/s]

    
  3%|████████                                                                                                                                                                                                                                                                                    | 161M/5.57G [00:01<00:45, 126MB/s]

    
  3%|████████▊                                                                                                                                                                                                                                                                                   | 176M/5.57G [00:01<00:44, 131MB/s]

    
  3%|█████████▌                                                                                                                                                                                                                                                                                  | 191M/5.57G [00:01<00:43, 134MB/s]

    
  4%|██████████▏                                                                                                                                                                                                                                                                                 | 205M/5.57G [00:01<00:42, 135MB/s]

    
  4%|██████████▉                                                                                                                                                                                                                                                                                 | 220M/5.57G [00:01<00:41, 137MB/s]

    
  4%|███████████▋                                                                                                                                                                                                                                                                                | 234M/5.57G [00:02<00:41, 138MB/s]

    
  4%|████████████▍                                                                                                                                                                                                                                                                               | 249M/5.57G [00:02<00:41, 139MB/s]

    
  5%|█████████████                                                                                                                                                                                                                                                                               | 263M/5.57G [00:02<00:40, 139MB/s]

    
  5%|█████████████▊                                                                                                                                                                                                                                                                              | 278M/5.57G [00:02<00:40, 140MB/s]

    
  5%|██████████████▌                                                                                                                                                                                                                                                                             | 293M/5.57G [00:02<00:40, 141MB/s]

    
  5%|███████████████▎                                                                                                                                                                                                                                                                            | 307M/5.57G [00:02<00:40, 140MB/s]

    
  6%|████████████████                                                                                                                                                                                                                                                                            | 322M/5.57G [00:02<00:40, 141MB/s]

    
  6%|████████████████▋                                                                                                                                                                                                                                                                           | 336M/5.57G [00:02<00:40, 140MB/s]

    
  6%|█████████████████▍                                                                                                                                                                                                                                                                          | 351M/5.57G [00:02<00:39, 141MB/s]

    
  6%|██████████████████▏                                                                                                                                                                                                                                                                         | 365M/5.57G [00:03<00:40, 140MB/s]

    
  7%|██████████████████▉                                                                                                                                                                                                                                                                         | 380M/5.57G [00:03<00:39, 141MB/s]

    
  7%|███████████████████▋                                                                                                                                                                                                                                                                        | 395M/5.57G [00:03<00:39, 140MB/s]

    
  7%|████████████████████▍                                                                                                                                                                                                                                                                       | 409M/5.57G [00:03<00:39, 140MB/s]

    
  7%|█████████████████████                                                                                                                                                                                                                                                                       | 424M/5.57G [00:03<00:39, 141MB/s]

    
  8%|█████████████████████▊                                                                                                                                                                                                                                                                      | 438M/5.57G [00:03<00:39, 140MB/s]

    
  8%|██████████████████████▌                                                                                                                                                                                                                                                                     | 453M/5.57G [00:03<00:39, 141MB/s]

    
  8%|███████████████████████▎                                                                                                                                                                                                                                                                    | 468M/5.57G [00:03<00:38, 141MB/s]

    
  8%|████████████████████████                                                                                                                                                                                                                                                                    | 482M/5.57G [00:03<00:38, 141MB/s]

    
  9%|████████████████████████▊                                                                                                                                                                                                                                                                   | 497M/5.57G [00:04<00:38, 142MB/s]

    
  9%|█████████████████████████▍                                                                                                                                                                                                                                                                  | 511M/5.57G [00:04<00:38, 140MB/s]

    
  9%|██████████████████████████▏                                                                                                                                                                                                                                                                 | 526M/5.57G [00:04<00:38, 141MB/s]

    
  9%|██████████████████████████▉                                                                                                                                                                                                                                                                 | 540M/5.57G [00:04<00:38, 140MB/s]

    
 10%|███████████████████████████▋                                                                                                                                                                                                                                                                | 555M/5.57G [00:04<00:38, 140MB/s]

    
 10%|████████████████████████████▍                                                                                                                                                                                                                                                               | 570M/5.57G [00:04<00:38, 141MB/s]

    
 10%|█████████████████████████████                                                                                                                                                                                                                                                               | 584M/5.57G [00:04<00:38, 141MB/s]

    
 11%|█████████████████████████████▊                                                                                                                                                                                                                                                              | 599M/5.57G [00:04<00:37, 141MB/s]

    
 11%|██████████████████████████████▌                                                                                                                                                                                                                                                             | 613M/5.57G [00:04<00:38, 140MB/s]

    
 11%|███████████████████████████████▎                                                                                                                                                                                                                                                            | 628M/5.57G [00:05<00:37, 141MB/s]

    
 11%|███████████████████████████████▉                                                                                                                                                                                                                                                            | 642M/5.57G [00:05<00:37, 140MB/s]

    
 12%|████████████████████████████████▋                                                                                                                                                                                                                                                           | 657M/5.57G [00:05<00:37, 141MB/s]

    
 12%|█████████████████████████████████▍                                                                                                                                                                                                                                                          | 672M/5.57G [00:05<00:37, 141MB/s]

    
 12%|██████████████████████████████████▏                                                                                                                                                                                                                                                         | 686M/5.57G [00:05<00:37, 141MB/s]

    
 12%|██████████████████████████████████▉                                                                                                                                                                                                                                                         | 701M/5.57G [00:05<00:37, 141MB/s]

    
 13%|███████████████████████████████████▌                                                                                                                                                                                                                                                        | 715M/5.57G [00:05<00:37, 141MB/s]

    
 13%|████████████████████████████████████▎                                                                                                                                                                                                                                                       | 730M/5.57G [00:05<00:36, 141MB/s]

    
 13%|█████████████████████████████████████                                                                                                                                                                                                                                                       | 744M/5.57G [00:05<00:37, 140MB/s]

    
 13%|█████████████████████████████████████▊                                                                                                                                                                                                                                                      | 759M/5.57G [00:05<00:36, 141MB/s]

    
 14%|██████████████████████████████████████▌                                                                                                                                                                                                                                                     | 774M/5.57G [00:06<00:36, 142MB/s]

    
 14%|███████████████████████████████████████▎                                                                                                                                                                                                                                                    | 788M/5.57G [00:06<00:36, 140MB/s]

    
 14%|████████████████████████████████████████                                                                                                                                                                                                                                                    | 803M/5.57G [00:06<00:36, 141MB/s]

    
 14%|████████████████████████████████████████▋                                                                                                                                                                                                                                                   | 817M/5.57G [00:06<00:36, 140MB/s]

    
 15%|█████████████████████████████████████████▍                                                                                                                                                                                                                                                  | 832M/5.57G [00:06<00:36, 141MB/s]

    
 15%|██████████████████████████████████████████▏                                                                                                                                                                                                                                                 | 847M/5.57G [00:06<00:35, 142MB/s]

    
 15%|██████████████████████████████████████████▉                                                                                                                                                                                                                                                 | 861M/5.57G [00:06<00:36, 140MB/s]

    
 15%|███████████████████████████████████████████▋                                                                                                                                                                                                                                                | 876M/5.57G [00:06<00:35, 141MB/s]

    
 16%|████████████████████████████████████████████▎                                                                                                                                                                                                                                               | 890M/5.57G [00:06<00:35, 141MB/s]

    
 16%|█████████████████████████████████████████████                                                                                                                                                                                                                                               | 905M/5.57G [00:07<00:35, 141MB/s]

    
 16%|█████████████████████████████████████████████▊                                                                                                                                                                                                                                              | 919M/5.57G [00:07<00:35, 140MB/s]

    
 16%|██████████████████████████████████████████████▌                                                                                                                                                                                                                                             | 934M/5.57G [00:07<00:35, 141MB/s]

    
 17%|███████████████████████████████████████████████▎                                                                                                                                                                                                                                            | 949M/5.57G [00:07<00:35, 141MB/s]

    
 17%|███████████████████████████████████████████████▉                                                                                                                                                                                                                                            | 963M/5.57G [00:07<00:35, 140MB/s]

    
 17%|████████████████████████████████████████████████▋                                                                                                                                                                                                                                           | 977M/5.57G [00:07<00:36, 135MB/s]

    
 17%|█████████████████████████████████████████████████▎                                                                                                                                                                                                                                          | 990M/5.57G [00:07<00:37, 131MB/s]

    
 18%|█████████████████████████████████████████████████▊                                                                                                                                                                                                                                         | 0.98G/5.57G [00:07<00:38, 129MB/s]

    
 18%|██████████████████████████████████████████████████▍                                                                                                                                                                                                                                        | 0.99G/5.57G [00:07<00:43, 113MB/s]

    
 18%|███████████████████████████████████████████████████                                                                                                                                                                                                                                        | 1.00G/5.57G [00:08<00:42, 115MB/s]

    
 18%|███████████████████████████████████████████████████▊                                                                                                                                                                                                                                       | 1.02G/5.57G [00:08<00:39, 122MB/s]

    
 19%|████████████████████████████████████████████████████▌                                                                                                                                                                                                                                      | 1.03G/5.57G [00:08<00:38, 128MB/s]

    
 19%|█████████████████████████████████████████████████████▏                                                                                                                                                                                                                                     | 1.05G/5.57G [00:08<00:36, 131MB/s]

    
 19%|█████████████████████████████████████████████████████▉                                                                                                                                                                                                                                     | 1.06G/5.57G [00:08<00:35, 135MB/s]

    
 19%|██████████████████████████████████████████████████████▋                                                                                                                                                                                                                                    | 1.08G/5.57G [00:08<00:35, 136MB/s]

    
 20%|███████████████████████████████████████████████████████▍                                                                                                                                                                                                                                   | 1.09G/5.57G [00:08<00:34, 138MB/s]

    
 20%|████████████████████████████████████████████████████████▏                                                                                                                                                                                                                                  | 1.10G/5.57G [00:08<00:34, 138MB/s]

    
 20%|████████████████████████████████████████████████████████▊                                                                                                                                                                                                                                  | 1.12G/5.57G [00:08<00:34, 140MB/s]

    
 20%|█████████████████████████████████████████████████████████▌                                                                                                                                                                                                                                 | 1.13G/5.57G [00:09<00:33, 141MB/s]

    
 21%|██████████████████████████████████████████████████████████▏                                                                                                                                                                                                                                | 1.15G/5.57G [00:09<00:33, 140MB/s]

    
 21%|██████████████████████████████████████████████████████████▉                                                                                                                                                                                                                                | 1.16G/5.57G [00:09<00:33, 139MB/s]

    
 21%|███████████████████████████████████████████████████████████▌                                                                                                                                                                                                                               | 1.17G/5.57G [00:09<00:34, 138MB/s]

    
 21%|████████████████████████████████████████████████████████████▎                                                                                                                                                                                                                              | 1.19G/5.57G [00:09<00:33, 139MB/s]

    
 22%|█████████████████████████████████████████████████████████████                                                                                                                                                                                                                              | 1.20G/5.57G [00:09<00:33, 139MB/s]

    
 22%|█████████████████████████████████████████████████████████████▊                                                                                                                                                                                                                             | 1.22G/5.57G [00:09<00:33, 140MB/s]

    
 22%|██████████████████████████████████████████████████████████████▌                                                                                                                                                                                                                            | 1.23G/5.57G [00:09<00:33, 138MB/s]

    
 22%|███████████████████████████████████████████████████████████████▏                                                                                                                                                                                                                           | 1.24G/5.57G [00:09<00:33, 139MB/s]

    
 23%|███████████████████████████████████████████████████████████████▉                                                                                                                                                                                                                           | 1.26G/5.57G [00:10<00:33, 140MB/s]

    
 23%|████████████████████████████████████████████████████████████████▋                                                                                                                                                                                                                          | 1.27G/5.57G [00:10<00:33, 140MB/s]

    
 23%|█████████████████████████████████████████████████████████████████▍                                                                                                                                                                                                                         | 1.29G/5.57G [00:10<00:32, 141MB/s]

    
 23%|██████████████████████████████████████████████████████████████████                                                                                                                                                                                                                         | 1.30G/5.57G [00:10<00:32, 141MB/s]

    
 24%|██████████████████████████████████████████████████████████████████▊                                                                                                                                                                                                                        | 1.31G/5.57G [00:10<00:32, 141MB/s]

    
 24%|███████████████████████████████████████████████████████████████████▍                                                                                                                                                                                                                       | 1.33G/5.57G [00:10<00:32, 140MB/s]

    
 24%|████████████████████████████████████████████████████████████████████▏                                                                                                                                                                                                                      | 1.34G/5.57G [00:10<00:32, 141MB/s]

    
 24%|████████████████████████████████████████████████████████████████████▊                                                                                                                                                                                                                      | 1.35G/5.57G [00:10<00:32, 140MB/s]

    
 25%|█████████████████████████████████████████████████████████████████████▌                                                                                                                                                                                                                     | 1.37G/5.57G [00:10<00:32, 139MB/s]

    
 25%|██████████████████████████████████████████████████████████████████████▏                                                                                                                                                                                                                    | 1.38G/5.57G [00:10<00:31, 141MB/s]

    
 25%|██████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                                                    | 1.40G/5.57G [00:11<00:31, 140MB/s]

    
 25%|███████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                                                   | 1.41G/5.57G [00:11<00:31, 141MB/s]

    
 26%|████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                                                                  | 1.42G/5.57G [00:11<00:31, 141MB/s]

    
 26%|█████████████████████████████████████████████████████████████████████████▏                                                                                                                                                                                                                 | 1.44G/5.57G [00:11<00:31, 140MB/s]

    
 26%|█████████████████████████████████████████████████████████████████████████▊                                                                                                                                                                                                                 | 1.45G/5.57G [00:11<00:31, 142MB/s]

    
 26%|██████████████████████████████████████████████████████████████████████████▌                                                                                                                                                                                                                | 1.47G/5.57G [00:11<00:31, 138MB/s]

    
 27%|███████████████████████████████████████████████████████████████████████████▏                                                                                                                                                                                                               | 1.48G/5.57G [00:11<00:31, 138MB/s]

    
 27%|███████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                                               | 1.49G/5.57G [00:11<00:31, 139MB/s]

    
 27%|████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                                              | 1.51G/5.57G [00:11<00:31, 140MB/s]

    
 27%|█████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                                                             | 1.52G/5.57G [00:12<00:31, 139MB/s]

    
 28%|██████████████████████████████████████████████████████████████████████████████                                                                                                                                                                                                             | 1.54G/5.57G [00:12<00:31, 139MB/s]

    
 28%|██████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                                                                            | 1.55G/5.57G [00:12<00:31, 139MB/s]

    
 28%|███████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                                                                           | 1.56G/5.57G [00:12<00:30, 140MB/s]

    
 28%|████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                                                          | 1.58G/5.57G [00:12<00:30, 141MB/s]

    
 29%|████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                                          | 1.59G/5.57G [00:12<00:30, 140MB/s]

    
 29%|█████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                                         | 1.61G/5.57G [00:12<00:30, 139MB/s]

    
 29%|██████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                                                        | 1.62G/5.57G [00:12<00:30, 140MB/s]

    
 29%|███████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                                                                       | 1.64G/5.57G [00:12<00:30, 140MB/s]

    
 30%|███████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                                       | 1.65G/5.57G [00:13<00:30, 138MB/s]

    
 30%|████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                                                                      | 1.66G/5.57G [00:13<00:30, 136MB/s]

    
 30%|█████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                                                                     | 1.68G/5.57G [00:13<00:31, 133MB/s]

    
 30%|█████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                                     | 1.69G/5.57G [00:13<00:31, 132MB/s]

    
 31%|██████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                                                                    | 1.70G/5.57G [00:13<00:30, 134MB/s]

    
 31%|███████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                                                   | 1.72G/5.57G [00:13<00:30, 137MB/s]

    
 31%|████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                                                   | 1.73G/5.57G [00:13<00:32, 128MB/s]

    
 31%|████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                                  | 1.74G/5.57G [00:13<00:34, 118MB/s]

    
 32%|█████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                                                 | 1.76G/5.57G [00:13<00:33, 123MB/s]

    
 32%|██████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                                                 | 1.77G/5.57G [00:14<00:31, 128MB/s]

    
 32%|██████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                                                                | 1.79G/5.57G [00:14<00:30, 132MB/s]

    
 32%|███████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                                               | 1.80G/5.57G [00:14<00:30, 134MB/s]

    
 33%|████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                                                              | 1.81G/5.57G [00:14<00:29, 136MB/s]

    
 33%|████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                              | 1.83G/5.57G [00:14<00:29, 138MB/s]

    
 33%|█████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                             | 1.84G/5.57G [00:14<00:28, 138MB/s]

    
 33%|██████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                                            | 1.86G/5.57G [00:14<00:28, 140MB/s]

    
 34%|███████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                                            | 1.87G/5.57G [00:14<00:28, 139MB/s]

    
 34%|███████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                                                           | 1.89G/5.57G [00:14<00:28, 140MB/s]

    
 34%|████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                                                          | 1.90G/5.57G [00:15<00:27, 142MB/s]

    
 34%|█████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                                         | 1.91G/5.57G [00:15<00:27, 142MB/s]

    
 35%|█████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                         | 1.93G/5.57G [00:15<00:27, 142MB/s]

    
 35%|██████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                        | 1.94G/5.57G [00:15<00:27, 141MB/s]

    
 35%|███████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                                       | 1.95G/5.57G [00:15<00:27, 140MB/s]

    
 35%|████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                                       | 1.97G/5.57G [00:15<00:27, 140MB/s]

    
 36%|████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                      | 1.98G/5.57G [00:15<00:27, 139MB/s]

    
 36%|█████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                                     | 2.00G/5.57G [00:15<00:27, 140MB/s]

    
 36%|██████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                                                    | 2.01G/5.57G [00:15<00:27, 140MB/s]

    
 36%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                                    | 2.02G/5.57G [00:16<00:27, 141MB/s]

    
 37%|███████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                                   | 2.04G/5.57G [00:16<00:26, 141MB/s]

    
 37%|████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                                  | 2.05G/5.57G [00:16<00:26, 140MB/s]

    
 37%|█████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                                  | 2.07G/5.57G [00:16<00:26, 140MB/s]

    
 37%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                                                 | 2.08G/5.57G [00:16<00:26, 140MB/s]

    
 38%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                                                | 2.10G/5.57G [00:16<00:26, 140MB/s]

    
 38%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                               | 2.11G/5.57G [00:16<00:26, 140MB/s]

    
 38%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                               | 2.12G/5.57G [00:16<00:26, 140MB/s]

    
 38%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                              | 2.14G/5.57G [00:16<00:26, 141MB/s]

    
 39%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                             | 2.15G/5.57G [00:16<00:26, 140MB/s]

    
 39%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                                            | 2.17G/5.57G [00:17<00:25, 141MB/s]

    
 39%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                                            | 2.18G/5.57G [00:17<00:25, 140MB/s]

    
 39%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                                           | 2.20G/5.57G [00:17<00:25, 141MB/s]

    
 40%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                          | 2.21G/5.57G [00:17<00:25, 140MB/s]

    
 40%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                          | 2.22G/5.57G [00:17<00:25, 141MB/s]

    
 40%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                                         | 2.24G/5.57G [00:17<00:25, 142MB/s]

    
 40%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                        | 2.25G/5.57G [00:17<00:25, 140MB/s]

    
 41%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                                       | 2.27G/5.57G [00:17<00:25, 140MB/s]

    
 41%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                                       | 2.28G/5.57G [00:17<00:25, 141MB/s]

    
 41%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                      | 2.29G/5.57G [00:18<00:24, 141MB/s]

    
 41%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                     | 2.31G/5.57G [00:18<00:24, 141MB/s]

    
 42%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                     | 2.32G/5.57G [00:18<00:24, 140MB/s]

    
 42%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                                    | 2.34G/5.57G [00:18<00:24, 141MB/s]

    
 42%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                                   | 2.35G/5.57G [00:18<00:24, 140MB/s]

    
 43%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                                  | 2.37G/5.57G [00:18<00:24, 140MB/s]

    
 43%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                                  | 2.38G/5.57G [00:18<00:24, 141MB/s]

    
 43%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                                 | 2.39G/5.57G [00:18<00:24, 140MB/s]

    
 43%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                                | 2.41G/5.57G [00:18<00:24, 140MB/s]

    
 44%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                               | 2.42G/5.57G [00:19<00:24, 140MB/s]

    
 44%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                               | 2.44G/5.57G [00:19<00:23, 141MB/s]

    
 44%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                              | 2.45G/5.57G [00:19<00:23, 141MB/s]

    
 44%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                             | 2.47G/5.57G [00:19<00:23, 140MB/s]

    
 45%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                             | 2.48G/5.57G [00:19<00:23, 141MB/s]

    
 45%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                            | 2.49G/5.57G [00:19<00:23, 140MB/s]

    
 45%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                           | 2.51G/5.57G [00:19<00:23, 141MB/s]

    
 45%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                          | 2.52G/5.57G [00:19<00:23, 140MB/s]

    
 46%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                          | 2.54G/5.57G [00:19<00:23, 141MB/s]

    
 46%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                         | 2.55G/5.57G [00:20<00:22, 141MB/s]

    
 46%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                        | 2.57G/5.57G [00:20<00:25, 127MB/s]

    
 46%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                        | 2.58G/5.57G [00:20<00:26, 120MB/s]

    
 47%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                       | 2.59G/5.57G [00:20<00:25, 125MB/s]

    
 47%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                                      | 2.61G/5.57G [00:20<00:24, 130MB/s]

    
 47%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                     | 2.62G/5.57G [00:20<00:24, 131MB/s]

    
 47%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                     | 2.63G/5.57G [00:20<00:23, 135MB/s]

    
 48%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                                    | 2.65G/5.57G [00:20<00:23, 136MB/s]

    
 48%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                   | 2.66G/5.57G [00:20<00:22, 137MB/s]

    
 48%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                                   | 2.67G/5.57G [00:21<00:22, 138MB/s]

    
 48%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                                  | 2.69G/5.57G [00:21<00:22, 139MB/s]

    
 49%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                                 | 2.70G/5.57G [00:21<00:22, 139MB/s]

    
 49%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                                 | 2.72G/5.57G [00:21<00:21, 140MB/s]

    
 49%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                                | 2.73G/5.57G [00:21<00:21, 140MB/s]

    
 49%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                               | 2.74G/5.57G [00:21<00:22, 136MB/s]

    
 50%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                              | 2.76G/5.57G [00:21<00:23, 128MB/s]

    
 50%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                              | 2.77G/5.57G [00:21<00:23, 128MB/s]

    
 50%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                             | 2.78G/5.57G [00:21<00:22, 131MB/s]

    
 50%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                            | 2.80G/5.57G [00:22<00:22, 134MB/s]

    
 51%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                            | 2.81G/5.57G [00:22<00:21, 136MB/s]

    
 51%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                           | 2.83G/5.57G [00:22<00:21, 137MB/s]

    
 51%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                          | 2.84G/5.57G [00:22<00:21, 138MB/s]

    
 51%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                          | 2.85G/5.57G [00:22<00:21, 139MB/s]

    
 52%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                         | 2.87G/5.57G [00:22<00:20, 140MB/s]

    
 52%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                        | 2.88G/5.57G [00:22<00:20, 140MB/s]

    
 52%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                       | 2.90G/5.57G [00:22<00:20, 140MB/s]

    
 52%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                       | 2.91G/5.57G [00:22<00:20, 140MB/s]

    
 53%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                                      | 2.93G/5.57G [00:23<00:20, 140MB/s]

    
 53%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                                     | 2.94G/5.57G [00:23<00:20, 141MB/s]

    
 53%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                    | 2.95G/5.57G [00:23<00:20, 140MB/s]

    
 53%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                    | 2.97G/5.57G [00:23<00:19, 141MB/s]

    
 54%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                   | 2.98G/5.57G [00:23<00:19, 140MB/s]

    
 54%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                                  | 3.00G/5.57G [00:23<00:19, 141MB/s]

    
 54%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                                  | 3.01G/5.57G [00:23<00:19, 140MB/s]

    
 54%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                                 | 3.03G/5.57G [00:23<00:19, 141MB/s]

    
 55%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                | 3.04G/5.57G [00:23<00:19, 142MB/s]

    
 55%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                               | 3.05G/5.57G [00:24<00:19, 140MB/s]

    
 55%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                               | 3.07G/5.57G [00:24<00:19, 141MB/s]

    
 55%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                              | 3.08G/5.57G [00:24<00:18, 141MB/s]

    
 56%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                             | 3.10G/5.57G [00:24<00:18, 141MB/s]

    
 56%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                             | 3.11G/5.57G [00:24<00:18, 140MB/s]

    
 56%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                            | 3.12G/5.57G [00:24<00:18, 141MB/s]

    
 56%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                           | 3.14G/5.57G [00:24<00:18, 141MB/s]

    
 57%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                          | 3.15G/5.57G [00:24<00:18, 140MB/s]

    
 57%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                          | 3.17G/5.57G [00:24<00:18, 141MB/s]

    
 57%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                         | 3.18G/5.57G [00:24<00:18, 141MB/s]

    
 57%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                        | 3.20G/5.57G [00:25<00:18, 141MB/s]

    
 58%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                       | 3.21G/5.57G [00:25<00:17, 142MB/s]

    
 58%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                       | 3.22G/5.57G [00:25<00:17, 141MB/s]

    
 58%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                      | 3.24G/5.57G [00:25<00:17, 142MB/s]

    
 58%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                     | 3.25G/5.57G [00:25<00:17, 141MB/s]

    
 59%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                     | 3.27G/5.57G [00:25<00:17, 141MB/s]

    
 59%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                    | 3.28G/5.57G [00:25<00:17, 141MB/s]

    
 59%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                                   | 3.29G/5.57G [00:25<00:17, 141MB/s]

    
 59%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                  | 3.31G/5.57G [00:25<00:17, 140MB/s]

    
 60%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                  | 3.32G/5.57G [00:26<00:17, 141MB/s]

    
 60%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                 | 3.34G/5.57G [00:26<00:17, 140MB/s]

    
 60%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                | 3.35G/5.57G [00:26<00:17, 140MB/s]

    
 60%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                | 3.36G/5.57G [00:26<00:16, 140MB/s]

    
 61%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                               | 3.38G/5.57G [00:26<00:16, 141MB/s]

    
 61%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                              | 3.39G/5.57G [00:26<00:16, 141MB/s]

    
 61%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                              | 3.40G/5.57G [00:26<00:16, 139MB/s]

    
 61%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                             | 3.42G/5.57G [00:26<00:18, 125MB/s]

    
 62%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                            | 3.43G/5.57G [00:26<00:19, 119MB/s]

    
 62%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                            | 3.44G/5.57G [00:27<00:18, 123MB/s]

    
 62%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                           | 3.46G/5.57G [00:27<00:17, 128MB/s]

    
 62%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                                          | 3.47G/5.57G [00:27<00:17, 131MB/s]

    
 63%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                         | 3.49G/5.57G [00:27<00:16, 134MB/s]

    
 63%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                         | 3.50G/5.57G [00:27<00:16, 135MB/s]

    
 63%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                        | 3.51G/5.57G [00:27<00:16, 136MB/s]

    
 63%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                       | 3.53G/5.57G [00:27<00:15, 138MB/s]

    
 64%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                       | 3.54G/5.57G [00:27<00:15, 138MB/s]

    
 64%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                                      | 3.56G/5.57G [00:27<00:15, 139MB/s]

    
 64%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                     | 3.57G/5.57G [00:28<00:15, 138MB/s]

    
 64%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                                    | 3.58G/5.57G [00:28<00:15, 140MB/s]

    
 65%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                    | 3.60G/5.57G [00:28<00:15, 141MB/s]

    
 65%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                   | 3.61G/5.57G [00:28<00:15, 140MB/s]

    
 65%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                  | 3.63G/5.57G [00:28<00:15, 136MB/s]

    
 65%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                  | 3.64G/5.57G [00:28<00:15, 133MB/s]

    
 66%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                 | 3.65G/5.57G [00:28<00:15, 131MB/s]

    
 66%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                | 3.66G/5.57G [00:28<00:15, 131MB/s]

    
 66%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                | 3.68G/5.57G [00:28<00:15, 131MB/s]

    
 66%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                               | 3.69G/5.57G [00:29<00:14, 135MB/s]

    
 67%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                              | 3.71G/5.57G [00:29<00:14, 136MB/s]

    
 67%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                              | 3.72G/5.57G [00:29<00:14, 137MB/s]

    
 67%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                             | 3.73G/5.57G [00:29<00:14, 139MB/s]

    
 67%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                            | 3.75G/5.57G [00:29<00:14, 139MB/s]

    
 68%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                           | 3.76G/5.57G [00:29<00:13, 140MB/s]

    
 68%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                           | 3.78G/5.57G [00:29<00:13, 139MB/s]

    
 68%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                          | 3.79G/5.57G [00:29<00:13, 141MB/s]

    
 68%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                         | 3.81G/5.57G [00:29<00:13, 141MB/s]

    
 69%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                        | 3.82G/5.57G [00:29<00:13, 139MB/s]

    
 69%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                        | 3.83G/5.57G [00:30<00:13, 139MB/s]

    
 69%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                       | 3.85G/5.57G [00:30<00:13, 140MB/s]

    
 69%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                      | 3.86G/5.57G [00:30<00:13, 141MB/s]

    
 70%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                      | 3.88G/5.57G [00:30<00:13, 140MB/s]

    
 70%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                     | 3.89G/5.57G [00:30<00:12, 140MB/s]

    
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                    | 3.91G/5.57G [00:30<00:12, 141MB/s]

    
 70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                   | 3.92G/5.57G [00:30<00:12, 141MB/s]

    
 71%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                   | 3.93G/5.57G [00:30<00:12, 141MB/s]

    
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                  | 3.95G/5.57G [00:30<00:12, 141MB/s]

    
 71%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                 | 3.96G/5.57G [00:31<00:12, 141MB/s]

    
 71%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                 | 3.98G/5.57G [00:31<00:12, 140MB/s]

    
 72%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                | 3.99G/5.57G [00:31<00:12, 140MB/s]

    
 72%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                               | 4.00G/5.57G [00:31<00:11, 142MB/s]

    
 72%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                              | 4.02G/5.57G [00:31<00:11, 140MB/s]

    
 72%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                              | 4.03G/5.57G [00:31<00:11, 141MB/s]

    
 73%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                             | 4.05G/5.57G [00:31<00:11, 140MB/s]

    
 73%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                            | 4.06G/5.57G [00:31<00:11, 141MB/s]

    
 73%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                           | 4.08G/5.57G [00:31<00:11, 142MB/s]

    
 73%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                           | 4.09G/5.57G [00:32<00:11, 140MB/s]

    
 74%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                          | 4.10G/5.57G [00:32<00:11, 140MB/s]

    
 74%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                         | 4.12G/5.57G [00:32<00:11, 141MB/s]

    
 74%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                         | 4.13G/5.57G [00:32<00:10, 141MB/s]

    
 74%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                        | 4.15G/5.57G [00:32<00:10, 142MB/s]

    
 75%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                       | 4.16G/5.57G [00:32<00:10, 141MB/s]

    
 75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                      | 4.17G/5.57G [00:32<00:10, 140MB/s]

    
 75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 4.19G/5.57G [00:32<00:10, 140MB/s]

    
 75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                     | 4.20G/5.57G [00:32<00:10, 140MB/s]

    
 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                    | 4.21G/5.57G [00:32<00:10, 141MB/s]

    
 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                    | 4.23G/5.57G [00:33<00:10, 142MB/s]

    
 76%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                   | 4.24G/5.57G [00:33<00:10, 141MB/s]

    
 76%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                  | 4.26G/5.57G [00:33<00:09, 142MB/s]

    
 77%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                  | 4.27G/5.57G [00:33<00:09, 141MB/s]

    
 77%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                 | 4.28G/5.57G [00:33<00:09, 140MB/s]

    
 77%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                | 4.30G/5.57G [00:33<00:10, 124MB/s]

    
 77%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                | 4.31G/5.57G [00:33<00:11, 121MB/s]

    
 78%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                               | 4.32G/5.57G [00:33<00:10, 125MB/s]

    
 78%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                              | 4.34G/5.57G [00:34<00:10, 129MB/s]

    
 78%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                             | 4.35G/5.57G [00:34<00:09, 134MB/s]

    
 78%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                             | 4.36G/5.57G [00:34<00:09, 134MB/s]

    
 79%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                            | 4.38G/5.57G [00:34<00:09, 136MB/s]

    
 79%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                           | 4.39G/5.57G [00:34<00:09, 138MB/s]

    
 79%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                           | 4.40G/5.57G [00:34<00:09, 138MB/s]

    
 79%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                          | 4.42G/5.57G [00:34<00:08, 140MB/s]

    
 80%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                         | 4.43G/5.57G [00:34<00:08, 139MB/s]

    
 80%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                         | 4.45G/5.57G [00:34<00:08, 139MB/s]

    
 80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                        | 4.46G/5.57G [00:34<00:08, 140MB/s]

    
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                       | 4.47G/5.57G [00:35<00:08, 139MB/s]

    
 81%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                       | 4.49G/5.57G [00:35<00:08, 140MB/s]

    
 81%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                      | 4.50G/5.57G [00:35<00:08, 141MB/s]

    
 81%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                     | 4.51G/5.57G [00:35<00:08, 140MB/s]

    
 81%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                    | 4.53G/5.57G [00:35<00:07, 140MB/s]

    
 82%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                    | 4.54G/5.57G [00:35<00:07, 140MB/s]

    
 82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                   | 4.55G/5.57G [00:35<00:07, 141MB/s]

    
 82%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                  | 4.57G/5.57G [00:35<00:07, 141MB/s]

    
 82%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                  | 4.58G/5.57G [00:35<00:07, 135MB/s]

    
 83%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                 | 4.59G/5.57G [00:36<00:08, 130MB/s]

    
 83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                | 4.61G/5.57G [00:36<00:08, 128MB/s]

    
 83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                | 4.62G/5.57G [00:36<00:07, 132MB/s]

    
 83%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                               | 4.63G/5.57G [00:36<00:07, 134MB/s]

    
 84%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                              | 4.65G/5.57G [00:36<00:07, 136MB/s]

    
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                              | 4.66G/5.57G [00:36<00:07, 136MB/s]

    
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                             | 4.68G/5.57G [00:36<00:06, 138MB/s]

    
 84%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                            | 4.69G/5.57G [00:36<00:06, 140MB/s]

    
 85%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                           | 4.71G/5.57G [00:36<00:06, 139MB/s]

    
 85%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                           | 4.72G/5.57G [00:36<00:06, 141MB/s]

    
 85%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                          | 4.73G/5.57G [00:37<00:06, 140MB/s]

    
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                         | 4.75G/5.57G [00:37<00:06, 140MB/s]

    
 86%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                         | 4.76G/5.57G [00:37<00:06, 142MB/s]

    
 86%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                        | 4.78G/5.57G [00:37<00:05, 142MB/s]

    
 86%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                       | 4.79G/5.57G [00:37<00:05, 141MB/s]

    
 86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                      | 4.80G/5.57G [00:37<00:05, 141MB/s]

    
 87%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                      | 4.82G/5.57G [00:37<00:05, 141MB/s]

    
 87%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                     | 4.83G/5.57G [00:37<00:05, 141MB/s]

    
 87%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                    | 4.84G/5.57G [00:37<00:05, 141MB/s]

    
 87%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                    | 4.86G/5.57G [00:38<00:05, 141MB/s]

    
 88%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                   | 4.87G/5.57G [00:38<00:05, 141MB/s]

    
 88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                  | 4.89G/5.57G [00:38<00:05, 142MB/s]

    
 88%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                  | 4.90G/5.57G [00:38<00:05, 142MB/s]

    
 88%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                 | 4.91G/5.57G [00:38<00:04, 141MB/s]

    
 89%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                | 4.93G/5.57G [00:38<00:04, 139MB/s]

    
 89%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                               | 4.94G/5.57G [00:38<00:04, 139MB/s]

    
 89%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                               | 4.96G/5.57G [00:38<00:04, 139MB/s]

    
 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                              | 4.97G/5.57G [00:38<00:04, 139MB/s]

    
 90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                             | 4.98G/5.57G [00:38<00:04, 140MB/s]

    
 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                            | 5.00G/5.57G [00:39<00:04, 140MB/s]

    
 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                            | 5.01G/5.57G [00:39<00:04, 139MB/s]

    
 90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                           | 5.03G/5.57G [00:39<00:04, 139MB/s]

    
 91%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                          | 5.04G/5.57G [00:39<00:04, 141MB/s]

    
 91%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                          | 5.06G/5.57G [00:39<00:03, 141MB/s]

    
 91%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                         | 5.07G/5.57G [00:39<00:03, 141MB/s]

    
 91%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                        | 5.08G/5.57G [00:39<00:03, 141MB/s]

    
 92%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                       | 5.10G/5.57G [00:39<00:03, 140MB/s]

    
 92%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                       | 5.11G/5.57G [00:39<00:03, 140MB/s]

    
 92%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                      | 5.13G/5.57G [00:40<00:03, 140MB/s]

    
 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                     | 5.14G/5.57G [00:40<00:03, 135MB/s]

    
 93%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                     | 5.15G/5.57G [00:40<00:03, 135MB/s]

    
 93%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                    | 5.17G/5.57G [00:40<00:03, 134MB/s]

    
 93%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                   | 5.18G/5.57G [00:40<00:03, 137MB/s]

    
 93%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                   | 5.19G/5.57G [00:40<00:02, 137MB/s]

    
 94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                  | 5.21G/5.57G [00:40<00:02, 138MB/s]

    
 94%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                 | 5.22G/5.57G [00:40<00:02, 139MB/s]

    
 94%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                | 5.24G/5.57G [00:40<00:02, 140MB/s]

    
 94%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                | 5.25G/5.57G [00:41<00:02, 139MB/s]

    
 95%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌               | 5.26G/5.57G [00:41<00:02, 141MB/s]

    
 95%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎              | 5.28G/5.57G [00:41<00:02, 140MB/s]

    
 95%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉              | 5.29G/5.57G [00:41<00:02, 140MB/s]

    
 95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋             | 5.30G/5.57G [00:41<00:02, 139MB/s]

    
 96%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎            | 5.32G/5.57G [00:41<00:01, 139MB/s]

    
 96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████            | 5.33G/5.57G [00:41<00:01, 141MB/s]

    
 96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊           | 5.35G/5.57G [00:41<00:01, 139MB/s]

    
 96%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍          | 5.36G/5.57G [00:41<00:01, 138MB/s]

    
 97%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏         | 5.37G/5.57G [00:42<00:01, 141MB/s]

    
 97%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉         | 5.39G/5.57G [00:42<00:01, 140MB/s]

    
 97%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌        | 5.40G/5.57G [00:42<00:01, 123MB/s]

    
 97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏       | 5.41G/5.57G [00:42<00:01, 121MB/s]

    
 97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉       | 5.43G/5.57G [00:42<00:01, 127MB/s]

    
 98%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌      | 5.44G/5.57G [00:42<00:01, 131MB/s]

    
 98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎     | 5.46G/5.57G [00:42<00:00, 134MB/s]

    
 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████     | 5.47G/5.57G [00:42<00:00, 135MB/s]

    
 99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊    | 5.48G/5.57G [00:42<00:00, 137MB/s]

    
 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍   | 5.50G/5.57G [00:43<00:00, 138MB/s]

    
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏  | 5.51G/5.57G [00:43<00:00, 140MB/s]

    
 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉  | 5.53G/5.57G [00:43<00:00, 139MB/s]

    
 99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 5.54G/5.57G [00:43<00:00, 139MB/s]

    
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎| 5.55G/5.57G [00:43<00:00, 140MB/s]

    
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉| 5.57G/5.57G [00:43<00:00, 139MB/s]

    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.57G/5.57G [00:43<00:00, 137MB/s]

    


<div class="k-default-codeblock">
```
Downloading to /home/kharshith/.cache/kagglehub/models/keras/stablediffusion3/keras/stable_diffusion_3_medium/5/preprocessor.json...
```
</div>

  0%|                                                                                                                                                                                                                                                                                                   | 0.00/4.08k [00:00<?, ?B/s]

    
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.08k/4.08k [00:00<00:00, 14.2MB/s]

    


<div class="k-default-codeblock">
```
Downloading to /home/kharshith/.cache/kagglehub/models/keras/stablediffusion3/keras/stable_diffusion_3_medium/5/assets/clip_l_tokenizer/vocabulary.json...
```
</div>

  0%|                                                                                                                                                                                                                                                                                                    | 0.00/976k [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 976k/976k [00:00<00:00, 5.03MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 976k/976k [00:00<00:00, 5.01MB/s]

    


<div class="k-default-codeblock">
```
Downloading to /home/kharshith/.cache/kagglehub/models/keras/stablediffusion3/keras/stable_diffusion_3_medium/5/assets/clip_l_tokenizer/merges.txt...
```
</div>

  0%|                                                                                                                                                                                                                                                                                                    | 0.00/512k [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 512k/512k [00:00<00:00, 3.08MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 512k/512k [00:00<00:00, 3.07MB/s]

    


<div class="k-default-codeblock">
```
Downloading to /home/kharshith/.cache/kagglehub/models/keras/stablediffusion3/keras/stable_diffusion_3_medium/5/assets/clip_g_tokenizer/vocabulary.json...
```
</div>

  0%|                                                                                                                                                                                                                                                                                                    | 0.00/976k [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 976k/976k [00:00<00:00, 5.05MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 976k/976k [00:00<00:00, 5.04MB/s]

    


<div class="k-default-codeblock">
```
Downloading to /home/kharshith/.cache/kagglehub/models/keras/stablediffusion3/keras/stable_diffusion_3_medium/5/assets/clip_g_tokenizer/merges.txt...
```
</div>

  0%|                                                                                                                                                                                                                                                                                                    | 0.00/512k [00:00<?, ?B/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 512k/512k [00:00<00:00, 3.52MB/s]

    
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 512k/512k [00:00<00:00, 3.50MB/s]

    


<div class="k-default-codeblock">
```
Encoding 2 unique prompts (instead of 400)...

Text embedding shapes: (200, 154, 4096), (200, 154, 4096)
Pooled embedding shapes: (200, 2048), (200, 2048)
```
</div>

---
## Prepare the images


```python

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

```

---
## Assemble dataset


```python

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

```

---
## Check shapes

Now that the dataset has been prepared, let's quickly check what's inside it.


```python

sample_batch = next(iter(train_dataset))
print(sample_batch[0].keys(), sample_batch[1].keys())

for k in sample_batch[0]:
    print(k, sample_batch[0][k].shape)

for k in sample_batch[1]:
    print(k, sample_batch[1][k].shape)

```

<div class="k-default-codeblock">
```
dict_keys(['instance_images', 'instance_embedded_texts', 'instance_pooled_embeddings']) dict_keys(['class_images', 'class_embedded_texts', 'class_pooled_embeddings'])
instance_images (1, 512, 512, 3)
instance_embedded_texts (1, 154, 4096)
instance_pooled_embeddings (1, 2048)
class_images (1, 512, 512, 3)
class_embedded_texts (1, 154, 4096)
class_pooled_embeddings (1, 2048)
```
</div>

During training, we make use of these keys to gather the images and text embeddings and
concat them accordingly.

---
## DreamBooth training loop

Our DreamBooth training loop is very much inspired by
[this script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)
provided by the Diffusers team at Hugging Face. However, there is an important
difference to note. We only fine-tune the diffusion model (the component responsible for predicting
noise / velocity) and don't fine-tune the text encoder in this example. If you're looking for an
implementation that also performs the additional fine-tuning of the text encoder, refer
to [this repository](https://github.com/sayakpaul/dreambooth-keras/).


```python

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

```

---
## Trainer initialization


```python

use_mp = True

keras.mixed_precision.set_global_policy("mixed_float16")

print("Reusing SD3 backbone from text encoding step...")

diffusion_model = sd3_backbone.diffuser

vae = sd3_backbone.vae

```

<div class="k-default-codeblock">
```
Reusing SD3 backbone from text encoding step...
```
</div>

---
## Train!

We first calculate the number of epochs, we need to train for.


```python

num_update_steps_per_epoch = len(train_dataset)
max_train_steps = 1200
epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
print(f"Training for {epochs} epochs.")

```

<div class="k-default-codeblock">
```
Training for 6 epochs.
```
</div>

And then we start training!


```python

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

```

<div class="k-default-codeblock">
```
Starting training (resolution: 512x512)...

Epoch 1/6

200/200 ━━━━━━━━━━━━━━━━━━━━ 2037s 10s/step - loss: 0.9379

Epoch 2/6

200/200 ━━━━━━━━━━━━━━━━━━━━ 1962s 10s/step - loss: 0.8031

Epoch 3/6

200/200 ━━━━━━━━━━━━━━━━━━━━ 1964s 10s/step - loss: 0.6908

Epoch 4/6

200/200 ━━━━━━━━━━━━━━━━━━━━ 1972s 10s/step - loss: 0.6907

Epoch 5/6

200/200 ━━━━━━━━━━━━━━━━━━━━ 1971s 10s/step - loss: 0.6913

Epoch 6/6

200/200 ━━━━━━━━━━━━━━━━━━━━ 1970s 10s/step - loss: 0.6932

<keras.src.callbacks.history.History at 0x75078cdb77d0>
```
</div>

---
## Experiments and inference

We ran various experiments with a slightly modified version of this example. Our
experiments are based on
[this repository](https://github.com/sayakpaul/dreambooth-keras/) and are inspired by
[this blog post](https://huggingface.co/blog/dreambooth) from Hugging Face.

First, let's see how we can use the fine-tuned checkpoint for running inference.


```python

import numpy as np

print("Loading Stable Diffusion 3 with 512x512 resolution (float32)...")
dreambooth_model_512 = keras_hub.models.StableDiffusion3TextToImage.from_preset(
    "stable_diffusion_3_medium",
    image_shape=(512, 512, 3),
    dtype="float32",
)

print(f"Loading fine-tuned weights from {ckpt_path}...")
dreambooth_model_512.backbone.diffuser.load_weights(ckpt_path)

```

<div class="k-default-codeblock">
```
Loading Stable Diffusion 3 with 512x512 resolution (float32)...

Loading fine-tuned weights from dreambooth-unet.weights.h5...
```
</div>

The default number of steps for generating an image with Stable Diffusion 3
is 50. Let's increase it to 100 for potentially better quality.


```python

prompt = f"A photo of {unique_id} {class_label} in a bucket"
print(f"Generating images for prompt: '{prompt}'...")

prompts = [prompt] * 3

images_dreamboothed = dreambooth_model_512.generate(prompts, num_steps=100, seed=42)

images_dreamboothed = np.array(images_dreamboothed)
if images_dreamboothed.ndim == 3:
    images_dreamboothed = np.expand_dims(images_dreamboothed, axis=0)

plot_images(images_dreamboothed, title=prompt)

```

<div class="k-default-codeblock">
```
Generating images for prompt: 'A photo of sks dog in a bucket'...
```
</div>

![png](/img/examples/generative/dreambooth/dreambooth_40_1.png)
    


Feel free to experiment with different prompts (don't forget to add the unique identifier
and the class label!) to see how the results change. We welcome you to check out our
codebase and more experimental results
[here](https://github.com/sayakpaul/dreambooth-keras#results). You can also read
[this blog post](https://huggingface.co/blog/dreambooth) to get more ideas.

---
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

---
## Relevant Chapters from Deep Learning with Python
- [Chapter 17: Image generation](https://deeplearningwithpython.io/chapters/chapter17_image-generation)

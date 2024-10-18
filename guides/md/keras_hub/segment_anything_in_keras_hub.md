# Segment Anything in KerasHub!

**Author:** Tirth Patel, Ian Stenbit, Divyashree Sreepathihalli<br><br>
**Date created:** 2024/10/1<br><br>
**Last modified:** 2024/10/1<br><br>
**Description:** Segment anything using text, box, and points prompts in KerasHub.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/segment_anything_in_keras_hub.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/segment_anything_in_keras_hub.py)



---
## Overview

The Segment Anything Model (SAM) produces high quality object masks from input prompts
such as points or boxes, and it can be used to generate masks for all objects in an
image. It has been trained on a
[dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1
billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

In this guide, we will show how to use KerasHub's implementation of the
[Segment Anything Model](https://github.com/facebookresearch/segment-anything)
and show how powerful TensorFlow's and JAX's performance boost is.

First, let's get all our dependencies and images for our demo.


```python
!!pip install -Uq git+https://github.com/keras-team/keras-hub.git
!!pip install -Uq keras
```




```python
!!wget -q https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg
```
<div class="k-default-codeblock">
```
[]

[]

```
</div>
---
## Choose your backend

With Keras 3, you can choose to use your favorite backend!


```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import timeit
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import ops
import keras_hub
```

---
## Helper functions

Let's define some helper functions for visulazing the images, prompts, and the
segmentation results.


```python

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    box = box.reshape(-1)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def inference_resizing(image, pad=True):
    # Compute Preprocess Shape
    image = ops.cast(image, dtype="float32")
    old_h, old_w = image.shape[0], image.shape[1]
    scale = 1024 * 1.0 / max(old_h, old_w)
    new_h = old_h * scale
    new_w = old_w * scale
    preprocess_shape = int(new_h + 0.5), int(new_w + 0.5)

    # Resize the image
    image = ops.image.resize(image[None, ...], preprocess_shape)[0]

    # Pad the shorter side
    if pad:
        pixel_mean = ops.array([123.675, 116.28, 103.53])
        pixel_std = ops.array([58.395, 57.12, 57.375])
        image = (image - pixel_mean) / pixel_std
        h, w = image.shape[0], image.shape[1]
        pad_h = 1024 - h
        pad_w = 1024 - w
        image = ops.pad(image, [(0, pad_h), (0, pad_w), (0, 0)])
        # KerasHub now rescales the images and normalizes them.
        # Just unnormalize such that when KerasHub normalizes them
        # again, the padded values map to 0.
        image = image * pixel_std + pixel_mean
    return image

```

---
## Get the pretrained SAM model

We can initialize a trained SAM model using KerasHub's `from_preset` factory method. Here,
we use the huge ViT backbone trained on the SA-1B dataset (`sam_huge_sa1b`) for
high-quality segmentation masks. You can also use one of the `sam_large_sa1b` or
`sam_base_sa1b` for better performance (at the cost of decreasing quality of segmentation
masks).


```python
model = keras_hub.models.SAMImageSegmenter.from_preset("sam_huge_sa1b")
```

<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/kerashub/sam/keras/sam_huge_sa1b/2/download/config.json...

100%|████████████████████████████████████████████████████| 3.06k/3.06k [00:00<00:00, 6.08MB/s]

Downloading from https://www.kaggle.com/api/v1/models/kerashub/sam/keras/sam_huge_sa1b/2/download/task.json...

100%|████████████████████████████████████████████████████| 5.76k/5.76k [00:00<00:00, 11.0MB/s]

Downloading from https://www.kaggle.com/api/v1/models/kerashub/sam/keras/sam_huge_sa1b/2/download/task.weights.h5...

100%|████████████████████████████████████████████████████| 2.39G/2.39G [00:26<00:00, 95.7MB/s]

Downloading from https://www.kaggle.com/api/v1/models/kerashub/sam/keras/sam_huge_sa1b/2/download/model.weights.h5...

100%|████████████████████████████████████████████████████| 2.39G/2.39G [00:32<00:00, 79.7MB/s]

```
</div>
---
## Understanding Prompts

Segment Anything allows prompting an image using points, boxes, and masks:

1. Point prompts are the most basic of all: the model tries to guess the object given a
point on an image. The point can either be a foreground point (i.e. the desired
segmentation mask contains the point in it) or a backround point (i.e. the point lies
outside the desired mask).
2. Another way to prompt the model is using boxes. Given a bounding box, the model tries
to segment the object contained in it.
3. Finally, the model can also be prompted using a mask itself. This is useful, for
instance, to refine the borders of a previously predicted or known segmentation mask.

What makes the model incredibly powerful is the ability to combine the prompts above.
Point, box, and mask prompts can be combined in several different ways to achieve the
best result.

Let's see the semantics of passing these prompts to the Segment Anything model in
KerasHub. Input to the SAM model is a dictionary with keys:

1. `"images"`: A batch of images to segment. Must be of shape `(B, 1024, 1024, 3)`.
2. `"points"`: A batch of point prompts. Each point is an `(x, y)` coordinate originating
from the top-left corner of the image. In other works, each point is of the form `(r, c)`
where `r` and `c` are the row and column of the pixel in the image. Must be of shape `(B,
N, 2)`.
3. `"labels"`: A batch of labels for the given points. `1` represents foreground points
and `0` represents background points. Must be of shape `(B, N)`.
4. `"boxes"`: A batch of boxes. Note that the model only accepts one box per batch.
Hence, the expected shape is `(B, 1, 2, 2)`. Each box is a collection of 2 points: the
top left corner and the bottom right corner of the box. The points here follow the same
semantics as the point prompts. Here the `1` in the second dimension represents the
presence of box prompts. If the box prompts are missing, a placeholder input of shape
`(B, 0, 2, 2)` must be passed.
5. `"masks"`: A batch of masks. Just like box prompts, only one mask prompt per image is
allowed. The shape of the input mask must be `(B, 1, 256, 256, 1)` if they are present
and `(B, 0, 256, 256, 1)` for missing mask prompt.

Placeholder prompts are only required when calling the model directly (i.e.
`model(...)`). When calling the `predict` method, missing prompts can be omitted from the
input dictionary.

---
## Point prompts

First, let's segment an image using point prompts. We load the image and resize it to
shape `(1024, 1024)`, the image size the pretrained SAM model expects.


```python
# Load our image
image = np.array(keras.utils.load_img("truck.jpg"))
image = inference_resizing(image)

plt.figure(figsize=(10, 10))
plt.imshow(ops.convert_to_numpy(image) / 255.0)
plt.axis("on")
plt.show()
```


    
![png](/img/guides/segment_anything_in_keras_hub/segment_anything_in_keras_hub_11_0.png)
    


Next, we will define the point on the object we want to segment. Let's try to segment the
truck's window pane at coordinates `(284, 213)`.


```python
# Define the input point prompt
input_point = np.array([[284, 213.5]])
input_label = np.array([1])

plt.figure(figsize=(10, 10))
plt.imshow(ops.convert_to_numpy(image) / 255.0)
show_points(input_point, input_label, plt.gca())
plt.axis("on")
plt.show()
```


    
![png](/img/guides/segment_anything_in_keras_hub/segment_anything_in_keras_hub_13_0.png)
    


Now let's call the `predict` method of our model to get the segmentation masks.

**Note**: We don't call the model directly (`model(...)`) since placeholder prompts are
required to do so. Missing prompts are handled automatically by the predict method so we
call it instead. Also, when no box prompts are present, the points and labels need to be
padded with a zero point prompt and `-1` label prompt respectively. The cell below
demonstrates how this works.


```python
outputs = model.predict(
    {
        "images": image[np.newaxis, ...],
        "points": np.concatenate(
            [input_point[np.newaxis, ...], np.zeros((1, 1, 2))], axis=1
        ),
        "labels": np.concatenate(
            [input_label[np.newaxis, ...], np.full((1, 1), fill_value=-1)], axis=1
        ),
    }
)
```

<div class="k-default-codeblock">
```
Could not load symbol cuFuncGetName. Error: /usr/lib64-nvidia/libcuda.so.1: undefined symbol: cuFuncGetName

 1/1 ━━━━━━━━━━━━━━━━━━━━ 24s 24s/step

```
</div>
`SegmentAnythingModel.predict` returns two outputs. First are logits (segmentation masks)
of shape `(1, 4, 256, 256)` and the other are the IoU confidence scores (of shape `(1,
4)`) for each mask predicted. The pretrained SAM model predicts four masks: the first is
the best mask the model could come up with for the given prompts, and the other 3 are the
alternative masks which can be used in case the best prediction doesn't contain the
desired object. The user can choose whichever mask they prefer.

Let's visualize the masks returned by the model!


```python
# Resize the mask to our image shape i.e. (1024, 1024)
mask = inference_resizing(outputs["masks"][0][0][..., None], pad=False)[..., 0]
# Convert the logits to a numpy array
# and convert the logits to a boolean mask
mask = ops.convert_to_numpy(mask) > 0.0
iou_score = ops.convert_to_numpy(outputs["iou_pred"][0][0])

plt.figure(figsize=(10, 10))
plt.imshow(ops.convert_to_numpy(image) / 255.0)
show_mask(mask, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.title(f"IoU Score: {iou_score:.3f}", fontsize=18)
plt.axis("off")
plt.show()
```


    
![png](/img/guides/segment_anything_in_keras_hub/segment_anything_in_keras_hub_17_0.png)
    


As expected, the model returns a segmentation mask for the truck's window pane. But, our
point prompt can also mean a range of other things. For example, another possible mask
that contains our point is just the right side of the window pane or the whole truck.

Let's also visualize the other masks the model has predicted.


```python
fig, ax = plt.subplots(1, 3, figsize=(20, 60))
masks, scores = outputs["masks"][0][1:], outputs["iou_pred"][0][1:]
for i, (mask, score) in enumerate(zip(masks, scores)):
    mask = inference_resizing(mask[..., None], pad=False)[..., 0]
    mask, score = map(ops.convert_to_numpy, (mask, score))
    mask = 1 * (mask > 0.0)
    ax[i].imshow(ops.convert_to_numpy(image) / 255.0)
    show_mask(mask, ax[i])
    show_points(input_point, input_label, ax[i])
    ax[i].set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=12)
    ax[i].axis("off")
plt.show()
```


    
![png](/img/guides/segment_anything_in_keras_hub/segment_anything_in_keras_hub_20_0.png)
    


Nice! SAM was able to capture the ambiguity of our point prompt and also returned other
possible segmentation masks.

---
## Box Prompts

Now, let's see how we can prompt the model using boxes. The box is specified using two
points, the top-left corner and the bottom-right corner of the bounding box in xyxy
format. Let's prompt the model using a bounding box around the left front tyre of the
truck.


```python
# Let's specify the box
input_box = np.array([[240, 340], [400, 500]])

outputs = model.predict(
    {"images": image[np.newaxis, ...], "boxes": input_box[np.newaxis, np.newaxis, ...]}
)
mask = inference_resizing(outputs["masks"][0][0][..., None], pad=False)[..., 0]
mask = ops.convert_to_numpy(mask) > 0.0

plt.figure(figsize=(10, 10))
plt.imshow(ops.convert_to_numpy(image) / 255.0)
show_mask(mask, plt.gca())
show_box(input_box, plt.gca())
plt.axis("off")
plt.show()
```

<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 10s 10s/step

```
</div>
    
![png](/img/guides/segment_anything_in_keras_hub/segment_anything_in_keras_hub_23_1.png)
    


Boom! The model perfectly segments out the left front tyre in our bounding box.

---
## Combining prompts

To get the true potential of the model out, let's combine box and point prompts and see
what the model does.


```python
# Let's specify the box
input_box = np.array([[240, 340], [400, 500]])
# Let's specify the point and mark it background
input_point = np.array([[325, 425]])
input_label = np.array([0])

outputs = model.predict(
    {
        "images": image[np.newaxis, ...],
        "points": input_point[np.newaxis, ...],
        "labels": input_label[np.newaxis, ...],
        "boxes": input_box[np.newaxis, np.newaxis, ...],
    }
)
mask = inference_resizing(outputs["masks"][0][0][..., None], pad=False)[..., 0]
mask = ops.convert_to_numpy(mask) > 0.0

plt.figure(figsize=(10, 10))
plt.imshow(ops.convert_to_numpy(image) / 255.0)
show_mask(mask, plt.gca())
show_box(input_box, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis("off")
plt.show()
```

<div class="k-default-codeblock">
```
 1/1 ━━━━━━━━━━━━━━━━━━━━ 14s 14s/step

```
</div>
    
![png](/img/guides/segment_anything_in_keras_hub/segment_anything_in_keras_hub_25_1.png)
    


Voila! The model understood that the object we wanted to exclude from our mask was the
rim of the tyre.

---
## Text prompts

Finally, let's see how text prompts can be used along with KerasHub's
`SegmentAnythingModel`.

For this demo, we will use the
[offical Grounding DINO model](https://github.com/IDEA-Research/GroundingDINO).
Grounding DINO is a model that
takes as input a `(image, text)` pair and generates a bounding box around the object in
the `image` described by the `text`. You can refer to the
[paper](https://arxiv.org/abs/2303.05499) for more details on the implementation of the
model.

For this part of the demo, we will need to install the `groundingdino` package from
source:

```
pip install -U git+https://github.com/IDEA-Research/GroundingDINO.git
```

Then, we can install the pretrained model's weights and config:


```python
!!pip install -U git+https://github.com/IDEA-Research/GroundingDINO.git
```




```python
!!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
!!wget -q https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/v0.1.0-alpha2/groundingdino/config/GroundingDINO_SwinT_OGC.py
```
```python
from groundingdino.util.inference import Model as GroundingDINO

CONFIG_PATH = "GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "groundingdino_swint_ogc.pth"

grounding_dino = GroundingDINO(CONFIG_PATH, WEIGHTS_PATH)
```
<div class="k-default-codeblock">
```
['Collecting git+https://github.com/IDEA-Research/GroundingDINO.git',
 '  Cloning https://github.com/IDEA-Research/GroundingDINO.git to /tmp/pip-req-build-m_hhz04_',
 '  Running command git clone --filter=blob:none --quiet https://github.com/IDEA-Research/GroundingDINO.git /tmp/pip-req-build-m_hhz04_',
 '  Resolved https://github.com/IDEA-Research/GroundingDINO.git to commit 856dde20aee659246248e20734ef9ba5214f5e44',
 '  Preparing metadata (setup.py) ... \x1b[?25l\x1b[?25hdone',
 'Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from groundingdino==0.1.0) (2.4.1+cu121)',
 'Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (from groundingdino==0.1.0) (0.19.1+cu121)',
 'Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (from groundingdino==0.1.0) (4.44.2)',
 'Collecting addict (from groundingdino==0.1.0)',
 '  Downloading addict-2.4.0-py3-none-any.whl.metadata (1.0 kB)',
 'Collecting yapf (from groundingdino==0.1.0)',
 '  Downloading yapf-0.40.2-py3-none-any.whl.metadata (45 kB)',
 '\x1b[?25l     \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m0.0/45.4 kB\x1b[0m \x1b[31m?\x1b[0m eta \x1b[36m-:--:--\x1b[0m',
 '\x1b[2K     \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m45.4/45.4 kB\x1b[0m \x1b[31m1.8 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m',
 '\x1b[?25hCollecting timm (from groundingdino==0.1.0)',
 '  Downloading timm-1.0.9-py3-none-any.whl.metadata (42 kB)',
 '\x1b[?25l     \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m0.0/42.4 kB\x1b[0m \x1b[31m?\x1b[0m eta \x1b[36m-:--:--\x1b[0m',
 '\x1b[2K     \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m42.4/42.4 kB\x1b[0m \x1b[31m1.8 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m',
 '\x1b[?25hRequirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from groundingdino==0.1.0) (1.26.4)',
 'Requirement already satisfied: opencv-python in /usr/local/lib/python3.10/dist-packages (from groundingdino==0.1.0) (4.10.0.84)',
 'Collecting supervision>=0.22.0 (from groundingdino==0.1.0)',
 '  Downloading supervision-0.23.0-py3-none-any.whl.metadata (14 kB)',
 'Requirement already satisfied: pycocotools in /usr/local/lib/python3.10/dist-packages (from groundingdino==0.1.0) (2.0.8)',
 'Requirement already satisfied: defusedxml<0.8.0,>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from supervision>=0.22.0->groundingdino==0.1.0) (0.7.1)',
 'Requirement already satisfied: matplotlib>=3.6.0 in /usr/local/lib/python3.10/dist-packages (from supervision>=0.22.0->groundingdino==0.1.0) (3.7.1)',
 'Requirement already satisfied: opencv-python-headless>=4.5.5.64 in /usr/local/lib/python3.10/dist-packages (from supervision>=0.22.0->groundingdino==0.1.0) (4.10.0.84)',
 'Requirement already satisfied: pillow>=9.4 in /usr/local/lib/python3.10/dist-packages (from supervision>=0.22.0->groundingdino==0.1.0) (10.4.0)',
 'Requirement already satisfied: pyyaml>=5.3 in /usr/local/lib/python3.10/dist-packages (from supervision>=0.22.0->groundingdino==0.1.0) (6.0.2)',
 'Requirement already satisfied: scipy<2.0.0,>=1.10.0 in /usr/local/lib/python3.10/dist-packages (from supervision>=0.22.0->groundingdino==0.1.0) (1.13.1)',
 'Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.10/dist-packages (from timm->groundingdino==0.1.0) (0.24.7)',
 'Requirement already satisfied: safetensors in /usr/local/lib/python3.10/dist-packages (from timm->groundingdino==0.1.0) (0.4.5)',
 'Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch->groundingdino==0.1.0) (3.16.1)',
 'Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch->groundingdino==0.1.0) (4.12.2)',
 'Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch->groundingdino==0.1.0) (1.13.3)',
 'Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch->groundingdino==0.1.0) (3.3)',
 'Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->groundingdino==0.1.0) (3.1.4)',
 'Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch->groundingdino==0.1.0) (2024.6.1)',
 'Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers->groundingdino==0.1.0) (24.1)',
 'Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers->groundingdino==0.1.0) (2024.9.11)',
 'Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers->groundingdino==0.1.0) (2.32.3)',
 'Requirement already satisfied: tokenizers<0.20,>=0.19 in /usr/local/lib/python3.10/dist-packages (from transformers->groundingdino==0.1.0) (0.19.1)',
 'Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers->groundingdino==0.1.0) (4.66.5)',
 'Requirement already satisfied: importlib-metadata>=6.6.0 in /usr/local/lib/python3.10/dist-packages (from yapf->groundingdino==0.1.0) (8.4.0)',
 'Requirement already satisfied: platformdirs>=3.5.1 in /usr/local/lib/python3.10/dist-packages (from yapf->groundingdino==0.1.0) (4.3.6)',
 'Requirement already satisfied: tomli>=2.0.1 in /usr/local/lib/python3.10/dist-packages (from yapf->groundingdino==0.1.0) (2.0.1)',
 'Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata>=6.6.0->yapf->groundingdino==0.1.0) (3.20.2)',
 'Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6.0->supervision>=0.22.0->groundingdino==0.1.0) (1.3.0)',
 'Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6.0->supervision>=0.22.0->groundingdino==0.1.0) (0.12.1)',
 'Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6.0->supervision>=0.22.0->groundingdino==0.1.0) (4.54.1)',
 'Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6.0->supervision>=0.22.0->groundingdino==0.1.0) (1.4.7)',
 'Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6.0->supervision>=0.22.0->groundingdino==0.1.0) (3.1.4)',
 'Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6.0->supervision>=0.22.0->groundingdino==0.1.0) (2.8.2)',
 'Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->groundingdino==0.1.0) (2.1.5)',
 'Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->groundingdino==0.1.0) (3.3.2)',
 'Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->groundingdino==0.1.0) (3.10)',
 'Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->groundingdino==0.1.0) (2.2.3)',
 'Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers->groundingdino==0.1.0) (2024.8.30)',
 'Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch->groundingdino==0.1.0) (1.3.0)',
 'Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=3.6.0->supervision>=0.22.0->groundingdino==0.1.0) (1.16.0)',
 'Downloading supervision-0.23.0-py3-none-any.whl (151 kB)',
 '\x1b[?25l   \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m0.0/151.5 kB\x1b[0m \x1b[31m?\x1b[0m eta \x1b[36m-:--:--\x1b[0m',
 '\x1b[2K   \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m151.5/151.5 kB\x1b[0m \x1b[31m6.0 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m',
 '\x1b[?25hDownloading addict-2.4.0-py3-none-any.whl (3.8 kB)',
 'Downloading timm-1.0.9-py3-none-any.whl (2.3 MB)',
 '\x1b[?25l   \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m0.0/2.3 MB\x1b[0m \x1b[31m?\x1b[0m eta \x1b[36m-:--:--\x1b[0m',
 '\x1b[2K   \x1b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m\x1b[90m╺\x1b[0m\x1b[90m━━━━━━━\x1b[0m \x1b[32m1.9/2.3 MB\x1b[0m \x1b[31m55.9 MB/s\x1b[0m eta \x1b[36m0:00:01\x1b[0m',
 '\x1b[2K   \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m2.3/2.3 MB\x1b[0m \x1b[31m42.4 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m',
 '\x1b[?25hDownloading yapf-0.40.2-py3-none-any.whl (254 kB)',
 '\x1b[?25l   \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m0.0/254.7 kB\x1b[0m \x1b[31m?\x1b[0m eta \x1b[36m-:--:--\x1b[0m',
 '\x1b[2K   \x1b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m \x1b[32m254.7/254.7 kB\x1b[0m \x1b[31m18.3 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m',
 '\x1b[?25hBuilding wheels for collected packages: groundingdino',
 '  Building wheel for groundingdino (setup.py) ... \x1b[?25l\x1b[?25hdone',
 '  Created wheel for groundingdino: filename=groundingdino-0.1.0-cp310-cp310-linux_x86_64.whl size=3038498 sha256=1e7306dfa5ebd4bebb340bfe814e13026800708bbc0223d37ae8963e90145fb2',
 '  Stored in directory: /tmp/pip-ephem-wheel-cache-multbs74/wheels/6b/06/d7/b57f601a4df56af41d262a5b1b496359b13c323bf5ef0434b2',
 'Successfully built groundingdino',
 'Installing collected packages: addict, yapf, supervision, timm, groundingdino',
 'Successfully installed addict-2.4.0 groundingdino-0.1.0 supervision-0.23.0 timm-1.0.9 yapf-0.40.2']

[]

UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3609.)

final text_encoder_type: bert-base-uncased

UserWarning: 
Error while fetching `HF_TOKEN` secret value from your vault: 'Requesting secret HF_TOKEN timed out. Secrets can only be fetched when running from the Colab UI.'.
You are not authenticated with the Hugging Face Hub in this notebook.
If the error persists, please let us know by opening an issue on GitHub (https://github.com/huggingface/huggingface_hub/issues/new).

tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]

config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]

vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]

tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]

FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884

model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]

FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.

```
</div>
Let's load an image of a dog for this part!


```python
filepath = keras.utils.get_file(
    origin="https://storage.googleapis.com/keras-cv/test-images/mountain-dog.jpeg"
)
image = np.array(keras.utils.load_img(filepath))
image = ops.convert_to_numpy(inference_resizing(image))

plt.figure(figsize=(10, 10))
plt.imshow(image / 255.0)
plt.axis("on")
plt.show()
```

<div class="k-default-codeblock">
```
Downloading data from https://storage.googleapis.com/keras-cv/test-images/mountain-dog.jpeg
 1236492/1236492 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step

WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

```
</div>
    
![png](/img/guides/segment_anything_in_keras_hub/segment_anything_in_keras_hub_31_2.png)
    


We first predict the bounding box of the object we want to segment using the Grounding
DINO model. Then, we prompt the SAM model using the bounding box to get the segmentation
mask.

Let's try to segment out the harness of the dog. Change the image and text below to
segment whatever you want using text from your image!


```python
# Let's predict the bounding box for the harness of the dog
boxes = grounding_dino.predict_with_caption(image.astype(np.uint8), "harness")
boxes = np.array(boxes[0].xyxy)

outputs = model.predict(
    {
        "images": np.repeat(image[np.newaxis, ...], boxes.shape[0], axis=0),
        "boxes": boxes.reshape(-1, 1, 2, 2),
    },
    batch_size=1,
)
```

<div class="k-default-codeblock">
```
FutureWarning: The `device` argument is deprecated and will be removed in v5 of Transformers.
UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
UserWarning: None of the inputs have requires_grad=True. Gradients will be None
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.

 1/1 ━━━━━━━━━━━━━━━━━━━━ 10s 10s/step

```
</div>
And that's it! We got a segmentation mask for our text prompt using the combination of
Gounding DINO + SAM! This is a very powerful technique to combine different models to
expand the applications!

Let's visualize the results.


```python
plt.figure(figsize=(10, 10))
plt.imshow(image / 255.0)

for mask in outputs["masks"]:
    mask = inference_resizing(mask[0][..., None], pad=False)[..., 0]
    mask = ops.convert_to_numpy(mask) > 0.0
    show_mask(mask, plt.gca())
    show_box(boxes, plt.gca())

plt.axis("off")
plt.show()
```

<div class="k-default-codeblock">
```
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

```
</div>
    
![png](/img/guides/segment_anything_in_keras_hub/segment_anything_in_keras_hub_35_1.png)
    


---
## Optimizing SAM

You can use `mixed_float16` or `bfloat16` dtype policies to gain huge speedups and memory
optimizations at releatively low precision loss.


```python
# Load our image
image = np.array(keras.utils.load_img("truck.jpg"))
image = inference_resizing(image)

# Specify the prompt
input_box = np.array([[240, 340], [400, 500]])

# Let's first see how fast the model is with float32 dtype
time_taken = timeit.repeat(
    'model.predict({"images": image[np.newaxis, ...], "boxes": input_box[np.newaxis, np.newaxis, ...]}, verbose=False)',
    repeat=3,
    number=3,
    globals=globals(),
)
print(f"Time taken with float32 dtype: {min(time_taken) / 3:.10f}s")

# Set the dtype policy in Keras
keras.mixed_precision.set_global_policy("mixed_float16")

model = keras_hub.models.SAMImageSegmenter.from_preset("sam_huge_sa1b")

time_taken = timeit.repeat(
    'model.predict({"images": image[np.newaxis, ...], "boxes": input_box[np.newaxis,np.newaxis, ...]}, verbose=False)',
    repeat=3,
    number=3,
    globals=globals(),
)
print(f"Time taken with float16 dtype: {min(time_taken) / 3:.10f}s")
```

<div class="k-default-codeblock">
```
Time taken with float32 dtype: 0.2298811787s

UserWarning: Skipping variable loading for optimizer 'loss_scale_optimizer', because it has 4 variables whereas the saved optimizer has 2 variables. 
UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 0 variables. 

Time taken with float16 dtype: 0.2068303013s

```
</div>
Here's a comparison of KerasHub's implementation with the original PyTorch
implementation!

![benchmark](https://github.com/tirthasheshpatel/segment_anything_keras/blob/main/benchmark.png?raw=true)

The script used to generate the benchmarks is present
[here](https://github.com/tirthasheshpatel/segment_anything_keras/blob/main/Segment_Anything_Benchmarks.ipynb).

---
## Conclusion

KerasHub's `SegmentAnythingModel` supports a variety of applications and, with the help of
Keras 3, enables running the model on TensorFlow, JAX, and PyTorch! With the help of XLA
in JAX and TensorFlow, the model runs several times faster than the original
implementation. Moreover, using Keras's mixed precision support helps optimize memory use
and computation time with just one line of code!

For more advanced uses, check out the
[Automatic Mask Generator demo](https://github.com/tirthasheshpatel/segment_anything_keras/blob/main/Segment_Anything_Automatic_Mask_Generator_Demo.ipynb).

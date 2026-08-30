"""
Title: Two-Stage Instance Segmentation with RetinaNet and SAM
Author: [Abdulrahim Kasim](https://www.linkedin.com/in/kasim-abdulrahim/)
Date created: 2026/08/29
Last modified: 2026/08/29
Description: Two-stage instance segmentation assembled entirely from pretrained KerasHub.
Accelerator: GPU
"""

"""
## Introduction

Instance segmentation asks two questions of every image: *what* the objects are, and
*which* pixels belong to each one. In this example, we build that capability from
pretrained models in the two-stage pattern: an object detector predicts classes, boxes,
and confidence scores, and the Segment Anything Model (SAM) turns each predicted box into
an instance mask. Everything runs without training, the two presets below do all the
work:

- `retinanet_resnet50_fpn_v2_coco` - RetinaNet object detector (classes, boxes, scores).
- `sam_huge_sa1b` - Segment Anything Model (box-prompted masks).

### References

- [Grounded-DINO + Meta AI's SAM = Instance Segmentation](https://medium.com/@amir_shakiba/grounded-dino-meta-ais-sam-instance-segmentation-386240de4825)
- [Segment Anything in
KerasHub](https://keras.io/keras_hub/guides/segment_anything_in_keras_hub/)
"""

"""
## Setup
We select the Jax backend, but PyTorch or TensorFlow can also be used. The backend must
be selected before importing `keras`.
COCO class IDs are translated to names with KerasHub's built-in `coco_id_to_name` map.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os

import keras
from keras import ops
import keras_hub

"""
## Object Detection with a Pretrained RetinaNet

KerasHub's `ObjectDetector` task wraps preprocessing, a RetinaNet head over a
ResNet50-FPN backbone, and postprocessing behind a single `from_preset()` call.
"""

# predicts classes, boxes, scores
detector = keras_hub.models.ObjectDetector.from_preset(
    "retinanet_resnet50_fpn_v2_coco",
    bounding_box_format="rel_xywh",
)
# get an image
path = keras.utils.get_file(
    origin="https://deeplearningwithpython.io/images/ch11/fruits.8cef44dc.png"
)
pil_image = keras.utils.load_img(path)
image_array = keras.utils.img_to_array(pil_image)
height, width = image_array.shape[:2]
print("Image:", image_array.shape)

prediction = detector.predict(np.array([image_array]))
print({k: v.shape for k, v in prediction.items()})
print("num_detections:", int(prediction["num_detections"][0]))
print(
    "Top-6 labels:",
    [keras_hub.utils.coco_id_to_name(int(l)) for l in prediction["labels"][0][:6]],
)
print("Top-6 scores:", np.round(prediction["confidence"][0][:6], 3))

"""
We visualize the top predictions: every class gets a distinct hue chosen by the golden
ratio, boxes are drawn on the units square, and each label carries the class name and its
confidence score.
"""


def label_to_color(idx):
    golden_ratio = (1 + 5**0.5) / 2
    hue = (idx * golden_ratio) % 1.0
    # Convert HSV to RGB for vibrant, distinct colors.
    return colors.hsv_to_rgb([hue, 0.85, 0.95])


def draw_box(box, label=None, score=None):
    x, y, width, height = box
    ax = plt.gca()
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            width,
            height,
            facecolor="none",
            edgecolor=label_to_color(label),
            linewidth=2,
        )
    )
    if label is not None:
        ax.text(
            x + 0.005,
            y + 0.02,
            f"{keras_hub.utils.coco_id_to_name(label)} - {score:.2f}",
            color="k",
            fontsize=9,
            bbox={"facecolor": label_to_color(label), "alpha": 0.7},
        )


def draw_image(image, boxes=None, labels=None, scores=None):
    ax = plt.gca()
    ax.set_axis_off()
    ax.imshow(image.astype("uint8"), extent=(0, 1, 1, 0))
    for box, label, score in zip(boxes, labels, scores):
        draw_box(box, label, score)


n = int(prediction["num_detections"][0])
scores = prediction["confidence"][0][:n]
keep = np.where(scores > 0.5)[0][:6]
draw_image(
    image_array,
    prediction["boxes"][0][keep],
    prediction["labels"][0][keep],
    scores[keep],
)
plt.show()

"""
## Promptable Instance Segmentation with SAM

SAM inverts the usual setup: instead of being asked **what** to find, it is told
**where** to look and segments whatever is there. A box prompt is a `(1, 1, 2, 2)` array
of corner pixels `[[x0, y0], [x1, y1]]` on the model's 1024x1024 input, which we produce
with the `resize` function; the padding lands in the center. For each prompt, SAM returns
four mask candidates along with `iou_pred`, its own estimate of each mask's quality; we
keep the argmax.
"""

# Load pretrained SAM.
sam = keras_hub.models.ImageSegmenter.from_preset("sam_huge_sa1b")
target_size = (1024, 1024)
print("SAM parameters:", f"{sam.count_params():,}")


def resize_and_pad(x):
    return ops.image.resize(x, target_size, pad_to_aspect_ratio=True)


def show_image(image, title=None):
    plt.imshow(image.astype(int) if image.max() > 4 else image)
    plt.axis("off")
    if title is not None:
        plt.title(title, color="#6B7A8F")


def show_mask(mask, color="#0066FF", alpha=0.5):
    mask = np.asarray(mask)[..., None]
    color = colors.to_rgb(color) + (alpha,)
    plt.imshow(np.where(mask, color, [0, 0, 0, 0]))


def show_box(box):
    box = box.reshape(-1)
    plt.gca().add_patch(
        plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            edgecolor="#FF0000",
            linewidth=2,
        )
    )


"""
## Building the Instance Segmentation Pipeline

The pipeline chains the two models together. `detect()` runs RetinaNet and keeps every
detection above `BOX_THRESHOLD`, so labels come from the detector alone. `segment()`
converts each `rel_xywh` box to corner pixels on the padded square, accounts for the
center padding, and prompts SAM. `annotate()` draws per-class colored masks, boxes, and
"class confidence" labels on a single image.
"""

BOX_THRESHOLD = 0.35


def detect(image_array):
    prediction = detector.predict(np.array([image_array]))
    n = int(prediction["num_detections"][0])
    boxes = prediction["boxes"][0][:n]
    scores = prediction["confidence"][0][:n]
    names = [
        keras_hub.utils.coco_id_to_name(int(l)) for l in prediction["labels"][0][:n]
    ]
    keep = scores >= BOX_THRESHOLD
    return (boxes[keep], scores[keep], [name for name, k in zip(names, keep) if k])


def rel_to_prompt_box(box, height, width):
    """Convert a box in relative coordinates to a prompt box in absolute coordinates."""
    scale = target_size[0] / max(height, width)
    pad_x = (target_size[1] - width * scale) / 2
    pad_y = (target_size[0] - height * scale) / 2
    x, y, w, h = box
    return np.array(
        [
            [x * width * scale + pad_x, y * height * scale + pad_y],
            [(x + w) * width * scale + pad_x, (y + h) * height * scale + pad_y],
        ]
    )


def segment(image_padded, boxes, height, width):
    masks, ious = [], []
    for box in boxes:
        outputs = sam.predict(
            {
                "images": np.expand_dims(image_padded, axis=0),
                "boxes": np.expand_dims(
                    rel_to_prompt_box(box, height, width), axis=(0, 1)
                ),
            }
        )
        iou_pred = ops.convert_to_numpy(outputs["iou_pred"])[0]  # (4,)
        best = int(np.argmax(iou_pred))
        ious.append(float(iou_pred[best]))
        masks.append(
            ops.convert_to_numpy(
                resize_and_pad(
                    ops.convert_to_numpy(outputs["masks"][0][best])[..., None]
                )
            )[..., 0]
            > 0
        )
    return masks, ious


def mask_to_box(mask):
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def annotate(padded_image, boxes, names, scores, masks, height, width):
    show_image(padded_image)
    for i, (mask, box, name, score) in enumerate(zip(masks, boxes, names, scores)):
        color = colors.to_hex(label_to_color(i))
        show_mask(mask, color=color)
        show_box(rel_to_prompt_box(box, height, width))
        x0, y0 = mask_to_box(mask)[:2]
        plt.text(
            x0 + 4,
            max(0, y0 - 4, 8),
            f"{name} - {score:.2f}",
            color="k",
            fontsize=9,
            bbox={"facecolor": color, "alpha": 0.7},
        )
    plt.show()


def plot_mask(image, boxes, names, masks):
    for i, (mask, box, name) in enumerate(zip(masks, boxes, names)):
        plt.figure(figsize=(6, 6))
        show_image(image)
        color = colors.to_hex(label_to_color(i))
        show_mask(mask, color=color)
        show_box(rel_to_prompt_box(box, image.shape[0], image.shape[1]))
        x0, y0 = mask_to_box(mask)[:2]
        plt.text(
            x0 + 4,
            max(0, y0 - 4, 8),
            f"{name}",
            color="k",
            fontsize=9,
            bbox={"facecolor": color, "alpha": 0.7},
        )
        plt.title(f"Instance {i + 1}")
        plt.show()


def run_pipeline(image_array):
    """Run the two-stage detection and segmentation pipeline on an image."""
    height, width = image_array.shape[:2]
    padded_image = ops.convert_to_numpy(resize_and_pad(image_array))

    boxes, scores, names = detect(image_array)
    print(f"Detected {len(boxes)} above {BOX_THRESHOLD} confidence threshold.")
    for name, score in zip(names, scores):
        print(f"  {name}: {score:.3f}")

    masks, ious = segment(padded_image, boxes, height, width)
    for name, score, iou in zip(names, scores, ious):
        print(f"  {name}: {score:.3f}, IoU: {iou:.3f}")

    annotate(padded_image, boxes, names, scores, masks, height, width)

    plot_mask(padded_image, boxes, names, masks)

    return


run_pipeline(image_array)

"""
## Relevant chapters from Deep Learning with Python

- [Chapter 11: Image
segmentation](https://deeplearningwithpython.io/chapters/chapter11_image-segmentation/)
- [Chapter 12: Object
detection](https://deeplearningwithpython.io/chapters/chapter12_object-detection/)
"""

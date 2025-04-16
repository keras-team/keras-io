# Object Detection with KerasHub

**Authors:** [Siva Sravana Kumar Neeli](https://github.com/sineeli), [Sachin Prasad](https://github.com/sachinprasadhs)<br>
**Date created:** 2025/03/21<br>
**Last modified:** 2025/03/21<br>
**Description:** RetinaNet Object Detection: Training, Fine-tuning, and Inference.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/object_detection_retinanet.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/object_detection_retinanet.py)



![](https://storage.googleapis.com/keras-hub/getting_started_guide/prof_keras_intermediate.png)

---
## Introduction
Object detection is a crucial computer vision task that goes beyond simple image
classification. It requires models to not only identify the types of objects
present in an image but also pinpoint their locations using bounding boxes. This
dual requirement of classification and localization makes object detection a
more complex and powerful tool.
Object detection models are broadly classified into two categories: "two-stage"
and "single-stage" detectors. Two-stage detectors often achieve higher accuracy
by first proposing regions of interest and then classifying them. However, this
approach can be computationally expensive. Single-stage detectors, on the other
hand, aim for speed by directly predicting object classes and bounding boxes in
a single pass.

In this tutorial, we'll be diving into `RetinaNet`, a powerful object detection
model known for its speed and precision. `RetinaNet` is a single-stage detector,
a design choice that allows it to be remarkably efficient. Its impressive
performance stems from two key architectural innovations:
1. **Feature Pyramid Network (FPN):** FPN equips `RetinaNet` with the ability to
seamlessly detect objects of all scales, from distant, tiny instances to large,
prominent ones.
2. **Focal Loss:** This ingenious loss function tackles the common challenge of
imbalanced data by focusing the model's learning on the most crucial and
challenging object examples, leading to enhanced accuracy without compromising
speed.

![retinanet](/img/guides/object_detection_retinanet/retinanet_architecture.png)

### References
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

---
## Setup and Imports

Let's install the dependencies and import the necessary modules.

To run this tutorial, you will need to install the following packages:

* `keras-hub`
* `keras`
* `opencv-python`


```python
!pip install -q --upgrade keras-hub
!pip install -q --upgrade keras
!pip install -q opencv-python
```


```python
import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import keras_hub
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
```

### Helper functions


```python
# @title Helper functions
import logging
import multiprocessing
from builtins import open
import os.path
import xml
from typing import Callable, Tuple, Dict, Any

import tensorflow_datasets as tfds

VOC_2007_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
)
VOC_2012_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
)

# Note that this list doesn't contain the background class. In the
# classification use case, the label is 0 based (aeroplane -> 0), whereas in
# segmentation use case, the 0 is reserved for background, so aeroplane maps to
# 1.
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
COCO_90_CLASS_MAPPING = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}
# This is used to map between string class to index.
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
INDEX_TO_CLASS = {index: name for index, name in enumerate(CLASSES)}


def get_image_ids(data_dir, split):
    """To get image ids from the "train", "eval" or "trainval" files of VOC data."""
    data_file_mapping = {
        "train": "train.txt",
        "eval": "val.txt",
        "trainval": "trainval.txt",
    }
    with open(
        os.path.join(data_dir, "ImageSets", "Main", data_file_mapping[split]),
        "r",
    ) as f:
        image_ids = f.read().splitlines()
        logging.info(f"Received {len(image_ids)} images for {split} dataset.")
        return image_ids


def load_images(example):
    """Loads VOC images for segmentation task from the provided paths"""
    image_file_path = example.pop("image/file_path")
    image = tf.io.read_file(image_file_path)
    image = tf.image.decode_jpeg(image)

    example.update(
        {
            "image": image,
        }
    )
    return example


def parse_annotation_data(annotation_file_path):
    """Parse the annotation XML file for the image.

    The annotation contains the metadata, as well as the object bounding box
    information.

    """
    with open(annotation_file_path, "r") as f:
        root = xml.etree.ElementTree.parse(f).getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        filename = root.find("filename").text

        objects = []
        for obj in root.findall("object"):
            # Get object's label name.
            label = CLASS_TO_INDEX[obj.find("name").text.lower()]
            bndbox = obj.find("bndbox")
            xmax = int(float(bndbox.find("xmax").text))
            xmin = int(float(bndbox.find("xmin").text))
            ymax = int(float(bndbox.find("ymax").text))
            ymin = int(float(bndbox.find("ymin").text))
            objects.append(
                {
                    "label": label,
                    "bbox": [ymin, xmin, ymax, xmax],
                }
            )

        return {
            "image/filename": filename,
            "width": width,
            "height": height,
            "objects": objects,
        }


def parse_single_image(annotation_file_path):
    """Creates metadata of VOC images and path."""
    data_dir, annotation_file_name = os.path.split(annotation_file_path)
    data_dir = os.path.normpath(os.path.join(data_dir, os.path.pardir))
    image_annotations = parse_annotation_data(annotation_file_path)

    result = {
        "image/file_path": os.path.join(
            data_dir, "JPEGImages", image_annotations["image/filename"]
        )
    }
    result.update(image_annotations)
    # Labels field should be same as the 'object.label'
    labels = list(set([o["label"] for o in result["objects"]]))
    result["labels"] = sorted(labels)
    return result


def build_metadata(data_dir, image_ids):
    """Transpose the metadata which convert from list of dict to dict of list."""
    # Parallel process all the images.
    image_file_paths = [
        os.path.join(data_dir, "JPEGImages", i + ".jpg") for i in image_ids
    ]
    annotation_file_paths = tf.io.gfile.glob(
        os.path.join(data_dir, "Annotations", "*.xml")
    )
    pool_size = 10 if len(image_ids) > 10 else len(annotation_file_paths)
    with multiprocessing.Pool(pool_size) as p:
        metadata = p.map(parse_single_image, annotation_file_paths)

    keys = [
        "image/filename",
        "image/file_path",
        "labels",
        "width",
        "height",
    ]
    result = {}
    for key in keys:
        values = [value[key] for value in metadata]
        result[key] = values

    # The ragged objects need some special handling
    for key in ["label", "bbox"]:
        values = []
        objects = [value["objects"] for value in metadata]
        for object in objects:
            values.append([o[key] for o in object])
        result["objects/" + key] = values
    return result


def build_dataset_from_metadata(metadata):
    """Builds TensorFlow dataset from the image metadata of VOC dataset."""
    # The objects need some manual conversion to ragged tensor.
    metadata["labels"] = tf.ragged.constant(metadata["labels"])
    metadata["objects/label"] = tf.ragged.constant(metadata["objects/label"])
    metadata["objects/bbox"] = tf.ragged.constant(
        metadata["objects/bbox"], ragged_rank=1
    )

    dataset = tf.data.Dataset.from_tensor_slices(metadata)
    dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def load_voc(
    year="2007",
    split="trainval",
    data_dir="./",
    voc_url=VOC_2007_URL,
):
    extracted_dir = os.path.join("VOCdevkit", f"VOC{year}")
    get_data = keras.utils.get_file(
        fname=os.path.basename(voc_url),
        origin=voc_url,
        cache_dir=data_dir,
        extract=True,
    )
    data_dir = os.path.join(get_data, extracted_dir)
    image_ids = get_image_ids(data_dir, split)
    metadata = build_metadata(data_dir, image_ids)
    dataset = build_dataset_from_metadata(metadata)

    return dataset

```

---
## Load the dataset


```python
train_ds_2007 = load_voc(
    year="2007",
    split="trainval",
    data_dir="./",
    voc_url=VOC_2007_URL,
)
train_ds_2012 = load_voc(
    year="2012",
    split="trainval",
    data_dir="./",
    voc_url=VOC_2012_URL,
)
```

<div class="k-default-codeblock">
```
Downloading data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

```
</div>


<div class="k-default-codeblock">
```

```
</div>
 1999639040/1999639040 ━━━━━━━━━━━━━━━━━━━━ 68s 0us/step


---
## Inference using a pre-trained object detector

Let's begin with the simplest `KerasHub` API: a pre-trained object detector. In
this example, we will construct an object detector that was pre-trained on the
`COCO` dataset. We'll use this model to detect objects in a sample image.

The highest-level module in KerasHub is a `task`. A `task` is a `keras.Model`
consisting of a (generally pre-trained) backbone model and task-specific layers.
Here's an example using `keras_hub.models.ImageObjectDetector` with the
`RetinaNet` model architecture and `ResNet50` as the backbone.

ResNet is a great starting model when constructing an image classification
pipeline. This architecture manages to achieve high accuracy while using a
relatively small number of parameters. If a ResNet isn't powerful enough for the
task you are hoping to solve, be sure to check out KerasHub's other available
backbones here https://keras.io/keras_hub/presets/


```python
object_detector = keras_hub.models.ImageObjectDetector.from_preset(
    "kaggle://keras/retinanet/keras/retinanet_resnet50_fpn_coco"
)
object_detector.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Preprocessor: "retina_net_object_detector_preprocessor"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                                                  </span>┃<span style="font-weight: bold">                                   Config </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ retina_net_image_converter (<span style="color: #0087ff; text-decoration-color: #0087ff">RetinaNetImageConverter</span>)          │                   Image size: (<span style="color: #00af00; text-decoration-color: #00af00">800</span>, <span style="color: #00af00; text-decoration-color: #00af00">800</span>) │
└───────────────────────────────────────────────────────────────┴──────────────────────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "retina_net_object_detector"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                  </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">         Param # </span>┃<span style="font-weight: bold"> Connected to               </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ images (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)     │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ retina_net_backbone           │ [(<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>), │      <span style="color: #00af00; text-decoration-color: #00af00">27,429,824</span> │ images[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]               │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">RetinaNetBackbone</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>),  │                 │                            │
│                               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>),  │                 │                            │
│                               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>),  │                 │                            │
│                               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)]  │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ box_head (<span style="color: #0087ff; text-decoration-color: #0087ff">PredictionHead</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">2,443,300</span> │ retina_net_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>], │
│                               │                           │                 │ retina_net_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">1</span>], │
│                               │                           │                 │ retina_net_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">2</span>], │
│                               │                           │                 │ retina_net_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">3</span>], │
│                               │                           │                 │ retina_net_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">4</span>]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ classification_head           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">819</span>)   │       <span style="color: #00af00; text-decoration-color: #00af00">4,248,115</span> │ retina_net_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>], │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">PredictionHead</span>)              │                           │                 │ retina_net_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">1</span>], │
│                               │                           │                 │ retina_net_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">2</span>], │
│                               │                           │                 │ retina_net_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">3</span>], │
│                               │                           │                 │ retina_net_backbone[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">4</span>]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ box_pred_P3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ box_head[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ box_pred_P4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ box_head[<span style="color: #00af00; text-decoration-color: #00af00">1</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ box_pred_P5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ box_head[<span style="color: #00af00; text-decoration-color: #00af00">2</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ box_pred_P6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ box_head[<span style="color: #00af00; text-decoration-color: #00af00">3</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ box_pred_P7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ box_head[<span style="color: #00af00; text-decoration-color: #00af00">4</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ cls_pred_P3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">91</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ classification_head[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ cls_pred_P4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">91</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ classification_head[<span style="color: #00af00; text-decoration-color: #00af00">1</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ cls_pred_P5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">91</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ classification_head[<span style="color: #00af00; text-decoration-color: #00af00">2</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ cls_pred_P6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">91</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ classification_head[<span style="color: #00af00; text-decoration-color: #00af00">3</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ cls_pred_P7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Reshape</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">91</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ classification_head[<span style="color: #00af00; text-decoration-color: #00af00">4</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bbox_regression (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ box_pred_P3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│                               │                           │                 │ box_pred_P4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│                               │                           │                 │ box_pred_P5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│                               │                           │                 │ box_pred_P6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│                               │                           │                 │ box_pred_P7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ cls_logits (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">91</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ cls_pred_P3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│                               │                           │                 │ cls_pred_P4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│                               │                           │                 │ cls_pred_P5[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│                               │                           │                 │ cls_pred_P6[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],         │
│                               │                           │                 │ cls_pred_P7[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]          │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,121,239</span> (130.16 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,068,119</span> (129.96 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">53,120</span> (207.50 KB)
</pre>



---
## Preprocessing Layers


```python
image_size = (800, 800)
batch_size = 4
gt_bbox_format = "yxyx"
epochs = 5

# Resizing layer: Resizes images and pads to maintain aspect ratio
resizing = keras.layers.Resizing(
    height=image_size[0],
    width=image_size[1],
    interpolation="bilinear",
    pad_to_aspect_ratio=True,
    bounding_box_format=gt_bbox_format,
)

# Max bounding box layer: Limits the number of bounding boxes per image
max_box_layer = keras.layers.MaxNumBoundingBoxes(
    max_number=100, bounding_box_format=gt_bbox_format
)
```

### Predict and Visualize
Next, let's get some predictions from our object detector:


```python
filepath = keras.utils.get_file(
    origin="http://farm4.staticflickr.com/3755/10245052896_958cbf4766_z.jpg"
)
image = keras.utils.load_img(filepath)
image = keras.ops.cast(image, "float32")
image = keras.ops.expand_dims(image, axis=0)

predictions = object_detector.predict(image, batch_size=1)

keras.visualization.plot_bounding_box_gallery(
    resizing(image),  # resize image as per prediction preprocessing pipeline
    bounding_box_format=gt_bbox_format,
    y_pred=predictions,
    scale=4,
    class_mapping=COCO_90_CLASS_MAPPING,
)
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 8s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 8s 8s/step



    
![png](/home/sachinprasad/projects/KERAS-IO/keras-io/guides/img/object_detection_retinanet/object_detection_retinanet_14_2.png)
    


---
## Fine tuning a pretrained object detector
In this guide, we'll assemble a full training pipeline for a KerasHub `RetinaNet`
object detection model. This includes data loading, augmentation, training, and
inference using Pascal VOC 2007 & 2012 dataset!

---
## TFDS Preprocessing
This preprocessing step prepares the TFDS dataset for object detection. It
includes:
- Merging the Pascal VOC 2007 and 2012 datasets.
- Resizing all images to a resolution of 800x800 pixels.
- Limiting the number of bounding boxes per image to a maximum of 100.
- Finally, the resulting dataset is batched into sets of 4 images and bounding
box annotations.


```python

def decode_custom_tfds(record: Dict[str, Any]) -> Dict[str, Any]:
    """Decodes a custom TFDS record into a dictionary.

    Args:
      record: A dictionary representing a single TFDS record.

    Returns:
      A dictionary with "images" and "bounding_boxes".
    """
    image = record["image"]
    boxes = record["objects/bbox"]
    labels = record["objects/label"]

    bounding_boxes = {"boxes": boxes, "labels": labels}

    return {"images": image, "bounding_boxes": bounding_boxes}


def convert_to_tuple(record: Dict[str, Any]) -> Tuple[tf.Tensor, Dict[str, Any]]:
    """Converts a decoded TFDS record to a tuple for keras-hub.

    Args:
      record: A dictionary returned by `decode_custom_tfds` or `decode_tfds`.

    Returns:
      A tuple (image, bounding_boxes).
    """
    return record["images"], {
        "boxes": record["bounding_boxes"]["boxes"],
        "labels": record["bounding_boxes"]["labels"],
    }


def decode_tfds(record: Dict[str, Any]) -> Dict[str, Any]:
    """Decodes a standard TFDS object detection record.

    Args:
      record: A dictionary representing a single TFDS record.

    Returns:
      A dictionary with "images" and "bounding_boxes".
    """
    image = record["image"]
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]
    boxes = keras.utils.bounding_boxes.convert_format(
        record["objects"]["bbox"],
        source="rel_yxyx",
        target=gt_bbox_format,
        height=height,
        width=width,
    )
    labels = record["objects"]["label"]

    bounding_boxes = {"boxes": boxes, "labels": labels}

    return {"images": image, "bounding_boxes": bounding_boxes}


def preprocess_tfds(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Preprocesses a TFDS dataset for object detection.

    Args:
        ds: The TFDS dataset.
        resizing: A resizing function.
        max_box_layer: A max box processing function.
        batch_size: The batch size.

    Returns:
      A preprocessed TFDS dataset.
    """
    ds = ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(max_box_layer, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

```

Now concatenate both 2007 and 2012 VOC data


```python
train_ds = train_ds_2007.concatenate(train_ds_2012)
train_ds = train_ds.map(decode_custom_tfds, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = preprocess_tfds(train_ds)
```

Load the eval data


```python
eval_ds = tfds.load("voc/2007", split="test")
eval_ds = eval_ds.map(decode_tfds, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = preprocess_tfds(eval_ds)
```

### Let's visualize batch of training data


```python
record = next(iter(train_ds.shuffle(100).take(1)))
keras.visualization.plot_bounding_box_gallery(
    record["images"],
    bounding_box_format=gt_bbox_format,
    y_true=record["bounding_boxes"],
    scale=3,
    rows=2,
    cols=2,
    class_mapping=INDEX_TO_CLASS,
)
```


    
![png](/home/sachinprasad/projects/KERAS-IO/keras-io/guides/img/object_detection_retinanet/object_detection_retinanet_23_0.png)
    


### Decoded TFDS record to a tuple for keras-hub


```python
train_ds = train_ds.map(convert_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

eval_ds = eval_ds.map(convert_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)
```

---
## Congiure RetinaNet Model
Configure the model with backbone, num_classes and preprocessor.
Use callbacks for recording logs and saving checkpoints.


```python

def get_callbacks(experiment_path: str):
    """Creates a list of callbacks for model training.

    Args:
      experiment_path (str): Path to the experiment directory.

    Returns:
      List of keras callback instances.
    """
    # models_path = os.path.join(experiment_path, "tf_models")
    tb_logs_path = os.path.join(experiment_path, "logs")
    ckpt_path = os.path.join(experiment_path, "weights")
    return [
        keras.callbacks.BackupAndRestore(ckpt_path, delete_checkpoint=False),
        keras.callbacks.TensorBoard(
            tb_logs_path,
            update_freq=1,
        ),
        keras.callbacks.ModelCheckpoint(
            ckpt_path + "/{epoch:04d}-{val_loss:.2f}.weights.h5",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

```

---
## Load backbone weights and preprocessor config


```python
backbone = keras_hub.models.Backbone.from_preset("retinanet_resnet50_fpn_coco")

preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor.from_preset(
    "retinanet_resnet50_fpn_coco"
)
model = keras_hub.models.RetinaNetObjectDetector(
    backbone=backbone, num_classes=len(CLASSES), preprocessor=preprocessor
)
model.compile(box_loss=keras.losses.MeanAbsoluteError(reduction="sum"))
```

<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/retinanet/keras/retinanet_resnet50_fpn_coco/2/download/config.json...

```
</div>
    
  0%|                                                                                                                                      | 0.00/1.59k [00:00<?, ?B/s]

    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.59k/1.59k [00:00<00:00, 2.99MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/retinanet/keras/retinanet_resnet50_fpn_coco/2/download/model.weights.h5...

```
</div>
    
  0%|                                                                                                                                       | 0.00/105M [00:00<?, ?B/s]

    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 105M/105M [00:01<00:00, 62.3MB/s]

    

<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/retinanet/keras/retinanet_resnet50_fpn_coco/2/download/preprocessor.json...

```
</div>
    
  0%|                                                                                                                                      | 0.00/1.65k [00:00<?, ?B/s]

    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.65k/1.65k [00:00<00:00, 3.13MB/s]

    


---
## Fit the model

**Note:** The model trained on L4 GPU, while training on T4 it takes significant time.


```python
model.fit(
    train_ds,
    epochs=epochs,
    validation_data=eval_ds,
    callbacks=get_callbacks("fine_tuning"),
)
```

<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
    1/5534 [37m━━━━━━━━━━━━━━━━━━━━  66:07:36 43s/step - bbox_regression_loss: 37.8961 - cls_logits_loss: 8512.3418 - loss: 8550.2383

<div class="k-default-codeblock">
```

```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 0s 113ms/step - bbox_regression_loss: 0.8882 - cls_logits_loss: 53.9798 - loss: 54.8681

    
<div class="k-default-codeblock">
```
Epoch 1: val_loss improved from inf to 0.28274, saving model to fine_tuning/weights/0001-0.28.weights.h5


```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 702s 119ms/step - bbox_regression_loss: 0.8881 - cls_logits_loss: 53.9714 - loss: 54.8596 - val_bbox_regression_loss: 0.1683 - val_cls_logits_loss: 0.1144 - val_loss: 0.2827


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
    1/5534 [37m━━━━━━━━━━━━━━━━━━━━  42:23:38 28s/step - bbox_regression_loss: 0.1529 - cls_logits_loss: 0.1173 - loss: 0.2702

<div class="k-default-codeblock">
```

```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 0s 107ms/step - bbox_regression_loss: 0.1751 - cls_logits_loss: 0.1020 - loss: 0.2771

    
<div class="k-default-codeblock">
```
Epoch 2: val_loss improved from 0.28274 to 0.21916, saving model to fine_tuning/weights/0002-0.22.weights.h5


```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 649s 112ms/step - bbox_regression_loss: 0.1751 - cls_logits_loss: 0.1020 - loss: 0.2771 - val_bbox_regression_loss: 0.1352 - val_cls_logits_loss: 0.0840 - val_loss: 0.2192


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
    1/5534 [37m━━━━━━━━━━━━━━━━━━━━  39:35 429ms/step - bbox_regression_loss: 0.1188 - cls_logits_loss: 0.0763 - loss: 0.1950

<div class="k-default-codeblock">
```

```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 0s 107ms/step - bbox_regression_loss: 0.1408 - cls_logits_loss: 0.0733 - loss: 0.2141

    
<div class="k-default-codeblock">
```
Epoch 3: val_loss improved from 0.21916 to 0.20594, saving model to fine_tuning/weights/0003-0.21.weights.h5


```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 621s 112ms/step - bbox_regression_loss: 0.1408 - cls_logits_loss: 0.0733 - loss: 0.2141 - val_bbox_regression_loss: 0.1406 - val_cls_logits_loss: 0.0654 - val_loss: 0.2059


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
    1/5534 [37m━━━━━━━━━━━━━━━━━━━━  41:22 449ms/step - bbox_regression_loss: 0.1649 - cls_logits_loss: 0.0613 - loss: 0.2262

<div class="k-default-codeblock">
```

```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 0s 106ms/step - bbox_regression_loss: 0.1222 - cls_logits_loss: 0.0561 - loss: 0.1782

    
<div class="k-default-codeblock">
```
Epoch 4: val_loss improved from 0.20594 to 0.17254, saving model to fine_tuning/weights/0004-0.17.weights.h5


```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 620s 112ms/step - bbox_regression_loss: 0.1222 - cls_logits_loss: 0.0561 - loss: 0.1782 - val_bbox_regression_loss: 0.1171 - val_cls_logits_loss: 0.0554 - val_loss: 0.1725


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
    1/5534 [37m━━━━━━━━━━━━━━━━━━━━  43:03 467ms/step - bbox_regression_loss: 0.1184 - cls_logits_loss: 0.0503 - loss: 0.1687

<div class="k-default-codeblock">
```

```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 0s 106ms/step - bbox_regression_loss: 0.1118 - cls_logits_loss: 0.0435 - loss: 0.1553

    
<div class="k-default-codeblock">
```
Epoch 5: val_loss did not improve from 0.17254


```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 617s 112ms/step - bbox_regression_loss: 0.1118 - cls_logits_loss: 0.0435 - loss: 0.1553 - val_bbox_regression_loss: 0.1300 - val_cls_logits_loss: 0.0536 - val_loss: 0.1835





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f8c60443810>

```
</div>
### Prediction on evaluation data


```python
images, y_true = next(iter(eval_ds.shuffle(50).take(1)))
y_pred = model.predict(images)
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 7s 7s/step


### Plot the predictions


```python
keras.visualization.plot_bounding_box_gallery(
    images,
    bounding_box_format=gt_bbox_format,
    y_true=y_true,
    y_pred=y_pred,
    scale=3,
    rows=2,
    cols=2,
    class_mapping=INDEX_TO_CLASS,
)
```


    
![png](/home/sachinprasad/projects/KERAS-IO/keras-io/guides/img/object_detection_retinanet/object_detection_retinanet_35_0.png)
    


---
## Custom training object detector
Additionally, you can customize the object detector by modifying the image
converter, selecting a different image encoder, etc.

### Image Converter
The `RetinaNetImageConverter` class prepares images for use with the `RetinaNet`
object detection model. Here's what it does:

- Scaling and Offsetting
- ImageNet Normalization
- Resizing


```python
image_converter = keras_hub.layers.RetinaNetImageConverter(scale=1 / 255)

preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor(
    image_converter=image_converter
)
```

### Image Encoder and RetinaNet Backbone
The image encoder, while typically initialized with pre-trained weights
(e.g., from ImageNet), can also be instantiated without them. This results in
the image encoder (and, consequently, the entire object detection network built
upon it) having randomly initialized weights.

This configuration is equivalent to training the model from scratch, as opposed
to fine-tuning a pre-trained model.

Training from scratch generally requires significantly more data and
computational resources to achieve performance comparable to fine-tuning.

**Note:**
`use_p5`: If True, the output of the last backbone layer (typically `P5` in an
`FPN`) is used as input to create higher-level feature maps (e.g., `P6`, `P7`)
through additional convolutional layers. If `False`, the original `P5` feature
map from the backbone is directly used as input for creating the coarser levels,
bypassing any further processing of `P5` within the feature pyramid. Defaults to
`False`.


```python
# Load a pre-trained ResNet50 model.
# This will serve as the base for extracting image features.
image_encoder = keras_hub.models.Backbone.from_preset("resnet_50_imagenet")

# Build the RetinaNet Feature Pyramid Network (FPN) on top of the ResNet50
# backbone. The FPN creates multi-scale feature maps for better object detection
# at different sizes.
backbone = keras_hub.models.RetinaNetBackbone(
    image_encoder=image_encoder, min_level=3, max_level=5, use_p5=True
)
```

<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/resnetv1/keras/resnet_50_imagenet/3/download/config.json...

```
</div>
    
  0%|                                                                                                                                        | 0.00/841 [00:00<?, ?B/s]

    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 841/841 [00:00<00:00, 1.62MB/s]

    


<div class="k-default-codeblock">
```
Downloading from https://www.kaggle.com/api/v1/models/keras/resnetv1/keras/resnet_50_imagenet/3/download/model.weights.h5...

```
</div>
    
  0%|                                                                                                                                      | 0.00/90.3M [00:00<?, ?B/s]
    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90.3M/90.3M [00:01<00:00, 68.5MB/s]

    


### Train and visulaize RetinaNet model

**Note:** Training the model (for demonstration purposes only 5 epochs). In a
real scenario, you would train for many more epochs (often hundreds) to achieve
good results.


```python
model = keras_hub.models.RetinaNetObjectDetector(
    backbone=backbone,
    num_classes=len(CLASSES),
    preprocessor=preprocessor,
    use_prediction_head_norm=True,
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    box_loss=keras.losses.MeanAbsoluteError(reduction="sum"),
)

model.fit(
    train_ds,
    epochs=epochs,
    validation_data=eval_ds,
    callbacks=get_callbacks("custom_training"),
)

images, y_true = next(iter(eval_ds.shuffle(50).take(1)))
y_pred = model.predict(images)

keras.visualization.plot_bounding_box_gallery(
    images,
    bounding_box_format=gt_bbox_format,
    y_true=y_true,
    y_pred=y_pred,
    scale=3,
    rows=2,
    cols=2,
    class_mapping=INDEX_TO_CLASS,
)
```

<div class="k-default-codeblock">
```
Epoch 1/5

```
</div>
    
    1/5534 [37m━━━━━━━━━━━━━━━━━━━━  51:01:21 33s/step - bbox_regression_loss: 5.1737 - cls_logits_loss: 7534.1587 - loss: 7539.3325

<div class="k-default-codeblock">
```

```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step - bbox_regression_loss: 0.2589 - cls_logits_loss: 12.9746 - loss: 13.2335

    
<div class="k-default-codeblock">
```
Epoch 1: val_loss improved from inf to 0.24076, saving model to custom_training/weights/0001-0.24.weights.h5


```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 687s 118ms/step - bbox_regression_loss: 0.2589 - cls_logits_loss: 12.9725 - loss: 13.2314 - val_bbox_regression_loss: 0.1338 - val_cls_logits_loss: 0.1069 - val_loss: 0.2408


<div class="k-default-codeblock">
```
Epoch 2/5

```
</div>
    
    1/5534 [37m━━━━━━━━━━━━━━━━━━━━  46:54:20 31s/step - bbox_regression_loss: 0.1345 - cls_logits_loss: 0.1115 - loss: 0.2460

<div class="k-default-codeblock">
```

```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 0s 106ms/step - bbox_regression_loss: 0.1356 - cls_logits_loss: 0.1146 - loss: 0.2502

    
<div class="k-default-codeblock">
```
Epoch 2: val_loss improved from 0.24076 to 0.21794, saving model to custom_training/weights/0002-0.22.weights.h5


```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 647s 111ms/step - bbox_regression_loss: 0.1356 - cls_logits_loss: 0.1146 - loss: 0.2502 - val_bbox_regression_loss: 0.1187 - val_cls_logits_loss: 0.0992 - val_loss: 0.2179


<div class="k-default-codeblock">
```
Epoch 3/5

```
</div>
    
    1/5534 [37m━━━━━━━━━━━━━━━━━━━━  32:28 352ms/step - bbox_regression_loss: 0.1244 - cls_logits_loss: 0.0842 - loss: 0.2085

<div class="k-default-codeblock">
```

```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 0s 106ms/step - bbox_regression_loss: 0.1145 - cls_logits_loss: 0.1014 - loss: 0.2158

    
<div class="k-default-codeblock">
```
Epoch 3: val_loss improved from 0.21794 to 0.19651, saving model to custom_training/weights/0003-0.20.weights.h5


```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 615s 111ms/step - bbox_regression_loss: 0.1144 - cls_logits_loss: 0.1014 - loss: 0.2158 - val_bbox_regression_loss: 0.1089 - val_cls_logits_loss: 0.0876 - val_loss: 0.1965


<div class="k-default-codeblock">
```
Epoch 4/5

```
</div>
    
    1/5534 [37m━━━━━━━━━━━━━━━━━━━━  32:10 349ms/step - bbox_regression_loss: 0.1100 - cls_logits_loss: 0.0812 - loss: 0.1912

<div class="k-default-codeblock">
```

```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 0s 105ms/step - bbox_regression_loss: 0.1017 - cls_logits_loss: 0.0812 - loss: 0.1829

    
<div class="k-default-codeblock">
```
Epoch 4: val_loss improved from 0.19651 to 0.18126, saving model to custom_training/weights/0004-0.18.weights.h5


```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 614s 111ms/step - bbox_regression_loss: 0.1017 - cls_logits_loss: 0.0812 - loss: 0.1829 - val_bbox_regression_loss: 0.1012 - val_cls_logits_loss: 0.0801 - val_loss: 0.1813


<div class="k-default-codeblock">
```
Epoch 5/5

```
</div>
    
    1/5534 [37m━━━━━━━━━━━━━━━━━━━━  31:39 343ms/step - bbox_regression_loss: 0.0963 - cls_logits_loss: 0.0638 - loss: 0.1600

<div class="k-default-codeblock">
```

```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 0s 105ms/step - bbox_regression_loss: 0.0902 - cls_logits_loss: 0.0648 - loss: 0.1550

    
<div class="k-default-codeblock">
```
Epoch 5: val_loss improved from 0.18126 to 0.17179, saving model to custom_training/weights/0005-0.17.weights.h5


```
</div>
 5534/5534 ━━━━━━━━━━━━━━━━━━━━ 615s 111ms/step - bbox_regression_loss: 0.0902 - cls_logits_loss: 0.0648 - loss: 0.1550 - val_bbox_regression_loss: 0.0983 - val_cls_logits_loss: 0.0735 - val_loss: 0.1718


    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 7s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 7s 7s/step



    
![png](/home/sachinprasad/projects/KERAS-IO/keras-io/guides/img/object_detection_retinanet/object_detection_retinanet_41_27687.png)
    


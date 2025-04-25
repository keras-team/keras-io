"""
Title: Object Detection with KerasHub
Authors: [Siva Sravana Kumar Neeli](https://github.com/sineeli), [Sachin Prasad](https://github.com/sachinprasadhs)
Date created: 2025/04/28
Last modified: 2025/04/28
Description: RetinaNet Object Detection: Training, Fine-tuning, and Inference.
Accelerator: GPU
"""

"""
![](https://storage.googleapis.com/keras-hub/getting_started_guide/prof_keras_intermediate.png)

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
"""

"""
## Setup and Imports

Let's install the dependencies and import the necessary modules.

To run this tutorial, you will need to install the following packages:

* `keras-hub`
* `keras`
* `opencv-python`
"""

"""shell
pip install -q --upgrade keras-hub
pip install -q --upgrade keras
pip install -q opencv-python
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import keras_hub
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

"""
### Helper functions
We download the Pascal VOC 2012 and 2007 datasets using these helper functions,
prepare them for the object detection task, and split them into training and
validation datasets.
"""
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


"""
## Load the dataset
Let's load the training data. Here, we load both the VOC 2007 and 2012 datasets
and split them into training and validation sets.
"""
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

"""
## Inference using a pre-trained object detector

Let's begin with the simplest `KerasHub` API: a pre-trained object detector. In
this example, we will construct an object detector that was pre-trained on the
`COCO` dataset. We'll use this model to detect objects in a sample image.

The highest-level module in KerasHub is a `task`. A `task` is a `keras.Model`
consisting of a (generally pre-trained) backbone model and task-specific layers.
Here's an example using `keras_hub.models.ImageObjectDetector` with the
`RetinaNet` model architecture and `ResNet50` as the backbone.

`ResNet` is a great starting model when constructing an image classification
pipeline. This architecture manages to achieve high accuracy while using a
relatively small number of parameters. If a ResNet isn't powerful enough for the
task you are hoping to solve, be sure to check out KerasHub's other available
backbones here https://keras.io/keras_hub/presets/
"""

object_detector = keras_hub.models.ImageObjectDetector.from_preset(
    "retinanet_resnet50_fpn_coco"
)
object_detector.summary()

"""
## Preprocessing Layers

Let's define the below preprocessing layers:

- Resizing Layer: Resizes the image and maintains the aspect ratio by applying
padding when `pad_to_aspect_ratio=True`. Also, sets the default bounding box
format for representing the data.
- Max Bounding Box Layer: Limits the maximum number of bounding boxes per image.
"""
image_size = (800, 800)
batch_size = 4
gt_bbox_format = "yxyx"
epochs = 5

resizing = keras.layers.Resizing(
    height=image_size[0],
    width=image_size[1],
    interpolation="bilinear",
    pad_to_aspect_ratio=True,
    bounding_box_format=gt_bbox_format,
)

max_box_layer = keras.layers.MaxNumBoundingBoxes(
    max_number=100, bounding_box_format=gt_bbox_format
)

"""
### Predict and Visualize
Next, let's obtain predictions from our object detector by loading the image and
visualizing them. We'll apply the preprocessing pipeline defined in the
preprocessing layers step.
"""

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

"""
## Fine tuning a pretrained object detector

In this guide, we'll assemble a full training pipeline for a KerasHub `RetinaNet`
object detection model. This includes data loading, augmentation, training, and
inference using Pascal VOC 2007 & 2012 dataset!
"""

"""
## TFDS Preprocessing
This preprocessing step prepares the TFDS dataset for object detection. It
includes:
- Merging the Pascal VOC 2007 and 2012 datasets.
- Resizing all images to a resolution of 800x800 pixels.
- Limiting the number of bounding boxes per image to a maximum of 100.
- Finally, the resulting dataset is batched into sets of 4 images and bounding
box annotations.
"""


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


"""
Now concatenate both 2007 and 2012 VOC data
"""
train_ds = train_ds_2007.concatenate(train_ds_2012)
train_ds = train_ds.map(decode_custom_tfds, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = preprocess_tfds(train_ds)

"""
Load the eval data
"""
eval_ds = tfds.load("voc/2007", split="test")
eval_ds = eval_ds.map(decode_tfds, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = preprocess_tfds(eval_ds)

"""
### Let's visualize batch of training data
"""
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

"""
### Decoded TFDS record to a tuple for keras-hub
"""
train_ds = train_ds.map(convert_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

eval_ds = eval_ds.map(convert_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)

"""
## Configure RetinaNet Model
Configure the model with `backbone`, `num_classes` and `preprocessor`.
Use callbacks for recording logs and saving checkpoints.
"""


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


"""
## Load backbone weights and preprocessor config

Let's use the "retinanet_resnet50_fpn_coco" pretrained weights as the backbone
model, applying its predefined configuration from the preprocessor of the
"retinanet_resnet50_fpn_coco" preset.
Define a RetinaNet object detector model with the backbone and preprocessor
specified above, and set `num_classes` to 20 to represent the object categories
from Pascal VOC.
Finally, compile the model using Mean Absolute Error (MAE) as the box loss.
"""

backbone = keras_hub.models.Backbone.from_preset("retinanet_resnet50_fpn_coco")

preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor.from_preset(
    "retinanet_resnet50_fpn_coco"
)
model = keras_hub.models.RetinaNetObjectDetector(
    backbone=backbone, num_classes=len(CLASSES), preprocessor=preprocessor
)
model.compile(box_loss=keras.losses.MeanAbsoluteError(reduction="sum"))

"""
## Train the model
Now that the object detector model is compiled, let's train it using the
training and validation data we created earlier.
For demonstration purposes, we have used a small number of epochs. You can
increase the number of epochs to achieve better results.

**Note:** The model trained on L4 GPU, while training on T4 it takes
significant time.
"""

model.fit(
    train_ds,
    epochs=epochs,
    validation_data=eval_ds,
    callbacks=get_callbacks("fine_tuning"),
)

"""
### Prediction on evaluation data
Let's predict the model using our evaluation dataset.
"""
images, y_true = next(iter(eval_ds.shuffle(50).take(1)))
y_pred = model.predict(images)

"""
### Plot the predictions
"""
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

"""
## Custom training object detector
Additionally, you can customize the object detector by modifying the image
converter, selecting a different image encoder, etc.

### Image Converter
The `RetinaNetImageConverter` class prepares images for use with the `RetinaNet`
object detection model. Here's what it does:

- Scaling and Offsetting
- ImageNet Normalization
- Resizing
"""

image_converter = keras_hub.layers.RetinaNetImageConverter(scale=1 / 255)

preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor(
    image_converter=image_converter
)

"""
### Image Encoder and RetinaNet Backbone
The image encoder, while typically initialized with pre-trained weights
(e.g., from ImageNet), can also be instantiated without them. This results in
the image encoder (and, consequently, the entire object detection network built
upon it) having randomly initialized weights.

Here we load pre-trained ResNet50 model.
This will serve as the base for extracting image features.

And then Build the RetinaNet Feature Pyramid Network (FPN) on top of the ResNet50
backbone. The FPN creates multi-scale feature maps for better object detection
at different sizes.

**Note:**
`use_p5`: If True, the output of the last backbone layer (typically `P5` in an
`FPN`) is used as input to create higher-level feature maps (e.g., `P6`, `P7`)
through additional convolutional layers. If `False`, the original `P5` feature
map from the backbone is directly used as input for creating the coarser levels,
bypassing any further processing of `P5` within the feature pyramid. Defaults to
`False`.
"""

image_encoder = keras_hub.models.Backbone.from_preset("resnet_50_imagenet")

backbone = keras_hub.models.RetinaNetBackbone(
    image_encoder=image_encoder, min_level=3, max_level=5, use_p5=True
)

"""
### Train and visualize RetinaNet model

**Note:** Training the model (for demonstration purposes only 5 epochs). In a
real scenario, you would train for many more epochs (often hundreds) to achieve
good results.
"""
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

"""
## Conclusion
In this tutorial, you learned how to custom train and fine-tune the RetinaNet
object detector.

You can experiment with different existing backbones trained on ImageNet as the
image encoder, or you can fine-tune your own backbone.

This configuration is equivalent to training the model from scratch, as opposed
to fine-tuning a pre-trained model.

Training from scratch generally requires significantly more data and
computational resources to achieve performance comparable to fine-tuning.

To achieve better results when fine-tuning the model, you can increase the
number of epochs and experiment with different hyperparameter values.
In addition to the training data used here, you can also use other object
detection datasets, but keep in mind that custom training these requires
high GPU memory.
"""

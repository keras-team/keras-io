# Semantic Segmentation with KerasHub

**Authors:** [Sachin Prasad](https://github.com/sachinprasadhs), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli), [Ian Stenbit](https://github.com/ianstenbit)<br>
**Date created:** 2024/10/11<br>
**Last modified:** 2024/10/22<br>
**Description:** DeepLabV3 training and inference with KerasHub.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_hub/semantic_segmentation_deeplab_v3.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_hub/semantic_segmentation_deeplab_v3.py)



![](https://storage.googleapis.com/keras-hub/getting_started_guide/prof_keras_intermediate.png)

---
## Background
Semantic segmentation is a type of computer vision task that involves assigning a
class label such as "person", "bike", or "background" to each individual pixel
of an image, effectively dividing the image into regions that correspond to
different object classes or categories.

![](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*z6ch-2BliDGLIHpOPFY_Sw.png)



KerasHub offers the DeepLabv3, DeepLabv3+, SegFormer, etc., models for semantic
segmentation.

This guide demonstrates how to fine-tune and use the DeepLabv3+ model, developed
by Google for image semantic segmentation with KerasHub. Its architecture
combines Atrous convolutions, contextual information aggregation, and powerful
backbones to achieve accurate and detailed semantic segmentation.

DeepLabv3+ extends DeepLabv3 by adding a simple yet effective decoder module to
refine the segmentation results, especially along object boundaries. Both models
have achieved state-of-the-art results on a variety of image segmentation
benchmarks.

### References
[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

---
## Setup and Imports

Let's install the dependencies and import the necessary modules.

To run this tutorial, you will need to install the following packages:

* `keras-hub`
* `keras`


```python
!pip install -q --upgrade keras-hub
!pip install -q --upgrade keras
```

After installing `keras` and `keras-hub`, set the backend for `keras`.
This guide can be run with any backend (Tensorflow, JAX, PyTorch).


```python
import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import ops
import keras_hub
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
```

---
## Perform semantic segmentation with a pretrained DeepLabv3+ model

The highest level API in the KerasHub semantic segmentation API is the
`keras_hub.models` API. This API includes fully pretrained semantic segmentation
models, such as `keras_hub.models.DeepLabV3ImageSegmenter`.

Let's get started by constructing a DeepLabv3 pretrained on the Pascal VOC
dataset.
Also, define the preprocessing function for the model to preprocess images and
labels.
**Note:** By default `from_preset()` method in KerasHub loads the pretrained
task weights with all the classes, 21 classes in this case.


```python
model = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
    "deeplab_v3_plus_resnet50_pascalvoc"
)

image_converter = keras_hub.layers.DeepLabV3ImageConverter(
    image_size=(512, 512),
    interpolation="bilinear",
)
preprocessor = keras_hub.models.DeepLabV3ImageSegmenterPreprocessor(image_converter)
```

<div class="k-default-codeblock">
```
/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/saving/saving_lib.py:719: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 185 variables. 
  saveable.load_own_variables(weights_store.get(inner_path))

```
</div>
Let us visualize the results of this pretrained model


```python
filepath = keras.utils.get_file(
    origin="https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png"
)
image = keras.utils.load_img(filepath)
image = keras.utils.img_to_array(image)

image = preprocessor(image)
image = keras.ops.expand_dims(image, axis=0)
preds = ops.expand_dims(ops.argmax(model(image), axis=-1), axis=-1)


def plot_segmentation(original_image, predicted_mask):
    original_image = np.squeeze(original_image, axis=0)
    original_image = np.clip(original_image / 255.0, 0, 1)
    predicted_mask = np.squeeze(predicted_mask, axis=0)
    plt.figure(figsize=(5, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


plot_segmentation(image, preds)
```

<div class="k-default-codeblock">
```
/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/models/functional.py:225: UserWarning: The structure of `inputs` doesn't match the expected structure: ['keras_tensor']. Received: the structure of inputs=*
  warnings.warn(

```
</div>
    
![png](/home/sachinprasad/projects/keras-io/guides/img/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_9_1.png)
    


---
## Train a custom semantic segmentation model
In this guide, we'll assemble a full training pipeline for a KerasHub DeepLabV3
semantic segmentation model. This includes data loading, augmentation, training,
metric evaluation, and inference!

---
## Download the data

We download Pascal VOC 2012 dataset with additional annotations provided here
[Semantic contours from inverse detectors](https://ieeexplore.ieee.org/document/6126343)
and split them into train dataset `train_ds` and `eval_ds`.


```python
# @title helper functions
import logging
import multiprocessing
from builtins import open
import os.path
import random
import xml

import tensorflow_datasets as tfds

VOC_URL = "https://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

SBD_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"

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
# This is used to map between string class to index.
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}

# For the mask data in the PNG file, the encoded raw pixel value need to be
# converted to the proper class index. In the following map, [0, 0, 0] will be
# convert to 0, and [128, 0, 0] will be converted to 1, so on so forth. Also
# note that the mask class is 1 base since class 0 is reserved for the
# background. The [128, 0, 0] (class 1) is mapped to `aeroplane`.
VOC_PNG_COLOR_VALUE = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]
# Will be populated by maybe_populate_voc_color_mapping() below.
VOC_PNG_COLOR_MAPPING = None


def maybe_populate_voc_color_mapping():
    """Lazy creation of VOC_PNG_COLOR_MAPPING, which could take 64M memory."""
    global VOC_PNG_COLOR_MAPPING
    if VOC_PNG_COLOR_MAPPING is None:
        VOC_PNG_COLOR_MAPPING = [0] * (256**3)
        for i, colormap in enumerate(VOC_PNG_COLOR_VALUE):
            VOC_PNG_COLOR_MAPPING[
                (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
            ] = i
        # There is a special mapping with [224, 224, 192] -> 255
        VOC_PNG_COLOR_MAPPING[224 * 256 * 256 + 224 * 256 + 192] = 255
        VOC_PNG_COLOR_MAPPING = tf.constant(VOC_PNG_COLOR_MAPPING)
    return VOC_PNG_COLOR_MAPPING


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

        objects = []
        for obj in root.findall("object"):
            # Get object's label name.
            label = CLASS_TO_INDEX[obj.find("name").text.lower()]
            # Get objects' pose name.
            pose = obj.find("pose").text.lower()
            is_truncated = obj.find("truncated").text == "1"
            is_difficult = obj.find("difficult").text == "1"
            bndbox = obj.find("bndbox")
            xmax = int(bndbox.find("xmax").text)
            xmin = int(bndbox.find("xmin").text)
            ymax = int(bndbox.find("ymax").text)
            ymin = int(bndbox.find("ymin").text)
            objects.append(
                {
                    "label": label,
                    "pose": pose,
                    "bbox": [ymin, xmin, ymax, xmax],
                    "is_truncated": is_truncated,
                    "is_difficult": is_difficult,
                }
            )

        return {"width": width, "height": height, "objects": objects}


def get_image_ids(data_dir, split):
    """To get image ids from the "train", "eval" or "trainval" files of VOC data."""
    data_file_mapping = {
        "train": "train.txt",
        "eval": "val.txt",
        "trainval": "trainval.txt",
    }
    with open(
        os.path.join(data_dir, "ImageSets", "Segmentation", data_file_mapping[split]),
        "r",
    ) as f:
        image_ids = f.read().splitlines()
        logging.info(f"Received {len(image_ids)} images for {split} dataset.")
        return image_ids


def get_sbd_image_ids(data_dir, split):
    """To get image ids from the "sbd_train", "sbd_eval" from files of SBD data."""
    data_file_mapping = {"sbd_train": "train.txt", "sbd_eval": "val.txt"}
    with open(
        os.path.join(data_dir, data_file_mapping[split]),
        "r",
    ) as f:
        image_ids = f.read().splitlines()
        logging.info(f"Received {len(image_ids)} images for {split} dataset.")
        return image_ids


def parse_single_image(image_file_path):
    """Creates metadata of VOC images and path."""
    data_dir, image_file_name = os.path.split(image_file_path)
    data_dir = os.path.normpath(os.path.join(data_dir, os.path.pardir))
    image_id, _ = os.path.splitext(image_file_name)
    class_segmentation_file_path = os.path.join(
        data_dir, "SegmentationClass", image_id + ".png"
    )
    object_segmentation_file_path = os.path.join(
        data_dir, "SegmentationObject", image_id + ".png"
    )
    annotation_file_path = os.path.join(data_dir, "Annotations", image_id + ".xml")
    image_annotations = parse_annotation_data(annotation_file_path)

    result = {
        "image/filename": image_id + ".jpg",
        "image/file_path": image_file_path,
        "segmentation/class/file_path": class_segmentation_file_path,
        "segmentation/object/file_path": object_segmentation_file_path,
    }
    result.update(image_annotations)
    # Labels field should be same as the 'object.label'
    labels = list(set([o["label"] for o in result["objects"]]))
    result["labels"] = sorted(labels)
    return result


def parse_single_sbd_image(image_file_path):
    """Creates metadata of SBD images and path."""
    data_dir, image_file_name = os.path.split(image_file_path)
    data_dir = os.path.normpath(os.path.join(data_dir, os.path.pardir))
    image_id, _ = os.path.splitext(image_file_name)
    class_segmentation_file_path = os.path.join(data_dir, "cls", image_id + ".mat")
    object_segmentation_file_path = os.path.join(data_dir, "inst", image_id + ".mat")
    result = {
        "image/filename": image_id + ".jpg",
        "image/file_path": image_file_path,
        "segmentation/class/file_path": class_segmentation_file_path,
        "segmentation/object/file_path": object_segmentation_file_path,
    }
    return result


def build_metadata(data_dir, image_ids):
    """Transpose the metadata which convert from list of dict to dict of list."""
    # Parallel process all the images.
    image_file_paths = [
        os.path.join(data_dir, "JPEGImages", i + ".jpg") for i in image_ids
    ]
    pool_size = 10 if len(image_ids) > 10 else len(image_ids)
    with multiprocessing.Pool(pool_size) as p:
        metadata = p.map(parse_single_image, image_file_paths)

    keys = [
        "image/filename",
        "image/file_path",
        "segmentation/class/file_path",
        "segmentation/object/file_path",
        "labels",
        "width",
        "height",
    ]
    result = {}
    for key in keys:
        values = [value[key] for value in metadata]
        result[key] = values

    # The ragged objects need some special handling
    for key in ["label", "pose", "bbox", "is_truncated", "is_difficult"]:
        values = []
        objects = [value["objects"] for value in metadata]
        for object in objects:
            values.append([o[key] for o in object])
        result["objects/" + key] = values
    return result


def build_sbd_metadata(data_dir, image_ids):
    """Transpose the metadata which convert from list of dict to dict of list."""
    # Parallel process all the images.
    image_file_paths = [os.path.join(data_dir, "img", i + ".jpg") for i in image_ids]
    pool_size = 10 if len(image_ids) > 10 else len(image_ids)
    with multiprocessing.Pool(pool_size) as p:
        metadata = p.map(parse_single_sbd_image, image_file_paths)

    keys = [
        "image/filename",
        "image/file_path",
        "segmentation/class/file_path",
        "segmentation/object/file_path",
    ]
    result = {}
    for key in keys:
        values = [value[key] for value in metadata]
        result[key] = values
    return result


def decode_png_mask(mask):
    """Decode the raw PNG image and convert it to 2D tensor with probably
    class."""
    # Cast the mask to int32 since the original uint8 will overflow when
    # multiplied with 256
    mask = tf.cast(mask, tf.int32)
    mask = mask[:, :, 0] * 256 * 256 + mask[:, :, 1] * 256 + mask[:, :, 2]
    mask = tf.expand_dims(tf.gather(VOC_PNG_COLOR_MAPPING, mask), -1)
    mask = tf.cast(mask, tf.uint8)
    return mask


def load_images(example):
    """Loads VOC images for segmentation task from the provided paths"""
    image_file_path = example.pop("image/file_path")
    segmentation_class_file_path = example.pop("segmentation/class/file_path")
    segmentation_object_file_path = example.pop("segmentation/object/file_path")
    image = tf.io.read_file(image_file_path)
    image = tf.image.decode_jpeg(image)

    segmentation_class_mask = tf.io.read_file(segmentation_class_file_path)
    segmentation_class_mask = tf.image.decode_png(segmentation_class_mask)
    segmentation_class_mask = decode_png_mask(segmentation_class_mask)

    segmentation_object_mask = tf.io.read_file(segmentation_object_file_path)
    segmentation_object_mask = tf.image.decode_png(segmentation_object_mask)
    segmentation_object_mask = decode_png_mask(segmentation_object_mask)

    example.update(
        {
            "image": image,
            "class_segmentation": segmentation_class_mask,
            "object_segmentation": segmentation_object_mask,
        }
    )
    return example


def load_sbd_images(image_file_path, seg_cls_file_path, seg_obj_file_path):
    """Loads SBD images for segmentation task from the provided paths"""
    image = tf.io.read_file(image_file_path)
    image = tf.image.decode_jpeg(image)

    segmentation_class_mask = tfds.core.lazy_imports.scipy.io.loadmat(seg_cls_file_path)
    segmentation_class_mask = segmentation_class_mask["GTcls"]["Segmentation"][0][0]
    segmentation_class_mask = segmentation_class_mask[..., np.newaxis]

    segmentation_object_mask = tfds.core.lazy_imports.scipy.io.loadmat(
        seg_obj_file_path
    )
    segmentation_object_mask = segmentation_object_mask["GTinst"]["Segmentation"][0][0]
    segmentation_object_mask = segmentation_object_mask[..., np.newaxis]

    return {
        "image": image,
        "class_segmentation": segmentation_class_mask,
        "object_segmentation": segmentation_object_mask,
    }


def build_dataset_from_metadata(metadata):
    """Builds TensorFlow dataset from the image metadata of VOC dataset."""
    # The objects need some manual conversion to ragged tensor.
    metadata["labels"] = tf.ragged.constant(metadata["labels"])
    metadata["objects/label"] = tf.ragged.constant(metadata["objects/label"])
    metadata["objects/pose"] = tf.ragged.constant(metadata["objects/pose"])
    metadata["objects/is_truncated"] = tf.ragged.constant(
        metadata["objects/is_truncated"]
    )
    metadata["objects/is_difficult"] = tf.ragged.constant(
        metadata["objects/is_difficult"]
    )
    metadata["objects/bbox"] = tf.ragged.constant(
        metadata["objects/bbox"], ragged_rank=1
    )

    dataset = tf.data.Dataset.from_tensor_slices(metadata)
    dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def build_sbd_dataset_from_metadata(metadata):
    """Builds TensorFlow dataset from the image metadata of SBD dataset."""
    img_filepath = metadata["image/file_path"]
    cls_filepath = metadata["segmentation/class/file_path"]
    obj_filepath = metadata["segmentation/object/file_path"]

    def md_gen():
        c = list(zip(img_filepath, cls_filepath, obj_filepath))
        # random shuffling for each generator boosts up the quality.
        random.shuffle(c)
        for fp in c:
            img_fp, cls_fp, obj_fp = fp
            yield load_sbd_images(img_fp, cls_fp, obj_fp)

    dataset = tf.data.Dataset.from_generator(
        md_gen,
        output_signature=(
            {
                "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                "class_segmentation": tf.TensorSpec(
                    shape=(None, None, 1), dtype=tf.uint8
                ),
                "object_segmentation": tf.TensorSpec(
                    shape=(None, None, 1), dtype=tf.uint8
                ),
            }
        ),
    )

    return dataset


def load(
    split="sbd_train",
    data_dir=None,
):
    """Load the Pacal VOC 2012 dataset.

    This function will download the data tar file from remote if needed, and
    untar to the local `data_dir`, and build dataset from it.

    It supports both VOC2012 and Semantic Boundaries Dataset (SBD).

    The returned segmentation masks will be int ranging from [0, num_classes),
    as well as 255 which is the boundary mask.

    Args:
        split: string, can be 'train', 'eval', 'trainval', 'sbd_train', or
            'sbd_eval'. 'sbd_train' represents the training dataset for SBD
            dataset, while 'train' represents the training dataset for VOC2012
            dataset. Defaults to `sbd_train`.
        data_dir: string, local directory path for the loaded data. This will be
            used to download the data file, and unzip. It will be used as a
            cache directory. Defaults to None, and `~/.keras/pascal_voc_2012`
            will be used.
    """
    supported_split_value = [
        "train",
        "eval",
        "trainval",
        "sbd_train",
        "sbd_eval",
    ]
    if split not in supported_split_value:
        raise ValueError(
            f"The support value for `split` are {supported_split_value}. "
            f"Got: {split}"
        )

    if data_dir is not None:
        data_dir = os.path.expanduser(data_dir)

    if "sbd" in split:
        return load_sbd(split, data_dir)
    else:
        return load_voc(split, data_dir)


def load_voc(
    split="train",
    data_dir=None,
):
    """This function will download VOC data from a URL. If the data is already
    present in the cache directory, it will load the data from that directory
    instead.
    """
    extracted_dir = os.path.join("VOCdevkit", "VOC2012")
    get_data = keras.utils.get_file(
        fname=os.path.basename(VOC_URL),
        origin=VOC_URL,
        cache_dir=data_dir,
        extract=True,
    )
    data_dir = os.path.join(os.path.dirname(get_data), extracted_dir)
    image_ids = get_image_ids(data_dir, split)
    # len(metadata) = #samples, metadata[i] is a dict.
    metadata = build_metadata(data_dir, image_ids)
    maybe_populate_voc_color_mapping()
    dataset = build_dataset_from_metadata(metadata)

    return dataset


def load_sbd(
    split="sbd_train",
    data_dir=None,
):
    """This function will download SBD data from a URL. If the data is already
    present in the cache directory, it will load the data from that directory
    instead.
    """
    extracted_dir = os.path.join("benchmark_RELEASE", "dataset")
    # get_data = keras.utils.get_file(
    #     fname=os.path.basename(SBD_URL),
    #     origin=SBD_URL,
    #     cache_dir=data_dir,
    #     extract=True,
    # )
    # data_dir = os.path.join(os.path.dirname(get_data), extracted_dir)
    data_dir = os.path.join("/home/sachinprasad/projects/", extracted_dir)
    image_ids = get_sbd_image_ids(data_dir, split)
    # len(metadata) = #samples, metadata[i] is a dict.
    metadata = build_sbd_metadata(data_dir, image_ids)

    dataset = build_sbd_dataset_from_metadata(metadata)
    return dataset

```

---
## Load the dataset

For training and evaluation, let's use "sbd_train" and "sbd_eval." You can also
choose any of these datasets for the `load` function: 'train', 'eval', 'trainval',
'sbd_train', or 'sbd_eval'. 'sbd_train' represents the training dataset for the
SBD dataset, while 'train' represents the training dataset for the VOC2012 dataset.


```python
train_ds = load(split="sbd_train", data_dir="segmentation")
eval_ds = load(split="sbd_eval", data_dir="segmentation")
```

<div class="k-default-codeblock">
```
/usr/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()

```
</div>
---
## Preprocess the data

The preprocess_inputs utility function preprocesses inputs, converting them into
a dictionary containing images and segmentation_masks. Both images and
segmentation masks are resized to 512x512. The resulting dataset is then batched
into groups of four image and segmentation mask pairs.


```python

def preprocess_inputs(inputs):
    def unpackage_inputs(inputs):
        return {
            "images": inputs["image"],
            "segmentation_masks": inputs["class_segmentation"],
        }

    outputs = inputs.map(unpackage_inputs)
    outputs = outputs.map(keras.layers.Resizing(height=512, width=512))
    outputs = outputs.batch(4, drop_remainder=True)
    return outputs


train_ds = preprocess_inputs(train_ds)
batch = train_ds.take(1).get_single_element()
```

A batch of this preprocessed input training data can be visualized using the
`plot_images_masks` function. This function takes a batch of images and
segmentation masks and prediction masks as input and displays them in a grid.


```python

def plot_images_masks(images, masks, pred_masks=None):
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    masks = (masks - np.min(masks)) / (np.max(masks) - np.min(masks))
    if pred_masks is not None:
        pred_masks = (pred_masks - pred_masks.min()) / (
            pred_masks.max() - pred_masks.min()
        )
    num_images = len(images)
    plt.figure(figsize=(8, 4))
    rows = 3 if pred_masks is not None else 2

    for i in range(num_images):
        plt.subplot(rows, num_images, i + 1)
        plt.imshow(images[i])
        plt.axis("off")

        plt.subplot(rows, num_images, num_images + i + 1)
        plt.imshow(masks[i])
        plt.axis("off")

        if pred_masks is not None:
            plt.subplot(rows, num_images, i + 1 + 2 * num_images)
            plt.imshow(pred_masks[i, ..., 0])
            plt.axis("off")

    plt.show()


plot_images_masks(batch["images"], batch["segmentation_masks"])
```


    
![png](/home/sachinprasad/projects/keras-io/guides/img/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_18_0.png)
    


The preprocessing is applied to the evaluation dataset `eval_ds`.


```python
eval_ds = preprocess_inputs(eval_ds)
```

---
## Data Augmentation

Keras provides a variety of image augmentation options. In this example, we will
use the `RandomFlip` augmentation to augment the training dataset. The
`RandomFlip` augmentation randomly flips the images in the training dataset
horizontally or vertically. This can help to improve the model's robustness to
changes in the orientation of the objects in the images.


```python
train_ds = train_ds.map(keras.layers.RandomFlip())
batch = train_ds.take(1).get_single_element()

plot_images_masks(batch["images"], batch["segmentation_masks"])
```


    
![png](/home/sachinprasad/projects/keras-io/guides/img/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_22_0.png)
    


---
## Model Configuration

Please feel free to modify the configurations for model training and note how the
training results changes. This is an great exercise to get a better
understanding of the training pipeline.

The learning rate schedule is used by the optimizer to calculate the learning
rate for each epoch. The optimizer then uses the learning rate to update the
weights of the model.
In this case, the learning rate schedule uses a cosine decay function. A cosine
decay function starts high and then decreases over time, eventually reaching
zero. The cardinality of the VOC dataset is 2124 with a batch size of 4. The
dataset cardinality is important for learning rate decay because it determines
how many steps the model will train for. The initial learning rate is
proportional to 0.007 and the decay steps are 2124. This means that the learning
rate will start at `INITIAL_LR` and then decrease to zero over 2124 steps.
![png](/img/guides/semantic_segmentation_deeplab_v3_plus/learning_rate_schedule.png)


```python
BATCH_SIZE = 4
INITIAL_LR = 0.007 * BATCH_SIZE / 16
EPOCHS = 1
NUM_CLASSES = 21
learning_rate = keras.optimizers.schedules.CosineDecay(
    INITIAL_LR,
    decay_steps=EPOCHS * 2124,
)
```

Let's take the `resnet_50_imagenet` pretrained weights as a image encoder for
the model, this implementation can be used both as DeepLabV3 and DeepLabV3+ with
additional decoder block.
For DeepLabV3+, we instantiate a DeepLabV3Backbone model by providing
`low_level_feature_key` as `P2` a pyramid level output to extract features from
`resnet_50_imagenet` which acts as a decoder block.
To use this model as DeepLabV3 architecture, ignore the `low_level_feature_key`
which defaults to `None`.

Then we create DeepLabV3ImageSegmenter instance.
The `num_classes` parameter specifies the number of classes that the model will
be trained to segment. `preprocessor`  argument to apply preprocessing to image
input and masks.


```python
image_encoder = keras_hub.models.Backbone.from_preset("resnet_50_imagenet")

deeplab_backbone = keras_hub.models.DeepLabV3Backbone(
    image_encoder=image_encoder,
    low_level_feature_key="P2",
    spatial_pyramid_pooling_key="P5",
    dilation_rates=[6, 12, 18],
    upsampling_size=8,
)

model = keras_hub.models.DeepLabV3ImageSegmenter(
    backbone=deeplab_backbone,
    num_classes=21,
    activation="softmax",
    preprocessor=preprocessor,
)
```

---
## Compile the model

The model.compile() function sets up the training process for the model. It defines the
- optimization algorithm - Stochastic Gradient Descent (SGD)
- the loss function - categorical cross-entropy
- the evaluation metrics - Mean IoU and categorical accuracy

Semantic segmentation evaluation metrics:

Mean Intersection over Union (MeanIoU):
MeanIoU measures how well a semantic segmentation model accurately identifies
and delineates different objects or regions in an image. It calculates the
overlap between predicted and actual object boundaries, providing a score
between 0 and 1, where 1 represents a perfect match.

Categorical Accuracy:
Categorical Accuracy measures the proportion of correctly classified pixels in
an image. It gives a simple percentage indicating how accurately the model
predicts the categories of pixels in the entire image.

In essence, MeanIoU emphasizes the accuracy of identifying specific object
boundaries, while Categorical Accuracy gives a broad overview of overall
pixel-level correctness.


```python
model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=learning_rate, weight_decay=0.0001, momentum=0.9, clipnorm=10.0
    ),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[
        keras.metrics.MeanIoU(
            num_classes=NUM_CLASSES, sparse_y_true=False, sparse_y_pred=False
        ),
        keras.metrics.CategoricalAccuracy(),
    ],
)

model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Preprocessor: "deep_lab_v3_image_segmenter_preprocessor"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                                                  </span>┃<span style="font-weight: bold">                                   Config </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ deep_lab_v3_image_converter (<span style="color: #0087ff; text-decoration-color: #0087ff">DeepLabV3ImageConverter</span>)         │                   Image size: (<span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>) │
└───────────────────────────────────────────────────────────────┴──────────────────────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "deep_lab_v3_image_segmenter"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                                  </span>┃<span style="font-weight: bold"> Output Shape                       </span>┃<span style="font-weight: bold">             Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ inputs (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)                           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │                   <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├───────────────────────────────────────────────┼────────────────────────────────────┼─────────────────────┤
│ deep_lab_v3_backbone (<span style="color: #0087ff; text-decoration-color: #0087ff">DeepLabV3Backbone</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │          <span style="color: #00af00; text-decoration-color: #00af00">39,190,656</span> │
├───────────────────────────────────────────────┼────────────────────────────────────┼─────────────────────┤
│ segmentation_output (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">21</span>)             │               <span style="color: #00af00; text-decoration-color: #00af00">5,376</span> │
└───────────────────────────────────────────────┴────────────────────────────────────┴─────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">39,196,032</span> (149.52 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">39,139,232</span> (149.30 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">56,800</span> (221.88 KB)
</pre>



The utility function `dict_to_tuple` effectively transforms the dictionaries of
training and validation datasets into tuples of images and one-hot encoded
segmentation masks, which is used during training and evaluation of the
DeepLabv3+ model.


```python

def dict_to_tuple(x):

    return x["images"], tf.one_hot(
        tf.cast(tf.squeeze(x["segmentation_masks"], axis=-1), "int32"), 21
    )


train_ds = train_ds.map(dict_to_tuple)
eval_ds = eval_ds.map(dict_to_tuple)

model.fit(train_ds, validation_data=eval_ds, epochs=EPOCHS)
```

<div class="k-default-codeblock">
```
/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/models/functional.py:225: UserWarning: The structure of `inputs` doesn't match the expected structure: ['keras_tensor_222']. Received: the structure of inputs=*
  warnings.warn(

/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/backend/jax/core.py:76: UserWarning: Explicitly requested dtype int64 requested in asarray is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return jnp.asarray(x, dtype=dtype)

/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/models/functional.py:225: UserWarning: The structure of `inputs` doesn't match the expected structure: ['keras_tensor_222']. Received: the structure of inputs=*
  warnings.warn(

/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/backend/jax/core.py:76: UserWarning: Explicitly requested dtype int64 requested in asarray is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return jnp.asarray(x, dtype=dtype)

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  40s 40s/step - categorical_accuracy: 0.1762 - loss: 2.7213 - mean_io_u: 0.0187


  2/Unknown  62s 21s/step - categorical_accuracy: 0.1811 - loss: 2.6949 - mean_io_u: 0.0185


  3/Unknown  62s 11s/step - categorical_accuracy: 0.1829 - loss: 2.6802 - mean_io_u: 0.0183


  4/Unknown  62s 7s/step - categorical_accuracy: 0.1828 - loss: 2.6807 - mean_io_u: 0.0180 


  5/Unknown  62s 5s/step - categorical_accuracy: 0.1856 - loss: 2.6964 - mean_io_u: 0.0178


  6/Unknown  62s 4s/step - categorical_accuracy: 0.1934 - loss: 2.6897 - mean_io_u: 0.0180


  7/Unknown  62s 4s/step - categorical_accuracy: 0.2016 - loss: 2.6850 - mean_io_u: 0.0184


  8/Unknown  62s 3s/step - categorical_accuracy: 0.2119 - loss: 2.6752 - mean_io_u: 0.0190


  9/Unknown  62s 3s/step - categorical_accuracy: 0.2222 - loss: 2.6638 - mean_io_u: 0.0196


 10/Unknown  62s 2s/step - categorical_accuracy: 0.2322 - loss: 2.6527 - mean_io_u: 0.0201


 11/Unknown  62s 2s/step - categorical_accuracy: 0.2424 - loss: 2.6423 - mean_io_u: 0.0206


 12/Unknown  63s 2s/step - categorical_accuracy: 0.2520 - loss: 2.6309 - mean_io_u: 0.0211


 13/Unknown  63s 2s/step - categorical_accuracy: 0.2614 - loss: 2.6180 - mean_io_u: 0.0215


 14/Unknown  63s 2s/step - categorical_accuracy: 0.2710 - loss: 2.6031 - mean_io_u: 0.0219


 15/Unknown  63s 2s/step - categorical_accuracy: 0.2799 - loss: 2.5887 - mean_io_u: 0.0223


 16/Unknown  63s 1s/step - categorical_accuracy: 0.2888 - loss: 2.5735 - mean_io_u: 0.0227


 17/Unknown  63s 1s/step - categorical_accuracy: 0.2975 - loss: 2.5579 - mean_io_u: 0.0230


 18/Unknown  63s 1s/step - categorical_accuracy: 0.3056 - loss: 2.5434 - mean_io_u: 0.0233


 19/Unknown  63s 1s/step - categorical_accuracy: 0.3139 - loss: 2.5270 - mean_io_u: 0.0236


 20/Unknown  63s 1s/step - categorical_accuracy: 0.3217 - loss: 2.5107 - mean_io_u: 0.0239


 21/Unknown  63s 1s/step - categorical_accuracy: 0.3295 - loss: 2.4937 - mean_io_u: 0.0242


 22/Unknown  63s 1s/step - categorical_accuracy: 0.3371 - loss: 2.4766 - mean_io_u: 0.0245


 23/Unknown  63s 1s/step - categorical_accuracy: 0.3441 - loss: 2.4615 - mean_io_u: 0.0248


 24/Unknown  64s 1s/step - categorical_accuracy: 0.3507 - loss: 2.4461 - mean_io_u: 0.0250


 25/Unknown  64s 966ms/step - categorical_accuracy: 0.3571 - loss: 2.4317 - mean_io_u: 0.0253


 26/Unknown  64s 933ms/step - categorical_accuracy: 0.3633 - loss: 2.4171 - mean_io_u: 0.0255


 27/Unknown  64s 902ms/step - categorical_accuracy: 0.3692 - loss: 2.4029 - mean_io_u: 0.0257


 28/Unknown  64s 873ms/step - categorical_accuracy: 0.3748 - loss: 2.3896 - mean_io_u: 0.0259


 29/Unknown  64s 846ms/step - categorical_accuracy: 0.3803 - loss: 2.3763 - mean_io_u: 0.0261


 30/Unknown  64s 821ms/step - categorical_accuracy: 0.3855 - loss: 2.3635 - mean_io_u: 0.0263


 31/Unknown  64s 796ms/step - categorical_accuracy: 0.3906 - loss: 2.3506 - mean_io_u: 0.0265


 32/Unknown  64s 773ms/step - categorical_accuracy: 0.3955 - loss: 2.3383 - mean_io_u: 0.0266


 33/Unknown  64s 752ms/step - categorical_accuracy: 0.4002 - loss: 2.3264 - mean_io_u: 0.0268


 34/Unknown  65s 733ms/step - categorical_accuracy: 0.4047 - loss: 2.3144 - mean_io_u: 0.0269


 35/Unknown  65s 714ms/step - categorical_accuracy: 0.4090 - loss: 2.3032 - mean_io_u: 0.0270


 36/Unknown  65s 696ms/step - categorical_accuracy: 0.4133 - loss: 2.2917 - mean_io_u: 0.0271


 37/Unknown  65s 679ms/step - categorical_accuracy: 0.4172 - loss: 2.2813 - mean_io_u: 0.0272


 38/Unknown  65s 663ms/step - categorical_accuracy: 0.4210 - loss: 2.2711 - mean_io_u: 0.0273


 39/Unknown  65s 648ms/step - categorical_accuracy: 0.4247 - loss: 2.2610 - mean_io_u: 0.0273


 40/Unknown  65s 633ms/step - categorical_accuracy: 0.4283 - loss: 2.2513 - mean_io_u: 0.0274


 41/Unknown  65s 621ms/step - categorical_accuracy: 0.4317 - loss: 2.2421 - mean_io_u: 0.0275


 42/Unknown  65s 608ms/step - categorical_accuracy: 0.4351 - loss: 2.2328 - mean_io_u: 0.0275


 43/Unknown  65s 597ms/step - categorical_accuracy: 0.4384 - loss: 2.2236 - mean_io_u: 0.0276


 44/Unknown  66s 585ms/step - categorical_accuracy: 0.4415 - loss: 2.2151 - mean_io_u: 0.0276


 45/Unknown  66s 574ms/step - categorical_accuracy: 0.4445 - loss: 2.2070 - mean_io_u: 0.0277


 46/Unknown  66s 564ms/step - categorical_accuracy: 0.4474 - loss: 2.1991 - mean_io_u: 0.0277


 47/Unknown  66s 553ms/step - categorical_accuracy: 0.4502 - loss: 2.1914 - mean_io_u: 0.0277


 48/Unknown  66s 544ms/step - categorical_accuracy: 0.4529 - loss: 2.1837 - mean_io_u: 0.0277


 49/Unknown  66s 534ms/step - categorical_accuracy: 0.4556 - loss: 2.1760 - mean_io_u: 0.0278


 50/Unknown  66s 525ms/step - categorical_accuracy: 0.4583 - loss: 2.1682 - mean_io_u: 0.0278


 51/Unknown  66s 517ms/step - categorical_accuracy: 0.4609 - loss: 2.1606 - mean_io_u: 0.0278


 52/Unknown  66s 509ms/step - categorical_accuracy: 0.4634 - loss: 2.1531 - mean_io_u: 0.0278


 53/Unknown  67s 502ms/step - categorical_accuracy: 0.4658 - loss: 2.1458 - mean_io_u: 0.0279


 54/Unknown  67s 494ms/step - categorical_accuracy: 0.4682 - loss: 2.1386 - mean_io_u: 0.0279


 55/Unknown  67s 487ms/step - categorical_accuracy: 0.4705 - loss: 2.1318 - mean_io_u: 0.0279


 56/Unknown  67s 479ms/step - categorical_accuracy: 0.4727 - loss: 2.1249 - mean_io_u: 0.0279


 57/Unknown  67s 473ms/step - categorical_accuracy: 0.4750 - loss: 2.1181 - mean_io_u: 0.0280


 58/Unknown  67s 467ms/step - categorical_accuracy: 0.4771 - loss: 2.1115 - mean_io_u: 0.0280


 59/Unknown  67s 461ms/step - categorical_accuracy: 0.4793 - loss: 2.1049 - mean_io_u: 0.0280


 60/Unknown  67s 454ms/step - categorical_accuracy: 0.4814 - loss: 2.0983 - mean_io_u: 0.0280


 61/Unknown  67s 449ms/step - categorical_accuracy: 0.4835 - loss: 2.0917 - mean_io_u: 0.0280


 62/Unknown  67s 443ms/step - categorical_accuracy: 0.4856 - loss: 2.0851 - mean_io_u: 0.0280


 63/Unknown  68s 438ms/step - categorical_accuracy: 0.4876 - loss: 2.0786 - mean_io_u: 0.0281


 64/Unknown  68s 432ms/step - categorical_accuracy: 0.4897 - loss: 2.0721 - mean_io_u: 0.0281


 65/Unknown  68s 427ms/step - categorical_accuracy: 0.4917 - loss: 2.0656 - mean_io_u: 0.0281


 66/Unknown  68s 421ms/step - categorical_accuracy: 0.4937 - loss: 2.0592 - mean_io_u: 0.0281


 67/Unknown  68s 416ms/step - categorical_accuracy: 0.4956 - loss: 2.0530 - mean_io_u: 0.0281


 68/Unknown  68s 412ms/step - categorical_accuracy: 0.4974 - loss: 2.0470 - mean_io_u: 0.0281


 69/Unknown  68s 407ms/step - categorical_accuracy: 0.4992 - loss: 2.0412 - mean_io_u: 0.0282


 70/Unknown  68s 402ms/step - categorical_accuracy: 0.5009 - loss: 2.0354 - mean_io_u: 0.0282


 71/Unknown  68s 398ms/step - categorical_accuracy: 0.5026 - loss: 2.0298 - mean_io_u: 0.0282


 72/Unknown  68s 394ms/step - categorical_accuracy: 0.5043 - loss: 2.0243 - mean_io_u: 0.0282


 73/Unknown  69s 390ms/step - categorical_accuracy: 0.5059 - loss: 2.0189 - mean_io_u: 0.0282


 74/Unknown  69s 386ms/step - categorical_accuracy: 0.5075 - loss: 2.0135 - mean_io_u: 0.0282


 75/Unknown  69s 382ms/step - categorical_accuracy: 0.5090 - loss: 2.0083 - mean_io_u: 0.0283


 76/Unknown  69s 379ms/step - categorical_accuracy: 0.5106 - loss: 2.0031 - mean_io_u: 0.0283


 77/Unknown  69s 375ms/step - categorical_accuracy: 0.5121 - loss: 1.9979 - mean_io_u: 0.0283


 78/Unknown  69s 372ms/step - categorical_accuracy: 0.5136 - loss: 1.9928 - mean_io_u: 0.0283


 79/Unknown  69s 368ms/step - categorical_accuracy: 0.5151 - loss: 1.9878 - mean_io_u: 0.0283


 80/Unknown  69s 365ms/step - categorical_accuracy: 0.5165 - loss: 1.9827 - mean_io_u: 0.0283


 81/Unknown  69s 361ms/step - categorical_accuracy: 0.5180 - loss: 1.9778 - mean_io_u: 0.0283


 82/Unknown  69s 358ms/step - categorical_accuracy: 0.5194 - loss: 1.9729 - mean_io_u: 0.0284


 83/Unknown  70s 355ms/step - categorical_accuracy: 0.5208 - loss: 1.9681 - mean_io_u: 0.0284


 84/Unknown  70s 351ms/step - categorical_accuracy: 0.5221 - loss: 1.9634 - mean_io_u: 0.0284


 85/Unknown  70s 349ms/step - categorical_accuracy: 0.5234 - loss: 1.9590 - mean_io_u: 0.0284


 86/Unknown  70s 346ms/step - categorical_accuracy: 0.5247 - loss: 1.9547 - mean_io_u: 0.0284


 87/Unknown  70s 343ms/step - categorical_accuracy: 0.5259 - loss: 1.9504 - mean_io_u: 0.0284


 88/Unknown  70s 340ms/step - categorical_accuracy: 0.5271 - loss: 1.9461 - mean_io_u: 0.0285


 89/Unknown  70s 337ms/step - categorical_accuracy: 0.5284 - loss: 1.9419 - mean_io_u: 0.0285


 90/Unknown  70s 334ms/step - categorical_accuracy: 0.5296 - loss: 1.9376 - mean_io_u: 0.0285


 91/Unknown  70s 331ms/step - categorical_accuracy: 0.5308 - loss: 1.9335 - mean_io_u: 0.0285


 92/Unknown  70s 329ms/step - categorical_accuracy: 0.5319 - loss: 1.9294 - mean_io_u: 0.0286


 93/Unknown  70s 326ms/step - categorical_accuracy: 0.5331 - loss: 1.9254 - mean_io_u: 0.0286


 94/Unknown  71s 324ms/step - categorical_accuracy: 0.5342 - loss: 1.9214 - mean_io_u: 0.0286


 95/Unknown  71s 321ms/step - categorical_accuracy: 0.5353 - loss: 1.9175 - mean_io_u: 0.0286


 96/Unknown  71s 318ms/step - categorical_accuracy: 0.5364 - loss: 1.9136 - mean_io_u: 0.0287


 97/Unknown  71s 316ms/step - categorical_accuracy: 0.5375 - loss: 1.9097 - mean_io_u: 0.0287


 98/Unknown  71s 313ms/step - categorical_accuracy: 0.5386 - loss: 1.9058 - mean_io_u: 0.0287


 99/Unknown  71s 311ms/step - categorical_accuracy: 0.5396 - loss: 1.9021 - mean_io_u: 0.0287


100/Unknown  71s 308ms/step - categorical_accuracy: 0.5407 - loss: 1.8984 - mean_io_u: 0.0288


101/Unknown  71s 306ms/step - categorical_accuracy: 0.5417 - loss: 1.8949 - mean_io_u: 0.0288


102/Unknown  71s 304ms/step - categorical_accuracy: 0.5427 - loss: 1.8914 - mean_io_u: 0.0288


103/Unknown  71s 301ms/step - categorical_accuracy: 0.5436 - loss: 1.8879 - mean_io_u: 0.0288


104/Unknown  71s 299ms/step - categorical_accuracy: 0.5446 - loss: 1.8845 - mean_io_u: 0.0289


105/Unknown  71s 298ms/step - categorical_accuracy: 0.5455 - loss: 1.8810 - mean_io_u: 0.0289


106/Unknown  71s 296ms/step - categorical_accuracy: 0.5465 - loss: 1.8776 - mean_io_u: 0.0289


107/Unknown  72s 294ms/step - categorical_accuracy: 0.5474 - loss: 1.8742 - mean_io_u: 0.0289


108/Unknown  72s 292ms/step - categorical_accuracy: 0.5484 - loss: 1.8709 - mean_io_u: 0.0290


109/Unknown  72s 290ms/step - categorical_accuracy: 0.5493 - loss: 1.8675 - mean_io_u: 0.0290


110/Unknown  72s 288ms/step - categorical_accuracy: 0.5502 - loss: 1.8642 - mean_io_u: 0.0290


111/Unknown  72s 286ms/step - categorical_accuracy: 0.5511 - loss: 1.8610 - mean_io_u: 0.0290


112/Unknown  72s 284ms/step - categorical_accuracy: 0.5520 - loss: 1.8578 - mean_io_u: 0.0291


113/Unknown  72s 283ms/step - categorical_accuracy: 0.5528 - loss: 1.8546 - mean_io_u: 0.0291


114/Unknown  72s 281ms/step - categorical_accuracy: 0.5537 - loss: 1.8514 - mean_io_u: 0.0291


115/Unknown  72s 280ms/step - categorical_accuracy: 0.5546 - loss: 1.8483 - mean_io_u: 0.0291


116/Unknown  72s 278ms/step - categorical_accuracy: 0.5554 - loss: 1.8452 - mean_io_u: 0.0292


117/Unknown  72s 276ms/step - categorical_accuracy: 0.5563 - loss: 1.8421 - mean_io_u: 0.0292


118/Unknown  73s 275ms/step - categorical_accuracy: 0.5571 - loss: 1.8390 - mean_io_u: 0.0292


119/Unknown  73s 273ms/step - categorical_accuracy: 0.5579 - loss: 1.8360 - mean_io_u: 0.0292


120/Unknown  73s 272ms/step - categorical_accuracy: 0.5587 - loss: 1.8330 - mean_io_u: 0.0293


121/Unknown  73s 270ms/step - categorical_accuracy: 0.5595 - loss: 1.8300 - mean_io_u: 0.0293


122/Unknown  73s 269ms/step - categorical_accuracy: 0.5603 - loss: 1.8271 - mean_io_u: 0.0293


123/Unknown  73s 267ms/step - categorical_accuracy: 0.5611 - loss: 1.8242 - mean_io_u: 0.0293


124/Unknown  73s 266ms/step - categorical_accuracy: 0.5619 - loss: 1.8213 - mean_io_u: 0.0293


125/Unknown  73s 265ms/step - categorical_accuracy: 0.5627 - loss: 1.8186 - mean_io_u: 0.0294


126/Unknown  73s 263ms/step - categorical_accuracy: 0.5634 - loss: 1.8158 - mean_io_u: 0.0294


127/Unknown  73s 262ms/step - categorical_accuracy: 0.5642 - loss: 1.8130 - mean_io_u: 0.0294


128/Unknown  74s 261ms/step - categorical_accuracy: 0.5649 - loss: 1.8103 - mean_io_u: 0.0294


129/Unknown  74s 260ms/step - categorical_accuracy: 0.5656 - loss: 1.8075 - mean_io_u: 0.0294


130/Unknown  74s 259ms/step - categorical_accuracy: 0.5663 - loss: 1.8049 - mean_io_u: 0.0295


131/Unknown  74s 258ms/step - categorical_accuracy: 0.5670 - loss: 1.8023 - mean_io_u: 0.0295


132/Unknown  74s 257ms/step - categorical_accuracy: 0.5677 - loss: 1.7997 - mean_io_u: 0.0295


133/Unknown  74s 255ms/step - categorical_accuracy: 0.5684 - loss: 1.7971 - mean_io_u: 0.0295


134/Unknown  74s 254ms/step - categorical_accuracy: 0.5691 - loss: 1.7945 - mean_io_u: 0.0295


135/Unknown  74s 253ms/step - categorical_accuracy: 0.5697 - loss: 1.7920 - mean_io_u: 0.0296


136/Unknown  74s 252ms/step - categorical_accuracy: 0.5704 - loss: 1.7894 - mean_io_u: 0.0296


137/Unknown  75s 251ms/step - categorical_accuracy: 0.5711 - loss: 1.7869 - mean_io_u: 0.0296


138/Unknown  75s 250ms/step - categorical_accuracy: 0.5717 - loss: 1.7844 - mean_io_u: 0.0296


139/Unknown  75s 249ms/step - categorical_accuracy: 0.5723 - loss: 1.7819 - mean_io_u: 0.0296


140/Unknown  75s 248ms/step - categorical_accuracy: 0.5730 - loss: 1.7795 - mean_io_u: 0.0297


141/Unknown  75s 247ms/step - categorical_accuracy: 0.5736 - loss: 1.7771 - mean_io_u: 0.0297


142/Unknown  75s 246ms/step - categorical_accuracy: 0.5742 - loss: 1.7747 - mean_io_u: 0.0297


143/Unknown  75s 245ms/step - categorical_accuracy: 0.5748 - loss: 1.7723 - mean_io_u: 0.0297


144/Unknown  75s 244ms/step - categorical_accuracy: 0.5754 - loss: 1.7699 - mean_io_u: 0.0298


145/Unknown  75s 243ms/step - categorical_accuracy: 0.5760 - loss: 1.7676 - mean_io_u: 0.0298


146/Unknown  76s 242ms/step - categorical_accuracy: 0.5766 - loss: 1.7653 - mean_io_u: 0.0298


147/Unknown  76s 241ms/step - categorical_accuracy: 0.5772 - loss: 1.7630 - mean_io_u: 0.0299


148/Unknown  76s 240ms/step - categorical_accuracy: 0.5778 - loss: 1.7607 - mean_io_u: 0.0299


149/Unknown  76s 239ms/step - categorical_accuracy: 0.5784 - loss: 1.7584 - mean_io_u: 0.0299


150/Unknown  76s 238ms/step - categorical_accuracy: 0.5789 - loss: 1.7561 - mean_io_u: 0.0299


151/Unknown  76s 237ms/step - categorical_accuracy: 0.5795 - loss: 1.7539 - mean_io_u: 0.0300


152/Unknown  76s 236ms/step - categorical_accuracy: 0.5801 - loss: 1.7517 - mean_io_u: 0.0300


153/Unknown  76s 235ms/step - categorical_accuracy: 0.5806 - loss: 1.7494 - mean_io_u: 0.0300


154/Unknown  76s 234ms/step - categorical_accuracy: 0.5812 - loss: 1.7473 - mean_io_u: 0.0301


155/Unknown  76s 234ms/step - categorical_accuracy: 0.5817 - loss: 1.7451 - mean_io_u: 0.0301


156/Unknown  77s 233ms/step - categorical_accuracy: 0.5823 - loss: 1.7429 - mean_io_u: 0.0301


157/Unknown  77s 232ms/step - categorical_accuracy: 0.5828 - loss: 1.7408 - mean_io_u: 0.0302


158/Unknown  77s 231ms/step - categorical_accuracy: 0.5834 - loss: 1.7386 - mean_io_u: 0.0302


159/Unknown  77s 231ms/step - categorical_accuracy: 0.5839 - loss: 1.7365 - mean_io_u: 0.0302


160/Unknown  77s 229ms/step - categorical_accuracy: 0.5845 - loss: 1.7344 - mean_io_u: 0.0303


161/Unknown  77s 229ms/step - categorical_accuracy: 0.5850 - loss: 1.7323 - mean_io_u: 0.0303


162/Unknown  77s 228ms/step - categorical_accuracy: 0.5855 - loss: 1.7302 - mean_io_u: 0.0304


163/Unknown  77s 227ms/step - categorical_accuracy: 0.5861 - loss: 1.7281 - mean_io_u: 0.0304


164/Unknown  77s 227ms/step - categorical_accuracy: 0.5866 - loss: 1.7260 - mean_io_u: 0.0304


165/Unknown  77s 226ms/step - categorical_accuracy: 0.5871 - loss: 1.7240 - mean_io_u: 0.0305


166/Unknown  78s 225ms/step - categorical_accuracy: 0.5876 - loss: 1.7219 - mean_io_u: 0.0305


167/Unknown  78s 224ms/step - categorical_accuracy: 0.5881 - loss: 1.7199 - mean_io_u: 0.0305


168/Unknown  78s 224ms/step - categorical_accuracy: 0.5886 - loss: 1.7179 - mean_io_u: 0.0306


169/Unknown  78s 223ms/step - categorical_accuracy: 0.5891 - loss: 1.7160 - mean_io_u: 0.0306


170/Unknown  78s 222ms/step - categorical_accuracy: 0.5896 - loss: 1.7140 - mean_io_u: 0.0307


171/Unknown  78s 221ms/step - categorical_accuracy: 0.5901 - loss: 1.7121 - mean_io_u: 0.0307


172/Unknown  78s 221ms/step - categorical_accuracy: 0.5906 - loss: 1.7101 - mean_io_u: 0.0307


173/Unknown  78s 220ms/step - categorical_accuracy: 0.5911 - loss: 1.7082 - mean_io_u: 0.0308


174/Unknown  78s 219ms/step - categorical_accuracy: 0.5916 - loss: 1.7063 - mean_io_u: 0.0308


175/Unknown  78s 219ms/step - categorical_accuracy: 0.5920 - loss: 1.7044 - mean_io_u: 0.0308


176/Unknown  79s 218ms/step - categorical_accuracy: 0.5925 - loss: 1.7025 - mean_io_u: 0.0309


177/Unknown  79s 217ms/step - categorical_accuracy: 0.5930 - loss: 1.7006 - mean_io_u: 0.0309


178/Unknown  79s 216ms/step - categorical_accuracy: 0.5934 - loss: 1.6988 - mean_io_u: 0.0309


179/Unknown  79s 216ms/step - categorical_accuracy: 0.5939 - loss: 1.6969 - mean_io_u: 0.0310


180/Unknown  79s 215ms/step - categorical_accuracy: 0.5944 - loss: 1.6951 - mean_io_u: 0.0310


181/Unknown  79s 214ms/step - categorical_accuracy: 0.5948 - loss: 1.6933 - mean_io_u: 0.0310


182/Unknown  79s 214ms/step - categorical_accuracy: 0.5953 - loss: 1.6915 - mean_io_u: 0.0311


183/Unknown  79s 213ms/step - categorical_accuracy: 0.5957 - loss: 1.6897 - mean_io_u: 0.0311


184/Unknown  79s 213ms/step - categorical_accuracy: 0.5961 - loss: 1.6879 - mean_io_u: 0.0311


185/Unknown  80s 212ms/step - categorical_accuracy: 0.5966 - loss: 1.6862 - mean_io_u: 0.0311


186/Unknown  80s 212ms/step - categorical_accuracy: 0.5970 - loss: 1.6844 - mean_io_u: 0.0312


187/Unknown  80s 211ms/step - categorical_accuracy: 0.5974 - loss: 1.6827 - mean_io_u: 0.0312


188/Unknown  80s 210ms/step - categorical_accuracy: 0.5978 - loss: 1.6811 - mean_io_u: 0.0312


189/Unknown  80s 210ms/step - categorical_accuracy: 0.5983 - loss: 1.6794 - mean_io_u: 0.0313


190/Unknown  80s 209ms/step - categorical_accuracy: 0.5987 - loss: 1.6777 - mean_io_u: 0.0313


191/Unknown  80s 209ms/step - categorical_accuracy: 0.5991 - loss: 1.6761 - mean_io_u: 0.0313


192/Unknown  80s 208ms/step - categorical_accuracy: 0.5995 - loss: 1.6744 - mean_io_u: 0.0314


193/Unknown  80s 207ms/step - categorical_accuracy: 0.5999 - loss: 1.6728 - mean_io_u: 0.0314


194/Unknown  80s 207ms/step - categorical_accuracy: 0.6003 - loss: 1.6712 - mean_io_u: 0.0314


195/Unknown  80s 206ms/step - categorical_accuracy: 0.6007 - loss: 1.6696 - mean_io_u: 0.0315


196/Unknown  81s 206ms/step - categorical_accuracy: 0.6011 - loss: 1.6680 - mean_io_u: 0.0315


197/Unknown  81s 205ms/step - categorical_accuracy: 0.6015 - loss: 1.6665 - mean_io_u: 0.0315


198/Unknown  81s 204ms/step - categorical_accuracy: 0.6018 - loss: 1.6649 - mean_io_u: 0.0315


199/Unknown  81s 204ms/step - categorical_accuracy: 0.6022 - loss: 1.6634 - mean_io_u: 0.0316


200/Unknown  81s 203ms/step - categorical_accuracy: 0.6026 - loss: 1.6619 - mean_io_u: 0.0316


201/Unknown  81s 203ms/step - categorical_accuracy: 0.6030 - loss: 1.6604 - mean_io_u: 0.0316


202/Unknown  81s 202ms/step - categorical_accuracy: 0.6033 - loss: 1.6588 - mean_io_u: 0.0317


203/Unknown  81s 202ms/step - categorical_accuracy: 0.6037 - loss: 1.6573 - mean_io_u: 0.0317


204/Unknown  81s 201ms/step - categorical_accuracy: 0.6041 - loss: 1.6558 - mean_io_u: 0.0317


205/Unknown  81s 201ms/step - categorical_accuracy: 0.6044 - loss: 1.6543 - mean_io_u: 0.0318


206/Unknown  81s 200ms/step - categorical_accuracy: 0.6048 - loss: 1.6529 - mean_io_u: 0.0318


207/Unknown  82s 200ms/step - categorical_accuracy: 0.6051 - loss: 1.6514 - mean_io_u: 0.0318


208/Unknown  82s 199ms/step - categorical_accuracy: 0.6055 - loss: 1.6499 - mean_io_u: 0.0319


209/Unknown  82s 198ms/step - categorical_accuracy: 0.6059 - loss: 1.6484 - mean_io_u: 0.0319


210/Unknown  82s 198ms/step - categorical_accuracy: 0.6062 - loss: 1.6469 - mean_io_u: 0.0319


211/Unknown  82s 197ms/step - categorical_accuracy: 0.6066 - loss: 1.6455 - mean_io_u: 0.0320


212/Unknown  82s 197ms/step - categorical_accuracy: 0.6069 - loss: 1.6440 - mean_io_u: 0.0320


213/Unknown  82s 196ms/step - categorical_accuracy: 0.6073 - loss: 1.6425 - mean_io_u: 0.0320


214/Unknown  82s 196ms/step - categorical_accuracy: 0.6076 - loss: 1.6411 - mean_io_u: 0.0321


215/Unknown  82s 195ms/step - categorical_accuracy: 0.6080 - loss: 1.6397 - mean_io_u: 0.0321


216/Unknown  82s 195ms/step - categorical_accuracy: 0.6083 - loss: 1.6382 - mean_io_u: 0.0321


217/Unknown  82s 194ms/step - categorical_accuracy: 0.6087 - loss: 1.6368 - mean_io_u: 0.0322


218/Unknown  82s 194ms/step - categorical_accuracy: 0.6090 - loss: 1.6354 - mean_io_u: 0.0322


219/Unknown  83s 193ms/step - categorical_accuracy: 0.6094 - loss: 1.6339 - mean_io_u: 0.0322


220/Unknown  83s 193ms/step - categorical_accuracy: 0.6097 - loss: 1.6325 - mean_io_u: 0.0323


221/Unknown  83s 192ms/step - categorical_accuracy: 0.6100 - loss: 1.6312 - mean_io_u: 0.0323


222/Unknown  83s 191ms/step - categorical_accuracy: 0.6103 - loss: 1.6298 - mean_io_u: 0.0323


223/Unknown  83s 191ms/step - categorical_accuracy: 0.6107 - loss: 1.6284 - mean_io_u: 0.0324


224/Unknown  83s 190ms/step - categorical_accuracy: 0.6110 - loss: 1.6270 - mean_io_u: 0.0324


225/Unknown  83s 190ms/step - categorical_accuracy: 0.6113 - loss: 1.6257 - mean_io_u: 0.0324


226/Unknown  83s 189ms/step - categorical_accuracy: 0.6116 - loss: 1.6243 - mean_io_u: 0.0325


227/Unknown  83s 189ms/step - categorical_accuracy: 0.6120 - loss: 1.6230 - mean_io_u: 0.0325


228/Unknown  83s 188ms/step - categorical_accuracy: 0.6123 - loss: 1.6217 - mean_io_u: 0.0325


229/Unknown  83s 188ms/step - categorical_accuracy: 0.6126 - loss: 1.6203 - mean_io_u: 0.0326


230/Unknown  83s 187ms/step - categorical_accuracy: 0.6129 - loss: 1.6190 - mean_io_u: 0.0326


231/Unknown  83s 187ms/step - categorical_accuracy: 0.6132 - loss: 1.6177 - mean_io_u: 0.0327


232/Unknown  83s 186ms/step - categorical_accuracy: 0.6135 - loss: 1.6164 - mean_io_u: 0.0327


233/Unknown  84s 186ms/step - categorical_accuracy: 0.6139 - loss: 1.6151 - mean_io_u: 0.0327


234/Unknown  84s 185ms/step - categorical_accuracy: 0.6142 - loss: 1.6138 - mean_io_u: 0.0328


235/Unknown  84s 185ms/step - categorical_accuracy: 0.6145 - loss: 1.6125 - mean_io_u: 0.0328


236/Unknown  84s 184ms/step - categorical_accuracy: 0.6148 - loss: 1.6113 - mean_io_u: 0.0328


237/Unknown  84s 184ms/step - categorical_accuracy: 0.6151 - loss: 1.6100 - mean_io_u: 0.0329


238/Unknown  84s 183ms/step - categorical_accuracy: 0.6153 - loss: 1.6088 - mean_io_u: 0.0329


239/Unknown  84s 183ms/step - categorical_accuracy: 0.6156 - loss: 1.6076 - mean_io_u: 0.0329


240/Unknown  84s 183ms/step - categorical_accuracy: 0.6159 - loss: 1.6064 - mean_io_u: 0.0329


241/Unknown  84s 182ms/step - categorical_accuracy: 0.6162 - loss: 1.6052 - mean_io_u: 0.0330


242/Unknown  84s 182ms/step - categorical_accuracy: 0.6165 - loss: 1.6040 - mean_io_u: 0.0330


243/Unknown  84s 181ms/step - categorical_accuracy: 0.6168 - loss: 1.6028 - mean_io_u: 0.0331


244/Unknown  84s 181ms/step - categorical_accuracy: 0.6171 - loss: 1.6016 - mean_io_u: 0.0331


245/Unknown  85s 181ms/step - categorical_accuracy: 0.6173 - loss: 1.6005 - mean_io_u: 0.0331


246/Unknown  85s 180ms/step - categorical_accuracy: 0.6176 - loss: 1.5993 - mean_io_u: 0.0332


247/Unknown  85s 180ms/step - categorical_accuracy: 0.6179 - loss: 1.5981 - mean_io_u: 0.0332


248/Unknown  85s 179ms/step - categorical_accuracy: 0.6182 - loss: 1.5970 - mean_io_u: 0.0332


249/Unknown  85s 179ms/step - categorical_accuracy: 0.6184 - loss: 1.5958 - mean_io_u: 0.0333


250/Unknown  85s 179ms/step - categorical_accuracy: 0.6187 - loss: 1.5947 - mean_io_u: 0.0333


251/Unknown  85s 178ms/step - categorical_accuracy: 0.6190 - loss: 1.5935 - mean_io_u: 0.0333


252/Unknown  85s 178ms/step - categorical_accuracy: 0.6193 - loss: 1.5924 - mean_io_u: 0.0334


253/Unknown  85s 178ms/step - categorical_accuracy: 0.6195 - loss: 1.5912 - mean_io_u: 0.0334


254/Unknown  85s 177ms/step - categorical_accuracy: 0.6198 - loss: 1.5901 - mean_io_u: 0.0334


255/Unknown  85s 177ms/step - categorical_accuracy: 0.6201 - loss: 1.5889 - mean_io_u: 0.0335


256/Unknown  85s 176ms/step - categorical_accuracy: 0.6203 - loss: 1.5878 - mean_io_u: 0.0335


257/Unknown  85s 176ms/step - categorical_accuracy: 0.6206 - loss: 1.5867 - mean_io_u: 0.0335


258/Unknown  86s 176ms/step - categorical_accuracy: 0.6208 - loss: 1.5856 - mean_io_u: 0.0336


259/Unknown  86s 175ms/step - categorical_accuracy: 0.6211 - loss: 1.5845 - mean_io_u: 0.0336


260/Unknown  86s 175ms/step - categorical_accuracy: 0.6214 - loss: 1.5834 - mean_io_u: 0.0337


261/Unknown  86s 174ms/step - categorical_accuracy: 0.6216 - loss: 1.5823 - mean_io_u: 0.0337


262/Unknown  86s 174ms/step - categorical_accuracy: 0.6219 - loss: 1.5812 - mean_io_u: 0.0337


263/Unknown  86s 174ms/step - categorical_accuracy: 0.6221 - loss: 1.5801 - mean_io_u: 0.0338


264/Unknown  86s 173ms/step - categorical_accuracy: 0.6223 - loss: 1.5791 - mean_io_u: 0.0338


265/Unknown  86s 173ms/step - categorical_accuracy: 0.6226 - loss: 1.5780 - mean_io_u: 0.0338


266/Unknown  86s 173ms/step - categorical_accuracy: 0.6228 - loss: 1.5770 - mean_io_u: 0.0339


267/Unknown  86s 172ms/step - categorical_accuracy: 0.6231 - loss: 1.5759 - mean_io_u: 0.0339


268/Unknown  86s 172ms/step - categorical_accuracy: 0.6233 - loss: 1.5749 - mean_io_u: 0.0340


269/Unknown  86s 172ms/step - categorical_accuracy: 0.6235 - loss: 1.5739 - mean_io_u: 0.0340


270/Unknown  87s 171ms/step - categorical_accuracy: 0.6238 - loss: 1.5728 - mean_io_u: 0.0340


271/Unknown  87s 171ms/step - categorical_accuracy: 0.6240 - loss: 1.5718 - mean_io_u: 0.0341


272/Unknown  87s 171ms/step - categorical_accuracy: 0.6243 - loss: 1.5708 - mean_io_u: 0.0341


273/Unknown  87s 170ms/step - categorical_accuracy: 0.6245 - loss: 1.5697 - mean_io_u: 0.0342


274/Unknown  87s 170ms/step - categorical_accuracy: 0.6247 - loss: 1.5687 - mean_io_u: 0.0342


275/Unknown  87s 170ms/step - categorical_accuracy: 0.6250 - loss: 1.5677 - mean_io_u: 0.0342


276/Unknown  87s 170ms/step - categorical_accuracy: 0.6252 - loss: 1.5666 - mean_io_u: 0.0343


277/Unknown  87s 169ms/step - categorical_accuracy: 0.6254 - loss: 1.5656 - mean_io_u: 0.0343


278/Unknown  87s 169ms/step - categorical_accuracy: 0.6257 - loss: 1.5646 - mean_io_u: 0.0344


279/Unknown  87s 169ms/step - categorical_accuracy: 0.6259 - loss: 1.5636 - mean_io_u: 0.0344


280/Unknown  88s 169ms/step - categorical_accuracy: 0.6261 - loss: 1.5626 - mean_io_u: 0.0345


281/Unknown  88s 168ms/step - categorical_accuracy: 0.6264 - loss: 1.5616 - mean_io_u: 0.0345


282/Unknown  88s 168ms/step - categorical_accuracy: 0.6266 - loss: 1.5605 - mean_io_u: 0.0345


283/Unknown  88s 168ms/step - categorical_accuracy: 0.6268 - loss: 1.5595 - mean_io_u: 0.0346


284/Unknown  88s 167ms/step - categorical_accuracy: 0.6271 - loss: 1.5585 - mean_io_u: 0.0346


285/Unknown  88s 167ms/step - categorical_accuracy: 0.6273 - loss: 1.5575 - mean_io_u: 0.0347


286/Unknown  88s 167ms/step - categorical_accuracy: 0.6275 - loss: 1.5566 - mean_io_u: 0.0347


287/Unknown  88s 167ms/step - categorical_accuracy: 0.6277 - loss: 1.5556 - mean_io_u: 0.0347


288/Unknown  88s 166ms/step - categorical_accuracy: 0.6280 - loss: 1.5546 - mean_io_u: 0.0348


289/Unknown  88s 166ms/step - categorical_accuracy: 0.6282 - loss: 1.5536 - mean_io_u: 0.0348


290/Unknown  88s 166ms/step - categorical_accuracy: 0.6284 - loss: 1.5526 - mean_io_u: 0.0349


291/Unknown  88s 165ms/step - categorical_accuracy: 0.6286 - loss: 1.5517 - mean_io_u: 0.0349


292/Unknown  89s 165ms/step - categorical_accuracy: 0.6288 - loss: 1.5507 - mean_io_u: 0.0349


293/Unknown  89s 165ms/step - categorical_accuracy: 0.6291 - loss: 1.5498 - mean_io_u: 0.0350


294/Unknown  89s 165ms/step - categorical_accuracy: 0.6293 - loss: 1.5488 - mean_io_u: 0.0350


295/Unknown  89s 164ms/step - categorical_accuracy: 0.6295 - loss: 1.5478 - mean_io_u: 0.0351


296/Unknown  89s 164ms/step - categorical_accuracy: 0.6297 - loss: 1.5469 - mean_io_u: 0.0351


297/Unknown  89s 164ms/step - categorical_accuracy: 0.6299 - loss: 1.5459 - mean_io_u: 0.0352


298/Unknown  89s 164ms/step - categorical_accuracy: 0.6302 - loss: 1.5450 - mean_io_u: 0.0352


299/Unknown  89s 163ms/step - categorical_accuracy: 0.6304 - loss: 1.5440 - mean_io_u: 0.0352


300/Unknown  89s 163ms/step - categorical_accuracy: 0.6306 - loss: 1.5430 - mean_io_u: 0.0353


301/Unknown  89s 163ms/step - categorical_accuracy: 0.6308 - loss: 1.5421 - mean_io_u: 0.0353


302/Unknown  89s 162ms/step - categorical_accuracy: 0.6310 - loss: 1.5411 - mean_io_u: 0.0354


303/Unknown  89s 162ms/step - categorical_accuracy: 0.6312 - loss: 1.5402 - mean_io_u: 0.0354


304/Unknown  90s 162ms/step - categorical_accuracy: 0.6315 - loss: 1.5392 - mean_io_u: 0.0355


305/Unknown  90s 162ms/step - categorical_accuracy: 0.6317 - loss: 1.5383 - mean_io_u: 0.0355


306/Unknown  90s 161ms/step - categorical_accuracy: 0.6319 - loss: 1.5373 - mean_io_u: 0.0355


307/Unknown  90s 161ms/step - categorical_accuracy: 0.6321 - loss: 1.5364 - mean_io_u: 0.0356


308/Unknown  90s 161ms/step - categorical_accuracy: 0.6323 - loss: 1.5354 - mean_io_u: 0.0356


309/Unknown  90s 161ms/step - categorical_accuracy: 0.6325 - loss: 1.5345 - mean_io_u: 0.0357


310/Unknown  90s 160ms/step - categorical_accuracy: 0.6327 - loss: 1.5336 - mean_io_u: 0.0357


311/Unknown  90s 160ms/step - categorical_accuracy: 0.6329 - loss: 1.5327 - mean_io_u: 0.0358


312/Unknown  90s 160ms/step - categorical_accuracy: 0.6331 - loss: 1.5317 - mean_io_u: 0.0358


313/Unknown  90s 159ms/step - categorical_accuracy: 0.6333 - loss: 1.5308 - mean_io_u: 0.0358


314/Unknown  90s 159ms/step - categorical_accuracy: 0.6336 - loss: 1.5299 - mean_io_u: 0.0359


315/Unknown  90s 159ms/step - categorical_accuracy: 0.6338 - loss: 1.5290 - mean_io_u: 0.0359


316/Unknown  90s 159ms/step - categorical_accuracy: 0.6340 - loss: 1.5281 - mean_io_u: 0.0360


317/Unknown  91s 158ms/step - categorical_accuracy: 0.6342 - loss: 1.5272 - mean_io_u: 0.0360


318/Unknown  91s 158ms/step - categorical_accuracy: 0.6344 - loss: 1.5263 - mean_io_u: 0.0360


319/Unknown  91s 158ms/step - categorical_accuracy: 0.6346 - loss: 1.5254 - mean_io_u: 0.0361


320/Unknown  91s 158ms/step - categorical_accuracy: 0.6348 - loss: 1.5245 - mean_io_u: 0.0361


321/Unknown  91s 157ms/step - categorical_accuracy: 0.6350 - loss: 1.5236 - mean_io_u: 0.0362


322/Unknown  91s 157ms/step - categorical_accuracy: 0.6352 - loss: 1.5227 - mean_io_u: 0.0362


323/Unknown  91s 157ms/step - categorical_accuracy: 0.6354 - loss: 1.5218 - mean_io_u: 0.0363


324/Unknown  91s 157ms/step - categorical_accuracy: 0.6356 - loss: 1.5210 - mean_io_u: 0.0363


325/Unknown  91s 156ms/step - categorical_accuracy: 0.6358 - loss: 1.5201 - mean_io_u: 0.0363


326/Unknown  91s 156ms/step - categorical_accuracy: 0.6360 - loss: 1.5192 - mean_io_u: 0.0364


327/Unknown  91s 156ms/step - categorical_accuracy: 0.6361 - loss: 1.5184 - mean_io_u: 0.0364


328/Unknown  91s 156ms/step - categorical_accuracy: 0.6363 - loss: 1.5175 - mean_io_u: 0.0365


329/Unknown  91s 155ms/step - categorical_accuracy: 0.6365 - loss: 1.5166 - mean_io_u: 0.0365


330/Unknown  91s 155ms/step - categorical_accuracy: 0.6367 - loss: 1.5158 - mean_io_u: 0.0365


331/Unknown  92s 155ms/step - categorical_accuracy: 0.6369 - loss: 1.5149 - mean_io_u: 0.0366


332/Unknown  92s 155ms/step - categorical_accuracy: 0.6371 - loss: 1.5141 - mean_io_u: 0.0366


333/Unknown  92s 154ms/step - categorical_accuracy: 0.6373 - loss: 1.5132 - mean_io_u: 0.0367


334/Unknown  92s 154ms/step - categorical_accuracy: 0.6375 - loss: 1.5124 - mean_io_u: 0.0367


335/Unknown  92s 154ms/step - categorical_accuracy: 0.6377 - loss: 1.5116 - mean_io_u: 0.0368


336/Unknown  92s 154ms/step - categorical_accuracy: 0.6379 - loss: 1.5107 - mean_io_u: 0.0368


337/Unknown  92s 153ms/step - categorical_accuracy: 0.6380 - loss: 1.5099 - mean_io_u: 0.0368


338/Unknown  92s 153ms/step - categorical_accuracy: 0.6382 - loss: 1.5091 - mean_io_u: 0.0369


339/Unknown  92s 153ms/step - categorical_accuracy: 0.6384 - loss: 1.5083 - mean_io_u: 0.0369


340/Unknown  92s 153ms/step - categorical_accuracy: 0.6386 - loss: 1.5074 - mean_io_u: 0.0370


341/Unknown  92s 152ms/step - categorical_accuracy: 0.6388 - loss: 1.5066 - mean_io_u: 0.0370


342/Unknown  92s 152ms/step - categorical_accuracy: 0.6390 - loss: 1.5058 - mean_io_u: 0.0371


343/Unknown  92s 152ms/step - categorical_accuracy: 0.6391 - loss: 1.5050 - mean_io_u: 0.0371


344/Unknown  92s 152ms/step - categorical_accuracy: 0.6393 - loss: 1.5042 - mean_io_u: 0.0371


345/Unknown  93s 152ms/step - categorical_accuracy: 0.6395 - loss: 1.5034 - mean_io_u: 0.0372


346/Unknown  93s 151ms/step - categorical_accuracy: 0.6397 - loss: 1.5026 - mean_io_u: 0.0372


347/Unknown  93s 151ms/step - categorical_accuracy: 0.6398 - loss: 1.5018 - mean_io_u: 0.0373


348/Unknown  93s 151ms/step - categorical_accuracy: 0.6400 - loss: 1.5010 - mean_io_u: 0.0373


349/Unknown  93s 151ms/step - categorical_accuracy: 0.6402 - loss: 1.5002 - mean_io_u: 0.0374


350/Unknown  93s 150ms/step - categorical_accuracy: 0.6404 - loss: 1.4995 - mean_io_u: 0.0374


351/Unknown  93s 150ms/step - categorical_accuracy: 0.6405 - loss: 1.4987 - mean_io_u: 0.0374


352/Unknown  93s 150ms/step - categorical_accuracy: 0.6407 - loss: 1.4979 - mean_io_u: 0.0375


353/Unknown  93s 150ms/step - categorical_accuracy: 0.6409 - loss: 1.4971 - mean_io_u: 0.0375


354/Unknown  93s 149ms/step - categorical_accuracy: 0.6411 - loss: 1.4963 - mean_io_u: 0.0376


355/Unknown  93s 149ms/step - categorical_accuracy: 0.6412 - loss: 1.4956 - mean_io_u: 0.0376


356/Unknown  93s 149ms/step - categorical_accuracy: 0.6414 - loss: 1.4948 - mean_io_u: 0.0377


357/Unknown  93s 149ms/step - categorical_accuracy: 0.6416 - loss: 1.4940 - mean_io_u: 0.0377


358/Unknown  93s 148ms/step - categorical_accuracy: 0.6417 - loss: 1.4933 - mean_io_u: 0.0378


359/Unknown  94s 148ms/step - categorical_accuracy: 0.6419 - loss: 1.4925 - mean_io_u: 0.0378


360/Unknown  94s 148ms/step - categorical_accuracy: 0.6421 - loss: 1.4918 - mean_io_u: 0.0378


361/Unknown  94s 148ms/step - categorical_accuracy: 0.6422 - loss: 1.4910 - mean_io_u: 0.0379


362/Unknown  94s 148ms/step - categorical_accuracy: 0.6424 - loss: 1.4903 - mean_io_u: 0.0379


363/Unknown  94s 147ms/step - categorical_accuracy: 0.6426 - loss: 1.4895 - mean_io_u: 0.0380


364/Unknown  94s 147ms/step - categorical_accuracy: 0.6427 - loss: 1.4888 - mean_io_u: 0.0380


365/Unknown  94s 147ms/step - categorical_accuracy: 0.6429 - loss: 1.4880 - mean_io_u: 0.0381


366/Unknown  94s 147ms/step - categorical_accuracy: 0.6430 - loss: 1.4873 - mean_io_u: 0.0381


367/Unknown  94s 147ms/step - categorical_accuracy: 0.6432 - loss: 1.4865 - mean_io_u: 0.0381


368/Unknown  94s 147ms/step - categorical_accuracy: 0.6434 - loss: 1.4858 - mean_io_u: 0.0382


369/Unknown  94s 146ms/step - categorical_accuracy: 0.6435 - loss: 1.4850 - mean_io_u: 0.0382


370/Unknown  94s 146ms/step - categorical_accuracy: 0.6437 - loss: 1.4843 - mean_io_u: 0.0383


371/Unknown  94s 146ms/step - categorical_accuracy: 0.6438 - loss: 1.4836 - mean_io_u: 0.0383


372/Unknown  95s 146ms/step - categorical_accuracy: 0.6440 - loss: 1.4828 - mean_io_u: 0.0384


373/Unknown  95s 146ms/step - categorical_accuracy: 0.6442 - loss: 1.4821 - mean_io_u: 0.0384


374/Unknown  95s 146ms/step - categorical_accuracy: 0.6443 - loss: 1.4814 - mean_io_u: 0.0385


375/Unknown  95s 145ms/step - categorical_accuracy: 0.6445 - loss: 1.4807 - mean_io_u: 0.0385


376/Unknown  95s 145ms/step - categorical_accuracy: 0.6446 - loss: 1.4799 - mean_io_u: 0.0385


377/Unknown  95s 145ms/step - categorical_accuracy: 0.6448 - loss: 1.4792 - mean_io_u: 0.0386


378/Unknown  95s 145ms/step - categorical_accuracy: 0.6449 - loss: 1.4785 - mean_io_u: 0.0386


379/Unknown  95s 145ms/step - categorical_accuracy: 0.6451 - loss: 1.4778 - mean_io_u: 0.0387


380/Unknown  95s 144ms/step - categorical_accuracy: 0.6452 - loss: 1.4770 - mean_io_u: 0.0387


381/Unknown  95s 144ms/step - categorical_accuracy: 0.6454 - loss: 1.4763 - mean_io_u: 0.0388


382/Unknown  95s 144ms/step - categorical_accuracy: 0.6456 - loss: 1.4756 - mean_io_u: 0.0388


383/Unknown  95s 144ms/step - categorical_accuracy: 0.6457 - loss: 1.4749 - mean_io_u: 0.0389


384/Unknown  95s 144ms/step - categorical_accuracy: 0.6459 - loss: 1.4742 - mean_io_u: 0.0389


385/Unknown  96s 144ms/step - categorical_accuracy: 0.6460 - loss: 1.4735 - mean_io_u: 0.0389


386/Unknown  96s 143ms/step - categorical_accuracy: 0.6462 - loss: 1.4728 - mean_io_u: 0.0390


387/Unknown  96s 143ms/step - categorical_accuracy: 0.6463 - loss: 1.4721 - mean_io_u: 0.0390


388/Unknown  96s 143ms/step - categorical_accuracy: 0.6465 - loss: 1.4714 - mean_io_u: 0.0391


389/Unknown  96s 143ms/step - categorical_accuracy: 0.6466 - loss: 1.4707 - mean_io_u: 0.0391


390/Unknown  96s 143ms/step - categorical_accuracy: 0.6468 - loss: 1.4700 - mean_io_u: 0.0392


391/Unknown  96s 143ms/step - categorical_accuracy: 0.6469 - loss: 1.4693 - mean_io_u: 0.0392


392/Unknown  96s 142ms/step - categorical_accuracy: 0.6470 - loss: 1.4686 - mean_io_u: 0.0392


393/Unknown  96s 142ms/step - categorical_accuracy: 0.6472 - loss: 1.4679 - mean_io_u: 0.0393


394/Unknown  96s 142ms/step - categorical_accuracy: 0.6473 - loss: 1.4672 - mean_io_u: 0.0393


395/Unknown  96s 142ms/step - categorical_accuracy: 0.6475 - loss: 1.4666 - mean_io_u: 0.0394


396/Unknown  96s 142ms/step - categorical_accuracy: 0.6476 - loss: 1.4659 - mean_io_u: 0.0394


397/Unknown  97s 142ms/step - categorical_accuracy: 0.6478 - loss: 1.4652 - mean_io_u: 0.0395


398/Unknown  97s 141ms/step - categorical_accuracy: 0.6479 - loss: 1.4645 - mean_io_u: 0.0395


399/Unknown  97s 141ms/step - categorical_accuracy: 0.6481 - loss: 1.4638 - mean_io_u: 0.0396


400/Unknown  97s 141ms/step - categorical_accuracy: 0.6482 - loss: 1.4631 - mean_io_u: 0.0396


401/Unknown  97s 141ms/step - categorical_accuracy: 0.6484 - loss: 1.4625 - mean_io_u: 0.0396


402/Unknown  97s 141ms/step - categorical_accuracy: 0.6485 - loss: 1.4618 - mean_io_u: 0.0397


403/Unknown  97s 141ms/step - categorical_accuracy: 0.6486 - loss: 1.4611 - mean_io_u: 0.0397


404/Unknown  97s 140ms/step - categorical_accuracy: 0.6488 - loss: 1.4604 - mean_io_u: 0.0398


405/Unknown  97s 140ms/step - categorical_accuracy: 0.6489 - loss: 1.4598 - mean_io_u: 0.0398


406/Unknown  97s 140ms/step - categorical_accuracy: 0.6491 - loss: 1.4591 - mean_io_u: 0.0399


407/Unknown  97s 140ms/step - categorical_accuracy: 0.6492 - loss: 1.4585 - mean_io_u: 0.0399


408/Unknown  97s 140ms/step - categorical_accuracy: 0.6493 - loss: 1.4578 - mean_io_u: 0.0400


409/Unknown  98s 140ms/step - categorical_accuracy: 0.6495 - loss: 1.4572 - mean_io_u: 0.0400


410/Unknown  98s 140ms/step - categorical_accuracy: 0.6496 - loss: 1.4565 - mean_io_u: 0.0401


411/Unknown  98s 140ms/step - categorical_accuracy: 0.6497 - loss: 1.4559 - mean_io_u: 0.0401


412/Unknown  98s 140ms/step - categorical_accuracy: 0.6499 - loss: 1.4552 - mean_io_u: 0.0401


413/Unknown  98s 139ms/step - categorical_accuracy: 0.6500 - loss: 1.4546 - mean_io_u: 0.0402


414/Unknown  98s 139ms/step - categorical_accuracy: 0.6502 - loss: 1.4539 - mean_io_u: 0.0402


415/Unknown  98s 139ms/step - categorical_accuracy: 0.6503 - loss: 1.4533 - mean_io_u: 0.0403


416/Unknown  98s 139ms/step - categorical_accuracy: 0.6504 - loss: 1.4526 - mean_io_u: 0.0403


417/Unknown  98s 139ms/step - categorical_accuracy: 0.6506 - loss: 1.4520 - mean_io_u: 0.0404


418/Unknown  98s 139ms/step - categorical_accuracy: 0.6507 - loss: 1.4513 - mean_io_u: 0.0404


419/Unknown  98s 138ms/step - categorical_accuracy: 0.6508 - loss: 1.4507 - mean_io_u: 0.0405


420/Unknown  98s 138ms/step - categorical_accuracy: 0.6510 - loss: 1.4501 - mean_io_u: 0.0405


421/Unknown  98s 138ms/step - categorical_accuracy: 0.6511 - loss: 1.4494 - mean_io_u: 0.0405


422/Unknown  98s 138ms/step - categorical_accuracy: 0.6512 - loss: 1.4488 - mean_io_u: 0.0406


423/Unknown  99s 138ms/step - categorical_accuracy: 0.6514 - loss: 1.4482 - mean_io_u: 0.0406


424/Unknown  99s 138ms/step - categorical_accuracy: 0.6515 - loss: 1.4475 - mean_io_u: 0.0407


425/Unknown  99s 137ms/step - categorical_accuracy: 0.6516 - loss: 1.4469 - mean_io_u: 0.0407


426/Unknown  99s 137ms/step - categorical_accuracy: 0.6518 - loss: 1.4463 - mean_io_u: 0.0408


427/Unknown  99s 137ms/step - categorical_accuracy: 0.6519 - loss: 1.4456 - mean_io_u: 0.0408


428/Unknown  99s 137ms/step - categorical_accuracy: 0.6520 - loss: 1.4450 - mean_io_u: 0.0409


429/Unknown  99s 137ms/step - categorical_accuracy: 0.6521 - loss: 1.4444 - mean_io_u: 0.0409


430/Unknown  99s 137ms/step - categorical_accuracy: 0.6523 - loss: 1.4438 - mean_io_u: 0.0410


431/Unknown  99s 137ms/step - categorical_accuracy: 0.6524 - loss: 1.4432 - mean_io_u: 0.0410


432/Unknown  99s 137ms/step - categorical_accuracy: 0.6525 - loss: 1.4425 - mean_io_u: 0.0410


433/Unknown  99s 136ms/step - categorical_accuracy: 0.6527 - loss: 1.4419 - mean_io_u: 0.0411


434/Unknown  99s 136ms/step - categorical_accuracy: 0.6528 - loss: 1.4413 - mean_io_u: 0.0411


435/Unknown  99s 136ms/step - categorical_accuracy: 0.6529 - loss: 1.4407 - mean_io_u: 0.0412


436/Unknown  100s 136ms/step - categorical_accuracy: 0.6531 - loss: 1.4401 - mean_io_u: 0.0412


437/Unknown  100s 136ms/step - categorical_accuracy: 0.6532 - loss: 1.4394 - mean_io_u: 0.0413


438/Unknown  100s 136ms/step - categorical_accuracy: 0.6533 - loss: 1.4388 - mean_io_u: 0.0413


439/Unknown  100s 136ms/step - categorical_accuracy: 0.6534 - loss: 1.4382 - mean_io_u: 0.0414


440/Unknown  100s 135ms/step - categorical_accuracy: 0.6536 - loss: 1.4376 - mean_io_u: 0.0414


441/Unknown  100s 135ms/step - categorical_accuracy: 0.6537 - loss: 1.4370 - mean_io_u: 0.0414


442/Unknown  100s 135ms/step - categorical_accuracy: 0.6538 - loss: 1.4364 - mean_io_u: 0.0415


443/Unknown  100s 135ms/step - categorical_accuracy: 0.6540 - loss: 1.4358 - mean_io_u: 0.0415


444/Unknown  100s 135ms/step - categorical_accuracy: 0.6541 - loss: 1.4352 - mean_io_u: 0.0416


445/Unknown  100s 135ms/step - categorical_accuracy: 0.6542 - loss: 1.4346 - mean_io_u: 0.0416


446/Unknown  100s 135ms/step - categorical_accuracy: 0.6543 - loss: 1.4340 - mean_io_u: 0.0417


447/Unknown  100s 135ms/step - categorical_accuracy: 0.6545 - loss: 1.4334 - mean_io_u: 0.0417


448/Unknown  101s 134ms/step - categorical_accuracy: 0.6546 - loss: 1.4328 - mean_io_u: 0.0418


449/Unknown  101s 134ms/step - categorical_accuracy: 0.6547 - loss: 1.4322 - mean_io_u: 0.0418


450/Unknown  101s 134ms/step - categorical_accuracy: 0.6548 - loss: 1.4316 - mean_io_u: 0.0418


451/Unknown  101s 134ms/step - categorical_accuracy: 0.6550 - loss: 1.4310 - mean_io_u: 0.0419


452/Unknown  101s 134ms/step - categorical_accuracy: 0.6551 - loss: 1.4304 - mean_io_u: 0.0419


453/Unknown  101s 134ms/step - categorical_accuracy: 0.6552 - loss: 1.4298 - mean_io_u: 0.0420


454/Unknown  101s 134ms/step - categorical_accuracy: 0.6553 - loss: 1.4292 - mean_io_u: 0.0420


455/Unknown  101s 134ms/step - categorical_accuracy: 0.6554 - loss: 1.4286 - mean_io_u: 0.0421


456/Unknown  101s 134ms/step - categorical_accuracy: 0.6556 - loss: 1.4281 - mean_io_u: 0.0421


457/Unknown  101s 133ms/step - categorical_accuracy: 0.6557 - loss: 1.4275 - mean_io_u: 0.0422


458/Unknown  101s 133ms/step - categorical_accuracy: 0.6558 - loss: 1.4269 - mean_io_u: 0.0422


459/Unknown  101s 133ms/step - categorical_accuracy: 0.6559 - loss: 1.4263 - mean_io_u: 0.0422


460/Unknown  102s 133ms/step - categorical_accuracy: 0.6560 - loss: 1.4258 - mean_io_u: 0.0423


461/Unknown  102s 133ms/step - categorical_accuracy: 0.6562 - loss: 1.4252 - mean_io_u: 0.0423


462/Unknown  102s 133ms/step - categorical_accuracy: 0.6563 - loss: 1.4246 - mean_io_u: 0.0424


463/Unknown  102s 133ms/step - categorical_accuracy: 0.6564 - loss: 1.4240 - mean_io_u: 0.0424


464/Unknown  102s 133ms/step - categorical_accuracy: 0.6565 - loss: 1.4235 - mean_io_u: 0.0425


465/Unknown  102s 132ms/step - categorical_accuracy: 0.6566 - loss: 1.4229 - mean_io_u: 0.0425


466/Unknown  102s 132ms/step - categorical_accuracy: 0.6567 - loss: 1.4224 - mean_io_u: 0.0426


467/Unknown  102s 132ms/step - categorical_accuracy: 0.6569 - loss: 1.4218 - mean_io_u: 0.0426


468/Unknown  102s 132ms/step - categorical_accuracy: 0.6570 - loss: 1.4212 - mean_io_u: 0.0427


469/Unknown  102s 132ms/step - categorical_accuracy: 0.6571 - loss: 1.4207 - mean_io_u: 0.0427


470/Unknown  102s 132ms/step - categorical_accuracy: 0.6572 - loss: 1.4201 - mean_io_u: 0.0427


471/Unknown  102s 132ms/step - categorical_accuracy: 0.6573 - loss: 1.4196 - mean_io_u: 0.0428


472/Unknown  102s 132ms/step - categorical_accuracy: 0.6574 - loss: 1.4190 - mean_io_u: 0.0428


473/Unknown  102s 131ms/step - categorical_accuracy: 0.6575 - loss: 1.4185 - mean_io_u: 0.0429


474/Unknown  103s 131ms/step - categorical_accuracy: 0.6577 - loss: 1.4179 - mean_io_u: 0.0429


475/Unknown  103s 131ms/step - categorical_accuracy: 0.6578 - loss: 1.4174 - mean_io_u: 0.0430


476/Unknown  103s 131ms/step - categorical_accuracy: 0.6579 - loss: 1.4168 - mean_io_u: 0.0430


477/Unknown  103s 131ms/step - categorical_accuracy: 0.6580 - loss: 1.4163 - mean_io_u: 0.0431


478/Unknown  103s 131ms/step - categorical_accuracy: 0.6581 - loss: 1.4157 - mean_io_u: 0.0431


479/Unknown  103s 131ms/step - categorical_accuracy: 0.6582 - loss: 1.4152 - mean_io_u: 0.0432


480/Unknown  103s 131ms/step - categorical_accuracy: 0.6583 - loss: 1.4146 - mean_io_u: 0.0432


481/Unknown  103s 131ms/step - categorical_accuracy: 0.6584 - loss: 1.4141 - mean_io_u: 0.0433


482/Unknown  103s 130ms/step - categorical_accuracy: 0.6585 - loss: 1.4135 - mean_io_u: 0.0433


483/Unknown  103s 130ms/step - categorical_accuracy: 0.6587 - loss: 1.4130 - mean_io_u: 0.0433


484/Unknown  103s 130ms/step - categorical_accuracy: 0.6588 - loss: 1.4125 - mean_io_u: 0.0434


485/Unknown  103s 130ms/step - categorical_accuracy: 0.6589 - loss: 1.4119 - mean_io_u: 0.0434


486/Unknown  103s 130ms/step - categorical_accuracy: 0.6590 - loss: 1.4114 - mean_io_u: 0.0435


487/Unknown  104s 130ms/step - categorical_accuracy: 0.6591 - loss: 1.4109 - mean_io_u: 0.0435


488/Unknown  104s 130ms/step - categorical_accuracy: 0.6592 - loss: 1.4103 - mean_io_u: 0.0436


489/Unknown  104s 130ms/step - categorical_accuracy: 0.6593 - loss: 1.4098 - mean_io_u: 0.0436


490/Unknown  104s 130ms/step - categorical_accuracy: 0.6594 - loss: 1.4093 - mean_io_u: 0.0437


491/Unknown  104s 130ms/step - categorical_accuracy: 0.6595 - loss: 1.4087 - mean_io_u: 0.0437


492/Unknown  104s 129ms/step - categorical_accuracy: 0.6596 - loss: 1.4082 - mean_io_u: 0.0438


493/Unknown  104s 129ms/step - categorical_accuracy: 0.6597 - loss: 1.4077 - mean_io_u: 0.0438


494/Unknown  104s 129ms/step - categorical_accuracy: 0.6599 - loss: 1.4071 - mean_io_u: 0.0439


495/Unknown  104s 129ms/step - categorical_accuracy: 0.6600 - loss: 1.4066 - mean_io_u: 0.0439


496/Unknown  104s 129ms/step - categorical_accuracy: 0.6601 - loss: 1.4061 - mean_io_u: 0.0439


497/Unknown  104s 129ms/step - categorical_accuracy: 0.6602 - loss: 1.4056 - mean_io_u: 0.0440


498/Unknown  104s 129ms/step - categorical_accuracy: 0.6603 - loss: 1.4050 - mean_io_u: 0.0440


499/Unknown  104s 129ms/step - categorical_accuracy: 0.6604 - loss: 1.4045 - mean_io_u: 0.0441


500/Unknown  105s 128ms/step - categorical_accuracy: 0.6605 - loss: 1.4040 - mean_io_u: 0.0441


501/Unknown  105s 128ms/step - categorical_accuracy: 0.6606 - loss: 1.4035 - mean_io_u: 0.0442


502/Unknown  105s 128ms/step - categorical_accuracy: 0.6607 - loss: 1.4029 - mean_io_u: 0.0442


503/Unknown  105s 128ms/step - categorical_accuracy: 0.6608 - loss: 1.4024 - mean_io_u: 0.0443


504/Unknown  105s 128ms/step - categorical_accuracy: 0.6609 - loss: 1.4019 - mean_io_u: 0.0443


505/Unknown  105s 128ms/step - categorical_accuracy: 0.6610 - loss: 1.4014 - mean_io_u: 0.0444


506/Unknown  105s 128ms/step - categorical_accuracy: 0.6611 - loss: 1.4009 - mean_io_u: 0.0444


507/Unknown  105s 127ms/step - categorical_accuracy: 0.6612 - loss: 1.4004 - mean_io_u: 0.0445


508/Unknown  105s 127ms/step - categorical_accuracy: 0.6613 - loss: 1.3998 - mean_io_u: 0.0445


509/Unknown  105s 127ms/step - categorical_accuracy: 0.6614 - loss: 1.3993 - mean_io_u: 0.0445


510/Unknown  105s 127ms/step - categorical_accuracy: 0.6615 - loss: 1.3988 - mean_io_u: 0.0446


511/Unknown  105s 127ms/step - categorical_accuracy: 0.6616 - loss: 1.3983 - mean_io_u: 0.0446


512/Unknown  105s 127ms/step - categorical_accuracy: 0.6617 - loss: 1.3978 - mean_io_u: 0.0447


513/Unknown  105s 127ms/step - categorical_accuracy: 0.6619 - loss: 1.3973 - mean_io_u: 0.0447


514/Unknown  105s 127ms/step - categorical_accuracy: 0.6620 - loss: 1.3968 - mean_io_u: 0.0448


515/Unknown  106s 127ms/step - categorical_accuracy: 0.6621 - loss: 1.3963 - mean_io_u: 0.0448


516/Unknown  106s 126ms/step - categorical_accuracy: 0.6622 - loss: 1.3958 - mean_io_u: 0.0449


517/Unknown  106s 126ms/step - categorical_accuracy: 0.6623 - loss: 1.3953 - mean_io_u: 0.0449


518/Unknown  106s 126ms/step - categorical_accuracy: 0.6624 - loss: 1.3948 - mean_io_u: 0.0450


519/Unknown  106s 126ms/step - categorical_accuracy: 0.6625 - loss: 1.3943 - mean_io_u: 0.0450


520/Unknown  106s 126ms/step - categorical_accuracy: 0.6626 - loss: 1.3938 - mean_io_u: 0.0451


521/Unknown  106s 126ms/step - categorical_accuracy: 0.6627 - loss: 1.3933 - mean_io_u: 0.0451


522/Unknown  106s 126ms/step - categorical_accuracy: 0.6628 - loss: 1.3928 - mean_io_u: 0.0451


523/Unknown  106s 126ms/step - categorical_accuracy: 0.6629 - loss: 1.3923 - mean_io_u: 0.0452


524/Unknown  106s 126ms/step - categorical_accuracy: 0.6630 - loss: 1.3918 - mean_io_u: 0.0452


525/Unknown  106s 126ms/step - categorical_accuracy: 0.6631 - loss: 1.3913 - mean_io_u: 0.0453


526/Unknown  106s 126ms/step - categorical_accuracy: 0.6632 - loss: 1.3908 - mean_io_u: 0.0453


527/Unknown  106s 125ms/step - categorical_accuracy: 0.6633 - loss: 1.3903 - mean_io_u: 0.0454


528/Unknown  107s 125ms/step - categorical_accuracy: 0.6634 - loss: 1.3899 - mean_io_u: 0.0454


529/Unknown  107s 125ms/step - categorical_accuracy: 0.6635 - loss: 1.3894 - mean_io_u: 0.0455


530/Unknown  107s 125ms/step - categorical_accuracy: 0.6636 - loss: 1.3889 - mean_io_u: 0.0455


531/Unknown  107s 125ms/step - categorical_accuracy: 0.6637 - loss: 1.3884 - mean_io_u: 0.0456


532/Unknown  107s 125ms/step - categorical_accuracy: 0.6637 - loss: 1.3879 - mean_io_u: 0.0456


533/Unknown  107s 125ms/step - categorical_accuracy: 0.6638 - loss: 1.3874 - mean_io_u: 0.0456


534/Unknown  107s 125ms/step - categorical_accuracy: 0.6639 - loss: 1.3870 - mean_io_u: 0.0457


535/Unknown  107s 125ms/step - categorical_accuracy: 0.6640 - loss: 1.3865 - mean_io_u: 0.0457


536/Unknown  107s 125ms/step - categorical_accuracy: 0.6641 - loss: 1.3860 - mean_io_u: 0.0458


537/Unknown  107s 125ms/step - categorical_accuracy: 0.6642 - loss: 1.3855 - mean_io_u: 0.0458


538/Unknown  107s 124ms/step - categorical_accuracy: 0.6643 - loss: 1.3850 - mean_io_u: 0.0459


539/Unknown  107s 124ms/step - categorical_accuracy: 0.6644 - loss: 1.3846 - mean_io_u: 0.0459


540/Unknown  107s 124ms/step - categorical_accuracy: 0.6645 - loss: 1.3841 - mean_io_u: 0.0460


541/Unknown  107s 124ms/step - categorical_accuracy: 0.6646 - loss: 1.3836 - mean_io_u: 0.0460


542/Unknown  108s 124ms/step - categorical_accuracy: 0.6647 - loss: 1.3832 - mean_io_u: 0.0461


543/Unknown  108s 124ms/step - categorical_accuracy: 0.6648 - loss: 1.3827 - mean_io_u: 0.0461


544/Unknown  108s 124ms/step - categorical_accuracy: 0.6649 - loss: 1.3822 - mean_io_u: 0.0462


545/Unknown  108s 124ms/step - categorical_accuracy: 0.6650 - loss: 1.3818 - mean_io_u: 0.0462


546/Unknown  108s 124ms/step - categorical_accuracy: 0.6651 - loss: 1.3813 - mean_io_u: 0.0462


547/Unknown  108s 124ms/step - categorical_accuracy: 0.6652 - loss: 1.3808 - mean_io_u: 0.0463


548/Unknown  108s 124ms/step - categorical_accuracy: 0.6653 - loss: 1.3804 - mean_io_u: 0.0463


549/Unknown  108s 124ms/step - categorical_accuracy: 0.6654 - loss: 1.3799 - mean_io_u: 0.0464


550/Unknown  108s 124ms/step - categorical_accuracy: 0.6655 - loss: 1.3794 - mean_io_u: 0.0464


551/Unknown  108s 124ms/step - categorical_accuracy: 0.6655 - loss: 1.3790 - mean_io_u: 0.0465


552/Unknown  108s 123ms/step - categorical_accuracy: 0.6656 - loss: 1.3785 - mean_io_u: 0.0465


553/Unknown  108s 123ms/step - categorical_accuracy: 0.6657 - loss: 1.3780 - mean_io_u: 0.0466


554/Unknown  109s 123ms/step - categorical_accuracy: 0.6658 - loss: 1.3776 - mean_io_u: 0.0466


555/Unknown  109s 123ms/step - categorical_accuracy: 0.6659 - loss: 1.3771 - mean_io_u: 0.0467


556/Unknown  109s 123ms/step - categorical_accuracy: 0.6660 - loss: 1.3767 - mean_io_u: 0.0467


557/Unknown  109s 123ms/step - categorical_accuracy: 0.6661 - loss: 1.3762 - mean_io_u: 0.0468


558/Unknown  109s 123ms/step - categorical_accuracy: 0.6662 - loss: 1.3757 - mean_io_u: 0.0468


559/Unknown  109s 123ms/step - categorical_accuracy: 0.6663 - loss: 1.3753 - mean_io_u: 0.0469


560/Unknown  109s 123ms/step - categorical_accuracy: 0.6664 - loss: 1.3748 - mean_io_u: 0.0469


561/Unknown  109s 123ms/step - categorical_accuracy: 0.6665 - loss: 1.3744 - mean_io_u: 0.0469


562/Unknown  109s 123ms/step - categorical_accuracy: 0.6666 - loss: 1.3739 - mean_io_u: 0.0470


563/Unknown  109s 123ms/step - categorical_accuracy: 0.6667 - loss: 1.3735 - mean_io_u: 0.0470


564/Unknown  109s 122ms/step - categorical_accuracy: 0.6667 - loss: 1.3730 - mean_io_u: 0.0471


565/Unknown  109s 122ms/step - categorical_accuracy: 0.6668 - loss: 1.3726 - mean_io_u: 0.0471


566/Unknown  110s 122ms/step - categorical_accuracy: 0.6669 - loss: 1.3721 - mean_io_u: 0.0472


567/Unknown  110s 122ms/step - categorical_accuracy: 0.6670 - loss: 1.3717 - mean_io_u: 0.0472


568/Unknown  110s 122ms/step - categorical_accuracy: 0.6671 - loss: 1.3712 - mean_io_u: 0.0473


569/Unknown  110s 122ms/step - categorical_accuracy: 0.6672 - loss: 1.3708 - mean_io_u: 0.0473


570/Unknown  110s 122ms/step - categorical_accuracy: 0.6673 - loss: 1.3704 - mean_io_u: 0.0474


571/Unknown  110s 122ms/step - categorical_accuracy: 0.6674 - loss: 1.3699 - mean_io_u: 0.0474


572/Unknown  110s 122ms/step - categorical_accuracy: 0.6675 - loss: 1.3695 - mean_io_u: 0.0475


573/Unknown  110s 122ms/step - categorical_accuracy: 0.6675 - loss: 1.3690 - mean_io_u: 0.0475


574/Unknown  110s 122ms/step - categorical_accuracy: 0.6676 - loss: 1.3686 - mean_io_u: 0.0476


575/Unknown  110s 122ms/step - categorical_accuracy: 0.6677 - loss: 1.3681 - mean_io_u: 0.0476


576/Unknown  110s 122ms/step - categorical_accuracy: 0.6678 - loss: 1.3677 - mean_io_u: 0.0476


577/Unknown  111s 122ms/step - categorical_accuracy: 0.6679 - loss: 1.3673 - mean_io_u: 0.0477


578/Unknown  111s 122ms/step - categorical_accuracy: 0.6680 - loss: 1.3668 - mean_io_u: 0.0477


579/Unknown  111s 122ms/step - categorical_accuracy: 0.6681 - loss: 1.3664 - mean_io_u: 0.0478


580/Unknown  111s 122ms/step - categorical_accuracy: 0.6682 - loss: 1.3660 - mean_io_u: 0.0478


581/Unknown  111s 122ms/step - categorical_accuracy: 0.6682 - loss: 1.3655 - mean_io_u: 0.0479


582/Unknown  111s 121ms/step - categorical_accuracy: 0.6683 - loss: 1.3651 - mean_io_u: 0.0479


583/Unknown  111s 121ms/step - categorical_accuracy: 0.6684 - loss: 1.3647 - mean_io_u: 0.0480


584/Unknown  111s 121ms/step - categorical_accuracy: 0.6685 - loss: 1.3642 - mean_io_u: 0.0480


585/Unknown  111s 121ms/step - categorical_accuracy: 0.6686 - loss: 1.3638 - mean_io_u: 0.0481


586/Unknown  111s 121ms/step - categorical_accuracy: 0.6687 - loss: 1.3634 - mean_io_u: 0.0481


587/Unknown  111s 121ms/step - categorical_accuracy: 0.6688 - loss: 1.3629 - mean_io_u: 0.0481


588/Unknown  111s 121ms/step - categorical_accuracy: 0.6688 - loss: 1.3625 - mean_io_u: 0.0482


589/Unknown  112s 121ms/step - categorical_accuracy: 0.6689 - loss: 1.3621 - mean_io_u: 0.0482


590/Unknown  112s 121ms/step - categorical_accuracy: 0.6690 - loss: 1.3616 - mean_io_u: 0.0483


591/Unknown  112s 121ms/step - categorical_accuracy: 0.6691 - loss: 1.3612 - mean_io_u: 0.0483


592/Unknown  112s 121ms/step - categorical_accuracy: 0.6692 - loss: 1.3608 - mean_io_u: 0.0484


593/Unknown  112s 121ms/step - categorical_accuracy: 0.6693 - loss: 1.3604 - mean_io_u: 0.0484


594/Unknown  112s 121ms/step - categorical_accuracy: 0.6694 - loss: 1.3599 - mean_io_u: 0.0485


595/Unknown  112s 120ms/step - categorical_accuracy: 0.6694 - loss: 1.3595 - mean_io_u: 0.0485


596/Unknown  112s 120ms/step - categorical_accuracy: 0.6695 - loss: 1.3591 - mean_io_u: 0.0486


597/Unknown  112s 120ms/step - categorical_accuracy: 0.6696 - loss: 1.3587 - mean_io_u: 0.0486


598/Unknown  112s 120ms/step - categorical_accuracy: 0.6697 - loss: 1.3582 - mean_io_u: 0.0486


599/Unknown  112s 120ms/step - categorical_accuracy: 0.6698 - loss: 1.3578 - mean_io_u: 0.0487


600/Unknown  112s 120ms/step - categorical_accuracy: 0.6699 - loss: 1.3574 - mean_io_u: 0.0487


601/Unknown  113s 120ms/step - categorical_accuracy: 0.6700 - loss: 1.3570 - mean_io_u: 0.0488


602/Unknown  113s 120ms/step - categorical_accuracy: 0.6700 - loss: 1.3566 - mean_io_u: 0.0488


603/Unknown  113s 120ms/step - categorical_accuracy: 0.6701 - loss: 1.3561 - mean_io_u: 0.0489


604/Unknown  113s 120ms/step - categorical_accuracy: 0.6702 - loss: 1.3557 - mean_io_u: 0.0489


605/Unknown  113s 120ms/step - categorical_accuracy: 0.6703 - loss: 1.3553 - mean_io_u: 0.0490


606/Unknown  113s 120ms/step - categorical_accuracy: 0.6704 - loss: 1.3549 - mean_io_u: 0.0490


607/Unknown  113s 120ms/step - categorical_accuracy: 0.6705 - loss: 1.3545 - mean_io_u: 0.0491


608/Unknown  113s 120ms/step - categorical_accuracy: 0.6705 - loss: 1.3541 - mean_io_u: 0.0491


609/Unknown  113s 120ms/step - categorical_accuracy: 0.6706 - loss: 1.3537 - mean_io_u: 0.0491


610/Unknown  113s 120ms/step - categorical_accuracy: 0.6707 - loss: 1.3532 - mean_io_u: 0.0492


611/Unknown  113s 120ms/step - categorical_accuracy: 0.6708 - loss: 1.3528 - mean_io_u: 0.0492


612/Unknown  113s 119ms/step - categorical_accuracy: 0.6709 - loss: 1.3524 - mean_io_u: 0.0493


613/Unknown  113s 119ms/step - categorical_accuracy: 0.6709 - loss: 1.3520 - mean_io_u: 0.0493


614/Unknown  114s 119ms/step - categorical_accuracy: 0.6710 - loss: 1.3516 - mean_io_u: 0.0494


615/Unknown  114s 119ms/step - categorical_accuracy: 0.6711 - loss: 1.3512 - mean_io_u: 0.0494


616/Unknown  114s 119ms/step - categorical_accuracy: 0.6712 - loss: 1.3508 - mean_io_u: 0.0495


617/Unknown  114s 119ms/step - categorical_accuracy: 0.6713 - loss: 1.3504 - mean_io_u: 0.0495


618/Unknown  114s 119ms/step - categorical_accuracy: 0.6714 - loss: 1.3500 - mean_io_u: 0.0496


619/Unknown  114s 119ms/step - categorical_accuracy: 0.6714 - loss: 1.3495 - mean_io_u: 0.0496


620/Unknown  114s 119ms/step - categorical_accuracy: 0.6715 - loss: 1.3491 - mean_io_u: 0.0497


621/Unknown  114s 119ms/step - categorical_accuracy: 0.6716 - loss: 1.3487 - mean_io_u: 0.0497


622/Unknown  114s 119ms/step - categorical_accuracy: 0.6717 - loss: 1.3483 - mean_io_u: 0.0497


623/Unknown  114s 119ms/step - categorical_accuracy: 0.6718 - loss: 1.3479 - mean_io_u: 0.0498


624/Unknown  114s 119ms/step - categorical_accuracy: 0.6718 - loss: 1.3475 - mean_io_u: 0.0498


625/Unknown  114s 119ms/step - categorical_accuracy: 0.6719 - loss: 1.3471 - mean_io_u: 0.0499


626/Unknown  115s 119ms/step - categorical_accuracy: 0.6720 - loss: 1.3467 - mean_io_u: 0.0499


627/Unknown  115s 119ms/step - categorical_accuracy: 0.6721 - loss: 1.3463 - mean_io_u: 0.0500


628/Unknown  115s 118ms/step - categorical_accuracy: 0.6722 - loss: 1.3459 - mean_io_u: 0.0500


629/Unknown  115s 118ms/step - categorical_accuracy: 0.6722 - loss: 1.3455 - mean_io_u: 0.0501


630/Unknown  115s 118ms/step - categorical_accuracy: 0.6723 - loss: 1.3451 - mean_io_u: 0.0501


631/Unknown  115s 118ms/step - categorical_accuracy: 0.6724 - loss: 1.3447 - mean_io_u: 0.0502


632/Unknown  115s 118ms/step - categorical_accuracy: 0.6725 - loss: 1.3443 - mean_io_u: 0.0502


633/Unknown  115s 118ms/step - categorical_accuracy: 0.6726 - loss: 1.3439 - mean_io_u: 0.0502


634/Unknown  115s 118ms/step - categorical_accuracy: 0.6726 - loss: 1.3435 - mean_io_u: 0.0503


635/Unknown  115s 118ms/step - categorical_accuracy: 0.6727 - loss: 1.3431 - mean_io_u: 0.0503


636/Unknown  115s 118ms/step - categorical_accuracy: 0.6728 - loss: 1.3428 - mean_io_u: 0.0504


637/Unknown  115s 118ms/step - categorical_accuracy: 0.6729 - loss: 1.3424 - mean_io_u: 0.0504


638/Unknown  115s 118ms/step - categorical_accuracy: 0.6730 - loss: 1.3420 - mean_io_u: 0.0505


639/Unknown  115s 118ms/step - categorical_accuracy: 0.6730 - loss: 1.3416 - mean_io_u: 0.0505


640/Unknown  116s 118ms/step - categorical_accuracy: 0.6731 - loss: 1.3412 - mean_io_u: 0.0506


641/Unknown  116s 118ms/step - categorical_accuracy: 0.6732 - loss: 1.3408 - mean_io_u: 0.0506


642/Unknown  116s 117ms/step - categorical_accuracy: 0.6733 - loss: 1.3404 - mean_io_u: 0.0507


643/Unknown  116s 117ms/step - categorical_accuracy: 0.6733 - loss: 1.3400 - mean_io_u: 0.0507


644/Unknown  116s 117ms/step - categorical_accuracy: 0.6734 - loss: 1.3396 - mean_io_u: 0.0507


645/Unknown  116s 117ms/step - categorical_accuracy: 0.6735 - loss: 1.3392 - mean_io_u: 0.0508


646/Unknown  116s 117ms/step - categorical_accuracy: 0.6736 - loss: 1.3389 - mean_io_u: 0.0508


647/Unknown  116s 117ms/step - categorical_accuracy: 0.6737 - loss: 1.3385 - mean_io_u: 0.0509


648/Unknown  116s 117ms/step - categorical_accuracy: 0.6737 - loss: 1.3381 - mean_io_u: 0.0509


649/Unknown  116s 117ms/step - categorical_accuracy: 0.6738 - loss: 1.3377 - mean_io_u: 0.0510


650/Unknown  116s 117ms/step - categorical_accuracy: 0.6739 - loss: 1.3373 - mean_io_u: 0.0510


651/Unknown  116s 117ms/step - categorical_accuracy: 0.6740 - loss: 1.3369 - mean_io_u: 0.0511


652/Unknown  116s 117ms/step - categorical_accuracy: 0.6740 - loss: 1.3365 - mean_io_u: 0.0511


653/Unknown  116s 117ms/step - categorical_accuracy: 0.6741 - loss: 1.3362 - mean_io_u: 0.0511


654/Unknown  116s 116ms/step - categorical_accuracy: 0.6742 - loss: 1.3358 - mean_io_u: 0.0512


655/Unknown  117s 116ms/step - categorical_accuracy: 0.6743 - loss: 1.3354 - mean_io_u: 0.0512


656/Unknown  117s 116ms/step - categorical_accuracy: 0.6743 - loss: 1.3350 - mean_io_u: 0.0513


657/Unknown  117s 116ms/step - categorical_accuracy: 0.6744 - loss: 1.3346 - mean_io_u: 0.0513


658/Unknown  117s 116ms/step - categorical_accuracy: 0.6745 - loss: 1.3343 - mean_io_u: 0.0514


659/Unknown  117s 116ms/step - categorical_accuracy: 0.6746 - loss: 1.3339 - mean_io_u: 0.0514


660/Unknown  117s 116ms/step - categorical_accuracy: 0.6747 - loss: 1.3335 - mean_io_u: 0.0515


661/Unknown  117s 116ms/step - categorical_accuracy: 0.6747 - loss: 1.3331 - mean_io_u: 0.0515


662/Unknown  117s 116ms/step - categorical_accuracy: 0.6748 - loss: 1.3327 - mean_io_u: 0.0515


663/Unknown  117s 116ms/step - categorical_accuracy: 0.6749 - loss: 1.3324 - mean_io_u: 0.0516


664/Unknown  117s 116ms/step - categorical_accuracy: 0.6750 - loss: 1.3320 - mean_io_u: 0.0516


665/Unknown  117s 116ms/step - categorical_accuracy: 0.6750 - loss: 1.3316 - mean_io_u: 0.0517


666/Unknown  117s 116ms/step - categorical_accuracy: 0.6751 - loss: 1.3312 - mean_io_u: 0.0517


667/Unknown  117s 116ms/step - categorical_accuracy: 0.6752 - loss: 1.3309 - mean_io_u: 0.0518


668/Unknown  118s 116ms/step - categorical_accuracy: 0.6753 - loss: 1.3305 - mean_io_u: 0.0518


669/Unknown  118s 116ms/step - categorical_accuracy: 0.6753 - loss: 1.3301 - mean_io_u: 0.0519


670/Unknown  118s 115ms/step - categorical_accuracy: 0.6754 - loss: 1.3297 - mean_io_u: 0.0519


671/Unknown  118s 115ms/step - categorical_accuracy: 0.6755 - loss: 1.3294 - mean_io_u: 0.0519


672/Unknown  118s 115ms/step - categorical_accuracy: 0.6756 - loss: 1.3290 - mean_io_u: 0.0520


673/Unknown  118s 115ms/step - categorical_accuracy: 0.6756 - loss: 1.3286 - mean_io_u: 0.0520


674/Unknown  118s 115ms/step - categorical_accuracy: 0.6757 - loss: 1.3283 - mean_io_u: 0.0521


675/Unknown  118s 115ms/step - categorical_accuracy: 0.6758 - loss: 1.3279 - mean_io_u: 0.0521


676/Unknown  118s 115ms/step - categorical_accuracy: 0.6759 - loss: 1.3275 - mean_io_u: 0.0522


677/Unknown  118s 115ms/step - categorical_accuracy: 0.6759 - loss: 1.3272 - mean_io_u: 0.0522


678/Unknown  118s 115ms/step - categorical_accuracy: 0.6760 - loss: 1.3268 - mean_io_u: 0.0522


679/Unknown  118s 115ms/step - categorical_accuracy: 0.6761 - loss: 1.3264 - mean_io_u: 0.0523


680/Unknown  118s 115ms/step - categorical_accuracy: 0.6761 - loss: 1.3261 - mean_io_u: 0.0523


681/Unknown  119s 115ms/step - categorical_accuracy: 0.6762 - loss: 1.3257 - mean_io_u: 0.0524


682/Unknown  119s 115ms/step - categorical_accuracy: 0.6763 - loss: 1.3253 - mean_io_u: 0.0524


683/Unknown  119s 115ms/step - categorical_accuracy: 0.6764 - loss: 1.3250 - mean_io_u: 0.0525


684/Unknown  119s 115ms/step - categorical_accuracy: 0.6764 - loss: 1.3246 - mean_io_u: 0.0525


685/Unknown  119s 115ms/step - categorical_accuracy: 0.6765 - loss: 1.3243 - mean_io_u: 0.0526


686/Unknown  119s 115ms/step - categorical_accuracy: 0.6766 - loss: 1.3239 - mean_io_u: 0.0526


687/Unknown  119s 115ms/step - categorical_accuracy: 0.6767 - loss: 1.3235 - mean_io_u: 0.0526


688/Unknown  119s 114ms/step - categorical_accuracy: 0.6767 - loss: 1.3232 - mean_io_u: 0.0527


689/Unknown  119s 114ms/step - categorical_accuracy: 0.6768 - loss: 1.3228 - mean_io_u: 0.0527


690/Unknown  119s 114ms/step - categorical_accuracy: 0.6769 - loss: 1.3225 - mean_io_u: 0.0528


691/Unknown  119s 114ms/step - categorical_accuracy: 0.6769 - loss: 1.3221 - mean_io_u: 0.0528


692/Unknown  119s 114ms/step - categorical_accuracy: 0.6770 - loss: 1.3218 - mean_io_u: 0.0528


693/Unknown  120s 114ms/step - categorical_accuracy: 0.6771 - loss: 1.3214 - mean_io_u: 0.0529


694/Unknown  120s 114ms/step - categorical_accuracy: 0.6771 - loss: 1.3211 - mean_io_u: 0.0529


695/Unknown  120s 114ms/step - categorical_accuracy: 0.6772 - loss: 1.3207 - mean_io_u: 0.0530


696/Unknown  120s 114ms/step - categorical_accuracy: 0.6773 - loss: 1.3204 - mean_io_u: 0.0530


697/Unknown  120s 114ms/step - categorical_accuracy: 0.6774 - loss: 1.3200 - mean_io_u: 0.0530


698/Unknown  120s 114ms/step - categorical_accuracy: 0.6774 - loss: 1.3197 - mean_io_u: 0.0531


699/Unknown  120s 114ms/step - categorical_accuracy: 0.6775 - loss: 1.3193 - mean_io_u: 0.0531


700/Unknown  120s 114ms/step - categorical_accuracy: 0.6776 - loss: 1.3190 - mean_io_u: 0.0532


701/Unknown  120s 114ms/step - categorical_accuracy: 0.6776 - loss: 1.3186 - mean_io_u: 0.0532


702/Unknown  120s 114ms/step - categorical_accuracy: 0.6777 - loss: 1.3183 - mean_io_u: 0.0533


703/Unknown  120s 114ms/step - categorical_accuracy: 0.6778 - loss: 1.3179 - mean_io_u: 0.0533


704/Unknown  120s 114ms/step - categorical_accuracy: 0.6778 - loss: 1.3176 - mean_io_u: 0.0533


705/Unknown  121s 114ms/step - categorical_accuracy: 0.6779 - loss: 1.3172 - mean_io_u: 0.0534


706/Unknown  121s 114ms/step - categorical_accuracy: 0.6780 - loss: 1.3169 - mean_io_u: 0.0534


707/Unknown  121s 114ms/step - categorical_accuracy: 0.6780 - loss: 1.3165 - mean_io_u: 0.0535


708/Unknown  121s 114ms/step - categorical_accuracy: 0.6781 - loss: 1.3162 - mean_io_u: 0.0535


709/Unknown  121s 114ms/step - categorical_accuracy: 0.6782 - loss: 1.3158 - mean_io_u: 0.0535


710/Unknown  121s 114ms/step - categorical_accuracy: 0.6783 - loss: 1.3155 - mean_io_u: 0.0536


711/Unknown  121s 114ms/step - categorical_accuracy: 0.6783 - loss: 1.3152 - mean_io_u: 0.0536


712/Unknown  121s 114ms/step - categorical_accuracy: 0.6784 - loss: 1.3148 - mean_io_u: 0.0537


713/Unknown  121s 114ms/step - categorical_accuracy: 0.6785 - loss: 1.3145 - mean_io_u: 0.0537


714/Unknown  121s 114ms/step - categorical_accuracy: 0.6785 - loss: 1.3141 - mean_io_u: 0.0538


715/Unknown  121s 114ms/step - categorical_accuracy: 0.6786 - loss: 1.3138 - mean_io_u: 0.0538


716/Unknown  122s 113ms/step - categorical_accuracy: 0.6787 - loss: 1.3135 - mean_io_u: 0.0538


717/Unknown  122s 113ms/step - categorical_accuracy: 0.6787 - loss: 1.3131 - mean_io_u: 0.0539


718/Unknown  122s 113ms/step - categorical_accuracy: 0.6788 - loss: 1.3128 - mean_io_u: 0.0539


719/Unknown  122s 113ms/step - categorical_accuracy: 0.6789 - loss: 1.3125 - mean_io_u: 0.0540


720/Unknown  122s 113ms/step - categorical_accuracy: 0.6789 - loss: 1.3121 - mean_io_u: 0.0540


721/Unknown  122s 113ms/step - categorical_accuracy: 0.6790 - loss: 1.3118 - mean_io_u: 0.0540


722/Unknown  122s 113ms/step - categorical_accuracy: 0.6791 - loss: 1.3115 - mean_io_u: 0.0541


723/Unknown  122s 113ms/step - categorical_accuracy: 0.6791 - loss: 1.3111 - mean_io_u: 0.0541


724/Unknown  122s 113ms/step - categorical_accuracy: 0.6792 - loss: 1.3108 - mean_io_u: 0.0542


725/Unknown  122s 113ms/step - categorical_accuracy: 0.6793 - loss: 1.3105 - mean_io_u: 0.0542


726/Unknown  122s 113ms/step - categorical_accuracy: 0.6793 - loss: 1.3101 - mean_io_u: 0.0542


727/Unknown  122s 113ms/step - categorical_accuracy: 0.6794 - loss: 1.3098 - mean_io_u: 0.0543


728/Unknown  123s 113ms/step - categorical_accuracy: 0.6795 - loss: 1.3095 - mean_io_u: 0.0543


729/Unknown  123s 113ms/step - categorical_accuracy: 0.6795 - loss: 1.3091 - mean_io_u: 0.0544


730/Unknown  123s 113ms/step - categorical_accuracy: 0.6796 - loss: 1.3088 - mean_io_u: 0.0544


731/Unknown  123s 113ms/step - categorical_accuracy: 0.6796 - loss: 1.3085 - mean_io_u: 0.0544


732/Unknown  123s 113ms/step - categorical_accuracy: 0.6797 - loss: 1.3081 - mean_io_u: 0.0545


733/Unknown  123s 113ms/step - categorical_accuracy: 0.6798 - loss: 1.3078 - mean_io_u: 0.0545


734/Unknown  123s 113ms/step - categorical_accuracy: 0.6798 - loss: 1.3075 - mean_io_u: 0.0546


735/Unknown  123s 113ms/step - categorical_accuracy: 0.6799 - loss: 1.3072 - mean_io_u: 0.0546


736/Unknown  123s 113ms/step - categorical_accuracy: 0.6800 - loss: 1.3068 - mean_io_u: 0.0546


737/Unknown  123s 112ms/step - categorical_accuracy: 0.6800 - loss: 1.3065 - mean_io_u: 0.0547


738/Unknown  123s 112ms/step - categorical_accuracy: 0.6801 - loss: 1.3062 - mean_io_u: 0.0547


739/Unknown  123s 112ms/step - categorical_accuracy: 0.6802 - loss: 1.3059 - mean_io_u: 0.0548


740/Unknown  123s 112ms/step - categorical_accuracy: 0.6802 - loss: 1.3055 - mean_io_u: 0.0548


741/Unknown  123s 112ms/step - categorical_accuracy: 0.6803 - loss: 1.3052 - mean_io_u: 0.0548


742/Unknown  124s 112ms/step - categorical_accuracy: 0.6804 - loss: 1.3049 - mean_io_u: 0.0549


743/Unknown  124s 112ms/step - categorical_accuracy: 0.6804 - loss: 1.3045 - mean_io_u: 0.0549


744/Unknown  124s 112ms/step - categorical_accuracy: 0.6805 - loss: 1.3042 - mean_io_u: 0.0550


745/Unknown  124s 112ms/step - categorical_accuracy: 0.6806 - loss: 1.3039 - mean_io_u: 0.0550


746/Unknown  124s 112ms/step - categorical_accuracy: 0.6806 - loss: 1.3036 - mean_io_u: 0.0550


747/Unknown  124s 112ms/step - categorical_accuracy: 0.6807 - loss: 1.3033 - mean_io_u: 0.0551


748/Unknown  124s 112ms/step - categorical_accuracy: 0.6808 - loss: 1.3029 - mean_io_u: 0.0551


749/Unknown  124s 112ms/step - categorical_accuracy: 0.6808 - loss: 1.3026 - mean_io_u: 0.0552


750/Unknown  124s 112ms/step - categorical_accuracy: 0.6809 - loss: 1.3023 - mean_io_u: 0.0552


751/Unknown  124s 112ms/step - categorical_accuracy: 0.6809 - loss: 1.3020 - mean_io_u: 0.0552


752/Unknown  124s 112ms/step - categorical_accuracy: 0.6810 - loss: 1.3017 - mean_io_u: 0.0553


753/Unknown  124s 112ms/step - categorical_accuracy: 0.6811 - loss: 1.3013 - mean_io_u: 0.0553


754/Unknown  125s 112ms/step - categorical_accuracy: 0.6811 - loss: 1.3010 - mean_io_u: 0.0554


755/Unknown  125s 112ms/step - categorical_accuracy: 0.6812 - loss: 1.3007 - mean_io_u: 0.0554


756/Unknown  125s 112ms/step - categorical_accuracy: 0.6813 - loss: 1.3004 - mean_io_u: 0.0554


757/Unknown  125s 112ms/step - categorical_accuracy: 0.6813 - loss: 1.3001 - mean_io_u: 0.0555


758/Unknown  125s 111ms/step - categorical_accuracy: 0.6814 - loss: 1.2998 - mean_io_u: 0.0555


759/Unknown  125s 111ms/step - categorical_accuracy: 0.6814 - loss: 1.2995 - mean_io_u: 0.0556


760/Unknown  125s 111ms/step - categorical_accuracy: 0.6815 - loss: 1.2991 - mean_io_u: 0.0556


761/Unknown  125s 111ms/step - categorical_accuracy: 0.6816 - loss: 1.2988 - mean_io_u: 0.0556


762/Unknown  125s 111ms/step - categorical_accuracy: 0.6816 - loss: 1.2985 - mean_io_u: 0.0557


763/Unknown  125s 111ms/step - categorical_accuracy: 0.6817 - loss: 1.2982 - mean_io_u: 0.0557


764/Unknown  125s 111ms/step - categorical_accuracy: 0.6818 - loss: 1.2979 - mean_io_u: 0.0557


765/Unknown  125s 111ms/step - categorical_accuracy: 0.6818 - loss: 1.2976 - mean_io_u: 0.0558


766/Unknown  125s 111ms/step - categorical_accuracy: 0.6819 - loss: 1.2973 - mean_io_u: 0.0558


767/Unknown  126s 111ms/step - categorical_accuracy: 0.6819 - loss: 1.2970 - mean_io_u: 0.0559


768/Unknown  126s 111ms/step - categorical_accuracy: 0.6820 - loss: 1.2967 - mean_io_u: 0.0559


769/Unknown  126s 111ms/step - categorical_accuracy: 0.6821 - loss: 1.2963 - mean_io_u: 0.0559


770/Unknown  126s 111ms/step - categorical_accuracy: 0.6821 - loss: 1.2960 - mean_io_u: 0.0560


771/Unknown  126s 111ms/step - categorical_accuracy: 0.6822 - loss: 1.2957 - mean_io_u: 0.0560


772/Unknown  126s 111ms/step - categorical_accuracy: 0.6822 - loss: 1.2954 - mean_io_u: 0.0561


773/Unknown  126s 111ms/step - categorical_accuracy: 0.6823 - loss: 1.2951 - mean_io_u: 0.0561


774/Unknown  126s 111ms/step - categorical_accuracy: 0.6824 - loss: 1.2948 - mean_io_u: 0.0561


775/Unknown  126s 111ms/step - categorical_accuracy: 0.6824 - loss: 1.2945 - mean_io_u: 0.0562


776/Unknown  126s 111ms/step - categorical_accuracy: 0.6825 - loss: 1.2942 - mean_io_u: 0.0562


777/Unknown  126s 111ms/step - categorical_accuracy: 0.6825 - loss: 1.2939 - mean_io_u: 0.0562


778/Unknown  126s 111ms/step - categorical_accuracy: 0.6826 - loss: 1.2936 - mean_io_u: 0.0563


779/Unknown  126s 111ms/step - categorical_accuracy: 0.6827 - loss: 1.2933 - mean_io_u: 0.0563


780/Unknown  127s 111ms/step - categorical_accuracy: 0.6827 - loss: 1.2930 - mean_io_u: 0.0564


781/Unknown  127s 110ms/step - categorical_accuracy: 0.6828 - loss: 1.2927 - mean_io_u: 0.0564


782/Unknown  127s 110ms/step - categorical_accuracy: 0.6829 - loss: 1.2924 - mean_io_u: 0.0564


783/Unknown  127s 110ms/step - categorical_accuracy: 0.6829 - loss: 1.2921 - mean_io_u: 0.0565


784/Unknown  127s 110ms/step - categorical_accuracy: 0.6830 - loss: 1.2918 - mean_io_u: 0.0565


785/Unknown  127s 110ms/step - categorical_accuracy: 0.6830 - loss: 1.2915 - mean_io_u: 0.0565


786/Unknown  127s 110ms/step - categorical_accuracy: 0.6831 - loss: 1.2912 - mean_io_u: 0.0566


787/Unknown  127s 110ms/step - categorical_accuracy: 0.6831 - loss: 1.2909 - mean_io_u: 0.0566


788/Unknown  127s 110ms/step - categorical_accuracy: 0.6832 - loss: 1.2906 - mean_io_u: 0.0567


789/Unknown  127s 110ms/step - categorical_accuracy: 0.6833 - loss: 1.2903 - mean_io_u: 0.0567


790/Unknown  127s 110ms/step - categorical_accuracy: 0.6833 - loss: 1.2900 - mean_io_u: 0.0567


791/Unknown  127s 110ms/step - categorical_accuracy: 0.6834 - loss: 1.2897 - mean_io_u: 0.0568


792/Unknown  127s 110ms/step - categorical_accuracy: 0.6834 - loss: 1.2894 - mean_io_u: 0.0568


793/Unknown  127s 110ms/step - categorical_accuracy: 0.6835 - loss: 1.2891 - mean_io_u: 0.0569


794/Unknown  127s 110ms/step - categorical_accuracy: 0.6836 - loss: 1.2888 - mean_io_u: 0.0569


795/Unknown  128s 110ms/step - categorical_accuracy: 0.6836 - loss: 1.2885 - mean_io_u: 0.0569


796/Unknown  128s 110ms/step - categorical_accuracy: 0.6837 - loss: 1.2882 - mean_io_u: 0.0570


797/Unknown  128s 110ms/step - categorical_accuracy: 0.6837 - loss: 1.2879 - mean_io_u: 0.0570


798/Unknown  128s 109ms/step - categorical_accuracy: 0.6838 - loss: 1.2876 - mean_io_u: 0.0570


799/Unknown  128s 109ms/step - categorical_accuracy: 0.6839 - loss: 1.2873 - mean_io_u: 0.0571


800/Unknown  128s 109ms/step - categorical_accuracy: 0.6839 - loss: 1.2870 - mean_io_u: 0.0571


801/Unknown  128s 109ms/step - categorical_accuracy: 0.6840 - loss: 1.2867 - mean_io_u: 0.0572


802/Unknown  128s 109ms/step - categorical_accuracy: 0.6840 - loss: 1.2864 - mean_io_u: 0.0572


803/Unknown  128s 109ms/step - categorical_accuracy: 0.6841 - loss: 1.2861 - mean_io_u: 0.0572


804/Unknown  128s 109ms/step - categorical_accuracy: 0.6841 - loss: 1.2858 - mean_io_u: 0.0573


805/Unknown  128s 109ms/step - categorical_accuracy: 0.6842 - loss: 1.2855 - mean_io_u: 0.0573


806/Unknown  128s 109ms/step - categorical_accuracy: 0.6843 - loss: 1.2853 - mean_io_u: 0.0573


807/Unknown  128s 109ms/step - categorical_accuracy: 0.6843 - loss: 1.2850 - mean_io_u: 0.0574


808/Unknown  128s 109ms/step - categorical_accuracy: 0.6844 - loss: 1.2847 - mean_io_u: 0.0574


809/Unknown  129s 109ms/step - categorical_accuracy: 0.6844 - loss: 1.2844 - mean_io_u: 0.0575


810/Unknown  129s 109ms/step - categorical_accuracy: 0.6845 - loss: 1.2841 - mean_io_u: 0.0575


811/Unknown  129s 109ms/step - categorical_accuracy: 0.6845 - loss: 1.2838 - mean_io_u: 0.0575


812/Unknown  129s 109ms/step - categorical_accuracy: 0.6846 - loss: 1.2835 - mean_io_u: 0.0576


813/Unknown  129s 109ms/step - categorical_accuracy: 0.6847 - loss: 1.2832 - mean_io_u: 0.0576


814/Unknown  129s 109ms/step - categorical_accuracy: 0.6847 - loss: 1.2829 - mean_io_u: 0.0577


815/Unknown  129s 109ms/step - categorical_accuracy: 0.6848 - loss: 1.2826 - mean_io_u: 0.0577


816/Unknown  129s 109ms/step - categorical_accuracy: 0.6848 - loss: 1.2823 - mean_io_u: 0.0577


817/Unknown  129s 109ms/step - categorical_accuracy: 0.6849 - loss: 1.2821 - mean_io_u: 0.0578


818/Unknown  129s 109ms/step - categorical_accuracy: 0.6849 - loss: 1.2818 - mean_io_u: 0.0578


819/Unknown  129s 109ms/step - categorical_accuracy: 0.6850 - loss: 1.2815 - mean_io_u: 0.0578


820/Unknown  129s 108ms/step - categorical_accuracy: 0.6851 - loss: 1.2812 - mean_io_u: 0.0579


821/Unknown  129s 108ms/step - categorical_accuracy: 0.6851 - loss: 1.2809 - mean_io_u: 0.0579


822/Unknown  129s 108ms/step - categorical_accuracy: 0.6852 - loss: 1.2806 - mean_io_u: 0.0580


823/Unknown  130s 108ms/step - categorical_accuracy: 0.6852 - loss: 1.2803 - mean_io_u: 0.0580


824/Unknown  130s 108ms/step - categorical_accuracy: 0.6853 - loss: 1.2800 - mean_io_u: 0.0580


825/Unknown  130s 108ms/step - categorical_accuracy: 0.6853 - loss: 1.2798 - mean_io_u: 0.0581


826/Unknown  130s 108ms/step - categorical_accuracy: 0.6854 - loss: 1.2795 - mean_io_u: 0.0581


827/Unknown  130s 108ms/step - categorical_accuracy: 0.6855 - loss: 1.2792 - mean_io_u: 0.0582


828/Unknown  130s 108ms/step - categorical_accuracy: 0.6855 - loss: 1.2789 - mean_io_u: 0.0582


829/Unknown  130s 108ms/step - categorical_accuracy: 0.6856 - loss: 1.2786 - mean_io_u: 0.0582


830/Unknown  130s 108ms/step - categorical_accuracy: 0.6856 - loss: 1.2783 - mean_io_u: 0.0583


831/Unknown  130s 108ms/step - categorical_accuracy: 0.6857 - loss: 1.2780 - mean_io_u: 0.0583


832/Unknown  130s 108ms/step - categorical_accuracy: 0.6857 - loss: 1.2778 - mean_io_u: 0.0583


833/Unknown  130s 108ms/step - categorical_accuracy: 0.6858 - loss: 1.2775 - mean_io_u: 0.0584


834/Unknown  130s 108ms/step - categorical_accuracy: 0.6858 - loss: 1.2772 - mean_io_u: 0.0584


835/Unknown  130s 108ms/step - categorical_accuracy: 0.6859 - loss: 1.2769 - mean_io_u: 0.0585


836/Unknown  131s 108ms/step - categorical_accuracy: 0.6860 - loss: 1.2766 - mean_io_u: 0.0585


837/Unknown  131s 108ms/step - categorical_accuracy: 0.6860 - loss: 1.2764 - mean_io_u: 0.0585


838/Unknown  131s 108ms/step - categorical_accuracy: 0.6861 - loss: 1.2761 - mean_io_u: 0.0586


839/Unknown  131s 108ms/step - categorical_accuracy: 0.6861 - loss: 1.2758 - mean_io_u: 0.0586


840/Unknown  131s 108ms/step - categorical_accuracy: 0.6862 - loss: 1.2755 - mean_io_u: 0.0587


841/Unknown  131s 108ms/step - categorical_accuracy: 0.6862 - loss: 1.2752 - mean_io_u: 0.0587


842/Unknown  131s 108ms/step - categorical_accuracy: 0.6863 - loss: 1.2750 - mean_io_u: 0.0587


843/Unknown  131s 108ms/step - categorical_accuracy: 0.6863 - loss: 1.2747 - mean_io_u: 0.0588


844/Unknown  131s 108ms/step - categorical_accuracy: 0.6864 - loss: 1.2744 - mean_io_u: 0.0588


845/Unknown  131s 108ms/step - categorical_accuracy: 0.6865 - loss: 1.2741 - mean_io_u: 0.0588


846/Unknown  131s 108ms/step - categorical_accuracy: 0.6865 - loss: 1.2738 - mean_io_u: 0.0589


847/Unknown  131s 108ms/step - categorical_accuracy: 0.6866 - loss: 1.2736 - mean_io_u: 0.0589


848/Unknown  131s 107ms/step - categorical_accuracy: 0.6866 - loss: 1.2733 - mean_io_u: 0.0590


849/Unknown  132s 107ms/step - categorical_accuracy: 0.6867 - loss: 1.2730 - mean_io_u: 0.0590


850/Unknown  132s 107ms/step - categorical_accuracy: 0.6867 - loss: 1.2727 - mean_io_u: 0.0590


851/Unknown  132s 107ms/step - categorical_accuracy: 0.6868 - loss: 1.2724 - mean_io_u: 0.0591


852/Unknown  132s 107ms/step - categorical_accuracy: 0.6868 - loss: 1.2722 - mean_io_u: 0.0591


853/Unknown  132s 107ms/step - categorical_accuracy: 0.6869 - loss: 1.2719 - mean_io_u: 0.0592


854/Unknown  132s 107ms/step - categorical_accuracy: 0.6869 - loss: 1.2716 - mean_io_u: 0.0592


855/Unknown  132s 107ms/step - categorical_accuracy: 0.6870 - loss: 1.2713 - mean_io_u: 0.0592


856/Unknown  132s 107ms/step - categorical_accuracy: 0.6871 - loss: 1.2711 - mean_io_u: 0.0593


857/Unknown  132s 107ms/step - categorical_accuracy: 0.6871 - loss: 1.2708 - mean_io_u: 0.0593


858/Unknown  132s 107ms/step - categorical_accuracy: 0.6872 - loss: 1.2705 - mean_io_u: 0.0593


859/Unknown  133s 107ms/step - categorical_accuracy: 0.6872 - loss: 1.2702 - mean_io_u: 0.0594


860/Unknown  133s 107ms/step - categorical_accuracy: 0.6873 - loss: 1.2700 - mean_io_u: 0.0594


861/Unknown  133s 107ms/step - categorical_accuracy: 0.6873 - loss: 1.2697 - mean_io_u: 0.0595


862/Unknown  133s 107ms/step - categorical_accuracy: 0.6874 - loss: 1.2694 - mean_io_u: 0.0595


863/Unknown  133s 107ms/step - categorical_accuracy: 0.6874 - loss: 1.2692 - mean_io_u: 0.0595


864/Unknown  133s 107ms/step - categorical_accuracy: 0.6875 - loss: 1.2689 - mean_io_u: 0.0596


865/Unknown  133s 107ms/step - categorical_accuracy: 0.6875 - loss: 1.2686 - mean_io_u: 0.0596


866/Unknown  133s 107ms/step - categorical_accuracy: 0.6876 - loss: 1.2683 - mean_io_u: 0.0597


867/Unknown  133s 107ms/step - categorical_accuracy: 0.6876 - loss: 1.2681 - mean_io_u: 0.0597


868/Unknown  133s 107ms/step - categorical_accuracy: 0.6877 - loss: 1.2678 - mean_io_u: 0.0597


869/Unknown  133s 107ms/step - categorical_accuracy: 0.6878 - loss: 1.2675 - mean_io_u: 0.0598


870/Unknown  133s 107ms/step - categorical_accuracy: 0.6878 - loss: 1.2673 - mean_io_u: 0.0598


871/Unknown  134s 107ms/step - categorical_accuracy: 0.6879 - loss: 1.2670 - mean_io_u: 0.0598


872/Unknown  134s 107ms/step - categorical_accuracy: 0.6879 - loss: 1.2667 - mean_io_u: 0.0599


873/Unknown  134s 107ms/step - categorical_accuracy: 0.6880 - loss: 1.2665 - mean_io_u: 0.0599


874/Unknown  134s 107ms/step - categorical_accuracy: 0.6880 - loss: 1.2662 - mean_io_u: 0.0600


875/Unknown  134s 107ms/step - categorical_accuracy: 0.6881 - loss: 1.2659 - mean_io_u: 0.0600


876/Unknown  134s 107ms/step - categorical_accuracy: 0.6881 - loss: 1.2657 - mean_io_u: 0.0600


877/Unknown  134s 107ms/step - categorical_accuracy: 0.6882 - loss: 1.2654 - mean_io_u: 0.0601


878/Unknown  134s 107ms/step - categorical_accuracy: 0.6882 - loss: 1.2651 - mean_io_u: 0.0601


879/Unknown  134s 107ms/step - categorical_accuracy: 0.6883 - loss: 1.2649 - mean_io_u: 0.0601


880/Unknown  134s 107ms/step - categorical_accuracy: 0.6883 - loss: 1.2646 - mean_io_u: 0.0602


881/Unknown  134s 107ms/step - categorical_accuracy: 0.6884 - loss: 1.2643 - mean_io_u: 0.0602


882/Unknown  134s 107ms/step - categorical_accuracy: 0.6884 - loss: 1.2641 - mean_io_u: 0.0603


883/Unknown  135s 107ms/step - categorical_accuracy: 0.6885 - loss: 1.2638 - mean_io_u: 0.0603


884/Unknown  135s 107ms/step - categorical_accuracy: 0.6885 - loss: 1.2635 - mean_io_u: 0.0603


885/Unknown  135s 107ms/step - categorical_accuracy: 0.6886 - loss: 1.2633 - mean_io_u: 0.0604


886/Unknown  135s 107ms/step - categorical_accuracy: 0.6886 - loss: 1.2630 - mean_io_u: 0.0604


887/Unknown  135s 107ms/step - categorical_accuracy: 0.6887 - loss: 1.2627 - mean_io_u: 0.0604


888/Unknown  135s 107ms/step - categorical_accuracy: 0.6888 - loss: 1.2625 - mean_io_u: 0.0605


889/Unknown  135s 107ms/step - categorical_accuracy: 0.6888 - loss: 1.2622 - mean_io_u: 0.0605


890/Unknown  135s 106ms/step - categorical_accuracy: 0.6889 - loss: 1.2620 - mean_io_u: 0.0606


891/Unknown  135s 106ms/step - categorical_accuracy: 0.6889 - loss: 1.2617 - mean_io_u: 0.0606


892/Unknown  135s 106ms/step - categorical_accuracy: 0.6890 - loss: 1.2614 - mean_io_u: 0.0606


893/Unknown  135s 106ms/step - categorical_accuracy: 0.6890 - loss: 1.2612 - mean_io_u: 0.0607


894/Unknown  135s 106ms/step - categorical_accuracy: 0.6891 - loss: 1.2609 - mean_io_u: 0.0607


895/Unknown  136s 106ms/step - categorical_accuracy: 0.6891 - loss: 1.2607 - mean_io_u: 0.0607


896/Unknown  136s 106ms/step - categorical_accuracy: 0.6892 - loss: 1.2604 - mean_io_u: 0.0608


897/Unknown  136s 106ms/step - categorical_accuracy: 0.6892 - loss: 1.2601 - mean_io_u: 0.0608


898/Unknown  136s 106ms/step - categorical_accuracy: 0.6893 - loss: 1.2599 - mean_io_u: 0.0608


899/Unknown  136s 106ms/step - categorical_accuracy: 0.6893 - loss: 1.2596 - mean_io_u: 0.0609


900/Unknown  136s 106ms/step - categorical_accuracy: 0.6894 - loss: 1.2594 - mean_io_u: 0.0609


901/Unknown  136s 106ms/step - categorical_accuracy: 0.6894 - loss: 1.2591 - mean_io_u: 0.0610


902/Unknown  136s 106ms/step - categorical_accuracy: 0.6895 - loss: 1.2589 - mean_io_u: 0.0610


903/Unknown  136s 106ms/step - categorical_accuracy: 0.6895 - loss: 1.2586 - mean_io_u: 0.0610


904/Unknown  136s 106ms/step - categorical_accuracy: 0.6896 - loss: 1.2583 - mean_io_u: 0.0611


905/Unknown  136s 106ms/step - categorical_accuracy: 0.6896 - loss: 1.2581 - mean_io_u: 0.0611


906/Unknown  136s 106ms/step - categorical_accuracy: 0.6897 - loss: 1.2578 - mean_io_u: 0.0611


907/Unknown  136s 106ms/step - categorical_accuracy: 0.6897 - loss: 1.2576 - mean_io_u: 0.0612


908/Unknown  137s 106ms/step - categorical_accuracy: 0.6898 - loss: 1.2573 - mean_io_u: 0.0612


909/Unknown  137s 106ms/step - categorical_accuracy: 0.6898 - loss: 1.2571 - mean_io_u: 0.0613


910/Unknown  137s 106ms/step - categorical_accuracy: 0.6899 - loss: 1.2568 - mean_io_u: 0.0613


911/Unknown  137s 106ms/step - categorical_accuracy: 0.6899 - loss: 1.2566 - mean_io_u: 0.0613


912/Unknown  137s 106ms/step - categorical_accuracy: 0.6900 - loss: 1.2563 - mean_io_u: 0.0614


913/Unknown  137s 106ms/step - categorical_accuracy: 0.6900 - loss: 1.2561 - mean_io_u: 0.0614


914/Unknown  137s 106ms/step - categorical_accuracy: 0.6901 - loss: 1.2558 - mean_io_u: 0.0614


915/Unknown  137s 106ms/step - categorical_accuracy: 0.6901 - loss: 1.2556 - mean_io_u: 0.0615


916/Unknown  137s 106ms/step - categorical_accuracy: 0.6902 - loss: 1.2553 - mean_io_u: 0.0615


917/Unknown  137s 106ms/step - categorical_accuracy: 0.6902 - loss: 1.2551 - mean_io_u: 0.0615


918/Unknown  137s 106ms/step - categorical_accuracy: 0.6903 - loss: 1.2548 - mean_io_u: 0.0616


919/Unknown  137s 106ms/step - categorical_accuracy: 0.6903 - loss: 1.2546 - mean_io_u: 0.0616


920/Unknown  137s 106ms/step - categorical_accuracy: 0.6904 - loss: 1.2543 - mean_io_u: 0.0617


921/Unknown  138s 106ms/step - categorical_accuracy: 0.6904 - loss: 1.2541 - mean_io_u: 0.0617


922/Unknown  138s 105ms/step - categorical_accuracy: 0.6905 - loss: 1.2538 - mean_io_u: 0.0617


923/Unknown  138s 105ms/step - categorical_accuracy: 0.6905 - loss: 1.2536 - mean_io_u: 0.0618


924/Unknown  138s 105ms/step - categorical_accuracy: 0.6906 - loss: 1.2533 - mean_io_u: 0.0618


925/Unknown  138s 105ms/step - categorical_accuracy: 0.6906 - loss: 1.2531 - mean_io_u: 0.0618


926/Unknown  138s 105ms/step - categorical_accuracy: 0.6907 - loss: 1.2528 - mean_io_u: 0.0619


927/Unknown  138s 105ms/step - categorical_accuracy: 0.6907 - loss: 1.2526 - mean_io_u: 0.0619


928/Unknown  138s 105ms/step - categorical_accuracy: 0.6908 - loss: 1.2523 - mean_io_u: 0.0619


929/Unknown  138s 105ms/step - categorical_accuracy: 0.6908 - loss: 1.2521 - mean_io_u: 0.0620


930/Unknown  138s 105ms/step - categorical_accuracy: 0.6909 - loss: 1.2518 - mean_io_u: 0.0620


931/Unknown  138s 105ms/step - categorical_accuracy: 0.6909 - loss: 1.2516 - mean_io_u: 0.0621


932/Unknown  138s 105ms/step - categorical_accuracy: 0.6910 - loss: 1.2513 - mean_io_u: 0.0621


933/Unknown  138s 105ms/step - categorical_accuracy: 0.6910 - loss: 1.2511 - mean_io_u: 0.0621


934/Unknown  138s 105ms/step - categorical_accuracy: 0.6911 - loss: 1.2509 - mean_io_u: 0.0622


935/Unknown  139s 105ms/step - categorical_accuracy: 0.6911 - loss: 1.2506 - mean_io_u: 0.0622


936/Unknown  139s 105ms/step - categorical_accuracy: 0.6911 - loss: 1.2504 - mean_io_u: 0.0622


937/Unknown  139s 105ms/step - categorical_accuracy: 0.6912 - loss: 1.2501 - mean_io_u: 0.0623


938/Unknown  139s 105ms/step - categorical_accuracy: 0.6912 - loss: 1.2499 - mean_io_u: 0.0623


939/Unknown  139s 105ms/step - categorical_accuracy: 0.6913 - loss: 1.2496 - mean_io_u: 0.0624


940/Unknown  139s 105ms/step - categorical_accuracy: 0.6913 - loss: 1.2494 - mean_io_u: 0.0624


941/Unknown  139s 105ms/step - categorical_accuracy: 0.6914 - loss: 1.2491 - mean_io_u: 0.0624


942/Unknown  139s 105ms/step - categorical_accuracy: 0.6914 - loss: 1.2489 - mean_io_u: 0.0625


943/Unknown  139s 105ms/step - categorical_accuracy: 0.6915 - loss: 1.2487 - mean_io_u: 0.0625


944/Unknown  139s 105ms/step - categorical_accuracy: 0.6915 - loss: 1.2484 - mean_io_u: 0.0625


945/Unknown  139s 105ms/step - categorical_accuracy: 0.6916 - loss: 1.2482 - mean_io_u: 0.0626


946/Unknown  139s 105ms/step - categorical_accuracy: 0.6916 - loss: 1.2479 - mean_io_u: 0.0626


947/Unknown  139s 104ms/step - categorical_accuracy: 0.6917 - loss: 1.2477 - mean_io_u: 0.0626


948/Unknown  139s 104ms/step - categorical_accuracy: 0.6917 - loss: 1.2474 - mean_io_u: 0.0627


949/Unknown  139s 104ms/step - categorical_accuracy: 0.6918 - loss: 1.2472 - mean_io_u: 0.0627


950/Unknown  139s 104ms/step - categorical_accuracy: 0.6918 - loss: 1.2470 - mean_io_u: 0.0628


951/Unknown  140s 104ms/step - categorical_accuracy: 0.6919 - loss: 1.2467 - mean_io_u: 0.0628


952/Unknown  140s 104ms/step - categorical_accuracy: 0.6919 - loss: 1.2465 - mean_io_u: 0.0628


953/Unknown  140s 104ms/step - categorical_accuracy: 0.6920 - loss: 1.2462 - mean_io_u: 0.0629


954/Unknown  140s 104ms/step - categorical_accuracy: 0.6920 - loss: 1.2460 - mean_io_u: 0.0629


955/Unknown  140s 104ms/step - categorical_accuracy: 0.6921 - loss: 1.2458 - mean_io_u: 0.0629


956/Unknown  140s 104ms/step - categorical_accuracy: 0.6921 - loss: 1.2455 - mean_io_u: 0.0630


957/Unknown  140s 104ms/step - categorical_accuracy: 0.6921 - loss: 1.2453 - mean_io_u: 0.0630


958/Unknown  140s 104ms/step - categorical_accuracy: 0.6922 - loss: 1.2451 - mean_io_u: 0.0630


959/Unknown  140s 104ms/step - categorical_accuracy: 0.6922 - loss: 1.2448 - mean_io_u: 0.0631


960/Unknown  140s 104ms/step - categorical_accuracy: 0.6923 - loss: 1.2446 - mean_io_u: 0.0631


961/Unknown  140s 104ms/step - categorical_accuracy: 0.6923 - loss: 1.2443 - mean_io_u: 0.0631


962/Unknown  140s 104ms/step - categorical_accuracy: 0.6924 - loss: 1.2441 - mean_io_u: 0.0632


963/Unknown  140s 104ms/step - categorical_accuracy: 0.6924 - loss: 1.2439 - mean_io_u: 0.0632


964/Unknown  141s 104ms/step - categorical_accuracy: 0.6925 - loss: 1.2436 - mean_io_u: 0.0633


965/Unknown  141s 104ms/step - categorical_accuracy: 0.6925 - loss: 1.2434 - mean_io_u: 0.0633


966/Unknown  141s 104ms/step - categorical_accuracy: 0.6926 - loss: 1.2432 - mean_io_u: 0.0633


967/Unknown  141s 104ms/step - categorical_accuracy: 0.6926 - loss: 1.2429 - mean_io_u: 0.0634


968/Unknown  141s 104ms/step - categorical_accuracy: 0.6927 - loss: 1.2427 - mean_io_u: 0.0634


969/Unknown  141s 104ms/step - categorical_accuracy: 0.6927 - loss: 1.2425 - mean_io_u: 0.0634


970/Unknown  141s 104ms/step - categorical_accuracy: 0.6928 - loss: 1.2422 - mean_io_u: 0.0635


971/Unknown  141s 104ms/step - categorical_accuracy: 0.6928 - loss: 1.2420 - mean_io_u: 0.0635


972/Unknown  141s 104ms/step - categorical_accuracy: 0.6929 - loss: 1.2417 - mean_io_u: 0.0635


973/Unknown  141s 104ms/step - categorical_accuracy: 0.6929 - loss: 1.2415 - mean_io_u: 0.0636


974/Unknown  141s 104ms/step - categorical_accuracy: 0.6929 - loss: 1.2413 - mean_io_u: 0.0636


975/Unknown  141s 104ms/step - categorical_accuracy: 0.6930 - loss: 1.2410 - mean_io_u: 0.0636


976/Unknown  141s 104ms/step - categorical_accuracy: 0.6930 - loss: 1.2408 - mean_io_u: 0.0637


977/Unknown  142s 104ms/step - categorical_accuracy: 0.6931 - loss: 1.2406 - mean_io_u: 0.0637


978/Unknown  142s 104ms/step - categorical_accuracy: 0.6931 - loss: 1.2403 - mean_io_u: 0.0637


979/Unknown  142s 104ms/step - categorical_accuracy: 0.6932 - loss: 1.2401 - mean_io_u: 0.0638


980/Unknown  142s 103ms/step - categorical_accuracy: 0.6932 - loss: 1.2399 - mean_io_u: 0.0638


981/Unknown  142s 103ms/step - categorical_accuracy: 0.6933 - loss: 1.2397 - mean_io_u: 0.0639


982/Unknown  142s 103ms/step - categorical_accuracy: 0.6933 - loss: 1.2394 - mean_io_u: 0.0639


983/Unknown  142s 103ms/step - categorical_accuracy: 0.6934 - loss: 1.2392 - mean_io_u: 0.0639


984/Unknown  142s 103ms/step - categorical_accuracy: 0.6934 - loss: 1.2390 - mean_io_u: 0.0640


985/Unknown  142s 103ms/step - categorical_accuracy: 0.6935 - loss: 1.2387 - mean_io_u: 0.0640


986/Unknown  142s 103ms/step - categorical_accuracy: 0.6935 - loss: 1.2385 - mean_io_u: 0.0640


987/Unknown  142s 103ms/step - categorical_accuracy: 0.6935 - loss: 1.2383 - mean_io_u: 0.0641


988/Unknown  142s 103ms/step - categorical_accuracy: 0.6936 - loss: 1.2381 - mean_io_u: 0.0641


989/Unknown  143s 103ms/step - categorical_accuracy: 0.6936 - loss: 1.2378 - mean_io_u: 0.0641


990/Unknown  143s 103ms/step - categorical_accuracy: 0.6937 - loss: 1.2376 - mean_io_u: 0.0642


991/Unknown  143s 103ms/step - categorical_accuracy: 0.6937 - loss: 1.2374 - mean_io_u: 0.0642


992/Unknown  143s 103ms/step - categorical_accuracy: 0.6938 - loss: 1.2371 - mean_io_u: 0.0642


993/Unknown  143s 103ms/step - categorical_accuracy: 0.6938 - loss: 1.2369 - mean_io_u: 0.0643


994/Unknown  143s 103ms/step - categorical_accuracy: 0.6939 - loss: 1.2367 - mean_io_u: 0.0643


995/Unknown  143s 103ms/step - categorical_accuracy: 0.6939 - loss: 1.2365 - mean_io_u: 0.0643


996/Unknown  143s 103ms/step - categorical_accuracy: 0.6940 - loss: 1.2362 - mean_io_u: 0.0644


997/Unknown  143s 103ms/step - categorical_accuracy: 0.6940 - loss: 1.2360 - mean_io_u: 0.0644


998/Unknown  143s 103ms/step - categorical_accuracy: 0.6940 - loss: 1.2358 - mean_io_u: 0.0644


999/Unknown  143s 103ms/step - categorical_accuracy: 0.6941 - loss: 1.2356 - mean_io_u: 0.0645


```
</div>
   1000/Unknown  143s 103ms/step - categorical_accuracy: 0.6941 - loss: 1.2353 - mean_io_u: 0.0645

<div class="k-default-codeblock">
```

```
</div>
   1001/Unknown  143s 103ms/step - categorical_accuracy: 0.6942 - loss: 1.2351 - mean_io_u: 0.0645

<div class="k-default-codeblock">
```

```
</div>
   1002/Unknown  143s 103ms/step - categorical_accuracy: 0.6942 - loss: 1.2349 - mean_io_u: 0.0646

<div class="k-default-codeblock">
```

```
</div>
   1003/Unknown  144s 103ms/step - categorical_accuracy: 0.6943 - loss: 1.2346 - mean_io_u: 0.0646

<div class="k-default-codeblock">
```

```
</div>
   1004/Unknown  144s 103ms/step - categorical_accuracy: 0.6943 - loss: 1.2344 - mean_io_u: 0.0646

<div class="k-default-codeblock">
```

```
</div>
   1005/Unknown  144s 103ms/step - categorical_accuracy: 0.6944 - loss: 1.2342 - mean_io_u: 0.0647

<div class="k-default-codeblock">
```

```
</div>
   1006/Unknown  144s 103ms/step - categorical_accuracy: 0.6944 - loss: 1.2340 - mean_io_u: 0.0647

<div class="k-default-codeblock">
```

```
</div>
   1007/Unknown  144s 103ms/step - categorical_accuracy: 0.6944 - loss: 1.2337 - mean_io_u: 0.0647

<div class="k-default-codeblock">
```

```
</div>
   1008/Unknown  144s 103ms/step - categorical_accuracy: 0.6945 - loss: 1.2335 - mean_io_u: 0.0648

<div class="k-default-codeblock">
```

```
</div>
   1009/Unknown  144s 103ms/step - categorical_accuracy: 0.6945 - loss: 1.2333 - mean_io_u: 0.0648

<div class="k-default-codeblock">
```

```
</div>
   1010/Unknown  144s 103ms/step - categorical_accuracy: 0.6946 - loss: 1.2331 - mean_io_u: 0.0648

<div class="k-default-codeblock">
```

```
</div>
   1011/Unknown  144s 103ms/step - categorical_accuracy: 0.6946 - loss: 1.2329 - mean_io_u: 0.0649

<div class="k-default-codeblock">
```

```
</div>
   1012/Unknown  144s 103ms/step - categorical_accuracy: 0.6947 - loss: 1.2326 - mean_io_u: 0.0649

<div class="k-default-codeblock">
```

```
</div>
   1013/Unknown  144s 103ms/step - categorical_accuracy: 0.6947 - loss: 1.2324 - mean_io_u: 0.0649

<div class="k-default-codeblock">
```

```
</div>
   1014/Unknown  144s 103ms/step - categorical_accuracy: 0.6948 - loss: 1.2322 - mean_io_u: 0.0650

<div class="k-default-codeblock">
```

```
</div>
   1015/Unknown  144s 103ms/step - categorical_accuracy: 0.6948 - loss: 1.2320 - mean_io_u: 0.0650

<div class="k-default-codeblock">
```

```
</div>
   1016/Unknown  145s 103ms/step - categorical_accuracy: 0.6948 - loss: 1.2317 - mean_io_u: 0.0650

<div class="k-default-codeblock">
```

```
</div>
   1017/Unknown  145s 103ms/step - categorical_accuracy: 0.6949 - loss: 1.2315 - mean_io_u: 0.0651

<div class="k-default-codeblock">
```

```
</div>
   1018/Unknown  145s 102ms/step - categorical_accuracy: 0.6949 - loss: 1.2313 - mean_io_u: 0.0651

<div class="k-default-codeblock">
```

```
</div>
   1019/Unknown  145s 102ms/step - categorical_accuracy: 0.6950 - loss: 1.2311 - mean_io_u: 0.0651

<div class="k-default-codeblock">
```

```
</div>
   1020/Unknown  145s 103ms/step - categorical_accuracy: 0.6950 - loss: 1.2309 - mean_io_u: 0.0652

<div class="k-default-codeblock">
```

```
</div>
   1021/Unknown  145s 103ms/step - categorical_accuracy: 0.6951 - loss: 1.2306 - mean_io_u: 0.0652

<div class="k-default-codeblock">
```

```
</div>
   1022/Unknown  145s 102ms/step - categorical_accuracy: 0.6951 - loss: 1.2304 - mean_io_u: 0.0652

<div class="k-default-codeblock">
```

```
</div>
   1023/Unknown  145s 102ms/step - categorical_accuracy: 0.6952 - loss: 1.2302 - mean_io_u: 0.0653

<div class="k-default-codeblock">
```

```
</div>
   1024/Unknown  145s 102ms/step - categorical_accuracy: 0.6952 - loss: 1.2300 - mean_io_u: 0.0653

<div class="k-default-codeblock">
```

```
</div>
   1025/Unknown  145s 102ms/step - categorical_accuracy: 0.6952 - loss: 1.2298 - mean_io_u: 0.0653

<div class="k-default-codeblock">
```

```
</div>
   1026/Unknown  145s 102ms/step - categorical_accuracy: 0.6953 - loss: 1.2295 - mean_io_u: 0.0654

<div class="k-default-codeblock">
```

```
</div>
   1027/Unknown  145s 102ms/step - categorical_accuracy: 0.6953 - loss: 1.2293 - mean_io_u: 0.0654

<div class="k-default-codeblock">
```

```
</div>
   1028/Unknown  146s 102ms/step - categorical_accuracy: 0.6954 - loss: 1.2291 - mean_io_u: 0.0654

<div class="k-default-codeblock">
```

```
</div>
   1029/Unknown  146s 102ms/step - categorical_accuracy: 0.6954 - loss: 1.2289 - mean_io_u: 0.0655

<div class="k-default-codeblock">
```

```
</div>
   1030/Unknown  146s 102ms/step - categorical_accuracy: 0.6955 - loss: 1.2287 - mean_io_u: 0.0655

<div class="k-default-codeblock">
```

```
</div>
   1031/Unknown  146s 102ms/step - categorical_accuracy: 0.6955 - loss: 1.2284 - mean_io_u: 0.0655

<div class="k-default-codeblock">
```

```
</div>
   1032/Unknown  146s 102ms/step - categorical_accuracy: 0.6955 - loss: 1.2282 - mean_io_u: 0.0656

<div class="k-default-codeblock">
```

```
</div>
   1033/Unknown  146s 102ms/step - categorical_accuracy: 0.6956 - loss: 1.2280 - mean_io_u: 0.0656

<div class="k-default-codeblock">
```

```
</div>
   1034/Unknown  146s 102ms/step - categorical_accuracy: 0.6956 - loss: 1.2278 - mean_io_u: 0.0656

<div class="k-default-codeblock">
```

```
</div>
   1035/Unknown  146s 102ms/step - categorical_accuracy: 0.6957 - loss: 1.2276 - mean_io_u: 0.0657

<div class="k-default-codeblock">
```

```
</div>
   1036/Unknown  146s 102ms/step - categorical_accuracy: 0.6957 - loss: 1.2274 - mean_io_u: 0.0657

<div class="k-default-codeblock">
```

```
</div>
   1037/Unknown  146s 102ms/step - categorical_accuracy: 0.6958 - loss: 1.2271 - mean_io_u: 0.0657

<div class="k-default-codeblock">
```

```
</div>
   1038/Unknown  146s 102ms/step - categorical_accuracy: 0.6958 - loss: 1.2269 - mean_io_u: 0.0658

<div class="k-default-codeblock">
```

```
</div>
   1039/Unknown  146s 102ms/step - categorical_accuracy: 0.6958 - loss: 1.2267 - mean_io_u: 0.0658

<div class="k-default-codeblock">
```

```
</div>
   1040/Unknown  146s 102ms/step - categorical_accuracy: 0.6959 - loss: 1.2265 - mean_io_u: 0.0658

<div class="k-default-codeblock">
```

```
</div>
   1041/Unknown  147s 102ms/step - categorical_accuracy: 0.6959 - loss: 1.2263 - mean_io_u: 0.0659

<div class="k-default-codeblock">
```

```
</div>
   1042/Unknown  147s 102ms/step - categorical_accuracy: 0.6960 - loss: 1.2261 - mean_io_u: 0.0659

<div class="k-default-codeblock">
```

```
</div>
   1043/Unknown  147s 102ms/step - categorical_accuracy: 0.6960 - loss: 1.2259 - mean_io_u: 0.0659

<div class="k-default-codeblock">
```

```
</div>
   1044/Unknown  147s 102ms/step - categorical_accuracy: 0.6961 - loss: 1.2256 - mean_io_u: 0.0660

<div class="k-default-codeblock">
```

```
</div>
   1045/Unknown  147s 102ms/step - categorical_accuracy: 0.6961 - loss: 1.2254 - mean_io_u: 0.0660

<div class="k-default-codeblock">
```

```
</div>
   1046/Unknown  147s 102ms/step - categorical_accuracy: 0.6961 - loss: 1.2252 - mean_io_u: 0.0660

<div class="k-default-codeblock">
```

```
</div>
   1047/Unknown  147s 102ms/step - categorical_accuracy: 0.6962 - loss: 1.2250 - mean_io_u: 0.0661

<div class="k-default-codeblock">
```

```
</div>
   1048/Unknown  147s 102ms/step - categorical_accuracy: 0.6962 - loss: 1.2248 - mean_io_u: 0.0661

<div class="k-default-codeblock">
```

```
</div>
   1049/Unknown  147s 102ms/step - categorical_accuracy: 0.6963 - loss: 1.2246 - mean_io_u: 0.0661

<div class="k-default-codeblock">
```

```
</div>
   1050/Unknown  147s 102ms/step - categorical_accuracy: 0.6963 - loss: 1.2244 - mean_io_u: 0.0662

<div class="k-default-codeblock">
```

```
</div>
   1051/Unknown  147s 102ms/step - categorical_accuracy: 0.6964 - loss: 1.2241 - mean_io_u: 0.0662

<div class="k-default-codeblock">
```

```
</div>
   1052/Unknown  147s 102ms/step - categorical_accuracy: 0.6964 - loss: 1.2239 - mean_io_u: 0.0662

<div class="k-default-codeblock">
```

```
</div>
   1053/Unknown  147s 102ms/step - categorical_accuracy: 0.6964 - loss: 1.2237 - mean_io_u: 0.0663

<div class="k-default-codeblock">
```

```
</div>
   1054/Unknown  148s 102ms/step - categorical_accuracy: 0.6965 - loss: 1.2235 - mean_io_u: 0.0663

<div class="k-default-codeblock">
```

```
</div>
   1055/Unknown  148s 102ms/step - categorical_accuracy: 0.6965 - loss: 1.2233 - mean_io_u: 0.0663

<div class="k-default-codeblock">
```

```
</div>
   1056/Unknown  148s 102ms/step - categorical_accuracy: 0.6966 - loss: 1.2231 - mean_io_u: 0.0664

<div class="k-default-codeblock">
```

```
</div>
   1057/Unknown  148s 102ms/step - categorical_accuracy: 0.6966 - loss: 1.2229 - mean_io_u: 0.0664

<div class="k-default-codeblock">
```

```
</div>
   1058/Unknown  148s 102ms/step - categorical_accuracy: 0.6967 - loss: 1.2226 - mean_io_u: 0.0664

<div class="k-default-codeblock">
```

```
</div>
   1059/Unknown  148s 102ms/step - categorical_accuracy: 0.6967 - loss: 1.2224 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1060/Unknown  148s 102ms/step - categorical_accuracy: 0.6967 - loss: 1.2222 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1061/Unknown  148s 102ms/step - categorical_accuracy: 0.6968 - loss: 1.2220 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1062/Unknown  148s 102ms/step - categorical_accuracy: 0.6968 - loss: 1.2218 - mean_io_u: 0.0666

<div class="k-default-codeblock">
```

```
</div>
   1063/Unknown  148s 101ms/step - categorical_accuracy: 0.6969 - loss: 1.2216 - mean_io_u: 0.0666

<div class="k-default-codeblock">
```

```
</div>
   1064/Unknown  148s 101ms/step - categorical_accuracy: 0.6969 - loss: 1.2214 - mean_io_u: 0.0666

<div class="k-default-codeblock">
```

```
</div>
   1065/Unknown  148s 101ms/step - categorical_accuracy: 0.6970 - loss: 1.2212 - mean_io_u: 0.0667

<div class="k-default-codeblock">
```

```
</div>
   1066/Unknown  148s 101ms/step - categorical_accuracy: 0.6970 - loss: 1.2210 - mean_io_u: 0.0667

<div class="k-default-codeblock">
```

```
</div>
   1067/Unknown  149s 101ms/step - categorical_accuracy: 0.6970 - loss: 1.2208 - mean_io_u: 0.0667

<div class="k-default-codeblock">
```

```
</div>
   1068/Unknown  149s 101ms/step - categorical_accuracy: 0.6971 - loss: 1.2205 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1069/Unknown  149s 101ms/step - categorical_accuracy: 0.6971 - loss: 1.2203 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1070/Unknown  149s 101ms/step - categorical_accuracy: 0.6972 - loss: 1.2201 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1071/Unknown  149s 101ms/step - categorical_accuracy: 0.6972 - loss: 1.2199 - mean_io_u: 0.0669

<div class="k-default-codeblock">
```

```
</div>
   1072/Unknown  149s 101ms/step - categorical_accuracy: 0.6972 - loss: 1.2197 - mean_io_u: 0.0669

<div class="k-default-codeblock">
```

```
</div>
   1073/Unknown  149s 101ms/step - categorical_accuracy: 0.6973 - loss: 1.2195 - mean_io_u: 0.0669

<div class="k-default-codeblock">
```

```
</div>
   1074/Unknown  149s 101ms/step - categorical_accuracy: 0.6973 - loss: 1.2193 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1075/Unknown  149s 101ms/step - categorical_accuracy: 0.6974 - loss: 1.2191 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1076/Unknown  149s 101ms/step - categorical_accuracy: 0.6974 - loss: 1.2189 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1077/Unknown  149s 101ms/step - categorical_accuracy: 0.6975 - loss: 1.2187 - mean_io_u: 0.0671

<div class="k-default-codeblock">
```

```
</div>
   1078/Unknown  149s 101ms/step - categorical_accuracy: 0.6975 - loss: 1.2185 - mean_io_u: 0.0671

<div class="k-default-codeblock">
```

```
</div>
   1079/Unknown  150s 101ms/step - categorical_accuracy: 0.6975 - loss: 1.2183 - mean_io_u: 0.0671

<div class="k-default-codeblock">
```

```
</div>
   1080/Unknown  150s 101ms/step - categorical_accuracy: 0.6976 - loss: 1.2180 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1081/Unknown  150s 101ms/step - categorical_accuracy: 0.6976 - loss: 1.2178 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1082/Unknown  150s 101ms/step - categorical_accuracy: 0.6977 - loss: 1.2176 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1083/Unknown  150s 101ms/step - categorical_accuracy: 0.6977 - loss: 1.2174 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1084/Unknown  150s 101ms/step - categorical_accuracy: 0.6977 - loss: 1.2172 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1085/Unknown  150s 101ms/step - categorical_accuracy: 0.6978 - loss: 1.2170 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1086/Unknown  150s 101ms/step - categorical_accuracy: 0.6978 - loss: 1.2168 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1087/Unknown  150s 101ms/step - categorical_accuracy: 0.6979 - loss: 1.2166 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1088/Unknown  150s 101ms/step - categorical_accuracy: 0.6979 - loss: 1.2164 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1089/Unknown  150s 101ms/step - categorical_accuracy: 0.6980 - loss: 1.2162 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1090/Unknown  150s 101ms/step - categorical_accuracy: 0.6980 - loss: 1.2160 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1091/Unknown  150s 101ms/step - categorical_accuracy: 0.6980 - loss: 1.2158 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1092/Unknown  150s 101ms/step - categorical_accuracy: 0.6981 - loss: 1.2156 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1093/Unknown  151s 101ms/step - categorical_accuracy: 0.6981 - loss: 1.2154 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1094/Unknown  151s 101ms/step - categorical_accuracy: 0.6982 - loss: 1.2152 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1095/Unknown  151s 101ms/step - categorical_accuracy: 0.6982 - loss: 1.2150 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1096/Unknown  151s 101ms/step - categorical_accuracy: 0.6982 - loss: 1.2148 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1097/Unknown  151s 101ms/step - categorical_accuracy: 0.6983 - loss: 1.2146 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1098/Unknown  151s 101ms/step - categorical_accuracy: 0.6983 - loss: 1.2144 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1099/Unknown  151s 101ms/step - categorical_accuracy: 0.6984 - loss: 1.2142 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1100/Unknown  151s 101ms/step - categorical_accuracy: 0.6984 - loss: 1.2140 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1101/Unknown  151s 101ms/step - categorical_accuracy: 0.6984 - loss: 1.2138 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1102/Unknown  151s 101ms/step - categorical_accuracy: 0.6985 - loss: 1.2135 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1103/Unknown  151s 101ms/step - categorical_accuracy: 0.6985 - loss: 1.2133 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1104/Unknown  151s 100ms/step - categorical_accuracy: 0.6986 - loss: 1.2131 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1105/Unknown  151s 100ms/step - categorical_accuracy: 0.6986 - loss: 1.2129 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1106/Unknown  151s 100ms/step - categorical_accuracy: 0.6986 - loss: 1.2127 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1107/Unknown  152s 100ms/step - categorical_accuracy: 0.6987 - loss: 1.2125 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1108/Unknown  152s 100ms/step - categorical_accuracy: 0.6987 - loss: 1.2123 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1109/Unknown  152s 100ms/step - categorical_accuracy: 0.6988 - loss: 1.2121 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1110/Unknown  152s 100ms/step - categorical_accuracy: 0.6988 - loss: 1.2119 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1111/Unknown  152s 100ms/step - categorical_accuracy: 0.6988 - loss: 1.2117 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1112/Unknown  152s 100ms/step - categorical_accuracy: 0.6989 - loss: 1.2115 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1113/Unknown  152s 100ms/step - categorical_accuracy: 0.6989 - loss: 1.2114 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1114/Unknown  152s 100ms/step - categorical_accuracy: 0.6990 - loss: 1.2112 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1115/Unknown  152s 100ms/step - categorical_accuracy: 0.6990 - loss: 1.2110 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1116/Unknown  152s 100ms/step - categorical_accuracy: 0.6990 - loss: 1.2108 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1117/Unknown  152s 100ms/step - categorical_accuracy: 0.6991 - loss: 1.2106 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1118/Unknown  152s 100ms/step - categorical_accuracy: 0.6991 - loss: 1.2104 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1119/Unknown  152s 100ms/step - categorical_accuracy: 0.6992 - loss: 1.2102 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1120/Unknown  152s 100ms/step - categorical_accuracy: 0.6992 - loss: 1.2100 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1121/Unknown  153s 100ms/step - categorical_accuracy: 0.6992 - loss: 1.2098 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1122/Unknown  153s 100ms/step - categorical_accuracy: 0.6993 - loss: 1.2096 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1123/Unknown  153s 100ms/step - categorical_accuracy: 0.6993 - loss: 1.2094 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1124/Unknown  153s 100ms/step - categorical_accuracy: 0.6994 - loss: 1.2092 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1125/Unknown  153s 100ms/step - categorical_accuracy: 0.6994 - loss: 1.2090 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1126/Unknown  153s 100ms/step - categorical_accuracy: 0.6994 - loss: 1.2088 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1127/Unknown  153s 100ms/step - categorical_accuracy: 0.6995 - loss: 1.2086 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1128/Unknown  153s 100ms/step - categorical_accuracy: 0.6995 - loss: 1.2084 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1129/Unknown  153s 100ms/step - categorical_accuracy: 0.6995 - loss: 1.2082 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1130/Unknown  153s 100ms/step - categorical_accuracy: 0.6996 - loss: 1.2080 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1131/Unknown  153s 100ms/step - categorical_accuracy: 0.6996 - loss: 1.2078 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1132/Unknown  153s 100ms/step - categorical_accuracy: 0.6997 - loss: 1.2076 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1133/Unknown  154s 100ms/step - categorical_accuracy: 0.6997 - loss: 1.2074 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1134/Unknown  154s 100ms/step - categorical_accuracy: 0.6997 - loss: 1.2072 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1135/Unknown  154s 100ms/step - categorical_accuracy: 0.6998 - loss: 1.2070 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1136/Unknown  154s 100ms/step - categorical_accuracy: 0.6998 - loss: 1.2068 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1137/Unknown  154s 100ms/step - categorical_accuracy: 0.6999 - loss: 1.2067 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1138/Unknown  154s 100ms/step - categorical_accuracy: 0.6999 - loss: 1.2065 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1139/Unknown  154s 100ms/step - categorical_accuracy: 0.6999 - loss: 1.2063 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1140/Unknown  154s 100ms/step - categorical_accuracy: 0.7000 - loss: 1.2061 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1141/Unknown  154s 100ms/step - categorical_accuracy: 0.7000 - loss: 1.2059 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1142/Unknown  154s 100ms/step - categorical_accuracy: 0.7000 - loss: 1.2057 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1143/Unknown  154s 100ms/step - categorical_accuracy: 0.7001 - loss: 1.2055 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1144/Unknown  154s 100ms/step - categorical_accuracy: 0.7001 - loss: 1.2053 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1145/Unknown  154s 100ms/step - categorical_accuracy: 0.7002 - loss: 1.2051 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1146/Unknown  155s 100ms/step - categorical_accuracy: 0.7002 - loss: 1.2049 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1147/Unknown  155s 100ms/step - categorical_accuracy: 0.7002 - loss: 1.2047 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1148/Unknown  155s 100ms/step - categorical_accuracy: 0.7003 - loss: 1.2045 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1149/Unknown  155s 100ms/step - categorical_accuracy: 0.7003 - loss: 1.2044 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1150/Unknown  155s 100ms/step - categorical_accuracy: 0.7003 - loss: 1.2042 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1151/Unknown  155s 100ms/step - categorical_accuracy: 0.7004 - loss: 1.2040 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1152/Unknown  155s 100ms/step - categorical_accuracy: 0.7004 - loss: 1.2038 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1153/Unknown  155s 100ms/step - categorical_accuracy: 0.7005 - loss: 1.2036 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1154/Unknown  155s 99ms/step - categorical_accuracy: 0.7005 - loss: 1.2034 - mean_io_u: 0.0695 

<div class="k-default-codeblock">
```

```
</div>
   1155/Unknown  155s 99ms/step - categorical_accuracy: 0.7005 - loss: 1.2032 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1156/Unknown  155s 99ms/step - categorical_accuracy: 0.7006 - loss: 1.2030 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1157/Unknown  155s 99ms/step - categorical_accuracy: 0.7006 - loss: 1.2028 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1158/Unknown  156s 99ms/step - categorical_accuracy: 0.7007 - loss: 1.2026 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1159/Unknown  156s 99ms/step - categorical_accuracy: 0.7007 - loss: 1.2024 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1160/Unknown  156s 99ms/step - categorical_accuracy: 0.7007 - loss: 1.2023 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1161/Unknown  156s 99ms/step - categorical_accuracy: 0.7008 - loss: 1.2021 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1162/Unknown  156s 99ms/step - categorical_accuracy: 0.7008 - loss: 1.2019 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1163/Unknown  156s 99ms/step - categorical_accuracy: 0.7008 - loss: 1.2017 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1164/Unknown  156s 99ms/step - categorical_accuracy: 0.7009 - loss: 1.2015 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1165/Unknown  156s 99ms/step - categorical_accuracy: 0.7009 - loss: 1.2013 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1166/Unknown  156s 99ms/step - categorical_accuracy: 0.7010 - loss: 1.2011 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1167/Unknown  156s 99ms/step - categorical_accuracy: 0.7010 - loss: 1.2009 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1168/Unknown  156s 99ms/step - categorical_accuracy: 0.7010 - loss: 1.2008 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1169/Unknown  156s 99ms/step - categorical_accuracy: 0.7011 - loss: 1.2006 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1170/Unknown  156s 99ms/step - categorical_accuracy: 0.7011 - loss: 1.2004 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1171/Unknown  157s 99ms/step - categorical_accuracy: 0.7011 - loss: 1.2002 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1172/Unknown  157s 99ms/step - categorical_accuracy: 0.7012 - loss: 1.2000 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1173/Unknown  157s 99ms/step - categorical_accuracy: 0.7012 - loss: 1.1998 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1174/Unknown  157s 99ms/step - categorical_accuracy: 0.7012 - loss: 1.1996 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1175/Unknown  157s 99ms/step - categorical_accuracy: 0.7013 - loss: 1.1995 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1176/Unknown  157s 99ms/step - categorical_accuracy: 0.7013 - loss: 1.1993 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1177/Unknown  157s 99ms/step - categorical_accuracy: 0.7014 - loss: 1.1991 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1178/Unknown  157s 99ms/step - categorical_accuracy: 0.7014 - loss: 1.1989 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1179/Unknown  157s 99ms/step - categorical_accuracy: 0.7014 - loss: 1.1987 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1180/Unknown  157s 99ms/step - categorical_accuracy: 0.7015 - loss: 1.1985 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1181/Unknown  157s 99ms/step - categorical_accuracy: 0.7015 - loss: 1.1983 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1182/Unknown  158s 99ms/step - categorical_accuracy: 0.7015 - loss: 1.1982 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1183/Unknown  158s 99ms/step - categorical_accuracy: 0.7016 - loss: 1.1980 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1184/Unknown  158s 99ms/step - categorical_accuracy: 0.7016 - loss: 1.1978 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1185/Unknown  158s 99ms/step - categorical_accuracy: 0.7017 - loss: 1.1976 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1186/Unknown  158s 99ms/step - categorical_accuracy: 0.7017 - loss: 1.1974 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1187/Unknown  158s 99ms/step - categorical_accuracy: 0.7017 - loss: 1.1972 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1188/Unknown  158s 99ms/step - categorical_accuracy: 0.7018 - loss: 1.1970 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1189/Unknown  158s 99ms/step - categorical_accuracy: 0.7018 - loss: 1.1969 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1190/Unknown  158s 99ms/step - categorical_accuracy: 0.7018 - loss: 1.1967 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1191/Unknown  158s 99ms/step - categorical_accuracy: 0.7019 - loss: 1.1965 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1192/Unknown  158s 99ms/step - categorical_accuracy: 0.7019 - loss: 1.1963 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1193/Unknown  158s 99ms/step - categorical_accuracy: 0.7019 - loss: 1.1961 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1194/Unknown  158s 99ms/step - categorical_accuracy: 0.7020 - loss: 1.1959 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1195/Unknown  159s 99ms/step - categorical_accuracy: 0.7020 - loss: 1.1958 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1196/Unknown  159s 99ms/step - categorical_accuracy: 0.7021 - loss: 1.1956 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1197/Unknown  159s 99ms/step - categorical_accuracy: 0.7021 - loss: 1.1954 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1198/Unknown  159s 99ms/step - categorical_accuracy: 0.7021 - loss: 1.1952 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1199/Unknown  159s 99ms/step - categorical_accuracy: 0.7022 - loss: 1.1950 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1200/Unknown  159s 99ms/step - categorical_accuracy: 0.7022 - loss: 1.1949 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1201/Unknown  159s 99ms/step - categorical_accuracy: 0.7022 - loss: 1.1947 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1202/Unknown  159s 99ms/step - categorical_accuracy: 0.7023 - loss: 1.1945 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1203/Unknown  159s 99ms/step - categorical_accuracy: 0.7023 - loss: 1.1943 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1204/Unknown  159s 99ms/step - categorical_accuracy: 0.7023 - loss: 1.1941 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1205/Unknown  159s 99ms/step - categorical_accuracy: 0.7024 - loss: 1.1940 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1206/Unknown  159s 99ms/step - categorical_accuracy: 0.7024 - loss: 1.1938 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1207/Unknown  159s 99ms/step - categorical_accuracy: 0.7024 - loss: 1.1936 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1208/Unknown  160s 99ms/step - categorical_accuracy: 0.7025 - loss: 1.1934 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1209/Unknown  160s 99ms/step - categorical_accuracy: 0.7025 - loss: 1.1932 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1210/Unknown  160s 99ms/step - categorical_accuracy: 0.7025 - loss: 1.1931 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1211/Unknown  160s 99ms/step - categorical_accuracy: 0.7026 - loss: 1.1929 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1212/Unknown  160s 99ms/step - categorical_accuracy: 0.7026 - loss: 1.1927 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1213/Unknown  160s 99ms/step - categorical_accuracy: 0.7027 - loss: 1.1925 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1214/Unknown  160s 99ms/step - categorical_accuracy: 0.7027 - loss: 1.1924 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1215/Unknown  160s 99ms/step - categorical_accuracy: 0.7027 - loss: 1.1922 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1216/Unknown  160s 99ms/step - categorical_accuracy: 0.7028 - loss: 1.1920 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1217/Unknown  160s 99ms/step - categorical_accuracy: 0.7028 - loss: 1.1918 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1218/Unknown  160s 99ms/step - categorical_accuracy: 0.7028 - loss: 1.1917 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1219/Unknown  160s 99ms/step - categorical_accuracy: 0.7029 - loss: 1.1915 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1220/Unknown  160s 98ms/step - categorical_accuracy: 0.7029 - loss: 1.1913 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1221/Unknown  161s 98ms/step - categorical_accuracy: 0.7029 - loss: 1.1911 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1222/Unknown  161s 98ms/step - categorical_accuracy: 0.7030 - loss: 1.1910 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1223/Unknown  161s 98ms/step - categorical_accuracy: 0.7030 - loss: 1.1908 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1224/Unknown  161s 98ms/step - categorical_accuracy: 0.7030 - loss: 1.1906 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1225/Unknown  161s 98ms/step - categorical_accuracy: 0.7031 - loss: 1.1904 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1226/Unknown  161s 98ms/step - categorical_accuracy: 0.7031 - loss: 1.1903 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1227/Unknown  161s 98ms/step - categorical_accuracy: 0.7031 - loss: 1.1901 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1228/Unknown  161s 98ms/step - categorical_accuracy: 0.7032 - loss: 1.1899 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1229/Unknown  161s 98ms/step - categorical_accuracy: 0.7032 - loss: 1.1897 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1230/Unknown  161s 98ms/step - categorical_accuracy: 0.7032 - loss: 1.1896 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1231/Unknown  161s 98ms/step - categorical_accuracy: 0.7033 - loss: 1.1894 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1232/Unknown  161s 98ms/step - categorical_accuracy: 0.7033 - loss: 1.1892 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1233/Unknown  162s 98ms/step - categorical_accuracy: 0.7033 - loss: 1.1891 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1234/Unknown  162s 98ms/step - categorical_accuracy: 0.7034 - loss: 1.1889 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1235/Unknown  162s 98ms/step - categorical_accuracy: 0.7034 - loss: 1.1887 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1236/Unknown  162s 98ms/step - categorical_accuracy: 0.7034 - loss: 1.1885 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1237/Unknown  162s 98ms/step - categorical_accuracy: 0.7035 - loss: 1.1884 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1238/Unknown  162s 98ms/step - categorical_accuracy: 0.7035 - loss: 1.1882 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1239/Unknown  162s 98ms/step - categorical_accuracy: 0.7035 - loss: 1.1880 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1240/Unknown  162s 98ms/step - categorical_accuracy: 0.7036 - loss: 1.1879 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1241/Unknown  162s 98ms/step - categorical_accuracy: 0.7036 - loss: 1.1877 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1242/Unknown  162s 98ms/step - categorical_accuracy: 0.7036 - loss: 1.1875 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1243/Unknown  162s 98ms/step - categorical_accuracy: 0.7037 - loss: 1.1873 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1244/Unknown  162s 98ms/step - categorical_accuracy: 0.7037 - loss: 1.1872 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1245/Unknown  162s 98ms/step - categorical_accuracy: 0.7037 - loss: 1.1870 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1246/Unknown  162s 98ms/step - categorical_accuracy: 0.7038 - loss: 1.1868 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1247/Unknown  162s 98ms/step - categorical_accuracy: 0.7038 - loss: 1.1867 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1248/Unknown  162s 98ms/step - categorical_accuracy: 0.7038 - loss: 1.1865 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1249/Unknown  163s 98ms/step - categorical_accuracy: 0.7039 - loss: 1.1863 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1250/Unknown  163s 98ms/step - categorical_accuracy: 0.7039 - loss: 1.1862 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1251/Unknown  163s 98ms/step - categorical_accuracy: 0.7039 - loss: 1.1860 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1252/Unknown  163s 98ms/step - categorical_accuracy: 0.7040 - loss: 1.1858 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1253/Unknown  163s 98ms/step - categorical_accuracy: 0.7040 - loss: 1.1856 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1254/Unknown  163s 98ms/step - categorical_accuracy: 0.7041 - loss: 1.1855 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1255/Unknown  163s 98ms/step - categorical_accuracy: 0.7041 - loss: 1.1853 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1256/Unknown  163s 98ms/step - categorical_accuracy: 0.7041 - loss: 1.1851 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1257/Unknown  163s 98ms/step - categorical_accuracy: 0.7041 - loss: 1.1850 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1258/Unknown  163s 98ms/step - categorical_accuracy: 0.7042 - loss: 1.1848 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1259/Unknown  163s 98ms/step - categorical_accuracy: 0.7042 - loss: 1.1846 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1260/Unknown  163s 98ms/step - categorical_accuracy: 0.7042 - loss: 1.1845 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1261/Unknown  163s 98ms/step - categorical_accuracy: 0.7043 - loss: 1.1843 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1262/Unknown  164s 98ms/step - categorical_accuracy: 0.7043 - loss: 1.1841 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1263/Unknown  164s 98ms/step - categorical_accuracy: 0.7043 - loss: 1.1840 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1264/Unknown  164s 98ms/step - categorical_accuracy: 0.7044 - loss: 1.1838 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1265/Unknown  164s 98ms/step - categorical_accuracy: 0.7044 - loss: 1.1836 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1266/Unknown  164s 98ms/step - categorical_accuracy: 0.7044 - loss: 1.1835 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1267/Unknown  164s 98ms/step - categorical_accuracy: 0.7045 - loss: 1.1833 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1268/Unknown  164s 97ms/step - categorical_accuracy: 0.7045 - loss: 1.1831 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1269/Unknown  164s 97ms/step - categorical_accuracy: 0.7045 - loss: 1.1830 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1270/Unknown  164s 97ms/step - categorical_accuracy: 0.7046 - loss: 1.1828 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1271/Unknown  164s 97ms/step - categorical_accuracy: 0.7046 - loss: 1.1826 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1272/Unknown  164s 97ms/step - categorical_accuracy: 0.7046 - loss: 1.1825 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1273/Unknown  164s 97ms/step - categorical_accuracy: 0.7047 - loss: 1.1823 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1274/Unknown  164s 97ms/step - categorical_accuracy: 0.7047 - loss: 1.1821 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1275/Unknown  164s 97ms/step - categorical_accuracy: 0.7047 - loss: 1.1820 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1276/Unknown  165s 97ms/step - categorical_accuracy: 0.7048 - loss: 1.1818 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1277/Unknown  165s 97ms/step - categorical_accuracy: 0.7048 - loss: 1.1816 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1278/Unknown  165s 97ms/step - categorical_accuracy: 0.7048 - loss: 1.1815 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1279/Unknown  165s 97ms/step - categorical_accuracy: 0.7049 - loss: 1.1813 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1280/Unknown  165s 97ms/step - categorical_accuracy: 0.7049 - loss: 1.1811 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1281/Unknown  165s 97ms/step - categorical_accuracy: 0.7049 - loss: 1.1810 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1282/Unknown  165s 97ms/step - categorical_accuracy: 0.7050 - loss: 1.1808 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1283/Unknown  165s 97ms/step - categorical_accuracy: 0.7050 - loss: 1.1807 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1284/Unknown  165s 97ms/step - categorical_accuracy: 0.7050 - loss: 1.1805 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1285/Unknown  165s 97ms/step - categorical_accuracy: 0.7051 - loss: 1.1803 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1286/Unknown  165s 97ms/step - categorical_accuracy: 0.7051 - loss: 1.1802 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1287/Unknown  165s 97ms/step - categorical_accuracy: 0.7051 - loss: 1.1800 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1288/Unknown  165s 97ms/step - categorical_accuracy: 0.7052 - loss: 1.1798 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1289/Unknown  166s 97ms/step - categorical_accuracy: 0.7052 - loss: 1.1797 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1290/Unknown  166s 97ms/step - categorical_accuracy: 0.7052 - loss: 1.1795 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1291/Unknown  166s 97ms/step - categorical_accuracy: 0.7053 - loss: 1.1794 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1292/Unknown  166s 97ms/step - categorical_accuracy: 0.7053 - loss: 1.1792 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1293/Unknown  166s 97ms/step - categorical_accuracy: 0.7053 - loss: 1.1790 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1294/Unknown  166s 97ms/step - categorical_accuracy: 0.7053 - loss: 1.1789 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1295/Unknown  166s 97ms/step - categorical_accuracy: 0.7054 - loss: 1.1787 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1296/Unknown  166s 97ms/step - categorical_accuracy: 0.7054 - loss: 1.1785 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1297/Unknown  166s 97ms/step - categorical_accuracy: 0.7054 - loss: 1.1784 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1298/Unknown  166s 97ms/step - categorical_accuracy: 0.7055 - loss: 1.1782 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1299/Unknown  166s 97ms/step - categorical_accuracy: 0.7055 - loss: 1.1781 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1300/Unknown  166s 97ms/step - categorical_accuracy: 0.7055 - loss: 1.1779 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1301/Unknown  166s 97ms/step - categorical_accuracy: 0.7056 - loss: 1.1777 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1302/Unknown  167s 97ms/step - categorical_accuracy: 0.7056 - loss: 1.1776 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1303/Unknown  167s 97ms/step - categorical_accuracy: 0.7056 - loss: 1.1774 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1304/Unknown  167s 97ms/step - categorical_accuracy: 0.7057 - loss: 1.1773 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1305/Unknown  167s 97ms/step - categorical_accuracy: 0.7057 - loss: 1.1771 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1306/Unknown  167s 97ms/step - categorical_accuracy: 0.7057 - loss: 1.1769 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1307/Unknown  167s 97ms/step - categorical_accuracy: 0.7058 - loss: 1.1768 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1308/Unknown  167s 97ms/step - categorical_accuracy: 0.7058 - loss: 1.1766 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1309/Unknown  167s 97ms/step - categorical_accuracy: 0.7058 - loss: 1.1765 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1310/Unknown  167s 97ms/step - categorical_accuracy: 0.7059 - loss: 1.1763 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1311/Unknown  167s 97ms/step - categorical_accuracy: 0.7059 - loss: 1.1761 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1312/Unknown  167s 97ms/step - categorical_accuracy: 0.7059 - loss: 1.1760 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1313/Unknown  168s 97ms/step - categorical_accuracy: 0.7059 - loss: 1.1758 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1314/Unknown  168s 97ms/step - categorical_accuracy: 0.7060 - loss: 1.1757 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1315/Unknown  168s 97ms/step - categorical_accuracy: 0.7060 - loss: 1.1755 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1316/Unknown  168s 97ms/step - categorical_accuracy: 0.7060 - loss: 1.1754 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1317/Unknown  168s 97ms/step - categorical_accuracy: 0.7061 - loss: 1.1752 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1318/Unknown  168s 97ms/step - categorical_accuracy: 0.7061 - loss: 1.1750 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1319/Unknown  168s 97ms/step - categorical_accuracy: 0.7061 - loss: 1.1749 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1320/Unknown  168s 97ms/step - categorical_accuracy: 0.7062 - loss: 1.1747 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1321/Unknown  168s 97ms/step - categorical_accuracy: 0.7062 - loss: 1.1746 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1322/Unknown  168s 97ms/step - categorical_accuracy: 0.7062 - loss: 1.1744 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1323/Unknown  168s 97ms/step - categorical_accuracy: 0.7063 - loss: 1.1743 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1324/Unknown  168s 97ms/step - categorical_accuracy: 0.7063 - loss: 1.1741 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1325/Unknown  169s 97ms/step - categorical_accuracy: 0.7063 - loss: 1.1739 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1326/Unknown  169s 97ms/step - categorical_accuracy: 0.7063 - loss: 1.1738 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1327/Unknown  169s 97ms/step - categorical_accuracy: 0.7064 - loss: 1.1736 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1328/Unknown  169s 97ms/step - categorical_accuracy: 0.7064 - loss: 1.1735 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1329/Unknown  169s 97ms/step - categorical_accuracy: 0.7064 - loss: 1.1733 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1330/Unknown  169s 97ms/step - categorical_accuracy: 0.7065 - loss: 1.1732 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1331/Unknown  169s 97ms/step - categorical_accuracy: 0.7065 - loss: 1.1730 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1332/Unknown  169s 97ms/step - categorical_accuracy: 0.7065 - loss: 1.1729 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1333/Unknown  169s 97ms/step - categorical_accuracy: 0.7066 - loss: 1.1727 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1334/Unknown  169s 97ms/step - categorical_accuracy: 0.7066 - loss: 1.1725 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1335/Unknown  169s 97ms/step - categorical_accuracy: 0.7066 - loss: 1.1724 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1336/Unknown  169s 97ms/step - categorical_accuracy: 0.7067 - loss: 1.1722 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1337/Unknown  169s 97ms/step - categorical_accuracy: 0.7067 - loss: 1.1721 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1338/Unknown  170s 97ms/step - categorical_accuracy: 0.7067 - loss: 1.1719 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1339/Unknown  170s 97ms/step - categorical_accuracy: 0.7067 - loss: 1.1718 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1340/Unknown  170s 97ms/step - categorical_accuracy: 0.7068 - loss: 1.1716 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1341/Unknown  170s 97ms/step - categorical_accuracy: 0.7068 - loss: 1.1715 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1342/Unknown  170s 97ms/step - categorical_accuracy: 0.7068 - loss: 1.1713 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1343/Unknown  170s 96ms/step - categorical_accuracy: 0.7069 - loss: 1.1712 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1344/Unknown  170s 96ms/step - categorical_accuracy: 0.7069 - loss: 1.1710 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1345/Unknown  170s 96ms/step - categorical_accuracy: 0.7069 - loss: 1.1709 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1346/Unknown  170s 96ms/step - categorical_accuracy: 0.7070 - loss: 1.1707 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1347/Unknown  170s 96ms/step - categorical_accuracy: 0.7070 - loss: 1.1706 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1348/Unknown  170s 96ms/step - categorical_accuracy: 0.7070 - loss: 1.1704 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1349/Unknown  170s 96ms/step - categorical_accuracy: 0.7070 - loss: 1.1703 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1350/Unknown  170s 96ms/step - categorical_accuracy: 0.7071 - loss: 1.1701 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1351/Unknown  171s 96ms/step - categorical_accuracy: 0.7071 - loss: 1.1699 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1352/Unknown  171s 96ms/step - categorical_accuracy: 0.7071 - loss: 1.1698 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1353/Unknown  171s 96ms/step - categorical_accuracy: 0.7072 - loss: 1.1696 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1354/Unknown  171s 96ms/step - categorical_accuracy: 0.7072 - loss: 1.1695 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1355/Unknown  171s 96ms/step - categorical_accuracy: 0.7072 - loss: 1.1693 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1356/Unknown  171s 96ms/step - categorical_accuracy: 0.7072 - loss: 1.1692 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1357/Unknown  171s 96ms/step - categorical_accuracy: 0.7073 - loss: 1.1690 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1358/Unknown  171s 96ms/step - categorical_accuracy: 0.7073 - loss: 1.1689 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1359/Unknown  171s 96ms/step - categorical_accuracy: 0.7073 - loss: 1.1687 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1360/Unknown  171s 96ms/step - categorical_accuracy: 0.7074 - loss: 1.1686 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1361/Unknown  171s 96ms/step - categorical_accuracy: 0.7074 - loss: 1.1684 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1362/Unknown  171s 96ms/step - categorical_accuracy: 0.7074 - loss: 1.1683 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1363/Unknown  171s 96ms/step - categorical_accuracy: 0.7075 - loss: 1.1681 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1364/Unknown  172s 96ms/step - categorical_accuracy: 0.7075 - loss: 1.1680 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1365/Unknown  172s 96ms/step - categorical_accuracy: 0.7075 - loss: 1.1678 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1366/Unknown  172s 96ms/step - categorical_accuracy: 0.7075 - loss: 1.1677 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1367/Unknown  172s 96ms/step - categorical_accuracy: 0.7076 - loss: 1.1676 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1368/Unknown  172s 96ms/step - categorical_accuracy: 0.7076 - loss: 1.1674 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1369/Unknown  172s 96ms/step - categorical_accuracy: 0.7076 - loss: 1.1673 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1370/Unknown  172s 96ms/step - categorical_accuracy: 0.7077 - loss: 1.1671 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1371/Unknown  172s 96ms/step - categorical_accuracy: 0.7077 - loss: 1.1670 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1372/Unknown  172s 96ms/step - categorical_accuracy: 0.7077 - loss: 1.1668 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1373/Unknown  172s 96ms/step - categorical_accuracy: 0.7077 - loss: 1.1667 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1374/Unknown  172s 96ms/step - categorical_accuracy: 0.7078 - loss: 1.1665 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1375/Unknown  172s 96ms/step - categorical_accuracy: 0.7078 - loss: 1.1664 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1376/Unknown  172s 96ms/step - categorical_accuracy: 0.7078 - loss: 1.1662 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1377/Unknown  173s 96ms/step - categorical_accuracy: 0.7079 - loss: 1.1661 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1378/Unknown  173s 96ms/step - categorical_accuracy: 0.7079 - loss: 1.1659 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1379/Unknown  173s 96ms/step - categorical_accuracy: 0.7079 - loss: 1.1658 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1380/Unknown  173s 96ms/step - categorical_accuracy: 0.7079 - loss: 1.1656 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1381/Unknown  173s 96ms/step - categorical_accuracy: 0.7080 - loss: 1.1655 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1382/Unknown  173s 96ms/step - categorical_accuracy: 0.7080 - loss: 1.1653 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1383/Unknown  173s 96ms/step - categorical_accuracy: 0.7080 - loss: 1.1652 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1384/Unknown  173s 96ms/step - categorical_accuracy: 0.7081 - loss: 1.1650 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1385/Unknown  173s 96ms/step - categorical_accuracy: 0.7081 - loss: 1.1649 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1386/Unknown  173s 96ms/step - categorical_accuracy: 0.7081 - loss: 1.1647 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1387/Unknown  173s 96ms/step - categorical_accuracy: 0.7081 - loss: 1.1646 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1388/Unknown  173s 96ms/step - categorical_accuracy: 0.7082 - loss: 1.1645 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1389/Unknown  173s 96ms/step - categorical_accuracy: 0.7082 - loss: 1.1643 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1390/Unknown  173s 96ms/step - categorical_accuracy: 0.7082 - loss: 1.1642 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1391/Unknown  173s 96ms/step - categorical_accuracy: 0.7083 - loss: 1.1640 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1392/Unknown  174s 96ms/step - categorical_accuracy: 0.7083 - loss: 1.1639 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1393/Unknown  174s 96ms/step - categorical_accuracy: 0.7083 - loss: 1.1637 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1394/Unknown  174s 96ms/step - categorical_accuracy: 0.7083 - loss: 1.1636 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1395/Unknown  174s 96ms/step - categorical_accuracy: 0.7084 - loss: 1.1634 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1396/Unknown  174s 96ms/step - categorical_accuracy: 0.7084 - loss: 1.1633 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1397/Unknown  174s 96ms/step - categorical_accuracy: 0.7084 - loss: 1.1631 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1398/Unknown  174s 96ms/step - categorical_accuracy: 0.7085 - loss: 1.1630 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1399/Unknown  174s 96ms/step - categorical_accuracy: 0.7085 - loss: 1.1629 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1400/Unknown  174s 96ms/step - categorical_accuracy: 0.7085 - loss: 1.1627 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1401/Unknown  174s 95ms/step - categorical_accuracy: 0.7085 - loss: 1.1626 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1402/Unknown  174s 95ms/step - categorical_accuracy: 0.7086 - loss: 1.1624 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1403/Unknown  174s 95ms/step - categorical_accuracy: 0.7086 - loss: 1.1623 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1404/Unknown  174s 95ms/step - categorical_accuracy: 0.7086 - loss: 1.1621 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1405/Unknown  174s 95ms/step - categorical_accuracy: 0.7087 - loss: 1.1620 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1406/Unknown  174s 95ms/step - categorical_accuracy: 0.7087 - loss: 1.1618 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1407/Unknown  175s 95ms/step - categorical_accuracy: 0.7087 - loss: 1.1617 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1408/Unknown  175s 95ms/step - categorical_accuracy: 0.7087 - loss: 1.1616 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1409/Unknown  175s 95ms/step - categorical_accuracy: 0.7088 - loss: 1.1614 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1410/Unknown  175s 95ms/step - categorical_accuracy: 0.7088 - loss: 1.1613 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1411/Unknown  175s 95ms/step - categorical_accuracy: 0.7088 - loss: 1.1611 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1412/Unknown  175s 95ms/step - categorical_accuracy: 0.7089 - loss: 1.1610 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1413/Unknown  175s 95ms/step - categorical_accuracy: 0.7089 - loss: 1.1608 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1414/Unknown  175s 95ms/step - categorical_accuracy: 0.7089 - loss: 1.1607 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1415/Unknown  175s 95ms/step - categorical_accuracy: 0.7089 - loss: 1.1605 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1416/Unknown  175s 95ms/step - categorical_accuracy: 0.7090 - loss: 1.1604 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1417/Unknown  175s 95ms/step - categorical_accuracy: 0.7090 - loss: 1.1603 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1418/Unknown  175s 95ms/step - categorical_accuracy: 0.7090 - loss: 1.1601 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1419/Unknown  175s 95ms/step - categorical_accuracy: 0.7091 - loss: 1.1600 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1420/Unknown  175s 95ms/step - categorical_accuracy: 0.7091 - loss: 1.1598 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1421/Unknown  176s 95ms/step - categorical_accuracy: 0.7091 - loss: 1.1597 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1422/Unknown  176s 95ms/step - categorical_accuracy: 0.7091 - loss: 1.1596 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1423/Unknown  176s 95ms/step - categorical_accuracy: 0.7092 - loss: 1.1594 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1424/Unknown  176s 95ms/step - categorical_accuracy: 0.7092 - loss: 1.1593 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1425/Unknown  176s 95ms/step - categorical_accuracy: 0.7092 - loss: 1.1591 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1426/Unknown  176s 95ms/step - categorical_accuracy: 0.7093 - loss: 1.1590 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1427/Unknown  176s 95ms/step - categorical_accuracy: 0.7093 - loss: 1.1588 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1428/Unknown  176s 95ms/step - categorical_accuracy: 0.7093 - loss: 1.1587 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1429/Unknown  176s 95ms/step - categorical_accuracy: 0.7093 - loss: 1.1586 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1430/Unknown  176s 95ms/step - categorical_accuracy: 0.7094 - loss: 1.1584 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1431/Unknown  176s 95ms/step - categorical_accuracy: 0.7094 - loss: 1.1583 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1432/Unknown  176s 95ms/step - categorical_accuracy: 0.7094 - loss: 1.1581 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1433/Unknown  176s 95ms/step - categorical_accuracy: 0.7094 - loss: 1.1580 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1434/Unknown  177s 95ms/step - categorical_accuracy: 0.7095 - loss: 1.1579 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1435/Unknown  177s 95ms/step - categorical_accuracy: 0.7095 - loss: 1.1577 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1436/Unknown  177s 95ms/step - categorical_accuracy: 0.7095 - loss: 1.1576 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1437/Unknown  177s 95ms/step - categorical_accuracy: 0.7096 - loss: 1.1574 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1438/Unknown  177s 95ms/step - categorical_accuracy: 0.7096 - loss: 1.1573 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1439/Unknown  177s 95ms/step - categorical_accuracy: 0.7096 - loss: 1.1572 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1440/Unknown  177s 95ms/step - categorical_accuracy: 0.7096 - loss: 1.1570 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1441/Unknown  177s 95ms/step - categorical_accuracy: 0.7097 - loss: 1.1569 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1442/Unknown  177s 95ms/step - categorical_accuracy: 0.7097 - loss: 1.1567 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1443/Unknown  177s 95ms/step - categorical_accuracy: 0.7097 - loss: 1.1566 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1444/Unknown  177s 95ms/step - categorical_accuracy: 0.7098 - loss: 1.1565 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1445/Unknown  177s 95ms/step - categorical_accuracy: 0.7098 - loss: 1.1563 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1446/Unknown  178s 95ms/step - categorical_accuracy: 0.7098 - loss: 1.1562 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1447/Unknown  178s 95ms/step - categorical_accuracy: 0.7098 - loss: 1.1560 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1448/Unknown  178s 95ms/step - categorical_accuracy: 0.7099 - loss: 1.1559 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1449/Unknown  178s 95ms/step - categorical_accuracy: 0.7099 - loss: 1.1558 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1450/Unknown  178s 95ms/step - categorical_accuracy: 0.7099 - loss: 1.1556 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1451/Unknown  178s 95ms/step - categorical_accuracy: 0.7099 - loss: 1.1555 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1452/Unknown  178s 95ms/step - categorical_accuracy: 0.7100 - loss: 1.1553 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1453/Unknown  178s 95ms/step - categorical_accuracy: 0.7100 - loss: 1.1552 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1454/Unknown  178s 95ms/step - categorical_accuracy: 0.7100 - loss: 1.1551 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1455/Unknown  178s 95ms/step - categorical_accuracy: 0.7101 - loss: 1.1549 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1456/Unknown  178s 95ms/step - categorical_accuracy: 0.7101 - loss: 1.1548 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1457/Unknown  178s 95ms/step - categorical_accuracy: 0.7101 - loss: 1.1546 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1458/Unknown  178s 95ms/step - categorical_accuracy: 0.7101 - loss: 1.1545 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1459/Unknown  179s 95ms/step - categorical_accuracy: 0.7102 - loss: 1.1544 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1460/Unknown  179s 95ms/step - categorical_accuracy: 0.7102 - loss: 1.1542 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1461/Unknown  179s 95ms/step - categorical_accuracy: 0.7102 - loss: 1.1541 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1462/Unknown  179s 95ms/step - categorical_accuracy: 0.7102 - loss: 1.1539 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1463/Unknown  179s 95ms/step - categorical_accuracy: 0.7103 - loss: 1.1538 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1464/Unknown  179s 95ms/step - categorical_accuracy: 0.7103 - loss: 1.1537 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1465/Unknown  179s 95ms/step - categorical_accuracy: 0.7103 - loss: 1.1535 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1466/Unknown  179s 95ms/step - categorical_accuracy: 0.7104 - loss: 1.1534 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1467/Unknown  179s 95ms/step - categorical_accuracy: 0.7104 - loss: 1.1533 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1468/Unknown  179s 95ms/step - categorical_accuracy: 0.7104 - loss: 1.1531 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1469/Unknown  179s 95ms/step - categorical_accuracy: 0.7104 - loss: 1.1530 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1470/Unknown  180s 95ms/step - categorical_accuracy: 0.7105 - loss: 1.1528 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1471/Unknown  180s 95ms/step - categorical_accuracy: 0.7105 - loss: 1.1527 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1472/Unknown  180s 95ms/step - categorical_accuracy: 0.7105 - loss: 1.1526 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1473/Unknown  180s 95ms/step - categorical_accuracy: 0.7105 - loss: 1.1524 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1474/Unknown  180s 95ms/step - categorical_accuracy: 0.7106 - loss: 1.1523 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1475/Unknown  180s 95ms/step - categorical_accuracy: 0.7106 - loss: 1.1522 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1476/Unknown  180s 95ms/step - categorical_accuracy: 0.7106 - loss: 1.1520 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1477/Unknown  180s 95ms/step - categorical_accuracy: 0.7107 - loss: 1.1519 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1478/Unknown  180s 95ms/step - categorical_accuracy: 0.7107 - loss: 1.1518 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1479/Unknown  180s 95ms/step - categorical_accuracy: 0.7107 - loss: 1.1516 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1480/Unknown  180s 95ms/step - categorical_accuracy: 0.7107 - loss: 1.1515 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1481/Unknown  180s 95ms/step - categorical_accuracy: 0.7108 - loss: 1.1513 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1482/Unknown  181s 95ms/step - categorical_accuracy: 0.7108 - loss: 1.1512 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1483/Unknown  181s 95ms/step - categorical_accuracy: 0.7108 - loss: 1.1511 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1484/Unknown  181s 95ms/step - categorical_accuracy: 0.7108 - loss: 1.1509 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1485/Unknown  181s 95ms/step - categorical_accuracy: 0.7109 - loss: 1.1508 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1486/Unknown  181s 95ms/step - categorical_accuracy: 0.7109 - loss: 1.1507 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1487/Unknown  181s 95ms/step - categorical_accuracy: 0.7109 - loss: 1.1505 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1488/Unknown  181s 95ms/step - categorical_accuracy: 0.7109 - loss: 1.1504 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1489/Unknown  181s 95ms/step - categorical_accuracy: 0.7110 - loss: 1.1503 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1490/Unknown  181s 95ms/step - categorical_accuracy: 0.7110 - loss: 1.1501 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1491/Unknown  181s 95ms/step - categorical_accuracy: 0.7110 - loss: 1.1500 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1492/Unknown  181s 95ms/step - categorical_accuracy: 0.7110 - loss: 1.1499 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1493/Unknown  181s 95ms/step - categorical_accuracy: 0.7111 - loss: 1.1497 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1494/Unknown  182s 95ms/step - categorical_accuracy: 0.7111 - loss: 1.1496 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1495/Unknown  182s 94ms/step - categorical_accuracy: 0.7111 - loss: 1.1495 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1496/Unknown  182s 94ms/step - categorical_accuracy: 0.7112 - loss: 1.1493 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1497/Unknown  182s 94ms/step - categorical_accuracy: 0.7112 - loss: 1.1492 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1498/Unknown  182s 94ms/step - categorical_accuracy: 0.7112 - loss: 1.1491 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1499/Unknown  182s 94ms/step - categorical_accuracy: 0.7112 - loss: 1.1489 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1500/Unknown  182s 94ms/step - categorical_accuracy: 0.7113 - loss: 1.1488 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1501/Unknown  182s 94ms/step - categorical_accuracy: 0.7113 - loss: 1.1487 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1502/Unknown  182s 94ms/step - categorical_accuracy: 0.7113 - loss: 1.1485 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1503/Unknown  182s 94ms/step - categorical_accuracy: 0.7113 - loss: 1.1484 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1504/Unknown  182s 94ms/step - categorical_accuracy: 0.7114 - loss: 1.1483 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1505/Unknown  182s 94ms/step - categorical_accuracy: 0.7114 - loss: 1.1481 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1506/Unknown  183s 94ms/step - categorical_accuracy: 0.7114 - loss: 1.1480 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1507/Unknown  183s 94ms/step - categorical_accuracy: 0.7114 - loss: 1.1479 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1508/Unknown  183s 94ms/step - categorical_accuracy: 0.7115 - loss: 1.1477 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1509/Unknown  183s 94ms/step - categorical_accuracy: 0.7115 - loss: 1.1476 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1510/Unknown  183s 94ms/step - categorical_accuracy: 0.7115 - loss: 1.1475 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1511/Unknown  183s 94ms/step - categorical_accuracy: 0.7115 - loss: 1.1473 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1512/Unknown  183s 94ms/step - categorical_accuracy: 0.7116 - loss: 1.1472 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1513/Unknown  183s 94ms/step - categorical_accuracy: 0.7116 - loss: 1.1471 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1514/Unknown  183s 94ms/step - categorical_accuracy: 0.7116 - loss: 1.1470 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1515/Unknown  183s 94ms/step - categorical_accuracy: 0.7116 - loss: 1.1468 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1516/Unknown  183s 94ms/step - categorical_accuracy: 0.7117 - loss: 1.1467 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1517/Unknown  183s 94ms/step - categorical_accuracy: 0.7117 - loss: 1.1466 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1518/Unknown  183s 94ms/step - categorical_accuracy: 0.7117 - loss: 1.1464 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1519/Unknown  184s 94ms/step - categorical_accuracy: 0.7117 - loss: 1.1463 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1520/Unknown  184s 94ms/step - categorical_accuracy: 0.7118 - loss: 1.1462 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1521/Unknown  184s 94ms/step - categorical_accuracy: 0.7118 - loss: 1.1460 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1522/Unknown  184s 94ms/step - categorical_accuracy: 0.7118 - loss: 1.1459 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1523/Unknown  184s 94ms/step - categorical_accuracy: 0.7119 - loss: 1.1458 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1524/Unknown  184s 94ms/step - categorical_accuracy: 0.7119 - loss: 1.1456 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1525/Unknown  184s 94ms/step - categorical_accuracy: 0.7119 - loss: 1.1455 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1526/Unknown  184s 94ms/step - categorical_accuracy: 0.7119 - loss: 1.1454 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1527/Unknown  184s 94ms/step - categorical_accuracy: 0.7120 - loss: 1.1453 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1528/Unknown  184s 94ms/step - categorical_accuracy: 0.7120 - loss: 1.1451 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1529/Unknown  184s 94ms/step - categorical_accuracy: 0.7120 - loss: 1.1450 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1530/Unknown  184s 94ms/step - categorical_accuracy: 0.7120 - loss: 1.1449 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1531/Unknown  184s 94ms/step - categorical_accuracy: 0.7121 - loss: 1.1447 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1532/Unknown  185s 94ms/step - categorical_accuracy: 0.7121 - loss: 1.1446 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1533/Unknown  185s 94ms/step - categorical_accuracy: 0.7121 - loss: 1.1445 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1534/Unknown  185s 94ms/step - categorical_accuracy: 0.7121 - loss: 1.1444 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1535/Unknown  185s 94ms/step - categorical_accuracy: 0.7122 - loss: 1.1442 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1536/Unknown  185s 94ms/step - categorical_accuracy: 0.7122 - loss: 1.1441 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1537/Unknown  185s 94ms/step - categorical_accuracy: 0.7122 - loss: 1.1440 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1538/Unknown  185s 94ms/step - categorical_accuracy: 0.7122 - loss: 1.1438 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1539/Unknown  185s 94ms/step - categorical_accuracy: 0.7123 - loss: 1.1437 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1540/Unknown  185s 94ms/step - categorical_accuracy: 0.7123 - loss: 1.1436 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1541/Unknown  185s 94ms/step - categorical_accuracy: 0.7123 - loss: 1.1435 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1542/Unknown  185s 94ms/step - categorical_accuracy: 0.7123 - loss: 1.1433 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1543/Unknown  185s 94ms/step - categorical_accuracy: 0.7124 - loss: 1.1432 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1544/Unknown  185s 94ms/step - categorical_accuracy: 0.7124 - loss: 1.1431 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1545/Unknown  185s 94ms/step - categorical_accuracy: 0.7124 - loss: 1.1429 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1546/Unknown  186s 94ms/step - categorical_accuracy: 0.7124 - loss: 1.1428 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1547/Unknown  186s 94ms/step - categorical_accuracy: 0.7125 - loss: 1.1427 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1548/Unknown  186s 94ms/step - categorical_accuracy: 0.7125 - loss: 1.1426 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1549/Unknown  186s 94ms/step - categorical_accuracy: 0.7125 - loss: 1.1424 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1550/Unknown  186s 94ms/step - categorical_accuracy: 0.7125 - loss: 1.1423 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1551/Unknown  186s 94ms/step - categorical_accuracy: 0.7126 - loss: 1.1422 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1552/Unknown  186s 94ms/step - categorical_accuracy: 0.7126 - loss: 1.1421 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1553/Unknown  186s 94ms/step - categorical_accuracy: 0.7126 - loss: 1.1419 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1554/Unknown  186s 94ms/step - categorical_accuracy: 0.7126 - loss: 1.1418 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1555/Unknown  186s 94ms/step - categorical_accuracy: 0.7127 - loss: 1.1417 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1556/Unknown  186s 94ms/step - categorical_accuracy: 0.7127 - loss: 1.1416 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1557/Unknown  186s 94ms/step - categorical_accuracy: 0.7127 - loss: 1.1414 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1558/Unknown  186s 94ms/step - categorical_accuracy: 0.7127 - loss: 1.1413 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1559/Unknown  186s 94ms/step - categorical_accuracy: 0.7128 - loss: 1.1412 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1560/Unknown  187s 94ms/step - categorical_accuracy: 0.7128 - loss: 1.1410 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1561/Unknown  187s 94ms/step - categorical_accuracy: 0.7128 - loss: 1.1409 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1562/Unknown  187s 94ms/step - categorical_accuracy: 0.7128 - loss: 1.1408 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1563/Unknown  187s 94ms/step - categorical_accuracy: 0.7129 - loss: 1.1407 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1564/Unknown  187s 94ms/step - categorical_accuracy: 0.7129 - loss: 1.1405 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1565/Unknown  187s 94ms/step - categorical_accuracy: 0.7129 - loss: 1.1404 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1566/Unknown  187s 94ms/step - categorical_accuracy: 0.7129 - loss: 1.1403 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1567/Unknown  187s 94ms/step - categorical_accuracy: 0.7130 - loss: 1.1402 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1568/Unknown  187s 94ms/step - categorical_accuracy: 0.7130 - loss: 1.1400 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1569/Unknown  187s 94ms/step - categorical_accuracy: 0.7130 - loss: 1.1399 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1570/Unknown  187s 94ms/step - categorical_accuracy: 0.7130 - loss: 1.1398 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1571/Unknown  187s 94ms/step - categorical_accuracy: 0.7130 - loss: 1.1397 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1572/Unknown  187s 94ms/step - categorical_accuracy: 0.7131 - loss: 1.1395 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1573/Unknown  187s 94ms/step - categorical_accuracy: 0.7131 - loss: 1.1394 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1574/Unknown  188s 94ms/step - categorical_accuracy: 0.7131 - loss: 1.1393 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1575/Unknown  188s 93ms/step - categorical_accuracy: 0.7131 - loss: 1.1392 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1576/Unknown  188s 93ms/step - categorical_accuracy: 0.7132 - loss: 1.1390 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1577/Unknown  188s 93ms/step - categorical_accuracy: 0.7132 - loss: 1.1389 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1578/Unknown  188s 93ms/step - categorical_accuracy: 0.7132 - loss: 1.1388 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1579/Unknown  188s 93ms/step - categorical_accuracy: 0.7132 - loss: 1.1387 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1580/Unknown  188s 93ms/step - categorical_accuracy: 0.7133 - loss: 1.1385 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1581/Unknown  188s 93ms/step - categorical_accuracy: 0.7133 - loss: 1.1384 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1582/Unknown  188s 93ms/step - categorical_accuracy: 0.7133 - loss: 1.1383 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1583/Unknown  188s 93ms/step - categorical_accuracy: 0.7133 - loss: 1.1382 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1584/Unknown  188s 93ms/step - categorical_accuracy: 0.7134 - loss: 1.1380 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1585/Unknown  188s 93ms/step - categorical_accuracy: 0.7134 - loss: 1.1379 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1586/Unknown  188s 93ms/step - categorical_accuracy: 0.7134 - loss: 1.1378 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1587/Unknown  189s 93ms/step - categorical_accuracy: 0.7134 - loss: 1.1377 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1588/Unknown  189s 93ms/step - categorical_accuracy: 0.7135 - loss: 1.1375 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1589/Unknown  189s 93ms/step - categorical_accuracy: 0.7135 - loss: 1.1374 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1590/Unknown  189s 93ms/step - categorical_accuracy: 0.7135 - loss: 1.1373 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1591/Unknown  189s 93ms/step - categorical_accuracy: 0.7135 - loss: 1.1372 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1592/Unknown  189s 93ms/step - categorical_accuracy: 0.7136 - loss: 1.1371 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1593/Unknown  189s 93ms/step - categorical_accuracy: 0.7136 - loss: 1.1369 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1594/Unknown  189s 93ms/step - categorical_accuracy: 0.7136 - loss: 1.1368 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1595/Unknown  189s 93ms/step - categorical_accuracy: 0.7136 - loss: 1.1367 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1596/Unknown  189s 93ms/step - categorical_accuracy: 0.7137 - loss: 1.1366 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1597/Unknown  189s 93ms/step - categorical_accuracy: 0.7137 - loss: 1.1364 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1598/Unknown  189s 93ms/step - categorical_accuracy: 0.7137 - loss: 1.1363 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1599/Unknown  190s 93ms/step - categorical_accuracy: 0.7137 - loss: 1.1362 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1600/Unknown  190s 93ms/step - categorical_accuracy: 0.7138 - loss: 1.1361 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1601/Unknown  190s 93ms/step - categorical_accuracy: 0.7138 - loss: 1.1359 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1602/Unknown  190s 93ms/step - categorical_accuracy: 0.7138 - loss: 1.1358 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1603/Unknown  190s 93ms/step - categorical_accuracy: 0.7138 - loss: 1.1357 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1604/Unknown  190s 93ms/step - categorical_accuracy: 0.7138 - loss: 1.1356 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1605/Unknown  190s 93ms/step - categorical_accuracy: 0.7139 - loss: 1.1355 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1606/Unknown  190s 93ms/step - categorical_accuracy: 0.7139 - loss: 1.1353 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1607/Unknown  190s 93ms/step - categorical_accuracy: 0.7139 - loss: 1.1352 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1608/Unknown  190s 93ms/step - categorical_accuracy: 0.7139 - loss: 1.1351 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1609/Unknown  190s 93ms/step - categorical_accuracy: 0.7140 - loss: 1.1350 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1610/Unknown  190s 93ms/step - categorical_accuracy: 0.7140 - loss: 1.1349 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1611/Unknown  190s 93ms/step - categorical_accuracy: 0.7140 - loss: 1.1347 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1612/Unknown  191s 93ms/step - categorical_accuracy: 0.7140 - loss: 1.1346 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1613/Unknown  191s 93ms/step - categorical_accuracy: 0.7141 - loss: 1.1345 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1614/Unknown  191s 93ms/step - categorical_accuracy: 0.7141 - loss: 1.1344 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1615/Unknown  191s 93ms/step - categorical_accuracy: 0.7141 - loss: 1.1342 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1616/Unknown  191s 93ms/step - categorical_accuracy: 0.7141 - loss: 1.1341 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1617/Unknown  191s 93ms/step - categorical_accuracy: 0.7142 - loss: 1.1340 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1618/Unknown  191s 93ms/step - categorical_accuracy: 0.7142 - loss: 1.1339 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1619/Unknown  191s 93ms/step - categorical_accuracy: 0.7142 - loss: 1.1338 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1620/Unknown  191s 93ms/step - categorical_accuracy: 0.7142 - loss: 1.1336 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1621/Unknown  191s 93ms/step - categorical_accuracy: 0.7142 - loss: 1.1335 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1622/Unknown  191s 93ms/step - categorical_accuracy: 0.7143 - loss: 1.1334 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1623/Unknown  191s 93ms/step - categorical_accuracy: 0.7143 - loss: 1.1333 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1624/Unknown  192s 93ms/step - categorical_accuracy: 0.7143 - loss: 1.1332 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1625/Unknown  192s 93ms/step - categorical_accuracy: 0.7143 - loss: 1.1331 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1626/Unknown  192s 93ms/step - categorical_accuracy: 0.7144 - loss: 1.1329 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1627/Unknown  192s 93ms/step - categorical_accuracy: 0.7144 - loss: 1.1328 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1628/Unknown  192s 93ms/step - categorical_accuracy: 0.7144 - loss: 1.1327 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1629/Unknown  192s 93ms/step - categorical_accuracy: 0.7144 - loss: 1.1326 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1630/Unknown  192s 93ms/step - categorical_accuracy: 0.7145 - loss: 1.1325 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1631/Unknown  192s 93ms/step - categorical_accuracy: 0.7145 - loss: 1.1323 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1632/Unknown  192s 93ms/step - categorical_accuracy: 0.7145 - loss: 1.1322 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1633/Unknown  192s 93ms/step - categorical_accuracy: 0.7145 - loss: 1.1321 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1634/Unknown  192s 93ms/step - categorical_accuracy: 0.7146 - loss: 1.1320 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1635/Unknown  192s 93ms/step - categorical_accuracy: 0.7146 - loss: 1.1319 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1636/Unknown  192s 93ms/step - categorical_accuracy: 0.7146 - loss: 1.1317 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1637/Unknown  193s 93ms/step - categorical_accuracy: 0.7146 - loss: 1.1316 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1638/Unknown  193s 93ms/step - categorical_accuracy: 0.7146 - loss: 1.1315 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1639/Unknown  193s 93ms/step - categorical_accuracy: 0.7147 - loss: 1.1314 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1640/Unknown  193s 93ms/step - categorical_accuracy: 0.7147 - loss: 1.1313 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1641/Unknown  193s 93ms/step - categorical_accuracy: 0.7147 - loss: 1.1312 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1642/Unknown  193s 93ms/step - categorical_accuracy: 0.7147 - loss: 1.1310 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1643/Unknown  193s 93ms/step - categorical_accuracy: 0.7148 - loss: 1.1309 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1644/Unknown  193s 93ms/step - categorical_accuracy: 0.7148 - loss: 1.1308 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1645/Unknown  193s 93ms/step - categorical_accuracy: 0.7148 - loss: 1.1307 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1646/Unknown  193s 93ms/step - categorical_accuracy: 0.7148 - loss: 1.1306 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1647/Unknown  193s 93ms/step - categorical_accuracy: 0.7148 - loss: 1.1305 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1648/Unknown  193s 93ms/step - categorical_accuracy: 0.7149 - loss: 1.1303 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1649/Unknown  193s 93ms/step - categorical_accuracy: 0.7149 - loss: 1.1302 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1650/Unknown  193s 93ms/step - categorical_accuracy: 0.7149 - loss: 1.1301 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1651/Unknown  194s 93ms/step - categorical_accuracy: 0.7149 - loss: 1.1300 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1652/Unknown  194s 93ms/step - categorical_accuracy: 0.7150 - loss: 1.1299 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1653/Unknown  194s 93ms/step - categorical_accuracy: 0.7150 - loss: 1.1298 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1654/Unknown  194s 93ms/step - categorical_accuracy: 0.7150 - loss: 1.1297 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1655/Unknown  194s 93ms/step - categorical_accuracy: 0.7150 - loss: 1.1295 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1656/Unknown  194s 93ms/step - categorical_accuracy: 0.7150 - loss: 1.1294 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1657/Unknown  194s 93ms/step - categorical_accuracy: 0.7151 - loss: 1.1293 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1658/Unknown  194s 93ms/step - categorical_accuracy: 0.7151 - loss: 1.1292 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1659/Unknown  194s 93ms/step - categorical_accuracy: 0.7151 - loss: 1.1291 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1660/Unknown  194s 93ms/step - categorical_accuracy: 0.7151 - loss: 1.1290 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1661/Unknown  194s 93ms/step - categorical_accuracy: 0.7152 - loss: 1.1289 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1662/Unknown  194s 93ms/step - categorical_accuracy: 0.7152 - loss: 1.1287 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1663/Unknown  194s 93ms/step - categorical_accuracy: 0.7152 - loss: 1.1286 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1664/Unknown  195s 93ms/step - categorical_accuracy: 0.7152 - loss: 1.1285 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1665/Unknown  195s 93ms/step - categorical_accuracy: 0.7153 - loss: 1.1284 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1666/Unknown  195s 93ms/step - categorical_accuracy: 0.7153 - loss: 1.1283 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1667/Unknown  195s 93ms/step - categorical_accuracy: 0.7153 - loss: 1.1282 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1668/Unknown  195s 93ms/step - categorical_accuracy: 0.7153 - loss: 1.1281 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1669/Unknown  195s 93ms/step - categorical_accuracy: 0.7153 - loss: 1.1279 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1670/Unknown  195s 93ms/step - categorical_accuracy: 0.7154 - loss: 1.1278 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1671/Unknown  195s 93ms/step - categorical_accuracy: 0.7154 - loss: 1.1277 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1672/Unknown  195s 93ms/step - categorical_accuracy: 0.7154 - loss: 1.1276 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1673/Unknown  195s 93ms/step - categorical_accuracy: 0.7154 - loss: 1.1275 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1674/Unknown  195s 93ms/step - categorical_accuracy: 0.7154 - loss: 1.1274 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1675/Unknown  195s 93ms/step - categorical_accuracy: 0.7155 - loss: 1.1273 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1676/Unknown  196s 93ms/step - categorical_accuracy: 0.7155 - loss: 1.1271 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1677/Unknown  196s 93ms/step - categorical_accuracy: 0.7155 - loss: 1.1270 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1678/Unknown  196s 93ms/step - categorical_accuracy: 0.7155 - loss: 1.1269 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1679/Unknown  196s 93ms/step - categorical_accuracy: 0.7156 - loss: 1.1268 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1680/Unknown  196s 93ms/step - categorical_accuracy: 0.7156 - loss: 1.1267 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1681/Unknown  196s 93ms/step - categorical_accuracy: 0.7156 - loss: 1.1266 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1682/Unknown  196s 93ms/step - categorical_accuracy: 0.7156 - loss: 1.1265 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1683/Unknown  196s 93ms/step - categorical_accuracy: 0.7156 - loss: 1.1264 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1684/Unknown  196s 93ms/step - categorical_accuracy: 0.7157 - loss: 1.1262 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1685/Unknown  196s 93ms/step - categorical_accuracy: 0.7157 - loss: 1.1261 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1686/Unknown  196s 93ms/step - categorical_accuracy: 0.7157 - loss: 1.1260 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1687/Unknown  196s 93ms/step - categorical_accuracy: 0.7157 - loss: 1.1259 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1688/Unknown  197s 93ms/step - categorical_accuracy: 0.7158 - loss: 1.1258 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1689/Unknown  197s 92ms/step - categorical_accuracy: 0.7158 - loss: 1.1257 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1690/Unknown  197s 92ms/step - categorical_accuracy: 0.7158 - loss: 1.1256 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1691/Unknown  197s 92ms/step - categorical_accuracy: 0.7158 - loss: 1.1255 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1692/Unknown  197s 92ms/step - categorical_accuracy: 0.7158 - loss: 1.1253 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1693/Unknown  197s 92ms/step - categorical_accuracy: 0.7159 - loss: 1.1252 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1694/Unknown  197s 92ms/step - categorical_accuracy: 0.7159 - loss: 1.1251 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1695/Unknown  197s 92ms/step - categorical_accuracy: 0.7159 - loss: 1.1250 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1696/Unknown  197s 92ms/step - categorical_accuracy: 0.7159 - loss: 1.1249 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1697/Unknown  197s 92ms/step - categorical_accuracy: 0.7160 - loss: 1.1248 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1698/Unknown  197s 92ms/step - categorical_accuracy: 0.7160 - loss: 1.1247 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1699/Unknown  197s 92ms/step - categorical_accuracy: 0.7160 - loss: 1.1246 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1700/Unknown  197s 92ms/step - categorical_accuracy: 0.7160 - loss: 1.1245 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1701/Unknown  197s 92ms/step - categorical_accuracy: 0.7160 - loss: 1.1243 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1702/Unknown  197s 92ms/step - categorical_accuracy: 0.7161 - loss: 1.1242 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1703/Unknown  197s 92ms/step - categorical_accuracy: 0.7161 - loss: 1.1241 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1704/Unknown  198s 92ms/step - categorical_accuracy: 0.7161 - loss: 1.1240 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1705/Unknown  198s 92ms/step - categorical_accuracy: 0.7161 - loss: 1.1239 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1706/Unknown  198s 92ms/step - categorical_accuracy: 0.7161 - loss: 1.1238 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1707/Unknown  198s 92ms/step - categorical_accuracy: 0.7162 - loss: 1.1237 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1708/Unknown  198s 92ms/step - categorical_accuracy: 0.7162 - loss: 1.1236 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1709/Unknown  198s 92ms/step - categorical_accuracy: 0.7162 - loss: 1.1235 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1710/Unknown  198s 92ms/step - categorical_accuracy: 0.7162 - loss: 1.1233 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1711/Unknown  198s 92ms/step - categorical_accuracy: 0.7163 - loss: 1.1232 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1712/Unknown  198s 92ms/step - categorical_accuracy: 0.7163 - loss: 1.1231 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1713/Unknown  198s 92ms/step - categorical_accuracy: 0.7163 - loss: 1.1230 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1714/Unknown  198s 92ms/step - categorical_accuracy: 0.7163 - loss: 1.1229 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1715/Unknown  198s 92ms/step - categorical_accuracy: 0.7163 - loss: 1.1228 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1716/Unknown  198s 92ms/step - categorical_accuracy: 0.7164 - loss: 1.1227 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1717/Unknown  198s 92ms/step - categorical_accuracy: 0.7164 - loss: 1.1226 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1718/Unknown  199s 92ms/step - categorical_accuracy: 0.7164 - loss: 1.1225 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1719/Unknown  199s 92ms/step - categorical_accuracy: 0.7164 - loss: 1.1224 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1720/Unknown  199s 92ms/step - categorical_accuracy: 0.7164 - loss: 1.1222 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1721/Unknown  199s 92ms/step - categorical_accuracy: 0.7165 - loss: 1.1221 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1722/Unknown  199s 92ms/step - categorical_accuracy: 0.7165 - loss: 1.1220 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1723/Unknown  199s 92ms/step - categorical_accuracy: 0.7165 - loss: 1.1219 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1724/Unknown  199s 92ms/step - categorical_accuracy: 0.7165 - loss: 1.1218 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1725/Unknown  199s 92ms/step - categorical_accuracy: 0.7166 - loss: 1.1217 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1726/Unknown  199s 92ms/step - categorical_accuracy: 0.7166 - loss: 1.1216 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1727/Unknown  199s 92ms/step - categorical_accuracy: 0.7166 - loss: 1.1215 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1728/Unknown  199s 92ms/step - categorical_accuracy: 0.7166 - loss: 1.1214 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1729/Unknown  199s 92ms/step - categorical_accuracy: 0.7166 - loss: 1.1213 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1730/Unknown  199s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1212 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1731/Unknown  200s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1210 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1732/Unknown  200s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1209 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1733/Unknown  200s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1208 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1734/Unknown  200s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1207 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1735/Unknown  200s 92ms/step - categorical_accuracy: 0.7168 - loss: 1.1206 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1736/Unknown  200s 92ms/step - categorical_accuracy: 0.7168 - loss: 1.1205 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1737/Unknown  200s 92ms/step - categorical_accuracy: 0.7168 - loss: 1.1204 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1738/Unknown  200s 92ms/step - categorical_accuracy: 0.7168 - loss: 1.1203 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1739/Unknown  200s 92ms/step - categorical_accuracy: 0.7168 - loss: 1.1202 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1740/Unknown  200s 92ms/step - categorical_accuracy: 0.7169 - loss: 1.1201 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1741/Unknown  200s 92ms/step - categorical_accuracy: 0.7169 - loss: 1.1200 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1742/Unknown  201s 92ms/step - categorical_accuracy: 0.7169 - loss: 1.1199 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1743/Unknown  201s 92ms/step - categorical_accuracy: 0.7169 - loss: 1.1198 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1744/Unknown  201s 92ms/step - categorical_accuracy: 0.7170 - loss: 1.1196 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1745/Unknown  201s 92ms/step - categorical_accuracy: 0.7170 - loss: 1.1195 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1746/Unknown  201s 92ms/step - categorical_accuracy: 0.7170 - loss: 1.1194 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1747/Unknown  201s 92ms/step - categorical_accuracy: 0.7170 - loss: 1.1193 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1748/Unknown  201s 92ms/step - categorical_accuracy: 0.7170 - loss: 1.1192 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1749/Unknown  201s 92ms/step - categorical_accuracy: 0.7171 - loss: 1.1191 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1750/Unknown  201s 92ms/step - categorical_accuracy: 0.7171 - loss: 1.1190 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1751/Unknown  201s 92ms/step - categorical_accuracy: 0.7171 - loss: 1.1189 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1752/Unknown  201s 92ms/step - categorical_accuracy: 0.7171 - loss: 1.1188 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1753/Unknown  201s 92ms/step - categorical_accuracy: 0.7171 - loss: 1.1187 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1754/Unknown  201s 92ms/step - categorical_accuracy: 0.7172 - loss: 1.1186 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1755/Unknown  201s 92ms/step - categorical_accuracy: 0.7172 - loss: 1.1185 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1756/Unknown  202s 92ms/step - categorical_accuracy: 0.7172 - loss: 1.1184 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1757/Unknown  202s 92ms/step - categorical_accuracy: 0.7172 - loss: 1.1183 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1758/Unknown  202s 92ms/step - categorical_accuracy: 0.7172 - loss: 1.1181 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1759/Unknown  202s 92ms/step - categorical_accuracy: 0.7173 - loss: 1.1180 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1760/Unknown  202s 92ms/step - categorical_accuracy: 0.7173 - loss: 1.1179 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1761/Unknown  202s 92ms/step - categorical_accuracy: 0.7173 - loss: 1.1178 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1762/Unknown  202s 92ms/step - categorical_accuracy: 0.7173 - loss: 1.1177 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1763/Unknown  202s 92ms/step - categorical_accuracy: 0.7173 - loss: 1.1176 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1764/Unknown  202s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1175 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1765/Unknown  202s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1174 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1766/Unknown  202s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1173 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1767/Unknown  202s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1172 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1768/Unknown  202s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1171 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1769/Unknown  203s 92ms/step - categorical_accuracy: 0.7175 - loss: 1.1170 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1770/Unknown  203s 92ms/step - categorical_accuracy: 0.7175 - loss: 1.1169 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1771/Unknown  203s 92ms/step - categorical_accuracy: 0.7175 - loss: 1.1168 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1772/Unknown  203s 92ms/step - categorical_accuracy: 0.7175 - loss: 1.1167 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1773/Unknown  203s 92ms/step - categorical_accuracy: 0.7176 - loss: 1.1166 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1774/Unknown  203s 92ms/step - categorical_accuracy: 0.7176 - loss: 1.1165 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1775/Unknown  203s 92ms/step - categorical_accuracy: 0.7176 - loss: 1.1164 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1776/Unknown  203s 92ms/step - categorical_accuracy: 0.7176 - loss: 1.1162 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1777/Unknown  203s 92ms/step - categorical_accuracy: 0.7176 - loss: 1.1161 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1778/Unknown  203s 92ms/step - categorical_accuracy: 0.7177 - loss: 1.1160 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1779/Unknown  203s 92ms/step - categorical_accuracy: 0.7177 - loss: 1.1159 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1780/Unknown  204s 92ms/step - categorical_accuracy: 0.7177 - loss: 1.1158 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1781/Unknown  204s 92ms/step - categorical_accuracy: 0.7177 - loss: 1.1157 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1782/Unknown  204s 92ms/step - categorical_accuracy: 0.7177 - loss: 1.1156 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1783/Unknown  204s 92ms/step - categorical_accuracy: 0.7178 - loss: 1.1155 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1784/Unknown  204s 92ms/step - categorical_accuracy: 0.7178 - loss: 1.1154 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1785/Unknown  204s 92ms/step - categorical_accuracy: 0.7178 - loss: 1.1153 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1786/Unknown  204s 92ms/step - categorical_accuracy: 0.7178 - loss: 1.1152 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1787/Unknown  204s 92ms/step - categorical_accuracy: 0.7178 - loss: 1.1151 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1788/Unknown  204s 92ms/step - categorical_accuracy: 0.7179 - loss: 1.1150 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1789/Unknown  204s 92ms/step - categorical_accuracy: 0.7179 - loss: 1.1149 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1790/Unknown  204s 92ms/step - categorical_accuracy: 0.7179 - loss: 1.1148 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1791/Unknown  204s 92ms/step - categorical_accuracy: 0.7179 - loss: 1.1147 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1792/Unknown  204s 92ms/step - categorical_accuracy: 0.7179 - loss: 1.1146 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1793/Unknown  205s 92ms/step - categorical_accuracy: 0.7180 - loss: 1.1145 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1794/Unknown  205s 92ms/step - categorical_accuracy: 0.7180 - loss: 1.1144 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1795/Unknown  205s 92ms/step - categorical_accuracy: 0.7180 - loss: 1.1143 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1796/Unknown  205s 92ms/step - categorical_accuracy: 0.7180 - loss: 1.1142 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1797/Unknown  205s 92ms/step - categorical_accuracy: 0.7180 - loss: 1.1141 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1798/Unknown  205s 92ms/step - categorical_accuracy: 0.7181 - loss: 1.1140 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1799/Unknown  205s 92ms/step - categorical_accuracy: 0.7181 - loss: 1.1139 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1800/Unknown  205s 92ms/step - categorical_accuracy: 0.7181 - loss: 1.1138 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1801/Unknown  205s 92ms/step - categorical_accuracy: 0.7181 - loss: 1.1136 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1802/Unknown  205s 92ms/step - categorical_accuracy: 0.7181 - loss: 1.1135 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1803/Unknown  205s 92ms/step - categorical_accuracy: 0.7182 - loss: 1.1134 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1804/Unknown  205s 92ms/step - categorical_accuracy: 0.7182 - loss: 1.1133 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1805/Unknown  206s 92ms/step - categorical_accuracy: 0.7182 - loss: 1.1132 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1806/Unknown  206s 92ms/step - categorical_accuracy: 0.7182 - loss: 1.1131 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1807/Unknown  206s 92ms/step - categorical_accuracy: 0.7182 - loss: 1.1130 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1808/Unknown  206s 92ms/step - categorical_accuracy: 0.7183 - loss: 1.1129 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1809/Unknown  206s 91ms/step - categorical_accuracy: 0.7183 - loss: 1.1128 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1810/Unknown  206s 91ms/step - categorical_accuracy: 0.7183 - loss: 1.1127 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1811/Unknown  206s 91ms/step - categorical_accuracy: 0.7183 - loss: 1.1126 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1812/Unknown  206s 91ms/step - categorical_accuracy: 0.7183 - loss: 1.1125 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1813/Unknown  206s 91ms/step - categorical_accuracy: 0.7184 - loss: 1.1124 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1814/Unknown  206s 91ms/step - categorical_accuracy: 0.7184 - loss: 1.1123 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1815/Unknown  206s 91ms/step - categorical_accuracy: 0.7184 - loss: 1.1122 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1816/Unknown  206s 91ms/step - categorical_accuracy: 0.7184 - loss: 1.1121 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1817/Unknown  207s 91ms/step - categorical_accuracy: 0.7184 - loss: 1.1120 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1818/Unknown  207s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1119 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1819/Unknown  207s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1118 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1820/Unknown  207s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1117 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1821/Unknown  207s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1116 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1822/Unknown  207s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1115 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1823/Unknown  207s 91ms/step - categorical_accuracy: 0.7186 - loss: 1.1114 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1824/Unknown  207s 91ms/step - categorical_accuracy: 0.7186 - loss: 1.1113 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1825/Unknown  207s 91ms/step - categorical_accuracy: 0.7186 - loss: 1.1112 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1826/Unknown  207s 91ms/step - categorical_accuracy: 0.7186 - loss: 1.1111 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1827/Unknown  207s 91ms/step - categorical_accuracy: 0.7186 - loss: 1.1110 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1828/Unknown  207s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1109 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1829/Unknown  207s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1108 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1830/Unknown  208s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1107 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1831/Unknown  208s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1106 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1832/Unknown  208s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1105 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1833/Unknown  208s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1104 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1834/Unknown  208s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1103 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1835/Unknown  208s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1102 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1836/Unknown  208s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1101 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1837/Unknown  208s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1100 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1838/Unknown  208s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1099 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1839/Unknown  208s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1098 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1840/Unknown  208s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1097 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1841/Unknown  208s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1096 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1842/Unknown  208s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1095 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1843/Unknown  208s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1094 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1844/Unknown  209s 91ms/step - categorical_accuracy: 0.7190 - loss: 1.1093 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1845/Unknown  209s 91ms/step - categorical_accuracy: 0.7190 - loss: 1.1092 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1846/Unknown  209s 91ms/step - categorical_accuracy: 0.7190 - loss: 1.1091 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1847/Unknown  209s 91ms/step - categorical_accuracy: 0.7190 - loss: 1.1090 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1848/Unknown  209s 91ms/step - categorical_accuracy: 0.7190 - loss: 1.1089 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1849/Unknown  209s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1088 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1850/Unknown  209s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1087 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1851/Unknown  209s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1086 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1852/Unknown  209s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1085 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1853/Unknown  209s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1084 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1854/Unknown  209s 91ms/step - categorical_accuracy: 0.7192 - loss: 1.1083 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1855/Unknown  209s 91ms/step - categorical_accuracy: 0.7192 - loss: 1.1082 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1856/Unknown  209s 91ms/step - categorical_accuracy: 0.7192 - loss: 1.1081 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1857/Unknown  209s 91ms/step - categorical_accuracy: 0.7192 - loss: 1.1080 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1858/Unknown  209s 91ms/step - categorical_accuracy: 0.7192 - loss: 1.1079 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1859/Unknown  210s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1078 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1860/Unknown  210s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1077 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1861/Unknown  210s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1076 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1862/Unknown  210s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1075 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1863/Unknown  210s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1074 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1864/Unknown  210s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1073 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1865/Unknown  210s 91ms/step - categorical_accuracy: 0.7194 - loss: 1.1072 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1866/Unknown  210s 91ms/step - categorical_accuracy: 0.7194 - loss: 1.1071 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1867/Unknown  210s 91ms/step - categorical_accuracy: 0.7194 - loss: 1.1071 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1868/Unknown  210s 91ms/step - categorical_accuracy: 0.7194 - loss: 1.1070 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1869/Unknown  210s 91ms/step - categorical_accuracy: 0.7194 - loss: 1.1069 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1870/Unknown  210s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1068 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1871/Unknown  210s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1067 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1872/Unknown  211s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1066 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1873/Unknown  211s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1065 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1874/Unknown  211s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1064 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1875/Unknown  211s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1063 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1876/Unknown  211s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1062 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1877/Unknown  211s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1061 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1878/Unknown  211s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1060 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1879/Unknown  211s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1059 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1880/Unknown  211s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1058 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1881/Unknown  211s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1057 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1882/Unknown  211s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1056 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1883/Unknown  211s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1055 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1884/Unknown  211s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1054 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1885/Unknown  212s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1053 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1886/Unknown  212s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1052 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1887/Unknown  212s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1051 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1888/Unknown  212s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1050 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1889/Unknown  212s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1049 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1890/Unknown  212s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1048 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1891/Unknown  212s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1047 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1892/Unknown  212s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1046 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1893/Unknown  212s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1045 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1894/Unknown  212s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1044 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1895/Unknown  212s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1044 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1896/Unknown  212s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1043 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1897/Unknown  212s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1042 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1898/Unknown  213s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1041 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1899/Unknown  213s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1040 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1900/Unknown  213s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1039 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1901/Unknown  213s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1038 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1902/Unknown  213s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1037 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1903/Unknown  213s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1036 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1904/Unknown  213s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1035 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1905/Unknown  213s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1034 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1906/Unknown  213s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1033 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1907/Unknown  213s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1032 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1908/Unknown  213s 91ms/step - categorical_accuracy: 0.7202 - loss: 1.1031 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1909/Unknown  213s 91ms/step - categorical_accuracy: 0.7202 - loss: 1.1030 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1910/Unknown  214s 91ms/step - categorical_accuracy: 0.7202 - loss: 1.1029 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1911/Unknown  214s 91ms/step - categorical_accuracy: 0.7202 - loss: 1.1028 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1912/Unknown  214s 91ms/step - categorical_accuracy: 0.7202 - loss: 1.1027 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1913/Unknown  214s 91ms/step - categorical_accuracy: 0.7203 - loss: 1.1026 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1914/Unknown  214s 91ms/step - categorical_accuracy: 0.7203 - loss: 1.1026 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1915/Unknown  214s 91ms/step - categorical_accuracy: 0.7203 - loss: 1.1025 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1916/Unknown  214s 91ms/step - categorical_accuracy: 0.7203 - loss: 1.1024 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1917/Unknown  214s 91ms/step - categorical_accuracy: 0.7203 - loss: 1.1023 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1918/Unknown  214s 91ms/step - categorical_accuracy: 0.7203 - loss: 1.1022 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1919/Unknown  214s 91ms/step - categorical_accuracy: 0.7204 - loss: 1.1021 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1920/Unknown  215s 91ms/step - categorical_accuracy: 0.7204 - loss: 1.1020 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1921/Unknown  215s 91ms/step - categorical_accuracy: 0.7204 - loss: 1.1019 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1922/Unknown  215s 91ms/step - categorical_accuracy: 0.7204 - loss: 1.1018 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1923/Unknown  215s 91ms/step - categorical_accuracy: 0.7204 - loss: 1.1017 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1924/Unknown  215s 91ms/step - categorical_accuracy: 0.7205 - loss: 1.1016 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1925/Unknown  215s 91ms/step - categorical_accuracy: 0.7205 - loss: 1.1015 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1926/Unknown  215s 91ms/step - categorical_accuracy: 0.7205 - loss: 1.1014 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1927/Unknown  215s 91ms/step - categorical_accuracy: 0.7205 - loss: 1.1013 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1928/Unknown  215s 91ms/step - categorical_accuracy: 0.7205 - loss: 1.1012 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1929/Unknown  215s 91ms/step - categorical_accuracy: 0.7205 - loss: 1.1011 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1930/Unknown  215s 91ms/step - categorical_accuracy: 0.7206 - loss: 1.1011 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1931/Unknown  215s 91ms/step - categorical_accuracy: 0.7206 - loss: 1.1010 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1932/Unknown  215s 91ms/step - categorical_accuracy: 0.7206 - loss: 1.1009 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1933/Unknown  216s 91ms/step - categorical_accuracy: 0.7206 - loss: 1.1008 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1934/Unknown  216s 91ms/step - categorical_accuracy: 0.7206 - loss: 1.1007 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1935/Unknown  216s 91ms/step - categorical_accuracy: 0.7207 - loss: 1.1006 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1936/Unknown  216s 91ms/step - categorical_accuracy: 0.7207 - loss: 1.1005 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1937/Unknown  216s 91ms/step - categorical_accuracy: 0.7207 - loss: 1.1004 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1938/Unknown  216s 91ms/step - categorical_accuracy: 0.7207 - loss: 1.1003 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1939/Unknown  216s 91ms/step - categorical_accuracy: 0.7207 - loss: 1.1002 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1940/Unknown  216s 91ms/step - categorical_accuracy: 0.7207 - loss: 1.1001 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1941/Unknown  216s 91ms/step - categorical_accuracy: 0.7208 - loss: 1.1000 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   1942/Unknown  216s 91ms/step - categorical_accuracy: 0.7208 - loss: 1.0999 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   1943/Unknown  216s 91ms/step - categorical_accuracy: 0.7208 - loss: 1.0999 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   1944/Unknown  216s 91ms/step - categorical_accuracy: 0.7208 - loss: 1.0998 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   1945/Unknown  216s 91ms/step - categorical_accuracy: 0.7208 - loss: 1.0997 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   1946/Unknown  217s 91ms/step - categorical_accuracy: 0.7209 - loss: 1.0996 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   1947/Unknown  217s 91ms/step - categorical_accuracy: 0.7209 - loss: 1.0995 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   1948/Unknown  217s 91ms/step - categorical_accuracy: 0.7209 - loss: 1.0994 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   1949/Unknown  217s 91ms/step - categorical_accuracy: 0.7209 - loss: 1.0993 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   1950/Unknown  217s 91ms/step - categorical_accuracy: 0.7209 - loss: 1.0992 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1951/Unknown  217s 90ms/step - categorical_accuracy: 0.7209 - loss: 1.0991 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1952/Unknown  217s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.0990 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1953/Unknown  217s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.0989 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1954/Unknown  217s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.0988 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1955/Unknown  217s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.0988 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   1956/Unknown  217s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.0987 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   1957/Unknown  217s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.0986 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   1958/Unknown  217s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.0985 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   1959/Unknown  217s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.0984 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   1960/Unknown  218s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.0983 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   1961/Unknown  218s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.0982 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   1962/Unknown  218s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.0981 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   1963/Unknown  218s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.0980 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   1964/Unknown  218s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.0979 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   1965/Unknown  218s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.0978 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   1966/Unknown  218s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.0977 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   1967/Unknown  218s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.0977 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   1968/Unknown  218s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.0976 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   1969/Unknown  218s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.0975 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   1970/Unknown  218s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.0974 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1971/Unknown  218s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.0973 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1972/Unknown  218s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.0972 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1973/Unknown  219s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.0971 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1974/Unknown  219s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.0970 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1975/Unknown  219s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.0969 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   1976/Unknown  219s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.0968 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   1977/Unknown  219s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.0968 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   1978/Unknown  219s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.0967 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   1979/Unknown  219s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.0966 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   1980/Unknown  219s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.0965 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   1981/Unknown  219s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.0964 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   1982/Unknown  219s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.0963 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   1983/Unknown  219s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.0962 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   1984/Unknown  219s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.0961 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   1985/Unknown  219s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.0960 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   1986/Unknown  220s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.0959 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   1987/Unknown  220s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.0959 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   1988/Unknown  220s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.0958 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   1989/Unknown  220s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.0957 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   1990/Unknown  220s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.0956 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   1991/Unknown  220s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.0955 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   1992/Unknown  220s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.0954 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   1993/Unknown  220s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.0953 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   1994/Unknown  220s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.0952 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   1995/Unknown  220s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.0951 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   1996/Unknown  220s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.0950 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   1997/Unknown  220s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.0950 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   1998/Unknown  220s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.0949 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   1999/Unknown  220s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.0948 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2000/Unknown  220s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.0947 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2001/Unknown  221s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.0946 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2002/Unknown  221s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.0945 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2003/Unknown  221s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.0944 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2004/Unknown  221s 90ms/step - categorical_accuracy: 0.7219 - loss: 1.0943 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2005/Unknown  221s 90ms/step - categorical_accuracy: 0.7219 - loss: 1.0942 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2006/Unknown  221s 90ms/step - categorical_accuracy: 0.7219 - loss: 1.0942 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2007/Unknown  221s 90ms/step - categorical_accuracy: 0.7219 - loss: 1.0941 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2008/Unknown  221s 90ms/step - categorical_accuracy: 0.7219 - loss: 1.0940 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2009/Unknown  221s 90ms/step - categorical_accuracy: 0.7220 - loss: 1.0939 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2010/Unknown  221s 90ms/step - categorical_accuracy: 0.7220 - loss: 1.0938 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2011/Unknown  221s 90ms/step - categorical_accuracy: 0.7220 - loss: 1.0937 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2012/Unknown  221s 90ms/step - categorical_accuracy: 0.7220 - loss: 1.0936 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2013/Unknown  221s 90ms/step - categorical_accuracy: 0.7220 - loss: 1.0935 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2014/Unknown  222s 90ms/step - categorical_accuracy: 0.7220 - loss: 1.0935 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2015/Unknown  222s 90ms/step - categorical_accuracy: 0.7221 - loss: 1.0934 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2016/Unknown  222s 90ms/step - categorical_accuracy: 0.7221 - loss: 1.0933 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2017/Unknown  222s 90ms/step - categorical_accuracy: 0.7221 - loss: 1.0932 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2018/Unknown  222s 90ms/step - categorical_accuracy: 0.7221 - loss: 1.0931 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2019/Unknown  222s 90ms/step - categorical_accuracy: 0.7221 - loss: 1.0930 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2020/Unknown  222s 90ms/step - categorical_accuracy: 0.7221 - loss: 1.0929 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2021/Unknown  222s 90ms/step - categorical_accuracy: 0.7222 - loss: 1.0928 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2022/Unknown  222s 90ms/step - categorical_accuracy: 0.7222 - loss: 1.0927 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2023/Unknown  222s 90ms/step - categorical_accuracy: 0.7222 - loss: 1.0927 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2024/Unknown  222s 90ms/step - categorical_accuracy: 0.7222 - loss: 1.0926 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2025/Unknown  222s 90ms/step - categorical_accuracy: 0.7222 - loss: 1.0925 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2026/Unknown  222s 90ms/step - categorical_accuracy: 0.7222 - loss: 1.0924 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2027/Unknown  223s 90ms/step - categorical_accuracy: 0.7223 - loss: 1.0923 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2028/Unknown  223s 90ms/step - categorical_accuracy: 0.7223 - loss: 1.0922 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2029/Unknown  223s 90ms/step - categorical_accuracy: 0.7223 - loss: 1.0921 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2030/Unknown  223s 90ms/step - categorical_accuracy: 0.7223 - loss: 1.0921 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2031/Unknown  223s 90ms/step - categorical_accuracy: 0.7223 - loss: 1.0920 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2032/Unknown  223s 90ms/step - categorical_accuracy: 0.7223 - loss: 1.0919 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2033/Unknown  223s 90ms/step - categorical_accuracy: 0.7224 - loss: 1.0918 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2034/Unknown  223s 90ms/step - categorical_accuracy: 0.7224 - loss: 1.0917 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2035/Unknown  223s 90ms/step - categorical_accuracy: 0.7224 - loss: 1.0916 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2036/Unknown  223s 90ms/step - categorical_accuracy: 0.7224 - loss: 1.0915 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2037/Unknown  223s 90ms/step - categorical_accuracy: 0.7224 - loss: 1.0914 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2038/Unknown  223s 90ms/step - categorical_accuracy: 0.7224 - loss: 1.0914 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2039/Unknown  223s 90ms/step - categorical_accuracy: 0.7225 - loss: 1.0913 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2040/Unknown  224s 90ms/step - categorical_accuracy: 0.7225 - loss: 1.0912 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2041/Unknown  224s 90ms/step - categorical_accuracy: 0.7225 - loss: 1.0911 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2042/Unknown  224s 90ms/step - categorical_accuracy: 0.7225 - loss: 1.0910 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2043/Unknown  224s 90ms/step - categorical_accuracy: 0.7225 - loss: 1.0909 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2044/Unknown  224s 90ms/step - categorical_accuracy: 0.7225 - loss: 1.0908 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2045/Unknown  224s 90ms/step - categorical_accuracy: 0.7226 - loss: 1.0908 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2046/Unknown  224s 90ms/step - categorical_accuracy: 0.7226 - loss: 1.0907 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2047/Unknown  224s 90ms/step - categorical_accuracy: 0.7226 - loss: 1.0906 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2048/Unknown  224s 90ms/step - categorical_accuracy: 0.7226 - loss: 1.0905 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2049/Unknown  224s 90ms/step - categorical_accuracy: 0.7226 - loss: 1.0904 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2050/Unknown  224s 90ms/step - categorical_accuracy: 0.7226 - loss: 1.0903 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2051/Unknown  224s 90ms/step - categorical_accuracy: 0.7227 - loss: 1.0902 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2052/Unknown  225s 90ms/step - categorical_accuracy: 0.7227 - loss: 1.0902 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2053/Unknown  225s 90ms/step - categorical_accuracy: 0.7227 - loss: 1.0901 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2054/Unknown  225s 90ms/step - categorical_accuracy: 0.7227 - loss: 1.0900 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2055/Unknown  225s 90ms/step - categorical_accuracy: 0.7227 - loss: 1.0899 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2056/Unknown  225s 90ms/step - categorical_accuracy: 0.7227 - loss: 1.0898 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2057/Unknown  225s 90ms/step - categorical_accuracy: 0.7228 - loss: 1.0897 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2058/Unknown  225s 90ms/step - categorical_accuracy: 0.7228 - loss: 1.0896 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2059/Unknown  225s 90ms/step - categorical_accuracy: 0.7228 - loss: 1.0896 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2060/Unknown  225s 90ms/step - categorical_accuracy: 0.7228 - loss: 1.0895 - mean_io_u: 0.0928

<div class="k-default-codeblock">
```

```
</div>
   2061/Unknown  225s 90ms/step - categorical_accuracy: 0.7228 - loss: 1.0894 - mean_io_u: 0.0928

<div class="k-default-codeblock">
```

```
</div>
   2062/Unknown  226s 90ms/step - categorical_accuracy: 0.7228 - loss: 1.0893 - mean_io_u: 0.0928

<div class="k-default-codeblock">
```

```
</div>
   2063/Unknown  226s 90ms/step - categorical_accuracy: 0.7229 - loss: 1.0892 - mean_io_u: 0.0928

<div class="k-default-codeblock">
```

```
</div>
   2064/Unknown  226s 90ms/step - categorical_accuracy: 0.7229 - loss: 1.0891 - mean_io_u: 0.0928

<div class="k-default-codeblock">
```

```
</div>
   2065/Unknown  226s 90ms/step - categorical_accuracy: 0.7229 - loss: 1.0890 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2066/Unknown  226s 90ms/step - categorical_accuracy: 0.7229 - loss: 1.0890 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2067/Unknown  226s 90ms/step - categorical_accuracy: 0.7229 - loss: 1.0889 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2068/Unknown  226s 90ms/step - categorical_accuracy: 0.7229 - loss: 1.0888 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2069/Unknown  226s 90ms/step - categorical_accuracy: 0.7230 - loss: 1.0887 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2070/Unknown  226s 90ms/step - categorical_accuracy: 0.7230 - loss: 1.0886 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2071/Unknown  226s 90ms/step - categorical_accuracy: 0.7230 - loss: 1.0885 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2072/Unknown  226s 90ms/step - categorical_accuracy: 0.7230 - loss: 1.0885 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2073/Unknown  226s 90ms/step - categorical_accuracy: 0.7230 - loss: 1.0884 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2074/Unknown  226s 90ms/step - categorical_accuracy: 0.7230 - loss: 1.0883 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2075/Unknown  227s 90ms/step - categorical_accuracy: 0.7231 - loss: 1.0882 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2076/Unknown  227s 90ms/step - categorical_accuracy: 0.7231 - loss: 1.0881 - mean_io_u: 0.0931

<div class="k-default-codeblock">
```

```
</div>
   2077/Unknown  227s 90ms/step - categorical_accuracy: 0.7231 - loss: 1.0880 - mean_io_u: 0.0931

<div class="k-default-codeblock">
```

```
</div>
   2078/Unknown  227s 90ms/step - categorical_accuracy: 0.7231 - loss: 1.0880 - mean_io_u: 0.0931

<div class="k-default-codeblock">
```

```
</div>
   2079/Unknown  227s 90ms/step - categorical_accuracy: 0.7231 - loss: 1.0879 - mean_io_u: 0.0931

<div class="k-default-codeblock">
```

```
</div>
   2080/Unknown  227s 90ms/step - categorical_accuracy: 0.7231 - loss: 1.0878 - mean_io_u: 0.0931

<div class="k-default-codeblock">
```

```
</div>
   2081/Unknown  227s 90ms/step - categorical_accuracy: 0.7232 - loss: 1.0877 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2082/Unknown  227s 90ms/step - categorical_accuracy: 0.7232 - loss: 1.0876 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2083/Unknown  227s 90ms/step - categorical_accuracy: 0.7232 - loss: 1.0875 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2084/Unknown  227s 90ms/step - categorical_accuracy: 0.7232 - loss: 1.0875 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2085/Unknown  227s 90ms/step - categorical_accuracy: 0.7232 - loss: 1.0874 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2086/Unknown  227s 90ms/step - categorical_accuracy: 0.7232 - loss: 1.0873 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2087/Unknown  228s 90ms/step - categorical_accuracy: 0.7232 - loss: 1.0872 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2088/Unknown  228s 90ms/step - categorical_accuracy: 0.7233 - loss: 1.0871 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2089/Unknown  228s 90ms/step - categorical_accuracy: 0.7233 - loss: 1.0870 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2090/Unknown  228s 90ms/step - categorical_accuracy: 0.7233 - loss: 1.0869 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2091/Unknown  228s 90ms/step - categorical_accuracy: 0.7233 - loss: 1.0869 - mean_io_u: 0.0934

<div class="k-default-codeblock">
```

```
</div>
   2092/Unknown  228s 90ms/step - categorical_accuracy: 0.7233 - loss: 1.0868 - mean_io_u: 0.0934

<div class="k-default-codeblock">
```

```
</div>
   2093/Unknown  228s 90ms/step - categorical_accuracy: 0.7233 - loss: 1.0867 - mean_io_u: 0.0934

<div class="k-default-codeblock">
```

```
</div>
   2094/Unknown  228s 90ms/step - categorical_accuracy: 0.7234 - loss: 1.0866 - mean_io_u: 0.0934

<div class="k-default-codeblock">
```

```
</div>
   2095/Unknown  228s 90ms/step - categorical_accuracy: 0.7234 - loss: 1.0865 - mean_io_u: 0.0934

<div class="k-default-codeblock">
```

```
</div>
   2096/Unknown  228s 90ms/step - categorical_accuracy: 0.7234 - loss: 1.0864 - mean_io_u: 0.0935

<div class="k-default-codeblock">
```

```
</div>
   2097/Unknown  228s 90ms/step - categorical_accuracy: 0.7234 - loss: 1.0864 - mean_io_u: 0.0935

<div class="k-default-codeblock">
```

```
</div>
   2098/Unknown  228s 90ms/step - categorical_accuracy: 0.7234 - loss: 1.0863 - mean_io_u: 0.0935

<div class="k-default-codeblock">
```

```
</div>
   2099/Unknown  229s 90ms/step - categorical_accuracy: 0.7234 - loss: 1.0862 - mean_io_u: 0.0935

<div class="k-default-codeblock">
```

```
</div>
   2100/Unknown  229s 90ms/step - categorical_accuracy: 0.7235 - loss: 1.0861 - mean_io_u: 0.0935

<div class="k-default-codeblock">
```

```
</div>
   2101/Unknown  229s 90ms/step - categorical_accuracy: 0.7235 - loss: 1.0860 - mean_io_u: 0.0935

<div class="k-default-codeblock">
```

```
</div>
   2102/Unknown  229s 90ms/step - categorical_accuracy: 0.7235 - loss: 1.0859 - mean_io_u: 0.0936

<div class="k-default-codeblock">
```

```
</div>
   2103/Unknown  229s 90ms/step - categorical_accuracy: 0.7235 - loss: 1.0859 - mean_io_u: 0.0936

<div class="k-default-codeblock">
```

```
</div>
   2104/Unknown  229s 90ms/step - categorical_accuracy: 0.7235 - loss: 1.0858 - mean_io_u: 0.0936

<div class="k-default-codeblock">
```

```
</div>
   2105/Unknown  229s 90ms/step - categorical_accuracy: 0.7235 - loss: 1.0857 - mean_io_u: 0.0936

<div class="k-default-codeblock">
```

```
</div>
   2106/Unknown  229s 90ms/step - categorical_accuracy: 0.7236 - loss: 1.0856 - mean_io_u: 0.0936

<div class="k-default-codeblock">
```

```
</div>
   2107/Unknown  229s 90ms/step - categorical_accuracy: 0.7236 - loss: 1.0855 - mean_io_u: 0.0937

<div class="k-default-codeblock">
```

```
</div>
   2108/Unknown  229s 90ms/step - categorical_accuracy: 0.7236 - loss: 1.0854 - mean_io_u: 0.0937

<div class="k-default-codeblock">
```

```
</div>
   2109/Unknown  229s 90ms/step - categorical_accuracy: 0.7236 - loss: 1.0854 - mean_io_u: 0.0937

<div class="k-default-codeblock">
```

```
</div>
   2110/Unknown  229s 90ms/step - categorical_accuracy: 0.7236 - loss: 1.0853 - mean_io_u: 0.0937

<div class="k-default-codeblock">
```

```
</div>
   2111/Unknown  229s 90ms/step - categorical_accuracy: 0.7236 - loss: 1.0852 - mean_io_u: 0.0937

<div class="k-default-codeblock">
```

```
</div>
   2112/Unknown  230s 90ms/step - categorical_accuracy: 0.7237 - loss: 1.0851 - mean_io_u: 0.0938

<div class="k-default-codeblock">
```

```
</div>
   2113/Unknown  230s 90ms/step - categorical_accuracy: 0.7237 - loss: 1.0850 - mean_io_u: 0.0938

<div class="k-default-codeblock">
```

```
</div>
   2114/Unknown  230s 90ms/step - categorical_accuracy: 0.7237 - loss: 1.0849 - mean_io_u: 0.0938

<div class="k-default-codeblock">
```

```
</div>
   2115/Unknown  230s 90ms/step - categorical_accuracy: 0.7237 - loss: 1.0849 - mean_io_u: 0.0938

<div class="k-default-codeblock">
```

```
</div>
   2116/Unknown  230s 90ms/step - categorical_accuracy: 0.7237 - loss: 1.0848 - mean_io_u: 0.0938

<div class="k-default-codeblock">
```

```
</div>
   2117/Unknown  230s 90ms/step - categorical_accuracy: 0.7237 - loss: 1.0847 - mean_io_u: 0.0939

<div class="k-default-codeblock">
```

```
</div>
   2118/Unknown  230s 90ms/step - categorical_accuracy: 0.7238 - loss: 1.0846 - mean_io_u: 0.0939

<div class="k-default-codeblock">
```

```
</div>
   2119/Unknown  230s 90ms/step - categorical_accuracy: 0.7238 - loss: 1.0845 - mean_io_u: 0.0939

<div class="k-default-codeblock">
```

```
</div>
   2120/Unknown  230s 90ms/step - categorical_accuracy: 0.7238 - loss: 1.0845 - mean_io_u: 0.0939

<div class="k-default-codeblock">
```

```
</div>
   2121/Unknown  230s 90ms/step - categorical_accuracy: 0.7238 - loss: 1.0844 - mean_io_u: 0.0939

<div class="k-default-codeblock">
```

```
</div>
   2122/Unknown  230s 90ms/step - categorical_accuracy: 0.7238 - loss: 1.0843 - mean_io_u: 0.0939

<div class="k-default-codeblock">
```

```
</div>
   2123/Unknown  230s 90ms/step - categorical_accuracy: 0.7238 - loss: 1.0841 - mean_io_u: 0.0940

<div class="k-default-codeblock">
```

```
</div>
   2124/Unknown  230s 89ms/step - categorical_accuracy: 0.7238 - loss: 1.0841 - mean_io_u: 0.0940

<div class="k-default-codeblock">
```
/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/models/functional.py:225: UserWarning: The structure of `inputs` doesn't match the expected structure: ['keras_tensor_222']. Received: the structure of inputs=*
  warnings.warn(
/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/backend/jax/core.py:76: UserWarning: Explicitly requested dtype int64 requested in asarray is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return jnp.asarray(x, dtype=dtype)


```
</div>
 2124/2124 ━━━━━━━━━━━━━━━━━━━━ 281s 113ms/step - categorical_accuracy: 0.7239 - loss: 1.0840 - mean_io_u: 0.0940 - val_categorical_accuracy: 0.8294 - val_loss: 0.5578 - val_mean_io_u: 0.3539





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7fbec838c8d0>

```
</div>
---
## Predictions with trained model
Now that the model training of DeepLabv3+ has completed, let's test it by making
predications
on a few sample images.
Note: For demonstration purpose the model has been trained on only 1 epoch, for
better accuracy and result train with more number of epochs.


```python
test_ds = load(split="sbd_eval")
test_ds = preprocess_inputs(test_ds)

images, masks = next(iter(train_ds.take(1)))
images = ops.convert_to_tensor(images)
masks = ops.convert_to_tensor(masks)
preds = ops.expand_dims(ops.argmax(model(images), axis=-1), axis=-1)
masks = ops.expand_dims(ops.argmax(masks, axis=-1), axis=-1)

plot_images_masks(images, masks, preds)
```

<div class="k-default-codeblock">
```
/usr/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()

/usr/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()

/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/models/functional.py:225: UserWarning: The structure of `inputs` doesn't match the expected structure: ['keras_tensor_222']. Received: the structure of inputs=*
  warnings.warn(

```
</div>
    
![png](/home/sachinprasad/projects/keras-io/guides/img/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_32_3.png)
    


Here are some additional tips for using the KerasHub DeepLabv3 model:

- The model can be trained on a variety of datasets, including the COCO dataset, the
PASCAL VOC dataset, and the Cityscapes dataset.
- The model can be fine-tuned on a custom dataset to improve its performance on a
specific task.
- The model can be used to perform real-time inference on images.
- Also, check out KerasHub's other segmentation models.

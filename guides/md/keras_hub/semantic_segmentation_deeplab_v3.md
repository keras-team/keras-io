# Semantic Segmentation with KerasHub

**Authors:** [Sachin Prasad](https://github.com/sachinprasadhs), [Divyashree Sreepathihalli](https://github.com/divyashreepathihalli), [Ian Stenbit](https://github.com/ianstenbit)<br>
**Date created:** 2024/10/11<br>
**Last modified:** 2024/10/11<br>
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
    origin="https://storage.googleapis.com/keras-cv/pictures/dog.jpeg"
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
    plt.imshow(predicted_mask, cmap="gray")
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
[Semantic contours from inverse detectors](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
and split them into train dataset `train_ds` and `eval_ds`.


```python
# @title helper functions
import logging
import multiprocess as multiprocessing
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
/home/sachinprasad/projects/env/lib/python3.11/site-packages/multiprocess/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()

/home/sachinprasad/projects/env/lib/python3.11/site-packages/multiprocess/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
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
        plt.imshow(masks[i], cmap="gray")
        plt.axis("off")

        if pred_masks is not None:
            plt.subplot(rows, num_images, i + 1 + 2 * num_images)
            plt.imshow(pred_masks[i, ..., 0], cmap="gray")
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
  1/Unknown  40s 40s/step - categorical_accuracy: 0.0494 - loss: 3.4081 - mean_io_u: 0.0112


  2/Unknown  61s 21s/step - categorical_accuracy: 0.0441 - loss: 3.4118 - mean_io_u: 0.0102


  3/Unknown  61s 11s/step - categorical_accuracy: 0.0415 - loss: 3.4205 - mean_io_u: 0.0100


  4/Unknown  61s 7s/step - categorical_accuracy: 0.0407 - loss: 3.4113 - mean_io_u: 0.0099 


  5/Unknown  61s 5s/step - categorical_accuracy: 0.0405 - loss: 3.3974 - mean_io_u: 0.0098


  6/Unknown  61s 4s/step - categorical_accuracy: 0.0420 - loss: 3.3817 - mean_io_u: 0.0099


  7/Unknown  61s 4s/step - categorical_accuracy: 0.0452 - loss: 3.3606 - mean_io_u: 0.0103


  8/Unknown  61s 3s/step - categorical_accuracy: 0.0499 - loss: 3.3363 - mean_io_u: 0.0107


  9/Unknown  61s 3s/step - categorical_accuracy: 0.0572 - loss: 3.3097 - mean_io_u: 0.0113


 10/Unknown  61s 2s/step - categorical_accuracy: 0.0659 - loss: 3.2798 - mean_io_u: 0.0119


 11/Unknown  62s 2s/step - categorical_accuracy: 0.0748 - loss: 3.2508 - mean_io_u: 0.0126


 12/Unknown  62s 2s/step - categorical_accuracy: 0.0847 - loss: 3.2212 - mean_io_u: 0.0133


 13/Unknown  62s 2s/step - categorical_accuracy: 0.0949 - loss: 3.1906 - mean_io_u: 0.0140


 14/Unknown  62s 2s/step - categorical_accuracy: 0.1057 - loss: 3.1589 - mean_io_u: 0.0147


 15/Unknown  62s 2s/step - categorical_accuracy: 0.1170 - loss: 3.1265 - mean_io_u: 0.0155


 16/Unknown  62s 1s/step - categorical_accuracy: 0.1289 - loss: 3.0923 - mean_io_u: 0.0163


 17/Unknown  62s 1s/step - categorical_accuracy: 0.1406 - loss: 3.0588 - mean_io_u: 0.0170


 18/Unknown  62s 1s/step - categorical_accuracy: 0.1526 - loss: 3.0248 - mean_io_u: 0.0178


 19/Unknown  62s 1s/step - categorical_accuracy: 0.1636 - loss: 2.9934 - mean_io_u: 0.0184


 20/Unknown  62s 1s/step - categorical_accuracy: 0.1745 - loss: 2.9624 - mean_io_u: 0.0191


 21/Unknown  62s 1s/step - categorical_accuracy: 0.1853 - loss: 2.9313 - mean_io_u: 0.0197


 22/Unknown  62s 1s/step - categorical_accuracy: 0.1956 - loss: 2.9016 - mean_io_u: 0.0203


 23/Unknown  62s 1s/step - categorical_accuracy: 0.2056 - loss: 2.8727 - mean_io_u: 0.0209


 24/Unknown  63s 994ms/step - categorical_accuracy: 0.2154 - loss: 2.8442 - mean_io_u: 0.0214


 25/Unknown  63s 955ms/step - categorical_accuracy: 0.2247 - loss: 2.8178 - mean_io_u: 0.0219


 26/Unknown  63s 919ms/step - categorical_accuracy: 0.2336 - loss: 2.7921 - mean_io_u: 0.0224


 27/Unknown  63s 887ms/step - categorical_accuracy: 0.2421 - loss: 2.7672 - mean_io_u: 0.0229


 28/Unknown  63s 858ms/step - categorical_accuracy: 0.2505 - loss: 2.7422 - mean_io_u: 0.0233


 29/Unknown  63s 832ms/step - categorical_accuracy: 0.2583 - loss: 2.7195 - mean_io_u: 0.0237


 30/Unknown  63s 808ms/step - categorical_accuracy: 0.2660 - loss: 2.6971 - mean_io_u: 0.0241


 31/Unknown  63s 785ms/step - categorical_accuracy: 0.2733 - loss: 2.6758 - mean_io_u: 0.0245


 32/Unknown  63s 764ms/step - categorical_accuracy: 0.2802 - loss: 2.6558 - mean_io_u: 0.0248


 33/Unknown  63s 742ms/step - categorical_accuracy: 0.2869 - loss: 2.6365 - mean_io_u: 0.0251


 34/Unknown  63s 722ms/step - categorical_accuracy: 0.2933 - loss: 2.6182 - mean_io_u: 0.0254


 35/Unknown  64s 704ms/step - categorical_accuracy: 0.2995 - loss: 2.6000 - mean_io_u: 0.0257


 36/Unknown  64s 688ms/step - categorical_accuracy: 0.3055 - loss: 2.5824 - mean_io_u: 0.0260


 37/Unknown  64s 672ms/step - categorical_accuracy: 0.3113 - loss: 2.5653 - mean_io_u: 0.0262


 38/Unknown  64s 657ms/step - categorical_accuracy: 0.3169 - loss: 2.5491 - mean_io_u: 0.0264


 39/Unknown  64s 642ms/step - categorical_accuracy: 0.3222 - loss: 2.5339 - mean_io_u: 0.0266


 40/Unknown  64s 628ms/step - categorical_accuracy: 0.3273 - loss: 2.5196 - mean_io_u: 0.0268


 41/Unknown  64s 615ms/step - categorical_accuracy: 0.3323 - loss: 2.5052 - mean_io_u: 0.0270


 42/Unknown  64s 604ms/step - categorical_accuracy: 0.3373 - loss: 2.4909 - mean_io_u: 0.0272


 43/Unknown  64s 591ms/step - categorical_accuracy: 0.3421 - loss: 2.4770 - mean_io_u: 0.0273


 44/Unknown  65s 579ms/step - categorical_accuracy: 0.3466 - loss: 2.4640 - mean_io_u: 0.0275


 45/Unknown  65s 568ms/step - categorical_accuracy: 0.3510 - loss: 2.4515 - mean_io_u: 0.0276


 46/Unknown  65s 558ms/step - categorical_accuracy: 0.3553 - loss: 2.4394 - mean_io_u: 0.0278


 47/Unknown  65s 548ms/step - categorical_accuracy: 0.3595 - loss: 2.4273 - mean_io_u: 0.0279


 48/Unknown  65s 538ms/step - categorical_accuracy: 0.3636 - loss: 2.4155 - mean_io_u: 0.0280


 49/Unknown  65s 530ms/step - categorical_accuracy: 0.3676 - loss: 2.4041 - mean_io_u: 0.0282


 50/Unknown  65s 522ms/step - categorical_accuracy: 0.3715 - loss: 2.3928 - mean_io_u: 0.0283


 51/Unknown  65s 513ms/step - categorical_accuracy: 0.3752 - loss: 2.3820 - mean_io_u: 0.0284


 52/Unknown  65s 505ms/step - categorical_accuracy: 0.3788 - loss: 2.3719 - mean_io_u: 0.0285


 53/Unknown  65s 497ms/step - categorical_accuracy: 0.3822 - loss: 2.3620 - mean_io_u: 0.0286


 54/Unknown  66s 490ms/step - categorical_accuracy: 0.3856 - loss: 2.3522 - mean_io_u: 0.0287


 55/Unknown  66s 482ms/step - categorical_accuracy: 0.3889 - loss: 2.3429 - mean_io_u: 0.0288


 56/Unknown  66s 475ms/step - categorical_accuracy: 0.3921 - loss: 2.3336 - mean_io_u: 0.0289


 57/Unknown  66s 469ms/step - categorical_accuracy: 0.3951 - loss: 2.3247 - mean_io_u: 0.0290


 58/Unknown  66s 463ms/step - categorical_accuracy: 0.3981 - loss: 2.3160 - mean_io_u: 0.0291


 59/Unknown  66s 456ms/step - categorical_accuracy: 0.4011 - loss: 2.3073 - mean_io_u: 0.0292


 60/Unknown  66s 450ms/step - categorical_accuracy: 0.4040 - loss: 2.2989 - mean_io_u: 0.0293


 61/Unknown  66s 445ms/step - categorical_accuracy: 0.4068 - loss: 2.2905 - mean_io_u: 0.0294


 62/Unknown  66s 439ms/step - categorical_accuracy: 0.4096 - loss: 2.2822 - mean_io_u: 0.0295


 63/Unknown  67s 434ms/step - categorical_accuracy: 0.4123 - loss: 2.2741 - mean_io_u: 0.0296


 64/Unknown  67s 428ms/step - categorical_accuracy: 0.4149 - loss: 2.2661 - mean_io_u: 0.0297


 65/Unknown  67s 423ms/step - categorical_accuracy: 0.4175 - loss: 2.2584 - mean_io_u: 0.0298


 66/Unknown  67s 418ms/step - categorical_accuracy: 0.4200 - loss: 2.2508 - mean_io_u: 0.0299


 67/Unknown  67s 413ms/step - categorical_accuracy: 0.4224 - loss: 2.2434 - mean_io_u: 0.0299


 68/Unknown  67s 408ms/step - categorical_accuracy: 0.4248 - loss: 2.2360 - mean_io_u: 0.0300


 69/Unknown  67s 403ms/step - categorical_accuracy: 0.4272 - loss: 2.2287 - mean_io_u: 0.0301


 70/Unknown  67s 399ms/step - categorical_accuracy: 0.4296 - loss: 2.2214 - mean_io_u: 0.0302


 71/Unknown  67s 394ms/step - categorical_accuracy: 0.4319 - loss: 2.2143 - mean_io_u: 0.0302


 72/Unknown  67s 390ms/step - categorical_accuracy: 0.4341 - loss: 2.2072 - mean_io_u: 0.0303


 73/Unknown  67s 385ms/step - categorical_accuracy: 0.4364 - loss: 2.2001 - mean_io_u: 0.0304


 74/Unknown  67s 381ms/step - categorical_accuracy: 0.4386 - loss: 2.1932 - mean_io_u: 0.0305


 75/Unknown  68s 377ms/step - categorical_accuracy: 0.4407 - loss: 2.1863 - mean_io_u: 0.0306


 76/Unknown  68s 374ms/step - categorical_accuracy: 0.4429 - loss: 2.1795 - mean_io_u: 0.0307


 77/Unknown  68s 370ms/step - categorical_accuracy: 0.4450 - loss: 2.1728 - mean_io_u: 0.0307


 78/Unknown  68s 367ms/step - categorical_accuracy: 0.4470 - loss: 2.1662 - mean_io_u: 0.0308


 79/Unknown  68s 363ms/step - categorical_accuracy: 0.4490 - loss: 2.1597 - mean_io_u: 0.0309


 80/Unknown  68s 360ms/step - categorical_accuracy: 0.4510 - loss: 2.1533 - mean_io_u: 0.0310


 81/Unknown  68s 356ms/step - categorical_accuracy: 0.4529 - loss: 2.1471 - mean_io_u: 0.0311


 82/Unknown  68s 354ms/step - categorical_accuracy: 0.4548 - loss: 2.1410 - mean_io_u: 0.0312


 83/Unknown  68s 350ms/step - categorical_accuracy: 0.4566 - loss: 2.1350 - mean_io_u: 0.0312


 84/Unknown  68s 347ms/step - categorical_accuracy: 0.4584 - loss: 2.1292 - mean_io_u: 0.0313


 85/Unknown  69s 345ms/step - categorical_accuracy: 0.4602 - loss: 2.1233 - mean_io_u: 0.0314


 86/Unknown  69s 342ms/step - categorical_accuracy: 0.4619 - loss: 2.1176 - mean_io_u: 0.0314


 87/Unknown  69s 339ms/step - categorical_accuracy: 0.4636 - loss: 2.1120 - mean_io_u: 0.0315


 88/Unknown  69s 336ms/step - categorical_accuracy: 0.4653 - loss: 2.1064 - mean_io_u: 0.0316


 89/Unknown  69s 334ms/step - categorical_accuracy: 0.4670 - loss: 2.1009 - mean_io_u: 0.0316


 90/Unknown  69s 331ms/step - categorical_accuracy: 0.4686 - loss: 2.0957 - mean_io_u: 0.0317


 91/Unknown  69s 328ms/step - categorical_accuracy: 0.4701 - loss: 2.0905 - mean_io_u: 0.0317


 92/Unknown  69s 325ms/step - categorical_accuracy: 0.4717 - loss: 2.0855 - mean_io_u: 0.0318


 93/Unknown  69s 323ms/step - categorical_accuracy: 0.4732 - loss: 2.0805 - mean_io_u: 0.0318


 94/Unknown  69s 321ms/step - categorical_accuracy: 0.4746 - loss: 2.0757 - mean_io_u: 0.0319


 95/Unknown  70s 318ms/step - categorical_accuracy: 0.4761 - loss: 2.0710 - mean_io_u: 0.0319


 96/Unknown  70s 315ms/step - categorical_accuracy: 0.4775 - loss: 2.0663 - mean_io_u: 0.0320


 97/Unknown  70s 313ms/step - categorical_accuracy: 0.4790 - loss: 2.0616 - mean_io_u: 0.0320


 98/Unknown  70s 311ms/step - categorical_accuracy: 0.4804 - loss: 2.0570 - mean_io_u: 0.0321


 99/Unknown  70s 308ms/step - categorical_accuracy: 0.4818 - loss: 2.0523 - mean_io_u: 0.0321


100/Unknown  70s 306ms/step - categorical_accuracy: 0.4831 - loss: 2.0478 - mean_io_u: 0.0322


101/Unknown  70s 304ms/step - categorical_accuracy: 0.4845 - loss: 2.0433 - mean_io_u: 0.0322


102/Unknown  70s 301ms/step - categorical_accuracy: 0.4858 - loss: 2.0389 - mean_io_u: 0.0323


103/Unknown  70s 299ms/step - categorical_accuracy: 0.4871 - loss: 2.0345 - mean_io_u: 0.0323


104/Unknown  70s 298ms/step - categorical_accuracy: 0.4884 - loss: 2.0301 - mean_io_u: 0.0323


105/Unknown  70s 296ms/step - categorical_accuracy: 0.4897 - loss: 2.0259 - mean_io_u: 0.0324


106/Unknown  70s 294ms/step - categorical_accuracy: 0.4910 - loss: 2.0217 - mean_io_u: 0.0324


107/Unknown  71s 292ms/step - categorical_accuracy: 0.4922 - loss: 2.0176 - mean_io_u: 0.0325


108/Unknown  71s 290ms/step - categorical_accuracy: 0.4934 - loss: 2.0136 - mean_io_u: 0.0325


109/Unknown  71s 288ms/step - categorical_accuracy: 0.4946 - loss: 2.0096 - mean_io_u: 0.0325


110/Unknown  71s 287ms/step - categorical_accuracy: 0.4957 - loss: 2.0057 - mean_io_u: 0.0326


111/Unknown  71s 285ms/step - categorical_accuracy: 0.4969 - loss: 2.0018 - mean_io_u: 0.0326


112/Unknown  71s 284ms/step - categorical_accuracy: 0.4980 - loss: 1.9980 - mean_io_u: 0.0326


113/Unknown  71s 282ms/step - categorical_accuracy: 0.4992 - loss: 1.9942 - mean_io_u: 0.0327


114/Unknown  71s 280ms/step - categorical_accuracy: 0.5003 - loss: 1.9903 - mean_io_u: 0.0327


115/Unknown  71s 278ms/step - categorical_accuracy: 0.5014 - loss: 1.9866 - mean_io_u: 0.0327


116/Unknown  72s 277ms/step - categorical_accuracy: 0.5025 - loss: 1.9829 - mean_io_u: 0.0328


117/Unknown  72s 276ms/step - categorical_accuracy: 0.5036 - loss: 1.9792 - mean_io_u: 0.0328


118/Unknown  72s 274ms/step - categorical_accuracy: 0.5046 - loss: 1.9755 - mean_io_u: 0.0328


119/Unknown  72s 273ms/step - categorical_accuracy: 0.5057 - loss: 1.9718 - mean_io_u: 0.0329


120/Unknown  72s 271ms/step - categorical_accuracy: 0.5068 - loss: 1.9682 - mean_io_u: 0.0329


121/Unknown  72s 270ms/step - categorical_accuracy: 0.5078 - loss: 1.9647 - mean_io_u: 0.0329


122/Unknown  72s 268ms/step - categorical_accuracy: 0.5088 - loss: 1.9611 - mean_io_u: 0.0330


123/Unknown  72s 267ms/step - categorical_accuracy: 0.5098 - loss: 1.9577 - mean_io_u: 0.0330


124/Unknown  72s 266ms/step - categorical_accuracy: 0.5108 - loss: 1.9542 - mean_io_u: 0.0331


125/Unknown  72s 264ms/step - categorical_accuracy: 0.5118 - loss: 1.9507 - mean_io_u: 0.0331


126/Unknown  73s 263ms/step - categorical_accuracy: 0.5128 - loss: 1.9473 - mean_io_u: 0.0331


127/Unknown  73s 262ms/step - categorical_accuracy: 0.5138 - loss: 1.9439 - mean_io_u: 0.0332


128/Unknown  73s 261ms/step - categorical_accuracy: 0.5148 - loss: 1.9405 - mean_io_u: 0.0332


129/Unknown  73s 259ms/step - categorical_accuracy: 0.5158 - loss: 1.9371 - mean_io_u: 0.0332


130/Unknown  73s 258ms/step - categorical_accuracy: 0.5168 - loss: 1.9337 - mean_io_u: 0.0333


131/Unknown  73s 257ms/step - categorical_accuracy: 0.5178 - loss: 1.9303 - mean_io_u: 0.0333


132/Unknown  73s 255ms/step - categorical_accuracy: 0.5187 - loss: 1.9270 - mean_io_u: 0.0333


133/Unknown  73s 254ms/step - categorical_accuracy: 0.5197 - loss: 1.9236 - mean_io_u: 0.0334


134/Unknown  73s 253ms/step - categorical_accuracy: 0.5206 - loss: 1.9203 - mean_io_u: 0.0334


135/Unknown  73s 252ms/step - categorical_accuracy: 0.5215 - loss: 1.9171 - mean_io_u: 0.0335


136/Unknown  74s 251ms/step - categorical_accuracy: 0.5224 - loss: 1.9139 - mean_io_u: 0.0335


137/Unknown  74s 250ms/step - categorical_accuracy: 0.5233 - loss: 1.9107 - mean_io_u: 0.0335


138/Unknown  74s 249ms/step - categorical_accuracy: 0.5243 - loss: 1.9075 - mean_io_u: 0.0336


139/Unknown  74s 248ms/step - categorical_accuracy: 0.5251 - loss: 1.9044 - mean_io_u: 0.0336


140/Unknown  74s 247ms/step - categorical_accuracy: 0.5260 - loss: 1.9013 - mean_io_u: 0.0336


141/Unknown  74s 246ms/step - categorical_accuracy: 0.5269 - loss: 1.8982 - mean_io_u: 0.0337


142/Unknown  74s 245ms/step - categorical_accuracy: 0.5278 - loss: 1.8950 - mean_io_u: 0.0337


143/Unknown  74s 244ms/step - categorical_accuracy: 0.5287 - loss: 1.8920 - mean_io_u: 0.0337


144/Unknown  74s 244ms/step - categorical_accuracy: 0.5295 - loss: 1.8889 - mean_io_u: 0.0338


145/Unknown  75s 243ms/step - categorical_accuracy: 0.5304 - loss: 1.8859 - mean_io_u: 0.0338


146/Unknown  75s 242ms/step - categorical_accuracy: 0.5312 - loss: 1.8829 - mean_io_u: 0.0338


147/Unknown  75s 240ms/step - categorical_accuracy: 0.5320 - loss: 1.8800 - mean_io_u: 0.0339


148/Unknown  75s 239ms/step - categorical_accuracy: 0.5329 - loss: 1.8771 - mean_io_u: 0.0339


149/Unknown  75s 238ms/step - categorical_accuracy: 0.5337 - loss: 1.8742 - mean_io_u: 0.0339


150/Unknown  75s 237ms/step - categorical_accuracy: 0.5345 - loss: 1.8714 - mean_io_u: 0.0340


151/Unknown  75s 236ms/step - categorical_accuracy: 0.5353 - loss: 1.8685 - mean_io_u: 0.0340


152/Unknown  75s 235ms/step - categorical_accuracy: 0.5361 - loss: 1.8657 - mean_io_u: 0.0340


153/Unknown  75s 235ms/step - categorical_accuracy: 0.5368 - loss: 1.8628 - mean_io_u: 0.0341


154/Unknown  75s 234ms/step - categorical_accuracy: 0.5376 - loss: 1.8600 - mean_io_u: 0.0341


155/Unknown  76s 233ms/step - categorical_accuracy: 0.5384 - loss: 1.8573 - mean_io_u: 0.0341


156/Unknown  76s 232ms/step - categorical_accuracy: 0.5391 - loss: 1.8546 - mean_io_u: 0.0342


157/Unknown  76s 231ms/step - categorical_accuracy: 0.5399 - loss: 1.8519 - mean_io_u: 0.0342


158/Unknown  76s 230ms/step - categorical_accuracy: 0.5406 - loss: 1.8492 - mean_io_u: 0.0343


159/Unknown  76s 229ms/step - categorical_accuracy: 0.5413 - loss: 1.8466 - mean_io_u: 0.0343


160/Unknown  76s 229ms/step - categorical_accuracy: 0.5421 - loss: 1.8440 - mean_io_u: 0.0343


161/Unknown  76s 228ms/step - categorical_accuracy: 0.5428 - loss: 1.8414 - mean_io_u: 0.0344


162/Unknown  76s 227ms/step - categorical_accuracy: 0.5435 - loss: 1.8388 - mean_io_u: 0.0344


163/Unknown  76s 226ms/step - categorical_accuracy: 0.5442 - loss: 1.8362 - mean_io_u: 0.0344


164/Unknown  76s 226ms/step - categorical_accuracy: 0.5449 - loss: 1.8336 - mean_io_u: 0.0345


165/Unknown  77s 225ms/step - categorical_accuracy: 0.5456 - loss: 1.8311 - mean_io_u: 0.0345


166/Unknown  77s 224ms/step - categorical_accuracy: 0.5463 - loss: 1.8286 - mean_io_u: 0.0346


167/Unknown  77s 223ms/step - categorical_accuracy: 0.5470 - loss: 1.8261 - mean_io_u: 0.0346


168/Unknown  77s 222ms/step - categorical_accuracy: 0.5476 - loss: 1.8237 - mean_io_u: 0.0346


169/Unknown  77s 222ms/step - categorical_accuracy: 0.5483 - loss: 1.8212 - mean_io_u: 0.0347


170/Unknown  77s 221ms/step - categorical_accuracy: 0.5490 - loss: 1.8188 - mean_io_u: 0.0347


171/Unknown  77s 220ms/step - categorical_accuracy: 0.5496 - loss: 1.8164 - mean_io_u: 0.0348


172/Unknown  77s 219ms/step - categorical_accuracy: 0.5503 - loss: 1.8140 - mean_io_u: 0.0348


173/Unknown  77s 219ms/step - categorical_accuracy: 0.5509 - loss: 1.8116 - mean_io_u: 0.0348


174/Unknown  77s 218ms/step - categorical_accuracy: 0.5515 - loss: 1.8093 - mean_io_u: 0.0349


175/Unknown  78s 218ms/step - categorical_accuracy: 0.5522 - loss: 1.8069 - mean_io_u: 0.0349


176/Unknown  78s 217ms/step - categorical_accuracy: 0.5528 - loss: 1.8046 - mean_io_u: 0.0350


177/Unknown  78s 216ms/step - categorical_accuracy: 0.5534 - loss: 1.8022 - mean_io_u: 0.0350


178/Unknown  78s 216ms/step - categorical_accuracy: 0.5540 - loss: 1.7999 - mean_io_u: 0.0351


179/Unknown  78s 215ms/step - categorical_accuracy: 0.5547 - loss: 1.7977 - mean_io_u: 0.0351


180/Unknown  78s 214ms/step - categorical_accuracy: 0.5553 - loss: 1.7954 - mean_io_u: 0.0351


181/Unknown  78s 214ms/step - categorical_accuracy: 0.5559 - loss: 1.7931 - mean_io_u: 0.0352


182/Unknown  78s 213ms/step - categorical_accuracy: 0.5565 - loss: 1.7909 - mean_io_u: 0.0352


183/Unknown  78s 212ms/step - categorical_accuracy: 0.5571 - loss: 1.7887 - mean_io_u: 0.0353


184/Unknown  78s 212ms/step - categorical_accuracy: 0.5576 - loss: 1.7865 - mean_io_u: 0.0353


185/Unknown  78s 211ms/step - categorical_accuracy: 0.5582 - loss: 1.7843 - mean_io_u: 0.0354


186/Unknown  79s 210ms/step - categorical_accuracy: 0.5588 - loss: 1.7822 - mean_io_u: 0.0354


187/Unknown  79s 210ms/step - categorical_accuracy: 0.5594 - loss: 1.7801 - mean_io_u: 0.0354


188/Unknown  79s 209ms/step - categorical_accuracy: 0.5599 - loss: 1.7779 - mean_io_u: 0.0355


189/Unknown  79s 209ms/step - categorical_accuracy: 0.5605 - loss: 1.7758 - mean_io_u: 0.0355


190/Unknown  79s 208ms/step - categorical_accuracy: 0.5611 - loss: 1.7737 - mean_io_u: 0.0356


191/Unknown  79s 208ms/step - categorical_accuracy: 0.5616 - loss: 1.7717 - mean_io_u: 0.0356


192/Unknown  79s 207ms/step - categorical_accuracy: 0.5622 - loss: 1.7696 - mean_io_u: 0.0357


193/Unknown  79s 207ms/step - categorical_accuracy: 0.5627 - loss: 1.7676 - mean_io_u: 0.0357


194/Unknown  79s 206ms/step - categorical_accuracy: 0.5633 - loss: 1.7655 - mean_io_u: 0.0357


195/Unknown  79s 205ms/step - categorical_accuracy: 0.5638 - loss: 1.7635 - mean_io_u: 0.0358


196/Unknown  80s 205ms/step - categorical_accuracy: 0.5643 - loss: 1.7615 - mean_io_u: 0.0358


197/Unknown  80s 204ms/step - categorical_accuracy: 0.5649 - loss: 1.7596 - mean_io_u: 0.0359


198/Unknown  80s 204ms/step - categorical_accuracy: 0.5654 - loss: 1.7576 - mean_io_u: 0.0359


199/Unknown  80s 203ms/step - categorical_accuracy: 0.5659 - loss: 1.7557 - mean_io_u: 0.0360


200/Unknown  80s 202ms/step - categorical_accuracy: 0.5664 - loss: 1.7537 - mean_io_u: 0.0360


201/Unknown  80s 202ms/step - categorical_accuracy: 0.5669 - loss: 1.7518 - mean_io_u: 0.0360


202/Unknown  80s 201ms/step - categorical_accuracy: 0.5674 - loss: 1.7499 - mean_io_u: 0.0361


203/Unknown  80s 201ms/step - categorical_accuracy: 0.5679 - loss: 1.7480 - mean_io_u: 0.0361


204/Unknown  80s 200ms/step - categorical_accuracy: 0.5685 - loss: 1.7461 - mean_io_u: 0.0362


205/Unknown  80s 200ms/step - categorical_accuracy: 0.5690 - loss: 1.7442 - mean_io_u: 0.0362


206/Unknown  80s 199ms/step - categorical_accuracy: 0.5695 - loss: 1.7423 - mean_io_u: 0.0363


207/Unknown  81s 199ms/step - categorical_accuracy: 0.5699 - loss: 1.7404 - mean_io_u: 0.0363


208/Unknown  81s 198ms/step - categorical_accuracy: 0.5704 - loss: 1.7386 - mean_io_u: 0.0364


209/Unknown  81s 197ms/step - categorical_accuracy: 0.5709 - loss: 1.7367 - mean_io_u: 0.0364


210/Unknown  81s 197ms/step - categorical_accuracy: 0.5714 - loss: 1.7348 - mean_io_u: 0.0365


211/Unknown  81s 196ms/step - categorical_accuracy: 0.5719 - loss: 1.7330 - mean_io_u: 0.0365


212/Unknown  81s 196ms/step - categorical_accuracy: 0.5724 - loss: 1.7311 - mean_io_u: 0.0366


213/Unknown  81s 195ms/step - categorical_accuracy: 0.5729 - loss: 1.7293 - mean_io_u: 0.0366


214/Unknown  81s 195ms/step - categorical_accuracy: 0.5734 - loss: 1.7275 - mean_io_u: 0.0367


215/Unknown  81s 194ms/step - categorical_accuracy: 0.5738 - loss: 1.7257 - mean_io_u: 0.0367


216/Unknown  81s 193ms/step - categorical_accuracy: 0.5743 - loss: 1.7239 - mean_io_u: 0.0368


217/Unknown  81s 193ms/step - categorical_accuracy: 0.5748 - loss: 1.7221 - mean_io_u: 0.0368


218/Unknown  81s 193ms/step - categorical_accuracy: 0.5752 - loss: 1.7204 - mean_io_u: 0.0368


219/Unknown  82s 192ms/step - categorical_accuracy: 0.5757 - loss: 1.7187 - mean_io_u: 0.0369


220/Unknown  82s 192ms/step - categorical_accuracy: 0.5761 - loss: 1.7169 - mean_io_u: 0.0369


221/Unknown  82s 191ms/step - categorical_accuracy: 0.5766 - loss: 1.7152 - mean_io_u: 0.0370


222/Unknown  82s 191ms/step - categorical_accuracy: 0.5770 - loss: 1.7135 - mean_io_u: 0.0370


223/Unknown  82s 190ms/step - categorical_accuracy: 0.5775 - loss: 1.7119 - mean_io_u: 0.0371


224/Unknown  82s 190ms/step - categorical_accuracy: 0.5779 - loss: 1.7102 - mean_io_u: 0.0371


225/Unknown  82s 189ms/step - categorical_accuracy: 0.5783 - loss: 1.7085 - mean_io_u: 0.0372


226/Unknown  82s 189ms/step - categorical_accuracy: 0.5788 - loss: 1.7069 - mean_io_u: 0.0372


227/Unknown  82s 188ms/step - categorical_accuracy: 0.5792 - loss: 1.7052 - mean_io_u: 0.0373


228/Unknown  82s 188ms/step - categorical_accuracy: 0.5796 - loss: 1.7036 - mean_io_u: 0.0373


229/Unknown  82s 187ms/step - categorical_accuracy: 0.5801 - loss: 1.7019 - mean_io_u: 0.0374


230/Unknown  82s 187ms/step - categorical_accuracy: 0.5805 - loss: 1.7003 - mean_io_u: 0.0374


231/Unknown  83s 186ms/step - categorical_accuracy: 0.5809 - loss: 1.6987 - mean_io_u: 0.0374


232/Unknown  83s 186ms/step - categorical_accuracy: 0.5813 - loss: 1.6971 - mean_io_u: 0.0375


233/Unknown  83s 185ms/step - categorical_accuracy: 0.5817 - loss: 1.6955 - mean_io_u: 0.0375


234/Unknown  83s 185ms/step - categorical_accuracy: 0.5822 - loss: 1.6939 - mean_io_u: 0.0376


235/Unknown  83s 185ms/step - categorical_accuracy: 0.5826 - loss: 1.6923 - mean_io_u: 0.0376


236/Unknown  83s 184ms/step - categorical_accuracy: 0.5830 - loss: 1.6908 - mean_io_u: 0.0377


237/Unknown  83s 184ms/step - categorical_accuracy: 0.5834 - loss: 1.6892 - mean_io_u: 0.0377


238/Unknown  83s 183ms/step - categorical_accuracy: 0.5838 - loss: 1.6876 - mean_io_u: 0.0378


239/Unknown  83s 183ms/step - categorical_accuracy: 0.5842 - loss: 1.6860 - mean_io_u: 0.0378


240/Unknown  83s 182ms/step - categorical_accuracy: 0.5846 - loss: 1.6845 - mean_io_u: 0.0379


241/Unknown  83s 182ms/step - categorical_accuracy: 0.5850 - loss: 1.6829 - mean_io_u: 0.0379


242/Unknown  83s 181ms/step - categorical_accuracy: 0.5854 - loss: 1.6814 - mean_io_u: 0.0380


243/Unknown  83s 181ms/step - categorical_accuracy: 0.5858 - loss: 1.6799 - mean_io_u: 0.0380


244/Unknown  84s 180ms/step - categorical_accuracy: 0.5862 - loss: 1.6783 - mean_io_u: 0.0380


245/Unknown  84s 180ms/step - categorical_accuracy: 0.5866 - loss: 1.6768 - mean_io_u: 0.0381


246/Unknown  84s 180ms/step - categorical_accuracy: 0.5870 - loss: 1.6753 - mean_io_u: 0.0381


247/Unknown  84s 179ms/step - categorical_accuracy: 0.5874 - loss: 1.6737 - mean_io_u: 0.0382


248/Unknown  84s 179ms/step - categorical_accuracy: 0.5878 - loss: 1.6722 - mean_io_u: 0.0382


249/Unknown  84s 179ms/step - categorical_accuracy: 0.5882 - loss: 1.6707 - mean_io_u: 0.0383


250/Unknown  84s 178ms/step - categorical_accuracy: 0.5886 - loss: 1.6693 - mean_io_u: 0.0383


251/Unknown  84s 178ms/step - categorical_accuracy: 0.5889 - loss: 1.6678 - mean_io_u: 0.0384


252/Unknown  84s 178ms/step - categorical_accuracy: 0.5893 - loss: 1.6663 - mean_io_u: 0.0384


253/Unknown  84s 177ms/step - categorical_accuracy: 0.5897 - loss: 1.6649 - mean_io_u: 0.0385


254/Unknown  84s 177ms/step - categorical_accuracy: 0.5901 - loss: 1.6634 - mean_io_u: 0.0385


255/Unknown  84s 176ms/step - categorical_accuracy: 0.5904 - loss: 1.6620 - mean_io_u: 0.0385


256/Unknown  85s 176ms/step - categorical_accuracy: 0.5908 - loss: 1.6606 - mean_io_u: 0.0386


257/Unknown  85s 175ms/step - categorical_accuracy: 0.5912 - loss: 1.6592 - mean_io_u: 0.0386


258/Unknown  85s 175ms/step - categorical_accuracy: 0.5915 - loss: 1.6577 - mean_io_u: 0.0387


259/Unknown  85s 175ms/step - categorical_accuracy: 0.5919 - loss: 1.6563 - mean_io_u: 0.0387


260/Unknown  85s 174ms/step - categorical_accuracy: 0.5922 - loss: 1.6549 - mean_io_u: 0.0387


261/Unknown  85s 174ms/step - categorical_accuracy: 0.5926 - loss: 1.6536 - mean_io_u: 0.0388


262/Unknown  85s 174ms/step - categorical_accuracy: 0.5929 - loss: 1.6522 - mean_io_u: 0.0388


263/Unknown  85s 173ms/step - categorical_accuracy: 0.5933 - loss: 1.6508 - mean_io_u: 0.0389


264/Unknown  85s 173ms/step - categorical_accuracy: 0.5936 - loss: 1.6495 - mean_io_u: 0.0389


265/Unknown  85s 173ms/step - categorical_accuracy: 0.5940 - loss: 1.6481 - mean_io_u: 0.0390


266/Unknown  85s 172ms/step - categorical_accuracy: 0.5943 - loss: 1.6467 - mean_io_u: 0.0390


267/Unknown  85s 172ms/step - categorical_accuracy: 0.5947 - loss: 1.6454 - mean_io_u: 0.0390


268/Unknown  86s 172ms/step - categorical_accuracy: 0.5950 - loss: 1.6440 - mean_io_u: 0.0391


269/Unknown  86s 172ms/step - categorical_accuracy: 0.5953 - loss: 1.6427 - mean_io_u: 0.0391


270/Unknown  86s 171ms/step - categorical_accuracy: 0.5957 - loss: 1.6414 - mean_io_u: 0.0392


271/Unknown  86s 171ms/step - categorical_accuracy: 0.5960 - loss: 1.6400 - mean_io_u: 0.0392


272/Unknown  86s 171ms/step - categorical_accuracy: 0.5964 - loss: 1.6387 - mean_io_u: 0.0393


273/Unknown  86s 170ms/step - categorical_accuracy: 0.5967 - loss: 1.6374 - mean_io_u: 0.0393


274/Unknown  86s 170ms/step - categorical_accuracy: 0.5970 - loss: 1.6361 - mean_io_u: 0.0393


275/Unknown  86s 170ms/step - categorical_accuracy: 0.5973 - loss: 1.6348 - mean_io_u: 0.0394


276/Unknown  86s 169ms/step - categorical_accuracy: 0.5977 - loss: 1.6336 - mean_io_u: 0.0394


277/Unknown  86s 169ms/step - categorical_accuracy: 0.5980 - loss: 1.6323 - mean_io_u: 0.0395


278/Unknown  86s 169ms/step - categorical_accuracy: 0.5983 - loss: 1.6311 - mean_io_u: 0.0395


279/Unknown  86s 168ms/step - categorical_accuracy: 0.5986 - loss: 1.6298 - mean_io_u: 0.0396


280/Unknown  87s 168ms/step - categorical_accuracy: 0.5989 - loss: 1.6286 - mean_io_u: 0.0396


281/Unknown  87s 168ms/step - categorical_accuracy: 0.5992 - loss: 1.6273 - mean_io_u: 0.0396


282/Unknown  87s 168ms/step - categorical_accuracy: 0.5996 - loss: 1.6261 - mean_io_u: 0.0397


283/Unknown  87s 167ms/step - categorical_accuracy: 0.5999 - loss: 1.6249 - mean_io_u: 0.0397


284/Unknown  87s 167ms/step - categorical_accuracy: 0.6002 - loss: 1.6236 - mean_io_u: 0.0398


285/Unknown  87s 167ms/step - categorical_accuracy: 0.6005 - loss: 1.6224 - mean_io_u: 0.0398


286/Unknown  87s 166ms/step - categorical_accuracy: 0.6008 - loss: 1.6212 - mean_io_u: 0.0398


287/Unknown  87s 166ms/step - categorical_accuracy: 0.6011 - loss: 1.6200 - mean_io_u: 0.0399


288/Unknown  87s 166ms/step - categorical_accuracy: 0.6014 - loss: 1.6188 - mean_io_u: 0.0399


289/Unknown  87s 165ms/step - categorical_accuracy: 0.6017 - loss: 1.6176 - mean_io_u: 0.0400


290/Unknown  87s 165ms/step - categorical_accuracy: 0.6020 - loss: 1.6164 - mean_io_u: 0.0400


291/Unknown  87s 165ms/step - categorical_accuracy: 0.6023 - loss: 1.6152 - mean_io_u: 0.0401


292/Unknown  88s 165ms/step - categorical_accuracy: 0.6026 - loss: 1.6140 - mean_io_u: 0.0401


293/Unknown  88s 164ms/step - categorical_accuracy: 0.6029 - loss: 1.6128 - mean_io_u: 0.0401


294/Unknown  88s 164ms/step - categorical_accuracy: 0.6032 - loss: 1.6116 - mean_io_u: 0.0402


295/Unknown  88s 164ms/step - categorical_accuracy: 0.6035 - loss: 1.6104 - mean_io_u: 0.0402


296/Unknown  88s 163ms/step - categorical_accuracy: 0.6038 - loss: 1.6092 - mean_io_u: 0.0403


297/Unknown  88s 163ms/step - categorical_accuracy: 0.6041 - loss: 1.6081 - mean_io_u: 0.0403


298/Unknown  88s 163ms/step - categorical_accuracy: 0.6044 - loss: 1.6069 - mean_io_u: 0.0404


299/Unknown  88s 163ms/step - categorical_accuracy: 0.6047 - loss: 1.6057 - mean_io_u: 0.0404


300/Unknown  88s 162ms/step - categorical_accuracy: 0.6050 - loss: 1.6045 - mean_io_u: 0.0404


301/Unknown  88s 162ms/step - categorical_accuracy: 0.6053 - loss: 1.6034 - mean_io_u: 0.0405


302/Unknown  88s 162ms/step - categorical_accuracy: 0.6056 - loss: 1.6022 - mean_io_u: 0.0405


303/Unknown  88s 162ms/step - categorical_accuracy: 0.6058 - loss: 1.6011 - mean_io_u: 0.0406


304/Unknown  89s 161ms/step - categorical_accuracy: 0.6061 - loss: 1.5999 - mean_io_u: 0.0406


305/Unknown  89s 161ms/step - categorical_accuracy: 0.6064 - loss: 1.5988 - mean_io_u: 0.0407


306/Unknown  89s 161ms/step - categorical_accuracy: 0.6067 - loss: 1.5976 - mean_io_u: 0.0407


307/Unknown  89s 161ms/step - categorical_accuracy: 0.6070 - loss: 1.5965 - mean_io_u: 0.0407


308/Unknown  89s 160ms/step - categorical_accuracy: 0.6073 - loss: 1.5954 - mean_io_u: 0.0408


309/Unknown  89s 160ms/step - categorical_accuracy: 0.6075 - loss: 1.5942 - mean_io_u: 0.0408


310/Unknown  89s 160ms/step - categorical_accuracy: 0.6078 - loss: 1.5931 - mean_io_u: 0.0409


311/Unknown  89s 160ms/step - categorical_accuracy: 0.6081 - loss: 1.5920 - mean_io_u: 0.0409


312/Unknown  89s 159ms/step - categorical_accuracy: 0.6084 - loss: 1.5909 - mean_io_u: 0.0409


313/Unknown  89s 159ms/step - categorical_accuracy: 0.6086 - loss: 1.5898 - mean_io_u: 0.0410


314/Unknown  89s 159ms/step - categorical_accuracy: 0.6089 - loss: 1.5887 - mean_io_u: 0.0410


315/Unknown  89s 158ms/step - categorical_accuracy: 0.6092 - loss: 1.5876 - mean_io_u: 0.0411


316/Unknown  89s 158ms/step - categorical_accuracy: 0.6095 - loss: 1.5866 - mean_io_u: 0.0411


317/Unknown  90s 158ms/step - categorical_accuracy: 0.6097 - loss: 1.5855 - mean_io_u: 0.0411


318/Unknown  90s 158ms/step - categorical_accuracy: 0.6100 - loss: 1.5844 - mean_io_u: 0.0412


319/Unknown  90s 157ms/step - categorical_accuracy: 0.6103 - loss: 1.5833 - mean_io_u: 0.0412


320/Unknown  90s 157ms/step - categorical_accuracy: 0.6105 - loss: 1.5822 - mean_io_u: 0.0413


321/Unknown  90s 157ms/step - categorical_accuracy: 0.6108 - loss: 1.5812 - mean_io_u: 0.0413


322/Unknown  90s 157ms/step - categorical_accuracy: 0.6111 - loss: 1.5801 - mean_io_u: 0.0413


323/Unknown  90s 156ms/step - categorical_accuracy: 0.6113 - loss: 1.5790 - mean_io_u: 0.0414


324/Unknown  90s 156ms/step - categorical_accuracy: 0.6116 - loss: 1.5780 - mean_io_u: 0.0414


325/Unknown  90s 156ms/step - categorical_accuracy: 0.6118 - loss: 1.5769 - mean_io_u: 0.0415


326/Unknown  90s 156ms/step - categorical_accuracy: 0.6121 - loss: 1.5759 - mean_io_u: 0.0415


327/Unknown  90s 155ms/step - categorical_accuracy: 0.6123 - loss: 1.5748 - mean_io_u: 0.0416


328/Unknown  90s 155ms/step - categorical_accuracy: 0.6126 - loss: 1.5738 - mean_io_u: 0.0416


329/Unknown  90s 155ms/step - categorical_accuracy: 0.6129 - loss: 1.5728 - mean_io_u: 0.0416


330/Unknown  90s 155ms/step - categorical_accuracy: 0.6131 - loss: 1.5718 - mean_io_u: 0.0417


331/Unknown  91s 154ms/step - categorical_accuracy: 0.6134 - loss: 1.5707 - mean_io_u: 0.0417


332/Unknown  91s 154ms/step - categorical_accuracy: 0.6136 - loss: 1.5697 - mean_io_u: 0.0418


333/Unknown  91s 154ms/step - categorical_accuracy: 0.6139 - loss: 1.5687 - mean_io_u: 0.0418


334/Unknown  91s 153ms/step - categorical_accuracy: 0.6141 - loss: 1.5677 - mean_io_u: 0.0418


335/Unknown  91s 153ms/step - categorical_accuracy: 0.6144 - loss: 1.5667 - mean_io_u: 0.0419


336/Unknown  91s 153ms/step - categorical_accuracy: 0.6146 - loss: 1.5657 - mean_io_u: 0.0419


337/Unknown  91s 153ms/step - categorical_accuracy: 0.6148 - loss: 1.5647 - mean_io_u: 0.0420


338/Unknown  91s 153ms/step - categorical_accuracy: 0.6151 - loss: 1.5637 - mean_io_u: 0.0420


339/Unknown  91s 152ms/step - categorical_accuracy: 0.6153 - loss: 1.5627 - mean_io_u: 0.0421


340/Unknown  91s 152ms/step - categorical_accuracy: 0.6156 - loss: 1.5617 - mean_io_u: 0.0421


341/Unknown  91s 152ms/step - categorical_accuracy: 0.6158 - loss: 1.5607 - mean_io_u: 0.0421


342/Unknown  91s 152ms/step - categorical_accuracy: 0.6161 - loss: 1.5597 - mean_io_u: 0.0422


343/Unknown  91s 151ms/step - categorical_accuracy: 0.6163 - loss: 1.5587 - mean_io_u: 0.0422


344/Unknown  91s 151ms/step - categorical_accuracy: 0.6165 - loss: 1.5577 - mean_io_u: 0.0423


345/Unknown  92s 151ms/step - categorical_accuracy: 0.6168 - loss: 1.5568 - mean_io_u: 0.0423


346/Unknown  92s 151ms/step - categorical_accuracy: 0.6170 - loss: 1.5558 - mean_io_u: 0.0423


347/Unknown  92s 150ms/step - categorical_accuracy: 0.6173 - loss: 1.5548 - mean_io_u: 0.0424


348/Unknown  92s 150ms/step - categorical_accuracy: 0.6175 - loss: 1.5539 - mean_io_u: 0.0424


349/Unknown  92s 150ms/step - categorical_accuracy: 0.6177 - loss: 1.5529 - mean_io_u: 0.0425


350/Unknown  92s 150ms/step - categorical_accuracy: 0.6180 - loss: 1.5520 - mean_io_u: 0.0425


351/Unknown  92s 150ms/step - categorical_accuracy: 0.6182 - loss: 1.5511 - mean_io_u: 0.0425


352/Unknown  92s 149ms/step - categorical_accuracy: 0.6184 - loss: 1.5501 - mean_io_u: 0.0426


353/Unknown  92s 149ms/step - categorical_accuracy: 0.6186 - loss: 1.5492 - mean_io_u: 0.0426


354/Unknown  92s 149ms/step - categorical_accuracy: 0.6189 - loss: 1.5483 - mean_io_u: 0.0427


355/Unknown  92s 149ms/step - categorical_accuracy: 0.6191 - loss: 1.5474 - mean_io_u: 0.0427


356/Unknown  92s 149ms/step - categorical_accuracy: 0.6193 - loss: 1.5464 - mean_io_u: 0.0427


357/Unknown  92s 148ms/step - categorical_accuracy: 0.6195 - loss: 1.5455 - mean_io_u: 0.0428


358/Unknown  92s 148ms/step - categorical_accuracy: 0.6197 - loss: 1.5446 - mean_io_u: 0.0428


359/Unknown  93s 148ms/step - categorical_accuracy: 0.6200 - loss: 1.5437 - mean_io_u: 0.0429


360/Unknown  93s 148ms/step - categorical_accuracy: 0.6202 - loss: 1.5428 - mean_io_u: 0.0429


361/Unknown  93s 147ms/step - categorical_accuracy: 0.6204 - loss: 1.5419 - mean_io_u: 0.0429


362/Unknown  93s 147ms/step - categorical_accuracy: 0.6206 - loss: 1.5410 - mean_io_u: 0.0430


363/Unknown  93s 147ms/step - categorical_accuracy: 0.6208 - loss: 1.5402 - mean_io_u: 0.0430


364/Unknown  93s 147ms/step - categorical_accuracy: 0.6211 - loss: 1.5393 - mean_io_u: 0.0430


365/Unknown  93s 146ms/step - categorical_accuracy: 0.6213 - loss: 1.5384 - mean_io_u: 0.0431


366/Unknown  93s 146ms/step - categorical_accuracy: 0.6215 - loss: 1.5375 - mean_io_u: 0.0431


367/Unknown  93s 146ms/step - categorical_accuracy: 0.6217 - loss: 1.5367 - mean_io_u: 0.0432


368/Unknown  93s 146ms/step - categorical_accuracy: 0.6219 - loss: 1.5358 - mean_io_u: 0.0432


369/Unknown  93s 146ms/step - categorical_accuracy: 0.6221 - loss: 1.5349 - mean_io_u: 0.0432


370/Unknown  93s 145ms/step - categorical_accuracy: 0.6223 - loss: 1.5341 - mean_io_u: 0.0433


371/Unknown  93s 145ms/step - categorical_accuracy: 0.6225 - loss: 1.5332 - mean_io_u: 0.0433


372/Unknown  94s 145ms/step - categorical_accuracy: 0.6227 - loss: 1.5324 - mean_io_u: 0.0434


373/Unknown  94s 145ms/step - categorical_accuracy: 0.6229 - loss: 1.5315 - mean_io_u: 0.0434


374/Unknown  94s 145ms/step - categorical_accuracy: 0.6232 - loss: 1.5307 - mean_io_u: 0.0434


375/Unknown  94s 145ms/step - categorical_accuracy: 0.6234 - loss: 1.5298 - mean_io_u: 0.0435


376/Unknown  94s 144ms/step - categorical_accuracy: 0.6236 - loss: 1.5290 - mean_io_u: 0.0435


377/Unknown  94s 144ms/step - categorical_accuracy: 0.6238 - loss: 1.5281 - mean_io_u: 0.0436


378/Unknown  94s 144ms/step - categorical_accuracy: 0.6240 - loss: 1.5273 - mean_io_u: 0.0436


379/Unknown  94s 144ms/step - categorical_accuracy: 0.6242 - loss: 1.5264 - mean_io_u: 0.0436


380/Unknown  94s 144ms/step - categorical_accuracy: 0.6244 - loss: 1.5256 - mean_io_u: 0.0437


381/Unknown  94s 144ms/step - categorical_accuracy: 0.6246 - loss: 1.5248 - mean_io_u: 0.0437


382/Unknown  94s 143ms/step - categorical_accuracy: 0.6248 - loss: 1.5240 - mean_io_u: 0.0438


383/Unknown  94s 143ms/step - categorical_accuracy: 0.6250 - loss: 1.5232 - mean_io_u: 0.0438


384/Unknown  94s 143ms/step - categorical_accuracy: 0.6252 - loss: 1.5223 - mean_io_u: 0.0438


385/Unknown  95s 143ms/step - categorical_accuracy: 0.6254 - loss: 1.5215 - mean_io_u: 0.0439


386/Unknown  95s 143ms/step - categorical_accuracy: 0.6256 - loss: 1.5207 - mean_io_u: 0.0439


387/Unknown  95s 143ms/step - categorical_accuracy: 0.6258 - loss: 1.5199 - mean_io_u: 0.0440


388/Unknown  95s 142ms/step - categorical_accuracy: 0.6259 - loss: 1.5191 - mean_io_u: 0.0440


389/Unknown  95s 142ms/step - categorical_accuracy: 0.6261 - loss: 1.5183 - mean_io_u: 0.0440


390/Unknown  95s 142ms/step - categorical_accuracy: 0.6263 - loss: 1.5175 - mean_io_u: 0.0441


391/Unknown  95s 142ms/step - categorical_accuracy: 0.6265 - loss: 1.5167 - mean_io_u: 0.0441


392/Unknown  95s 142ms/step - categorical_accuracy: 0.6267 - loss: 1.5159 - mean_io_u: 0.0442


393/Unknown  95s 141ms/step - categorical_accuracy: 0.6269 - loss: 1.5151 - mean_io_u: 0.0442


394/Unknown  95s 141ms/step - categorical_accuracy: 0.6271 - loss: 1.5143 - mean_io_u: 0.0442


395/Unknown  95s 141ms/step - categorical_accuracy: 0.6273 - loss: 1.5135 - mean_io_u: 0.0443


396/Unknown  95s 141ms/step - categorical_accuracy: 0.6275 - loss: 1.5128 - mean_io_u: 0.0443


397/Unknown  95s 141ms/step - categorical_accuracy: 0.6277 - loss: 1.5120 - mean_io_u: 0.0444


398/Unknown  95s 141ms/step - categorical_accuracy: 0.6279 - loss: 1.5112 - mean_io_u: 0.0444


399/Unknown  96s 140ms/step - categorical_accuracy: 0.6280 - loss: 1.5104 - mean_io_u: 0.0444


400/Unknown  96s 140ms/step - categorical_accuracy: 0.6282 - loss: 1.5096 - mean_io_u: 0.0445


401/Unknown  96s 140ms/step - categorical_accuracy: 0.6284 - loss: 1.5089 - mean_io_u: 0.0445


402/Unknown  96s 140ms/step - categorical_accuracy: 0.6286 - loss: 1.5081 - mean_io_u: 0.0446


403/Unknown  96s 140ms/step - categorical_accuracy: 0.6288 - loss: 1.5073 - mean_io_u: 0.0446


404/Unknown  96s 140ms/step - categorical_accuracy: 0.6290 - loss: 1.5065 - mean_io_u: 0.0446


405/Unknown  96s 140ms/step - categorical_accuracy: 0.6292 - loss: 1.5058 - mean_io_u: 0.0447


406/Unknown  96s 139ms/step - categorical_accuracy: 0.6293 - loss: 1.5050 - mean_io_u: 0.0447


407/Unknown  96s 139ms/step - categorical_accuracy: 0.6295 - loss: 1.5042 - mean_io_u: 0.0448


408/Unknown  96s 139ms/step - categorical_accuracy: 0.6297 - loss: 1.5035 - mean_io_u: 0.0448


409/Unknown  96s 139ms/step - categorical_accuracy: 0.6299 - loss: 1.5027 - mean_io_u: 0.0448


410/Unknown  96s 139ms/step - categorical_accuracy: 0.6301 - loss: 1.5020 - mean_io_u: 0.0449


411/Unknown  96s 139ms/step - categorical_accuracy: 0.6303 - loss: 1.5012 - mean_io_u: 0.0449


412/Unknown  97s 138ms/step - categorical_accuracy: 0.6304 - loss: 1.5004 - mean_io_u: 0.0450


413/Unknown  97s 138ms/step - categorical_accuracy: 0.6306 - loss: 1.4997 - mean_io_u: 0.0450


414/Unknown  97s 138ms/step - categorical_accuracy: 0.6308 - loss: 1.4989 - mean_io_u: 0.0450


415/Unknown  97s 138ms/step - categorical_accuracy: 0.6310 - loss: 1.4982 - mean_io_u: 0.0451


416/Unknown  97s 138ms/step - categorical_accuracy: 0.6312 - loss: 1.4974 - mean_io_u: 0.0451


417/Unknown  97s 138ms/step - categorical_accuracy: 0.6313 - loss: 1.4967 - mean_io_u: 0.0452


418/Unknown  97s 138ms/step - categorical_accuracy: 0.6315 - loss: 1.4960 - mean_io_u: 0.0452


419/Unknown  97s 137ms/step - categorical_accuracy: 0.6317 - loss: 1.4952 - mean_io_u: 0.0452


420/Unknown  97s 137ms/step - categorical_accuracy: 0.6319 - loss: 1.4945 - mean_io_u: 0.0453


421/Unknown  97s 137ms/step - categorical_accuracy: 0.6320 - loss: 1.4938 - mean_io_u: 0.0453


422/Unknown  97s 137ms/step - categorical_accuracy: 0.6322 - loss: 1.4931 - mean_io_u: 0.0453


423/Unknown  97s 137ms/step - categorical_accuracy: 0.6324 - loss: 1.4923 - mean_io_u: 0.0454


424/Unknown  97s 137ms/step - categorical_accuracy: 0.6325 - loss: 1.4916 - mean_io_u: 0.0454


425/Unknown  98s 137ms/step - categorical_accuracy: 0.6327 - loss: 1.4909 - mean_io_u: 0.0455


426/Unknown  98s 136ms/step - categorical_accuracy: 0.6329 - loss: 1.4902 - mean_io_u: 0.0455


427/Unknown  98s 136ms/step - categorical_accuracy: 0.6331 - loss: 1.4895 - mean_io_u: 0.0455


428/Unknown  98s 136ms/step - categorical_accuracy: 0.6332 - loss: 1.4888 - mean_io_u: 0.0456


429/Unknown  98s 136ms/step - categorical_accuracy: 0.6334 - loss: 1.4881 - mean_io_u: 0.0456


430/Unknown  98s 136ms/step - categorical_accuracy: 0.6336 - loss: 1.4874 - mean_io_u: 0.0457


431/Unknown  98s 136ms/step - categorical_accuracy: 0.6337 - loss: 1.4867 - mean_io_u: 0.0457


432/Unknown  98s 135ms/step - categorical_accuracy: 0.6339 - loss: 1.4859 - mean_io_u: 0.0457


433/Unknown  98s 135ms/step - categorical_accuracy: 0.6341 - loss: 1.4852 - mean_io_u: 0.0458


434/Unknown  98s 135ms/step - categorical_accuracy: 0.6342 - loss: 1.4845 - mean_io_u: 0.0458


435/Unknown  98s 135ms/step - categorical_accuracy: 0.6344 - loss: 1.4838 - mean_io_u: 0.0458


436/Unknown  98s 135ms/step - categorical_accuracy: 0.6346 - loss: 1.4831 - mean_io_u: 0.0459


437/Unknown  98s 135ms/step - categorical_accuracy: 0.6347 - loss: 1.4824 - mean_io_u: 0.0459


438/Unknown  99s 135ms/step - categorical_accuracy: 0.6349 - loss: 1.4818 - mean_io_u: 0.0460


439/Unknown  99s 135ms/step - categorical_accuracy: 0.6351 - loss: 1.4811 - mean_io_u: 0.0460


440/Unknown  99s 134ms/step - categorical_accuracy: 0.6352 - loss: 1.4804 - mean_io_u: 0.0460


441/Unknown  99s 134ms/step - categorical_accuracy: 0.6354 - loss: 1.4797 - mean_io_u: 0.0461


442/Unknown  99s 134ms/step - categorical_accuracy: 0.6356 - loss: 1.4790 - mean_io_u: 0.0461


443/Unknown  99s 134ms/step - categorical_accuracy: 0.6357 - loss: 1.4783 - mean_io_u: 0.0461


444/Unknown  99s 134ms/step - categorical_accuracy: 0.6359 - loss: 1.4776 - mean_io_u: 0.0462


445/Unknown  99s 134ms/step - categorical_accuracy: 0.6360 - loss: 1.4769 - mean_io_u: 0.0462


446/Unknown  99s 134ms/step - categorical_accuracy: 0.6362 - loss: 1.4763 - mean_io_u: 0.0463


447/Unknown  99s 134ms/step - categorical_accuracy: 0.6364 - loss: 1.4756 - mean_io_u: 0.0463


448/Unknown  99s 133ms/step - categorical_accuracy: 0.6365 - loss: 1.4749 - mean_io_u: 0.0463


449/Unknown  99s 133ms/step - categorical_accuracy: 0.6367 - loss: 1.4742 - mean_io_u: 0.0464


450/Unknown  99s 133ms/step - categorical_accuracy: 0.6369 - loss: 1.4735 - mean_io_u: 0.0464


451/Unknown  100s 133ms/step - categorical_accuracy: 0.6370 - loss: 1.4729 - mean_io_u: 0.0464


452/Unknown  100s 133ms/step - categorical_accuracy: 0.6372 - loss: 1.4722 - mean_io_u: 0.0465


453/Unknown  100s 133ms/step - categorical_accuracy: 0.6373 - loss: 1.4715 - mean_io_u: 0.0465


454/Unknown  100s 133ms/step - categorical_accuracy: 0.6375 - loss: 1.4708 - mean_io_u: 0.0466


455/Unknown  100s 133ms/step - categorical_accuracy: 0.6376 - loss: 1.4702 - mean_io_u: 0.0466


456/Unknown  100s 132ms/step - categorical_accuracy: 0.6378 - loss: 1.4695 - mean_io_u: 0.0466


457/Unknown  100s 132ms/step - categorical_accuracy: 0.6380 - loss: 1.4689 - mean_io_u: 0.0467


458/Unknown  100s 132ms/step - categorical_accuracy: 0.6381 - loss: 1.4682 - mean_io_u: 0.0467


459/Unknown  100s 132ms/step - categorical_accuracy: 0.6383 - loss: 1.4676 - mean_io_u: 0.0467


460/Unknown  100s 132ms/step - categorical_accuracy: 0.6384 - loss: 1.4669 - mean_io_u: 0.0468


461/Unknown  100s 132ms/step - categorical_accuracy: 0.6386 - loss: 1.4663 - mean_io_u: 0.0468


462/Unknown  100s 132ms/step - categorical_accuracy: 0.6387 - loss: 1.4656 - mean_io_u: 0.0469


463/Unknown  100s 132ms/step - categorical_accuracy: 0.6389 - loss: 1.4650 - mean_io_u: 0.0469


464/Unknown  101s 131ms/step - categorical_accuracy: 0.6390 - loss: 1.4643 - mean_io_u: 0.0469


465/Unknown  101s 131ms/step - categorical_accuracy: 0.6392 - loss: 1.4637 - mean_io_u: 0.0470


466/Unknown  101s 131ms/step - categorical_accuracy: 0.6393 - loss: 1.4631 - mean_io_u: 0.0470


467/Unknown  101s 131ms/step - categorical_accuracy: 0.6395 - loss: 1.4624 - mean_io_u: 0.0470


468/Unknown  101s 131ms/step - categorical_accuracy: 0.6396 - loss: 1.4618 - mean_io_u: 0.0471


469/Unknown  101s 131ms/step - categorical_accuracy: 0.6398 - loss: 1.4611 - mean_io_u: 0.0471


470/Unknown  101s 131ms/step - categorical_accuracy: 0.6399 - loss: 1.4605 - mean_io_u: 0.0471


471/Unknown  101s 131ms/step - categorical_accuracy: 0.6401 - loss: 1.4599 - mean_io_u: 0.0472


472/Unknown  101s 131ms/step - categorical_accuracy: 0.6402 - loss: 1.4593 - mean_io_u: 0.0472


473/Unknown  101s 130ms/step - categorical_accuracy: 0.6404 - loss: 1.4586 - mean_io_u: 0.0473


474/Unknown  101s 130ms/step - categorical_accuracy: 0.6405 - loss: 1.4580 - mean_io_u: 0.0473


475/Unknown  101s 130ms/step - categorical_accuracy: 0.6407 - loss: 1.4574 - mean_io_u: 0.0473


476/Unknown  101s 130ms/step - categorical_accuracy: 0.6408 - loss: 1.4567 - mean_io_u: 0.0474


477/Unknown  102s 130ms/step - categorical_accuracy: 0.6410 - loss: 1.4561 - mean_io_u: 0.0474


478/Unknown  102s 130ms/step - categorical_accuracy: 0.6411 - loss: 1.4555 - mean_io_u: 0.0474


479/Unknown  102s 130ms/step - categorical_accuracy: 0.6413 - loss: 1.4549 - mean_io_u: 0.0475


480/Unknown  102s 130ms/step - categorical_accuracy: 0.6414 - loss: 1.4543 - mean_io_u: 0.0475


481/Unknown  102s 130ms/step - categorical_accuracy: 0.6416 - loss: 1.4536 - mean_io_u: 0.0476


482/Unknown  102s 130ms/step - categorical_accuracy: 0.6417 - loss: 1.4530 - mean_io_u: 0.0476


483/Unknown  102s 130ms/step - categorical_accuracy: 0.6418 - loss: 1.4524 - mean_io_u: 0.0476


484/Unknown  102s 129ms/step - categorical_accuracy: 0.6420 - loss: 1.4518 - mean_io_u: 0.0477


485/Unknown  102s 129ms/step - categorical_accuracy: 0.6421 - loss: 1.4512 - mean_io_u: 0.0477


486/Unknown  102s 129ms/step - categorical_accuracy: 0.6423 - loss: 1.4506 - mean_io_u: 0.0478


487/Unknown  102s 129ms/step - categorical_accuracy: 0.6424 - loss: 1.4499 - mean_io_u: 0.0478


488/Unknown  103s 129ms/step - categorical_accuracy: 0.6426 - loss: 1.4493 - mean_io_u: 0.0478


489/Unknown  103s 129ms/step - categorical_accuracy: 0.6427 - loss: 1.4487 - mean_io_u: 0.0479


490/Unknown  103s 129ms/step - categorical_accuracy: 0.6429 - loss: 1.4481 - mean_io_u: 0.0479


491/Unknown  103s 129ms/step - categorical_accuracy: 0.6430 - loss: 1.4475 - mean_io_u: 0.0479


492/Unknown  103s 129ms/step - categorical_accuracy: 0.6431 - loss: 1.4469 - mean_io_u: 0.0480


493/Unknown  103s 128ms/step - categorical_accuracy: 0.6433 - loss: 1.4463 - mean_io_u: 0.0480


494/Unknown  103s 128ms/step - categorical_accuracy: 0.6434 - loss: 1.4457 - mean_io_u: 0.0481


495/Unknown  103s 128ms/step - categorical_accuracy: 0.6436 - loss: 1.4451 - mean_io_u: 0.0481


496/Unknown  103s 128ms/step - categorical_accuracy: 0.6437 - loss: 1.4445 - mean_io_u: 0.0481


497/Unknown  103s 128ms/step - categorical_accuracy: 0.6438 - loss: 1.4439 - mean_io_u: 0.0482


498/Unknown  103s 128ms/step - categorical_accuracy: 0.6440 - loss: 1.4433 - mean_io_u: 0.0482


499/Unknown  103s 128ms/step - categorical_accuracy: 0.6441 - loss: 1.4427 - mean_io_u: 0.0483


500/Unknown  103s 128ms/step - categorical_accuracy: 0.6443 - loss: 1.4421 - mean_io_u: 0.0483


501/Unknown  103s 128ms/step - categorical_accuracy: 0.6444 - loss: 1.4415 - mean_io_u: 0.0483


502/Unknown  103s 127ms/step - categorical_accuracy: 0.6445 - loss: 1.4409 - mean_io_u: 0.0484


503/Unknown  104s 127ms/step - categorical_accuracy: 0.6447 - loss: 1.4403 - mean_io_u: 0.0484


504/Unknown  104s 127ms/step - categorical_accuracy: 0.6448 - loss: 1.4397 - mean_io_u: 0.0484


505/Unknown  104s 127ms/step - categorical_accuracy: 0.6449 - loss: 1.4391 - mean_io_u: 0.0485


506/Unknown  104s 127ms/step - categorical_accuracy: 0.6451 - loss: 1.4386 - mean_io_u: 0.0485


507/Unknown  104s 127ms/step - categorical_accuracy: 0.6452 - loss: 1.4380 - mean_io_u: 0.0486


508/Unknown  104s 127ms/step - categorical_accuracy: 0.6454 - loss: 1.4374 - mean_io_u: 0.0486


509/Unknown  104s 127ms/step - categorical_accuracy: 0.6455 - loss: 1.4368 - mean_io_u: 0.0486


510/Unknown  104s 127ms/step - categorical_accuracy: 0.6456 - loss: 1.4362 - mean_io_u: 0.0487


511/Unknown  104s 126ms/step - categorical_accuracy: 0.6458 - loss: 1.4357 - mean_io_u: 0.0487


512/Unknown  104s 126ms/step - categorical_accuracy: 0.6459 - loss: 1.4351 - mean_io_u: 0.0488


513/Unknown  104s 126ms/step - categorical_accuracy: 0.6460 - loss: 1.4345 - mean_io_u: 0.0488


514/Unknown  104s 126ms/step - categorical_accuracy: 0.6462 - loss: 1.4340 - mean_io_u: 0.0488


515/Unknown  104s 126ms/step - categorical_accuracy: 0.6463 - loss: 1.4334 - mean_io_u: 0.0489


516/Unknown  104s 126ms/step - categorical_accuracy: 0.6464 - loss: 1.4328 - mean_io_u: 0.0489


517/Unknown  104s 126ms/step - categorical_accuracy: 0.6466 - loss: 1.4323 - mean_io_u: 0.0489


518/Unknown  105s 126ms/step - categorical_accuracy: 0.6467 - loss: 1.4317 - mean_io_u: 0.0490


519/Unknown  105s 126ms/step - categorical_accuracy: 0.6468 - loss: 1.4312 - mean_io_u: 0.0490


520/Unknown  105s 125ms/step - categorical_accuracy: 0.6469 - loss: 1.4306 - mean_io_u: 0.0490


521/Unknown  105s 125ms/step - categorical_accuracy: 0.6471 - loss: 1.4300 - mean_io_u: 0.0491


522/Unknown  105s 125ms/step - categorical_accuracy: 0.6472 - loss: 1.4295 - mean_io_u: 0.0491


523/Unknown  105s 125ms/step - categorical_accuracy: 0.6473 - loss: 1.4289 - mean_io_u: 0.0492


524/Unknown  105s 125ms/step - categorical_accuracy: 0.6475 - loss: 1.4284 - mean_io_u: 0.0492


525/Unknown  105s 125ms/step - categorical_accuracy: 0.6476 - loss: 1.4278 - mean_io_u: 0.0492


526/Unknown  105s 125ms/step - categorical_accuracy: 0.6477 - loss: 1.4273 - mean_io_u: 0.0493


527/Unknown  105s 125ms/step - categorical_accuracy: 0.6478 - loss: 1.4267 - mean_io_u: 0.0493


528/Unknown  105s 125ms/step - categorical_accuracy: 0.6480 - loss: 1.4262 - mean_io_u: 0.0493


529/Unknown  105s 125ms/step - categorical_accuracy: 0.6481 - loss: 1.4256 - mean_io_u: 0.0494


530/Unknown  106s 124ms/step - categorical_accuracy: 0.6482 - loss: 1.4251 - mean_io_u: 0.0494


531/Unknown  106s 124ms/step - categorical_accuracy: 0.6484 - loss: 1.4245 - mean_io_u: 0.0494


532/Unknown  106s 124ms/step - categorical_accuracy: 0.6485 - loss: 1.4240 - mean_io_u: 0.0495


533/Unknown  106s 124ms/step - categorical_accuracy: 0.6486 - loss: 1.4234 - mean_io_u: 0.0495


534/Unknown  106s 124ms/step - categorical_accuracy: 0.6487 - loss: 1.4229 - mean_io_u: 0.0496


535/Unknown  106s 124ms/step - categorical_accuracy: 0.6489 - loss: 1.4224 - mean_io_u: 0.0496


536/Unknown  106s 124ms/step - categorical_accuracy: 0.6490 - loss: 1.4218 - mean_io_u: 0.0496


537/Unknown  106s 124ms/step - categorical_accuracy: 0.6491 - loss: 1.4213 - mean_io_u: 0.0497


538/Unknown  106s 124ms/step - categorical_accuracy: 0.6492 - loss: 1.4207 - mean_io_u: 0.0497


539/Unknown  106s 124ms/step - categorical_accuracy: 0.6494 - loss: 1.4202 - mean_io_u: 0.0497


540/Unknown  106s 124ms/step - categorical_accuracy: 0.6495 - loss: 1.4197 - mean_io_u: 0.0498


541/Unknown  106s 124ms/step - categorical_accuracy: 0.6496 - loss: 1.4191 - mean_io_u: 0.0498


542/Unknown  106s 123ms/step - categorical_accuracy: 0.6497 - loss: 1.4186 - mean_io_u: 0.0498


543/Unknown  107s 123ms/step - categorical_accuracy: 0.6499 - loss: 1.4181 - mean_io_u: 0.0499


544/Unknown  107s 123ms/step - categorical_accuracy: 0.6500 - loss: 1.4175 - mean_io_u: 0.0499


545/Unknown  107s 123ms/step - categorical_accuracy: 0.6501 - loss: 1.4170 - mean_io_u: 0.0500


546/Unknown  107s 123ms/step - categorical_accuracy: 0.6502 - loss: 1.4165 - mean_io_u: 0.0500


547/Unknown  107s 123ms/step - categorical_accuracy: 0.6504 - loss: 1.4159 - mean_io_u: 0.0500


548/Unknown  107s 123ms/step - categorical_accuracy: 0.6505 - loss: 1.4154 - mean_io_u: 0.0501


549/Unknown  107s 123ms/step - categorical_accuracy: 0.6506 - loss: 1.4149 - mean_io_u: 0.0501


550/Unknown  107s 123ms/step - categorical_accuracy: 0.6507 - loss: 1.4144 - mean_io_u: 0.0501


551/Unknown  107s 123ms/step - categorical_accuracy: 0.6508 - loss: 1.4138 - mean_io_u: 0.0502


552/Unknown  107s 123ms/step - categorical_accuracy: 0.6510 - loss: 1.4133 - mean_io_u: 0.0502


553/Unknown  107s 123ms/step - categorical_accuracy: 0.6511 - loss: 1.4128 - mean_io_u: 0.0502


554/Unknown  108s 123ms/step - categorical_accuracy: 0.6512 - loss: 1.4123 - mean_io_u: 0.0503


555/Unknown  108s 123ms/step - categorical_accuracy: 0.6513 - loss: 1.4118 - mean_io_u: 0.0503


556/Unknown  108s 123ms/step - categorical_accuracy: 0.6515 - loss: 1.4112 - mean_io_u: 0.0504


557/Unknown  108s 122ms/step - categorical_accuracy: 0.6516 - loss: 1.4107 - mean_io_u: 0.0504


558/Unknown  108s 122ms/step - categorical_accuracy: 0.6517 - loss: 1.4102 - mean_io_u: 0.0504


559/Unknown  108s 122ms/step - categorical_accuracy: 0.6518 - loss: 1.4097 - mean_io_u: 0.0505


560/Unknown  108s 122ms/step - categorical_accuracy: 0.6519 - loss: 1.4092 - mean_io_u: 0.0505


561/Unknown  108s 122ms/step - categorical_accuracy: 0.6520 - loss: 1.4087 - mean_io_u: 0.0505


562/Unknown  108s 122ms/step - categorical_accuracy: 0.6522 - loss: 1.4082 - mean_io_u: 0.0506


563/Unknown  108s 122ms/step - categorical_accuracy: 0.6523 - loss: 1.4077 - mean_io_u: 0.0506


564/Unknown  108s 122ms/step - categorical_accuracy: 0.6524 - loss: 1.4072 - mean_io_u: 0.0506


565/Unknown  108s 122ms/step - categorical_accuracy: 0.6525 - loss: 1.4066 - mean_io_u: 0.0507


566/Unknown  108s 122ms/step - categorical_accuracy: 0.6526 - loss: 1.4061 - mean_io_u: 0.0507


567/Unknown  108s 122ms/step - categorical_accuracy: 0.6528 - loss: 1.4056 - mean_io_u: 0.0508


568/Unknown  109s 122ms/step - categorical_accuracy: 0.6529 - loss: 1.4051 - mean_io_u: 0.0508


569/Unknown  109s 122ms/step - categorical_accuracy: 0.6530 - loss: 1.4046 - mean_io_u: 0.0508


570/Unknown  109s 121ms/step - categorical_accuracy: 0.6531 - loss: 1.4041 - mean_io_u: 0.0509


571/Unknown  109s 121ms/step - categorical_accuracy: 0.6532 - loss: 1.4036 - mean_io_u: 0.0509


572/Unknown  109s 121ms/step - categorical_accuracy: 0.6533 - loss: 1.4031 - mean_io_u: 0.0509


573/Unknown  109s 121ms/step - categorical_accuracy: 0.6535 - loss: 1.4026 - mean_io_u: 0.0510


574/Unknown  109s 121ms/step - categorical_accuracy: 0.6536 - loss: 1.4021 - mean_io_u: 0.0510


575/Unknown  109s 121ms/step - categorical_accuracy: 0.6537 - loss: 1.4016 - mean_io_u: 0.0511


576/Unknown  109s 121ms/step - categorical_accuracy: 0.6538 - loss: 1.4012 - mean_io_u: 0.0511


577/Unknown  109s 121ms/step - categorical_accuracy: 0.6539 - loss: 1.4007 - mean_io_u: 0.0511


578/Unknown  109s 121ms/step - categorical_accuracy: 0.6540 - loss: 1.4002 - mean_io_u: 0.0512


579/Unknown  109s 121ms/step - categorical_accuracy: 0.6541 - loss: 1.3997 - mean_io_u: 0.0512


580/Unknown  110s 121ms/step - categorical_accuracy: 0.6542 - loss: 1.3992 - mean_io_u: 0.0512


581/Unknown  110s 121ms/step - categorical_accuracy: 0.6544 - loss: 1.3987 - mean_io_u: 0.0513


582/Unknown  110s 121ms/step - categorical_accuracy: 0.6545 - loss: 1.3982 - mean_io_u: 0.0513


583/Unknown  110s 121ms/step - categorical_accuracy: 0.6546 - loss: 1.3977 - mean_io_u: 0.0514


584/Unknown  110s 121ms/step - categorical_accuracy: 0.6547 - loss: 1.3973 - mean_io_u: 0.0514


585/Unknown  110s 121ms/step - categorical_accuracy: 0.6548 - loss: 1.3968 - mean_io_u: 0.0514


586/Unknown  110s 120ms/step - categorical_accuracy: 0.6549 - loss: 1.3963 - mean_io_u: 0.0515


587/Unknown  110s 120ms/step - categorical_accuracy: 0.6550 - loss: 1.3958 - mean_io_u: 0.0515


588/Unknown  110s 120ms/step - categorical_accuracy: 0.6551 - loss: 1.3953 - mean_io_u: 0.0515


589/Unknown  110s 120ms/step - categorical_accuracy: 0.6553 - loss: 1.3949 - mean_io_u: 0.0516


590/Unknown  110s 120ms/step - categorical_accuracy: 0.6554 - loss: 1.3944 - mean_io_u: 0.0516


591/Unknown  111s 120ms/step - categorical_accuracy: 0.6555 - loss: 1.3939 - mean_io_u: 0.0516


592/Unknown  111s 120ms/step - categorical_accuracy: 0.6556 - loss: 1.3934 - mean_io_u: 0.0517


593/Unknown  111s 120ms/step - categorical_accuracy: 0.6557 - loss: 1.3930 - mean_io_u: 0.0517


594/Unknown  111s 120ms/step - categorical_accuracy: 0.6558 - loss: 1.3925 - mean_io_u: 0.0518


595/Unknown  111s 120ms/step - categorical_accuracy: 0.6559 - loss: 1.3920 - mean_io_u: 0.0518


596/Unknown  111s 120ms/step - categorical_accuracy: 0.6560 - loss: 1.3915 - mean_io_u: 0.0518


597/Unknown  111s 120ms/step - categorical_accuracy: 0.6561 - loss: 1.3911 - mean_io_u: 0.0519


598/Unknown  111s 120ms/step - categorical_accuracy: 0.6562 - loss: 1.3906 - mean_io_u: 0.0519


599/Unknown  111s 120ms/step - categorical_accuracy: 0.6563 - loss: 1.3901 - mean_io_u: 0.0519


600/Unknown  111s 120ms/step - categorical_accuracy: 0.6564 - loss: 1.3897 - mean_io_u: 0.0520


601/Unknown  111s 120ms/step - categorical_accuracy: 0.6566 - loss: 1.3892 - mean_io_u: 0.0520


602/Unknown  111s 119ms/step - categorical_accuracy: 0.6567 - loss: 1.3887 - mean_io_u: 0.0521


603/Unknown  112s 119ms/step - categorical_accuracy: 0.6568 - loss: 1.3883 - mean_io_u: 0.0521


604/Unknown  112s 119ms/step - categorical_accuracy: 0.6569 - loss: 1.3878 - mean_io_u: 0.0521


605/Unknown  112s 119ms/step - categorical_accuracy: 0.6570 - loss: 1.3873 - mean_io_u: 0.0522


606/Unknown  112s 119ms/step - categorical_accuracy: 0.6571 - loss: 1.3869 - mean_io_u: 0.0522


607/Unknown  112s 119ms/step - categorical_accuracy: 0.6572 - loss: 1.3864 - mean_io_u: 0.0522


608/Unknown  112s 119ms/step - categorical_accuracy: 0.6573 - loss: 1.3859 - mean_io_u: 0.0523


609/Unknown  112s 119ms/step - categorical_accuracy: 0.6574 - loss: 1.3855 - mean_io_u: 0.0523


610/Unknown  112s 119ms/step - categorical_accuracy: 0.6575 - loss: 1.3850 - mean_io_u: 0.0524


611/Unknown  112s 119ms/step - categorical_accuracy: 0.6576 - loss: 1.3846 - mean_io_u: 0.0524


612/Unknown  112s 119ms/step - categorical_accuracy: 0.6577 - loss: 1.3841 - mean_io_u: 0.0524


613/Unknown  112s 119ms/step - categorical_accuracy: 0.6578 - loss: 1.3836 - mean_io_u: 0.0525


614/Unknown  112s 119ms/step - categorical_accuracy: 0.6579 - loss: 1.3832 - mean_io_u: 0.0525


615/Unknown  113s 119ms/step - categorical_accuracy: 0.6580 - loss: 1.3827 - mean_io_u: 0.0525


616/Unknown  113s 119ms/step - categorical_accuracy: 0.6581 - loss: 1.3823 - mean_io_u: 0.0526


617/Unknown  113s 119ms/step - categorical_accuracy: 0.6583 - loss: 1.3818 - mean_io_u: 0.0526


618/Unknown  113s 119ms/step - categorical_accuracy: 0.6584 - loss: 1.3814 - mean_io_u: 0.0527


619/Unknown  113s 118ms/step - categorical_accuracy: 0.6585 - loss: 1.3809 - mean_io_u: 0.0527


620/Unknown  113s 118ms/step - categorical_accuracy: 0.6586 - loss: 1.3805 - mean_io_u: 0.0527


621/Unknown  113s 118ms/step - categorical_accuracy: 0.6587 - loss: 1.3800 - mean_io_u: 0.0528


622/Unknown  113s 118ms/step - categorical_accuracy: 0.6588 - loss: 1.3796 - mean_io_u: 0.0528


623/Unknown  113s 118ms/step - categorical_accuracy: 0.6589 - loss: 1.3791 - mean_io_u: 0.0529


624/Unknown  113s 118ms/step - categorical_accuracy: 0.6590 - loss: 1.3787 - mean_io_u: 0.0529


625/Unknown  113s 118ms/step - categorical_accuracy: 0.6591 - loss: 1.3782 - mean_io_u: 0.0529


626/Unknown  113s 118ms/step - categorical_accuracy: 0.6592 - loss: 1.3778 - mean_io_u: 0.0530


627/Unknown  114s 118ms/step - categorical_accuracy: 0.6593 - loss: 1.3774 - mean_io_u: 0.0530


628/Unknown  114s 118ms/step - categorical_accuracy: 0.6594 - loss: 1.3769 - mean_io_u: 0.0530


629/Unknown  114s 118ms/step - categorical_accuracy: 0.6595 - loss: 1.3765 - mean_io_u: 0.0531


630/Unknown  114s 118ms/step - categorical_accuracy: 0.6596 - loss: 1.3760 - mean_io_u: 0.0531


631/Unknown  114s 118ms/step - categorical_accuracy: 0.6597 - loss: 1.3756 - mean_io_u: 0.0532


632/Unknown  114s 118ms/step - categorical_accuracy: 0.6598 - loss: 1.3751 - mean_io_u: 0.0532


633/Unknown  114s 118ms/step - categorical_accuracy: 0.6599 - loss: 1.3747 - mean_io_u: 0.0532


634/Unknown  114s 118ms/step - categorical_accuracy: 0.6600 - loss: 1.3743 - mean_io_u: 0.0533


635/Unknown  114s 117ms/step - categorical_accuracy: 0.6601 - loss: 1.3738 - mean_io_u: 0.0533


636/Unknown  114s 117ms/step - categorical_accuracy: 0.6602 - loss: 1.3734 - mean_io_u: 0.0533


637/Unknown  114s 117ms/step - categorical_accuracy: 0.6603 - loss: 1.3730 - mean_io_u: 0.0534


638/Unknown  114s 117ms/step - categorical_accuracy: 0.6604 - loss: 1.3725 - mean_io_u: 0.0534


639/Unknown  114s 117ms/step - categorical_accuracy: 0.6605 - loss: 1.3721 - mean_io_u: 0.0535


640/Unknown  115s 117ms/step - categorical_accuracy: 0.6606 - loss: 1.3717 - mean_io_u: 0.0535


641/Unknown  115s 117ms/step - categorical_accuracy: 0.6607 - loss: 1.3712 - mean_io_u: 0.0535


642/Unknown  115s 117ms/step - categorical_accuracy: 0.6608 - loss: 1.3708 - mean_io_u: 0.0536


643/Unknown  115s 117ms/step - categorical_accuracy: 0.6609 - loss: 1.3704 - mean_io_u: 0.0536


644/Unknown  115s 117ms/step - categorical_accuracy: 0.6610 - loss: 1.3699 - mean_io_u: 0.0536


645/Unknown  115s 117ms/step - categorical_accuracy: 0.6611 - loss: 1.3695 - mean_io_u: 0.0537


646/Unknown  115s 117ms/step - categorical_accuracy: 0.6612 - loss: 1.3691 - mean_io_u: 0.0537


647/Unknown  115s 117ms/step - categorical_accuracy: 0.6613 - loss: 1.3686 - mean_io_u: 0.0537


648/Unknown  115s 117ms/step - categorical_accuracy: 0.6614 - loss: 1.3682 - mean_io_u: 0.0538


649/Unknown  115s 117ms/step - categorical_accuracy: 0.6615 - loss: 1.3678 - mean_io_u: 0.0538


650/Unknown  115s 116ms/step - categorical_accuracy: 0.6616 - loss: 1.3674 - mean_io_u: 0.0539


651/Unknown  115s 116ms/step - categorical_accuracy: 0.6617 - loss: 1.3670 - mean_io_u: 0.0539


652/Unknown  115s 116ms/step - categorical_accuracy: 0.6617 - loss: 1.3665 - mean_io_u: 0.0539


653/Unknown  115s 116ms/step - categorical_accuracy: 0.6618 - loss: 1.3661 - mean_io_u: 0.0540


654/Unknown  115s 116ms/step - categorical_accuracy: 0.6619 - loss: 1.3657 - mean_io_u: 0.0540


655/Unknown  116s 116ms/step - categorical_accuracy: 0.6620 - loss: 1.3653 - mean_io_u: 0.0540


656/Unknown  116s 116ms/step - categorical_accuracy: 0.6621 - loss: 1.3648 - mean_io_u: 0.0541


657/Unknown  116s 116ms/step - categorical_accuracy: 0.6622 - loss: 1.3644 - mean_io_u: 0.0541


658/Unknown  116s 116ms/step - categorical_accuracy: 0.6623 - loss: 1.3640 - mean_io_u: 0.0542


659/Unknown  116s 116ms/step - categorical_accuracy: 0.6624 - loss: 1.3636 - mean_io_u: 0.0542


660/Unknown  116s 116ms/step - categorical_accuracy: 0.6625 - loss: 1.3632 - mean_io_u: 0.0542


661/Unknown  116s 116ms/step - categorical_accuracy: 0.6626 - loss: 1.3628 - mean_io_u: 0.0543


662/Unknown  116s 116ms/step - categorical_accuracy: 0.6627 - loss: 1.3623 - mean_io_u: 0.0543


663/Unknown  116s 115ms/step - categorical_accuracy: 0.6628 - loss: 1.3619 - mean_io_u: 0.0544


664/Unknown  116s 115ms/step - categorical_accuracy: 0.6629 - loss: 1.3615 - mean_io_u: 0.0544


665/Unknown  116s 115ms/step - categorical_accuracy: 0.6630 - loss: 1.3611 - mean_io_u: 0.0544


666/Unknown  116s 115ms/step - categorical_accuracy: 0.6631 - loss: 1.3607 - mean_io_u: 0.0545


667/Unknown  116s 115ms/step - categorical_accuracy: 0.6632 - loss: 1.3603 - mean_io_u: 0.0545


668/Unknown  116s 115ms/step - categorical_accuracy: 0.6633 - loss: 1.3599 - mean_io_u: 0.0545


669/Unknown  117s 115ms/step - categorical_accuracy: 0.6634 - loss: 1.3595 - mean_io_u: 0.0546


670/Unknown  117s 115ms/step - categorical_accuracy: 0.6634 - loss: 1.3591 - mean_io_u: 0.0546


671/Unknown  117s 115ms/step - categorical_accuracy: 0.6635 - loss: 1.3587 - mean_io_u: 0.0547


672/Unknown  117s 115ms/step - categorical_accuracy: 0.6636 - loss: 1.3583 - mean_io_u: 0.0547


673/Unknown  117s 115ms/step - categorical_accuracy: 0.6637 - loss: 1.3579 - mean_io_u: 0.0547


674/Unknown  117s 115ms/step - categorical_accuracy: 0.6638 - loss: 1.3575 - mean_io_u: 0.0548


675/Unknown  117s 115ms/step - categorical_accuracy: 0.6639 - loss: 1.3570 - mean_io_u: 0.0548


676/Unknown  117s 115ms/step - categorical_accuracy: 0.6640 - loss: 1.3566 - mean_io_u: 0.0548


677/Unknown  117s 115ms/step - categorical_accuracy: 0.6641 - loss: 1.3562 - mean_io_u: 0.0549


678/Unknown  117s 115ms/step - categorical_accuracy: 0.6642 - loss: 1.3558 - mean_io_u: 0.0549


679/Unknown  117s 115ms/step - categorical_accuracy: 0.6643 - loss: 1.3555 - mean_io_u: 0.0550


680/Unknown  117s 115ms/step - categorical_accuracy: 0.6644 - loss: 1.3551 - mean_io_u: 0.0550


681/Unknown  117s 114ms/step - categorical_accuracy: 0.6644 - loss: 1.3547 - mean_io_u: 0.0550


682/Unknown  118s 114ms/step - categorical_accuracy: 0.6645 - loss: 1.3543 - mean_io_u: 0.0551


683/Unknown  118s 114ms/step - categorical_accuracy: 0.6646 - loss: 1.3539 - mean_io_u: 0.0551


684/Unknown  118s 114ms/step - categorical_accuracy: 0.6647 - loss: 1.3535 - mean_io_u: 0.0551


685/Unknown  118s 114ms/step - categorical_accuracy: 0.6648 - loss: 1.3531 - mean_io_u: 0.0552


686/Unknown  118s 114ms/step - categorical_accuracy: 0.6649 - loss: 1.3527 - mean_io_u: 0.0552


687/Unknown  118s 114ms/step - categorical_accuracy: 0.6650 - loss: 1.3523 - mean_io_u: 0.0553


688/Unknown  118s 114ms/step - categorical_accuracy: 0.6651 - loss: 1.3519 - mean_io_u: 0.0553


689/Unknown  118s 114ms/step - categorical_accuracy: 0.6652 - loss: 1.3515 - mean_io_u: 0.0553


690/Unknown  118s 114ms/step - categorical_accuracy: 0.6652 - loss: 1.3511 - mean_io_u: 0.0554


691/Unknown  118s 114ms/step - categorical_accuracy: 0.6653 - loss: 1.3507 - mean_io_u: 0.0554


692/Unknown  118s 114ms/step - categorical_accuracy: 0.6654 - loss: 1.3504 - mean_io_u: 0.0554


693/Unknown  118s 114ms/step - categorical_accuracy: 0.6655 - loss: 1.3500 - mean_io_u: 0.0555


694/Unknown  118s 114ms/step - categorical_accuracy: 0.6656 - loss: 1.3496 - mean_io_u: 0.0555


695/Unknown  119s 114ms/step - categorical_accuracy: 0.6657 - loss: 1.3492 - mean_io_u: 0.0555


696/Unknown  119s 114ms/step - categorical_accuracy: 0.6658 - loss: 1.3488 - mean_io_u: 0.0556


697/Unknown  119s 114ms/step - categorical_accuracy: 0.6658 - loss: 1.3485 - mean_io_u: 0.0556


698/Unknown  119s 114ms/step - categorical_accuracy: 0.6659 - loss: 1.3481 - mean_io_u: 0.0557


699/Unknown  119s 114ms/step - categorical_accuracy: 0.6660 - loss: 1.3477 - mean_io_u: 0.0557


700/Unknown  119s 113ms/step - categorical_accuracy: 0.6661 - loss: 1.3473 - mean_io_u: 0.0557


701/Unknown  119s 113ms/step - categorical_accuracy: 0.6662 - loss: 1.3469 - mean_io_u: 0.0558


702/Unknown  119s 113ms/step - categorical_accuracy: 0.6663 - loss: 1.3466 - mean_io_u: 0.0558


703/Unknown  119s 113ms/step - categorical_accuracy: 0.6664 - loss: 1.3462 - mean_io_u: 0.0558


704/Unknown  119s 113ms/step - categorical_accuracy: 0.6664 - loss: 1.3458 - mean_io_u: 0.0559


705/Unknown  119s 113ms/step - categorical_accuracy: 0.6665 - loss: 1.3454 - mean_io_u: 0.0559


706/Unknown  119s 113ms/step - categorical_accuracy: 0.6666 - loss: 1.3450 - mean_io_u: 0.0560


707/Unknown  119s 113ms/step - categorical_accuracy: 0.6667 - loss: 1.3447 - mean_io_u: 0.0560


708/Unknown  120s 113ms/step - categorical_accuracy: 0.6668 - loss: 1.3443 - mean_io_u: 0.0560


709/Unknown  120s 113ms/step - categorical_accuracy: 0.6669 - loss: 1.3439 - mean_io_u: 0.0561


710/Unknown  120s 113ms/step - categorical_accuracy: 0.6669 - loss: 1.3436 - mean_io_u: 0.0561


711/Unknown  120s 113ms/step - categorical_accuracy: 0.6670 - loss: 1.3432 - mean_io_u: 0.0561


712/Unknown  120s 113ms/step - categorical_accuracy: 0.6671 - loss: 1.3428 - mean_io_u: 0.0562


713/Unknown  120s 113ms/step - categorical_accuracy: 0.6672 - loss: 1.3424 - mean_io_u: 0.0562


714/Unknown  120s 113ms/step - categorical_accuracy: 0.6673 - loss: 1.3421 - mean_io_u: 0.0563


715/Unknown  120s 113ms/step - categorical_accuracy: 0.6674 - loss: 1.3417 - mean_io_u: 0.0563


716/Unknown  120s 113ms/step - categorical_accuracy: 0.6674 - loss: 1.3413 - mean_io_u: 0.0563


717/Unknown  120s 113ms/step - categorical_accuracy: 0.6675 - loss: 1.3410 - mean_io_u: 0.0564


718/Unknown  120s 113ms/step - categorical_accuracy: 0.6676 - loss: 1.3406 - mean_io_u: 0.0564


719/Unknown  121s 113ms/step - categorical_accuracy: 0.6677 - loss: 1.3403 - mean_io_u: 0.0564


720/Unknown  121s 113ms/step - categorical_accuracy: 0.6678 - loss: 1.3399 - mean_io_u: 0.0565


721/Unknown  121s 113ms/step - categorical_accuracy: 0.6679 - loss: 1.3395 - mean_io_u: 0.0565


722/Unknown  121s 112ms/step - categorical_accuracy: 0.6679 - loss: 1.3392 - mean_io_u: 0.0566


723/Unknown  121s 112ms/step - categorical_accuracy: 0.6680 - loss: 1.3388 - mean_io_u: 0.0566


724/Unknown  121s 112ms/step - categorical_accuracy: 0.6681 - loss: 1.3384 - mean_io_u: 0.0566


725/Unknown  121s 112ms/step - categorical_accuracy: 0.6682 - loss: 1.3381 - mean_io_u: 0.0567


726/Unknown  121s 112ms/step - categorical_accuracy: 0.6683 - loss: 1.3377 - mean_io_u: 0.0567


727/Unknown  121s 112ms/step - categorical_accuracy: 0.6683 - loss: 1.3374 - mean_io_u: 0.0567


728/Unknown  121s 112ms/step - categorical_accuracy: 0.6684 - loss: 1.3370 - mean_io_u: 0.0568


729/Unknown  121s 112ms/step - categorical_accuracy: 0.6685 - loss: 1.3367 - mean_io_u: 0.0568


730/Unknown  121s 112ms/step - categorical_accuracy: 0.6686 - loss: 1.3363 - mean_io_u: 0.0568


731/Unknown  121s 112ms/step - categorical_accuracy: 0.6687 - loss: 1.3359 - mean_io_u: 0.0569


732/Unknown  121s 112ms/step - categorical_accuracy: 0.6687 - loss: 1.3356 - mean_io_u: 0.0569


733/Unknown  122s 112ms/step - categorical_accuracy: 0.6688 - loss: 1.3352 - mean_io_u: 0.0570


734/Unknown  122s 112ms/step - categorical_accuracy: 0.6689 - loss: 1.3349 - mean_io_u: 0.0570


735/Unknown  122s 112ms/step - categorical_accuracy: 0.6690 - loss: 1.3345 - mean_io_u: 0.0570


736/Unknown  122s 112ms/step - categorical_accuracy: 0.6691 - loss: 1.3342 - mean_io_u: 0.0571


737/Unknown  122s 112ms/step - categorical_accuracy: 0.6691 - loss: 1.3338 - mean_io_u: 0.0571


738/Unknown  122s 112ms/step - categorical_accuracy: 0.6692 - loss: 1.3335 - mean_io_u: 0.0571


739/Unknown  122s 112ms/step - categorical_accuracy: 0.6693 - loss: 1.3331 - mean_io_u: 0.0572


740/Unknown  122s 112ms/step - categorical_accuracy: 0.6694 - loss: 1.3328 - mean_io_u: 0.0572


741/Unknown  122s 112ms/step - categorical_accuracy: 0.6694 - loss: 1.3324 - mean_io_u: 0.0573


742/Unknown  122s 112ms/step - categorical_accuracy: 0.6695 - loss: 1.3321 - mean_io_u: 0.0573


743/Unknown  122s 112ms/step - categorical_accuracy: 0.6696 - loss: 1.3317 - mean_io_u: 0.0573


744/Unknown  123s 112ms/step - categorical_accuracy: 0.6697 - loss: 1.3314 - mean_io_u: 0.0574


745/Unknown  123s 111ms/step - categorical_accuracy: 0.6698 - loss: 1.3310 - mean_io_u: 0.0574


746/Unknown  123s 111ms/step - categorical_accuracy: 0.6698 - loss: 1.3307 - mean_io_u: 0.0574


747/Unknown  123s 111ms/step - categorical_accuracy: 0.6699 - loss: 1.3303 - mean_io_u: 0.0575


748/Unknown  123s 111ms/step - categorical_accuracy: 0.6700 - loss: 1.3300 - mean_io_u: 0.0575


749/Unknown  123s 111ms/step - categorical_accuracy: 0.6701 - loss: 1.3297 - mean_io_u: 0.0575


750/Unknown  123s 111ms/step - categorical_accuracy: 0.6701 - loss: 1.3293 - mean_io_u: 0.0576


751/Unknown  123s 111ms/step - categorical_accuracy: 0.6702 - loss: 1.3290 - mean_io_u: 0.0576


752/Unknown  123s 111ms/step - categorical_accuracy: 0.6703 - loss: 1.3286 - mean_io_u: 0.0577


753/Unknown  123s 111ms/step - categorical_accuracy: 0.6704 - loss: 1.3283 - mean_io_u: 0.0577


754/Unknown  123s 111ms/step - categorical_accuracy: 0.6704 - loss: 1.3279 - mean_io_u: 0.0577


755/Unknown  123s 111ms/step - categorical_accuracy: 0.6705 - loss: 1.3276 - mean_io_u: 0.0578


756/Unknown  123s 111ms/step - categorical_accuracy: 0.6706 - loss: 1.3273 - mean_io_u: 0.0578


757/Unknown  124s 111ms/step - categorical_accuracy: 0.6707 - loss: 1.3269 - mean_io_u: 0.0578


758/Unknown  124s 111ms/step - categorical_accuracy: 0.6707 - loss: 1.3266 - mean_io_u: 0.0579


759/Unknown  124s 111ms/step - categorical_accuracy: 0.6708 - loss: 1.3262 - mean_io_u: 0.0579


760/Unknown  124s 111ms/step - categorical_accuracy: 0.6709 - loss: 1.3259 - mean_io_u: 0.0579


761/Unknown  124s 111ms/step - categorical_accuracy: 0.6710 - loss: 1.3256 - mean_io_u: 0.0580


762/Unknown  124s 111ms/step - categorical_accuracy: 0.6710 - loss: 1.3252 - mean_io_u: 0.0580


763/Unknown  124s 111ms/step - categorical_accuracy: 0.6711 - loss: 1.3249 - mean_io_u: 0.0580


764/Unknown  124s 111ms/step - categorical_accuracy: 0.6712 - loss: 1.3246 - mean_io_u: 0.0581


765/Unknown  124s 111ms/step - categorical_accuracy: 0.6713 - loss: 1.3242 - mean_io_u: 0.0581


766/Unknown  124s 111ms/step - categorical_accuracy: 0.6713 - loss: 1.3239 - mean_io_u: 0.0582


767/Unknown  124s 110ms/step - categorical_accuracy: 0.6714 - loss: 1.3236 - mean_io_u: 0.0582


768/Unknown  124s 110ms/step - categorical_accuracy: 0.6715 - loss: 1.3232 - mean_io_u: 0.0582


769/Unknown  124s 110ms/step - categorical_accuracy: 0.6716 - loss: 1.3229 - mean_io_u: 0.0583


770/Unknown  125s 110ms/step - categorical_accuracy: 0.6716 - loss: 1.3226 - mean_io_u: 0.0583


771/Unknown  125s 110ms/step - categorical_accuracy: 0.6717 - loss: 1.3222 - mean_io_u: 0.0583


772/Unknown  125s 110ms/step - categorical_accuracy: 0.6718 - loss: 1.3219 - mean_io_u: 0.0584


773/Unknown  125s 110ms/step - categorical_accuracy: 0.6719 - loss: 1.3216 - mean_io_u: 0.0584


774/Unknown  125s 110ms/step - categorical_accuracy: 0.6719 - loss: 1.3213 - mean_io_u: 0.0584


775/Unknown  125s 110ms/step - categorical_accuracy: 0.6720 - loss: 1.3209 - mean_io_u: 0.0585


776/Unknown  125s 110ms/step - categorical_accuracy: 0.6721 - loss: 1.3206 - mean_io_u: 0.0585


777/Unknown  125s 110ms/step - categorical_accuracy: 0.6721 - loss: 1.3203 - mean_io_u: 0.0585


778/Unknown  125s 110ms/step - categorical_accuracy: 0.6722 - loss: 1.3199 - mean_io_u: 0.0586


779/Unknown  125s 110ms/step - categorical_accuracy: 0.6723 - loss: 1.3196 - mean_io_u: 0.0586


780/Unknown  125s 110ms/step - categorical_accuracy: 0.6724 - loss: 1.3193 - mean_io_u: 0.0586


781/Unknown  125s 110ms/step - categorical_accuracy: 0.6724 - loss: 1.3190 - mean_io_u: 0.0587


782/Unknown  126s 110ms/step - categorical_accuracy: 0.6725 - loss: 1.3186 - mean_io_u: 0.0587


783/Unknown  126s 110ms/step - categorical_accuracy: 0.6726 - loss: 1.3183 - mean_io_u: 0.0587


784/Unknown  126s 110ms/step - categorical_accuracy: 0.6726 - loss: 1.3180 - mean_io_u: 0.0588


785/Unknown  126s 110ms/step - categorical_accuracy: 0.6727 - loss: 1.3177 - mean_io_u: 0.0588


786/Unknown  126s 110ms/step - categorical_accuracy: 0.6728 - loss: 1.3173 - mean_io_u: 0.0589


787/Unknown  126s 110ms/step - categorical_accuracy: 0.6729 - loss: 1.3170 - mean_io_u: 0.0589


788/Unknown  126s 110ms/step - categorical_accuracy: 0.6729 - loss: 1.3167 - mean_io_u: 0.0589


789/Unknown  126s 110ms/step - categorical_accuracy: 0.6730 - loss: 1.3164 - mean_io_u: 0.0590


790/Unknown  126s 110ms/step - categorical_accuracy: 0.6731 - loss: 1.3161 - mean_io_u: 0.0590


791/Unknown  126s 110ms/step - categorical_accuracy: 0.6731 - loss: 1.3158 - mean_io_u: 0.0590


792/Unknown  126s 110ms/step - categorical_accuracy: 0.6732 - loss: 1.3154 - mean_io_u: 0.0591


793/Unknown  126s 109ms/step - categorical_accuracy: 0.6733 - loss: 1.3151 - mean_io_u: 0.0591


794/Unknown  126s 109ms/step - categorical_accuracy: 0.6734 - loss: 1.3148 - mean_io_u: 0.0591


795/Unknown  126s 109ms/step - categorical_accuracy: 0.6734 - loss: 1.3145 - mean_io_u: 0.0592


796/Unknown  127s 109ms/step - categorical_accuracy: 0.6735 - loss: 1.3142 - mean_io_u: 0.0592


797/Unknown  127s 109ms/step - categorical_accuracy: 0.6736 - loss: 1.3139 - mean_io_u: 0.0592


798/Unknown  127s 109ms/step - categorical_accuracy: 0.6736 - loss: 1.3136 - mean_io_u: 0.0593


799/Unknown  127s 109ms/step - categorical_accuracy: 0.6737 - loss: 1.3132 - mean_io_u: 0.0593


800/Unknown  127s 109ms/step - categorical_accuracy: 0.6738 - loss: 1.3129 - mean_io_u: 0.0593


801/Unknown  127s 109ms/step - categorical_accuracy: 0.6738 - loss: 1.3126 - mean_io_u: 0.0594


802/Unknown  127s 109ms/step - categorical_accuracy: 0.6739 - loss: 1.3123 - mean_io_u: 0.0594


803/Unknown  127s 109ms/step - categorical_accuracy: 0.6740 - loss: 1.3120 - mean_io_u: 0.0594


804/Unknown  127s 109ms/step - categorical_accuracy: 0.6740 - loss: 1.3117 - mean_io_u: 0.0595


805/Unknown  127s 109ms/step - categorical_accuracy: 0.6741 - loss: 1.3114 - mean_io_u: 0.0595


806/Unknown  127s 109ms/step - categorical_accuracy: 0.6742 - loss: 1.3111 - mean_io_u: 0.0595


807/Unknown  127s 109ms/step - categorical_accuracy: 0.6742 - loss: 1.3108 - mean_io_u: 0.0596


808/Unknown  127s 109ms/step - categorical_accuracy: 0.6743 - loss: 1.3104 - mean_io_u: 0.0596


809/Unknown  127s 109ms/step - categorical_accuracy: 0.6744 - loss: 1.3101 - mean_io_u: 0.0596


810/Unknown  127s 109ms/step - categorical_accuracy: 0.6744 - loss: 1.3098 - mean_io_u: 0.0597


811/Unknown  128s 108ms/step - categorical_accuracy: 0.6745 - loss: 1.3095 - mean_io_u: 0.0597


812/Unknown  128s 108ms/step - categorical_accuracy: 0.6746 - loss: 1.3092 - mean_io_u: 0.0597


813/Unknown  128s 108ms/step - categorical_accuracy: 0.6747 - loss: 1.3089 - mean_io_u: 0.0598


814/Unknown  128s 108ms/step - categorical_accuracy: 0.6747 - loss: 1.3086 - mean_io_u: 0.0598


815/Unknown  128s 108ms/step - categorical_accuracy: 0.6748 - loss: 1.3083 - mean_io_u: 0.0599


816/Unknown  128s 108ms/step - categorical_accuracy: 0.6749 - loss: 1.3080 - mean_io_u: 0.0599


817/Unknown  128s 108ms/step - categorical_accuracy: 0.6749 - loss: 1.3077 - mean_io_u: 0.0599


818/Unknown  128s 108ms/step - categorical_accuracy: 0.6750 - loss: 1.3074 - mean_io_u: 0.0600


819/Unknown  128s 108ms/step - categorical_accuracy: 0.6751 - loss: 1.3071 - mean_io_u: 0.0600


820/Unknown  128s 108ms/step - categorical_accuracy: 0.6751 - loss: 1.3068 - mean_io_u: 0.0600


821/Unknown  128s 108ms/step - categorical_accuracy: 0.6752 - loss: 1.3065 - mean_io_u: 0.0601


822/Unknown  128s 108ms/step - categorical_accuracy: 0.6753 - loss: 1.3062 - mean_io_u: 0.0601


823/Unknown  128s 108ms/step - categorical_accuracy: 0.6753 - loss: 1.3059 - mean_io_u: 0.0601


824/Unknown  129s 108ms/step - categorical_accuracy: 0.6754 - loss: 1.3056 - mean_io_u: 0.0602


825/Unknown  129s 108ms/step - categorical_accuracy: 0.6754 - loss: 1.3053 - mean_io_u: 0.0602


826/Unknown  129s 108ms/step - categorical_accuracy: 0.6755 - loss: 1.3050 - mean_io_u: 0.0602


827/Unknown  129s 108ms/step - categorical_accuracy: 0.6756 - loss: 1.3047 - mean_io_u: 0.0603


828/Unknown  129s 108ms/step - categorical_accuracy: 0.6756 - loss: 1.3044 - mean_io_u: 0.0603


829/Unknown  129s 108ms/step - categorical_accuracy: 0.6757 - loss: 1.3041 - mean_io_u: 0.0603


830/Unknown  129s 108ms/step - categorical_accuracy: 0.6758 - loss: 1.3038 - mean_io_u: 0.0604


831/Unknown  129s 108ms/step - categorical_accuracy: 0.6758 - loss: 1.3035 - mean_io_u: 0.0604


832/Unknown  129s 108ms/step - categorical_accuracy: 0.6759 - loss: 1.3032 - mean_io_u: 0.0604


833/Unknown  129s 108ms/step - categorical_accuracy: 0.6760 - loss: 1.3029 - mean_io_u: 0.0605


834/Unknown  129s 108ms/step - categorical_accuracy: 0.6760 - loss: 1.3026 - mean_io_u: 0.0605


835/Unknown  129s 108ms/step - categorical_accuracy: 0.6761 - loss: 1.3023 - mean_io_u: 0.0605


836/Unknown  129s 107ms/step - categorical_accuracy: 0.6762 - loss: 1.3020 - mean_io_u: 0.0606


837/Unknown  129s 107ms/step - categorical_accuracy: 0.6762 - loss: 1.3017 - mean_io_u: 0.0606


838/Unknown  130s 107ms/step - categorical_accuracy: 0.6763 - loss: 1.3014 - mean_io_u: 0.0606


839/Unknown  130s 107ms/step - categorical_accuracy: 0.6764 - loss: 1.3011 - mean_io_u: 0.0607


840/Unknown  130s 107ms/step - categorical_accuracy: 0.6764 - loss: 1.3008 - mean_io_u: 0.0607


841/Unknown  130s 107ms/step - categorical_accuracy: 0.6765 - loss: 1.3005 - mean_io_u: 0.0607


842/Unknown  130s 107ms/step - categorical_accuracy: 0.6766 - loss: 1.3002 - mean_io_u: 0.0608


843/Unknown  130s 107ms/step - categorical_accuracy: 0.6766 - loss: 1.2999 - mean_io_u: 0.0608


844/Unknown  130s 107ms/step - categorical_accuracy: 0.6767 - loss: 1.2996 - mean_io_u: 0.0608


845/Unknown  130s 107ms/step - categorical_accuracy: 0.6767 - loss: 1.2994 - mean_io_u: 0.0609


846/Unknown  130s 107ms/step - categorical_accuracy: 0.6768 - loss: 1.2991 - mean_io_u: 0.0609


847/Unknown  130s 107ms/step - categorical_accuracy: 0.6769 - loss: 1.2988 - mean_io_u: 0.0610


848/Unknown  130s 107ms/step - categorical_accuracy: 0.6769 - loss: 1.2985 - mean_io_u: 0.0610


849/Unknown  130s 107ms/step - categorical_accuracy: 0.6770 - loss: 1.2982 - mean_io_u: 0.0610


850/Unknown  131s 107ms/step - categorical_accuracy: 0.6771 - loss: 1.2979 - mean_io_u: 0.0611


851/Unknown  131s 107ms/step - categorical_accuracy: 0.6771 - loss: 1.2976 - mean_io_u: 0.0611


852/Unknown  131s 107ms/step - categorical_accuracy: 0.6772 - loss: 1.2973 - mean_io_u: 0.0611


853/Unknown  131s 107ms/step - categorical_accuracy: 0.6773 - loss: 1.2970 - mean_io_u: 0.0612


854/Unknown  131s 107ms/step - categorical_accuracy: 0.6773 - loss: 1.2967 - mean_io_u: 0.0612


855/Unknown  131s 107ms/step - categorical_accuracy: 0.6774 - loss: 1.2964 - mean_io_u: 0.0612


856/Unknown  131s 107ms/step - categorical_accuracy: 0.6774 - loss: 1.2962 - mean_io_u: 0.0613


857/Unknown  131s 107ms/step - categorical_accuracy: 0.6775 - loss: 1.2959 - mean_io_u: 0.0613


858/Unknown  131s 107ms/step - categorical_accuracy: 0.6776 - loss: 1.2956 - mean_io_u: 0.0613


859/Unknown  131s 107ms/step - categorical_accuracy: 0.6776 - loss: 1.2953 - mean_io_u: 0.0614


860/Unknown  131s 107ms/step - categorical_accuracy: 0.6777 - loss: 1.2950 - mean_io_u: 0.0614


861/Unknown  131s 107ms/step - categorical_accuracy: 0.6778 - loss: 1.2947 - mean_io_u: 0.0614


862/Unknown  131s 107ms/step - categorical_accuracy: 0.6778 - loss: 1.2944 - mean_io_u: 0.0615


863/Unknown  132s 107ms/step - categorical_accuracy: 0.6779 - loss: 1.2941 - mean_io_u: 0.0615


864/Unknown  132s 107ms/step - categorical_accuracy: 0.6780 - loss: 1.2939 - mean_io_u: 0.0615


865/Unknown  132s 107ms/step - categorical_accuracy: 0.6780 - loss: 1.2936 - mean_io_u: 0.0616


866/Unknown  132s 107ms/step - categorical_accuracy: 0.6781 - loss: 1.2933 - mean_io_u: 0.0616


867/Unknown  132s 107ms/step - categorical_accuracy: 0.6781 - loss: 1.2930 - mean_io_u: 0.0616


868/Unknown  132s 107ms/step - categorical_accuracy: 0.6782 - loss: 1.2927 - mean_io_u: 0.0617


869/Unknown  132s 107ms/step - categorical_accuracy: 0.6783 - loss: 1.2924 - mean_io_u: 0.0617


870/Unknown  132s 107ms/step - categorical_accuracy: 0.6783 - loss: 1.2922 - mean_io_u: 0.0617


871/Unknown  132s 106ms/step - categorical_accuracy: 0.6784 - loss: 1.2919 - mean_io_u: 0.0618


872/Unknown  132s 106ms/step - categorical_accuracy: 0.6784 - loss: 1.2916 - mean_io_u: 0.0618


873/Unknown  132s 106ms/step - categorical_accuracy: 0.6785 - loss: 1.2913 - mean_io_u: 0.0618


874/Unknown  133s 106ms/step - categorical_accuracy: 0.6786 - loss: 1.2910 - mean_io_u: 0.0619


875/Unknown  133s 106ms/step - categorical_accuracy: 0.6786 - loss: 1.2907 - mean_io_u: 0.0619


876/Unknown  133s 106ms/step - categorical_accuracy: 0.6787 - loss: 1.2905 - mean_io_u: 0.0619


877/Unknown  133s 106ms/step - categorical_accuracy: 0.6788 - loss: 1.2902 - mean_io_u: 0.0620


878/Unknown  133s 106ms/step - categorical_accuracy: 0.6788 - loss: 1.2899 - mean_io_u: 0.0620


879/Unknown  133s 106ms/step - categorical_accuracy: 0.6789 - loss: 1.2896 - mean_io_u: 0.0620


880/Unknown  133s 106ms/step - categorical_accuracy: 0.6789 - loss: 1.2893 - mean_io_u: 0.0621


881/Unknown  133s 106ms/step - categorical_accuracy: 0.6790 - loss: 1.2891 - mean_io_u: 0.0621


882/Unknown  133s 106ms/step - categorical_accuracy: 0.6791 - loss: 1.2888 - mean_io_u: 0.0621


883/Unknown  133s 106ms/step - categorical_accuracy: 0.6791 - loss: 1.2885 - mean_io_u: 0.0622


884/Unknown  133s 106ms/step - categorical_accuracy: 0.6792 - loss: 1.2882 - mean_io_u: 0.0622


885/Unknown  133s 106ms/step - categorical_accuracy: 0.6792 - loss: 1.2879 - mean_io_u: 0.0622


886/Unknown  134s 106ms/step - categorical_accuracy: 0.6793 - loss: 1.2877 - mean_io_u: 0.0623


887/Unknown  134s 106ms/step - categorical_accuracy: 0.6794 - loss: 1.2874 - mean_io_u: 0.0623


888/Unknown  134s 106ms/step - categorical_accuracy: 0.6794 - loss: 1.2871 - mean_io_u: 0.0623


889/Unknown  134s 106ms/step - categorical_accuracy: 0.6795 - loss: 1.2868 - mean_io_u: 0.0624


890/Unknown  134s 106ms/step - categorical_accuracy: 0.6795 - loss: 1.2866 - mean_io_u: 0.0624


891/Unknown  134s 106ms/step - categorical_accuracy: 0.6796 - loss: 1.2863 - mean_io_u: 0.0624


892/Unknown  134s 106ms/step - categorical_accuracy: 0.6797 - loss: 1.2860 - mean_io_u: 0.0625


893/Unknown  134s 106ms/step - categorical_accuracy: 0.6797 - loss: 1.2857 - mean_io_u: 0.0625


894/Unknown  134s 106ms/step - categorical_accuracy: 0.6798 - loss: 1.2855 - mean_io_u: 0.0625


895/Unknown  134s 106ms/step - categorical_accuracy: 0.6798 - loss: 1.2852 - mean_io_u: 0.0626


896/Unknown  134s 106ms/step - categorical_accuracy: 0.6799 - loss: 1.2849 - mean_io_u: 0.0626


897/Unknown  134s 106ms/step - categorical_accuracy: 0.6800 - loss: 1.2846 - mean_io_u: 0.0626


898/Unknown  134s 106ms/step - categorical_accuracy: 0.6800 - loss: 1.2844 - mean_io_u: 0.0627


899/Unknown  135s 106ms/step - categorical_accuracy: 0.6801 - loss: 1.2841 - mean_io_u: 0.0627


900/Unknown  135s 106ms/step - categorical_accuracy: 0.6801 - loss: 1.2838 - mean_io_u: 0.0627


901/Unknown  135s 106ms/step - categorical_accuracy: 0.6802 - loss: 1.2835 - mean_io_u: 0.0628


902/Unknown  135s 106ms/step - categorical_accuracy: 0.6803 - loss: 1.2833 - mean_io_u: 0.0628


903/Unknown  135s 106ms/step - categorical_accuracy: 0.6803 - loss: 1.2830 - mean_io_u: 0.0628


904/Unknown  135s 106ms/step - categorical_accuracy: 0.6804 - loss: 1.2827 - mean_io_u: 0.0629


905/Unknown  135s 105ms/step - categorical_accuracy: 0.6804 - loss: 1.2824 - mean_io_u: 0.0629


906/Unknown  135s 105ms/step - categorical_accuracy: 0.6805 - loss: 1.2822 - mean_io_u: 0.0629


907/Unknown  135s 105ms/step - categorical_accuracy: 0.6806 - loss: 1.2819 - mean_io_u: 0.0630


908/Unknown  135s 105ms/step - categorical_accuracy: 0.6806 - loss: 1.2816 - mean_io_u: 0.0630


909/Unknown  135s 105ms/step - categorical_accuracy: 0.6807 - loss: 1.2814 - mean_io_u: 0.0630


910/Unknown  135s 105ms/step - categorical_accuracy: 0.6807 - loss: 1.2811 - mean_io_u: 0.0630


911/Unknown  136s 105ms/step - categorical_accuracy: 0.6808 - loss: 1.2808 - mean_io_u: 0.0631


912/Unknown  136s 105ms/step - categorical_accuracy: 0.6809 - loss: 1.2806 - mean_io_u: 0.0631


913/Unknown  136s 105ms/step - categorical_accuracy: 0.6809 - loss: 1.2803 - mean_io_u: 0.0631


914/Unknown  136s 105ms/step - categorical_accuracy: 0.6810 - loss: 1.2800 - mean_io_u: 0.0632


915/Unknown  136s 105ms/step - categorical_accuracy: 0.6810 - loss: 1.2797 - mean_io_u: 0.0632


916/Unknown  136s 105ms/step - categorical_accuracy: 0.6811 - loss: 1.2795 - mean_io_u: 0.0632


917/Unknown  136s 105ms/step - categorical_accuracy: 0.6812 - loss: 1.2792 - mean_io_u: 0.0633


918/Unknown  136s 105ms/step - categorical_accuracy: 0.6812 - loss: 1.2789 - mean_io_u: 0.0633


919/Unknown  136s 105ms/step - categorical_accuracy: 0.6813 - loss: 1.2787 - mean_io_u: 0.0633


920/Unknown  136s 105ms/step - categorical_accuracy: 0.6813 - loss: 1.2784 - mean_io_u: 0.0634


921/Unknown  136s 105ms/step - categorical_accuracy: 0.6814 - loss: 1.2781 - mean_io_u: 0.0634


922/Unknown  136s 105ms/step - categorical_accuracy: 0.6814 - loss: 1.2779 - mean_io_u: 0.0634


923/Unknown  136s 105ms/step - categorical_accuracy: 0.6815 - loss: 1.2776 - mean_io_u: 0.0635


924/Unknown  137s 105ms/step - categorical_accuracy: 0.6816 - loss: 1.2773 - mean_io_u: 0.0635


925/Unknown  137s 105ms/step - categorical_accuracy: 0.6816 - loss: 1.2771 - mean_io_u: 0.0635


926/Unknown  137s 105ms/step - categorical_accuracy: 0.6817 - loss: 1.2768 - mean_io_u: 0.0636


927/Unknown  137s 105ms/step - categorical_accuracy: 0.6817 - loss: 1.2766 - mean_io_u: 0.0636


928/Unknown  137s 105ms/step - categorical_accuracy: 0.6818 - loss: 1.2763 - mean_io_u: 0.0636


929/Unknown  137s 105ms/step - categorical_accuracy: 0.6818 - loss: 1.2760 - mean_io_u: 0.0637


930/Unknown  137s 105ms/step - categorical_accuracy: 0.6819 - loss: 1.2758 - mean_io_u: 0.0637


931/Unknown  137s 105ms/step - categorical_accuracy: 0.6820 - loss: 1.2755 - mean_io_u: 0.0637


932/Unknown  137s 105ms/step - categorical_accuracy: 0.6820 - loss: 1.2752 - mean_io_u: 0.0638


933/Unknown  137s 105ms/step - categorical_accuracy: 0.6821 - loss: 1.2750 - mean_io_u: 0.0638


934/Unknown  137s 105ms/step - categorical_accuracy: 0.6821 - loss: 1.2747 - mean_io_u: 0.0638


935/Unknown  137s 105ms/step - categorical_accuracy: 0.6822 - loss: 1.2745 - mean_io_u: 0.0639


936/Unknown  137s 105ms/step - categorical_accuracy: 0.6822 - loss: 1.2742 - mean_io_u: 0.0639


937/Unknown  138s 105ms/step - categorical_accuracy: 0.6823 - loss: 1.2740 - mean_io_u: 0.0639


938/Unknown  138s 105ms/step - categorical_accuracy: 0.6824 - loss: 1.2737 - mean_io_u: 0.0640


939/Unknown  138s 104ms/step - categorical_accuracy: 0.6824 - loss: 1.2734 - mean_io_u: 0.0640


940/Unknown  138s 104ms/step - categorical_accuracy: 0.6825 - loss: 1.2732 - mean_io_u: 0.0640


941/Unknown  138s 104ms/step - categorical_accuracy: 0.6825 - loss: 1.2729 - mean_io_u: 0.0641


942/Unknown  138s 104ms/step - categorical_accuracy: 0.6826 - loss: 1.2727 - mean_io_u: 0.0641


943/Unknown  138s 104ms/step - categorical_accuracy: 0.6826 - loss: 1.2724 - mean_io_u: 0.0641


944/Unknown  138s 104ms/step - categorical_accuracy: 0.6827 - loss: 1.2722 - mean_io_u: 0.0642


945/Unknown  138s 104ms/step - categorical_accuracy: 0.6827 - loss: 1.2719 - mean_io_u: 0.0642


946/Unknown  138s 104ms/step - categorical_accuracy: 0.6828 - loss: 1.2717 - mean_io_u: 0.0642


947/Unknown  138s 104ms/step - categorical_accuracy: 0.6828 - loss: 1.2714 - mean_io_u: 0.0642


948/Unknown  138s 104ms/step - categorical_accuracy: 0.6829 - loss: 1.2712 - mean_io_u: 0.0643


949/Unknown  138s 104ms/step - categorical_accuracy: 0.6830 - loss: 1.2709 - mean_io_u: 0.0643


950/Unknown  138s 104ms/step - categorical_accuracy: 0.6830 - loss: 1.2707 - mean_io_u: 0.0643


951/Unknown  139s 104ms/step - categorical_accuracy: 0.6831 - loss: 1.2704 - mean_io_u: 0.0644


952/Unknown  139s 104ms/step - categorical_accuracy: 0.6831 - loss: 1.2702 - mean_io_u: 0.0644


953/Unknown  139s 104ms/step - categorical_accuracy: 0.6832 - loss: 1.2699 - mean_io_u: 0.0644


954/Unknown  139s 104ms/step - categorical_accuracy: 0.6832 - loss: 1.2697 - mean_io_u: 0.0645


955/Unknown  139s 104ms/step - categorical_accuracy: 0.6833 - loss: 1.2694 - mean_io_u: 0.0645


956/Unknown  139s 104ms/step - categorical_accuracy: 0.6833 - loss: 1.2692 - mean_io_u: 0.0645


957/Unknown  139s 104ms/step - categorical_accuracy: 0.6834 - loss: 1.2689 - mean_io_u: 0.0646


958/Unknown  139s 104ms/step - categorical_accuracy: 0.6834 - loss: 1.2687 - mean_io_u: 0.0646


959/Unknown  139s 104ms/step - categorical_accuracy: 0.6835 - loss: 1.2684 - mean_io_u: 0.0646


960/Unknown  139s 104ms/step - categorical_accuracy: 0.6835 - loss: 1.2682 - mean_io_u: 0.0647


961/Unknown  139s 104ms/step - categorical_accuracy: 0.6836 - loss: 1.2679 - mean_io_u: 0.0647


962/Unknown  139s 104ms/step - categorical_accuracy: 0.6837 - loss: 1.2677 - mean_io_u: 0.0647


963/Unknown  139s 104ms/step - categorical_accuracy: 0.6837 - loss: 1.2675 - mean_io_u: 0.0648


964/Unknown  139s 104ms/step - categorical_accuracy: 0.6838 - loss: 1.2672 - mean_io_u: 0.0648


965/Unknown  139s 104ms/step - categorical_accuracy: 0.6838 - loss: 1.2670 - mean_io_u: 0.0648


966/Unknown  140s 104ms/step - categorical_accuracy: 0.6839 - loss: 1.2667 - mean_io_u: 0.0648


967/Unknown  140s 103ms/step - categorical_accuracy: 0.6839 - loss: 1.2665 - mean_io_u: 0.0649


968/Unknown  140s 103ms/step - categorical_accuracy: 0.6840 - loss: 1.2662 - mean_io_u: 0.0649


969/Unknown  140s 103ms/step - categorical_accuracy: 0.6840 - loss: 1.2660 - mean_io_u: 0.0649


970/Unknown  140s 103ms/step - categorical_accuracy: 0.6841 - loss: 1.2658 - mean_io_u: 0.0650


971/Unknown  140s 103ms/step - categorical_accuracy: 0.6841 - loss: 1.2655 - mean_io_u: 0.0650


972/Unknown  140s 103ms/step - categorical_accuracy: 0.6842 - loss: 1.2653 - mean_io_u: 0.0650


973/Unknown  140s 103ms/step - categorical_accuracy: 0.6842 - loss: 1.2650 - mean_io_u: 0.0651


974/Unknown  140s 103ms/step - categorical_accuracy: 0.6843 - loss: 1.2648 - mean_io_u: 0.0651


975/Unknown  140s 103ms/step - categorical_accuracy: 0.6843 - loss: 1.2646 - mean_io_u: 0.0651


976/Unknown  140s 103ms/step - categorical_accuracy: 0.6844 - loss: 1.2643 - mean_io_u: 0.0652


977/Unknown  140s 103ms/step - categorical_accuracy: 0.6844 - loss: 1.2641 - mean_io_u: 0.0652


978/Unknown  140s 103ms/step - categorical_accuracy: 0.6845 - loss: 1.2638 - mean_io_u: 0.0652


979/Unknown  140s 103ms/step - categorical_accuracy: 0.6845 - loss: 1.2636 - mean_io_u: 0.0652


980/Unknown  141s 103ms/step - categorical_accuracy: 0.6846 - loss: 1.2634 - mean_io_u: 0.0653


981/Unknown  141s 103ms/step - categorical_accuracy: 0.6846 - loss: 1.2631 - mean_io_u: 0.0653


982/Unknown  141s 103ms/step - categorical_accuracy: 0.6847 - loss: 1.2629 - mean_io_u: 0.0653


983/Unknown  141s 103ms/step - categorical_accuracy: 0.6848 - loss: 1.2626 - mean_io_u: 0.0654


984/Unknown  141s 103ms/step - categorical_accuracy: 0.6848 - loss: 1.2624 - mean_io_u: 0.0654


985/Unknown  141s 103ms/step - categorical_accuracy: 0.6849 - loss: 1.2622 - mean_io_u: 0.0654


986/Unknown  141s 103ms/step - categorical_accuracy: 0.6849 - loss: 1.2619 - mean_io_u: 0.0655


987/Unknown  141s 103ms/step - categorical_accuracy: 0.6850 - loss: 1.2617 - mean_io_u: 0.0655


988/Unknown  141s 103ms/step - categorical_accuracy: 0.6850 - loss: 1.2615 - mean_io_u: 0.0655


989/Unknown  141s 103ms/step - categorical_accuracy: 0.6851 - loss: 1.2612 - mean_io_u: 0.0656


990/Unknown  141s 103ms/step - categorical_accuracy: 0.6851 - loss: 1.2610 - mean_io_u: 0.0656


991/Unknown  141s 103ms/step - categorical_accuracy: 0.6852 - loss: 1.2608 - mean_io_u: 0.0656


992/Unknown  141s 103ms/step - categorical_accuracy: 0.6852 - loss: 1.2605 - mean_io_u: 0.0656


993/Unknown  142s 103ms/step - categorical_accuracy: 0.6853 - loss: 1.2603 - mean_io_u: 0.0657


994/Unknown  142s 103ms/step - categorical_accuracy: 0.6853 - loss: 1.2601 - mean_io_u: 0.0657


995/Unknown  142s 103ms/step - categorical_accuracy: 0.6854 - loss: 1.2598 - mean_io_u: 0.0657


996/Unknown  142s 103ms/step - categorical_accuracy: 0.6854 - loss: 1.2596 - mean_io_u: 0.0658


997/Unknown  142s 103ms/step - categorical_accuracy: 0.6855 - loss: 1.2594 - mean_io_u: 0.0658


998/Unknown  142s 103ms/step - categorical_accuracy: 0.6855 - loss: 1.2591 - mean_io_u: 0.0658


999/Unknown  142s 103ms/step - categorical_accuracy: 0.6856 - loss: 1.2589 - mean_io_u: 0.0659


```
</div>
   1000/Unknown  142s 103ms/step - categorical_accuracy: 0.6856 - loss: 1.2587 - mean_io_u: 0.0659

<div class="k-default-codeblock">
```

```
</div>
   1001/Unknown  142s 102ms/step - categorical_accuracy: 0.6857 - loss: 1.2584 - mean_io_u: 0.0659

<div class="k-default-codeblock">
```

```
</div>
   1002/Unknown  142s 102ms/step - categorical_accuracy: 0.6857 - loss: 1.2582 - mean_io_u: 0.0659

<div class="k-default-codeblock">
```

```
</div>
   1003/Unknown  142s 102ms/step - categorical_accuracy: 0.6858 - loss: 1.2580 - mean_io_u: 0.0660

<div class="k-default-codeblock">
```

```
</div>
   1004/Unknown  142s 102ms/step - categorical_accuracy: 0.6858 - loss: 1.2577 - mean_io_u: 0.0660

<div class="k-default-codeblock">
```

```
</div>
   1005/Unknown  142s 102ms/step - categorical_accuracy: 0.6859 - loss: 1.2575 - mean_io_u: 0.0660

<div class="k-default-codeblock">
```

```
</div>
   1006/Unknown  142s 102ms/step - categorical_accuracy: 0.6859 - loss: 1.2573 - mean_io_u: 0.0661

<div class="k-default-codeblock">
```

```
</div>
   1007/Unknown  143s 102ms/step - categorical_accuracy: 0.6860 - loss: 1.2571 - mean_io_u: 0.0661

<div class="k-default-codeblock">
```

```
</div>
   1008/Unknown  143s 102ms/step - categorical_accuracy: 0.6860 - loss: 1.2568 - mean_io_u: 0.0661

<div class="k-default-codeblock">
```

```
</div>
   1009/Unknown  143s 102ms/step - categorical_accuracy: 0.6861 - loss: 1.2566 - mean_io_u: 0.0662

<div class="k-default-codeblock">
```

```
</div>
   1010/Unknown  143s 102ms/step - categorical_accuracy: 0.6861 - loss: 1.2564 - mean_io_u: 0.0662

<div class="k-default-codeblock">
```

```
</div>
   1011/Unknown  143s 102ms/step - categorical_accuracy: 0.6862 - loss: 1.2561 - mean_io_u: 0.0662

<div class="k-default-codeblock">
```

```
</div>
   1012/Unknown  143s 102ms/step - categorical_accuracy: 0.6862 - loss: 1.2559 - mean_io_u: 0.0662

<div class="k-default-codeblock">
```

```
</div>
   1013/Unknown  143s 102ms/step - categorical_accuracy: 0.6863 - loss: 1.2557 - mean_io_u: 0.0663

<div class="k-default-codeblock">
```

```
</div>
   1014/Unknown  143s 102ms/step - categorical_accuracy: 0.6863 - loss: 1.2554 - mean_io_u: 0.0663

<div class="k-default-codeblock">
```

```
</div>
   1015/Unknown  143s 102ms/step - categorical_accuracy: 0.6864 - loss: 1.2552 - mean_io_u: 0.0663

<div class="k-default-codeblock">
```

```
</div>
   1016/Unknown  143s 102ms/step - categorical_accuracy: 0.6864 - loss: 1.2550 - mean_io_u: 0.0664

<div class="k-default-codeblock">
```

```
</div>
   1017/Unknown  144s 102ms/step - categorical_accuracy: 0.6865 - loss: 1.2548 - mean_io_u: 0.0664

<div class="k-default-codeblock">
```

```
</div>
   1018/Unknown  144s 102ms/step - categorical_accuracy: 0.6865 - loss: 1.2545 - mean_io_u: 0.0664

<div class="k-default-codeblock">
```

```
</div>
   1019/Unknown  144s 102ms/step - categorical_accuracy: 0.6866 - loss: 1.2543 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1020/Unknown  144s 102ms/step - categorical_accuracy: 0.6866 - loss: 1.2541 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1021/Unknown  144s 102ms/step - categorical_accuracy: 0.6867 - loss: 1.2538 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1022/Unknown  144s 102ms/step - categorical_accuracy: 0.6867 - loss: 1.2536 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1023/Unknown  144s 102ms/step - categorical_accuracy: 0.6867 - loss: 1.2534 - mean_io_u: 0.0666

<div class="k-default-codeblock">
```

```
</div>
   1024/Unknown  144s 102ms/step - categorical_accuracy: 0.6868 - loss: 1.2532 - mean_io_u: 0.0666

<div class="k-default-codeblock">
```

```
</div>
   1025/Unknown  144s 102ms/step - categorical_accuracy: 0.6868 - loss: 1.2529 - mean_io_u: 0.0666

<div class="k-default-codeblock">
```

```
</div>
   1026/Unknown  144s 102ms/step - categorical_accuracy: 0.6869 - loss: 1.2527 - mean_io_u: 0.0667

<div class="k-default-codeblock">
```

```
</div>
   1027/Unknown  144s 102ms/step - categorical_accuracy: 0.6869 - loss: 1.2525 - mean_io_u: 0.0667

<div class="k-default-codeblock">
```

```
</div>
   1028/Unknown  144s 102ms/step - categorical_accuracy: 0.6870 - loss: 1.2523 - mean_io_u: 0.0667

<div class="k-default-codeblock">
```

```
</div>
   1029/Unknown  144s 102ms/step - categorical_accuracy: 0.6870 - loss: 1.2520 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1030/Unknown  145s 102ms/step - categorical_accuracy: 0.6871 - loss: 1.2518 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1031/Unknown  145s 102ms/step - categorical_accuracy: 0.6871 - loss: 1.2516 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1032/Unknown  145s 102ms/step - categorical_accuracy: 0.6872 - loss: 1.2514 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1033/Unknown  145s 102ms/step - categorical_accuracy: 0.6872 - loss: 1.2511 - mean_io_u: 0.0669

<div class="k-default-codeblock">
```

```
</div>
   1034/Unknown  145s 102ms/step - categorical_accuracy: 0.6873 - loss: 1.2509 - mean_io_u: 0.0669

<div class="k-default-codeblock">
```

```
</div>
   1035/Unknown  145s 102ms/step - categorical_accuracy: 0.6873 - loss: 1.2507 - mean_io_u: 0.0669

<div class="k-default-codeblock">
```

```
</div>
   1036/Unknown  145s 102ms/step - categorical_accuracy: 0.6874 - loss: 1.2505 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1037/Unknown  145s 102ms/step - categorical_accuracy: 0.6874 - loss: 1.2502 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1038/Unknown  145s 102ms/step - categorical_accuracy: 0.6875 - loss: 1.2500 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1039/Unknown  145s 102ms/step - categorical_accuracy: 0.6875 - loss: 1.2498 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1040/Unknown  145s 102ms/step - categorical_accuracy: 0.6876 - loss: 1.2496 - mean_io_u: 0.0671

<div class="k-default-codeblock">
```

```
</div>
   1041/Unknown  145s 102ms/step - categorical_accuracy: 0.6876 - loss: 1.2493 - mean_io_u: 0.0671

<div class="k-default-codeblock">
```

```
</div>
   1042/Unknown  146s 102ms/step - categorical_accuracy: 0.6877 - loss: 1.2491 - mean_io_u: 0.0671

<div class="k-default-codeblock">
```

```
</div>
   1043/Unknown  146s 102ms/step - categorical_accuracy: 0.6877 - loss: 1.2489 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1044/Unknown  146s 102ms/step - categorical_accuracy: 0.6878 - loss: 1.2487 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1045/Unknown  146s 102ms/step - categorical_accuracy: 0.6878 - loss: 1.2485 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1046/Unknown  146s 102ms/step - categorical_accuracy: 0.6879 - loss: 1.2482 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1047/Unknown  146s 102ms/step - categorical_accuracy: 0.6879 - loss: 1.2480 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1048/Unknown  146s 102ms/step - categorical_accuracy: 0.6880 - loss: 1.2478 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1049/Unknown  146s 102ms/step - categorical_accuracy: 0.6880 - loss: 1.2476 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1050/Unknown  146s 102ms/step - categorical_accuracy: 0.6881 - loss: 1.2474 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1051/Unknown  146s 102ms/step - categorical_accuracy: 0.6881 - loss: 1.2471 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1052/Unknown  146s 101ms/step - categorical_accuracy: 0.6881 - loss: 1.2469 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1053/Unknown  146s 101ms/step - categorical_accuracy: 0.6882 - loss: 1.2467 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1054/Unknown  146s 101ms/step - categorical_accuracy: 0.6882 - loss: 1.2465 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1055/Unknown  147s 101ms/step - categorical_accuracy: 0.6883 - loss: 1.2463 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1056/Unknown  147s 101ms/step - categorical_accuracy: 0.6883 - loss: 1.2460 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1057/Unknown  147s 101ms/step - categorical_accuracy: 0.6884 - loss: 1.2458 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1058/Unknown  147s 101ms/step - categorical_accuracy: 0.6884 - loss: 1.2456 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1059/Unknown  147s 101ms/step - categorical_accuracy: 0.6885 - loss: 1.2454 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1060/Unknown  147s 101ms/step - categorical_accuracy: 0.6885 - loss: 1.2452 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1061/Unknown  147s 101ms/step - categorical_accuracy: 0.6886 - loss: 1.2450 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1062/Unknown  147s 101ms/step - categorical_accuracy: 0.6886 - loss: 1.2447 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1063/Unknown  147s 101ms/step - categorical_accuracy: 0.6887 - loss: 1.2445 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1064/Unknown  147s 101ms/step - categorical_accuracy: 0.6887 - loss: 1.2443 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1065/Unknown  147s 101ms/step - categorical_accuracy: 0.6888 - loss: 1.2441 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1066/Unknown  147s 101ms/step - categorical_accuracy: 0.6888 - loss: 1.2439 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1067/Unknown  147s 101ms/step - categorical_accuracy: 0.6888 - loss: 1.2437 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1068/Unknown  148s 101ms/step - categorical_accuracy: 0.6889 - loss: 1.2434 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1069/Unknown  148s 101ms/step - categorical_accuracy: 0.6889 - loss: 1.2432 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1070/Unknown  148s 101ms/step - categorical_accuracy: 0.6890 - loss: 1.2430 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1071/Unknown  148s 101ms/step - categorical_accuracy: 0.6890 - loss: 1.2428 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1072/Unknown  148s 101ms/step - categorical_accuracy: 0.6891 - loss: 1.2426 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1073/Unknown  148s 101ms/step - categorical_accuracy: 0.6891 - loss: 1.2424 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1074/Unknown  148s 101ms/step - categorical_accuracy: 0.6892 - loss: 1.2422 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1075/Unknown  148s 101ms/step - categorical_accuracy: 0.6892 - loss: 1.2419 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1076/Unknown  148s 101ms/step - categorical_accuracy: 0.6893 - loss: 1.2417 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1077/Unknown  148s 101ms/step - categorical_accuracy: 0.6893 - loss: 1.2415 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1078/Unknown  148s 101ms/step - categorical_accuracy: 0.6894 - loss: 1.2413 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1079/Unknown  148s 101ms/step - categorical_accuracy: 0.6894 - loss: 1.2411 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1080/Unknown  149s 101ms/step - categorical_accuracy: 0.6894 - loss: 1.2409 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1081/Unknown  149s 101ms/step - categorical_accuracy: 0.6895 - loss: 1.2407 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1082/Unknown  149s 101ms/step - categorical_accuracy: 0.6895 - loss: 1.2405 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1083/Unknown  149s 101ms/step - categorical_accuracy: 0.6896 - loss: 1.2402 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1084/Unknown  149s 101ms/step - categorical_accuracy: 0.6896 - loss: 1.2400 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1085/Unknown  149s 101ms/step - categorical_accuracy: 0.6897 - loss: 1.2398 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1086/Unknown  149s 101ms/step - categorical_accuracy: 0.6897 - loss: 1.2396 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1087/Unknown  149s 101ms/step - categorical_accuracy: 0.6898 - loss: 1.2394 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1088/Unknown  149s 101ms/step - categorical_accuracy: 0.6898 - loss: 1.2392 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1089/Unknown  149s 101ms/step - categorical_accuracy: 0.6899 - loss: 1.2390 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1090/Unknown  149s 101ms/step - categorical_accuracy: 0.6899 - loss: 1.2388 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1091/Unknown  149s 101ms/step - categorical_accuracy: 0.6899 - loss: 1.2386 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1092/Unknown  150s 101ms/step - categorical_accuracy: 0.6900 - loss: 1.2383 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1093/Unknown  150s 101ms/step - categorical_accuracy: 0.6900 - loss: 1.2381 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1094/Unknown  150s 101ms/step - categorical_accuracy: 0.6901 - loss: 1.2379 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1095/Unknown  150s 101ms/step - categorical_accuracy: 0.6901 - loss: 1.2377 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1096/Unknown  150s 101ms/step - categorical_accuracy: 0.6902 - loss: 1.2375 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1097/Unknown  150s 101ms/step - categorical_accuracy: 0.6902 - loss: 1.2373 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1098/Unknown  150s 101ms/step - categorical_accuracy: 0.6903 - loss: 1.2371 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1099/Unknown  150s 100ms/step - categorical_accuracy: 0.6903 - loss: 1.2369 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1100/Unknown  150s 100ms/step - categorical_accuracy: 0.6903 - loss: 1.2367 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1101/Unknown  150s 100ms/step - categorical_accuracy: 0.6904 - loss: 1.2365 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1102/Unknown  150s 100ms/step - categorical_accuracy: 0.6904 - loss: 1.2362 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1103/Unknown  150s 100ms/step - categorical_accuracy: 0.6905 - loss: 1.2360 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1104/Unknown  150s 100ms/step - categorical_accuracy: 0.6905 - loss: 1.2358 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1105/Unknown  150s 100ms/step - categorical_accuracy: 0.6906 - loss: 1.2356 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1106/Unknown  150s 100ms/step - categorical_accuracy: 0.6906 - loss: 1.2354 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1107/Unknown  150s 100ms/step - categorical_accuracy: 0.6907 - loss: 1.2352 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1108/Unknown  151s 100ms/step - categorical_accuracy: 0.6907 - loss: 1.2350 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1109/Unknown  151s 100ms/step - categorical_accuracy: 0.6907 - loss: 1.2348 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1110/Unknown  151s 100ms/step - categorical_accuracy: 0.6908 - loss: 1.2346 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1111/Unknown  151s 100ms/step - categorical_accuracy: 0.6908 - loss: 1.2344 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1112/Unknown  151s 100ms/step - categorical_accuracy: 0.6909 - loss: 1.2342 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1113/Unknown  151s 100ms/step - categorical_accuracy: 0.6909 - loss: 1.2340 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1114/Unknown  151s 100ms/step - categorical_accuracy: 0.6910 - loss: 1.2338 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1115/Unknown  151s 100ms/step - categorical_accuracy: 0.6910 - loss: 1.2336 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1116/Unknown  151s 100ms/step - categorical_accuracy: 0.6911 - loss: 1.2334 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1117/Unknown  151s 100ms/step - categorical_accuracy: 0.6911 - loss: 1.2332 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1118/Unknown  151s 100ms/step - categorical_accuracy: 0.6911 - loss: 1.2329 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1119/Unknown  151s 100ms/step - categorical_accuracy: 0.6912 - loss: 1.2327 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1120/Unknown  151s 100ms/step - categorical_accuracy: 0.6912 - loss: 1.2325 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1121/Unknown  152s 100ms/step - categorical_accuracy: 0.6913 - loss: 1.2323 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1122/Unknown  152s 100ms/step - categorical_accuracy: 0.6913 - loss: 1.2321 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1123/Unknown  152s 100ms/step - categorical_accuracy: 0.6914 - loss: 1.2319 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1124/Unknown  152s 100ms/step - categorical_accuracy: 0.6914 - loss: 1.2317 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1125/Unknown  152s 100ms/step - categorical_accuracy: 0.6914 - loss: 1.2315 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1126/Unknown  152s 100ms/step - categorical_accuracy: 0.6915 - loss: 1.2313 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1127/Unknown  152s 100ms/step - categorical_accuracy: 0.6915 - loss: 1.2311 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1128/Unknown  152s 100ms/step - categorical_accuracy: 0.6916 - loss: 1.2309 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1129/Unknown  152s 100ms/step - categorical_accuracy: 0.6916 - loss: 1.2307 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1130/Unknown  152s 100ms/step - categorical_accuracy: 0.6917 - loss: 1.2305 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1131/Unknown  152s 100ms/step - categorical_accuracy: 0.6917 - loss: 1.2303 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1132/Unknown  152s 100ms/step - categorical_accuracy: 0.6918 - loss: 1.2301 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1133/Unknown  152s 100ms/step - categorical_accuracy: 0.6918 - loss: 1.2299 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1134/Unknown  153s 100ms/step - categorical_accuracy: 0.6918 - loss: 1.2297 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1135/Unknown  153s 100ms/step - categorical_accuracy: 0.6919 - loss: 1.2295 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1136/Unknown  153s 100ms/step - categorical_accuracy: 0.6919 - loss: 1.2293 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1137/Unknown  153s 100ms/step - categorical_accuracy: 0.6920 - loss: 1.2291 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1138/Unknown  153s 100ms/step - categorical_accuracy: 0.6920 - loss: 1.2289 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1139/Unknown  153s 100ms/step - categorical_accuracy: 0.6921 - loss: 1.2287 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1140/Unknown  153s 100ms/step - categorical_accuracy: 0.6921 - loss: 1.2285 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1141/Unknown  153s 99ms/step - categorical_accuracy: 0.6921 - loss: 1.2283 - mean_io_u: 0.0700 

<div class="k-default-codeblock">
```

```
</div>
   1142/Unknown  153s 99ms/step - categorical_accuracy: 0.6922 - loss: 1.2281 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1143/Unknown  153s 99ms/step - categorical_accuracy: 0.6922 - loss: 1.2279 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1144/Unknown  153s 99ms/step - categorical_accuracy: 0.6923 - loss: 1.2277 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1145/Unknown  153s 99ms/step - categorical_accuracy: 0.6923 - loss: 1.2275 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1146/Unknown  153s 99ms/step - categorical_accuracy: 0.6924 - loss: 1.2273 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1147/Unknown  154s 99ms/step - categorical_accuracy: 0.6924 - loss: 1.2271 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1148/Unknown  154s 99ms/step - categorical_accuracy: 0.6924 - loss: 1.2269 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1149/Unknown  154s 99ms/step - categorical_accuracy: 0.6925 - loss: 1.2267 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1150/Unknown  154s 99ms/step - categorical_accuracy: 0.6925 - loss: 1.2265 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1151/Unknown  154s 99ms/step - categorical_accuracy: 0.6926 - loss: 1.2263 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1152/Unknown  154s 99ms/step - categorical_accuracy: 0.6926 - loss: 1.2261 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1153/Unknown  154s 99ms/step - categorical_accuracy: 0.6926 - loss: 1.2259 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1154/Unknown  154s 99ms/step - categorical_accuracy: 0.6927 - loss: 1.2257 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1155/Unknown  154s 99ms/step - categorical_accuracy: 0.6927 - loss: 1.2255 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1156/Unknown  154s 99ms/step - categorical_accuracy: 0.6928 - loss: 1.2254 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1157/Unknown  154s 99ms/step - categorical_accuracy: 0.6928 - loss: 1.2252 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1158/Unknown  154s 99ms/step - categorical_accuracy: 0.6929 - loss: 1.2250 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1159/Unknown  154s 99ms/step - categorical_accuracy: 0.6929 - loss: 1.2248 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1160/Unknown  155s 99ms/step - categorical_accuracy: 0.6929 - loss: 1.2246 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1161/Unknown  155s 99ms/step - categorical_accuracy: 0.6930 - loss: 1.2244 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1162/Unknown  155s 99ms/step - categorical_accuracy: 0.6930 - loss: 1.2242 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1163/Unknown  155s 99ms/step - categorical_accuracy: 0.6931 - loss: 1.2240 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1164/Unknown  155s 99ms/step - categorical_accuracy: 0.6931 - loss: 1.2238 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1165/Unknown  155s 99ms/step - categorical_accuracy: 0.6931 - loss: 1.2236 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1166/Unknown  155s 99ms/step - categorical_accuracy: 0.6932 - loss: 1.2234 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1167/Unknown  155s 99ms/step - categorical_accuracy: 0.6932 - loss: 1.2232 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1168/Unknown  155s 99ms/step - categorical_accuracy: 0.6933 - loss: 1.2230 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1169/Unknown  155s 99ms/step - categorical_accuracy: 0.6933 - loss: 1.2228 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1170/Unknown  155s 99ms/step - categorical_accuracy: 0.6934 - loss: 1.2226 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1171/Unknown  155s 99ms/step - categorical_accuracy: 0.6934 - loss: 1.2224 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1172/Unknown  156s 99ms/step - categorical_accuracy: 0.6934 - loss: 1.2222 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1173/Unknown  156s 99ms/step - categorical_accuracy: 0.6935 - loss: 1.2220 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1174/Unknown  156s 99ms/step - categorical_accuracy: 0.6935 - loss: 1.2219 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1175/Unknown  156s 99ms/step - categorical_accuracy: 0.6936 - loss: 1.2217 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1176/Unknown  156s 99ms/step - categorical_accuracy: 0.6936 - loss: 1.2215 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1177/Unknown  156s 99ms/step - categorical_accuracy: 0.6936 - loss: 1.2213 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1178/Unknown  156s 99ms/step - categorical_accuracy: 0.6937 - loss: 1.2211 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1179/Unknown  156s 99ms/step - categorical_accuracy: 0.6937 - loss: 1.2209 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1180/Unknown  156s 99ms/step - categorical_accuracy: 0.6938 - loss: 1.2207 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1181/Unknown  156s 99ms/step - categorical_accuracy: 0.6938 - loss: 1.2205 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1182/Unknown  156s 99ms/step - categorical_accuracy: 0.6938 - loss: 1.2203 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1183/Unknown  156s 99ms/step - categorical_accuracy: 0.6939 - loss: 1.2201 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1184/Unknown  157s 99ms/step - categorical_accuracy: 0.6939 - loss: 1.2200 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1185/Unknown  157s 99ms/step - categorical_accuracy: 0.6940 - loss: 1.2198 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1186/Unknown  157s 99ms/step - categorical_accuracy: 0.6940 - loss: 1.2196 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1187/Unknown  157s 99ms/step - categorical_accuracy: 0.6940 - loss: 1.2194 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1188/Unknown  157s 99ms/step - categorical_accuracy: 0.6941 - loss: 1.2192 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1189/Unknown  157s 99ms/step - categorical_accuracy: 0.6941 - loss: 1.2190 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1190/Unknown  157s 99ms/step - categorical_accuracy: 0.6942 - loss: 1.2188 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1191/Unknown  157s 99ms/step - categorical_accuracy: 0.6942 - loss: 1.2186 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1192/Unknown  157s 99ms/step - categorical_accuracy: 0.6942 - loss: 1.2184 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1193/Unknown  157s 99ms/step - categorical_accuracy: 0.6943 - loss: 1.2183 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1194/Unknown  157s 99ms/step - categorical_accuracy: 0.6943 - loss: 1.2181 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1195/Unknown  157s 99ms/step - categorical_accuracy: 0.6944 - loss: 1.2179 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1196/Unknown  157s 99ms/step - categorical_accuracy: 0.6944 - loss: 1.2177 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1197/Unknown  158s 99ms/step - categorical_accuracy: 0.6944 - loss: 1.2175 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1198/Unknown  158s 99ms/step - categorical_accuracy: 0.6945 - loss: 1.2173 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1199/Unknown  158s 99ms/step - categorical_accuracy: 0.6945 - loss: 1.2171 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1200/Unknown  158s 98ms/step - categorical_accuracy: 0.6946 - loss: 1.2170 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1201/Unknown  158s 98ms/step - categorical_accuracy: 0.6946 - loss: 1.2168 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1202/Unknown  158s 98ms/step - categorical_accuracy: 0.6946 - loss: 1.2166 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1203/Unknown  158s 98ms/step - categorical_accuracy: 0.6947 - loss: 1.2164 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1204/Unknown  158s 98ms/step - categorical_accuracy: 0.6947 - loss: 1.2162 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1205/Unknown  158s 98ms/step - categorical_accuracy: 0.6948 - loss: 1.2160 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1206/Unknown  158s 98ms/step - categorical_accuracy: 0.6948 - loss: 1.2159 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1207/Unknown  158s 98ms/step - categorical_accuracy: 0.6948 - loss: 1.2157 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1208/Unknown  158s 98ms/step - categorical_accuracy: 0.6949 - loss: 1.2155 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1209/Unknown  159s 98ms/step - categorical_accuracy: 0.6949 - loss: 1.2153 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1210/Unknown  159s 98ms/step - categorical_accuracy: 0.6950 - loss: 1.2151 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1211/Unknown  159s 98ms/step - categorical_accuracy: 0.6950 - loss: 1.2149 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1212/Unknown  159s 98ms/step - categorical_accuracy: 0.6950 - loss: 1.2148 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1213/Unknown  159s 98ms/step - categorical_accuracy: 0.6951 - loss: 1.2146 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1214/Unknown  159s 98ms/step - categorical_accuracy: 0.6951 - loss: 1.2144 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1215/Unknown  159s 98ms/step - categorical_accuracy: 0.6952 - loss: 1.2142 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1216/Unknown  159s 98ms/step - categorical_accuracy: 0.6952 - loss: 1.2140 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1217/Unknown  159s 98ms/step - categorical_accuracy: 0.6952 - loss: 1.2139 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1218/Unknown  159s 98ms/step - categorical_accuracy: 0.6953 - loss: 1.2137 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1219/Unknown  159s 98ms/step - categorical_accuracy: 0.6953 - loss: 1.2135 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1220/Unknown  159s 98ms/step - categorical_accuracy: 0.6953 - loss: 1.2133 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1221/Unknown  159s 98ms/step - categorical_accuracy: 0.6954 - loss: 1.2131 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1222/Unknown  160s 98ms/step - categorical_accuracy: 0.6954 - loss: 1.2130 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1223/Unknown  160s 98ms/step - categorical_accuracy: 0.6955 - loss: 1.2128 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1224/Unknown  160s 98ms/step - categorical_accuracy: 0.6955 - loss: 1.2126 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1225/Unknown  160s 98ms/step - categorical_accuracy: 0.6955 - loss: 1.2124 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1226/Unknown  160s 98ms/step - categorical_accuracy: 0.6956 - loss: 1.2122 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1227/Unknown  160s 98ms/step - categorical_accuracy: 0.6956 - loss: 1.2121 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1228/Unknown  160s 98ms/step - categorical_accuracy: 0.6956 - loss: 1.2119 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1229/Unknown  160s 98ms/step - categorical_accuracy: 0.6957 - loss: 1.2117 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1230/Unknown  160s 98ms/step - categorical_accuracy: 0.6957 - loss: 1.2115 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1231/Unknown  160s 98ms/step - categorical_accuracy: 0.6958 - loss: 1.2113 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1232/Unknown  160s 98ms/step - categorical_accuracy: 0.6958 - loss: 1.2112 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1233/Unknown  160s 98ms/step - categorical_accuracy: 0.6958 - loss: 1.2110 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1234/Unknown  160s 98ms/step - categorical_accuracy: 0.6959 - loss: 1.2108 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1235/Unknown  161s 98ms/step - categorical_accuracy: 0.6959 - loss: 1.2106 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1236/Unknown  161s 98ms/step - categorical_accuracy: 0.6960 - loss: 1.2105 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1237/Unknown  161s 98ms/step - categorical_accuracy: 0.6960 - loss: 1.2103 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1238/Unknown  161s 98ms/step - categorical_accuracy: 0.6960 - loss: 1.2101 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1239/Unknown  161s 98ms/step - categorical_accuracy: 0.6961 - loss: 1.2099 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1240/Unknown  161s 98ms/step - categorical_accuracy: 0.6961 - loss: 1.2097 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1241/Unknown  161s 98ms/step - categorical_accuracy: 0.6961 - loss: 1.2096 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1242/Unknown  161s 98ms/step - categorical_accuracy: 0.6962 - loss: 1.2094 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1243/Unknown  161s 98ms/step - categorical_accuracy: 0.6962 - loss: 1.2092 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1244/Unknown  161s 98ms/step - categorical_accuracy: 0.6963 - loss: 1.2090 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1245/Unknown  161s 98ms/step - categorical_accuracy: 0.6963 - loss: 1.2089 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1246/Unknown  161s 98ms/step - categorical_accuracy: 0.6963 - loss: 1.2087 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1247/Unknown  161s 98ms/step - categorical_accuracy: 0.6964 - loss: 1.2085 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1248/Unknown  161s 98ms/step - categorical_accuracy: 0.6964 - loss: 1.2083 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1249/Unknown  162s 98ms/step - categorical_accuracy: 0.6964 - loss: 1.2082 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1250/Unknown  162s 98ms/step - categorical_accuracy: 0.6965 - loss: 1.2080 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1251/Unknown  162s 98ms/step - categorical_accuracy: 0.6965 - loss: 1.2078 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1252/Unknown  162s 98ms/step - categorical_accuracy: 0.6966 - loss: 1.2076 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1253/Unknown  162s 98ms/step - categorical_accuracy: 0.6966 - loss: 1.2075 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1254/Unknown  162s 97ms/step - categorical_accuracy: 0.6966 - loss: 1.2073 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1255/Unknown  162s 97ms/step - categorical_accuracy: 0.6967 - loss: 1.2071 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1256/Unknown  162s 97ms/step - categorical_accuracy: 0.6967 - loss: 1.2069 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1257/Unknown  162s 97ms/step - categorical_accuracy: 0.6967 - loss: 1.2068 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1258/Unknown  162s 97ms/step - categorical_accuracy: 0.6968 - loss: 1.2066 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1259/Unknown  162s 97ms/step - categorical_accuracy: 0.6968 - loss: 1.2064 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1260/Unknown  162s 97ms/step - categorical_accuracy: 0.6968 - loss: 1.2062 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1261/Unknown  162s 97ms/step - categorical_accuracy: 0.6969 - loss: 1.2061 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1262/Unknown  162s 97ms/step - categorical_accuracy: 0.6969 - loss: 1.2059 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1263/Unknown  162s 97ms/step - categorical_accuracy: 0.6970 - loss: 1.2057 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1264/Unknown  163s 97ms/step - categorical_accuracy: 0.6970 - loss: 1.2056 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1265/Unknown  163s 97ms/step - categorical_accuracy: 0.6970 - loss: 1.2054 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1266/Unknown  163s 97ms/step - categorical_accuracy: 0.6971 - loss: 1.2052 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1267/Unknown  163s 97ms/step - categorical_accuracy: 0.6971 - loss: 1.2050 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1268/Unknown  163s 97ms/step - categorical_accuracy: 0.6971 - loss: 1.2049 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1269/Unknown  163s 97ms/step - categorical_accuracy: 0.6972 - loss: 1.2047 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1270/Unknown  163s 97ms/step - categorical_accuracy: 0.6972 - loss: 1.2045 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1271/Unknown  163s 97ms/step - categorical_accuracy: 0.6972 - loss: 1.2044 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1272/Unknown  163s 97ms/step - categorical_accuracy: 0.6973 - loss: 1.2042 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1273/Unknown  163s 97ms/step - categorical_accuracy: 0.6973 - loss: 1.2040 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1274/Unknown  163s 97ms/step - categorical_accuracy: 0.6974 - loss: 1.2038 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1275/Unknown  163s 97ms/step - categorical_accuracy: 0.6974 - loss: 1.2037 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1276/Unknown  163s 97ms/step - categorical_accuracy: 0.6974 - loss: 1.2035 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1277/Unknown  164s 97ms/step - categorical_accuracy: 0.6975 - loss: 1.2033 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1278/Unknown  164s 97ms/step - categorical_accuracy: 0.6975 - loss: 1.2032 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1279/Unknown  164s 97ms/step - categorical_accuracy: 0.6975 - loss: 1.2030 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1280/Unknown  164s 97ms/step - categorical_accuracy: 0.6976 - loss: 1.2028 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1281/Unknown  164s 97ms/step - categorical_accuracy: 0.6976 - loss: 1.2026 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1282/Unknown  164s 97ms/step - categorical_accuracy: 0.6976 - loss: 1.2025 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1283/Unknown  164s 97ms/step - categorical_accuracy: 0.6977 - loss: 1.2023 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1284/Unknown  164s 97ms/step - categorical_accuracy: 0.6977 - loss: 1.2021 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1285/Unknown  164s 97ms/step - categorical_accuracy: 0.6978 - loss: 1.2020 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1286/Unknown  164s 97ms/step - categorical_accuracy: 0.6978 - loss: 1.2018 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1287/Unknown  164s 97ms/step - categorical_accuracy: 0.6978 - loss: 1.2016 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1288/Unknown  164s 97ms/step - categorical_accuracy: 0.6979 - loss: 1.2015 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1289/Unknown  165s 97ms/step - categorical_accuracy: 0.6979 - loss: 1.2013 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1290/Unknown  165s 97ms/step - categorical_accuracy: 0.6979 - loss: 1.2011 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1291/Unknown  165s 97ms/step - categorical_accuracy: 0.6980 - loss: 1.2010 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1292/Unknown  165s 97ms/step - categorical_accuracy: 0.6980 - loss: 1.2008 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1293/Unknown  165s 97ms/step - categorical_accuracy: 0.6980 - loss: 1.2006 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1294/Unknown  165s 97ms/step - categorical_accuracy: 0.6981 - loss: 1.2005 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1295/Unknown  165s 97ms/step - categorical_accuracy: 0.6981 - loss: 1.2003 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1296/Unknown  165s 97ms/step - categorical_accuracy: 0.6981 - loss: 1.2001 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1297/Unknown  165s 97ms/step - categorical_accuracy: 0.6982 - loss: 1.2000 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1298/Unknown  165s 97ms/step - categorical_accuracy: 0.6982 - loss: 1.1998 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1299/Unknown  165s 97ms/step - categorical_accuracy: 0.6982 - loss: 1.1996 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1300/Unknown  165s 97ms/step - categorical_accuracy: 0.6983 - loss: 1.1995 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1301/Unknown  165s 97ms/step - categorical_accuracy: 0.6983 - loss: 1.1993 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1302/Unknown  166s 97ms/step - categorical_accuracy: 0.6984 - loss: 1.1991 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1303/Unknown  166s 97ms/step - categorical_accuracy: 0.6984 - loss: 1.1990 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1304/Unknown  166s 97ms/step - categorical_accuracy: 0.6984 - loss: 1.1988 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1305/Unknown  166s 97ms/step - categorical_accuracy: 0.6985 - loss: 1.1986 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1306/Unknown  166s 97ms/step - categorical_accuracy: 0.6985 - loss: 1.1985 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1307/Unknown  166s 97ms/step - categorical_accuracy: 0.6985 - loss: 1.1983 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1308/Unknown  166s 97ms/step - categorical_accuracy: 0.6986 - loss: 1.1982 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1309/Unknown  166s 97ms/step - categorical_accuracy: 0.6986 - loss: 1.1980 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1310/Unknown  166s 97ms/step - categorical_accuracy: 0.6986 - loss: 1.1978 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1311/Unknown  166s 97ms/step - categorical_accuracy: 0.6987 - loss: 1.1977 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1312/Unknown  166s 97ms/step - categorical_accuracy: 0.6987 - loss: 1.1975 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1313/Unknown  166s 97ms/step - categorical_accuracy: 0.6987 - loss: 1.1973 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1314/Unknown  166s 97ms/step - categorical_accuracy: 0.6988 - loss: 1.1972 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1315/Unknown  166s 96ms/step - categorical_accuracy: 0.6988 - loss: 1.1970 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1316/Unknown  167s 96ms/step - categorical_accuracy: 0.6988 - loss: 1.1968 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1317/Unknown  167s 96ms/step - categorical_accuracy: 0.6989 - loss: 1.1967 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1318/Unknown  167s 96ms/step - categorical_accuracy: 0.6989 - loss: 1.1965 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1319/Unknown  167s 96ms/step - categorical_accuracy: 0.6989 - loss: 1.1964 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1320/Unknown  167s 96ms/step - categorical_accuracy: 0.6990 - loss: 1.1962 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1321/Unknown  167s 96ms/step - categorical_accuracy: 0.6990 - loss: 1.1960 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1322/Unknown  167s 96ms/step - categorical_accuracy: 0.6990 - loss: 1.1959 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1323/Unknown  167s 96ms/step - categorical_accuracy: 0.6991 - loss: 1.1957 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1324/Unknown  167s 96ms/step - categorical_accuracy: 0.6991 - loss: 1.1955 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1325/Unknown  167s 96ms/step - categorical_accuracy: 0.6991 - loss: 1.1954 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1326/Unknown  167s 96ms/step - categorical_accuracy: 0.6992 - loss: 1.1952 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1327/Unknown  168s 96ms/step - categorical_accuracy: 0.6992 - loss: 1.1951 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1328/Unknown  168s 96ms/step - categorical_accuracy: 0.6992 - loss: 1.1949 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1329/Unknown  168s 96ms/step - categorical_accuracy: 0.6993 - loss: 1.1947 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1330/Unknown  168s 96ms/step - categorical_accuracy: 0.6993 - loss: 1.1946 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1331/Unknown  168s 96ms/step - categorical_accuracy: 0.6993 - loss: 1.1944 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1332/Unknown  168s 96ms/step - categorical_accuracy: 0.6994 - loss: 1.1943 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1333/Unknown  168s 96ms/step - categorical_accuracy: 0.6994 - loss: 1.1941 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1334/Unknown  168s 96ms/step - categorical_accuracy: 0.6994 - loss: 1.1940 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1335/Unknown  168s 96ms/step - categorical_accuracy: 0.6995 - loss: 1.1938 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1336/Unknown  168s 96ms/step - categorical_accuracy: 0.6995 - loss: 1.1936 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1337/Unknown  168s 96ms/step - categorical_accuracy: 0.6995 - loss: 1.1935 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1338/Unknown  168s 96ms/step - categorical_accuracy: 0.6996 - loss: 1.1933 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1339/Unknown  169s 96ms/step - categorical_accuracy: 0.6996 - loss: 1.1932 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1340/Unknown  169s 96ms/step - categorical_accuracy: 0.6996 - loss: 1.1930 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1341/Unknown  169s 96ms/step - categorical_accuracy: 0.6997 - loss: 1.1928 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1342/Unknown  169s 96ms/step - categorical_accuracy: 0.6997 - loss: 1.1927 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1343/Unknown  169s 96ms/step - categorical_accuracy: 0.6997 - loss: 1.1925 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1344/Unknown  169s 96ms/step - categorical_accuracy: 0.6998 - loss: 1.1924 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1345/Unknown  169s 96ms/step - categorical_accuracy: 0.6998 - loss: 1.1922 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1346/Unknown  169s 96ms/step - categorical_accuracy: 0.6998 - loss: 1.1921 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1347/Unknown  169s 96ms/step - categorical_accuracy: 0.6999 - loss: 1.1919 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1348/Unknown  169s 96ms/step - categorical_accuracy: 0.6999 - loss: 1.1917 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1349/Unknown  169s 96ms/step - categorical_accuracy: 0.6999 - loss: 1.1916 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1350/Unknown  169s 96ms/step - categorical_accuracy: 0.7000 - loss: 1.1914 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1351/Unknown  170s 96ms/step - categorical_accuracy: 0.7000 - loss: 1.1913 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1352/Unknown  170s 96ms/step - categorical_accuracy: 0.7000 - loss: 1.1911 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1353/Unknown  170s 96ms/step - categorical_accuracy: 0.7001 - loss: 1.1909 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1354/Unknown  170s 96ms/step - categorical_accuracy: 0.7001 - loss: 1.1908 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1355/Unknown  170s 96ms/step - categorical_accuracy: 0.7001 - loss: 1.1906 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1356/Unknown  170s 96ms/step - categorical_accuracy: 0.7002 - loss: 1.1905 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1357/Unknown  170s 96ms/step - categorical_accuracy: 0.7002 - loss: 1.1903 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1358/Unknown  170s 96ms/step - categorical_accuracy: 0.7002 - loss: 1.1902 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1359/Unknown  170s 96ms/step - categorical_accuracy: 0.7003 - loss: 1.1900 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1360/Unknown  170s 96ms/step - categorical_accuracy: 0.7003 - loss: 1.1899 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1361/Unknown  170s 96ms/step - categorical_accuracy: 0.7003 - loss: 1.1897 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1362/Unknown  170s 96ms/step - categorical_accuracy: 0.7004 - loss: 1.1895 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1363/Unknown  170s 96ms/step - categorical_accuracy: 0.7004 - loss: 1.1894 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1364/Unknown  171s 96ms/step - categorical_accuracy: 0.7004 - loss: 1.1892 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1365/Unknown  171s 96ms/step - categorical_accuracy: 0.7005 - loss: 1.1891 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1366/Unknown  171s 96ms/step - categorical_accuracy: 0.7005 - loss: 1.1889 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1367/Unknown  171s 96ms/step - categorical_accuracy: 0.7005 - loss: 1.1888 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1368/Unknown  171s 96ms/step - categorical_accuracy: 0.7006 - loss: 1.1886 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1369/Unknown  171s 96ms/step - categorical_accuracy: 0.7006 - loss: 1.1885 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1370/Unknown  171s 96ms/step - categorical_accuracy: 0.7006 - loss: 1.1883 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1371/Unknown  171s 96ms/step - categorical_accuracy: 0.7007 - loss: 1.1881 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1372/Unknown  171s 96ms/step - categorical_accuracy: 0.7007 - loss: 1.1880 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1373/Unknown  171s 96ms/step - categorical_accuracy: 0.7007 - loss: 1.1878 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1374/Unknown  171s 96ms/step - categorical_accuracy: 0.7008 - loss: 1.1877 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1375/Unknown  171s 96ms/step - categorical_accuracy: 0.7008 - loss: 1.1875 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1376/Unknown  171s 96ms/step - categorical_accuracy: 0.7008 - loss: 1.1874 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1377/Unknown  171s 96ms/step - categorical_accuracy: 0.7009 - loss: 1.1872 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1378/Unknown  172s 96ms/step - categorical_accuracy: 0.7009 - loss: 1.1871 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1379/Unknown  172s 96ms/step - categorical_accuracy: 0.7009 - loss: 1.1869 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1380/Unknown  172s 96ms/step - categorical_accuracy: 0.7010 - loss: 1.1868 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1381/Unknown  172s 96ms/step - categorical_accuracy: 0.7010 - loss: 1.1866 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1382/Unknown  172s 96ms/step - categorical_accuracy: 0.7010 - loss: 1.1865 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1383/Unknown  172s 96ms/step - categorical_accuracy: 0.7010 - loss: 1.1863 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1384/Unknown  172s 96ms/step - categorical_accuracy: 0.7011 - loss: 1.1862 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1385/Unknown  172s 96ms/step - categorical_accuracy: 0.7011 - loss: 1.1860 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1386/Unknown  172s 96ms/step - categorical_accuracy: 0.7011 - loss: 1.1859 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1387/Unknown  172s 96ms/step - categorical_accuracy: 0.7012 - loss: 1.1857 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1388/Unknown  172s 96ms/step - categorical_accuracy: 0.7012 - loss: 1.1855 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1389/Unknown  172s 96ms/step - categorical_accuracy: 0.7012 - loss: 1.1854 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1390/Unknown  172s 96ms/step - categorical_accuracy: 0.7013 - loss: 1.1852 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1391/Unknown  173s 96ms/step - categorical_accuracy: 0.7013 - loss: 1.1851 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1392/Unknown  173s 96ms/step - categorical_accuracy: 0.7013 - loss: 1.1849 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1393/Unknown  173s 96ms/step - categorical_accuracy: 0.7014 - loss: 1.1848 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1394/Unknown  173s 96ms/step - categorical_accuracy: 0.7014 - loss: 1.1846 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1395/Unknown  173s 96ms/step - categorical_accuracy: 0.7014 - loss: 1.1845 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1396/Unknown  173s 96ms/step - categorical_accuracy: 0.7015 - loss: 1.1843 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1397/Unknown  173s 95ms/step - categorical_accuracy: 0.7015 - loss: 1.1842 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1398/Unknown  173s 95ms/step - categorical_accuracy: 0.7015 - loss: 1.1840 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1399/Unknown  173s 95ms/step - categorical_accuracy: 0.7016 - loss: 1.1839 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1400/Unknown  173s 95ms/step - categorical_accuracy: 0.7016 - loss: 1.1837 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1401/Unknown  173s 95ms/step - categorical_accuracy: 0.7016 - loss: 1.1836 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1402/Unknown  173s 95ms/step - categorical_accuracy: 0.7016 - loss: 1.1834 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1403/Unknown  173s 95ms/step - categorical_accuracy: 0.7017 - loss: 1.1833 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1404/Unknown  173s 95ms/step - categorical_accuracy: 0.7017 - loss: 1.1831 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1405/Unknown  173s 95ms/step - categorical_accuracy: 0.7017 - loss: 1.1830 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1406/Unknown  174s 95ms/step - categorical_accuracy: 0.7018 - loss: 1.1828 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1407/Unknown  174s 95ms/step - categorical_accuracy: 0.7018 - loss: 1.1827 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1408/Unknown  174s 95ms/step - categorical_accuracy: 0.7018 - loss: 1.1825 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1409/Unknown  174s 95ms/step - categorical_accuracy: 0.7019 - loss: 1.1824 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1410/Unknown  174s 95ms/step - categorical_accuracy: 0.7019 - loss: 1.1822 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1411/Unknown  174s 95ms/step - categorical_accuracy: 0.7019 - loss: 1.1821 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1412/Unknown  174s 95ms/step - categorical_accuracy: 0.7020 - loss: 1.1819 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1413/Unknown  174s 95ms/step - categorical_accuracy: 0.7020 - loss: 1.1818 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1414/Unknown  174s 95ms/step - categorical_accuracy: 0.7020 - loss: 1.1817 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1415/Unknown  174s 95ms/step - categorical_accuracy: 0.7021 - loss: 1.1815 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1416/Unknown  174s 95ms/step - categorical_accuracy: 0.7021 - loss: 1.1814 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1417/Unknown  174s 95ms/step - categorical_accuracy: 0.7021 - loss: 1.1812 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1418/Unknown  174s 95ms/step - categorical_accuracy: 0.7021 - loss: 1.1811 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1419/Unknown  175s 95ms/step - categorical_accuracy: 0.7022 - loss: 1.1809 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1420/Unknown  175s 95ms/step - categorical_accuracy: 0.7022 - loss: 1.1808 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1421/Unknown  175s 95ms/step - categorical_accuracy: 0.7022 - loss: 1.1806 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1422/Unknown  175s 95ms/step - categorical_accuracy: 0.7023 - loss: 1.1805 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1423/Unknown  175s 95ms/step - categorical_accuracy: 0.7023 - loss: 1.1803 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1424/Unknown  175s 95ms/step - categorical_accuracy: 0.7023 - loss: 1.1802 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1425/Unknown  175s 95ms/step - categorical_accuracy: 0.7024 - loss: 1.1800 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1426/Unknown  175s 95ms/step - categorical_accuracy: 0.7024 - loss: 1.1799 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1427/Unknown  175s 95ms/step - categorical_accuracy: 0.7024 - loss: 1.1798 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1428/Unknown  175s 95ms/step - categorical_accuracy: 0.7024 - loss: 1.1796 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1429/Unknown  175s 95ms/step - categorical_accuracy: 0.7025 - loss: 1.1795 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1430/Unknown  175s 95ms/step - categorical_accuracy: 0.7025 - loss: 1.1793 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1431/Unknown  175s 95ms/step - categorical_accuracy: 0.7025 - loss: 1.1792 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1432/Unknown  176s 95ms/step - categorical_accuracy: 0.7026 - loss: 1.1790 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1433/Unknown  176s 95ms/step - categorical_accuracy: 0.7026 - loss: 1.1789 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1434/Unknown  176s 95ms/step - categorical_accuracy: 0.7026 - loss: 1.1787 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1435/Unknown  176s 95ms/step - categorical_accuracy: 0.7027 - loss: 1.1786 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1436/Unknown  176s 95ms/step - categorical_accuracy: 0.7027 - loss: 1.1784 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1437/Unknown  176s 95ms/step - categorical_accuracy: 0.7027 - loss: 1.1783 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1438/Unknown  176s 95ms/step - categorical_accuracy: 0.7028 - loss: 1.1782 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1439/Unknown  176s 95ms/step - categorical_accuracy: 0.7028 - loss: 1.1780 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1440/Unknown  176s 95ms/step - categorical_accuracy: 0.7028 - loss: 1.1779 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1441/Unknown  176s 95ms/step - categorical_accuracy: 0.7028 - loss: 1.1777 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1442/Unknown  176s 95ms/step - categorical_accuracy: 0.7029 - loss: 1.1776 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1443/Unknown  176s 95ms/step - categorical_accuracy: 0.7029 - loss: 1.1774 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1444/Unknown  176s 95ms/step - categorical_accuracy: 0.7029 - loss: 1.1773 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1445/Unknown  176s 95ms/step - categorical_accuracy: 0.7030 - loss: 1.1771 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1446/Unknown  177s 95ms/step - categorical_accuracy: 0.7030 - loss: 1.1770 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1447/Unknown  177s 95ms/step - categorical_accuracy: 0.7030 - loss: 1.1768 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1448/Unknown  177s 95ms/step - categorical_accuracy: 0.7031 - loss: 1.1767 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1449/Unknown  177s 95ms/step - categorical_accuracy: 0.7031 - loss: 1.1766 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1450/Unknown  177s 95ms/step - categorical_accuracy: 0.7031 - loss: 1.1764 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1451/Unknown  177s 95ms/step - categorical_accuracy: 0.7031 - loss: 1.1763 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1452/Unknown  177s 95ms/step - categorical_accuracy: 0.7032 - loss: 1.1761 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1453/Unknown  177s 95ms/step - categorical_accuracy: 0.7032 - loss: 1.1760 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1454/Unknown  177s 95ms/step - categorical_accuracy: 0.7032 - loss: 1.1758 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1455/Unknown  177s 95ms/step - categorical_accuracy: 0.7033 - loss: 1.1757 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1456/Unknown  177s 95ms/step - categorical_accuracy: 0.7033 - loss: 1.1756 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1457/Unknown  177s 95ms/step - categorical_accuracy: 0.7033 - loss: 1.1754 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1458/Unknown  177s 95ms/step - categorical_accuracy: 0.7034 - loss: 1.1753 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1459/Unknown  177s 95ms/step - categorical_accuracy: 0.7034 - loss: 1.1751 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1460/Unknown  178s 95ms/step - categorical_accuracy: 0.7034 - loss: 1.1750 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1461/Unknown  178s 94ms/step - categorical_accuracy: 0.7034 - loss: 1.1748 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1462/Unknown  178s 94ms/step - categorical_accuracy: 0.7035 - loss: 1.1747 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1463/Unknown  178s 94ms/step - categorical_accuracy: 0.7035 - loss: 1.1745 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1464/Unknown  178s 94ms/step - categorical_accuracy: 0.7035 - loss: 1.1744 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1465/Unknown  178s 94ms/step - categorical_accuracy: 0.7036 - loss: 1.1743 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1466/Unknown  178s 94ms/step - categorical_accuracy: 0.7036 - loss: 1.1741 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1467/Unknown  178s 94ms/step - categorical_accuracy: 0.7036 - loss: 1.1740 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1468/Unknown  178s 94ms/step - categorical_accuracy: 0.7037 - loss: 1.1738 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1469/Unknown  178s 94ms/step - categorical_accuracy: 0.7037 - loss: 1.1737 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1470/Unknown  178s 94ms/step - categorical_accuracy: 0.7037 - loss: 1.1736 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1471/Unknown  178s 94ms/step - categorical_accuracy: 0.7037 - loss: 1.1734 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1472/Unknown  179s 94ms/step - categorical_accuracy: 0.7038 - loss: 1.1733 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1473/Unknown  179s 94ms/step - categorical_accuracy: 0.7038 - loss: 1.1731 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1474/Unknown  179s 94ms/step - categorical_accuracy: 0.7038 - loss: 1.1730 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1475/Unknown  179s 94ms/step - categorical_accuracy: 0.7039 - loss: 1.1728 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1476/Unknown  179s 94ms/step - categorical_accuracy: 0.7039 - loss: 1.1727 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1477/Unknown  179s 94ms/step - categorical_accuracy: 0.7039 - loss: 1.1726 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1478/Unknown  179s 94ms/step - categorical_accuracy: 0.7039 - loss: 1.1724 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1479/Unknown  179s 94ms/step - categorical_accuracy: 0.7040 - loss: 1.1723 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1480/Unknown  179s 94ms/step - categorical_accuracy: 0.7040 - loss: 1.1721 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1481/Unknown  179s 94ms/step - categorical_accuracy: 0.7040 - loss: 1.1720 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1482/Unknown  179s 94ms/step - categorical_accuracy: 0.7041 - loss: 1.1719 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1483/Unknown  179s 94ms/step - categorical_accuracy: 0.7041 - loss: 1.1717 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1484/Unknown  179s 94ms/step - categorical_accuracy: 0.7041 - loss: 1.1716 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1485/Unknown  179s 94ms/step - categorical_accuracy: 0.7041 - loss: 1.1714 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1486/Unknown  180s 94ms/step - categorical_accuracy: 0.7042 - loss: 1.1713 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1487/Unknown  180s 94ms/step - categorical_accuracy: 0.7042 - loss: 1.1712 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1488/Unknown  180s 94ms/step - categorical_accuracy: 0.7042 - loss: 1.1710 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1489/Unknown  180s 94ms/step - categorical_accuracy: 0.7043 - loss: 1.1709 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1490/Unknown  180s 94ms/step - categorical_accuracy: 0.7043 - loss: 1.1707 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1491/Unknown  180s 94ms/step - categorical_accuracy: 0.7043 - loss: 1.1706 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1492/Unknown  180s 94ms/step - categorical_accuracy: 0.7043 - loss: 1.1705 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1493/Unknown  180s 94ms/step - categorical_accuracy: 0.7044 - loss: 1.1703 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1494/Unknown  180s 94ms/step - categorical_accuracy: 0.7044 - loss: 1.1702 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1495/Unknown  180s 94ms/step - categorical_accuracy: 0.7044 - loss: 1.1701 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1496/Unknown  180s 94ms/step - categorical_accuracy: 0.7045 - loss: 1.1699 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1497/Unknown  180s 94ms/step - categorical_accuracy: 0.7045 - loss: 1.1698 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1498/Unknown  181s 94ms/step - categorical_accuracy: 0.7045 - loss: 1.1696 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1499/Unknown  181s 94ms/step - categorical_accuracy: 0.7046 - loss: 1.1695 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1500/Unknown  181s 94ms/step - categorical_accuracy: 0.7046 - loss: 1.1694 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1501/Unknown  181s 94ms/step - categorical_accuracy: 0.7046 - loss: 1.1692 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1502/Unknown  181s 94ms/step - categorical_accuracy: 0.7046 - loss: 1.1691 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1503/Unknown  181s 94ms/step - categorical_accuracy: 0.7047 - loss: 1.1689 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1504/Unknown  181s 94ms/step - categorical_accuracy: 0.7047 - loss: 1.1688 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1505/Unknown  181s 94ms/step - categorical_accuracy: 0.7047 - loss: 1.1687 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1506/Unknown  181s 94ms/step - categorical_accuracy: 0.7047 - loss: 1.1685 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1507/Unknown  181s 94ms/step - categorical_accuracy: 0.7048 - loss: 1.1684 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1508/Unknown  181s 94ms/step - categorical_accuracy: 0.7048 - loss: 1.1683 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1509/Unknown  181s 94ms/step - categorical_accuracy: 0.7048 - loss: 1.1681 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1510/Unknown  181s 94ms/step - categorical_accuracy: 0.7049 - loss: 1.1680 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1511/Unknown  182s 94ms/step - categorical_accuracy: 0.7049 - loss: 1.1678 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1512/Unknown  182s 94ms/step - categorical_accuracy: 0.7049 - loss: 1.1677 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1513/Unknown  182s 94ms/step - categorical_accuracy: 0.7049 - loss: 1.1676 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1514/Unknown  182s 94ms/step - categorical_accuracy: 0.7050 - loss: 1.1674 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1515/Unknown  182s 94ms/step - categorical_accuracy: 0.7050 - loss: 1.1673 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1516/Unknown  182s 94ms/step - categorical_accuracy: 0.7050 - loss: 1.1672 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1517/Unknown  182s 94ms/step - categorical_accuracy: 0.7051 - loss: 1.1670 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1518/Unknown  182s 94ms/step - categorical_accuracy: 0.7051 - loss: 1.1669 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1519/Unknown  182s 94ms/step - categorical_accuracy: 0.7051 - loss: 1.1668 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1520/Unknown  182s 94ms/step - categorical_accuracy: 0.7051 - loss: 1.1666 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1521/Unknown  182s 94ms/step - categorical_accuracy: 0.7052 - loss: 1.1665 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1522/Unknown  182s 94ms/step - categorical_accuracy: 0.7052 - loss: 1.1664 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1523/Unknown  182s 94ms/step - categorical_accuracy: 0.7052 - loss: 1.1662 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1524/Unknown  182s 94ms/step - categorical_accuracy: 0.7053 - loss: 1.1661 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1525/Unknown  183s 94ms/step - categorical_accuracy: 0.7053 - loss: 1.1659 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1526/Unknown  183s 94ms/step - categorical_accuracy: 0.7053 - loss: 1.1658 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1527/Unknown  183s 94ms/step - categorical_accuracy: 0.7053 - loss: 1.1657 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1528/Unknown  183s 94ms/step - categorical_accuracy: 0.7054 - loss: 1.1655 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1529/Unknown  183s 94ms/step - categorical_accuracy: 0.7054 - loss: 1.1654 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1530/Unknown  183s 94ms/step - categorical_accuracy: 0.7054 - loss: 1.1653 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1531/Unknown  183s 94ms/step - categorical_accuracy: 0.7055 - loss: 1.1651 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1532/Unknown  183s 94ms/step - categorical_accuracy: 0.7055 - loss: 1.1650 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1533/Unknown  183s 94ms/step - categorical_accuracy: 0.7055 - loss: 1.1649 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1534/Unknown  183s 94ms/step - categorical_accuracy: 0.7055 - loss: 1.1647 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1535/Unknown  183s 94ms/step - categorical_accuracy: 0.7056 - loss: 1.1646 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1536/Unknown  183s 94ms/step - categorical_accuracy: 0.7056 - loss: 1.1645 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1537/Unknown  183s 94ms/step - categorical_accuracy: 0.7056 - loss: 1.1643 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1538/Unknown  184s 94ms/step - categorical_accuracy: 0.7056 - loss: 1.1642 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1539/Unknown  184s 94ms/step - categorical_accuracy: 0.7057 - loss: 1.1641 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1540/Unknown  184s 94ms/step - categorical_accuracy: 0.7057 - loss: 1.1639 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1541/Unknown  184s 94ms/step - categorical_accuracy: 0.7057 - loss: 1.1638 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1542/Unknown  184s 94ms/step - categorical_accuracy: 0.7058 - loss: 1.1637 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1543/Unknown  184s 94ms/step - categorical_accuracy: 0.7058 - loss: 1.1635 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1544/Unknown  184s 94ms/step - categorical_accuracy: 0.7058 - loss: 1.1634 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1545/Unknown  184s 94ms/step - categorical_accuracy: 0.7058 - loss: 1.1633 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1546/Unknown  184s 94ms/step - categorical_accuracy: 0.7059 - loss: 1.1631 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1547/Unknown  184s 94ms/step - categorical_accuracy: 0.7059 - loss: 1.1630 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1548/Unknown  184s 94ms/step - categorical_accuracy: 0.7059 - loss: 1.1629 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1549/Unknown  184s 94ms/step - categorical_accuracy: 0.7059 - loss: 1.1627 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1550/Unknown  184s 93ms/step - categorical_accuracy: 0.7060 - loss: 1.1626 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1551/Unknown  185s 93ms/step - categorical_accuracy: 0.7060 - loss: 1.1625 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1552/Unknown  185s 93ms/step - categorical_accuracy: 0.7060 - loss: 1.1623 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1553/Unknown  185s 93ms/step - categorical_accuracy: 0.7061 - loss: 1.1622 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1554/Unknown  185s 93ms/step - categorical_accuracy: 0.7061 - loss: 1.1621 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1555/Unknown  185s 93ms/step - categorical_accuracy: 0.7061 - loss: 1.1619 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1556/Unknown  185s 93ms/step - categorical_accuracy: 0.7061 - loss: 1.1618 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1557/Unknown  185s 93ms/step - categorical_accuracy: 0.7062 - loss: 1.1617 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1558/Unknown  185s 93ms/step - categorical_accuracy: 0.7062 - loss: 1.1615 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1559/Unknown  185s 93ms/step - categorical_accuracy: 0.7062 - loss: 1.1614 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1560/Unknown  185s 93ms/step - categorical_accuracy: 0.7062 - loss: 1.1613 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1561/Unknown  185s 93ms/step - categorical_accuracy: 0.7063 - loss: 1.1612 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1562/Unknown  185s 93ms/step - categorical_accuracy: 0.7063 - loss: 1.1610 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1563/Unknown  185s 93ms/step - categorical_accuracy: 0.7063 - loss: 1.1609 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1564/Unknown  185s 93ms/step - categorical_accuracy: 0.7064 - loss: 1.1608 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1565/Unknown  185s 93ms/step - categorical_accuracy: 0.7064 - loss: 1.1606 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1566/Unknown  186s 93ms/step - categorical_accuracy: 0.7064 - loss: 1.1605 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1567/Unknown  186s 93ms/step - categorical_accuracy: 0.7064 - loss: 1.1604 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1568/Unknown  186s 93ms/step - categorical_accuracy: 0.7065 - loss: 1.1602 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1569/Unknown  186s 93ms/step - categorical_accuracy: 0.7065 - loss: 1.1601 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1570/Unknown  186s 93ms/step - categorical_accuracy: 0.7065 - loss: 1.1600 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1571/Unknown  186s 93ms/step - categorical_accuracy: 0.7065 - loss: 1.1598 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1572/Unknown  186s 93ms/step - categorical_accuracy: 0.7066 - loss: 1.1597 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1573/Unknown  186s 93ms/step - categorical_accuracy: 0.7066 - loss: 1.1596 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1574/Unknown  186s 93ms/step - categorical_accuracy: 0.7066 - loss: 1.1595 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1575/Unknown  186s 93ms/step - categorical_accuracy: 0.7067 - loss: 1.1593 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1576/Unknown  186s 93ms/step - categorical_accuracy: 0.7067 - loss: 1.1592 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1577/Unknown  186s 93ms/step - categorical_accuracy: 0.7067 - loss: 1.1591 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1578/Unknown  186s 93ms/step - categorical_accuracy: 0.7067 - loss: 1.1589 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1579/Unknown  187s 93ms/step - categorical_accuracy: 0.7068 - loss: 1.1588 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1580/Unknown  187s 93ms/step - categorical_accuracy: 0.7068 - loss: 1.1587 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1581/Unknown  187s 93ms/step - categorical_accuracy: 0.7068 - loss: 1.1585 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1582/Unknown  187s 93ms/step - categorical_accuracy: 0.7068 - loss: 1.1584 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1583/Unknown  187s 93ms/step - categorical_accuracy: 0.7069 - loss: 1.1583 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1584/Unknown  187s 93ms/step - categorical_accuracy: 0.7069 - loss: 1.1582 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1585/Unknown  187s 93ms/step - categorical_accuracy: 0.7069 - loss: 1.1580 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1586/Unknown  187s 93ms/step - categorical_accuracy: 0.7069 - loss: 1.1579 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1587/Unknown  187s 93ms/step - categorical_accuracy: 0.7070 - loss: 1.1578 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1588/Unknown  187s 93ms/step - categorical_accuracy: 0.7070 - loss: 1.1576 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1589/Unknown  187s 93ms/step - categorical_accuracy: 0.7070 - loss: 1.1575 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1590/Unknown  187s 93ms/step - categorical_accuracy: 0.7071 - loss: 1.1574 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1591/Unknown  187s 93ms/step - categorical_accuracy: 0.7071 - loss: 1.1573 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1592/Unknown  188s 93ms/step - categorical_accuracy: 0.7071 - loss: 1.1571 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1593/Unknown  188s 93ms/step - categorical_accuracy: 0.7071 - loss: 1.1570 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1594/Unknown  188s 93ms/step - categorical_accuracy: 0.7072 - loss: 1.1569 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1595/Unknown  188s 93ms/step - categorical_accuracy: 0.7072 - loss: 1.1567 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1596/Unknown  188s 93ms/step - categorical_accuracy: 0.7072 - loss: 1.1566 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1597/Unknown  188s 93ms/step - categorical_accuracy: 0.7072 - loss: 1.1565 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1598/Unknown  188s 93ms/step - categorical_accuracy: 0.7073 - loss: 1.1564 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1599/Unknown  188s 93ms/step - categorical_accuracy: 0.7073 - loss: 1.1562 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1600/Unknown  188s 93ms/step - categorical_accuracy: 0.7073 - loss: 1.1561 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1601/Unknown  188s 93ms/step - categorical_accuracy: 0.7073 - loss: 1.1560 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1602/Unknown  188s 93ms/step - categorical_accuracy: 0.7074 - loss: 1.1558 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1603/Unknown  188s 93ms/step - categorical_accuracy: 0.7074 - loss: 1.1556 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1604/Unknown  188s 93ms/step - categorical_accuracy: 0.7074 - loss: 1.1556 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1605/Unknown  189s 93ms/step - categorical_accuracy: 0.7075 - loss: 1.1555 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1606/Unknown  189s 93ms/step - categorical_accuracy: 0.7075 - loss: 1.1553 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1607/Unknown  189s 93ms/step - categorical_accuracy: 0.7075 - loss: 1.1552 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1608/Unknown  189s 93ms/step - categorical_accuracy: 0.7075 - loss: 1.1551 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1609/Unknown  189s 93ms/step - categorical_accuracy: 0.7076 - loss: 1.1549 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1610/Unknown  189s 93ms/step - categorical_accuracy: 0.7076 - loss: 1.1548 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1611/Unknown  189s 93ms/step - categorical_accuracy: 0.7076 - loss: 1.1547 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1612/Unknown  189s 93ms/step - categorical_accuracy: 0.7076 - loss: 1.1546 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1613/Unknown  189s 93ms/step - categorical_accuracy: 0.7077 - loss: 1.1544 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1614/Unknown  189s 93ms/step - categorical_accuracy: 0.7077 - loss: 1.1543 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1615/Unknown  189s 93ms/step - categorical_accuracy: 0.7077 - loss: 1.1542 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1616/Unknown  189s 93ms/step - categorical_accuracy: 0.7077 - loss: 1.1541 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1617/Unknown  190s 93ms/step - categorical_accuracy: 0.7078 - loss: 1.1539 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1618/Unknown  190s 93ms/step - categorical_accuracy: 0.7078 - loss: 1.1538 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1619/Unknown  190s 93ms/step - categorical_accuracy: 0.7078 - loss: 1.1537 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1620/Unknown  190s 93ms/step - categorical_accuracy: 0.7078 - loss: 1.1535 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1621/Unknown  190s 93ms/step - categorical_accuracy: 0.7079 - loss: 1.1534 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1622/Unknown  190s 93ms/step - categorical_accuracy: 0.7079 - loss: 1.1533 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1623/Unknown  190s 93ms/step - categorical_accuracy: 0.7079 - loss: 1.1532 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1624/Unknown  190s 93ms/step - categorical_accuracy: 0.7080 - loss: 1.1530 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1625/Unknown  190s 93ms/step - categorical_accuracy: 0.7080 - loss: 1.1529 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1626/Unknown  190s 93ms/step - categorical_accuracy: 0.7080 - loss: 1.1528 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1627/Unknown  190s 93ms/step - categorical_accuracy: 0.7080 - loss: 1.1527 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1628/Unknown  190s 93ms/step - categorical_accuracy: 0.7081 - loss: 1.1525 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1629/Unknown  191s 93ms/step - categorical_accuracy: 0.7081 - loss: 1.1524 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1630/Unknown  191s 93ms/step - categorical_accuracy: 0.7081 - loss: 1.1523 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1631/Unknown  191s 93ms/step - categorical_accuracy: 0.7081 - loss: 1.1522 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1632/Unknown  191s 93ms/step - categorical_accuracy: 0.7082 - loss: 1.1520 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1633/Unknown  191s 93ms/step - categorical_accuracy: 0.7082 - loss: 1.1519 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1634/Unknown  191s 93ms/step - categorical_accuracy: 0.7082 - loss: 1.1518 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1635/Unknown  191s 93ms/step - categorical_accuracy: 0.7082 - loss: 1.1517 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1636/Unknown  191s 93ms/step - categorical_accuracy: 0.7083 - loss: 1.1515 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1637/Unknown  191s 93ms/step - categorical_accuracy: 0.7083 - loss: 1.1514 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1638/Unknown  191s 93ms/step - categorical_accuracy: 0.7083 - loss: 1.1513 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1639/Unknown  191s 93ms/step - categorical_accuracy: 0.7083 - loss: 1.1512 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1640/Unknown  191s 93ms/step - categorical_accuracy: 0.7084 - loss: 1.1510 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1641/Unknown  192s 93ms/step - categorical_accuracy: 0.7084 - loss: 1.1509 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1642/Unknown  192s 93ms/step - categorical_accuracy: 0.7084 - loss: 1.1508 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1643/Unknown  192s 93ms/step - categorical_accuracy: 0.7084 - loss: 1.1507 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1644/Unknown  192s 93ms/step - categorical_accuracy: 0.7085 - loss: 1.1505 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1645/Unknown  192s 93ms/step - categorical_accuracy: 0.7085 - loss: 1.1504 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1646/Unknown  192s 93ms/step - categorical_accuracy: 0.7085 - loss: 1.1503 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1647/Unknown  192s 93ms/step - categorical_accuracy: 0.7085 - loss: 1.1502 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1648/Unknown  192s 93ms/step - categorical_accuracy: 0.7086 - loss: 1.1500 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1649/Unknown  192s 93ms/step - categorical_accuracy: 0.7086 - loss: 1.1499 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1650/Unknown  192s 92ms/step - categorical_accuracy: 0.7086 - loss: 1.1498 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1651/Unknown  192s 92ms/step - categorical_accuracy: 0.7086 - loss: 1.1497 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1652/Unknown  192s 92ms/step - categorical_accuracy: 0.7087 - loss: 1.1495 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1653/Unknown  192s 92ms/step - categorical_accuracy: 0.7087 - loss: 1.1494 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1654/Unknown  192s 92ms/step - categorical_accuracy: 0.7087 - loss: 1.1493 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1655/Unknown  193s 92ms/step - categorical_accuracy: 0.7087 - loss: 1.1492 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1656/Unknown  193s 92ms/step - categorical_accuracy: 0.7088 - loss: 1.1491 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1657/Unknown  193s 92ms/step - categorical_accuracy: 0.7088 - loss: 1.1489 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1658/Unknown  193s 92ms/step - categorical_accuracy: 0.7088 - loss: 1.1488 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1659/Unknown  193s 92ms/step - categorical_accuracy: 0.7088 - loss: 1.1487 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1660/Unknown  193s 92ms/step - categorical_accuracy: 0.7089 - loss: 1.1486 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1661/Unknown  193s 92ms/step - categorical_accuracy: 0.7089 - loss: 1.1484 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1662/Unknown  193s 92ms/step - categorical_accuracy: 0.7089 - loss: 1.1483 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1663/Unknown  193s 92ms/step - categorical_accuracy: 0.7089 - loss: 1.1482 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1664/Unknown  193s 92ms/step - categorical_accuracy: 0.7090 - loss: 1.1481 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1665/Unknown  193s 92ms/step - categorical_accuracy: 0.7090 - loss: 1.1480 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1666/Unknown  193s 92ms/step - categorical_accuracy: 0.7090 - loss: 1.1478 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1667/Unknown  193s 92ms/step - categorical_accuracy: 0.7091 - loss: 1.1477 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1668/Unknown  194s 92ms/step - categorical_accuracy: 0.7091 - loss: 1.1476 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1669/Unknown  194s 92ms/step - categorical_accuracy: 0.7091 - loss: 1.1475 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1670/Unknown  194s 92ms/step - categorical_accuracy: 0.7091 - loss: 1.1473 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1671/Unknown  194s 92ms/step - categorical_accuracy: 0.7092 - loss: 1.1472 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1672/Unknown  194s 92ms/step - categorical_accuracy: 0.7092 - loss: 1.1471 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1673/Unknown  194s 92ms/step - categorical_accuracy: 0.7092 - loss: 1.1470 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1674/Unknown  194s 92ms/step - categorical_accuracy: 0.7092 - loss: 1.1469 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1675/Unknown  194s 92ms/step - categorical_accuracy: 0.7093 - loss: 1.1467 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1676/Unknown  194s 92ms/step - categorical_accuracy: 0.7093 - loss: 1.1466 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1677/Unknown  194s 92ms/step - categorical_accuracy: 0.7093 - loss: 1.1465 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1678/Unknown  194s 92ms/step - categorical_accuracy: 0.7093 - loss: 1.1464 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1679/Unknown  194s 92ms/step - categorical_accuracy: 0.7094 - loss: 1.1463 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1680/Unknown  194s 92ms/step - categorical_accuracy: 0.7094 - loss: 1.1461 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1681/Unknown  195s 92ms/step - categorical_accuracy: 0.7094 - loss: 1.1460 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1682/Unknown  195s 92ms/step - categorical_accuracy: 0.7094 - loss: 1.1459 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1683/Unknown  195s 92ms/step - categorical_accuracy: 0.7095 - loss: 1.1458 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1684/Unknown  195s 92ms/step - categorical_accuracy: 0.7095 - loss: 1.1456 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1685/Unknown  195s 92ms/step - categorical_accuracy: 0.7095 - loss: 1.1455 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1686/Unknown  195s 92ms/step - categorical_accuracy: 0.7095 - loss: 1.1454 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1687/Unknown  195s 92ms/step - categorical_accuracy: 0.7096 - loss: 1.1453 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1688/Unknown  195s 92ms/step - categorical_accuracy: 0.7096 - loss: 1.1452 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1689/Unknown  195s 92ms/step - categorical_accuracy: 0.7096 - loss: 1.1450 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1690/Unknown  195s 92ms/step - categorical_accuracy: 0.7096 - loss: 1.1449 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1691/Unknown  195s 92ms/step - categorical_accuracy: 0.7096 - loss: 1.1448 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1692/Unknown  195s 92ms/step - categorical_accuracy: 0.7097 - loss: 1.1447 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1693/Unknown  195s 92ms/step - categorical_accuracy: 0.7097 - loss: 1.1446 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1694/Unknown  195s 92ms/step - categorical_accuracy: 0.7097 - loss: 1.1444 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1695/Unknown  196s 92ms/step - categorical_accuracy: 0.7097 - loss: 1.1443 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1696/Unknown  196s 92ms/step - categorical_accuracy: 0.7098 - loss: 1.1442 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1697/Unknown  196s 92ms/step - categorical_accuracy: 0.7098 - loss: 1.1441 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1698/Unknown  196s 92ms/step - categorical_accuracy: 0.7098 - loss: 1.1440 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1699/Unknown  196s 92ms/step - categorical_accuracy: 0.7098 - loss: 1.1439 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1700/Unknown  196s 92ms/step - categorical_accuracy: 0.7099 - loss: 1.1437 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1701/Unknown  196s 92ms/step - categorical_accuracy: 0.7099 - loss: 1.1436 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1702/Unknown  196s 92ms/step - categorical_accuracy: 0.7099 - loss: 1.1435 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1703/Unknown  196s 92ms/step - categorical_accuracy: 0.7099 - loss: 1.1434 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1704/Unknown  196s 92ms/step - categorical_accuracy: 0.7100 - loss: 1.1433 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1705/Unknown  196s 92ms/step - categorical_accuracy: 0.7100 - loss: 1.1431 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1706/Unknown  196s 92ms/step - categorical_accuracy: 0.7100 - loss: 1.1430 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1707/Unknown  196s 92ms/step - categorical_accuracy: 0.7100 - loss: 1.1429 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1708/Unknown  196s 92ms/step - categorical_accuracy: 0.7101 - loss: 1.1428 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1709/Unknown  197s 92ms/step - categorical_accuracy: 0.7101 - loss: 1.1427 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1710/Unknown  197s 92ms/step - categorical_accuracy: 0.7101 - loss: 1.1426 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1711/Unknown  197s 92ms/step - categorical_accuracy: 0.7101 - loss: 1.1424 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1712/Unknown  197s 92ms/step - categorical_accuracy: 0.7102 - loss: 1.1423 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1713/Unknown  197s 92ms/step - categorical_accuracy: 0.7102 - loss: 1.1422 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1714/Unknown  197s 92ms/step - categorical_accuracy: 0.7102 - loss: 1.1421 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1715/Unknown  197s 92ms/step - categorical_accuracy: 0.7102 - loss: 1.1420 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1716/Unknown  197s 92ms/step - categorical_accuracy: 0.7103 - loss: 1.1419 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1717/Unknown  197s 92ms/step - categorical_accuracy: 0.7103 - loss: 1.1417 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1718/Unknown  197s 92ms/step - categorical_accuracy: 0.7103 - loss: 1.1416 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1719/Unknown  197s 92ms/step - categorical_accuracy: 0.7103 - loss: 1.1415 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1720/Unknown  197s 92ms/step - categorical_accuracy: 0.7103 - loss: 1.1414 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1721/Unknown  197s 92ms/step - categorical_accuracy: 0.7104 - loss: 1.1413 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1722/Unknown  197s 92ms/step - categorical_accuracy: 0.7104 - loss: 1.1412 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1723/Unknown  197s 92ms/step - categorical_accuracy: 0.7104 - loss: 1.1410 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1724/Unknown  198s 92ms/step - categorical_accuracy: 0.7104 - loss: 1.1409 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1725/Unknown  198s 92ms/step - categorical_accuracy: 0.7105 - loss: 1.1408 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1726/Unknown  198s 92ms/step - categorical_accuracy: 0.7105 - loss: 1.1407 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1727/Unknown  198s 92ms/step - categorical_accuracy: 0.7105 - loss: 1.1406 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1728/Unknown  198s 92ms/step - categorical_accuracy: 0.7105 - loss: 1.1405 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1729/Unknown  198s 92ms/step - categorical_accuracy: 0.7106 - loss: 1.1404 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1730/Unknown  198s 92ms/step - categorical_accuracy: 0.7106 - loss: 1.1402 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1731/Unknown  198s 92ms/step - categorical_accuracy: 0.7106 - loss: 1.1401 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1732/Unknown  198s 92ms/step - categorical_accuracy: 0.7106 - loss: 1.1400 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1733/Unknown  198s 92ms/step - categorical_accuracy: 0.7107 - loss: 1.1399 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1734/Unknown  198s 92ms/step - categorical_accuracy: 0.7107 - loss: 1.1398 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1735/Unknown  198s 91ms/step - categorical_accuracy: 0.7107 - loss: 1.1397 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1736/Unknown  198s 91ms/step - categorical_accuracy: 0.7107 - loss: 1.1396 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1737/Unknown  198s 91ms/step - categorical_accuracy: 0.7108 - loss: 1.1394 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1738/Unknown  199s 91ms/step - categorical_accuracy: 0.7108 - loss: 1.1393 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1739/Unknown  199s 91ms/step - categorical_accuracy: 0.7108 - loss: 1.1392 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1740/Unknown  199s 91ms/step - categorical_accuracy: 0.7108 - loss: 1.1391 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1741/Unknown  199s 91ms/step - categorical_accuracy: 0.7108 - loss: 1.1390 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1742/Unknown  199s 91ms/step - categorical_accuracy: 0.7109 - loss: 1.1389 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1743/Unknown  199s 91ms/step - categorical_accuracy: 0.7109 - loss: 1.1388 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1744/Unknown  199s 91ms/step - categorical_accuracy: 0.7109 - loss: 1.1386 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1745/Unknown  199s 91ms/step - categorical_accuracy: 0.7109 - loss: 1.1385 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1746/Unknown  199s 91ms/step - categorical_accuracy: 0.7110 - loss: 1.1384 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1747/Unknown  199s 91ms/step - categorical_accuracy: 0.7110 - loss: 1.1383 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1748/Unknown  199s 91ms/step - categorical_accuracy: 0.7110 - loss: 1.1382 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1749/Unknown  199s 91ms/step - categorical_accuracy: 0.7110 - loss: 1.1381 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1750/Unknown  199s 91ms/step - categorical_accuracy: 0.7111 - loss: 1.1380 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1751/Unknown  200s 91ms/step - categorical_accuracy: 0.7111 - loss: 1.1378 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1752/Unknown  200s 91ms/step - categorical_accuracy: 0.7111 - loss: 1.1377 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1753/Unknown  200s 91ms/step - categorical_accuracy: 0.7111 - loss: 1.1376 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1754/Unknown  200s 91ms/step - categorical_accuracy: 0.7111 - loss: 1.1375 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1755/Unknown  200s 91ms/step - categorical_accuracy: 0.7112 - loss: 1.1374 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1756/Unknown  200s 91ms/step - categorical_accuracy: 0.7112 - loss: 1.1373 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1757/Unknown  200s 91ms/step - categorical_accuracy: 0.7112 - loss: 1.1372 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1758/Unknown  200s 91ms/step - categorical_accuracy: 0.7112 - loss: 1.1371 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1759/Unknown  200s 91ms/step - categorical_accuracy: 0.7113 - loss: 1.1369 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1760/Unknown  200s 91ms/step - categorical_accuracy: 0.7113 - loss: 1.1368 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1761/Unknown  200s 91ms/step - categorical_accuracy: 0.7113 - loss: 1.1367 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1762/Unknown  200s 91ms/step - categorical_accuracy: 0.7113 - loss: 1.1366 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1763/Unknown  201s 91ms/step - categorical_accuracy: 0.7114 - loss: 1.1365 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1764/Unknown  201s 91ms/step - categorical_accuracy: 0.7114 - loss: 1.1364 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1765/Unknown  201s 91ms/step - categorical_accuracy: 0.7114 - loss: 1.1363 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1766/Unknown  201s 91ms/step - categorical_accuracy: 0.7114 - loss: 1.1362 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1767/Unknown  201s 91ms/step - categorical_accuracy: 0.7114 - loss: 1.1360 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1768/Unknown  201s 91ms/step - categorical_accuracy: 0.7115 - loss: 1.1359 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1769/Unknown  201s 91ms/step - categorical_accuracy: 0.7115 - loss: 1.1358 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1770/Unknown  201s 91ms/step - categorical_accuracy: 0.7115 - loss: 1.1357 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1771/Unknown  201s 91ms/step - categorical_accuracy: 0.7115 - loss: 1.1356 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1772/Unknown  201s 91ms/step - categorical_accuracy: 0.7116 - loss: 1.1355 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1773/Unknown  201s 91ms/step - categorical_accuracy: 0.7116 - loss: 1.1354 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1774/Unknown  201s 91ms/step - categorical_accuracy: 0.7116 - loss: 1.1353 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1775/Unknown  202s 91ms/step - categorical_accuracy: 0.7116 - loss: 1.1351 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1776/Unknown  202s 91ms/step - categorical_accuracy: 0.7117 - loss: 1.1350 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1777/Unknown  202s 91ms/step - categorical_accuracy: 0.7117 - loss: 1.1349 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1778/Unknown  202s 91ms/step - categorical_accuracy: 0.7117 - loss: 1.1348 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1779/Unknown  202s 91ms/step - categorical_accuracy: 0.7117 - loss: 1.1347 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1780/Unknown  202s 91ms/step - categorical_accuracy: 0.7117 - loss: 1.1346 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1781/Unknown  202s 91ms/step - categorical_accuracy: 0.7118 - loss: 1.1345 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1782/Unknown  202s 91ms/step - categorical_accuracy: 0.7118 - loss: 1.1344 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1783/Unknown  202s 91ms/step - categorical_accuracy: 0.7118 - loss: 1.1343 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1784/Unknown  202s 91ms/step - categorical_accuracy: 0.7118 - loss: 1.1341 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1785/Unknown  202s 91ms/step - categorical_accuracy: 0.7119 - loss: 1.1340 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1786/Unknown  202s 91ms/step - categorical_accuracy: 0.7119 - loss: 1.1339 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1787/Unknown  202s 91ms/step - categorical_accuracy: 0.7119 - loss: 1.1338 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1788/Unknown  203s 91ms/step - categorical_accuracy: 0.7119 - loss: 1.1337 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1789/Unknown  203s 91ms/step - categorical_accuracy: 0.7120 - loss: 1.1336 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1790/Unknown  203s 91ms/step - categorical_accuracy: 0.7120 - loss: 1.1335 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1791/Unknown  203s 91ms/step - categorical_accuracy: 0.7120 - loss: 1.1334 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1792/Unknown  203s 91ms/step - categorical_accuracy: 0.7120 - loss: 1.1333 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1793/Unknown  203s 91ms/step - categorical_accuracy: 0.7120 - loss: 1.1331 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1794/Unknown  203s 91ms/step - categorical_accuracy: 0.7121 - loss: 1.1330 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1795/Unknown  203s 91ms/step - categorical_accuracy: 0.7121 - loss: 1.1329 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1796/Unknown  203s 91ms/step - categorical_accuracy: 0.7121 - loss: 1.1328 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1797/Unknown  203s 91ms/step - categorical_accuracy: 0.7121 - loss: 1.1327 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1798/Unknown  203s 91ms/step - categorical_accuracy: 0.7122 - loss: 1.1326 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1799/Unknown  203s 91ms/step - categorical_accuracy: 0.7122 - loss: 1.1325 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1800/Unknown  204s 91ms/step - categorical_accuracy: 0.7122 - loss: 1.1324 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1801/Unknown  204s 91ms/step - categorical_accuracy: 0.7122 - loss: 1.1323 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1802/Unknown  204s 91ms/step - categorical_accuracy: 0.7122 - loss: 1.1322 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1803/Unknown  204s 91ms/step - categorical_accuracy: 0.7123 - loss: 1.1320 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1804/Unknown  204s 91ms/step - categorical_accuracy: 0.7123 - loss: 1.1319 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1805/Unknown  204s 91ms/step - categorical_accuracy: 0.7123 - loss: 1.1318 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1806/Unknown  204s 91ms/step - categorical_accuracy: 0.7123 - loss: 1.1317 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1807/Unknown  204s 91ms/step - categorical_accuracy: 0.7124 - loss: 1.1316 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1808/Unknown  204s 91ms/step - categorical_accuracy: 0.7124 - loss: 1.1315 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1809/Unknown  204s 91ms/step - categorical_accuracy: 0.7124 - loss: 1.1314 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1810/Unknown  204s 91ms/step - categorical_accuracy: 0.7124 - loss: 1.1313 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1811/Unknown  204s 91ms/step - categorical_accuracy: 0.7125 - loss: 1.1312 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1812/Unknown  205s 91ms/step - categorical_accuracy: 0.7125 - loss: 1.1311 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1813/Unknown  205s 91ms/step - categorical_accuracy: 0.7125 - loss: 1.1309 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1814/Unknown  205s 91ms/step - categorical_accuracy: 0.7125 - loss: 1.1308 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1815/Unknown  205s 91ms/step - categorical_accuracy: 0.7125 - loss: 1.1307 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1816/Unknown  205s 91ms/step - categorical_accuracy: 0.7126 - loss: 1.1306 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1817/Unknown  205s 91ms/step - categorical_accuracy: 0.7126 - loss: 1.1305 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1818/Unknown  205s 91ms/step - categorical_accuracy: 0.7126 - loss: 1.1304 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1819/Unknown  205s 91ms/step - categorical_accuracy: 0.7126 - loss: 1.1303 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1820/Unknown  205s 91ms/step - categorical_accuracy: 0.7127 - loss: 1.1302 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1821/Unknown  205s 91ms/step - categorical_accuracy: 0.7127 - loss: 1.1301 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1822/Unknown  205s 91ms/step - categorical_accuracy: 0.7127 - loss: 1.1300 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1823/Unknown  205s 91ms/step - categorical_accuracy: 0.7127 - loss: 1.1299 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1824/Unknown  205s 91ms/step - categorical_accuracy: 0.7127 - loss: 1.1298 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1825/Unknown  206s 91ms/step - categorical_accuracy: 0.7128 - loss: 1.1296 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1826/Unknown  206s 91ms/step - categorical_accuracy: 0.7128 - loss: 1.1295 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1827/Unknown  206s 91ms/step - categorical_accuracy: 0.7128 - loss: 1.1294 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1828/Unknown  206s 91ms/step - categorical_accuracy: 0.7128 - loss: 1.1293 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1829/Unknown  206s 91ms/step - categorical_accuracy: 0.7129 - loss: 1.1292 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1830/Unknown  206s 91ms/step - categorical_accuracy: 0.7129 - loss: 1.1291 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1831/Unknown  206s 91ms/step - categorical_accuracy: 0.7129 - loss: 1.1290 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1832/Unknown  206s 91ms/step - categorical_accuracy: 0.7129 - loss: 1.1289 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1833/Unknown  206s 91ms/step - categorical_accuracy: 0.7129 - loss: 1.1288 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1834/Unknown  206s 91ms/step - categorical_accuracy: 0.7130 - loss: 1.1287 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1835/Unknown  206s 91ms/step - categorical_accuracy: 0.7130 - loss: 1.1286 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1836/Unknown  206s 91ms/step - categorical_accuracy: 0.7130 - loss: 1.1285 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1837/Unknown  206s 91ms/step - categorical_accuracy: 0.7130 - loss: 1.1283 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1838/Unknown  207s 91ms/step - categorical_accuracy: 0.7131 - loss: 1.1282 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1839/Unknown  207s 91ms/step - categorical_accuracy: 0.7131 - loss: 1.1281 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1840/Unknown  207s 91ms/step - categorical_accuracy: 0.7131 - loss: 1.1280 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1841/Unknown  207s 91ms/step - categorical_accuracy: 0.7131 - loss: 1.1279 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1842/Unknown  207s 91ms/step - categorical_accuracy: 0.7131 - loss: 1.1278 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1843/Unknown  207s 91ms/step - categorical_accuracy: 0.7132 - loss: 1.1277 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1844/Unknown  207s 91ms/step - categorical_accuracy: 0.7132 - loss: 1.1276 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1845/Unknown  207s 91ms/step - categorical_accuracy: 0.7132 - loss: 1.1275 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1846/Unknown  207s 91ms/step - categorical_accuracy: 0.7132 - loss: 1.1274 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1847/Unknown  207s 91ms/step - categorical_accuracy: 0.7133 - loss: 1.1273 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1848/Unknown  207s 91ms/step - categorical_accuracy: 0.7133 - loss: 1.1272 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1849/Unknown  207s 91ms/step - categorical_accuracy: 0.7133 - loss: 1.1271 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1850/Unknown  207s 91ms/step - categorical_accuracy: 0.7133 - loss: 1.1269 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1851/Unknown  208s 91ms/step - categorical_accuracy: 0.7133 - loss: 1.1268 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1852/Unknown  208s 91ms/step - categorical_accuracy: 0.7134 - loss: 1.1267 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1853/Unknown  208s 91ms/step - categorical_accuracy: 0.7134 - loss: 1.1266 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1854/Unknown  208s 91ms/step - categorical_accuracy: 0.7134 - loss: 1.1265 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1855/Unknown  208s 91ms/step - categorical_accuracy: 0.7134 - loss: 1.1264 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1856/Unknown  208s 91ms/step - categorical_accuracy: 0.7135 - loss: 1.1263 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1857/Unknown  208s 91ms/step - categorical_accuracy: 0.7135 - loss: 1.1262 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1858/Unknown  208s 91ms/step - categorical_accuracy: 0.7135 - loss: 1.1261 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1859/Unknown  208s 91ms/step - categorical_accuracy: 0.7135 - loss: 1.1260 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1860/Unknown  208s 91ms/step - categorical_accuracy: 0.7135 - loss: 1.1259 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1861/Unknown  208s 91ms/step - categorical_accuracy: 0.7136 - loss: 1.1258 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1862/Unknown  208s 91ms/step - categorical_accuracy: 0.7136 - loss: 1.1257 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1863/Unknown  208s 91ms/step - categorical_accuracy: 0.7136 - loss: 1.1256 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1864/Unknown  208s 91ms/step - categorical_accuracy: 0.7136 - loss: 1.1254 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1865/Unknown  209s 91ms/step - categorical_accuracy: 0.7137 - loss: 1.1253 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1866/Unknown  209s 91ms/step - categorical_accuracy: 0.7137 - loss: 1.1252 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1867/Unknown  209s 91ms/step - categorical_accuracy: 0.7137 - loss: 1.1251 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1868/Unknown  209s 91ms/step - categorical_accuracy: 0.7137 - loss: 1.1250 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1869/Unknown  209s 91ms/step - categorical_accuracy: 0.7137 - loss: 1.1249 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1870/Unknown  209s 91ms/step - categorical_accuracy: 0.7138 - loss: 1.1248 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1871/Unknown  209s 90ms/step - categorical_accuracy: 0.7138 - loss: 1.1247 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1872/Unknown  209s 90ms/step - categorical_accuracy: 0.7138 - loss: 1.1246 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1873/Unknown  209s 90ms/step - categorical_accuracy: 0.7138 - loss: 1.1245 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1874/Unknown  209s 90ms/step - categorical_accuracy: 0.7139 - loss: 1.1244 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1875/Unknown  209s 90ms/step - categorical_accuracy: 0.7139 - loss: 1.1243 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1876/Unknown  209s 90ms/step - categorical_accuracy: 0.7139 - loss: 1.1242 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1877/Unknown  209s 90ms/step - categorical_accuracy: 0.7139 - loss: 1.1241 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1878/Unknown  209s 90ms/step - categorical_accuracy: 0.7139 - loss: 1.1240 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1879/Unknown  209s 90ms/step - categorical_accuracy: 0.7140 - loss: 1.1239 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1880/Unknown  210s 90ms/step - categorical_accuracy: 0.7140 - loss: 1.1238 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1881/Unknown  210s 90ms/step - categorical_accuracy: 0.7140 - loss: 1.1237 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1882/Unknown  210s 90ms/step - categorical_accuracy: 0.7140 - loss: 1.1235 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1883/Unknown  210s 90ms/step - categorical_accuracy: 0.7140 - loss: 1.1234 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1884/Unknown  210s 90ms/step - categorical_accuracy: 0.7141 - loss: 1.1233 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1885/Unknown  210s 90ms/step - categorical_accuracy: 0.7141 - loss: 1.1232 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1886/Unknown  210s 90ms/step - categorical_accuracy: 0.7141 - loss: 1.1231 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1887/Unknown  210s 90ms/step - categorical_accuracy: 0.7141 - loss: 1.1230 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1888/Unknown  210s 90ms/step - categorical_accuracy: 0.7142 - loss: 1.1229 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1889/Unknown  210s 90ms/step - categorical_accuracy: 0.7142 - loss: 1.1228 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1890/Unknown  210s 90ms/step - categorical_accuracy: 0.7142 - loss: 1.1227 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1891/Unknown  210s 90ms/step - categorical_accuracy: 0.7142 - loss: 1.1226 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1892/Unknown  210s 90ms/step - categorical_accuracy: 0.7142 - loss: 1.1225 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1893/Unknown  211s 90ms/step - categorical_accuracy: 0.7143 - loss: 1.1224 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1894/Unknown  211s 90ms/step - categorical_accuracy: 0.7143 - loss: 1.1223 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1895/Unknown  211s 90ms/step - categorical_accuracy: 0.7143 - loss: 1.1222 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1896/Unknown  211s 90ms/step - categorical_accuracy: 0.7143 - loss: 1.1221 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1897/Unknown  211s 90ms/step - categorical_accuracy: 0.7143 - loss: 1.1220 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1898/Unknown  211s 90ms/step - categorical_accuracy: 0.7144 - loss: 1.1219 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1899/Unknown  211s 90ms/step - categorical_accuracy: 0.7144 - loss: 1.1218 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1900/Unknown  211s 90ms/step - categorical_accuracy: 0.7144 - loss: 1.1217 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1901/Unknown  211s 90ms/step - categorical_accuracy: 0.7144 - loss: 1.1216 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1902/Unknown  211s 90ms/step - categorical_accuracy: 0.7145 - loss: 1.1215 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1903/Unknown  211s 90ms/step - categorical_accuracy: 0.7145 - loss: 1.1214 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1904/Unknown  211s 90ms/step - categorical_accuracy: 0.7145 - loss: 1.1213 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1905/Unknown  211s 90ms/step - categorical_accuracy: 0.7145 - loss: 1.1212 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1906/Unknown  212s 90ms/step - categorical_accuracy: 0.7145 - loss: 1.1211 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1907/Unknown  212s 90ms/step - categorical_accuracy: 0.7146 - loss: 1.1210 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1908/Unknown  212s 90ms/step - categorical_accuracy: 0.7146 - loss: 1.1208 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1909/Unknown  212s 90ms/step - categorical_accuracy: 0.7146 - loss: 1.1207 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1910/Unknown  212s 90ms/step - categorical_accuracy: 0.7146 - loss: 1.1206 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1911/Unknown  212s 90ms/step - categorical_accuracy: 0.7146 - loss: 1.1205 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1912/Unknown  212s 90ms/step - categorical_accuracy: 0.7147 - loss: 1.1204 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1913/Unknown  212s 90ms/step - categorical_accuracy: 0.7147 - loss: 1.1203 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1914/Unknown  212s 90ms/step - categorical_accuracy: 0.7147 - loss: 1.1202 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1915/Unknown  212s 90ms/step - categorical_accuracy: 0.7147 - loss: 1.1201 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1916/Unknown  212s 90ms/step - categorical_accuracy: 0.7148 - loss: 1.1200 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1917/Unknown  212s 90ms/step - categorical_accuracy: 0.7148 - loss: 1.1199 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1918/Unknown  213s 90ms/step - categorical_accuracy: 0.7148 - loss: 1.1198 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1919/Unknown  213s 90ms/step - categorical_accuracy: 0.7148 - loss: 1.1197 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1920/Unknown  213s 90ms/step - categorical_accuracy: 0.7148 - loss: 1.1196 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1921/Unknown  213s 90ms/step - categorical_accuracy: 0.7149 - loss: 1.1195 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1922/Unknown  213s 90ms/step - categorical_accuracy: 0.7149 - loss: 1.1194 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1923/Unknown  213s 90ms/step - categorical_accuracy: 0.7149 - loss: 1.1193 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1924/Unknown  213s 90ms/step - categorical_accuracy: 0.7149 - loss: 1.1192 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1925/Unknown  213s 90ms/step - categorical_accuracy: 0.7149 - loss: 1.1191 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1926/Unknown  213s 90ms/step - categorical_accuracy: 0.7150 - loss: 1.1190 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1927/Unknown  213s 90ms/step - categorical_accuracy: 0.7150 - loss: 1.1189 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1928/Unknown  213s 90ms/step - categorical_accuracy: 0.7150 - loss: 1.1188 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1929/Unknown  213s 90ms/step - categorical_accuracy: 0.7150 - loss: 1.1187 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1930/Unknown  213s 90ms/step - categorical_accuracy: 0.7150 - loss: 1.1186 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1931/Unknown  214s 90ms/step - categorical_accuracy: 0.7151 - loss: 1.1185 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1932/Unknown  214s 90ms/step - categorical_accuracy: 0.7151 - loss: 1.1184 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1933/Unknown  214s 90ms/step - categorical_accuracy: 0.7151 - loss: 1.1183 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1934/Unknown  214s 90ms/step - categorical_accuracy: 0.7151 - loss: 1.1182 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1935/Unknown  214s 90ms/step - categorical_accuracy: 0.7151 - loss: 1.1181 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1936/Unknown  214s 90ms/step - categorical_accuracy: 0.7152 - loss: 1.1180 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1937/Unknown  214s 90ms/step - categorical_accuracy: 0.7152 - loss: 1.1179 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1938/Unknown  214s 90ms/step - categorical_accuracy: 0.7152 - loss: 1.1178 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1939/Unknown  214s 90ms/step - categorical_accuracy: 0.7152 - loss: 1.1177 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1940/Unknown  214s 90ms/step - categorical_accuracy: 0.7153 - loss: 1.1176 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1941/Unknown  214s 90ms/step - categorical_accuracy: 0.7153 - loss: 1.1175 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1942/Unknown  214s 90ms/step - categorical_accuracy: 0.7153 - loss: 1.1174 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1943/Unknown  215s 90ms/step - categorical_accuracy: 0.7153 - loss: 1.1173 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1944/Unknown  215s 90ms/step - categorical_accuracy: 0.7153 - loss: 1.1172 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1945/Unknown  215s 90ms/step - categorical_accuracy: 0.7154 - loss: 1.1171 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1946/Unknown  215s 90ms/step - categorical_accuracy: 0.7154 - loss: 1.1170 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1947/Unknown  215s 90ms/step - categorical_accuracy: 0.7154 - loss: 1.1169 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1948/Unknown  215s 90ms/step - categorical_accuracy: 0.7154 - loss: 1.1168 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1949/Unknown  215s 90ms/step - categorical_accuracy: 0.7154 - loss: 1.1167 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1950/Unknown  215s 90ms/step - categorical_accuracy: 0.7155 - loss: 1.1166 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1951/Unknown  215s 90ms/step - categorical_accuracy: 0.7155 - loss: 1.1165 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1952/Unknown  215s 90ms/step - categorical_accuracy: 0.7155 - loss: 1.1164 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1953/Unknown  215s 90ms/step - categorical_accuracy: 0.7155 - loss: 1.1163 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1954/Unknown  215s 90ms/step - categorical_accuracy: 0.7155 - loss: 1.1162 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1955/Unknown  216s 90ms/step - categorical_accuracy: 0.7156 - loss: 1.1161 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1956/Unknown  216s 90ms/step - categorical_accuracy: 0.7156 - loss: 1.1160 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1957/Unknown  216s 90ms/step - categorical_accuracy: 0.7156 - loss: 1.1159 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1958/Unknown  216s 90ms/step - categorical_accuracy: 0.7156 - loss: 1.1158 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   1959/Unknown  216s 90ms/step - categorical_accuracy: 0.7156 - loss: 1.1157 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   1960/Unknown  216s 90ms/step - categorical_accuracy: 0.7157 - loss: 1.1156 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   1961/Unknown  216s 90ms/step - categorical_accuracy: 0.7157 - loss: 1.1155 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   1962/Unknown  216s 90ms/step - categorical_accuracy: 0.7157 - loss: 1.1154 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   1963/Unknown  216s 90ms/step - categorical_accuracy: 0.7157 - loss: 1.1153 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   1964/Unknown  216s 90ms/step - categorical_accuracy: 0.7157 - loss: 1.1152 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   1965/Unknown  216s 90ms/step - categorical_accuracy: 0.7158 - loss: 1.1151 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   1966/Unknown  216s 90ms/step - categorical_accuracy: 0.7158 - loss: 1.1150 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   1967/Unknown  217s 90ms/step - categorical_accuracy: 0.7158 - loss: 1.1149 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   1968/Unknown  217s 90ms/step - categorical_accuracy: 0.7158 - loss: 1.1148 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1969/Unknown  217s 90ms/step - categorical_accuracy: 0.7158 - loss: 1.1147 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1970/Unknown  217s 90ms/step - categorical_accuracy: 0.7159 - loss: 1.1146 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1971/Unknown  217s 90ms/step - categorical_accuracy: 0.7159 - loss: 1.1145 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1972/Unknown  217s 90ms/step - categorical_accuracy: 0.7159 - loss: 1.1144 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1973/Unknown  217s 90ms/step - categorical_accuracy: 0.7159 - loss: 1.1143 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   1974/Unknown  217s 90ms/step - categorical_accuracy: 0.7159 - loss: 1.1142 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   1975/Unknown  217s 90ms/step - categorical_accuracy: 0.7160 - loss: 1.1141 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   1976/Unknown  217s 90ms/step - categorical_accuracy: 0.7160 - loss: 1.1140 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   1977/Unknown  217s 90ms/step - categorical_accuracy: 0.7160 - loss: 1.1139 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   1978/Unknown  217s 90ms/step - categorical_accuracy: 0.7160 - loss: 1.1138 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   1979/Unknown  217s 90ms/step - categorical_accuracy: 0.7160 - loss: 1.1137 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   1980/Unknown  217s 90ms/step - categorical_accuracy: 0.7161 - loss: 1.1136 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   1981/Unknown  218s 90ms/step - categorical_accuracy: 0.7161 - loss: 1.1135 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   1982/Unknown  218s 90ms/step - categorical_accuracy: 0.7161 - loss: 1.1135 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   1983/Unknown  218s 90ms/step - categorical_accuracy: 0.7161 - loss: 1.1134 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   1984/Unknown  218s 90ms/step - categorical_accuracy: 0.7161 - loss: 1.1133 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   1985/Unknown  218s 90ms/step - categorical_accuracy: 0.7162 - loss: 1.1132 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   1986/Unknown  218s 90ms/step - categorical_accuracy: 0.7162 - loss: 1.1131 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   1987/Unknown  218s 90ms/step - categorical_accuracy: 0.7162 - loss: 1.1130 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   1988/Unknown  218s 90ms/step - categorical_accuracy: 0.7162 - loss: 1.1129 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   1989/Unknown  218s 90ms/step - categorical_accuracy: 0.7162 - loss: 1.1128 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1990/Unknown  218s 90ms/step - categorical_accuracy: 0.7163 - loss: 1.1127 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1991/Unknown  218s 90ms/step - categorical_accuracy: 0.7163 - loss: 1.1126 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1992/Unknown  218s 90ms/step - categorical_accuracy: 0.7163 - loss: 1.1125 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1993/Unknown  219s 90ms/step - categorical_accuracy: 0.7163 - loss: 1.1124 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1994/Unknown  219s 90ms/step - categorical_accuracy: 0.7163 - loss: 1.1123 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   1995/Unknown  219s 90ms/step - categorical_accuracy: 0.7164 - loss: 1.1122 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   1996/Unknown  219s 90ms/step - categorical_accuracy: 0.7164 - loss: 1.1121 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   1997/Unknown  219s 90ms/step - categorical_accuracy: 0.7164 - loss: 1.1120 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   1998/Unknown  219s 90ms/step - categorical_accuracy: 0.7164 - loss: 1.1119 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   1999/Unknown  219s 90ms/step - categorical_accuracy: 0.7164 - loss: 1.1118 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2000/Unknown  219s 90ms/step - categorical_accuracy: 0.7165 - loss: 1.1117 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2001/Unknown  219s 90ms/step - categorical_accuracy: 0.7165 - loss: 1.1116 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2002/Unknown  219s 90ms/step - categorical_accuracy: 0.7165 - loss: 1.1115 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2003/Unknown  219s 90ms/step - categorical_accuracy: 0.7165 - loss: 1.1114 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2004/Unknown  219s 90ms/step - categorical_accuracy: 0.7165 - loss: 1.1113 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2005/Unknown  219s 90ms/step - categorical_accuracy: 0.7166 - loss: 1.1112 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2006/Unknown  219s 90ms/step - categorical_accuracy: 0.7166 - loss: 1.1111 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2007/Unknown  220s 90ms/step - categorical_accuracy: 0.7166 - loss: 1.1111 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2008/Unknown  220s 90ms/step - categorical_accuracy: 0.7166 - loss: 1.1110 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2009/Unknown  220s 90ms/step - categorical_accuracy: 0.7166 - loss: 1.1109 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2010/Unknown  220s 90ms/step - categorical_accuracy: 0.7167 - loss: 1.1108 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2011/Unknown  220s 90ms/step - categorical_accuracy: 0.7167 - loss: 1.1107 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2012/Unknown  220s 90ms/step - categorical_accuracy: 0.7167 - loss: 1.1106 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2013/Unknown  220s 90ms/step - categorical_accuracy: 0.7167 - loss: 1.1105 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2014/Unknown  220s 90ms/step - categorical_accuracy: 0.7167 - loss: 1.1104 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2015/Unknown  220s 90ms/step - categorical_accuracy: 0.7168 - loss: 1.1103 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2016/Unknown  220s 90ms/step - categorical_accuracy: 0.7168 - loss: 1.1102 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2017/Unknown  220s 90ms/step - categorical_accuracy: 0.7168 - loss: 1.1101 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2018/Unknown  220s 90ms/step - categorical_accuracy: 0.7168 - loss: 1.1100 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2019/Unknown  220s 90ms/step - categorical_accuracy: 0.7168 - loss: 1.1099 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2020/Unknown  220s 90ms/step - categorical_accuracy: 0.7169 - loss: 1.1098 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2021/Unknown  221s 90ms/step - categorical_accuracy: 0.7169 - loss: 1.1097 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2022/Unknown  221s 90ms/step - categorical_accuracy: 0.7169 - loss: 1.1096 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2023/Unknown  221s 90ms/step - categorical_accuracy: 0.7169 - loss: 1.1095 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2024/Unknown  221s 90ms/step - categorical_accuracy: 0.7169 - loss: 1.1094 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2025/Unknown  221s 89ms/step - categorical_accuracy: 0.7169 - loss: 1.1093 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2026/Unknown  221s 89ms/step - categorical_accuracy: 0.7170 - loss: 1.1093 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2027/Unknown  221s 90ms/step - categorical_accuracy: 0.7170 - loss: 1.1092 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2028/Unknown  221s 89ms/step - categorical_accuracy: 0.7170 - loss: 1.1091 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2029/Unknown  221s 89ms/step - categorical_accuracy: 0.7170 - loss: 1.1090 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2030/Unknown  221s 89ms/step - categorical_accuracy: 0.7170 - loss: 1.1089 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2031/Unknown  221s 89ms/step - categorical_accuracy: 0.7171 - loss: 1.1088 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2032/Unknown  221s 89ms/step - categorical_accuracy: 0.7171 - loss: 1.1087 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2033/Unknown  221s 89ms/step - categorical_accuracy: 0.7171 - loss: 1.1086 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2034/Unknown  222s 89ms/step - categorical_accuracy: 0.7171 - loss: 1.1085 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2035/Unknown  222s 89ms/step - categorical_accuracy: 0.7171 - loss: 1.1084 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2036/Unknown  222s 89ms/step - categorical_accuracy: 0.7172 - loss: 1.1083 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2037/Unknown  222s 89ms/step - categorical_accuracy: 0.7172 - loss: 1.1082 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2038/Unknown  222s 89ms/step - categorical_accuracy: 0.7172 - loss: 1.1081 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2039/Unknown  222s 89ms/step - categorical_accuracy: 0.7172 - loss: 1.1080 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2040/Unknown  222s 89ms/step - categorical_accuracy: 0.7172 - loss: 1.1080 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2041/Unknown  222s 89ms/step - categorical_accuracy: 0.7173 - loss: 1.1079 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2042/Unknown  222s 89ms/step - categorical_accuracy: 0.7173 - loss: 1.1078 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2043/Unknown  222s 89ms/step - categorical_accuracy: 0.7173 - loss: 1.1077 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2044/Unknown  222s 89ms/step - categorical_accuracy: 0.7173 - loss: 1.1076 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2045/Unknown  222s 89ms/step - categorical_accuracy: 0.7173 - loss: 1.1075 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2046/Unknown  222s 89ms/step - categorical_accuracy: 0.7173 - loss: 1.1074 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2047/Unknown  223s 89ms/step - categorical_accuracy: 0.7174 - loss: 1.1073 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2048/Unknown  223s 89ms/step - categorical_accuracy: 0.7174 - loss: 1.1072 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2049/Unknown  223s 89ms/step - categorical_accuracy: 0.7174 - loss: 1.1071 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2050/Unknown  223s 89ms/step - categorical_accuracy: 0.7174 - loss: 1.1070 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2051/Unknown  223s 89ms/step - categorical_accuracy: 0.7174 - loss: 1.1069 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2052/Unknown  223s 89ms/step - categorical_accuracy: 0.7175 - loss: 1.1068 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2053/Unknown  223s 89ms/step - categorical_accuracy: 0.7175 - loss: 1.1068 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2054/Unknown  223s 89ms/step - categorical_accuracy: 0.7175 - loss: 1.1067 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2055/Unknown  223s 89ms/step - categorical_accuracy: 0.7175 - loss: 1.1066 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2056/Unknown  223s 89ms/step - categorical_accuracy: 0.7175 - loss: 1.1065 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2057/Unknown  223s 89ms/step - categorical_accuracy: 0.7176 - loss: 1.1064 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2058/Unknown  223s 89ms/step - categorical_accuracy: 0.7176 - loss: 1.1063 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2059/Unknown  224s 89ms/step - categorical_accuracy: 0.7176 - loss: 1.1062 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2060/Unknown  224s 89ms/step - categorical_accuracy: 0.7176 - loss: 1.1061 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2061/Unknown  224s 89ms/step - categorical_accuracy: 0.7176 - loss: 1.1060 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2062/Unknown  224s 89ms/step - categorical_accuracy: 0.7176 - loss: 1.1059 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2063/Unknown  224s 89ms/step - categorical_accuracy: 0.7177 - loss: 1.1058 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2064/Unknown  224s 89ms/step - categorical_accuracy: 0.7177 - loss: 1.1058 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2065/Unknown  224s 89ms/step - categorical_accuracy: 0.7177 - loss: 1.1057 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2066/Unknown  224s 89ms/step - categorical_accuracy: 0.7177 - loss: 1.1056 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2067/Unknown  224s 89ms/step - categorical_accuracy: 0.7177 - loss: 1.1055 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2068/Unknown  224s 89ms/step - categorical_accuracy: 0.7178 - loss: 1.1054 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2069/Unknown  224s 89ms/step - categorical_accuracy: 0.7178 - loss: 1.1053 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2070/Unknown  224s 89ms/step - categorical_accuracy: 0.7178 - loss: 1.1052 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2071/Unknown  224s 89ms/step - categorical_accuracy: 0.7178 - loss: 1.1051 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2072/Unknown  225s 89ms/step - categorical_accuracy: 0.7178 - loss: 1.1050 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2073/Unknown  225s 89ms/step - categorical_accuracy: 0.7179 - loss: 1.1049 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2074/Unknown  225s 89ms/step - categorical_accuracy: 0.7179 - loss: 1.1048 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2075/Unknown  225s 89ms/step - categorical_accuracy: 0.7179 - loss: 1.1048 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2076/Unknown  225s 89ms/step - categorical_accuracy: 0.7179 - loss: 1.1047 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2077/Unknown  225s 89ms/step - categorical_accuracy: 0.7179 - loss: 1.1046 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2078/Unknown  225s 89ms/step - categorical_accuracy: 0.7179 - loss: 1.1045 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2079/Unknown  225s 89ms/step - categorical_accuracy: 0.7180 - loss: 1.1044 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2080/Unknown  225s 89ms/step - categorical_accuracy: 0.7180 - loss: 1.1043 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2081/Unknown  225s 89ms/step - categorical_accuracy: 0.7180 - loss: 1.1042 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2082/Unknown  225s 89ms/step - categorical_accuracy: 0.7180 - loss: 1.1041 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```

```
</div>
   2083/Unknown  226s 89ms/step - categorical_accuracy: 0.7180 - loss: 1.1040 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2084/Unknown  226s 89ms/step - categorical_accuracy: 0.7181 - loss: 1.1039 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2085/Unknown  226s 89ms/step - categorical_accuracy: 0.7181 - loss: 1.1039 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2086/Unknown  226s 89ms/step - categorical_accuracy: 0.7181 - loss: 1.1038 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2087/Unknown  226s 89ms/step - categorical_accuracy: 0.7181 - loss: 1.1037 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2088/Unknown  226s 89ms/step - categorical_accuracy: 0.7181 - loss: 1.1036 - mean_io_u: 0.0927

<div class="k-default-codeblock">
```

```
</div>
   2089/Unknown  226s 89ms/step - categorical_accuracy: 0.7181 - loss: 1.1035 - mean_io_u: 0.0928

<div class="k-default-codeblock">
```

```
</div>
   2090/Unknown  226s 89ms/step - categorical_accuracy: 0.7182 - loss: 1.1034 - mean_io_u: 0.0928

<div class="k-default-codeblock">
```

```
</div>
   2091/Unknown  226s 89ms/step - categorical_accuracy: 0.7182 - loss: 1.1033 - mean_io_u: 0.0928

<div class="k-default-codeblock">
```

```
</div>
   2092/Unknown  226s 89ms/step - categorical_accuracy: 0.7182 - loss: 1.1032 - mean_io_u: 0.0928

<div class="k-default-codeblock">
```

```
</div>
   2093/Unknown  226s 89ms/step - categorical_accuracy: 0.7182 - loss: 1.1031 - mean_io_u: 0.0928

<div class="k-default-codeblock">
```

```
</div>
   2094/Unknown  226s 89ms/step - categorical_accuracy: 0.7182 - loss: 1.1031 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2095/Unknown  227s 89ms/step - categorical_accuracy: 0.7183 - loss: 1.1030 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2096/Unknown  227s 89ms/step - categorical_accuracy: 0.7183 - loss: 1.1029 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2097/Unknown  227s 89ms/step - categorical_accuracy: 0.7183 - loss: 1.1028 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2098/Unknown  227s 89ms/step - categorical_accuracy: 0.7183 - loss: 1.1027 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2099/Unknown  227s 89ms/step - categorical_accuracy: 0.7183 - loss: 1.1026 - mean_io_u: 0.0929

<div class="k-default-codeblock">
```

```
</div>
   2100/Unknown  227s 89ms/step - categorical_accuracy: 0.7183 - loss: 1.1025 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2101/Unknown  227s 89ms/step - categorical_accuracy: 0.7184 - loss: 1.1024 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2102/Unknown  227s 89ms/step - categorical_accuracy: 0.7184 - loss: 1.1023 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2103/Unknown  227s 89ms/step - categorical_accuracy: 0.7184 - loss: 1.1023 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2104/Unknown  227s 89ms/step - categorical_accuracy: 0.7184 - loss: 1.1022 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2105/Unknown  227s 89ms/step - categorical_accuracy: 0.7184 - loss: 1.1021 - mean_io_u: 0.0930

<div class="k-default-codeblock">
```

```
</div>
   2106/Unknown  227s 89ms/step - categorical_accuracy: 0.7185 - loss: 1.1020 - mean_io_u: 0.0931

<div class="k-default-codeblock">
```

```
</div>
   2107/Unknown  227s 89ms/step - categorical_accuracy: 0.7185 - loss: 1.1019 - mean_io_u: 0.0931

<div class="k-default-codeblock">
```

```
</div>
   2108/Unknown  228s 89ms/step - categorical_accuracy: 0.7185 - loss: 1.1018 - mean_io_u: 0.0931

<div class="k-default-codeblock">
```

```
</div>
   2109/Unknown  228s 89ms/step - categorical_accuracy: 0.7185 - loss: 1.1017 - mean_io_u: 0.0931

<div class="k-default-codeblock">
```

```
</div>
   2110/Unknown  228s 89ms/step - categorical_accuracy: 0.7185 - loss: 1.1016 - mean_io_u: 0.0931

<div class="k-default-codeblock">
```

```
</div>
   2111/Unknown  228s 89ms/step - categorical_accuracy: 0.7185 - loss: 1.1015 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2112/Unknown  228s 89ms/step - categorical_accuracy: 0.7186 - loss: 1.1015 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2113/Unknown  228s 89ms/step - categorical_accuracy: 0.7186 - loss: 1.1014 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2114/Unknown  228s 89ms/step - categorical_accuracy: 0.7186 - loss: 1.1013 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2115/Unknown  228s 89ms/step - categorical_accuracy: 0.7186 - loss: 1.1012 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2116/Unknown  228s 89ms/step - categorical_accuracy: 0.7186 - loss: 1.1011 - mean_io_u: 0.0932

<div class="k-default-codeblock">
```

```
</div>
   2117/Unknown  228s 89ms/step - categorical_accuracy: 0.7187 - loss: 1.1010 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2118/Unknown  228s 89ms/step - categorical_accuracy: 0.7187 - loss: 1.1009 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2119/Unknown  228s 89ms/step - categorical_accuracy: 0.7187 - loss: 1.1008 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2120/Unknown  229s 89ms/step - categorical_accuracy: 0.7187 - loss: 1.1008 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2121/Unknown  229s 89ms/step - categorical_accuracy: 0.7187 - loss: 1.1007 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2122/Unknown  229s 89ms/step - categorical_accuracy: 0.7187 - loss: 1.1006 - mean_io_u: 0.0933

<div class="k-default-codeblock">
```

```
</div>
   2123/Unknown  229s 89ms/step - categorical_accuracy: 0.7188 - loss: 1.1004 - mean_io_u: 0.0934

<div class="k-default-codeblock">
```

```
</div>
   2124/Unknown  229s 89ms/step - categorical_accuracy: 0.7188 - loss: 1.1004 - mean_io_u: 0.0934

<div class="k-default-codeblock">
```
/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/models/functional.py:225: UserWarning: The structure of `inputs` doesn't match the expected structure: ['keras_tensor_222']. Received: the structure of inputs=*
  warnings.warn(
/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/backend/jax/core.py:76: UserWarning: Explicitly requested dtype int64 requested in asarray is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return jnp.asarray(x, dtype=dtype)


```
</div>
 2124/2124 ━━━━━━━━━━━━━━━━━━━━ 279s 113ms/step - categorical_accuracy: 0.7188 - loss: 1.1003 - mean_io_u: 0.0934 - val_categorical_accuracy: 0.8222 - val_loss: 0.5761 - val_mean_io_u: 0.3481





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f6868c16490>

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
/home/sachinprasad/projects/env/lib/python3.11/site-packages/multiprocess/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()

/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/models/functional.py:225: UserWarning: The structure of `inputs` doesn't match the expected structure: ['keras_tensor_222']. Received: the structure of inputs=*
  warnings.warn(

```
</div>
    
![png](/home/sachinprasad/projects/keras-io/guides/img/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_32_2.png)
    


Here are some additional tips for using the KerasHub DeepLabv3 model:

- The model can be trained on a variety of datasets, including the COCO dataset, the
PASCAL VOC dataset, and the Cityscapes dataset.
- The model can be fine-tuned on a custom dataset to improve its performance on a
specific task.
- The model can be used to perform real-time inference on images.
- Also, check out KerasHub's other segmentation models.

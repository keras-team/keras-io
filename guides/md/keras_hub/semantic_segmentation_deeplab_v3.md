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
image = np.array(image)

image = preprocessor(image)
image = keras.ops.expand_dims(image, axis=0)
preds = ops.expand_dims(ops.argmax(model.predict(image), axis=-1), axis=-1)


def plot_segmentation(original_image, predicted_mask):
    plt.figure(figsize=(5, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image[0] / 255)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask[0])
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
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 5s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 5s 5s/step



    
![png](/home/sachinprasad/projects/keras-io/guides/img/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_9_3.png)
    


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
    num_images = len(images)
    plt.figure(figsize=(8, 4))
    rows = 3 if pred_masks is not None else 2

    for i in range(num_images):
        plt.subplot(rows, num_images, i + 1)
        plt.imshow(images[i] / 255)
        plt.axis("off")

        plt.subplot(rows, num_images, num_images + i + 1)
        plt.imshow(masks[i])
        plt.axis("off")

        if pred_masks is not None:
            plt.subplot(rows, num_images, i + 1 + 2 * num_images)
            plt.imshow(pred_masks[i])
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
  1/Unknown  40s 40s/step - categorical_accuracy: 0.0998 - loss: 3.2794 - mean_io_u: 0.0095


  2/Unknown  62s 22s/step - categorical_accuracy: 0.1057 - loss: 3.2194 - mean_io_u: 0.0103


  3/Unknown  62s 11s/step - categorical_accuracy: 0.1102 - loss: 3.1906 - mean_io_u: 0.0128


  4/Unknown  62s 7s/step - categorical_accuracy: 0.1175 - loss: 3.1606 - mean_io_u: 0.0128 


  5/Unknown  62s 5s/step - categorical_accuracy: 0.1242 - loss: 3.1374 - mean_io_u: 0.0134


  6/Unknown  62s 4s/step - categorical_accuracy: 0.1304 - loss: 3.1136 - mean_io_u: 0.0140


  7/Unknown  62s 4s/step - categorical_accuracy: 0.1371 - loss: 3.0899 - mean_io_u: 0.0147


  8/Unknown  62s 3s/step - categorical_accuracy: 0.1462 - loss: 3.0626 - mean_io_u: 0.0155


  9/Unknown  62s 3s/step - categorical_accuracy: 0.1549 - loss: 3.0381 - mean_io_u: 0.0161


 10/Unknown  62s 2s/step - categorical_accuracy: 0.1653 - loss: 3.0111 - mean_io_u: 0.0168


 11/Unknown  62s 2s/step - categorical_accuracy: 0.1770 - loss: 2.9810 - mean_io_u: 0.0176


 12/Unknown  62s 2s/step - categorical_accuracy: 0.1884 - loss: 2.9519 - mean_io_u: 0.0183


 13/Unknown  62s 2s/step - categorical_accuracy: 0.1995 - loss: 2.9241 - mean_io_u: 0.0190


 14/Unknown  62s 2s/step - categorical_accuracy: 0.2109 - loss: 2.8951 - mean_io_u: 0.0197


 15/Unknown  62s 2s/step - categorical_accuracy: 0.2226 - loss: 2.8649 - mean_io_u: 0.0204


 16/Unknown  63s 2s/step - categorical_accuracy: 0.2337 - loss: 2.8359 - mean_io_u: 0.0210


 17/Unknown  63s 1s/step - categorical_accuracy: 0.2446 - loss: 2.8071 - mean_io_u: 0.0216


 18/Unknown  63s 1s/step - categorical_accuracy: 0.2553 - loss: 2.7779 - mean_io_u: 0.0222


 19/Unknown  63s 1s/step - categorical_accuracy: 0.2655 - loss: 2.7503 - mean_io_u: 0.0228


 20/Unknown  63s 1s/step - categorical_accuracy: 0.2752 - loss: 2.7236 - mean_io_u: 0.0233


 21/Unknown  63s 1s/step - categorical_accuracy: 0.2844 - loss: 2.6979 - mean_io_u: 0.0237


 22/Unknown  63s 1s/step - categorical_accuracy: 0.2932 - loss: 2.6736 - mean_io_u: 0.0242


 23/Unknown  63s 1s/step - categorical_accuracy: 0.3014 - loss: 2.6515 - mean_io_u: 0.0246


 24/Unknown  63s 1s/step - categorical_accuracy: 0.3091 - loss: 2.6306 - mean_io_u: 0.0249


 25/Unknown  63s 979ms/step - categorical_accuracy: 0.3164 - loss: 2.6108 - mean_io_u: 0.0253


 26/Unknown  63s 945ms/step - categorical_accuracy: 0.3237 - loss: 2.5904 - mean_io_u: 0.0256


 27/Unknown  63s 912ms/step - categorical_accuracy: 0.3307 - loss: 2.5708 - mean_io_u: 0.0259


 28/Unknown  64s 882ms/step - categorical_accuracy: 0.3374 - loss: 2.5518 - mean_io_u: 0.0262


 29/Unknown  64s 854ms/step - categorical_accuracy: 0.3440 - loss: 2.5328 - mean_io_u: 0.0265


 30/Unknown  64s 828ms/step - categorical_accuracy: 0.3504 - loss: 2.5141 - mean_io_u: 0.0268


 31/Unknown  64s 804ms/step - categorical_accuracy: 0.3567 - loss: 2.4954 - mean_io_u: 0.0271


 32/Unknown  64s 781ms/step - categorical_accuracy: 0.3627 - loss: 2.4773 - mean_io_u: 0.0273


 33/Unknown  64s 760ms/step - categorical_accuracy: 0.3686 - loss: 2.4595 - mean_io_u: 0.0276


 34/Unknown  64s 740ms/step - categorical_accuracy: 0.3743 - loss: 2.4424 - mean_io_u: 0.0278


 35/Unknown  64s 722ms/step - categorical_accuracy: 0.3798 - loss: 2.4259 - mean_io_u: 0.0280


 36/Unknown  64s 705ms/step - categorical_accuracy: 0.3850 - loss: 2.4102 - mean_io_u: 0.0281


 37/Unknown  65s 688ms/step - categorical_accuracy: 0.3900 - loss: 2.3950 - mean_io_u: 0.0283


 38/Unknown  65s 673ms/step - categorical_accuracy: 0.3948 - loss: 2.3807 - mean_io_u: 0.0284


 39/Unknown  65s 658ms/step - categorical_accuracy: 0.3995 - loss: 2.3669 - mean_io_u: 0.0285


 40/Unknown  65s 643ms/step - categorical_accuracy: 0.4040 - loss: 2.3533 - mean_io_u: 0.0286


 41/Unknown  65s 629ms/step - categorical_accuracy: 0.4085 - loss: 2.3400 - mean_io_u: 0.0287


 42/Unknown  65s 615ms/step - categorical_accuracy: 0.4127 - loss: 2.3271 - mean_io_u: 0.0288


 43/Unknown  65s 602ms/step - categorical_accuracy: 0.4169 - loss: 2.3144 - mean_io_u: 0.0289


 44/Unknown  65s 590ms/step - categorical_accuracy: 0.4210 - loss: 2.3021 - mean_io_u: 0.0289


 45/Unknown  65s 579ms/step - categorical_accuracy: 0.4250 - loss: 2.2897 - mean_io_u: 0.0290


 46/Unknown  65s 568ms/step - categorical_accuracy: 0.4289 - loss: 2.2776 - mean_io_u: 0.0291


 47/Unknown  65s 558ms/step - categorical_accuracy: 0.4328 - loss: 2.2656 - mean_io_u: 0.0291


 48/Unknown  66s 549ms/step - categorical_accuracy: 0.4365 - loss: 2.2540 - mean_io_u: 0.0292


 49/Unknown  66s 540ms/step - categorical_accuracy: 0.4401 - loss: 2.2428 - mean_io_u: 0.0292


 50/Unknown  66s 531ms/step - categorical_accuracy: 0.4436 - loss: 2.2318 - mean_io_u: 0.0292


 51/Unknown  66s 523ms/step - categorical_accuracy: 0.4469 - loss: 2.2211 - mean_io_u: 0.0293


 52/Unknown  66s 515ms/step - categorical_accuracy: 0.4502 - loss: 2.2108 - mean_io_u: 0.0293


 53/Unknown  66s 507ms/step - categorical_accuracy: 0.4532 - loss: 2.2014 - mean_io_u: 0.0293


 54/Unknown  66s 499ms/step - categorical_accuracy: 0.4561 - loss: 2.1923 - mean_io_u: 0.0294


 55/Unknown  66s 492ms/step - categorical_accuracy: 0.4590 - loss: 2.1833 - mean_io_u: 0.0294


 56/Unknown  66s 485ms/step - categorical_accuracy: 0.4618 - loss: 2.1745 - mean_io_u: 0.0294


 57/Unknown  67s 478ms/step - categorical_accuracy: 0.4645 - loss: 2.1657 - mean_io_u: 0.0294


 58/Unknown  67s 471ms/step - categorical_accuracy: 0.4672 - loss: 2.1574 - mean_io_u: 0.0294


 59/Unknown  67s 465ms/step - categorical_accuracy: 0.4697 - loss: 2.1494 - mean_io_u: 0.0294


 60/Unknown  67s 458ms/step - categorical_accuracy: 0.4721 - loss: 2.1418 - mean_io_u: 0.0294


 61/Unknown  67s 452ms/step - categorical_accuracy: 0.4745 - loss: 2.1343 - mean_io_u: 0.0294


 62/Unknown  67s 447ms/step - categorical_accuracy: 0.4768 - loss: 2.1269 - mean_io_u: 0.0295


 63/Unknown  67s 442ms/step - categorical_accuracy: 0.4790 - loss: 2.1197 - mean_io_u: 0.0295


 64/Unknown  67s 436ms/step - categorical_accuracy: 0.4812 - loss: 2.1127 - mean_io_u: 0.0295


 65/Unknown  67s 431ms/step - categorical_accuracy: 0.4833 - loss: 2.1057 - mean_io_u: 0.0295


 66/Unknown  67s 426ms/step - categorical_accuracy: 0.4854 - loss: 2.0988 - mean_io_u: 0.0295


 67/Unknown  68s 421ms/step - categorical_accuracy: 0.4875 - loss: 2.0920 - mean_io_u: 0.0295


 68/Unknown  68s 416ms/step - categorical_accuracy: 0.4896 - loss: 2.0853 - mean_io_u: 0.0295


 69/Unknown  68s 411ms/step - categorical_accuracy: 0.4915 - loss: 2.0788 - mean_io_u: 0.0296


 70/Unknown  68s 406ms/step - categorical_accuracy: 0.4934 - loss: 2.0725 - mean_io_u: 0.0296


 71/Unknown  68s 402ms/step - categorical_accuracy: 0.4954 - loss: 2.0663 - mean_io_u: 0.0296


 72/Unknown  68s 398ms/step - categorical_accuracy: 0.4972 - loss: 2.0602 - mean_io_u: 0.0296


 73/Unknown  68s 394ms/step - categorical_accuracy: 0.4990 - loss: 2.0544 - mean_io_u: 0.0296


 74/Unknown  68s 390ms/step - categorical_accuracy: 0.5007 - loss: 2.0486 - mean_io_u: 0.0297


 75/Unknown  68s 386ms/step - categorical_accuracy: 0.5025 - loss: 2.0428 - mean_io_u: 0.0297


 76/Unknown  68s 382ms/step - categorical_accuracy: 0.5042 - loss: 2.0372 - mean_io_u: 0.0298


 77/Unknown  68s 378ms/step - categorical_accuracy: 0.5058 - loss: 2.0317 - mean_io_u: 0.0298


 78/Unknown  69s 374ms/step - categorical_accuracy: 0.5074 - loss: 2.0263 - mean_io_u: 0.0298


 79/Unknown  69s 371ms/step - categorical_accuracy: 0.5090 - loss: 2.0209 - mean_io_u: 0.0299


 80/Unknown  69s 367ms/step - categorical_accuracy: 0.5105 - loss: 2.0156 - mean_io_u: 0.0299


 81/Unknown  69s 364ms/step - categorical_accuracy: 0.5121 - loss: 2.0104 - mean_io_u: 0.0299


 82/Unknown  69s 361ms/step - categorical_accuracy: 0.5136 - loss: 2.0051 - mean_io_u: 0.0300


 83/Unknown  69s 357ms/step - categorical_accuracy: 0.5151 - loss: 2.0001 - mean_io_u: 0.0300


 84/Unknown  69s 354ms/step - categorical_accuracy: 0.5165 - loss: 1.9950 - mean_io_u: 0.0300


 85/Unknown  69s 352ms/step - categorical_accuracy: 0.5179 - loss: 1.9901 - mean_io_u: 0.0301


 86/Unknown  69s 349ms/step - categorical_accuracy: 0.5193 - loss: 1.9853 - mean_io_u: 0.0301


 87/Unknown  70s 346ms/step - categorical_accuracy: 0.5207 - loss: 1.9805 - mean_io_u: 0.0302


 88/Unknown  70s 343ms/step - categorical_accuracy: 0.5221 - loss: 1.9757 - mean_io_u: 0.0302


 89/Unknown  70s 340ms/step - categorical_accuracy: 0.5234 - loss: 1.9711 - mean_io_u: 0.0302


 90/Unknown  70s 337ms/step - categorical_accuracy: 0.5247 - loss: 1.9664 - mean_io_u: 0.0303


 91/Unknown  70s 335ms/step - categorical_accuracy: 0.5260 - loss: 1.9618 - mean_io_u: 0.0303


 92/Unknown  70s 333ms/step - categorical_accuracy: 0.5273 - loss: 1.9572 - mean_io_u: 0.0303


 93/Unknown  70s 330ms/step - categorical_accuracy: 0.5286 - loss: 1.9528 - mean_io_u: 0.0304


 94/Unknown  70s 328ms/step - categorical_accuracy: 0.5298 - loss: 1.9484 - mean_io_u: 0.0304


 95/Unknown  70s 325ms/step - categorical_accuracy: 0.5310 - loss: 1.9441 - mean_io_u: 0.0304


 96/Unknown  70s 323ms/step - categorical_accuracy: 0.5321 - loss: 1.9399 - mean_io_u: 0.0304


 97/Unknown  71s 320ms/step - categorical_accuracy: 0.5333 - loss: 1.9357 - mean_io_u: 0.0305


 98/Unknown  71s 318ms/step - categorical_accuracy: 0.5344 - loss: 1.9316 - mean_io_u: 0.0305


 99/Unknown  71s 316ms/step - categorical_accuracy: 0.5355 - loss: 1.9274 - mean_io_u: 0.0305


100/Unknown  71s 314ms/step - categorical_accuracy: 0.5367 - loss: 1.9233 - mean_io_u: 0.0306


101/Unknown  71s 312ms/step - categorical_accuracy: 0.5378 - loss: 1.9193 - mean_io_u: 0.0306


102/Unknown  71s 309ms/step - categorical_accuracy: 0.5389 - loss: 1.9153 - mean_io_u: 0.0306


103/Unknown  71s 307ms/step - categorical_accuracy: 0.5399 - loss: 1.9113 - mean_io_u: 0.0306


104/Unknown  71s 305ms/step - categorical_accuracy: 0.5410 - loss: 1.9073 - mean_io_u: 0.0307


105/Unknown  71s 303ms/step - categorical_accuracy: 0.5421 - loss: 1.9034 - mean_io_u: 0.0307


106/Unknown  71s 301ms/step - categorical_accuracy: 0.5431 - loss: 1.8995 - mean_io_u: 0.0307


107/Unknown  71s 299ms/step - categorical_accuracy: 0.5442 - loss: 1.8956 - mean_io_u: 0.0308


108/Unknown  72s 297ms/step - categorical_accuracy: 0.5452 - loss: 1.8918 - mean_io_u: 0.0308


109/Unknown  72s 295ms/step - categorical_accuracy: 0.5462 - loss: 1.8880 - mean_io_u: 0.0308


110/Unknown  72s 294ms/step - categorical_accuracy: 0.5472 - loss: 1.8843 - mean_io_u: 0.0309


111/Unknown  72s 292ms/step - categorical_accuracy: 0.5482 - loss: 1.8805 - mean_io_u: 0.0309


112/Unknown  72s 290ms/step - categorical_accuracy: 0.5492 - loss: 1.8768 - mean_io_u: 0.0309


113/Unknown  72s 289ms/step - categorical_accuracy: 0.5502 - loss: 1.8731 - mean_io_u: 0.0310


114/Unknown  72s 287ms/step - categorical_accuracy: 0.5512 - loss: 1.8694 - mean_io_u: 0.0310


115/Unknown  72s 285ms/step - categorical_accuracy: 0.5521 - loss: 1.8658 - mean_io_u: 0.0311


116/Unknown  72s 283ms/step - categorical_accuracy: 0.5531 - loss: 1.8623 - mean_io_u: 0.0311


117/Unknown  72s 281ms/step - categorical_accuracy: 0.5540 - loss: 1.8588 - mean_io_u: 0.0311


118/Unknown  72s 280ms/step - categorical_accuracy: 0.5549 - loss: 1.8554 - mean_io_u: 0.0311


119/Unknown  73s 278ms/step - categorical_accuracy: 0.5558 - loss: 1.8519 - mean_io_u: 0.0312


120/Unknown  73s 276ms/step - categorical_accuracy: 0.5567 - loss: 1.8485 - mean_io_u: 0.0312


121/Unknown  73s 275ms/step - categorical_accuracy: 0.5575 - loss: 1.8452 - mean_io_u: 0.0313


122/Unknown  73s 273ms/step - categorical_accuracy: 0.5584 - loss: 1.8419 - mean_io_u: 0.0313


123/Unknown  73s 271ms/step - categorical_accuracy: 0.5593 - loss: 1.8387 - mean_io_u: 0.0313


124/Unknown  73s 270ms/step - categorical_accuracy: 0.5601 - loss: 1.8355 - mean_io_u: 0.0314


125/Unknown  73s 268ms/step - categorical_accuracy: 0.5609 - loss: 1.8325 - mean_io_u: 0.0314


126/Unknown  73s 267ms/step - categorical_accuracy: 0.5617 - loss: 1.8295 - mean_io_u: 0.0314


127/Unknown  73s 265ms/step - categorical_accuracy: 0.5624 - loss: 1.8266 - mean_io_u: 0.0314


128/Unknown  73s 264ms/step - categorical_accuracy: 0.5632 - loss: 1.8236 - mean_io_u: 0.0315


129/Unknown  73s 263ms/step - categorical_accuracy: 0.5640 - loss: 1.8208 - mean_io_u: 0.0315


130/Unknown  73s 261ms/step - categorical_accuracy: 0.5647 - loss: 1.8179 - mean_io_u: 0.0315


131/Unknown  74s 260ms/step - categorical_accuracy: 0.5655 - loss: 1.8151 - mean_io_u: 0.0316


132/Unknown  74s 259ms/step - categorical_accuracy: 0.5662 - loss: 1.8122 - mean_io_u: 0.0316


133/Unknown  74s 258ms/step - categorical_accuracy: 0.5669 - loss: 1.8094 - mean_io_u: 0.0316


134/Unknown  74s 257ms/step - categorical_accuracy: 0.5676 - loss: 1.8067 - mean_io_u: 0.0316


135/Unknown  74s 256ms/step - categorical_accuracy: 0.5684 - loss: 1.8040 - mean_io_u: 0.0317


136/Unknown  74s 254ms/step - categorical_accuracy: 0.5691 - loss: 1.8012 - mean_io_u: 0.0317


137/Unknown  74s 253ms/step - categorical_accuracy: 0.5698 - loss: 1.7985 - mean_io_u: 0.0317


138/Unknown  74s 252ms/step - categorical_accuracy: 0.5704 - loss: 1.7959 - mean_io_u: 0.0318


139/Unknown  74s 251ms/step - categorical_accuracy: 0.5711 - loss: 1.7932 - mean_io_u: 0.0318


140/Unknown  74s 250ms/step - categorical_accuracy: 0.5718 - loss: 1.7906 - mean_io_u: 0.0319


141/Unknown  75s 249ms/step - categorical_accuracy: 0.5724 - loss: 1.7880 - mean_io_u: 0.0319


142/Unknown  75s 248ms/step - categorical_accuracy: 0.5731 - loss: 1.7854 - mean_io_u: 0.0319


143/Unknown  75s 247ms/step - categorical_accuracy: 0.5737 - loss: 1.7828 - mean_io_u: 0.0320


144/Unknown  75s 245ms/step - categorical_accuracy: 0.5744 - loss: 1.7803 - mean_io_u: 0.0320


145/Unknown  75s 245ms/step - categorical_accuracy: 0.5750 - loss: 1.7778 - mean_io_u: 0.0321


146/Unknown  75s 243ms/step - categorical_accuracy: 0.5757 - loss: 1.7752 - mean_io_u: 0.0321


147/Unknown  75s 243ms/step - categorical_accuracy: 0.5763 - loss: 1.7728 - mean_io_u: 0.0321


148/Unknown  75s 242ms/step - categorical_accuracy: 0.5769 - loss: 1.7703 - mean_io_u: 0.0322


149/Unknown  75s 241ms/step - categorical_accuracy: 0.5775 - loss: 1.7679 - mean_io_u: 0.0322


150/Unknown  76s 240ms/step - categorical_accuracy: 0.5781 - loss: 1.7655 - mean_io_u: 0.0323


151/Unknown  76s 239ms/step - categorical_accuracy: 0.5787 - loss: 1.7631 - mean_io_u: 0.0323


152/Unknown  76s 238ms/step - categorical_accuracy: 0.5793 - loss: 1.7607 - mean_io_u: 0.0323


153/Unknown  76s 237ms/step - categorical_accuracy: 0.5799 - loss: 1.7584 - mean_io_u: 0.0324


154/Unknown  76s 236ms/step - categorical_accuracy: 0.5805 - loss: 1.7560 - mean_io_u: 0.0324


155/Unknown  76s 235ms/step - categorical_accuracy: 0.5811 - loss: 1.7537 - mean_io_u: 0.0325


156/Unknown  76s 234ms/step - categorical_accuracy: 0.5817 - loss: 1.7514 - mean_io_u: 0.0325


157/Unknown  76s 234ms/step - categorical_accuracy: 0.5822 - loss: 1.7492 - mean_io_u: 0.0325


158/Unknown  76s 233ms/step - categorical_accuracy: 0.5828 - loss: 1.7469 - mean_io_u: 0.0326


159/Unknown  76s 232ms/step - categorical_accuracy: 0.5833 - loss: 1.7447 - mean_io_u: 0.0326


160/Unknown  76s 231ms/step - categorical_accuracy: 0.5839 - loss: 1.7424 - mean_io_u: 0.0327


161/Unknown  77s 230ms/step - categorical_accuracy: 0.5845 - loss: 1.7403 - mean_io_u: 0.0327


162/Unknown  77s 229ms/step - categorical_accuracy: 0.5850 - loss: 1.7381 - mean_io_u: 0.0327


163/Unknown  77s 229ms/step - categorical_accuracy: 0.5855 - loss: 1.7359 - mean_io_u: 0.0328


164/Unknown  77s 228ms/step - categorical_accuracy: 0.5861 - loss: 1.7338 - mean_io_u: 0.0328


165/Unknown  77s 227ms/step - categorical_accuracy: 0.5866 - loss: 1.7316 - mean_io_u: 0.0329


166/Unknown  77s 227ms/step - categorical_accuracy: 0.5871 - loss: 1.7295 - mean_io_u: 0.0329


167/Unknown  77s 226ms/step - categorical_accuracy: 0.5876 - loss: 1.7274 - mean_io_u: 0.0330


168/Unknown  77s 226ms/step - categorical_accuracy: 0.5882 - loss: 1.7253 - mean_io_u: 0.0330


169/Unknown  78s 225ms/step - categorical_accuracy: 0.5887 - loss: 1.7232 - mean_io_u: 0.0330


170/Unknown  78s 225ms/step - categorical_accuracy: 0.5892 - loss: 1.7211 - mean_io_u: 0.0331


171/Unknown  78s 224ms/step - categorical_accuracy: 0.5897 - loss: 1.7190 - mean_io_u: 0.0331


172/Unknown  78s 223ms/step - categorical_accuracy: 0.5902 - loss: 1.7169 - mean_io_u: 0.0332


173/Unknown  78s 222ms/step - categorical_accuracy: 0.5908 - loss: 1.7148 - mean_io_u: 0.0332


174/Unknown  78s 222ms/step - categorical_accuracy: 0.5913 - loss: 1.7128 - mean_io_u: 0.0333


175/Unknown  78s 221ms/step - categorical_accuracy: 0.5918 - loss: 1.7107 - mean_io_u: 0.0333


176/Unknown  78s 221ms/step - categorical_accuracy: 0.5923 - loss: 1.7086 - mean_io_u: 0.0334


177/Unknown  78s 220ms/step - categorical_accuracy: 0.5928 - loss: 1.7066 - mean_io_u: 0.0334


178/Unknown  79s 219ms/step - categorical_accuracy: 0.5933 - loss: 1.7045 - mean_io_u: 0.0335


179/Unknown  79s 219ms/step - categorical_accuracy: 0.5938 - loss: 1.7025 - mean_io_u: 0.0335


180/Unknown  79s 218ms/step - categorical_accuracy: 0.5943 - loss: 1.7006 - mean_io_u: 0.0335


181/Unknown  79s 218ms/step - categorical_accuracy: 0.5948 - loss: 1.6986 - mean_io_u: 0.0336


182/Unknown  79s 217ms/step - categorical_accuracy: 0.5953 - loss: 1.6966 - mean_io_u: 0.0336


183/Unknown  79s 216ms/step - categorical_accuracy: 0.5958 - loss: 1.6947 - mean_io_u: 0.0337


184/Unknown  79s 216ms/step - categorical_accuracy: 0.5963 - loss: 1.6927 - mean_io_u: 0.0337


185/Unknown  79s 215ms/step - categorical_accuracy: 0.5967 - loss: 1.6908 - mean_io_u: 0.0338


186/Unknown  79s 214ms/step - categorical_accuracy: 0.5972 - loss: 1.6888 - mean_io_u: 0.0338


187/Unknown  80s 214ms/step - categorical_accuracy: 0.5977 - loss: 1.6869 - mean_io_u: 0.0338


188/Unknown  80s 213ms/step - categorical_accuracy: 0.5982 - loss: 1.6850 - mean_io_u: 0.0339


189/Unknown  80s 213ms/step - categorical_accuracy: 0.5986 - loss: 1.6832 - mean_io_u: 0.0339


190/Unknown  80s 212ms/step - categorical_accuracy: 0.5991 - loss: 1.6813 - mean_io_u: 0.0340


191/Unknown  80s 211ms/step - categorical_accuracy: 0.5995 - loss: 1.6795 - mean_io_u: 0.0340


192/Unknown  80s 211ms/step - categorical_accuracy: 0.6000 - loss: 1.6777 - mean_io_u: 0.0340


193/Unknown  80s 210ms/step - categorical_accuracy: 0.6004 - loss: 1.6759 - mean_io_u: 0.0341


194/Unknown  80s 210ms/step - categorical_accuracy: 0.6008 - loss: 1.6742 - mean_io_u: 0.0341


195/Unknown  80s 209ms/step - categorical_accuracy: 0.6013 - loss: 1.6724 - mean_io_u: 0.0342


196/Unknown  80s 209ms/step - categorical_accuracy: 0.6017 - loss: 1.6707 - mean_io_u: 0.0342


197/Unknown  81s 208ms/step - categorical_accuracy: 0.6021 - loss: 1.6689 - mean_io_u: 0.0342


198/Unknown  81s 207ms/step - categorical_accuracy: 0.6026 - loss: 1.6672 - mean_io_u: 0.0343


199/Unknown  81s 207ms/step - categorical_accuracy: 0.6030 - loss: 1.6655 - mean_io_u: 0.0343


200/Unknown  81s 206ms/step - categorical_accuracy: 0.6034 - loss: 1.6638 - mean_io_u: 0.0344


201/Unknown  81s 206ms/step - categorical_accuracy: 0.6038 - loss: 1.6621 - mean_io_u: 0.0344


202/Unknown  81s 205ms/step - categorical_accuracy: 0.6042 - loss: 1.6604 - mean_io_u: 0.0344


203/Unknown  81s 205ms/step - categorical_accuracy: 0.6046 - loss: 1.6587 - mean_io_u: 0.0345


204/Unknown  81s 204ms/step - categorical_accuracy: 0.6050 - loss: 1.6571 - mean_io_u: 0.0345


205/Unknown  81s 204ms/step - categorical_accuracy: 0.6054 - loss: 1.6554 - mean_io_u: 0.0345


206/Unknown  81s 203ms/step - categorical_accuracy: 0.6059 - loss: 1.6538 - mean_io_u: 0.0346


207/Unknown  81s 203ms/step - categorical_accuracy: 0.6063 - loss: 1.6522 - mean_io_u: 0.0346


208/Unknown  82s 202ms/step - categorical_accuracy: 0.6067 - loss: 1.6505 - mean_io_u: 0.0347


209/Unknown  82s 202ms/step - categorical_accuracy: 0.6070 - loss: 1.6489 - mean_io_u: 0.0347


210/Unknown  82s 201ms/step - categorical_accuracy: 0.6074 - loss: 1.6474 - mean_io_u: 0.0347


211/Unknown  82s 201ms/step - categorical_accuracy: 0.6078 - loss: 1.6458 - mean_io_u: 0.0348


212/Unknown  82s 200ms/step - categorical_accuracy: 0.6082 - loss: 1.6442 - mean_io_u: 0.0348


213/Unknown  82s 200ms/step - categorical_accuracy: 0.6086 - loss: 1.6427 - mean_io_u: 0.0349


214/Unknown  82s 199ms/step - categorical_accuracy: 0.6090 - loss: 1.6411 - mean_io_u: 0.0349


215/Unknown  82s 199ms/step - categorical_accuracy: 0.6093 - loss: 1.6396 - mean_io_u: 0.0349


216/Unknown  82s 198ms/step - categorical_accuracy: 0.6097 - loss: 1.6381 - mean_io_u: 0.0350


217/Unknown  82s 198ms/step - categorical_accuracy: 0.6101 - loss: 1.6366 - mean_io_u: 0.0350


218/Unknown  83s 197ms/step - categorical_accuracy: 0.6104 - loss: 1.6351 - mean_io_u: 0.0351


219/Unknown  83s 197ms/step - categorical_accuracy: 0.6108 - loss: 1.6336 - mean_io_u: 0.0351


220/Unknown  83s 196ms/step - categorical_accuracy: 0.6112 - loss: 1.6321 - mean_io_u: 0.0351


221/Unknown  83s 196ms/step - categorical_accuracy: 0.6115 - loss: 1.6306 - mean_io_u: 0.0352


222/Unknown  83s 195ms/step - categorical_accuracy: 0.6119 - loss: 1.6292 - mean_io_u: 0.0352


223/Unknown  83s 194ms/step - categorical_accuracy: 0.6122 - loss: 1.6278 - mean_io_u: 0.0352


224/Unknown  83s 194ms/step - categorical_accuracy: 0.6126 - loss: 1.6263 - mean_io_u: 0.0353


225/Unknown  83s 193ms/step - categorical_accuracy: 0.6129 - loss: 1.6249 - mean_io_u: 0.0353


226/Unknown  83s 193ms/step - categorical_accuracy: 0.6133 - loss: 1.6235 - mean_io_u: 0.0354


227/Unknown  83s 192ms/step - categorical_accuracy: 0.6136 - loss: 1.6221 - mean_io_u: 0.0354


228/Unknown  83s 192ms/step - categorical_accuracy: 0.6139 - loss: 1.6207 - mean_io_u: 0.0354


229/Unknown  83s 191ms/step - categorical_accuracy: 0.6143 - loss: 1.6194 - mean_io_u: 0.0355


230/Unknown  83s 190ms/step - categorical_accuracy: 0.6146 - loss: 1.6181 - mean_io_u: 0.0355


231/Unknown  83s 190ms/step - categorical_accuracy: 0.6149 - loss: 1.6167 - mean_io_u: 0.0356


232/Unknown  84s 189ms/step - categorical_accuracy: 0.6152 - loss: 1.6154 - mean_io_u: 0.0356


233/Unknown  84s 189ms/step - categorical_accuracy: 0.6156 - loss: 1.6141 - mean_io_u: 0.0356


234/Unknown  84s 188ms/step - categorical_accuracy: 0.6159 - loss: 1.6128 - mean_io_u: 0.0357


235/Unknown  84s 188ms/step - categorical_accuracy: 0.6162 - loss: 1.6115 - mean_io_u: 0.0357


236/Unknown  84s 187ms/step - categorical_accuracy: 0.6165 - loss: 1.6102 - mean_io_u: 0.0357


237/Unknown  84s 187ms/step - categorical_accuracy: 0.6168 - loss: 1.6089 - mean_io_u: 0.0358


238/Unknown  84s 186ms/step - categorical_accuracy: 0.6171 - loss: 1.6076 - mean_io_u: 0.0358


239/Unknown  84s 186ms/step - categorical_accuracy: 0.6174 - loss: 1.6063 - mean_io_u: 0.0359


240/Unknown  84s 185ms/step - categorical_accuracy: 0.6177 - loss: 1.6051 - mean_io_u: 0.0359


241/Unknown  84s 185ms/step - categorical_accuracy: 0.6180 - loss: 1.6038 - mean_io_u: 0.0359


242/Unknown  84s 184ms/step - categorical_accuracy: 0.6184 - loss: 1.6026 - mean_io_u: 0.0360


243/Unknown  84s 184ms/step - categorical_accuracy: 0.6187 - loss: 1.6013 - mean_io_u: 0.0360


244/Unknown  84s 183ms/step - categorical_accuracy: 0.6190 - loss: 1.6000 - mean_io_u: 0.0361


245/Unknown  84s 183ms/step - categorical_accuracy: 0.6193 - loss: 1.5988 - mean_io_u: 0.0361


246/Unknown  84s 182ms/step - categorical_accuracy: 0.6196 - loss: 1.5976 - mean_io_u: 0.0361


247/Unknown  84s 182ms/step - categorical_accuracy: 0.6198 - loss: 1.5963 - mean_io_u: 0.0362


248/Unknown  85s 181ms/step - categorical_accuracy: 0.6201 - loss: 1.5951 - mean_io_u: 0.0362


249/Unknown  85s 181ms/step - categorical_accuracy: 0.6204 - loss: 1.5939 - mean_io_u: 0.0363


250/Unknown  85s 181ms/step - categorical_accuracy: 0.6207 - loss: 1.5927 - mean_io_u: 0.0363


251/Unknown  85s 180ms/step - categorical_accuracy: 0.6210 - loss: 1.5915 - mean_io_u: 0.0363


252/Unknown  85s 180ms/step - categorical_accuracy: 0.6213 - loss: 1.5903 - mean_io_u: 0.0364


253/Unknown  85s 179ms/step - categorical_accuracy: 0.6216 - loss: 1.5891 - mean_io_u: 0.0364


254/Unknown  85s 179ms/step - categorical_accuracy: 0.6219 - loss: 1.5880 - mean_io_u: 0.0364


255/Unknown  85s 179ms/step - categorical_accuracy: 0.6221 - loss: 1.5868 - mean_io_u: 0.0365


256/Unknown  85s 178ms/step - categorical_accuracy: 0.6224 - loss: 1.5856 - mean_io_u: 0.0365


257/Unknown  85s 178ms/step - categorical_accuracy: 0.6227 - loss: 1.5845 - mean_io_u: 0.0366


258/Unknown  85s 177ms/step - categorical_accuracy: 0.6230 - loss: 1.5833 - mean_io_u: 0.0366


259/Unknown  85s 177ms/step - categorical_accuracy: 0.6232 - loss: 1.5822 - mean_io_u: 0.0366


260/Unknown  85s 177ms/step - categorical_accuracy: 0.6235 - loss: 1.5810 - mean_io_u: 0.0367


261/Unknown  86s 176ms/step - categorical_accuracy: 0.6238 - loss: 1.5799 - mean_io_u: 0.0367


262/Unknown  86s 176ms/step - categorical_accuracy: 0.6241 - loss: 1.5787 - mean_io_u: 0.0368


263/Unknown  86s 175ms/step - categorical_accuracy: 0.6243 - loss: 1.5776 - mean_io_u: 0.0368


264/Unknown  86s 175ms/step - categorical_accuracy: 0.6246 - loss: 1.5765 - mean_io_u: 0.0368


265/Unknown  86s 175ms/step - categorical_accuracy: 0.6248 - loss: 1.5754 - mean_io_u: 0.0369


266/Unknown  86s 174ms/step - categorical_accuracy: 0.6251 - loss: 1.5743 - mean_io_u: 0.0369


267/Unknown  86s 174ms/step - categorical_accuracy: 0.6254 - loss: 1.5732 - mean_io_u: 0.0370


268/Unknown  86s 173ms/step - categorical_accuracy: 0.6256 - loss: 1.5721 - mean_io_u: 0.0370


269/Unknown  86s 173ms/step - categorical_accuracy: 0.6259 - loss: 1.5710 - mean_io_u: 0.0370


270/Unknown  86s 173ms/step - categorical_accuracy: 0.6261 - loss: 1.5700 - mean_io_u: 0.0371


271/Unknown  86s 172ms/step - categorical_accuracy: 0.6264 - loss: 1.5689 - mean_io_u: 0.0371


272/Unknown  86s 172ms/step - categorical_accuracy: 0.6267 - loss: 1.5678 - mean_io_u: 0.0371


273/Unknown  86s 172ms/step - categorical_accuracy: 0.6269 - loss: 1.5667 - mean_io_u: 0.0372


274/Unknown  87s 171ms/step - categorical_accuracy: 0.6272 - loss: 1.5657 - mean_io_u: 0.0372


275/Unknown  87s 171ms/step - categorical_accuracy: 0.6274 - loss: 1.5646 - mean_io_u: 0.0373


276/Unknown  87s 171ms/step - categorical_accuracy: 0.6277 - loss: 1.5636 - mean_io_u: 0.0373


277/Unknown  87s 170ms/step - categorical_accuracy: 0.6279 - loss: 1.5625 - mean_io_u: 0.0373


278/Unknown  87s 170ms/step - categorical_accuracy: 0.6281 - loss: 1.5615 - mean_io_u: 0.0374


279/Unknown  87s 170ms/step - categorical_accuracy: 0.6284 - loss: 1.5605 - mean_io_u: 0.0374


280/Unknown  87s 170ms/step - categorical_accuracy: 0.6286 - loss: 1.5594 - mean_io_u: 0.0374


281/Unknown  87s 169ms/step - categorical_accuracy: 0.6289 - loss: 1.5584 - mean_io_u: 0.0375


282/Unknown  87s 169ms/step - categorical_accuracy: 0.6291 - loss: 1.5574 - mean_io_u: 0.0375


283/Unknown  87s 169ms/step - categorical_accuracy: 0.6293 - loss: 1.5564 - mean_io_u: 0.0376


284/Unknown  87s 168ms/step - categorical_accuracy: 0.6296 - loss: 1.5555 - mean_io_u: 0.0376


285/Unknown  87s 168ms/step - categorical_accuracy: 0.6298 - loss: 1.5545 - mean_io_u: 0.0376


286/Unknown  88s 167ms/step - categorical_accuracy: 0.6300 - loss: 1.5535 - mean_io_u: 0.0377


287/Unknown  88s 167ms/step - categorical_accuracy: 0.6303 - loss: 1.5525 - mean_io_u: 0.0377


288/Unknown  88s 167ms/step - categorical_accuracy: 0.6305 - loss: 1.5516 - mean_io_u: 0.0377


289/Unknown  88s 166ms/step - categorical_accuracy: 0.6307 - loss: 1.5506 - mean_io_u: 0.0378


290/Unknown  88s 166ms/step - categorical_accuracy: 0.6309 - loss: 1.5497 - mean_io_u: 0.0378


291/Unknown  88s 166ms/step - categorical_accuracy: 0.6312 - loss: 1.5487 - mean_io_u: 0.0379


292/Unknown  88s 165ms/step - categorical_accuracy: 0.6314 - loss: 1.5478 - mean_io_u: 0.0379


293/Unknown  88s 165ms/step - categorical_accuracy: 0.6316 - loss: 1.5468 - mean_io_u: 0.0379


294/Unknown  88s 165ms/step - categorical_accuracy: 0.6318 - loss: 1.5459 - mean_io_u: 0.0380


295/Unknown  88s 165ms/step - categorical_accuracy: 0.6320 - loss: 1.5449 - mean_io_u: 0.0380


296/Unknown  88s 164ms/step - categorical_accuracy: 0.6323 - loss: 1.5440 - mean_io_u: 0.0380


297/Unknown  88s 164ms/step - categorical_accuracy: 0.6325 - loss: 1.5431 - mean_io_u: 0.0381


298/Unknown  88s 164ms/step - categorical_accuracy: 0.6327 - loss: 1.5421 - mean_io_u: 0.0381


299/Unknown  88s 164ms/step - categorical_accuracy: 0.6329 - loss: 1.5412 - mean_io_u: 0.0382


300/Unknown  89s 163ms/step - categorical_accuracy: 0.6331 - loss: 1.5403 - mean_io_u: 0.0382


301/Unknown  89s 163ms/step - categorical_accuracy: 0.6333 - loss: 1.5394 - mean_io_u: 0.0382


302/Unknown  89s 163ms/step - categorical_accuracy: 0.6335 - loss: 1.5385 - mean_io_u: 0.0383


303/Unknown  89s 163ms/step - categorical_accuracy: 0.6337 - loss: 1.5376 - mean_io_u: 0.0383


304/Unknown  89s 162ms/step - categorical_accuracy: 0.6340 - loss: 1.5367 - mean_io_u: 0.0383


305/Unknown  89s 162ms/step - categorical_accuracy: 0.6342 - loss: 1.5358 - mean_io_u: 0.0384


306/Unknown  89s 162ms/step - categorical_accuracy: 0.6344 - loss: 1.5349 - mean_io_u: 0.0384


307/Unknown  89s 161ms/step - categorical_accuracy: 0.6346 - loss: 1.5340 - mean_io_u: 0.0385


308/Unknown  89s 161ms/step - categorical_accuracy: 0.6348 - loss: 1.5332 - mean_io_u: 0.0385


309/Unknown  89s 161ms/step - categorical_accuracy: 0.6350 - loss: 1.5323 - mean_io_u: 0.0385


310/Unknown  89s 161ms/step - categorical_accuracy: 0.6352 - loss: 1.5314 - mean_io_u: 0.0386


311/Unknown  89s 160ms/step - categorical_accuracy: 0.6354 - loss: 1.5305 - mean_io_u: 0.0386


312/Unknown  90s 160ms/step - categorical_accuracy: 0.6356 - loss: 1.5297 - mean_io_u: 0.0386


313/Unknown  90s 160ms/step - categorical_accuracy: 0.6358 - loss: 1.5288 - mean_io_u: 0.0387


314/Unknown  90s 159ms/step - categorical_accuracy: 0.6360 - loss: 1.5279 - mean_io_u: 0.0387


315/Unknown  90s 159ms/step - categorical_accuracy: 0.6362 - loss: 1.5271 - mean_io_u: 0.0387


316/Unknown  90s 159ms/step - categorical_accuracy: 0.6364 - loss: 1.5262 - mean_io_u: 0.0388


317/Unknown  90s 159ms/step - categorical_accuracy: 0.6366 - loss: 1.5254 - mean_io_u: 0.0388


318/Unknown  90s 158ms/step - categorical_accuracy: 0.6368 - loss: 1.5245 - mean_io_u: 0.0389


319/Unknown  90s 158ms/step - categorical_accuracy: 0.6369 - loss: 1.5237 - mean_io_u: 0.0389


320/Unknown  90s 158ms/step - categorical_accuracy: 0.6371 - loss: 1.5228 - mean_io_u: 0.0389


321/Unknown  90s 158ms/step - categorical_accuracy: 0.6373 - loss: 1.5220 - mean_io_u: 0.0390


322/Unknown  90s 157ms/step - categorical_accuracy: 0.6375 - loss: 1.5211 - mean_io_u: 0.0390


323/Unknown  90s 157ms/step - categorical_accuracy: 0.6377 - loss: 1.5203 - mean_io_u: 0.0390


324/Unknown  90s 157ms/step - categorical_accuracy: 0.6379 - loss: 1.5195 - mean_io_u: 0.0391


325/Unknown  91s 157ms/step - categorical_accuracy: 0.6381 - loss: 1.5187 - mean_io_u: 0.0391


326/Unknown  91s 157ms/step - categorical_accuracy: 0.6383 - loss: 1.5178 - mean_io_u: 0.0391


327/Unknown  91s 156ms/step - categorical_accuracy: 0.6385 - loss: 1.5170 - mean_io_u: 0.0392


328/Unknown  91s 156ms/step - categorical_accuracy: 0.6386 - loss: 1.5162 - mean_io_u: 0.0392


329/Unknown  91s 156ms/step - categorical_accuracy: 0.6388 - loss: 1.5154 - mean_io_u: 0.0393


330/Unknown  91s 156ms/step - categorical_accuracy: 0.6390 - loss: 1.5146 - mean_io_u: 0.0393


331/Unknown  91s 155ms/step - categorical_accuracy: 0.6392 - loss: 1.5137 - mean_io_u: 0.0393


332/Unknown  91s 155ms/step - categorical_accuracy: 0.6394 - loss: 1.5129 - mean_io_u: 0.0394


333/Unknown  91s 155ms/step - categorical_accuracy: 0.6396 - loss: 1.5121 - mean_io_u: 0.0394


334/Unknown  91s 155ms/step - categorical_accuracy: 0.6398 - loss: 1.5113 - mean_io_u: 0.0394


335/Unknown  91s 154ms/step - categorical_accuracy: 0.6399 - loss: 1.5105 - mean_io_u: 0.0395


336/Unknown  91s 154ms/step - categorical_accuracy: 0.6401 - loss: 1.5097 - mean_io_u: 0.0395


337/Unknown  92s 154ms/step - categorical_accuracy: 0.6403 - loss: 1.5089 - mean_io_u: 0.0396


338/Unknown  92s 154ms/step - categorical_accuracy: 0.6405 - loss: 1.5081 - mean_io_u: 0.0396


339/Unknown  92s 154ms/step - categorical_accuracy: 0.6406 - loss: 1.5073 - mean_io_u: 0.0396


340/Unknown  92s 153ms/step - categorical_accuracy: 0.6408 - loss: 1.5066 - mean_io_u: 0.0397


341/Unknown  92s 153ms/step - categorical_accuracy: 0.6410 - loss: 1.5058 - mean_io_u: 0.0397


342/Unknown  92s 153ms/step - categorical_accuracy: 0.6412 - loss: 1.5050 - mean_io_u: 0.0397


343/Unknown  92s 153ms/step - categorical_accuracy: 0.6413 - loss: 1.5043 - mean_io_u: 0.0398


344/Unknown  92s 152ms/step - categorical_accuracy: 0.6415 - loss: 1.5035 - mean_io_u: 0.0398


345/Unknown  92s 152ms/step - categorical_accuracy: 0.6417 - loss: 1.5027 - mean_io_u: 0.0399


346/Unknown  92s 152ms/step - categorical_accuracy: 0.6419 - loss: 1.5020 - mean_io_u: 0.0399


347/Unknown  92s 152ms/step - categorical_accuracy: 0.6420 - loss: 1.5012 - mean_io_u: 0.0399


348/Unknown  92s 151ms/step - categorical_accuracy: 0.6422 - loss: 1.5005 - mean_io_u: 0.0400


349/Unknown  92s 151ms/step - categorical_accuracy: 0.6424 - loss: 1.4997 - mean_io_u: 0.0400


350/Unknown  92s 151ms/step - categorical_accuracy: 0.6425 - loss: 1.4990 - mean_io_u: 0.0401


351/Unknown  93s 151ms/step - categorical_accuracy: 0.6427 - loss: 1.4982 - mean_io_u: 0.0401


352/Unknown  93s 151ms/step - categorical_accuracy: 0.6428 - loss: 1.4975 - mean_io_u: 0.0401


353/Unknown  93s 150ms/step - categorical_accuracy: 0.6430 - loss: 1.4967 - mean_io_u: 0.0402


354/Unknown  93s 150ms/step - categorical_accuracy: 0.6432 - loss: 1.4960 - mean_io_u: 0.0402


355/Unknown  93s 150ms/step - categorical_accuracy: 0.6433 - loss: 1.4953 - mean_io_u: 0.0402


356/Unknown  93s 150ms/step - categorical_accuracy: 0.6435 - loss: 1.4945 - mean_io_u: 0.0403


357/Unknown  93s 150ms/step - categorical_accuracy: 0.6437 - loss: 1.4938 - mean_io_u: 0.0403


358/Unknown  93s 149ms/step - categorical_accuracy: 0.6438 - loss: 1.4930 - mean_io_u: 0.0404


359/Unknown  93s 149ms/step - categorical_accuracy: 0.6440 - loss: 1.4923 - mean_io_u: 0.0404


360/Unknown  93s 149ms/step - categorical_accuracy: 0.6442 - loss: 1.4916 - mean_io_u: 0.0404


361/Unknown  93s 149ms/step - categorical_accuracy: 0.6443 - loss: 1.4909 - mean_io_u: 0.0405


362/Unknown  93s 149ms/step - categorical_accuracy: 0.6445 - loss: 1.4901 - mean_io_u: 0.0405


363/Unknown  94s 148ms/step - categorical_accuracy: 0.6446 - loss: 1.4894 - mean_io_u: 0.0406


364/Unknown  94s 148ms/step - categorical_accuracy: 0.6448 - loss: 1.4887 - mean_io_u: 0.0406


365/Unknown  94s 148ms/step - categorical_accuracy: 0.6449 - loss: 1.4880 - mean_io_u: 0.0406


366/Unknown  94s 148ms/step - categorical_accuracy: 0.6451 - loss: 1.4873 - mean_io_u: 0.0407


367/Unknown  94s 148ms/step - categorical_accuracy: 0.6453 - loss: 1.4866 - mean_io_u: 0.0407


368/Unknown  94s 147ms/step - categorical_accuracy: 0.6454 - loss: 1.4859 - mean_io_u: 0.0407


369/Unknown  94s 147ms/step - categorical_accuracy: 0.6456 - loss: 1.4851 - mean_io_u: 0.0408


370/Unknown  94s 147ms/step - categorical_accuracy: 0.6457 - loss: 1.4844 - mean_io_u: 0.0408


371/Unknown  94s 147ms/step - categorical_accuracy: 0.6459 - loss: 1.4837 - mean_io_u: 0.0409


372/Unknown  94s 147ms/step - categorical_accuracy: 0.6460 - loss: 1.4831 - mean_io_u: 0.0409


373/Unknown  94s 147ms/step - categorical_accuracy: 0.6462 - loss: 1.4824 - mean_io_u: 0.0409


374/Unknown  94s 146ms/step - categorical_accuracy: 0.6463 - loss: 1.4817 - mean_io_u: 0.0410


375/Unknown  94s 146ms/step - categorical_accuracy: 0.6465 - loss: 1.4810 - mean_io_u: 0.0410


376/Unknown  94s 146ms/step - categorical_accuracy: 0.6466 - loss: 1.4803 - mean_io_u: 0.0410


377/Unknown  95s 146ms/step - categorical_accuracy: 0.6468 - loss: 1.4796 - mean_io_u: 0.0411


378/Unknown  95s 146ms/step - categorical_accuracy: 0.6469 - loss: 1.4789 - mean_io_u: 0.0411


379/Unknown  95s 145ms/step - categorical_accuracy: 0.6471 - loss: 1.4783 - mean_io_u: 0.0412


380/Unknown  95s 145ms/step - categorical_accuracy: 0.6472 - loss: 1.4776 - mean_io_u: 0.0412


381/Unknown  95s 145ms/step - categorical_accuracy: 0.6474 - loss: 1.4769 - mean_io_u: 0.0412


382/Unknown  95s 145ms/step - categorical_accuracy: 0.6475 - loss: 1.4763 - mean_io_u: 0.0413


383/Unknown  95s 145ms/step - categorical_accuracy: 0.6477 - loss: 1.4756 - mean_io_u: 0.0413


384/Unknown  95s 145ms/step - categorical_accuracy: 0.6478 - loss: 1.4749 - mean_io_u: 0.0413


385/Unknown  95s 144ms/step - categorical_accuracy: 0.6479 - loss: 1.4743 - mean_io_u: 0.0414


386/Unknown  95s 144ms/step - categorical_accuracy: 0.6481 - loss: 1.4736 - mean_io_u: 0.0414


387/Unknown  95s 144ms/step - categorical_accuracy: 0.6482 - loss: 1.4730 - mean_io_u: 0.0415


388/Unknown  95s 144ms/step - categorical_accuracy: 0.6484 - loss: 1.4723 - mean_io_u: 0.0415


389/Unknown  95s 143ms/step - categorical_accuracy: 0.6485 - loss: 1.4716 - mean_io_u: 0.0415


390/Unknown  95s 143ms/step - categorical_accuracy: 0.6487 - loss: 1.4710 - mean_io_u: 0.0416


391/Unknown  96s 143ms/step - categorical_accuracy: 0.6488 - loss: 1.4703 - mean_io_u: 0.0416


392/Unknown  96s 143ms/step - categorical_accuracy: 0.6489 - loss: 1.4697 - mean_io_u: 0.0417


393/Unknown  96s 143ms/step - categorical_accuracy: 0.6491 - loss: 1.4690 - mean_io_u: 0.0417


394/Unknown  96s 142ms/step - categorical_accuracy: 0.6492 - loss: 1.4684 - mean_io_u: 0.0417


395/Unknown  96s 142ms/step - categorical_accuracy: 0.6493 - loss: 1.4678 - mean_io_u: 0.0418


396/Unknown  96s 142ms/step - categorical_accuracy: 0.6495 - loss: 1.4671 - mean_io_u: 0.0418


397/Unknown  96s 142ms/step - categorical_accuracy: 0.6496 - loss: 1.4665 - mean_io_u: 0.0419


398/Unknown  96s 142ms/step - categorical_accuracy: 0.6498 - loss: 1.4658 - mean_io_u: 0.0419


399/Unknown  96s 141ms/step - categorical_accuracy: 0.6499 - loss: 1.4652 - mean_io_u: 0.0419


400/Unknown  96s 141ms/step - categorical_accuracy: 0.6500 - loss: 1.4646 - mean_io_u: 0.0420


401/Unknown  96s 141ms/step - categorical_accuracy: 0.6502 - loss: 1.4639 - mean_io_u: 0.0420


402/Unknown  96s 141ms/step - categorical_accuracy: 0.6503 - loss: 1.4633 - mean_io_u: 0.0421


403/Unknown  96s 141ms/step - categorical_accuracy: 0.6504 - loss: 1.4627 - mean_io_u: 0.0421


404/Unknown  97s 141ms/step - categorical_accuracy: 0.6506 - loss: 1.4620 - mean_io_u: 0.0421


405/Unknown  97s 141ms/step - categorical_accuracy: 0.6507 - loss: 1.4614 - mean_io_u: 0.0422


406/Unknown  97s 140ms/step - categorical_accuracy: 0.6508 - loss: 1.4608 - mean_io_u: 0.0422


407/Unknown  97s 140ms/step - categorical_accuracy: 0.6510 - loss: 1.4601 - mean_io_u: 0.0423


408/Unknown  97s 140ms/step - categorical_accuracy: 0.6511 - loss: 1.4595 - mean_io_u: 0.0423


409/Unknown  97s 140ms/step - categorical_accuracy: 0.6512 - loss: 1.4589 - mean_io_u: 0.0423


410/Unknown  97s 140ms/step - categorical_accuracy: 0.6514 - loss: 1.4583 - mean_io_u: 0.0424


411/Unknown  97s 140ms/step - categorical_accuracy: 0.6515 - loss: 1.4576 - mean_io_u: 0.0424


412/Unknown  97s 139ms/step - categorical_accuracy: 0.6516 - loss: 1.4570 - mean_io_u: 0.0425


413/Unknown  97s 139ms/step - categorical_accuracy: 0.6518 - loss: 1.4564 - mean_io_u: 0.0425


414/Unknown  97s 139ms/step - categorical_accuracy: 0.6519 - loss: 1.4558 - mean_io_u: 0.0425


415/Unknown  97s 139ms/step - categorical_accuracy: 0.6520 - loss: 1.4552 - mean_io_u: 0.0426


416/Unknown  97s 139ms/step - categorical_accuracy: 0.6522 - loss: 1.4546 - mean_io_u: 0.0426


417/Unknown  97s 139ms/step - categorical_accuracy: 0.6523 - loss: 1.4540 - mean_io_u: 0.0427


418/Unknown  98s 139ms/step - categorical_accuracy: 0.6524 - loss: 1.4534 - mean_io_u: 0.0427


419/Unknown  98s 138ms/step - categorical_accuracy: 0.6526 - loss: 1.4528 - mean_io_u: 0.0427


420/Unknown  98s 138ms/step - categorical_accuracy: 0.6527 - loss: 1.4522 - mean_io_u: 0.0428


421/Unknown  98s 138ms/step - categorical_accuracy: 0.6528 - loss: 1.4516 - mean_io_u: 0.0428


422/Unknown  98s 138ms/step - categorical_accuracy: 0.6529 - loss: 1.4510 - mean_io_u: 0.0429


423/Unknown  98s 138ms/step - categorical_accuracy: 0.6531 - loss: 1.4504 - mean_io_u: 0.0429


424/Unknown  98s 138ms/step - categorical_accuracy: 0.6532 - loss: 1.4498 - mean_io_u: 0.0429


425/Unknown  98s 138ms/step - categorical_accuracy: 0.6533 - loss: 1.4492 - mean_io_u: 0.0430


426/Unknown  98s 137ms/step - categorical_accuracy: 0.6534 - loss: 1.4486 - mean_io_u: 0.0430


427/Unknown  98s 137ms/step - categorical_accuracy: 0.6536 - loss: 1.4480 - mean_io_u: 0.0431


428/Unknown  98s 137ms/step - categorical_accuracy: 0.6537 - loss: 1.4474 - mean_io_u: 0.0431


429/Unknown  98s 137ms/step - categorical_accuracy: 0.6538 - loss: 1.4468 - mean_io_u: 0.0431


430/Unknown  99s 137ms/step - categorical_accuracy: 0.6539 - loss: 1.4462 - mean_io_u: 0.0432


431/Unknown  99s 137ms/step - categorical_accuracy: 0.6541 - loss: 1.4457 - mean_io_u: 0.0432


432/Unknown  99s 137ms/step - categorical_accuracy: 0.6542 - loss: 1.4451 - mean_io_u: 0.0432


433/Unknown  99s 137ms/step - categorical_accuracy: 0.6543 - loss: 1.4445 - mean_io_u: 0.0433


434/Unknown  99s 137ms/step - categorical_accuracy: 0.6544 - loss: 1.4439 - mean_io_u: 0.0433


435/Unknown  99s 136ms/step - categorical_accuracy: 0.6546 - loss: 1.4433 - mean_io_u: 0.0434


436/Unknown  99s 136ms/step - categorical_accuracy: 0.6547 - loss: 1.4428 - mean_io_u: 0.0434


437/Unknown  99s 136ms/step - categorical_accuracy: 0.6548 - loss: 1.4422 - mean_io_u: 0.0434


438/Unknown  99s 136ms/step - categorical_accuracy: 0.6549 - loss: 1.4416 - mean_io_u: 0.0435


439/Unknown  99s 136ms/step - categorical_accuracy: 0.6551 - loss: 1.4411 - mean_io_u: 0.0435


440/Unknown  99s 136ms/step - categorical_accuracy: 0.6552 - loss: 1.4405 - mean_io_u: 0.0435


441/Unknown  99s 135ms/step - categorical_accuracy: 0.6553 - loss: 1.4399 - mean_io_u: 0.0436


442/Unknown  99s 135ms/step - categorical_accuracy: 0.6554 - loss: 1.4394 - mean_io_u: 0.0436


443/Unknown  100s 135ms/step - categorical_accuracy: 0.6555 - loss: 1.4388 - mean_io_u: 0.0437


444/Unknown  100s 135ms/step - categorical_accuracy: 0.6557 - loss: 1.4382 - mean_io_u: 0.0437


445/Unknown  100s 135ms/step - categorical_accuracy: 0.6558 - loss: 1.4377 - mean_io_u: 0.0437


446/Unknown  100s 135ms/step - categorical_accuracy: 0.6559 - loss: 1.4371 - mean_io_u: 0.0438


447/Unknown  100s 135ms/step - categorical_accuracy: 0.6560 - loss: 1.4366 - mean_io_u: 0.0438


448/Unknown  100s 134ms/step - categorical_accuracy: 0.6561 - loss: 1.4360 - mean_io_u: 0.0439


449/Unknown  100s 134ms/step - categorical_accuracy: 0.6562 - loss: 1.4355 - mean_io_u: 0.0439


450/Unknown  100s 134ms/step - categorical_accuracy: 0.6564 - loss: 1.4349 - mean_io_u: 0.0439


451/Unknown  100s 134ms/step - categorical_accuracy: 0.6565 - loss: 1.4344 - mean_io_u: 0.0440


452/Unknown  100s 134ms/step - categorical_accuracy: 0.6566 - loss: 1.4338 - mean_io_u: 0.0440


453/Unknown  100s 134ms/step - categorical_accuracy: 0.6567 - loss: 1.4333 - mean_io_u: 0.0441


454/Unknown  100s 134ms/step - categorical_accuracy: 0.6568 - loss: 1.4328 - mean_io_u: 0.0441


455/Unknown  100s 134ms/step - categorical_accuracy: 0.6569 - loss: 1.4322 - mean_io_u: 0.0441


456/Unknown  101s 134ms/step - categorical_accuracy: 0.6570 - loss: 1.4317 - mean_io_u: 0.0442


457/Unknown  101s 134ms/step - categorical_accuracy: 0.6572 - loss: 1.4311 - mean_io_u: 0.0442


458/Unknown  101s 133ms/step - categorical_accuracy: 0.6573 - loss: 1.4306 - mean_io_u: 0.0442


459/Unknown  101s 133ms/step - categorical_accuracy: 0.6574 - loss: 1.4301 - mean_io_u: 0.0443


460/Unknown  101s 133ms/step - categorical_accuracy: 0.6575 - loss: 1.4295 - mean_io_u: 0.0443


461/Unknown  101s 133ms/step - categorical_accuracy: 0.6576 - loss: 1.4290 - mean_io_u: 0.0444


462/Unknown  101s 133ms/step - categorical_accuracy: 0.6577 - loss: 1.4285 - mean_io_u: 0.0444


463/Unknown  101s 133ms/step - categorical_accuracy: 0.6578 - loss: 1.4279 - mean_io_u: 0.0444


464/Unknown  101s 133ms/step - categorical_accuracy: 0.6579 - loss: 1.4274 - mean_io_u: 0.0445


465/Unknown  101s 133ms/step - categorical_accuracy: 0.6581 - loss: 1.4269 - mean_io_u: 0.0445


466/Unknown  101s 132ms/step - categorical_accuracy: 0.6582 - loss: 1.4264 - mean_io_u: 0.0445


467/Unknown  101s 132ms/step - categorical_accuracy: 0.6583 - loss: 1.4259 - mean_io_u: 0.0446


468/Unknown  102s 132ms/step - categorical_accuracy: 0.6584 - loss: 1.4253 - mean_io_u: 0.0446


469/Unknown  102s 132ms/step - categorical_accuracy: 0.6585 - loss: 1.4248 - mean_io_u: 0.0447


470/Unknown  102s 132ms/step - categorical_accuracy: 0.6586 - loss: 1.4243 - mean_io_u: 0.0447


471/Unknown  102s 132ms/step - categorical_accuracy: 0.6587 - loss: 1.4238 - mean_io_u: 0.0447


472/Unknown  102s 132ms/step - categorical_accuracy: 0.6588 - loss: 1.4233 - mean_io_u: 0.0448


473/Unknown  102s 132ms/step - categorical_accuracy: 0.6589 - loss: 1.4228 - mean_io_u: 0.0448


474/Unknown  102s 132ms/step - categorical_accuracy: 0.6590 - loss: 1.4223 - mean_io_u: 0.0449


475/Unknown  102s 132ms/step - categorical_accuracy: 0.6591 - loss: 1.4217 - mean_io_u: 0.0449


476/Unknown  102s 131ms/step - categorical_accuracy: 0.6592 - loss: 1.4212 - mean_io_u: 0.0449


477/Unknown  102s 131ms/step - categorical_accuracy: 0.6593 - loss: 1.4207 - mean_io_u: 0.0450


478/Unknown  102s 131ms/step - categorical_accuracy: 0.6595 - loss: 1.4202 - mean_io_u: 0.0450


479/Unknown  102s 131ms/step - categorical_accuracy: 0.6596 - loss: 1.4197 - mean_io_u: 0.0450


480/Unknown  102s 131ms/step - categorical_accuracy: 0.6597 - loss: 1.4192 - mean_io_u: 0.0451


481/Unknown  103s 131ms/step - categorical_accuracy: 0.6598 - loss: 1.4187 - mean_io_u: 0.0451


482/Unknown  103s 131ms/step - categorical_accuracy: 0.6599 - loss: 1.4182 - mean_io_u: 0.0452


483/Unknown  103s 131ms/step - categorical_accuracy: 0.6600 - loss: 1.4177 - mean_io_u: 0.0452


484/Unknown  103s 130ms/step - categorical_accuracy: 0.6601 - loss: 1.4172 - mean_io_u: 0.0452


485/Unknown  103s 130ms/step - categorical_accuracy: 0.6602 - loss: 1.4167 - mean_io_u: 0.0453


486/Unknown  103s 130ms/step - categorical_accuracy: 0.6603 - loss: 1.4162 - mean_io_u: 0.0453


487/Unknown  103s 130ms/step - categorical_accuracy: 0.6604 - loss: 1.4157 - mean_io_u: 0.0454


488/Unknown  103s 130ms/step - categorical_accuracy: 0.6605 - loss: 1.4152 - mean_io_u: 0.0454


489/Unknown  103s 130ms/step - categorical_accuracy: 0.6606 - loss: 1.4147 - mean_io_u: 0.0454


490/Unknown  103s 130ms/step - categorical_accuracy: 0.6607 - loss: 1.4142 - mean_io_u: 0.0455


491/Unknown  103s 130ms/step - categorical_accuracy: 0.6608 - loss: 1.4137 - mean_io_u: 0.0455


492/Unknown  103s 130ms/step - categorical_accuracy: 0.6609 - loss: 1.4133 - mean_io_u: 0.0456


493/Unknown  104s 130ms/step - categorical_accuracy: 0.6610 - loss: 1.4128 - mean_io_u: 0.0456


494/Unknown  104s 130ms/step - categorical_accuracy: 0.6611 - loss: 1.4123 - mean_io_u: 0.0456


495/Unknown  104s 129ms/step - categorical_accuracy: 0.6612 - loss: 1.4118 - mean_io_u: 0.0457


496/Unknown  104s 129ms/step - categorical_accuracy: 0.6613 - loss: 1.4113 - mean_io_u: 0.0457


497/Unknown  104s 129ms/step - categorical_accuracy: 0.6614 - loss: 1.4108 - mean_io_u: 0.0458


498/Unknown  104s 129ms/step - categorical_accuracy: 0.6615 - loss: 1.4103 - mean_io_u: 0.0458


499/Unknown  104s 129ms/step - categorical_accuracy: 0.6616 - loss: 1.4098 - mean_io_u: 0.0458


500/Unknown  104s 129ms/step - categorical_accuracy: 0.6617 - loss: 1.4093 - mean_io_u: 0.0459


501/Unknown  104s 129ms/step - categorical_accuracy: 0.6618 - loss: 1.4089 - mean_io_u: 0.0459


502/Unknown  104s 129ms/step - categorical_accuracy: 0.6619 - loss: 1.4084 - mean_io_u: 0.0460


503/Unknown  104s 129ms/step - categorical_accuracy: 0.6620 - loss: 1.4079 - mean_io_u: 0.0460


504/Unknown  104s 128ms/step - categorical_accuracy: 0.6621 - loss: 1.4074 - mean_io_u: 0.0460


505/Unknown  104s 128ms/step - categorical_accuracy: 0.6622 - loss: 1.4070 - mean_io_u: 0.0461


506/Unknown  105s 128ms/step - categorical_accuracy: 0.6623 - loss: 1.4065 - mean_io_u: 0.0461


507/Unknown  105s 128ms/step - categorical_accuracy: 0.6624 - loss: 1.4060 - mean_io_u: 0.0462


508/Unknown  105s 128ms/step - categorical_accuracy: 0.6625 - loss: 1.4056 - mean_io_u: 0.0462


509/Unknown  105s 128ms/step - categorical_accuracy: 0.6626 - loss: 1.4051 - mean_io_u: 0.0463


510/Unknown  105s 128ms/step - categorical_accuracy: 0.6627 - loss: 1.4046 - mean_io_u: 0.0463


511/Unknown  105s 128ms/step - categorical_accuracy: 0.6628 - loss: 1.4042 - mean_io_u: 0.0463


512/Unknown  105s 128ms/step - categorical_accuracy: 0.6629 - loss: 1.4037 - mean_io_u: 0.0464


513/Unknown  105s 127ms/step - categorical_accuracy: 0.6630 - loss: 1.4032 - mean_io_u: 0.0464


514/Unknown  105s 127ms/step - categorical_accuracy: 0.6631 - loss: 1.4028 - mean_io_u: 0.0465


515/Unknown  105s 127ms/step - categorical_accuracy: 0.6632 - loss: 1.4023 - mean_io_u: 0.0465


516/Unknown  105s 127ms/step - categorical_accuracy: 0.6633 - loss: 1.4018 - mean_io_u: 0.0465


517/Unknown  105s 127ms/step - categorical_accuracy: 0.6634 - loss: 1.4014 - mean_io_u: 0.0466


518/Unknown  105s 127ms/step - categorical_accuracy: 0.6635 - loss: 1.4009 - mean_io_u: 0.0466


519/Unknown  106s 127ms/step - categorical_accuracy: 0.6636 - loss: 1.4004 - mean_io_u: 0.0467


520/Unknown  106s 127ms/step - categorical_accuracy: 0.6637 - loss: 1.4000 - mean_io_u: 0.0467


521/Unknown  106s 127ms/step - categorical_accuracy: 0.6638 - loss: 1.3995 - mean_io_u: 0.0468


522/Unknown  106s 127ms/step - categorical_accuracy: 0.6639 - loss: 1.3991 - mean_io_u: 0.0468


523/Unknown  106s 127ms/step - categorical_accuracy: 0.6640 - loss: 1.3986 - mean_io_u: 0.0468


524/Unknown  106s 127ms/step - categorical_accuracy: 0.6641 - loss: 1.3981 - mean_io_u: 0.0469


525/Unknown  106s 126ms/step - categorical_accuracy: 0.6642 - loss: 1.3977 - mean_io_u: 0.0469


526/Unknown  106s 126ms/step - categorical_accuracy: 0.6643 - loss: 1.3972 - mean_io_u: 0.0470


527/Unknown  106s 126ms/step - categorical_accuracy: 0.6644 - loss: 1.3968 - mean_io_u: 0.0470


528/Unknown  106s 126ms/step - categorical_accuracy: 0.6645 - loss: 1.3963 - mean_io_u: 0.0471


529/Unknown  106s 126ms/step - categorical_accuracy: 0.6645 - loss: 1.3958 - mean_io_u: 0.0471


530/Unknown  106s 126ms/step - categorical_accuracy: 0.6646 - loss: 1.3954 - mean_io_u: 0.0471


531/Unknown  106s 126ms/step - categorical_accuracy: 0.6647 - loss: 1.3949 - mean_io_u: 0.0472


532/Unknown  107s 126ms/step - categorical_accuracy: 0.6648 - loss: 1.3945 - mean_io_u: 0.0472


533/Unknown  107s 126ms/step - categorical_accuracy: 0.6649 - loss: 1.3940 - mean_io_u: 0.0473


534/Unknown  107s 126ms/step - categorical_accuracy: 0.6650 - loss: 1.3935 - mean_io_u: 0.0473


535/Unknown  107s 126ms/step - categorical_accuracy: 0.6651 - loss: 1.3931 - mean_io_u: 0.0474


536/Unknown  107s 125ms/step - categorical_accuracy: 0.6652 - loss: 1.3926 - mean_io_u: 0.0474


537/Unknown  107s 125ms/step - categorical_accuracy: 0.6653 - loss: 1.3922 - mean_io_u: 0.0474


538/Unknown  107s 125ms/step - categorical_accuracy: 0.6654 - loss: 1.3917 - mean_io_u: 0.0475


539/Unknown  107s 125ms/step - categorical_accuracy: 0.6655 - loss: 1.3913 - mean_io_u: 0.0475


540/Unknown  107s 125ms/step - categorical_accuracy: 0.6656 - loss: 1.3908 - mean_io_u: 0.0476


541/Unknown  107s 125ms/step - categorical_accuracy: 0.6657 - loss: 1.3904 - mean_io_u: 0.0476


542/Unknown  107s 125ms/step - categorical_accuracy: 0.6658 - loss: 1.3899 - mean_io_u: 0.0477


543/Unknown  107s 125ms/step - categorical_accuracy: 0.6659 - loss: 1.3895 - mean_io_u: 0.0477


544/Unknown  107s 124ms/step - categorical_accuracy: 0.6660 - loss: 1.3890 - mean_io_u: 0.0477


545/Unknown  107s 124ms/step - categorical_accuracy: 0.6661 - loss: 1.3886 - mean_io_u: 0.0478


546/Unknown  108s 124ms/step - categorical_accuracy: 0.6661 - loss: 1.3881 - mean_io_u: 0.0478


547/Unknown  108s 124ms/step - categorical_accuracy: 0.6662 - loss: 1.3877 - mean_io_u: 0.0479


548/Unknown  108s 124ms/step - categorical_accuracy: 0.6663 - loss: 1.3873 - mean_io_u: 0.0479


549/Unknown  108s 124ms/step - categorical_accuracy: 0.6664 - loss: 1.3868 - mean_io_u: 0.0480


550/Unknown  108s 124ms/step - categorical_accuracy: 0.6665 - loss: 1.3864 - mean_io_u: 0.0480


551/Unknown  108s 124ms/step - categorical_accuracy: 0.6666 - loss: 1.3860 - mean_io_u: 0.0481


552/Unknown  108s 124ms/step - categorical_accuracy: 0.6667 - loss: 1.3855 - mean_io_u: 0.0481


553/Unknown  108s 124ms/step - categorical_accuracy: 0.6668 - loss: 1.3851 - mean_io_u: 0.0481


554/Unknown  108s 124ms/step - categorical_accuracy: 0.6669 - loss: 1.3846 - mean_io_u: 0.0482


555/Unknown  108s 123ms/step - categorical_accuracy: 0.6670 - loss: 1.3842 - mean_io_u: 0.0482


556/Unknown  108s 123ms/step - categorical_accuracy: 0.6670 - loss: 1.3838 - mean_io_u: 0.0483


557/Unknown  108s 123ms/step - categorical_accuracy: 0.6671 - loss: 1.3834 - mean_io_u: 0.0483


558/Unknown  108s 123ms/step - categorical_accuracy: 0.6672 - loss: 1.3829 - mean_io_u: 0.0484


559/Unknown  108s 123ms/step - categorical_accuracy: 0.6673 - loss: 1.3825 - mean_io_u: 0.0484


560/Unknown  109s 123ms/step - categorical_accuracy: 0.6674 - loss: 1.3821 - mean_io_u: 0.0484


561/Unknown  109s 123ms/step - categorical_accuracy: 0.6675 - loss: 1.3817 - mean_io_u: 0.0485


562/Unknown  109s 123ms/step - categorical_accuracy: 0.6676 - loss: 1.3812 - mean_io_u: 0.0485


563/Unknown  109s 123ms/step - categorical_accuracy: 0.6677 - loss: 1.3808 - mean_io_u: 0.0486


564/Unknown  109s 123ms/step - categorical_accuracy: 0.6677 - loss: 1.3804 - mean_io_u: 0.0486


565/Unknown  109s 123ms/step - categorical_accuracy: 0.6678 - loss: 1.3800 - mean_io_u: 0.0487


566/Unknown  109s 123ms/step - categorical_accuracy: 0.6679 - loss: 1.3795 - mean_io_u: 0.0487


567/Unknown  109s 123ms/step - categorical_accuracy: 0.6680 - loss: 1.3791 - mean_io_u: 0.0488


568/Unknown  109s 122ms/step - categorical_accuracy: 0.6681 - loss: 1.3787 - mean_io_u: 0.0488


569/Unknown  109s 122ms/step - categorical_accuracy: 0.6682 - loss: 1.3783 - mean_io_u: 0.0488


570/Unknown  109s 122ms/step - categorical_accuracy: 0.6683 - loss: 1.3779 - mean_io_u: 0.0489


571/Unknown  109s 122ms/step - categorical_accuracy: 0.6683 - loss: 1.3774 - mean_io_u: 0.0489


572/Unknown  109s 122ms/step - categorical_accuracy: 0.6684 - loss: 1.3770 - mean_io_u: 0.0490


573/Unknown  110s 122ms/step - categorical_accuracy: 0.6685 - loss: 1.3766 - mean_io_u: 0.0490


574/Unknown  110s 122ms/step - categorical_accuracy: 0.6686 - loss: 1.3762 - mean_io_u: 0.0491


575/Unknown  110s 122ms/step - categorical_accuracy: 0.6687 - loss: 1.3758 - mean_io_u: 0.0491


576/Unknown  110s 122ms/step - categorical_accuracy: 0.6688 - loss: 1.3754 - mean_io_u: 0.0491


577/Unknown  110s 122ms/step - categorical_accuracy: 0.6688 - loss: 1.3750 - mean_io_u: 0.0492


578/Unknown  110s 122ms/step - categorical_accuracy: 0.6689 - loss: 1.3746 - mean_io_u: 0.0492


579/Unknown  110s 122ms/step - categorical_accuracy: 0.6690 - loss: 1.3742 - mean_io_u: 0.0493


580/Unknown  110s 122ms/step - categorical_accuracy: 0.6691 - loss: 1.3737 - mean_io_u: 0.0493


581/Unknown  110s 122ms/step - categorical_accuracy: 0.6692 - loss: 1.3733 - mean_io_u: 0.0493


582/Unknown  110s 122ms/step - categorical_accuracy: 0.6693 - loss: 1.3729 - mean_io_u: 0.0494


583/Unknown  110s 121ms/step - categorical_accuracy: 0.6693 - loss: 1.3725 - mean_io_u: 0.0494


584/Unknown  111s 121ms/step - categorical_accuracy: 0.6694 - loss: 1.3721 - mean_io_u: 0.0495


585/Unknown  111s 121ms/step - categorical_accuracy: 0.6695 - loss: 1.3717 - mean_io_u: 0.0495


586/Unknown  111s 121ms/step - categorical_accuracy: 0.6696 - loss: 1.3713 - mean_io_u: 0.0496


587/Unknown  111s 121ms/step - categorical_accuracy: 0.6697 - loss: 1.3709 - mean_io_u: 0.0496


588/Unknown  111s 121ms/step - categorical_accuracy: 0.6697 - loss: 1.3705 - mean_io_u: 0.0496


589/Unknown  111s 121ms/step - categorical_accuracy: 0.6698 - loss: 1.3701 - mean_io_u: 0.0497


590/Unknown  111s 121ms/step - categorical_accuracy: 0.6699 - loss: 1.3697 - mean_io_u: 0.0497


591/Unknown  111s 121ms/step - categorical_accuracy: 0.6700 - loss: 1.3693 - mean_io_u: 0.0498


592/Unknown  111s 121ms/step - categorical_accuracy: 0.6701 - loss: 1.3689 - mean_io_u: 0.0498


593/Unknown  111s 121ms/step - categorical_accuracy: 0.6701 - loss: 1.3686 - mean_io_u: 0.0499


594/Unknown  111s 121ms/step - categorical_accuracy: 0.6702 - loss: 1.3682 - mean_io_u: 0.0499


595/Unknown  111s 121ms/step - categorical_accuracy: 0.6703 - loss: 1.3678 - mean_io_u: 0.0499


596/Unknown  111s 120ms/step - categorical_accuracy: 0.6704 - loss: 1.3674 - mean_io_u: 0.0500


597/Unknown  112s 120ms/step - categorical_accuracy: 0.6705 - loss: 1.3670 - mean_io_u: 0.0500


598/Unknown  112s 120ms/step - categorical_accuracy: 0.6705 - loss: 1.3666 - mean_io_u: 0.0501


599/Unknown  112s 120ms/step - categorical_accuracy: 0.6706 - loss: 1.3662 - mean_io_u: 0.0501


600/Unknown  112s 120ms/step - categorical_accuracy: 0.6707 - loss: 1.3658 - mean_io_u: 0.0501


601/Unknown  112s 120ms/step - categorical_accuracy: 0.6708 - loss: 1.3654 - mean_io_u: 0.0502


602/Unknown  112s 120ms/step - categorical_accuracy: 0.6709 - loss: 1.3651 - mean_io_u: 0.0502


603/Unknown  112s 120ms/step - categorical_accuracy: 0.6709 - loss: 1.3647 - mean_io_u: 0.0503


604/Unknown  112s 120ms/step - categorical_accuracy: 0.6710 - loss: 1.3643 - mean_io_u: 0.0503


605/Unknown  112s 120ms/step - categorical_accuracy: 0.6711 - loss: 1.3639 - mean_io_u: 0.0504


606/Unknown  112s 120ms/step - categorical_accuracy: 0.6712 - loss: 1.3635 - mean_io_u: 0.0504


607/Unknown  112s 120ms/step - categorical_accuracy: 0.6712 - loss: 1.3632 - mean_io_u: 0.0504


608/Unknown  113s 120ms/step - categorical_accuracy: 0.6713 - loss: 1.3628 - mean_io_u: 0.0505


609/Unknown  113s 120ms/step - categorical_accuracy: 0.6714 - loss: 1.3624 - mean_io_u: 0.0505


610/Unknown  113s 120ms/step - categorical_accuracy: 0.6715 - loss: 1.3620 - mean_io_u: 0.0506


611/Unknown  113s 120ms/step - categorical_accuracy: 0.6715 - loss: 1.3616 - mean_io_u: 0.0506


612/Unknown  113s 120ms/step - categorical_accuracy: 0.6716 - loss: 1.3613 - mean_io_u: 0.0506


613/Unknown  113s 120ms/step - categorical_accuracy: 0.6717 - loss: 1.3609 - mean_io_u: 0.0507


614/Unknown  113s 120ms/step - categorical_accuracy: 0.6718 - loss: 1.3605 - mean_io_u: 0.0507


615/Unknown  113s 119ms/step - categorical_accuracy: 0.6719 - loss: 1.3601 - mean_io_u: 0.0508


616/Unknown  113s 119ms/step - categorical_accuracy: 0.6719 - loss: 1.3598 - mean_io_u: 0.0508


617/Unknown  113s 119ms/step - categorical_accuracy: 0.6720 - loss: 1.3594 - mean_io_u: 0.0508


618/Unknown  113s 119ms/step - categorical_accuracy: 0.6721 - loss: 1.3590 - mean_io_u: 0.0509


619/Unknown  113s 119ms/step - categorical_accuracy: 0.6722 - loss: 1.3587 - mean_io_u: 0.0509


620/Unknown  114s 119ms/step - categorical_accuracy: 0.6722 - loss: 1.3583 - mean_io_u: 0.0510


621/Unknown  114s 119ms/step - categorical_accuracy: 0.6723 - loss: 1.3579 - mean_io_u: 0.0510


622/Unknown  114s 119ms/step - categorical_accuracy: 0.6724 - loss: 1.3575 - mean_io_u: 0.0510


623/Unknown  114s 119ms/step - categorical_accuracy: 0.6725 - loss: 1.3572 - mean_io_u: 0.0511


624/Unknown  114s 119ms/step - categorical_accuracy: 0.6725 - loss: 1.3568 - mean_io_u: 0.0511


625/Unknown  114s 119ms/step - categorical_accuracy: 0.6726 - loss: 1.3564 - mean_io_u: 0.0512


626/Unknown  114s 119ms/step - categorical_accuracy: 0.6727 - loss: 1.3561 - mean_io_u: 0.0512


627/Unknown  114s 119ms/step - categorical_accuracy: 0.6728 - loss: 1.3557 - mean_io_u: 0.0513


628/Unknown  114s 119ms/step - categorical_accuracy: 0.6728 - loss: 1.3553 - mean_io_u: 0.0513


629/Unknown  114s 119ms/step - categorical_accuracy: 0.6729 - loss: 1.3550 - mean_io_u: 0.0513


630/Unknown  114s 119ms/step - categorical_accuracy: 0.6730 - loss: 1.3546 - mean_io_u: 0.0514


631/Unknown  114s 118ms/step - categorical_accuracy: 0.6731 - loss: 1.3542 - mean_io_u: 0.0514


632/Unknown  114s 118ms/step - categorical_accuracy: 0.6731 - loss: 1.3538 - mean_io_u: 0.0515


633/Unknown  115s 118ms/step - categorical_accuracy: 0.6732 - loss: 1.3535 - mean_io_u: 0.0515


634/Unknown  115s 118ms/step - categorical_accuracy: 0.6733 - loss: 1.3531 - mean_io_u: 0.0515


635/Unknown  115s 118ms/step - categorical_accuracy: 0.6734 - loss: 1.3527 - mean_io_u: 0.0516


636/Unknown  115s 118ms/step - categorical_accuracy: 0.6734 - loss: 1.3524 - mean_io_u: 0.0516


637/Unknown  115s 118ms/step - categorical_accuracy: 0.6735 - loss: 1.3520 - mean_io_u: 0.0517


638/Unknown  115s 118ms/step - categorical_accuracy: 0.6736 - loss: 1.3516 - mean_io_u: 0.0517


639/Unknown  115s 118ms/step - categorical_accuracy: 0.6737 - loss: 1.3513 - mean_io_u: 0.0517


640/Unknown  115s 118ms/step - categorical_accuracy: 0.6737 - loss: 1.3509 - mean_io_u: 0.0518


641/Unknown  115s 118ms/step - categorical_accuracy: 0.6738 - loss: 1.3506 - mean_io_u: 0.0518


642/Unknown  115s 118ms/step - categorical_accuracy: 0.6739 - loss: 1.3502 - mean_io_u: 0.0519


643/Unknown  115s 118ms/step - categorical_accuracy: 0.6739 - loss: 1.3498 - mean_io_u: 0.0519


644/Unknown  115s 118ms/step - categorical_accuracy: 0.6740 - loss: 1.3495 - mean_io_u: 0.0519


645/Unknown  115s 118ms/step - categorical_accuracy: 0.6741 - loss: 1.3491 - mean_io_u: 0.0520


646/Unknown  116s 117ms/step - categorical_accuracy: 0.6742 - loss: 1.3488 - mean_io_u: 0.0520


647/Unknown  116s 117ms/step - categorical_accuracy: 0.6742 - loss: 1.3484 - mean_io_u: 0.0521


648/Unknown  116s 117ms/step - categorical_accuracy: 0.6743 - loss: 1.3480 - mean_io_u: 0.0521


649/Unknown  116s 117ms/step - categorical_accuracy: 0.6744 - loss: 1.3477 - mean_io_u: 0.0521


650/Unknown  116s 117ms/step - categorical_accuracy: 0.6745 - loss: 1.3473 - mean_io_u: 0.0522


651/Unknown  116s 117ms/step - categorical_accuracy: 0.6745 - loss: 1.3470 - mean_io_u: 0.0522


652/Unknown  116s 117ms/step - categorical_accuracy: 0.6746 - loss: 1.3466 - mean_io_u: 0.0523


653/Unknown  116s 117ms/step - categorical_accuracy: 0.6747 - loss: 1.3462 - mean_io_u: 0.0523


654/Unknown  116s 117ms/step - categorical_accuracy: 0.6747 - loss: 1.3459 - mean_io_u: 0.0523


655/Unknown  116s 117ms/step - categorical_accuracy: 0.6748 - loss: 1.3455 - mean_io_u: 0.0524


656/Unknown  116s 117ms/step - categorical_accuracy: 0.6749 - loss: 1.3452 - mean_io_u: 0.0524


657/Unknown  116s 117ms/step - categorical_accuracy: 0.6750 - loss: 1.3448 - mean_io_u: 0.0525


658/Unknown  117s 117ms/step - categorical_accuracy: 0.6750 - loss: 1.3445 - mean_io_u: 0.0525


659/Unknown  117s 117ms/step - categorical_accuracy: 0.6751 - loss: 1.3441 - mean_io_u: 0.0525


660/Unknown  117s 117ms/step - categorical_accuracy: 0.6752 - loss: 1.3438 - mean_io_u: 0.0526


661/Unknown  117s 117ms/step - categorical_accuracy: 0.6752 - loss: 1.3434 - mean_io_u: 0.0526


662/Unknown  117s 117ms/step - categorical_accuracy: 0.6753 - loss: 1.3431 - mean_io_u: 0.0526


663/Unknown  117s 116ms/step - categorical_accuracy: 0.6754 - loss: 1.3427 - mean_io_u: 0.0527


664/Unknown  117s 116ms/step - categorical_accuracy: 0.6755 - loss: 1.3424 - mean_io_u: 0.0527


665/Unknown  117s 116ms/step - categorical_accuracy: 0.6755 - loss: 1.3420 - mean_io_u: 0.0528


666/Unknown  117s 116ms/step - categorical_accuracy: 0.6756 - loss: 1.3417 - mean_io_u: 0.0528


667/Unknown  117s 116ms/step - categorical_accuracy: 0.6757 - loss: 1.3413 - mean_io_u: 0.0528


668/Unknown  117s 116ms/step - categorical_accuracy: 0.6757 - loss: 1.3410 - mean_io_u: 0.0529


669/Unknown  117s 116ms/step - categorical_accuracy: 0.6758 - loss: 1.3407 - mean_io_u: 0.0529


670/Unknown  117s 116ms/step - categorical_accuracy: 0.6759 - loss: 1.3403 - mean_io_u: 0.0530


671/Unknown  117s 116ms/step - categorical_accuracy: 0.6759 - loss: 1.3400 - mean_io_u: 0.0530


672/Unknown  118s 116ms/step - categorical_accuracy: 0.6760 - loss: 1.3396 - mean_io_u: 0.0530


673/Unknown  118s 116ms/step - categorical_accuracy: 0.6761 - loss: 1.3393 - mean_io_u: 0.0531


674/Unknown  118s 116ms/step - categorical_accuracy: 0.6762 - loss: 1.3389 - mean_io_u: 0.0531


675/Unknown  118s 116ms/step - categorical_accuracy: 0.6762 - loss: 1.3386 - mean_io_u: 0.0532


676/Unknown  118s 116ms/step - categorical_accuracy: 0.6763 - loss: 1.3383 - mean_io_u: 0.0532


677/Unknown  118s 116ms/step - categorical_accuracy: 0.6764 - loss: 1.3379 - mean_io_u: 0.0532


678/Unknown  118s 116ms/step - categorical_accuracy: 0.6764 - loss: 1.3376 - mean_io_u: 0.0533


679/Unknown  118s 115ms/step - categorical_accuracy: 0.6765 - loss: 1.3372 - mean_io_u: 0.0533


680/Unknown  118s 115ms/step - categorical_accuracy: 0.6766 - loss: 1.3369 - mean_io_u: 0.0533


681/Unknown  118s 115ms/step - categorical_accuracy: 0.6766 - loss: 1.3366 - mean_io_u: 0.0534


682/Unknown  118s 115ms/step - categorical_accuracy: 0.6767 - loss: 1.3362 - mean_io_u: 0.0534


683/Unknown  118s 115ms/step - categorical_accuracy: 0.6768 - loss: 1.3359 - mean_io_u: 0.0535


684/Unknown  118s 115ms/step - categorical_accuracy: 0.6768 - loss: 1.3355 - mean_io_u: 0.0535


685/Unknown  118s 115ms/step - categorical_accuracy: 0.6769 - loss: 1.3352 - mean_io_u: 0.0535


686/Unknown  119s 115ms/step - categorical_accuracy: 0.6770 - loss: 1.3349 - mean_io_u: 0.0536


687/Unknown  119s 115ms/step - categorical_accuracy: 0.6770 - loss: 1.3345 - mean_io_u: 0.0536


688/Unknown  119s 115ms/step - categorical_accuracy: 0.6771 - loss: 1.3342 - mean_io_u: 0.0537


689/Unknown  119s 115ms/step - categorical_accuracy: 0.6772 - loss: 1.3339 - mean_io_u: 0.0537


690/Unknown  119s 115ms/step - categorical_accuracy: 0.6772 - loss: 1.3335 - mean_io_u: 0.0537


691/Unknown  119s 115ms/step - categorical_accuracy: 0.6773 - loss: 1.3332 - mean_io_u: 0.0538


692/Unknown  119s 114ms/step - categorical_accuracy: 0.6774 - loss: 1.3329 - mean_io_u: 0.0538


693/Unknown  119s 114ms/step - categorical_accuracy: 0.6774 - loss: 1.3325 - mean_io_u: 0.0539


694/Unknown  119s 114ms/step - categorical_accuracy: 0.6775 - loss: 1.3322 - mean_io_u: 0.0539


695/Unknown  119s 114ms/step - categorical_accuracy: 0.6776 - loss: 1.3319 - mean_io_u: 0.0539


696/Unknown  119s 114ms/step - categorical_accuracy: 0.6776 - loss: 1.3315 - mean_io_u: 0.0540


697/Unknown  119s 114ms/step - categorical_accuracy: 0.6777 - loss: 1.3312 - mean_io_u: 0.0540


698/Unknown  119s 114ms/step - categorical_accuracy: 0.6778 - loss: 1.3309 - mean_io_u: 0.0540


699/Unknown  119s 114ms/step - categorical_accuracy: 0.6778 - loss: 1.3305 - mean_io_u: 0.0541


700/Unknown  120s 114ms/step - categorical_accuracy: 0.6779 - loss: 1.3302 - mean_io_u: 0.0541


701/Unknown  120s 114ms/step - categorical_accuracy: 0.6780 - loss: 1.3299 - mean_io_u: 0.0542


702/Unknown  120s 114ms/step - categorical_accuracy: 0.6780 - loss: 1.3295 - mean_io_u: 0.0542


703/Unknown  120s 114ms/step - categorical_accuracy: 0.6781 - loss: 1.3292 - mean_io_u: 0.0542


704/Unknown  120s 114ms/step - categorical_accuracy: 0.6782 - loss: 1.3289 - mean_io_u: 0.0543


705/Unknown  120s 114ms/step - categorical_accuracy: 0.6782 - loss: 1.3286 - mean_io_u: 0.0543


706/Unknown  120s 114ms/step - categorical_accuracy: 0.6783 - loss: 1.3282 - mean_io_u: 0.0543


707/Unknown  120s 114ms/step - categorical_accuracy: 0.6784 - loss: 1.3279 - mean_io_u: 0.0544


708/Unknown  120s 114ms/step - categorical_accuracy: 0.6784 - loss: 1.3276 - mean_io_u: 0.0544


709/Unknown  120s 114ms/step - categorical_accuracy: 0.6785 - loss: 1.3273 - mean_io_u: 0.0545


710/Unknown  120s 114ms/step - categorical_accuracy: 0.6786 - loss: 1.3269 - mean_io_u: 0.0545


711/Unknown  120s 114ms/step - categorical_accuracy: 0.6786 - loss: 1.3266 - mean_io_u: 0.0545


712/Unknown  120s 113ms/step - categorical_accuracy: 0.6787 - loss: 1.3263 - mean_io_u: 0.0546


713/Unknown  121s 113ms/step - categorical_accuracy: 0.6788 - loss: 1.3260 - mean_io_u: 0.0546


714/Unknown  121s 113ms/step - categorical_accuracy: 0.6788 - loss: 1.3257 - mean_io_u: 0.0546


715/Unknown  121s 113ms/step - categorical_accuracy: 0.6789 - loss: 1.3253 - mean_io_u: 0.0547


716/Unknown  121s 113ms/step - categorical_accuracy: 0.6790 - loss: 1.3250 - mean_io_u: 0.0547


717/Unknown  121s 113ms/step - categorical_accuracy: 0.6790 - loss: 1.3247 - mean_io_u: 0.0547


718/Unknown  121s 113ms/step - categorical_accuracy: 0.6791 - loss: 1.3244 - mean_io_u: 0.0548


719/Unknown  121s 113ms/step - categorical_accuracy: 0.6791 - loss: 1.3240 - mean_io_u: 0.0548


720/Unknown  121s 113ms/step - categorical_accuracy: 0.6792 - loss: 1.3237 - mean_io_u: 0.0549


721/Unknown  121s 113ms/step - categorical_accuracy: 0.6793 - loss: 1.3234 - mean_io_u: 0.0549


722/Unknown  121s 113ms/step - categorical_accuracy: 0.6793 - loss: 1.3231 - mean_io_u: 0.0549


723/Unknown  121s 113ms/step - categorical_accuracy: 0.6794 - loss: 1.3228 - mean_io_u: 0.0550


724/Unknown  121s 113ms/step - categorical_accuracy: 0.6795 - loss: 1.3224 - mean_io_u: 0.0550


725/Unknown  122s 113ms/step - categorical_accuracy: 0.6795 - loss: 1.3221 - mean_io_u: 0.0550


726/Unknown  122s 113ms/step - categorical_accuracy: 0.6796 - loss: 1.3218 - mean_io_u: 0.0551


727/Unknown  122s 113ms/step - categorical_accuracy: 0.6797 - loss: 1.3215 - mean_io_u: 0.0551


728/Unknown  122s 113ms/step - categorical_accuracy: 0.6797 - loss: 1.3212 - mean_io_u: 0.0551


729/Unknown  122s 113ms/step - categorical_accuracy: 0.6798 - loss: 1.3209 - mean_io_u: 0.0552


730/Unknown  122s 113ms/step - categorical_accuracy: 0.6798 - loss: 1.3205 - mean_io_u: 0.0552


731/Unknown  122s 113ms/step - categorical_accuracy: 0.6799 - loss: 1.3202 - mean_io_u: 0.0553


732/Unknown  122s 113ms/step - categorical_accuracy: 0.6800 - loss: 1.3199 - mean_io_u: 0.0553


733/Unknown  122s 113ms/step - categorical_accuracy: 0.6800 - loss: 1.3196 - mean_io_u: 0.0553


734/Unknown  122s 113ms/step - categorical_accuracy: 0.6801 - loss: 1.3193 - mean_io_u: 0.0554


735/Unknown  122s 113ms/step - categorical_accuracy: 0.6802 - loss: 1.3190 - mean_io_u: 0.0554


736/Unknown  122s 112ms/step - categorical_accuracy: 0.6802 - loss: 1.3187 - mean_io_u: 0.0554


737/Unknown  123s 112ms/step - categorical_accuracy: 0.6803 - loss: 1.3184 - mean_io_u: 0.0555


738/Unknown  123s 112ms/step - categorical_accuracy: 0.6803 - loss: 1.3181 - mean_io_u: 0.0555


739/Unknown  123s 112ms/step - categorical_accuracy: 0.6804 - loss: 1.3177 - mean_io_u: 0.0555


740/Unknown  123s 112ms/step - categorical_accuracy: 0.6805 - loss: 1.3174 - mean_io_u: 0.0556


741/Unknown  123s 112ms/step - categorical_accuracy: 0.6805 - loss: 1.3171 - mean_io_u: 0.0556


742/Unknown  123s 112ms/step - categorical_accuracy: 0.6806 - loss: 1.3168 - mean_io_u: 0.0556


743/Unknown  123s 112ms/step - categorical_accuracy: 0.6806 - loss: 1.3165 - mean_io_u: 0.0557


744/Unknown  123s 112ms/step - categorical_accuracy: 0.6807 - loss: 1.3162 - mean_io_u: 0.0557


745/Unknown  123s 112ms/step - categorical_accuracy: 0.6808 - loss: 1.3159 - mean_io_u: 0.0557


746/Unknown  123s 112ms/step - categorical_accuracy: 0.6808 - loss: 1.3156 - mean_io_u: 0.0558


747/Unknown  123s 112ms/step - categorical_accuracy: 0.6809 - loss: 1.3153 - mean_io_u: 0.0558


748/Unknown  124s 112ms/step - categorical_accuracy: 0.6809 - loss: 1.3150 - mean_io_u: 0.0559


749/Unknown  124s 112ms/step - categorical_accuracy: 0.6810 - loss: 1.3147 - mean_io_u: 0.0559


750/Unknown  124s 112ms/step - categorical_accuracy: 0.6811 - loss: 1.3144 - mean_io_u: 0.0559


751/Unknown  124s 112ms/step - categorical_accuracy: 0.6811 - loss: 1.3141 - mean_io_u: 0.0560


752/Unknown  124s 112ms/step - categorical_accuracy: 0.6812 - loss: 1.3138 - mean_io_u: 0.0560


753/Unknown  124s 112ms/step - categorical_accuracy: 0.6812 - loss: 1.3135 - mean_io_u: 0.0560


754/Unknown  124s 112ms/step - categorical_accuracy: 0.6813 - loss: 1.3132 - mean_io_u: 0.0561


755/Unknown  124s 112ms/step - categorical_accuracy: 0.6814 - loss: 1.3129 - mean_io_u: 0.0561


756/Unknown  124s 112ms/step - categorical_accuracy: 0.6814 - loss: 1.3125 - mean_io_u: 0.0561


757/Unknown  124s 112ms/step - categorical_accuracy: 0.6815 - loss: 1.3122 - mean_io_u: 0.0562


758/Unknown  124s 112ms/step - categorical_accuracy: 0.6815 - loss: 1.3119 - mean_io_u: 0.0562


759/Unknown  124s 112ms/step - categorical_accuracy: 0.6816 - loss: 1.3116 - mean_io_u: 0.0562


760/Unknown  124s 112ms/step - categorical_accuracy: 0.6817 - loss: 1.3113 - mean_io_u: 0.0563


761/Unknown  125s 112ms/step - categorical_accuracy: 0.6817 - loss: 1.3110 - mean_io_u: 0.0563


762/Unknown  125s 111ms/step - categorical_accuracy: 0.6818 - loss: 1.3107 - mean_io_u: 0.0563


763/Unknown  125s 111ms/step - categorical_accuracy: 0.6818 - loss: 1.3104 - mean_io_u: 0.0564


764/Unknown  125s 111ms/step - categorical_accuracy: 0.6819 - loss: 1.3101 - mean_io_u: 0.0564


765/Unknown  125s 111ms/step - categorical_accuracy: 0.6820 - loss: 1.3098 - mean_io_u: 0.0565


766/Unknown  125s 111ms/step - categorical_accuracy: 0.6820 - loss: 1.3096 - mean_io_u: 0.0565


767/Unknown  125s 111ms/step - categorical_accuracy: 0.6821 - loss: 1.3093 - mean_io_u: 0.0565


768/Unknown  125s 111ms/step - categorical_accuracy: 0.6821 - loss: 1.3090 - mean_io_u: 0.0566


769/Unknown  125s 111ms/step - categorical_accuracy: 0.6822 - loss: 1.3087 - mean_io_u: 0.0566


770/Unknown  125s 111ms/step - categorical_accuracy: 0.6822 - loss: 1.3084 - mean_io_u: 0.0566


771/Unknown  125s 111ms/step - categorical_accuracy: 0.6823 - loss: 1.3081 - mean_io_u: 0.0567


772/Unknown  125s 111ms/step - categorical_accuracy: 0.6824 - loss: 1.3078 - mean_io_u: 0.0567


773/Unknown  126s 111ms/step - categorical_accuracy: 0.6824 - loss: 1.3075 - mean_io_u: 0.0567


774/Unknown  126s 111ms/step - categorical_accuracy: 0.6825 - loss: 1.3072 - mean_io_u: 0.0568


775/Unknown  126s 111ms/step - categorical_accuracy: 0.6825 - loss: 1.3069 - mean_io_u: 0.0568


776/Unknown  126s 111ms/step - categorical_accuracy: 0.6826 - loss: 1.3066 - mean_io_u: 0.0568


777/Unknown  126s 111ms/step - categorical_accuracy: 0.6826 - loss: 1.3063 - mean_io_u: 0.0569


778/Unknown  126s 111ms/step - categorical_accuracy: 0.6827 - loss: 1.3061 - mean_io_u: 0.0569


779/Unknown  126s 111ms/step - categorical_accuracy: 0.6828 - loss: 1.3058 - mean_io_u: 0.0569


780/Unknown  126s 111ms/step - categorical_accuracy: 0.6828 - loss: 1.3055 - mean_io_u: 0.0570


781/Unknown  126s 111ms/step - categorical_accuracy: 0.6829 - loss: 1.3052 - mean_io_u: 0.0570


782/Unknown  126s 111ms/step - categorical_accuracy: 0.6829 - loss: 1.3049 - mean_io_u: 0.0570


783/Unknown  126s 111ms/step - categorical_accuracy: 0.6830 - loss: 1.3046 - mean_io_u: 0.0571


784/Unknown  127s 111ms/step - categorical_accuracy: 0.6830 - loss: 1.3043 - mean_io_u: 0.0571


785/Unknown  127s 111ms/step - categorical_accuracy: 0.6831 - loss: 1.3040 - mean_io_u: 0.0572


786/Unknown  127s 111ms/step - categorical_accuracy: 0.6831 - loss: 1.3038 - mean_io_u: 0.0572


787/Unknown  127s 111ms/step - categorical_accuracy: 0.6832 - loss: 1.3035 - mean_io_u: 0.0572


788/Unknown  127s 111ms/step - categorical_accuracy: 0.6833 - loss: 1.3032 - mean_io_u: 0.0573


789/Unknown  127s 111ms/step - categorical_accuracy: 0.6833 - loss: 1.3029 - mean_io_u: 0.0573


790/Unknown  127s 110ms/step - categorical_accuracy: 0.6834 - loss: 1.3026 - mean_io_u: 0.0573


791/Unknown  127s 110ms/step - categorical_accuracy: 0.6834 - loss: 1.3023 - mean_io_u: 0.0574


792/Unknown  127s 110ms/step - categorical_accuracy: 0.6835 - loss: 1.3021 - mean_io_u: 0.0574


793/Unknown  127s 110ms/step - categorical_accuracy: 0.6835 - loss: 1.3018 - mean_io_u: 0.0574


794/Unknown  127s 110ms/step - categorical_accuracy: 0.6836 - loss: 1.3015 - mean_io_u: 0.0575


795/Unknown  127s 110ms/step - categorical_accuracy: 0.6836 - loss: 1.3012 - mean_io_u: 0.0575


796/Unknown  127s 110ms/step - categorical_accuracy: 0.6837 - loss: 1.3009 - mean_io_u: 0.0575


797/Unknown  127s 110ms/step - categorical_accuracy: 0.6838 - loss: 1.3007 - mean_io_u: 0.0576


798/Unknown  128s 110ms/step - categorical_accuracy: 0.6838 - loss: 1.3004 - mean_io_u: 0.0576


799/Unknown  128s 110ms/step - categorical_accuracy: 0.6839 - loss: 1.3001 - mean_io_u: 0.0576


800/Unknown  128s 110ms/step - categorical_accuracy: 0.6839 - loss: 1.2998 - mean_io_u: 0.0577


801/Unknown  128s 110ms/step - categorical_accuracy: 0.6840 - loss: 1.2996 - mean_io_u: 0.0577


802/Unknown  128s 110ms/step - categorical_accuracy: 0.6840 - loss: 1.2993 - mean_io_u: 0.0577


803/Unknown  128s 110ms/step - categorical_accuracy: 0.6841 - loss: 1.2990 - mean_io_u: 0.0578


804/Unknown  128s 110ms/step - categorical_accuracy: 0.6841 - loss: 1.2987 - mean_io_u: 0.0578


805/Unknown  128s 110ms/step - categorical_accuracy: 0.6842 - loss: 1.2984 - mean_io_u: 0.0578


806/Unknown  128s 110ms/step - categorical_accuracy: 0.6842 - loss: 1.2982 - mean_io_u: 0.0579


807/Unknown  128s 110ms/step - categorical_accuracy: 0.6843 - loss: 1.2979 - mean_io_u: 0.0579


808/Unknown  128s 110ms/step - categorical_accuracy: 0.6843 - loss: 1.2976 - mean_io_u: 0.0579


809/Unknown  128s 110ms/step - categorical_accuracy: 0.6844 - loss: 1.2974 - mean_io_u: 0.0580


810/Unknown  129s 110ms/step - categorical_accuracy: 0.6844 - loss: 1.2971 - mean_io_u: 0.0580


811/Unknown  129s 110ms/step - categorical_accuracy: 0.6845 - loss: 1.2968 - mean_io_u: 0.0580


812/Unknown  129s 110ms/step - categorical_accuracy: 0.6846 - loss: 1.2965 - mean_io_u: 0.0581


813/Unknown  129s 110ms/step - categorical_accuracy: 0.6846 - loss: 1.2963 - mean_io_u: 0.0581


814/Unknown  129s 110ms/step - categorical_accuracy: 0.6847 - loss: 1.2960 - mean_io_u: 0.0581


815/Unknown  129s 109ms/step - categorical_accuracy: 0.6847 - loss: 1.2957 - mean_io_u: 0.0582


816/Unknown  129s 109ms/step - categorical_accuracy: 0.6848 - loss: 1.2955 - mean_io_u: 0.0582


817/Unknown  129s 109ms/step - categorical_accuracy: 0.6848 - loss: 1.2952 - mean_io_u: 0.0582


818/Unknown  129s 109ms/step - categorical_accuracy: 0.6849 - loss: 1.2949 - mean_io_u: 0.0583


819/Unknown  129s 109ms/step - categorical_accuracy: 0.6849 - loss: 1.2947 - mean_io_u: 0.0583


820/Unknown  129s 109ms/step - categorical_accuracy: 0.6850 - loss: 1.2944 - mean_io_u: 0.0583


821/Unknown  129s 109ms/step - categorical_accuracy: 0.6850 - loss: 1.2941 - mean_io_u: 0.0584


822/Unknown  129s 109ms/step - categorical_accuracy: 0.6851 - loss: 1.2939 - mean_io_u: 0.0584


823/Unknown  130s 109ms/step - categorical_accuracy: 0.6851 - loss: 1.2936 - mean_io_u: 0.0584


824/Unknown  130s 109ms/step - categorical_accuracy: 0.6852 - loss: 1.2933 - mean_io_u: 0.0585


825/Unknown  130s 109ms/step - categorical_accuracy: 0.6852 - loss: 1.2931 - mean_io_u: 0.0585


826/Unknown  130s 109ms/step - categorical_accuracy: 0.6853 - loss: 1.2928 - mean_io_u: 0.0585


827/Unknown  130s 109ms/step - categorical_accuracy: 0.6853 - loss: 1.2925 - mean_io_u: 0.0586


828/Unknown  130s 109ms/step - categorical_accuracy: 0.6854 - loss: 1.2923 - mean_io_u: 0.0586


829/Unknown  130s 109ms/step - categorical_accuracy: 0.6854 - loss: 1.2920 - mean_io_u: 0.0586


830/Unknown  130s 109ms/step - categorical_accuracy: 0.6855 - loss: 1.2917 - mean_io_u: 0.0587


831/Unknown  130s 109ms/step - categorical_accuracy: 0.6855 - loss: 1.2915 - mean_io_u: 0.0587


832/Unknown  130s 109ms/step - categorical_accuracy: 0.6856 - loss: 1.2912 - mean_io_u: 0.0587


833/Unknown  130s 109ms/step - categorical_accuracy: 0.6856 - loss: 1.2909 - mean_io_u: 0.0588


834/Unknown  130s 109ms/step - categorical_accuracy: 0.6857 - loss: 1.2907 - mean_io_u: 0.0588


835/Unknown  130s 109ms/step - categorical_accuracy: 0.6857 - loss: 1.2904 - mean_io_u: 0.0588


836/Unknown  130s 108ms/step - categorical_accuracy: 0.6858 - loss: 1.2901 - mean_io_u: 0.0589


837/Unknown  130s 108ms/step - categorical_accuracy: 0.6858 - loss: 1.2899 - mean_io_u: 0.0589


838/Unknown  130s 108ms/step - categorical_accuracy: 0.6859 - loss: 1.2896 - mean_io_u: 0.0589


839/Unknown  131s 108ms/step - categorical_accuracy: 0.6859 - loss: 1.2894 - mean_io_u: 0.0590


840/Unknown  131s 108ms/step - categorical_accuracy: 0.6860 - loss: 1.2891 - mean_io_u: 0.0590


841/Unknown  131s 108ms/step - categorical_accuracy: 0.6860 - loss: 1.2888 - mean_io_u: 0.0590


842/Unknown  131s 108ms/step - categorical_accuracy: 0.6861 - loss: 1.2886 - mean_io_u: 0.0591


843/Unknown  131s 108ms/step - categorical_accuracy: 0.6862 - loss: 1.2883 - mean_io_u: 0.0591


844/Unknown  131s 108ms/step - categorical_accuracy: 0.6862 - loss: 1.2880 - mean_io_u: 0.0591


845/Unknown  131s 108ms/step - categorical_accuracy: 0.6863 - loss: 1.2878 - mean_io_u: 0.0592


846/Unknown  131s 108ms/step - categorical_accuracy: 0.6863 - loss: 1.2875 - mean_io_u: 0.0592


847/Unknown  131s 108ms/step - categorical_accuracy: 0.6864 - loss: 1.2873 - mean_io_u: 0.0592


848/Unknown  131s 108ms/step - categorical_accuracy: 0.6864 - loss: 1.2870 - mean_io_u: 0.0593


849/Unknown  131s 108ms/step - categorical_accuracy: 0.6865 - loss: 1.2867 - mean_io_u: 0.0593


850/Unknown  131s 108ms/step - categorical_accuracy: 0.6865 - loss: 1.2865 - mean_io_u: 0.0593


851/Unknown  131s 108ms/step - categorical_accuracy: 0.6866 - loss: 1.2862 - mean_io_u: 0.0594


852/Unknown  132s 108ms/step - categorical_accuracy: 0.6866 - loss: 1.2860 - mean_io_u: 0.0594


853/Unknown  132s 108ms/step - categorical_accuracy: 0.6867 - loss: 1.2857 - mean_io_u: 0.0594


854/Unknown  132s 108ms/step - categorical_accuracy: 0.6867 - loss: 1.2854 - mean_io_u: 0.0595


855/Unknown  132s 108ms/step - categorical_accuracy: 0.6868 - loss: 1.2852 - mean_io_u: 0.0595


856/Unknown  132s 108ms/step - categorical_accuracy: 0.6868 - loss: 1.2849 - mean_io_u: 0.0595


857/Unknown  132s 108ms/step - categorical_accuracy: 0.6869 - loss: 1.2847 - mean_io_u: 0.0596


858/Unknown  132s 108ms/step - categorical_accuracy: 0.6869 - loss: 1.2844 - mean_io_u: 0.0596


859/Unknown  132s 108ms/step - categorical_accuracy: 0.6870 - loss: 1.2842 - mean_io_u: 0.0596


860/Unknown  132s 108ms/step - categorical_accuracy: 0.6870 - loss: 1.2839 - mean_io_u: 0.0597


861/Unknown  132s 108ms/step - categorical_accuracy: 0.6871 - loss: 1.2836 - mean_io_u: 0.0597


862/Unknown  132s 107ms/step - categorical_accuracy: 0.6871 - loss: 1.2834 - mean_io_u: 0.0597


863/Unknown  132s 107ms/step - categorical_accuracy: 0.6872 - loss: 1.2831 - mean_io_u: 0.0598


864/Unknown  132s 107ms/step - categorical_accuracy: 0.6872 - loss: 1.2829 - mean_io_u: 0.0598


865/Unknown  133s 107ms/step - categorical_accuracy: 0.6873 - loss: 1.2826 - mean_io_u: 0.0598


866/Unknown  133s 107ms/step - categorical_accuracy: 0.6873 - loss: 1.2824 - mean_io_u: 0.0599


867/Unknown  133s 107ms/step - categorical_accuracy: 0.6874 - loss: 1.2821 - mean_io_u: 0.0599


868/Unknown  133s 107ms/step - categorical_accuracy: 0.6874 - loss: 1.2818 - mean_io_u: 0.0599


869/Unknown  133s 107ms/step - categorical_accuracy: 0.6875 - loss: 1.2816 - mean_io_u: 0.0600


870/Unknown  133s 107ms/step - categorical_accuracy: 0.6875 - loss: 1.2813 - mean_io_u: 0.0600


871/Unknown  133s 107ms/step - categorical_accuracy: 0.6876 - loss: 1.2811 - mean_io_u: 0.0600


872/Unknown  133s 107ms/step - categorical_accuracy: 0.6876 - loss: 1.2808 - mean_io_u: 0.0601


873/Unknown  133s 107ms/step - categorical_accuracy: 0.6877 - loss: 1.2806 - mean_io_u: 0.0601


874/Unknown  133s 107ms/step - categorical_accuracy: 0.6877 - loss: 1.2803 - mean_io_u: 0.0601


875/Unknown  133s 107ms/step - categorical_accuracy: 0.6878 - loss: 1.2801 - mean_io_u: 0.0602


876/Unknown  133s 107ms/step - categorical_accuracy: 0.6878 - loss: 1.2798 - mean_io_u: 0.0602


877/Unknown  134s 107ms/step - categorical_accuracy: 0.6879 - loss: 1.2796 - mean_io_u: 0.0602


878/Unknown  134s 107ms/step - categorical_accuracy: 0.6879 - loss: 1.2793 - mean_io_u: 0.0603


879/Unknown  134s 107ms/step - categorical_accuracy: 0.6880 - loss: 1.2791 - mean_io_u: 0.0603


880/Unknown  134s 107ms/step - categorical_accuracy: 0.6880 - loss: 1.2788 - mean_io_u: 0.0603


881/Unknown  134s 107ms/step - categorical_accuracy: 0.6881 - loss: 1.2785 - mean_io_u: 0.0604


882/Unknown  134s 107ms/step - categorical_accuracy: 0.6881 - loss: 1.2783 - mean_io_u: 0.0604


883/Unknown  134s 107ms/step - categorical_accuracy: 0.6881 - loss: 1.2780 - mean_io_u: 0.0604


884/Unknown  134s 107ms/step - categorical_accuracy: 0.6882 - loss: 1.2778 - mean_io_u: 0.0604


885/Unknown  134s 107ms/step - categorical_accuracy: 0.6882 - loss: 1.2775 - mean_io_u: 0.0605


886/Unknown  134s 107ms/step - categorical_accuracy: 0.6883 - loss: 1.2773 - mean_io_u: 0.0605


887/Unknown  134s 107ms/step - categorical_accuracy: 0.6883 - loss: 1.2770 - mean_io_u: 0.0605


888/Unknown  135s 107ms/step - categorical_accuracy: 0.6884 - loss: 1.2768 - mean_io_u: 0.0606


889/Unknown  135s 107ms/step - categorical_accuracy: 0.6884 - loss: 1.2765 - mean_io_u: 0.0606


890/Unknown  135s 107ms/step - categorical_accuracy: 0.6885 - loss: 1.2763 - mean_io_u: 0.0606


891/Unknown  135s 107ms/step - categorical_accuracy: 0.6885 - loss: 1.2761 - mean_io_u: 0.0607


892/Unknown  135s 107ms/step - categorical_accuracy: 0.6886 - loss: 1.2758 - mean_io_u: 0.0607


893/Unknown  135s 107ms/step - categorical_accuracy: 0.6886 - loss: 1.2756 - mean_io_u: 0.0607


894/Unknown  135s 107ms/step - categorical_accuracy: 0.6887 - loss: 1.2753 - mean_io_u: 0.0608


895/Unknown  135s 107ms/step - categorical_accuracy: 0.6887 - loss: 1.2751 - mean_io_u: 0.0608


896/Unknown  135s 107ms/step - categorical_accuracy: 0.6888 - loss: 1.2748 - mean_io_u: 0.0608


897/Unknown  135s 107ms/step - categorical_accuracy: 0.6888 - loss: 1.2746 - mean_io_u: 0.0609


898/Unknown  136s 107ms/step - categorical_accuracy: 0.6889 - loss: 1.2743 - mean_io_u: 0.0609


899/Unknown  136s 107ms/step - categorical_accuracy: 0.6889 - loss: 1.2741 - mean_io_u: 0.0609


900/Unknown  136s 107ms/step - categorical_accuracy: 0.6890 - loss: 1.2739 - mean_io_u: 0.0610


901/Unknown  136s 107ms/step - categorical_accuracy: 0.6890 - loss: 1.2736 - mean_io_u: 0.0610


902/Unknown  136s 107ms/step - categorical_accuracy: 0.6891 - loss: 1.2734 - mean_io_u: 0.0610


903/Unknown  136s 107ms/step - categorical_accuracy: 0.6891 - loss: 1.2731 - mean_io_u: 0.0611


904/Unknown  136s 107ms/step - categorical_accuracy: 0.6892 - loss: 1.2729 - mean_io_u: 0.0611


905/Unknown  136s 106ms/step - categorical_accuracy: 0.6892 - loss: 1.2726 - mean_io_u: 0.0611


906/Unknown  136s 106ms/step - categorical_accuracy: 0.6892 - loss: 1.2724 - mean_io_u: 0.0612


907/Unknown  136s 106ms/step - categorical_accuracy: 0.6893 - loss: 1.2722 - mean_io_u: 0.0612


908/Unknown  136s 106ms/step - categorical_accuracy: 0.6893 - loss: 1.2719 - mean_io_u: 0.0612


909/Unknown  136s 106ms/step - categorical_accuracy: 0.6894 - loss: 1.2717 - mean_io_u: 0.0612


910/Unknown  136s 106ms/step - categorical_accuracy: 0.6894 - loss: 1.2714 - mean_io_u: 0.0613


911/Unknown  137s 106ms/step - categorical_accuracy: 0.6895 - loss: 1.2712 - mean_io_u: 0.0613


912/Unknown  137s 106ms/step - categorical_accuracy: 0.6895 - loss: 1.2710 - mean_io_u: 0.0613


913/Unknown  137s 106ms/step - categorical_accuracy: 0.6896 - loss: 1.2707 - mean_io_u: 0.0614


914/Unknown  137s 106ms/step - categorical_accuracy: 0.6896 - loss: 1.2705 - mean_io_u: 0.0614


915/Unknown  137s 106ms/step - categorical_accuracy: 0.6897 - loss: 1.2702 - mean_io_u: 0.0614


916/Unknown  137s 106ms/step - categorical_accuracy: 0.6897 - loss: 1.2700 - mean_io_u: 0.0615


917/Unknown  137s 106ms/step - categorical_accuracy: 0.6898 - loss: 1.2698 - mean_io_u: 0.0615


918/Unknown  137s 106ms/step - categorical_accuracy: 0.6898 - loss: 1.2695 - mean_io_u: 0.0615


919/Unknown  137s 106ms/step - categorical_accuracy: 0.6899 - loss: 1.2693 - mean_io_u: 0.0616


920/Unknown  137s 106ms/step - categorical_accuracy: 0.6899 - loss: 1.2691 - mean_io_u: 0.0616


921/Unknown  137s 106ms/step - categorical_accuracy: 0.6899 - loss: 1.2688 - mean_io_u: 0.0616


922/Unknown  137s 106ms/step - categorical_accuracy: 0.6900 - loss: 1.2686 - mean_io_u: 0.0617


923/Unknown  138s 106ms/step - categorical_accuracy: 0.6900 - loss: 1.2684 - mean_io_u: 0.0617


924/Unknown  138s 106ms/step - categorical_accuracy: 0.6901 - loss: 1.2681 - mean_io_u: 0.0617


925/Unknown  138s 106ms/step - categorical_accuracy: 0.6901 - loss: 1.2679 - mean_io_u: 0.0618


926/Unknown  138s 106ms/step - categorical_accuracy: 0.6902 - loss: 1.2677 - mean_io_u: 0.0618


927/Unknown  138s 106ms/step - categorical_accuracy: 0.6902 - loss: 1.2674 - mean_io_u: 0.0618


928/Unknown  138s 106ms/step - categorical_accuracy: 0.6903 - loss: 1.2672 - mean_io_u: 0.0619


929/Unknown  138s 106ms/step - categorical_accuracy: 0.6903 - loss: 1.2670 - mean_io_u: 0.0619


930/Unknown  138s 106ms/step - categorical_accuracy: 0.6904 - loss: 1.2667 - mean_io_u: 0.0619


931/Unknown  138s 106ms/step - categorical_accuracy: 0.6904 - loss: 1.2665 - mean_io_u: 0.0619


932/Unknown  138s 106ms/step - categorical_accuracy: 0.6904 - loss: 1.2663 - mean_io_u: 0.0620


933/Unknown  138s 106ms/step - categorical_accuracy: 0.6905 - loss: 1.2660 - mean_io_u: 0.0620


934/Unknown  138s 106ms/step - categorical_accuracy: 0.6905 - loss: 1.2658 - mean_io_u: 0.0620


935/Unknown  138s 106ms/step - categorical_accuracy: 0.6906 - loss: 1.2656 - mean_io_u: 0.0621


936/Unknown  139s 106ms/step - categorical_accuracy: 0.6906 - loss: 1.2653 - mean_io_u: 0.0621


937/Unknown  139s 106ms/step - categorical_accuracy: 0.6907 - loss: 1.2651 - mean_io_u: 0.0621


938/Unknown  139s 106ms/step - categorical_accuracy: 0.6907 - loss: 1.2649 - mean_io_u: 0.0622


939/Unknown  139s 105ms/step - categorical_accuracy: 0.6908 - loss: 1.2646 - mean_io_u: 0.0622


940/Unknown  139s 105ms/step - categorical_accuracy: 0.6908 - loss: 1.2644 - mean_io_u: 0.0622


941/Unknown  139s 105ms/step - categorical_accuracy: 0.6908 - loss: 1.2642 - mean_io_u: 0.0623


942/Unknown  139s 105ms/step - categorical_accuracy: 0.6909 - loss: 1.2639 - mean_io_u: 0.0623


943/Unknown  139s 105ms/step - categorical_accuracy: 0.6909 - loss: 1.2637 - mean_io_u: 0.0623


944/Unknown  139s 105ms/step - categorical_accuracy: 0.6910 - loss: 1.2635 - mean_io_u: 0.0624


945/Unknown  139s 105ms/step - categorical_accuracy: 0.6910 - loss: 1.2633 - mean_io_u: 0.0624


946/Unknown  139s 105ms/step - categorical_accuracy: 0.6911 - loss: 1.2630 - mean_io_u: 0.0624


947/Unknown  139s 105ms/step - categorical_accuracy: 0.6911 - loss: 1.2628 - mean_io_u: 0.0625


948/Unknown  139s 105ms/step - categorical_accuracy: 0.6912 - loss: 1.2626 - mean_io_u: 0.0625


949/Unknown  140s 105ms/step - categorical_accuracy: 0.6912 - loss: 1.2623 - mean_io_u: 0.0625


950/Unknown  140s 105ms/step - categorical_accuracy: 0.6912 - loss: 1.2621 - mean_io_u: 0.0625


951/Unknown  140s 105ms/step - categorical_accuracy: 0.6913 - loss: 1.2619 - mean_io_u: 0.0626


952/Unknown  140s 105ms/step - categorical_accuracy: 0.6913 - loss: 1.2617 - mean_io_u: 0.0626


953/Unknown  140s 105ms/step - categorical_accuracy: 0.6914 - loss: 1.2614 - mean_io_u: 0.0626


954/Unknown  140s 105ms/step - categorical_accuracy: 0.6914 - loss: 1.2612 - mean_io_u: 0.0627


955/Unknown  140s 105ms/step - categorical_accuracy: 0.6915 - loss: 1.2610 - mean_io_u: 0.0627


956/Unknown  140s 105ms/step - categorical_accuracy: 0.6915 - loss: 1.2608 - mean_io_u: 0.0627


957/Unknown  140s 105ms/step - categorical_accuracy: 0.6916 - loss: 1.2605 - mean_io_u: 0.0628


958/Unknown  140s 105ms/step - categorical_accuracy: 0.6916 - loss: 1.2603 - mean_io_u: 0.0628


959/Unknown  140s 105ms/step - categorical_accuracy: 0.6916 - loss: 1.2601 - mean_io_u: 0.0628


960/Unknown  140s 105ms/step - categorical_accuracy: 0.6917 - loss: 1.2599 - mean_io_u: 0.0629


961/Unknown  140s 105ms/step - categorical_accuracy: 0.6917 - loss: 1.2597 - mean_io_u: 0.0629


962/Unknown  140s 105ms/step - categorical_accuracy: 0.6918 - loss: 1.2594 - mean_io_u: 0.0629


963/Unknown  141s 105ms/step - categorical_accuracy: 0.6918 - loss: 1.2592 - mean_io_u: 0.0630


964/Unknown  141s 105ms/step - categorical_accuracy: 0.6919 - loss: 1.2590 - mean_io_u: 0.0630


965/Unknown  141s 105ms/step - categorical_accuracy: 0.6919 - loss: 1.2588 - mean_io_u: 0.0630


966/Unknown  141s 105ms/step - categorical_accuracy: 0.6919 - loss: 1.2586 - mean_io_u: 0.0630


967/Unknown  141s 105ms/step - categorical_accuracy: 0.6920 - loss: 1.2583 - mean_io_u: 0.0631


968/Unknown  141s 105ms/step - categorical_accuracy: 0.6920 - loss: 1.2581 - mean_io_u: 0.0631


969/Unknown  141s 105ms/step - categorical_accuracy: 0.6921 - loss: 1.2579 - mean_io_u: 0.0631


970/Unknown  141s 105ms/step - categorical_accuracy: 0.6921 - loss: 1.2577 - mean_io_u: 0.0632


971/Unknown  141s 105ms/step - categorical_accuracy: 0.6922 - loss: 1.2575 - mean_io_u: 0.0632


972/Unknown  141s 104ms/step - categorical_accuracy: 0.6922 - loss: 1.2572 - mean_io_u: 0.0632


973/Unknown  141s 104ms/step - categorical_accuracy: 0.6922 - loss: 1.2570 - mean_io_u: 0.0633


974/Unknown  141s 104ms/step - categorical_accuracy: 0.6923 - loss: 1.2568 - mean_io_u: 0.0633


975/Unknown  141s 104ms/step - categorical_accuracy: 0.6923 - loss: 1.2566 - mean_io_u: 0.0633


976/Unknown  141s 104ms/step - categorical_accuracy: 0.6924 - loss: 1.2564 - mean_io_u: 0.0634


977/Unknown  142s 104ms/step - categorical_accuracy: 0.6924 - loss: 1.2562 - mean_io_u: 0.0634


978/Unknown  142s 104ms/step - categorical_accuracy: 0.6924 - loss: 1.2560 - mean_io_u: 0.0634


979/Unknown  142s 104ms/step - categorical_accuracy: 0.6925 - loss: 1.2557 - mean_io_u: 0.0634


980/Unknown  142s 104ms/step - categorical_accuracy: 0.6925 - loss: 1.2555 - mean_io_u: 0.0635


981/Unknown  142s 104ms/step - categorical_accuracy: 0.6926 - loss: 1.2553 - mean_io_u: 0.0635


982/Unknown  142s 104ms/step - categorical_accuracy: 0.6926 - loss: 1.2551 - mean_io_u: 0.0635


983/Unknown  142s 104ms/step - categorical_accuracy: 0.6926 - loss: 1.2549 - mean_io_u: 0.0636


984/Unknown  142s 104ms/step - categorical_accuracy: 0.6927 - loss: 1.2547 - mean_io_u: 0.0636


985/Unknown  142s 104ms/step - categorical_accuracy: 0.6927 - loss: 1.2545 - mean_io_u: 0.0636


986/Unknown  142s 104ms/step - categorical_accuracy: 0.6928 - loss: 1.2542 - mean_io_u: 0.0637


987/Unknown  142s 104ms/step - categorical_accuracy: 0.6928 - loss: 1.2540 - mean_io_u: 0.0637


988/Unknown  142s 104ms/step - categorical_accuracy: 0.6929 - loss: 1.2538 - mean_io_u: 0.0637


989/Unknown  142s 104ms/step - categorical_accuracy: 0.6929 - loss: 1.2536 - mean_io_u: 0.0637


990/Unknown  142s 104ms/step - categorical_accuracy: 0.6929 - loss: 1.2534 - mean_io_u: 0.0638


991/Unknown  143s 104ms/step - categorical_accuracy: 0.6930 - loss: 1.2532 - mean_io_u: 0.0638


992/Unknown  143s 104ms/step - categorical_accuracy: 0.6930 - loss: 1.2530 - mean_io_u: 0.0638


993/Unknown  143s 104ms/step - categorical_accuracy: 0.6931 - loss: 1.2528 - mean_io_u: 0.0639


994/Unknown  143s 104ms/step - categorical_accuracy: 0.6931 - loss: 1.2526 - mean_io_u: 0.0639


995/Unknown  143s 104ms/step - categorical_accuracy: 0.6931 - loss: 1.2523 - mean_io_u: 0.0639


996/Unknown  143s 104ms/step - categorical_accuracy: 0.6932 - loss: 1.2521 - mean_io_u: 0.0640


997/Unknown  143s 104ms/step - categorical_accuracy: 0.6932 - loss: 1.2519 - mean_io_u: 0.0640


998/Unknown  143s 104ms/step - categorical_accuracy: 0.6933 - loss: 1.2517 - mean_io_u: 0.0640


999/Unknown  143s 104ms/step - categorical_accuracy: 0.6933 - loss: 1.2515 - mean_io_u: 0.0640


```
</div>
   1000/Unknown  143s 104ms/step - categorical_accuracy: 0.6933 - loss: 1.2513 - mean_io_u: 0.0641

<div class="k-default-codeblock">
```

```
</div>
   1001/Unknown  143s 104ms/step - categorical_accuracy: 0.6934 - loss: 1.2511 - mean_io_u: 0.0641

<div class="k-default-codeblock">
```

```
</div>
   1002/Unknown  143s 104ms/step - categorical_accuracy: 0.6934 - loss: 1.2509 - mean_io_u: 0.0641

<div class="k-default-codeblock">
```

```
</div>
   1003/Unknown  143s 103ms/step - categorical_accuracy: 0.6935 - loss: 1.2507 - mean_io_u: 0.0642

<div class="k-default-codeblock">
```

```
</div>
   1004/Unknown  144s 103ms/step - categorical_accuracy: 0.6935 - loss: 1.2505 - mean_io_u: 0.0642

<div class="k-default-codeblock">
```

```
</div>
   1005/Unknown  144s 103ms/step - categorical_accuracy: 0.6935 - loss: 1.2503 - mean_io_u: 0.0642

<div class="k-default-codeblock">
```

```
</div>
   1006/Unknown  144s 103ms/step - categorical_accuracy: 0.6936 - loss: 1.2501 - mean_io_u: 0.0643

<div class="k-default-codeblock">
```

```
</div>
   1007/Unknown  144s 103ms/step - categorical_accuracy: 0.6936 - loss: 1.2498 - mean_io_u: 0.0643

<div class="k-default-codeblock">
```

```
</div>
   1008/Unknown  144s 103ms/step - categorical_accuracy: 0.6937 - loss: 1.2496 - mean_io_u: 0.0643

<div class="k-default-codeblock">
```

```
</div>
   1009/Unknown  144s 103ms/step - categorical_accuracy: 0.6937 - loss: 1.2494 - mean_io_u: 0.0643

<div class="k-default-codeblock">
```

```
</div>
   1010/Unknown  144s 103ms/step - categorical_accuracy: 0.6937 - loss: 1.2492 - mean_io_u: 0.0644

<div class="k-default-codeblock">
```

```
</div>
   1011/Unknown  144s 103ms/step - categorical_accuracy: 0.6938 - loss: 1.2490 - mean_io_u: 0.0644

<div class="k-default-codeblock">
```

```
</div>
   1012/Unknown  144s 103ms/step - categorical_accuracy: 0.6938 - loss: 1.2488 - mean_io_u: 0.0644

<div class="k-default-codeblock">
```

```
</div>
   1013/Unknown  144s 103ms/step - categorical_accuracy: 0.6939 - loss: 1.2486 - mean_io_u: 0.0645

<div class="k-default-codeblock">
```

```
</div>
   1014/Unknown  144s 103ms/step - categorical_accuracy: 0.6939 - loss: 1.2484 - mean_io_u: 0.0645

<div class="k-default-codeblock">
```

```
</div>
   1015/Unknown  144s 103ms/step - categorical_accuracy: 0.6939 - loss: 1.2482 - mean_io_u: 0.0645

<div class="k-default-codeblock">
```

```
</div>
   1016/Unknown  144s 103ms/step - categorical_accuracy: 0.6940 - loss: 1.2480 - mean_io_u: 0.0646

<div class="k-default-codeblock">
```

```
</div>
   1017/Unknown  145s 103ms/step - categorical_accuracy: 0.6940 - loss: 1.2478 - mean_io_u: 0.0646

<div class="k-default-codeblock">
```

```
</div>
   1018/Unknown  145s 103ms/step - categorical_accuracy: 0.6941 - loss: 1.2476 - mean_io_u: 0.0646

<div class="k-default-codeblock">
```

```
</div>
   1019/Unknown  145s 103ms/step - categorical_accuracy: 0.6941 - loss: 1.2474 - mean_io_u: 0.0646

<div class="k-default-codeblock">
```

```
</div>
   1020/Unknown  145s 103ms/step - categorical_accuracy: 0.6941 - loss: 1.2472 - mean_io_u: 0.0647

<div class="k-default-codeblock">
```

```
</div>
   1021/Unknown  145s 103ms/step - categorical_accuracy: 0.6942 - loss: 1.2470 - mean_io_u: 0.0647

<div class="k-default-codeblock">
```

```
</div>
   1022/Unknown  145s 103ms/step - categorical_accuracy: 0.6942 - loss: 1.2467 - mean_io_u: 0.0647

<div class="k-default-codeblock">
```

```
</div>
   1023/Unknown  145s 103ms/step - categorical_accuracy: 0.6942 - loss: 1.2465 - mean_io_u: 0.0648

<div class="k-default-codeblock">
```

```
</div>
   1024/Unknown  145s 103ms/step - categorical_accuracy: 0.6943 - loss: 1.2463 - mean_io_u: 0.0648

<div class="k-default-codeblock">
```

```
</div>
   1025/Unknown  145s 103ms/step - categorical_accuracy: 0.6943 - loss: 1.2461 - mean_io_u: 0.0648

<div class="k-default-codeblock">
```

```
</div>
   1026/Unknown  145s 103ms/step - categorical_accuracy: 0.6944 - loss: 1.2459 - mean_io_u: 0.0649

<div class="k-default-codeblock">
```

```
</div>
   1027/Unknown  145s 103ms/step - categorical_accuracy: 0.6944 - loss: 1.2457 - mean_io_u: 0.0649

<div class="k-default-codeblock">
```

```
</div>
   1028/Unknown  145s 103ms/step - categorical_accuracy: 0.6944 - loss: 1.2455 - mean_io_u: 0.0649

<div class="k-default-codeblock">
```

```
</div>
   1029/Unknown  145s 103ms/step - categorical_accuracy: 0.6945 - loss: 1.2453 - mean_io_u: 0.0649

<div class="k-default-codeblock">
```

```
</div>
   1030/Unknown  146s 103ms/step - categorical_accuracy: 0.6945 - loss: 1.2451 - mean_io_u: 0.0650

<div class="k-default-codeblock">
```

```
</div>
   1031/Unknown  146s 103ms/step - categorical_accuracy: 0.6946 - loss: 1.2449 - mean_io_u: 0.0650

<div class="k-default-codeblock">
```

```
</div>
   1032/Unknown  146s 103ms/step - categorical_accuracy: 0.6946 - loss: 1.2447 - mean_io_u: 0.0650

<div class="k-default-codeblock">
```

```
</div>
   1033/Unknown  146s 103ms/step - categorical_accuracy: 0.6946 - loss: 1.2445 - mean_io_u: 0.0651

<div class="k-default-codeblock">
```

```
</div>
   1034/Unknown  146s 103ms/step - categorical_accuracy: 0.6947 - loss: 1.2443 - mean_io_u: 0.0651

<div class="k-default-codeblock">
```

```
</div>
   1035/Unknown  146s 103ms/step - categorical_accuracy: 0.6947 - loss: 1.2441 - mean_io_u: 0.0651

<div class="k-default-codeblock">
```

```
</div>
   1036/Unknown  146s 103ms/step - categorical_accuracy: 0.6948 - loss: 1.2439 - mean_io_u: 0.0652

<div class="k-default-codeblock">
```

```
</div>
   1037/Unknown  146s 103ms/step - categorical_accuracy: 0.6948 - loss: 1.2437 - mean_io_u: 0.0652

<div class="k-default-codeblock">
```

```
</div>
   1038/Unknown  146s 103ms/step - categorical_accuracy: 0.6948 - loss: 1.2435 - mean_io_u: 0.0652

<div class="k-default-codeblock">
```

```
</div>
   1039/Unknown  146s 103ms/step - categorical_accuracy: 0.6949 - loss: 1.2433 - mean_io_u: 0.0652

<div class="k-default-codeblock">
```

```
</div>
   1040/Unknown  146s 103ms/step - categorical_accuracy: 0.6949 - loss: 1.2431 - mean_io_u: 0.0653

<div class="k-default-codeblock">
```

```
</div>
   1041/Unknown  147s 103ms/step - categorical_accuracy: 0.6949 - loss: 1.2429 - mean_io_u: 0.0653

<div class="k-default-codeblock">
```

```
</div>
   1042/Unknown  147s 103ms/step - categorical_accuracy: 0.6950 - loss: 1.2427 - mean_io_u: 0.0653

<div class="k-default-codeblock">
```

```
</div>
   1043/Unknown  147s 103ms/step - categorical_accuracy: 0.6950 - loss: 1.2425 - mean_io_u: 0.0654

<div class="k-default-codeblock">
```

```
</div>
   1044/Unknown  147s 103ms/step - categorical_accuracy: 0.6951 - loss: 1.2423 - mean_io_u: 0.0654

<div class="k-default-codeblock">
```

```
</div>
   1045/Unknown  147s 103ms/step - categorical_accuracy: 0.6951 - loss: 1.2421 - mean_io_u: 0.0654

<div class="k-default-codeblock">
```

```
</div>
   1046/Unknown  147s 103ms/step - categorical_accuracy: 0.6951 - loss: 1.2419 - mean_io_u: 0.0655

<div class="k-default-codeblock">
```

```
</div>
   1047/Unknown  147s 103ms/step - categorical_accuracy: 0.6952 - loss: 1.2417 - mean_io_u: 0.0655

<div class="k-default-codeblock">
```

```
</div>
   1048/Unknown  147s 103ms/step - categorical_accuracy: 0.6952 - loss: 1.2415 - mean_io_u: 0.0655

<div class="k-default-codeblock">
```

```
</div>
   1049/Unknown  147s 103ms/step - categorical_accuracy: 0.6952 - loss: 1.2413 - mean_io_u: 0.0655

<div class="k-default-codeblock">
```

```
</div>
   1050/Unknown  147s 103ms/step - categorical_accuracy: 0.6953 - loss: 1.2411 - mean_io_u: 0.0656

<div class="k-default-codeblock">
```

```
</div>
   1051/Unknown  147s 103ms/step - categorical_accuracy: 0.6953 - loss: 1.2409 - mean_io_u: 0.0656

<div class="k-default-codeblock">
```

```
</div>
   1052/Unknown  148s 103ms/step - categorical_accuracy: 0.6954 - loss: 1.2408 - mean_io_u: 0.0656

<div class="k-default-codeblock">
```

```
</div>
   1053/Unknown  148s 103ms/step - categorical_accuracy: 0.6954 - loss: 1.2406 - mean_io_u: 0.0657

<div class="k-default-codeblock">
```

```
</div>
   1054/Unknown  148s 102ms/step - categorical_accuracy: 0.6954 - loss: 1.2404 - mean_io_u: 0.0657

<div class="k-default-codeblock">
```

```
</div>
   1055/Unknown  148s 102ms/step - categorical_accuracy: 0.6955 - loss: 1.2402 - mean_io_u: 0.0657

<div class="k-default-codeblock">
```

```
</div>
   1056/Unknown  148s 102ms/step - categorical_accuracy: 0.6955 - loss: 1.2400 - mean_io_u: 0.0657

<div class="k-default-codeblock">
```

```
</div>
   1057/Unknown  148s 102ms/step - categorical_accuracy: 0.6955 - loss: 1.2398 - mean_io_u: 0.0658

<div class="k-default-codeblock">
```

```
</div>
   1058/Unknown  148s 102ms/step - categorical_accuracy: 0.6956 - loss: 1.2396 - mean_io_u: 0.0658

<div class="k-default-codeblock">
```

```
</div>
   1059/Unknown  148s 102ms/step - categorical_accuracy: 0.6956 - loss: 1.2394 - mean_io_u: 0.0658

<div class="k-default-codeblock">
```

```
</div>
   1060/Unknown  148s 102ms/step - categorical_accuracy: 0.6957 - loss: 1.2392 - mean_io_u: 0.0659

<div class="k-default-codeblock">
```

```
</div>
   1061/Unknown  148s 102ms/step - categorical_accuracy: 0.6957 - loss: 1.2390 - mean_io_u: 0.0659

<div class="k-default-codeblock">
```

```
</div>
   1062/Unknown  148s 102ms/step - categorical_accuracy: 0.6957 - loss: 1.2388 - mean_io_u: 0.0659

<div class="k-default-codeblock">
```

```
</div>
   1063/Unknown  148s 102ms/step - categorical_accuracy: 0.6958 - loss: 1.2386 - mean_io_u: 0.0660

<div class="k-default-codeblock">
```

```
</div>
   1064/Unknown  148s 102ms/step - categorical_accuracy: 0.6958 - loss: 1.2384 - mean_io_u: 0.0660

<div class="k-default-codeblock">
```

```
</div>
   1065/Unknown  149s 102ms/step - categorical_accuracy: 0.6958 - loss: 1.2382 - mean_io_u: 0.0660

<div class="k-default-codeblock">
```

```
</div>
   1066/Unknown  149s 102ms/step - categorical_accuracy: 0.6959 - loss: 1.2380 - mean_io_u: 0.0660

<div class="k-default-codeblock">
```

```
</div>
   1067/Unknown  149s 102ms/step - categorical_accuracy: 0.6959 - loss: 1.2378 - mean_io_u: 0.0661

<div class="k-default-codeblock">
```

```
</div>
   1068/Unknown  149s 102ms/step - categorical_accuracy: 0.6959 - loss: 1.2376 - mean_io_u: 0.0661

<div class="k-default-codeblock">
```

```
</div>
   1069/Unknown  149s 102ms/step - categorical_accuracy: 0.6960 - loss: 1.2375 - mean_io_u: 0.0661

<div class="k-default-codeblock">
```

```
</div>
   1070/Unknown  149s 102ms/step - categorical_accuracy: 0.6960 - loss: 1.2373 - mean_io_u: 0.0662

<div class="k-default-codeblock">
```

```
</div>
   1071/Unknown  149s 102ms/step - categorical_accuracy: 0.6961 - loss: 1.2371 - mean_io_u: 0.0662

<div class="k-default-codeblock">
```

```
</div>
   1072/Unknown  149s 102ms/step - categorical_accuracy: 0.6961 - loss: 1.2369 - mean_io_u: 0.0662

<div class="k-default-codeblock">
```

```
</div>
   1073/Unknown  149s 102ms/step - categorical_accuracy: 0.6961 - loss: 1.2367 - mean_io_u: 0.0663

<div class="k-default-codeblock">
```

```
</div>
   1074/Unknown  149s 102ms/step - categorical_accuracy: 0.6962 - loss: 1.2365 - mean_io_u: 0.0663

<div class="k-default-codeblock">
```

```
</div>
   1075/Unknown  149s 102ms/step - categorical_accuracy: 0.6962 - loss: 1.2363 - mean_io_u: 0.0663

<div class="k-default-codeblock">
```

```
</div>
   1076/Unknown  149s 102ms/step - categorical_accuracy: 0.6962 - loss: 1.2361 - mean_io_u: 0.0663

<div class="k-default-codeblock">
```

```
</div>
   1077/Unknown  149s 102ms/step - categorical_accuracy: 0.6963 - loss: 1.2359 - mean_io_u: 0.0664

<div class="k-default-codeblock">
```

```
</div>
   1078/Unknown  150s 102ms/step - categorical_accuracy: 0.6963 - loss: 1.2357 - mean_io_u: 0.0664

<div class="k-default-codeblock">
```

```
</div>
   1079/Unknown  150s 102ms/step - categorical_accuracy: 0.6964 - loss: 1.2355 - mean_io_u: 0.0664

<div class="k-default-codeblock">
```

```
</div>
   1080/Unknown  150s 102ms/step - categorical_accuracy: 0.6964 - loss: 1.2354 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1081/Unknown  150s 102ms/step - categorical_accuracy: 0.6964 - loss: 1.2352 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1082/Unknown  150s 102ms/step - categorical_accuracy: 0.6965 - loss: 1.2350 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1083/Unknown  150s 102ms/step - categorical_accuracy: 0.6965 - loss: 1.2348 - mean_io_u: 0.0665

<div class="k-default-codeblock">
```

```
</div>
   1084/Unknown  150s 102ms/step - categorical_accuracy: 0.6965 - loss: 1.2346 - mean_io_u: 0.0666

<div class="k-default-codeblock">
```

```
</div>
   1085/Unknown  150s 102ms/step - categorical_accuracy: 0.6966 - loss: 1.2344 - mean_io_u: 0.0666

<div class="k-default-codeblock">
```

```
</div>
   1086/Unknown  150s 102ms/step - categorical_accuracy: 0.6966 - loss: 1.2342 - mean_io_u: 0.0666

<div class="k-default-codeblock">
```

```
</div>
   1087/Unknown  150s 102ms/step - categorical_accuracy: 0.6966 - loss: 1.2340 - mean_io_u: 0.0667

<div class="k-default-codeblock">
```

```
</div>
   1088/Unknown  150s 102ms/step - categorical_accuracy: 0.6967 - loss: 1.2339 - mean_io_u: 0.0667

<div class="k-default-codeblock">
```

```
</div>
   1089/Unknown  150s 102ms/step - categorical_accuracy: 0.6967 - loss: 1.2337 - mean_io_u: 0.0667

<div class="k-default-codeblock">
```

```
</div>
   1090/Unknown  150s 102ms/step - categorical_accuracy: 0.6967 - loss: 1.2335 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1091/Unknown  151s 102ms/step - categorical_accuracy: 0.6968 - loss: 1.2333 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1092/Unknown  151s 102ms/step - categorical_accuracy: 0.6968 - loss: 1.2331 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1093/Unknown  151s 102ms/step - categorical_accuracy: 0.6969 - loss: 1.2329 - mean_io_u: 0.0668

<div class="k-default-codeblock">
```

```
</div>
   1094/Unknown  151s 102ms/step - categorical_accuracy: 0.6969 - loss: 1.2327 - mean_io_u: 0.0669

<div class="k-default-codeblock">
```

```
</div>
   1095/Unknown  151s 102ms/step - categorical_accuracy: 0.6969 - loss: 1.2326 - mean_io_u: 0.0669

<div class="k-default-codeblock">
```

```
</div>
   1096/Unknown  151s 102ms/step - categorical_accuracy: 0.6970 - loss: 1.2324 - mean_io_u: 0.0669

<div class="k-default-codeblock">
```

```
</div>
   1097/Unknown  151s 102ms/step - categorical_accuracy: 0.6970 - loss: 1.2322 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1098/Unknown  151s 102ms/step - categorical_accuracy: 0.6970 - loss: 1.2320 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1099/Unknown  151s 102ms/step - categorical_accuracy: 0.6971 - loss: 1.2318 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1100/Unknown  151s 102ms/step - categorical_accuracy: 0.6971 - loss: 1.2316 - mean_io_u: 0.0670

<div class="k-default-codeblock">
```

```
</div>
   1101/Unknown  151s 101ms/step - categorical_accuracy: 0.6971 - loss: 1.2314 - mean_io_u: 0.0671

<div class="k-default-codeblock">
```

```
</div>
   1102/Unknown  151s 101ms/step - categorical_accuracy: 0.6972 - loss: 1.2313 - mean_io_u: 0.0671

<div class="k-default-codeblock">
```

```
</div>
   1103/Unknown  152s 101ms/step - categorical_accuracy: 0.6972 - loss: 1.2311 - mean_io_u: 0.0671

<div class="k-default-codeblock">
```

```
</div>
   1104/Unknown  152s 101ms/step - categorical_accuracy: 0.6972 - loss: 1.2309 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1105/Unknown  152s 101ms/step - categorical_accuracy: 0.6973 - loss: 1.2307 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1106/Unknown  152s 101ms/step - categorical_accuracy: 0.6973 - loss: 1.2305 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1107/Unknown  152s 101ms/step - categorical_accuracy: 0.6974 - loss: 1.2303 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1108/Unknown  152s 101ms/step - categorical_accuracy: 0.6974 - loss: 1.2301 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1109/Unknown  152s 101ms/step - categorical_accuracy: 0.6974 - loss: 1.2300 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1110/Unknown  152s 101ms/step - categorical_accuracy: 0.6975 - loss: 1.2298 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1111/Unknown  152s 101ms/step - categorical_accuracy: 0.6975 - loss: 1.2296 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1112/Unknown  152s 101ms/step - categorical_accuracy: 0.6975 - loss: 1.2294 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1113/Unknown  152s 101ms/step - categorical_accuracy: 0.6976 - loss: 1.2292 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1114/Unknown  152s 101ms/step - categorical_accuracy: 0.6976 - loss: 1.2290 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1115/Unknown  153s 101ms/step - categorical_accuracy: 0.6976 - loss: 1.2289 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1116/Unknown  153s 101ms/step - categorical_accuracy: 0.6977 - loss: 1.2287 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1117/Unknown  153s 101ms/step - categorical_accuracy: 0.6977 - loss: 1.2285 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1118/Unknown  153s 101ms/step - categorical_accuracy: 0.6977 - loss: 1.2283 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1120/Unknown  153s 101ms/step - categorical_accuracy: 0.6978 - loss: 1.2280 - mean_io_u: 0.0676
   1119/Unknown  153s 101ms/step - categorical_accuracy: 0.6978 - loss: 1.2280 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1121/Unknown  153s 101ms/step - categorical_accuracy: 0.6978 - loss: 1.2278 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1122/Unknown  153s 101ms/step - categorical_accuracy: 0.6979 - loss: 1.2276 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1123/Unknown  153s 101ms/step - categorical_accuracy: 0.6979 - loss: 1.2274 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1124/Unknown  153s 101ms/step - categorical_accuracy: 0.6980 - loss: 1.2272 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1125/Unknown  153s 101ms/step - categorical_accuracy: 0.6980 - loss: 1.2270 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1126/Unknown  153s 101ms/step - categorical_accuracy: 0.6980 - loss: 1.2269 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1127/Unknown  153s 101ms/step - categorical_accuracy: 0.6981 - loss: 1.2267 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1128/Unknown  153s 101ms/step - categorical_accuracy: 0.6981 - loss: 1.2265 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1129/Unknown  153s 101ms/step - categorical_accuracy: 0.6981 - loss: 1.2263 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1130/Unknown  154s 101ms/step - categorical_accuracy: 0.6982 - loss: 1.2261 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1131/Unknown  154s 101ms/step - categorical_accuracy: 0.6982 - loss: 1.2260 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1132/Unknown  154s 101ms/step - categorical_accuracy: 0.6982 - loss: 1.2258 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1133/Unknown  154s 101ms/step - categorical_accuracy: 0.6983 - loss: 1.2256 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1134/Unknown  154s 101ms/step - categorical_accuracy: 0.6983 - loss: 1.2254 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1135/Unknown  154s 101ms/step - categorical_accuracy: 0.6983 - loss: 1.2252 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1136/Unknown  154s 101ms/step - categorical_accuracy: 0.6984 - loss: 1.2251 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1137/Unknown  154s 101ms/step - categorical_accuracy: 0.6984 - loss: 1.2249 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1138/Unknown  154s 101ms/step - categorical_accuracy: 0.6984 - loss: 1.2247 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1139/Unknown  154s 101ms/step - categorical_accuracy: 0.6985 - loss: 1.2245 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1140/Unknown  154s 101ms/step - categorical_accuracy: 0.6985 - loss: 1.2243 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1141/Unknown  154s 101ms/step - categorical_accuracy: 0.6985 - loss: 1.2242 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1142/Unknown  154s 100ms/step - categorical_accuracy: 0.6986 - loss: 1.2240 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1143/Unknown  154s 100ms/step - categorical_accuracy: 0.6986 - loss: 1.2238 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1144/Unknown  155s 100ms/step - categorical_accuracy: 0.6986 - loss: 1.2236 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1145/Unknown  155s 100ms/step - categorical_accuracy: 0.6987 - loss: 1.2234 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1146/Unknown  155s 100ms/step - categorical_accuracy: 0.6987 - loss: 1.2233 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1147/Unknown  155s 100ms/step - categorical_accuracy: 0.6987 - loss: 1.2231 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1148/Unknown  155s 100ms/step - categorical_accuracy: 0.6988 - loss: 1.2229 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1149/Unknown  155s 100ms/step - categorical_accuracy: 0.6988 - loss: 1.2227 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1150/Unknown  155s 100ms/step - categorical_accuracy: 0.6988 - loss: 1.2225 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1151/Unknown  155s 100ms/step - categorical_accuracy: 0.6989 - loss: 1.2224 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1152/Unknown  155s 100ms/step - categorical_accuracy: 0.6989 - loss: 1.2222 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1153/Unknown  155s 100ms/step - categorical_accuracy: 0.6989 - loss: 1.2220 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1154/Unknown  155s 100ms/step - categorical_accuracy: 0.6990 - loss: 1.2218 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1155/Unknown  155s 100ms/step - categorical_accuracy: 0.6990 - loss: 1.2217 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1156/Unknown  155s 100ms/step - categorical_accuracy: 0.6991 - loss: 1.2215 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1157/Unknown  156s 100ms/step - categorical_accuracy: 0.6991 - loss: 1.2213 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1158/Unknown  156s 100ms/step - categorical_accuracy: 0.6991 - loss: 1.2211 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1159/Unknown  156s 100ms/step - categorical_accuracy: 0.6992 - loss: 1.2209 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1160/Unknown  156s 100ms/step - categorical_accuracy: 0.6992 - loss: 1.2208 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1161/Unknown  156s 100ms/step - categorical_accuracy: 0.6992 - loss: 1.2206 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1162/Unknown  156s 100ms/step - categorical_accuracy: 0.6993 - loss: 1.2204 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1163/Unknown  156s 100ms/step - categorical_accuracy: 0.6993 - loss: 1.2202 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1164/Unknown  156s 100ms/step - categorical_accuracy: 0.6993 - loss: 1.2201 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1165/Unknown  156s 100ms/step - categorical_accuracy: 0.6994 - loss: 1.2199 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1166/Unknown  156s 100ms/step - categorical_accuracy: 0.6994 - loss: 1.2197 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1167/Unknown  156s 100ms/step - categorical_accuracy: 0.6994 - loss: 1.2195 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1168/Unknown  156s 100ms/step - categorical_accuracy: 0.6995 - loss: 1.2194 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1169/Unknown  157s 100ms/step - categorical_accuracy: 0.6995 - loss: 1.2192 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1170/Unknown  157s 100ms/step - categorical_accuracy: 0.6995 - loss: 1.2190 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1171/Unknown  157s 100ms/step - categorical_accuracy: 0.6996 - loss: 1.2188 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1172/Unknown  157s 100ms/step - categorical_accuracy: 0.6996 - loss: 1.2187 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1173/Unknown  157s 100ms/step - categorical_accuracy: 0.6996 - loss: 1.2185 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1174/Unknown  157s 100ms/step - categorical_accuracy: 0.6997 - loss: 1.2183 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1175/Unknown  157s 100ms/step - categorical_accuracy: 0.6997 - loss: 1.2181 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1176/Unknown  157s 100ms/step - categorical_accuracy: 0.6997 - loss: 1.2180 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1177/Unknown  157s 100ms/step - categorical_accuracy: 0.6998 - loss: 1.2178 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1178/Unknown  157s 100ms/step - categorical_accuracy: 0.6998 - loss: 1.2176 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1179/Unknown  157s 100ms/step - categorical_accuracy: 0.6998 - loss: 1.2174 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1180/Unknown  158s 100ms/step - categorical_accuracy: 0.6999 - loss: 1.2173 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1181/Unknown  158s 100ms/step - categorical_accuracy: 0.6999 - loss: 1.2171 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1182/Unknown  158s 100ms/step - categorical_accuracy: 0.6999 - loss: 1.2169 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1183/Unknown  158s 100ms/step - categorical_accuracy: 0.7000 - loss: 1.2167 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1184/Unknown  158s 100ms/step - categorical_accuracy: 0.7000 - loss: 1.2166 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1185/Unknown  158s 100ms/step - categorical_accuracy: 0.7000 - loss: 1.2164 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1186/Unknown  158s 100ms/step - categorical_accuracy: 0.7001 - loss: 1.2162 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1187/Unknown  158s 100ms/step - categorical_accuracy: 0.7001 - loss: 1.2160 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1188/Unknown  158s 100ms/step - categorical_accuracy: 0.7001 - loss: 1.2159 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1189/Unknown  158s 100ms/step - categorical_accuracy: 0.7002 - loss: 1.2157 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1190/Unknown  158s 100ms/step - categorical_accuracy: 0.7002 - loss: 1.2155 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1191/Unknown  158s 100ms/step - categorical_accuracy: 0.7002 - loss: 1.2153 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1192/Unknown  159s 100ms/step - categorical_accuracy: 0.7003 - loss: 1.2152 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1193/Unknown  159s 100ms/step - categorical_accuracy: 0.7003 - loss: 1.2150 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1194/Unknown  159s 100ms/step - categorical_accuracy: 0.7003 - loss: 1.2148 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1195/Unknown  159s 100ms/step - categorical_accuracy: 0.7004 - loss: 1.2146 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1196/Unknown  159s 100ms/step - categorical_accuracy: 0.7004 - loss: 1.2145 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1197/Unknown  159s 100ms/step - categorical_accuracy: 0.7004 - loss: 1.2143 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1198/Unknown  159s 100ms/step - categorical_accuracy: 0.7005 - loss: 1.2141 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1199/Unknown  159s 100ms/step - categorical_accuracy: 0.7005 - loss: 1.2140 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1200/Unknown  159s 100ms/step - categorical_accuracy: 0.7005 - loss: 1.2138 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1201/Unknown  159s 100ms/step - categorical_accuracy: 0.7006 - loss: 1.2136 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1202/Unknown  159s 100ms/step - categorical_accuracy: 0.7006 - loss: 1.2134 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1203/Unknown  159s 99ms/step - categorical_accuracy: 0.7006 - loss: 1.2133 - mean_io_u: 0.0700 

<div class="k-default-codeblock">
```

```
</div>
   1204/Unknown  159s 99ms/step - categorical_accuracy: 0.7007 - loss: 1.2131 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1205/Unknown  159s 99ms/step - categorical_accuracy: 0.7007 - loss: 1.2129 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1206/Unknown  160s 99ms/step - categorical_accuracy: 0.7007 - loss: 1.2128 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1207/Unknown  160s 99ms/step - categorical_accuracy: 0.7008 - loss: 1.2126 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1208/Unknown  160s 99ms/step - categorical_accuracy: 0.7008 - loss: 1.2124 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1209/Unknown  160s 99ms/step - categorical_accuracy: 0.7008 - loss: 1.2122 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1210/Unknown  160s 99ms/step - categorical_accuracy: 0.7009 - loss: 1.2121 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1211/Unknown  160s 99ms/step - categorical_accuracy: 0.7009 - loss: 1.2119 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1212/Unknown  160s 99ms/step - categorical_accuracy: 0.7009 - loss: 1.2117 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1213/Unknown  160s 99ms/step - categorical_accuracy: 0.7010 - loss: 1.2116 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1214/Unknown  160s 99ms/step - categorical_accuracy: 0.7010 - loss: 1.2114 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1215/Unknown  160s 99ms/step - categorical_accuracy: 0.7010 - loss: 1.2112 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1216/Unknown  160s 99ms/step - categorical_accuracy: 0.7011 - loss: 1.2111 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1217/Unknown  160s 99ms/step - categorical_accuracy: 0.7011 - loss: 1.2109 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1218/Unknown  161s 99ms/step - categorical_accuracy: 0.7011 - loss: 1.2107 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1219/Unknown  161s 99ms/step - categorical_accuracy: 0.7011 - loss: 1.2105 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1220/Unknown  161s 99ms/step - categorical_accuracy: 0.7012 - loss: 1.2104 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1221/Unknown  161s 99ms/step - categorical_accuracy: 0.7012 - loss: 1.2102 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1222/Unknown  161s 99ms/step - categorical_accuracy: 0.7012 - loss: 1.2100 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1223/Unknown  161s 99ms/step - categorical_accuracy: 0.7013 - loss: 1.2099 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1224/Unknown  161s 99ms/step - categorical_accuracy: 0.7013 - loss: 1.2097 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1225/Unknown  161s 99ms/step - categorical_accuracy: 0.7013 - loss: 1.2095 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1226/Unknown  161s 99ms/step - categorical_accuracy: 0.7014 - loss: 1.2094 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1227/Unknown  161s 99ms/step - categorical_accuracy: 0.7014 - loss: 1.2092 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1228/Unknown  161s 99ms/step - categorical_accuracy: 0.7014 - loss: 1.2090 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1229/Unknown  162s 99ms/step - categorical_accuracy: 0.7015 - loss: 1.2089 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1230/Unknown  162s 99ms/step - categorical_accuracy: 0.7015 - loss: 1.2087 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1231/Unknown  162s 99ms/step - categorical_accuracy: 0.7015 - loss: 1.2085 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1232/Unknown  162s 99ms/step - categorical_accuracy: 0.7016 - loss: 1.2084 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1233/Unknown  162s 99ms/step - categorical_accuracy: 0.7016 - loss: 1.2082 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1234/Unknown  162s 99ms/step - categorical_accuracy: 0.7016 - loss: 1.2080 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1235/Unknown  162s 99ms/step - categorical_accuracy: 0.7017 - loss: 1.2079 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1236/Unknown  162s 99ms/step - categorical_accuracy: 0.7017 - loss: 1.2077 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1237/Unknown  162s 99ms/step - categorical_accuracy: 0.7017 - loss: 1.2075 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1238/Unknown  162s 99ms/step - categorical_accuracy: 0.7018 - loss: 1.2074 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1239/Unknown  162s 99ms/step - categorical_accuracy: 0.7018 - loss: 1.2072 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1240/Unknown  162s 99ms/step - categorical_accuracy: 0.7018 - loss: 1.2070 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1241/Unknown  162s 99ms/step - categorical_accuracy: 0.7018 - loss: 1.2069 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1242/Unknown  163s 99ms/step - categorical_accuracy: 0.7019 - loss: 1.2067 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1243/Unknown  163s 99ms/step - categorical_accuracy: 0.7019 - loss: 1.2065 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1244/Unknown  163s 99ms/step - categorical_accuracy: 0.7019 - loss: 1.2064 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1245/Unknown  163s 99ms/step - categorical_accuracy: 0.7020 - loss: 1.2062 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1246/Unknown  163s 99ms/step - categorical_accuracy: 0.7020 - loss: 1.2061 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1247/Unknown  163s 99ms/step - categorical_accuracy: 0.7020 - loss: 1.2059 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1248/Unknown  163s 99ms/step - categorical_accuracy: 0.7021 - loss: 1.2057 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1249/Unknown  163s 99ms/step - categorical_accuracy: 0.7021 - loss: 1.2056 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1250/Unknown  163s 99ms/step - categorical_accuracy: 0.7021 - loss: 1.2054 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1251/Unknown  163s 99ms/step - categorical_accuracy: 0.7022 - loss: 1.2052 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1252/Unknown  163s 99ms/step - categorical_accuracy: 0.7022 - loss: 1.2051 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1253/Unknown  163s 99ms/step - categorical_accuracy: 0.7022 - loss: 1.2049 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1254/Unknown  163s 99ms/step - categorical_accuracy: 0.7023 - loss: 1.2048 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1255/Unknown  163s 99ms/step - categorical_accuracy: 0.7023 - loss: 1.2046 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1256/Unknown  164s 99ms/step - categorical_accuracy: 0.7023 - loss: 1.2044 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1257/Unknown  164s 99ms/step - categorical_accuracy: 0.7023 - loss: 1.2043 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1258/Unknown  164s 99ms/step - categorical_accuracy: 0.7024 - loss: 1.2041 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1259/Unknown  164s 99ms/step - categorical_accuracy: 0.7024 - loss: 1.2040 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1260/Unknown  164s 99ms/step - categorical_accuracy: 0.7024 - loss: 1.2038 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1261/Unknown  164s 99ms/step - categorical_accuracy: 0.7025 - loss: 1.2036 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1262/Unknown  164s 99ms/step - categorical_accuracy: 0.7025 - loss: 1.2035 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1263/Unknown  164s 98ms/step - categorical_accuracy: 0.7025 - loss: 1.2033 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1264/Unknown  164s 98ms/step - categorical_accuracy: 0.7026 - loss: 1.2032 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1265/Unknown  164s 98ms/step - categorical_accuracy: 0.7026 - loss: 1.2030 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1266/Unknown  164s 98ms/step - categorical_accuracy: 0.7026 - loss: 1.2028 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1267/Unknown  164s 98ms/step - categorical_accuracy: 0.7026 - loss: 1.2027 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1268/Unknown  164s 98ms/step - categorical_accuracy: 0.7027 - loss: 1.2025 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1269/Unknown  164s 98ms/step - categorical_accuracy: 0.7027 - loss: 1.2024 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1270/Unknown  164s 98ms/step - categorical_accuracy: 0.7027 - loss: 1.2022 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1271/Unknown  165s 98ms/step - categorical_accuracy: 0.7028 - loss: 1.2021 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1272/Unknown  165s 98ms/step - categorical_accuracy: 0.7028 - loss: 1.2019 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1273/Unknown  165s 98ms/step - categorical_accuracy: 0.7028 - loss: 1.2017 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1274/Unknown  165s 98ms/step - categorical_accuracy: 0.7029 - loss: 1.2016 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1275/Unknown  165s 98ms/step - categorical_accuracy: 0.7029 - loss: 1.2014 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1276/Unknown  165s 98ms/step - categorical_accuracy: 0.7029 - loss: 1.2013 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1277/Unknown  165s 98ms/step - categorical_accuracy: 0.7029 - loss: 1.2011 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1278/Unknown  165s 98ms/step - categorical_accuracy: 0.7030 - loss: 1.2010 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1279/Unknown  165s 98ms/step - categorical_accuracy: 0.7030 - loss: 1.2008 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1280/Unknown  165s 98ms/step - categorical_accuracy: 0.7030 - loss: 1.2006 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1281/Unknown  165s 98ms/step - categorical_accuracy: 0.7031 - loss: 1.2005 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1282/Unknown  165s 98ms/step - categorical_accuracy: 0.7031 - loss: 1.2003 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1283/Unknown  165s 98ms/step - categorical_accuracy: 0.7031 - loss: 1.2002 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1284/Unknown  165s 98ms/step - categorical_accuracy: 0.7032 - loss: 1.2000 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1285/Unknown  165s 98ms/step - categorical_accuracy: 0.7032 - loss: 1.1999 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1286/Unknown  166s 98ms/step - categorical_accuracy: 0.7032 - loss: 1.1997 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1287/Unknown  166s 98ms/step - categorical_accuracy: 0.7032 - loss: 1.1995 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1288/Unknown  166s 98ms/step - categorical_accuracy: 0.7033 - loss: 1.1994 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1289/Unknown  166s 98ms/step - categorical_accuracy: 0.7033 - loss: 1.1992 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1290/Unknown  166s 98ms/step - categorical_accuracy: 0.7033 - loss: 1.1991 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1291/Unknown  166s 98ms/step - categorical_accuracy: 0.7034 - loss: 1.1989 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1292/Unknown  166s 98ms/step - categorical_accuracy: 0.7034 - loss: 1.1988 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1293/Unknown  166s 98ms/step - categorical_accuracy: 0.7034 - loss: 1.1986 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1294/Unknown  166s 98ms/step - categorical_accuracy: 0.7035 - loss: 1.1985 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1295/Unknown  166s 98ms/step - categorical_accuracy: 0.7035 - loss: 1.1983 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1296/Unknown  166s 98ms/step - categorical_accuracy: 0.7035 - loss: 1.1981 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1297/Unknown  166s 98ms/step - categorical_accuracy: 0.7035 - loss: 1.1980 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1298/Unknown  166s 98ms/step - categorical_accuracy: 0.7036 - loss: 1.1978 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1299/Unknown  167s 98ms/step - categorical_accuracy: 0.7036 - loss: 1.1977 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1300/Unknown  167s 98ms/step - categorical_accuracy: 0.7036 - loss: 1.1975 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1301/Unknown  167s 98ms/step - categorical_accuracy: 0.7037 - loss: 1.1974 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1302/Unknown  167s 98ms/step - categorical_accuracy: 0.7037 - loss: 1.1972 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1303/Unknown  167s 98ms/step - categorical_accuracy: 0.7037 - loss: 1.1971 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1304/Unknown  167s 98ms/step - categorical_accuracy: 0.7038 - loss: 1.1969 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1305/Unknown  167s 98ms/step - categorical_accuracy: 0.7038 - loss: 1.1967 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1306/Unknown  167s 98ms/step - categorical_accuracy: 0.7038 - loss: 1.1966 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1307/Unknown  167s 98ms/step - categorical_accuracy: 0.7038 - loss: 1.1964 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1308/Unknown  167s 98ms/step - categorical_accuracy: 0.7039 - loss: 1.1963 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1309/Unknown  167s 97ms/step - categorical_accuracy: 0.7039 - loss: 1.1961 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1310/Unknown  167s 97ms/step - categorical_accuracy: 0.7039 - loss: 1.1960 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1311/Unknown  167s 97ms/step - categorical_accuracy: 0.7040 - loss: 1.1958 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1312/Unknown  168s 97ms/step - categorical_accuracy: 0.7040 - loss: 1.1957 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1313/Unknown  168s 97ms/step - categorical_accuracy: 0.7040 - loss: 1.1955 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1314/Unknown  168s 97ms/step - categorical_accuracy: 0.7041 - loss: 1.1954 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1315/Unknown  168s 97ms/step - categorical_accuracy: 0.7041 - loss: 1.1952 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1316/Unknown  168s 97ms/step - categorical_accuracy: 0.7041 - loss: 1.1951 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1317/Unknown  168s 97ms/step - categorical_accuracy: 0.7041 - loss: 1.1949 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1318/Unknown  168s 97ms/step - categorical_accuracy: 0.7042 - loss: 1.1948 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1319/Unknown  168s 97ms/step - categorical_accuracy: 0.7042 - loss: 1.1946 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1320/Unknown  168s 97ms/step - categorical_accuracy: 0.7042 - loss: 1.1945 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1321/Unknown  168s 97ms/step - categorical_accuracy: 0.7043 - loss: 1.1943 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1322/Unknown  168s 97ms/step - categorical_accuracy: 0.7043 - loss: 1.1941 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1323/Unknown  168s 97ms/step - categorical_accuracy: 0.7043 - loss: 1.1940 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1324/Unknown  168s 97ms/step - categorical_accuracy: 0.7043 - loss: 1.1938 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1325/Unknown  169s 97ms/step - categorical_accuracy: 0.7044 - loss: 1.1937 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1326/Unknown  169s 97ms/step - categorical_accuracy: 0.7044 - loss: 1.1935 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1327/Unknown  169s 97ms/step - categorical_accuracy: 0.7044 - loss: 1.1934 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1328/Unknown  169s 97ms/step - categorical_accuracy: 0.7045 - loss: 1.1932 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1329/Unknown  169s 97ms/step - categorical_accuracy: 0.7045 - loss: 1.1931 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1330/Unknown  169s 97ms/step - categorical_accuracy: 0.7045 - loss: 1.1929 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1331/Unknown  169s 97ms/step - categorical_accuracy: 0.7045 - loss: 1.1928 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1332/Unknown  169s 97ms/step - categorical_accuracy: 0.7046 - loss: 1.1926 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1333/Unknown  169s 97ms/step - categorical_accuracy: 0.7046 - loss: 1.1925 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1334/Unknown  169s 97ms/step - categorical_accuracy: 0.7046 - loss: 1.1923 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1335/Unknown  169s 97ms/step - categorical_accuracy: 0.7047 - loss: 1.1922 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1336/Unknown  170s 97ms/step - categorical_accuracy: 0.7047 - loss: 1.1920 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1337/Unknown  170s 97ms/step - categorical_accuracy: 0.7047 - loss: 1.1919 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1338/Unknown  170s 97ms/step - categorical_accuracy: 0.7047 - loss: 1.1917 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1339/Unknown  170s 97ms/step - categorical_accuracy: 0.7048 - loss: 1.1916 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1340/Unknown  170s 97ms/step - categorical_accuracy: 0.7048 - loss: 1.1914 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1341/Unknown  170s 97ms/step - categorical_accuracy: 0.7048 - loss: 1.1913 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1342/Unknown  170s 97ms/step - categorical_accuracy: 0.7049 - loss: 1.1911 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1343/Unknown  170s 97ms/step - categorical_accuracy: 0.7049 - loss: 1.1910 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1344/Unknown  170s 97ms/step - categorical_accuracy: 0.7049 - loss: 1.1908 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1345/Unknown  170s 97ms/step - categorical_accuracy: 0.7049 - loss: 1.1907 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1346/Unknown  170s 97ms/step - categorical_accuracy: 0.7050 - loss: 1.1905 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1347/Unknown  170s 97ms/step - categorical_accuracy: 0.7050 - loss: 1.1904 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1348/Unknown  171s 97ms/step - categorical_accuracy: 0.7050 - loss: 1.1902 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1349/Unknown  171s 97ms/step - categorical_accuracy: 0.7051 - loss: 1.1901 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1350/Unknown  171s 97ms/step - categorical_accuracy: 0.7051 - loss: 1.1900 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1351/Unknown  171s 97ms/step - categorical_accuracy: 0.7051 - loss: 1.1898 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1352/Unknown  171s 97ms/step - categorical_accuracy: 0.7051 - loss: 1.1897 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1353/Unknown  171s 97ms/step - categorical_accuracy: 0.7052 - loss: 1.1895 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1354/Unknown  171s 97ms/step - categorical_accuracy: 0.7052 - loss: 1.1894 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1355/Unknown  171s 97ms/step - categorical_accuracy: 0.7052 - loss: 1.1892 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1356/Unknown  171s 97ms/step - categorical_accuracy: 0.7053 - loss: 1.1891 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1357/Unknown  171s 97ms/step - categorical_accuracy: 0.7053 - loss: 1.1889 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1358/Unknown  171s 97ms/step - categorical_accuracy: 0.7053 - loss: 1.1888 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1359/Unknown  171s 97ms/step - categorical_accuracy: 0.7053 - loss: 1.1886 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1360/Unknown  171s 97ms/step - categorical_accuracy: 0.7054 - loss: 1.1885 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1361/Unknown  172s 97ms/step - categorical_accuracy: 0.7054 - loss: 1.1883 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1362/Unknown  172s 97ms/step - categorical_accuracy: 0.7054 - loss: 1.1882 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1363/Unknown  172s 97ms/step - categorical_accuracy: 0.7055 - loss: 1.1880 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1364/Unknown  172s 97ms/step - categorical_accuracy: 0.7055 - loss: 1.1879 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1365/Unknown  172s 97ms/step - categorical_accuracy: 0.7055 - loss: 1.1877 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1366/Unknown  172s 97ms/step - categorical_accuracy: 0.7055 - loss: 1.1876 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1367/Unknown  172s 97ms/step - categorical_accuracy: 0.7056 - loss: 1.1874 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1368/Unknown  172s 97ms/step - categorical_accuracy: 0.7056 - loss: 1.1873 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1369/Unknown  172s 97ms/step - categorical_accuracy: 0.7056 - loss: 1.1871 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1370/Unknown  172s 97ms/step - categorical_accuracy: 0.7057 - loss: 1.1870 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1371/Unknown  172s 97ms/step - categorical_accuracy: 0.7057 - loss: 1.1868 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1372/Unknown  172s 97ms/step - categorical_accuracy: 0.7057 - loss: 1.1867 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1373/Unknown  173s 97ms/step - categorical_accuracy: 0.7057 - loss: 1.1866 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1374/Unknown  173s 97ms/step - categorical_accuracy: 0.7058 - loss: 1.1864 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1375/Unknown  173s 97ms/step - categorical_accuracy: 0.7058 - loss: 1.1863 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1376/Unknown  173s 97ms/step - categorical_accuracy: 0.7058 - loss: 1.1861 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1377/Unknown  173s 97ms/step - categorical_accuracy: 0.7059 - loss: 1.1860 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1378/Unknown  173s 97ms/step - categorical_accuracy: 0.7059 - loss: 1.1858 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1379/Unknown  173s 97ms/step - categorical_accuracy: 0.7059 - loss: 1.1857 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1380/Unknown  173s 97ms/step - categorical_accuracy: 0.7059 - loss: 1.1855 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1381/Unknown  173s 97ms/step - categorical_accuracy: 0.7060 - loss: 1.1854 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1382/Unknown  173s 97ms/step - categorical_accuracy: 0.7060 - loss: 1.1852 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1383/Unknown  173s 97ms/step - categorical_accuracy: 0.7060 - loss: 1.1851 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1384/Unknown  173s 97ms/step - categorical_accuracy: 0.7061 - loss: 1.1850 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1385/Unknown  173s 97ms/step - categorical_accuracy: 0.7061 - loss: 1.1848 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1386/Unknown  174s 97ms/step - categorical_accuracy: 0.7061 - loss: 1.1847 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1387/Unknown  174s 97ms/step - categorical_accuracy: 0.7061 - loss: 1.1845 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1388/Unknown  174s 97ms/step - categorical_accuracy: 0.7062 - loss: 1.1844 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1389/Unknown  174s 97ms/step - categorical_accuracy: 0.7062 - loss: 1.1842 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1390/Unknown  174s 97ms/step - categorical_accuracy: 0.7062 - loss: 1.1841 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1391/Unknown  174s 97ms/step - categorical_accuracy: 0.7063 - loss: 1.1839 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1392/Unknown  174s 97ms/step - categorical_accuracy: 0.7063 - loss: 1.1838 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1393/Unknown  174s 97ms/step - categorical_accuracy: 0.7063 - loss: 1.1837 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1394/Unknown  174s 97ms/step - categorical_accuracy: 0.7063 - loss: 1.1835 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1395/Unknown  174s 97ms/step - categorical_accuracy: 0.7064 - loss: 1.1834 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1396/Unknown  174s 97ms/step - categorical_accuracy: 0.7064 - loss: 1.1832 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1397/Unknown  174s 96ms/step - categorical_accuracy: 0.7064 - loss: 1.1831 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1398/Unknown  175s 96ms/step - categorical_accuracy: 0.7064 - loss: 1.1829 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1399/Unknown  175s 96ms/step - categorical_accuracy: 0.7065 - loss: 1.1828 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1400/Unknown  175s 96ms/step - categorical_accuracy: 0.7065 - loss: 1.1826 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1401/Unknown  175s 96ms/step - categorical_accuracy: 0.7065 - loss: 1.1825 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1402/Unknown  175s 96ms/step - categorical_accuracy: 0.7066 - loss: 1.1824 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1403/Unknown  175s 96ms/step - categorical_accuracy: 0.7066 - loss: 1.1822 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1404/Unknown  175s 96ms/step - categorical_accuracy: 0.7066 - loss: 1.1821 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1405/Unknown  175s 96ms/step - categorical_accuracy: 0.7066 - loss: 1.1819 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1406/Unknown  175s 96ms/step - categorical_accuracy: 0.7067 - loss: 1.1818 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1407/Unknown  175s 96ms/step - categorical_accuracy: 0.7067 - loss: 1.1817 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1408/Unknown  175s 96ms/step - categorical_accuracy: 0.7067 - loss: 1.1815 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1409/Unknown  175s 96ms/step - categorical_accuracy: 0.7068 - loss: 1.1814 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1410/Unknown  175s 96ms/step - categorical_accuracy: 0.7068 - loss: 1.1812 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1411/Unknown  175s 96ms/step - categorical_accuracy: 0.7068 - loss: 1.1811 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1412/Unknown  176s 96ms/step - categorical_accuracy: 0.7068 - loss: 1.1809 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1413/Unknown  176s 96ms/step - categorical_accuracy: 0.7069 - loss: 1.1808 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1414/Unknown  176s 96ms/step - categorical_accuracy: 0.7069 - loss: 1.1807 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1415/Unknown  176s 96ms/step - categorical_accuracy: 0.7069 - loss: 1.1805 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1416/Unknown  176s 96ms/step - categorical_accuracy: 0.7069 - loss: 1.1804 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1417/Unknown  176s 96ms/step - categorical_accuracy: 0.7070 - loss: 1.1802 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1418/Unknown  176s 96ms/step - categorical_accuracy: 0.7070 - loss: 1.1801 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1419/Unknown  176s 96ms/step - categorical_accuracy: 0.7070 - loss: 1.1800 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1420/Unknown  176s 96ms/step - categorical_accuracy: 0.7071 - loss: 1.1798 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1421/Unknown  176s 96ms/step - categorical_accuracy: 0.7071 - loss: 1.1797 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1422/Unknown  176s 96ms/step - categorical_accuracy: 0.7071 - loss: 1.1795 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1423/Unknown  176s 96ms/step - categorical_accuracy: 0.7071 - loss: 1.1794 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1424/Unknown  176s 96ms/step - categorical_accuracy: 0.7072 - loss: 1.1792 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1425/Unknown  176s 96ms/step - categorical_accuracy: 0.7072 - loss: 1.1791 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1426/Unknown  176s 96ms/step - categorical_accuracy: 0.7072 - loss: 1.1790 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1427/Unknown  177s 96ms/step - categorical_accuracy: 0.7072 - loss: 1.1788 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1428/Unknown  177s 96ms/step - categorical_accuracy: 0.7073 - loss: 1.1787 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1429/Unknown  177s 96ms/step - categorical_accuracy: 0.7073 - loss: 1.1785 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1430/Unknown  177s 96ms/step - categorical_accuracy: 0.7073 - loss: 1.1784 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1431/Unknown  177s 96ms/step - categorical_accuracy: 0.7074 - loss: 1.1783 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1432/Unknown  177s 96ms/step - categorical_accuracy: 0.7074 - loss: 1.1781 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1433/Unknown  177s 96ms/step - categorical_accuracy: 0.7074 - loss: 1.1780 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1434/Unknown  177s 96ms/step - categorical_accuracy: 0.7074 - loss: 1.1779 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1435/Unknown  177s 96ms/step - categorical_accuracy: 0.7075 - loss: 1.1777 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1436/Unknown  177s 96ms/step - categorical_accuracy: 0.7075 - loss: 1.1776 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1437/Unknown  177s 96ms/step - categorical_accuracy: 0.7075 - loss: 1.1774 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1438/Unknown  177s 96ms/step - categorical_accuracy: 0.7075 - loss: 1.1773 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1439/Unknown  177s 96ms/step - categorical_accuracy: 0.7076 - loss: 1.1772 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1440/Unknown  177s 96ms/step - categorical_accuracy: 0.7076 - loss: 1.1770 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1441/Unknown  178s 96ms/step - categorical_accuracy: 0.7076 - loss: 1.1769 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1442/Unknown  178s 96ms/step - categorical_accuracy: 0.7076 - loss: 1.1767 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1443/Unknown  178s 96ms/step - categorical_accuracy: 0.7077 - loss: 1.1766 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1444/Unknown  178s 96ms/step - categorical_accuracy: 0.7077 - loss: 1.1765 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1445/Unknown  178s 96ms/step - categorical_accuracy: 0.7077 - loss: 1.1763 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1446/Unknown  178s 96ms/step - categorical_accuracy: 0.7078 - loss: 1.1762 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1447/Unknown  178s 96ms/step - categorical_accuracy: 0.7078 - loss: 1.1761 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1448/Unknown  178s 96ms/step - categorical_accuracy: 0.7078 - loss: 1.1759 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1449/Unknown  178s 96ms/step - categorical_accuracy: 0.7078 - loss: 1.1758 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1450/Unknown  178s 96ms/step - categorical_accuracy: 0.7079 - loss: 1.1756 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1451/Unknown  178s 96ms/step - categorical_accuracy: 0.7079 - loss: 1.1755 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1452/Unknown  178s 96ms/step - categorical_accuracy: 0.7079 - loss: 1.1754 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1453/Unknown  178s 96ms/step - categorical_accuracy: 0.7079 - loss: 1.1752 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1454/Unknown  179s 96ms/step - categorical_accuracy: 0.7080 - loss: 1.1751 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1455/Unknown  179s 96ms/step - categorical_accuracy: 0.7080 - loss: 1.1750 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1456/Unknown  179s 95ms/step - categorical_accuracy: 0.7080 - loss: 1.1748 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1457/Unknown  179s 95ms/step - categorical_accuracy: 0.7080 - loss: 1.1747 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1458/Unknown  179s 95ms/step - categorical_accuracy: 0.7081 - loss: 1.1745 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1459/Unknown  179s 95ms/step - categorical_accuracy: 0.7081 - loss: 1.1744 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1460/Unknown  179s 95ms/step - categorical_accuracy: 0.7081 - loss: 1.1743 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1461/Unknown  179s 95ms/step - categorical_accuracy: 0.7082 - loss: 1.1741 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1462/Unknown  179s 95ms/step - categorical_accuracy: 0.7082 - loss: 1.1740 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1463/Unknown  179s 95ms/step - categorical_accuracy: 0.7082 - loss: 1.1739 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1464/Unknown  179s 95ms/step - categorical_accuracy: 0.7082 - loss: 1.1737 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1465/Unknown  179s 95ms/step - categorical_accuracy: 0.7083 - loss: 1.1736 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1466/Unknown  180s 95ms/step - categorical_accuracy: 0.7083 - loss: 1.1734 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1467/Unknown  180s 95ms/step - categorical_accuracy: 0.7083 - loss: 1.1733 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1468/Unknown  180s 95ms/step - categorical_accuracy: 0.7083 - loss: 1.1732 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1469/Unknown  180s 95ms/step - categorical_accuracy: 0.7084 - loss: 1.1730 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1470/Unknown  180s 95ms/step - categorical_accuracy: 0.7084 - loss: 1.1729 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1471/Unknown  180s 95ms/step - categorical_accuracy: 0.7084 - loss: 1.1728 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1472/Unknown  180s 95ms/step - categorical_accuracy: 0.7084 - loss: 1.1726 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1473/Unknown  180s 95ms/step - categorical_accuracy: 0.7085 - loss: 1.1725 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1474/Unknown  180s 95ms/step - categorical_accuracy: 0.7085 - loss: 1.1724 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1475/Unknown  180s 95ms/step - categorical_accuracy: 0.7085 - loss: 1.1722 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1476/Unknown  180s 95ms/step - categorical_accuracy: 0.7085 - loss: 1.1721 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1477/Unknown  180s 95ms/step - categorical_accuracy: 0.7086 - loss: 1.1720 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1478/Unknown  180s 95ms/step - categorical_accuracy: 0.7086 - loss: 1.1718 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1479/Unknown  180s 95ms/step - categorical_accuracy: 0.7086 - loss: 1.1717 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1480/Unknown  181s 95ms/step - categorical_accuracy: 0.7086 - loss: 1.1716 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1481/Unknown  181s 95ms/step - categorical_accuracy: 0.7087 - loss: 1.1714 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1482/Unknown  181s 95ms/step - categorical_accuracy: 0.7087 - loss: 1.1713 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1483/Unknown  181s 95ms/step - categorical_accuracy: 0.7087 - loss: 1.1712 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1484/Unknown  181s 95ms/step - categorical_accuracy: 0.7088 - loss: 1.1710 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1485/Unknown  181s 95ms/step - categorical_accuracy: 0.7088 - loss: 1.1709 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1486/Unknown  181s 95ms/step - categorical_accuracy: 0.7088 - loss: 1.1707 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1487/Unknown  181s 95ms/step - categorical_accuracy: 0.7088 - loss: 1.1706 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1488/Unknown  181s 95ms/step - categorical_accuracy: 0.7089 - loss: 1.1705 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1489/Unknown  181s 95ms/step - categorical_accuracy: 0.7089 - loss: 1.1703 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1490/Unknown  181s 95ms/step - categorical_accuracy: 0.7089 - loss: 1.1702 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1491/Unknown  182s 95ms/step - categorical_accuracy: 0.7089 - loss: 1.1701 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1492/Unknown  182s 95ms/step - categorical_accuracy: 0.7090 - loss: 1.1699 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1493/Unknown  182s 95ms/step - categorical_accuracy: 0.7090 - loss: 1.1698 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1494/Unknown  182s 95ms/step - categorical_accuracy: 0.7090 - loss: 1.1697 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1495/Unknown  182s 95ms/step - categorical_accuracy: 0.7090 - loss: 1.1695 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1496/Unknown  182s 95ms/step - categorical_accuracy: 0.7091 - loss: 1.1694 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1497/Unknown  182s 95ms/step - categorical_accuracy: 0.7091 - loss: 1.1693 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1498/Unknown  182s 95ms/step - categorical_accuracy: 0.7091 - loss: 1.1691 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1499/Unknown  182s 95ms/step - categorical_accuracy: 0.7091 - loss: 1.1690 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1500/Unknown  182s 95ms/step - categorical_accuracy: 0.7092 - loss: 1.1689 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1501/Unknown  182s 95ms/step - categorical_accuracy: 0.7092 - loss: 1.1687 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1502/Unknown  182s 95ms/step - categorical_accuracy: 0.7092 - loss: 1.1686 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1503/Unknown  183s 95ms/step - categorical_accuracy: 0.7092 - loss: 1.1685 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1504/Unknown  183s 95ms/step - categorical_accuracy: 0.7093 - loss: 1.1684 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1505/Unknown  183s 95ms/step - categorical_accuracy: 0.7093 - loss: 1.1682 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1506/Unknown  183s 95ms/step - categorical_accuracy: 0.7093 - loss: 1.1681 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1507/Unknown  183s 95ms/step - categorical_accuracy: 0.7093 - loss: 1.1680 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1508/Unknown  183s 95ms/step - categorical_accuracy: 0.7094 - loss: 1.1678 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1509/Unknown  183s 95ms/step - categorical_accuracy: 0.7094 - loss: 1.1677 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1510/Unknown  183s 95ms/step - categorical_accuracy: 0.7094 - loss: 1.1676 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1511/Unknown  183s 95ms/step - categorical_accuracy: 0.7094 - loss: 1.1674 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1512/Unknown  183s 95ms/step - categorical_accuracy: 0.7095 - loss: 1.1673 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1513/Unknown  183s 95ms/step - categorical_accuracy: 0.7095 - loss: 1.1672 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1514/Unknown  183s 95ms/step - categorical_accuracy: 0.7095 - loss: 1.1670 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1515/Unknown  184s 95ms/step - categorical_accuracy: 0.7095 - loss: 1.1669 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1516/Unknown  184s 95ms/step - categorical_accuracy: 0.7096 - loss: 1.1668 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1517/Unknown  184s 95ms/step - categorical_accuracy: 0.7096 - loss: 1.1666 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1518/Unknown  184s 95ms/step - categorical_accuracy: 0.7096 - loss: 1.1665 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1519/Unknown  184s 95ms/step - categorical_accuracy: 0.7096 - loss: 1.1664 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1520/Unknown  184s 95ms/step - categorical_accuracy: 0.7097 - loss: 1.1662 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1521/Unknown  184s 95ms/step - categorical_accuracy: 0.7097 - loss: 1.1661 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1522/Unknown  184s 95ms/step - categorical_accuracy: 0.7097 - loss: 1.1660 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1523/Unknown  184s 95ms/step - categorical_accuracy: 0.7097 - loss: 1.1658 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1524/Unknown  184s 95ms/step - categorical_accuracy: 0.7098 - loss: 1.1657 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1525/Unknown  184s 95ms/step - categorical_accuracy: 0.7098 - loss: 1.1656 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1526/Unknown  184s 95ms/step - categorical_accuracy: 0.7098 - loss: 1.1655 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1527/Unknown  184s 95ms/step - categorical_accuracy: 0.7098 - loss: 1.1653 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1528/Unknown  185s 95ms/step - categorical_accuracy: 0.7099 - loss: 1.1652 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1529/Unknown  185s 95ms/step - categorical_accuracy: 0.7099 - loss: 1.1651 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1530/Unknown  185s 95ms/step - categorical_accuracy: 0.7099 - loss: 1.1649 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1531/Unknown  185s 95ms/step - categorical_accuracy: 0.7099 - loss: 1.1648 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1532/Unknown  185s 95ms/step - categorical_accuracy: 0.7100 - loss: 1.1647 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1533/Unknown  185s 95ms/step - categorical_accuracy: 0.7100 - loss: 1.1645 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1534/Unknown  185s 95ms/step - categorical_accuracy: 0.7100 - loss: 1.1644 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1535/Unknown  185s 95ms/step - categorical_accuracy: 0.7100 - loss: 1.1643 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1536/Unknown  185s 95ms/step - categorical_accuracy: 0.7101 - loss: 1.1642 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1537/Unknown  185s 95ms/step - categorical_accuracy: 0.7101 - loss: 1.1640 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1538/Unknown  185s 95ms/step - categorical_accuracy: 0.7101 - loss: 1.1639 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1539/Unknown  185s 95ms/step - categorical_accuracy: 0.7101 - loss: 1.1638 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1540/Unknown  185s 95ms/step - categorical_accuracy: 0.7102 - loss: 1.1636 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1541/Unknown  186s 95ms/step - categorical_accuracy: 0.7102 - loss: 1.1635 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1542/Unknown  186s 95ms/step - categorical_accuracy: 0.7102 - loss: 1.1634 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1543/Unknown  186s 95ms/step - categorical_accuracy: 0.7102 - loss: 1.1633 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1544/Unknown  186s 95ms/step - categorical_accuracy: 0.7103 - loss: 1.1631 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1545/Unknown  186s 95ms/step - categorical_accuracy: 0.7103 - loss: 1.1630 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1546/Unknown  186s 95ms/step - categorical_accuracy: 0.7103 - loss: 1.1629 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1547/Unknown  186s 95ms/step - categorical_accuracy: 0.7103 - loss: 1.1627 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1548/Unknown  186s 95ms/step - categorical_accuracy: 0.7104 - loss: 1.1626 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1549/Unknown  186s 95ms/step - categorical_accuracy: 0.7104 - loss: 1.1625 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1550/Unknown  186s 95ms/step - categorical_accuracy: 0.7104 - loss: 1.1624 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1551/Unknown  186s 95ms/step - categorical_accuracy: 0.7104 - loss: 1.1622 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1552/Unknown  186s 95ms/step - categorical_accuracy: 0.7105 - loss: 1.1621 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1553/Unknown  186s 95ms/step - categorical_accuracy: 0.7105 - loss: 1.1620 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1554/Unknown  187s 94ms/step - categorical_accuracy: 0.7105 - loss: 1.1618 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1555/Unknown  187s 94ms/step - categorical_accuracy: 0.7105 - loss: 1.1617 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1556/Unknown  187s 94ms/step - categorical_accuracy: 0.7106 - loss: 1.1616 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1557/Unknown  187s 94ms/step - categorical_accuracy: 0.7106 - loss: 1.1615 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1558/Unknown  187s 94ms/step - categorical_accuracy: 0.7106 - loss: 1.1613 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1559/Unknown  187s 94ms/step - categorical_accuracy: 0.7106 - loss: 1.1612 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1560/Unknown  187s 94ms/step - categorical_accuracy: 0.7107 - loss: 1.1611 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1561/Unknown  187s 94ms/step - categorical_accuracy: 0.7107 - loss: 1.1610 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1562/Unknown  187s 94ms/step - categorical_accuracy: 0.7107 - loss: 1.1608 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1563/Unknown  187s 94ms/step - categorical_accuracy: 0.7107 - loss: 1.1607 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1564/Unknown  187s 94ms/step - categorical_accuracy: 0.7108 - loss: 1.1606 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1565/Unknown  187s 94ms/step - categorical_accuracy: 0.7108 - loss: 1.1604 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1566/Unknown  187s 94ms/step - categorical_accuracy: 0.7108 - loss: 1.1603 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1567/Unknown  187s 94ms/step - categorical_accuracy: 0.7108 - loss: 1.1602 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1568/Unknown  188s 94ms/step - categorical_accuracy: 0.7109 - loss: 1.1601 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1569/Unknown  188s 94ms/step - categorical_accuracy: 0.7109 - loss: 1.1599 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1570/Unknown  188s 94ms/step - categorical_accuracy: 0.7109 - loss: 1.1598 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1571/Unknown  188s 94ms/step - categorical_accuracy: 0.7109 - loss: 1.1597 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1572/Unknown  188s 94ms/step - categorical_accuracy: 0.7110 - loss: 1.1596 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1573/Unknown  188s 94ms/step - categorical_accuracy: 0.7110 - loss: 1.1594 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1574/Unknown  188s 94ms/step - categorical_accuracy: 0.7110 - loss: 1.1593 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1575/Unknown  188s 94ms/step - categorical_accuracy: 0.7110 - loss: 1.1592 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1576/Unknown  188s 94ms/step - categorical_accuracy: 0.7111 - loss: 1.1591 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1577/Unknown  188s 94ms/step - categorical_accuracy: 0.7111 - loss: 1.1589 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1578/Unknown  188s 94ms/step - categorical_accuracy: 0.7111 - loss: 1.1588 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1579/Unknown  188s 94ms/step - categorical_accuracy: 0.7111 - loss: 1.1587 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1580/Unknown  188s 94ms/step - categorical_accuracy: 0.7111 - loss: 1.1585 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1581/Unknown  188s 94ms/step - categorical_accuracy: 0.7112 - loss: 1.1584 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1582/Unknown  188s 94ms/step - categorical_accuracy: 0.7112 - loss: 1.1583 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1583/Unknown  189s 94ms/step - categorical_accuracy: 0.7112 - loss: 1.1582 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1584/Unknown  189s 94ms/step - categorical_accuracy: 0.7112 - loss: 1.1580 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1585/Unknown  189s 94ms/step - categorical_accuracy: 0.7113 - loss: 1.1579 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1586/Unknown  189s 94ms/step - categorical_accuracy: 0.7113 - loss: 1.1578 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1587/Unknown  189s 94ms/step - categorical_accuracy: 0.7113 - loss: 1.1577 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1588/Unknown  189s 94ms/step - categorical_accuracy: 0.7113 - loss: 1.1575 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1589/Unknown  189s 94ms/step - categorical_accuracy: 0.7114 - loss: 1.1574 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1590/Unknown  189s 94ms/step - categorical_accuracy: 0.7114 - loss: 1.1573 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1591/Unknown  189s 94ms/step - categorical_accuracy: 0.7114 - loss: 1.1572 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1592/Unknown  189s 94ms/step - categorical_accuracy: 0.7114 - loss: 1.1571 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1593/Unknown  189s 94ms/step - categorical_accuracy: 0.7115 - loss: 1.1569 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1594/Unknown  189s 94ms/step - categorical_accuracy: 0.7115 - loss: 1.1568 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1595/Unknown  190s 94ms/step - categorical_accuracy: 0.7115 - loss: 1.1567 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1596/Unknown  190s 94ms/step - categorical_accuracy: 0.7115 - loss: 1.1566 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1597/Unknown  190s 94ms/step - categorical_accuracy: 0.7116 - loss: 1.1564 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1598/Unknown  190s 94ms/step - categorical_accuracy: 0.7116 - loss: 1.1563 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1599/Unknown  190s 94ms/step - categorical_accuracy: 0.7116 - loss: 1.1562 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1600/Unknown  190s 94ms/step - categorical_accuracy: 0.7116 - loss: 1.1561 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1601/Unknown  190s 94ms/step - categorical_accuracy: 0.7116 - loss: 1.1559 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1602/Unknown  190s 94ms/step - categorical_accuracy: 0.7117 - loss: 1.1558 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1603/Unknown  190s 94ms/step - categorical_accuracy: 0.7117 - loss: 1.1557 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1604/Unknown  190s 94ms/step - categorical_accuracy: 0.7117 - loss: 1.1556 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1605/Unknown  190s 94ms/step - categorical_accuracy: 0.7117 - loss: 1.1554 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1606/Unknown  190s 94ms/step - categorical_accuracy: 0.7118 - loss: 1.1553 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1607/Unknown  190s 94ms/step - categorical_accuracy: 0.7118 - loss: 1.1552 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1608/Unknown  191s 94ms/step - categorical_accuracy: 0.7118 - loss: 1.1551 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1609/Unknown  191s 94ms/step - categorical_accuracy: 0.7118 - loss: 1.1549 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1610/Unknown  191s 94ms/step - categorical_accuracy: 0.7119 - loss: 1.1548 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1611/Unknown  191s 94ms/step - categorical_accuracy: 0.7119 - loss: 1.1547 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1612/Unknown  191s 94ms/step - categorical_accuracy: 0.7119 - loss: 1.1546 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1613/Unknown  191s 94ms/step - categorical_accuracy: 0.7119 - loss: 1.1545 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1614/Unknown  191s 94ms/step - categorical_accuracy: 0.7120 - loss: 1.1543 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1615/Unknown  191s 94ms/step - categorical_accuracy: 0.7120 - loss: 1.1542 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1616/Unknown  191s 94ms/step - categorical_accuracy: 0.7120 - loss: 1.1541 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1617/Unknown  191s 94ms/step - categorical_accuracy: 0.7120 - loss: 1.1540 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1618/Unknown  191s 94ms/step - categorical_accuracy: 0.7121 - loss: 1.1538 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1619/Unknown  191s 94ms/step - categorical_accuracy: 0.7121 - loss: 1.1537 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1620/Unknown  191s 94ms/step - categorical_accuracy: 0.7121 - loss: 1.1536 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1621/Unknown  191s 94ms/step - categorical_accuracy: 0.7121 - loss: 1.1535 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1622/Unknown  192s 94ms/step - categorical_accuracy: 0.7121 - loss: 1.1534 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1623/Unknown  192s 94ms/step - categorical_accuracy: 0.7122 - loss: 1.1532 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1624/Unknown  192s 94ms/step - categorical_accuracy: 0.7122 - loss: 1.1531 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1625/Unknown  192s 94ms/step - categorical_accuracy: 0.7122 - loss: 1.1530 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1626/Unknown  192s 94ms/step - categorical_accuracy: 0.7122 - loss: 1.1529 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1627/Unknown  192s 94ms/step - categorical_accuracy: 0.7123 - loss: 1.1527 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1628/Unknown  192s 94ms/step - categorical_accuracy: 0.7123 - loss: 1.1526 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1629/Unknown  192s 94ms/step - categorical_accuracy: 0.7123 - loss: 1.1525 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1630/Unknown  192s 94ms/step - categorical_accuracy: 0.7123 - loss: 1.1524 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1631/Unknown  192s 94ms/step - categorical_accuracy: 0.7124 - loss: 1.1523 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1632/Unknown  192s 94ms/step - categorical_accuracy: 0.7124 - loss: 1.1521 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1633/Unknown  192s 94ms/step - categorical_accuracy: 0.7124 - loss: 1.1520 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1634/Unknown  192s 94ms/step - categorical_accuracy: 0.7124 - loss: 1.1519 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1635/Unknown  193s 94ms/step - categorical_accuracy: 0.7125 - loss: 1.1518 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1636/Unknown  193s 94ms/step - categorical_accuracy: 0.7125 - loss: 1.1517 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1637/Unknown  193s 94ms/step - categorical_accuracy: 0.7125 - loss: 1.1515 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1638/Unknown  193s 93ms/step - categorical_accuracy: 0.7125 - loss: 1.1514 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1639/Unknown  193s 94ms/step - categorical_accuracy: 0.7125 - loss: 1.1513 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1640/Unknown  193s 93ms/step - categorical_accuracy: 0.7126 - loss: 1.1512 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1641/Unknown  193s 93ms/step - categorical_accuracy: 0.7126 - loss: 1.1510 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1642/Unknown  193s 93ms/step - categorical_accuracy: 0.7126 - loss: 1.1509 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1643/Unknown  193s 93ms/step - categorical_accuracy: 0.7126 - loss: 1.1508 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1644/Unknown  193s 93ms/step - categorical_accuracy: 0.7127 - loss: 1.1507 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1645/Unknown  193s 93ms/step - categorical_accuracy: 0.7127 - loss: 1.1506 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1646/Unknown  194s 93ms/step - categorical_accuracy: 0.7127 - loss: 1.1505 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1647/Unknown  194s 93ms/step - categorical_accuracy: 0.7127 - loss: 1.1503 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1648/Unknown  194s 93ms/step - categorical_accuracy: 0.7127 - loss: 1.1502 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1649/Unknown  194s 93ms/step - categorical_accuracy: 0.7128 - loss: 1.1501 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1650/Unknown  194s 93ms/step - categorical_accuracy: 0.7128 - loss: 1.1500 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1651/Unknown  194s 93ms/step - categorical_accuracy: 0.7128 - loss: 1.1499 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1652/Unknown  194s 93ms/step - categorical_accuracy: 0.7128 - loss: 1.1497 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1653/Unknown  194s 93ms/step - categorical_accuracy: 0.7129 - loss: 1.1496 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1654/Unknown  194s 93ms/step - categorical_accuracy: 0.7129 - loss: 1.1495 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1655/Unknown  194s 93ms/step - categorical_accuracy: 0.7129 - loss: 1.1494 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1656/Unknown  194s 93ms/step - categorical_accuracy: 0.7129 - loss: 1.1493 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1657/Unknown  194s 93ms/step - categorical_accuracy: 0.7130 - loss: 1.1491 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1658/Unknown  195s 93ms/step - categorical_accuracy: 0.7130 - loss: 1.1490 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1659/Unknown  195s 93ms/step - categorical_accuracy: 0.7130 - loss: 1.1489 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1660/Unknown  195s 93ms/step - categorical_accuracy: 0.7130 - loss: 1.1488 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1661/Unknown  195s 93ms/step - categorical_accuracy: 0.7130 - loss: 1.1487 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1662/Unknown  195s 93ms/step - categorical_accuracy: 0.7131 - loss: 1.1486 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1663/Unknown  195s 93ms/step - categorical_accuracy: 0.7131 - loss: 1.1484 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1664/Unknown  195s 93ms/step - categorical_accuracy: 0.7131 - loss: 1.1483 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1665/Unknown  195s 93ms/step - categorical_accuracy: 0.7131 - loss: 1.1482 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1666/Unknown  195s 93ms/step - categorical_accuracy: 0.7132 - loss: 1.1481 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1667/Unknown  195s 93ms/step - categorical_accuracy: 0.7132 - loss: 1.1480 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1668/Unknown  195s 93ms/step - categorical_accuracy: 0.7132 - loss: 1.1479 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1669/Unknown  195s 93ms/step - categorical_accuracy: 0.7132 - loss: 1.1477 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1670/Unknown  196s 93ms/step - categorical_accuracy: 0.7132 - loss: 1.1476 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1671/Unknown  196s 93ms/step - categorical_accuracy: 0.7133 - loss: 1.1475 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1672/Unknown  196s 93ms/step - categorical_accuracy: 0.7133 - loss: 1.1474 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1673/Unknown  196s 93ms/step - categorical_accuracy: 0.7133 - loss: 1.1473 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1674/Unknown  196s 93ms/step - categorical_accuracy: 0.7133 - loss: 1.1471 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1675/Unknown  196s 93ms/step - categorical_accuracy: 0.7134 - loss: 1.1470 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1676/Unknown  196s 93ms/step - categorical_accuracy: 0.7134 - loss: 1.1469 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1677/Unknown  196s 93ms/step - categorical_accuracy: 0.7134 - loss: 1.1468 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1678/Unknown  196s 93ms/step - categorical_accuracy: 0.7134 - loss: 1.1467 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1679/Unknown  196s 93ms/step - categorical_accuracy: 0.7134 - loss: 1.1466 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1680/Unknown  196s 93ms/step - categorical_accuracy: 0.7135 - loss: 1.1464 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1681/Unknown  197s 93ms/step - categorical_accuracy: 0.7135 - loss: 1.1463 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1682/Unknown  197s 93ms/step - categorical_accuracy: 0.7135 - loss: 1.1462 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1683/Unknown  197s 93ms/step - categorical_accuracy: 0.7135 - loss: 1.1461 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1684/Unknown  197s 93ms/step - categorical_accuracy: 0.7136 - loss: 1.1460 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1685/Unknown  197s 93ms/step - categorical_accuracy: 0.7136 - loss: 1.1459 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1686/Unknown  197s 93ms/step - categorical_accuracy: 0.7136 - loss: 1.1457 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1687/Unknown  197s 93ms/step - categorical_accuracy: 0.7136 - loss: 1.1456 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1688/Unknown  197s 93ms/step - categorical_accuracy: 0.7137 - loss: 1.1455 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1689/Unknown  197s 93ms/step - categorical_accuracy: 0.7137 - loss: 1.1454 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1690/Unknown  197s 93ms/step - categorical_accuracy: 0.7137 - loss: 1.1453 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1691/Unknown  197s 93ms/step - categorical_accuracy: 0.7137 - loss: 1.1452 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1692/Unknown  197s 93ms/step - categorical_accuracy: 0.7137 - loss: 1.1450 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1693/Unknown  197s 93ms/step - categorical_accuracy: 0.7138 - loss: 1.1449 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1694/Unknown  197s 93ms/step - categorical_accuracy: 0.7138 - loss: 1.1448 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1695/Unknown  198s 93ms/step - categorical_accuracy: 0.7138 - loss: 1.1447 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1696/Unknown  198s 93ms/step - categorical_accuracy: 0.7138 - loss: 1.1446 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1697/Unknown  198s 93ms/step - categorical_accuracy: 0.7139 - loss: 1.1445 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1698/Unknown  198s 93ms/step - categorical_accuracy: 0.7139 - loss: 1.1444 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1699/Unknown  198s 93ms/step - categorical_accuracy: 0.7139 - loss: 1.1442 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1700/Unknown  198s 93ms/step - categorical_accuracy: 0.7139 - loss: 1.1441 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1701/Unknown  198s 93ms/step - categorical_accuracy: 0.7139 - loss: 1.1440 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1702/Unknown  198s 93ms/step - categorical_accuracy: 0.7140 - loss: 1.1439 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1703/Unknown  198s 93ms/step - categorical_accuracy: 0.7140 - loss: 1.1438 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1704/Unknown  198s 93ms/step - categorical_accuracy: 0.7140 - loss: 1.1437 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1705/Unknown  198s 93ms/step - categorical_accuracy: 0.7140 - loss: 1.1436 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1706/Unknown  198s 93ms/step - categorical_accuracy: 0.7140 - loss: 1.1434 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1707/Unknown  198s 93ms/step - categorical_accuracy: 0.7141 - loss: 1.1433 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1708/Unknown  199s 93ms/step - categorical_accuracy: 0.7141 - loss: 1.1432 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1709/Unknown  199s 93ms/step - categorical_accuracy: 0.7141 - loss: 1.1431 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1710/Unknown  199s 93ms/step - categorical_accuracy: 0.7141 - loss: 1.1430 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1711/Unknown  199s 93ms/step - categorical_accuracy: 0.7142 - loss: 1.1429 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1712/Unknown  199s 93ms/step - categorical_accuracy: 0.7142 - loss: 1.1428 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1713/Unknown  199s 93ms/step - categorical_accuracy: 0.7142 - loss: 1.1426 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1714/Unknown  199s 93ms/step - categorical_accuracy: 0.7142 - loss: 1.1425 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1715/Unknown  199s 93ms/step - categorical_accuracy: 0.7142 - loss: 1.1424 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1716/Unknown  199s 93ms/step - categorical_accuracy: 0.7143 - loss: 1.1423 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1717/Unknown  199s 93ms/step - categorical_accuracy: 0.7143 - loss: 1.1422 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1718/Unknown  199s 93ms/step - categorical_accuracy: 0.7143 - loss: 1.1421 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1719/Unknown  199s 93ms/step - categorical_accuracy: 0.7143 - loss: 1.1420 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1720/Unknown  199s 93ms/step - categorical_accuracy: 0.7144 - loss: 1.1418 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1721/Unknown  199s 93ms/step - categorical_accuracy: 0.7144 - loss: 1.1417 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1722/Unknown  199s 93ms/step - categorical_accuracy: 0.7144 - loss: 1.1416 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1723/Unknown  200s 93ms/step - categorical_accuracy: 0.7144 - loss: 1.1415 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1724/Unknown  200s 93ms/step - categorical_accuracy: 0.7144 - loss: 1.1414 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1725/Unknown  200s 93ms/step - categorical_accuracy: 0.7145 - loss: 1.1413 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1726/Unknown  200s 93ms/step - categorical_accuracy: 0.7145 - loss: 1.1412 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1727/Unknown  200s 93ms/step - categorical_accuracy: 0.7145 - loss: 1.1411 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1728/Unknown  200s 93ms/step - categorical_accuracy: 0.7145 - loss: 1.1409 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1729/Unknown  200s 93ms/step - categorical_accuracy: 0.7146 - loss: 1.1408 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1730/Unknown  200s 93ms/step - categorical_accuracy: 0.7146 - loss: 1.1407 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1731/Unknown  200s 93ms/step - categorical_accuracy: 0.7146 - loss: 1.1406 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1732/Unknown  200s 93ms/step - categorical_accuracy: 0.7146 - loss: 1.1405 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1733/Unknown  200s 93ms/step - categorical_accuracy: 0.7146 - loss: 1.1404 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1734/Unknown  200s 93ms/step - categorical_accuracy: 0.7147 - loss: 1.1403 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1735/Unknown  200s 93ms/step - categorical_accuracy: 0.7147 - loss: 1.1402 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1736/Unknown  200s 93ms/step - categorical_accuracy: 0.7147 - loss: 1.1400 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1737/Unknown  201s 93ms/step - categorical_accuracy: 0.7147 - loss: 1.1399 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1738/Unknown  201s 93ms/step - categorical_accuracy: 0.7147 - loss: 1.1398 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1739/Unknown  201s 93ms/step - categorical_accuracy: 0.7148 - loss: 1.1397 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1740/Unknown  201s 93ms/step - categorical_accuracy: 0.7148 - loss: 1.1396 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1741/Unknown  201s 93ms/step - categorical_accuracy: 0.7148 - loss: 1.1395 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1742/Unknown  201s 93ms/step - categorical_accuracy: 0.7148 - loss: 1.1394 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1743/Unknown  201s 93ms/step - categorical_accuracy: 0.7149 - loss: 1.1393 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1744/Unknown  201s 93ms/step - categorical_accuracy: 0.7149 - loss: 1.1392 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1745/Unknown  201s 93ms/step - categorical_accuracy: 0.7149 - loss: 1.1390 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1746/Unknown  201s 93ms/step - categorical_accuracy: 0.7149 - loss: 1.1389 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1747/Unknown  201s 93ms/step - categorical_accuracy: 0.7149 - loss: 1.1388 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1748/Unknown  201s 93ms/step - categorical_accuracy: 0.7150 - loss: 1.1387 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1749/Unknown  201s 93ms/step - categorical_accuracy: 0.7150 - loss: 1.1386 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1750/Unknown  202s 92ms/step - categorical_accuracy: 0.7150 - loss: 1.1385 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1751/Unknown  202s 92ms/step - categorical_accuracy: 0.7150 - loss: 1.1384 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1752/Unknown  202s 92ms/step - categorical_accuracy: 0.7150 - loss: 1.1383 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1753/Unknown  202s 92ms/step - categorical_accuracy: 0.7151 - loss: 1.1382 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1754/Unknown  202s 92ms/step - categorical_accuracy: 0.7151 - loss: 1.1381 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1755/Unknown  202s 92ms/step - categorical_accuracy: 0.7151 - loss: 1.1379 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1756/Unknown  202s 92ms/step - categorical_accuracy: 0.7151 - loss: 1.1378 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1757/Unknown  202s 92ms/step - categorical_accuracy: 0.7151 - loss: 1.1377 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1758/Unknown  202s 92ms/step - categorical_accuracy: 0.7152 - loss: 1.1376 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1759/Unknown  202s 92ms/step - categorical_accuracy: 0.7152 - loss: 1.1375 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1760/Unknown  202s 92ms/step - categorical_accuracy: 0.7152 - loss: 1.1374 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1761/Unknown  202s 92ms/step - categorical_accuracy: 0.7152 - loss: 1.1373 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1762/Unknown  203s 92ms/step - categorical_accuracy: 0.7153 - loss: 1.1372 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1763/Unknown  203s 92ms/step - categorical_accuracy: 0.7153 - loss: 1.1371 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1764/Unknown  203s 92ms/step - categorical_accuracy: 0.7153 - loss: 1.1370 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1765/Unknown  203s 92ms/step - categorical_accuracy: 0.7153 - loss: 1.1368 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1766/Unknown  203s 92ms/step - categorical_accuracy: 0.7153 - loss: 1.1367 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1767/Unknown  203s 92ms/step - categorical_accuracy: 0.7154 - loss: 1.1366 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1768/Unknown  203s 92ms/step - categorical_accuracy: 0.7154 - loss: 1.1365 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1769/Unknown  203s 92ms/step - categorical_accuracy: 0.7154 - loss: 1.1364 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1770/Unknown  203s 92ms/step - categorical_accuracy: 0.7154 - loss: 1.1363 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1771/Unknown  203s 92ms/step - categorical_accuracy: 0.7154 - loss: 1.1362 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1772/Unknown  203s 92ms/step - categorical_accuracy: 0.7155 - loss: 1.1361 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1773/Unknown  203s 92ms/step - categorical_accuracy: 0.7155 - loss: 1.1360 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1774/Unknown  203s 92ms/step - categorical_accuracy: 0.7155 - loss: 1.1359 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1775/Unknown  204s 92ms/step - categorical_accuracy: 0.7155 - loss: 1.1358 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1776/Unknown  204s 92ms/step - categorical_accuracy: 0.7155 - loss: 1.1357 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1777/Unknown  204s 92ms/step - categorical_accuracy: 0.7156 - loss: 1.1355 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1778/Unknown  204s 92ms/step - categorical_accuracy: 0.7156 - loss: 1.1354 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1779/Unknown  204s 92ms/step - categorical_accuracy: 0.7156 - loss: 1.1353 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1780/Unknown  204s 92ms/step - categorical_accuracy: 0.7156 - loss: 1.1352 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1781/Unknown  204s 92ms/step - categorical_accuracy: 0.7156 - loss: 1.1351 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1782/Unknown  204s 92ms/step - categorical_accuracy: 0.7157 - loss: 1.1350 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1783/Unknown  204s 92ms/step - categorical_accuracy: 0.7157 - loss: 1.1349 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1784/Unknown  204s 92ms/step - categorical_accuracy: 0.7157 - loss: 1.1348 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1785/Unknown  204s 92ms/step - categorical_accuracy: 0.7157 - loss: 1.1347 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1786/Unknown  204s 92ms/step - categorical_accuracy: 0.7157 - loss: 1.1346 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1787/Unknown  205s 92ms/step - categorical_accuracy: 0.7158 - loss: 1.1345 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1788/Unknown  205s 92ms/step - categorical_accuracy: 0.7158 - loss: 1.1344 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1789/Unknown  205s 92ms/step - categorical_accuracy: 0.7158 - loss: 1.1343 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1790/Unknown  205s 92ms/step - categorical_accuracy: 0.7158 - loss: 1.1341 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1791/Unknown  205s 92ms/step - categorical_accuracy: 0.7158 - loss: 1.1340 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1792/Unknown  205s 92ms/step - categorical_accuracy: 0.7159 - loss: 1.1339 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1793/Unknown  205s 92ms/step - categorical_accuracy: 0.7159 - loss: 1.1338 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1794/Unknown  205s 92ms/step - categorical_accuracy: 0.7159 - loss: 1.1337 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1795/Unknown  205s 92ms/step - categorical_accuracy: 0.7159 - loss: 1.1336 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1796/Unknown  205s 92ms/step - categorical_accuracy: 0.7160 - loss: 1.1335 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1797/Unknown  205s 92ms/step - categorical_accuracy: 0.7160 - loss: 1.1334 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1798/Unknown  205s 92ms/step - categorical_accuracy: 0.7160 - loss: 1.1333 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1799/Unknown  206s 92ms/step - categorical_accuracy: 0.7160 - loss: 1.1332 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1800/Unknown  206s 92ms/step - categorical_accuracy: 0.7160 - loss: 1.1331 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1801/Unknown  206s 92ms/step - categorical_accuracy: 0.7161 - loss: 1.1330 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1802/Unknown  206s 92ms/step - categorical_accuracy: 0.7161 - loss: 1.1329 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1803/Unknown  206s 92ms/step - categorical_accuracy: 0.7161 - loss: 1.1328 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1804/Unknown  206s 92ms/step - categorical_accuracy: 0.7161 - loss: 1.1327 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1805/Unknown  206s 92ms/step - categorical_accuracy: 0.7161 - loss: 1.1326 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1806/Unknown  206s 92ms/step - categorical_accuracy: 0.7162 - loss: 1.1325 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1807/Unknown  206s 92ms/step - categorical_accuracy: 0.7162 - loss: 1.1323 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1808/Unknown  206s 92ms/step - categorical_accuracy: 0.7162 - loss: 1.1322 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1809/Unknown  206s 92ms/step - categorical_accuracy: 0.7162 - loss: 1.1321 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1810/Unknown  206s 92ms/step - categorical_accuracy: 0.7162 - loss: 1.1320 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1811/Unknown  206s 92ms/step - categorical_accuracy: 0.7163 - loss: 1.1319 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1812/Unknown  207s 92ms/step - categorical_accuracy: 0.7163 - loss: 1.1318 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1813/Unknown  207s 92ms/step - categorical_accuracy: 0.7163 - loss: 1.1317 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1814/Unknown  207s 92ms/step - categorical_accuracy: 0.7163 - loss: 1.1316 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1815/Unknown  207s 92ms/step - categorical_accuracy: 0.7163 - loss: 1.1315 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1816/Unknown  207s 92ms/step - categorical_accuracy: 0.7164 - loss: 1.1314 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1817/Unknown  207s 92ms/step - categorical_accuracy: 0.7164 - loss: 1.1313 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1818/Unknown  207s 92ms/step - categorical_accuracy: 0.7164 - loss: 1.1312 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1819/Unknown  207s 92ms/step - categorical_accuracy: 0.7164 - loss: 1.1311 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1820/Unknown  207s 92ms/step - categorical_accuracy: 0.7164 - loss: 1.1310 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1821/Unknown  207s 92ms/step - categorical_accuracy: 0.7165 - loss: 1.1309 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1822/Unknown  207s 92ms/step - categorical_accuracy: 0.7165 - loss: 1.1308 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1823/Unknown  207s 92ms/step - categorical_accuracy: 0.7165 - loss: 1.1307 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1824/Unknown  208s 92ms/step - categorical_accuracy: 0.7165 - loss: 1.1306 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1825/Unknown  208s 92ms/step - categorical_accuracy: 0.7165 - loss: 1.1305 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1826/Unknown  208s 92ms/step - categorical_accuracy: 0.7166 - loss: 1.1304 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1827/Unknown  208s 92ms/step - categorical_accuracy: 0.7166 - loss: 1.1303 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1828/Unknown  208s 92ms/step - categorical_accuracy: 0.7166 - loss: 1.1302 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1829/Unknown  208s 92ms/step - categorical_accuracy: 0.7166 - loss: 1.1300 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1830/Unknown  208s 92ms/step - categorical_accuracy: 0.7166 - loss: 1.1299 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1831/Unknown  208s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1298 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1832/Unknown  208s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1297 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1833/Unknown  208s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1296 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1834/Unknown  208s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1295 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1835/Unknown  208s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1294 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1836/Unknown  208s 92ms/step - categorical_accuracy: 0.7167 - loss: 1.1293 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1837/Unknown  209s 92ms/step - categorical_accuracy: 0.7168 - loss: 1.1292 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1838/Unknown  209s 92ms/step - categorical_accuracy: 0.7168 - loss: 1.1291 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1839/Unknown  209s 92ms/step - categorical_accuracy: 0.7168 - loss: 1.1290 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1840/Unknown  209s 92ms/step - categorical_accuracy: 0.7168 - loss: 1.1289 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1841/Unknown  209s 92ms/step - categorical_accuracy: 0.7168 - loss: 1.1288 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1842/Unknown  209s 92ms/step - categorical_accuracy: 0.7169 - loss: 1.1287 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1843/Unknown  209s 92ms/step - categorical_accuracy: 0.7169 - loss: 1.1286 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1844/Unknown  209s 92ms/step - categorical_accuracy: 0.7169 - loss: 1.1285 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1845/Unknown  209s 92ms/step - categorical_accuracy: 0.7169 - loss: 1.1284 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1846/Unknown  209s 92ms/step - categorical_accuracy: 0.7169 - loss: 1.1283 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1847/Unknown  209s 92ms/step - categorical_accuracy: 0.7170 - loss: 1.1282 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1848/Unknown  209s 92ms/step - categorical_accuracy: 0.7170 - loss: 1.1281 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1849/Unknown  209s 92ms/step - categorical_accuracy: 0.7170 - loss: 1.1280 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1850/Unknown  210s 92ms/step - categorical_accuracy: 0.7170 - loss: 1.1279 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1851/Unknown  210s 92ms/step - categorical_accuracy: 0.7170 - loss: 1.1278 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1852/Unknown  210s 92ms/step - categorical_accuracy: 0.7171 - loss: 1.1277 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1853/Unknown  210s 92ms/step - categorical_accuracy: 0.7171 - loss: 1.1276 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1854/Unknown  210s 92ms/step - categorical_accuracy: 0.7171 - loss: 1.1275 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1855/Unknown  210s 92ms/step - categorical_accuracy: 0.7171 - loss: 1.1274 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1856/Unknown  210s 92ms/step - categorical_accuracy: 0.7171 - loss: 1.1273 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1857/Unknown  210s 92ms/step - categorical_accuracy: 0.7172 - loss: 1.1272 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1858/Unknown  210s 92ms/step - categorical_accuracy: 0.7172 - loss: 1.1271 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1859/Unknown  210s 92ms/step - categorical_accuracy: 0.7172 - loss: 1.1270 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1860/Unknown  210s 92ms/step - categorical_accuracy: 0.7172 - loss: 1.1269 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1861/Unknown  210s 92ms/step - categorical_accuracy: 0.7172 - loss: 1.1268 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1862/Unknown  210s 92ms/step - categorical_accuracy: 0.7173 - loss: 1.1267 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1863/Unknown  210s 92ms/step - categorical_accuracy: 0.7173 - loss: 1.1266 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1864/Unknown  210s 92ms/step - categorical_accuracy: 0.7173 - loss: 1.1265 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1866/Unknown  211s 92ms/step - categorical_accuracy: 0.7173 - loss: 1.1263 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1865/Unknown  211s 92ms/step - categorical_accuracy: 0.7173 - loss: 1.1263 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1867/Unknown  211s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1262 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1869/Unknown  211s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1260 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1868/Unknown  211s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1260 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1870/Unknown  211s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1259 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1871/Unknown  211s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1258 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1872/Unknown  211s 92ms/step - categorical_accuracy: 0.7174 - loss: 1.1257 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1873/Unknown  211s 92ms/step - categorical_accuracy: 0.7175 - loss: 1.1256 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1874/Unknown  211s 92ms/step - categorical_accuracy: 0.7175 - loss: 1.1255 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1875/Unknown  211s 91ms/step - categorical_accuracy: 0.7175 - loss: 1.1254 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1876/Unknown  211s 91ms/step - categorical_accuracy: 0.7175 - loss: 1.1253 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1877/Unknown  211s 91ms/step - categorical_accuracy: 0.7175 - loss: 1.1252 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1878/Unknown  211s 91ms/step - categorical_accuracy: 0.7176 - loss: 1.1251 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1879/Unknown  212s 91ms/step - categorical_accuracy: 0.7176 - loss: 1.1250 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1880/Unknown  212s 91ms/step - categorical_accuracy: 0.7176 - loss: 1.1249 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1881/Unknown  212s 91ms/step - categorical_accuracy: 0.7176 - loss: 1.1248 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1882/Unknown  212s 91ms/step - categorical_accuracy: 0.7176 - loss: 1.1247 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1883/Unknown  212s 91ms/step - categorical_accuracy: 0.7177 - loss: 1.1246 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1884/Unknown  212s 91ms/step - categorical_accuracy: 0.7177 - loss: 1.1245 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1885/Unknown  212s 91ms/step - categorical_accuracy: 0.7177 - loss: 1.1244 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1886/Unknown  212s 91ms/step - categorical_accuracy: 0.7177 - loss: 1.1243 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1887/Unknown  212s 91ms/step - categorical_accuracy: 0.7177 - loss: 1.1242 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1888/Unknown  212s 91ms/step - categorical_accuracy: 0.7178 - loss: 1.1241 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1889/Unknown  212s 91ms/step - categorical_accuracy: 0.7178 - loss: 1.1240 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1890/Unknown  212s 91ms/step - categorical_accuracy: 0.7178 - loss: 1.1239 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1891/Unknown  212s 91ms/step - categorical_accuracy: 0.7178 - loss: 1.1238 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1892/Unknown  213s 91ms/step - categorical_accuracy: 0.7178 - loss: 1.1237 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1893/Unknown  213s 91ms/step - categorical_accuracy: 0.7178 - loss: 1.1236 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1894/Unknown  213s 91ms/step - categorical_accuracy: 0.7179 - loss: 1.1235 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1895/Unknown  213s 91ms/step - categorical_accuracy: 0.7179 - loss: 1.1234 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1896/Unknown  213s 91ms/step - categorical_accuracy: 0.7179 - loss: 1.1233 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1897/Unknown  213s 91ms/step - categorical_accuracy: 0.7179 - loss: 1.1232 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1898/Unknown  213s 91ms/step - categorical_accuracy: 0.7179 - loss: 1.1231 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1899/Unknown  213s 91ms/step - categorical_accuracy: 0.7180 - loss: 1.1230 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1900/Unknown  213s 91ms/step - categorical_accuracy: 0.7180 - loss: 1.1229 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1901/Unknown  213s 91ms/step - categorical_accuracy: 0.7180 - loss: 1.1228 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1902/Unknown  213s 91ms/step - categorical_accuracy: 0.7180 - loss: 1.1227 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1903/Unknown  213s 91ms/step - categorical_accuracy: 0.7180 - loss: 1.1226 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1904/Unknown  213s 91ms/step - categorical_accuracy: 0.7181 - loss: 1.1225 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1905/Unknown  214s 91ms/step - categorical_accuracy: 0.7181 - loss: 1.1224 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1906/Unknown  214s 91ms/step - categorical_accuracy: 0.7181 - loss: 1.1223 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1907/Unknown  214s 91ms/step - categorical_accuracy: 0.7181 - loss: 1.1222 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1908/Unknown  214s 91ms/step - categorical_accuracy: 0.7181 - loss: 1.1221 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1909/Unknown  214s 91ms/step - categorical_accuracy: 0.7182 - loss: 1.1220 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1910/Unknown  214s 91ms/step - categorical_accuracy: 0.7182 - loss: 1.1219 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1911/Unknown  214s 91ms/step - categorical_accuracy: 0.7182 - loss: 1.1218 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1912/Unknown  214s 91ms/step - categorical_accuracy: 0.7182 - loss: 1.1217 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1913/Unknown  214s 91ms/step - categorical_accuracy: 0.7182 - loss: 1.1216 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1914/Unknown  214s 91ms/step - categorical_accuracy: 0.7182 - loss: 1.1215 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1915/Unknown  214s 91ms/step - categorical_accuracy: 0.7183 - loss: 1.1214 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1916/Unknown  214s 91ms/step - categorical_accuracy: 0.7183 - loss: 1.1213 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1917/Unknown  214s 91ms/step - categorical_accuracy: 0.7183 - loss: 1.1212 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1918/Unknown  215s 91ms/step - categorical_accuracy: 0.7183 - loss: 1.1211 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1919/Unknown  215s 91ms/step - categorical_accuracy: 0.7183 - loss: 1.1210 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1920/Unknown  215s 91ms/step - categorical_accuracy: 0.7184 - loss: 1.1209 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1921/Unknown  215s 91ms/step - categorical_accuracy: 0.7184 - loss: 1.1208 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1922/Unknown  215s 91ms/step - categorical_accuracy: 0.7184 - loss: 1.1207 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1923/Unknown  215s 91ms/step - categorical_accuracy: 0.7184 - loss: 1.1206 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1924/Unknown  215s 91ms/step - categorical_accuracy: 0.7184 - loss: 1.1205 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1925/Unknown  215s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1204 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1926/Unknown  215s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1203 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1927/Unknown  215s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1202 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1928/Unknown  216s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1201 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1929/Unknown  216s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1200 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1930/Unknown  216s 91ms/step - categorical_accuracy: 0.7185 - loss: 1.1199 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1931/Unknown  216s 91ms/step - categorical_accuracy: 0.7186 - loss: 1.1198 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1932/Unknown  216s 91ms/step - categorical_accuracy: 0.7186 - loss: 1.1197 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1933/Unknown  216s 91ms/step - categorical_accuracy: 0.7186 - loss: 1.1196 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1934/Unknown  216s 91ms/step - categorical_accuracy: 0.7186 - loss: 1.1195 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1935/Unknown  216s 91ms/step - categorical_accuracy: 0.7186 - loss: 1.1194 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1936/Unknown  216s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1193 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1937/Unknown  216s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1192 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1938/Unknown  216s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1191 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1939/Unknown  216s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1190 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1940/Unknown  217s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1190 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1941/Unknown  217s 91ms/step - categorical_accuracy: 0.7187 - loss: 1.1189 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1942/Unknown  217s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1188 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1943/Unknown  217s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1187 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1944/Unknown  217s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1186 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1945/Unknown  217s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1185 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1946/Unknown  217s 91ms/step - categorical_accuracy: 0.7188 - loss: 1.1184 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1947/Unknown  217s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1183 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1948/Unknown  217s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1182 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1949/Unknown  217s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1181 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1950/Unknown  217s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1180 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1951/Unknown  217s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1179 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1952/Unknown  217s 91ms/step - categorical_accuracy: 0.7189 - loss: 1.1178 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1953/Unknown  218s 91ms/step - categorical_accuracy: 0.7190 - loss: 1.1177 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1954/Unknown  218s 91ms/step - categorical_accuracy: 0.7190 - loss: 1.1176 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1955/Unknown  218s 91ms/step - categorical_accuracy: 0.7190 - loss: 1.1175 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1956/Unknown  218s 91ms/step - categorical_accuracy: 0.7190 - loss: 1.1174 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1957/Unknown  218s 91ms/step - categorical_accuracy: 0.7190 - loss: 1.1173 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1958/Unknown  218s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1172 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1959/Unknown  218s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1171 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1960/Unknown  218s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1170 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1961/Unknown  218s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1169 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1962/Unknown  218s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1168 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1963/Unknown  218s 91ms/step - categorical_accuracy: 0.7191 - loss: 1.1168 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1964/Unknown  218s 91ms/step - categorical_accuracy: 0.7192 - loss: 1.1167 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1965/Unknown  218s 91ms/step - categorical_accuracy: 0.7192 - loss: 1.1166 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1966/Unknown  219s 91ms/step - categorical_accuracy: 0.7192 - loss: 1.1165 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1967/Unknown  219s 91ms/step - categorical_accuracy: 0.7192 - loss: 1.1164 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1968/Unknown  219s 91ms/step - categorical_accuracy: 0.7192 - loss: 1.1163 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1969/Unknown  219s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1162 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1970/Unknown  219s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1161 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1971/Unknown  219s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1160 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1972/Unknown  219s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1159 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1973/Unknown  219s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1158 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1974/Unknown  219s 91ms/step - categorical_accuracy: 0.7193 - loss: 1.1157 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1975/Unknown  219s 91ms/step - categorical_accuracy: 0.7194 - loss: 1.1156 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1976/Unknown  219s 91ms/step - categorical_accuracy: 0.7194 - loss: 1.1155 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1977/Unknown  219s 91ms/step - categorical_accuracy: 0.7194 - loss: 1.1154 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1978/Unknown  220s 91ms/step - categorical_accuracy: 0.7194 - loss: 1.1153 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1979/Unknown  220s 91ms/step - categorical_accuracy: 0.7194 - loss: 1.1152 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1980/Unknown  220s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1151 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1981/Unknown  220s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1151 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1982/Unknown  220s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1150 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1983/Unknown  220s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1149 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1984/Unknown  220s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1148 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1985/Unknown  220s 91ms/step - categorical_accuracy: 0.7195 - loss: 1.1147 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1986/Unknown  220s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1146 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1987/Unknown  220s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1145 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1988/Unknown  220s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1144 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1989/Unknown  220s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1143 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1990/Unknown  220s 91ms/step - categorical_accuracy: 0.7196 - loss: 1.1142 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1991/Unknown  221s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1141 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1992/Unknown  221s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1140 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1993/Unknown  221s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1139 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1994/Unknown  221s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1138 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1995/Unknown  221s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1137 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1996/Unknown  221s 91ms/step - categorical_accuracy: 0.7197 - loss: 1.1137 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1997/Unknown  221s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1136 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1998/Unknown  221s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1135 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1999/Unknown  221s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1134 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   2000/Unknown  221s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1133 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   2001/Unknown  221s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1132 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   2002/Unknown  221s 91ms/step - categorical_accuracy: 0.7198 - loss: 1.1131 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   2003/Unknown  221s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1130 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   2004/Unknown  222s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1129 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   2005/Unknown  222s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1128 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   2006/Unknown  222s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1127 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   2007/Unknown  222s 91ms/step - categorical_accuracy: 0.7199 - loss: 1.1126 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   2008/Unknown  222s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1125 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   2009/Unknown  222s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1125 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   2010/Unknown  222s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1124 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   2011/Unknown  222s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1123 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   2012/Unknown  222s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1122 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   2013/Unknown  222s 91ms/step - categorical_accuracy: 0.7200 - loss: 1.1121 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   2014/Unknown  222s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1120 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   2015/Unknown  222s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1119 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   2016/Unknown  222s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1118 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   2017/Unknown  222s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1117 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   2018/Unknown  222s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1116 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   2019/Unknown  222s 91ms/step - categorical_accuracy: 0.7201 - loss: 1.1115 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   2020/Unknown  223s 91ms/step - categorical_accuracy: 0.7202 - loss: 1.1115 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   2021/Unknown  223s 91ms/step - categorical_accuracy: 0.7202 - loss: 1.1114 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   2022/Unknown  223s 91ms/step - categorical_accuracy: 0.7202 - loss: 1.1113 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   2023/Unknown  223s 91ms/step - categorical_accuracy: 0.7202 - loss: 1.1112 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   2024/Unknown  223s 91ms/step - categorical_accuracy: 0.7202 - loss: 1.1111 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   2025/Unknown  223s 90ms/step - categorical_accuracy: 0.7202 - loss: 1.1110 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   2026/Unknown  223s 90ms/step - categorical_accuracy: 0.7203 - loss: 1.1109 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   2027/Unknown  223s 90ms/step - categorical_accuracy: 0.7203 - loss: 1.1108 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   2028/Unknown  223s 90ms/step - categorical_accuracy: 0.7203 - loss: 1.1107 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   2029/Unknown  223s 90ms/step - categorical_accuracy: 0.7203 - loss: 1.1106 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   2030/Unknown  223s 90ms/step - categorical_accuracy: 0.7203 - loss: 1.1106 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   2031/Unknown  223s 90ms/step - categorical_accuracy: 0.7203 - loss: 1.1105 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   2032/Unknown  223s 90ms/step - categorical_accuracy: 0.7204 - loss: 1.1104 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   2033/Unknown  224s 90ms/step - categorical_accuracy: 0.7204 - loss: 1.1103 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   2034/Unknown  224s 90ms/step - categorical_accuracy: 0.7204 - loss: 1.1102 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   2035/Unknown  224s 90ms/step - categorical_accuracy: 0.7204 - loss: 1.1101 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   2036/Unknown  224s 90ms/step - categorical_accuracy: 0.7204 - loss: 1.1100 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   2037/Unknown  224s 90ms/step - categorical_accuracy: 0.7205 - loss: 1.1099 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   2038/Unknown  224s 90ms/step - categorical_accuracy: 0.7205 - loss: 1.1098 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   2039/Unknown  224s 90ms/step - categorical_accuracy: 0.7205 - loss: 1.1098 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   2040/Unknown  224s 90ms/step - categorical_accuracy: 0.7205 - loss: 1.1097 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   2041/Unknown  224s 90ms/step - categorical_accuracy: 0.7205 - loss: 1.1096 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   2042/Unknown  224s 90ms/step - categorical_accuracy: 0.7205 - loss: 1.1095 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   2043/Unknown  224s 90ms/step - categorical_accuracy: 0.7206 - loss: 1.1094 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   2044/Unknown  224s 90ms/step - categorical_accuracy: 0.7206 - loss: 1.1093 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   2045/Unknown  224s 90ms/step - categorical_accuracy: 0.7206 - loss: 1.1092 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   2046/Unknown  225s 90ms/step - categorical_accuracy: 0.7206 - loss: 1.1091 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   2047/Unknown  225s 90ms/step - categorical_accuracy: 0.7206 - loss: 1.1090 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   2048/Unknown  225s 90ms/step - categorical_accuracy: 0.7206 - loss: 1.1089 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   2049/Unknown  225s 90ms/step - categorical_accuracy: 0.7207 - loss: 1.1089 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   2050/Unknown  225s 90ms/step - categorical_accuracy: 0.7207 - loss: 1.1088 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   2051/Unknown  225s 90ms/step - categorical_accuracy: 0.7207 - loss: 1.1087 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   2052/Unknown  225s 90ms/step - categorical_accuracy: 0.7207 - loss: 1.1086 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   2053/Unknown  225s 90ms/step - categorical_accuracy: 0.7207 - loss: 1.1085 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   2054/Unknown  225s 90ms/step - categorical_accuracy: 0.7207 - loss: 1.1084 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   2055/Unknown  225s 90ms/step - categorical_accuracy: 0.7208 - loss: 1.1083 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   2056/Unknown  225s 90ms/step - categorical_accuracy: 0.7208 - loss: 1.1082 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2057/Unknown  225s 90ms/step - categorical_accuracy: 0.7208 - loss: 1.1081 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2058/Unknown  225s 90ms/step - categorical_accuracy: 0.7208 - loss: 1.1081 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2059/Unknown  226s 90ms/step - categorical_accuracy: 0.7208 - loss: 1.1080 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2060/Unknown  226s 90ms/step - categorical_accuracy: 0.7208 - loss: 1.1079 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2061/Unknown  226s 90ms/step - categorical_accuracy: 0.7209 - loss: 1.1078 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2062/Unknown  226s 90ms/step - categorical_accuracy: 0.7209 - loss: 1.1077 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   2063/Unknown  226s 90ms/step - categorical_accuracy: 0.7209 - loss: 1.1076 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   2064/Unknown  226s 90ms/step - categorical_accuracy: 0.7209 - loss: 1.1075 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   2065/Unknown  226s 90ms/step - categorical_accuracy: 0.7209 - loss: 1.1074 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   2066/Unknown  226s 90ms/step - categorical_accuracy: 0.7209 - loss: 1.1074 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   2067/Unknown  226s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.1073 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2068/Unknown  226s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.1072 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2069/Unknown  226s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.1071 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2070/Unknown  226s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.1070 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2071/Unknown  226s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.1069 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2072/Unknown  227s 90ms/step - categorical_accuracy: 0.7210 - loss: 1.1068 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   2073/Unknown  227s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.1067 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   2074/Unknown  227s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.1067 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   2075/Unknown  227s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.1066 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   2076/Unknown  227s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.1065 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   2077/Unknown  227s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.1064 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2078/Unknown  227s 90ms/step - categorical_accuracy: 0.7211 - loss: 1.1063 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2079/Unknown  227s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.1062 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2080/Unknown  227s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.1061 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2081/Unknown  227s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.1060 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2082/Unknown  227s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.1060 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2083/Unknown  228s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.1059 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2084/Unknown  228s 90ms/step - categorical_accuracy: 0.7212 - loss: 1.1058 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2085/Unknown  228s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.1057 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2086/Unknown  228s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.1056 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2087/Unknown  228s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.1055 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2088/Unknown  228s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.1054 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   2089/Unknown  228s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.1053 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   2090/Unknown  228s 90ms/step - categorical_accuracy: 0.7213 - loss: 1.1053 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   2091/Unknown  228s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.1052 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   2092/Unknown  228s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.1051 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   2093/Unknown  228s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.1050 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2094/Unknown  228s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.1049 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2095/Unknown  229s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.1048 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2096/Unknown  229s 90ms/step - categorical_accuracy: 0.7214 - loss: 1.1047 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2097/Unknown  229s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.1047 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2098/Unknown  229s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.1046 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2099/Unknown  229s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.1045 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2100/Unknown  229s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.1044 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2101/Unknown  229s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.1043 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2102/Unknown  229s 90ms/step - categorical_accuracy: 0.7215 - loss: 1.1042 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2103/Unknown  229s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.1041 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2104/Unknown  229s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.1040 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2105/Unknown  229s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.1040 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2106/Unknown  229s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.1039 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2107/Unknown  229s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.1038 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2108/Unknown  230s 90ms/step - categorical_accuracy: 0.7216 - loss: 1.1037 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2109/Unknown  230s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.1036 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2110/Unknown  230s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.1035 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2111/Unknown  230s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.1034 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2112/Unknown  230s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.1034 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2113/Unknown  230s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.1033 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2114/Unknown  230s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.1032 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2115/Unknown  230s 90ms/step - categorical_accuracy: 0.7217 - loss: 1.1031 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2116/Unknown  230s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.1030 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2117/Unknown  230s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.1029 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2118/Unknown  230s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.1028 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2119/Unknown  230s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.1028 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2120/Unknown  230s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.1027 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2121/Unknown  230s 90ms/step - categorical_accuracy: 0.7218 - loss: 1.1026 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2122/Unknown  231s 90ms/step - categorical_accuracy: 0.7219 - loss: 1.1025 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2123/Unknown  231s 90ms/step - categorical_accuracy: 0.7219 - loss: 1.1023 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2124/Unknown  231s 90ms/step - categorical_accuracy: 0.7219 - loss: 1.1023 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```
/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/models/functional.py:225: UserWarning: The structure of `inputs` doesn't match the expected structure: ['keras_tensor_222']. Received: the structure of inputs=*
  warnings.warn(

/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/backend/jax/core.py:76: UserWarning: Explicitly requested dtype int64 requested in asarray is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return jnp.asarray(x, dtype=dtype)


```
</div>
 2124/2124 ━━━━━━━━━━━━━━━━━━━━ 281s 114ms/step - categorical_accuracy: 0.7219 - loss: 1.1023 - mean_io_u: 0.0916 - val_categorical_accuracy: 0.8199 - val_loss: 0.5963 - val_mean_io_u: 0.3422





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7f4353cec610>

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
preds = ops.expand_dims(ops.argmax(model.predict(images), axis=-1), axis=-1)
masks = ops.expand_dims(ops.argmax(masks, axis=-1), axis=-1)

plot_images_masks(images, masks, preds)
```

<div class="k-default-codeblock">
```
/usr/lib/python3.11/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()

/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/models/functional.py:225: UserWarning: The structure of `inputs` doesn't match the expected structure: ['keras_tensor_222']. Received: the structure of inputs=*
  warnings.warn(

```
</div>
    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 3s 3s/step



    
![png](/home/sachinprasad/projects/keras-io/guides/img/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_32_4.png)
    


Here are some additional tips for using the KerasHub DeepLabv3 model:

- The model can be trained on a variety of datasets, including the COCO dataset, the
PASCAL VOC dataset, and the Cityscapes dataset.
- The model can be fine-tuned on a custom dataset to improve its performance on a
specific task.
- The model can be used to perform real-time inference on images.
- Also, check out KerasHub's other segmentation models.

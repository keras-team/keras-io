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
  1/Unknown  40s 40s/step - categorical_accuracy: 0.1191 - loss: 3.0568 - mean_io_u: 0.0118


  2/Unknown  62s 22s/step - categorical_accuracy: 0.1355 - loss: 2.9447 - mean_io_u: 0.0151


  3/Unknown  62s 11s/step - categorical_accuracy: 0.1482 - loss: 2.8838 - mean_io_u: 0.0164


  4/Unknown  62s 7s/step - categorical_accuracy: 0.1571 - loss: 2.8481 - mean_io_u: 0.0171 


  5/Unknown  62s 6s/step - categorical_accuracy: 0.1672 - loss: 2.8169 - mean_io_u: 0.0180


  6/Unknown  62s 4s/step - categorical_accuracy: 0.1771 - loss: 2.7875 - mean_io_u: 0.0187


  7/Unknown  62s 4s/step - categorical_accuracy: 0.1880 - loss: 2.7602 - mean_io_u: 0.0195


  8/Unknown  62s 3s/step - categorical_accuracy: 0.1992 - loss: 2.7335 - mean_io_u: 0.0202


  9/Unknown  62s 3s/step - categorical_accuracy: 0.2108 - loss: 2.7103 - mean_io_u: 0.0208


 10/Unknown  62s 2s/step - categorical_accuracy: 0.2230 - loss: 2.6871 - mean_io_u: 0.0215


 11/Unknown  62s 2s/step - categorical_accuracy: 0.2346 - loss: 2.6652 - mean_io_u: 0.0221


 12/Unknown  62s 2s/step - categorical_accuracy: 0.2467 - loss: 2.6421 - mean_io_u: 0.0227


 13/Unknown  62s 2s/step - categorical_accuracy: 0.2590 - loss: 2.6172 - mean_io_u: 0.0234


 14/Unknown  62s 2s/step - categorical_accuracy: 0.2706 - loss: 2.5954 - mean_io_u: 0.0240


 15/Unknown  63s 2s/step - categorical_accuracy: 0.2819 - loss: 2.5732 - mean_io_u: 0.0246


 16/Unknown  63s 2s/step - categorical_accuracy: 0.2928 - loss: 2.5512 - mean_io_u: 0.0251


 17/Unknown  63s 1s/step - categorical_accuracy: 0.3037 - loss: 2.5284 - mean_io_u: 0.0257


 18/Unknown  63s 1s/step - categorical_accuracy: 0.3138 - loss: 2.5063 - mean_io_u: 0.0262


 19/Unknown  63s 1s/step - categorical_accuracy: 0.3232 - loss: 2.4850 - mean_io_u: 0.0266


 20/Unknown  63s 1s/step - categorical_accuracy: 0.3323 - loss: 2.4637 - mean_io_u: 0.0271


 21/Unknown  63s 1s/step - categorical_accuracy: 0.3407 - loss: 2.4438 - mean_io_u: 0.0275


 22/Unknown  63s 1s/step - categorical_accuracy: 0.3487 - loss: 2.4247 - mean_io_u: 0.0278


 23/Unknown  63s 1s/step - categorical_accuracy: 0.3564 - loss: 2.4057 - mean_io_u: 0.0282


 24/Unknown  63s 1s/step - categorical_accuracy: 0.3640 - loss: 2.3866 - mean_io_u: 0.0285


 25/Unknown  63s 984ms/step - categorical_accuracy: 0.3712 - loss: 2.3687 - mean_io_u: 0.0288


 26/Unknown  63s 948ms/step - categorical_accuracy: 0.3780 - loss: 2.3514 - mean_io_u: 0.0291


 27/Unknown  63s 916ms/step - categorical_accuracy: 0.3845 - loss: 2.3348 - mean_io_u: 0.0294


 28/Unknown  64s 884ms/step - categorical_accuracy: 0.3906 - loss: 2.3197 - mean_io_u: 0.0297


 29/Unknown  64s 855ms/step - categorical_accuracy: 0.3963 - loss: 2.3055 - mean_io_u: 0.0299


 30/Unknown  64s 831ms/step - categorical_accuracy: 0.4018 - loss: 2.2919 - mean_io_u: 0.0301


 31/Unknown  64s 806ms/step - categorical_accuracy: 0.4070 - loss: 2.2786 - mean_io_u: 0.0303


 32/Unknown  64s 783ms/step - categorical_accuracy: 0.4120 - loss: 2.2658 - mean_io_u: 0.0305


 33/Unknown  64s 762ms/step - categorical_accuracy: 0.4167 - loss: 2.2535 - mean_io_u: 0.0307


 34/Unknown  64s 742ms/step - categorical_accuracy: 0.4212 - loss: 2.2423 - mean_io_u: 0.0308


 35/Unknown  64s 722ms/step - categorical_accuracy: 0.4256 - loss: 2.2308 - mean_io_u: 0.0309


 36/Unknown  64s 704ms/step - categorical_accuracy: 0.4298 - loss: 2.2197 - mean_io_u: 0.0310


 37/Unknown  64s 687ms/step - categorical_accuracy: 0.4339 - loss: 2.2087 - mean_io_u: 0.0311


 38/Unknown  64s 671ms/step - categorical_accuracy: 0.4379 - loss: 2.1979 - mean_io_u: 0.0312


 39/Unknown  65s 655ms/step - categorical_accuracy: 0.4418 - loss: 2.1874 - mean_io_u: 0.0313


 40/Unknown  65s 640ms/step - categorical_accuracy: 0.4455 - loss: 2.1772 - mean_io_u: 0.0314


 41/Unknown  65s 626ms/step - categorical_accuracy: 0.4492 - loss: 2.1669 - mean_io_u: 0.0314


 42/Unknown  65s 612ms/step - categorical_accuracy: 0.4528 - loss: 2.1570 - mean_io_u: 0.0315


 43/Unknown  65s 602ms/step - categorical_accuracy: 0.4562 - loss: 2.1475 - mean_io_u: 0.0315


 44/Unknown  65s 589ms/step - categorical_accuracy: 0.4594 - loss: 2.1383 - mean_io_u: 0.0316


 45/Unknown  65s 577ms/step - categorical_accuracy: 0.4626 - loss: 2.1292 - mean_io_u: 0.0316


 46/Unknown  65s 566ms/step - categorical_accuracy: 0.4657 - loss: 2.1204 - mean_io_u: 0.0317


 47/Unknown  65s 556ms/step - categorical_accuracy: 0.4687 - loss: 2.1118 - mean_io_u: 0.0317


 48/Unknown  65s 546ms/step - categorical_accuracy: 0.4717 - loss: 2.1032 - mean_io_u: 0.0317


 49/Unknown  65s 537ms/step - categorical_accuracy: 0.4746 - loss: 2.0947 - mean_io_u: 0.0318


 50/Unknown  66s 528ms/step - categorical_accuracy: 0.4774 - loss: 2.0863 - mean_io_u: 0.0318


 51/Unknown  66s 519ms/step - categorical_accuracy: 0.4802 - loss: 2.0782 - mean_io_u: 0.0318


 52/Unknown  66s 511ms/step - categorical_accuracy: 0.4828 - loss: 2.0703 - mean_io_u: 0.0318


 53/Unknown  66s 503ms/step - categorical_accuracy: 0.4853 - loss: 2.0629 - mean_io_u: 0.0318


 54/Unknown  66s 495ms/step - categorical_accuracy: 0.4878 - loss: 2.0554 - mean_io_u: 0.0318


 55/Unknown  66s 488ms/step - categorical_accuracy: 0.4902 - loss: 2.0482 - mean_io_u: 0.0318


 56/Unknown  66s 480ms/step - categorical_accuracy: 0.4926 - loss: 2.0410 - mean_io_u: 0.0319


 57/Unknown  66s 473ms/step - categorical_accuracy: 0.4949 - loss: 2.0342 - mean_io_u: 0.0319


 58/Unknown  66s 467ms/step - categorical_accuracy: 0.4971 - loss: 2.0274 - mean_io_u: 0.0319


 59/Unknown  66s 460ms/step - categorical_accuracy: 0.4993 - loss: 2.0207 - mean_io_u: 0.0319


 60/Unknown  66s 454ms/step - categorical_accuracy: 0.5014 - loss: 2.0141 - mean_io_u: 0.0319


 61/Unknown  67s 448ms/step - categorical_accuracy: 0.5035 - loss: 2.0077 - mean_io_u: 0.0319


 62/Unknown  67s 443ms/step - categorical_accuracy: 0.5056 - loss: 2.0013 - mean_io_u: 0.0319


 63/Unknown  67s 437ms/step - categorical_accuracy: 0.5076 - loss: 1.9949 - mean_io_u: 0.0319


 64/Unknown  67s 432ms/step - categorical_accuracy: 0.5096 - loss: 1.9886 - mean_io_u: 0.0319


 65/Unknown  67s 426ms/step - categorical_accuracy: 0.5116 - loss: 1.9824 - mean_io_u: 0.0319


 66/Unknown  67s 421ms/step - categorical_accuracy: 0.5135 - loss: 1.9764 - mean_io_u: 0.0319


 67/Unknown  67s 417ms/step - categorical_accuracy: 0.5153 - loss: 1.9705 - mean_io_u: 0.0319


 68/Unknown  67s 412ms/step - categorical_accuracy: 0.5171 - loss: 1.9648 - mean_io_u: 0.0319


 69/Unknown  67s 407ms/step - categorical_accuracy: 0.5189 - loss: 1.9591 - mean_io_u: 0.0319


 70/Unknown  67s 403ms/step - categorical_accuracy: 0.5206 - loss: 1.9536 - mean_io_u: 0.0319


 71/Unknown  68s 399ms/step - categorical_accuracy: 0.5223 - loss: 1.9482 - mean_io_u: 0.0319


 72/Unknown  68s 395ms/step - categorical_accuracy: 0.5240 - loss: 1.9428 - mean_io_u: 0.0319


 73/Unknown  68s 391ms/step - categorical_accuracy: 0.5256 - loss: 1.9377 - mean_io_u: 0.0320


 74/Unknown  68s 386ms/step - categorical_accuracy: 0.5271 - loss: 1.9326 - mean_io_u: 0.0320


 75/Unknown  68s 382ms/step - categorical_accuracy: 0.5287 - loss: 1.9276 - mean_io_u: 0.0320


 76/Unknown  68s 378ms/step - categorical_accuracy: 0.5302 - loss: 1.9227 - mean_io_u: 0.0320


 77/Unknown  68s 374ms/step - categorical_accuracy: 0.5316 - loss: 1.9178 - mean_io_u: 0.0320


 78/Unknown  68s 371ms/step - categorical_accuracy: 0.5331 - loss: 1.9130 - mean_io_u: 0.0320


 79/Unknown  68s 367ms/step - categorical_accuracy: 0.5345 - loss: 1.9083 - mean_io_u: 0.0320


 80/Unknown  68s 364ms/step - categorical_accuracy: 0.5359 - loss: 1.9035 - mean_io_u: 0.0320


 81/Unknown  69s 361ms/step - categorical_accuracy: 0.5373 - loss: 1.8987 - mean_io_u: 0.0320


 82/Unknown  69s 358ms/step - categorical_accuracy: 0.5387 - loss: 1.8941 - mean_io_u: 0.0321


 83/Unknown  69s 356ms/step - categorical_accuracy: 0.5400 - loss: 1.8894 - mean_io_u: 0.0321


 84/Unknown  69s 352ms/step - categorical_accuracy: 0.5414 - loss: 1.8848 - mean_io_u: 0.0321


 85/Unknown  69s 349ms/step - categorical_accuracy: 0.5427 - loss: 1.8803 - mean_io_u: 0.0321


 86/Unknown  69s 346ms/step - categorical_accuracy: 0.5440 - loss: 1.8758 - mean_io_u: 0.0321


 87/Unknown  69s 343ms/step - categorical_accuracy: 0.5453 - loss: 1.8714 - mean_io_u: 0.0322


 88/Unknown  69s 341ms/step - categorical_accuracy: 0.5465 - loss: 1.8670 - mean_io_u: 0.0322


 89/Unknown  69s 338ms/step - categorical_accuracy: 0.5477 - loss: 1.8627 - mean_io_u: 0.0322


 90/Unknown  69s 335ms/step - categorical_accuracy: 0.5489 - loss: 1.8584 - mean_io_u: 0.0322


 91/Unknown  70s 333ms/step - categorical_accuracy: 0.5501 - loss: 1.8542 - mean_io_u: 0.0323


 92/Unknown  70s 331ms/step - categorical_accuracy: 0.5513 - loss: 1.8500 - mean_io_u: 0.0323


 93/Unknown  70s 328ms/step - categorical_accuracy: 0.5525 - loss: 1.8458 - mean_io_u: 0.0323


 94/Unknown  70s 326ms/step - categorical_accuracy: 0.5537 - loss: 1.8417 - mean_io_u: 0.0324


 95/Unknown  70s 324ms/step - categorical_accuracy: 0.5548 - loss: 1.8375 - mean_io_u: 0.0324


 96/Unknown  70s 321ms/step - categorical_accuracy: 0.5559 - loss: 1.8334 - mean_io_u: 0.0324


 97/Unknown  70s 319ms/step - categorical_accuracy: 0.5570 - loss: 1.8295 - mean_io_u: 0.0325


 98/Unknown  70s 317ms/step - categorical_accuracy: 0.5581 - loss: 1.8255 - mean_io_u: 0.0325


 99/Unknown  70s 315ms/step - categorical_accuracy: 0.5592 - loss: 1.8217 - mean_io_u: 0.0325


100/Unknown  71s 313ms/step - categorical_accuracy: 0.5602 - loss: 1.8178 - mean_io_u: 0.0326


101/Unknown  71s 311ms/step - categorical_accuracy: 0.5613 - loss: 1.8141 - mean_io_u: 0.0326


102/Unknown  71s 309ms/step - categorical_accuracy: 0.5623 - loss: 1.8104 - mean_io_u: 0.0326


103/Unknown  71s 307ms/step - categorical_accuracy: 0.5633 - loss: 1.8067 - mean_io_u: 0.0327


104/Unknown  71s 305ms/step - categorical_accuracy: 0.5643 - loss: 1.8030 - mean_io_u: 0.0327


105/Unknown  71s 302ms/step - categorical_accuracy: 0.5653 - loss: 1.7993 - mean_io_u: 0.0328


106/Unknown  71s 301ms/step - categorical_accuracy: 0.5663 - loss: 1.7956 - mean_io_u: 0.0328


107/Unknown  71s 299ms/step - categorical_accuracy: 0.5673 - loss: 1.7920 - mean_io_u: 0.0328


108/Unknown  71s 297ms/step - categorical_accuracy: 0.5682 - loss: 1.7885 - mean_io_u: 0.0329


109/Unknown  72s 295ms/step - categorical_accuracy: 0.5692 - loss: 1.7851 - mean_io_u: 0.0329


110/Unknown  72s 294ms/step - categorical_accuracy: 0.5701 - loss: 1.7817 - mean_io_u: 0.0330


111/Unknown  72s 292ms/step - categorical_accuracy: 0.5710 - loss: 1.7784 - mean_io_u: 0.0330


112/Unknown  72s 290ms/step - categorical_accuracy: 0.5719 - loss: 1.7751 - mean_io_u: 0.0330


113/Unknown  72s 289ms/step - categorical_accuracy: 0.5728 - loss: 1.7718 - mean_io_u: 0.0331


114/Unknown  72s 287ms/step - categorical_accuracy: 0.5737 - loss: 1.7685 - mean_io_u: 0.0331


115/Unknown  72s 285ms/step - categorical_accuracy: 0.5746 - loss: 1.7653 - mean_io_u: 0.0331


116/Unknown  72s 283ms/step - categorical_accuracy: 0.5754 - loss: 1.7621 - mean_io_u: 0.0332


117/Unknown  72s 282ms/step - categorical_accuracy: 0.5763 - loss: 1.7590 - mean_io_u: 0.0332


118/Unknown  72s 280ms/step - categorical_accuracy: 0.5771 - loss: 1.7560 - mean_io_u: 0.0332


119/Unknown  73s 279ms/step - categorical_accuracy: 0.5779 - loss: 1.7530 - mean_io_u: 0.0333


120/Unknown  73s 277ms/step - categorical_accuracy: 0.5787 - loss: 1.7500 - mean_io_u: 0.0333


121/Unknown  73s 276ms/step - categorical_accuracy: 0.5795 - loss: 1.7470 - mean_io_u: 0.0333


122/Unknown  73s 274ms/step - categorical_accuracy: 0.5803 - loss: 1.7441 - mean_io_u: 0.0333


123/Unknown  73s 273ms/step - categorical_accuracy: 0.5811 - loss: 1.7412 - mean_io_u: 0.0334


124/Unknown  73s 272ms/step - categorical_accuracy: 0.5819 - loss: 1.7383 - mean_io_u: 0.0334


125/Unknown  73s 270ms/step - categorical_accuracy: 0.5826 - loss: 1.7356 - mean_io_u: 0.0334


126/Unknown  73s 269ms/step - categorical_accuracy: 0.5834 - loss: 1.7328 - mean_io_u: 0.0334


127/Unknown  73s 267ms/step - categorical_accuracy: 0.5841 - loss: 1.7301 - mean_io_u: 0.0334


128/Unknown  73s 266ms/step - categorical_accuracy: 0.5848 - loss: 1.7274 - mean_io_u: 0.0335


129/Unknown  74s 265ms/step - categorical_accuracy: 0.5856 - loss: 1.7247 - mean_io_u: 0.0335


130/Unknown  74s 263ms/step - categorical_accuracy: 0.5863 - loss: 1.7219 - mean_io_u: 0.0335


131/Unknown  74s 262ms/step - categorical_accuracy: 0.5870 - loss: 1.7193 - mean_io_u: 0.0335


132/Unknown  74s 261ms/step - categorical_accuracy: 0.5877 - loss: 1.7166 - mean_io_u: 0.0336


133/Unknown  74s 260ms/step - categorical_accuracy: 0.5884 - loss: 1.7140 - mean_io_u: 0.0336


134/Unknown  74s 259ms/step - categorical_accuracy: 0.5891 - loss: 1.7114 - mean_io_u: 0.0336


135/Unknown  74s 257ms/step - categorical_accuracy: 0.5898 - loss: 1.7088 - mean_io_u: 0.0336


136/Unknown  74s 256ms/step - categorical_accuracy: 0.5905 - loss: 1.7062 - mean_io_u: 0.0337


137/Unknown  74s 255ms/step - categorical_accuracy: 0.5912 - loss: 1.7037 - mean_io_u: 0.0337


138/Unknown  74s 254ms/step - categorical_accuracy: 0.5918 - loss: 1.7012 - mean_io_u: 0.0337


139/Unknown  75s 253ms/step - categorical_accuracy: 0.5925 - loss: 1.6988 - mean_io_u: 0.0338


140/Unknown  75s 252ms/step - categorical_accuracy: 0.5931 - loss: 1.6964 - mean_io_u: 0.0338


141/Unknown  75s 251ms/step - categorical_accuracy: 0.5937 - loss: 1.6940 - mean_io_u: 0.0338


142/Unknown  75s 249ms/step - categorical_accuracy: 0.5944 - loss: 1.6916 - mean_io_u: 0.0338


143/Unknown  75s 248ms/step - categorical_accuracy: 0.5950 - loss: 1.6892 - mean_io_u: 0.0339


144/Unknown  75s 247ms/step - categorical_accuracy: 0.5956 - loss: 1.6869 - mean_io_u: 0.0339


145/Unknown  75s 246ms/step - categorical_accuracy: 0.5962 - loss: 1.6846 - mean_io_u: 0.0339


146/Unknown  75s 245ms/step - categorical_accuracy: 0.5968 - loss: 1.6823 - mean_io_u: 0.0340


147/Unknown  75s 244ms/step - categorical_accuracy: 0.5974 - loss: 1.6801 - mean_io_u: 0.0340


148/Unknown  75s 242ms/step - categorical_accuracy: 0.5980 - loss: 1.6778 - mean_io_u: 0.0340


149/Unknown  75s 241ms/step - categorical_accuracy: 0.5985 - loss: 1.6756 - mean_io_u: 0.0341


150/Unknown  75s 240ms/step - categorical_accuracy: 0.5991 - loss: 1.6734 - mean_io_u: 0.0341


151/Unknown  76s 240ms/step - categorical_accuracy: 0.5997 - loss: 1.6712 - mean_io_u: 0.0341


152/Unknown  76s 239ms/step - categorical_accuracy: 0.6002 - loss: 1.6691 - mean_io_u: 0.0342


153/Unknown  76s 238ms/step - categorical_accuracy: 0.6008 - loss: 1.6669 - mean_io_u: 0.0342


154/Unknown  76s 237ms/step - categorical_accuracy: 0.6013 - loss: 1.6648 - mean_io_u: 0.0342


155/Unknown  76s 236ms/step - categorical_accuracy: 0.6019 - loss: 1.6627 - mean_io_u: 0.0343


156/Unknown  76s 235ms/step - categorical_accuracy: 0.6024 - loss: 1.6605 - mean_io_u: 0.0343


157/Unknown  76s 234ms/step - categorical_accuracy: 0.6029 - loss: 1.6584 - mean_io_u: 0.0344


158/Unknown  76s 233ms/step - categorical_accuracy: 0.6035 - loss: 1.6563 - mean_io_u: 0.0344


159/Unknown  76s 232ms/step - categorical_accuracy: 0.6040 - loss: 1.6543 - mean_io_u: 0.0345


160/Unknown  76s 231ms/step - categorical_accuracy: 0.6045 - loss: 1.6522 - mean_io_u: 0.0345


161/Unknown  77s 230ms/step - categorical_accuracy: 0.6050 - loss: 1.6501 - mean_io_u: 0.0346


162/Unknown  77s 230ms/step - categorical_accuracy: 0.6056 - loss: 1.6481 - mean_io_u: 0.0346


163/Unknown  77s 229ms/step - categorical_accuracy: 0.6061 - loss: 1.6461 - mean_io_u: 0.0347


164/Unknown  77s 228ms/step - categorical_accuracy: 0.6066 - loss: 1.6440 - mean_io_u: 0.0347


165/Unknown  77s 227ms/step - categorical_accuracy: 0.6071 - loss: 1.6420 - mean_io_u: 0.0348


166/Unknown  77s 226ms/step - categorical_accuracy: 0.6076 - loss: 1.6400 - mean_io_u: 0.0348


167/Unknown  77s 226ms/step - categorical_accuracy: 0.6081 - loss: 1.6381 - mean_io_u: 0.0349


168/Unknown  77s 225ms/step - categorical_accuracy: 0.6086 - loss: 1.6361 - mean_io_u: 0.0349


169/Unknown  77s 224ms/step - categorical_accuracy: 0.6091 - loss: 1.6341 - mean_io_u: 0.0350


170/Unknown  77s 224ms/step - categorical_accuracy: 0.6096 - loss: 1.6322 - mean_io_u: 0.0350


171/Unknown  77s 223ms/step - categorical_accuracy: 0.6101 - loss: 1.6302 - mean_io_u: 0.0351


172/Unknown  78s 222ms/step - categorical_accuracy: 0.6105 - loss: 1.6282 - mean_io_u: 0.0352


173/Unknown  78s 222ms/step - categorical_accuracy: 0.6110 - loss: 1.6264 - mean_io_u: 0.0352


174/Unknown  78s 221ms/step - categorical_accuracy: 0.6115 - loss: 1.6245 - mean_io_u: 0.0353


175/Unknown  78s 220ms/step - categorical_accuracy: 0.6120 - loss: 1.6226 - mean_io_u: 0.0353


176/Unknown  78s 220ms/step - categorical_accuracy: 0.6124 - loss: 1.6207 - mean_io_u: 0.0354


177/Unknown  78s 219ms/step - categorical_accuracy: 0.6129 - loss: 1.6188 - mean_io_u: 0.0354


178/Unknown  78s 218ms/step - categorical_accuracy: 0.6133 - loss: 1.6170 - mean_io_u: 0.0355


179/Unknown  78s 218ms/step - categorical_accuracy: 0.6138 - loss: 1.6151 - mean_io_u: 0.0355


180/Unknown  78s 217ms/step - categorical_accuracy: 0.6142 - loss: 1.6133 - mean_io_u: 0.0356


181/Unknown  79s 216ms/step - categorical_accuracy: 0.6147 - loss: 1.6115 - mean_io_u: 0.0356


182/Unknown  79s 216ms/step - categorical_accuracy: 0.6151 - loss: 1.6097 - mean_io_u: 0.0357


183/Unknown  79s 215ms/step - categorical_accuracy: 0.6156 - loss: 1.6079 - mean_io_u: 0.0357


184/Unknown  79s 215ms/step - categorical_accuracy: 0.6160 - loss: 1.6061 - mean_io_u: 0.0358


185/Unknown  79s 214ms/step - categorical_accuracy: 0.6164 - loss: 1.6043 - mean_io_u: 0.0358


186/Unknown  79s 214ms/step - categorical_accuracy: 0.6169 - loss: 1.6026 - mean_io_u: 0.0359


187/Unknown  79s 213ms/step - categorical_accuracy: 0.6173 - loss: 1.6009 - mean_io_u: 0.0359


188/Unknown  79s 213ms/step - categorical_accuracy: 0.6177 - loss: 1.5991 - mean_io_u: 0.0360


189/Unknown  79s 212ms/step - categorical_accuracy: 0.6181 - loss: 1.5974 - mean_io_u: 0.0361


190/Unknown  80s 212ms/step - categorical_accuracy: 0.6185 - loss: 1.5957 - mean_io_u: 0.0361


191/Unknown  80s 211ms/step - categorical_accuracy: 0.6189 - loss: 1.5940 - mean_io_u: 0.0362


192/Unknown  80s 211ms/step - categorical_accuracy: 0.6193 - loss: 1.5923 - mean_io_u: 0.0362


193/Unknown  80s 210ms/step - categorical_accuracy: 0.6198 - loss: 1.5907 - mean_io_u: 0.0363


194/Unknown  80s 209ms/step - categorical_accuracy: 0.6202 - loss: 1.5890 - mean_io_u: 0.0363


195/Unknown  80s 209ms/step - categorical_accuracy: 0.6205 - loss: 1.5873 - mean_io_u: 0.0364


196/Unknown  80s 208ms/step - categorical_accuracy: 0.6209 - loss: 1.5857 - mean_io_u: 0.0365


197/Unknown  80s 208ms/step - categorical_accuracy: 0.6213 - loss: 1.5841 - mean_io_u: 0.0365


198/Unknown  80s 207ms/step - categorical_accuracy: 0.6217 - loss: 1.5825 - mean_io_u: 0.0366


199/Unknown  81s 207ms/step - categorical_accuracy: 0.6221 - loss: 1.5809 - mean_io_u: 0.0366


200/Unknown  81s 206ms/step - categorical_accuracy: 0.6225 - loss: 1.5793 - mean_io_u: 0.0367


201/Unknown  81s 206ms/step - categorical_accuracy: 0.6229 - loss: 1.5777 - mean_io_u: 0.0367


202/Unknown  81s 205ms/step - categorical_accuracy: 0.6232 - loss: 1.5761 - mean_io_u: 0.0368


203/Unknown  81s 205ms/step - categorical_accuracy: 0.6236 - loss: 1.5746 - mean_io_u: 0.0368


204/Unknown  81s 205ms/step - categorical_accuracy: 0.6240 - loss: 1.5730 - mean_io_u: 0.0369


205/Unknown  81s 204ms/step - categorical_accuracy: 0.6244 - loss: 1.5715 - mean_io_u: 0.0370


206/Unknown  81s 204ms/step - categorical_accuracy: 0.6247 - loss: 1.5699 - mean_io_u: 0.0370


207/Unknown  81s 203ms/step - categorical_accuracy: 0.6251 - loss: 1.5684 - mean_io_u: 0.0371


208/Unknown  82s 203ms/step - categorical_accuracy: 0.6254 - loss: 1.5669 - mean_io_u: 0.0371


209/Unknown  82s 202ms/step - categorical_accuracy: 0.6258 - loss: 1.5653 - mean_io_u: 0.0372


210/Unknown  82s 202ms/step - categorical_accuracy: 0.6262 - loss: 1.5638 - mean_io_u: 0.0372


211/Unknown  82s 201ms/step - categorical_accuracy: 0.6265 - loss: 1.5623 - mean_io_u: 0.0373


212/Unknown  82s 201ms/step - categorical_accuracy: 0.6269 - loss: 1.5608 - mean_io_u: 0.0374


213/Unknown  82s 200ms/step - categorical_accuracy: 0.6272 - loss: 1.5593 - mean_io_u: 0.0374


214/Unknown  82s 200ms/step - categorical_accuracy: 0.6276 - loss: 1.5579 - mean_io_u: 0.0375


215/Unknown  82s 199ms/step - categorical_accuracy: 0.6279 - loss: 1.5564 - mean_io_u: 0.0375


216/Unknown  82s 199ms/step - categorical_accuracy: 0.6282 - loss: 1.5550 - mean_io_u: 0.0376


217/Unknown  82s 198ms/step - categorical_accuracy: 0.6286 - loss: 1.5535 - mean_io_u: 0.0376


218/Unknown  83s 198ms/step - categorical_accuracy: 0.6289 - loss: 1.5521 - mean_io_u: 0.0377


219/Unknown  83s 197ms/step - categorical_accuracy: 0.6292 - loss: 1.5507 - mean_io_u: 0.0377


220/Unknown  83s 197ms/step - categorical_accuracy: 0.6296 - loss: 1.5493 - mean_io_u: 0.0378


221/Unknown  83s 196ms/step - categorical_accuracy: 0.6299 - loss: 1.5479 - mean_io_u: 0.0379


222/Unknown  83s 195ms/step - categorical_accuracy: 0.6302 - loss: 1.5465 - mean_io_u: 0.0379


223/Unknown  83s 195ms/step - categorical_accuracy: 0.6305 - loss: 1.5451 - mean_io_u: 0.0380


224/Unknown  83s 194ms/step - categorical_accuracy: 0.6309 - loss: 1.5437 - mean_io_u: 0.0380


225/Unknown  83s 194ms/step - categorical_accuracy: 0.6312 - loss: 1.5424 - mean_io_u: 0.0381


226/Unknown  83s 193ms/step - categorical_accuracy: 0.6315 - loss: 1.5410 - mean_io_u: 0.0381


227/Unknown  83s 193ms/step - categorical_accuracy: 0.6318 - loss: 1.5397 - mean_io_u: 0.0382


228/Unknown  83s 192ms/step - categorical_accuracy: 0.6321 - loss: 1.5383 - mean_io_u: 0.0382


229/Unknown  83s 192ms/step - categorical_accuracy: 0.6324 - loss: 1.5370 - mean_io_u: 0.0383


230/Unknown  83s 191ms/step - categorical_accuracy: 0.6327 - loss: 1.5357 - mean_io_u: 0.0383


231/Unknown  84s 191ms/step - categorical_accuracy: 0.6330 - loss: 1.5344 - mean_io_u: 0.0384


232/Unknown  84s 190ms/step - categorical_accuracy: 0.6333 - loss: 1.5331 - mean_io_u: 0.0384


233/Unknown  84s 190ms/step - categorical_accuracy: 0.6336 - loss: 1.5318 - mean_io_u: 0.0385


234/Unknown  84s 189ms/step - categorical_accuracy: 0.6339 - loss: 1.5305 - mean_io_u: 0.0385


235/Unknown  84s 189ms/step - categorical_accuracy: 0.6342 - loss: 1.5292 - mean_io_u: 0.0386


236/Unknown  84s 188ms/step - categorical_accuracy: 0.6345 - loss: 1.5279 - mean_io_u: 0.0386


237/Unknown  84s 188ms/step - categorical_accuracy: 0.6348 - loss: 1.5267 - mean_io_u: 0.0387


238/Unknown  84s 188ms/step - categorical_accuracy: 0.6351 - loss: 1.5255 - mean_io_u: 0.0387


239/Unknown  84s 187ms/step - categorical_accuracy: 0.6354 - loss: 1.5242 - mean_io_u: 0.0388


240/Unknown  84s 187ms/step - categorical_accuracy: 0.6356 - loss: 1.5230 - mean_io_u: 0.0388


241/Unknown  84s 186ms/step - categorical_accuracy: 0.6359 - loss: 1.5218 - mean_io_u: 0.0389


242/Unknown  84s 186ms/step - categorical_accuracy: 0.6362 - loss: 1.5206 - mean_io_u: 0.0389


243/Unknown  84s 185ms/step - categorical_accuracy: 0.6365 - loss: 1.5195 - mean_io_u: 0.0390


244/Unknown  85s 185ms/step - categorical_accuracy: 0.6367 - loss: 1.5183 - mean_io_u: 0.0390


245/Unknown  85s 184ms/step - categorical_accuracy: 0.6370 - loss: 1.5171 - mean_io_u: 0.0391


246/Unknown  85s 184ms/step - categorical_accuracy: 0.6373 - loss: 1.5160 - mean_io_u: 0.0391


247/Unknown  85s 184ms/step - categorical_accuracy: 0.6375 - loss: 1.5149 - mean_io_u: 0.0392


248/Unknown  85s 183ms/step - categorical_accuracy: 0.6378 - loss: 1.5137 - mean_io_u: 0.0392


249/Unknown  85s 183ms/step - categorical_accuracy: 0.6380 - loss: 1.5126 - mean_io_u: 0.0393


250/Unknown  85s 182ms/step - categorical_accuracy: 0.6383 - loss: 1.5115 - mean_io_u: 0.0393


251/Unknown  85s 182ms/step - categorical_accuracy: 0.6386 - loss: 1.5104 - mean_io_u: 0.0393


252/Unknown  85s 181ms/step - categorical_accuracy: 0.6388 - loss: 1.5093 - mean_io_u: 0.0394


253/Unknown  85s 181ms/step - categorical_accuracy: 0.6390 - loss: 1.5082 - mean_io_u: 0.0394


254/Unknown  85s 181ms/step - categorical_accuracy: 0.6393 - loss: 1.5071 - mean_io_u: 0.0395


255/Unknown  85s 180ms/step - categorical_accuracy: 0.6395 - loss: 1.5061 - mean_io_u: 0.0395


256/Unknown  85s 180ms/step - categorical_accuracy: 0.6398 - loss: 1.5050 - mean_io_u: 0.0396


257/Unknown  86s 179ms/step - categorical_accuracy: 0.6400 - loss: 1.5040 - mean_io_u: 0.0396


258/Unknown  86s 179ms/step - categorical_accuracy: 0.6402 - loss: 1.5029 - mean_io_u: 0.0397


259/Unknown  86s 178ms/step - categorical_accuracy: 0.6405 - loss: 1.5019 - mean_io_u: 0.0397


260/Unknown  86s 178ms/step - categorical_accuracy: 0.6407 - loss: 1.5009 - mean_io_u: 0.0398


261/Unknown  86s 177ms/step - categorical_accuracy: 0.6409 - loss: 1.4998 - mean_io_u: 0.0398


262/Unknown  86s 177ms/step - categorical_accuracy: 0.6412 - loss: 1.4988 - mean_io_u: 0.0399


263/Unknown  86s 177ms/step - categorical_accuracy: 0.6414 - loss: 1.4978 - mean_io_u: 0.0399


264/Unknown  86s 176ms/step - categorical_accuracy: 0.6416 - loss: 1.4968 - mean_io_u: 0.0400


265/Unknown  86s 176ms/step - categorical_accuracy: 0.6419 - loss: 1.4958 - mean_io_u: 0.0400


266/Unknown  86s 175ms/step - categorical_accuracy: 0.6421 - loss: 1.4948 - mean_io_u: 0.0401


267/Unknown  86s 175ms/step - categorical_accuracy: 0.6423 - loss: 1.4938 - mean_io_u: 0.0401


268/Unknown  86s 174ms/step - categorical_accuracy: 0.6425 - loss: 1.4928 - mean_io_u: 0.0402


269/Unknown  86s 174ms/step - categorical_accuracy: 0.6428 - loss: 1.4918 - mean_io_u: 0.0402


270/Unknown  86s 174ms/step - categorical_accuracy: 0.6430 - loss: 1.4908 - mean_io_u: 0.0403


271/Unknown  86s 173ms/step - categorical_accuracy: 0.6432 - loss: 1.4899 - mean_io_u: 0.0403


272/Unknown  86s 173ms/step - categorical_accuracy: 0.6434 - loss: 1.4889 - mean_io_u: 0.0404


273/Unknown  87s 173ms/step - categorical_accuracy: 0.6437 - loss: 1.4879 - mean_io_u: 0.0404


274/Unknown  87s 172ms/step - categorical_accuracy: 0.6439 - loss: 1.4870 - mean_io_u: 0.0405


275/Unknown  87s 172ms/step - categorical_accuracy: 0.6441 - loss: 1.4860 - mean_io_u: 0.0405


276/Unknown  87s 172ms/step - categorical_accuracy: 0.6443 - loss: 1.4851 - mean_io_u: 0.0405


277/Unknown  87s 171ms/step - categorical_accuracy: 0.6445 - loss: 1.4841 - mean_io_u: 0.0406


278/Unknown  87s 171ms/step - categorical_accuracy: 0.6447 - loss: 1.4832 - mean_io_u: 0.0406


279/Unknown  87s 170ms/step - categorical_accuracy: 0.6449 - loss: 1.4823 - mean_io_u: 0.0407


280/Unknown  87s 170ms/step - categorical_accuracy: 0.6451 - loss: 1.4813 - mean_io_u: 0.0407


281/Unknown  87s 170ms/step - categorical_accuracy: 0.6454 - loss: 1.4804 - mean_io_u: 0.0408


282/Unknown  87s 169ms/step - categorical_accuracy: 0.6456 - loss: 1.4795 - mean_io_u: 0.0408


283/Unknown  87s 169ms/step - categorical_accuracy: 0.6458 - loss: 1.4786 - mean_io_u: 0.0409


284/Unknown  87s 169ms/step - categorical_accuracy: 0.6460 - loss: 1.4776 - mean_io_u: 0.0409


285/Unknown  87s 168ms/step - categorical_accuracy: 0.6462 - loss: 1.4767 - mean_io_u: 0.0410


286/Unknown  88s 168ms/step - categorical_accuracy: 0.6464 - loss: 1.4758 - mean_io_u: 0.0410


287/Unknown  88s 168ms/step - categorical_accuracy: 0.6466 - loss: 1.4749 - mean_io_u: 0.0411


288/Unknown  88s 167ms/step - categorical_accuracy: 0.6468 - loss: 1.4740 - mean_io_u: 0.0411


289/Unknown  88s 167ms/step - categorical_accuracy: 0.6470 - loss: 1.4731 - mean_io_u: 0.0412


290/Unknown  88s 167ms/step - categorical_accuracy: 0.6472 - loss: 1.4722 - mean_io_u: 0.0412


291/Unknown  88s 166ms/step - categorical_accuracy: 0.6474 - loss: 1.4713 - mean_io_u: 0.0413


292/Unknown  88s 166ms/step - categorical_accuracy: 0.6476 - loss: 1.4704 - mean_io_u: 0.0413


293/Unknown  88s 166ms/step - categorical_accuracy: 0.6478 - loss: 1.4695 - mean_io_u: 0.0414


294/Unknown  88s 166ms/step - categorical_accuracy: 0.6480 - loss: 1.4686 - mean_io_u: 0.0414


295/Unknown  88s 165ms/step - categorical_accuracy: 0.6482 - loss: 1.4677 - mean_io_u: 0.0415


296/Unknown  88s 165ms/step - categorical_accuracy: 0.6484 - loss: 1.4669 - mean_io_u: 0.0415


297/Unknown  88s 165ms/step - categorical_accuracy: 0.6486 - loss: 1.4660 - mean_io_u: 0.0416


298/Unknown  88s 164ms/step - categorical_accuracy: 0.6488 - loss: 1.4651 - mean_io_u: 0.0416


299/Unknown  89s 164ms/step - categorical_accuracy: 0.6490 - loss: 1.4642 - mean_io_u: 0.0417


300/Unknown  89s 164ms/step - categorical_accuracy: 0.6492 - loss: 1.4634 - mean_io_u: 0.0417


301/Unknown  89s 163ms/step - categorical_accuracy: 0.6494 - loss: 1.4625 - mean_io_u: 0.0418


302/Unknown  89s 163ms/step - categorical_accuracy: 0.6495 - loss: 1.4617 - mean_io_u: 0.0418


303/Unknown  89s 163ms/step - categorical_accuracy: 0.6497 - loss: 1.4608 - mean_io_u: 0.0418


304/Unknown  89s 163ms/step - categorical_accuracy: 0.6499 - loss: 1.4600 - mean_io_u: 0.0419


305/Unknown  89s 162ms/step - categorical_accuracy: 0.6501 - loss: 1.4591 - mean_io_u: 0.0419


306/Unknown  89s 162ms/step - categorical_accuracy: 0.6503 - loss: 1.4583 - mean_io_u: 0.0420


307/Unknown  89s 162ms/step - categorical_accuracy: 0.6505 - loss: 1.4574 - mean_io_u: 0.0420


308/Unknown  89s 162ms/step - categorical_accuracy: 0.6507 - loss: 1.4566 - mean_io_u: 0.0421


309/Unknown  89s 162ms/step - categorical_accuracy: 0.6509 - loss: 1.4558 - mean_io_u: 0.0421


310/Unknown  90s 161ms/step - categorical_accuracy: 0.6510 - loss: 1.4550 - mean_io_u: 0.0422


311/Unknown  90s 161ms/step - categorical_accuracy: 0.6512 - loss: 1.4541 - mean_io_u: 0.0422


312/Unknown  90s 161ms/step - categorical_accuracy: 0.6514 - loss: 1.4533 - mean_io_u: 0.0422


313/Unknown  90s 161ms/step - categorical_accuracy: 0.6516 - loss: 1.4525 - mean_io_u: 0.0423


314/Unknown  90s 160ms/step - categorical_accuracy: 0.6518 - loss: 1.4517 - mean_io_u: 0.0423


315/Unknown  90s 160ms/step - categorical_accuracy: 0.6519 - loss: 1.4509 - mean_io_u: 0.0424


316/Unknown  90s 160ms/step - categorical_accuracy: 0.6521 - loss: 1.4501 - mean_io_u: 0.0424


317/Unknown  90s 159ms/step - categorical_accuracy: 0.6523 - loss: 1.4493 - mean_io_u: 0.0425


318/Unknown  90s 159ms/step - categorical_accuracy: 0.6525 - loss: 1.4485 - mean_io_u: 0.0425


319/Unknown  90s 159ms/step - categorical_accuracy: 0.6527 - loss: 1.4477 - mean_io_u: 0.0426


320/Unknown  90s 159ms/step - categorical_accuracy: 0.6528 - loss: 1.4469 - mean_io_u: 0.0426


321/Unknown  90s 158ms/step - categorical_accuracy: 0.6530 - loss: 1.4461 - mean_io_u: 0.0426


322/Unknown  90s 158ms/step - categorical_accuracy: 0.6532 - loss: 1.4453 - mean_io_u: 0.0427


323/Unknown  90s 158ms/step - categorical_accuracy: 0.6534 - loss: 1.4445 - mean_io_u: 0.0427


324/Unknown  91s 158ms/step - categorical_accuracy: 0.6535 - loss: 1.4437 - mean_io_u: 0.0428


325/Unknown  91s 157ms/step - categorical_accuracy: 0.6537 - loss: 1.4429 - mean_io_u: 0.0428


326/Unknown  91s 157ms/step - categorical_accuracy: 0.6539 - loss: 1.4421 - mean_io_u: 0.0429


327/Unknown  91s 157ms/step - categorical_accuracy: 0.6541 - loss: 1.4414 - mean_io_u: 0.0429


328/Unknown  91s 157ms/step - categorical_accuracy: 0.6542 - loss: 1.4406 - mean_io_u: 0.0429


329/Unknown  91s 157ms/step - categorical_accuracy: 0.6544 - loss: 1.4398 - mean_io_u: 0.0430


330/Unknown  91s 156ms/step - categorical_accuracy: 0.6546 - loss: 1.4390 - mean_io_u: 0.0430


331/Unknown  91s 156ms/step - categorical_accuracy: 0.6547 - loss: 1.4383 - mean_io_u: 0.0431


332/Unknown  91s 156ms/step - categorical_accuracy: 0.6549 - loss: 1.4375 - mean_io_u: 0.0431


333/Unknown  91s 156ms/step - categorical_accuracy: 0.6551 - loss: 1.4367 - mean_io_u: 0.0432


334/Unknown  91s 156ms/step - categorical_accuracy: 0.6553 - loss: 1.4360 - mean_io_u: 0.0432


335/Unknown  91s 155ms/step - categorical_accuracy: 0.6554 - loss: 1.4352 - mean_io_u: 0.0432


336/Unknown  92s 155ms/step - categorical_accuracy: 0.6556 - loss: 1.4344 - mean_io_u: 0.0433


337/Unknown  92s 155ms/step - categorical_accuracy: 0.6558 - loss: 1.4337 - mean_io_u: 0.0433


338/Unknown  92s 155ms/step - categorical_accuracy: 0.6559 - loss: 1.4329 - mean_io_u: 0.0434


339/Unknown  92s 154ms/step - categorical_accuracy: 0.6561 - loss: 1.4322 - mean_io_u: 0.0434


340/Unknown  92s 154ms/step - categorical_accuracy: 0.6563 - loss: 1.4314 - mean_io_u: 0.0435


341/Unknown  92s 154ms/step - categorical_accuracy: 0.6564 - loss: 1.4307 - mean_io_u: 0.0435


342/Unknown  92s 154ms/step - categorical_accuracy: 0.6566 - loss: 1.4300 - mean_io_u: 0.0435


343/Unknown  92s 154ms/step - categorical_accuracy: 0.6568 - loss: 1.4292 - mean_io_u: 0.0436


344/Unknown  92s 153ms/step - categorical_accuracy: 0.6569 - loss: 1.4285 - mean_io_u: 0.0436


345/Unknown  92s 153ms/step - categorical_accuracy: 0.6571 - loss: 1.4277 - mean_io_u: 0.0437


346/Unknown  92s 153ms/step - categorical_accuracy: 0.6572 - loss: 1.4270 - mean_io_u: 0.0437


347/Unknown  93s 153ms/step - categorical_accuracy: 0.6574 - loss: 1.4263 - mean_io_u: 0.0438


348/Unknown  93s 153ms/step - categorical_accuracy: 0.6576 - loss: 1.4255 - mean_io_u: 0.0438


349/Unknown  93s 152ms/step - categorical_accuracy: 0.6577 - loss: 1.4248 - mean_io_u: 0.0438


350/Unknown  93s 152ms/step - categorical_accuracy: 0.6579 - loss: 1.4241 - mean_io_u: 0.0439


351/Unknown  93s 152ms/step - categorical_accuracy: 0.6581 - loss: 1.4234 - mean_io_u: 0.0439


352/Unknown  93s 152ms/step - categorical_accuracy: 0.6582 - loss: 1.4227 - mean_io_u: 0.0440


353/Unknown  93s 152ms/step - categorical_accuracy: 0.6584 - loss: 1.4219 - mean_io_u: 0.0440


354/Unknown  93s 151ms/step - categorical_accuracy: 0.6585 - loss: 1.4212 - mean_io_u: 0.0441


355/Unknown  93s 151ms/step - categorical_accuracy: 0.6587 - loss: 1.4205 - mean_io_u: 0.0441


356/Unknown  93s 151ms/step - categorical_accuracy: 0.6588 - loss: 1.4198 - mean_io_u: 0.0441


357/Unknown  93s 151ms/step - categorical_accuracy: 0.6590 - loss: 1.4191 - mean_io_u: 0.0442


358/Unknown  93s 151ms/step - categorical_accuracy: 0.6592 - loss: 1.4184 - mean_io_u: 0.0442


359/Unknown  93s 150ms/step - categorical_accuracy: 0.6593 - loss: 1.4177 - mean_io_u: 0.0443


360/Unknown  94s 150ms/step - categorical_accuracy: 0.6595 - loss: 1.4171 - mean_io_u: 0.0443


361/Unknown  94s 150ms/step - categorical_accuracy: 0.6596 - loss: 1.4164 - mean_io_u: 0.0443


362/Unknown  94s 150ms/step - categorical_accuracy: 0.6598 - loss: 1.4157 - mean_io_u: 0.0444


363/Unknown  94s 150ms/step - categorical_accuracy: 0.6599 - loss: 1.4150 - mean_io_u: 0.0444


364/Unknown  94s 149ms/step - categorical_accuracy: 0.6601 - loss: 1.4143 - mean_io_u: 0.0445


365/Unknown  94s 149ms/step - categorical_accuracy: 0.6602 - loss: 1.4137 - mean_io_u: 0.0445


366/Unknown  94s 149ms/step - categorical_accuracy: 0.6603 - loss: 1.4130 - mean_io_u: 0.0445


367/Unknown  94s 149ms/step - categorical_accuracy: 0.6605 - loss: 1.4123 - mean_io_u: 0.0446


368/Unknown  94s 148ms/step - categorical_accuracy: 0.6606 - loss: 1.4117 - mean_io_u: 0.0446


369/Unknown  94s 148ms/step - categorical_accuracy: 0.6608 - loss: 1.4110 - mean_io_u: 0.0447


370/Unknown  94s 148ms/step - categorical_accuracy: 0.6609 - loss: 1.4103 - mean_io_u: 0.0447


371/Unknown  94s 148ms/step - categorical_accuracy: 0.6611 - loss: 1.4097 - mean_io_u: 0.0447


372/Unknown  94s 148ms/step - categorical_accuracy: 0.6612 - loss: 1.4090 - mean_io_u: 0.0448


373/Unknown  95s 148ms/step - categorical_accuracy: 0.6614 - loss: 1.4084 - mean_io_u: 0.0448


374/Unknown  95s 147ms/step - categorical_accuracy: 0.6615 - loss: 1.4078 - mean_io_u: 0.0449


375/Unknown  95s 147ms/step - categorical_accuracy: 0.6616 - loss: 1.4071 - mean_io_u: 0.0449


376/Unknown  95s 147ms/step - categorical_accuracy: 0.6618 - loss: 1.4065 - mean_io_u: 0.0449


377/Unknown  95s 147ms/step - categorical_accuracy: 0.6619 - loss: 1.4058 - mean_io_u: 0.0450


378/Unknown  95s 147ms/step - categorical_accuracy: 0.6620 - loss: 1.4052 - mean_io_u: 0.0450


379/Unknown  95s 147ms/step - categorical_accuracy: 0.6622 - loss: 1.4046 - mean_io_u: 0.0450


380/Unknown  95s 146ms/step - categorical_accuracy: 0.6623 - loss: 1.4039 - mean_io_u: 0.0451


381/Unknown  95s 146ms/step - categorical_accuracy: 0.6625 - loss: 1.4033 - mean_io_u: 0.0451


382/Unknown  95s 146ms/step - categorical_accuracy: 0.6626 - loss: 1.4027 - mean_io_u: 0.0452


383/Unknown  95s 146ms/step - categorical_accuracy: 0.6627 - loss: 1.4021 - mean_io_u: 0.0452


384/Unknown  95s 146ms/step - categorical_accuracy: 0.6628 - loss: 1.4015 - mean_io_u: 0.0452


385/Unknown  96s 146ms/step - categorical_accuracy: 0.6630 - loss: 1.4009 - mean_io_u: 0.0453


386/Unknown  96s 146ms/step - categorical_accuracy: 0.6631 - loss: 1.4003 - mean_io_u: 0.0453


387/Unknown  96s 145ms/step - categorical_accuracy: 0.6632 - loss: 1.3997 - mean_io_u: 0.0454


388/Unknown  96s 145ms/step - categorical_accuracy: 0.6634 - loss: 1.3991 - mean_io_u: 0.0454


389/Unknown  96s 145ms/step - categorical_accuracy: 0.6635 - loss: 1.3984 - mean_io_u: 0.0454


390/Unknown  96s 145ms/step - categorical_accuracy: 0.6636 - loss: 1.3978 - mean_io_u: 0.0455


391/Unknown  96s 145ms/step - categorical_accuracy: 0.6638 - loss: 1.3973 - mean_io_u: 0.0455


392/Unknown  96s 145ms/step - categorical_accuracy: 0.6639 - loss: 1.3967 - mean_io_u: 0.0455


393/Unknown  96s 144ms/step - categorical_accuracy: 0.6640 - loss: 1.3961 - mean_io_u: 0.0456


394/Unknown  96s 144ms/step - categorical_accuracy: 0.6642 - loss: 1.3955 - mean_io_u: 0.0456


395/Unknown  96s 144ms/step - categorical_accuracy: 0.6643 - loss: 1.3949 - mean_io_u: 0.0457


396/Unknown  96s 144ms/step - categorical_accuracy: 0.6644 - loss: 1.3943 - mean_io_u: 0.0457


397/Unknown  97s 144ms/step - categorical_accuracy: 0.6645 - loss: 1.3937 - mean_io_u: 0.0457


398/Unknown  97s 143ms/step - categorical_accuracy: 0.6647 - loss: 1.3931 - mean_io_u: 0.0458


399/Unknown  97s 143ms/step - categorical_accuracy: 0.6648 - loss: 1.3925 - mean_io_u: 0.0458


400/Unknown  97s 143ms/step - categorical_accuracy: 0.6649 - loss: 1.3919 - mean_io_u: 0.0459


401/Unknown  97s 143ms/step - categorical_accuracy: 0.6650 - loss: 1.3914 - mean_io_u: 0.0459


402/Unknown  97s 143ms/step - categorical_accuracy: 0.6652 - loss: 1.3908 - mean_io_u: 0.0459


403/Unknown  97s 143ms/step - categorical_accuracy: 0.6653 - loss: 1.3902 - mean_io_u: 0.0460


404/Unknown  97s 143ms/step - categorical_accuracy: 0.6654 - loss: 1.3896 - mean_io_u: 0.0460


405/Unknown  97s 142ms/step - categorical_accuracy: 0.6655 - loss: 1.3891 - mean_io_u: 0.0460


406/Unknown  97s 142ms/step - categorical_accuracy: 0.6657 - loss: 1.3885 - mean_io_u: 0.0461


407/Unknown  97s 142ms/step - categorical_accuracy: 0.6658 - loss: 1.3879 - mean_io_u: 0.0461


408/Unknown  97s 142ms/step - categorical_accuracy: 0.6659 - loss: 1.3874 - mean_io_u: 0.0461


409/Unknown  97s 142ms/step - categorical_accuracy: 0.6660 - loss: 1.3868 - mean_io_u: 0.0462


410/Unknown  97s 141ms/step - categorical_accuracy: 0.6661 - loss: 1.3862 - mean_io_u: 0.0462


411/Unknown  98s 141ms/step - categorical_accuracy: 0.6663 - loss: 1.3857 - mean_io_u: 0.0463


412/Unknown  98s 141ms/step - categorical_accuracy: 0.6664 - loss: 1.3851 - mean_io_u: 0.0463


413/Unknown  98s 141ms/step - categorical_accuracy: 0.6665 - loss: 1.3845 - mean_io_u: 0.0463


414/Unknown  98s 141ms/step - categorical_accuracy: 0.6666 - loss: 1.3840 - mean_io_u: 0.0464


415/Unknown  98s 141ms/step - categorical_accuracy: 0.6668 - loss: 1.3834 - mean_io_u: 0.0464


416/Unknown  98s 140ms/step - categorical_accuracy: 0.6669 - loss: 1.3828 - mean_io_u: 0.0465


417/Unknown  98s 140ms/step - categorical_accuracy: 0.6670 - loss: 1.3823 - mean_io_u: 0.0465


418/Unknown  98s 140ms/step - categorical_accuracy: 0.6671 - loss: 1.3817 - mean_io_u: 0.0465


419/Unknown  98s 140ms/step - categorical_accuracy: 0.6672 - loss: 1.3812 - mean_io_u: 0.0466


420/Unknown  98s 140ms/step - categorical_accuracy: 0.6673 - loss: 1.3806 - mean_io_u: 0.0466


421/Unknown  98s 140ms/step - categorical_accuracy: 0.6675 - loss: 1.3801 - mean_io_u: 0.0466


422/Unknown  98s 140ms/step - categorical_accuracy: 0.6676 - loss: 1.3795 - mean_io_u: 0.0467


423/Unknown  99s 140ms/step - categorical_accuracy: 0.6677 - loss: 1.3790 - mean_io_u: 0.0467


424/Unknown  99s 139ms/step - categorical_accuracy: 0.6678 - loss: 1.3784 - mean_io_u: 0.0468


425/Unknown  99s 139ms/step - categorical_accuracy: 0.6679 - loss: 1.3779 - mean_io_u: 0.0468


426/Unknown  99s 139ms/step - categorical_accuracy: 0.6681 - loss: 1.3773 - mean_io_u: 0.0468


427/Unknown  99s 139ms/step - categorical_accuracy: 0.6682 - loss: 1.3768 - mean_io_u: 0.0469


428/Unknown  99s 139ms/step - categorical_accuracy: 0.6683 - loss: 1.3762 - mean_io_u: 0.0469


429/Unknown  99s 139ms/step - categorical_accuracy: 0.6684 - loss: 1.3757 - mean_io_u: 0.0469


430/Unknown  99s 138ms/step - categorical_accuracy: 0.6685 - loss: 1.3751 - mean_io_u: 0.0470


431/Unknown  99s 138ms/step - categorical_accuracy: 0.6686 - loss: 1.3746 - mean_io_u: 0.0470


432/Unknown  99s 138ms/step - categorical_accuracy: 0.6688 - loss: 1.3740 - mean_io_u: 0.0471


433/Unknown  99s 138ms/step - categorical_accuracy: 0.6689 - loss: 1.3735 - mean_io_u: 0.0471


434/Unknown  99s 138ms/step - categorical_accuracy: 0.6690 - loss: 1.3730 - mean_io_u: 0.0471


435/Unknown  99s 138ms/step - categorical_accuracy: 0.6691 - loss: 1.3724 - mean_io_u: 0.0472


436/Unknown  100s 138ms/step - categorical_accuracy: 0.6692 - loss: 1.3719 - mean_io_u: 0.0472


437/Unknown  100s 137ms/step - categorical_accuracy: 0.6693 - loss: 1.3714 - mean_io_u: 0.0473


438/Unknown  100s 137ms/step - categorical_accuracy: 0.6694 - loss: 1.3708 - mean_io_u: 0.0473


439/Unknown  100s 137ms/step - categorical_accuracy: 0.6696 - loss: 1.3703 - mean_io_u: 0.0473


440/Unknown  100s 137ms/step - categorical_accuracy: 0.6697 - loss: 1.3698 - mean_io_u: 0.0474


441/Unknown  100s 137ms/step - categorical_accuracy: 0.6698 - loss: 1.3693 - mean_io_u: 0.0474


442/Unknown  100s 137ms/step - categorical_accuracy: 0.6699 - loss: 1.3687 - mean_io_u: 0.0474


443/Unknown  100s 137ms/step - categorical_accuracy: 0.6700 - loss: 1.3682 - mean_io_u: 0.0475


444/Unknown  100s 137ms/step - categorical_accuracy: 0.6701 - loss: 1.3677 - mean_io_u: 0.0475


445/Unknown  100s 137ms/step - categorical_accuracy: 0.6702 - loss: 1.3672 - mean_io_u: 0.0475


446/Unknown  100s 136ms/step - categorical_accuracy: 0.6703 - loss: 1.3667 - mean_io_u: 0.0476


447/Unknown  100s 136ms/step - categorical_accuracy: 0.6704 - loss: 1.3661 - mean_io_u: 0.0476


448/Unknown  100s 136ms/step - categorical_accuracy: 0.6706 - loss: 1.3656 - mean_io_u: 0.0477


449/Unknown  101s 136ms/step - categorical_accuracy: 0.6707 - loss: 1.3651 - mean_io_u: 0.0477


450/Unknown  101s 136ms/step - categorical_accuracy: 0.6708 - loss: 1.3646 - mean_io_u: 0.0477


451/Unknown  101s 136ms/step - categorical_accuracy: 0.6709 - loss: 1.3641 - mean_io_u: 0.0478


452/Unknown  101s 136ms/step - categorical_accuracy: 0.6710 - loss: 1.3636 - mean_io_u: 0.0478


453/Unknown  101s 136ms/step - categorical_accuracy: 0.6711 - loss: 1.3631 - mean_io_u: 0.0478


454/Unknown  101s 135ms/step - categorical_accuracy: 0.6712 - loss: 1.3626 - mean_io_u: 0.0479


455/Unknown  101s 135ms/step - categorical_accuracy: 0.6713 - loss: 1.3621 - mean_io_u: 0.0479


456/Unknown  101s 135ms/step - categorical_accuracy: 0.6714 - loss: 1.3616 - mean_io_u: 0.0480


457/Unknown  101s 135ms/step - categorical_accuracy: 0.6715 - loss: 1.3611 - mean_io_u: 0.0480


458/Unknown  101s 135ms/step - categorical_accuracy: 0.6716 - loss: 1.3606 - mean_io_u: 0.0480


459/Unknown  101s 135ms/step - categorical_accuracy: 0.6717 - loss: 1.3601 - mean_io_u: 0.0481


460/Unknown  101s 135ms/step - categorical_accuracy: 0.6718 - loss: 1.3596 - mean_io_u: 0.0481


461/Unknown  101s 134ms/step - categorical_accuracy: 0.6719 - loss: 1.3591 - mean_io_u: 0.0481


462/Unknown  102s 134ms/step - categorical_accuracy: 0.6720 - loss: 1.3586 - mean_io_u: 0.0482


463/Unknown  102s 134ms/step - categorical_accuracy: 0.6722 - loss: 1.3581 - mean_io_u: 0.0482


464/Unknown  102s 134ms/step - categorical_accuracy: 0.6723 - loss: 1.3576 - mean_io_u: 0.0483


465/Unknown  102s 134ms/step - categorical_accuracy: 0.6724 - loss: 1.3572 - mean_io_u: 0.0483


466/Unknown  102s 134ms/step - categorical_accuracy: 0.6725 - loss: 1.3567 - mean_io_u: 0.0483


467/Unknown  102s 134ms/step - categorical_accuracy: 0.6726 - loss: 1.3562 - mean_io_u: 0.0484


468/Unknown  102s 134ms/step - categorical_accuracy: 0.6727 - loss: 1.3557 - mean_io_u: 0.0484


469/Unknown  102s 133ms/step - categorical_accuracy: 0.6728 - loss: 1.3552 - mean_io_u: 0.0485


470/Unknown  102s 133ms/step - categorical_accuracy: 0.6729 - loss: 1.3547 - mean_io_u: 0.0485


471/Unknown  102s 133ms/step - categorical_accuracy: 0.6730 - loss: 1.3542 - mean_io_u: 0.0485


472/Unknown  102s 133ms/step - categorical_accuracy: 0.6731 - loss: 1.3537 - mean_io_u: 0.0486


473/Unknown  102s 133ms/step - categorical_accuracy: 0.6732 - loss: 1.3533 - mean_io_u: 0.0486


474/Unknown  102s 133ms/step - categorical_accuracy: 0.6733 - loss: 1.3528 - mean_io_u: 0.0486


475/Unknown  103s 133ms/step - categorical_accuracy: 0.6734 - loss: 1.3523 - mean_io_u: 0.0487


476/Unknown  103s 133ms/step - categorical_accuracy: 0.6735 - loss: 1.3518 - mean_io_u: 0.0487


477/Unknown  103s 133ms/step - categorical_accuracy: 0.6736 - loss: 1.3514 - mean_io_u: 0.0488


478/Unknown  103s 132ms/step - categorical_accuracy: 0.6737 - loss: 1.3509 - mean_io_u: 0.0488


479/Unknown  103s 132ms/step - categorical_accuracy: 0.6738 - loss: 1.3504 - mean_io_u: 0.0488


480/Unknown  103s 132ms/step - categorical_accuracy: 0.6739 - loss: 1.3499 - mean_io_u: 0.0489


481/Unknown  103s 132ms/step - categorical_accuracy: 0.6740 - loss: 1.3495 - mean_io_u: 0.0489


482/Unknown  103s 132ms/step - categorical_accuracy: 0.6741 - loss: 1.3490 - mean_io_u: 0.0489


483/Unknown  103s 132ms/step - categorical_accuracy: 0.6742 - loss: 1.3485 - mean_io_u: 0.0490


484/Unknown  103s 132ms/step - categorical_accuracy: 0.6743 - loss: 1.3480 - mean_io_u: 0.0490


485/Unknown  103s 132ms/step - categorical_accuracy: 0.6744 - loss: 1.3476 - mean_io_u: 0.0491


486/Unknown  104s 132ms/step - categorical_accuracy: 0.6745 - loss: 1.3471 - mean_io_u: 0.0491


487/Unknown  104s 132ms/step - categorical_accuracy: 0.6746 - loss: 1.3466 - mean_io_u: 0.0491


488/Unknown  104s 132ms/step - categorical_accuracy: 0.6747 - loss: 1.3462 - mean_io_u: 0.0492


489/Unknown  104s 132ms/step - categorical_accuracy: 0.6748 - loss: 1.3457 - mean_io_u: 0.0492


490/Unknown  104s 131ms/step - categorical_accuracy: 0.6749 - loss: 1.3452 - mean_io_u: 0.0493


491/Unknown  104s 131ms/step - categorical_accuracy: 0.6750 - loss: 1.3447 - mean_io_u: 0.0493


492/Unknown  104s 131ms/step - categorical_accuracy: 0.6751 - loss: 1.3443 - mean_io_u: 0.0493


493/Unknown  104s 131ms/step - categorical_accuracy: 0.6752 - loss: 1.3438 - mean_io_u: 0.0494


494/Unknown  104s 131ms/step - categorical_accuracy: 0.6753 - loss: 1.3433 - mean_io_u: 0.0494


495/Unknown  104s 131ms/step - categorical_accuracy: 0.6754 - loss: 1.3429 - mean_io_u: 0.0494


496/Unknown  104s 131ms/step - categorical_accuracy: 0.6755 - loss: 1.3424 - mean_io_u: 0.0495


497/Unknown  104s 131ms/step - categorical_accuracy: 0.6756 - loss: 1.3419 - mean_io_u: 0.0495


498/Unknown  105s 131ms/step - categorical_accuracy: 0.6757 - loss: 1.3415 - mean_io_u: 0.0496


499/Unknown  105s 131ms/step - categorical_accuracy: 0.6758 - loss: 1.3410 - mean_io_u: 0.0496


500/Unknown  105s 130ms/step - categorical_accuracy: 0.6759 - loss: 1.3406 - mean_io_u: 0.0496


501/Unknown  105s 130ms/step - categorical_accuracy: 0.6760 - loss: 1.3401 - mean_io_u: 0.0497


502/Unknown  105s 130ms/step - categorical_accuracy: 0.6761 - loss: 1.3396 - mean_io_u: 0.0497


503/Unknown  105s 130ms/step - categorical_accuracy: 0.6762 - loss: 1.3392 - mean_io_u: 0.0498


504/Unknown  105s 130ms/step - categorical_accuracy: 0.6763 - loss: 1.3387 - mean_io_u: 0.0498


505/Unknown  105s 130ms/step - categorical_accuracy: 0.6764 - loss: 1.3383 - mean_io_u: 0.0498


506/Unknown  105s 130ms/step - categorical_accuracy: 0.6765 - loss: 1.3378 - mean_io_u: 0.0499


507/Unknown  105s 130ms/step - categorical_accuracy: 0.6766 - loss: 1.3374 - mean_io_u: 0.0499


508/Unknown  105s 130ms/step - categorical_accuracy: 0.6767 - loss: 1.3369 - mean_io_u: 0.0500


509/Unknown  105s 130ms/step - categorical_accuracy: 0.6768 - loss: 1.3365 - mean_io_u: 0.0500


510/Unknown  106s 129ms/step - categorical_accuracy: 0.6769 - loss: 1.3360 - mean_io_u: 0.0500


511/Unknown  106s 129ms/step - categorical_accuracy: 0.6770 - loss: 1.3356 - mean_io_u: 0.0501


512/Unknown  106s 129ms/step - categorical_accuracy: 0.6771 - loss: 1.3351 - mean_io_u: 0.0501


513/Unknown  106s 129ms/step - categorical_accuracy: 0.6771 - loss: 1.3347 - mean_io_u: 0.0502


514/Unknown  106s 129ms/step - categorical_accuracy: 0.6772 - loss: 1.3343 - mean_io_u: 0.0502


515/Unknown  106s 129ms/step - categorical_accuracy: 0.6773 - loss: 1.3338 - mean_io_u: 0.0502


516/Unknown  106s 129ms/step - categorical_accuracy: 0.6774 - loss: 1.3334 - mean_io_u: 0.0503


517/Unknown  106s 129ms/step - categorical_accuracy: 0.6775 - loss: 1.3330 - mean_io_u: 0.0503


518/Unknown  106s 129ms/step - categorical_accuracy: 0.6776 - loss: 1.3325 - mean_io_u: 0.0504


519/Unknown  106s 128ms/step - categorical_accuracy: 0.6777 - loss: 1.3321 - mean_io_u: 0.0504


520/Unknown  106s 128ms/step - categorical_accuracy: 0.6778 - loss: 1.3317 - mean_io_u: 0.0504


521/Unknown  106s 128ms/step - categorical_accuracy: 0.6779 - loss: 1.3312 - mean_io_u: 0.0505


522/Unknown  106s 128ms/step - categorical_accuracy: 0.6780 - loss: 1.3308 - mean_io_u: 0.0505


523/Unknown  106s 128ms/step - categorical_accuracy: 0.6781 - loss: 1.3304 - mean_io_u: 0.0505


524/Unknown  107s 128ms/step - categorical_accuracy: 0.6781 - loss: 1.3300 - mean_io_u: 0.0506


525/Unknown  107s 128ms/step - categorical_accuracy: 0.6782 - loss: 1.3295 - mean_io_u: 0.0506


526/Unknown  107s 128ms/step - categorical_accuracy: 0.6783 - loss: 1.3291 - mean_io_u: 0.0507


527/Unknown  107s 128ms/step - categorical_accuracy: 0.6784 - loss: 1.3287 - mean_io_u: 0.0507


528/Unknown  107s 128ms/step - categorical_accuracy: 0.6785 - loss: 1.3283 - mean_io_u: 0.0507


529/Unknown  107s 128ms/step - categorical_accuracy: 0.6786 - loss: 1.3278 - mean_io_u: 0.0508


530/Unknown  107s 127ms/step - categorical_accuracy: 0.6787 - loss: 1.3274 - mean_io_u: 0.0508


531/Unknown  107s 127ms/step - categorical_accuracy: 0.6788 - loss: 1.3270 - mean_io_u: 0.0509


532/Unknown  107s 127ms/step - categorical_accuracy: 0.6789 - loss: 1.3266 - mean_io_u: 0.0509


533/Unknown  107s 127ms/step - categorical_accuracy: 0.6789 - loss: 1.3262 - mean_io_u: 0.0509


534/Unknown  107s 127ms/step - categorical_accuracy: 0.6790 - loss: 1.3257 - mean_io_u: 0.0510


535/Unknown  107s 127ms/step - categorical_accuracy: 0.6791 - loss: 1.3253 - mean_io_u: 0.0510


536/Unknown  108s 127ms/step - categorical_accuracy: 0.6792 - loss: 1.3249 - mean_io_u: 0.0511


537/Unknown  108s 127ms/step - categorical_accuracy: 0.6793 - loss: 1.3245 - mean_io_u: 0.0511


538/Unknown  108s 127ms/step - categorical_accuracy: 0.6794 - loss: 1.3241 - mean_io_u: 0.0512


539/Unknown  108s 127ms/step - categorical_accuracy: 0.6795 - loss: 1.3237 - mean_io_u: 0.0512


540/Unknown  108s 126ms/step - categorical_accuracy: 0.6796 - loss: 1.3233 - mean_io_u: 0.0512


541/Unknown  108s 126ms/step - categorical_accuracy: 0.6796 - loss: 1.3228 - mean_io_u: 0.0513


542/Unknown  108s 126ms/step - categorical_accuracy: 0.6797 - loss: 1.3224 - mean_io_u: 0.0513


543/Unknown  108s 126ms/step - categorical_accuracy: 0.6798 - loss: 1.3220 - mean_io_u: 0.0514


544/Unknown  108s 126ms/step - categorical_accuracy: 0.6799 - loss: 1.3216 - mean_io_u: 0.0514


545/Unknown  108s 126ms/step - categorical_accuracy: 0.6800 - loss: 1.3212 - mean_io_u: 0.0514


546/Unknown  108s 126ms/step - categorical_accuracy: 0.6801 - loss: 1.3208 - mean_io_u: 0.0515


547/Unknown  108s 126ms/step - categorical_accuracy: 0.6802 - loss: 1.3204 - mean_io_u: 0.0515


548/Unknown  108s 126ms/step - categorical_accuracy: 0.6802 - loss: 1.3200 - mean_io_u: 0.0516


549/Unknown  108s 126ms/step - categorical_accuracy: 0.6803 - loss: 1.3196 - mean_io_u: 0.0516


550/Unknown  109s 125ms/step - categorical_accuracy: 0.6804 - loss: 1.3192 - mean_io_u: 0.0516


551/Unknown  109s 125ms/step - categorical_accuracy: 0.6805 - loss: 1.3188 - mean_io_u: 0.0517


552/Unknown  109s 125ms/step - categorical_accuracy: 0.6806 - loss: 1.3184 - mean_io_u: 0.0517


553/Unknown  109s 125ms/step - categorical_accuracy: 0.6807 - loss: 1.3180 - mean_io_u: 0.0518


554/Unknown  109s 125ms/step - categorical_accuracy: 0.6807 - loss: 1.3176 - mean_io_u: 0.0518


555/Unknown  109s 125ms/step - categorical_accuracy: 0.6808 - loss: 1.3172 - mean_io_u: 0.0518


556/Unknown  109s 125ms/step - categorical_accuracy: 0.6809 - loss: 1.3168 - mean_io_u: 0.0519


557/Unknown  109s 125ms/step - categorical_accuracy: 0.6810 - loss: 1.3164 - mean_io_u: 0.0519


558/Unknown  109s 125ms/step - categorical_accuracy: 0.6811 - loss: 1.3160 - mean_io_u: 0.0520


559/Unknown  109s 124ms/step - categorical_accuracy: 0.6811 - loss: 1.3156 - mean_io_u: 0.0520


560/Unknown  109s 124ms/step - categorical_accuracy: 0.6812 - loss: 1.3152 - mean_io_u: 0.0520


561/Unknown  109s 124ms/step - categorical_accuracy: 0.6813 - loss: 1.3149 - mean_io_u: 0.0521


562/Unknown  109s 124ms/step - categorical_accuracy: 0.6814 - loss: 1.3145 - mean_io_u: 0.0521


563/Unknown  109s 124ms/step - categorical_accuracy: 0.6815 - loss: 1.3141 - mean_io_u: 0.0522


564/Unknown  110s 124ms/step - categorical_accuracy: 0.6815 - loss: 1.3137 - mean_io_u: 0.0522


565/Unknown  110s 124ms/step - categorical_accuracy: 0.6816 - loss: 1.3133 - mean_io_u: 0.0522


566/Unknown  110s 124ms/step - categorical_accuracy: 0.6817 - loss: 1.3130 - mean_io_u: 0.0523


567/Unknown  110s 124ms/step - categorical_accuracy: 0.6818 - loss: 1.3126 - mean_io_u: 0.0523


568/Unknown  110s 124ms/step - categorical_accuracy: 0.6819 - loss: 1.3122 - mean_io_u: 0.0524


569/Unknown  110s 124ms/step - categorical_accuracy: 0.6819 - loss: 1.3118 - mean_io_u: 0.0524


570/Unknown  110s 124ms/step - categorical_accuracy: 0.6820 - loss: 1.3115 - mean_io_u: 0.0524


571/Unknown  110s 123ms/step - categorical_accuracy: 0.6821 - loss: 1.3111 - mean_io_u: 0.0525


572/Unknown  110s 123ms/step - categorical_accuracy: 0.6822 - loss: 1.3107 - mean_io_u: 0.0525


573/Unknown  110s 123ms/step - categorical_accuracy: 0.6822 - loss: 1.3104 - mean_io_u: 0.0526


574/Unknown  110s 123ms/step - categorical_accuracy: 0.6823 - loss: 1.3100 - mean_io_u: 0.0526


575/Unknown  110s 123ms/step - categorical_accuracy: 0.6824 - loss: 1.3096 - mean_io_u: 0.0526


576/Unknown  110s 123ms/step - categorical_accuracy: 0.6825 - loss: 1.3092 - mean_io_u: 0.0527


577/Unknown  110s 123ms/step - categorical_accuracy: 0.6825 - loss: 1.3089 - mean_io_u: 0.0527


578/Unknown  111s 123ms/step - categorical_accuracy: 0.6826 - loss: 1.3085 - mean_io_u: 0.0527


579/Unknown  111s 123ms/step - categorical_accuracy: 0.6827 - loss: 1.3081 - mean_io_u: 0.0528


580/Unknown  111s 123ms/step - categorical_accuracy: 0.6828 - loss: 1.3078 - mean_io_u: 0.0528


581/Unknown  111s 123ms/step - categorical_accuracy: 0.6828 - loss: 1.3074 - mean_io_u: 0.0529


582/Unknown  111s 123ms/step - categorical_accuracy: 0.6829 - loss: 1.3070 - mean_io_u: 0.0529


583/Unknown  111s 122ms/step - categorical_accuracy: 0.6830 - loss: 1.3067 - mean_io_u: 0.0529


584/Unknown  111s 122ms/step - categorical_accuracy: 0.6831 - loss: 1.3063 - mean_io_u: 0.0530


585/Unknown  111s 122ms/step - categorical_accuracy: 0.6831 - loss: 1.3060 - mean_io_u: 0.0530


586/Unknown  111s 122ms/step - categorical_accuracy: 0.6832 - loss: 1.3056 - mean_io_u: 0.0531


587/Unknown  111s 122ms/step - categorical_accuracy: 0.6833 - loss: 1.3052 - mean_io_u: 0.0531


588/Unknown  111s 122ms/step - categorical_accuracy: 0.6834 - loss: 1.3049 - mean_io_u: 0.0531


589/Unknown  111s 122ms/step - categorical_accuracy: 0.6834 - loss: 1.3045 - mean_io_u: 0.0532


590/Unknown  111s 122ms/step - categorical_accuracy: 0.6835 - loss: 1.3041 - mean_io_u: 0.0532


591/Unknown  112s 122ms/step - categorical_accuracy: 0.6836 - loss: 1.3038 - mean_io_u: 0.0533


592/Unknown  112s 122ms/step - categorical_accuracy: 0.6837 - loss: 1.3034 - mean_io_u: 0.0533


593/Unknown  112s 122ms/step - categorical_accuracy: 0.6837 - loss: 1.3031 - mean_io_u: 0.0533


594/Unknown  112s 122ms/step - categorical_accuracy: 0.6838 - loss: 1.3027 - mean_io_u: 0.0534


595/Unknown  112s 122ms/step - categorical_accuracy: 0.6839 - loss: 1.3023 - mean_io_u: 0.0534


596/Unknown  112s 122ms/step - categorical_accuracy: 0.6840 - loss: 1.3020 - mean_io_u: 0.0534


597/Unknown  112s 122ms/step - categorical_accuracy: 0.6840 - loss: 1.3016 - mean_io_u: 0.0535


598/Unknown  112s 121ms/step - categorical_accuracy: 0.6841 - loss: 1.3013 - mean_io_u: 0.0535


599/Unknown  112s 121ms/step - categorical_accuracy: 0.6842 - loss: 1.3009 - mean_io_u: 0.0536


600/Unknown  112s 121ms/step - categorical_accuracy: 0.6843 - loss: 1.3006 - mean_io_u: 0.0536


601/Unknown  112s 121ms/step - categorical_accuracy: 0.6843 - loss: 1.3002 - mean_io_u: 0.0536


602/Unknown  112s 121ms/step - categorical_accuracy: 0.6844 - loss: 1.2998 - mean_io_u: 0.0537


603/Unknown  112s 121ms/step - categorical_accuracy: 0.6845 - loss: 1.2995 - mean_io_u: 0.0537


604/Unknown  113s 121ms/step - categorical_accuracy: 0.6846 - loss: 1.2991 - mean_io_u: 0.0538


605/Unknown  113s 121ms/step - categorical_accuracy: 0.6846 - loss: 1.2988 - mean_io_u: 0.0538


606/Unknown  113s 121ms/step - categorical_accuracy: 0.6847 - loss: 1.2984 - mean_io_u: 0.0538


607/Unknown  113s 121ms/step - categorical_accuracy: 0.6848 - loss: 1.2981 - mean_io_u: 0.0539


608/Unknown  113s 121ms/step - categorical_accuracy: 0.6848 - loss: 1.2977 - mean_io_u: 0.0539


609/Unknown  113s 121ms/step - categorical_accuracy: 0.6849 - loss: 1.2974 - mean_io_u: 0.0540


610/Unknown  113s 120ms/step - categorical_accuracy: 0.6850 - loss: 1.2970 - mean_io_u: 0.0540


611/Unknown  113s 120ms/step - categorical_accuracy: 0.6851 - loss: 1.2967 - mean_io_u: 0.0540


612/Unknown  113s 120ms/step - categorical_accuracy: 0.6851 - loss: 1.2963 - mean_io_u: 0.0541


613/Unknown  113s 120ms/step - categorical_accuracy: 0.6852 - loss: 1.2960 - mean_io_u: 0.0541


614/Unknown  113s 120ms/step - categorical_accuracy: 0.6853 - loss: 1.2957 - mean_io_u: 0.0541


615/Unknown  113s 120ms/step - categorical_accuracy: 0.6853 - loss: 1.2953 - mean_io_u: 0.0542


616/Unknown  114s 120ms/step - categorical_accuracy: 0.6854 - loss: 1.2950 - mean_io_u: 0.0542


617/Unknown  114s 120ms/step - categorical_accuracy: 0.6855 - loss: 1.2946 - mean_io_u: 0.0543


618/Unknown  114s 120ms/step - categorical_accuracy: 0.6856 - loss: 1.2943 - mean_io_u: 0.0543


619/Unknown  114s 120ms/step - categorical_accuracy: 0.6856 - loss: 1.2939 - mean_io_u: 0.0543


620/Unknown  114s 120ms/step - categorical_accuracy: 0.6857 - loss: 1.2936 - mean_io_u: 0.0544


621/Unknown  114s 120ms/step - categorical_accuracy: 0.6858 - loss: 1.2933 - mean_io_u: 0.0544


622/Unknown  114s 120ms/step - categorical_accuracy: 0.6858 - loss: 1.2929 - mean_io_u: 0.0545


623/Unknown  114s 120ms/step - categorical_accuracy: 0.6859 - loss: 1.2926 - mean_io_u: 0.0545


624/Unknown  114s 120ms/step - categorical_accuracy: 0.6860 - loss: 1.2922 - mean_io_u: 0.0545


625/Unknown  114s 120ms/step - categorical_accuracy: 0.6860 - loss: 1.2919 - mean_io_u: 0.0546


626/Unknown  114s 120ms/step - categorical_accuracy: 0.6861 - loss: 1.2916 - mean_io_u: 0.0546


627/Unknown  114s 120ms/step - categorical_accuracy: 0.6862 - loss: 1.2912 - mean_io_u: 0.0546


628/Unknown  115s 119ms/step - categorical_accuracy: 0.6862 - loss: 1.2909 - mean_io_u: 0.0547


629/Unknown  115s 119ms/step - categorical_accuracy: 0.6863 - loss: 1.2906 - mean_io_u: 0.0547


630/Unknown  115s 119ms/step - categorical_accuracy: 0.6864 - loss: 1.2902 - mean_io_u: 0.0548


631/Unknown  115s 119ms/step - categorical_accuracy: 0.6865 - loss: 1.2899 - mean_io_u: 0.0548


632/Unknown  115s 119ms/step - categorical_accuracy: 0.6865 - loss: 1.2896 - mean_io_u: 0.0548


633/Unknown  115s 119ms/step - categorical_accuracy: 0.6866 - loss: 1.2892 - mean_io_u: 0.0549


634/Unknown  115s 119ms/step - categorical_accuracy: 0.6867 - loss: 1.2889 - mean_io_u: 0.0549


635/Unknown  115s 119ms/step - categorical_accuracy: 0.6867 - loss: 1.2886 - mean_io_u: 0.0549


636/Unknown  115s 119ms/step - categorical_accuracy: 0.6868 - loss: 1.2882 - mean_io_u: 0.0550


637/Unknown  115s 119ms/step - categorical_accuracy: 0.6869 - loss: 1.2879 - mean_io_u: 0.0550


638/Unknown  115s 119ms/step - categorical_accuracy: 0.6869 - loss: 1.2876 - mean_io_u: 0.0551


639/Unknown  115s 119ms/step - categorical_accuracy: 0.6870 - loss: 1.2872 - mean_io_u: 0.0551


640/Unknown  115s 119ms/step - categorical_accuracy: 0.6871 - loss: 1.2869 - mean_io_u: 0.0551


641/Unknown  116s 119ms/step - categorical_accuracy: 0.6871 - loss: 1.2866 - mean_io_u: 0.0552


642/Unknown  116s 119ms/step - categorical_accuracy: 0.6872 - loss: 1.2863 - mean_io_u: 0.0552


643/Unknown  116s 118ms/step - categorical_accuracy: 0.6873 - loss: 1.2859 - mean_io_u: 0.0552


644/Unknown  116s 118ms/step - categorical_accuracy: 0.6873 - loss: 1.2856 - mean_io_u: 0.0553


645/Unknown  116s 118ms/step - categorical_accuracy: 0.6874 - loss: 1.2853 - mean_io_u: 0.0553


646/Unknown  116s 118ms/step - categorical_accuracy: 0.6875 - loss: 1.2850 - mean_io_u: 0.0554


647/Unknown  116s 118ms/step - categorical_accuracy: 0.6875 - loss: 1.2846 - mean_io_u: 0.0554


648/Unknown  116s 118ms/step - categorical_accuracy: 0.6876 - loss: 1.2843 - mean_io_u: 0.0554


649/Unknown  116s 118ms/step - categorical_accuracy: 0.6877 - loss: 1.2840 - mean_io_u: 0.0555


650/Unknown  116s 118ms/step - categorical_accuracy: 0.6877 - loss: 1.2837 - mean_io_u: 0.0555


651/Unknown  116s 118ms/step - categorical_accuracy: 0.6878 - loss: 1.2833 - mean_io_u: 0.0555


652/Unknown  116s 118ms/step - categorical_accuracy: 0.6879 - loss: 1.2830 - mean_io_u: 0.0556


653/Unknown  117s 118ms/step - categorical_accuracy: 0.6879 - loss: 1.2827 - mean_io_u: 0.0556


654/Unknown  117s 118ms/step - categorical_accuracy: 0.6880 - loss: 1.2824 - mean_io_u: 0.0557


655/Unknown  117s 118ms/step - categorical_accuracy: 0.6881 - loss: 1.2821 - mean_io_u: 0.0557


656/Unknown  117s 118ms/step - categorical_accuracy: 0.6881 - loss: 1.2817 - mean_io_u: 0.0557


657/Unknown  117s 118ms/step - categorical_accuracy: 0.6882 - loss: 1.2814 - mean_io_u: 0.0558


658/Unknown  117s 118ms/step - categorical_accuracy: 0.6882 - loss: 1.2811 - mean_io_u: 0.0558


659/Unknown  117s 118ms/step - categorical_accuracy: 0.6883 - loss: 1.2808 - mean_io_u: 0.0559


660/Unknown  117s 117ms/step - categorical_accuracy: 0.6884 - loss: 1.2805 - mean_io_u: 0.0559


661/Unknown  117s 117ms/step - categorical_accuracy: 0.6884 - loss: 1.2802 - mean_io_u: 0.0559


662/Unknown  117s 117ms/step - categorical_accuracy: 0.6885 - loss: 1.2798 - mean_io_u: 0.0560


663/Unknown  117s 117ms/step - categorical_accuracy: 0.6886 - loss: 1.2795 - mean_io_u: 0.0560


664/Unknown  117s 117ms/step - categorical_accuracy: 0.6886 - loss: 1.2792 - mean_io_u: 0.0560


665/Unknown  117s 117ms/step - categorical_accuracy: 0.6887 - loss: 1.2789 - mean_io_u: 0.0561


666/Unknown  118s 117ms/step - categorical_accuracy: 0.6888 - loss: 1.2786 - mean_io_u: 0.0561


667/Unknown  118s 117ms/step - categorical_accuracy: 0.6888 - loss: 1.2782 - mean_io_u: 0.0562


668/Unknown  118s 117ms/step - categorical_accuracy: 0.6889 - loss: 1.2779 - mean_io_u: 0.0562


669/Unknown  118s 117ms/step - categorical_accuracy: 0.6890 - loss: 1.2776 - mean_io_u: 0.0562


670/Unknown  118s 117ms/step - categorical_accuracy: 0.6890 - loss: 1.2773 - mean_io_u: 0.0563


671/Unknown  118s 117ms/step - categorical_accuracy: 0.6891 - loss: 1.2770 - mean_io_u: 0.0563


672/Unknown  118s 117ms/step - categorical_accuracy: 0.6891 - loss: 1.2767 - mean_io_u: 0.0563


673/Unknown  118s 117ms/step - categorical_accuracy: 0.6892 - loss: 1.2764 - mean_io_u: 0.0564


674/Unknown  118s 117ms/step - categorical_accuracy: 0.6893 - loss: 1.2761 - mean_io_u: 0.0564


675/Unknown  118s 117ms/step - categorical_accuracy: 0.6893 - loss: 1.2757 - mean_io_u: 0.0564


676/Unknown  118s 117ms/step - categorical_accuracy: 0.6894 - loss: 1.2754 - mean_io_u: 0.0565


677/Unknown  118s 117ms/step - categorical_accuracy: 0.6895 - loss: 1.2751 - mean_io_u: 0.0565


678/Unknown  118s 116ms/step - categorical_accuracy: 0.6895 - loss: 1.2748 - mean_io_u: 0.0566


679/Unknown  119s 116ms/step - categorical_accuracy: 0.6896 - loss: 1.2745 - mean_io_u: 0.0566


680/Unknown  119s 116ms/step - categorical_accuracy: 0.6896 - loss: 1.2742 - mean_io_u: 0.0566


681/Unknown  119s 116ms/step - categorical_accuracy: 0.6897 - loss: 1.2739 - mean_io_u: 0.0567


682/Unknown  119s 116ms/step - categorical_accuracy: 0.6898 - loss: 1.2736 - mean_io_u: 0.0567


683/Unknown  119s 116ms/step - categorical_accuracy: 0.6898 - loss: 1.2733 - mean_io_u: 0.0567


684/Unknown  119s 116ms/step - categorical_accuracy: 0.6899 - loss: 1.2730 - mean_io_u: 0.0568


685/Unknown  119s 116ms/step - categorical_accuracy: 0.6900 - loss: 1.2727 - mean_io_u: 0.0568


686/Unknown  119s 116ms/step - categorical_accuracy: 0.6900 - loss: 1.2723 - mean_io_u: 0.0569


687/Unknown  119s 116ms/step - categorical_accuracy: 0.6901 - loss: 1.2720 - mean_io_u: 0.0569


688/Unknown  119s 116ms/step - categorical_accuracy: 0.6901 - loss: 1.2717 - mean_io_u: 0.0569


689/Unknown  119s 116ms/step - categorical_accuracy: 0.6902 - loss: 1.2714 - mean_io_u: 0.0570


690/Unknown  119s 116ms/step - categorical_accuracy: 0.6903 - loss: 1.2711 - mean_io_u: 0.0570


691/Unknown  120s 116ms/step - categorical_accuracy: 0.6903 - loss: 1.2708 - mean_io_u: 0.0570


692/Unknown  120s 116ms/step - categorical_accuracy: 0.6904 - loss: 1.2705 - mean_io_u: 0.0571


693/Unknown  120s 116ms/step - categorical_accuracy: 0.6904 - loss: 1.2702 - mean_io_u: 0.0571


694/Unknown  120s 116ms/step - categorical_accuracy: 0.6905 - loss: 1.2699 - mean_io_u: 0.0571


695/Unknown  120s 116ms/step - categorical_accuracy: 0.6906 - loss: 1.2696 - mean_io_u: 0.0572


696/Unknown  120s 115ms/step - categorical_accuracy: 0.6906 - loss: 1.2693 - mean_io_u: 0.0572


697/Unknown  120s 115ms/step - categorical_accuracy: 0.6907 - loss: 1.2690 - mean_io_u: 0.0573


698/Unknown  120s 115ms/step - categorical_accuracy: 0.6907 - loss: 1.2687 - mean_io_u: 0.0573


699/Unknown  120s 115ms/step - categorical_accuracy: 0.6908 - loss: 1.2685 - mean_io_u: 0.0573


700/Unknown  120s 115ms/step - categorical_accuracy: 0.6908 - loss: 1.2682 - mean_io_u: 0.0574


701/Unknown  120s 115ms/step - categorical_accuracy: 0.6909 - loss: 1.2679 - mean_io_u: 0.0574


702/Unknown  120s 115ms/step - categorical_accuracy: 0.6910 - loss: 1.2676 - mean_io_u: 0.0574


703/Unknown  120s 115ms/step - categorical_accuracy: 0.6910 - loss: 1.2673 - mean_io_u: 0.0575


704/Unknown  120s 115ms/step - categorical_accuracy: 0.6911 - loss: 1.2670 - mean_io_u: 0.0575


705/Unknown  120s 115ms/step - categorical_accuracy: 0.6911 - loss: 1.2667 - mean_io_u: 0.0575


706/Unknown  121s 115ms/step - categorical_accuracy: 0.6912 - loss: 1.2664 - mean_io_u: 0.0576


707/Unknown  121s 115ms/step - categorical_accuracy: 0.6912 - loss: 1.2661 - mean_io_u: 0.0576


708/Unknown  121s 115ms/step - categorical_accuracy: 0.6913 - loss: 1.2658 - mean_io_u: 0.0576


709/Unknown  121s 115ms/step - categorical_accuracy: 0.6914 - loss: 1.2656 - mean_io_u: 0.0577


710/Unknown  121s 114ms/step - categorical_accuracy: 0.6914 - loss: 1.2653 - mean_io_u: 0.0577


711/Unknown  121s 114ms/step - categorical_accuracy: 0.6915 - loss: 1.2650 - mean_io_u: 0.0577


712/Unknown  121s 114ms/step - categorical_accuracy: 0.6915 - loss: 1.2647 - mean_io_u: 0.0578


713/Unknown  121s 114ms/step - categorical_accuracy: 0.6916 - loss: 1.2644 - mean_io_u: 0.0578


714/Unknown  121s 114ms/step - categorical_accuracy: 0.6916 - loss: 1.2641 - mean_io_u: 0.0579


715/Unknown  121s 114ms/step - categorical_accuracy: 0.6917 - loss: 1.2639 - mean_io_u: 0.0579


716/Unknown  121s 114ms/step - categorical_accuracy: 0.6918 - loss: 1.2636 - mean_io_u: 0.0579


717/Unknown  121s 114ms/step - categorical_accuracy: 0.6918 - loss: 1.2633 - mean_io_u: 0.0580


718/Unknown  121s 114ms/step - categorical_accuracy: 0.6919 - loss: 1.2630 - mean_io_u: 0.0580


719/Unknown  121s 114ms/step - categorical_accuracy: 0.6919 - loss: 1.2627 - mean_io_u: 0.0580


720/Unknown  121s 114ms/step - categorical_accuracy: 0.6920 - loss: 1.2624 - mean_io_u: 0.0581


721/Unknown  122s 114ms/step - categorical_accuracy: 0.6920 - loss: 1.2622 - mean_io_u: 0.0581


722/Unknown  122s 114ms/step - categorical_accuracy: 0.6921 - loss: 1.2619 - mean_io_u: 0.0581


723/Unknown  122s 114ms/step - categorical_accuracy: 0.6921 - loss: 1.2616 - mean_io_u: 0.0582


724/Unknown  122s 114ms/step - categorical_accuracy: 0.6922 - loss: 1.2613 - mean_io_u: 0.0582


725/Unknown  122s 114ms/step - categorical_accuracy: 0.6923 - loss: 1.2610 - mean_io_u: 0.0582


726/Unknown  122s 113ms/step - categorical_accuracy: 0.6923 - loss: 1.2607 - mean_io_u: 0.0583


727/Unknown  122s 113ms/step - categorical_accuracy: 0.6924 - loss: 1.2605 - mean_io_u: 0.0583


728/Unknown  122s 113ms/step - categorical_accuracy: 0.6924 - loss: 1.2602 - mean_io_u: 0.0583


729/Unknown  122s 113ms/step - categorical_accuracy: 0.6925 - loss: 1.2599 - mean_io_u: 0.0584


730/Unknown  122s 113ms/step - categorical_accuracy: 0.6925 - loss: 1.2596 - mean_io_u: 0.0584


731/Unknown  122s 113ms/step - categorical_accuracy: 0.6926 - loss: 1.2594 - mean_io_u: 0.0584


732/Unknown  122s 113ms/step - categorical_accuracy: 0.6926 - loss: 1.2591 - mean_io_u: 0.0585


733/Unknown  122s 113ms/step - categorical_accuracy: 0.6927 - loss: 1.2588 - mean_io_u: 0.0585


734/Unknown  123s 113ms/step - categorical_accuracy: 0.6927 - loss: 1.2585 - mean_io_u: 0.0585


735/Unknown  123s 113ms/step - categorical_accuracy: 0.6928 - loss: 1.2583 - mean_io_u: 0.0586


736/Unknown  123s 113ms/step - categorical_accuracy: 0.6929 - loss: 1.2580 - mean_io_u: 0.0586


737/Unknown  123s 113ms/step - categorical_accuracy: 0.6929 - loss: 1.2577 - mean_io_u: 0.0586


738/Unknown  123s 113ms/step - categorical_accuracy: 0.6930 - loss: 1.2574 - mean_io_u: 0.0587


739/Unknown  123s 113ms/step - categorical_accuracy: 0.6930 - loss: 1.2572 - mean_io_u: 0.0587


740/Unknown  123s 113ms/step - categorical_accuracy: 0.6931 - loss: 1.2569 - mean_io_u: 0.0588


741/Unknown  123s 113ms/step - categorical_accuracy: 0.6931 - loss: 1.2566 - mean_io_u: 0.0588


742/Unknown  123s 113ms/step - categorical_accuracy: 0.6932 - loss: 1.2563 - mean_io_u: 0.0588


743/Unknown  123s 113ms/step - categorical_accuracy: 0.6932 - loss: 1.2561 - mean_io_u: 0.0589


744/Unknown  123s 113ms/step - categorical_accuracy: 0.6933 - loss: 1.2558 - mean_io_u: 0.0589


745/Unknown  123s 113ms/step - categorical_accuracy: 0.6933 - loss: 1.2555 - mean_io_u: 0.0589


746/Unknown  124s 113ms/step - categorical_accuracy: 0.6934 - loss: 1.2553 - mean_io_u: 0.0590


747/Unknown  124s 113ms/step - categorical_accuracy: 0.6934 - loss: 1.2550 - mean_io_u: 0.0590


748/Unknown  124s 113ms/step - categorical_accuracy: 0.6935 - loss: 1.2547 - mean_io_u: 0.0590


749/Unknown  124s 113ms/step - categorical_accuracy: 0.6936 - loss: 1.2544 - mean_io_u: 0.0591


750/Unknown  124s 113ms/step - categorical_accuracy: 0.6936 - loss: 1.2542 - mean_io_u: 0.0591


751/Unknown  124s 112ms/step - categorical_accuracy: 0.6937 - loss: 1.2539 - mean_io_u: 0.0591


752/Unknown  124s 112ms/step - categorical_accuracy: 0.6937 - loss: 1.2536 - mean_io_u: 0.0592


753/Unknown  124s 112ms/step - categorical_accuracy: 0.6938 - loss: 1.2534 - mean_io_u: 0.0592


754/Unknown  124s 112ms/step - categorical_accuracy: 0.6938 - loss: 1.2531 - mean_io_u: 0.0592


755/Unknown  124s 112ms/step - categorical_accuracy: 0.6939 - loss: 1.2528 - mean_io_u: 0.0593


756/Unknown  124s 112ms/step - categorical_accuracy: 0.6939 - loss: 1.2526 - mean_io_u: 0.0593


757/Unknown  124s 112ms/step - categorical_accuracy: 0.6940 - loss: 1.2523 - mean_io_u: 0.0593


758/Unknown  124s 112ms/step - categorical_accuracy: 0.6940 - loss: 1.2520 - mean_io_u: 0.0594


759/Unknown  125s 112ms/step - categorical_accuracy: 0.6941 - loss: 1.2518 - mean_io_u: 0.0594


760/Unknown  125s 112ms/step - categorical_accuracy: 0.6941 - loss: 1.2515 - mean_io_u: 0.0594


761/Unknown  125s 112ms/step - categorical_accuracy: 0.6942 - loss: 1.2512 - mean_io_u: 0.0595


762/Unknown  125s 112ms/step - categorical_accuracy: 0.6942 - loss: 1.2510 - mean_io_u: 0.0595


763/Unknown  125s 112ms/step - categorical_accuracy: 0.6943 - loss: 1.2507 - mean_io_u: 0.0595


764/Unknown  125s 112ms/step - categorical_accuracy: 0.6943 - loss: 1.2504 - mean_io_u: 0.0596


765/Unknown  125s 112ms/step - categorical_accuracy: 0.6944 - loss: 1.2502 - mean_io_u: 0.0596


766/Unknown  125s 112ms/step - categorical_accuracy: 0.6944 - loss: 1.2499 - mean_io_u: 0.0596


767/Unknown  125s 112ms/step - categorical_accuracy: 0.6945 - loss: 1.2496 - mean_io_u: 0.0597


768/Unknown  125s 112ms/step - categorical_accuracy: 0.6946 - loss: 1.2494 - mean_io_u: 0.0597


769/Unknown  125s 112ms/step - categorical_accuracy: 0.6946 - loss: 1.2491 - mean_io_u: 0.0597


770/Unknown  125s 112ms/step - categorical_accuracy: 0.6947 - loss: 1.2489 - mean_io_u: 0.0598


771/Unknown  126s 112ms/step - categorical_accuracy: 0.6947 - loss: 1.2486 - mean_io_u: 0.0598


772/Unknown  126s 112ms/step - categorical_accuracy: 0.6948 - loss: 1.2483 - mean_io_u: 0.0598


773/Unknown  126s 112ms/step - categorical_accuracy: 0.6948 - loss: 1.2481 - mean_io_u: 0.0599


774/Unknown  126s 112ms/step - categorical_accuracy: 0.6949 - loss: 1.2478 - mean_io_u: 0.0599


775/Unknown  126s 111ms/step - categorical_accuracy: 0.6949 - loss: 1.2475 - mean_io_u: 0.0599


776/Unknown  126s 111ms/step - categorical_accuracy: 0.6950 - loss: 1.2473 - mean_io_u: 0.0600


777/Unknown  126s 111ms/step - categorical_accuracy: 0.6950 - loss: 1.2470 - mean_io_u: 0.0600


778/Unknown  126s 111ms/step - categorical_accuracy: 0.6951 - loss: 1.2468 - mean_io_u: 0.0601


779/Unknown  126s 111ms/step - categorical_accuracy: 0.6951 - loss: 1.2465 - mean_io_u: 0.0601


780/Unknown  126s 111ms/step - categorical_accuracy: 0.6952 - loss: 1.2463 - mean_io_u: 0.0601


781/Unknown  126s 111ms/step - categorical_accuracy: 0.6952 - loss: 1.2460 - mean_io_u: 0.0602


782/Unknown  126s 111ms/step - categorical_accuracy: 0.6953 - loss: 1.2457 - mean_io_u: 0.0602


783/Unknown  127s 111ms/step - categorical_accuracy: 0.6953 - loss: 1.2455 - mean_io_u: 0.0602


784/Unknown  127s 111ms/step - categorical_accuracy: 0.6954 - loss: 1.2452 - mean_io_u: 0.0603


785/Unknown  127s 111ms/step - categorical_accuracy: 0.6954 - loss: 1.2450 - mean_io_u: 0.0603


786/Unknown  127s 111ms/step - categorical_accuracy: 0.6955 - loss: 1.2447 - mean_io_u: 0.0603


787/Unknown  127s 111ms/step - categorical_accuracy: 0.6955 - loss: 1.2444 - mean_io_u: 0.0604


788/Unknown  127s 111ms/step - categorical_accuracy: 0.6956 - loss: 1.2442 - mean_io_u: 0.0604


789/Unknown  127s 111ms/step - categorical_accuracy: 0.6956 - loss: 1.2439 - mean_io_u: 0.0604


790/Unknown  127s 111ms/step - categorical_accuracy: 0.6957 - loss: 1.2437 - mean_io_u: 0.0605


791/Unknown  127s 111ms/step - categorical_accuracy: 0.6957 - loss: 1.2434 - mean_io_u: 0.0605


792/Unknown  127s 111ms/step - categorical_accuracy: 0.6958 - loss: 1.2432 - mean_io_u: 0.0605


793/Unknown  127s 111ms/step - categorical_accuracy: 0.6958 - loss: 1.2429 - mean_io_u: 0.0606


794/Unknown  127s 111ms/step - categorical_accuracy: 0.6959 - loss: 1.2427 - mean_io_u: 0.0606


795/Unknown  128s 111ms/step - categorical_accuracy: 0.6959 - loss: 1.2424 - mean_io_u: 0.0606


796/Unknown  128s 111ms/step - categorical_accuracy: 0.6960 - loss: 1.2422 - mean_io_u: 0.0607


797/Unknown  128s 111ms/step - categorical_accuracy: 0.6960 - loss: 1.2419 - mean_io_u: 0.0607


798/Unknown  128s 111ms/step - categorical_accuracy: 0.6961 - loss: 1.2416 - mean_io_u: 0.0607


799/Unknown  128s 111ms/step - categorical_accuracy: 0.6961 - loss: 1.2414 - mean_io_u: 0.0608


800/Unknown  128s 111ms/step - categorical_accuracy: 0.6962 - loss: 1.2411 - mean_io_u: 0.0608


801/Unknown  128s 110ms/step - categorical_accuracy: 0.6962 - loss: 1.2409 - mean_io_u: 0.0608


802/Unknown  128s 110ms/step - categorical_accuracy: 0.6963 - loss: 1.2406 - mean_io_u: 0.0609


803/Unknown  128s 110ms/step - categorical_accuracy: 0.6963 - loss: 1.2404 - mean_io_u: 0.0609


804/Unknown  128s 110ms/step - categorical_accuracy: 0.6964 - loss: 1.2401 - mean_io_u: 0.0609


805/Unknown  128s 110ms/step - categorical_accuracy: 0.6964 - loss: 1.2399 - mean_io_u: 0.0610


806/Unknown  128s 110ms/step - categorical_accuracy: 0.6965 - loss: 1.2396 - mean_io_u: 0.0610


807/Unknown  128s 110ms/step - categorical_accuracy: 0.6965 - loss: 1.2394 - mean_io_u: 0.0610


808/Unknown  129s 110ms/step - categorical_accuracy: 0.6966 - loss: 1.2391 - mean_io_u: 0.0611


809/Unknown  129s 110ms/step - categorical_accuracy: 0.6966 - loss: 1.2389 - mean_io_u: 0.0611


810/Unknown  129s 110ms/step - categorical_accuracy: 0.6967 - loss: 1.2386 - mean_io_u: 0.0611


811/Unknown  129s 110ms/step - categorical_accuracy: 0.6967 - loss: 1.2384 - mean_io_u: 0.0612


812/Unknown  129s 110ms/step - categorical_accuracy: 0.6968 - loss: 1.2381 - mean_io_u: 0.0612


813/Unknown  129s 110ms/step - categorical_accuracy: 0.6968 - loss: 1.2379 - mean_io_u: 0.0612


814/Unknown  129s 110ms/step - categorical_accuracy: 0.6969 - loss: 1.2376 - mean_io_u: 0.0613


815/Unknown  129s 110ms/step - categorical_accuracy: 0.6969 - loss: 1.2374 - mean_io_u: 0.0613


816/Unknown  129s 110ms/step - categorical_accuracy: 0.6970 - loss: 1.2371 - mean_io_u: 0.0613


817/Unknown  129s 110ms/step - categorical_accuracy: 0.6970 - loss: 1.2369 - mean_io_u: 0.0614


818/Unknown  129s 110ms/step - categorical_accuracy: 0.6971 - loss: 1.2366 - mean_io_u: 0.0614


819/Unknown  129s 110ms/step - categorical_accuracy: 0.6971 - loss: 1.2364 - mean_io_u: 0.0614


820/Unknown  129s 110ms/step - categorical_accuracy: 0.6972 - loss: 1.2361 - mean_io_u: 0.0615


821/Unknown  130s 110ms/step - categorical_accuracy: 0.6972 - loss: 1.2359 - mean_io_u: 0.0615


822/Unknown  130s 110ms/step - categorical_accuracy: 0.6973 - loss: 1.2356 - mean_io_u: 0.0615


823/Unknown  130s 110ms/step - categorical_accuracy: 0.6973 - loss: 1.2354 - mean_io_u: 0.0616


824/Unknown  130s 109ms/step - categorical_accuracy: 0.6974 - loss: 1.2352 - mean_io_u: 0.0616


825/Unknown  130s 109ms/step - categorical_accuracy: 0.6974 - loss: 1.2349 - mean_io_u: 0.0616


826/Unknown  130s 109ms/step - categorical_accuracy: 0.6975 - loss: 1.2347 - mean_io_u: 0.0617


827/Unknown  130s 109ms/step - categorical_accuracy: 0.6975 - loss: 1.2344 - mean_io_u: 0.0617


828/Unknown  130s 109ms/step - categorical_accuracy: 0.6976 - loss: 1.2342 - mean_io_u: 0.0617


829/Unknown  130s 109ms/step - categorical_accuracy: 0.6976 - loss: 1.2339 - mean_io_u: 0.0618


830/Unknown  130s 109ms/step - categorical_accuracy: 0.6977 - loss: 1.2337 - mean_io_u: 0.0618


831/Unknown  130s 109ms/step - categorical_accuracy: 0.6977 - loss: 1.2334 - mean_io_u: 0.0618


832/Unknown  130s 109ms/step - categorical_accuracy: 0.6978 - loss: 1.2332 - mean_io_u: 0.0619


833/Unknown  130s 109ms/step - categorical_accuracy: 0.6978 - loss: 1.2330 - mean_io_u: 0.0619


834/Unknown  131s 109ms/step - categorical_accuracy: 0.6979 - loss: 1.2327 - mean_io_u: 0.0619


835/Unknown  131s 109ms/step - categorical_accuracy: 0.6979 - loss: 1.2325 - mean_io_u: 0.0620


836/Unknown  131s 109ms/step - categorical_accuracy: 0.6980 - loss: 1.2322 - mean_io_u: 0.0620


837/Unknown  131s 109ms/step - categorical_accuracy: 0.6980 - loss: 1.2320 - mean_io_u: 0.0620


838/Unknown  131s 109ms/step - categorical_accuracy: 0.6981 - loss: 1.2317 - mean_io_u: 0.0621


839/Unknown  131s 109ms/step - categorical_accuracy: 0.6981 - loss: 1.2315 - mean_io_u: 0.0621


840/Unknown  131s 109ms/step - categorical_accuracy: 0.6981 - loss: 1.2313 - mean_io_u: 0.0621


841/Unknown  131s 109ms/step - categorical_accuracy: 0.6982 - loss: 1.2310 - mean_io_u: 0.0622


842/Unknown  131s 109ms/step - categorical_accuracy: 0.6982 - loss: 1.2308 - mean_io_u: 0.0622


843/Unknown  131s 109ms/step - categorical_accuracy: 0.6983 - loss: 1.2305 - mean_io_u: 0.0622


844/Unknown  131s 109ms/step - categorical_accuracy: 0.6983 - loss: 1.2303 - mean_io_u: 0.0623


845/Unknown  131s 109ms/step - categorical_accuracy: 0.6984 - loss: 1.2301 - mean_io_u: 0.0623


846/Unknown  131s 109ms/step - categorical_accuracy: 0.6984 - loss: 1.2298 - mean_io_u: 0.0623


847/Unknown  131s 109ms/step - categorical_accuracy: 0.6985 - loss: 1.2296 - mean_io_u: 0.0624


848/Unknown  132s 109ms/step - categorical_accuracy: 0.6985 - loss: 1.2293 - mean_io_u: 0.0624


849/Unknown  132s 108ms/step - categorical_accuracy: 0.6986 - loss: 1.2291 - mean_io_u: 0.0624


850/Unknown  132s 108ms/step - categorical_accuracy: 0.6986 - loss: 1.2289 - mean_io_u: 0.0625


851/Unknown  132s 108ms/step - categorical_accuracy: 0.6987 - loss: 1.2286 - mean_io_u: 0.0625


852/Unknown  132s 108ms/step - categorical_accuracy: 0.6987 - loss: 1.2284 - mean_io_u: 0.0625


853/Unknown  132s 108ms/step - categorical_accuracy: 0.6988 - loss: 1.2282 - mean_io_u: 0.0626


854/Unknown  132s 108ms/step - categorical_accuracy: 0.6988 - loss: 1.2279 - mean_io_u: 0.0626


855/Unknown  132s 108ms/step - categorical_accuracy: 0.6988 - loss: 1.2277 - mean_io_u: 0.0626


856/Unknown  132s 108ms/step - categorical_accuracy: 0.6989 - loss: 1.2275 - mean_io_u: 0.0627


857/Unknown  132s 108ms/step - categorical_accuracy: 0.6989 - loss: 1.2272 - mean_io_u: 0.0627


858/Unknown  132s 108ms/step - categorical_accuracy: 0.6990 - loss: 1.2270 - mean_io_u: 0.0627


859/Unknown  132s 108ms/step - categorical_accuracy: 0.6990 - loss: 1.2268 - mean_io_u: 0.0628


860/Unknown  132s 108ms/step - categorical_accuracy: 0.6991 - loss: 1.2265 - mean_io_u: 0.0628


861/Unknown  132s 108ms/step - categorical_accuracy: 0.6991 - loss: 1.2263 - mean_io_u: 0.0628


862/Unknown  132s 108ms/step - categorical_accuracy: 0.6992 - loss: 1.2261 - mean_io_u: 0.0629


863/Unknown  133s 108ms/step - categorical_accuracy: 0.6992 - loss: 1.2258 - mean_io_u: 0.0629


864/Unknown  133s 108ms/step - categorical_accuracy: 0.6993 - loss: 1.2256 - mean_io_u: 0.0629


865/Unknown  133s 108ms/step - categorical_accuracy: 0.6993 - loss: 1.2254 - mean_io_u: 0.0630


866/Unknown  133s 108ms/step - categorical_accuracy: 0.6994 - loss: 1.2251 - mean_io_u: 0.0630


867/Unknown  133s 108ms/step - categorical_accuracy: 0.6994 - loss: 1.2249 - mean_io_u: 0.0630


868/Unknown  133s 108ms/step - categorical_accuracy: 0.6994 - loss: 1.2247 - mean_io_u: 0.0631


869/Unknown  133s 108ms/step - categorical_accuracy: 0.6995 - loss: 1.2244 - mean_io_u: 0.0631


870/Unknown  133s 108ms/step - categorical_accuracy: 0.6995 - loss: 1.2242 - mean_io_u: 0.0631


871/Unknown  133s 107ms/step - categorical_accuracy: 0.6996 - loss: 1.2240 - mean_io_u: 0.0632


872/Unknown  133s 107ms/step - categorical_accuracy: 0.6996 - loss: 1.2238 - mean_io_u: 0.0632


873/Unknown  133s 107ms/step - categorical_accuracy: 0.6997 - loss: 1.2235 - mean_io_u: 0.0632


874/Unknown  133s 107ms/step - categorical_accuracy: 0.6997 - loss: 1.2233 - mean_io_u: 0.0632


875/Unknown  133s 107ms/step - categorical_accuracy: 0.6998 - loss: 1.2231 - mean_io_u: 0.0633


876/Unknown  134s 107ms/step - categorical_accuracy: 0.6998 - loss: 1.2228 - mean_io_u: 0.0633


877/Unknown  134s 107ms/step - categorical_accuracy: 0.6998 - loss: 1.2226 - mean_io_u: 0.0633


878/Unknown  134s 107ms/step - categorical_accuracy: 0.6999 - loss: 1.2224 - mean_io_u: 0.0634


879/Unknown  134s 107ms/step - categorical_accuracy: 0.6999 - loss: 1.2222 - mean_io_u: 0.0634


880/Unknown  134s 107ms/step - categorical_accuracy: 0.7000 - loss: 1.2219 - mean_io_u: 0.0634


881/Unknown  134s 107ms/step - categorical_accuracy: 0.7000 - loss: 1.2217 - mean_io_u: 0.0635


882/Unknown  134s 107ms/step - categorical_accuracy: 0.7001 - loss: 1.2215 - mean_io_u: 0.0635


883/Unknown  134s 107ms/step - categorical_accuracy: 0.7001 - loss: 1.2213 - mean_io_u: 0.0635


884/Unknown  134s 107ms/step - categorical_accuracy: 0.7002 - loss: 1.2210 - mean_io_u: 0.0636


885/Unknown  134s 107ms/step - categorical_accuracy: 0.7002 - loss: 1.2208 - mean_io_u: 0.0636


886/Unknown  134s 107ms/step - categorical_accuracy: 0.7002 - loss: 1.2206 - mean_io_u: 0.0636


887/Unknown  134s 107ms/step - categorical_accuracy: 0.7003 - loss: 1.2204 - mean_io_u: 0.0637


888/Unknown  134s 107ms/step - categorical_accuracy: 0.7003 - loss: 1.2201 - mean_io_u: 0.0637


889/Unknown  134s 107ms/step - categorical_accuracy: 0.7004 - loss: 1.2199 - mean_io_u: 0.0637


890/Unknown  135s 107ms/step - categorical_accuracy: 0.7004 - loss: 1.2197 - mean_io_u: 0.0638


891/Unknown  135s 107ms/step - categorical_accuracy: 0.7005 - loss: 1.2195 - mean_io_u: 0.0638


892/Unknown  135s 107ms/step - categorical_accuracy: 0.7005 - loss: 1.2192 - mean_io_u: 0.0638


893/Unknown  135s 107ms/step - categorical_accuracy: 0.7005 - loss: 1.2190 - mean_io_u: 0.0639


894/Unknown  135s 107ms/step - categorical_accuracy: 0.7006 - loss: 1.2188 - mean_io_u: 0.0639


895/Unknown  135s 107ms/step - categorical_accuracy: 0.7006 - loss: 1.2186 - mean_io_u: 0.0639


896/Unknown  135s 107ms/step - categorical_accuracy: 0.7007 - loss: 1.2183 - mean_io_u: 0.0640


897/Unknown  135s 107ms/step - categorical_accuracy: 0.7007 - loss: 1.2181 - mean_io_u: 0.0640


898/Unknown  135s 107ms/step - categorical_accuracy: 0.7008 - loss: 1.2179 - mean_io_u: 0.0640


899/Unknown  135s 106ms/step - categorical_accuracy: 0.7008 - loss: 1.2177 - mean_io_u: 0.0641


900/Unknown  135s 106ms/step - categorical_accuracy: 0.7009 - loss: 1.2175 - mean_io_u: 0.0641


901/Unknown  135s 107ms/step - categorical_accuracy: 0.7009 - loss: 1.2172 - mean_io_u: 0.0641


902/Unknown  136s 106ms/step - categorical_accuracy: 0.7009 - loss: 1.2170 - mean_io_u: 0.0642


903/Unknown  136s 106ms/step - categorical_accuracy: 0.7010 - loss: 1.2168 - mean_io_u: 0.0642


904/Unknown  136s 106ms/step - categorical_accuracy: 0.7010 - loss: 1.2166 - mean_io_u: 0.0642


905/Unknown  136s 106ms/step - categorical_accuracy: 0.7011 - loss: 1.2164 - mean_io_u: 0.0642


906/Unknown  136s 106ms/step - categorical_accuracy: 0.7011 - loss: 1.2161 - mean_io_u: 0.0643


907/Unknown  136s 106ms/step - categorical_accuracy: 0.7012 - loss: 1.2159 - mean_io_u: 0.0643


908/Unknown  136s 106ms/step - categorical_accuracy: 0.7012 - loss: 1.2157 - mean_io_u: 0.0643


909/Unknown  136s 106ms/step - categorical_accuracy: 0.7012 - loss: 1.2155 - mean_io_u: 0.0644


910/Unknown  136s 106ms/step - categorical_accuracy: 0.7013 - loss: 1.2153 - mean_io_u: 0.0644


911/Unknown  136s 106ms/step - categorical_accuracy: 0.7013 - loss: 1.2150 - mean_io_u: 0.0644


912/Unknown  136s 106ms/step - categorical_accuracy: 0.7014 - loss: 1.2148 - mean_io_u: 0.0645


913/Unknown  136s 106ms/step - categorical_accuracy: 0.7014 - loss: 1.2146 - mean_io_u: 0.0645


914/Unknown  136s 106ms/step - categorical_accuracy: 0.7015 - loss: 1.2144 - mean_io_u: 0.0645


915/Unknown  136s 106ms/step - categorical_accuracy: 0.7015 - loss: 1.2142 - mean_io_u: 0.0646


916/Unknown  137s 106ms/step - categorical_accuracy: 0.7015 - loss: 1.2140 - mean_io_u: 0.0646


917/Unknown  137s 106ms/step - categorical_accuracy: 0.7016 - loss: 1.2137 - mean_io_u: 0.0646


918/Unknown  137s 106ms/step - categorical_accuracy: 0.7016 - loss: 1.2135 - mean_io_u: 0.0647


919/Unknown  137s 106ms/step - categorical_accuracy: 0.7017 - loss: 1.2133 - mean_io_u: 0.0647


920/Unknown  137s 106ms/step - categorical_accuracy: 0.7017 - loss: 1.2131 - mean_io_u: 0.0647


921/Unknown  137s 106ms/step - categorical_accuracy: 0.7018 - loss: 1.2129 - mean_io_u: 0.0648


922/Unknown  137s 106ms/step - categorical_accuracy: 0.7018 - loss: 1.2127 - mean_io_u: 0.0648


923/Unknown  137s 106ms/step - categorical_accuracy: 0.7018 - loss: 1.2124 - mean_io_u: 0.0648


924/Unknown  137s 106ms/step - categorical_accuracy: 0.7019 - loss: 1.2122 - mean_io_u: 0.0648


925/Unknown  137s 106ms/step - categorical_accuracy: 0.7019 - loss: 1.2120 - mean_io_u: 0.0649


926/Unknown  137s 106ms/step - categorical_accuracy: 0.7020 - loss: 1.2118 - mean_io_u: 0.0649


927/Unknown  138s 106ms/step - categorical_accuracy: 0.7020 - loss: 1.2116 - mean_io_u: 0.0649


928/Unknown  138s 106ms/step - categorical_accuracy: 0.7020 - loss: 1.2114 - mean_io_u: 0.0650


929/Unknown  138s 106ms/step - categorical_accuracy: 0.7021 - loss: 1.2112 - mean_io_u: 0.0650


930/Unknown  138s 106ms/step - categorical_accuracy: 0.7021 - loss: 1.2109 - mean_io_u: 0.0650


931/Unknown  138s 106ms/step - categorical_accuracy: 0.7022 - loss: 1.2107 - mean_io_u: 0.0651


932/Unknown  138s 106ms/step - categorical_accuracy: 0.7022 - loss: 1.2105 - mean_io_u: 0.0651


933/Unknown  138s 106ms/step - categorical_accuracy: 0.7023 - loss: 1.2103 - mean_io_u: 0.0651


934/Unknown  138s 106ms/step - categorical_accuracy: 0.7023 - loss: 1.2101 - mean_io_u: 0.0652


935/Unknown  138s 106ms/step - categorical_accuracy: 0.7023 - loss: 1.2099 - mean_io_u: 0.0652


936/Unknown  138s 106ms/step - categorical_accuracy: 0.7024 - loss: 1.2097 - mean_io_u: 0.0652


937/Unknown  138s 106ms/step - categorical_accuracy: 0.7024 - loss: 1.2095 - mean_io_u: 0.0652


938/Unknown  138s 105ms/step - categorical_accuracy: 0.7025 - loss: 1.2093 - mean_io_u: 0.0653


939/Unknown  139s 105ms/step - categorical_accuracy: 0.7025 - loss: 1.2091 - mean_io_u: 0.0653


940/Unknown  139s 105ms/step - categorical_accuracy: 0.7025 - loss: 1.2088 - mean_io_u: 0.0653


941/Unknown  139s 105ms/step - categorical_accuracy: 0.7026 - loss: 1.2086 - mean_io_u: 0.0654


942/Unknown  139s 105ms/step - categorical_accuracy: 0.7026 - loss: 1.2084 - mean_io_u: 0.0654


943/Unknown  139s 105ms/step - categorical_accuracy: 0.7027 - loss: 1.2082 - mean_io_u: 0.0654


944/Unknown  139s 105ms/step - categorical_accuracy: 0.7027 - loss: 1.2080 - mean_io_u: 0.0655


945/Unknown  139s 105ms/step - categorical_accuracy: 0.7027 - loss: 1.2078 - mean_io_u: 0.0655


946/Unknown  139s 105ms/step - categorical_accuracy: 0.7028 - loss: 1.2076 - mean_io_u: 0.0655


947/Unknown  139s 105ms/step - categorical_accuracy: 0.7028 - loss: 1.2074 - mean_io_u: 0.0656


948/Unknown  139s 105ms/step - categorical_accuracy: 0.7029 - loss: 1.2072 - mean_io_u: 0.0656


949/Unknown  139s 105ms/step - categorical_accuracy: 0.7029 - loss: 1.2070 - mean_io_u: 0.0656


950/Unknown  140s 105ms/step - categorical_accuracy: 0.7029 - loss: 1.2068 - mean_io_u: 0.0656


951/Unknown  140s 105ms/step - categorical_accuracy: 0.7030 - loss: 1.2066 - mean_io_u: 0.0657


952/Unknown  140s 105ms/step - categorical_accuracy: 0.7030 - loss: 1.2064 - mean_io_u: 0.0657


953/Unknown  140s 105ms/step - categorical_accuracy: 0.7031 - loss: 1.2062 - mean_io_u: 0.0657


954/Unknown  140s 105ms/step - categorical_accuracy: 0.7031 - loss: 1.2060 - mean_io_u: 0.0658


955/Unknown  140s 105ms/step - categorical_accuracy: 0.7031 - loss: 1.2058 - mean_io_u: 0.0658


956/Unknown  140s 105ms/step - categorical_accuracy: 0.7032 - loss: 1.2056 - mean_io_u: 0.0658


957/Unknown  140s 105ms/step - categorical_accuracy: 0.7032 - loss: 1.2054 - mean_io_u: 0.0659


958/Unknown  140s 105ms/step - categorical_accuracy: 0.7033 - loss: 1.2052 - mean_io_u: 0.0659


959/Unknown  140s 105ms/step - categorical_accuracy: 0.7033 - loss: 1.2050 - mean_io_u: 0.0659


960/Unknown  140s 105ms/step - categorical_accuracy: 0.7033 - loss: 1.2048 - mean_io_u: 0.0660


961/Unknown  140s 105ms/step - categorical_accuracy: 0.7034 - loss: 1.2046 - mean_io_u: 0.0660


962/Unknown  140s 105ms/step - categorical_accuracy: 0.7034 - loss: 1.2044 - mean_io_u: 0.0660


963/Unknown  141s 105ms/step - categorical_accuracy: 0.7034 - loss: 1.2041 - mean_io_u: 0.0660


964/Unknown  141s 105ms/step - categorical_accuracy: 0.7035 - loss: 1.2039 - mean_io_u: 0.0661


965/Unknown  141s 105ms/step - categorical_accuracy: 0.7035 - loss: 1.2037 - mean_io_u: 0.0661


966/Unknown  141s 105ms/step - categorical_accuracy: 0.7036 - loss: 1.2035 - mean_io_u: 0.0661


967/Unknown  141s 105ms/step - categorical_accuracy: 0.7036 - loss: 1.2033 - mean_io_u: 0.0662


968/Unknown  141s 105ms/step - categorical_accuracy: 0.7036 - loss: 1.2031 - mean_io_u: 0.0662


969/Unknown  141s 105ms/step - categorical_accuracy: 0.7037 - loss: 1.2029 - mean_io_u: 0.0662


970/Unknown  141s 105ms/step - categorical_accuracy: 0.7037 - loss: 1.2027 - mean_io_u: 0.0663


971/Unknown  141s 105ms/step - categorical_accuracy: 0.7038 - loss: 1.2025 - mean_io_u: 0.0663


972/Unknown  141s 105ms/step - categorical_accuracy: 0.7038 - loss: 1.2023 - mean_io_u: 0.0663


973/Unknown  141s 105ms/step - categorical_accuracy: 0.7038 - loss: 1.2021 - mean_io_u: 0.0663


974/Unknown  141s 105ms/step - categorical_accuracy: 0.7039 - loss: 1.2019 - mean_io_u: 0.0664


975/Unknown  141s 105ms/step - categorical_accuracy: 0.7039 - loss: 1.2017 - mean_io_u: 0.0664


976/Unknown  142s 105ms/step - categorical_accuracy: 0.7040 - loss: 1.2015 - mean_io_u: 0.0664


977/Unknown  142s 105ms/step - categorical_accuracy: 0.7040 - loss: 1.2013 - mean_io_u: 0.0665


978/Unknown  142s 104ms/step - categorical_accuracy: 0.7040 - loss: 1.2011 - mean_io_u: 0.0665


979/Unknown  142s 104ms/step - categorical_accuracy: 0.7041 - loss: 1.2009 - mean_io_u: 0.0665


980/Unknown  142s 104ms/step - categorical_accuracy: 0.7041 - loss: 1.2007 - mean_io_u: 0.0666


981/Unknown  142s 104ms/step - categorical_accuracy: 0.7041 - loss: 1.2006 - mean_io_u: 0.0666


982/Unknown  142s 104ms/step - categorical_accuracy: 0.7042 - loss: 1.2004 - mean_io_u: 0.0666


983/Unknown  142s 104ms/step - categorical_accuracy: 0.7042 - loss: 1.2002 - mean_io_u: 0.0666


984/Unknown  142s 104ms/step - categorical_accuracy: 0.7043 - loss: 1.2000 - mean_io_u: 0.0667


985/Unknown  142s 104ms/step - categorical_accuracy: 0.7043 - loss: 1.1998 - mean_io_u: 0.0667


986/Unknown  142s 104ms/step - categorical_accuracy: 0.7043 - loss: 1.1996 - mean_io_u: 0.0667


987/Unknown  142s 104ms/step - categorical_accuracy: 0.7044 - loss: 1.1994 - mean_io_u: 0.0668


988/Unknown  142s 104ms/step - categorical_accuracy: 0.7044 - loss: 1.1992 - mean_io_u: 0.0668


989/Unknown  143s 104ms/step - categorical_accuracy: 0.7045 - loss: 1.1990 - mean_io_u: 0.0668


990/Unknown  143s 104ms/step - categorical_accuracy: 0.7045 - loss: 1.1988 - mean_io_u: 0.0669


991/Unknown  143s 104ms/step - categorical_accuracy: 0.7045 - loss: 1.1986 - mean_io_u: 0.0669


992/Unknown  143s 104ms/step - categorical_accuracy: 0.7046 - loss: 1.1984 - mean_io_u: 0.0669


993/Unknown  143s 104ms/step - categorical_accuracy: 0.7046 - loss: 1.1982 - mean_io_u: 0.0669


994/Unknown  143s 104ms/step - categorical_accuracy: 0.7047 - loss: 1.1980 - mean_io_u: 0.0670


995/Unknown  143s 104ms/step - categorical_accuracy: 0.7047 - loss: 1.1978 - mean_io_u: 0.0670


996/Unknown  143s 104ms/step - categorical_accuracy: 0.7047 - loss: 1.1976 - mean_io_u: 0.0670


997/Unknown  143s 104ms/step - categorical_accuracy: 0.7048 - loss: 1.1974 - mean_io_u: 0.0671


998/Unknown  143s 104ms/step - categorical_accuracy: 0.7048 - loss: 1.1972 - mean_io_u: 0.0671


999/Unknown  143s 104ms/step - categorical_accuracy: 0.7048 - loss: 1.1970 - mean_io_u: 0.0671


```
</div>
   1000/Unknown  143s 104ms/step - categorical_accuracy: 0.7049 - loss: 1.1968 - mean_io_u: 0.0671

<div class="k-default-codeblock">
```

```
</div>
   1001/Unknown  143s 104ms/step - categorical_accuracy: 0.7049 - loss: 1.1966 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1002/Unknown  143s 104ms/step - categorical_accuracy: 0.7050 - loss: 1.1964 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1003/Unknown  144s 104ms/step - categorical_accuracy: 0.7050 - loss: 1.1962 - mean_io_u: 0.0672

<div class="k-default-codeblock">
```

```
</div>
   1004/Unknown  144s 104ms/step - categorical_accuracy: 0.7050 - loss: 1.1960 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1005/Unknown  144s 104ms/step - categorical_accuracy: 0.7051 - loss: 1.1958 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1006/Unknown  144s 104ms/step - categorical_accuracy: 0.7051 - loss: 1.1956 - mean_io_u: 0.0673

<div class="k-default-codeblock">
```

```
</div>
   1007/Unknown  144s 104ms/step - categorical_accuracy: 0.7051 - loss: 1.1954 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1008/Unknown  144s 104ms/step - categorical_accuracy: 0.7052 - loss: 1.1953 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1009/Unknown  144s 103ms/step - categorical_accuracy: 0.7052 - loss: 1.1951 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1010/Unknown  144s 103ms/step - categorical_accuracy: 0.7053 - loss: 1.1949 - mean_io_u: 0.0674

<div class="k-default-codeblock">
```

```
</div>
   1011/Unknown  144s 103ms/step - categorical_accuracy: 0.7053 - loss: 1.1947 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1012/Unknown  144s 103ms/step - categorical_accuracy: 0.7053 - loss: 1.1945 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1013/Unknown  144s 103ms/step - categorical_accuracy: 0.7054 - loss: 1.1943 - mean_io_u: 0.0675

<div class="k-default-codeblock">
```

```
</div>
   1014/Unknown  144s 103ms/step - categorical_accuracy: 0.7054 - loss: 1.1941 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1015/Unknown  144s 103ms/step - categorical_accuracy: 0.7054 - loss: 1.1939 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1016/Unknown  144s 103ms/step - categorical_accuracy: 0.7055 - loss: 1.1937 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1017/Unknown  144s 103ms/step - categorical_accuracy: 0.7055 - loss: 1.1935 - mean_io_u: 0.0676

<div class="k-default-codeblock">
```

```
</div>
   1018/Unknown  145s 103ms/step - categorical_accuracy: 0.7056 - loss: 1.1933 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1019/Unknown  145s 103ms/step - categorical_accuracy: 0.7056 - loss: 1.1932 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1020/Unknown  145s 103ms/step - categorical_accuracy: 0.7056 - loss: 1.1930 - mean_io_u: 0.0677

<div class="k-default-codeblock">
```

```
</div>
   1021/Unknown  145s 103ms/step - categorical_accuracy: 0.7057 - loss: 1.1928 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1022/Unknown  145s 103ms/step - categorical_accuracy: 0.7057 - loss: 1.1926 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1023/Unknown  145s 103ms/step - categorical_accuracy: 0.7057 - loss: 1.1924 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1024/Unknown  145s 103ms/step - categorical_accuracy: 0.7058 - loss: 1.1922 - mean_io_u: 0.0678

<div class="k-default-codeblock">
```

```
</div>
   1025/Unknown  145s 103ms/step - categorical_accuracy: 0.7058 - loss: 1.1920 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1026/Unknown  145s 103ms/step - categorical_accuracy: 0.7058 - loss: 1.1918 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1027/Unknown  145s 103ms/step - categorical_accuracy: 0.7059 - loss: 1.1917 - mean_io_u: 0.0679

<div class="k-default-codeblock">
```

```
</div>
   1028/Unknown  145s 103ms/step - categorical_accuracy: 0.7059 - loss: 1.1915 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1029/Unknown  145s 103ms/step - categorical_accuracy: 0.7060 - loss: 1.1913 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1030/Unknown  145s 103ms/step - categorical_accuracy: 0.7060 - loss: 1.1911 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1031/Unknown  146s 103ms/step - categorical_accuracy: 0.7060 - loss: 1.1909 - mean_io_u: 0.0680

<div class="k-default-codeblock">
```

```
</div>
   1032/Unknown  146s 103ms/step - categorical_accuracy: 0.7061 - loss: 1.1907 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1033/Unknown  146s 103ms/step - categorical_accuracy: 0.7061 - loss: 1.1905 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1034/Unknown  146s 103ms/step - categorical_accuracy: 0.7061 - loss: 1.1904 - mean_io_u: 0.0681

<div class="k-default-codeblock">
```

```
</div>
   1035/Unknown  146s 103ms/step - categorical_accuracy: 0.7062 - loss: 1.1902 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1036/Unknown  146s 103ms/step - categorical_accuracy: 0.7062 - loss: 1.1900 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1037/Unknown  146s 103ms/step - categorical_accuracy: 0.7062 - loss: 1.1898 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1038/Unknown  146s 103ms/step - categorical_accuracy: 0.7063 - loss: 1.1896 - mean_io_u: 0.0682

<div class="k-default-codeblock">
```

```
</div>
   1039/Unknown  146s 103ms/step - categorical_accuracy: 0.7063 - loss: 1.1894 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1040/Unknown  146s 103ms/step - categorical_accuracy: 0.7063 - loss: 1.1893 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1041/Unknown  146s 103ms/step - categorical_accuracy: 0.7064 - loss: 1.1891 - mean_io_u: 0.0683

<div class="k-default-codeblock">
```

```
</div>
   1042/Unknown  146s 103ms/step - categorical_accuracy: 0.7064 - loss: 1.1889 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1043/Unknown  146s 102ms/step - categorical_accuracy: 0.7065 - loss: 1.1887 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1044/Unknown  147s 102ms/step - categorical_accuracy: 0.7065 - loss: 1.1885 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1045/Unknown  147s 102ms/step - categorical_accuracy: 0.7065 - loss: 1.1883 - mean_io_u: 0.0684

<div class="k-default-codeblock">
```

```
</div>
   1046/Unknown  147s 102ms/step - categorical_accuracy: 0.7066 - loss: 1.1882 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1047/Unknown  147s 102ms/step - categorical_accuracy: 0.7066 - loss: 1.1880 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1048/Unknown  147s 102ms/step - categorical_accuracy: 0.7066 - loss: 1.1878 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1049/Unknown  147s 102ms/step - categorical_accuracy: 0.7067 - loss: 1.1876 - mean_io_u: 0.0685

<div class="k-default-codeblock">
```

```
</div>
   1050/Unknown  147s 102ms/step - categorical_accuracy: 0.7067 - loss: 1.1875 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1051/Unknown  147s 102ms/step - categorical_accuracy: 0.7067 - loss: 1.1873 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1052/Unknown  147s 102ms/step - categorical_accuracy: 0.7068 - loss: 1.1871 - mean_io_u: 0.0686

<div class="k-default-codeblock">
```

```
</div>
   1053/Unknown  147s 102ms/step - categorical_accuracy: 0.7068 - loss: 1.1869 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1054/Unknown  147s 102ms/step - categorical_accuracy: 0.7068 - loss: 1.1867 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1055/Unknown  147s 102ms/step - categorical_accuracy: 0.7069 - loss: 1.1866 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1056/Unknown  147s 102ms/step - categorical_accuracy: 0.7069 - loss: 1.1864 - mean_io_u: 0.0687

<div class="k-default-codeblock">
```

```
</div>
   1057/Unknown  148s 102ms/step - categorical_accuracy: 0.7069 - loss: 1.1862 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1058/Unknown  148s 102ms/step - categorical_accuracy: 0.7070 - loss: 1.1860 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1059/Unknown  148s 102ms/step - categorical_accuracy: 0.7070 - loss: 1.1859 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1060/Unknown  148s 102ms/step - categorical_accuracy: 0.7070 - loss: 1.1857 - mean_io_u: 0.0688

<div class="k-default-codeblock">
```

```
</div>
   1061/Unknown  148s 102ms/step - categorical_accuracy: 0.7071 - loss: 1.1855 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1062/Unknown  148s 102ms/step - categorical_accuracy: 0.7071 - loss: 1.1853 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1063/Unknown  148s 102ms/step - categorical_accuracy: 0.7071 - loss: 1.1852 - mean_io_u: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   1064/Unknown  148s 102ms/step - categorical_accuracy: 0.7072 - loss: 1.1850 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1065/Unknown  148s 102ms/step - categorical_accuracy: 0.7072 - loss: 1.1848 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1066/Unknown  148s 102ms/step - categorical_accuracy: 0.7072 - loss: 1.1846 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1067/Unknown  148s 102ms/step - categorical_accuracy: 0.7073 - loss: 1.1845 - mean_io_u: 0.0690

<div class="k-default-codeblock">
```

```
</div>
   1068/Unknown  148s 102ms/step - categorical_accuracy: 0.7073 - loss: 1.1843 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1069/Unknown  149s 102ms/step - categorical_accuracy: 0.7073 - loss: 1.1841 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1070/Unknown  149s 102ms/step - categorical_accuracy: 0.7074 - loss: 1.1840 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1071/Unknown  149s 102ms/step - categorical_accuracy: 0.7074 - loss: 1.1838 - mean_io_u: 0.0691

<div class="k-default-codeblock">
```

```
</div>
   1072/Unknown  149s 102ms/step - categorical_accuracy: 0.7074 - loss: 1.1836 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1073/Unknown  149s 102ms/step - categorical_accuracy: 0.7075 - loss: 1.1834 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1074/Unknown  149s 102ms/step - categorical_accuracy: 0.7075 - loss: 1.1833 - mean_io_u: 0.0692

<div class="k-default-codeblock">
```

```
</div>
   1075/Unknown  149s 102ms/step - categorical_accuracy: 0.7075 - loss: 1.1831 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1076/Unknown  149s 102ms/step - categorical_accuracy: 0.7076 - loss: 1.1829 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1077/Unknown  149s 102ms/step - categorical_accuracy: 0.7076 - loss: 1.1828 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1078/Unknown  149s 102ms/step - categorical_accuracy: 0.7076 - loss: 1.1826 - mean_io_u: 0.0693

<div class="k-default-codeblock">
```

```
</div>
   1079/Unknown  149s 102ms/step - categorical_accuracy: 0.7077 - loss: 1.1824 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1080/Unknown  149s 102ms/step - categorical_accuracy: 0.7077 - loss: 1.1822 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1081/Unknown  149s 102ms/step - categorical_accuracy: 0.7077 - loss: 1.1821 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1082/Unknown  150s 102ms/step - categorical_accuracy: 0.7078 - loss: 1.1819 - mean_io_u: 0.0694

<div class="k-default-codeblock">
```

```
</div>
   1083/Unknown  150s 102ms/step - categorical_accuracy: 0.7078 - loss: 1.1817 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1084/Unknown  150s 102ms/step - categorical_accuracy: 0.7078 - loss: 1.1816 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1085/Unknown  150s 102ms/step - categorical_accuracy: 0.7079 - loss: 1.1814 - mean_io_u: 0.0695

<div class="k-default-codeblock">
```

```
</div>
   1086/Unknown  150s 102ms/step - categorical_accuracy: 0.7079 - loss: 1.1812 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1087/Unknown  150s 102ms/step - categorical_accuracy: 0.7079 - loss: 1.1811 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1088/Unknown  150s 102ms/step - categorical_accuracy: 0.7080 - loss: 1.1809 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1089/Unknown  150s 102ms/step - categorical_accuracy: 0.7080 - loss: 1.1807 - mean_io_u: 0.0696

<div class="k-default-codeblock">
```

```
</div>
   1090/Unknown  150s 102ms/step - categorical_accuracy: 0.7080 - loss: 1.1806 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1091/Unknown  150s 102ms/step - categorical_accuracy: 0.7080 - loss: 1.1804 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1092/Unknown  150s 102ms/step - categorical_accuracy: 0.7081 - loss: 1.1802 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1093/Unknown  150s 101ms/step - categorical_accuracy: 0.7081 - loss: 1.1801 - mean_io_u: 0.0697

<div class="k-default-codeblock">
```

```
</div>
   1094/Unknown  151s 102ms/step - categorical_accuracy: 0.7081 - loss: 1.1799 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1095/Unknown  151s 101ms/step - categorical_accuracy: 0.7082 - loss: 1.1797 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1096/Unknown  151s 101ms/step - categorical_accuracy: 0.7082 - loss: 1.1796 - mean_io_u: 0.0698

<div class="k-default-codeblock">
```

```
</div>
   1097/Unknown  151s 101ms/step - categorical_accuracy: 0.7082 - loss: 1.1794 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1098/Unknown  151s 101ms/step - categorical_accuracy: 0.7083 - loss: 1.1792 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1099/Unknown  151s 101ms/step - categorical_accuracy: 0.7083 - loss: 1.1791 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1100/Unknown  151s 101ms/step - categorical_accuracy: 0.7083 - loss: 1.1789 - mean_io_u: 0.0699

<div class="k-default-codeblock">
```

```
</div>
   1101/Unknown  151s 101ms/step - categorical_accuracy: 0.7084 - loss: 1.1787 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1102/Unknown  151s 101ms/step - categorical_accuracy: 0.7084 - loss: 1.1786 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1103/Unknown  151s 101ms/step - categorical_accuracy: 0.7084 - loss: 1.1784 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1104/Unknown  151s 101ms/step - categorical_accuracy: 0.7085 - loss: 1.1783 - mean_io_u: 0.0700

<div class="k-default-codeblock">
```

```
</div>
   1105/Unknown  151s 101ms/step - categorical_accuracy: 0.7085 - loss: 1.1781 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1106/Unknown  152s 101ms/step - categorical_accuracy: 0.7085 - loss: 1.1779 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1107/Unknown  152s 101ms/step - categorical_accuracy: 0.7085 - loss: 1.1778 - mean_io_u: 0.0701

<div class="k-default-codeblock">
```

```
</div>
   1108/Unknown  152s 101ms/step - categorical_accuracy: 0.7086 - loss: 1.1776 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1109/Unknown  152s 101ms/step - categorical_accuracy: 0.7086 - loss: 1.1774 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1110/Unknown  152s 101ms/step - categorical_accuracy: 0.7086 - loss: 1.1773 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1111/Unknown  152s 101ms/step - categorical_accuracy: 0.7087 - loss: 1.1771 - mean_io_u: 0.0702

<div class="k-default-codeblock">
```

```
</div>
   1112/Unknown  152s 101ms/step - categorical_accuracy: 0.7087 - loss: 1.1770 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1113/Unknown  152s 101ms/step - categorical_accuracy: 0.7087 - loss: 1.1768 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1114/Unknown  152s 101ms/step - categorical_accuracy: 0.7088 - loss: 1.1766 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1115/Unknown  152s 101ms/step - categorical_accuracy: 0.7088 - loss: 1.1765 - mean_io_u: 0.0703

<div class="k-default-codeblock">
```

```
</div>
   1116/Unknown  152s 101ms/step - categorical_accuracy: 0.7088 - loss: 1.1763 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1117/Unknown  152s 101ms/step - categorical_accuracy: 0.7089 - loss: 1.1761 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1118/Unknown  152s 101ms/step - categorical_accuracy: 0.7089 - loss: 1.1760 - mean_io_u: 0.0704

<div class="k-default-codeblock">
```

```
</div>
   1119/Unknown  153s 101ms/step - categorical_accuracy: 0.7089 - loss: 1.1758 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1120/Unknown  153s 101ms/step - categorical_accuracy: 0.7089 - loss: 1.1757 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1121/Unknown  153s 101ms/step - categorical_accuracy: 0.7090 - loss: 1.1755 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1122/Unknown  153s 101ms/step - categorical_accuracy: 0.7090 - loss: 1.1753 - mean_io_u: 0.0705

<div class="k-default-codeblock">
```

```
</div>
   1123/Unknown  153s 101ms/step - categorical_accuracy: 0.7090 - loss: 1.1752 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1124/Unknown  153s 101ms/step - categorical_accuracy: 0.7091 - loss: 1.1750 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1125/Unknown  153s 101ms/step - categorical_accuracy: 0.7091 - loss: 1.1749 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1126/Unknown  153s 101ms/step - categorical_accuracy: 0.7091 - loss: 1.1747 - mean_io_u: 0.0706

<div class="k-default-codeblock">
```

```
</div>
   1127/Unknown  153s 101ms/step - categorical_accuracy: 0.7092 - loss: 1.1745 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1128/Unknown  153s 101ms/step - categorical_accuracy: 0.7092 - loss: 1.1744 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1129/Unknown  153s 101ms/step - categorical_accuracy: 0.7092 - loss: 1.1742 - mean_io_u: 0.0707

<div class="k-default-codeblock">
```

```
</div>
   1130/Unknown  153s 101ms/step - categorical_accuracy: 0.7093 - loss: 1.1741 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1131/Unknown  154s 101ms/step - categorical_accuracy: 0.7093 - loss: 1.1739 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1132/Unknown  154s 101ms/step - categorical_accuracy: 0.7093 - loss: 1.1738 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1133/Unknown  154s 101ms/step - categorical_accuracy: 0.7093 - loss: 1.1736 - mean_io_u: 0.0708

<div class="k-default-codeblock">
```

```
</div>
   1134/Unknown  154s 101ms/step - categorical_accuracy: 0.7094 - loss: 1.1734 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1135/Unknown  154s 101ms/step - categorical_accuracy: 0.7094 - loss: 1.1733 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1136/Unknown  154s 101ms/step - categorical_accuracy: 0.7094 - loss: 1.1731 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1137/Unknown  154s 101ms/step - categorical_accuracy: 0.7095 - loss: 1.1730 - mean_io_u: 0.0709

<div class="k-default-codeblock">
```

```
</div>
   1138/Unknown  154s 101ms/step - categorical_accuracy: 0.7095 - loss: 1.1728 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1139/Unknown  154s 101ms/step - categorical_accuracy: 0.7095 - loss: 1.1727 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1140/Unknown  154s 101ms/step - categorical_accuracy: 0.7096 - loss: 1.1725 - mean_io_u: 0.0710

<div class="k-default-codeblock">
```

```
</div>
   1141/Unknown  154s 101ms/step - categorical_accuracy: 0.7096 - loss: 1.1723 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1142/Unknown  154s 101ms/step - categorical_accuracy: 0.7096 - loss: 1.1722 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1143/Unknown  154s 101ms/step - categorical_accuracy: 0.7096 - loss: 1.1720 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1144/Unknown  155s 100ms/step - categorical_accuracy: 0.7097 - loss: 1.1719 - mean_io_u: 0.0711

<div class="k-default-codeblock">
```

```
</div>
   1145/Unknown  155s 100ms/step - categorical_accuracy: 0.7097 - loss: 1.1717 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1146/Unknown  155s 100ms/step - categorical_accuracy: 0.7097 - loss: 1.1716 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1147/Unknown  155s 100ms/step - categorical_accuracy: 0.7098 - loss: 1.1714 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1148/Unknown  155s 100ms/step - categorical_accuracy: 0.7098 - loss: 1.1712 - mean_io_u: 0.0712

<div class="k-default-codeblock">
```

```
</div>
   1149/Unknown  155s 100ms/step - categorical_accuracy: 0.7098 - loss: 1.1711 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1150/Unknown  155s 100ms/step - categorical_accuracy: 0.7099 - loss: 1.1709 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1151/Unknown  155s 100ms/step - categorical_accuracy: 0.7099 - loss: 1.1708 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1152/Unknown  155s 100ms/step - categorical_accuracy: 0.7099 - loss: 1.1706 - mean_io_u: 0.0713

<div class="k-default-codeblock">
```

```
</div>
   1153/Unknown  155s 100ms/step - categorical_accuracy: 0.7099 - loss: 1.1705 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1154/Unknown  155s 100ms/step - categorical_accuracy: 0.7100 - loss: 1.1703 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1155/Unknown  155s 100ms/step - categorical_accuracy: 0.7100 - loss: 1.1702 - mean_io_u: 0.0714

<div class="k-default-codeblock">
```

```
</div>
   1156/Unknown  155s 100ms/step - categorical_accuracy: 0.7100 - loss: 1.1700 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1157/Unknown  155s 100ms/step - categorical_accuracy: 0.7101 - loss: 1.1699 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1158/Unknown  156s 100ms/step - categorical_accuracy: 0.7101 - loss: 1.1697 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1159/Unknown  156s 100ms/step - categorical_accuracy: 0.7101 - loss: 1.1696 - mean_io_u: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   1160/Unknown  156s 100ms/step - categorical_accuracy: 0.7101 - loss: 1.1694 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1161/Unknown  156s 100ms/step - categorical_accuracy: 0.7102 - loss: 1.1693 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1162/Unknown  156s 100ms/step - categorical_accuracy: 0.7102 - loss: 1.1691 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1163/Unknown  156s 100ms/step - categorical_accuracy: 0.7102 - loss: 1.1689 - mean_io_u: 0.0716

<div class="k-default-codeblock">
```

```
</div>
   1164/Unknown  156s 100ms/step - categorical_accuracy: 0.7103 - loss: 1.1688 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1165/Unknown  156s 100ms/step - categorical_accuracy: 0.7103 - loss: 1.1686 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1166/Unknown  156s 100ms/step - categorical_accuracy: 0.7103 - loss: 1.1685 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1167/Unknown  156s 100ms/step - categorical_accuracy: 0.7103 - loss: 1.1683 - mean_io_u: 0.0717

<div class="k-default-codeblock">
```

```
</div>
   1168/Unknown  156s 100ms/step - categorical_accuracy: 0.7104 - loss: 1.1682 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1169/Unknown  156s 100ms/step - categorical_accuracy: 0.7104 - loss: 1.1680 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1170/Unknown  156s 100ms/step - categorical_accuracy: 0.7104 - loss: 1.1679 - mean_io_u: 0.0718

<div class="k-default-codeblock">
```

```
</div>
   1171/Unknown  156s 100ms/step - categorical_accuracy: 0.7105 - loss: 1.1677 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1172/Unknown  157s 100ms/step - categorical_accuracy: 0.7105 - loss: 1.1676 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1173/Unknown  157s 100ms/step - categorical_accuracy: 0.7105 - loss: 1.1674 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1174/Unknown  157s 100ms/step - categorical_accuracy: 0.7105 - loss: 1.1673 - mean_io_u: 0.0719

<div class="k-default-codeblock">
```

```
</div>
   1175/Unknown  157s 100ms/step - categorical_accuracy: 0.7106 - loss: 1.1671 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1176/Unknown  157s 100ms/step - categorical_accuracy: 0.7106 - loss: 1.1670 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1177/Unknown  157s 100ms/step - categorical_accuracy: 0.7106 - loss: 1.1668 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1178/Unknown  157s 100ms/step - categorical_accuracy: 0.7107 - loss: 1.1667 - mean_io_u: 0.0720

<div class="k-default-codeblock">
```

```
</div>
   1179/Unknown  157s 100ms/step - categorical_accuracy: 0.7107 - loss: 1.1665 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1180/Unknown  157s 100ms/step - categorical_accuracy: 0.7107 - loss: 1.1664 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1181/Unknown  157s 100ms/step - categorical_accuracy: 0.7108 - loss: 1.1662 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1182/Unknown  157s 100ms/step - categorical_accuracy: 0.7108 - loss: 1.1661 - mean_io_u: 0.0721

<div class="k-default-codeblock">
```

```
</div>
   1183/Unknown  157s 100ms/step - categorical_accuracy: 0.7108 - loss: 1.1659 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1184/Unknown  157s 100ms/step - categorical_accuracy: 0.7108 - loss: 1.1658 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1185/Unknown  158s 100ms/step - categorical_accuracy: 0.7109 - loss: 1.1656 - mean_io_u: 0.0722

<div class="k-default-codeblock">
```

```
</div>
   1186/Unknown  158s 100ms/step - categorical_accuracy: 0.7109 - loss: 1.1655 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1187/Unknown  158s 100ms/step - categorical_accuracy: 0.7109 - loss: 1.1653 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1188/Unknown  158s 100ms/step - categorical_accuracy: 0.7110 - loss: 1.1652 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1189/Unknown  158s 100ms/step - categorical_accuracy: 0.7110 - loss: 1.1650 - mean_io_u: 0.0723

<div class="k-default-codeblock">
```

```
</div>
   1190/Unknown  158s 99ms/step - categorical_accuracy: 0.7110 - loss: 1.1649 - mean_io_u: 0.0724 

<div class="k-default-codeblock">
```

```
</div>
   1191/Unknown  158s 99ms/step - categorical_accuracy: 0.7110 - loss: 1.1647 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1192/Unknown  158s 99ms/step - categorical_accuracy: 0.7111 - loss: 1.1646 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1193/Unknown  158s 99ms/step - categorical_accuracy: 0.7111 - loss: 1.1644 - mean_io_u: 0.0724

<div class="k-default-codeblock">
```

```
</div>
   1194/Unknown  158s 99ms/step - categorical_accuracy: 0.7111 - loss: 1.1643 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1195/Unknown  158s 99ms/step - categorical_accuracy: 0.7112 - loss: 1.1641 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1196/Unknown  158s 99ms/step - categorical_accuracy: 0.7112 - loss: 1.1640 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1197/Unknown  159s 99ms/step - categorical_accuracy: 0.7112 - loss: 1.1638 - mean_io_u: 0.0725

<div class="k-default-codeblock">
```

```
</div>
   1198/Unknown  159s 99ms/step - categorical_accuracy: 0.7112 - loss: 1.1637 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1199/Unknown  159s 99ms/step - categorical_accuracy: 0.7113 - loss: 1.1635 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1200/Unknown  159s 99ms/step - categorical_accuracy: 0.7113 - loss: 1.1634 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1201/Unknown  159s 99ms/step - categorical_accuracy: 0.7113 - loss: 1.1632 - mean_io_u: 0.0726

<div class="k-default-codeblock">
```

```
</div>
   1202/Unknown  159s 99ms/step - categorical_accuracy: 0.7114 - loss: 1.1631 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1203/Unknown  159s 99ms/step - categorical_accuracy: 0.7114 - loss: 1.1629 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1204/Unknown  159s 99ms/step - categorical_accuracy: 0.7114 - loss: 1.1628 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1205/Unknown  159s 99ms/step - categorical_accuracy: 0.7114 - loss: 1.1626 - mean_io_u: 0.0727

<div class="k-default-codeblock">
```

```
</div>
   1206/Unknown  159s 99ms/step - categorical_accuracy: 0.7115 - loss: 1.1625 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1207/Unknown  159s 99ms/step - categorical_accuracy: 0.7115 - loss: 1.1623 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1208/Unknown  159s 99ms/step - categorical_accuracy: 0.7115 - loss: 1.1622 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1209/Unknown  159s 99ms/step - categorical_accuracy: 0.7115 - loss: 1.1621 - mean_io_u: 0.0728

<div class="k-default-codeblock">
```

```
</div>
   1210/Unknown  159s 99ms/step - categorical_accuracy: 0.7116 - loss: 1.1619 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1211/Unknown  160s 99ms/step - categorical_accuracy: 0.7116 - loss: 1.1618 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1212/Unknown  160s 99ms/step - categorical_accuracy: 0.7116 - loss: 1.1616 - mean_io_u: 0.0729

<div class="k-default-codeblock">
```

```
</div>
   1213/Unknown  160s 99ms/step - categorical_accuracy: 0.7117 - loss: 1.1615 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1214/Unknown  160s 99ms/step - categorical_accuracy: 0.7117 - loss: 1.1613 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1215/Unknown  160s 99ms/step - categorical_accuracy: 0.7117 - loss: 1.1612 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1216/Unknown  160s 99ms/step - categorical_accuracy: 0.7117 - loss: 1.1610 - mean_io_u: 0.0730

<div class="k-default-codeblock">
```

```
</div>
   1217/Unknown  160s 99ms/step - categorical_accuracy: 0.7118 - loss: 1.1609 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1218/Unknown  160s 99ms/step - categorical_accuracy: 0.7118 - loss: 1.1607 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1219/Unknown  160s 99ms/step - categorical_accuracy: 0.7118 - loss: 1.1606 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1220/Unknown  160s 99ms/step - categorical_accuracy: 0.7119 - loss: 1.1605 - mean_io_u: 0.0731

<div class="k-default-codeblock">
```

```
</div>
   1221/Unknown  160s 99ms/step - categorical_accuracy: 0.7119 - loss: 1.1603 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1222/Unknown  160s 99ms/step - categorical_accuracy: 0.7119 - loss: 1.1602 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1223/Unknown  161s 99ms/step - categorical_accuracy: 0.7119 - loss: 1.1600 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1224/Unknown  161s 99ms/step - categorical_accuracy: 0.7120 - loss: 1.1599 - mean_io_u: 0.0732

<div class="k-default-codeblock">
```

```
</div>
   1225/Unknown  161s 99ms/step - categorical_accuracy: 0.7120 - loss: 1.1597 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1226/Unknown  161s 99ms/step - categorical_accuracy: 0.7120 - loss: 1.1596 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1227/Unknown  161s 99ms/step - categorical_accuracy: 0.7120 - loss: 1.1595 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1228/Unknown  161s 99ms/step - categorical_accuracy: 0.7121 - loss: 1.1593 - mean_io_u: 0.0733

<div class="k-default-codeblock">
```

```
</div>
   1229/Unknown  161s 99ms/step - categorical_accuracy: 0.7121 - loss: 1.1592 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1230/Unknown  161s 99ms/step - categorical_accuracy: 0.7121 - loss: 1.1590 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1231/Unknown  161s 99ms/step - categorical_accuracy: 0.7122 - loss: 1.1589 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1232/Unknown  161s 99ms/step - categorical_accuracy: 0.7122 - loss: 1.1587 - mean_io_u: 0.0734

<div class="k-default-codeblock">
```

```
</div>
   1233/Unknown  161s 99ms/step - categorical_accuracy: 0.7122 - loss: 1.1586 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1234/Unknown  161s 99ms/step - categorical_accuracy: 0.7122 - loss: 1.1585 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1235/Unknown  162s 99ms/step - categorical_accuracy: 0.7123 - loss: 1.1583 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1236/Unknown  162s 99ms/step - categorical_accuracy: 0.7123 - loss: 1.1582 - mean_io_u: 0.0735

<div class="k-default-codeblock">
```

```
</div>
   1237/Unknown  162s 99ms/step - categorical_accuracy: 0.7123 - loss: 1.1580 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1238/Unknown  162s 99ms/step - categorical_accuracy: 0.7123 - loss: 1.1579 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1239/Unknown  162s 99ms/step - categorical_accuracy: 0.7124 - loss: 1.1577 - mean_io_u: 0.0736

<div class="k-default-codeblock">
```

```
</div>
   1240/Unknown  162s 99ms/step - categorical_accuracy: 0.7124 - loss: 1.1576 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1241/Unknown  162s 99ms/step - categorical_accuracy: 0.7124 - loss: 1.1575 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1242/Unknown  162s 99ms/step - categorical_accuracy: 0.7124 - loss: 1.1573 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1243/Unknown  162s 99ms/step - categorical_accuracy: 0.7125 - loss: 1.1572 - mean_io_u: 0.0737

<div class="k-default-codeblock">
```

```
</div>
   1244/Unknown  162s 99ms/step - categorical_accuracy: 0.7125 - loss: 1.1570 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1245/Unknown  162s 99ms/step - categorical_accuracy: 0.7125 - loss: 1.1569 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1246/Unknown  162s 99ms/step - categorical_accuracy: 0.7126 - loss: 1.1568 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1247/Unknown  162s 99ms/step - categorical_accuracy: 0.7126 - loss: 1.1566 - mean_io_u: 0.0738

<div class="k-default-codeblock">
```

```
</div>
   1248/Unknown  163s 99ms/step - categorical_accuracy: 0.7126 - loss: 1.1565 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1249/Unknown  163s 99ms/step - categorical_accuracy: 0.7126 - loss: 1.1563 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1250/Unknown  163s 99ms/step - categorical_accuracy: 0.7127 - loss: 1.1562 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1251/Unknown  163s 99ms/step - categorical_accuracy: 0.7127 - loss: 1.1561 - mean_io_u: 0.0739

<div class="k-default-codeblock">
```

```
</div>
   1252/Unknown  163s 99ms/step - categorical_accuracy: 0.7127 - loss: 1.1559 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1253/Unknown  163s 99ms/step - categorical_accuracy: 0.7127 - loss: 1.1558 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1254/Unknown  163s 98ms/step - categorical_accuracy: 0.7128 - loss: 1.1556 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1255/Unknown  163s 98ms/step - categorical_accuracy: 0.7128 - loss: 1.1555 - mean_io_u: 0.0740

<div class="k-default-codeblock">
```

```
</div>
   1256/Unknown  163s 98ms/step - categorical_accuracy: 0.7128 - loss: 1.1554 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1257/Unknown  163s 98ms/step - categorical_accuracy: 0.7128 - loss: 1.1552 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1258/Unknown  163s 98ms/step - categorical_accuracy: 0.7129 - loss: 1.1551 - mean_io_u: 0.0741

<div class="k-default-codeblock">
```

```
</div>
   1259/Unknown  163s 98ms/step - categorical_accuracy: 0.7129 - loss: 1.1549 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1260/Unknown  163s 98ms/step - categorical_accuracy: 0.7129 - loss: 1.1548 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1261/Unknown  164s 98ms/step - categorical_accuracy: 0.7130 - loss: 1.1547 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1262/Unknown  164s 98ms/step - categorical_accuracy: 0.7130 - loss: 1.1545 - mean_io_u: 0.0742

<div class="k-default-codeblock">
```

```
</div>
   1263/Unknown  164s 98ms/step - categorical_accuracy: 0.7130 - loss: 1.1544 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1264/Unknown  164s 98ms/step - categorical_accuracy: 0.7130 - loss: 1.1543 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1265/Unknown  164s 98ms/step - categorical_accuracy: 0.7131 - loss: 1.1541 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1266/Unknown  164s 98ms/step - categorical_accuracy: 0.7131 - loss: 1.1540 - mean_io_u: 0.0743

<div class="k-default-codeblock">
```

```
</div>
   1267/Unknown  164s 98ms/step - categorical_accuracy: 0.7131 - loss: 1.1538 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1268/Unknown  164s 98ms/step - categorical_accuracy: 0.7131 - loss: 1.1537 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1269/Unknown  164s 98ms/step - categorical_accuracy: 0.7132 - loss: 1.1536 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1270/Unknown  164s 98ms/step - categorical_accuracy: 0.7132 - loss: 1.1534 - mean_io_u: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   1271/Unknown  164s 98ms/step - categorical_accuracy: 0.7132 - loss: 1.1533 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1272/Unknown  164s 98ms/step - categorical_accuracy: 0.7132 - loss: 1.1532 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1273/Unknown  164s 98ms/step - categorical_accuracy: 0.7133 - loss: 1.1530 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1274/Unknown  165s 98ms/step - categorical_accuracy: 0.7133 - loss: 1.1529 - mean_io_u: 0.0745

<div class="k-default-codeblock">
```

```
</div>
   1275/Unknown  165s 98ms/step - categorical_accuracy: 0.7133 - loss: 1.1527 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1276/Unknown  165s 98ms/step - categorical_accuracy: 0.7133 - loss: 1.1526 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1277/Unknown  165s 98ms/step - categorical_accuracy: 0.7134 - loss: 1.1525 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1278/Unknown  165s 98ms/step - categorical_accuracy: 0.7134 - loss: 1.1523 - mean_io_u: 0.0746

<div class="k-default-codeblock">
```

```
</div>
   1279/Unknown  165s 98ms/step - categorical_accuracy: 0.7134 - loss: 1.1522 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1280/Unknown  165s 98ms/step - categorical_accuracy: 0.7134 - loss: 1.1521 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1281/Unknown  165s 98ms/step - categorical_accuracy: 0.7135 - loss: 1.1519 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1282/Unknown  165s 98ms/step - categorical_accuracy: 0.7135 - loss: 1.1518 - mean_io_u: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   1283/Unknown  165s 98ms/step - categorical_accuracy: 0.7135 - loss: 1.1517 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1284/Unknown  165s 98ms/step - categorical_accuracy: 0.7135 - loss: 1.1515 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1285/Unknown  165s 98ms/step - categorical_accuracy: 0.7136 - loss: 1.1514 - mean_io_u: 0.0748

<div class="k-default-codeblock">
```

```
</div>
   1286/Unknown  166s 98ms/step - categorical_accuracy: 0.7136 - loss: 1.1513 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1287/Unknown  166s 98ms/step - categorical_accuracy: 0.7136 - loss: 1.1511 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1288/Unknown  166s 98ms/step - categorical_accuracy: 0.7136 - loss: 1.1510 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1289/Unknown  166s 98ms/step - categorical_accuracy: 0.7137 - loss: 1.1509 - mean_io_u: 0.0749

<div class="k-default-codeblock">
```

```
</div>
   1290/Unknown  166s 98ms/step - categorical_accuracy: 0.7137 - loss: 1.1507 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1291/Unknown  166s 98ms/step - categorical_accuracy: 0.7137 - loss: 1.1506 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1292/Unknown  166s 98ms/step - categorical_accuracy: 0.7137 - loss: 1.1505 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1293/Unknown  166s 98ms/step - categorical_accuracy: 0.7138 - loss: 1.1503 - mean_io_u: 0.0750

<div class="k-default-codeblock">
```

```
</div>
   1294/Unknown  166s 98ms/step - categorical_accuracy: 0.7138 - loss: 1.1502 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1295/Unknown  166s 98ms/step - categorical_accuracy: 0.7138 - loss: 1.1501 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1296/Unknown  166s 98ms/step - categorical_accuracy: 0.7138 - loss: 1.1499 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1297/Unknown  166s 98ms/step - categorical_accuracy: 0.7139 - loss: 1.1498 - mean_io_u: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   1298/Unknown  166s 98ms/step - categorical_accuracy: 0.7139 - loss: 1.1497 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1299/Unknown  166s 98ms/step - categorical_accuracy: 0.7139 - loss: 1.1495 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1300/Unknown  167s 98ms/step - categorical_accuracy: 0.7139 - loss: 1.1494 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1301/Unknown  167s 98ms/step - categorical_accuracy: 0.7140 - loss: 1.1493 - mean_io_u: 0.0752

<div class="k-default-codeblock">
```

```
</div>
   1302/Unknown  167s 98ms/step - categorical_accuracy: 0.7140 - loss: 1.1491 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1303/Unknown  167s 98ms/step - categorical_accuracy: 0.7140 - loss: 1.1490 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1304/Unknown  167s 98ms/step - categorical_accuracy: 0.7140 - loss: 1.1489 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1305/Unknown  167s 98ms/step - categorical_accuracy: 0.7141 - loss: 1.1487 - mean_io_u: 0.0753

<div class="k-default-codeblock">
```

```
</div>
   1306/Unknown  167s 98ms/step - categorical_accuracy: 0.7141 - loss: 1.1486 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1307/Unknown  167s 98ms/step - categorical_accuracy: 0.7141 - loss: 1.1485 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1308/Unknown  167s 98ms/step - categorical_accuracy: 0.7141 - loss: 1.1483 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1309/Unknown  167s 97ms/step - categorical_accuracy: 0.7142 - loss: 1.1482 - mean_io_u: 0.0754

<div class="k-default-codeblock">
```

```
</div>
   1310/Unknown  167s 97ms/step - categorical_accuracy: 0.7142 - loss: 1.1481 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1311/Unknown  167s 97ms/step - categorical_accuracy: 0.7142 - loss: 1.1480 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1312/Unknown  167s 97ms/step - categorical_accuracy: 0.7142 - loss: 1.1478 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1313/Unknown  167s 97ms/step - categorical_accuracy: 0.7143 - loss: 1.1477 - mean_io_u: 0.0755

<div class="k-default-codeblock">
```

```
</div>
   1314/Unknown  167s 97ms/step - categorical_accuracy: 0.7143 - loss: 1.1476 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1315/Unknown  168s 97ms/step - categorical_accuracy: 0.7143 - loss: 1.1474 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1316/Unknown  168s 97ms/step - categorical_accuracy: 0.7143 - loss: 1.1473 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1317/Unknown  168s 97ms/step - categorical_accuracy: 0.7144 - loss: 1.1472 - mean_io_u: 0.0756

<div class="k-default-codeblock">
```

```
</div>
   1318/Unknown  168s 97ms/step - categorical_accuracy: 0.7144 - loss: 1.1471 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1319/Unknown  168s 97ms/step - categorical_accuracy: 0.7144 - loss: 1.1469 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1320/Unknown  168s 97ms/step - categorical_accuracy: 0.7144 - loss: 1.1468 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1321/Unknown  168s 97ms/step - categorical_accuracy: 0.7145 - loss: 1.1467 - mean_io_u: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   1322/Unknown  168s 97ms/step - categorical_accuracy: 0.7145 - loss: 1.1465 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1323/Unknown  168s 97ms/step - categorical_accuracy: 0.7145 - loss: 1.1464 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1324/Unknown  168s 97ms/step - categorical_accuracy: 0.7145 - loss: 1.1463 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1325/Unknown  168s 97ms/step - categorical_accuracy: 0.7146 - loss: 1.1462 - mean_io_u: 0.0758

<div class="k-default-codeblock">
```

```
</div>
   1326/Unknown  168s 97ms/step - categorical_accuracy: 0.7146 - loss: 1.1460 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1327/Unknown  168s 97ms/step - categorical_accuracy: 0.7146 - loss: 1.1459 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1328/Unknown  169s 97ms/step - categorical_accuracy: 0.7146 - loss: 1.1458 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1329/Unknown  169s 97ms/step - categorical_accuracy: 0.7147 - loss: 1.1456 - mean_io_u: 0.0759

<div class="k-default-codeblock">
```

```
</div>
   1330/Unknown  169s 97ms/step - categorical_accuracy: 0.7147 - loss: 1.1455 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1331/Unknown  169s 97ms/step - categorical_accuracy: 0.7147 - loss: 1.1454 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1332/Unknown  169s 97ms/step - categorical_accuracy: 0.7147 - loss: 1.1453 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1333/Unknown  169s 97ms/step - categorical_accuracy: 0.7148 - loss: 1.1451 - mean_io_u: 0.0760

<div class="k-default-codeblock">
```

```
</div>
   1334/Unknown  169s 97ms/step - categorical_accuracy: 0.7148 - loss: 1.1450 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1335/Unknown  169s 97ms/step - categorical_accuracy: 0.7148 - loss: 1.1449 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1336/Unknown  169s 97ms/step - categorical_accuracy: 0.7148 - loss: 1.1448 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1337/Unknown  169s 97ms/step - categorical_accuracy: 0.7148 - loss: 1.1446 - mean_io_u: 0.0761

<div class="k-default-codeblock">
```

```
</div>
   1338/Unknown  169s 97ms/step - categorical_accuracy: 0.7149 - loss: 1.1445 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1339/Unknown  169s 97ms/step - categorical_accuracy: 0.7149 - loss: 1.1444 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1340/Unknown  169s 97ms/step - categorical_accuracy: 0.7149 - loss: 1.1443 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1341/Unknown  169s 97ms/step - categorical_accuracy: 0.7149 - loss: 1.1441 - mean_io_u: 0.0762

<div class="k-default-codeblock">
```

```
</div>
   1342/Unknown  170s 97ms/step - categorical_accuracy: 0.7150 - loss: 1.1440 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1343/Unknown  170s 97ms/step - categorical_accuracy: 0.7150 - loss: 1.1439 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1344/Unknown  170s 97ms/step - categorical_accuracy: 0.7150 - loss: 1.1438 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1345/Unknown  170s 97ms/step - categorical_accuracy: 0.7150 - loss: 1.1436 - mean_io_u: 0.0763

<div class="k-default-codeblock">
```

```
</div>
   1346/Unknown  170s 97ms/step - categorical_accuracy: 0.7151 - loss: 1.1435 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1347/Unknown  170s 97ms/step - categorical_accuracy: 0.7151 - loss: 1.1434 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1348/Unknown  170s 97ms/step - categorical_accuracy: 0.7151 - loss: 1.1433 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1349/Unknown  170s 97ms/step - categorical_accuracy: 0.7151 - loss: 1.1431 - mean_io_u: 0.0764

<div class="k-default-codeblock">
```

```
</div>
   1350/Unknown  170s 97ms/step - categorical_accuracy: 0.7152 - loss: 1.1430 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1351/Unknown  170s 97ms/step - categorical_accuracy: 0.7152 - loss: 1.1429 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1352/Unknown  170s 97ms/step - categorical_accuracy: 0.7152 - loss: 1.1428 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1353/Unknown  170s 97ms/step - categorical_accuracy: 0.7152 - loss: 1.1426 - mean_io_u: 0.0765

<div class="k-default-codeblock">
```

```
</div>
   1354/Unknown  171s 97ms/step - categorical_accuracy: 0.7152 - loss: 1.1425 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1355/Unknown  171s 97ms/step - categorical_accuracy: 0.7153 - loss: 1.1424 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1356/Unknown  171s 97ms/step - categorical_accuracy: 0.7153 - loss: 1.1423 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1357/Unknown  171s 97ms/step - categorical_accuracy: 0.7153 - loss: 1.1421 - mean_io_u: 0.0766

<div class="k-default-codeblock">
```

```
</div>
   1358/Unknown  171s 97ms/step - categorical_accuracy: 0.7153 - loss: 1.1420 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1359/Unknown  171s 97ms/step - categorical_accuracy: 0.7154 - loss: 1.1419 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1360/Unknown  171s 97ms/step - categorical_accuracy: 0.7154 - loss: 1.1418 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1361/Unknown  171s 97ms/step - categorical_accuracy: 0.7154 - loss: 1.1416 - mean_io_u: 0.0767

<div class="k-default-codeblock">
```

```
</div>
   1362/Unknown  171s 97ms/step - categorical_accuracy: 0.7154 - loss: 1.1415 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1363/Unknown  171s 97ms/step - categorical_accuracy: 0.7155 - loss: 1.1414 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1364/Unknown  171s 97ms/step - categorical_accuracy: 0.7155 - loss: 1.1413 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1365/Unknown  171s 97ms/step - categorical_accuracy: 0.7155 - loss: 1.1411 - mean_io_u: 0.0768

<div class="k-default-codeblock">
```

```
</div>
   1366/Unknown  171s 97ms/step - categorical_accuracy: 0.7155 - loss: 1.1410 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1367/Unknown  172s 97ms/step - categorical_accuracy: 0.7155 - loss: 1.1409 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1368/Unknown  172s 97ms/step - categorical_accuracy: 0.7156 - loss: 1.1408 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1369/Unknown  172s 97ms/step - categorical_accuracy: 0.7156 - loss: 1.1407 - mean_io_u: 0.0769

<div class="k-default-codeblock">
```

```
</div>
   1370/Unknown  172s 97ms/step - categorical_accuracy: 0.7156 - loss: 1.1405 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1371/Unknown  172s 97ms/step - categorical_accuracy: 0.7156 - loss: 1.1404 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1372/Unknown  172s 97ms/step - categorical_accuracy: 0.7157 - loss: 1.1403 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1373/Unknown  172s 97ms/step - categorical_accuracy: 0.7157 - loss: 1.1402 - mean_io_u: 0.0770

<div class="k-default-codeblock">
```

```
</div>
   1374/Unknown  172s 97ms/step - categorical_accuracy: 0.7157 - loss: 1.1400 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1375/Unknown  172s 97ms/step - categorical_accuracy: 0.7157 - loss: 1.1399 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1376/Unknown  172s 97ms/step - categorical_accuracy: 0.7157 - loss: 1.1398 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1377/Unknown  173s 97ms/step - categorical_accuracy: 0.7158 - loss: 1.1397 - mean_io_u: 0.0771

<div class="k-default-codeblock">
```

```
</div>
   1378/Unknown  173s 97ms/step - categorical_accuracy: 0.7158 - loss: 1.1396 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1379/Unknown  173s 97ms/step - categorical_accuracy: 0.7158 - loss: 1.1394 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1380/Unknown  173s 97ms/step - categorical_accuracy: 0.7158 - loss: 1.1393 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1381/Unknown  173s 97ms/step - categorical_accuracy: 0.7159 - loss: 1.1392 - mean_io_u: 0.0772

<div class="k-default-codeblock">
```

```
</div>
   1382/Unknown  173s 97ms/step - categorical_accuracy: 0.7159 - loss: 1.1391 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1383/Unknown  173s 96ms/step - categorical_accuracy: 0.7159 - loss: 1.1389 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1384/Unknown  173s 96ms/step - categorical_accuracy: 0.7159 - loss: 1.1388 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1385/Unknown  173s 96ms/step - categorical_accuracy: 0.7160 - loss: 1.1387 - mean_io_u: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   1386/Unknown  173s 96ms/step - categorical_accuracy: 0.7160 - loss: 1.1386 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1387/Unknown  173s 96ms/step - categorical_accuracy: 0.7160 - loss: 1.1385 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1388/Unknown  173s 96ms/step - categorical_accuracy: 0.7160 - loss: 1.1383 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1389/Unknown  173s 96ms/step - categorical_accuracy: 0.7160 - loss: 1.1382 - mean_io_u: 0.0774

<div class="k-default-codeblock">
```

```
</div>
   1390/Unknown  174s 96ms/step - categorical_accuracy: 0.7161 - loss: 1.1381 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1391/Unknown  174s 96ms/step - categorical_accuracy: 0.7161 - loss: 1.1380 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1392/Unknown  174s 96ms/step - categorical_accuracy: 0.7161 - loss: 1.1379 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1393/Unknown  174s 96ms/step - categorical_accuracy: 0.7161 - loss: 1.1377 - mean_io_u: 0.0775

<div class="k-default-codeblock">
```

```
</div>
   1394/Unknown  174s 96ms/step - categorical_accuracy: 0.7162 - loss: 1.1376 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1395/Unknown  174s 96ms/step - categorical_accuracy: 0.7162 - loss: 1.1375 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1396/Unknown  174s 96ms/step - categorical_accuracy: 0.7162 - loss: 1.1374 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1397/Unknown  174s 96ms/step - categorical_accuracy: 0.7162 - loss: 1.1373 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1398/Unknown  174s 96ms/step - categorical_accuracy: 0.7162 - loss: 1.1371 - mean_io_u: 0.0776

<div class="k-default-codeblock">
```

```
</div>
   1399/Unknown  174s 96ms/step - categorical_accuracy: 0.7163 - loss: 1.1370 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1400/Unknown  174s 96ms/step - categorical_accuracy: 0.7163 - loss: 1.1369 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1401/Unknown  174s 96ms/step - categorical_accuracy: 0.7163 - loss: 1.1368 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1402/Unknown  174s 96ms/step - categorical_accuracy: 0.7163 - loss: 1.1367 - mean_io_u: 0.0777

<div class="k-default-codeblock">
```

```
</div>
   1403/Unknown  175s 96ms/step - categorical_accuracy: 0.7164 - loss: 1.1365 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1404/Unknown  175s 96ms/step - categorical_accuracy: 0.7164 - loss: 1.1364 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1405/Unknown  175s 96ms/step - categorical_accuracy: 0.7164 - loss: 1.1363 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1406/Unknown  175s 96ms/step - categorical_accuracy: 0.7164 - loss: 1.1362 - mean_io_u: 0.0778

<div class="k-default-codeblock">
```

```
</div>
   1407/Unknown  175s 96ms/step - categorical_accuracy: 0.7164 - loss: 1.1361 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1408/Unknown  175s 96ms/step - categorical_accuracy: 0.7165 - loss: 1.1359 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1409/Unknown  175s 96ms/step - categorical_accuracy: 0.7165 - loss: 1.1358 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1410/Unknown  175s 96ms/step - categorical_accuracy: 0.7165 - loss: 1.1357 - mean_io_u: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   1411/Unknown  175s 96ms/step - categorical_accuracy: 0.7165 - loss: 1.1356 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1412/Unknown  175s 96ms/step - categorical_accuracy: 0.7166 - loss: 1.1355 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1413/Unknown  175s 96ms/step - categorical_accuracy: 0.7166 - loss: 1.1353 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1414/Unknown  175s 96ms/step - categorical_accuracy: 0.7166 - loss: 1.1352 - mean_io_u: 0.0780

<div class="k-default-codeblock">
```

```
</div>
   1415/Unknown  175s 96ms/step - categorical_accuracy: 0.7166 - loss: 1.1351 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1416/Unknown  175s 96ms/step - categorical_accuracy: 0.7167 - loss: 1.1350 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1417/Unknown  176s 96ms/step - categorical_accuracy: 0.7167 - loss: 1.1349 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1418/Unknown  176s 96ms/step - categorical_accuracy: 0.7167 - loss: 1.1347 - mean_io_u: 0.0781

<div class="k-default-codeblock">
```

```
</div>
   1419/Unknown  176s 96ms/step - categorical_accuracy: 0.7167 - loss: 1.1346 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1420/Unknown  176s 96ms/step - categorical_accuracy: 0.7167 - loss: 1.1345 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1421/Unknown  176s 96ms/step - categorical_accuracy: 0.7168 - loss: 1.1344 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1422/Unknown  176s 96ms/step - categorical_accuracy: 0.7168 - loss: 1.1343 - mean_io_u: 0.0782

<div class="k-default-codeblock">
```

```
</div>
   1423/Unknown  176s 96ms/step - categorical_accuracy: 0.7168 - loss: 1.1342 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1424/Unknown  176s 96ms/step - categorical_accuracy: 0.7168 - loss: 1.1340 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1425/Unknown  176s 96ms/step - categorical_accuracy: 0.7168 - loss: 1.1339 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1426/Unknown  176s 96ms/step - categorical_accuracy: 0.7169 - loss: 1.1338 - mean_io_u: 0.0783

<div class="k-default-codeblock">
```

```
</div>
   1427/Unknown  176s 96ms/step - categorical_accuracy: 0.7169 - loss: 1.1337 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1428/Unknown  176s 96ms/step - categorical_accuracy: 0.7169 - loss: 1.1336 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1429/Unknown  177s 96ms/step - categorical_accuracy: 0.7169 - loss: 1.1334 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1430/Unknown  177s 96ms/step - categorical_accuracy: 0.7170 - loss: 1.1333 - mean_io_u: 0.0784

<div class="k-default-codeblock">
```

```
</div>
   1431/Unknown  177s 96ms/step - categorical_accuracy: 0.7170 - loss: 1.1332 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1432/Unknown  177s 96ms/step - categorical_accuracy: 0.7170 - loss: 1.1331 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1433/Unknown  177s 96ms/step - categorical_accuracy: 0.7170 - loss: 1.1330 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1434/Unknown  177s 96ms/step - categorical_accuracy: 0.7170 - loss: 1.1329 - mean_io_u: 0.0785

<div class="k-default-codeblock">
```

```
</div>
   1435/Unknown  177s 96ms/step - categorical_accuracy: 0.7171 - loss: 1.1327 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1436/Unknown  177s 96ms/step - categorical_accuracy: 0.7171 - loss: 1.1326 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1437/Unknown  177s 96ms/step - categorical_accuracy: 0.7171 - loss: 1.1325 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1438/Unknown  177s 96ms/step - categorical_accuracy: 0.7171 - loss: 1.1324 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1439/Unknown  177s 96ms/step - categorical_accuracy: 0.7172 - loss: 1.1323 - mean_io_u: 0.0786

<div class="k-default-codeblock">
```

```
</div>
   1440/Unknown  177s 96ms/step - categorical_accuracy: 0.7172 - loss: 1.1322 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1441/Unknown  178s 96ms/step - categorical_accuracy: 0.7172 - loss: 1.1320 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1442/Unknown  178s 96ms/step - categorical_accuracy: 0.7172 - loss: 1.1319 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1443/Unknown  178s 96ms/step - categorical_accuracy: 0.7172 - loss: 1.1318 - mean_io_u: 0.0787

<div class="k-default-codeblock">
```

```
</div>
   1444/Unknown  178s 96ms/step - categorical_accuracy: 0.7173 - loss: 1.1317 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1445/Unknown  178s 96ms/step - categorical_accuracy: 0.7173 - loss: 1.1316 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1446/Unknown  178s 96ms/step - categorical_accuracy: 0.7173 - loss: 1.1315 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1447/Unknown  178s 96ms/step - categorical_accuracy: 0.7173 - loss: 1.1313 - mean_io_u: 0.0788

<div class="k-default-codeblock">
```

```
</div>
   1448/Unknown  178s 96ms/step - categorical_accuracy: 0.7174 - loss: 1.1312 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1449/Unknown  178s 96ms/step - categorical_accuracy: 0.7174 - loss: 1.1311 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1450/Unknown  178s 96ms/step - categorical_accuracy: 0.7174 - loss: 1.1310 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1451/Unknown  178s 96ms/step - categorical_accuracy: 0.7174 - loss: 1.1309 - mean_io_u: 0.0789

<div class="k-default-codeblock">
```

```
</div>
   1452/Unknown  178s 96ms/step - categorical_accuracy: 0.7174 - loss: 1.1308 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1453/Unknown  178s 96ms/step - categorical_accuracy: 0.7175 - loss: 1.1306 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1454/Unknown  178s 96ms/step - categorical_accuracy: 0.7175 - loss: 1.1305 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1455/Unknown  179s 96ms/step - categorical_accuracy: 0.7175 - loss: 1.1304 - mean_io_u: 0.0790

<div class="k-default-codeblock">
```

```
</div>
   1456/Unknown  179s 95ms/step - categorical_accuracy: 0.7175 - loss: 1.1303 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1457/Unknown  179s 95ms/step - categorical_accuracy: 0.7176 - loss: 1.1302 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1458/Unknown  179s 95ms/step - categorical_accuracy: 0.7176 - loss: 1.1301 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1459/Unknown  179s 95ms/step - categorical_accuracy: 0.7176 - loss: 1.1300 - mean_io_u: 0.0791

<div class="k-default-codeblock">
```

```
</div>
   1460/Unknown  179s 95ms/step - categorical_accuracy: 0.7176 - loss: 1.1298 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1461/Unknown  179s 95ms/step - categorical_accuracy: 0.7176 - loss: 1.1297 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1462/Unknown  179s 95ms/step - categorical_accuracy: 0.7177 - loss: 1.1296 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1463/Unknown  179s 95ms/step - categorical_accuracy: 0.7177 - loss: 1.1295 - mean_io_u: 0.0792

<div class="k-default-codeblock">
```

```
</div>
   1464/Unknown  179s 95ms/step - categorical_accuracy: 0.7177 - loss: 1.1294 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1465/Unknown  179s 95ms/step - categorical_accuracy: 0.7177 - loss: 1.1293 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1466/Unknown  179s 95ms/step - categorical_accuracy: 0.7177 - loss: 1.1292 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1467/Unknown  179s 95ms/step - categorical_accuracy: 0.7178 - loss: 1.1290 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1468/Unknown  179s 95ms/step - categorical_accuracy: 0.7178 - loss: 1.1289 - mean_io_u: 0.0793

<div class="k-default-codeblock">
```

```
</div>
   1469/Unknown  180s 95ms/step - categorical_accuracy: 0.7178 - loss: 1.1288 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1470/Unknown  180s 95ms/step - categorical_accuracy: 0.7178 - loss: 1.1287 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1471/Unknown  180s 95ms/step - categorical_accuracy: 0.7178 - loss: 1.1286 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1472/Unknown  180s 95ms/step - categorical_accuracy: 0.7179 - loss: 1.1285 - mean_io_u: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   1473/Unknown  180s 95ms/step - categorical_accuracy: 0.7179 - loss: 1.1284 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1474/Unknown  180s 95ms/step - categorical_accuracy: 0.7179 - loss: 1.1283 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1475/Unknown  180s 95ms/step - categorical_accuracy: 0.7179 - loss: 1.1281 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1476/Unknown  180s 95ms/step - categorical_accuracy: 0.7180 - loss: 1.1280 - mean_io_u: 0.0795

<div class="k-default-codeblock">
```

```
</div>
   1477/Unknown  180s 95ms/step - categorical_accuracy: 0.7180 - loss: 1.1279 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1478/Unknown  180s 95ms/step - categorical_accuracy: 0.7180 - loss: 1.1278 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1479/Unknown  180s 95ms/step - categorical_accuracy: 0.7180 - loss: 1.1277 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1480/Unknown  180s 95ms/step - categorical_accuracy: 0.7180 - loss: 1.1276 - mean_io_u: 0.0796

<div class="k-default-codeblock">
```

```
</div>
   1481/Unknown  181s 95ms/step - categorical_accuracy: 0.7181 - loss: 1.1275 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1482/Unknown  181s 95ms/step - categorical_accuracy: 0.7181 - loss: 1.1274 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1483/Unknown  181s 95ms/step - categorical_accuracy: 0.7181 - loss: 1.1272 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1484/Unknown  181s 95ms/step - categorical_accuracy: 0.7181 - loss: 1.1271 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1485/Unknown  181s 95ms/step - categorical_accuracy: 0.7181 - loss: 1.1270 - mean_io_u: 0.0797

<div class="k-default-codeblock">
```

```
</div>
   1486/Unknown  181s 95ms/step - categorical_accuracy: 0.7182 - loss: 1.1269 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1487/Unknown  181s 95ms/step - categorical_accuracy: 0.7182 - loss: 1.1268 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1488/Unknown  181s 95ms/step - categorical_accuracy: 0.7182 - loss: 1.1267 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1489/Unknown  181s 95ms/step - categorical_accuracy: 0.7182 - loss: 1.1266 - mean_io_u: 0.0798

<div class="k-default-codeblock">
```

```
</div>
   1490/Unknown  181s 95ms/step - categorical_accuracy: 0.7182 - loss: 1.1265 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1491/Unknown  181s 95ms/step - categorical_accuracy: 0.7183 - loss: 1.1264 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1492/Unknown  181s 95ms/step - categorical_accuracy: 0.7183 - loss: 1.1262 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1493/Unknown  181s 95ms/step - categorical_accuracy: 0.7183 - loss: 1.1261 - mean_io_u: 0.0799

<div class="k-default-codeblock">
```

```
</div>
   1494/Unknown  182s 95ms/step - categorical_accuracy: 0.7183 - loss: 1.1260 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1495/Unknown  182s 95ms/step - categorical_accuracy: 0.7183 - loss: 1.1259 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1496/Unknown  182s 95ms/step - categorical_accuracy: 0.7184 - loss: 1.1258 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1497/Unknown  182s 95ms/step - categorical_accuracy: 0.7184 - loss: 1.1257 - mean_io_u: 0.0800

<div class="k-default-codeblock">
```

```
</div>
   1498/Unknown  182s 95ms/step - categorical_accuracy: 0.7184 - loss: 1.1256 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1499/Unknown  182s 95ms/step - categorical_accuracy: 0.7184 - loss: 1.1255 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1500/Unknown  182s 95ms/step - categorical_accuracy: 0.7184 - loss: 1.1254 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1501/Unknown  182s 95ms/step - categorical_accuracy: 0.7185 - loss: 1.1253 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1502/Unknown  182s 95ms/step - categorical_accuracy: 0.7185 - loss: 1.1251 - mean_io_u: 0.0801

<div class="k-default-codeblock">
```

```
</div>
   1503/Unknown  182s 95ms/step - categorical_accuracy: 0.7185 - loss: 1.1250 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1504/Unknown  182s 95ms/step - categorical_accuracy: 0.7185 - loss: 1.1249 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1505/Unknown  182s 95ms/step - categorical_accuracy: 0.7186 - loss: 1.1248 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1506/Unknown  182s 95ms/step - categorical_accuracy: 0.7186 - loss: 1.1247 - mean_io_u: 0.0802

<div class="k-default-codeblock">
```

```
</div>
   1507/Unknown  183s 95ms/step - categorical_accuracy: 0.7186 - loss: 1.1246 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1508/Unknown  183s 95ms/step - categorical_accuracy: 0.7186 - loss: 1.1245 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1509/Unknown  183s 95ms/step - categorical_accuracy: 0.7186 - loss: 1.1244 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1510/Unknown  183s 95ms/step - categorical_accuracy: 0.7187 - loss: 1.1243 - mean_io_u: 0.0803

<div class="k-default-codeblock">
```

```
</div>
   1511/Unknown  183s 95ms/step - categorical_accuracy: 0.7187 - loss: 1.1242 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1512/Unknown  183s 95ms/step - categorical_accuracy: 0.7187 - loss: 1.1241 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1513/Unknown  183s 95ms/step - categorical_accuracy: 0.7187 - loss: 1.1239 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1514/Unknown  183s 95ms/step - categorical_accuracy: 0.7187 - loss: 1.1238 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1515/Unknown  183s 95ms/step - categorical_accuracy: 0.7188 - loss: 1.1237 - mean_io_u: 0.0804

<div class="k-default-codeblock">
```

```
</div>
   1516/Unknown  183s 95ms/step - categorical_accuracy: 0.7188 - loss: 1.1236 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1517/Unknown  183s 95ms/step - categorical_accuracy: 0.7188 - loss: 1.1235 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1518/Unknown  183s 95ms/step - categorical_accuracy: 0.7188 - loss: 1.1234 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1519/Unknown  183s 95ms/step - categorical_accuracy: 0.7188 - loss: 1.1233 - mean_io_u: 0.0805

<div class="k-default-codeblock">
```

```
</div>
   1520/Unknown  184s 95ms/step - categorical_accuracy: 0.7189 - loss: 1.1232 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1521/Unknown  184s 95ms/step - categorical_accuracy: 0.7189 - loss: 1.1231 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1522/Unknown  184s 95ms/step - categorical_accuracy: 0.7189 - loss: 1.1230 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1523/Unknown  184s 95ms/step - categorical_accuracy: 0.7189 - loss: 1.1229 - mean_io_u: 0.0806

<div class="k-default-codeblock">
```

```
</div>
   1524/Unknown  184s 95ms/step - categorical_accuracy: 0.7189 - loss: 1.1228 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1525/Unknown  184s 95ms/step - categorical_accuracy: 0.7190 - loss: 1.1227 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1526/Unknown  184s 95ms/step - categorical_accuracy: 0.7190 - loss: 1.1225 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1527/Unknown  184s 95ms/step - categorical_accuracy: 0.7190 - loss: 1.1224 - mean_io_u: 0.0807

<div class="k-default-codeblock">
```

```
</div>
   1528/Unknown  184s 95ms/step - categorical_accuracy: 0.7190 - loss: 1.1223 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1529/Unknown  184s 95ms/step - categorical_accuracy: 0.7190 - loss: 1.1222 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1530/Unknown  184s 95ms/step - categorical_accuracy: 0.7191 - loss: 1.1221 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1531/Unknown  184s 95ms/step - categorical_accuracy: 0.7191 - loss: 1.1220 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1532/Unknown  185s 95ms/step - categorical_accuracy: 0.7191 - loss: 1.1219 - mean_io_u: 0.0808

<div class="k-default-codeblock">
```

```
</div>
   1533/Unknown  185s 95ms/step - categorical_accuracy: 0.7191 - loss: 1.1218 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1534/Unknown  185s 95ms/step - categorical_accuracy: 0.7191 - loss: 1.1217 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1535/Unknown  185s 95ms/step - categorical_accuracy: 0.7192 - loss: 1.1216 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1536/Unknown  185s 95ms/step - categorical_accuracy: 0.7192 - loss: 1.1215 - mean_io_u: 0.0809

<div class="k-default-codeblock">
```

```
</div>
   1537/Unknown  185s 95ms/step - categorical_accuracy: 0.7192 - loss: 1.1214 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1538/Unknown  185s 95ms/step - categorical_accuracy: 0.7192 - loss: 1.1213 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1539/Unknown  185s 95ms/step - categorical_accuracy: 0.7192 - loss: 1.1212 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1540/Unknown  185s 95ms/step - categorical_accuracy: 0.7192 - loss: 1.1211 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1541/Unknown  185s 95ms/step - categorical_accuracy: 0.7193 - loss: 1.1209 - mean_io_u: 0.0810

<div class="k-default-codeblock">
```

```
</div>
   1542/Unknown  185s 95ms/step - categorical_accuracy: 0.7193 - loss: 1.1208 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1543/Unknown  185s 95ms/step - categorical_accuracy: 0.7193 - loss: 1.1207 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1544/Unknown  186s 95ms/step - categorical_accuracy: 0.7193 - loss: 1.1206 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1545/Unknown  186s 95ms/step - categorical_accuracy: 0.7193 - loss: 1.1205 - mean_io_u: 0.0811

<div class="k-default-codeblock">
```

```
</div>
   1546/Unknown  186s 95ms/step - categorical_accuracy: 0.7194 - loss: 1.1204 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1547/Unknown  186s 95ms/step - categorical_accuracy: 0.7194 - loss: 1.1203 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1548/Unknown  186s 95ms/step - categorical_accuracy: 0.7194 - loss: 1.1202 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1549/Unknown  186s 95ms/step - categorical_accuracy: 0.7194 - loss: 1.1201 - mean_io_u: 0.0812

<div class="k-default-codeblock">
```

```
</div>
   1550/Unknown  186s 94ms/step - categorical_accuracy: 0.7194 - loss: 1.1200 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1551/Unknown  186s 94ms/step - categorical_accuracy: 0.7195 - loss: 1.1199 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1552/Unknown  186s 94ms/step - categorical_accuracy: 0.7195 - loss: 1.1198 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1553/Unknown  186s 94ms/step - categorical_accuracy: 0.7195 - loss: 1.1197 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1554/Unknown  186s 94ms/step - categorical_accuracy: 0.7195 - loss: 1.1196 - mean_io_u: 0.0813

<div class="k-default-codeblock">
```

```
</div>
   1555/Unknown  186s 94ms/step - categorical_accuracy: 0.7195 - loss: 1.1195 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1556/Unknown  186s 94ms/step - categorical_accuracy: 0.7196 - loss: 1.1194 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1557/Unknown  187s 94ms/step - categorical_accuracy: 0.7196 - loss: 1.1193 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1558/Unknown  187s 94ms/step - categorical_accuracy: 0.7196 - loss: 1.1192 - mean_io_u: 0.0814

<div class="k-default-codeblock">
```

```
</div>
   1559/Unknown  187s 94ms/step - categorical_accuracy: 0.7196 - loss: 1.1190 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1560/Unknown  187s 94ms/step - categorical_accuracy: 0.7196 - loss: 1.1189 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1561/Unknown  187s 94ms/step - categorical_accuracy: 0.7197 - loss: 1.1188 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1562/Unknown  187s 94ms/step - categorical_accuracy: 0.7197 - loss: 1.1187 - mean_io_u: 0.0815

<div class="k-default-codeblock">
```

```
</div>
   1563/Unknown  187s 94ms/step - categorical_accuracy: 0.7197 - loss: 1.1186 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1564/Unknown  187s 94ms/step - categorical_accuracy: 0.7197 - loss: 1.1185 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1565/Unknown  187s 94ms/step - categorical_accuracy: 0.7197 - loss: 1.1184 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1566/Unknown  187s 94ms/step - categorical_accuracy: 0.7198 - loss: 1.1183 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1567/Unknown  187s 94ms/step - categorical_accuracy: 0.7198 - loss: 1.1182 - mean_io_u: 0.0816

<div class="k-default-codeblock">
```

```
</div>
   1568/Unknown  187s 94ms/step - categorical_accuracy: 0.7198 - loss: 1.1181 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1569/Unknown  188s 94ms/step - categorical_accuracy: 0.7198 - loss: 1.1180 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1570/Unknown  188s 94ms/step - categorical_accuracy: 0.7198 - loss: 1.1179 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1571/Unknown  188s 94ms/step - categorical_accuracy: 0.7199 - loss: 1.1178 - mean_io_u: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   1572/Unknown  188s 94ms/step - categorical_accuracy: 0.7199 - loss: 1.1177 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1573/Unknown  188s 94ms/step - categorical_accuracy: 0.7199 - loss: 1.1176 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1574/Unknown  188s 94ms/step - categorical_accuracy: 0.7199 - loss: 1.1175 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1575/Unknown  188s 94ms/step - categorical_accuracy: 0.7199 - loss: 1.1174 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1576/Unknown  188s 94ms/step - categorical_accuracy: 0.7200 - loss: 1.1173 - mean_io_u: 0.0818

<div class="k-default-codeblock">
```

```
</div>
   1577/Unknown  188s 94ms/step - categorical_accuracy: 0.7200 - loss: 1.1172 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1578/Unknown  188s 94ms/step - categorical_accuracy: 0.7200 - loss: 1.1171 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1579/Unknown  188s 94ms/step - categorical_accuracy: 0.7200 - loss: 1.1170 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1580/Unknown  188s 94ms/step - categorical_accuracy: 0.7200 - loss: 1.1169 - mean_io_u: 0.0819

<div class="k-default-codeblock">
```

```
</div>
   1581/Unknown  188s 94ms/step - categorical_accuracy: 0.7201 - loss: 1.1168 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1582/Unknown  189s 94ms/step - categorical_accuracy: 0.7201 - loss: 1.1167 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1583/Unknown  189s 94ms/step - categorical_accuracy: 0.7201 - loss: 1.1165 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1584/Unknown  189s 94ms/step - categorical_accuracy: 0.7201 - loss: 1.1164 - mean_io_u: 0.0820

<div class="k-default-codeblock">
```

```
</div>
   1585/Unknown  189s 94ms/step - categorical_accuracy: 0.7201 - loss: 1.1163 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1586/Unknown  189s 94ms/step - categorical_accuracy: 0.7201 - loss: 1.1162 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1587/Unknown  189s 94ms/step - categorical_accuracy: 0.7202 - loss: 1.1161 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1588/Unknown  189s 94ms/step - categorical_accuracy: 0.7202 - loss: 1.1160 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1589/Unknown  189s 94ms/step - categorical_accuracy: 0.7202 - loss: 1.1159 - mean_io_u: 0.0821

<div class="k-default-codeblock">
```

```
</div>
   1590/Unknown  189s 94ms/step - categorical_accuracy: 0.7202 - loss: 1.1158 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1591/Unknown  189s 94ms/step - categorical_accuracy: 0.7202 - loss: 1.1157 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1592/Unknown  189s 94ms/step - categorical_accuracy: 0.7203 - loss: 1.1156 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1593/Unknown  189s 94ms/step - categorical_accuracy: 0.7203 - loss: 1.1155 - mean_io_u: 0.0822

<div class="k-default-codeblock">
```

```
</div>
   1594/Unknown  189s 94ms/step - categorical_accuracy: 0.7203 - loss: 1.1154 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1595/Unknown  190s 94ms/step - categorical_accuracy: 0.7203 - loss: 1.1153 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1596/Unknown  190s 94ms/step - categorical_accuracy: 0.7203 - loss: 1.1152 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1597/Unknown  190s 94ms/step - categorical_accuracy: 0.7204 - loss: 1.1151 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1598/Unknown  190s 94ms/step - categorical_accuracy: 0.7204 - loss: 1.1150 - mean_io_u: 0.0823

<div class="k-default-codeblock">
```

```
</div>
   1599/Unknown  190s 94ms/step - categorical_accuracy: 0.7204 - loss: 1.1149 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1600/Unknown  190s 94ms/step - categorical_accuracy: 0.7204 - loss: 1.1148 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1601/Unknown  190s 94ms/step - categorical_accuracy: 0.7204 - loss: 1.1147 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1602/Unknown  190s 94ms/step - categorical_accuracy: 0.7205 - loss: 1.1146 - mean_io_u: 0.0824

<div class="k-default-codeblock">
```

```
</div>
   1603/Unknown  190s 94ms/step - categorical_accuracy: 0.7205 - loss: 1.1145 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1604/Unknown  190s 94ms/step - categorical_accuracy: 0.7205 - loss: 1.1144 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1605/Unknown  190s 94ms/step - categorical_accuracy: 0.7205 - loss: 1.1143 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1606/Unknown  190s 94ms/step - categorical_accuracy: 0.7205 - loss: 1.1142 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1607/Unknown  190s 94ms/step - categorical_accuracy: 0.7205 - loss: 1.1141 - mean_io_u: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   1608/Unknown  190s 94ms/step - categorical_accuracy: 0.7206 - loss: 1.1140 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1609/Unknown  190s 94ms/step - categorical_accuracy: 0.7206 - loss: 1.1139 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1610/Unknown  191s 94ms/step - categorical_accuracy: 0.7206 - loss: 1.1138 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1611/Unknown  191s 94ms/step - categorical_accuracy: 0.7206 - loss: 1.1137 - mean_io_u: 0.0826

<div class="k-default-codeblock">
```

```
</div>
   1612/Unknown  191s 94ms/step - categorical_accuracy: 0.7206 - loss: 1.1136 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1613/Unknown  191s 94ms/step - categorical_accuracy: 0.7207 - loss: 1.1135 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1614/Unknown  191s 94ms/step - categorical_accuracy: 0.7207 - loss: 1.1134 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1615/Unknown  191s 94ms/step - categorical_accuracy: 0.7207 - loss: 1.1133 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1616/Unknown  191s 94ms/step - categorical_accuracy: 0.7207 - loss: 1.1132 - mean_io_u: 0.0827

<div class="k-default-codeblock">
```

```
</div>
   1617/Unknown  191s 94ms/step - categorical_accuracy: 0.7207 - loss: 1.1131 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1618/Unknown  191s 94ms/step - categorical_accuracy: 0.7207 - loss: 1.1130 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1619/Unknown  191s 94ms/step - categorical_accuracy: 0.7208 - loss: 1.1129 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1620/Unknown  191s 94ms/step - categorical_accuracy: 0.7208 - loss: 1.1128 - mean_io_u: 0.0828

<div class="k-default-codeblock">
```

```
</div>
   1621/Unknown  191s 94ms/step - categorical_accuracy: 0.7208 - loss: 1.1127 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1622/Unknown  191s 94ms/step - categorical_accuracy: 0.7208 - loss: 1.1126 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1623/Unknown  192s 94ms/step - categorical_accuracy: 0.7208 - loss: 1.1125 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1624/Unknown  192s 94ms/step - categorical_accuracy: 0.7209 - loss: 1.1124 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1625/Unknown  192s 94ms/step - categorical_accuracy: 0.7209 - loss: 1.1123 - mean_io_u: 0.0829

<div class="k-default-codeblock">
```

```
</div>
   1626/Unknown  192s 94ms/step - categorical_accuracy: 0.7209 - loss: 1.1122 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1627/Unknown  192s 94ms/step - categorical_accuracy: 0.7209 - loss: 1.1121 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1628/Unknown  192s 94ms/step - categorical_accuracy: 0.7209 - loss: 1.1120 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1629/Unknown  192s 94ms/step - categorical_accuracy: 0.7210 - loss: 1.1119 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1630/Unknown  192s 94ms/step - categorical_accuracy: 0.7210 - loss: 1.1118 - mean_io_u: 0.0830

<div class="k-default-codeblock">
```

```
</div>
   1631/Unknown  192s 94ms/step - categorical_accuracy: 0.7210 - loss: 1.1117 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1632/Unknown  192s 94ms/step - categorical_accuracy: 0.7210 - loss: 1.1116 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1633/Unknown  192s 94ms/step - categorical_accuracy: 0.7210 - loss: 1.1115 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1634/Unknown  192s 94ms/step - categorical_accuracy: 0.7210 - loss: 1.1114 - mean_io_u: 0.0831

<div class="k-default-codeblock">
```

```
</div>
   1635/Unknown  192s 94ms/step - categorical_accuracy: 0.7211 - loss: 1.1113 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1636/Unknown  193s 94ms/step - categorical_accuracy: 0.7211 - loss: 1.1112 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1637/Unknown  193s 93ms/step - categorical_accuracy: 0.7211 - loss: 1.1111 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1638/Unknown  193s 93ms/step - categorical_accuracy: 0.7211 - loss: 1.1110 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1639/Unknown  193s 93ms/step - categorical_accuracy: 0.7211 - loss: 1.1109 - mean_io_u: 0.0832

<div class="k-default-codeblock">
```

```
</div>
   1640/Unknown  193s 93ms/step - categorical_accuracy: 0.7212 - loss: 1.1108 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1641/Unknown  193s 93ms/step - categorical_accuracy: 0.7212 - loss: 1.1108 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1642/Unknown  193s 93ms/step - categorical_accuracy: 0.7212 - loss: 1.1107 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1643/Unknown  193s 93ms/step - categorical_accuracy: 0.7212 - loss: 1.1106 - mean_io_u: 0.0833

<div class="k-default-codeblock">
```

```
</div>
   1644/Unknown  193s 93ms/step - categorical_accuracy: 0.7212 - loss: 1.1105 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1645/Unknown  193s 93ms/step - categorical_accuracy: 0.7212 - loss: 1.1104 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1646/Unknown  193s 93ms/step - categorical_accuracy: 0.7213 - loss: 1.1103 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1647/Unknown  193s 93ms/step - categorical_accuracy: 0.7213 - loss: 1.1102 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1648/Unknown  193s 93ms/step - categorical_accuracy: 0.7213 - loss: 1.1101 - mean_io_u: 0.0834

<div class="k-default-codeblock">
```

```
</div>
   1649/Unknown  194s 93ms/step - categorical_accuracy: 0.7213 - loss: 1.1100 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1650/Unknown  194s 93ms/step - categorical_accuracy: 0.7213 - loss: 1.1099 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1651/Unknown  194s 93ms/step - categorical_accuracy: 0.7214 - loss: 1.1098 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1652/Unknown  194s 93ms/step - categorical_accuracy: 0.7214 - loss: 1.1097 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1653/Unknown  194s 93ms/step - categorical_accuracy: 0.7214 - loss: 1.1096 - mean_io_u: 0.0835

<div class="k-default-codeblock">
```

```
</div>
   1654/Unknown  194s 93ms/step - categorical_accuracy: 0.7214 - loss: 1.1095 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1655/Unknown  194s 93ms/step - categorical_accuracy: 0.7214 - loss: 1.1094 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1656/Unknown  194s 93ms/step - categorical_accuracy: 0.7214 - loss: 1.1093 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1657/Unknown  194s 93ms/step - categorical_accuracy: 0.7215 - loss: 1.1092 - mean_io_u: 0.0836

<div class="k-default-codeblock">
```

```
</div>
   1658/Unknown  194s 93ms/step - categorical_accuracy: 0.7215 - loss: 1.1091 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1659/Unknown  194s 93ms/step - categorical_accuracy: 0.7215 - loss: 1.1090 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1660/Unknown  194s 93ms/step - categorical_accuracy: 0.7215 - loss: 1.1089 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1661/Unknown  195s 93ms/step - categorical_accuracy: 0.7215 - loss: 1.1088 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1662/Unknown  195s 93ms/step - categorical_accuracy: 0.7216 - loss: 1.1087 - mean_io_u: 0.0837

<div class="k-default-codeblock">
```

```
</div>
   1663/Unknown  195s 93ms/step - categorical_accuracy: 0.7216 - loss: 1.1086 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1664/Unknown  195s 93ms/step - categorical_accuracy: 0.7216 - loss: 1.1085 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1665/Unknown  195s 93ms/step - categorical_accuracy: 0.7216 - loss: 1.1084 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1666/Unknown  195s 93ms/step - categorical_accuracy: 0.7216 - loss: 1.1083 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1667/Unknown  195s 93ms/step - categorical_accuracy: 0.7216 - loss: 1.1082 - mean_io_u: 0.0838

<div class="k-default-codeblock">
```

```
</div>
   1668/Unknown  195s 93ms/step - categorical_accuracy: 0.7217 - loss: 1.1081 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1669/Unknown  195s 93ms/step - categorical_accuracy: 0.7217 - loss: 1.1080 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1670/Unknown  195s 93ms/step - categorical_accuracy: 0.7217 - loss: 1.1079 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1671/Unknown  195s 93ms/step - categorical_accuracy: 0.7217 - loss: 1.1079 - mean_io_u: 0.0839

<div class="k-default-codeblock">
```

```
</div>
   1672/Unknown  195s 93ms/step - categorical_accuracy: 0.7217 - loss: 1.1078 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1673/Unknown  196s 93ms/step - categorical_accuracy: 0.7218 - loss: 1.1077 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1674/Unknown  196s 93ms/step - categorical_accuracy: 0.7218 - loss: 1.1076 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1675/Unknown  196s 93ms/step - categorical_accuracy: 0.7218 - loss: 1.1075 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1676/Unknown  196s 93ms/step - categorical_accuracy: 0.7218 - loss: 1.1074 - mean_io_u: 0.0840

<div class="k-default-codeblock">
```

```
</div>
   1677/Unknown  196s 93ms/step - categorical_accuracy: 0.7218 - loss: 1.1073 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1678/Unknown  196s 93ms/step - categorical_accuracy: 0.7218 - loss: 1.1072 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1679/Unknown  196s 93ms/step - categorical_accuracy: 0.7219 - loss: 1.1071 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1680/Unknown  196s 93ms/step - categorical_accuracy: 0.7219 - loss: 1.1070 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1681/Unknown  196s 93ms/step - categorical_accuracy: 0.7219 - loss: 1.1069 - mean_io_u: 0.0841

<div class="k-default-codeblock">
```

```
</div>
   1682/Unknown  196s 93ms/step - categorical_accuracy: 0.7219 - loss: 1.1068 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1683/Unknown  196s 93ms/step - categorical_accuracy: 0.7219 - loss: 1.1067 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1684/Unknown  196s 93ms/step - categorical_accuracy: 0.7219 - loss: 1.1066 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1685/Unknown  196s 93ms/step - categorical_accuracy: 0.7220 - loss: 1.1065 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1686/Unknown  197s 93ms/step - categorical_accuracy: 0.7220 - loss: 1.1064 - mean_io_u: 0.0842

<div class="k-default-codeblock">
```

```
</div>
   1687/Unknown  197s 93ms/step - categorical_accuracy: 0.7220 - loss: 1.1063 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1688/Unknown  197s 93ms/step - categorical_accuracy: 0.7220 - loss: 1.1063 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1689/Unknown  197s 93ms/step - categorical_accuracy: 0.7220 - loss: 1.1062 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1690/Unknown  197s 93ms/step - categorical_accuracy: 0.7220 - loss: 1.1061 - mean_io_u: 0.0843

<div class="k-default-codeblock">
```

```
</div>
   1691/Unknown  197s 93ms/step - categorical_accuracy: 0.7221 - loss: 1.1060 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1692/Unknown  197s 93ms/step - categorical_accuracy: 0.7221 - loss: 1.1059 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1693/Unknown  197s 93ms/step - categorical_accuracy: 0.7221 - loss: 1.1058 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1694/Unknown  197s 93ms/step - categorical_accuracy: 0.7221 - loss: 1.1057 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1695/Unknown  197s 93ms/step - categorical_accuracy: 0.7221 - loss: 1.1056 - mean_io_u: 0.0844

<div class="k-default-codeblock">
```

```
</div>
   1696/Unknown  197s 93ms/step - categorical_accuracy: 0.7222 - loss: 1.1055 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1697/Unknown  197s 93ms/step - categorical_accuracy: 0.7222 - loss: 1.1054 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1698/Unknown  198s 93ms/step - categorical_accuracy: 0.7222 - loss: 1.1053 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1699/Unknown  198s 93ms/step - categorical_accuracy: 0.7222 - loss: 1.1052 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1700/Unknown  198s 93ms/step - categorical_accuracy: 0.7222 - loss: 1.1051 - mean_io_u: 0.0845

<div class="k-default-codeblock">
```

```
</div>
   1701/Unknown  198s 93ms/step - categorical_accuracy: 0.7222 - loss: 1.1050 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1702/Unknown  198s 93ms/step - categorical_accuracy: 0.7223 - loss: 1.1049 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1703/Unknown  198s 93ms/step - categorical_accuracy: 0.7223 - loss: 1.1048 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1704/Unknown  198s 93ms/step - categorical_accuracy: 0.7223 - loss: 1.1048 - mean_io_u: 0.0846

<div class="k-default-codeblock">
```

```
</div>
   1705/Unknown  198s 93ms/step - categorical_accuracy: 0.7223 - loss: 1.1047 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1706/Unknown  198s 93ms/step - categorical_accuracy: 0.7223 - loss: 1.1046 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1707/Unknown  198s 93ms/step - categorical_accuracy: 0.7223 - loss: 1.1045 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1708/Unknown  198s 93ms/step - categorical_accuracy: 0.7224 - loss: 1.1044 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1709/Unknown  198s 93ms/step - categorical_accuracy: 0.7224 - loss: 1.1043 - mean_io_u: 0.0847

<div class="k-default-codeblock">
```

```
</div>
   1710/Unknown  198s 93ms/step - categorical_accuracy: 0.7224 - loss: 1.1042 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1711/Unknown  199s 93ms/step - categorical_accuracy: 0.7224 - loss: 1.1041 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1712/Unknown  199s 93ms/step - categorical_accuracy: 0.7224 - loss: 1.1040 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1713/Unknown  199s 93ms/step - categorical_accuracy: 0.7224 - loss: 1.1039 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1714/Unknown  199s 93ms/step - categorical_accuracy: 0.7225 - loss: 1.1038 - mean_io_u: 0.0848

<div class="k-default-codeblock">
```

```
</div>
   1715/Unknown  199s 93ms/step - categorical_accuracy: 0.7225 - loss: 1.1037 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1716/Unknown  199s 93ms/step - categorical_accuracy: 0.7225 - loss: 1.1036 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1717/Unknown  199s 93ms/step - categorical_accuracy: 0.7225 - loss: 1.1035 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1718/Unknown  199s 93ms/step - categorical_accuracy: 0.7225 - loss: 1.1035 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1719/Unknown  199s 93ms/step - categorical_accuracy: 0.7225 - loss: 1.1034 - mean_io_u: 0.0849

<div class="k-default-codeblock">
```

```
</div>
   1720/Unknown  199s 93ms/step - categorical_accuracy: 0.7226 - loss: 1.1033 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1721/Unknown  199s 93ms/step - categorical_accuracy: 0.7226 - loss: 1.1032 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1722/Unknown  199s 93ms/step - categorical_accuracy: 0.7226 - loss: 1.1031 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1723/Unknown  200s 93ms/step - categorical_accuracy: 0.7226 - loss: 1.1030 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1724/Unknown  200s 93ms/step - categorical_accuracy: 0.7226 - loss: 1.1029 - mean_io_u: 0.0850

<div class="k-default-codeblock">
```

```
</div>
   1725/Unknown  200s 93ms/step - categorical_accuracy: 0.7226 - loss: 1.1028 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1726/Unknown  200s 93ms/step - categorical_accuracy: 0.7227 - loss: 1.1027 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1727/Unknown  200s 93ms/step - categorical_accuracy: 0.7227 - loss: 1.1026 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1728/Unknown  200s 93ms/step - categorical_accuracy: 0.7227 - loss: 1.1025 - mean_io_u: 0.0851

<div class="k-default-codeblock">
```

```
</div>
   1729/Unknown  200s 93ms/step - categorical_accuracy: 0.7227 - loss: 1.1024 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1730/Unknown  200s 93ms/step - categorical_accuracy: 0.7227 - loss: 1.1024 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1731/Unknown  200s 93ms/step - categorical_accuracy: 0.7228 - loss: 1.1023 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1732/Unknown  200s 93ms/step - categorical_accuracy: 0.7228 - loss: 1.1022 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1733/Unknown  200s 93ms/step - categorical_accuracy: 0.7228 - loss: 1.1021 - mean_io_u: 0.0852

<div class="k-default-codeblock">
```

```
</div>
   1734/Unknown  200s 93ms/step - categorical_accuracy: 0.7228 - loss: 1.1020 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1735/Unknown  201s 93ms/step - categorical_accuracy: 0.7228 - loss: 1.1019 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1736/Unknown  201s 93ms/step - categorical_accuracy: 0.7228 - loss: 1.1018 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1737/Unknown  201s 93ms/step - categorical_accuracy: 0.7229 - loss: 1.1017 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1738/Unknown  201s 93ms/step - categorical_accuracy: 0.7229 - loss: 1.1016 - mean_io_u: 0.0853

<div class="k-default-codeblock">
```

```
</div>
   1739/Unknown  201s 93ms/step - categorical_accuracy: 0.7229 - loss: 1.1015 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1740/Unknown  201s 93ms/step - categorical_accuracy: 0.7229 - loss: 1.1015 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1741/Unknown  201s 93ms/step - categorical_accuracy: 0.7229 - loss: 1.1014 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1742/Unknown  201s 93ms/step - categorical_accuracy: 0.7229 - loss: 1.1013 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1743/Unknown  201s 93ms/step - categorical_accuracy: 0.7230 - loss: 1.1012 - mean_io_u: 0.0854

<div class="k-default-codeblock">
```

```
</div>
   1744/Unknown  201s 93ms/step - categorical_accuracy: 0.7230 - loss: 1.1011 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1745/Unknown  201s 93ms/step - categorical_accuracy: 0.7230 - loss: 1.1010 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1746/Unknown  201s 93ms/step - categorical_accuracy: 0.7230 - loss: 1.1009 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1747/Unknown  201s 93ms/step - categorical_accuracy: 0.7230 - loss: 1.1008 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1748/Unknown  201s 93ms/step - categorical_accuracy: 0.7230 - loss: 1.1007 - mean_io_u: 0.0855

<div class="k-default-codeblock">
```

```
</div>
   1749/Unknown  202s 93ms/step - categorical_accuracy: 0.7231 - loss: 1.1006 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1750/Unknown  202s 93ms/step - categorical_accuracy: 0.7231 - loss: 1.1006 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1751/Unknown  202s 93ms/step - categorical_accuracy: 0.7231 - loss: 1.1005 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1752/Unknown  202s 93ms/step - categorical_accuracy: 0.7231 - loss: 1.1004 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1753/Unknown  202s 93ms/step - categorical_accuracy: 0.7231 - loss: 1.1003 - mean_io_u: 0.0856

<div class="k-default-codeblock">
```

```
</div>
   1754/Unknown  202s 93ms/step - categorical_accuracy: 0.7231 - loss: 1.1002 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1755/Unknown  202s 93ms/step - categorical_accuracy: 0.7231 - loss: 1.1001 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1756/Unknown  202s 92ms/step - categorical_accuracy: 0.7232 - loss: 1.1000 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1757/Unknown  202s 92ms/step - categorical_accuracy: 0.7232 - loss: 1.0999 - mean_io_u: 0.0857

<div class="k-default-codeblock">
```

```
</div>
   1758/Unknown  202s 93ms/step - categorical_accuracy: 0.7232 - loss: 1.0998 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1759/Unknown  202s 92ms/step - categorical_accuracy: 0.7232 - loss: 1.0998 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1760/Unknown  202s 92ms/step - categorical_accuracy: 0.7232 - loss: 1.0997 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1761/Unknown  202s 92ms/step - categorical_accuracy: 0.7232 - loss: 1.0996 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1762/Unknown  202s 92ms/step - categorical_accuracy: 0.7233 - loss: 1.0995 - mean_io_u: 0.0858

<div class="k-default-codeblock">
```

```
</div>
   1763/Unknown  202s 92ms/step - categorical_accuracy: 0.7233 - loss: 1.0994 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1764/Unknown  203s 92ms/step - categorical_accuracy: 0.7233 - loss: 1.0993 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1765/Unknown  203s 92ms/step - categorical_accuracy: 0.7233 - loss: 1.0992 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1766/Unknown  203s 92ms/step - categorical_accuracy: 0.7233 - loss: 1.0991 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1767/Unknown  203s 92ms/step - categorical_accuracy: 0.7233 - loss: 1.0991 - mean_io_u: 0.0859

<div class="k-default-codeblock">
```

```
</div>
   1768/Unknown  203s 92ms/step - categorical_accuracy: 0.7234 - loss: 1.0990 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1769/Unknown  203s 92ms/step - categorical_accuracy: 0.7234 - loss: 1.0989 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1770/Unknown  203s 92ms/step - categorical_accuracy: 0.7234 - loss: 1.0988 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1771/Unknown  203s 92ms/step - categorical_accuracy: 0.7234 - loss: 1.0987 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1772/Unknown  203s 92ms/step - categorical_accuracy: 0.7234 - loss: 1.0986 - mean_io_u: 0.0860

<div class="k-default-codeblock">
```

```
</div>
   1773/Unknown  203s 92ms/step - categorical_accuracy: 0.7234 - loss: 1.0985 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1774/Unknown  203s 92ms/step - categorical_accuracy: 0.7235 - loss: 1.0984 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1775/Unknown  203s 92ms/step - categorical_accuracy: 0.7235 - loss: 1.0984 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1776/Unknown  204s 92ms/step - categorical_accuracy: 0.7235 - loss: 1.0983 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1777/Unknown  204s 92ms/step - categorical_accuracy: 0.7235 - loss: 1.0982 - mean_io_u: 0.0861

<div class="k-default-codeblock">
```

```
</div>
   1778/Unknown  204s 92ms/step - categorical_accuracy: 0.7235 - loss: 1.0981 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1779/Unknown  204s 92ms/step - categorical_accuracy: 0.7235 - loss: 1.0980 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1780/Unknown  204s 92ms/step - categorical_accuracy: 0.7236 - loss: 1.0979 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1781/Unknown  204s 92ms/step - categorical_accuracy: 0.7236 - loss: 1.0978 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1782/Unknown  204s 92ms/step - categorical_accuracy: 0.7236 - loss: 1.0977 - mean_io_u: 0.0862

<div class="k-default-codeblock">
```

```
</div>
   1783/Unknown  204s 92ms/step - categorical_accuracy: 0.7236 - loss: 1.0977 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1784/Unknown  204s 92ms/step - categorical_accuracy: 0.7236 - loss: 1.0976 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1785/Unknown  204s 92ms/step - categorical_accuracy: 0.7236 - loss: 1.0975 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1786/Unknown  204s 92ms/step - categorical_accuracy: 0.7237 - loss: 1.0974 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1787/Unknown  204s 92ms/step - categorical_accuracy: 0.7237 - loss: 1.0973 - mean_io_u: 0.0863

<div class="k-default-codeblock">
```

```
</div>
   1788/Unknown  204s 92ms/step - categorical_accuracy: 0.7237 - loss: 1.0972 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1789/Unknown  205s 92ms/step - categorical_accuracy: 0.7237 - loss: 1.0971 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1790/Unknown  205s 92ms/step - categorical_accuracy: 0.7237 - loss: 1.0970 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1791/Unknown  205s 92ms/step - categorical_accuracy: 0.7237 - loss: 1.0970 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1792/Unknown  205s 92ms/step - categorical_accuracy: 0.7237 - loss: 1.0969 - mean_io_u: 0.0864

<div class="k-default-codeblock">
```

```
</div>
   1793/Unknown  205s 92ms/step - categorical_accuracy: 0.7238 - loss: 1.0968 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1794/Unknown  205s 92ms/step - categorical_accuracy: 0.7238 - loss: 1.0967 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1795/Unknown  205s 92ms/step - categorical_accuracy: 0.7238 - loss: 1.0966 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1796/Unknown  205s 92ms/step - categorical_accuracy: 0.7238 - loss: 1.0965 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1797/Unknown  205s 92ms/step - categorical_accuracy: 0.7238 - loss: 1.0964 - mean_io_u: 0.0865

<div class="k-default-codeblock">
```

```
</div>
   1798/Unknown  205s 92ms/step - categorical_accuracy: 0.7238 - loss: 1.0964 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1799/Unknown  205s 92ms/step - categorical_accuracy: 0.7239 - loss: 1.0963 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1800/Unknown  205s 92ms/step - categorical_accuracy: 0.7239 - loss: 1.0962 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1801/Unknown  205s 92ms/step - categorical_accuracy: 0.7239 - loss: 1.0961 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1802/Unknown  206s 92ms/step - categorical_accuracy: 0.7239 - loss: 1.0960 - mean_io_u: 0.0866

<div class="k-default-codeblock">
```

```
</div>
   1803/Unknown  206s 92ms/step - categorical_accuracy: 0.7239 - loss: 1.0959 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1804/Unknown  206s 92ms/step - categorical_accuracy: 0.7239 - loss: 1.0958 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1805/Unknown  206s 92ms/step - categorical_accuracy: 0.7240 - loss: 1.0958 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1806/Unknown  206s 92ms/step - categorical_accuracy: 0.7240 - loss: 1.0957 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1807/Unknown  206s 92ms/step - categorical_accuracy: 0.7240 - loss: 1.0956 - mean_io_u: 0.0867

<div class="k-default-codeblock">
```

```
</div>
   1808/Unknown  206s 92ms/step - categorical_accuracy: 0.7240 - loss: 1.0955 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1809/Unknown  206s 92ms/step - categorical_accuracy: 0.7240 - loss: 1.0954 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1810/Unknown  206s 92ms/step - categorical_accuracy: 0.7240 - loss: 1.0953 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1811/Unknown  206s 92ms/step - categorical_accuracy: 0.7240 - loss: 1.0952 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1812/Unknown  206s 92ms/step - categorical_accuracy: 0.7241 - loss: 1.0952 - mean_io_u: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   1813/Unknown  206s 92ms/step - categorical_accuracy: 0.7241 - loss: 1.0951 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1814/Unknown  206s 92ms/step - categorical_accuracy: 0.7241 - loss: 1.0950 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1815/Unknown  207s 92ms/step - categorical_accuracy: 0.7241 - loss: 1.0949 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1816/Unknown  207s 92ms/step - categorical_accuracy: 0.7241 - loss: 1.0948 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1817/Unknown  207s 92ms/step - categorical_accuracy: 0.7241 - loss: 1.0947 - mean_io_u: 0.0869

<div class="k-default-codeblock">
```

```
</div>
   1818/Unknown  207s 92ms/step - categorical_accuracy: 0.7242 - loss: 1.0946 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1819/Unknown  207s 92ms/step - categorical_accuracy: 0.7242 - loss: 1.0946 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1820/Unknown  207s 92ms/step - categorical_accuracy: 0.7242 - loss: 1.0945 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1821/Unknown  207s 92ms/step - categorical_accuracy: 0.7242 - loss: 1.0944 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1822/Unknown  207s 92ms/step - categorical_accuracy: 0.7242 - loss: 1.0943 - mean_io_u: 0.0870

<div class="k-default-codeblock">
```

```
</div>
   1823/Unknown  207s 92ms/step - categorical_accuracy: 0.7242 - loss: 1.0942 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1824/Unknown  207s 92ms/step - categorical_accuracy: 0.7242 - loss: 1.0941 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1825/Unknown  207s 92ms/step - categorical_accuracy: 0.7243 - loss: 1.0941 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1826/Unknown  207s 92ms/step - categorical_accuracy: 0.7243 - loss: 1.0940 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1827/Unknown  207s 92ms/step - categorical_accuracy: 0.7243 - loss: 1.0939 - mean_io_u: 0.0871

<div class="k-default-codeblock">
```

```
</div>
   1828/Unknown  208s 92ms/step - categorical_accuracy: 0.7243 - loss: 1.0938 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1829/Unknown  208s 92ms/step - categorical_accuracy: 0.7243 - loss: 1.0937 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1830/Unknown  208s 92ms/step - categorical_accuracy: 0.7243 - loss: 1.0936 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1831/Unknown  208s 92ms/step - categorical_accuracy: 0.7244 - loss: 1.0936 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1832/Unknown  208s 92ms/step - categorical_accuracy: 0.7244 - loss: 1.0935 - mean_io_u: 0.0872

<div class="k-default-codeblock">
```

```
</div>
   1833/Unknown  208s 92ms/step - categorical_accuracy: 0.7244 - loss: 1.0934 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1834/Unknown  208s 92ms/step - categorical_accuracy: 0.7244 - loss: 1.0933 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1835/Unknown  208s 92ms/step - categorical_accuracy: 0.7244 - loss: 1.0932 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1836/Unknown  208s 92ms/step - categorical_accuracy: 0.7244 - loss: 1.0931 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1837/Unknown  208s 92ms/step - categorical_accuracy: 0.7244 - loss: 1.0931 - mean_io_u: 0.0873

<div class="k-default-codeblock">
```

```
</div>
   1838/Unknown  208s 92ms/step - categorical_accuracy: 0.7245 - loss: 1.0930 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1839/Unknown  208s 92ms/step - categorical_accuracy: 0.7245 - loss: 1.0929 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1840/Unknown  209s 92ms/step - categorical_accuracy: 0.7245 - loss: 1.0928 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1841/Unknown  209s 92ms/step - categorical_accuracy: 0.7245 - loss: 1.0927 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1842/Unknown  209s 92ms/step - categorical_accuracy: 0.7245 - loss: 1.0926 - mean_io_u: 0.0874

<div class="k-default-codeblock">
```

```
</div>
   1843/Unknown  209s 92ms/step - categorical_accuracy: 0.7245 - loss: 1.0926 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1844/Unknown  209s 92ms/step - categorical_accuracy: 0.7246 - loss: 1.0925 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1845/Unknown  209s 92ms/step - categorical_accuracy: 0.7246 - loss: 1.0924 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1846/Unknown  209s 92ms/step - categorical_accuracy: 0.7246 - loss: 1.0923 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1847/Unknown  209s 92ms/step - categorical_accuracy: 0.7246 - loss: 1.0922 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1848/Unknown  209s 92ms/step - categorical_accuracy: 0.7246 - loss: 1.0921 - mean_io_u: 0.0875

<div class="k-default-codeblock">
```

```
</div>
   1849/Unknown  209s 92ms/step - categorical_accuracy: 0.7246 - loss: 1.0921 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1850/Unknown  209s 92ms/step - categorical_accuracy: 0.7246 - loss: 1.0920 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1851/Unknown  209s 92ms/step - categorical_accuracy: 0.7247 - loss: 1.0919 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1852/Unknown  210s 92ms/step - categorical_accuracy: 0.7247 - loss: 1.0918 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1853/Unknown  210s 92ms/step - categorical_accuracy: 0.7247 - loss: 1.0917 - mean_io_u: 0.0876

<div class="k-default-codeblock">
```

```
</div>
   1854/Unknown  210s 92ms/step - categorical_accuracy: 0.7247 - loss: 1.0916 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1855/Unknown  210s 92ms/step - categorical_accuracy: 0.7247 - loss: 1.0916 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1856/Unknown  210s 92ms/step - categorical_accuracy: 0.7247 - loss: 1.0915 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1857/Unknown  210s 92ms/step - categorical_accuracy: 0.7248 - loss: 1.0914 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1858/Unknown  210s 92ms/step - categorical_accuracy: 0.7248 - loss: 1.0913 - mean_io_u: 0.0877

<div class="k-default-codeblock">
```

```
</div>
   1859/Unknown  210s 92ms/step - categorical_accuracy: 0.7248 - loss: 1.0912 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1860/Unknown  210s 92ms/step - categorical_accuracy: 0.7248 - loss: 1.0911 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1861/Unknown  210s 92ms/step - categorical_accuracy: 0.7248 - loss: 1.0911 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1862/Unknown  210s 92ms/step - categorical_accuracy: 0.7248 - loss: 1.0910 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1863/Unknown  210s 92ms/step - categorical_accuracy: 0.7248 - loss: 1.0909 - mean_io_u: 0.0878

<div class="k-default-codeblock">
```

```
</div>
   1864/Unknown  210s 92ms/step - categorical_accuracy: 0.7249 - loss: 1.0908 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1865/Unknown  211s 92ms/step - categorical_accuracy: 0.7249 - loss: 1.0907 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1866/Unknown  211s 92ms/step - categorical_accuracy: 0.7249 - loss: 1.0906 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1867/Unknown  211s 92ms/step - categorical_accuracy: 0.7249 - loss: 1.0906 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1868/Unknown  211s 92ms/step - categorical_accuracy: 0.7249 - loss: 1.0905 - mean_io_u: 0.0879

<div class="k-default-codeblock">
```

```
</div>
   1869/Unknown  211s 92ms/step - categorical_accuracy: 0.7249 - loss: 1.0904 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1870/Unknown  211s 92ms/step - categorical_accuracy: 0.7249 - loss: 1.0903 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1871/Unknown  211s 92ms/step - categorical_accuracy: 0.7250 - loss: 1.0902 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1872/Unknown  211s 92ms/step - categorical_accuracy: 0.7250 - loss: 1.0902 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1873/Unknown  211s 92ms/step - categorical_accuracy: 0.7250 - loss: 1.0901 - mean_io_u: 0.0880

<div class="k-default-codeblock">
```

```
</div>
   1874/Unknown  211s 92ms/step - categorical_accuracy: 0.7250 - loss: 1.0900 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1875/Unknown  211s 92ms/step - categorical_accuracy: 0.7250 - loss: 1.0899 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1876/Unknown  211s 92ms/step - categorical_accuracy: 0.7250 - loss: 1.0898 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1877/Unknown  211s 92ms/step - categorical_accuracy: 0.7251 - loss: 1.0897 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1878/Unknown  212s 92ms/step - categorical_accuracy: 0.7251 - loss: 1.0897 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1879/Unknown  212s 92ms/step - categorical_accuracy: 0.7251 - loss: 1.0896 - mean_io_u: 0.0881

<div class="k-default-codeblock">
```

```
</div>
   1880/Unknown  212s 92ms/step - categorical_accuracy: 0.7251 - loss: 1.0895 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1881/Unknown  212s 92ms/step - categorical_accuracy: 0.7251 - loss: 1.0894 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1882/Unknown  212s 92ms/step - categorical_accuracy: 0.7251 - loss: 1.0893 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1883/Unknown  212s 92ms/step - categorical_accuracy: 0.7251 - loss: 1.0893 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1884/Unknown  212s 92ms/step - categorical_accuracy: 0.7252 - loss: 1.0892 - mean_io_u: 0.0882

<div class="k-default-codeblock">
```

```
</div>
   1885/Unknown  212s 92ms/step - categorical_accuracy: 0.7252 - loss: 1.0891 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1886/Unknown  212s 92ms/step - categorical_accuracy: 0.7252 - loss: 1.0890 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1887/Unknown  212s 92ms/step - categorical_accuracy: 0.7252 - loss: 1.0889 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1888/Unknown  212s 92ms/step - categorical_accuracy: 0.7252 - loss: 1.0888 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1889/Unknown  212s 92ms/step - categorical_accuracy: 0.7252 - loss: 1.0888 - mean_io_u: 0.0883

<div class="k-default-codeblock">
```

```
</div>
   1890/Unknown  212s 91ms/step - categorical_accuracy: 0.7252 - loss: 1.0887 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1891/Unknown  213s 91ms/step - categorical_accuracy: 0.7253 - loss: 1.0886 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1892/Unknown  213s 91ms/step - categorical_accuracy: 0.7253 - loss: 1.0885 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1893/Unknown  213s 91ms/step - categorical_accuracy: 0.7253 - loss: 1.0884 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1894/Unknown  213s 91ms/step - categorical_accuracy: 0.7253 - loss: 1.0884 - mean_io_u: 0.0884

<div class="k-default-codeblock">
```

```
</div>
   1895/Unknown  213s 91ms/step - categorical_accuracy: 0.7253 - loss: 1.0883 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1896/Unknown  213s 91ms/step - categorical_accuracy: 0.7253 - loss: 1.0882 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1897/Unknown  213s 91ms/step - categorical_accuracy: 0.7254 - loss: 1.0881 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1898/Unknown  213s 91ms/step - categorical_accuracy: 0.7254 - loss: 1.0880 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1899/Unknown  213s 91ms/step - categorical_accuracy: 0.7254 - loss: 1.0880 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1900/Unknown  213s 91ms/step - categorical_accuracy: 0.7254 - loss: 1.0879 - mean_io_u: 0.0885

<div class="k-default-codeblock">
```

```
</div>
   1901/Unknown  213s 91ms/step - categorical_accuracy: 0.7254 - loss: 1.0878 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1902/Unknown  213s 91ms/step - categorical_accuracy: 0.7254 - loss: 1.0877 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1903/Unknown  213s 91ms/step - categorical_accuracy: 0.7254 - loss: 1.0876 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1904/Unknown  213s 91ms/step - categorical_accuracy: 0.7255 - loss: 1.0876 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1905/Unknown  214s 91ms/step - categorical_accuracy: 0.7255 - loss: 1.0875 - mean_io_u: 0.0886

<div class="k-default-codeblock">
```

```
</div>
   1906/Unknown  214s 91ms/step - categorical_accuracy: 0.7255 - loss: 1.0874 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1907/Unknown  214s 91ms/step - categorical_accuracy: 0.7255 - loss: 1.0873 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1908/Unknown  214s 91ms/step - categorical_accuracy: 0.7255 - loss: 1.0872 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1909/Unknown  214s 91ms/step - categorical_accuracy: 0.7255 - loss: 1.0872 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1910/Unknown  214s 91ms/step - categorical_accuracy: 0.7255 - loss: 1.0871 - mean_io_u: 0.0887

<div class="k-default-codeblock">
```

```
</div>
   1911/Unknown  214s 91ms/step - categorical_accuracy: 0.7256 - loss: 1.0870 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1912/Unknown  214s 91ms/step - categorical_accuracy: 0.7256 - loss: 1.0869 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1913/Unknown  214s 91ms/step - categorical_accuracy: 0.7256 - loss: 1.0868 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1914/Unknown  214s 91ms/step - categorical_accuracy: 0.7256 - loss: 1.0868 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1915/Unknown  214s 91ms/step - categorical_accuracy: 0.7256 - loss: 1.0867 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1916/Unknown  214s 91ms/step - categorical_accuracy: 0.7256 - loss: 1.0866 - mean_io_u: 0.0888

<div class="k-default-codeblock">
```

```
</div>
   1917/Unknown  214s 91ms/step - categorical_accuracy: 0.7256 - loss: 1.0865 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1918/Unknown  214s 91ms/step - categorical_accuracy: 0.7257 - loss: 1.0864 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1919/Unknown  215s 91ms/step - categorical_accuracy: 0.7257 - loss: 1.0864 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1920/Unknown  215s 91ms/step - categorical_accuracy: 0.7257 - loss: 1.0863 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1921/Unknown  215s 91ms/step - categorical_accuracy: 0.7257 - loss: 1.0862 - mean_io_u: 0.0889

<div class="k-default-codeblock">
```

```
</div>
   1922/Unknown  215s 91ms/step - categorical_accuracy: 0.7257 - loss: 1.0861 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1923/Unknown  215s 91ms/step - categorical_accuracy: 0.7257 - loss: 1.0860 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1924/Unknown  215s 91ms/step - categorical_accuracy: 0.7257 - loss: 1.0860 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1925/Unknown  215s 91ms/step - categorical_accuracy: 0.7258 - loss: 1.0859 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1926/Unknown  215s 91ms/step - categorical_accuracy: 0.7258 - loss: 1.0858 - mean_io_u: 0.0890

<div class="k-default-codeblock">
```

```
</div>
   1927/Unknown  215s 91ms/step - categorical_accuracy: 0.7258 - loss: 1.0857 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1928/Unknown  215s 91ms/step - categorical_accuracy: 0.7258 - loss: 1.0857 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1929/Unknown  215s 91ms/step - categorical_accuracy: 0.7258 - loss: 1.0856 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1930/Unknown  215s 91ms/step - categorical_accuracy: 0.7258 - loss: 1.0855 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1931/Unknown  215s 91ms/step - categorical_accuracy: 0.7259 - loss: 1.0854 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1932/Unknown  215s 91ms/step - categorical_accuracy: 0.7259 - loss: 1.0853 - mean_io_u: 0.0891

<div class="k-default-codeblock">
```

```
</div>
   1933/Unknown  216s 91ms/step - categorical_accuracy: 0.7259 - loss: 1.0853 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1934/Unknown  216s 91ms/step - categorical_accuracy: 0.7259 - loss: 1.0852 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1935/Unknown  216s 91ms/step - categorical_accuracy: 0.7259 - loss: 1.0851 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1936/Unknown  216s 91ms/step - categorical_accuracy: 0.7259 - loss: 1.0850 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1937/Unknown  216s 91ms/step - categorical_accuracy: 0.7259 - loss: 1.0849 - mean_io_u: 0.0892

<div class="k-default-codeblock">
```

```
</div>
   1938/Unknown  216s 91ms/step - categorical_accuracy: 0.7260 - loss: 1.0849 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1939/Unknown  216s 91ms/step - categorical_accuracy: 0.7260 - loss: 1.0848 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1940/Unknown  216s 91ms/step - categorical_accuracy: 0.7260 - loss: 1.0847 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1941/Unknown  216s 91ms/step - categorical_accuracy: 0.7260 - loss: 1.0846 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1942/Unknown  216s 91ms/step - categorical_accuracy: 0.7260 - loss: 1.0845 - mean_io_u: 0.0893

<div class="k-default-codeblock">
```

```
</div>
   1943/Unknown  216s 91ms/step - categorical_accuracy: 0.7260 - loss: 1.0845 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1944/Unknown  216s 91ms/step - categorical_accuracy: 0.7260 - loss: 1.0844 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1945/Unknown  216s 91ms/step - categorical_accuracy: 0.7261 - loss: 1.0843 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1946/Unknown  217s 91ms/step - categorical_accuracy: 0.7261 - loss: 1.0842 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1947/Unknown  217s 91ms/step - categorical_accuracy: 0.7261 - loss: 1.0842 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1948/Unknown  217s 91ms/step - categorical_accuracy: 0.7261 - loss: 1.0841 - mean_io_u: 0.0894

<div class="k-default-codeblock">
```

```
</div>
   1949/Unknown  217s 91ms/step - categorical_accuracy: 0.7261 - loss: 1.0840 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1950/Unknown  217s 91ms/step - categorical_accuracy: 0.7261 - loss: 1.0839 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1951/Unknown  217s 91ms/step - categorical_accuracy: 0.7261 - loss: 1.0838 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1952/Unknown  217s 91ms/step - categorical_accuracy: 0.7262 - loss: 1.0838 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1953/Unknown  217s 91ms/step - categorical_accuracy: 0.7262 - loss: 1.0837 - mean_io_u: 0.0895

<div class="k-default-codeblock">
```

```
</div>
   1954/Unknown  217s 91ms/step - categorical_accuracy: 0.7262 - loss: 1.0836 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1955/Unknown  217s 91ms/step - categorical_accuracy: 0.7262 - loss: 1.0835 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1956/Unknown  217s 91ms/step - categorical_accuracy: 0.7262 - loss: 1.0834 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1957/Unknown  217s 91ms/step - categorical_accuracy: 0.7262 - loss: 1.0834 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1958/Unknown  217s 91ms/step - categorical_accuracy: 0.7262 - loss: 1.0833 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1959/Unknown  218s 91ms/step - categorical_accuracy: 0.7263 - loss: 1.0832 - mean_io_u: 0.0896

<div class="k-default-codeblock">
```

```
</div>
   1960/Unknown  218s 91ms/step - categorical_accuracy: 0.7263 - loss: 1.0831 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1961/Unknown  218s 91ms/step - categorical_accuracy: 0.7263 - loss: 1.0831 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1962/Unknown  218s 91ms/step - categorical_accuracy: 0.7263 - loss: 1.0830 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1963/Unknown  218s 91ms/step - categorical_accuracy: 0.7263 - loss: 1.0829 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1964/Unknown  218s 91ms/step - categorical_accuracy: 0.7263 - loss: 1.0828 - mean_io_u: 0.0897

<div class="k-default-codeblock">
```

```
</div>
   1965/Unknown  218s 91ms/step - categorical_accuracy: 0.7263 - loss: 1.0827 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1966/Unknown  218s 91ms/step - categorical_accuracy: 0.7264 - loss: 1.0827 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1967/Unknown  218s 91ms/step - categorical_accuracy: 0.7264 - loss: 1.0826 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1968/Unknown  218s 91ms/step - categorical_accuracy: 0.7264 - loss: 1.0825 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1969/Unknown  218s 91ms/step - categorical_accuracy: 0.7264 - loss: 1.0824 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1970/Unknown  219s 91ms/step - categorical_accuracy: 0.7264 - loss: 1.0824 - mean_io_u: 0.0898

<div class="k-default-codeblock">
```

```
</div>
   1971/Unknown  219s 91ms/step - categorical_accuracy: 0.7264 - loss: 1.0823 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1972/Unknown  219s 91ms/step - categorical_accuracy: 0.7264 - loss: 1.0822 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1973/Unknown  219s 91ms/step - categorical_accuracy: 0.7265 - loss: 1.0821 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1974/Unknown  219s 91ms/step - categorical_accuracy: 0.7265 - loss: 1.0820 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1975/Unknown  219s 91ms/step - categorical_accuracy: 0.7265 - loss: 1.0820 - mean_io_u: 0.0899

<div class="k-default-codeblock">
```

```
</div>
   1976/Unknown  219s 91ms/step - categorical_accuracy: 0.7265 - loss: 1.0819 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1977/Unknown  219s 91ms/step - categorical_accuracy: 0.7265 - loss: 1.0818 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1978/Unknown  219s 91ms/step - categorical_accuracy: 0.7265 - loss: 1.0817 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1979/Unknown  219s 91ms/step - categorical_accuracy: 0.7265 - loss: 1.0817 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1980/Unknown  219s 91ms/step - categorical_accuracy: 0.7266 - loss: 1.0816 - mean_io_u: 0.0900

<div class="k-default-codeblock">
```

```
</div>
   1981/Unknown  219s 91ms/step - categorical_accuracy: 0.7266 - loss: 1.0815 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1982/Unknown  220s 91ms/step - categorical_accuracy: 0.7266 - loss: 1.0814 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1983/Unknown  220s 91ms/step - categorical_accuracy: 0.7266 - loss: 1.0813 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1984/Unknown  220s 91ms/step - categorical_accuracy: 0.7266 - loss: 1.0813 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1985/Unknown  220s 91ms/step - categorical_accuracy: 0.7266 - loss: 1.0812 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1986/Unknown  220s 91ms/step - categorical_accuracy: 0.7266 - loss: 1.0811 - mean_io_u: 0.0901

<div class="k-default-codeblock">
```

```
</div>
   1987/Unknown  220s 91ms/step - categorical_accuracy: 0.7267 - loss: 1.0810 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1988/Unknown  220s 91ms/step - categorical_accuracy: 0.7267 - loss: 1.0810 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1989/Unknown  220s 91ms/step - categorical_accuracy: 0.7267 - loss: 1.0809 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1990/Unknown  220s 91ms/step - categorical_accuracy: 0.7267 - loss: 1.0808 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1991/Unknown  220s 91ms/step - categorical_accuracy: 0.7267 - loss: 1.0807 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1992/Unknown  220s 91ms/step - categorical_accuracy: 0.7267 - loss: 1.0807 - mean_io_u: 0.0902

<div class="k-default-codeblock">
```

```
</div>
   1993/Unknown  220s 91ms/step - categorical_accuracy: 0.7267 - loss: 1.0806 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1994/Unknown  221s 91ms/step - categorical_accuracy: 0.7268 - loss: 1.0805 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1995/Unknown  221s 91ms/step - categorical_accuracy: 0.7268 - loss: 1.0804 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1996/Unknown  221s 91ms/step - categorical_accuracy: 0.7268 - loss: 1.0803 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1997/Unknown  221s 91ms/step - categorical_accuracy: 0.7268 - loss: 1.0803 - mean_io_u: 0.0903

<div class="k-default-codeblock">
```

```
</div>
   1998/Unknown  221s 91ms/step - categorical_accuracy: 0.7268 - loss: 1.0802 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   1999/Unknown  221s 91ms/step - categorical_accuracy: 0.7268 - loss: 1.0801 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2000/Unknown  221s 91ms/step - categorical_accuracy: 0.7268 - loss: 1.0800 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2001/Unknown  221s 91ms/step - categorical_accuracy: 0.7269 - loss: 1.0800 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2002/Unknown  221s 91ms/step - categorical_accuracy: 0.7269 - loss: 1.0799 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2003/Unknown  221s 91ms/step - categorical_accuracy: 0.7269 - loss: 1.0798 - mean_io_u: 0.0904

<div class="k-default-codeblock">
```

```
</div>
   2004/Unknown  221s 91ms/step - categorical_accuracy: 0.7269 - loss: 1.0797 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   2005/Unknown  221s 91ms/step - categorical_accuracy: 0.7269 - loss: 1.0797 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   2006/Unknown  222s 91ms/step - categorical_accuracy: 0.7269 - loss: 1.0796 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   2007/Unknown  222s 91ms/step - categorical_accuracy: 0.7269 - loss: 1.0795 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   2008/Unknown  222s 91ms/step - categorical_accuracy: 0.7270 - loss: 1.0794 - mean_io_u: 0.0905

<div class="k-default-codeblock">
```

```
</div>
   2009/Unknown  222s 91ms/step - categorical_accuracy: 0.7270 - loss: 1.0794 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2010/Unknown  222s 91ms/step - categorical_accuracy: 0.7270 - loss: 1.0793 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2011/Unknown  222s 91ms/step - categorical_accuracy: 0.7270 - loss: 1.0792 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2012/Unknown  222s 91ms/step - categorical_accuracy: 0.7270 - loss: 1.0791 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2013/Unknown  222s 91ms/step - categorical_accuracy: 0.7270 - loss: 1.0791 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2014/Unknown  222s 91ms/step - categorical_accuracy: 0.7270 - loss: 1.0790 - mean_io_u: 0.0906

<div class="k-default-codeblock">
```

```
</div>
   2015/Unknown  222s 91ms/step - categorical_accuracy: 0.7271 - loss: 1.0789 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   2016/Unknown  222s 91ms/step - categorical_accuracy: 0.7271 - loss: 1.0788 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   2017/Unknown  222s 91ms/step - categorical_accuracy: 0.7271 - loss: 1.0787 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   2018/Unknown  222s 91ms/step - categorical_accuracy: 0.7271 - loss: 1.0787 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   2019/Unknown  222s 91ms/step - categorical_accuracy: 0.7271 - loss: 1.0786 - mean_io_u: 0.0907

<div class="k-default-codeblock">
```

```
</div>
   2020/Unknown  223s 91ms/step - categorical_accuracy: 0.7271 - loss: 1.0785 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2021/Unknown  223s 91ms/step - categorical_accuracy: 0.7271 - loss: 1.0784 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2022/Unknown  223s 91ms/step - categorical_accuracy: 0.7272 - loss: 1.0784 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2023/Unknown  223s 91ms/step - categorical_accuracy: 0.7272 - loss: 1.0783 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2024/Unknown  223s 91ms/step - categorical_accuracy: 0.7272 - loss: 1.0782 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2025/Unknown  223s 91ms/step - categorical_accuracy: 0.7272 - loss: 1.0781 - mean_io_u: 0.0908

<div class="k-default-codeblock">
```

```
</div>
   2026/Unknown  223s 91ms/step - categorical_accuracy: 0.7272 - loss: 1.0781 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2027/Unknown  223s 91ms/step - categorical_accuracy: 0.7272 - loss: 1.0780 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2028/Unknown  223s 91ms/step - categorical_accuracy: 0.7272 - loss: 1.0779 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2029/Unknown  223s 91ms/step - categorical_accuracy: 0.7273 - loss: 1.0778 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2030/Unknown  223s 91ms/step - categorical_accuracy: 0.7273 - loss: 1.0778 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2031/Unknown  223s 91ms/step - categorical_accuracy: 0.7273 - loss: 1.0777 - mean_io_u: 0.0909

<div class="k-default-codeblock">
```

```
</div>
   2032/Unknown  224s 91ms/step - categorical_accuracy: 0.7273 - loss: 1.0776 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   2033/Unknown  224s 91ms/step - categorical_accuracy: 0.7273 - loss: 1.0775 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   2034/Unknown  224s 91ms/step - categorical_accuracy: 0.7273 - loss: 1.0775 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   2035/Unknown  224s 91ms/step - categorical_accuracy: 0.7273 - loss: 1.0774 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   2036/Unknown  224s 91ms/step - categorical_accuracy: 0.7274 - loss: 1.0773 - mean_io_u: 0.0910

<div class="k-default-codeblock">
```

```
</div>
   2037/Unknown  224s 91ms/step - categorical_accuracy: 0.7274 - loss: 1.0772 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2038/Unknown  224s 90ms/step - categorical_accuracy: 0.7274 - loss: 1.0772 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2039/Unknown  224s 90ms/step - categorical_accuracy: 0.7274 - loss: 1.0771 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2040/Unknown  224s 90ms/step - categorical_accuracy: 0.7274 - loss: 1.0770 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2041/Unknown  224s 90ms/step - categorical_accuracy: 0.7274 - loss: 1.0769 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2042/Unknown  224s 90ms/step - categorical_accuracy: 0.7274 - loss: 1.0769 - mean_io_u: 0.0911

<div class="k-default-codeblock">
```

```
</div>
   2043/Unknown  224s 90ms/step - categorical_accuracy: 0.7275 - loss: 1.0768 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2044/Unknown  224s 90ms/step - categorical_accuracy: 0.7275 - loss: 1.0767 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2045/Unknown  225s 90ms/step - categorical_accuracy: 0.7275 - loss: 1.0766 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2046/Unknown  225s 90ms/step - categorical_accuracy: 0.7275 - loss: 1.0766 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2047/Unknown  225s 90ms/step - categorical_accuracy: 0.7275 - loss: 1.0765 - mean_io_u: 0.0912

<div class="k-default-codeblock">
```

```
</div>
   2048/Unknown  225s 90ms/step - categorical_accuracy: 0.7275 - loss: 1.0764 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2049/Unknown  225s 90ms/step - categorical_accuracy: 0.7275 - loss: 1.0763 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2050/Unknown  225s 90ms/step - categorical_accuracy: 0.7276 - loss: 1.0762 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2051/Unknown  225s 90ms/step - categorical_accuracy: 0.7276 - loss: 1.0762 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2052/Unknown  225s 90ms/step - categorical_accuracy: 0.7276 - loss: 1.0761 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2053/Unknown  225s 90ms/step - categorical_accuracy: 0.7276 - loss: 1.0760 - mean_io_u: 0.0913

<div class="k-default-codeblock">
```

```
</div>
   2054/Unknown  225s 90ms/step - categorical_accuracy: 0.7276 - loss: 1.0759 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2055/Unknown  225s 90ms/step - categorical_accuracy: 0.7276 - loss: 1.0759 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2056/Unknown  225s 90ms/step - categorical_accuracy: 0.7276 - loss: 1.0758 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2057/Unknown  225s 90ms/step - categorical_accuracy: 0.7277 - loss: 1.0757 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2058/Unknown  225s 90ms/step - categorical_accuracy: 0.7277 - loss: 1.0756 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2059/Unknown  225s 90ms/step - categorical_accuracy: 0.7277 - loss: 1.0756 - mean_io_u: 0.0914

<div class="k-default-codeblock">
```

```
</div>
   2060/Unknown  225s 90ms/step - categorical_accuracy: 0.7277 - loss: 1.0755 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2061/Unknown  226s 90ms/step - categorical_accuracy: 0.7277 - loss: 1.0754 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2062/Unknown  226s 90ms/step - categorical_accuracy: 0.7277 - loss: 1.0753 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2063/Unknown  226s 90ms/step - categorical_accuracy: 0.7277 - loss: 1.0753 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2064/Unknown  226s 90ms/step - categorical_accuracy: 0.7278 - loss: 1.0752 - mean_io_u: 0.0915

<div class="k-default-codeblock">
```

```
</div>
   2065/Unknown  226s 90ms/step - categorical_accuracy: 0.7278 - loss: 1.0751 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2066/Unknown  226s 90ms/step - categorical_accuracy: 0.7278 - loss: 1.0750 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2067/Unknown  226s 90ms/step - categorical_accuracy: 0.7278 - loss: 1.0750 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2068/Unknown  226s 90ms/step - categorical_accuracy: 0.7278 - loss: 1.0749 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2069/Unknown  226s 90ms/step - categorical_accuracy: 0.7278 - loss: 1.0748 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2070/Unknown  226s 90ms/step - categorical_accuracy: 0.7278 - loss: 1.0747 - mean_io_u: 0.0916

<div class="k-default-codeblock">
```

```
</div>
   2071/Unknown  226s 90ms/step - categorical_accuracy: 0.7278 - loss: 1.0747 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2072/Unknown  226s 90ms/step - categorical_accuracy: 0.7279 - loss: 1.0746 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2073/Unknown  226s 90ms/step - categorical_accuracy: 0.7279 - loss: 1.0745 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2074/Unknown  227s 90ms/step - categorical_accuracy: 0.7279 - loss: 1.0745 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2075/Unknown  227s 90ms/step - categorical_accuracy: 0.7279 - loss: 1.0744 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2076/Unknown  227s 90ms/step - categorical_accuracy: 0.7279 - loss: 1.0743 - mean_io_u: 0.0917

<div class="k-default-codeblock">
```

```
</div>
   2077/Unknown  227s 90ms/step - categorical_accuracy: 0.7279 - loss: 1.0742 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2078/Unknown  227s 90ms/step - categorical_accuracy: 0.7279 - loss: 1.0742 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2079/Unknown  227s 90ms/step - categorical_accuracy: 0.7280 - loss: 1.0741 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2080/Unknown  227s 90ms/step - categorical_accuracy: 0.7280 - loss: 1.0740 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2081/Unknown  227s 90ms/step - categorical_accuracy: 0.7280 - loss: 1.0739 - mean_io_u: 0.0918

<div class="k-default-codeblock">
```

```
</div>
   2082/Unknown  227s 90ms/step - categorical_accuracy: 0.7280 - loss: 1.0739 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2083/Unknown  227s 90ms/step - categorical_accuracy: 0.7280 - loss: 1.0738 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2084/Unknown  227s 90ms/step - categorical_accuracy: 0.7280 - loss: 1.0737 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2085/Unknown  227s 90ms/step - categorical_accuracy: 0.7280 - loss: 1.0736 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2086/Unknown  227s 90ms/step - categorical_accuracy: 0.7281 - loss: 1.0736 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2087/Unknown  228s 90ms/step - categorical_accuracy: 0.7281 - loss: 1.0735 - mean_io_u: 0.0919

<div class="k-default-codeblock">
```

```
</div>
   2088/Unknown  228s 90ms/step - categorical_accuracy: 0.7281 - loss: 1.0734 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2089/Unknown  228s 90ms/step - categorical_accuracy: 0.7281 - loss: 1.0733 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2090/Unknown  228s 90ms/step - categorical_accuracy: 0.7281 - loss: 1.0733 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2091/Unknown  228s 90ms/step - categorical_accuracy: 0.7281 - loss: 1.0732 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2092/Unknown  228s 90ms/step - categorical_accuracy: 0.7281 - loss: 1.0731 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2093/Unknown  228s 90ms/step - categorical_accuracy: 0.7282 - loss: 1.0730 - mean_io_u: 0.0920

<div class="k-default-codeblock">
```

```
</div>
   2094/Unknown  228s 90ms/step - categorical_accuracy: 0.7282 - loss: 1.0730 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2095/Unknown  228s 90ms/step - categorical_accuracy: 0.7282 - loss: 1.0729 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2096/Unknown  228s 90ms/step - categorical_accuracy: 0.7282 - loss: 1.0728 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2097/Unknown  228s 90ms/step - categorical_accuracy: 0.7282 - loss: 1.0728 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2098/Unknown  228s 90ms/step - categorical_accuracy: 0.7282 - loss: 1.0727 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2099/Unknown  229s 90ms/step - categorical_accuracy: 0.7282 - loss: 1.0726 - mean_io_u: 0.0921

<div class="k-default-codeblock">
```

```
</div>
   2100/Unknown  229s 90ms/step - categorical_accuracy: 0.7282 - loss: 1.0725 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2101/Unknown  229s 90ms/step - categorical_accuracy: 0.7283 - loss: 1.0725 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2102/Unknown  229s 90ms/step - categorical_accuracy: 0.7283 - loss: 1.0724 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2103/Unknown  229s 90ms/step - categorical_accuracy: 0.7283 - loss: 1.0723 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2104/Unknown  229s 90ms/step - categorical_accuracy: 0.7283 - loss: 1.0722 - mean_io_u: 0.0922

<div class="k-default-codeblock">
```

```
</div>
   2105/Unknown  229s 90ms/step - categorical_accuracy: 0.7283 - loss: 1.0722 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2106/Unknown  229s 90ms/step - categorical_accuracy: 0.7283 - loss: 1.0721 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2107/Unknown  229s 90ms/step - categorical_accuracy: 0.7283 - loss: 1.0720 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2108/Unknown  229s 90ms/step - categorical_accuracy: 0.7284 - loss: 1.0720 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2109/Unknown  229s 90ms/step - categorical_accuracy: 0.7284 - loss: 1.0719 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2110/Unknown  229s 90ms/step - categorical_accuracy: 0.7284 - loss: 1.0718 - mean_io_u: 0.0923

<div class="k-default-codeblock">
```

```
</div>
   2111/Unknown  229s 90ms/step - categorical_accuracy: 0.7284 - loss: 1.0717 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2112/Unknown  229s 90ms/step - categorical_accuracy: 0.7284 - loss: 1.0717 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2113/Unknown  230s 90ms/step - categorical_accuracy: 0.7284 - loss: 1.0716 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2114/Unknown  230s 90ms/step - categorical_accuracy: 0.7284 - loss: 1.0715 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2115/Unknown  230s 90ms/step - categorical_accuracy: 0.7285 - loss: 1.0714 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2116/Unknown  230s 90ms/step - categorical_accuracy: 0.7285 - loss: 1.0714 - mean_io_u: 0.0924

<div class="k-default-codeblock">
```

```
</div>
   2117/Unknown  230s 90ms/step - categorical_accuracy: 0.7285 - loss: 1.0713 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2118/Unknown  230s 90ms/step - categorical_accuracy: 0.7285 - loss: 1.0712 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2119/Unknown  230s 90ms/step - categorical_accuracy: 0.7285 - loss: 1.0712 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2120/Unknown  230s 90ms/step - categorical_accuracy: 0.7285 - loss: 1.0711 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2121/Unknown  230s 90ms/step - categorical_accuracy: 0.7285 - loss: 1.0710 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2122/Unknown  230s 90ms/step - categorical_accuracy: 0.7285 - loss: 1.0709 - mean_io_u: 0.0925

<div class="k-default-codeblock">
```

```
</div>
   2123/Unknown  230s 90ms/step - categorical_accuracy: 0.7286 - loss: 1.0708 - mean_io_u: 0.0926
   2124/Unknown  230s 90ms/step - categorical_accuracy: 0.7286 - loss: 1.0708 - mean_io_u: 0.0926

<div class="k-default-codeblock">
```
/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/models/functional.py:225: UserWarning: The structure of `inputs` doesn't match the expected structure: ['keras_tensor_222']. Received: the structure of inputs=*
  warnings.warn(

/home/sachinprasad/projects/env/lib/python3.11/site-packages/keras/src/backend/jax/core.py:76: UserWarning: Explicitly requested dtype int64 requested in asarray is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return jnp.asarray(x, dtype=dtype)


```
</div>
 2124/2124 ━━━━━━━━━━━━━━━━━━━━ 281s 114ms/step - categorical_accuracy: 0.7286 - loss: 1.0707 - mean_io_u: 0.0926 - val_categorical_accuracy: 0.8199 - val_loss: 0.5900 - val_mean_io_u: 0.3265





<div class="k-default-codeblock">
```
<keras.src.callbacks.history.History at 0x7fd7a897f8d0>

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

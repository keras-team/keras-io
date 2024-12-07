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

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 5s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 5s 5s/step



    
![png](/img/guides/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_9_3.png)
    


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
    get_data = keras.utils.get_file(
        fname=os.path.basename(SBD_URL),
        origin=SBD_URL,
        cache_dir=data_dir,
        extract=True,
    )
    data_dir = os.path.join(os.path.dirname(get_data), extracted_dir)
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


    
![png](/img/guides/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_18_0.png)
    


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


    
![png](/img/guides/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_22_0.png)
    


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
![png](/img/guides/semantic_segmentation_deeplab_v3/learning_rate_schedule.png)


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
  1/Unknown  40s 40s/step - categorical_accuracy: 0.1191 - loss: 3.0568 - mean_io_u: 0.0118

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

images, masks = next(iter(test_ds.take(1)))
images = ops.convert_to_tensor(images)
masks = ops.convert_to_tensor(masks)
preds = ops.expand_dims(ops.argmax(model.predict(images), axis=-1), axis=-1)
masks = ops.expand_dims(ops.argmax(masks, axis=-1), axis=-1)

plot_images_masks(images, masks, preds)
```

    
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step

<div class="k-default-codeblock">
```

```
</div>
 1/1 ━━━━━━━━━━━━━━━━━━━━ 3s 3s/step



    
![png](/img/guides/semantic_segmentation_deeplab_v3/semantic_segmentation_deeplab_v3_32_2.png)
    


Here are some additional tips for using the KerasHub DeepLabv3 model:

- The model can be trained on a variety of datasets, including the COCO dataset, the
PASCAL VOC dataset, and the Cityscapes dataset.
- The model can be fine-tuned on a custom dataset to improve its performance on a
specific task.
- The model can be used to perform real-time inference on images.
- Also, check out KerasHub's other segmentation models.

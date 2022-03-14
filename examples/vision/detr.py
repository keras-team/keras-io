"""
Title: DETR : End-to-End Object Detection with Transformers
Author: [Ayyuce Demirbas](https://twitter.com/demirbasayyuce)
Date created: 2022/03/13
Last modified: 2022/03/13
Description: TensorFlow implementation of [End-to-End Object Detection with Transformers paper](https://arxiv.org/pdf/2005.12872.pdf)
"""
"""


"""

"""
## Introduction 

Unlike traditional computer vision techniques, DETR approaches object detection as a
direct set prediction problem. It consists of a set-based global loss, which forces
unique predictions via bipartite matching, and a Transformer encoder-decoder
architecture. Given a fixed small set of learned object queries, DETR reasons about the
relations of the objects and the global image context to directly output the final set of
predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.
[1]

![](https://raw.githubusercontent.com/facebookresearch/detr/main/.github/DETR.png)
Figure 1 [2]
"""

"""
## Downloading the data
"""

"""shell
!mkdir data/
!mkdir data/hardhat/
"""

"""shell
!curl -L "https://public.roboflow.com/ds/uaTeWnewYu?key=HB9EuKDeJF" >
data/hardhat/hard-hat-workers-dataset.zip
"""

"""shell
!unzip data/hardhat/hard-hat-workers-dataset.zip -d data/hardhat/
"""

"""
# Setting up a custom dataset
"""

"""
### Create your config
"""

import tensorflow as tf
import os


class TrainingConfig:
    def __init__(self):

        # Dataset info
        self.data_dir, self.img_dir, self.ann_dir, self.ann_file = (
            None,
            None,
            None,
            None,
        )
        self.data = DataConfig(data_dir=None, img_dir=None, ann_file=None, ann_dir=None)
        self.background_class = 0
        self.image_size = 376, 672

        # What to train
        self.train_backbone = False
        self.train_transformers = False
        self.train_nlayers = False

        # How to train
        self.finetuning = False
        self.batch_size = 1
        self.gradient_norm_clipping = 0.1
        # Batch aggregate before to backprop
        self.target_batch = 1

        # Learning rate
        # Set as tf.Variable so that the variable can be update during the training while
        # keeping the same graph
        self.backbone_lr = tf.Variable(1e-5)
        self.transformers_lr = tf.Variable(1e-4)
        self.nlayers_lr = tf.Variable(1e-4)
        self.nlayers = []

        # Training progress
        self.global_step = 0
        self.log = False

        # Pipeline variables
        self.normalized_method = "torch_resnet"

    def add_nlayers(self, layers):
        """Set the new layers to train on the training config"""
        self.nlayers = [l.name for l in layers]

    def update_from_args(self, args):
        """Update the training config from args"""
        args = vars(args)
        for key in args:
            if isinstance(getattr(self, key), tf.Variable):
                getattr(self, key).assign(args[key])
            else:
                setattr(self, key, args[key])

        # Set the config on the data class

        self.data = DataConfig(
            data_dir=self.data_dir,
            img_dir=self.img_dir,
            ann_file=self.ann_file,
            ann_dir=self.ann_dir,
        )


class DataConfig:
    def __init__(self, data_dir=None, img_dir=None, ann_file=None, ann_dir=None):
        self.data_dir = data_dir
        self.img_dir = (
            os.path.join(data_dir, img_dir)
            if data_dir is not None and img_dir is not None
            else None
        )
        self.ann_file = (
            os.path.join(self.data_dir, ann_file) if ann_file is not None else None
        )
        self.ann_dir = (
            os.path.join(self.data_dir, ann_dir) if ann_dir is not None else None
        )


config = TrainingConfig()

from os.path import expanduser
import os


class CustomConfig(TrainingConfig):
    def __init__(self):
        super().__init__()

        # Dataset info
        self.datadir = os.path.join(expanduser("."), "data/hardhat/")
        # The model is trained using fixed size images.
        # The following is the desired target image size
        self.image_size = (480, 720)


config = CustomConfig()

"""
### Load the dataset
"""

import pandas as pd
import os

# Open annotation file
anns = pd.read_csv(os.path.join(config.datadir, "train/_annotations.csv"))

# Set the class name.
CLASS_NAMES = anns["class"].unique().tolist()
# Add the background class at the begining
CLASS_NAMES = ["background"] + CLASS_NAMES

print("CLASS_NAMES", CLASS_NAMES)
anns

"""
Display one image with the associated bounding box
"""

import imageio
import matplotlib.pyplot as plt
import os

# Select all unique filenames. We'll be usefull later to setup the datapipeline
filenames = anns["filename"].unique().tolist()

img_id = filenames.index("003225_jpg.rf.03205181938e7f642273dd4f106ea3e1.jpg")
# Select all the annotatio (bbox and class) on this image
image_anns = anns[anns["filename"] == filenames[img_id]]  # filenames[img_id]]

# Open the image
image = imageio.imread(os.path.join(config.datadir, "train", filenames[img_id]))
plt.imshow(image)

"""
Retrive the targert class, bbox and display the result:
"""

from typing import Union, Dict, Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import cv2


def bbox_xcycwh_to_x1y1x2y2(bbox_xcycwh: np.array):
    """
    Rescale a list of bbox to the image size
    @bbox_xcycwh: [[xc, yc, w, h], ...]
    @img_size (height, width)
    """
    bbox_x1y1x2y2 = np.zeros_like((bbox_xcycwh))
    bbox_x1y1x2y2[:, 0] = bbox_xcycwh[:, 0] - (bbox_xcycwh[:, 2] / 2)
    bbox_x1y1x2y2[:, 2] = bbox_xcycwh[:, 0] + (bbox_xcycwh[:, 2] / 2)
    bbox_x1y1x2y2[:, 1] = bbox_xcycwh[:, 1] - (bbox_xcycwh[:, 3] / 2)
    bbox_x1y1x2y2[:, 3] = bbox_xcycwh[:, 1] + (bbox_xcycwh[:, 3] / 2)
    bbox_x1y1x2y2 = bbox_x1y1x2y2.astype(np.int32)
    return bbox_x1y1x2y2


def intersect(box_a: tf.Tensor, box_b: tf.Tensor) -> tf.Tensor:
    """
    Compute the intersection area between two sets of boxes.
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        The intersection area [a, b] between each bbox. zero if no intersection
    """
    # resize both tensors to [A,B,2] with the tile function to compare
    # each bbox with the anchors:
    # [A,2] -> [A,1,2] -> [A,B,2]
    # [B,2] -> [1,B,2] -> [A,B,2]
    # Then we compute the area of intersect between box_a and box_b.
    # box_a: (tensor) bounding boxes, Shape: [n, A, 4].
    # box_b: (tensor) bounding boxes, Shape: [n, B, 4].
    # Return: (tensor) intersection area, Shape: [n,A,B].

    A = tf.shape(box_a)[0]  # Number of possible bbox
    B = tf.shape(box_b)[0]  # Number of anchors

    # print(A, B, box_a.shape, box_b.shape)
    # Above Right Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a_xymax = tf.tile(tf.expand_dims(box_a[:, 2:], axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b_xymax = tf.tile(tf.expand_dims(box_b[:, 2:], axis=0), [A, 1, 1])
    # Select the lower right corner of the intersect area
    above_right_corner = tf.math.minimum(tiled_box_a_xymax, tiled_box_b_xymax)

    # Upper Left Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a_xymin = tf.tile(tf.expand_dims(box_a[:, :2], axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b_xymin = tf.tile(tf.expand_dims(box_b[:, :2], axis=0), [A, 1, 1])
    # Select the lower right corner of the intersect area
    upper_left_corner = tf.math.maximum(tiled_box_a_xymin, tiled_box_b_xymin)

    # If there is some intersection, both must be > 0
    inter = tf.nn.relu(above_right_corner - upper_left_corner)
    inter = inter[:, :, 0] * inter[:, :, 1]
    return inter


def jaccard(box_a: tf.Tensor, box_b: tf.Tensor, return_union=False) -> tf.Tensor:
    """
    Compute the IoU of two sets of boxes.
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        The Jaccard overlap [a, b] between each bbox
    """
    # Get the intersectin area
    inter = intersect(box_a, box_b)

    # Compute the A area
    # (xmax - xmin) * (ymax - ymin)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    # Tile the area to match the anchors area
    area_a = tf.tile(tf.expand_dims(area_a, axis=-1), [1, tf.shape(inter)[-1]])

    # Compute the B area
    # (xmax - xmin) * (ymax - ymin)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    # Tile the area to match the gt areas
    area_b = tf.tile(tf.expand_dims(area_b, axis=-2), [tf.shape(inter)[-2], 1])

    union = area_a + area_b - inter

    if return_union is False:
        # Return the intesect over union
        return inter / union
    else:
        return inter / union, union


def merge(box_a: tf.Tensor, box_b: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Merged two set of boxes so that operations ca be run to compare them
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        Return the two same tensor tiled: (a, b, 4)
    """
    A = tf.shape(box_a)[0]  # Number of bbox in box_a
    B = tf.shape(box_b)[0]  # Number of bbox in box b
    # Above Right Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a = tf.tile(tf.expand_dims(box_a, axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b = tf.tile(tf.expand_dims(box_b, axis=0), [A, 1, 1])

    return tiled_box_a, tiled_box_b


def xy_min_xy_max_to_yx_min_yx_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    return tf.concat([bbox[:, 1:2], bbox[:, 0:1], bbox[:, 3:4], bbox[:, 2:3]], axis=-1)


def yx_min_yx_max_to_xy_min_xy_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    return tf.concat([bbox[:, 1:2], bbox[:, 0:1], bbox[:, 3:4], bbox[:, 2:3]], axis=-1)


def xy_min_xy_max_to_xcycwh(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xmin, ymin, xmax, ymax] to [xc, yc, w, h]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xmin, ymin, xmax, ymax] to [x_center, y_center, w, h]
    bbox_xcycwh = tf.concat(
        [bbox[:, :2] + ((bbox[:, 2:] - bbox[:, :2]) / 2), bbox[:, 2:] - bbox[:, :2]],
        axis=-1,
    )
    return bbox_xcycwh


def xcycwh_to_xy_min_xy_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xyxy = tf.concat(
        [bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1
    )
    # Be sure to keep the values btw 0 and 1
    bbox_xyxy = tf.clip_by_value(bbox_xyxy, 0.0, 1.0)
    return bbox_xyxy


def xcycwh_to_yx_min_yx_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xc, yc, w, h] to [ymin, xmin, ymax, xmax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    bbox = xcycwh_to_xy_min_xy_max(bbox)
    bbox = xy_min_xy_max_to_yx_min_yx_max(bbox)
    return bbox


def yx_min_yx_max_to_xcycwh(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [ymin, xmin, ymax, xmax] to [xc, yc, w, h]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    bbox = yx_min_yx_max_to_xy_min_xy_max(bbox)
    bbox = xy_min_xy_max_to_xcycwh(bbox)
    return bbox


"""
Numpy Transformations
"""


def xy_min_xy_max_to_xcycwh(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [xmin, ymin, xmax, ymax] to [xc, yc, w, h]
    Args:
        bbox A (np.array) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xmin, ymin, xmax, ymax] to [x_center, y_center, w, h]
    bbox_xcycwh = np.concatenate(
        [bbox[:, :2] + ((bbox[:, 2:] - bbox[:, :2]) / 2), bbox[:, 2:] - bbox[:, :2]],
        axis=-1,
    )
    return bbox_xcycwh


def np_xcycwh_to_xy_min_xy_max(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xy = np.concatenate(
        [bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1
    )
    return bbox_xy


def np_yx_min_yx_max_to_xy_min_xy_max(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (np.array) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    return np.concatenate(
        [bbox[:, 1:2], bbox[:, 0:1], bbox[:, 3:4], bbox[:, 2:3]], axis=-1
    )


def np_rescale_bbox_xcycwh(bbox_xcycwh: np.array, img_size: tuple):
    """
    Rescale a list of bbox to the image size
    @bbox_xcycwh: [[xc, yc, w, h], ...]
    @img_size (height, width)
    """
    bbox_xcycwh = np.array(bbox_xcycwh)  # Be sure to work with a numpy array
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    bbox_xcycwh_rescaled = bbox_xcycwh * scale
    return bbox_xcycwh_rescaled


def np_rescale_bbox_yx_min_yx_max(bbox_xcycwh: np.array, img_size: tuple):
    """
    Rescale a list of bbox to the image size
    @bbox_xcycwh: [[y_min, x_min, y_max, x_max], ...]
    @img_size (height, width)
    """
    bbox_xcycwh = np.array(bbox_xcycwh)  # Be sure to work with a numpy array
    scale = np.array([img_size[0], img_size[1], img_size[0], img_size[1]])
    bbox_xcycwh_rescaled = bbox_xcycwh * scale
    return bbox_xcycwh_rescaled


def np_rescale_bbox_xy_min_xy_max(bbox: np.array, img_size: tuple):
    """
    Rescale a list of bbox to the image size
    @bbox: [[x_min, y_min, x_max, y_max], ...]
    @img_size (height, width)
    """
    bbox = np.array(bbox)  # Be sure to work with a numpy array
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    bbox_rescaled = bbox * scale
    return bbox_rescaled


import numpy as np
import cv2


CLASS_COLOR_MAP = np.random.randint(0, 255, (100, 3))


def numpy_bbox_to_image(
    image, bbox_list, labels=None, scores=None, class_name=[], config=None
):
    """Numpy function used to display the bbox (target or prediction)"""
    assert (
        image.dtype == np.float32
        and image.dtype == np.float32
        and len(image.shape) == 3
    )

    if config is not None and config.normalized_method == "torch_resnet":
        channel_avg = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        image = (image * channel_std) + channel_avg
        image = (image * 255).astype(np.uint8)
    elif config is not None and config.normalized_method == "tf_resnet":
        image = image + mean
        image = image[..., ::-1]
        image = image / 255

    bbox_xcycwh = np_rescale_bbox_xcycwh(bbox_list, (image.shape[0], image.shape[1]))
    bbox_x1y1x2y2 = np_xcycwh_to_xy_min_xy_max(bbox_xcycwh)

    # Set the labels if not defined
    if labels is None:
        labels = np.zeros((bbox_x1y1x2y2.shape[0]))

    bbox_area = []
    # Go through each bbox
    for b in range(0, bbox_x1y1x2y2.shape[0]):
        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        bbox_area.append((x2 - x1) * (y2 - y1))

    # Go through each bbox
    for b in np.argsort(bbox_area)[::-1]:
        # Take a new color at reandon for this instance
        instance_color = np.random.randint(0, 255, (3))

        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1, x2, y2 = (
            max(0, x1),
            max(0, y1),
            min(image.shape[1], x2),
            min(image.shape[0], y2),
        )

        # Select the class associated with this bbox
        class_id = labels[int(b)]

        if scores is not None and len(scores) > 0:
            label_name = class_name[int(class_id)]
            label_name = "%s:%.2f" % (label_name, scores[b])
        else:
            label_name = class_name[int(class_id)]

        class_color = CLASS_COLOR_MAP[int(class_id)]

        color = instance_color

        multiplier = image.shape[0] / 500
        cv2.rectangle(
            image,
            (x1, y1),
            (x1 + int(multiplier * 15) * len(label_name), y1 + 20),
            class_color.tolist(),
            -10,
        )
        cv2.putText(
            image,
            label_name,
            (x1 + 2, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * multiplier,
            (0, 0, 0),
            1,
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), tuple(class_color.tolist()), 2)

    return image


def get_model_inference(m_outputs: dict, background_class, bbox_format="xy_center"):

    predicted_bbox = m_outputs["pred_boxes"][0]
    predicted_labels = m_outputs["pred_logits"][0]

    softmax = tf.nn.softmax(predicted_labels)
    predicted_scores = tf.reduce_max(softmax, axis=-1)
    predicted_labels = tf.argmax(softmax, axis=-1)

    indices = tf.where(predicted_labels != background_class)
    indices = tf.squeeze(indices, axis=-1)

    predicted_scores = tf.gather(predicted_scores, indices)
    predicted_labels = tf.gather(predicted_labels, indices)
    predicted_bbox = tf.gather(predicted_bbox, indices)

    if bbox_format == "xy_center":
        predicted_bbox = predicted_bbox
    elif bbox_format == "xyxy":
        predicted_bbox = xcycwh_to_xy_min_xy_max(predicted_bbox)
    elif bbox_format == "yxyx":
        predicted_bbox = xcycwh_to_yx_min_yx_max(predicted_bbox)
    else:
        raise NotImplementedError()

    return predicted_bbox, predicted_labels, predicted_scores


# first convert all string class to number (the target class)
t_class = image_anns["class"].map(lambda x: CLASS_NAMES.index(x)).to_numpy()
# Select the width of each image (should be the same since all the ann belongs to the
same image)
width = image_anns["width"].to_numpy()
# Select the height of each image
height = image_anns["height"].to_numpy()
# Select the xmin, ymin, xmax and ymax of each bbox
# Then, normalized the bbox to be between and 0 and 1
# Finally, convert the bbox from xmin,ymin,xmax,ymax to x_center,y_center,width,height
bbox_list = image_anns[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
bbox_list = bbox_list / [width[0], height[0], width[0], height[0]]
t_bbox = xy_min_xy_max_to_xcycwh(bbox_list)

print("CLASS_NAMES", CLASS_NAMES)
print("t_class", t_class)

# Now we can apply the target bbox and the target class on the image
img = (image / 255).astype(np.float32)
img = numpy_bbox_to_image(img, t_bbox, t_class, scores=None, class_name=CLASS_NAMES)
plt.imshow(img)

"""shell
!pip install imgaug==0.4.0
"""

import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import tensorflow as tf


def bbox_xcyc_wh_to_imgaug_bbox(bbox, target_class, height, width):

    img_aug_bbox = []

    for b in range(0, len(bbox)):
        bbox_xcyc_wh = bbox[b]
        # Convert size form 0.1 to height/width
        bbox_xcyc_wh = [
            bbox_xcyc_wh[0] * width,
            bbox_xcyc_wh[1] * height,
            bbox_xcyc_wh[2] * width,
            bbox_xcyc_wh[3] * height,
        ]
        x1 = bbox_xcyc_wh[0] - (bbox_xcyc_wh[2] / 2)
        x2 = bbox_xcyc_wh[0] + (bbox_xcyc_wh[2] / 2)
        y1 = bbox_xcyc_wh[1] - (bbox_xcyc_wh[3] / 2)
        y2 = bbox_xcyc_wh[1] + (bbox_xcyc_wh[3] / 2)

        n_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=target_class[b])

        img_aug_bbox.append(n_bbox)
    img_aug_bbox
    return img_aug_bbox


def prepare_aug_inputs(image, bbox, t_class):

    images_batch = []
    bbox_batch = []

    images_batch.append(image)

    # Create the Imgaug bbox
    bbs_original = bbox_xcyc_wh_to_imgaug_bbox(
        bbox, t_class, image.shape[0], image.shape[1]
    )
    bbs_original = BoundingBoxesOnImage(bbs_original, shape=image.shape)
    bbox_batch.append(bbs_original)

    for i in range(len(images_batch)):
        images_batch[i] = images_batch[i].astype(np.uint8)

    return images_batch, bbox_batch


def detr_aug_seq(image, config, augmenation):

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    target_min_side_size = 480

    # According to the paper
    min_side_min = 480
    min_side_max = 800
    max_side_max = 1333

    image_size = config.image_size
    if augmenation:

        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontal flips
                sometimes(
                    iaa.OneOf(
                        [
                            # Resize complety the image
                            iaa.Resize(
                                {"width": image_size[1], "height": image_size[0]},
                                interpolation=ia.ALL,
                            ),
                            # Crop into the image
                            iaa.CropToFixedSize(image_size[1], image_size[0]),
                            # Affine transform
                            iaa.Affine(
                                scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                            ),
                        ]
                    )
                ),
                # Be sure to resize to the target image size
                iaa.Resize(
                    {"width": image_size[1], "height": image_size[0]},
                    interpolation=ia.ALL,
                ),
            ],
            random_order=False,
        )  # apply augmenters in random order

        return seq

    else:

        seq = iaa.Sequential(
            [
                # Be sure to resize to the target image size
                iaa.Resize({"width": image_size[1], "height": image_size[0]})
            ],
            random_order=False,
        )  # apply augmenters in random order

        return seq

        """ Mode paper evaluation
        # Evaluation mode, we took the largest min side the model is trained on
        target_min_side_size = 480 
        image_min_side = min(float(image.shape[0]), float(image.shape[1]))
        image_max_side = max(float(image.shape[0]), float(image.shape[1]))
        
        min_side_scaling = target_min_side_size / image_min_side
        max_side_scaling = max_side_max / image_max_side
        scaling = min(min_side_scaling, max_side_scaling)
        n_height = int(scaling * image.shape[0])
        n_width = int(scaling * image.shape[1])
        seq = iaa.Sequential([
            iaa.Resize({"height": n_height, "width": n_width}),
        ])
        """

    return seq


def imgaug_bbox_to_xcyc_wh(bbs_aug, height, width):

    bbox_xcyc_wh = []
    t_class = []

    nb_bbox = 0

    for b, bbox in enumerate(bbs_aug):

        h = bbox.y2 - bbox.y1
        w = bbox.x2 - bbox.x1
        xc = bbox.x1 + (w / 2)
        yc = bbox.y1 + (h / 2)

        assert bbox.label != None

        bbox_xcyc_wh.append([xc / width, yc / height, w / width, h / height])
        t_class.append(bbox.label)

        nb_bbox += 1

    # bbox_xcyc_wh[0][0] = nb_bbox
    bbox_xcyc_wh = np.array(bbox_xcyc_wh)

    return bbox_xcyc_wh, t_class


def retrieve_outputs(augmented_images, augmented_bbox):

    outputs_dict = {}
    image_shape = None

    # We expect only one image here for now
    image = augmented_images[0].astype(np.float32)
    augmented_bbox = augmented_bbox[0]

    bbox, t_class = imgaug_bbox_to_xcyc_wh(
        augmented_bbox, image.shape[0], image.shape[1]
    )

    bbox = np.array(bbox)
    t_class = np.array(t_class)

    return image, bbox, t_class


def detr_transform(image, bbox, t_class, config, augmentation):

    images_batch, bbox_batch = prepare_aug_inputs(image, bbox, t_class)

    seq = detr_aug_seq(image, config, augmentation)

    # Run the pipeline in a deterministic manner
    seq_det = seq.to_deterministic()

    augmented_images = []
    augmented_bbox = []
    augmented_class = []

    for img, bbox, t_cls in zip(images_batch, bbox_batch, t_class):

        img_aug = seq_det.augment_image(img)
        bbox_aug = seq_det.augment_bounding_boxes(bbox)

        for b, bbox_instance in enumerate(bbox_aug.items):
            setattr(bbox_instance, "instance_id", b + 1)

        bbox_aug = bbox_aug.remove_out_of_image_fraction(0.7)
        segmap_aug = None
        bbox_aug = bbox_aug.clip_out_of_image()

        augmented_images.append(img_aug)
        augmented_bbox.append(bbox_aug)

    return retrieve_outputs(augmented_images, augmented_bbox)


"""
DETR Transformation / Augmentations
"""

# Transform/Resize images without any augmentation -> for validation
val_img, val_t_bbox, val_t_class = detr_transform(
    image, t_bbox, t_class, config, augmentation=False
)
# Transform/Resize images with augmentations operations -> for training
train_img, train_t_bbox, train_t_class = detr_transform(
    image, t_bbox, t_class, config, augmentation=True
)

# Display image with resize only
display_img = numpy_bbox_to_image(
    val_img / 255.0, val_t_bbox, val_t_class, scores=None, class_name=CLASS_NAMES
)
plt.imshow(display_img)
plt.show()
# Display image with transformations
display_img = numpy_bbox_to_image(
    train_img / 255, train_t_bbox, train_t_class, scores=None, class_name=CLASS_NAMES
)
plt.imshow(display_img)
plt.show()

"""
Image normalization:
"""


def normalized_images(image, config):
    """Normalized images. torch_resnet is used on finetuning
    since the weights are based on the  original paper training code
    from pytorch. tf_resnet is used when training from scratch with a
    resnet50 traine don tensorflow.
    """
    if config.normalized_method == "torch_resnet":
        channel_avg = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        image = (image / 255.0 - channel_avg) / channel_std
        return image.astype(np.float32)
    elif config.normalized_method == "tf_resnet":
        mean = [103.939, 116.779, 123.68]
        image = image[..., ::-1]
        image = image - mean
        return image.astype(np.float32)
    else:
        raise Exception("Can't handler thid normalized method")


def numpy_fc(idx, fc, outputs_types=(tf.float32, tf.float32, tf.int64), **params):
    """
Call a numpy function on each given ID (`idx`) and load the associated image and labels
(bbbox and cls)
    """

    def _np_function(_idx):
        return fc(_idx, **params)

    return tf.numpy_function(_np_function, [idx], outputs_types)


def pad_labels(images: tf.Tensor, t_bbox: tf.Tensor, t_class: tf.Tensor):
    """Pad the bbox by adding [0, 0, 0, 0] at the end
    and one header to indicate how maby bbox are set.
    Do the same with the labels.
    """
    nb_bbox = tf.shape(t_bbox)[0]

    bbox_header = tf.expand_dims(nb_bbox, axis=0)
    bbox_header = tf.expand_dims(bbox_header, axis=0)
    bbox_header = tf.pad(bbox_header, [[0, 0], [0, 3]])
    bbox_header = tf.cast(bbox_header, tf.float32)
    cls_header = tf.constant([[0]], dtype=tf.int64)

    # Padd bbox and class
    t_bbox = tf.pad(
        t_bbox, [[0, 100 - 1 - nb_bbox], [0, 0]], mode="CONSTANT", constant_values=0
    )
    t_class = tf.pad(
        t_class, [[0, 100 - 1 - nb_bbox], [0, 0]], mode="CONSTANT", constant_values=0
    )

    t_bbox = tf.concat([bbox_header, t_bbox], axis=0)
    t_class = tf.concat([cls_header, t_class], axis=0)

    return images, t_bbox, t_class


normalized_image = normalized_images(image, config)

# The method now use the config class to know the normalization to applied on the image
# before to render the outputs
display_img = numpy_bbox_to_image(
    normalized_image,
    t_bbox,
    t_class,
    scores=None,
    class_name=CLASS_NAMES,
    config=config,
)

plt.imshow(display_img)

"""
### Setting up a new TensorFlow dataset
"""

from random import shuffle


def load_hardhat(train_val, batch_size, config, augmentation=False):
    """Load the hardhat dataset"""
    # Open annotation file (train or val)
    anns = pd.read_csv(os.path.join(config.datadir, f"{train_val}/_annotations.csv"))
    # Set the class name and add the background at the begining
    CLASS_NAMES = ["background"] + anns["class"].unique().tolist()
    # Select all the unique images in this dataset
    filenames = anns["filename"].unique().tolist()
    indexes = list(range(0, len(filenames)))
    shuffle(indexes)
    dataset = tf.data.Dataset.from_tensor_slices(indexes)
    return dataset


dataset = load_hardhat("test", 1, config, augmentation=False)
for i, index in enumerate(dataset):
    print("Image indice", index)
    if i > 10:
        break


def load_hardhat_data_from_index(
    index, filenames, train_val, anns, config, augmentation
):
    # Open the image
    image = imageio.imread(
        os.path.join(config.datadir, f"{train_val}", filenames[img_id])
    )
    # Select all the annotatiom (bbox and class) on this image
    image_anns = anns[anns["filename"] == filenames[img_id]]

    # Convert all string class to number (the target class)
    t_class = image_anns["class"].map(lambda x: CLASS_NAMES.index(x)).to_numpy()
# Select the width&height of each image (should be the same since all the ann belongs to
the same image)
    width = image_anns["width"].to_numpy()
    height = image_anns["height"].to_numpy()
# Select the xmin, ymin, xmax and ymax of each bbox, Then, normalized the bbox to be
between and 0 and 1
    # Finally, convert the bbox from xmin,ymin,xmax,ymax to x_center,y_center,width,height
    bbox_list = image_anns[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    bbox_list = bbox_list / [width[0], height[0], width[0], height[0]]
    t_bbox = xy_min_xy_max_to_xcycwh(bbox_list)

    # Transform and augment image with bbox and class if needed
    image, t_bbox, t_class = detr_transform(
        image, t_bbox, t_class, config, augmentation=augmentation
    )

    # Normalized image
    image = normalized_images(image, config)

    return (
        image.astype(np.float32),
        t_bbox.astype(np.float32),
        np.expand_dims(t_class, axis=-1),
    )


def load_hardhat(train_val, batch_size, config, augmentation=False):
    """Load the hardhat dataset"""
    anns = pd.read_csv(os.path.join(config.datadir, f"{train_val}/_annotations.csv"))
    CLASS_NAMES = ["background"] + anns["class"].unique().tolist()
    filenames = anns["filename"].unique().tolist()
    indexes = list(range(0, len(filenames)))
    shuffle(indexes)

    dataset = tf.data.Dataset.from_tensor_slices(indexes)
    dataset = dataset.map(
        lambda idx: numpy_fc(
            idx,
            load_hardhat_data_from_index,
            filenames=filenames,
            train_val=train_val,
            anns=anns,
            config=config,
            augmentation=augmentation,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    return dataset


dataset = load_hardhat("test", 1, config, augmentation=False)
for i, (image, t_bbox, t_labels) in enumerate(dataset):
    print("shapes", image.shape, t_bbox.shape, t_labels.shape)
    if i > 10:
        break


def load_hardhat(train_val, batch_size, config, augmentation=False):
    """Load the hardhat dataset"""
    anns = pd.read_csv(os.path.join(config.datadir, f"{train_val}/_annotations.csv"))
    CLASS_NAMES = ["background"] + anns["class"].unique().tolist()
    filenames = anns["filename"].unique().tolist()
    indexes = list(range(0, len(filenames)))
    shuffle(indexes)

    dataset = tf.data.Dataset.from_tensor_slices(indexes)
    dataset = dataset.map(
        lambda idx: numpy_fc(
            idx,
            load_hardhat_data_from_index,
            filenames=filenames,
            train_val=train_val,
            anns=anns,
            config=config,
            augmentation=augmentation,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Filter labels to be sure to keep only sample with at least one bbox
    dataset = dataset.filter(lambda imgs, tbbox, tclass: tf.shape(tbbox)[0] > 0)
    # Pad bbox and labels
    dataset = dataset.map(pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


dataset = load_hardhat("test", 8, config, augmentation=False)
for i, (image, t_bbox, t_labels) in enumerate(dataset):
    print("shapes", image.shape, t_bbox.shape, t_labels.shape)
    if i > 10:
        break


"""
# Fine-tuning
"""

"""
Create a new training config:
"""

from os.path import expanduser
import os


class CustomConfig(TrainingConfig):
    def __init__(self):
        super().__init__()
        # Dataset info
        self.datadir = os.path.join(expanduser("."), "data/hardhat/")
        # The model is trained using fixed size images.
# The following is the desired target image size, but it can be change based on
your
        # dataset
        self.image_size = (480, 720)
        # Batch size
        self.batch_size = 1
# Using the target batch size , the training loop will agregate the gradient on 38
steps
        # before to update the weights
        self.target_batch = 8


config = CustomConfig()

import tensorflow as tf
from random import shuffle
import pandas as pd
import numpy as np
import imageio
import os


def load_data_from_index(
    index, class_names, filenames, anns, config, augmentation, img_dir
):
    # Open the image
    image = imageio.imread(
        os.path.join(config.data.data_dir, img_dir, filenames[index])
    )
    # Select all the annotatiom (bbox and class) on this image
    image_anns = anns[anns["filename"] == filenames[index]]

    # Convert all string class to number (the target class)
    t_class = image_anns["class"].map(lambda x: class_names.index(x)).to_numpy()
# Select the width&height of each image (should be the same since all the ann belongs to
the same image)
    width = image_anns["width"].to_numpy()
    height = image_anns["height"].to_numpy()
# Select the xmin, ymin, xmax and ymax of each bbox, Then, normalized the bbox to be
between and 0 and 1
    # Finally, convert the bbox from xmin,ymin,xmax,ymax to x_center,y_center,width,height
    bbox_list = image_anns[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    bbox_list = bbox_list / [width[0], height[0], width[0], height[0]]
    t_bbox = xy_min_xy_max_to_xcycwh(bbox_list)

    # Transform and augment image with bbox and class if needed
    image, t_bbox, t_class = detr_transform(
        image, t_bbox, t_class, config, augmentation=augmentation
    )

    # Normalized image
    image = normalized_images(image, config)

    return (
        image.astype(np.float32),
        t_bbox.astype(np.float32),
        np.expand_dims(t_class, axis=-1).astype(np.int64),
    )


def load_hardhat(train_val, batch_size, config, augmentation=False):
    """Load the hardhat dataset"""
    anns = pd.read_csv(os.path.join(config.datadir, f"{train_val}/_annotations.csv"))
    CLASS_NAMES = ["background"] + anns["class"].unique().tolist()
    filenames = anns["filename"].unique().tolist()
    indexes = list(range(0, len(filenames)))
    shuffle(indexes)

    dataset = tf.data.Dataset.from_tensor_slices(indexes)
    dataset = dataset.map(
        lambda idx: numpy_fc(
            idx,
            load_hardhat_data_from_index,
            filenames=filenames,
            train_val=train_val,
            anns=anns,
            config=config,
            augmentation=augmentation,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Filter labels to be sure to keep only sample with at least one bbox
    dataset = dataset.filter(lambda imgs, tbbox, tclass: tf.shape(tbbox)[0] > 0)
    # Pad bbox and labels
    dataset = dataset.map(pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset, CLASS_NAMES


def load_tfcsv_dataset(
    config,
    batch_size,
    augmentation=False,
    exclude=[],
    ann_dir=None,
    ann_file=None,
    img_dir=None,
    path=None,
):
    """Load the hardhat dataset"""
    ann_dir = config.data.ann_dir if ann_dir is None else ann_dir
    ann_file = config.data.ann_file if ann_file is None else ann_file
    img_dir = config.data.img_dir if img_dir is None else img_dir

    anns = pd.read_csv(os.path.join("./data/hardhat/" + path, "_annotations.csv"))
    for name in exclude:
        anns = anns[anns["class"] != name]

    unique_class = anns["class"].unique()
    unique_class.sort()

    # Set the background class to 0
    config.background_class = 0
    class_names = ["background"] + unique_class.tolist()

    filenames = anns["filename"].unique().tolist()
    # print(filenames)
    indexes = list(range(0, len(filenames)))
    shuffle(indexes)
    # print(indexes)

    dataset = tf.data.Dataset.from_tensor_slices(indexes)
    dataset = dataset.map(
        lambda idx: numpy_fc(
            idx,
            load_data_from_index,
            class_names=class_names,
            filenames=filenames,
            anns=anns,
            config=config,
            augmentation=augmentation,
            img_dir=img_dir,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    print(type(config))

    # Filter labels to be sure to keep only sample with at least one bbox
    dataset = dataset.filter(lambda imgs, tbbox, tclass: tf.shape(tbbox)[0] > 0)
    # Pad bbox and labels
    dataset = dataset.map(pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    dataset = dataset.batch(1, drop_remainder=True)

    return dataset, class_names


"""
### Load the Dataset
"""

train_iterator, class_names = dataset = load_hardhat(
    "train", config.batch_size, config, augmentation=True
)
valid_iterator, class_names = dataset = load_hardhat(
    "test", config.batch_size, config, augmentation=False
)

"""
We have 4 classes:
"""

print("class_names", class_names)

import numpy as np
import tensorflow as tf


class PositionEmbeddingSine(tf.keras.Model):
    def __init__(
        self,
        num_pos_features=64,
        temperature=10000,
        normalize=False,
        scale=None,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_pos_features = num_pos_features
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
        self.eps = eps

    def call(self, mask):
        not_mask = tf.cast(~mask, tf.float32)
        y_embed = tf.math.cumsum(not_mask, axis=1)
        x_embed = tf.math.cumsum(not_mask, axis=2)

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = tf.range(self.num_pos_features, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_features)

        pos_x = x_embed[..., tf.newaxis] / dim_t
        pos_y = y_embed[..., tf.newaxis] / dim_t

        pos_x = tf.stack(
            [tf.math.sin(pos_x[..., 0::2]), tf.math.cos(pos_x[..., 1::2])], axis=4
        )

        pos_y = tf.stack(
            [tf.math.sin(pos_y[..., 0::2]), tf.math.cos(pos_y[..., 1::2])], axis=4
        )

        shape = [tf.shape(pos_x)[i] for i in range(3)] + [-1]
        pos_x = tf.reshape(pos_x, shape)
        pos_y = tf.reshape(pos_y, shape)

        pos_emb = tf.concat([pos_y, pos_x], axis=3)
        return pos_emb


"""
### Custom layers
"""


class FrozenBatchNorm2D(tf.keras.layers.Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=[input_shape[-1]],
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=False,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[input_shape[-1]],
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=False,
        )
        self.running_mean = self.add_weight(
            name="running_mean",
            shape=[input_shape[-1]],
            initializer="zeros",
            trainable=False,
        )
        self.running_var = self.add_weight(
            name="running_var",
            shape=[input_shape[-1]],
            initializer="ones",
            trainable=False,
        )

    def call(self, x):
        scale = self.weight * tf.math.rsqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        return x * scale + shift

    def compute_output_shape(self, input_shape):
        return input_shape


class Linear(tf.keras.layers.Layer):
    """
    Use this custom layer instead of tf.keras.layers.Dense to allow
    loading converted PyTorch Dense weights that have shape (output_dim, input_dim)
    """

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[self.output_dim, input_shape[-1]],
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[self.output_dim],
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )

    def call(self, x):
        return tf.matmul(x, self.kernel, transpose_b=True) + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape.as_list()[:-1] + [self.output_dim]


class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_shape, **kwargs):
        super().__init__(**kwargs)
        self.embed_shape = embed_shape

    def build(self, input_shape):
        self.w = self.add_weight(
            name="kernel",
            shape=self.embed_shape,
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )

    def call(self, x=None):
        return self.w


"""
### ResNet Backbone
"""

from tensorflow.keras.layers import ZeroPadding2D, Conv2D, ReLU, MaxPool2D


class ResNetBase(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pad1 = ZeroPadding2D(3, name="pad1")
        self.conv1 = Conv2D(
            64, kernel_size=7, strides=2, padding="valid", use_bias=False, name="conv1"
        )
        self.bn1 = FrozenBatchNorm2D(name="bn1")
        self.relu = ReLU(name="relu")
        self.pad2 = ZeroPadding2D(1, name="pad2")
        self.maxpool = MaxPool2D(pool_size=3, strides=2, padding="valid")

    def call(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet50Backbone(ResNetBase):
    def __init__(self, replace_stride_with_dilation=[False, False, False], **kwargs):
        super().__init__(**kwargs)

        self.layer1 = ResidualBlock(
            num_bottlenecks=3,
            dim1=64,
            dim2=256,
            strides=1,
            replace_stride_with_dilation=False,
            name="layer1",
        )
        self.layer2 = ResidualBlock(
            num_bottlenecks=4,
            dim1=128,
            dim2=512,
            strides=2,
            replace_stride_with_dilation=replace_stride_with_dilation[0],
            name="layer2",
        )
        self.layer3 = ResidualBlock(
            num_bottlenecks=6,
            dim1=256,
            dim2=1024,
            strides=2,
            replace_stride_with_dilation=replace_stride_with_dilation[1],
            name="layer3",
        )
        self.layer4 = ResidualBlock(
            num_bottlenecks=3,
            dim1=512,
            dim2=2048,
            strides=2,
            replace_stride_with_dilation=replace_stride_with_dilation[2],
            name="layer4",
        )


class ResNet101Backbone(ResNetBase):
    def __init__(self, replace_stride_with_dilation=[False, False, False], **kwargs):
        super().__init__(**kwargs)

        self.layer1 = ResidualBlock(
            num_bottlenecks=3,
            dim1=64,
            dim2=256,
            strides=1,
            replace_stride_with_dilation=False,
            name="layer1",
        )
        self.layer2 = ResidualBlock(
            num_bottlenecks=4,
            dim1=128,
            dim2=512,
            strides=2,
            replace_stride_with_dilation=replace_stride_with_dilation[0],
            name="layer2",
        )
        self.layer3 = ResidualBlock(
            num_bottlenecks=23,
            dim1=256,
            dim2=1024,
            strides=2,
            replace_stride_with_dilation=replace_stride_with_dilation[1],
            name="layer3",
        )
        self.layer4 = ResidualBlock(
            num_bottlenecks=3,
            dim1=512,
            dim2=2048,
            strides=2,
            replace_stride_with_dilation=replace_stride_with_dilation[2],
            name="layer4",
        )


class ResidualBlock(tf.keras.Model):
    def __init__(
        self,
        num_bottlenecks,
        dim1,
        dim2,
        strides=1,
        replace_stride_with_dilation=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if replace_stride_with_dilation:
            strides = 1
            dilation = 2
        else:
            dilation = 1

        self.bottlenecks = [
            BottleNeck(dim1, dim2, strides=strides, downsample=True, name="0")
        ]

        for idx in range(1, num_bottlenecks):
            self.bottlenecks.append(
                BottleNeck(dim1, dim2, name=str(idx), dilation=dilation)
            )

    def call(self, x):
        for btn in self.bottlenecks:
            x = btn(x)
        return x


class BottleNeck(tf.keras.Model):
    def __init__(self, dim1, dim2, strides=1, dilation=1, downsample=False, **kwargs):
        super().__init__(**kwargs)
        self.downsample = downsample
        self.pad = ZeroPadding2D(dilation)
        self.relu = ReLU(name="relu")

        self.conv1 = Conv2D(dim1, kernel_size=1, use_bias=False, name="conv1")
        self.bn1 = FrozenBatchNorm2D(name="bn1")

        self.conv2 = Conv2D(
            dim1,
            kernel_size=3,
            strides=strides,
            dilation_rate=dilation,
            use_bias=False,
            name="conv2",
        )
        self.bn2 = FrozenBatchNorm2D(name="bn2")

        self.conv3 = Conv2D(dim2, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = FrozenBatchNorm2D(name="bn3")

        self.downsample_conv = Conv2D(
            dim2, kernel_size=1, strides=strides, use_bias=False, name="downsample_0"
        )
        self.downsample_bn = FrozenBatchNorm2D(name="downsample_1")

    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.downsample_bn(self.downsample_conv(x))

        out += identity
        out = self.relu(out)

        return out


"""
### Transformer
"""

from tensorflow.keras.layers import Dropout, Activation, LayerNormalization


class Transformer(tf.keras.Model):
    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads

        enc_norm = (
            LayerNormalization(epsilon=1e-5, name="norm_pre")
            if normalize_before
            else None
        )
        self.encoder = TransformerEncoder(
            model_dim,
            num_heads,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            enc_norm,
            num_encoder_layers,
            name="encoder",
        )

        dec_norm = LayerNormalization(epsilon=1e-5, name="norm")
        self.decoder = TransformerDecoder(
            model_dim,
            num_heads,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            dec_norm,
            num_decoder_layers,
            name="decoder",
            return_intermediate=return_intermediate_dec,
        )

    def call(self, source, mask, query_encoding, pos_encoding, training=False):

        batch_size, rows, cols = [tf.shape(source)[i] for i in range(3)]
        source = tf.reshape(source, [batch_size, -1, self.model_dim])
        source = tf.transpose(source, [1, 0, 2])

        pos_encoding = tf.reshape(pos_encoding, [batch_size, -1, self.model_dim])
        pos_encoding = tf.transpose(pos_encoding, [1, 0, 2])

        query_encoding = tf.expand_dims(query_encoding, axis=1)
        query_encoding = tf.tile(query_encoding, [1, batch_size, 1])

        mask = tf.reshape(mask, [batch_size, -1])

        target = tf.zeros_like(query_encoding)

        memory = self.encoder(
            source,
            source_key_padding_mask=mask,
            pos_encoding=pos_encoding,
            training=training,
        )
        hs = self.decoder(
            target,
            memory,
            memory_key_padding_mask=mask,
            pos_encoding=pos_encoding,
            query_encoding=query_encoding,
            training=training,
        )

        hs = tf.transpose(hs, [0, 2, 1, 3])
        memory = tf.transpose(memory, [1, 0, 2])
        memory = tf.reshape(memory, [batch_size, rows, cols, self.model_dim])

        return hs, memory


class TransformerEncoder(tf.keras.Model):
    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        norm=None,
        num_encoder_layers=6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.enc_layers = [
            EncoderLayer(
                model_dim,
                num_heads,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
                name="layer_%d" % i,
            )
            for i in range(num_encoder_layers)
        ]

        self.norm = norm

    def call(
        self,
        source,
        mask=None,
        source_key_padding_mask=None,
        pos_encoding=None,
        training=False,
    ):
        x = source

        for layer in self.enc_layers:
            x = layer(
                x,
                source_mask=mask,
                source_key_padding_mask=source_key_padding_mask,
                pos_encoding=pos_encoding,
                training=training,
            )

        if self.norm:
            x = self.norm(x)

        return x


class TransformerDecoder(tf.keras.Model):
    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        norm=None,
        num_decoder_layers=6,
        return_intermediate=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dec_layers = [
            DecoderLayer(
                model_dim,
                num_heads,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
                name="layer_%d" % i,
            )
            for i in range(num_decoder_layers)
        ]

        self.norm = norm
        self.return_intermediate = return_intermediate

    def call(
        self,
        target,
        memory,
        target_mask=None,
        memory_mask=None,
        target_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_encoding=None,
        query_encoding=None,
        training=False,
    ):

        x = target
        intermediate = []

        for layer in self.dec_layers:
            x = layer(
                x,
                memory,
                target_mask=target_mask,
                memory_mask=memory_mask,
                target_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_encoding=pos_encoding,
                query_encoding=query_encoding,
            )

            if self.return_intermediate:
                if self.norm:
                    intermediate.append(self.norm(x))
                else:
                    intermediate.append(x)

        if self.return_intermediate:
            return tf.stack(intermediate, axis=0)

        if self.norm:
            x = self.norm(x)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.self_attn = MultiHeadAttention(
            model_dim, num_heads, dropout=dropout, name="self_attn"
        )

        self.dropout = Dropout(dropout)
        self.activation = Activation(activation)

        self.linear1 = Linear(dim_feedforward, name="linear1")
        self.linear2 = Linear(model_dim, name="linear2")

        self.norm1 = LayerNormalization(epsilon=1e-5, name="norm1")
        self.norm2 = LayerNormalization(epsilon=1e-5, name="norm2")

        self.normalize_before = normalize_before

    def call(
        self,
        source,
        source_mask=None,
        source_key_padding_mask=None,
        pos_encoding=None,
        training=False,
    ):

        if pos_encoding is None:
            query = key = source
        else:
            query = key = source + pos_encoding

        attn_source = self.self_attn(
            (query, key, source),
            attn_mask=source_mask,
            key_padding_mask=source_key_padding_mask,
            need_weights=False,
        )
        source += self.dropout(attn_source, training=training)
        source = self.norm1(source)

        x = self.linear1(source)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.linear2(x)
        source += self.dropout(x, training=training)
        source = self.norm2(source)

        return source


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.self_attn = MultiHeadAttention(
            model_dim, num_heads, dropout=dropout, name="self_attn"
        )
        self.multihead_attn = MultiHeadAttention(
            model_dim, num_heads, dropout=dropout, name="multihead_attn"
        )

        self.dropout = Dropout(dropout)
        self.activation = Activation(activation)

        self.linear1 = Linear(dim_feedforward, name="linear1")
        self.linear2 = Linear(model_dim, name="linear2")

        self.norm1 = LayerNormalization(epsilon=1e-5, name="norm1")
        self.norm2 = LayerNormalization(epsilon=1e-5, name="norm2")
        self.norm3 = LayerNormalization(epsilon=1e-5, name="norm3")

        self.normalize_before = normalize_before

    def call(
        self,
        target,
        memory,
        target_mask=None,
        memory_mask=None,
        target_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_encoding=None,
        query_encoding=None,
        training=False,
    ):

        query_tgt = key_tgt = target + query_encoding
        attn_target = self.self_attn(
            (query_tgt, key_tgt, target),
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
            need_weights=False,
        )
        target += self.dropout(attn_target, training=training)
        target = self.norm1(target)

        query_tgt = target + query_encoding
        key_mem = memory + pos_encoding

        attn_target2 = self.multihead_attn(
            (query_tgt, key_mem, memory),
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        target += self.dropout(attn_target2, training=training)
        target = self.norm2(target)

        x = self.linear1(target)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.linear2(x)
        target += self.dropout(x, training=training)
        target = self.norm3(target)

        return target


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads

        assert model_dim % num_heads == 0
        self.head_dim = model_dim // num_heads

        self.dropout = Dropout(rate=dropout)

    def build(self, input_shapes):
        in_dim = sum([shape[-1] for shape in input_shapes[:3]])

        self.in_proj_weight = self.add_weight(
            name="in_proj_kernel",
            shape=(in_dim, self.model_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True,
        )
        self.in_proj_bias = self.add_weight(
            name="in_proj_bias",
            shape=(in_dim,),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True,
        )
        self.out_proj_weight = self.add_weight(
            name="out_proj_kernel",
            shape=(self.model_dim, self.model_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True,
        )
        self.out_proj_bias = self.add_weight(
            name="out_proj_bias",
            shape=(self.model_dim,),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True,
        )

        # self.in_proj_weight = tf.Variable(
        #    tf.zeros((in_dim, self.model_dim), dtype=tf.float32), name='in_proj_kernel')
        # self.in_proj_bias = tf.Variable(tf.zeros((in_dim,), dtype=tf.float32),
        #                                name='in_proj_bias')

        # self.out_proj_weight = tf.Variable(
#    tf.zeros((self.model_dim, self.model_dim), dtype=tf.float32),
name='out_proj_kernel')
        # self.out_proj_bias = tf.Variable(
        #    tf.zeros((self.model_dim,), dtype=tf.float32), name='out_proj_bias')

    def call(
        self,
        inputs,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=True,
        training=False,
    ):

        query, key, value = inputs

        batch_size = tf.shape(query)[1]
        target_len = tf.shape(query)[0]
        source_len = tf.shape(key)[0]

        W = self.in_proj_weight[: self.model_dim, :]
        b = self.in_proj_bias[: self.model_dim]

        WQ = tf.matmul(query, W, transpose_b=True) + b

        W = self.in_proj_weight[self.model_dim : 2 * self.model_dim, :]
        b = self.in_proj_bias[self.model_dim : 2 * self.model_dim]
        WK = tf.matmul(key, W, transpose_b=True) + b

        W = self.in_proj_weight[2 * self.model_dim :, :]
        b = self.in_proj_bias[2 * self.model_dim :]
        WV = tf.matmul(value, W, transpose_b=True) + b

        WQ *= float(self.head_dim) ** -0.5
        WQ = tf.reshape(WQ, [target_len, batch_size * self.num_heads, self.head_dim])
        WQ = tf.transpose(WQ, [1, 0, 2])

        WK = tf.reshape(WK, [source_len, batch_size * self.num_heads, self.head_dim])
        WK = tf.transpose(WK, [1, 0, 2])

        WV = tf.reshape(WV, [source_len, batch_size * self.num_heads, self.head_dim])
        WV = tf.transpose(WV, [1, 0, 2])

        attn_output_weights = tf.matmul(WQ, WK, transpose_b=True)

        if attn_mask is not None:
            attn_output_weights += attn_mask

        """
        if key_padding_mask is not None:
            attn_output_weights = tf.reshape(attn_output_weights,
                                [batch_size, self.num_heads, target_len, source_len])
            key_padding_mask = tf.expand_dims(key_padding_mask, 1)
            key_padding_mask = tf.expand_dims(key_padding_mask, 2)
key_padding_mask = tf.tile(key_padding_mask, [1, self.num_heads, target_len,
1])
            #print("before attn_output_weights", attn_output_weights.shape)
            attn_output_weights = tf.where(key_padding_mask,
tf.zeros_like(attn_output_weights) +
float('-inf'),
                                           attn_output_weights)
            attn_output_weights = tf.reshape(attn_output_weights,
                                [batch_size * self.num_heads, target_len, source_len])
        """

        attn_output_weights = tf.nn.softmax(attn_output_weights, axis=-1)
        attn_output_weights = self.dropout(attn_output_weights, training=training)

        attn_output = tf.matmul(attn_output_weights, WV)
        attn_output = tf.transpose(attn_output, [1, 0, 2])
        attn_output = tf.reshape(attn_output, [target_len, batch_size, self.model_dim])
        attn_output = (
            tf.matmul(attn_output, self.out_proj_weight, transpose_b=True)
            + self.out_proj_bias
        )

        if need_weights:
            attn_output_weights = tf.reshape(
                attn_output_weights,
                [batch_size, self.num_heads, target_len, source_len],
            )
            # Retrun the average weight over the heads
            avg_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, avg_weights

        return attn_output


import os
import requests


WEIGHT_NAME_TO_CKPT = {
    "detr": [
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/checkpoint",
"https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.data-00000-of-00001",
"https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.data-00000-of-00001",
"https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.index",
"https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.index",
    ]
}


def load_weights(model, weights: str):
    """Load weight on a given model
weights are supposed to be sotred in the weight folder at the root of the repository. If
weights
    does not exists, but are publicly known, the weight will be download from gcloud.
    """
    if not os.path.exists("weights"):
        os.makedirs("weights")

    if "ckpt" in "weights":
        model.load(weights)
    elif weights in WEIGHT_NAME_TO_CKPT:
        wdir = f"weights/{weights}"
        if not os.path.exists(wdir):
            os.makedirs(wdir)
        for f in WEIGHT_NAME_TO_CKPT[weights]:
            fname = f.split("/")[-1]
            if not os.path.exists(os.path.join(wdir, fname)):
                print("Download....", f)
                r = requests.get(f, allow_redirects=True)
                open(os.path.join(wdir, fname), "wb").write(r.content)
        print("Load weights from", os.path.join(wdir, f"{weights}.ckpt"))
        l = model.load_weights(os.path.join(wdir, f"{weights}.ckpt"))
        l.expect_partial()
    else:
        raise Exception(f"Cant load the weights: {weights}")


"""
### DETR Model
"""

import pickle
import tensorflow as tf
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path


class DETR(tf.keras.Model):
    def __init__(
        self,
        num_classes=92,
        num_queries=100,
        backbone=None,
        pos_encoder=None,
        transformer=None,
        num_encoder_layers=6,
        num_decoder_layers=6,
        return_intermediate_dec=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_queries = num_queries

        self.backbone = ResNet50Backbone(name="backbone")
        self.transformer = transformer or Transformer(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            return_intermediate_dec=return_intermediate_dec,
            name="transformer",
        )

        self.model_dim = self.transformer.model_dim

        self.pos_encoder = pos_encoder or PositionEmbeddingSine(
            num_pos_features=self.model_dim // 2,
            normalize=True,
            name="position_embedding_sine",
        )

        self.input_proj = tf.keras.layers.Conv2D(
            self.model_dim, kernel_size=1, name="input_proj"
        )

        self.query_embed = FixedEmbedding(
            (num_queries, self.model_dim), name="query_embed"
        )

        self.class_embed = Linear(num_classes, name="class_embed")

        self.bbox_embed_linear1 = Linear(self.model_dim, name="bbox_embed_0")
        self.bbox_embed_linear2 = Linear(self.model_dim, name="bbox_embed_1")
        self.bbox_embed_linear3 = Linear(4, name="bbox_embed_2")
        self.activation = tf.keras.layers.ReLU(name="re_lu")

    def downsample_masks(self, masks, x):
        masks = tf.cast(masks, tf.int32)
        masks = tf.expand_dims(masks, -1)
        masks = tf.compat.v1.image.resize_nearest_neighbor(
            masks, tf.shape(x)[1:3], align_corners=False, half_pixel_centers=False
        )
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks, tf.bool)
        return masks

    def call(self, inp, training=False, post_process=False):
        x, masks = inp
        x = self.backbone(x, training=training)
        masks = self.downsample_masks(masks, x)

        pos_encoding = self.pos_encoder(masks)

        hs = self.transformer(
            self.input_proj(x),
            masks,
            self.query_embed(None),
            pos_encoding,
            training=training,
        )[0]

        outputs_class = self.class_embed(hs)

        box_ftmps = self.activation(self.bbox_embed_linear1(hs))
        box_ftmps = self.activation(self.bbox_embed_linear2(box_ftmps))
        outputs_coord = tf.sigmoid(self.bbox_embed_linear3(box_ftmps))

        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        if post_process:
            output = self.post_process(output)
        return output

    def build(self, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [(None, None, None, 3), (None, None, None)]
        super().build(input_shape, **kwargs)


def add_heads_nlayers(config, detr, nb_class):
    image_input = tf.keras.Input((None, None, 3))
    # Setup the new layers
    cls_layer = tf.keras.layers.Dense(nb_class, name="cls_layer")
    pos_layer = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(4, activation="sigmoid"),
        ],
        name="pos_layer",
    )
    config.add_nlayers([cls_layer, pos_layer])

    transformer_output = detr(image_input)
    cls_preds = cls_layer(transformer_output)
    pos_preds = pos_layer(transformer_output)

    # Define the main outputs along with the auxialiary loss
    outputs = {"pred_logits": cls_preds[-1], "pred_boxes": pos_preds[-1]}
    outputs["aux"] = [
        {"pred_logits": cls_preds[i], "pred_boxes": pos_preds[i]} for i in range(0, 5)
    ]

    n_detr = tf.keras.Model(image_input, outputs, name="detr_finetuning")
    return n_detr


def get_detr_model(
    config,
    include_top=False,
    nb_class=None,
    weights=None,
    tf_backbone=False,
    num_decoder_layers=6,
    num_encoder_layers=6,
):
    """Get the DETR model
    Parameters
    ----------
    include_top: bool
        If false, the last layers of the transformers used to predict the bbox position
and cls will not be include. And therefore could be replace for finetuning if the
`weight` parameter
        is set.
    nb_class: int
If include_top is False and nb_class is set, then, this method will automaticly add two
new heads to predict
        the bbox pos and the bbox class on the decoder.
    weights: str
        Name of the weights to load. Only "detr" is avaiable to get started for now.
        More weight as detr-r101 will be added soon.
    tf_backbone:
        Using the pretrained weight from pytorch, the resnet backbone does not used
tf.keras.application to load the weight. If you do want to load the tf backbone, and
not
        laod the weights from pytorch, set this variable to True.
    """
    detr = DETR(
        num_decoder_layers=num_decoder_layers, num_encoder_layers=num_encoder_layers
    )

    if weights is not None:
        load_weights(detr, weights)

    image_input = tf.keras.Input((None, None, 3))

    # Backbone
    if not tf_backbone:
        backbone = detr.get_layer("backbone")
    else:
        config.normalized_method = "tf_resnet"
        backbone = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=(None, None, 3)
        )

    # Transformer
    transformer = detr.get_layer("transformer")
    # Positional embedding of the feature map
    position_embedding_sine = detr.get_layer("position_embedding_sine")
    # Used to project the feature map before to fit in into the encoder
    input_proj = detr.get_layer("input_proj")
    # Decoder objects query embedding
    query_embed = detr.get_layer("query_embed")

    # Used to project the  output of the decoder into a class prediction
    # This layer will be replace for finetuning
    class_embed = detr.get_layer("class_embed")

    # Predict the bbox pos
    bbox_embed_linear1 = detr.get_layer("bbox_embed_0")
    bbox_embed_linear2 = detr.get_layer("bbox_embed_1")
    bbox_embed_linear3 = detr.get_layer("bbox_embed_2")
    activation = detr.get_layer("re_lu")

    x = backbone(image_input)

    masks = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]), tf.bool)
    pos_encoding = position_embedding_sine(masks)

    hs = transformer(input_proj(x), masks, query_embed(None), pos_encoding)[0]

    detr = tf.keras.Model(image_input, hs, name="detr")
    if include_top is False and nb_class is None:
        return detr
    elif include_top is False and nb_class is not None:
        return add_heads_nlayers(config, detr, nb_class)

    transformer_output = detr(image_input)

    outputs_class = class_embed(transformer_output)
    box_ftmps = activation(bbox_embed_linear1(transformer_output))
    box_ftmps = activation(bbox_embed_linear2(box_ftmps))
    outputs_coord = tf.sigmoid(bbox_embed_linear3(box_ftmps))

    outputs = {}

    output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

    output["aux"] = []
    for i in range(0, num_decoder_layers - 1):
        out_class = outputs_class[i]
        pred_boxes = outputs_coord[i]
        output["aux"].append({"pred_logits": out_class, "pred_boxes": pred_boxes})

    return tf.keras.Model(image_input, output, name="detr_finetuning")


"""
### Load the DETR model
"""

detr = get_detr_model(config, include_top=False, nb_class=4, weights="detr")
detr.summary()

"""
Change the config to train only the layers on top of the transformers
"""

# Train/finetune the transformers only
config.train_backbone = tf.Variable(False)
config.train_transformers = tf.Variable(False)
config.train_nlayers = tf.Variable(True)

config.nlayers_lr = tf.Variable(1e-3)


def disable_batchnorm_training(model):
    for l in model.layers:
        if hasattr(l, "layers"):
            disable_batchnorm_training(l)
        elif isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False


def get_transformers_trainable_variables(model, exclude=[]):
    transformers_variables = []

    # Transformers variables
    transformers_variables = (
        model.get_layer("detr").get_layer("transformer").trainable_variables
    )

    for layer in model.layers[2:]:
        if layer.name not in exclude:
            transformers_variables += layer.trainable_variables
        else:
            pass

    return transformers_variables


def get_backbone_trainable_variables(model):
    backbone_variables = []
    # layer [1] is the detr model including the backbone and the transformers

    detr = model.get_layer("detr")
    tr_index = [l.name for l in detr.layers].index("transformer")

    for l, layer in enumerate(detr.layers):
        if l != tr_index:
            backbone_variables += layer.trainable_variables

    return backbone_variables


def get_nlayers_trainables_variables(model, nlayers_names):
    nlayers_variables = []
    for nlayer_name in nlayers_names:
        nlayers_variables += model.get_layer(nlayer_name).trainable_variables
    return nlayers_variables


def get_trainable_variables(model, config):

    disable_batchnorm_training(model)

    backbone_variables = []
    transformers_variables = []
    nlayers_variables = []

    # Retrieve the gradient ofr each trainable variables
    # if config.train_backbone:
    backbone_variables = get_backbone_trainable_variables(model)
    # if config.train_transformers:
    transformers_variables = get_transformers_trainable_variables(
        model, exclude=config.nlayers
    )
    # if config.train_nlayers:
    nlayers_variables = get_nlayers_trainables_variables(model, config.nlayers)

    return backbone_variables, transformers_variables, nlayers_variables


def setup_optimizers(model, config):
    """Method call by the Scheduler to init user data"""

    @tf.function
    def get_backbone_learning_rate():
        return config.backbone_lr

    @tf.function
    def get_transformers_learning_rate():
        return config.transformers_lr

    @tf.function
    def get_nlayers_learning_rate():
        return config.nlayers_lr

    # Disable batch norm on the backbone
    disable_batchnorm_training(model)

    # Optimizers
    backbone_optimizer = tf.keras.optimizers.Adam(
        learning_rate=get_backbone_learning_rate, clipnorm=config.gradient_norm_clipping
    )
    transformers_optimizer = tf.keras.optimizers.Adam(
        learning_rate=get_transformers_learning_rate,
        clipnorm=config.gradient_norm_clipping,
    )
    nlayers_optimizer = tf.keras.optimizers.Adam(
        learning_rate=get_nlayers_learning_rate, clipnorm=config.gradient_norm_clipping
    )

    # Set trainable variables

    backbone_variables, transformers_variables, nlayers_variables = [], [], []

    backbone_variables = get_backbone_trainable_variables(model)
    transformers_variables = get_transformers_trainable_variables(
        model, exclude=config.nlayers
    )
    nlayers_variables = get_nlayers_trainables_variables(model, config.nlayers)

    return {
        "backbone_optimizer": backbone_optimizer,
        "transformers_optimizer": transformers_optimizer,
        "nlayers_optimizer": nlayers_optimizer,
        "backbone_variables": backbone_variables,
        "transformers_variables": transformers_variables,
        "nlayers_variables": nlayers_variables,
    }


def gather_gradient(model, optimizers, total_loss, tape, config, log):

    (
        backbone_variables,
        transformers_variables,
        nlayers_variables,
    ) = get_trainable_variables(model, config)
    trainables_variables = (
        backbone_variables + transformers_variables + nlayers_variables
    )

    gradients = tape.gradient(total_loss, trainables_variables)

    # Retrieve the gradients from the tap
    backbone_gradients = gradients[: len(optimizers["backbone_variables"])]
    transformers_gradients = gradients[
        len(optimizers["backbone_variables"]) : len(optimizers["backbone_variables"])
        + len(optimizers["transformers_variables"])
    ]
    nlayers_gradients = gradients[
        len(optimizers["backbone_variables"])
        + len(optimizers["transformers_variables"]) :
    ]

    gradient_steps = {}

    gradient_steps["backbone"] = {"gradients": backbone_gradients}
    gradient_steps["transformers"] = {"gradients": transformers_gradients}
    gradient_steps["nlayers"] = {"gradients": nlayers_gradients}

    log.update(
        {
            "backbone_lr": optimizers["backbone_optimizer"]._serialize_hyperparameter(
                "learning_rate"
            )
        }
    )
    log.update(
        {
            "transformers_lr": optimizers[
                "transformers_optimizer"
            ]._serialize_hyperparameter("learning_rate")
        }
    )
    log.update(
        {
            "nlayers_lr": optimizers["nlayers_optimizer"]._serialize_hyperparameter(
                "learning_rate"
            )
        }
    )

    return gradient_steps


def aggregate_grad_and_apply(name, optimizers, gradients, step, config):

    gradient_aggregate = None
    if config.target_batch is not None:
        gradient_aggregate = int(config.target_batch // config.batch_size)

    gradient_name = "{}_gradients".format(name)
    optimizer_name = "{}_optimizer".format(name)
    variables_name = "{}_variables".format(name)
    train_part_name = "train_{}".format(name)

    if getattr(config, train_part_name):

        # Init the aggregate gradient
        if gradient_aggregate is not None and step % gradient_aggregate == 0:
            optimizers[gradient_name] = [
                tf.zeros_like(tv) for tv in optimizers[variables_name]
            ]

        if gradient_aggregate is not None:
            # Aggregate the gradient
            optimizers[gradient_name] = [
                (gradient + n_gradient) if n_gradient is not None else None
                for gradient, n_gradient in zip(optimizers[gradient_name], gradients)
            ]
        else:
            optimizers[gradient_name] = gradients

# Apply gradient if no gradient aggregate or if we finished gathering gradient
oversteps
        if gradient_aggregate is None or (step + 1) % gradient_aggregate == 0:
            optimizers[optimizer_name].apply_gradients(
                zip(optimizers[gradient_name], optimizers[variables_name])
            )


optimzers = setup_optimizers(detr, config)

from scipy.special import softmax
import matplotlib.pyplot as plt
from itertools import product
import tensorflow as tf
import numpy as np
import argparse
import random
import json
import time
import cv2
import os

from collections import OrderedDict


class APDataObject:
"""Stores all the information necessary to calculate the AP for one IoU and one
class."""

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """Call this once per image."""
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """Warning: result not cached."""

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

# Compute the precision-recall curve. The x axis is recalls and the y axis
precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

# Smooth the curve by computing [max(precisions[i:]) for i in
range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

# Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length
riemann summation with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

# I realize this is weird, but all it does is find the nearest precision(x) for a given x
in x_range.
# Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) =
precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side="left")
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > 0.5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > 0.5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def calc_map(ap_data, iou_thresholds, class_name, print_result=False):
    # print('Calculating mAP...')
    aps = [{"box": [], "mask": []} for _ in iou_thresholds]

    for _class in range(len(class_name)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ("box", "mask"):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {"box": OrderedDict(), "mask": OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ("box", "mask"):
        all_maps[iou_type]["all"] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = (
                sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100
                if len(aps[i][iou_type]) > 0
                else 0
            )
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]["all"] = sum(all_maps[iou_type].values()) / (
            len(all_maps[iou_type].values()) - 1
        )

    if print_result:
        print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps


def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (" %5s |" * len(vals)) % tuple(vals)
    make_sep = lambda n: ("-------+" * n)

    print()
    print(
        make_row(
            [""]
            + [
                (".%d " % x if isinstance(x, int) else x + " ")
                for x in all_maps["box"].keys()
            ]
        )
    )
    print(make_sep(len(all_maps["box"]) + 1))
    for iou_type in ("box", "mask"):
        print(
            make_row(
                [iou_type]
                + [
                    "%.2f" % x if x < 100 else "%.1f" % x
                    for x in all_maps[iou_type].values()
                ]
            )
        )
    print(make_sep(len(all_maps["box"]) + 1))
    print()


def cal_map(
    p_bbox,
    p_labels,
    p_scores,
    p_mask,
    t_bbox,
    gt_classes,
    t_mask,
    ap_data,
    iou_thresholds,
):

    # print("p_bbox", p_bbox.shape)
    # print("p_labels", p_labels.shape)
    # print("p_scores", p_scores.shape)
    # print("p_mask", p_mask.shape)
    # print("t_bbox", t_bbox.shape)
    # print("gt_classes", gt_classes)
    # print("t_mask", t_mask.shape)

    num_crowd = 0

    classes = list(np.array(p_labels).astype(int))
    scores = list(np.array(p_scores).astype(float))

    box_scores = scores
    mask_scores = scores

    masks = p_mask

    num_pred = len(classes)
    num_gt = len(gt_classes)

    mask_iou_cache = compute_overlaps_masks(masks, t_mask)
    bbox_iou_cache = compute_overlaps(p_bbox, t_bbox)

    crowd_mask_iou_cache = None
    crowd_bbox_iou_cache = None

    box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
    mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

    iou_types = [
        (
            "box",
            lambda i, j: bbox_iou_cache[i, j].item(),
            lambda i, j: crowd_bbox_iou_cache[i, j].item(),
            lambda i: box_scores[i],
            box_indices,
        ),
        (
            "mask",
            lambda i, j: mask_iou_cache[i, j].item(),
            lambda i, j: crowd_mask_iou_cache[i, j].item(),
            lambda i: mask_scores[i],
            mask_indices,
        ),
    ]
    # print("run", list(classes), list(gt_classes))
    # print(classes + gt_classes)
    for _class in set(list(classes) + list(gt_classes)):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]
            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j

                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue

                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

# All this crowd code so that we can make sure that our eval code gives
the
# same result as COCOEval. There aren't even that many crowd annotations
to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)


from typing import Union, Dict, Tuple
from itertools import product
import tensorflow as tf
import numpy as np

from scipy.optimize import linear_sum_assignment


def get_offsets(anchors_xywh, target_bbox_xywh):
    # Return the offset between the boxes in anchors_xywh and the boxes
    # in anchors_xywh

    variances = [0.1, 0.2]

    tiled_a_bbox, tiled_t_bbox = merge(anchors_xywh, target_bbox_xywh)

    g_cxcy = tiled_t_bbox[:, :, :2] - tiled_a_bbox[:, :, :2]
    g_cxcy = g_cxcy / (variances[0] * tiled_a_bbox[:, :, 2:])

    g_wh = tiled_t_bbox[:, :, 2:] / tiled_a_bbox[:, :, 2:]
    g_wh = tf.math.log(g_wh) / variances[1]

    return tf.concat([g_cxcy, g_wh], axis=-1)


def np_tf_linear_sum_assignment(matrix):

    indices = linear_sum_assignment(matrix)
    target_indices = indices[0]
    pred_indices = indices[1]

    # print(matrix.shape, target_indices, pred_indices)

    target_selector = np.zeros(matrix.shape[0])
    target_selector[target_indices] = 1
    target_selector = target_selector.astype(np.bool)

    pred_selector = np.zeros(matrix.shape[1])
    pred_selector[pred_indices] = 1
    pred_selector = pred_selector.astype(np.bool)

    # print('target_indices', target_indices)
    # print("pred_indices", pred_indices)

    return [target_indices, pred_indices, target_selector, pred_selector]


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()

    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def loss_labels(outputs, targets, indices, num_boxes, log=True):
    """Classification loss (NLL)
targets dicts must contain the key "labels" containing a tensor of dim
[nb_target_boxes]
    """
    assert "pred_logits" in outputs
    src_logits = outputs["pred_logits"]

    idx = _get_src_permutation_idx(indices)
    target_classes_o = torch.cat(
        [t["labels"][J] for t, (_, J) in zip(targets, indices)]
    )
    target_classes = torch.full(
        src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
    )
    target_classes[idx] = target_classes_o

    empty_weight = torch.ones(81)
    empty_weight[0] = 0.1

    # print("log_softmax(input, 1)", F.softmax(src_logits, 1).mean())
    # print("src_logits", src_logits.shape)
    # print("target_classes", target_classes, target_classes.shape)

    # print("target_classes", target_classes)

    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
    # print('>loss_ce', loss_ce)
    losses = {"loss_ce": loss_ce}

    # if log:
    #    # TODO this should probably be a separate loss, not hacked in this one here
    #    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
    return losses


def loss_boxes(outputs, targets, indices, num_boxes):
"""Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU
loss
targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes,
4]
The target boxes are expected in format (center_x, center_y, w, h), normalized by the
image size.
    """
    # print("------")
    # assert 'pred_boxes' in outputs
    idx = _get_src_permutation_idx(indices)
    src_boxes = outputs["pred_boxes"][idx]
    target_boxes = torch.cat(
        [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
    )

    # print("target_boxes", target_boxes)
    # print("src_boxes", src_boxes)

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
    # print("loss_bbox", loss_bbox)
    losses = {}
    losses["loss_bbox"] = loss_bbox.sum() / target_boxes.shape[0]
    # print(">loss_bbox", losses['loss_bbox'])

    loss_giou = 1 - torch.diag(
        generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
        )
    )
    # print('>loss_giou', loss_giou)
    losses["loss_giou"] = loss_giou.sum() / target_boxes.shape[0]
    # print(">loss_giou", losses['loss_giou'])
    return losses


def hungarian_matching(
    t_bbox,
    t_class,
    p_bbox,
    p_class,
    fcost_class=1,
    fcost_bbox=5,
    fcost_giou=2,
    slice_preds=True,
) -> tuple:

    if slice_preds:
        size = tf.cast(t_bbox[0][0], tf.int32)
        t_bbox = tf.slice(t_bbox, [1, 0], [size, 4])
        t_class = tf.slice(t_class, [1, 0], [size, -1])
        t_class = tf.squeeze(t_class, axis=-1)

    # Convert frpm [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    p_bbox_xy = xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = xcycwh_to_xy_min_xy_max(t_bbox)

    softmax = tf.nn.softmax(p_class)

    # Classification cost for the Hungarian algorithom
    # On each prediction. We select the prob of the expected class
    cost_class = -tf.gather(softmax, t_class, axis=1)
    # print(cost_class)

    # L1 cost for the hungarian algorithm
    _p_bbox, _t_bbox = merge(p_bbox, t_bbox)
    cost_bbox = tf.norm(_p_bbox - _t_bbox, ord=1, axis=-1)

    # Generalized IOU
    iou, union = jaccard(p_bbox_xy, t_bbox_xy, return_union=True)
    _p_bbox_xy, _t_bbox_xy = merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:, :, :2], _t_bbox_xy[:, :, :2])
    bottom_right = tf.math.maximum(_p_bbox_xy[:, :, 2:], _t_bbox_xy[:, :, 2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:, :, 0] * size[:, :, 1]
    cost_giou = -(iou - (area - union) / area)

    # Final hungarian cost matrix
    cost_matrix = (
        fcost_bbox * cost_bbox + fcost_class * cost_class + fcost_giou * cost_giou
    )

    selectors = tf.numpy_function(
        np_tf_linear_sum_assignment,
        [cost_matrix],
        [tf.int64, tf.int64, tf.bool, tf.bool],
    )
    target_indices = selectors[0]
    pred_indices = selectors[1]
    target_selector = selectors[2]
    pred_selector = selectors[3]

    return pred_indices, target_indices, pred_selector, target_selector, t_bbox, t_class


def get_total_losss(losses):
    """
    Get model total losss including auxiliary loss
    """
    train_loss = ["label_cost", "giou_loss", "l1_loss"]
    loss_weights = [1, 2, 5]

    total_loss = 0
    for key in losses:
        selector = [l for l, loss_name in enumerate(train_loss) if loss_name in key]
        if len(selector) == 1:
            # print("Add to the total loss", key, losses[key], loss_weights[selector[0]])
            total_loss += losses[key] * loss_weights[selector[0]]
    return total_loss


def get_losses(m_outputs, t_bbox, t_class, config):
    losses = get_detr_losses(m_outputs, t_bbox, t_class, config)

    # Get auxiliary loss for each auxiliary output
    if "aux" in m_outputs:
        for a, aux_m_outputs in enumerate(m_outputs["aux"]):
            aux_losses = get_detr_losses(
                aux_m_outputs, t_bbox, t_class, config, suffix="_{}".format(a)
            )
            losses.update(aux_losses)

    # Compute the total loss
    total_loss = get_total_losss(losses)

    return total_loss, losses


def loss_labels(
    p_bbox,
    p_class,
    t_bbox,
    t_class,
    t_indices,
    p_indices,
    t_selector,
    p_selector,
    background_class=0,
):

    neg_indices = tf.squeeze(tf.where(p_selector == False), axis=-1)
    neg_p_class = tf.gather(p_class, neg_indices)
    neg_t_class = tf.zeros((tf.shape(neg_p_class)[0],), tf.int64) + background_class

    neg_weights = tf.zeros((tf.shape(neg_indices)[0],)) + 0.1
    pos_weights = tf.zeros((tf.shape(t_indices)[0],)) + 1.0
    weights = tf.concat([neg_weights, pos_weights], axis=0)

    pos_p_class = tf.gather(p_class, p_indices)
    pos_t_class = tf.gather(t_class, t_indices)

    #############
    # Metrics
    #############
    # True negative
    cls_neg_p_class = tf.argmax(neg_p_class, axis=-1)
    true_neg = tf.reduce_mean(tf.cast(cls_neg_p_class == background_class, tf.float32))
    # True positive
    cls_pos_p_class = tf.argmax(pos_p_class, axis=-1)
    true_pos = tf.reduce_mean(tf.cast(cls_pos_p_class != background_class, tf.float32))
    # True accuracy
    cls_pos_p_class = tf.argmax(pos_p_class, axis=-1)
    pos_accuracy = tf.reduce_mean(tf.cast(cls_pos_p_class == pos_t_class, tf.float32))

    targets = tf.concat([neg_t_class, pos_t_class], axis=0)
    preds = tf.concat([neg_p_class, pos_p_class], axis=0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(targets, preds)
    loss = tf.reduce_sum(loss * weights) / tf.reduce_sum(weights)

    return loss, true_neg, true_pos, pos_accuracy


def loss_boxes(
    p_bbox, p_class, t_bbox, t_class, t_indices, p_indices, t_selector, p_selector
):
    # print("------")
    p_bbox = tf.gather(p_bbox, p_indices)
    t_bbox = tf.gather(t_bbox, t_indices)

    p_bbox_xy = xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = xcycwh_to_xy_min_xy_max(t_bbox)

    l1_loss = tf.abs(p_bbox - t_bbox)
    l1_loss = tf.reduce_sum(l1_loss) / tf.cast(tf.shape(p_bbox)[0], tf.float32)

    iou, union = jaccard(p_bbox_xy, t_bbox_xy, return_union=True)

    _p_bbox_xy, _t_bbox_xy = merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:, :, :2], _t_bbox_xy[:, :, :2])
    bottom_right = tf.math.maximum(_p_bbox_xy[:, :, 2:], _t_bbox_xy[:, :, 2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:, :, 0] * size[:, :, 1]
    giou = iou - (area - union) / area
    loss_giou = 1 - tf.linalg.diag_part(giou)

    loss_giou = tf.reduce_sum(loss_giou) / tf.cast(tf.shape(p_bbox)[0], tf.float32)

    return loss_giou, l1_loss


def get_detr_losses(m_outputs, target_bbox, target_label, config, suffix=""):

    predicted_bbox = m_outputs["pred_boxes"]
    predicted_label = m_outputs["pred_logits"]

    all_target_bbox = []
    all_target_class = []
    all_predicted_bbox = []
    all_predicted_class = []
    all_target_indices = []
    all_predcted_indices = []
    all_target_selector = []
    all_predcted_selector = []

    t_offset = 0
    p_offset = 0

    for b in range(predicted_bbox.shape[0]):

        p_bbox, p_class, t_bbox, t_class = (
            predicted_bbox[b],
            predicted_label[b],
            target_bbox[b],
            target_label[b],
        )
        (
            t_indices,
            p_indices,
            t_selector,
            p_selector,
            t_bbox,
            t_class,
        ) = hungarian_matching(t_bbox, t_class, p_bbox, p_class, slice_preds=True)

        t_indices = t_indices + tf.cast(t_offset, tf.int64)
        p_indices = p_indices + tf.cast(p_offset, tf.int64)

        all_target_bbox.append(t_bbox)
        all_target_class.append(t_class)
        all_predicted_bbox.append(p_bbox)
        all_predicted_class.append(p_class)
        all_target_indices.append(t_indices)
        all_predcted_indices.append(p_indices)
        all_target_selector.append(t_selector)
        all_predcted_selector.append(p_selector)

        t_offset += tf.shape(t_bbox)[0]
        p_offset += tf.shape(p_bbox)[0]

    all_target_bbox = tf.concat(all_target_bbox, axis=0)
    all_target_class = tf.concat(all_target_class, axis=0)
    all_predicted_bbox = tf.concat(all_predicted_bbox, axis=0)
    all_predicted_class = tf.concat(all_predicted_class, axis=0)
    all_target_indices = tf.concat(all_target_indices, axis=0)
    all_predcted_indices = tf.concat(all_predcted_indices, axis=0)
    all_target_selector = tf.concat(all_target_selector, axis=0)
    all_predcted_selector = tf.concat(all_predcted_selector, axis=0)

    label_cost, true_neg, true_pos, pos_accuracy = loss_labels(
        all_predicted_bbox,
        all_predicted_class,
        all_target_bbox,
        all_target_class,
        all_target_indices,
        all_predcted_indices,
        all_target_selector,
        all_predcted_selector,
        background_class=config.background_class,
    )

    giou_loss, l1_loss = loss_boxes(
        all_predicted_bbox,
        all_predicted_class,
        all_target_bbox,
        all_target_class,
        all_target_indices,
        all_predcted_indices,
        all_target_selector,
        all_predcted_selector,
    )

    label_cost = label_cost
    giou_loss = giou_loss
    l1_loss = l1_loss

    return {
        "label_cost{}".format(suffix): label_cost,
        "true_neg{}".format(suffix): true_neg,
        "true_pos{}".format(suffix): true_pos,
        "pos_accuracy{}".format(suffix): pos_accuracy,
        "giou_loss{}".format(suffix): giou_loss,
        "l1_loss{}".format(suffix): l1_loss,
    }


"""
    This scripts is used to send training logs to Wandb.
"""
from typing import Union, Dict, Tuple
import tensorflow as tf
import numpy as np

try:
    # Should be optional
    import wandb
except:
    wandb = None

import cv2


class WandbSender(object):
    """
    Class used within the Yolact project to send data to Wandb to
    log experiments.
    """

    IOU_THRESHOLDS = [x / 100.0 for x in range(50, 100, 5)]
    AP_DATA = None
    NB_CLASS = None

    def __init__(self):
        self.init_buffer()

    @staticmethod
    def init_ap_data(nb_class=None):
        """Init the ap data used to compute the Map metrics.
        If nb_class is not provided, used the last provided nb_class.
        """
        if nb_class is not None:
            WandbSender.NB_CLASS = nb_class

        if WandbSender.NB_CLASS is None:
            raise ValueError("NB_CLASS is not sed in WandbSender")

        if WandbSender.AP_DATA is None:
            WandbSender.AP_DATA = {
                "box": [
                    [
                        APDataObject()
                        for _ in [f"class_{i}" for i in range(WandbSender.NB_CLASS)]
                    ]
                    for _ in [x / 100.0 for x in range(50, 100, 5)]
                ],
                "mask": [
                    [
                        APDataObject()
                        for _ in [f"class_{i}" for i in range(WandbSender.NB_CLASS)]
                    ]
                    for _ in [x / 100.0 for x in range(50, 100, 5)]
                ],
            }

    def init_buffer(self):
        """Init list used to store the information from a batch of data.
        Onced the list is filled, the send method
        send all images online.
        """
        self.images = []
        self.queries = []
        self.images_mask_ground_truth = []
        self.images_mask_prediction = []
        self.p_labels_batch = []
        self.t_labels_batch = []
        self.batch_mAP = []

    @staticmethod
    @tf.autograph.experimental.do_not_convert()
    def compute_map(
        p_bbox: np.array,
        p_labels: np.array,
        p_scores: np.array,
        t_bbox: np.array,
        t_labels: np.array,
        b: int,
        batch: int,
        prefix: str,
        step: int,
        send: bool,
        p_mask: np.array,
        t_mask: np.array,
    ):
        """
For some reason, autograph is trying to understand what I'm doing here. With some
failure.
Thus, @tf.autograph.experimental.do_not_convert() is used to prevent autograph to scan
this method.
        Args:
            p_bbox/t_bbox: List of bbox (n, 4) [y1, x2, y2, x2]
            p_labels/t_labels: List of labels index (n)
p_mask/t_mask: predicted/target mask (n, h, w) with h and w the size of the
mask
            p_scores: List of predicted scores (n)
            b: Batch indice
            batch: size of a batch
            prefix: Prefix to use to log something on wandb
            step: Step number
            send: Whether to send the result of all computed map to wandb.
        """

        # Init Ap Data
        if WandbSender.AP_DATA is None:
            WandbSender.init_ap_data()

# Set fake class name. (we do not really need the real name of each class at this
point)
        class_name = [f"class_{i}" for i in range(WandbSender.NB_CLASS)]

        try:
            # Compyute
            cal_map(
                p_bbox,
                p_labels,
                p_scores,
                p_mask,
                t_bbox,
                t_labels,
                t_mask,
                WandbSender.AP_DATA,
                WandbSender.IOU_THRESHOLDS,
            )

            # Last element of the validation set.

            if send and b + 1 == batch:

                all_maps = calc_map(
                    WandbSender.AP_DATA,
                    WandbSender.IOU_THRESHOLDS,
                    class_name,
                    print_result=True,
                )
                wandb.log(
                    {
                        f"val/map50_bbox": all_maps["box"][50],
                        f"val/map50_mask": all_maps["mask"][50],
                        f"val/map_bbox": all_maps["box"]["all"],
                        f"val/map_mask": all_maps["mask"]["all"],
                    },
                    step=step,
                )
                wandb.run.summary.update(
                    {
                        f"val/map50_bbox": all_maps["box"][50],
                        f"val/map50_mask": all_maps["mask"][50],
                        f"val/map_bbox": all_maps["box"]["all"],
                        f"val/map_mask": all_maps["mask"]["all"],
                    }
                )

                WandbSender.AP_DATA = None
                WandbSender.init_ap_data()

            return np.array([0.0, 0.0], np.float64)

        except Exception as e:
            print("compute_map error. e=", e)
            # raise e
            return np.array([0.0, 0.0], np.float64)
        return np.array([0.0, 0.0], np.float64)

    def get_wandb_bbox_mask_image(
        self,
        image: np.array,
        bbox: np.array,
        labels: np.array,
        masks=None,
        scores=None,
        class_name=[],
    ) -> Tuple[list, np.array]:
        """
        Serialize the model inference into a dict and an image ready to be send to wandb.
        Args:
            image: (550, 550, 3)
            bbox: List of bbox (n, 4) [x1, y2, x2, y2]
            labels: List of labels index (n)
            masks: predicted/target mask (n, h, w) with h and w the size of the mask
            scores: List of predicted scores (n) (Optional)
            class_name; List of class name for each label
        Return:
           A dict with the box data for wandb
           and a copy of the  original image with the instance masks
        """
        height, width = image.shape[0], image.shape[1]
        image_mask = np.copy(image)
        instance_id = 1
        box_data = []

        for b in range(len(bbox)):
            # Sample a new color for the mask instance
            instance_color = np.random.uniform(0, 1, (3))
            # Retrive bbox coordinates
            x1, y1, x2, y2 = bbox[b]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

            # Fill the mask
            if masks is not None:
                mask = masks[:, :, b]
                mask = cv2.resize(mask, (width, height))
                mask = mask[
                    int(y1 * height) : int(y2 * height),
                    int(x1 * width) : int(x2 * width),
                ]
                image_mask[
                    int(y1 * height) : int(y2 * height),
                    int(x1 * width) : int(x2 * width),
                ][mask > 0.5] = (
                    0.5
                    * image[
                        int(y1 * height) : int(y2 * height),
                        int(x1 * width) : int(x2 * width),
                    ][mask > 0.5]
                    + 0.5 * instance_color
                )

            image_mask = cv2.rectangle(
                image_mask,
                (int(x1 * width), int(y1 * height)),
                (int(x2 * width), int(y2 * height)),
                (1, 1, 0),
                3,
            )

            # if scores is None:
            box_caption = "%s" % (class_name[int(labels[b])])
            # else:
#    box_caption = "%s-{:.2f}" % (class_name[int(labels[b])],
float(scores[b]))

            box_dict = {
                "position": {"minX": x1, "maxX": x2, "minY": y1, "maxY": y2},
                "class_id": int(labels[b]),
                "box_caption": box_caption,
            }
# b < len(scores) for some reason sometime scores is not of the same length than the
bbox
            if scores is not None and b < len(scores):
                box_dict["scores"] = {"conf": float(scores[b])}
            # print("append", box_dict)
            box_data.append(box_dict)
            instance_id += 1

        return box_data, image_mask

    def gather_inference(
        self,
        image: np.array,
        p_bbox: np.array,
        p_scores: np.array,
        t_bbox: np.array,
        p_labels: np.array,
        t_labels: np.array,
        p_masks=None,
        t_masks=None,
        class_name=[],
    ):
        self.class_name = class_name

        # This is what wandb expext to get as input to display images with bbox.
        boxes = {"ground_truth": {"box_data": []}, "predictions": {"box_data": []}}

        # Ground Truth
        box_data, _ = self.get_wandb_bbox_mask_image(
            image, t_bbox, t_labels, t_masks, class_name=class_name, scores=p_scores
        )
        boxes["ground_truth"]["box_data"] = box_data
        boxes["ground_truth"]["class_labels"] = {
            _id: str(label) for _id, label in enumerate(class_name)
        }

        # Predictions
        box_data, _ = self.get_wandb_bbox_mask_image(
            image, p_bbox, p_labels, p_masks, class_name=class_name, scores=p_scores
        )
        boxes["predictions"]["box_data"] = box_data
        boxes["predictions"]["class_labels"] = {
            _id: str(label) for _id, label in enumerate(class_name)
        }

        # Append the target and the predictions to the buffer
        self.images.append(wandb.Image(image, boxes=boxes))

        return np.array(0, dtype=np.int64)

    @tf.autograph.experimental.do_not_convert()
    def send(self, step: tf.Tensor, prefix=""):
        """
For some reason, autograph is trying to understand what I'm doing here. With some
failure.
Thus, @tf.autograph.experimental.do_not_convert() is used to prevent autograph to scan
this method.
        Send the buffer to wandb
        Args:
            step: The global training step as eager tensor
            prefix: Prefix used before each log name.
        """
        step = int(step)

        wandb.log({f"{prefix}Images bbox": self.images}, step=step)

        if len(self.batch_mAP) > 0:
            wandb.log({f"{prefix}mAp": np.mean(self.batch_mAP)}, step=step)

        self.init_buffer()

        return np.array(0, dtype=np.int64)

    @staticmethod
    @tf.autograph.experimental.do_not_convert()
    def send_depth(depth_map, step: np.array, prefix=""):
        """
For some reason, autograph is trying to understand what I'm doing here. With some
failure.
Thus, @tf.autograph.experimental.do_not_convert() is used to prevent autograph to scan
this method.
        Send the depth map to wandb
        Args:
           depth_map: (8, h, w, 1) Depth images used to train the model
           step: The global training step as eager tensor
           prefix: Prefix used before each log name
        """
        step = int(step)
        depth_map_images = []
        for depth in depth_map:
            depth_map_images.append(wandb.Image(depth))
        wandb.log({f"{prefix}Depth map": depth_map_images}, step=step)
        return np.array(0, dtype=np.int64)

    @staticmethod
    @tf.autograph.experimental.do_not_convert()
    def send_proto_sample(
        proto_map: np.array,
        proto_sample: np.array,
        proto_targets: np.array,
        step: np.array,
        prefix="",
    ):
        """
For some reason, autograph is trying to understand what I'm doing here. With some
failure.
Thus, @tf.autograph.experimental.do_not_convert() is used to prevent autograph to scan
this method.
        Send the proto images logs to wandb.
        Args:
           proto_map: The k (32 by default) proto map of the proto network (h, w, k)
proto_sample: Some generated mask from the network for a batch (n, h, w) with n the
number of mask
proto_targets: The target mask for each generated mask. (n, h, w) with n the number of
mask
           step: The global training step as eager tensor
           prefix: Prefix used before each log name
        """
        step = int(step)

        proto_map_images = []
        proto_sample_images = []
        proto_targets_images = []

        for p in range(proto_map.shape[-1]):
            proto_map_images.append(
                wandb.Image(np.clip(proto_map[:, :, p] * 100, 0, 255))
            )
        for p in range(len(proto_sample)):
            proto_sample_images.append(wandb.Image(proto_sample[p]))
            proto_targets_images.append(wandb.Image(proto_targets[p]))

        wandb.log({f"{prefix}Proto Map": proto_map_images}, step=step)
        wandb.log(
            {f"{prefix}Instance segmentation prediction": proto_sample_images},
            step=step,
        )
        wandb.log(
            {f"{prefix}Instance segmentation target": proto_targets_images}, step=step
        )
        return np.array(0, dtype=np.int64)

    @staticmethod
    @tf.autograph.experimental.do_not_convert()
    def send_images(
        images,
        step: np.array,
        name: str,
        captions=None,
        masks_prediction=None,
        masks_target=None,
    ):
        """
For some reason, autograph is trying to understand what I'm doing here. With some
failure.
Thus, @tf.autograph.experimental.do_not_convert() is used to prevent autograph to scan
this method.
        Send some images to wandb
        Args:
           images: (8, h, w, c) Images to log in wandb
           step: The global training step as eager tensor
           name: Image names
        """
        class_labels = {
            0: "background",
            1: "0",
            2: "1",
            3: "2",
            4: "3",
            5: "4",
            6: "5",
            7: "6",
            8: "7",
            9: "8",
            10: "9",
        }

        step = int(step)
        images_list = []
        for i, img in enumerate(images):
            img_params = {}
            if captions is not None:
                img_params["caption"] = captions[i]

            if masks_prediction is not None:
                mask_pred = cv2.resize(
                    masks_prediction[i],
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_pred = mask_pred.astype(np.int32)
                if "masks" not in img_params:
                    img_params["masks"] = {}

                # seg = np.expand_dims(masks[i].astype(np.int32), axis=-1)
                img_params["masks"]["predictions"] = {
                    "mask_data": mask_pred,
                    "class_labels": class_labels,
                }

            if masks_target is not None:
                if "masks" not in img_params:
                    img_params["masks"] = {}

                mask_target = masks_target[i].astype(np.int32)
                # seg = np.expand_dims(masks[i].astype(np.int32), axis=-1)
                print(mask_target.shape)
                img_params["masks"]["groud_truth"] = {
                    "mask_data": mask_target,
                    "class_labels": class_labels,
                }

            images_list.append(wandb.Image(img, **img_params))

        wandb.log({name: images_list}, step=step)
        return np.array(0, dtype=np.int64)


from typing import Union, Dict, Tuple
import tensorflow as tf


import numpy as np
import cv2


if int(tf.__version__.split(".")[1]) >= 4:
    RAGGED = True
else:
    RAGGED = False


def tf_send_batch_log_to_wandb(
    images,
    target_bbox,
    target_class,
    m_outputs: dict,
    config,
    class_name=[],
    step=None,
    prefix="",
):

# Warning: In graph mode, this class is init only once. In eager mode, this class is init
at each step.
    img_sender = WandbSender()

    predicted_bbox = m_outputs["pred_boxes"]
    for b in range(predicted_bbox.shape[0]):
        # Select within the batch the elements at indice b
        image = images[b]

        elem_m_outputs = {
            key: m_outputs[key][b : b + 1]
            if (m_outputs[key] is not None and not isinstance(m_outputs[key], list))
            else m_outputs[key]
            for key in m_outputs
        }

        # Target
        t_bbox, t_class = target_bbox[b], target_class[b]

        if not RAGGED:
            size = tf.cast(t_bbox[0][0], tf.int32)
            t_bbox = tf.slice(t_bbox, [1, 0], [size, 4])
            t_bbox = xcycwh_to_xy_min_xy_max(t_bbox)
            t_class = tf.slice(t_class, [1, 0], [size, -1])
            t_class = tf.squeeze(t_class, axis=-1)

        # Predictions
        predicted_bbox, predicted_labels, predicted_scores = get_model_inference(
            elem_m_outputs, config.background_class, bbox_format="xyxy"
        )

        np_func_params = {
            "image": image,
            "p_bbox": np.array(predicted_bbox),
            "p_scores": np.array(predicted_scores),
            "t_bbox": np.array(t_bbox),
            "p_labels": np.array(predicted_labels),
            "t_labels": np.array(t_class),
            "class_name": class_name,
        }
        img_sender.gather_inference(**np_func_params)

    img_sender.send(step=step, prefix=prefix)


def compute_map_on_batch(
    images,
    target_bbox,
    target_class,
    m_outputs: dict,
    config,
    class_name=[],
    step=None,
    send=True,
    prefix="",
):
    predicted_bbox = m_outputs["pred_boxes"]
    batch_size = predicted_bbox.shape[0]
    for b in range(batch_size):

        image = images[b]
        elem_m_outputs = {
            key: m_outputs[key][b : b + 1]
            if (m_outputs[key] is not None and not isinstance(m_outputs[key], list))
            else m_outputs[key]
            for key in m_outputs
        }

        # Target
        t_bbox, t_class = target_bbox[b], target_class[b]

        if not RAGGED:
            size = tf.cast(t_bbox[0][0], tf.int32)
            t_bbox = tf.slice(t_bbox, [1, 0], [size, 4])
            t_bbox = xcycwh_to_yx_min_yx_max(t_bbox)
            t_class = tf.slice(t_class, [1, 0], [size, -1])
            t_class = tf.squeeze(t_class, axis=-1)

        # Inference ops
        predicted_bbox, predicted_labels, predicted_scores = get_model_inference(
            elem_m_outputs, config.background_class, bbox_format="yxyx"
        )
        pred_mask = None

        pred_mask = np.zeros((138, 138, len(predicted_bbox)))
        target_mask = np.zeros((138, 138, len(t_bbox)))
        WandbSender.compute_map(
            np.array(predicted_bbox),
            np.array(predicted_labels),
            np.array(predicted_scores),
            np.array(t_bbox),
            np.array(t_class),
            b,
            batch_size,
            prefix,
            step,
            send,
            pred_mask,
            target_mask,
        )


def train_log(
    images,
    t_bbox,
    t_class,
    m_outputs: dict,
    config,
    step,
    class_name=[],
    prefix="train/",
):
    # Every 1000 steps, log some progress of the training
    # (Images with bbox and images logs)
    if step % 100 == 0:
        tf_send_batch_log_to_wandb(
            images,
            t_bbox,
            t_class,
            m_outputs,
            config,
            class_name=class_name,
            step=step,
            prefix=prefix,
        )


def valid_log(
    images,
    t_bbox,
    t_class,
    m_outputs: dict,
    config,
    step,
    global_step,
    class_name=[],
    evaluation_step=200,
    prefix="train/",
):

    # Set the number of class
    WandbSender.init_ap_data(nb_class=len(class_name))
    map_list = compute_map_on_batch(
        images,
        t_bbox,
        t_class,
        m_outputs,
        config,
        class_name=class_name,
        step=global_step,
        send=(step + 1 == evaluation_step),
        prefix="val/",
    )

    if step == 0:
        tf_send_batch_log_to_wandb(
            images,
            t_bbox,
            t_class,
            m_outputs,
            config,
            class_name=class_name,
            step=global_step,
            prefix="val/",
        )


"""shell
!pip install wandb
"""

import time
import wandb


@tf.function
def run_train_step(model, images, t_bbox, t_class, optimizers, config):

    if config.target_batch is not None:
        gradient_aggregate = int(config.target_batch // config.batch_size)
    else:
        gradient_aggregate = 1

    with tf.GradientTape() as tape:
        m_outputs = model(images, training=True)
        total_loss, log = get_losses(m_outputs, t_bbox, t_class, config)
        total_loss = total_loss / gradient_aggregate

    # Compute gradient for each part of the network
    gradient_steps = gather_gradient(model, optimizers, total_loss, tape, config, log)

    return m_outputs, total_loss, log, gradient_steps


@tf.function
def run_val_step(model, images, t_bbox, t_class, config):
    m_outputs = model(images, training=False)
    total_loss, log = get_losses(m_outputs, t_bbox, t_class, config)
    return m_outputs, total_loss, log


def fit(model, train_dt, optimizers, config, epoch_nb, class_names):
    """Train the model for one epoch"""
    # Aggregate the gradient for bigger batch and better convergence
    gradient_aggregate = None
    if config.target_batch is not None:
        gradient_aggregate = int(config.target_batch // config.batch_size)
    t = None
    for epoch_step, (images, t_bbox, t_class) in enumerate(train_dt):

        # Run the prediction and retrieve the gradient step for each part of the network
        m_outputs, total_loss, log, gradient_steps = run_train_step(
            model, images, t_bbox, t_class, optimizers, config
        )

        # Load the predictions
        if config.log:
            train_log(
                images,
                t_bbox,
                t_class,
                m_outputs,
                config,
                config.global_step,
                class_names,
                prefix="train/",
            )

        # Aggregate and apply the gradient
        for name in gradient_steps:
            aggregate_grad_and_apply(
                name, optimizers, gradient_steps[name]["gradients"], epoch_step, config
            )

        # Log every 100 steps
        if epoch_step % 100 == 0:
            t = t if t is not None else time.time()
            elapsed = time.time() - t
            print(
f"Epoch: [{epoch_nb}], \t Step: [{epoch_step}], \t ce: [{log['label_cost']:.2f}] \t giou
: [{log['giou_loss']:.2f}] \t l1 : [{log['l1_loss']:.2f}] \t time : [{elapsed:.2f}]"
            )
            if config.log:
                wandb.log({f"train/{k}": log[k] for k in log}, step=config.global_step)
            t = time.time()

        config.global_step += 1


def eval(model, valid_dt, config, class_name, evaluation_step=200):
    """Evaluate the model on the validation set"""
    t = None
    for val_step, (images, t_bbox, t_class) in enumerate(valid_dt):
        # Run prediction
        m_outputs, total_loss, log = run_val_step(
            model, images, t_bbox, t_class, config
        )
        # Log the predictions
        if config.log:
            valid_log(
                images,
                t_bbox,
                t_class,
                m_outputs,
                config,
                val_step,
                config.global_step,
                class_name,
                evaluation_step=evaluation_step,
                prefix="train/",
            )
        # Log the metrics
        if config.log and val_step == 0:
            wandb.log({f"val/{k}": log[k] for k in log}, step=config.global_step)
        # Log the progress
        if val_step % 10 == 0:
            t = t if t is not None else time.time()
            elapsed = time.time() - t
            print(
f"Validation step: [{val_step}], \t ce: [{log['label_cost']:.2f}] \t giou :
[{log['giou_loss']:.2f}] \t l1 : [{log['l1_loss']:.2f}] \t time : [{elapsed:.2f}]"
            )
        if val_step + 1 >= evaluation_step:
            break


"""
Train the last layers on top of the transformer for one epoch


*   ce is the cross entropy loss of the layer that predicts the bbox class

*   giou and l1 loss are the positional loss of the layer that predicts the bbox pos
(center_x, center_y, width, height)


"""

fit(detr, train_iterator, optimzers, config, epoch_nb=0, class_names=class_names)

CLASS_COLOR_MAP = np.random.randint(0, 255, (100, 3))


def numpy_bbox_to_image(
    image, bbox_list, labels=None, scores=None, class_name=[], config=None
):
    """Numpy function used to display the bbox (target or prediction)"""
    assert (
        image.dtype == np.float32
        and image.dtype == np.float32
        and len(image.shape) == 3
    )

    if config is not None and config.normalized_method == "torch_resnet":
        channel_avg = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        image = (image * channel_std) + channel_avg
        image = (image * 255).astype(np.uint8)
    elif config is not None and config.normalized_method == "tf_resnet":
        image = image + mean
        image = image[..., ::-1]
        image = image / 255

    bbox_xcycwh = np_rescale_bbox_xcycwh(bbox_list, (image.shape[0], image.shape[1]))
    bbox_x1y1x2y2 = np_xcycwh_to_xy_min_xy_max(bbox_xcycwh)

    # Set the labels if not defined
    if labels is None:
        labels = np.zeros((bbox_x1y1x2y2.shape[0]))

    bbox_area = []
    # Go through each bbox
    for b in range(0, bbox_x1y1x2y2.shape[0]):
        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        bbox_area.append((x2 - x1) * (y2 - y1))

    # Go through each bbox
    for b in np.argsort(bbox_area)[::-1]:
        # Take a new color at reandon for this instance
        instance_color = np.random.randint(0, 255, (3))

        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1, x2, y2 = (
            max(0, x1),
            max(0, y1),
            min(image.shape[1], x2),
            min(image.shape[0], y2),
        )

        # Select the class associated with this bbox
        class_id = labels[int(b)]

        if scores is not None and len(scores) > 0:
            label_name = class_name[int(class_id)]
            label_name = "%s:%.2f" % (label_name, scores[b])
        else:
            label_name = class_name[int(class_id)]

        class_color = CLASS_COLOR_MAP[int(class_id)]

        color = instance_color

        multiplier = image.shape[0] / 500
        cv2.rectangle(
            image,
            (x1, y1),
            (x1 + int(multiplier * 15) * len(label_name), y1 + 20),
            class_color.tolist(),
            -10,
        )
        cv2.putText(
            image,
            label_name,
            (x1 + 2, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * multiplier,
            (0, 0, 0),
            1,
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), tuple(class_color.tolist()), 2)

    return image


def get_model_inference(m_outputs: dict, background_class, bbox_format="xy_center"):

    predicted_bbox = m_outputs["pred_boxes"][0]
    predicted_labels = m_outputs["pred_logits"][0]

    softmax = tf.nn.softmax(predicted_labels)
    predicted_scores = tf.reduce_max(softmax, axis=-1)
    predicted_labels = tf.argmax(softmax, axis=-1)

    indices = tf.where(predicted_labels != background_class)
    indices = tf.squeeze(indices, axis=-1)

    predicted_scores = tf.gather(predicted_scores, indices)
    predicted_labels = tf.gather(predicted_labels, indices)
    predicted_bbox = tf.gather(predicted_bbox, indices)

    if bbox_format == "xy_center":
        predicted_bbox = predicted_bbox
    elif bbox_format == "xyxy":
        predicted_bbox = xcycwh_to_xy_min_xy_max(predicted_bbox)
    elif bbox_format == "yxyx":
        predicted_bbox = xcycwh_to_yx_min_yx_max(predicted_bbox)
    else:
        raise NotImplementedError()

    return predicted_bbox, predicted_labels, predicted_scores


"""
### The training results
"""

for valid_images, target_bbox, target_class in valid_iterator:

    m_outputs = detr(valid_images, training=False)
    predicted_bbox, predicted_labels, predicted_scores = get_model_inference(
        m_outputs, config.background_class, bbox_format="xy_center"
    )

    result = numpy_bbox_to_image(
        np.array(valid_images[0]),
        np.array(predicted_bbox),
        np.array(predicted_labels),
        scores=np.array(predicted_scores),
        class_name=class_names,
        config=config,
    )
    plt.imshow(result)
    plt.show()
    break

"""
## References

[1] https://github.com/facebookresearch/detr

[2] https://arxiv.org/pdf/2005.12872.pdf

[3] https://github.com/Visual-Behavior/detr-tensorflow
"""

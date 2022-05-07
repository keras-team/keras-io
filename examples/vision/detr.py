"""
Title: DETR : End-to-End Object Detection with Transformers
Author: [Ayyuce Demirbas](https://twitter.com/demirbasayyuce)
Date created: 2022/05/07
Last modified: 2022/05/07
Description: TensorFlow implementation of [End-to-End Object Detection with Transformers paper](https://arxiv.org/pdf/2005.12872.pdf)
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
# Download the dataset
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
import pandas as pd


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


class TrainingConfig:
    def __init__(self):
        self.data_dir, self.img_dir, self.ann_dir, self.ann_file = (
            None,
            None,
            None,
            None,
        )
        self.data = DataConfig(data_dir=None, img_dir=None, ann_file=None, ann_dir=None)
        self.background_class = 0
        self.image_size = 376, 672
        self.train_backbone = False
        self.train_transformers = False
        self.train_nlayers = False
        self.finetuning = False
        self.batch_size = 1
        self.gradient_norm_clipping = 0.1
        self.target_batch = 1
        self.backbone_lr = tf.Variable(1e-5)
        self.transformers_lr = tf.Variable(1e-4)
        self.nlayers_lr = tf.Variable(1e-4)
        self.nlayers = []
        self.global_step = 0
        self.log = False
        self.normalized_method = "torch_resnet"

    def add_nlayers(self, layers):
        self.nlayers = [l.name for l in layers]


from typing import Union, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2


def intersect(box_a: tf.Tensor, box_b: tf.Tensor) -> tf.Tensor:
    A = tf.shape(box_a)[0]  # Number of possible bbox
    B = tf.shape(box_b)[0]  # Number of anchors
    tiled_box_a_xymax = tf.tile(tf.expand_dims(box_a[:, 2:], axis=1), [1, B, 1])
    tiled_box_b_xymax = tf.tile(tf.expand_dims(box_b[:, 2:], axis=0), [A, 1, 1])
    above_right_corner = tf.math.minimum(tiled_box_a_xymax, tiled_box_b_xymax)
    tiled_box_a_xymin = tf.tile(tf.expand_dims(box_a[:, :2], axis=1), [1, B, 1])
    tiled_box_b_xymin = tf.tile(tf.expand_dims(box_b[:, :2], axis=0), [A, 1, 1])
    upper_left_corner = tf.math.maximum(tiled_box_a_xymin, tiled_box_b_xymin)
    inter = tf.nn.relu(above_right_corner - upper_left_corner)
    inter = inter[:, :, 0] * inter[:, :, 1]
    return inter


def jaccard(box_a: tf.Tensor, box_b: tf.Tensor, return_union=False) -> tf.Tensor:
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_a = tf.tile(tf.expand_dims(area_a, axis=-1), [1, tf.shape(inter)[-1]])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    area_b = tf.tile(tf.expand_dims(area_b, axis=-2), [tf.shape(inter)[-2], 1])
    union = area_a + area_b - inter
    if return_union is False:
        return inter / union
    else:
        return inter / union, union


def merge(box_a: tf.Tensor, box_b: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    A = tf.shape(box_a)[0]  # Number of bbox in box_a
    B = tf.shape(box_b)[0]  # Number of bbox in box b
    tiled_box_a = tf.tile(tf.expand_dims(box_a, axis=1), [1, B, 1])
    tiled_box_b = tf.tile(tf.expand_dims(box_b, axis=0), [A, 1, 1])
    return tiled_box_a, tiled_box_b


def xy_min_xy_max_to_xcycwh(bbox: tf.Tensor) -> tf.Tensor:
    bbox_xcycwh = tf.concat(
        [bbox[:, :2] + ((bbox[:, 2:] - bbox[:, :2]) / 2), bbox[:, 2:] - bbox[:, :2]],
        axis=-1,
    )
    return bbox_xcycwh


def xcycwh_to_xy_min_xy_max(bbox: tf.Tensor) -> tf.Tensor:
    bbox_xyxy = tf.concat(
        [bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1
    )
    bbox_xyxy = tf.clip_by_value(bbox_xyxy, 0.0, 1.0)
    return bbox_xyxy


def xcycwh_to_yx_min_yx_max(bbox: tf.Tensor) -> tf.Tensor:
    bbox = xcycwh_to_xy_min_xy_max(bbox)
    bbox = tf.concat([bbox[:, 1:2], bbox[:, 0:1], bbox[:, 3:4], bbox[:, 2:3]], axis=-1)
    return bbox


def xy_min_xy_max_to_xcycwh(bbox: np.array) -> np.array:
    bbox_xcycwh = np.concatenate(
        [bbox[:, :2] + ((bbox[:, 2:] - bbox[:, :2]) / 2), bbox[:, 2:] - bbox[:, :2]],
        axis=-1,
    )
    return bbox_xcycwh


def np_xcycwh_to_xy_min_xy_max(bbox: np.array) -> np.array:
    bbox_xy = np.concatenate(
        [bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1
    )
    return bbox_xy


def np_rescale_bbox_xcycwh(bbox_xcycwh: np.array, img_size: tuple):
    bbox_xcycwh = np.array(bbox_xcycwh)  # Be sure to work with a numpy array
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    bbox_xcycwh_rescaled = bbox_xcycwh * scale
    return bbox_xcycwh_rescaled


CLASS_COLOR_MAP = np.random.randint(0, 255, (100, 3))


def numpy_bbox_to_image(
    image, bbox_list, labels=None, scores=None, class_name=[], config=None
):
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
    if labels is None:
        labels = np.zeros((bbox_x1y1x2y2.shape[0]))
    bbox_area = []
    for b in range(0, bbox_x1y1x2y2.shape[0]):
        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        bbox_area.append((x2 - x1) * (y2 - y1))
    for b in np.argsort(bbox_area)[::-1]:
        instance_color = np.random.randint(0, 255, (3))
        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1, x2, y2 = (
            max(0, x1),
            max(0, y1),
            min(image.shape[1], x2),
            min(image.shape[0], y2),
        )
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


"""shell
!pip install imgaug==0.4.0
"""

import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def bbox_xcyc_wh_to_imgaug_bbox(bbox, target_class, height, width):
    img_aug_bbox = []
    for b in range(0, len(bbox)):
        bbox_xcyc_wh = bbox[b]
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
                            iaa.Resize(
                                {"width": image_size[1], "height": image_size[0]},
                                interpolation=ia.ALL,
                            ),
                            iaa.CropToFixedSize(image_size[1], image_size[0]),
                            iaa.Affine(
                                scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                            ),
                        ]
                    )
                ),
                iaa.Resize(
                    {"width": image_size[1], "height": image_size[0]},
                    interpolation=ia.ALL,
                ),
            ],
            random_order=False,
        )
        return seq
    else:
        seq = iaa.Sequential(
            [iaa.Resize({"width": image_size[1], "height": image_size[0]})],
            random_order=False,
        )
        return seq
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
    bbox_xcyc_wh = np.array(bbox_xcyc_wh)
    return bbox_xcyc_wh, t_class


def retrieve_outputs(augmented_images, augmented_bbox):
    outputs_dict = {}
    image_shape = None
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


def normalized_images(image, config):
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
    def _np_function(_idx):
        return fc(_idx, **params)

    return tf.numpy_function(_np_function, [idx], outputs_types)


def pad_labels(images: tf.Tensor, t_bbox: tf.Tensor, t_class: tf.Tensor):
    nb_bbox = tf.shape(t_bbox)[0]
    bbox_header = tf.expand_dims(nb_bbox, axis=0)
    bbox_header = tf.expand_dims(bbox_header, axis=0)
    bbox_header = tf.pad(bbox_header, [[0, 0], [0, 3]])
    bbox_header = tf.cast(bbox_header, tf.float32)
    cls_header = tf.constant([[0]], dtype=tf.int64)
    t_bbox = tf.pad(
        t_bbox, [[0, 100 - 1 - nb_bbox], [0, 0]], mode="CONSTANT", constant_values=0
    )
    t_class = tf.pad(
        t_class, [[0, 100 - 1 - nb_bbox], [0, 0]], mode="CONSTANT", constant_values=0
    )
    t_bbox = tf.concat([bbox_header, t_bbox], axis=0)
    t_class = tf.concat([cls_header, t_class], axis=0)
    return images, t_bbox, t_class


CLASS_NAMES = ["background", "head", "helmet", "person"]

"""
# Fine-tuning
"""

"""
Create a new training config:
"""

from os.path import expanduser


class CustomConfig(TrainingConfig):
    def __init__(self):
        super().__init__()
        self.datadir = os.path.join(expanduser("."), "data/hardhat/")
        self.image_size = (480, 720)
        self.batch_size = 1
        self.target_batch = 8


config = CustomConfig()


def load_hardhat_data_from_index(
    index, filenames, train_val, anns, config, augmentation
):
    image = imageio.imread(
        os.path.join(config.datadir, f"{train_val}", filenames[index])
    )
    image_anns = anns[anns["filename"] == filenames[index]]
    t_class = image_anns["class"].map(lambda x: CLASS_NAMES.index(x)).to_numpy()
    width = image_anns["width"].to_numpy()
    height = image_anns["height"].to_numpy()
    bbox_list = image_anns[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    bbox_list = bbox_list / [width[0], height[0], width[0], height[0]]
    t_bbox = xy_min_xy_max_to_xcycwh(bbox_list)
    image, t_bbox, t_class = detr_transform(
        image, t_bbox, t_class, config, augmentation=augmentation
    )
    image = normalized_images(image, config)

    return (
        image.astype(np.float32),
        t_bbox.astype(np.float32),
        np.expand_dims(t_class, axis=-1),
    )


from random import shuffle
import pandas as pd
import imageio


def load_hardhat(train_val, batch_size, config, augmentation=False):
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
    dataset = dataset.filter(lambda imgs, tbbox, tclass: tf.shape(tbbox)[0] > 0)
    dataset = dataset.map(pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset, CLASS_NAMES


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
            avg_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, avg_weights
        return attn_output


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
                r = requests.get(f, allow_redirects=True)
                open(os.path.join(wdir, fname), "wb").write(r.content)
        l = model.load_weights(os.path.join(wdir, f"{weights}.ckpt"))
        l.expect_partial()
    else:
        raise Exception(f"Cant load the weights: {weights}")


"""
### DETR Model
"""

import pickle
import time
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
    detr = DETR(
        num_decoder_layers=num_decoder_layers, num_encoder_layers=num_encoder_layers
    )
    if weights is not None:
        load_weights(detr, weights)
    image_input = tf.keras.Input((None, None, 3))
    if not tf_backbone:
        backbone = detr.get_layer("backbone")
    else:
        config.normalized_method = "tf_resnet"
        backbone = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=(None, None, 3)
        )
    transformer = detr.get_layer("transformer")
    position_embedding_sine = detr.get_layer("position_embedding_sine")
    input_proj = detr.get_layer("input_proj")
    query_embed = detr.get_layer("query_embed")
    class_embed = detr.get_layer("class_embed")
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
    backbone_variables = get_backbone_trainable_variables(model)
    transformers_variables = get_transformers_trainable_variables(
        model, exclude=config.nlayers
    )
    nlayers_variables = get_nlayers_trainables_variables(model, config.nlayers)
    return backbone_variables, transformers_variables, nlayers_variables


def setup_optimizers(model, config):
    @tf.function
    def get_backbone_learning_rate():
        return config.backbone_lr

    @tf.function
    def get_transformers_learning_rate():
        return config.transformers_lr

    @tf.function
    def get_nlayers_learning_rate():
        return config.nlayers_lr

    disable_batchnorm_training(model)
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
        if gradient_aggregate is not None and step % gradient_aggregate == 0:
            optimizers[gradient_name] = [
                tf.zeros_like(tv) for tv in optimizers[variables_name]
            ]
        if gradient_aggregate is not None:
            optimizers[gradient_name] = [
                (gradient + n_gradient) if n_gradient is not None else None
                for gradient, n_gradient in zip(optimizers[gradient_name], gradients)
            ]
        else:
            optimizers[gradient_name] = gradients
        if gradient_aggregate is None or (step + 1) % gradient_aggregate == 0:
            optimizers[optimizer_name].apply_gradients(
                zip(optimizers[gradient_name], optimizers[variables_name])
            )


optimzers = setup_optimizers(detr, config)

from scipy.optimize import linear_sum_assignment


def np_tf_linear_sum_assignment(matrix):
    indices = linear_sum_assignment(matrix)
    target_indices = indices[0]
    pred_indices = indices[1]
    target_selector = np.zeros(matrix.shape[0])
    target_selector[target_indices] = 1
    target_selector = target_selector.astype(np.bool)
    pred_selector = np.zeros(matrix.shape[1])
    pred_selector[pred_indices] = 1
    pred_selector = pred_selector.astype(np.bool)
    return [target_indices, pred_indices, target_selector, pred_selector]


def _get_src_permutation_idx(indices):
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


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
    p_bbox_xy = xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = xcycwh_to_xy_min_xy_max(t_bbox)
    softmax = tf.nn.softmax(p_class)
    cost_class = -tf.gather(softmax, t_class, axis=1)
    _p_bbox, _t_bbox = merge(p_bbox, t_bbox)
    cost_bbox = tf.norm(_p_bbox - _t_bbox, ord=1, axis=-1)
    iou, union = jaccard(p_bbox_xy, t_bbox_xy, return_union=True)
    _p_bbox_xy, _t_bbox_xy = merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:, :, :2], _t_bbox_xy[:, :, :2])
    bottom_right = tf.math.maximum(_p_bbox_xy[:, :, 2:], _t_bbox_xy[:, :, 2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:, :, 0] * size[:, :, 1]
    cost_giou = -(iou - (area - union) / area)
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
    train_loss = ["label_cost", "giou_loss", "l1_loss"]
    loss_weights = [1, 2, 5]
    total_loss = 0
    for key in losses:
        selector = [l for l, loss_name in enumerate(train_loss) if loss_name in key]
        if len(selector) == 1:
            total_loss += losses[key] * loss_weights[selector[0]]
    return total_loss


def get_losses(m_outputs, t_bbox, t_class, config):
    losses = get_detr_losses(m_outputs, t_bbox, t_class, config)
    if "aux" in m_outputs:
        for a, aux_m_outputs in enumerate(m_outputs["aux"]):
            aux_losses = get_detr_losses(
                aux_m_outputs, t_bbox, t_class, config, suffix="_{}".format(a)
            )
            losses.update(aux_losses)
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
    cls_neg_p_class = tf.argmax(neg_p_class, axis=-1)
    true_neg = tf.reduce_mean(tf.cast(cls_neg_p_class == background_class, tf.float32))
    cls_pos_p_class = tf.argmax(pos_p_class, axis=-1)
    true_pos = tf.reduce_mean(tf.cast(cls_pos_p_class != background_class, tf.float32))
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
    gradient_steps = gather_gradient(model, optimizers, total_loss, tape, config, log)
    return m_outputs, total_loss, log, gradient_steps


@tf.function
def run_val_step(model, images, t_bbox, t_class, config):
    m_outputs = model(images, training=False)
    total_loss, log = get_losses(m_outputs, t_bbox, t_class, config)
    return m_outputs, total_loss, log


def fit(model, train_dt, optimizers, config, epoch_nb, class_names):
    gradient_aggregate = None
    if config.target_batch is not None:
        gradient_aggregate = int(config.target_batch // config.batch_size)
    t = None
    for epoch_step, (images, t_bbox, t_class) in enumerate(train_dt):
        m_outputs, total_loss, log, gradient_steps = run_train_step(
            model, images, t_bbox, t_class, optimizers, config
        )
        for name in gradient_steps:
            aggregate_grad_and_apply(
                name, optimizers, gradient_steps[name]["gradients"], epoch_step, config
            )
        if epoch_step % 100 == 0:
            t = t if t is not None else time.time()
            elapsed = time.time() - t
            print(
f"Epoch: [{epoch_nb}], \t Step: [{epoch_step}], \t ce: [{log['label_cost']:.2f}] \t giou
: [{log['giou_loss']:.2f}] \t l1 : [{log['l1_loss']:.2f}] \t time : [{elapsed:.2f}]"
            )
            t = time.time()
        config.global_step += 1


def eval(model, valid_dt, config, class_name, evaluation_step=200):
    t = None
    for val_step, (images, t_bbox, t_class) in enumerate(valid_dt):
        m_outputs, total_loss, log = run_val_step(
            model, images, t_bbox, t_class, config
        )
        if val_step % 10 == 0:
            t = t if t is not None else time.time()
            elapsed = time.time() - t
            print(
f"Validation step: [{val_step}], \t ce: [{log['label_cost']:.2f}] \t giou :
[{log['giou_loss']:.2f}] \t l1 : [{log['l1_loss']:.2f}] \t time : [{elapsed:.2f}]"
            )
        if val_step + 1 >= evaluation_step:
            break


fit(detr, train_iterator, optimzers, config, epoch_nb=0, class_names=class_names)

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

"""
Title: Object Detection with Yolo v3
Author: [Parsa Mazaheri](https://twitter.com/ParsaMazaheri)
Date created: 2020/05/21
Last modified: 2020/05/21
Description: Using Yolo algorithm for object detection.
"""

"""
## Introduction
"Yolo (You only look once!) is a real-time object detection algorithm with high accuracy.
The algorithm applies a single neural network to the full image, and then divides the
image into
bounding boxes(regions) and predicts probabilities for each region.
This example shows how to do object detection and localization using pre-trained weights.
**Refrences:**
- based on [keras-yolo3](https://github.com/experiencor/keras-yolo3) project
- Yolo [Original Paper](https://arxiv.org/abs/1804.02767)

"""

"""
## Setup

"""

import cv2
import numpy as np
from tensorflow.keras import backend as K
from keras import layers
from keras.models import Model
from keras.utils import get_file
import struct


"""
## Getting model weights and config
Download weights and config file from yolo site and coco dataset labels for labeling
objects

"""

"""shell
!wget https://pjreddie.com/media/files/yolov3.weights
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
!wget
https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt

"""

"""
## Load network structure
We're going to extract network structure form config to build the model

"""


def load_config(cfg):
    file = open(cfg, "r")
    # getting rid of comments spaces and empty lines
    lines = file.read().split("\n")
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != "#"]
    lines = [x.rstrip().lstrip() for x in lines]
    block, blocks = {}, []
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    # reading labels
    f = open("coco-labels-2014_2017.txt", "r")
    labels = f.read().split("\n")
    # the first and last block are networks hype-parameters
    return {**blocks[0], **blocks[-1]}, blocks[1:], labels


# hyper-parameters / net / labels
hyperParam, blocks, labels = load_config("yolov3.cfg")
# threshold / anchors
thresh = 0.5
anchors = [
    [116, 90, 156, 198, 373, 326],
    [30, 61, 62, 45, 59, 119],
    [10, 13, 16, 30, 33, 23],
]


"""
## Build the model
blocks are divided into 4 types [`convolutional`, `route`, `shortcut`, `upsample`] and
the last
block is the yolo hyper-parameters for network structure.

"""


def create_model(blocks):
    prev_layer = input_layer = layers.Input(shape=(None, None, 3))
    all_layers, out_index = [], []
    # for addressing and assign weights to layers later in code
    i = 0
    for b in blocks:
        if b["type"] == "convolutional":
            # Add the convolutional layer
            if int(b["stride"]) > 1:
                prev_layer = layers.ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)
            conv = layers.Conv2D(
                int(b["filters"]),
                int(b["size"]),
                strides=int(b["stride"]),
                name="conv_" + str(i),
                padding="same"
                if int(b["pad"]) == 1 and int(b["stride"]) == 1
                else "valid",
                use_bias=not "batch_normalize" in b,
            )(prev_layer)
            # Add Batch Norm Layer
            if "batch_normalize" in b:
                conv = layers.BatchNormalization(name="bnorm_" + str(i))(conv)
            prev_layer = conv
            if b["activation"] == "leaky":
                prev_layer = layers.LeakyReLU(alpha=0.1, name="leaky_" + str(i),)(
                    prev_layer
                )
            all_layers.append(prev_layer)
        elif b["type"] == "route":
            ids = [int(i) for i in b["layers"].split(",")]
            _layers = [all_layers[i] for i in ids]
            if len(_layers) > 1:
                # Concatenate layer
                temp_layer = layers.Concatenate()(_layers)
            else:
                # Skip layer
                temp_layer = _layers[0]
            # add to layers and update prev_layer
            all_layers.append(temp_layer)
            prev_layer = temp_layer
        elif b["type"] == "shortcut":
            index, activation = int(b["from"]), b["activation"]
            all_layers.append(layers.Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]
        elif b["type"] == "upsample":
            stride = int(b["stride"])
            all_layers.append(layers.UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]
        elif b["type"] == "yolo":
            out_index.append(len(all_layers) - 1)
            all_layers.append(None)
            prev_layer = all_layers[-1]
        i += 1
    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    model.summary()
    return model


# make the yolov3 model
yolov3 = create_model(blocks)


"""
## Load weights
We're going to use struct to unpack weights for assigning it to model

"""


class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, "rb") as w_f:
            (major,) = struct.unpack("i", w_f.read(4))
            (minor,) = struct.unpack("i", w_f.read(4))
            (revision,) = struct.unpack("i", w_f.read(4))
            if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)
            transpose = (major > 1000) or (minor > 1000)
            binary = w_f.read()
        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype="float32")

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size : self.offset]

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer("conv_" + str(i))
                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer("bnorm_" + str(i))
                    size = np.prod(norm_layer.get_weights()[0].shape)
                    beta = self.read_bytes(size)  # bias
                    gamma = self.read_bytes(size)  # scale
                    mean = self.read_bytes(size)  # mean
                    var = self.read_bytes(size)  # variance
                    weights = norm_layer.set_weights([gamma, beta, mean, var])
                if len(conv_layer.get_weights()) > 1:
                    bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(
                        list(reversed(conv_layer.get_weights()[0].shape))
                    )
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(
                        list(reversed(conv_layer.get_weights()[0].shape))
                    )
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                pass


# load trained weights
weight_reader = WeightReader("yolov3.weights")
weight_reader.load_weights(yolov3)


"""
## Preprocess data
keras takes inputs as batch (here we have a batch of size of 1) we also have to store
original
image size for later to draw object boxes on it

"""


# scale image and turn it to a batch of 1 images
def preprocess_input(image, shape):
    new_h, new_w, _ = image.shape
    net_h, net_w = shape[0], shape[1]
    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h
    new_h, new_w = int(new_h), int(new_w)
    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1] / 255.0, (int(new_w), int(new_h)))
    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[
        int((net_h - new_h) // 2) : int((net_h + new_h) // 2),
        int((net_w - new_w) // 2) : int((net_w + new_w) // 2),
        :,
    ] = resized
    new_image = np.expand_dims(new_image, 0)
    return new_image


shape = 416, 416
# preprocess the image
# delete next 2 lines and add your own image
img_path = get_file(
    "nyc.jpg", "https://unsplash.com/photos/j0Deh-kkkFo/download?force=true&w=640"
)
image = cv2.imread(img_path)
image_h, image_w = image.shape[0], image.shape[1]
new_image = preprocess_input(image, shape)


"""
Bound box for objects

"""

# for stroing detected objects
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return "{:.2f}".format(self.score)


def decode_netout(netout, anchors, thresh, shape):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > thresh
    for i in range(grid_h * grid_w):
        row, col = i / grid_w, i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            # objectness = netout[..., :4]
            if objectness.all() <= thresh:
                continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / shape[1]  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / shape[0]  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(
                x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes
            )
            boxes.append(box)
    return boxes


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


# sigmoid function
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Intersection over Union
def iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


# box pos correction
def correct_yolo_boxes(boxes, image_h, image_w, shape):
    if (float(shape[1]) / image_w) < (float(shape[0]) / image_h):
        new_w = shape[1]
        new_h = (image_h * shape[1]) / image_w
    else:
        new_h = shape[1]
        new_w = (image_w * shape[0]) / image_h
    for i in range(len(boxes)):
        x_offset, x_scale = (shape[1] - new_w) / 2.0 / shape[1], float(new_w) / shape[1]
        y_offset, y_scale = (shape[0] - new_h) / 2.0 / shape[0], float(new_h) / shape[0]
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


# Non-max suppression (detecting each object once)
def non_max_suppression(boxes, thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0:
                continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if iou(boxes[index_i], boxes[index_j]) >= thresh:
                    boxes[index_j].classes[c] = 0


"""
Evaluate the model

"""

# run the prediction
yolos = yolov3.predict(new_image)
boxes = []
for i in range(len(yolos)):
    # decode the output of the network
    boxes += decode_netout(yolos[i][0], anchors[i], thresh, shape)
# correct the sizes of the bounding boxes
correct_yolo_boxes(boxes, image_h, image_w, shape)
# non max suppression (detecting each object once)
non_max_suppression(boxes, thresh)


"""
## Draw detected objects on image

"""

for box in boxes:
    label_str, label = "", -1
    # print detected objects
    for i in range(len(labels)):
        if box.classes[i] > thresh:
            label_str += labels[i]
            label = i
            print("{} : {:.3f}%".format(labels[i], box.classes[i] * 100))
    # draw detected objects
    if label >= 0:
        cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 2)
        cv2.putText(
            image,
            label_str + " " + str(box.get_score()),
            (box.xmin, box.ymin - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            1e-3 * image.shape[0],
            (0, 255, 0),
            2,
        )
# save and show the image
cv2.imwrite("detected.jpg", image.astype("uint8"))
cv2.imshow("Detected in image", image)
cv2.waitKey()

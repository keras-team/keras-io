"""
Title: Using KerasCV COCO Metrics
Author: [lukewood](https://lukewood.xyz)
Date created: 2022/04/13
Last modified: 2022/04/13
Description: Use KerasCV COCO metrics to evaluate object detection models.
"""

"""
## Overview

With KerasCV's COCO metrics implementation, you can easily evaluate your object
detection model's performance all from within the TensorFlow graph. This guide
shows you how to use KerasCV's COCO metrics and integrate it into your own model
evaluation pipeline.  Historically, users have evaluated COCO metrics as a post training
step.  KerasCV offers an in graph implementation of COCO metrics, enabling users to
evaluate COCO metrics *during* training!

Let's get started using KerasCV's COCO metrics.
"""

"""
## Input Format

KerasCV COCO metrics require a specific input format.

The metrics expect `y_true` and be a `float` Tensor with the shape `[batch,
num_images, num_boxes, 5]`. The final axis stores the locational and class
information for each specific bounding box. The dimensions in order are: `[left,
top, right, bottom, class]`.

The metrics expect `y_pred` and be a `float` Tensor with the shape `[batch,
num_images, num_boxes, 56]`. The final axis stores the locational and class
information for each specific bounding box. The dimensions in order are: `[left,
top, right, bottom, class, confidence]`.

Due to the fact that each image may have a different number of bounding boxes,
the `num_boxes` dimension may actually have a mismatching shape between images.
KerasCV works around this by allowing you to either pass a `RaggedTensor` as an
input to the KerasCV COCO metrics, or padding unused bounding boxes with `-1`.

Utility functions to manipulate bounding boxes, transform between formats, and
pad bounding box Tensors with `-1s` are available at
[`keras_cv.utils.bounding_box`](https://github.com/keras-team/keras-cv/blob/master/keras_cv/utils/bounding_box.py).

"""

"""
## Independent Metric Use

The usage first pattern for KerasCV COCO metrics is to manually call
`update_state()` and `result()` methods. This pattern is recommended for users
who want finer grained control of their metric evaluation, or want to use a
different format for `y_pred` in their model.

Let's run through a quick code example.
"""

"""
1.) First, we must construct our metric:
"""

# import all modules we will need in this example
import tensorflow as tf
import keras_cv

# only consider boxes with areas less than a 32x32 square.
metric = keras_cv.metrics.COCORecall(class_ids=[1, 2, 3], area_range=(0, 32 ** 2))

"""
2.) Create Some Bounding Boxes:
"""

y_true = tf.ragged.stack(
    [
        # image 1
        tf.constant([
            [0, 0, 10, 10, 1],
            [11, 12, 30, 30, 2]
        ], tf.float32),
        # image 2
        tf.constant([
            [0, 0, 10, 10, 1]
        ], tf.float32),
    ]
)
y_pred = tf.ragged.stack(
    [
        # predictions for image 1
        tf.constant([
            [5, 5, 10, 10, 1, 0.9]
        ], tf.float32),
        # predictions for image 2
        tf.constant([
            [0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.9]
        ], tf.float32),
    ]
)

"""
3.) Update metric state:
"""

metric.update_state(y_true, y_pred)

"""
4.) Evaluate the result:
"""

metric.result()

"""
Evaluating COCORecall for your object detection model is as simple as that!
"""

"""
## Metric Use in a Model

You can also leverage COCORecall in your model's training loop.  Let's walk through this
process.

1.) Construct your the metric and a dummy model
"""

import tensorflow as tf
from tensorflow import keras
import keras_cv

i = keras.layers.Input((None, None, 6))
model = keras.Model(i, i)

"""
2.) Create some fake bounding boxes:
"""

y_true = tf.constant([
    [
      [0, 0, 10, 10, 1],
      [5, 5, 10, 10, 1]
    ]
], tf.float32)
y_pred = tf.constant([
    [
        [0, 0, 10, 10, 1, 1.0],
        [5, 5, 10, 10, 1, 0.9]
    ]
], tf.float32)

"""
3.) Create the metric and compile the model
"""

recall = keras_cv.metrics.COCORecall(
    max_detections=100,
    class_ids=[1],
    area_range=(0, 64**2),
    name='coco_recall'
)
model.compile(metrics=[recall])

"""
4.) Use `model.evaluate()` to evaluate the metric
"""

model.evaluate(y_pred, y_true, return_dict=True)

"""
Looks great!  That's all it takes to use KerasCV's COCO metrics to evaluate object
detection models.
"""

"""
## Supported Constructor Parameters

KerasCV COCO Metrics are sufficiently parameterized to support all of the
permutations evaluated in the original COCO challenge, all metrics evaluated in
the accompanying `pycocotools` library, and more!

### COCORecall

The COCORecall constructor supports the following parameters

| Name            | Usage                                                      |
| --------------- | ---------------------------------------------------------- |
| iou\_thresholds | iou\_thresholds expects an iterable. This value is used as |
:                 : a cutoff to determine the minimum intersection of unions   :
:                 : required for a classification sample to be considered a    :
:                 : true positive. If an iterable is passed, the result is the :
:                 : average across IoU values passed in the                    :
:                 : iterable.<br>Defaults to `range(0.5, 0.95, incr=0.05)`     :
| area\_range     | area\_range specifies a range over which to evaluate the   |
:                 : metric. Only ground truth objects within the area\_range   :
:                 : are considered in the scoring.<br>Defaults to\: `\[0,      :
:                 : 1e5\*\*2\]`                                                :
| max\_detections | max\_detections is a value specifying the max number of    |
:                 : detections a model is allowed to make.<br>Defaults to\:    :
:                 : `100`                                                      :
| class\_ids      | When class\_ids is not None, the metric will only consider |
:                 : boxes of the matching class label. This is useful when a   :
:                 : specific class is considered high priority. An example of  :
:                 : this would be providing the human and animal class indices :
:                 : in the case of self driving cars.<br>To evaluate all       :
:                 : categories, users will pass `range(0, num\_classes)`.      :

### COCOMeanAveragePrecision

The COCOMeanAveragePrecision constructor supports the following parameters

| Name               | Usage                                                   |
| ------------------ | ------------------------------------------------------- |
| \*\*kwargs         | Passed to COCOBase.super()                              |
| recall\_thresholds | recall\_thresholds is a list containing the             |
:                    : recall\_thresholds over which to consider in the        :
:                    : computation of MeanAveragePrecision.                    :
| iou\_thresholds    | iou\_thresholds expects an iterable. This value is used |
:                    : as a cutoff to determine the minimum intersection of    :
:                    : unions required for a classification sample to be       :
:                    : considered a true positive. If an iterable is passed,   :
:                    : the result is the average across IoU values passed in   :
:                    : the iterable.<br>Defaults to `range(0.5, 0.95,          :
:                    : incr=0.05)`                                             :
| area\_range        | area\_range specifies a range over which to evaluate    |
:                    : the metric. Only ground truth objects within the        :
:                    : area\_range are considered in the                       :
:                    : scoring.<br><br>Defaults to\: `\[0, 1e5\*\*2\]`         :
| max\_detections    | max\_detections is a value specifying the max number of |
:                    : detections a model is allowed to make.<br><br>Defaults  :
:                    : to\: `100`                                              :
| class\_ids         | When class\_ids is not None, the metric will only       |
:                    : consider boxes of the matching class label. This is     :
:                    : useful when a specific class is considered high         :
:                    : priority. An example of this would be providing the     :
:                    : human and animal class indices in the case of self      :
:                    : driving cars.<br>To evaluate all categories, users will :
:                    : pass `range(0, num\_classes)`.                          :
"""

"""
## Conclusion & Next Steps
KerasCV makes it easier than ever before to evaluate a Keras object detection model.
Historically, users had to perform post training evaluation.  With KerasCV, you can
perform train time evaluation to see how these metrics evolve over time!

As an additional exercise for readers, you can:

- Configure `iou_thresholds`, `max_detections`, and `area_range` to reproduce the suite
of metrics evaluted in `pycocotools`
- Integrate COCO metrics into a RetinaNet using the
[keras.io RetinaNet example](https://keras.io/examples/vision/retinanet/)
"""

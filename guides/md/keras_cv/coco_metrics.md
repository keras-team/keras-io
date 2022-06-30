# Using KerasCV COCO Metrics

**Author:** [lukewood](https://lukewood.xyz)<br>
**Date created:** 2022/04/13<br>
**Last modified:** 2022/04/13<br>
**Description:** Use KerasCV COCO metrics to evaluate object detection models.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_cv/coco_metrics.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/coco_metrics.py)



---
## Overview

With KerasCV's COCO metrics implementation, you can easily evaluate your object
detection model's performance all from within the TensorFlow graph. This guide
shows you how to use KerasCV's COCO metrics and integrate it into your own model
evaluation pipeline.  Historically, users have evaluated COCO metrics as a post training
step.  KerasCV offers an in graph implementation of COCO metrics, enabling users to
evaluate COCO metrics *during* training!

Let's get started using KerasCV's COCO metrics.

---
## Input format

All KerasCV components that process bounding boxes, including COCO metrics, require a
`bounding_box_format` parameter.  This parameter is used to tell the components what
format your bounding boxes are in.  While this guide uses the `xyxy` format, a full
list of supported formats is available in
[the bounding_box API documentation](https://keras.io/api/keras_cv/bounding_box/formats/).

The metrics expect `y_true` and be a `float` Tensor with the shape `[batch,
num_images, num_boxes, 5]`, with the ordering of last set of axes determined by the
provided format.  The same is true of `y_pred`, except that an additional `confidence`
axis must be provided.

Due to the fact that each image may have a different number of bounding boxes,
the `num_boxes` dimension may actually have a mismatching shape between images.
KerasCV works around this by allowing you to either pass a `RaggedTensor` as an
input to the KerasCV COCO metrics, or padding unused bounding boxes with `-1`.

Utility functions to manipulate bounding boxes, transform between formats, and
pad bounding box Tensors with `-1s` are available from the
[`keras_cv.bounding_box`](https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box)
package.

---
## Independent metric use

The usage first pattern for KerasCV COCO metrics is to manually call
`update_state()` and `result()` methods. This pattern is recommended for users
who want finer grained control of their metric evaluation, or want to use a
different format for `y_pred` in their model.

Let's run through a quick code example.

1.) First, we must construct our metric:


```python
import keras_cv

# import all modules we will need in this example
import tensorflow as tf
from tensorflow import keras

# only consider boxes with areas less than a 32x32 square.
metric = keras_cv.metrics.COCORecall(
    bounding_box_format="xyxy", class_ids=[1, 2, 3], area_range=(0, 32**2)
)
```

2.) Create Some Bounding Boxes:


```python
y_true = tf.ragged.stack(
    [
        # image 1
        tf.constant([[0, 0, 10, 10, 1], [11, 12, 30, 30, 2]], tf.float32),
        # image 2
        tf.constant([[0, 0, 10, 10, 1]], tf.float32),
    ]
)
y_pred = tf.ragged.stack(
    [
        # predictions for image 1
        tf.constant([[5, 5, 10, 10, 1, 0.9]], tf.float32),
        # predictions for image 2
        tf.constant([[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.9]], tf.float32),
    ]
)
```

3.) Update metric state:


```python
metric.update_state(y_true, y_pred)
```

4.) Evaluate the result:


```python
metric.result()
```




<div class="k-default-codeblock">
```
<tf.Tensor: shape=(), dtype=float32, numpy=0.25>

```
</div>
Evaluating COCORecall for your object detection model is as simple as that!

---
## Metric use in a model

You can also leverage COCORecall in your model's training loop.  Let's walk through this
process.

1.) Construct your the metric and a dummy model


```python
i = keras.layers.Input((None, 6))
model = keras.Model(i, i)
```

2.) Create some fake bounding boxes:


```python
y_true = tf.constant([[[0, 0, 10, 10, 1], [5, 5, 10, 10, 1]]], tf.float32)
y_pred = tf.constant([[[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.9]]], tf.float32)
```

3.) Create the metric and compile the model


```python
recall = keras_cv.metrics.COCORecall(
    bounding_box_format="xyxy",
    max_detections=100,
    class_ids=[1],
    area_range=(0, 64**2),
    name="coco_recall",
)
model.compile(metrics=[recall])
```

4.) Use `model.evaluate()` to evaluate the metric


```python
model.evaluate(y_pred, y_true, return_dict=True)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 1s 1s/step - loss: 0.0000e+00 - coco_recall: 1.0000

{'loss': 0.0, 'coco_recall': 1.0}

```
</div>
Looks great!  That's all it takes to use KerasCV's COCO metrics to evaluate object
detection models.

---
## Supported constructor parameters

KerasCV COCO Metrics are sufficiently parameterized to support all of the
permutations evaluated in the original COCO challenge, all metrics evaluated in
the accompanying `pycocotools` library, and more!

Check out the full documentation for [`COCORecall`](/api/keras_cv/metrics/coco_recall/) and
[`COCOMeanAveragePrecision`](/api/keras_cv/metrics/coco_mean_average_precision/).


---
## Conclusion & next steps
KerasCV makes it easier than ever before to evaluate a Keras object detection model.
Historically, users had to perform post training evaluation.  With KerasCV, you can
perform train time evaluation to see how these metrics evolve over time!

As an additional exercise for readers, you can:

- Configure `iou_thresholds`, `max_detections`, and `area_range` to reproduce the suite
of metrics evaluted in `pycocotools`
- Integrate COCO metrics into a RetinaNet using the
[keras.io RetinaNet example](https://keras.io/examples/vision/retinanet/)

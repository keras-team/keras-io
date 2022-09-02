# Train an Object Detection Model on Pascal VOC 2007 using KerasCV

**Author:** [lukewood](https://lukewood.xyz)<br>
**Date created:** 2022/08/22<br>
**Last modified:** 2022/08/22<br>
**Description:** Use KerasCV to train a RetinaNet on Pascal VOC 2007.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_cv/retina_net_overview.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/retina_net_overview.py)



---
## Overview

KerasCV offers a complete set of APIs to train your own state-of-the-art,
production-grade object detection model.  These APIs include object detection specific
data augmentation techniques, models, and COCO metrics.

To get started, let's sort out all of our imports and define global configuration parameters.


```python
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers

import keras_cv
from keras_cv import bounding_box
import os

BATCH_SIZE = 8
EPOCHS = int(os.getenv("EPOCHS", "1"))
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint")
```

---
## Data loading

In this guide, we use the data-loading function: `keras_cv.datasets.pascal_voc.load()`.
KerasCV supports a `bounding_box_format` argument in all components that process
bounding boxes.  To match the KerasCV API style, it is recommended that when writing a
custom data loader, you also support a `bounding_box_format` argument.
This makes it clear to those invoking your data loader what format the bounding boxes
are in.

For example:

```python
train_ds, ds_info = keras_cv.datasets.pascal_voc.load(
    split='train', bounding_box_format='xywh', batch_size=8
)
```

Clearly yields bounding boxes in the format `xywh`.  You can read more about
KerasCV bounding box formats [in the API docs](https://keras.io/api/keras_cv/bounding_box/formats/).

Our data comesloaded into the format
`{"images": images, "bounding_boxes": bounding_boxes}`.  This format is supported in all
KerasCV preprocessing components.

Let's load some data and verify that our data looks as we expect it to.


```python
dataset, dataset_info = keras_cv.datasets.pascal_voc.load(
    split="train", bounding_box_format="xywh", batch_size=9
)


def visualize_dataset(dataset, bounding_box_format):
    color = tf.constant(((255.0, 0, 0),))
    plt.figure(figsize=(10, 10))
    for i, example in enumerate(dataset.take(9)):
        images, boxes = example["images"], example["bounding_boxes"]
        boxes = keras_cv.bounding_box.convert_format(
            boxes, source=bounding_box_format, target="rel_yxyx", images=images
        )
        boxes = boxes.to_tensor(default_value=-1)
        plotted_images = tf.image.draw_bounding_boxes(images, boxes[..., :4], color)
        plt.subplot(9 // 3, 9 // 3, i + 1)
        plt.imshow(plotted_images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


visualize_dataset(dataset, bounding_box_format="xywh")
```

<div class="k-default-codeblock">
```
[1mDownloading and preparing dataset 868.85 MiB (download: 868.85 MiB, generated: Unknown size, total: 868.85 MiB) to ~/tensorflow_datasets/voc/2007/4.0.0...[0m

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/3 [00:00<?, ? splits/s]

Generating test examples...:   0%|          | 0/4952 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2007/4.0.0.incompleteG8GEQ1/voc-test.tfrecord*...:   0%|          | 0/4952…

Generating train examples...:   0%|          | 0/2501 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2007/4.0.0.incompleteG8GEQ1/voc-train.tfrecord*...:   0%|          | 0/250…

Generating validation examples...:   0%|          | 0/2510 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2007/4.0.0.incompleteG8GEQ1/voc-validation.tfrecord*...:   0%|          | …

[1mDataset voc downloaded and prepared to ~/tensorflow_datasets/voc/2007/4.0.0. Subsequent calls will reuse this data.[0m

```
</div>
    
![png](../guides/img/retina_net_overview/retina_net_overview_4_12.png)
    


Looks like everything is structured as expected.  Now we can move on to constructing our
data augmentation pipeline.

---
## Data augmentation

One of the most labor-intensive tasks when constructing object detection pipelines is
data augmentation.  Image augmentation techniques must be aware of the underlying
bounding boxes, and must update them accordingly.

Luckily, KerasCV natively supports bounding box augmentation with its extensive library
of [data augmentation layers](https://keras.io/api/keras_cv/layers/preprocessing/).
The code below loads the Pascal VOC dataset, and performs on-the-fly bounding box
friendly data augmentation inside of a `tf.data` pipeline.


```python
# train_ds is batched as a (images, bounding_boxes) tuple
# bounding_boxes are ragged
train_ds, train_dataset_info = keras_cv.datasets.pascal_voc.load(
    bounding_box_format="xywh", split="train", batch_size=BATCH_SIZE
)
val_ds, val_dataset_info = keras_cv.datasets.pascal_voc.load(
    bounding_box_format="xywh", split="validation", batch_size=BATCH_SIZE
)

augmenter = keras_cv.layers.RandomChoice(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
        keras_cv.layers.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=0.1,
            contrast_factor=0.1,
            saturation_factor=0.1,
            hue_factor=0.1,
        ),
        keras_cv.layers.RandomSharpness(value_range=(0, 255), factor=0.1),
    ]
)

# train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
visualize_dataset(train_ds, bounding_box_format="xywh")
```


    
![png](../guides/img/retina_net_overview/retina_net_overview_7_0.png)
    


Great!  We now have a bounding box friendly augmentation pipeline.

Next, let's unpackage our inputs from the preprocessing dictionary, and prepare to feed
the inputs into our model.


```python

def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
```

Our data pipeline is now complete.  We can now move on to model creation and training.

---
## Model creation

We'll use the KerasCV API to construct a RetinaNet model.  In this tutorial we use
a pretrained ResNet50 backbone using weights.  In order to perform fine-tuning, we
freeze the backbone before training.  When `include_rescaling=True` is set, inputs to
the model are expected to be in the range `[0, 255]`.


```python
model = keras_cv.models.RetinaNet(
    # number of classes to be used in box classification
    classes=20,
    # For more info on supported bounding box formats, visit
    # https://keras.io/api/keras_cv/bounding_box/
    bounding_box_format="xywh",
    # KerasCV offers a set of pre-configured backbones
    backbone="resnet50",
    # Each backbone comes with multiple pre-trained weights
    # These weights match the weights available in the `keras_cv.model` class.
    backbone_weights="imagenet",
    # include_rescaling tells the model whether your input images are in the default
    # pixel range (0, 255) or if you have already rescaled your inputs to the range
    # (0, 1).  In our case, we feed our model images with inputs in the range (0, 255).
    include_rescaling=True,
    # Typically, you'll want to set this to False when training a real model.
    # evaluate_train_time_metrics=True makes `train_step()` incompatible with TPU,
    # and also causes a massive performance hit.
    evaluate_train_time_metrics=False,
)
# Fine-tuning a RetinaNet is as simple as setting backbone.trainable = False
model.backbone.trainable = False
```

That is all it takes to construct a KerasCV RetinaNet.  The RetinaNet accepts tuples of
dense image Tensors and ragged bounding box Tensors to `fit()` and `train_on_batch()`
This matches what we have constructed in our input pipeline above.

The RetinaNet `call()` method outputs two values: training targets and inference targets.
In this guide, we are primarily concerned with the inference targets.  Internally, the
training targets are used by `box_loss` and `classification_loss` to train the
network.

---
## Training our model

All that is left to do is train our model.  KerasCV object detection models follow the
standard Keras workflow, leveraging `compile()` and `fit()`.

Let's compile our model:


```python
optimizer = tf.optimizers.SGD(learning_rate=0.1, momentum=0.9, global_clipnorm=10.0)
model.compile(
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    optimizer=optimizer,
)
```

All that is left to do is construct some callbacks:


```python
callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.EarlyStopping(patience=15),
    keras.callbacks.ReduceLROnPlateau(patience=10),
    # Uncomment to train your own RetinaNet
    # keras.callbacks.ModelCheckpoint("checkpoint/", save_weights_only=True),
]
```

And run `model.fit()`!


```python
# model.fit(
#    train_ds,
#    validation_data=val_ds.take(20),
#    epochs=EPOCHS,
#    callbacks=callbacks,
# )
```

An important nuance to note is that by default the KerasCV RetinaNet does not evaluate
metrics at train time.  This is to ensure optimal GPU performance and TPU compatibility.
If you want to evaluate train time metrics, you may pass
`evaluate_train_time_metrics=True` to the `keras_cv.models.RetinaNet` constructor.

---
## Evaluation with COCO Metrics

KerasCV offers a suite of in-graph COCO metrics that support batch-wise evaluation.
More information on these metrics is available in:

- [Efficient Graph-Friendly COCO Metric Computation for Train-Time Model Evaluation](https://arxiv.org/abs/2207.12120)
- [Using KerasCV COCO Metrics](https://keras.io/guides/keras_cv/coco_metrics/)

Let's construct two COCO metrics, an instance of
`keras_cv.metrics.COCOMeanAveragePrecision` with the parameterization to match the
standard COCO Mean Average Precision metric, and `keras_cv.metrics.COCORecall`
parameterized to match the standard COCO Recall metric.


```python
metrics = [
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=range(20),
        bounding_box_format="xywh",
        name="Mean Average Precision",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(20),
        bounding_box_format="xywh",
        max_detections=100,
        name="Recall",
    ),
]

```

Next, we can evaluate the metrics by re-compiling the model, and running
`model.evaluate()`:


```python
model.load_weights(CHECKPOINT_PATH)
model.compile(
    metrics=metrics,
    box_loss=model.box_loss,
    classification_loss=model.classification_loss,
    optimizer=model.optimizer,
)
# metrics = model.evaluate(val_ds.take(20), return_dict=True)
# print(metrics)
# {"Mean Average Precision": 0.612, "Recall": 0.767}
```

---
## Inference

KerasCV makes object detection inference simple.  `model.predict(images)` returns a
RaggedTensor of bounding boxes.  By default, `RetinaNet.predict()` will perform
a non max suppression operation for you.


```python
model = keras_cv.models.RetinaNet(
    classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
)
model.load_weights(CHECKPOINT_PATH)


def visualize_detections(model):
    train_ds, val_dataset_info = keras_cv.datasets.pascal_voc.load(
        bounding_box_format="xywh", split="train", batch_size=9
    )
    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    images, labels = next(iter(train_ds.take(1)))
    predictions = model.predict(images)
    color = tf.constant(((255.0, 0, 0),))
    plt.figure(figsize=(10, 10))
    predictions = keras_cv.bounding_box.convert_format(
        predictions, source="xywh", target="rel_yxyx", images=images
    )
    predictions = predictions.to_tensor(default_value=-1)
    plotted_images = tf.image.draw_bounding_boxes(images, predictions[..., :4], color)
    for i in range(9):
        plt.subplot(9 // 3, 9 // 3, i + 1)
        plt.imshow(plotted_images[i].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


visualize_detections(model)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 20s 20s/step

```
</div>
    
![png](../guides/img/retina_net_overview/retina_net_overview_26_1.png)
    


To get good results, you should train for at least 100 epochs.  You also need to
tune the prediction decoder layer.  This can be done by passing a custom prediction
decoder to the RetinaNet constructor as follows.

Luckily, tuning a prediction decoder does not require any sort of retraining - so it
may be done iteratively.  Below is an example showing how to do this:


```python
prediction_decoder = keras_cv.layers.NmsPredictionDecoder(
    bounding_box_format="xywh",
    anchor_generator=keras_cv.models.RetinaNet.default_anchor_generator(
        bounding_box_format="xywh"
    ),
    suppression_layer=keras_cv.layers.NonMaxSuppression(
        bounding_box_format="xywh", classes=20, confidence_threshold=0.15
    ),
)
model = keras_cv.models.RetinaNet(
    classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
    prediction_decoder=prediction_decoder,
)
model.load_weights(CHECKPOINT_PATH)
visualize_detections(model)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Inconsistent references when loading the checkpoint into this object graph. For example, in the saved checkpoint object, `model.layer.weight` and `model.layer_copy.weight` reference the same variable, while in the current object these are two different variables. The referenced variables are:(<keras_cv.layers.object_detection.anchor_generator.AnchorGenerator object at 0x7fe9602dcb00> and <keras_cv.layers.object_detection.anchor_generator.AnchorGenerator object at 0x7fe9602dcc18>).

WARNING:tensorflow:Inconsistent references when loading the checkpoint into this object graph. For example, in the saved checkpoint object, `model.layer.weight` and `model.layer_copy.weight` reference the same variable, while in the current object these are two different variables. The referenced variables are:(<keras_cv.layers.object_detection.anchor_generator.AnchorGenerator object at 0x7fe9602dcb00> and <keras_cv.layers.object_detection.anchor_generator.AnchorGenerator object at 0x7fe9602dcc18>).

WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.

WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.momentum

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.momentum

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c3_1x1.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c3_1x1.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c3_1x1.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c3_1x1.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c4_1x1.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c4_1x1.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c4_1x1.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c4_1x1.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c5_1x1.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c5_1x1.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c5_1x1.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c5_1x1.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c3_3x3.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c3_3x3.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c3_3x3.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c3_3x3.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c4_3x3.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c4_3x3.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c4_3x3.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c4_3x3.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c5_3x3.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c5_3x3.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c5_3x3.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c5_3x3.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c6_3x3.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c6_3x3.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c6_3x3.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c6_3x3.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c7_3x3.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c7_3x3.kernel

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c7_3x3.bias

WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer's state 'momentum' for (root).feature_pyramid.conv_c7_3x3.bias

1/1 [==============================] - 16s 16s/step

```
</div>
    
![png](../guides/img/retina_net_overview/retina_net_overview_28_45.png)
    


---
## Results and conclusions

KerasCV makes it easy to construct state-of-the-art object detection pipelines.  All of
the KerasCV object detection components can be used independently, but also have deep
integration with each other.  With KerasCV, bounding box augmentation, train-time COCO
metrics evaluation, and more, are all made simple and consistent.

By default, this script runs for a single epoch.  To run training to convergence,
invoke the script with a command line flag `--epochs=500`.  To save you the effort of
running the script for 500 epochs, we have produced a Weights and Biases report covering
the training results below!  As a bonus, the report includes a training run with and
without data augmentation.

[Metrics from a 500 epoch Weights and Biases Run are available here](
    https://tinyurl.com/y34xx65w
)

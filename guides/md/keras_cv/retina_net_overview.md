# Train an Object Detection Model on Pascal VOC 2007 using KerasCV

**Author:** [lukewood](https://twitter.com/luke_wood_ml)<br>
**Date created:** 2022/08/22<br>
**Last modified:** 2022/08/22<br>
**Description:** Use KerasCV to train a RetinaNet on Pascal VOC 2007.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_cv/retina_net_overview.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/retina_net_overview.py)



---
## Overview

KerasCV offers a complete set of APIs to train your own state-of-the-art,
production-grade object detection model.  These APIs include object detection specific
data augmentation techniques, and batteries included object detection models.

To get started, let's sort out all of our imports and define global configuration parameters.


```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers

import keras_cv
from keras_cv import bounding_box
import os
import resource
from luketils import visualization

BATCH_SIZE = 16
EPOCHS = int(os.getenv("EPOCHS", "1"))
# To fully train a RetinaNet, comment out this line.
# EPOCHS = 50
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint/")
INFERENCE_CHECKPOINT_PATH = os.getenv("INFERENCE_CHECKPOINT_PATH", CHECKPOINT_PATH)

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
```

<div class="k-default-codeblock">
```
You do not have Waymo Open Dataset installed, so KerasCV Waymo metrics are not available.

```
</div>
---
## Data loading

KerasCV has a predefined specificication for bounding boxes.  To comply with this, you
should package your bounding boxes into a dictionary matching the speciciation below:

```
bounding_boxes = {
    # num_boxes may be a Ragged dimension
    'boxes': Tensor(shape=[batch, num_boxes, 4]),
    'classes': Tensor(shape=[batch, num_boxes])
}
```


`bounding_boxes['boxes']` contains the coordinates of your bounding box in a KerasCV
supported `bounding_box_format`.
KerasCV requires a `bounding_box_format` argument in all components that process
bounding boxes.
This is done to maximize users' ability to plug and play individual components into their
object detection components.

To match the KerasCV API style, it is recommended that when writing a
custom data loader, you also support a `bounding_box_format` argument.
This makes it clear to those invoking your data loader what format the bounding boxes
are in.

For example:

```python
train_ds, ds_info = your_data_loader.load(
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

def unpackage_tfds_inputs(inputs):
    image = inputs["image"]
    boxes = keras_cv.bounding_box.convert_format(
        inputs["objects"]["bbox"],
        images=image,
        source="rel_yxyx",
        target="xywh",
    )
    bounding_boxes = {
        "classes": tf.cast(inputs["objects"]["label"], dtype=tf.float32),
        "boxes": tf.cast(boxes, dtype=tf.float32),
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


train_ds = tfds.load(
    "voc/2007", split="train+validation", with_info=False, shuffle_files=True
)
# add pascal 2012 dataset to augment the training set
train_ds = train_ds.concatenate(
    tfds.load("voc/2012", split="train+validation", with_info=False, shuffle_files=True)
)
eval_ds = tfds.load("voc/2007", split="test", with_info=False)

train_ds = train_ds.map(unpackage_tfds_inputs, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_tfds_inputs, num_parallel_calls=tf.data.AUTOTUNE)
```

<div class="k-default-codeblock">
```
[1mDownloading and preparing dataset 868.85 MiB (download: 868.85 MiB, generated: Unknown size, total: 868.85 MiB) to ~/tensorflow_datasets/voc/2007/4.0.0...[0m

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/3 [00:00<?, ? splits/s]

Generating test examples...:   0%|          | 0/4952 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2007/4.0.0.incompleteV7B5OW/voc-test.tfrecord*...:   0%|          | 0/4952…

Generating train examples...:   0%|          | 0/2501 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2007/4.0.0.incompleteV7B5OW/voc-train.tfrecord*...:   0%|          | 0/250…

Generating validation examples...:   0%|          | 0/2510 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2007/4.0.0.incompleteV7B5OW/voc-validation.tfrecord*...:   0%|          | …

[1mDataset voc downloaded and prepared to ~/tensorflow_datasets/voc/2007/4.0.0. Subsequent calls will reuse this data.[0m
[1mDownloading and preparing dataset 3.59 GiB (download: 3.59 GiB, generated: Unknown size, total: 3.59 GiB) to ~/tensorflow_datasets/voc/2012/4.0.0...[0m

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/3 [00:00<?, ? splits/s]

Generating test examples...:   0%|          | 0/10991 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2012/4.0.0.incompleteDREP08/voc-test.tfrecord*...:   0%|          | 0/1099…

Generating train examples...:   0%|          | 0/5717 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2012/4.0.0.incompleteDREP08/voc-train.tfrecord*...:   0%|          | 0/571…

Generating validation examples...:   0%|          | 0/5823 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2012/4.0.0.incompleteDREP08/voc-validation.tfrecord*...:   0%|          | …

[1mDataset voc downloaded and prepared to ~/tensorflow_datasets/voc/2012/4.0.0. Subsequent calls will reuse this data.[0m

```
</div>
Next, lets batch our data.  In KerasCV object detection tasks it is recommended that
users use ragged batches.  This is due to the fact that images may be of different
sizes in PascalVOC and that there may be different numbers of bounding boxes per image.

The easiest way to construct a ragged dataset in a `tf.data` pipeline is to use
`tf.data.experimental.dense_to_ragged_batch`.


```python
train_ds = train_ds.apply(tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE))
eval_ds = eval_ds.apply(tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE))
```

Let's make sure our datasets look as we expect them to:


```python
class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))


def visualize_dataset(dataset, bounding_box_format):
    sample = next(iter(dataset))
    images, boxes = sample["images"], sample["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=boxes,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        thickness=4,
        font_scale=1,
        class_mapping=class_mapping,
    )


visualize_dataset(train_ds, bounding_box_format="xywh")
```


    
![png](/img/guides/retina_net_overview/retina_net_overview_8_0.png)
    


and our eval set:


```python
visualize_dataset(eval_ds, bounding_box_format="xywh")
```


    
![png](/img/guides/retina_net_overview/retina_net_overview_10_0.png)
    


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
augment = keras_cv.layers.Augmenter(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
        keras_cv.layers.RandAugment(
            value_range=(0, 255),
            rate=0.5,
            magnitude=0.25,
            augmentations_per_image=2,
            geometric=False,
        ),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xywh"
        ),
    ]
)

train_ds = train_ds.map(
    lambda x: augment(x, training=True), num_parallel_calls=tf.data.AUTOTUNE
)
visualize_dataset(train_ds, bounding_box_format="xywh")

```

<div class="k-default-codeblock">
```
WARNING:tensorflow:From /home/lukewood/.local/lib/python3.7/site-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
Instructions for updating:
Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089

WARNING:tensorflow:From /home/lukewood/.local/lib/python3.7/site-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
Instructions for updating:
Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089

```
</div>
    
![png](/img/guides/retina_net_overview/retina_net_overview_13_2.png)
    


Great!  We now have a bounding box friendly augmentation pipeline.

Next, let's construct our eval pipeline:


```python
inference_resizing = keras_cv.layers.Resizing(
    640, 640, bounding_box_format="xywh", pad_to_aspect_ratio=True
)
eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
visualize_dataset(eval_ds, bounding_box_format="xywh")
```


    
![png](/img/guides/retina_net_overview/retina_net_overview_15_0.png)
    


Finally, let's unpackage our inputs from the preprocessing dictionary, and prepare to feed
the inputs into our model.


```python

def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )


train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)
```

Our data pipeline is now complete.  We can now move on to model creation and training.

---
## Model creation

We'll use the KerasCV API to construct a RetinaNet model.  In this tutorial we use
a pretrained ResNet50 backbone, initializing the weights to weights produced by training
on the imagenet dataset.  In order to perform fine-tuning, we
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
    backbone=keras_cv.models.ResNet50(
        include_top=False, weights="imagenet", include_rescaling=True
    ).as_backbone(),
)
# For faster convergence, freeze the feature extraction filters by setting:
model.backbone.trainable = False
```

That is all it takes to construct a KerasCV RetinaNet.  The RetinaNet accepts tuples of
dense image Tensors and bounding box dictionaries to `fit()` and `train_on_batch()`
This matches what we have constructed in our input pipeline above.


```python
callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.ReduceLROnPlateau(patience=5),
    keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True),
]

```

---
## Training our model

All that is left to do is train our model.  KerasCV object detection models follow the
standard Keras workflow, leveraging `compile()` and `fit()`.

Let's compile our model:


```python
# including a global_clipnorm is extremely important in object detection tasks
optimizer = tf.optimizers.SGD(global_clipnorm=10.0)
model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=optimizer,
)
```

And run `model.fit()`!


```python
model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)
# you can also save model weights with: `model.save_weights(CHECKPOINT_PATH)`
```

<div class="k-default-codeblock">
```
Epoch 1/20
1035/1035 [==============================] - 192s 169ms/step - loss: 1.3442 - box_loss: 0.6097 - cls_loss: 0.7345 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 1.1719 - val_box_loss: 0.5512 - val_cls_loss: 0.6207 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 2/20
1035/1035 [==============================] - 172s 166ms/step - loss: 1.0251 - box_loss: 0.4797 - cls_loss: 0.5454 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 1.0314 - val_box_loss: 0.4870 - val_cls_loss: 0.5445 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 3/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.8987 - box_loss: 0.4284 - cls_loss: 0.4703 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.9336 - val_box_loss: 0.4460 - val_cls_loss: 0.4875 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 4/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.8242 - box_loss: 0.4019 - cls_loss: 0.4224 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.8650 - val_box_loss: 0.4248 - val_cls_loss: 0.4402 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 5/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.7752 - box_loss: 0.3841 - cls_loss: 0.3910 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.8208 - val_box_loss: 0.4077 - val_cls_loss: 0.4131 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 6/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.7408 - box_loss: 0.3723 - cls_loss: 0.3685 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.7843 - val_box_loss: 0.3960 - val_cls_loss: 0.3883 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 7/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.7131 - box_loss: 0.3611 - cls_loss: 0.3519 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.7672 - val_box_loss: 0.3877 - val_cls_loss: 0.3795 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 8/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.6920 - box_loss: 0.3527 - cls_loss: 0.3392 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.7527 - val_box_loss: 0.3825 - val_cls_loss: 0.3702 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 9/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.6731 - box_loss: 0.3451 - cls_loss: 0.3281 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.7325 - val_box_loss: 0.3747 - val_cls_loss: 0.3578 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 10/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.6585 - box_loss: 0.3390 - cls_loss: 0.3195 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.7161 - val_box_loss: 0.3645 - val_cls_loss: 0.3516 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 11/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.6444 - box_loss: 0.3326 - cls_loss: 0.3117 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6987 - val_box_loss: 0.3614 - val_cls_loss: 0.3373 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 12/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.6336 - box_loss: 0.3286 - cls_loss: 0.3050 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6872 - val_box_loss: 0.3586 - val_cls_loss: 0.3286 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 13/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.6216 - box_loss: 0.3236 - cls_loss: 0.2980 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6761 - val_box_loss: 0.3525 - val_cls_loss: 0.3235 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 14/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.6136 - box_loss: 0.3196 - cls_loss: 0.2940 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6668 - val_box_loss: 0.3480 - val_cls_loss: 0.3188 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 15/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.6037 - box_loss: 0.3153 - cls_loss: 0.2884 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6634 - val_box_loss: 0.3440 - val_cls_loss: 0.3194 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 16/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.5964 - box_loss: 0.3119 - cls_loss: 0.2845 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6611 - val_box_loss: 0.3424 - val_cls_loss: 0.3186 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 17/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.5904 - box_loss: 0.3096 - cls_loss: 0.2808 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6453 - val_box_loss: 0.3390 - val_cls_loss: 0.3063 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 18/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.5812 - box_loss: 0.3049 - cls_loss: 0.2763 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6448 - val_box_loss: 0.3350 - val_cls_loss: 0.3098 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 19/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.5771 - box_loss: 0.3035 - cls_loss: 0.2735 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6388 - val_box_loss: 0.3315 - val_cls_loss: 0.3073 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 20/20
1035/1035 [==============================] - 172s 166ms/step - loss: 0.5702 - box_loss: 0.2998 - cls_loss: 0.2704 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6259 - val_box_loss: 0.3291 - val_cls_loss: 0.2968 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100

<keras.callbacks.History at 0x7f35e01ccf60>

```
</div>
---
## Inference

KerasCV makes object detection inference simple.  `model.predict(images)` returns a
RaggedTensor of bounding boxes.  By default, `RetinaNet.predict()` will perform
a non max suppression operation for you.


```python
model.load_weights(INFERENCE_CHECKPOINT_PATH)


def visualize_detections(model, bounding_box_format):
    images, y_true = next(iter(train_ds.take(1)))
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=3,
        cols=3,
        show=True,
        thickness=4,
        font_scale=1,
        class_mapping=class_mapping,
    )


visualize_detections(model, bounding_box_format="xywh")
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 2s 2s/step

```
</div>
    
![png](/img/guides/retina_net_overview/retina_net_overview_28_1.png)
    


To get good results, you should train for at least 50~ epochs.  You also may need to
tune the prediction decoder layer.  This can be done by passing a custom prediction
decoder to the RetinaNet constructor as follows:


```python
prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xywh",
    from_logits=True,
    iou_threshold=0.75,
    confidence_threshold=0.85,
)
model.prediction_decoder = prediction_decoder
visualize_detections(model, bounding_box_format="xywh")
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 2s 2s/step

```
</div>
    
![png](/img/guides/retina_net_overview/retina_net_overview_30_1.png)
    


---
## Results and conclusions

KerasCV makes it easy to construct state-of-the-art object detection pipelines.  All of
the KerasCV object detection components can be used independently, but also have deep
integration with each other.  With KerasCV, bounding box augmentation and more,
are all made simple and consistent.

Some follow up exercises for the reader:

- add additional augmentation techniques to improve model performance
- grid search `confidence_threshold` and `iou_threshold` on `NmsPredictionDecoder` to
    achieve an optimal Mean Average Precision
- tune the hyperparameters and data augmentation used to produce high quality results
- train an object detection model on another dataset

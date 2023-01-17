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

Shuffling ~/tensorflow_datasets/voc/2007/4.0.0.incomplete2B4Z2E/voc-test.tfrecord*...:   0%|          | 0/4952…

Generating train examples...:   0%|          | 0/2501 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2007/4.0.0.incomplete2B4Z2E/voc-train.tfrecord*...:   0%|          | 0/250…

Generating validation examples...:   0%|          | 0/2510 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2007/4.0.0.incomplete2B4Z2E/voc-validation.tfrecord*...:   0%|          | …

[1mDataset voc downloaded and prepared to ~/tensorflow_datasets/voc/2007/4.0.0. Subsequent calls will reuse this data.[0m
[1mDownloading and preparing dataset 3.59 GiB (download: 3.59 GiB, generated: Unknown size, total: 3.59 GiB) to ~/tensorflow_datasets/voc/2012/4.0.0...[0m

Dl Completed...: 0 url [00:00, ? url/s]

Dl Size...: 0 MiB [00:00, ? MiB/s]

Extraction completed...: 0 file [00:00, ? file/s]

Generating splits...:   0%|          | 0/3 [00:00<?, ? splits/s]

Generating test examples...:   0%|          | 0/10991 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2012/4.0.0.incomplete0A3FX8/voc-test.tfrecord*...:   0%|          | 0/1099…

Generating train examples...:   0%|          | 0/5717 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2012/4.0.0.incomplete0A3FX8/voc-train.tfrecord*...:   0%|          | 0/571…

Generating validation examples...:   0%|          | 0/5823 [00:00<?, ? examples/s]

Shuffling ~/tensorflow_datasets/voc/2012/4.0.0.incomplete0A3FX8/voc-validation.tfrecord*...:   0%|          | …

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
Epoch 1/50
1035/1035 [==============================] - 190s 167ms/step - loss: 1.3557 - box_loss: 0.6093 - cls_loss: 0.7464 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 1.1934 - val_box_loss: 0.5566 - val_cls_loss: 0.6368 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 2/50
1035/1035 [==============================] - 170s 164ms/step - loss: 1.0342 - box_loss: 0.4797 - cls_loss: 0.5545 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 1.0235 - val_box_loss: 0.4824 - val_cls_loss: 0.5411 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 3/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.9026 - box_loss: 0.4278 - cls_loss: 0.4749 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.9165 - val_box_loss: 0.4401 - val_cls_loss: 0.4764 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 4/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.8267 - box_loss: 0.4017 - cls_loss: 0.4250 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.8756 - val_box_loss: 0.4197 - val_cls_loss: 0.4559 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 5/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.7770 - box_loss: 0.3837 - cls_loss: 0.3933 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.8308 - val_box_loss: 0.4076 - val_cls_loss: 0.4232 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 6/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.7412 - box_loss: 0.3707 - cls_loss: 0.3705 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.7828 - val_box_loss: 0.3915 - val_cls_loss: 0.3913 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 7/50
1035/1035 [==============================] - 170s 163ms/step - loss: 0.7143 - box_loss: 0.3602 - cls_loss: 0.3541 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.7604 - val_box_loss: 0.3852 - val_cls_loss: 0.3751 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 8/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.6930 - box_loss: 0.3517 - cls_loss: 0.3412 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.7382 - val_box_loss: 0.3761 - val_cls_loss: 0.3621 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 9/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.6738 - box_loss: 0.3442 - cls_loss: 0.3296 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.7193 - val_box_loss: 0.3684 - val_cls_loss: 0.3509 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 10/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.6579 - box_loss: 0.3382 - cls_loss: 0.3197 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.7063 - val_box_loss: 0.3639 - val_cls_loss: 0.3424 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 11/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.6427 - box_loss: 0.3315 - cls_loss: 0.3111 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6943 - val_box_loss: 0.3592 - val_cls_loss: 0.3351 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 12/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.6334 - box_loss: 0.3277 - cls_loss: 0.3058 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6889 - val_box_loss: 0.3560 - val_cls_loss: 0.3330 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 13/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.6234 - box_loss: 0.3237 - cls_loss: 0.2997 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6752 - val_box_loss: 0.3488 - val_cls_loss: 0.3264 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 14/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.6133 - box_loss: 0.3192 - cls_loss: 0.2941 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6635 - val_box_loss: 0.3460 - val_cls_loss: 0.3175 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 15/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.6038 - box_loss: 0.3153 - cls_loss: 0.2885 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6566 - val_box_loss: 0.3429 - val_cls_loss: 0.3137 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 16/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5965 - box_loss: 0.3116 - cls_loss: 0.2849 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6623 - val_box_loss: 0.3443 - val_cls_loss: 0.3180 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 17/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5907 - box_loss: 0.3092 - cls_loss: 0.2815 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6401 - val_box_loss: 0.3354 - val_cls_loss: 0.3047 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 18/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5833 - box_loss: 0.3061 - cls_loss: 0.2773 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6367 - val_box_loss: 0.3350 - val_cls_loss: 0.3017 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 19/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5774 - box_loss: 0.3026 - cls_loss: 0.2748 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6311 - val_box_loss: 0.3332 - val_cls_loss: 0.2979 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 20/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5708 - box_loss: 0.2999 - cls_loss: 0.2709 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6306 - val_box_loss: 0.3294 - val_cls_loss: 0.3013 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 21/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5656 - box_loss: 0.2976 - cls_loss: 0.2680 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6218 - val_box_loss: 0.3272 - val_cls_loss: 0.2946 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 22/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5608 - box_loss: 0.2952 - cls_loss: 0.2656 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6161 - val_box_loss: 0.3244 - val_cls_loss: 0.2917 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 23/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5555 - box_loss: 0.2931 - cls_loss: 0.2624 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6129 - val_box_loss: 0.3242 - val_cls_loss: 0.2887 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 24/50
1035/1035 [==============================] - 169s 163ms/step - loss: 0.5501 - box_loss: 0.2900 - cls_loss: 0.2600 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6121 - val_box_loss: 0.3201 - val_cls_loss: 0.2920 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 25/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5473 - box_loss: 0.2885 - cls_loss: 0.2588 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.6076 - val_box_loss: 0.3181 - val_cls_loss: 0.2895 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 26/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5415 - box_loss: 0.2856 - cls_loss: 0.2559 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6005 - val_box_loss: 0.3164 - val_cls_loss: 0.2840 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 27/50
1035/1035 [==============================] - 169s 163ms/step - loss: 0.5392 - box_loss: 0.2851 - cls_loss: 0.2541 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.6017 - val_box_loss: 0.3158 - val_cls_loss: 0.2859 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 28/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5346 - box_loss: 0.2826 - cls_loss: 0.2519 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5963 - val_box_loss: 0.3147 - val_cls_loss: 0.2816 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 29/50
1035/1035 [==============================] - 170s 163ms/step - loss: 0.5297 - box_loss: 0.2806 - cls_loss: 0.2492 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5906 - val_box_loss: 0.3124 - val_cls_loss: 0.2782 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 30/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5271 - box_loss: 0.2791 - cls_loss: 0.2480 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5885 - val_box_loss: 0.3116 - val_cls_loss: 0.2769 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 31/50
1035/1035 [==============================] - 170s 163ms/step - loss: 0.5225 - box_loss: 0.2769 - cls_loss: 0.2457 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.5892 - val_box_loss: 0.3105 - val_cls_loss: 0.2787 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 32/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5197 - box_loss: 0.2753 - cls_loss: 0.2444 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5902 - val_box_loss: 0.3098 - val_cls_loss: 0.2803 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 33/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5159 - box_loss: 0.2731 - cls_loss: 0.2428 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5836 - val_box_loss: 0.3081 - val_cls_loss: 0.2755 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 34/50
1035/1035 [==============================] - 170s 163ms/step - loss: 0.5140 - box_loss: 0.2721 - cls_loss: 0.2419 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.5790 - val_box_loss: 0.3059 - val_cls_loss: 0.2732 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 35/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5103 - box_loss: 0.2705 - cls_loss: 0.2398 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5804 - val_box_loss: 0.3061 - val_cls_loss: 0.2743 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 36/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5074 - box_loss: 0.2691 - cls_loss: 0.2383 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.5755 - val_box_loss: 0.3050 - val_cls_loss: 0.2704 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 37/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5044 - box_loss: 0.2678 - cls_loss: 0.2367 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.5731 - val_box_loss: 0.3030 - val_cls_loss: 0.2700 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 38/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5020 - box_loss: 0.2664 - cls_loss: 0.2356 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5753 - val_box_loss: 0.3009 - val_cls_loss: 0.2744 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 39/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.5000 - box_loss: 0.2655 - cls_loss: 0.2345 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5720 - val_box_loss: 0.3005 - val_cls_loss: 0.2715 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 40/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4964 - box_loss: 0.2639 - cls_loss: 0.2325 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.5750 - val_box_loss: 0.3056 - val_cls_loss: 0.2694 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 41/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4943 - box_loss: 0.2625 - cls_loss: 0.2317 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5671 - val_box_loss: 0.2996 - val_cls_loss: 0.2675 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 42/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4904 - box_loss: 0.2605 - cls_loss: 0.2299 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.5653 - val_box_loss: 0.2979 - val_cls_loss: 0.2674 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 43/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4884 - box_loss: 0.2591 - cls_loss: 0.2293 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5698 - val_box_loss: 0.2977 - val_cls_loss: 0.2721 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 44/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4852 - box_loss: 0.2576 - cls_loss: 0.2276 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5596 - val_box_loss: 0.2959 - val_cls_loss: 0.2637 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 45/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4832 - box_loss: 0.2571 - cls_loss: 0.2261 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5588 - val_box_loss: 0.2956 - val_cls_loss: 0.2632 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 46/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4799 - box_loss: 0.2554 - cls_loss: 0.2245 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5605 - val_box_loss: 0.2948 - val_cls_loss: 0.2657 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 47/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4801 - box_loss: 0.2554 - cls_loss: 0.2247 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.5574 - val_box_loss: 0.2937 - val_cls_loss: 0.2637 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 48/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4765 - box_loss: 0.2537 - cls_loss: 0.2229 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5562 - val_box_loss: 0.2918 - val_cls_loss: 0.2644 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 49/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4744 - box_loss: 0.2524 - cls_loss: 0.2220 - percent_boxes_matched_with_anchor: 0.9111 - val_loss: 0.5585 - val_box_loss: 0.2930 - val_cls_loss: 0.2655 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100
Epoch 50/50
1035/1035 [==============================] - 170s 164ms/step - loss: 0.4718 - box_loss: 0.2506 - cls_loss: 0.2212 - percent_boxes_matched_with_anchor: 0.9110 - val_loss: 0.5530 - val_box_loss: 0.2920 - val_cls_loss: 0.2610 - val_percent_boxes_matched_with_anchor: 0.9056 - lr: 0.0100

<keras.callbacks.History at 0x7f5fb007be10>

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

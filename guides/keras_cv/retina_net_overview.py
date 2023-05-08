"""
Title: Train an Object Detection Model on Pascal VOC 2007 using KerasCV
Author: [lukewood](https://twitter.com/luke_wood_ml)
Date created: 2022/08/22
Last modified: 2022/08/22
Description: Use KerasCV to train a RetinaNet on Pascal VOC 2007.
Accelerator: GPU
"""

"""
## Overview

KerasCV offers a complete set of APIs to train your own state-of-the-art,
production-grade object detection model.  These APIs include object detection specific
data augmentation techniques, and batteries included object detection models.

To get started, let's sort out all of our imports and define global configuration parameters.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import optimizers
import math
from keras.optimizers import schedules

import keras_cv
from keras_cv import bounding_box
import os
import resource
from luketils import visualization

BATCH_SIZE = 16
EPOCHS = int(os.getenv("EPOCHS", "1"))
# To fully train a RetinaNet, comment out this line.
# EPOCHS = 100
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint/")
INFERENCE_CHECKPOINT_PATH = os.getenv("INFERENCE_CHECKPOINT_PATH", CHECKPOINT_PATH)

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

"""
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
"""


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

"""
Next, lets batch our data.  In KerasCV object detection tasks it is recommended that
users use ragged batches.  This is due to the fact that images may be of different
sizes in PascalVOC and that there may be different numbers of bounding boxes per image.

The easiest way to construct a ragged dataset in a `tf.data` pipeline is to use
`tf.data.experimental.dense_to_ragged_batch`.
"""

train_ds = train_ds.apply(tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE))
eval_ds = eval_ds.apply(tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE))

"""
Let's make sure our datasets look as we expect them to:
"""

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

"""
and our eval set:
"""

visualize_dataset(eval_ds, bounding_box_format="xywh")

"""
Looks like everything is structured as expected.  Now we can move on to constructing our
data augmentation pipeline.
"""

"""
## Data augmentation

One of the most labor-intensive tasks when constructing object detection pipelines is
data augmentation.  Image augmentation techniques must be aware of the underlying
bounding boxes, and must update them accordingly.

Luckily, KerasCV natively supports bounding box augmentation with its extensive library
of [data augmentation layers](https://keras.io/api/keras_cv/layers/preprocessing/).
The code below loads the Pascal VOC dataset, and performs on-the-fly bounding box
friendly data augmentation inside of a `tf.data` pipeline.
"""

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


"""
Great!  We now have a bounding box friendly augmentation pipeline.

Next, let's construct our eval pipeline:
"""
inference_resizing = keras_cv.layers.Resizing(
    640, 640, bounding_box_format="xywh", pad_to_aspect_ratio=True
)
eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
visualize_dataset(eval_ds, bounding_box_format="xywh")

"""
Finally, let's unpackage our inputs from the preprocessing dictionary, and prepare to feed
the inputs into our model.
"""


def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )


train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)

"""
Our data pipeline is now complete.  We can now move on to model creation and training.
"""

"""
## Model creation

We'll use the KerasCV API to construct a RetinaNet model.  In this tutorial we use
a pretrained ResNet50 backbone, initializing the weights to weights produced by training
on the imagenet dataset.  In order to perform fine-tuning, we
freeze the backbone before training.  When `include_rescaling=True` is set, inputs to
the model are expected to be in the range `[0, 255]`.
"""

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

"""
That is all it takes to construct a KerasCV RetinaNet.  The RetinaNet accepts tuples of
dense image Tensors and bounding box dictionaries to `fit()` and `train_on_batch()`
This matches what we have constructed in our input pipeline above.
"""

callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True),
]


"""
## Training our model

All that is left to do is train our model.  KerasCV object detection models follow the
standard Keras workflow, leveraging `compile()` and `fit()`.

Let's compile our model:
"""

# including a global_clipnorm is extremely important in object detection tasks
base_lr = 0.01
lr_decay = schedules.PiecewiseConstantDecay(
    boundaries=[12000 * 16, 16000 * 16],
    values=[base_lr, 0.1 * base_lr, 0.01 * base_lr],
)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=lr_decay, momentum=0.9, global_clipnorm=10.0
)
model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=optimizer,
)

"""
And run `model.fit()`!
"""

model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)
# you can also save model weights with: `model.save_weights(CHECKPOINT_PATH)`

"""
## Inference

KerasCV makes object detection inference simple.  `model.predict(images)` returns a
RaggedTensor of bounding boxes.  By default, `RetinaNet.predict()` will perform
a non max suppression operation for you.
"""

model.load_weights(INFERENCE_CHECKPOINT_PATH)


def visualize_detections(model, bounding_box_format):
    images, y_true = next(iter(eval_ds.take(1)))
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

"""
To get good results, you should train for at least 50~ epochs.  You also may need to
tune the prediction decoder layer.  This can be done by passing a custom prediction
decoder to the RetinaNet constructor as follows:
"""

prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xywh",
    from_logits=True,
    # Decrease the required threshold to make predictions get pruned out
    iou_threshold=0.35,
    # Tune confidence threshold for predictions to pass NMS
    confidence_threshold=0.95,
)
model.prediction_decoder = prediction_decoder
visualize_detections(model, bounding_box_format="xywh")
"""
## Results and conclusions

KerasCV makes it easy to construct state-of-the-art object detection pipelines.  All of
the KerasCV object detection components can be used independently, but also have deep
integration with each other.  With KerasCV, bounding box augmentation and more,
are all made simple and consistent.

Some follow up exercises for the reader:

- add additional augmentation techniques to improve model performance
- grid search `confidence_threshold` and `iou_threshold` on `MultiClassNonMaxSuppression` to
    achieve an optimal Mean Average Precision
- tune the hyperparameters and data augmentation used to produce high quality results
- train an object detection model on another dataset
"""

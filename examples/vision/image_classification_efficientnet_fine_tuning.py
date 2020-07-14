"""
Title: Image classification via fine-tuning with EfficientNet
Author: [Yixing Fu](https://github.com/yixingfu)
Date created: 2020/06/30
Last modified: 2020/07/08
Description: Use EfficientNet with weights pre-trained on imagenet for CIFAR-100 classification.
"""
"""

## Introduction: what is EfficientNet

EfficientNet, first introduced in [Tan and Le, 2019](https://arxiv.org/abs/1905.11946)
is among the most efficient models (i.e. requiring least FLOPS for inference)
that reaches State-of-the-Art accuracy on both
imagenet and common image classification transfer learning tasks.

The smallest base model is similar to [MnasNet](https://arxiv.org/abs/1807.11626), which
reached near-SOTA with a significantly smaller model. By introducing a heuristic way to
scale the model, EfficientNet provides a family of models (B0 to B7) that represents a
good combination of efficiency and accuracy on a variety of scales. Such a scaling
heuristics (compound-scaling, details see
[Tan and Le, 2019](https://arxiv.org/abs/1905.11946)) allows the
efficiency-oriented base model (B0) to surpass models at every scale, while avoiding
extensive grid-search of hyperparameters.

A summary of the latest updates on the model is available at
[here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), where various
augmentation schemes and semi-supervised learning approaches are applied to further
improve the imagenet performance of the models. These extensions of the model can be used
by updating weights without changing model architecture.

## B0 to B7 variants of EfficientNet

*(This section provides some details on the compound scaling, and can be skipped
if only interested in using the models)*

Based on the [original paper](https://arxiv.org/abs/1905.11946) people may have the
impression that EfficientNet is a continuous family of models created by arbitrarily
choosing scaling factor in as Eq.(3) of the paper.  However, choice of resolution,
depth and width are also restricted by many factors:

- Resolution: Resolutions not divisible by 8, 16, etc. cause zero-padding near boundaries
of some layers which wastes computational resources. This especially applies to smaller
variants of the model, hence the input resolution for B0 and B1 are chosen as 224 and
240.

- Depth and width: The building blocks of EfficientNet demands channel size to be
multiples of 8.

- Resource limit: Memory limitation may bottleneck resolution when depth
and width can still increase. In such a situation, increasing depth and/or
width but keep resolution can still improve performance.

As a result, the depth, width and resolution of each variant of the EfficientNet models
are hand-picked and proven to produce good results, though they may be significantly
off from the compound scaling formula.
Therefore, the keras implementation (detailed below) only provide these 8 models, B0 to B7,
instead of allowing arbitray choice of width / depth / resolution parameters.

## Keras implementation of EfficientNet

An implementation of EfficientNet B0 to B7 has been shipped with tf.keras since TF2.3. To
use EfficientNetB0 for classifying 1000 classes of images from imagenet, run
```
from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights='imagenet')
```

This model takes input images of shape (224, 224, 3), and the input data should range
[0, 255]. Normalization is included as part of the model.

Because training EfficientNet on ImageNet takes a tremendous amount of resources and
several techniques that are not a part of the model architecture itself. Hence the Keras
implementation by default loads pre-trained weights obtained via training with
[AutoAugment](https://arxiv.org/abs/1805.09501).

For B0 to B7 base models, the input shapes are different. Here is a list of input shape
expected for each model:

| Base model | resolution|
|----------------|-----|
| EfficientNetB0 | 224 |
| EfficientNetB1 | 240 |
| EfficientNetB2 | 260 |
| EfficientNetB3 | 300 |
| EfficientNetB4 | 380 |
| EfficientNetB5 | 456 |
| EfficientNetB6 | 528 |
| EfficientNetB7 | 600 |

When the model is intended for transfer learning, the Keras implementation
provides a option to remove the top layers:
```
model = EfficientNetB0(include_top=False, weights='imagenet')
```
This option excludes the final `Dense` layer that turns 1280 features on the penultimate
layer into prediction of the 1000 ImageNet classes. Replacing the top layer with custom
layers allows using EfficientNet as a feature extractor in a transfer learning workflow.

Another argument in the model constructor worth noticing is `drop_connect_rate` which controls
the dropout rate responsible for [stochastic depth](https://arxiv.org/abs/1603.09382).
This parameter serves as a toggle for extra regularization in finetuning, but does not
affect loaded weights. For example, when stronger regularization is desired, try:
```
model = EfficientNetB0(weights='imagenet', drop_connect_rate=0.4)
```
The default value is 0.2.

## Example: EfficientNetB0 for CIFAR-100.

EfficientNet is capable of a wide range of image classification tasks.
This makes it a good model for transfer learning.
As an end-to-end example, we will show using pre-trained EfficientNetB0 on CIFAR-100.

Notice that CIFAR-100 dataset has image size 32x32. This is much smaller than
the intended input resolution (224x224) for EfficientNetB0.
When fine-tuning a model that has been pre-trained on higher resolution dataset for
application to lower resolution image, we must up-sample the image.
When the up-sampling ratio is so large, fine-tuning show significant advantage over
training from scratch, and fine-tuning in the right way becomes important.

"""
# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224
"""

## Setup and data loading

This example requires TensorFlow 2.3 or above.

To use TPU, the TPU runtime must match current running TensorFlow
version. If there is a mismatch, try
```
from cloud_tpu_client import Client
c = Client()
c.configure_tpu_version(tf.__version__, restart_type="always")
```
"""

import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()


"""
### Loading data

Here we load data from [tensorflow_datasets](https://www.tensorflow.org/datasets)
(hereafter TFDS).
[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is provided in
TFDS as [cifar100](https://www.tensorflow.org/datasets/catalog/cifar100).

By simply changing `dataset_name` below, you may also try this notebook for
other datasets in TFDS such as
[cifar10](https://www.tensorflow.org/datasets/catalog/cifar10),
[food101](https://www.tensorflow.org/datasets/catalog/food101),
[stanford_dogs](https://www.tensorflow.org/datasets/catalog/stanford_dogs),
etc.

For TPU: if using TFDS datasets,
a [GCS bucket](https://cloud.google.com/storage/docs/key-terms#buckets)
location is required to save the datasets. For example,
```
tfds.load(dataset_name, data_dir="gs://example-bucket/datapath")
```
Also, both the current environment and the TPU service account have
proper [access](https://cloud.google.com/tpu/docs/storage-buckets#authorize_the_service_account)
to the bucket. Alternatively, for small datasets you may try loading data
into the memory and use `tf.data.Dataset.from_tensor_slices()`.
"""


import tensorflow_datasets as tfds

batch_size = 64

dataset_name = "cifar100"
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train", "test"], with_info=True, as_supervised=True
)
NUM_CLASSES = ds_info.features["label"].num_classes


"""
### Visualizing the data

The following code shows the first 9 images with their labels both
in numeric form and text.
"""
import matplotlib.pyplot as plt

label_info = ds_info.features["label"]
for i, (image, label) in enumerate(ds_train.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title("{}, {}".format((label), label_info.int2str(label)))
    plt.axis("off")


"""
### Data augmentation

We can use preprocessing layers APIs for image augmentation.
"""

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

"""
This `Sequential` model object can be used both as a part of
the model we later build, and as a function to preprocess
data before feeding into the model. Using them as function makes
it easy to visualize the augmented images. Here we plot 9 examples
of augmentation result of a given figure.
"""

for image, label in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(tf.expand_dims(image, axis=0))
        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title("{}, {}".format((label), label_info.int2str(label)))
        plt.axis("off")


"""
### Prepare inputs

Once we verify the input data and augmentation are working correctly,
we prepare dataset for training. The input data are resized to uniform
`IMG_SIZE`. The labels are put into one-hot
(a.k.a. categorical) encoding. The dataset is batched.

Note: `cache`, `prefetch` and `AUTOTUNE` may in some situation improve
performance, but depends on environment and the specific dataset used.
See this [guide](https://www.tensorflow.org/guide/data_performance)
for more information on data pipeline performance.
"""

# One-hot / categorical encoding
def input_preprocess(image, label):
    resize = preprocessing.Resizing(IMG_SIZE, IMG_SIZE, interpolation="bilinear")
    image = resize(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_train = ds_train.cache()
ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)


"""
## Training a model from scratch

We build an EfficientNetB0 with 100 output classes, that is initialized from scratch:

Note: the model will start noticeably overfitting after ~20 epochs.
"""

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import SGD

with strategy.scope():
    inputs = layers.Input(shape=(None, None, 3))
    x = inputs

    x = img_augmentation(x)

    x = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(x)

    model = tf.keras.Model(inputs, x)

    sgd = SGD(learning_rate=0.2, momentum=0.1, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.005
)

epochs = 20  # @param {type: "slider", min:5, max:50}
hist = model.fit(
    ds_train, epochs=epochs, validation_data=ds_test, callbacks=[reduce_lr]
)


"""
Training the model is relatively fast (takes only 20 seconds per epoch on TPUv2 that is
available on colab). This might make it sounds easy to simply train EfficientNet on any
dataset wanted from scratch. However, training EfficientNet on smaller datasets,
especially those with lower resolution like CIFAR-100, faces the significant challenge of
overfitting or getting trapped in local extrema.

Hence traning from scratch requires very careful choice of hyperparameters and is
difficult to find suitable regularization. Plotting the training and validation accuracy
makes it clear that validation accuracy stagnates at very low value.
"""

import matplotlib.pyplot as plt

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

plot_hist(hist)

"""
## Transfer learning from pre-trained weights

Here we initialize the model with pre-trained ImageNet weights,
and we fine-tune it on our own dataset.
"""

from tensorflow.keras.layers.experimental import preprocessing


def build_model(n_classes):
    inputs = layers.Input(shape=(None, None, 3))
    x = inputs

    x = img_augmentation(x)

    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, x, name="EfficientNet")
    sgd = SGD(learning_rate=0.2, momentum=0.1, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


"""
The first step to transfer learning is to freeze all layers and train only the top
layers. For this step, a relatively large learning rate (~0.1) can be used to start with,
while applying some learning rate decay (either `ExponentialDecay` or use the `ReduceLROnPlateau`
callback). On CIFAR-100 with `EfficientNetB0`, this step will take validation accuracy to
~70% with suitable image augmentation. For this stage, using
`EfficientNetB0`, validation accuracy and loss will be consistently better than training
accuracy and loss. This is because the regularization is strong, which only
suppresses train time metrics.

Note that the convergence may take up to 50 epochs. If no data augmentation layer is
applied, expect the validation accuracy to reach only ~60% even for many epochs.
"""

from tensorflow.keras.callbacks import ReduceLROnPlateau

with strategy.scope():
    model = build_model(n_classes=NUM_CLASSES)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001)

epochs = 25  # @param {type: "slider", min:8, max:80}
hist = model.fit(
    ds_train, epochs=epochs, validation_data=ds_test, callbacks=[reduce_lr]
)
plot_hist(hist)

"""
The second step is to unfreeze a number of layers. Unfreezing layers and fine-tuning them is
usually thought to only provide incremental improvements on validation accuracy, but for
the case of `EfficientNetB0` it boosts validation accuracy by about 10% to pass 80%
(reaching ~87% as in the original paper requires including AutoAugmentation or Random
Augmentaion).

Note that the convergence may take more than 50 epochs. If no data augmentation layer is
applied, expect the validation accuracy to reach only ~70% even for many epochs.
"""

def unfreeze_model(model):
    for l in model.layers:
        if "bn" in l.name:
            print(f"{l.name} is staying untrainable")
        else:
            l.trainable = True

    sgd = SGD(learning_rate=0.0004)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = unfreeze_model(model)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
)
epochs = 25  # @param {type: "slider", min:8, max:80}
hist = model.fit(
    ds_train, epochs=epochs, validation_data=ds_test, callbacks=[reduce_lr],
)
plot_hist(hist)

"""
### Tips for fine tuning EfficientNet

On unfreezing layers:

- The `BathcNormalization` layers need to be kept frozen ([more details](https://keras.io/guides/transfer_learning/)).
If they are also turned to trainable, the
first epoch after unfreezing will significantly reduce accuracy.
- In some cases it may be beneficial to open up only a portion of layers instead of
unfreezing all. This will make fine tuning much faster when going to larger models like
B7.
- Each block needs to be all turned on or off. This is because the architecture includes
a shortcut from the first layer to the last layer for each block. Not respecting blocks
also significantly harms the final performance.


Some other tips for utilizing EfficientNet:

- Larger variants of EfficientNet do not guarantee improved performance, especially for
tasks with less data or fewer classes. In such a case, the larger variant of EfficientNet
chosen, the harder it is to tune hyperparameters.
- EMA (Exponential Moving Average) is very helpful in training EfficientNet from scratch,
but not so much for transfer learning.
- Do not use the RMSprop setup as in the original paper for transfer learning. The
momentum and learning rate are too high for transfer learning. It will easily corrupt the
pretrained weight and blow up the loss. A quick check is to see if loss (as categorical
cross entropy) is getting significantly larger than log(NUM_CLASSES) after the same
epoch. If so, the initial learning rate/momentum is too high.
- Smaller batch size benefit validation accuracy, possibly due to effectively providing
regularization.

## Using the latest EfficientNet weights

Since the initial paper, the EfficientNet has been improved by various methods for data
preprocessing and for using unlabelled data to enhance learning results. These
improvements are relatively hard and computationally costly to reproduce, and require
extra code; but the weights are readily available in the form of TF checkpoint files. The
model architecture has not changed, so loading the improved checkpoints is possible.

To use a checkpoint provided at
[the official model repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), first
download the checkpoint. As example, here we download noisy-student version of B1:

```
!wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet\
       /noisystudent/noisy_student_efficientnet-b1.tar.gz
!tar -xf noisy_student_efficientnet-b1.tar.gz
```

Then use the script efficientnet_weight_update_util.py to convert ckpt file to h5 file.

```
!python efficientnet_weight_update_util.py --model b1 --notop --ckpt \
        efficientnet-b1/model.ckpt --o efficientnetb1_notop.h5
```

When creating model, use the following to load new weight:

```
model = EfficientNetB0(weights="efficientnetb1_notop.h5", include_top=False)
```
"""

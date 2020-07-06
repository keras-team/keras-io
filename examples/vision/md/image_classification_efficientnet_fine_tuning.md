# Image classification using EfficientNet and fine-tuning

**Author:** Yixing Fu<br>
**Date created:** 2020/06/30<br>
**Last modified:** 2020/07/06<br>
**Description:** Use EfficientNet with weights pre-trained on imagenet for CIFAR-100 classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_efficientnet_fine_tuning.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_efficientnet_fine_tuning.py)



---
## What is EfficientNet
EfficientNet, first introduced in https://arxiv.org/abs/1905.11946 is among the most
efficient models (i.e. requiring least FLOPS for inference) that reaches SOTA in both
imagenet and common image classification transfer learning tasks.

The smallest base model is similar to MnasNet (https://arxiv.org/abs/1807.11626), which
reached near-SOTA with a significantly smaller model. By introducing a heuristic way to
scale the model, EfficientNet provides a family of models (B0 to B7) that represents a
good combination of efficiency and accuracy on a variety of scales. Such a scaling
heuristics (compound-scaling, details see https://arxiv.org/abs/1905.11946) allows the
efficiency-oriented base model (B0) to surpass models at every scale, while avoiding
extensive grid-search of hyperparameters.

A summary of the latest updates on the model is available at
https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet, where various
augmentation schemes and semi-supervised learning approaches are applied to further
improve the imagenet performance of the models. These extensions of the model can be used
by updating weights without changing model architecture.

---
## Compound scaling

The EfficientNet models are approximately created using compound scaling. Starting from
the base model B0, as model size scales from B0 to B7, the extra computational resource
is proportioned into width, depth and resolution of the model by requiring each of the
three dimensions to grow at the same power of a set of fixed ratios.

However, it must be noted that the ratios are not taken accurately. A few points need to
be taken into account:
Resolution. Resolutions not divisible by 8, 16, etc. cause zero-padding near boundaries
of some layers which wastes computational resources. This especially applies to smaller
variants of the model, hence the input resolution for B0 and B1 are chosen as 224 and
240.
Depth and width. Channel size is always rounded to 8/16/32 because of the architecture.
Resource limit. Perfect compound scaling would assume spatial (memory) and time allowance
for the computation to grow simultaneously, but OOM may further bottleneck the scaling of
resolution.

As a result, compound scaling factor is significantly off from
https://arxiv.org/abs/1905.11946. Hence it is important to understand the compound
scaling as a rule of thumb that leads to this family of base models, rather than an exact
optimization scheme. This also justifies that in the keras implementation (detailed
below), only these 8 models, B0 to B7, are exposed to the user and arbitrary width /
depth / resolution is not allowed.

---
## Keras implementation of EfficientNet

An implementation of EfficientNet B0 to B7 has been shipped with tf.keras since TF2.3. To
use EfficientNetB0 for classifying 1000 classes of images from imagenet, run
```
from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights='imagenet')
```

This model takes input images of shape (224, 224, 3), and the input data should range
[0,255]. Resizing and normalization are included as part of the model.

Because training EfficientNet on imagenet takes a tremendous amount of resources and
several techniques that are not a part of the model architecture itself. Hence the Keras
implementation by default loads pre-trained weights with AutoAugment
(https://arxiv.org/abs/1805.09501).

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

When the use of the model is intended for transfer learning, the Keras implementation
provides a option to remove the top layers:
```
model = EfficientNetB0(include_top=False, weights='imagenet')
```
This option excludes the final Dense layer that turns 1280 features on the penultimate
layer into prediction of the 1000 classes in imagenet. Replacing the top with custom
layers allows using EfficientNet as a feature extractor and transfers the pretrained
weights to other tasks.

Another keyword in the model builder worth noticing is `drop_connect_rate` which controls
the dropout rate responsible for stochastic depth (https://arxiv.org/abs/1603.09382).
This parameter serves as a toggle for extra regularization in finetuning, but does not
alter loaded weights.



---
## Example: EfficientNetB0 for CIFAR-100.

As an architecture, EfficientNet is capable of a wide range of image classification
tasks. For example, we will show using pre-trained EfficientNetB0 on CIFAR-100. For
EfficientNetB0, image size is 224.


```python
# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224
```

### prepare


```python
!!pip install --quiet tensorflow==2.3.0rc0
!!pip install --quiet cloud-tpu-client
```




```python
import tensorflow as tf

try:
    from cloud_tpu_client import Client

    c = Client()
    c.configure_tpu_version(tf.__version__, restart_type="always")
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

```
<div class="k-default-codeblock">
```
['WARNING: You are using pip version 20.1; however, version 20.1.1 is available.',
 "You should consider upgrading via the '/usr/bin/python -m pip install --upgrade pip' command."]

Running on TPU  ['10.240.1.2:8470']
INFO:tensorflow:Initializing the TPU system: tpututorial

INFO:tensorflow:Initializing the TPU system: tpututorial

INFO:tensorflow:Clearing out eager caches

INFO:tensorflow:Clearing out eager caches

INFO:tensorflow:Finished initializing TPU system.

INFO:tensorflow:Finished initializing TPU system.
WARNING:absl:`tf.distribute.experimental.TPUStrategy` is deprecated, please use  the non experimental symbol `tf.distribute.TPUStrategy` instead.

INFO:tensorflow:Found TPU system:

INFO:tensorflow:Found TPU system:

INFO:tensorflow:*** Num TPU Cores: 8

INFO:tensorflow:*** Num TPU Cores: 8

INFO:tensorflow:*** Num TPU Workers: 1

INFO:tensorflow:*** Num TPU Workers: 1

INFO:tensorflow:*** Num TPU Cores Per Worker: 8

INFO:tensorflow:*** Num TPU Cores Per Worker: 8

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)

```
</div>
Below is example code for loading data.
To see sensible result, you need to load entire dataset and adjust epochs for
training; but you may truncate data for a quick verification of the workflow.
Expect the notebook to run at least an hour for GPU, while much faster on TPU if
using hosted Colab session.


```python
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

batch_size = 64

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
NUM_CLASSES = 100

x_train = tf.cast(x_train, tf.int32)
x_test = tf.cast(x_test, tf.int32)

truncate_data = False  # @param {type: "boolean"}
if truncate_data:
    x_train = x_train[0:5000]
    y_train = y_train[0:5000]
    x_test = x_test[0:1000]
    y_test = y_test[0:1000]


# one-hot / categorical
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.cache()
ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)
```

### training from scratch
To build model that use EfficientNetB0 with 100 classes that is initiated from scratch:

Note: to better see validation peeling off from training accuracy, run ~20 epochs.


```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental.preprocessing import (
    Resizing,
    RandomFlip,
    RandomContrast,
    # RandomHeight,
)
from tensorflow.keras.optimizers import SGD

with strategy.scope():
    inputs = keras.layers.Input(shape=(32, 32, 3))
    x = inputs

    x = RandomFlip()(x)
    x = RandomContrast(0.1)(x)
    # x = RandomHeight(0.1)(x)
    x = Resizing(IMG_SIZE, IMG_SIZE, interpolation="bilinear")(x)

    x = EfficientNetB0(include_top=True, weights=None, classes=100)(x)

    model = keras.Model(inputs, x)

    sgd = SGD(learning_rate=0.2, momentum=0.1, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.005, verbose=2
)

epochs = 20  # @param {type: "slider", min:5, max:50}
hist = model.fit(
    ds_train, epochs=epochs, validation_data=ds_test, callbacks=[reduce_lr], verbose=2
)

```

<div class="k-default-codeblock">
```
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
random_flip (RandomFlip)     (None, 32, 32, 3)         0         
_________________________________________________________________
random_contrast (RandomContr (None, 32, 32, 3)         0         
_________________________________________________________________
resizing (Resizing)          (None, 224, 224, 3)       0         
_________________________________________________________________
efficientnetb0 (Functional)  (None, 100)               4177671   
=================================================================
Total params: 4,177,671
Trainable params: 4,135,648
Non-trainable params: 42,023
_________________________________________________________________
Epoch 1/20
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0091s vs `on_train_batch_end` time: 0.0426s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0091s vs `on_train_batch_end` time: 0.0426s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0073s vs `on_test_batch_end` time: 0.0213s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0073s vs `on_test_batch_end` time: 0.0213s). Check your callbacks.

781/781 - 51s - loss: 4.0524 - accuracy: 0.0782 - val_loss: 3.7740 - val_accuracy: 0.1373
Epoch 2/20
781/781 - 42s - loss: 3.2597 - accuracy: 0.1983 - val_loss: 2.7339 - val_accuracy: 0.3057
Epoch 3/20
781/781 - 44s - loss: 2.8151 - accuracy: 0.2848 - val_loss: 2.5497 - val_accuracy: 0.3458
Epoch 4/20
781/781 - 42s - loss: 2.5210 - accuracy: 0.3480 - val_loss: 2.3207 - val_accuracy: 0.3932
Epoch 5/20
781/781 - 43s - loss: 2.2938 - accuracy: 0.3978 - val_loss: 2.1789 - val_accuracy: 0.4312
Epoch 6/20
781/781 - 45s - loss: 2.1156 - accuracy: 0.4350 - val_loss: 2.1087 - val_accuracy: 0.4514
Epoch 7/20
781/781 - 45s - loss: 1.9626 - accuracy: 0.4706 - val_loss: 2.0596 - val_accuracy: 0.4624
Epoch 8/20
781/781 - 43s - loss: 1.8248 - accuracy: 0.5040 - val_loss: 1.8574 - val_accuracy: 0.5069
Epoch 9/20
781/781 - 45s - loss: 1.7056 - accuracy: 0.5312 - val_loss: 1.8604 - val_accuracy: 0.5112
Epoch 10/20
781/781 - 45s - loss: 1.5971 - accuracy: 0.5553 - val_loss: 1.8866 - val_accuracy: 0.5125
Epoch 11/20
781/781 - 46s - loss: 1.4993 - accuracy: 0.5820 - val_loss: 1.8002 - val_accuracy: 0.5268
Epoch 12/20
781/781 - 44s - loss: 1.4071 - accuracy: 0.6038 - val_loss: 1.8472 - val_accuracy: 0.5356
Epoch 13/20
781/781 - 45s - loss: 1.3200 - accuracy: 0.6253 - val_loss: 1.7777 - val_accuracy: 0.5474
Epoch 14/20
781/781 - 45s - loss: 1.2380 - accuracy: 0.6438 - val_loss: 1.7694 - val_accuracy: 0.5518
Epoch 15/20
781/781 - 44s - loss: 1.1666 - accuracy: 0.6633 - val_loss: 1.7883 - val_accuracy: 0.5597
Epoch 16/20
781/781 - 44s - loss: 1.0872 - accuracy: 0.6859 - val_loss: 1.7724 - val_accuracy: 0.5678
Epoch 17/20
781/781 - 43s - loss: 1.0258 - accuracy: 0.7002 - val_loss: 1.7608 - val_accuracy: 0.5672
Epoch 18/20
781/781 - 45s - loss: 0.9678 - accuracy: 0.7154 - val_loss: 1.7667 - val_accuracy: 0.5756
Epoch 19/20
781/781 - 46s - loss: 0.9077 - accuracy: 0.7294 - val_loss: 1.7843 - val_accuracy: 0.5742
Epoch 20/20
781/781 - 45s - loss: 0.8556 - accuracy: 0.7439 - val_loss: 1.8463 - val_accuracy: 0.5807

```
</div>
Training the model is relatively fast (takes only 20 seconds per epoch on TPUv2 that is
available on colab). This might make it sounds easy to simply train EfficientNet on any
dataset wanted from scratch. However, training EfficientNet on smaller datasets,
especially those with lower resolution like CIFAR-100, faces the significant challenge of
overfitting or getting trapped in local extrema.

Hence traning from scratch requires very careful choice of hyperparameters and is
difficult to find suitable regularization. Plotting the training and validation accuracy
makes it clear that validation accuracy stagnates at very low value.


```python
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
```


![png](/img/examples/vision/image_classification_efficientnet_fine_tuning/image_classification_efficientnet_fine_tuning_11_0.png)


### transfer learning from pretrained weight
Using pre-trained imagenet weights and only transfer learn (fine-tune) the model allows
utilizing the power of EfficientNet much easier. To use pretrained weight, the model can
be initiated through


```python
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import (
    Resizing,
    RandomContrast,
)


def build_model(n_classes):
    inputs = keras.layers.Input(shape=(32, 32, 3))
    x = inputs

    x = RandomFlip()(x)
    x = RandomContrast(0.1)(x)
    x = Resizing(IMG_SIZE, IMG_SIZE, interpolation="bilinear")(x)
    # other preprocessing layers can be used similar to Resizing and RandomRotation

    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # freeze the pretrained weights
    for l in model.layers:
        l.trainable = False

    # rebuild top
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = keras.layers.Dense(100, activation="softmax", name="pred")(x)

    # compile
    model = keras.Model(inputs, x, name="EfficientNet")
    sgd = SGD(learning_rate=0.2, momentum=0.1, nesterov=True)
    # sgd = tfa.optimizers.MovingAverage(sgd)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

```

Note that it is also possible to freeze pre-trained part entirely by
```
model.trainable = False
```
instead of setting each layer separately.


The first step to transfer learning is to freeze all layers and train only the top
layers. For this step a relatively large learning rate (~0.1) can be used to start with,
while applying some learning rate decay (either ExponentialDecay or use ReduceLROnPlateau
callback). On CIFAR-100 with EfficientNetB0, this step will take validation accuracy to
~70% with suitable (but not absolutely optimal) image augmentation. For this stage, using
EfficientNetB0, validation accuracy and loss will be consistently better than training
accuracy and loss. This is because the regularization is strong, which only
suppresses train time metrics.

Note that the convergence may take up to 50 epochs. If no data augmentation layer is
applied, expect the validation accuracy to reach only ~60% even for many epochs.


```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

with strategy.scope():
    model = build_model(n_classes=NUM_CLASSES)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001, verbose=2
)

epochs = 25  # @param {type: "slider", min:8, max:80}
hist = model.fit(
    ds_train, epochs=epochs, validation_data=ds_test, callbacks=[reduce_lr], verbose=2,
)
plot_hist(hist)
```

<div class="k-default-codeblock">
```
Epoch 1/25
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0055s vs `on_train_batch_end` time: 0.0227s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0055s vs `on_train_batch_end` time: 0.0227s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0076s vs `on_test_batch_end` time: 0.0226s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0076s vs `on_test_batch_end` time: 0.0226s). Check your callbacks.

781/781 - 31s - loss: 2.3805 - accuracy: 0.4403 - val_loss: 1.5734 - val_accuracy: 0.6037
Epoch 2/25
781/781 - 25s - loss: 1.8293 - accuracy: 0.5397 - val_loss: 1.4319 - val_accuracy: 0.6242
Epoch 3/25
781/781 - 24s - loss: 1.6594 - accuracy: 0.5684 - val_loss: 1.3463 - val_accuracy: 0.6377
Epoch 4/25
781/781 - 25s - loss: 1.5293 - accuracy: 0.5893 - val_loss: 1.2889 - val_accuracy: 0.6501
Epoch 5/25
781/781 - 24s - loss: 1.4747 - accuracy: 0.5993 - val_loss: 1.2679 - val_accuracy: 0.6539
Epoch 6/25
781/781 - 24s - loss: 1.4256 - accuracy: 0.6109 - val_loss: 1.2539 - val_accuracy: 0.6541
Epoch 7/25
781/781 - 25s - loss: 1.4071 - accuracy: 0.6139 - val_loss: 1.2367 - val_accuracy: 0.6586
Epoch 8/25
781/781 - 25s - loss: 1.3694 - accuracy: 0.6206 - val_loss: 1.2173 - val_accuracy: 0.6638
Epoch 9/25
781/781 - 24s - loss: 1.3479 - accuracy: 0.6266 - val_loss: 1.2174 - val_accuracy: 0.6603
Epoch 10/25
781/781 - 25s - loss: 1.3418 - accuracy: 0.6295 - val_loss: 1.2127 - val_accuracy: 0.6646
Epoch 11/25
781/781 - 25s - loss: 1.3347 - accuracy: 0.6288 - val_loss: 1.2062 - val_accuracy: 0.6666
Epoch 12/25
781/781 - 24s - loss: 1.3233 - accuracy: 0.6316 - val_loss: 1.2018 - val_accuracy: 0.6687
Epoch 13/25
781/781 - 26s - loss: 1.3147 - accuracy: 0.6333 - val_loss: 1.2025 - val_accuracy: 0.6674
Epoch 14/25
781/781 - 25s - loss: 1.3042 - accuracy: 0.6358 - val_loss: 1.1976 - val_accuracy: 0.6686
Epoch 15/25
781/781 - 24s - loss: 1.2929 - accuracy: 0.6370 - val_loss: 1.1899 - val_accuracy: 0.6703
Epoch 16/25
781/781 - 24s - loss: 1.2998 - accuracy: 0.6375 - val_loss: 1.1880 - val_accuracy: 0.6702
Epoch 17/25
781/781 - 24s - loss: 1.2855 - accuracy: 0.6385 - val_loss: 1.1901 - val_accuracy: 0.6707
Epoch 18/25
781/781 - 25s - loss: 1.2831 - accuracy: 0.6373 - val_loss: 1.1825 - val_accuracy: 0.6709
Epoch 19/25
781/781 - 23s - loss: 1.2779 - accuracy: 0.6417 - val_loss: 1.1843 - val_accuracy: 0.6714
Epoch 20/25
781/781 - 24s - loss: 1.2765 - accuracy: 0.6392 - val_loss: 1.1883 - val_accuracy: 0.6706
Epoch 21/25
781/781 - 24s - loss: 1.2670 - accuracy: 0.6428 - val_loss: 1.1849 - val_accuracy: 0.6747
Epoch 22/25
781/781 - 25s - loss: 1.2673 - accuracy: 0.6428 - val_loss: 1.1878 - val_accuracy: 0.6711
Epoch 23/25
781/781 - 25s - loss: 1.2580 - accuracy: 0.6484 - val_loss: 1.1808 - val_accuracy: 0.6719
Epoch 24/25
781/781 - 25s - loss: 1.2620 - accuracy: 0.6442 - val_loss: 1.1875 - val_accuracy: 0.6707
Epoch 25/25
781/781 - 24s - loss: 1.2658 - accuracy: 0.6438 - val_loss: 1.1895 - val_accuracy: 0.6697

```
</div>
![png](/img/examples/vision/image_classification_efficientnet_fine_tuning/image_classification_efficientnet_fine_tuning_15_5.png)


The second step is to unfreeze a number of layers. Unfreezing layers and fine tuning is
usually thought to only provide incremental improvements on validation accuracy, but for
the case of EfficientNetB0 it boosts validation accuracy by about 10% to pass 80%
(reaching ~87% as in the original paper requires including AutoAugmentation or Random
Augmentaion).

Note that the convergence may take more than 50 epochs. If no data augmentation layer is
applied, expect the validation accuracy to reach only ~70% even for many epochs.


```python

def unfreeze_model(model):
    for l in model.layers:
        if "bn" in l.name:
            print(f"{l.name} is staying untrainable")
        else:
            l.trainable = True

    sgd = SGD(learning_rate=0.005)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = unfreeze_model(model)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001, verbose=2
)
epochs = 25  # @param {type: "slider", min:8, max:80}
hist = model.fit(
    ds_train, epochs=epochs, validation_data=ds_test, callbacks=[reduce_lr], verbose=2,
)
plot_hist(hist)
```

<div class="k-default-codeblock">
```
stem_bn is staying untrainable
block1a_bn is staying untrainable
block1a_project_bn is staying untrainable
block2a_expand_bn is staying untrainable
block2a_bn is staying untrainable
block2a_project_bn is staying untrainable
block2b_expand_bn is staying untrainable
block2b_bn is staying untrainable
block2b_project_bn is staying untrainable
block3a_expand_bn is staying untrainable
block3a_bn is staying untrainable
block3a_project_bn is staying untrainable
block3b_expand_bn is staying untrainable
block3b_bn is staying untrainable
block3b_project_bn is staying untrainable
block4a_expand_bn is staying untrainable
block4a_bn is staying untrainable
block4a_project_bn is staying untrainable
block4b_expand_bn is staying untrainable
block4b_bn is staying untrainable
block4b_project_bn is staying untrainable
block4c_expand_bn is staying untrainable
block4c_bn is staying untrainable
block4c_project_bn is staying untrainable
block5a_expand_bn is staying untrainable
block5a_bn is staying untrainable
block5a_project_bn is staying untrainable
block5b_expand_bn is staying untrainable
block5b_bn is staying untrainable
block5b_project_bn is staying untrainable
block5c_expand_bn is staying untrainable
block5c_bn is staying untrainable
block5c_project_bn is staying untrainable
block6a_expand_bn is staying untrainable
block6a_bn is staying untrainable
block6a_project_bn is staying untrainable
block6b_expand_bn is staying untrainable
block6b_bn is staying untrainable
block6b_project_bn is staying untrainable
block6c_expand_bn is staying untrainable
block6c_bn is staying untrainable
block6c_project_bn is staying untrainable
block6d_expand_bn is staying untrainable
block6d_bn is staying untrainable
block6d_project_bn is staying untrainable
block7a_expand_bn is staying untrainable
block7a_bn is staying untrainable
block7a_project_bn is staying untrainable
top_bn is staying untrainable
Epoch 1/25
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0052s vs `on_train_batch_end` time: 0.0326s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0052s vs `on_train_batch_end` time: 0.0326s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0056s vs `on_test_batch_end` time: 0.0223s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0056s vs `on_test_batch_end` time: 0.0223s). Check your callbacks.

781/781 - 38s - loss: 1.1118 - accuracy: 0.6843 - val_loss: 0.9062 - val_accuracy: 0.7412
Epoch 2/25
781/781 - 31s - loss: 0.9105 - accuracy: 0.7319 - val_loss: 0.8623 - val_accuracy: 0.7543
Epoch 3/25
781/781 - 32s - loss: 0.8261 - accuracy: 0.7544 - val_loss: 0.8047 - val_accuracy: 0.7662
Epoch 4/25
781/781 - 32s - loss: 0.7714 - accuracy: 0.7681 - val_loss: 0.8077 - val_accuracy: 0.7680
Epoch 5/25
781/781 - 32s - loss: 0.7204 - accuracy: 0.7827 - val_loss: 0.8034 - val_accuracy: 0.7710
Epoch 6/25
781/781 - 32s - loss: 0.6885 - accuracy: 0.7906 - val_loss: 0.7699 - val_accuracy: 0.7757
Epoch 7/25
781/781 - 32s - loss: 0.6576 - accuracy: 0.7979 - val_loss: 0.7785 - val_accuracy: 0.7775
Epoch 8/25
781/781 - 32s - loss: 0.6249 - accuracy: 0.8105 - val_loss: 0.7680 - val_accuracy: 0.7785
Epoch 9/25
781/781 - 33s - loss: 0.5929 - accuracy: 0.8196 - val_loss: 0.7495 - val_accuracy: 0.7849
Epoch 10/25
781/781 - 34s - loss: 0.5666 - accuracy: 0.8249 - val_loss: 0.7364 - val_accuracy: 0.7874
Epoch 11/25
781/781 - 34s - loss: 0.5482 - accuracy: 0.8311 - val_loss: 0.7419 - val_accuracy: 0.7889
Epoch 12/25
781/781 - 32s - loss: 0.5244 - accuracy: 0.8380 - val_loss: 0.7304 - val_accuracy: 0.7916
Epoch 13/25
781/781 - 32s - loss: 0.5037 - accuracy: 0.8443 - val_loss: 0.7240 - val_accuracy: 0.7928
Epoch 14/25
781/781 - 34s - loss: 0.4816 - accuracy: 0.8484 - val_loss: 0.7234 - val_accuracy: 0.7937
Epoch 15/25
781/781 - 33s - loss: 0.4683 - accuracy: 0.8543 - val_loss: 0.7181 - val_accuracy: 0.7938
Epoch 16/25
781/781 - 33s - loss: 0.4451 - accuracy: 0.8603 - val_loss: 0.7072 - val_accuracy: 0.7989
Epoch 17/25
781/781 - 33s - loss: 0.4304 - accuracy: 0.8659 - val_loss: 0.6935 - val_accuracy: 0.8019
Epoch 18/25
781/781 - 32s - loss: 0.4137 - accuracy: 0.8711 - val_loss: 0.7098 - val_accuracy: 0.8005
Epoch 19/25
781/781 - 32s - loss: 0.4063 - accuracy: 0.8731 - val_loss: 0.6853 - val_accuracy: 0.8061
Epoch 20/25
781/781 - 33s - loss: 0.3839 - accuracy: 0.8811 - val_loss: 0.7232 - val_accuracy: 0.7994
Epoch 21/25
781/781 - 34s - loss: 0.3735 - accuracy: 0.8845 - val_loss: 0.7016 - val_accuracy: 0.8043
Epoch 22/25
781/781 - 33s - loss: 0.3617 - accuracy: 0.8888 - val_loss: 0.6951 - val_accuracy: 0.8048
Epoch 23/25
781/781 - 32s - loss: 0.3481 - accuracy: 0.8914 - val_loss: 0.6822 - val_accuracy: 0.8075
Epoch 24/25
781/781 - 32s - loss: 0.3379 - accuracy: 0.8956 - val_loss: 0.6938 - val_accuracy: 0.8050
Epoch 25/25
781/781 - 32s - loss: 0.3275 - accuracy: 0.8972 - val_loss: 0.6968 - val_accuracy: 0.8072

```
</div>
![png](/img/examples/vision/image_classification_efficientnet_fine_tuning/image_classification_efficientnet_fine_tuning_17_5.png)


### tips for fine tuning EfficientNet

On unfreezing layers:
- The batch normalization layers need to be kept untrainable
(https://keras.io/guides/transfer_learning/). If they are also turned to trainable, the
first epoch after unfreezing will significantly reduce accuracy.
- In some cases it may be beneficial to open up only a portion of layers instead of
unfreezing all. This will make fine tuning much faster when going to larger models like
B7.
- Each block needs to be all turned on or off. This is because the architecture includes
a shortcut from the first layer to the last layer for each block. Not respecting blocks
also significantly harms the final performance.


Some other tips for utilizing EfficientNet
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

---
## Using the latest EfficientNet weights

Since the initial paper, the EfficientNet has been improved by various methods for data
preprocessing and for using unlabelled data to enhance learning results. These
improvements are relatively hard and computationally costly to reproduce, and require
extra code; but the weights are readily available in the form of TF checkpoint files. The
model architecture has not changed, so loading the improved checkpoints is possible.

To use a checkpoint provided at
(https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), first
download the checkpoint. As example, here we download noisy-student version of B1


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

# Image classification using EfficientNet and fine-tuning

**Author:** Yixing Fu<br>
**Date created:** 2020/06/30<br>
**Last modified:** 2020/06/30<br>
**Description:** Use EfficientNet with weights pre-trained on imagenet for CIFAR-100 classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_efficientnet_fine_tuning.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_efficientnet_fine_tuning.py)



# image classification using EfficientNet and fine-tuning

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
from cloud_tpu_client import Client

try:
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

Note: to better see validation peeling off from training accuracy, try increasing epochs
to ~20


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

epochs = 30  # @param {type: "slider", min:5, max:50}
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
Epoch 1/30
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0081s vs `on_train_batch_end` time: 0.0442s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0081s vs `on_train_batch_end` time: 0.0442s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0052s vs `on_test_batch_end` time: 0.0239s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0052s vs `on_test_batch_end` time: 0.0239s). Check your callbacks.

781/781 - 49s - loss: 4.0741 - accuracy: 0.0781 - val_loss: 3.9473 - val_accuracy: 0.1017
Epoch 2/30
781/781 - 45s - loss: 3.3167 - accuracy: 0.1906 - val_loss: 2.9555 - val_accuracy: 0.2670
Epoch 3/30
781/781 - 47s - loss: 2.8729 - accuracy: 0.2733 - val_loss: 2.5347 - val_accuracy: 0.3403
Epoch 4/30
781/781 - 47s - loss: 2.5771 - accuracy: 0.3349 - val_loss: 2.3892 - val_accuracy: 0.3789
Epoch 5/30
781/781 - 47s - loss: 2.3370 - accuracy: 0.3831 - val_loss: 2.1458 - val_accuracy: 0.4329
Epoch 6/30
781/781 - 47s - loss: 2.1519 - accuracy: 0.4254 - val_loss: 2.0155 - val_accuracy: 0.4648
Epoch 7/30
781/781 - 47s - loss: 1.9968 - accuracy: 0.4623 - val_loss: 2.0167 - val_accuracy: 0.4698
Epoch 8/30
781/781 - 47s - loss: 1.8530 - accuracy: 0.4953 - val_loss: 1.9190 - val_accuracy: 0.4921
Epoch 9/30
781/781 - 44s - loss: 1.7393 - accuracy: 0.5230 - val_loss: 1.8984 - val_accuracy: 0.5043
Epoch 10/30
781/781 - 44s - loss: 1.6313 - accuracy: 0.5470 - val_loss: 1.8370 - val_accuracy: 0.5195
Epoch 11/30
781/781 - 44s - loss: 1.5302 - accuracy: 0.5705 - val_loss: 1.8223 - val_accuracy: 0.5263
Epoch 12/30
781/781 - 45s - loss: 1.4380 - accuracy: 0.5967 - val_loss: 1.8480 - val_accuracy: 0.5249
Epoch 13/30
781/781 - 47s - loss: 1.3606 - accuracy: 0.6150 - val_loss: 1.7727 - val_accuracy: 0.5445
Epoch 14/30
781/781 - 45s - loss: 1.2772 - accuracy: 0.6363 - val_loss: 1.7700 - val_accuracy: 0.5535
Epoch 15/30
781/781 - 46s - loss: 1.2002 - accuracy: 0.6554 - val_loss: 1.7437 - val_accuracy: 0.5524
Epoch 16/30
781/781 - 46s - loss: 1.1316 - accuracy: 0.6723 - val_loss: 1.7600 - val_accuracy: 0.5626
Epoch 17/30
781/781 - 44s - loss: 1.0650 - accuracy: 0.6895 - val_loss: 1.8324 - val_accuracy: 0.5612
Epoch 18/30
781/781 - 46s - loss: 0.9986 - accuracy: 0.7089 - val_loss: 1.8082 - val_accuracy: 0.5686
Epoch 19/30
781/781 - 44s - loss: 0.9392 - accuracy: 0.7235 - val_loss: 1.8596 - val_accuracy: 0.5661
Epoch 20/30
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 00020: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
781/781 - 44s - loss: 0.8879 - accuracy: 0.7375 - val_loss: 1.8436 - val_accuracy: 0.5726
Epoch 21/30
781/781 - 46s - loss: 0.6507 - accuracy: 0.8081 - val_loss: 1.5143 - val_accuracy: 0.6267
Epoch 22/30
781/781 - 46s - loss: 0.5729 - accuracy: 0.8344 - val_loss: 1.5188 - val_accuracy: 0.6304
Epoch 23/30
781/781 - 46s - loss: 0.5278 - accuracy: 0.8468 - val_loss: 1.5461 - val_accuracy: 0.6323
Epoch 24/30
781/781 - 46s - loss: 0.4950 - accuracy: 0.8580 - val_loss: 1.5467 - val_accuracy: 0.6342
Epoch 25/30
781/781 - 46s - loss: 0.4653 - accuracy: 0.8659 - val_loss: 1.5705 - val_accuracy: 0.6322
Epoch 26/30
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.007999999821186066.
781/781 - 44s - loss: 0.4440 - accuracy: 0.8716 - val_loss: 1.5969 - val_accuracy: 0.6300
Epoch 27/30
781/781 - 47s - loss: 0.4060 - accuracy: 0.8849 - val_loss: 1.5527 - val_accuracy: 0.6417
Epoch 28/30
781/781 - 45s - loss: 0.3869 - accuracy: 0.8901 - val_loss: 1.5526 - val_accuracy: 0.6427
Epoch 29/30
781/781 - 45s - loss: 0.3840 - accuracy: 0.8926 - val_loss: 1.5545 - val_accuracy: 0.6413
Epoch 30/30
781/781 - 47s - loss: 0.3724 - accuracy: 0.8946 - val_loss: 1.5602 - val_accuracy: 0.6428

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


![png](/img/examples/vision/image_classification_efficientnet_fine_tuning/image_classification_efficientnet_fine_tuning_13_0.png)


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
accuracy and loss. This is because the regularization is relatively strong, and it only
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

epochs = 30  # @param {type: "slider", min:8, max:80}
hist = model.fit(
    ds_train, epochs=epochs, validation_data=ds_test, callbacks=[reduce_lr], verbose=2,
)
plot_hist(hist)
```

<div class="k-default-codeblock">
```
Epoch 1/30
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0050s vs `on_train_batch_end` time: 0.0229s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0050s vs `on_train_batch_end` time: 0.0229s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0053s vs `on_test_batch_end` time: 0.0228s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0053s vs `on_test_batch_end` time: 0.0228s). Check your callbacks.

781/781 - 29s - loss: 2.3756 - accuracy: 0.4435 - val_loss: 1.5460 - val_accuracy: 0.6075
Epoch 2/30
781/781 - 25s - loss: 1.8381 - accuracy: 0.5355 - val_loss: 1.4279 - val_accuracy: 0.6305
Epoch 3/30
781/781 - 24s - loss: 1.6588 - accuracy: 0.5662 - val_loss: 1.3371 - val_accuracy: 0.6436
Epoch 4/30
781/781 - 27s - loss: 1.5331 - accuracy: 0.5874 - val_loss: 1.2881 - val_accuracy: 0.6499
Epoch 5/30
781/781 - 25s - loss: 1.4793 - accuracy: 0.5962 - val_loss: 1.2637 - val_accuracy: 0.6525
Epoch 6/30
781/781 - 26s - loss: 1.4345 - accuracy: 0.6062 - val_loss: 1.2428 - val_accuracy: 0.6574
Epoch 7/30
781/781 - 25s - loss: 1.3948 - accuracy: 0.6158 - val_loss: 1.2417 - val_accuracy: 0.6602
Epoch 8/30
781/781 - 25s - loss: 1.3779 - accuracy: 0.6175 - val_loss: 1.2206 - val_accuracy: 0.6600
Epoch 9/30
781/781 - 25s - loss: 1.3537 - accuracy: 0.6225 - val_loss: 1.2235 - val_accuracy: 0.6650
Epoch 10/30
781/781 - 26s - loss: 1.3482 - accuracy: 0.6243 - val_loss: 1.2112 - val_accuracy: 0.6697
Epoch 11/30
781/781 - 26s - loss: 1.3284 - accuracy: 0.6285 - val_loss: 1.2027 - val_accuracy: 0.6665
Epoch 12/30
781/781 - 25s - loss: 1.3218 - accuracy: 0.6300 - val_loss: 1.2049 - val_accuracy: 0.6674
Epoch 13/30
781/781 - 25s - loss: 1.3160 - accuracy: 0.6326 - val_loss: 1.1988 - val_accuracy: 0.6700
Epoch 14/30
781/781 - 26s - loss: 1.3065 - accuracy: 0.6351 - val_loss: 1.1933 - val_accuracy: 0.6712
Epoch 15/30
781/781 - 26s - loss: 1.3067 - accuracy: 0.6339 - val_loss: 1.1996 - val_accuracy: 0.6684
Epoch 16/30
781/781 - 27s - loss: 1.2887 - accuracy: 0.6395 - val_loss: 1.1903 - val_accuracy: 0.6688
Epoch 17/30
781/781 - 26s - loss: 1.2797 - accuracy: 0.6403 - val_loss: 1.1939 - val_accuracy: 0.6730
Epoch 18/30
781/781 - 26s - loss: 1.2867 - accuracy: 0.6388 - val_loss: 1.1873 - val_accuracy: 0.6715
Epoch 19/30
781/781 - 26s - loss: 1.2773 - accuracy: 0.6399 - val_loss: 1.1935 - val_accuracy: 0.6683
Epoch 20/30
781/781 - 26s - loss: 1.2719 - accuracy: 0.6419 - val_loss: 1.1918 - val_accuracy: 0.6730
Epoch 21/30
781/781 - 26s - loss: 1.2697 - accuracy: 0.6423 - val_loss: 1.1891 - val_accuracy: 0.6716
Epoch 22/30
781/781 - 25s - loss: 1.2683 - accuracy: 0.6437 - val_loss: 1.1914 - val_accuracy: 0.6727
Epoch 23/30
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 00023: ReduceLROnPlateau reducing learning rate to 0.04000000059604645.
781/781 - 25s - loss: 1.2651 - accuracy: 0.6437 - val_loss: 1.1908 - val_accuracy: 0.6699
Epoch 24/30
781/781 - 27s - loss: 1.2035 - accuracy: 0.6608 - val_loss: 1.1589 - val_accuracy: 0.6800
Epoch 25/30
781/781 - 26s - loss: 1.1825 - accuracy: 0.6666 - val_loss: 1.1541 - val_accuracy: 0.6827
Epoch 26/30
781/781 - 26s - loss: 1.1808 - accuracy: 0.6652 - val_loss: 1.1526 - val_accuracy: 0.6815
Epoch 27/30
781/781 - 25s - loss: 1.1838 - accuracy: 0.6647 - val_loss: 1.1489 - val_accuracy: 0.6842
Epoch 28/30
781/781 - 25s - loss: 1.1805 - accuracy: 0.6639 - val_loss: 1.1493 - val_accuracy: 0.6844
Epoch 29/30
781/781 - 26s - loss: 1.1714 - accuracy: 0.6681 - val_loss: 1.1473 - val_accuracy: 0.6869
Epoch 30/30
781/781 - 25s - loss: 1.1710 - accuracy: 0.6695 - val_loss: 1.1478 - val_accuracy: 0.6846

```
</div>
![png](/img/examples/vision/image_classification_efficientnet_fine_tuning/image_classification_efficientnet_fine_tuning_17_5.png)


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
epochs = 30  # @param {type: "slider", min:8, max:80}
hist3 = model.fit(
    ds_train, epochs=epochs, validation_data=ds_test, callbacks=[reduce_lr], verbose=2,
)
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
Epoch 1/30
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0051s vs `on_train_batch_end` time: 0.0334s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0051s vs `on_train_batch_end` time: 0.0334s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0051s vs `on_test_batch_end` time: 0.0222s). Check your callbacks.

WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0051s vs `on_test_batch_end` time: 0.0222s). Check your callbacks.

781/781 - 37s - loss: 1.0832 - accuracy: 0.6904 - val_loss: 0.9120 - val_accuracy: 0.7397
Epoch 2/30
781/781 - 33s - loss: 0.8862 - accuracy: 0.7375 - val_loss: 0.8477 - val_accuracy: 0.7557
Epoch 3/30
781/781 - 33s - loss: 0.8143 - accuracy: 0.7588 - val_loss: 0.8277 - val_accuracy: 0.7621
Epoch 4/30
781/781 - 32s - loss: 0.7590 - accuracy: 0.7721 - val_loss: 0.7909 - val_accuracy: 0.7704
Epoch 5/30
781/781 - 32s - loss: 0.7245 - accuracy: 0.7796 - val_loss: 0.7812 - val_accuracy: 0.7731
Epoch 6/30
781/781 - 34s - loss: 0.6768 - accuracy: 0.7934 - val_loss: 0.7767 - val_accuracy: 0.7766
Epoch 7/30
781/781 - 33s - loss: 0.6429 - accuracy: 0.8044 - val_loss: 0.7469 - val_accuracy: 0.7823
Epoch 8/30
781/781 - 32s - loss: 0.6172 - accuracy: 0.8121 - val_loss: 0.7430 - val_accuracy: 0.7847
Epoch 9/30
781/781 - 32s - loss: 0.5830 - accuracy: 0.8206 - val_loss: 0.7322 - val_accuracy: 0.7899
Epoch 10/30
781/781 - 32s - loss: 0.5598 - accuracy: 0.8272 - val_loss: 0.7280 - val_accuracy: 0.7890
Epoch 11/30
781/781 - 33s - loss: 0.5349 - accuracy: 0.8347 - val_loss: 0.7229 - val_accuracy: 0.7912
Epoch 12/30
781/781 - 33s - loss: 0.5134 - accuracy: 0.8415 - val_loss: 0.7231 - val_accuracy: 0.7930
Epoch 13/30
781/781 - 32s - loss: 0.4944 - accuracy: 0.8461 - val_loss: 0.7246 - val_accuracy: 0.7932
Epoch 14/30
781/781 - 33s - loss: 0.4759 - accuracy: 0.8525 - val_loss: 0.7138 - val_accuracy: 0.7933
Epoch 15/30
781/781 - 33s - loss: 0.4527 - accuracy: 0.8580 - val_loss: 0.7143 - val_accuracy: 0.7956
Epoch 16/30
781/781 - 34s - loss: 0.4474 - accuracy: 0.8613 - val_loss: 0.7156 - val_accuracy: 0.7965
Epoch 17/30
781/781 - 34s - loss: 0.4283 - accuracy: 0.8657 - val_loss: 0.7107 - val_accuracy: 0.7949
Epoch 18/30
781/781 - 32s - loss: 0.4110 - accuracy: 0.8730 - val_loss: 0.6902 - val_accuracy: 0.8018
Epoch 19/30
781/781 - 32s - loss: 0.3964 - accuracy: 0.8774 - val_loss: 0.6890 - val_accuracy: 0.8048
Epoch 20/30
781/781 - 33s - loss: 0.3813 - accuracy: 0.8815 - val_loss: 0.6890 - val_accuracy: 0.8024
Epoch 21/30
781/781 - 32s - loss: 0.3693 - accuracy: 0.8848 - val_loss: 0.6923 - val_accuracy: 0.8033
Epoch 22/30
781/781 - 31s - loss: 0.3563 - accuracy: 0.8900 - val_loss: 0.6945 - val_accuracy: 0.8038
Epoch 23/30
781/781 - 33s - loss: 0.3444 - accuracy: 0.8925 - val_loss: 0.6905 - val_accuracy: 0.8062
Epoch 24/30
781/781 - 32s - loss: 0.3285 - accuracy: 0.8990 - val_loss: 0.6844 - val_accuracy: 0.8071
Epoch 25/30
781/781 - 31s - loss: 0.3226 - accuracy: 0.8995 - val_loss: 0.6898 - val_accuracy: 0.8049
Epoch 26/30
781/781 - 32s - loss: 0.3127 - accuracy: 0.9024 - val_loss: 0.6916 - val_accuracy: 0.8056
Epoch 27/30
781/781 - 33s - loss: 0.3031 - accuracy: 0.9070 - val_loss: 0.6864 - val_accuracy: 0.8075
Epoch 28/30
781/781 - 33s - loss: 0.2884 - accuracy: 0.9126 - val_loss: 0.6951 - val_accuracy: 0.8047
Epoch 29/30
```
</div>
    
<div class="k-default-codeblock">
```
Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.0009999999776482583.
781/781 - 32s - loss: 0.2826 - accuracy: 0.9139 - val_loss: 0.6895 - val_accuracy: 0.8108
Epoch 30/30
781/781 - 34s - loss: 0.2585 - accuracy: 0.9206 - val_loss: 0.6584 - val_accuracy: 0.8160

```
</div>
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


```python
!!wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet\
!       /noisystudent/noisy_student_efficientnet-b1.tar.gz
!!tar -xf noisy_student_efficientnet-b1.tar.gz
```




<div class="k-default-codeblock">
```
['tar: noisy_student_efficientnet-b1.tar.gz: Cannot open: No such file or directory',
 'tar: Error is not recoverable: exiting now']

```
</div>
Then use the script efficientnet_weight_update_util.py to convert ckpt file to h5 file.


```python
!!python efficientnet_weight_update_util.py --model b1 --notop --ckpt \
!        efficientnet-b1/model.ckpt --o efficientnetb1_notop.h5
```




<div class="k-default-codeblock">
```
["python: can't open file 'efficientnet_weight_update_util.py': [Errno 2] No such file or directory"]

```
</div>
When creating model, use the following to load new weight:

```
model = EfficientNetB0(weights="efficientnetb1_notop.h5", include_top=False)
```

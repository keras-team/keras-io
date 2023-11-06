# FixRes: Fixing train-test resolution discrepancy

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2021/10/08<br>
**Last modified:** 2021/10/10<br>
**Description:** Mitigating resolution discrepancy between training and test sets.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/fixres.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/fixres.py)



---
## Introduction

It is a common practice to use the same input image resolution while training and testing
vision models. However, as investigated in
[Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423)
(Touvron et al.), this practice leads to suboptimal performance. Data augmentation
is an indispensable part of the training process of deep neural networks. For vision models, we
typically use random resized crops during training and center crops during inference.
This introduces a discrepancy in the object sizes seen during training and inference.
As shown by Touvron et al., if we can fix this discrepancy, we can significantly
boost model performance.

In this example, we implement the **FixRes** techniques introduced by Touvron et al.
to fix this discrepancy.

---
## Imports


```python
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import matplotlib.pyplot as plt
```

---
## Load the `tf_flowers` dataset


```python
train_dataset, val_dataset = tfds.load(
    "tf_flowers", split=["train[:90%]", "train[90%:]"], as_supervised=True
)

num_train = train_dataset.cardinality()
num_val = val_dataset.cardinality()
print(f"Number of training examples: {num_train}")
print(f"Number of validation examples: {num_val}")
```

<div class="k-default-codeblock">
```
Number of training examples: 3303
Number of validation examples: 367

```
</div>
---
## Data preprocessing utilities

We create three datasets:

1. A dataset with a smaller resolution - 128x128.
2. Two datasets with a larger resolution - 224x224.

We will apply different augmentation transforms to the larger-resolution datasets.

The idea of FixRes is to first train a model on a smaller resolution dataset and then fine-tune
it on a larger resolution dataset. This simple yet effective recipe leads to non-trivial performance
improvements. Please refer to the [original paper](https://arxiv.org/abs/1906.06423) for
results.


```python
# Reference: https://github.com/facebookresearch/FixRes/blob/main/transforms_v2.py.

batch_size = 128
auto = tf.data.AUTOTUNE
smaller_size = 128
bigger_size = 224

size_for_resizing = int((bigger_size / smaller_size) * bigger_size)
central_crop_layer = layers.CenterCrop(bigger_size, bigger_size)


def preprocess_initial(train, image_size):
    """Initial preprocessing function for training on smaller resolution.

    For training, do random_horizontal_flip -> random_crop.
    For validation, just resize.
    No color-jittering has been used.
    """

    def _pp(image, label, train):
        if train:
            channels = image.shape[-1]
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                tf.zeros([0, 0, 4], tf.float32),
                area_range=(0.05, 1.0),
                min_object_covered=0,
                use_image_if_no_bounding_boxes=True,
            )
            image = tf.slice(image, begin, size)

            image.set_shape([None, None, channels])
            image = tf.image.resize(image, [image_size, image_size])
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, [image_size, image_size])

        return image, label

    return _pp


def preprocess_finetune(image, label, train):
    """Preprocessing function for fine-tuning on a higher resolution.

    For training, resize to a bigger resolution to maintain the ratio ->
        random_horizontal_flip -> center_crop.
    For validation, do the same without any horizontal flipping.
    No color-jittering has been used.
    """
    image = tf.image.resize(image, [size_for_resizing, size_for_resizing])
    if train:
        image = tf.image.random_flip_left_right(image)
    image = central_crop_layer(image[None, ...])[0]

    return image, label


def make_dataset(
    dataset: tf.data.Dataset,
    train: bool,
    image_size: int = smaller_size,
    fixres: bool = True,
    num_parallel_calls=auto,
):
    if image_size not in [smaller_size, bigger_size]:
        raise ValueError(f"{image_size} resolution is not supported.")

    # Determine which preprocessing function we are using.
    if image_size == smaller_size:
        preprocess_func = preprocess_initial(train, image_size)
    elif not fixres and image_size == bigger_size:
        preprocess_func = preprocess_initial(train, image_size)
    else:
        preprocess_func = preprocess_finetune

    if train:
        dataset = dataset.shuffle(batch_size * 10)

    return (
        dataset.map(
            lambda x, y: preprocess_func(x, y, train),
            num_parallel_calls=num_parallel_calls,
        )
        .batch(batch_size)
        .prefetch(num_parallel_calls)
    )

```

Notice how the augmentation transforms vary for the kind of dataset we are preparing.

---
## Prepare datasets


```python
initial_train_dataset = make_dataset(train_dataset, train=True, image_size=smaller_size)
initial_val_dataset = make_dataset(val_dataset, train=False, image_size=smaller_size)

finetune_train_dataset = make_dataset(train_dataset, train=True, image_size=bigger_size)
finetune_val_dataset = make_dataset(val_dataset, train=False, image_size=bigger_size)

vanilla_train_dataset = make_dataset(
    train_dataset, train=True, image_size=bigger_size, fixres=False
)
vanilla_val_dataset = make_dataset(
    val_dataset, train=False, image_size=bigger_size, fixres=False
)
```

---
## Visualize the datasets


```python

def visualize_dataset(batch_images):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(batch_images[n].numpy().astype("int"))
        plt.axis("off")
    plt.show()

    print(f"Batch shape: {batch_images.shape}.")


# Smaller resolution.
initial_sample_images, _ = next(iter(initial_train_dataset))
visualize_dataset(initial_sample_images)

# Bigger resolution, only for fine-tuning.
finetune_sample_images, _ = next(iter(finetune_train_dataset))
visualize_dataset(finetune_sample_images)

# Bigger resolution, with the same augmentation transforms as
# the smaller resolution dataset.
vanilla_sample_images, _ = next(iter(vanilla_train_dataset))
visualize_dataset(vanilla_sample_images)
```

<div class="k-default-codeblock">
```
2021-10-11 02:05:26.638594: W tensorflow/core/kernels/data/cache_dataset_ops.cc:768] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.

```
</div>
    
![png](/img/examples/vision/fixres/fixres_13_1.png)
    


<div class="k-default-codeblock">
```
Batch shape: (128, 128, 128, 3).

2021-10-11 02:05:28.509752: W tensorflow/core/kernels/data/cache_dataset_ops.cc:768] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.

```
</div>
    
![png](/img/examples/vision/fixres/fixres_13_4.png)
    


<div class="k-default-codeblock">
```
Batch shape: (128, 224, 224, 3).

2021-10-11 02:05:30.108623: W tensorflow/core/kernels/data/cache_dataset_ops.cc:768] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.

```
</div>
    
![png](/img/examples/vision/fixres/fixres_13_7.png)
    


<div class="k-default-codeblock">
```
Batch shape: (128, 224, 224, 3).

```
</div>
---
## Model training utilities

We train multiple variants of ResNet50V2
([He et al.](https://arxiv.org/abs/1603.05027)):

1. On the smaller resolution dataset (128x128). It will be trained from scratch.
2. Then fine-tune the model from 1 on the larger resolution (224x224) dataset.
3. Train another ResNet50V2 from scratch on the larger resolution dataset.

As a reminder, the larger resolution datasets differ in terms of their augmentation
transforms.


```python

def get_training_model(num_classes=5):
    inputs = layers.Input((None, None, 3))
    resnet_base = keras.applications.ResNet50V2(
        include_top=False, weights=None, pooling="avg"
    )
    resnet_base.trainable = True

    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)
    x = resnet_base(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def train_and_evaluate(
    model, train_ds, val_ds, epochs, learning_rate=1e-3, use_early_stopping=False
):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if use_early_stopping:
        es_callback = keras.callbacks.EarlyStopping(patience=5)
        callbacks = [es_callback]
    else:
        callbacks = None

    model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks,
    )

    _, accuracy = model.evaluate(val_ds)
    print(f"Top-1 accuracy on the validation set: {accuracy*100:.2f}%.")
    return model

```

---
## Experiment 1: Train on 128x128 and then fine-tune on 224x224


```python
epochs = 30

smaller_res_model = get_training_model()
smaller_res_model = train_and_evaluate(
    smaller_res_model, initial_train_dataset, initial_val_dataset, epochs
)
```

<div class="k-default-codeblock">
```
Epoch 1/30
26/26 [==============================] - 14s 226ms/step - loss: 1.6476 - accuracy: 0.4345 - val_loss: 9.8213 - val_accuracy: 0.2044
Epoch 2/30
26/26 [==============================] - 3s 123ms/step - loss: 1.1561 - accuracy: 0.5495 - val_loss: 6.5521 - val_accuracy: 0.2071
Epoch 3/30
26/26 [==============================] - 3s 123ms/step - loss: 1.0989 - accuracy: 0.5722 - val_loss: 2.6216 - val_accuracy: 0.1935
Epoch 4/30
26/26 [==============================] - 3s 122ms/step - loss: 1.0373 - accuracy: 0.5895 - val_loss: 1.9918 - val_accuracy: 0.2125
Epoch 5/30
26/26 [==============================] - 3s 122ms/step - loss: 0.9960 - accuracy: 0.6119 - val_loss: 2.8505 - val_accuracy: 0.2262
Epoch 6/30
26/26 [==============================] - 3s 122ms/step - loss: 0.9458 - accuracy: 0.6331 - val_loss: 1.8974 - val_accuracy: 0.2834
Epoch 7/30
26/26 [==============================] - 3s 122ms/step - loss: 0.8949 - accuracy: 0.6606 - val_loss: 2.1164 - val_accuracy: 0.2834
Epoch 8/30
26/26 [==============================] - 3s 122ms/step - loss: 0.8581 - accuracy: 0.6709 - val_loss: 1.8858 - val_accuracy: 0.3815
Epoch 9/30
26/26 [==============================] - 3s 123ms/step - loss: 0.8436 - accuracy: 0.6776 - val_loss: 1.5671 - val_accuracy: 0.4687
Epoch 10/30
26/26 [==============================] - 3s 123ms/step - loss: 0.8632 - accuracy: 0.6685 - val_loss: 1.5005 - val_accuracy: 0.5504
Epoch 11/30
26/26 [==============================] - 3s 123ms/step - loss: 0.8316 - accuracy: 0.6918 - val_loss: 1.1421 - val_accuracy: 0.6594
Epoch 12/30
26/26 [==============================] - 3s 123ms/step - loss: 0.7981 - accuracy: 0.6951 - val_loss: 1.2036 - val_accuracy: 0.6403
Epoch 13/30
26/26 [==============================] - 3s 122ms/step - loss: 0.8275 - accuracy: 0.6806 - val_loss: 2.2632 - val_accuracy: 0.5177
Epoch 14/30
26/26 [==============================] - 3s 122ms/step - loss: 0.8156 - accuracy: 0.6994 - val_loss: 1.1023 - val_accuracy: 0.6649
Epoch 15/30
26/26 [==============================] - 3s 122ms/step - loss: 0.7572 - accuracy: 0.7091 - val_loss: 1.6248 - val_accuracy: 0.6049
Epoch 16/30
26/26 [==============================] - 3s 123ms/step - loss: 0.7757 - accuracy: 0.7024 - val_loss: 2.0600 - val_accuracy: 0.6294
Epoch 17/30
26/26 [==============================] - 3s 122ms/step - loss: 0.7600 - accuracy: 0.7087 - val_loss: 1.5731 - val_accuracy: 0.6131
Epoch 18/30
26/26 [==============================] - 3s 122ms/step - loss: 0.7385 - accuracy: 0.7215 - val_loss: 1.8312 - val_accuracy: 0.5749
Epoch 19/30
26/26 [==============================] - 3s 122ms/step - loss: 0.7493 - accuracy: 0.7224 - val_loss: 3.0382 - val_accuracy: 0.4986
Epoch 20/30
26/26 [==============================] - 3s 122ms/step - loss: 0.7746 - accuracy: 0.7048 - val_loss: 7.8191 - val_accuracy: 0.5123
Epoch 21/30
26/26 [==============================] - 3s 123ms/step - loss: 0.7367 - accuracy: 0.7405 - val_loss: 1.9607 - val_accuracy: 0.6676
Epoch 22/30
26/26 [==============================] - 3s 122ms/step - loss: 0.6970 - accuracy: 0.7357 - val_loss: 3.1944 - val_accuracy: 0.4496
Epoch 23/30
26/26 [==============================] - 3s 122ms/step - loss: 0.7299 - accuracy: 0.7212 - val_loss: 1.4012 - val_accuracy: 0.6567
Epoch 24/30
26/26 [==============================] - 3s 122ms/step - loss: 0.6965 - accuracy: 0.7315 - val_loss: 1.9781 - val_accuracy: 0.6403
Epoch 25/30
26/26 [==============================] - 3s 124ms/step - loss: 0.6811 - accuracy: 0.7408 - val_loss: 0.9287 - val_accuracy: 0.6839
Epoch 26/30
26/26 [==============================] - 3s 123ms/step - loss: 0.6732 - accuracy: 0.7487 - val_loss: 2.9406 - val_accuracy: 0.5504
Epoch 27/30
26/26 [==============================] - 3s 122ms/step - loss: 0.6571 - accuracy: 0.7560 - val_loss: 1.6268 - val_accuracy: 0.5804
Epoch 28/30
26/26 [==============================] - 3s 122ms/step - loss: 0.6662 - accuracy: 0.7548 - val_loss: 0.9067 - val_accuracy: 0.7357
Epoch 29/30
26/26 [==============================] - 3s 122ms/step - loss: 0.6443 - accuracy: 0.7520 - val_loss: 0.7760 - val_accuracy: 0.7520
Epoch 30/30
26/26 [==============================] - 3s 122ms/step - loss: 0.6617 - accuracy: 0.7539 - val_loss: 0.6026 - val_accuracy: 0.7766
3/3 [==============================] - 0s 37ms/step - loss: 0.6026 - accuracy: 0.7766
Top-1 accuracy on the validation set: 77.66%.

```
</div>
### Freeze all the layers except for the final Batch Normalization layer

For fine-tuning, we train only two layers:

* The final Batch Normalization ([Ioffe et al.](https://arxiv.org/abs/1502.03167)) layer.
* The classification layer.

We are unfreezing the final Batch Normalization layer to compensate for the change in
activation statistics before the global average pooling layer. As shown in
[the paper](https://arxiv.org/abs/1906.06423), unfreezing the final Batch
Normalization layer is enough.

For a comprehensive guide on fine-tuning models in Keras, refer to
[this tutorial](https://keras.io/guides/transfer_learning/).


```python
for layer in smaller_res_model.layers[2].layers:
    layer.trainable = False

smaller_res_model.layers[2].get_layer("post_bn").trainable = True

epochs = 10

# Use a lower learning rate during fine-tuning.
bigger_res_model = train_and_evaluate(
    smaller_res_model,
    finetune_train_dataset,
    finetune_val_dataset,
    epochs,
    learning_rate=1e-4,
)
```

<div class="k-default-codeblock">
```
Epoch 1/10
26/26 [==============================] - 9s 201ms/step - loss: 0.7912 - accuracy: 0.7856 - val_loss: 0.6808 - val_accuracy: 0.7575
Epoch 2/10
26/26 [==============================] - 3s 115ms/step - loss: 0.7732 - accuracy: 0.7938 - val_loss: 0.7028 - val_accuracy: 0.7684
Epoch 3/10
26/26 [==============================] - 3s 115ms/step - loss: 0.7658 - accuracy: 0.7923 - val_loss: 0.7136 - val_accuracy: 0.7629
Epoch 4/10
26/26 [==============================] - 3s 115ms/step - loss: 0.7536 - accuracy: 0.7872 - val_loss: 0.7161 - val_accuracy: 0.7684
Epoch 5/10
26/26 [==============================] - 3s 115ms/step - loss: 0.7346 - accuracy: 0.7947 - val_loss: 0.7154 - val_accuracy: 0.7711
Epoch 6/10
26/26 [==============================] - 3s 115ms/step - loss: 0.7183 - accuracy: 0.7990 - val_loss: 0.7139 - val_accuracy: 0.7684
Epoch 7/10
26/26 [==============================] - 3s 116ms/step - loss: 0.7059 - accuracy: 0.7962 - val_loss: 0.7071 - val_accuracy: 0.7738
Epoch 8/10
26/26 [==============================] - 3s 115ms/step - loss: 0.6959 - accuracy: 0.7923 - val_loss: 0.7002 - val_accuracy: 0.7738
Epoch 9/10
26/26 [==============================] - 3s 116ms/step - loss: 0.6871 - accuracy: 0.8011 - val_loss: 0.6967 - val_accuracy: 0.7711
Epoch 10/10
26/26 [==============================] - 3s 116ms/step - loss: 0.6761 - accuracy: 0.8044 - val_loss: 0.6887 - val_accuracy: 0.7738
3/3 [==============================] - 0s 95ms/step - loss: 0.6887 - accuracy: 0.7738
Top-1 accuracy on the validation set: 77.38%.

```
</div>
---
## Experiment 2: Train a model on 224x224 resolution from scratch

Now, we train another model from scratch on the larger resolution dataset. Recall that
the augmentation transforms used in this dataset are different from before.


```python
epochs = 30

vanilla_bigger_res_model = get_training_model()
vanilla_bigger_res_model = train_and_evaluate(
    vanilla_bigger_res_model, vanilla_train_dataset, vanilla_val_dataset, epochs
)
```

<div class="k-default-codeblock">
```
Epoch 1/30
26/26 [==============================] - 15s 389ms/step - loss: 1.5339 - accuracy: 0.4569 - val_loss: 177.5233 - val_accuracy: 0.1907
Epoch 2/30
26/26 [==============================] - 8s 314ms/step - loss: 1.1472 - accuracy: 0.5483 - val_loss: 17.5804 - val_accuracy: 0.1907
Epoch 3/30
26/26 [==============================] - 8s 315ms/step - loss: 1.0708 - accuracy: 0.5792 - val_loss: 2.2719 - val_accuracy: 0.2480
Epoch 4/30
26/26 [==============================] - 8s 315ms/step - loss: 1.0225 - accuracy: 0.6170 - val_loss: 2.1274 - val_accuracy: 0.2398
Epoch 5/30
26/26 [==============================] - 8s 316ms/step - loss: 1.0001 - accuracy: 0.6206 - val_loss: 2.0375 - val_accuracy: 0.2834
Epoch 6/30
26/26 [==============================] - 8s 315ms/step - loss: 0.9602 - accuracy: 0.6355 - val_loss: 1.4412 - val_accuracy: 0.3978
Epoch 7/30
26/26 [==============================] - 8s 316ms/step - loss: 0.9418 - accuracy: 0.6461 - val_loss: 1.5257 - val_accuracy: 0.4305
Epoch 8/30
26/26 [==============================] - 8s 316ms/step - loss: 0.8911 - accuracy: 0.6649 - val_loss: 1.1530 - val_accuracy: 0.5858
Epoch 9/30
26/26 [==============================] - 8s 316ms/step - loss: 0.8834 - accuracy: 0.6694 - val_loss: 1.2026 - val_accuracy: 0.5531
Epoch 10/30
26/26 [==============================] - 8s 316ms/step - loss: 0.8752 - accuracy: 0.6724 - val_loss: 1.4917 - val_accuracy: 0.5695
Epoch 11/30
26/26 [==============================] - 8s 316ms/step - loss: 0.8690 - accuracy: 0.6594 - val_loss: 1.4115 - val_accuracy: 0.6022
Epoch 12/30
26/26 [==============================] - 8s 314ms/step - loss: 0.8586 - accuracy: 0.6761 - val_loss: 1.0692 - val_accuracy: 0.6349
Epoch 13/30
26/26 [==============================] - 8s 315ms/step - loss: 0.8120 - accuracy: 0.6894 - val_loss: 1.5233 - val_accuracy: 0.6567
Epoch 14/30
26/26 [==============================] - 8s 316ms/step - loss: 0.8275 - accuracy: 0.6857 - val_loss: 1.9079 - val_accuracy: 0.5804
Epoch 15/30
26/26 [==============================] - 8s 316ms/step - loss: 0.7624 - accuracy: 0.7127 - val_loss: 0.9543 - val_accuracy: 0.6540
Epoch 16/30
26/26 [==============================] - 8s 315ms/step - loss: 0.7595 - accuracy: 0.7266 - val_loss: 4.5757 - val_accuracy: 0.4877
Epoch 17/30
26/26 [==============================] - 8s 315ms/step - loss: 0.7577 - accuracy: 0.7154 - val_loss: 1.8411 - val_accuracy: 0.5749
Epoch 18/30
26/26 [==============================] - 8s 316ms/step - loss: 0.7596 - accuracy: 0.7163 - val_loss: 1.0660 - val_accuracy: 0.6703
Epoch 19/30
26/26 [==============================] - 8s 315ms/step - loss: 0.7492 - accuracy: 0.7160 - val_loss: 1.2462 - val_accuracy: 0.6485
Epoch 20/30
26/26 [==============================] - 8s 315ms/step - loss: 0.7269 - accuracy: 0.7330 - val_loss: 5.8287 - val_accuracy: 0.3379
Epoch 21/30
26/26 [==============================] - 8s 315ms/step - loss: 0.7193 - accuracy: 0.7275 - val_loss: 4.7058 - val_accuracy: 0.6049
Epoch 22/30
26/26 [==============================] - 8s 316ms/step - loss: 0.7251 - accuracy: 0.7318 - val_loss: 1.5608 - val_accuracy: 0.6485
Epoch 23/30
26/26 [==============================] - 8s 314ms/step - loss: 0.6888 - accuracy: 0.7466 - val_loss: 1.7914 - val_accuracy: 0.6240
Epoch 24/30
26/26 [==============================] - 8s 314ms/step - loss: 0.7051 - accuracy: 0.7339 - val_loss: 2.0918 - val_accuracy: 0.6158
Epoch 25/30
26/26 [==============================] - 8s 315ms/step - loss: 0.6920 - accuracy: 0.7454 - val_loss: 0.7284 - val_accuracy: 0.7575
Epoch 26/30
26/26 [==============================] - 8s 316ms/step - loss: 0.6502 - accuracy: 0.7523 - val_loss: 2.5474 - val_accuracy: 0.5313
Epoch 27/30
26/26 [==============================] - 8s 315ms/step - loss: 0.7101 - accuracy: 0.7330 - val_loss: 26.8117 - val_accuracy: 0.3297
Epoch 28/30
26/26 [==============================] - 8s 315ms/step - loss: 0.6632 - accuracy: 0.7548 - val_loss: 20.1011 - val_accuracy: 0.3243
Epoch 29/30
26/26 [==============================] - 8s 315ms/step - loss: 0.6682 - accuracy: 0.7505 - val_loss: 11.5872 - val_accuracy: 0.3297
Epoch 30/30
26/26 [==============================] - 8s 315ms/step - loss: 0.6758 - accuracy: 0.7514 - val_loss: 5.7229 - val_accuracy: 0.4305
3/3 [==============================] - 0s 95ms/step - loss: 5.7229 - accuracy: 0.4305
Top-1 accuracy on the validation set: 43.05%.

```
</div>
As we can notice from the above cells, FixRes leads to a better performance. Another
advantage of FixRes is the improved total training time and reduction in GPU memory usage.
FixRes is model-agnostic, you can use it on any image classification model
to potentially boost performance.

You can find more results
[here](https://tensorboard.dev/experiment/BQOg28w0TlmvuJYeqsVntw)
that were gathered by running the same code with different random seeds.

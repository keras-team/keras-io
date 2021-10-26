# MixUp augmentation for image classification

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2021/03/06<br>
**Last modified:** 2021/03/06<br>
**Description:** Data augmentation using the mixup technique for image classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/mixup.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/mixup.py)



---
## Introduction

_mixup_ is a *domain-agnostic* data augmentation technique proposed in [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
by Zhang et al. It's implemented with the following formulas:

![](https://i.ibb.co/DRyHYww/image.png)

(Note that the lambda values are values with the [0, 1] range and are sampled from the
[Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution).)

The technique is quite systematically named - we are literally mixing up the features and
their corresponding labels. Implementation-wise it's simple. Neural networks are prone
to [memorizing corrupt labels](https://arxiv.org/abs/1611.03530). mixup relaxes this by
combining different features with one another (same happens for the labels too) so that
a network does not get overconfident about the relationship between the features and
their labels.

mixup is specifically useful when we are not sure about selecting a set of augmentation
transforms for a given dataset, medical imaging datasets, for example. mixup can be
extended to a variety of data modalities such as computer vision, naturallanguage
processing, speech, and so on.

This example requires TensorFlow 2.4 or higher.

---
## Setup


```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
```

---
## Prepare the dataset

In this example, we will be using the [FashionMNIST](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/) dataset. But this same recipe can
be used for other classification datasets as well.


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))
y_train = tf.one_hot(y_train, 10)

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))
y_test = tf.one_hot(y_test, 10)
```

---
## Define hyperparameters


```python
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 64
EPOCHS = 10
```

---
## Convert the data into TensorFlow `Dataset` objects


```python
# Put aside a few samples to create our validation set
val_samples = 2000
x_val, y_val = x_train[:val_samples], y_train[:val_samples]
new_x_train, new_y_train = x_train[val_samples:], y_train[val_samples:]

train_ds_one = (
    tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
)
train_ds_two = (
    tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train))
    .shuffle(BATCH_SIZE * 100)
    .batch(BATCH_SIZE)
)
# Because we will be mixing up the images and their corresponding labels, we will be
# combining two shuffled datasets from the same training data.
train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
```

---
## Define the mixup technique function

To perform the mixup routine, we create new virtual datasets using the training data from
the same dataset, and apply a lambda value within the [0, 1] range sampled from a [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
— such that, for example, `new_x = lambda * x1 + (1 - lambda) * x2` (where
`x1` and `x2` are images) and the same equation is applied to the labels as well.


```python

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)

```

**Note** that here , we are combining two images to create a single one. Theoretically,
we can combine as many we want but that comes at an increased computation cost. In
certain cases, it may not help improve the performance as well.

---
## Visualize the new augmented dataset


```python
# First create the new dataset using our `mix_up` utility
train_ds_mu = train_ds.map(
    lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
)

# Let's preview 9 samples from the dataset
sample_images, sample_labels = next(iter(train_ds_mu))
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().squeeze())
    print(label.numpy().tolist())
    plt.axis("off")
```

<div class="k-default-codeblock">
```
[0.01706075668334961, 0.0, 0.0, 0.9829392433166504, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.5761554837226868, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42384451627731323, 0.0]
[0.0, 0.0, 0.9999957084655762, 0.0, 4.291534423828125e-06, 0.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.03438800573348999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.96561199426651, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.9808260202407837, 0.0, 0.0, 0.0, 0.01917397230863571, 0.0, 0.0, 0.0]
[0.0, 0.9999748468399048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5153160095214844e-05]
[0.0, 0.0, 0.0, 0.0002035107754636556, 0.0, 0.9997965097427368, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2410212755203247, 0.0, 0.0, 0.7589787244796753]

```
</div>
![png](/img/examples/vision/mixup/mixup_15_1.png)


---
## Model building


```python

def get_training_model():
    model = tf.keras.Sequential(
        [
            layers.Conv2D(16, (5, 5), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, (5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.2),
            layers.GlobalAvgPool2D(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model

```

For the sake of reproducibility, we serialize the initial random weights of our shallow
network.


```python
initial_model = get_training_model()
initial_model.save_weights("initial_weights.h5")
```

---
## 1. Train the model with the mixed up dataset


```python
model = get_training_model()
model.load_weights("initial_weights.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_ds_mu, validation_data=val_ds, epochs=EPOCHS)
_, test_acc = model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_acc * 100))
```

<div class="k-default-codeblock">
```
Epoch 1/10
907/907 [==============================] - 38s 41ms/step - loss: 1.4440 - accuracy: 0.5173 - val_loss: 0.7120 - val_accuracy: 0.7405
Epoch 2/10
907/907 [==============================] - 38s 42ms/step - loss: 0.9869 - accuracy: 0.7074 - val_loss: 0.5996 - val_accuracy: 0.7780
Epoch 3/10
907/907 [==============================] - 38s 42ms/step - loss: 0.9096 - accuracy: 0.7451 - val_loss: 0.5197 - val_accuracy: 0.8285
Epoch 4/10
907/907 [==============================] - 38s 42ms/step - loss: 0.8485 - accuracy: 0.7741 - val_loss: 0.4830 - val_accuracy: 0.8380
Epoch 5/10
907/907 [==============================] - 38s 42ms/step - loss: 0.8032 - accuracy: 0.7916 - val_loss: 0.4543 - val_accuracy: 0.8445
Epoch 6/10
907/907 [==============================] - 38s 42ms/step - loss: 0.7675 - accuracy: 0.8032 - val_loss: 0.4398 - val_accuracy: 0.8470
Epoch 7/10
907/907 [==============================] - 38s 42ms/step - loss: 0.7474 - accuracy: 0.8098 - val_loss: 0.4262 - val_accuracy: 0.8495
Epoch 8/10
907/907 [==============================] - 38s 42ms/step - loss: 0.7337 - accuracy: 0.8145 - val_loss: 0.3950 - val_accuracy: 0.8650
Epoch 9/10
907/907 [==============================] - 38s 42ms/step - loss: 0.7154 - accuracy: 0.8218 - val_loss: 0.3822 - val_accuracy: 0.8725
Epoch 10/10
907/907 [==============================] - 38s 42ms/step - loss: 0.7095 - accuracy: 0.8224 - val_loss: 0.3563 - val_accuracy: 0.8720
157/157 [==============================] - 2s 14ms/step - loss: 0.3821 - accuracy: 0.8726
Test accuracy: 87.26%

```
</div>
---
## 2. Train the model *without* the mixed up dataset


```python
model = get_training_model()
model.load_weights("initial_weights.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Notice that we are NOT using the mixed up dataset here
model.fit(train_ds_one, validation_data=val_ds, epochs=EPOCHS)
_, test_acc = model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_acc * 100))
```

<div class="k-default-codeblock">
```
Epoch 1/10
907/907 [==============================] - 37s 40ms/step - loss: 1.2037 - accuracy: 0.5553 - val_loss: 0.6732 - val_accuracy: 0.7565
Epoch 2/10
907/907 [==============================] - 37s 40ms/step - loss: 0.6724 - accuracy: 0.7462 - val_loss: 0.5715 - val_accuracy: 0.7940
Epoch 3/10
907/907 [==============================] - 37s 40ms/step - loss: 0.5828 - accuracy: 0.7897 - val_loss: 0.5042 - val_accuracy: 0.8210
Epoch 4/10
907/907 [==============================] - 37s 40ms/step - loss: 0.5203 - accuracy: 0.8115 - val_loss: 0.4587 - val_accuracy: 0.8405
Epoch 5/10
907/907 [==============================] - 36s 40ms/step - loss: 0.4802 - accuracy: 0.8255 - val_loss: 0.4602 - val_accuracy: 0.8340
Epoch 6/10
907/907 [==============================] - 36s 40ms/step - loss: 0.4566 - accuracy: 0.8351 - val_loss: 0.3985 - val_accuracy: 0.8700
Epoch 7/10
907/907 [==============================] - 37s 40ms/step - loss: 0.4273 - accuracy: 0.8457 - val_loss: 0.3764 - val_accuracy: 0.8685
Epoch 8/10
907/907 [==============================] - 36s 40ms/step - loss: 0.4133 - accuracy: 0.8481 - val_loss: 0.3704 - val_accuracy: 0.8735
Epoch 9/10
907/907 [==============================] - 36s 40ms/step - loss: 0.3951 - accuracy: 0.8543 - val_loss: 0.3715 - val_accuracy: 0.8680
Epoch 10/10
907/907 [==============================] - 36s 40ms/step - loss: 0.3850 - accuracy: 0.8586 - val_loss: 0.3458 - val_accuracy: 0.8735
157/157 [==============================] - 2s 13ms/step - loss: 0.3817 - accuracy: 0.8636
Test accuracy: 86.36%

```
</div>
Readers are encouraged to try out mixup on different datasets from different domains and
experiment with the lambda parameter. You are strongly advised to check out the
[original paper](https://arxiv.org/abs/1710.09412) as well - the authors present several ablation studies on mixup
showing how it can improve generalization, as well as show their results of combining
more than two images to create a single one.

---
## Notes

* With mixup, you can create synthetic examples — especially when you lack a large
dataset - without incurring high computational costs.
* [Label smoothing](https://www.pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/) and mixup usually do not work well together because label smoothing
already modifies the hard labels by some factor.
* mixup does not work well when you are using [Supervised Contrastive
Learning](https://arxiv.org/abs/2004.11362) (SCL) since SCL expects the true labels
during its pre-training phase.
* A few other benefits of mixup include (as described in the [paper](https://arxiv.org/abs/1710.09412)) robustness to
adversarial examples and stabilized GAN (Generative Adversarial Networks) training.
* There are a number of data augmentation techniques that extend mixup such as
[CutMix](https://arxiv.org/abs/1905.04899) and [AugMix](https://arxiv.org/abs/1912.02781).

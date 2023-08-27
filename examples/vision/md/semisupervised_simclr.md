# [KerasCV] Semi-supervised image classification using contrastive pretraining with SimCLR

**Author:** [András Béres](https://www.linkedin.com/in/andras-beres-789190210), updated by [Aritra Roy Gosthipaty](https://twitter.com/ariG23498)<br>
**Date created:** 2021/04/24<br>
**Last modified:** 2023/07/06<br>
**Description:** Contrastive pretraining with SimCLR for semi-supervised image classification on the STL-10 dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/semisupervised_simclr.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/semisupervised_simclr.py)



---
## Introduction

### Semi-supervised learning

Semi-supervised learning is a machine learning paradigm that deals with
**partially labeled datasets**. When applying deep learning in the real world,
one usually has to gather a large dataset to make it work well. However, while
the cost of labeling scales linearly with the dataset size (labeling each
example takes a constant time), model performance only scales
[sublinearly](https://arxiv.org/abs/2001.08361) with it. This means that
labeling more and more samples becomes less and less cost-efficient, while
gathering unlabeled data is generally cheap, as it is usually readily available
in large quantities.

Semi-supervised learning offers to solve this problem by only requiring a
partially labeled dataset, and by being label-efficient by utilizing the
unlabeled examples for learning as well.

In this example, we will pretrain an encoder with contrastive learning on the
[STL-10](https://ai.stanford.edu/~acoates/stl10/) semi-supervised dataset using
no labels at all, and then fine-tune it using only its labeled subset.

### Contrastive learning

On the highest level, the main idea behind contrastive learning is to **learn
representations that are invariant to image augmentations** in a self-supervised
manner. One problem with this objective is that it has a trivial degenerate
solution: the case where the representations are constant, and do not depend at all on the
input images.

Contrastive learning avoids this trap by modifying the objective in the
following way: it pulls representations of augmented versions/views of the same
image closer to each other (contracting positives), while simultaneously pushing
different images away from each other (contrasting negatives) in representation
space.

One such contrastive approach is [SimCLR](https://arxiv.org/abs/2002.05709),
which essentially identifies the core components needed to optimize this
objective, and can achieve high performance by scaling this simple approach.

Another approach is [SimSiam](https://arxiv.org/abs/2011.10566)
([Keras example](https://keras.io/examples/vision/simsiam/)),
whose main difference from
SimCLR is that the former does not use any negatives in its loss. Therefore, it does not
explicitly prevent the trivial solution, and, instead, avoids it implicitly by
architecture design (asymmetric encoding paths using a predictor network and
batch normalization (BatchNorm) are applied in the final layers).

For further reading about SimCLR, check out
[the official Google AI blog post](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html),
and for an overview of self-supervised learning across both vision and language
check out
[this blog post](https://ai.facebook.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/).

---
## Setup

For this tutorial we will need [KerasCV](https://keras.io/keras_cv/) which can be installed with the following command:
`pip install keras-cv`


```python
import keras
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

tfds.disable_progress_bar()
```

---
## Hyperparameter setup

Please feel free to change the hyperparameters and train the model. Here we make the following choices
due to hardware restrictions and good training logs.


```python
# Dataset hyperparameters
IMAGE_SIZE = 96
IMAGE_CHANNELS = 3
NUM_CLASSES = 10

# Algorithm hyperparameter
UNLABELED_BATCH_SIZE = 1024
LABELED_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
PROJECTION_WIDTH = 128
TEMPERATURE = 0.1

# Stronger augmentations for contrastive
CONTRASTIVE_AUGMENTATION = {
    "crop_area_factor": (0.08, 1.0),
    "aspect_ratio_factor": (3 / 4, 4 / 3),
    "color_jitter_rate": 0.8,
    "brightness_factor": 0.2,
    "contrast_factor": 0.8,
    "saturation_factor": (0.3, 0.7),
    "hue_factor": 0.2,
}

# Weaker ones for supervised training
CLASSIFICATION_AUGMENTATION = {
    "crop_area_factor": (0.8, 1.0),
    "aspect_ratio_factor": (3 / 4, 4 / 3),
    "color_jitter_rate": 0.05,
    "brightness_factor": 0.1,
    "contrast_factor": 0.1,
    "saturation_factor": (0.1, 0.1),
    "hue_factor": 0.2,
}

AUTOTUNE = tf.data.AUTOTUNE
```

---
## Dataset

The dataset has three splits:
- Training Unlabelled: This dataset is used to train the encoder in the contrastive setting.
- Training Lablelled: This dataset is used to train the baseline encoder (supervised) and also
    fine tune the pre-trained encoder.
- Testing Labelled: This dataset is used to evaluate the models.


```python

def prepare_dataset():
    unlabeled_train_dataset = (
        tfds.load("stl10", data_dir="dataset", split="unlabelled", as_supervised=True)
        .map(lambda image, _: image, num_parallel_calls=AUTOTUNE)
        .shuffle(buffer_size=2 * UNLABELED_BATCH_SIZE)
        .batch(UNLABELED_BATCH_SIZE, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    labeled_train_dataset = (
        tfds.load("stl10", data_dir="dataset", split="train", as_supervised=True)
        .shuffle(buffer_size=10 * LABELED_BATCH_SIZE)
        .batch(LABELED_BATCH_SIZE, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    test_dataset = (
        tfds.load("stl10", data_dir="dataset", split="test", as_supervised=True)
        .batch(TEST_BATCH_SIZE, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    return unlabeled_train_dataset, labeled_train_dataset, test_dataset


# Load STL10 dataset
unlabeled_train_dataset, labeled_train_dataset, test_dataset = prepare_dataset()
```

<div class="k-default-codeblock">
```
 Downloading and preparing dataset 2.46 GiB (download: 2.46 GiB, generated: 1.86 GiB, total: 4.32 GiB) to dataset/stl10/1.0.0...
 Dataset stl10 downloaded and prepared to dataset/stl10/1.0.0. Subsequent calls will reuse this data.

```
</div>
---
## Image augmentations

The two most important image augmentations for contrastive learning are the
following:

- **Cropping**: forces the model to encode different parts of the same image
similarly.
- **Color jitter**: prevents a trivial color histogram-based solution to the task by
distorting color histograms. A principled way to implement that is by affine
transformations in color space.

Stronger augmentations are applied for contrastive learning, along with weaker
ones for supervised classification to avoid overfitting on the few labeled examples.

We implement the augmentations using the KerasCV library.


```python

def get_augmenter(
    crop_area_factor,
    aspect_ratio_factor,
    color_jitter_rate,
    brightness_factor,
    contrast_factor,
    saturation_factor,
    hue_factor,
):
    return keras.Sequential(
        [
            keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)),
            keras_cv.layers.Rescaling(scale=1.0 / 255),
            keras_cv.layers.RandomFlip("horizontal"),
            keras_cv.layers.RandomCropAndResize(
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                crop_area_factor=crop_area_factor,
                aspect_ratio_factor=aspect_ratio_factor,
            ),
            keras_cv.layers.RandomApply(
                keras_cv.layers.RandomColorJitter(
                    value_range=(0, 1),
                    brightness_factor=brightness_factor,
                    contrast_factor=contrast_factor,
                    saturation_factor=saturation_factor,
                    hue_factor=hue_factor,
                ),
                rate=color_jitter_rate,
            ),
        ]
    )

```

---
## Visualize the dataset

Let's first visualize the original dataset.


```python
# Original Images
unlabeled_images = next(iter(unlabeled_train_dataset))
keras_cv.visualization.plot_image_gallery(
    images=unlabeled_images,
    value_range=(0, 255),
    rows=3,
    cols=3,
)
```


    
![png](/img/examples/vision/semisupervised_simclr/semisupervised_simclr_11_0.png)
    


Using the contrastive augmentation pipleine we notice how
the dataset has changed.


```python
# Contrastive Augmentations
contrastive_augmenter = get_augmenter(**CONTRASTIVE_AUGMENTATION)
augmented_images = contrastive_augmenter(unlabeled_images)
keras_cv.visualization.plot_image_gallery(
    images=augmented_images,
    value_range=(0, 1),
    rows=3,
    cols=3,
)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

```
</div>
    
![png](/img/examples/vision/semisupervised_simclr/semisupervised_simclr_13_4.png)
    


Let's now apply the classification augmentation pipeline on the
dataset.


```python
# Classification Augmentations
classification_augmenter = get_augmenter(**CLASSIFICATION_AUGMENTATION)
augmented_images = classification_augmenter(unlabeled_images)
keras_cv.visualization.plot_image_gallery(
    images=augmented_images,
    value_range=(0, 1),
    rows=3,
    cols=3,
)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

```
</div>
    
![png](/img/examples/vision/semisupervised_simclr/semisupervised_simclr_15_4.png)
    


---
## Encoder architecture

We use the `ResNet18Backbone` from the KerasCV library. Try out different
backbones and check whether any model trains better in this paradigm. Also
try to reason out why that happened.


```python

# Define the encoder architecture
def get_encoder():
    return keras.Sequential(
        [
            keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)),
            keras_cv.models.ResNet18Backbone(include_rescaling=False),
            keras.layers.GlobalAveragePooling2D(name="avg_pool"),
        ],
        name="encoder",
    )

```

---
## Supervised baseline model

A baseline supervised model is trained using random initialization.


```python
# Baseline supervised training with random initialization
baseline_model = keras.Sequential(
    [
        keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)),
        get_augmenter(**CLASSIFICATION_AUGMENTATION),
        get_encoder(),
        keras.layers.Dense(NUM_CLASSES),
    ],
    name="baseline_model",
)
baseline_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

baseline_history = baseline_model.fit(
    labeled_train_dataset, epochs=20, validation_data=test_dataset
)

print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(baseline_history.history["val_acc"]) * 100
    )
)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

Epoch 1/20
WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

40/40 [==============================] - ETA: 0s - loss: 1.9072 - acc: 0.3252WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

40/40 [==============================] - 25s 265ms/step - loss: 1.9072 - acc: 0.3252 - val_loss: 4.4865 - val_acc: 0.1130
Epoch 2/20
40/40 [==============================] - 8s 207ms/step - loss: 1.4727 - acc: 0.4508 - val_loss: 4.0150 - val_acc: 0.1520
Epoch 3/20
40/40 [==============================] - 8s 208ms/step - loss: 1.3147 - acc: 0.5110 - val_loss: 3.3695 - val_acc: 0.1713
Epoch 4/20
40/40 [==============================] - 8s 208ms/step - loss: 1.2389 - acc: 0.5450 - val_loss: 2.9845 - val_acc: 0.1803
Epoch 5/20
40/40 [==============================] - 9s 211ms/step - loss: 1.1386 - acc: 0.5868 - val_loss: 5.7640 - val_acc: 0.1326
Epoch 6/20
40/40 [==============================] - 9s 211ms/step - loss: 1.0558 - acc: 0.6090 - val_loss: 3.6970 - val_acc: 0.1614
Epoch 7/20
40/40 [==============================] - 9s 213ms/step - loss: 0.9654 - acc: 0.6510 - val_loss: 3.5209 - val_acc: 0.2023
Epoch 8/20
40/40 [==============================] - 9s 213ms/step - loss: 0.9862 - acc: 0.6368 - val_loss: 3.3486 - val_acc: 0.2212
Epoch 9/20
40/40 [==============================] - 8s 206ms/step - loss: 0.8777 - acc: 0.6776 - val_loss: 2.2990 - val_acc: 0.3305
Epoch 10/20
40/40 [==============================] - 8s 204ms/step - loss: 0.8297 - acc: 0.7016 - val_loss: 3.6051 - val_acc: 0.2769
Epoch 11/20
40/40 [==============================] - 8s 205ms/step - loss: 0.7952 - acc: 0.7092 - val_loss: 1.8223 - val_acc: 0.4650
Epoch 12/20
40/40 [==============================] - 8s 208ms/step - loss: 0.8468 - acc: 0.6998 - val_loss: 1.6880 - val_acc: 0.5008
Epoch 13/20
40/40 [==============================] - 9s 213ms/step - loss: 0.7948 - acc: 0.7208 - val_loss: 1.9914 - val_acc: 0.4221
Epoch 14/20
40/40 [==============================] - 8s 207ms/step - loss: 0.7430 - acc: 0.7338 - val_loss: 3.7770 - val_acc: 0.3709
Epoch 15/20
40/40 [==============================] - 9s 217ms/step - loss: 0.7464 - acc: 0.7358 - val_loss: 4.6517 - val_acc: 0.2849
Epoch 16/20
40/40 [==============================] - 8s 209ms/step - loss: 0.6132 - acc: 0.7828 - val_loss: 1.5031 - val_acc: 0.5433
Epoch 17/20
40/40 [==============================] - 8s 202ms/step - loss: 0.6846 - acc: 0.7554 - val_loss: 1.4208 - val_acc: 0.5611
Epoch 18/20
40/40 [==============================] - 8s 207ms/step - loss: 0.5599 - acc: 0.8032 - val_loss: 1.2669 - val_acc: 0.5866
Epoch 19/20
40/40 [==============================] - 8s 210ms/step - loss: 0.4973 - acc: 0.8242 - val_loss: 2.0523 - val_acc: 0.4749
Epoch 20/20
40/40 [==============================] - 8s 204ms/step - loss: 0.6079 - acc: 0.7858 - val_loss: 1.8732 - val_acc: 0.5054
Maximal validation accuracy: 58.66%

```
</div>
---
## Self-supervised model for contrastive pretraining

We pretrain an encoder on unlabeled images with a contrastive loss.
A nonlinear projection head is attached to the top of the encoder, as it
improves the quality of representations of the encoder.

We use the InfoNCE/NT-Xent/N-pairs loss (KerasCV already has this implemented as the `SimCLRLoss`),
which can be interpreted in the following way:

1. We treat each image in the batch as if it had its own class.
2. Then, we have two examples (a pair of augmented views) for each "class".
3. Each view's representation is compared to every possible pair's one (for both
  augmented versions).
4. We use the temperature-scaled cosine similarity of compared representations as
  logits.
5. Finally, we use categorical cross-entropy as the "classification" loss

We subclass the `ContrastiveTrainer` from the KerasCV library to build the `SimCLRTrainer`.


```python

class SimCLRTrainer(keras_cv.training.ContrastiveTrainer):
    def __init__(self, encoder, augmenter, projector, probe=None, **kwargs):
        super().__init__(
            encoder=encoder,
            augmenter=augmenter,
            projector=projector,
            probe=probe,
            **kwargs,
        )


simclr_model = SimCLRTrainer(
    encoder=get_encoder(),
    augmenter=get_augmenter(**CONTRASTIVE_AUGMENTATION),
    projector=keras.Sequential(
        [
            keras.layers.Dense(PROJECTION_WIDTH, activation="relu"),
            keras.layers.Dense(PROJECTION_WIDTH),
            keras.layers.BatchNormalization(),
        ],
        name="projector",
    ),
)

simclr_model.compile(
    encoder_optimizer=keras.optimizers.Adam(),
    encoder_loss=keras_cv.losses.SimCLRLoss(
        temperature=TEMPERATURE,
    ),
)

simclr_history = simclr_model.fit(
    unlabeled_train_dataset,
    epochs=20,
)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

Epoch 1/20
98/98 [==============================] - 123s 1s/step - loss: 11.6321
Epoch 2/20
98/98 [==============================] - 110s 1s/step - loss: 8.3731
Epoch 3/20
98/98 [==============================] - 110s 1s/step - loss: 7.0380
Epoch 4/20
98/98 [==============================] - 111s 1s/step - loss: 6.2318
Epoch 5/20
98/98 [==============================] - 110s 1s/step - loss: 5.6933
Epoch 6/20
98/98 [==============================] - 111s 1s/step - loss: 5.2573
Epoch 7/20
98/98 [==============================] - 112s 1s/step - loss: 4.9030
Epoch 8/20
98/98 [==============================] - 110s 1s/step - loss: 4.6462
Epoch 9/20
98/98 [==============================] - 112s 1s/step - loss: 4.4500
Epoch 10/20
98/98 [==============================] - 114s 1s/step - loss: 4.2191
Epoch 11/20
98/98 [==============================] - 113s 1s/step - loss: 4.0687
Epoch 12/20
98/98 [==============================] - 112s 1s/step - loss: 3.9270
Epoch 13/20
98/98 [==============================] - 113s 1s/step - loss: 3.8176
Epoch 14/20
98/98 [==============================] - 113s 1s/step - loss: 3.6935
Epoch 15/20
98/98 [==============================] - 112s 1s/step - loss: 3.6033
Epoch 16/20
98/98 [==============================] - 112s 1s/step - loss: 3.5326
Epoch 17/20
98/98 [==============================] - 111s 1s/step - loss: 3.4492
Epoch 18/20
98/98 [==============================] - 111s 1s/step - loss: 3.4024
Epoch 19/20
98/98 [==============================] - 116s 1s/step - loss: 3.3422
Epoch 20/20
98/98 [==============================] - 113s 1s/step - loss: 3.2761

```
</div>
---
## Supervised finetuning of the pretrained encoder

We then finetune the encoder on the labeled examples, by attaching
a single randomly initalized fully connected classification layer on its top.


```python
# Supervised finetuning of the pretrained encoder
finetune_model = keras.Sequential(
    [
        keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)),
        get_augmenter(**CLASSIFICATION_AUGMENTATION),
        simclr_model.encoder,
        keras.layers.Dense(NUM_CLASSES),
    ],
    name="finetuning_model",
)
finetune_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

finetune_history = finetune_model.fit(
    labeled_train_dataset, epochs=20, validation_data=test_dataset
)

print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(finetune_history.history["val_acc"]) * 100
    )
)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

Epoch 1/20
WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

39/40 [============================>.] - ETA: 0s - loss: 1.4232 - acc: 0.5112WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

WARNING:tensorflow:Using a while_loop for converting CropAndResize cause there is no registered converter for this op.

40/40 [==============================] - 21s 249ms/step - loss: 1.4221 - acc: 0.5118 - val_loss: 5.5473 - val_acc: 0.2649
Epoch 2/20
40/40 [==============================] - 9s 217ms/step - loss: 0.9671 - acc: 0.6480 - val_loss: 5.6939 - val_acc: 0.2358
Epoch 3/20
40/40 [==============================] - 9s 214ms/step - loss: 0.8753 - acc: 0.6822 - val_loss: 2.0208 - val_acc: 0.4498
Epoch 4/20
40/40 [==============================] - 9s 210ms/step - loss: 0.7816 - acc: 0.7200 - val_loss: 2.7762 - val_acc: 0.3365
Epoch 5/20
40/40 [==============================] - 8s 208ms/step - loss: 0.7641 - acc: 0.7222 - val_loss: 3.0242 - val_acc: 0.4688
Epoch 6/20
40/40 [==============================] - 9s 216ms/step - loss: 0.6752 - acc: 0.7566 - val_loss: 1.8544 - val_acc: 0.4789
Epoch 7/20
40/40 [==============================] - 9s 213ms/step - loss: 0.6603 - acc: 0.7590 - val_loss: 1.4286 - val_acc: 0.5669
Epoch 8/20
40/40 [==============================] - 9s 213ms/step - loss: 0.6717 - acc: 0.7666 - val_loss: 1.6336 - val_acc: 0.5460
Epoch 9/20
40/40 [==============================] - 9s 214ms/step - loss: 0.5979 - acc: 0.7878 - val_loss: 3.0925 - val_acc: 0.3101
Epoch 10/20
40/40 [==============================] - 8s 208ms/step - loss: 0.7213 - acc: 0.7460 - val_loss: 1.2885 - val_acc: 0.5832
Epoch 11/20
40/40 [==============================] - 9s 212ms/step - loss: 0.4963 - acc: 0.8282 - val_loss: 1.3040 - val_acc: 0.6034
Epoch 12/20
40/40 [==============================] - 8s 209ms/step - loss: 0.4354 - acc: 0.8488 - val_loss: 1.1805 - val_acc: 0.6398
Epoch 13/20
40/40 [==============================] - 8s 208ms/step - loss: 0.3205 - acc: 0.8894 - val_loss: 1.4723 - val_acc: 0.5899
Epoch 14/20
40/40 [==============================] - 8s 208ms/step - loss: 0.3937 - acc: 0.8648 - val_loss: 1.2627 - val_acc: 0.6215
Epoch 15/20
40/40 [==============================] - 8s 210ms/step - loss: 0.4112 - acc: 0.8582 - val_loss: 1.4905 - val_acc: 0.5803
Epoch 16/20
40/40 [==============================] - 9s 220ms/step - loss: 0.3344 - acc: 0.8822 - val_loss: 1.6081 - val_acc: 0.5771
Epoch 17/20
40/40 [==============================] - 9s 218ms/step - loss: 0.3794 - acc: 0.8694 - val_loss: 1.5366 - val_acc: 0.6008
Epoch 18/20
40/40 [==============================] - 8s 205ms/step - loss: 0.2635 - acc: 0.9074 - val_loss: 1.2707 - val_acc: 0.6463
Epoch 19/20
40/40 [==============================] - 8s 207ms/step - loss: 0.3174 - acc: 0.8844 - val_loss: 1.6366 - val_acc: 0.5904
Epoch 20/20
40/40 [==============================] - 9s 213ms/step - loss: 0.2809 - acc: 0.9058 - val_loss: 1.1887 - val_acc: 0.6668
Maximal validation accuracy: 66.68%

```
</div>
---
## Comparison against the baseline


```python

# The classification accuracies of the baseline and finetuning process:
def plot_training_curves(baseline_history, finetune_history):
    for metric_key, metric_name in zip(["acc", "loss"], ["accuracy", "loss"]):
        plt.figure(figsize=(8, 5), dpi=100)
        plt.plot(
            baseline_history.history[f"val_{metric_key}"], label="supervised baseline"
        )
        plt.plot(
            finetune_history.history[f"val_{metric_key}"],
            label="supervised finetuning",
        )
        plt.legend()
        plt.title(f"Classification {metric_name} during training")
        plt.xlabel("epochs")
        plt.ylabel(f"validation {metric_name}")


plot_training_curves(baseline_history, finetune_history)
```


    
![png](/img/examples/vision/semisupervised_simclr/semisupervised_simclr_25_0.png)
    



    
![png](/img/examples/vision/semisupervised_simclr/semisupervised_simclr_25_1.png)
    


By comparing the training curves, we can see that when using contrastive
pretraining, a higher validation accuracy can be reached, paired with a lower
validation loss, which means that the pretrained network was able to generalize
better when seeing only a small amount of labeled examples.

---
## Improving further

### Architecture

The experiment in the original paper demonstrated that increasing the width and depth of the
models improves performance at a higher rate than for supervised learning. Also,
using a [ResNet-50](https://keras.io/api/applications/resnet/#resnet50-function)
encoder is quite standard in the literature. However keep in mind, that more
powerful models will not only increase training time but will also require more
memory and will limit the maximal batch size you can use.

It has [been](https://arxiv.org/abs/1905.09272)
[reported](https://arxiv.org/abs/1911.05722) that the usage of BatchNorm layers
could sometimes degrade performance, as it introduces an intra-batch dependency
between samples, which is why I did not have used them in this example. In my
experiments however, using BatchNorm, especially in the projection head,
improves performance.

### Hyperparameters

The hyperparameters used in this example have been tuned manually for this task and
architecture. Therefore, without changing them, only marginal gains can be expected
from further hyperparameter tuning.

However for a different task or model architecture these would need tuning, so
here are my notes on the most important ones:

- **Batch size**: since the objective can be interpreted as a classification
over a batch of images (loosely speaking), the batch size is actually a more
important hyperparameter than usual. The higher, the better.
- **Temperature**: the temperature defines the "softness" of the softmax
distribution that is used in the cross-entropy loss, and is an important
hyperparameter. Lower values generally lead to a higher contrastive accuracy.
A recent trick (in [ALIGN](https://arxiv.org/abs/2102.05918)) is to learn
the temperature's value as well (which can be done by defining it as a
tf.Variable, and applying gradients on it). Even though this provides a good baseline
value, in my experiments the learned temperature was somewhat lower
than optimal, as it is optimized with respect to the contrastive loss, which is not a
perfect proxy for representation quality.
- **Image augmentation strength**: during pretraining stronger augmentations
increase the difficulty of the task, however after a point too strong
augmentations will degrade performance. During finetuning stronger
augmentations reduce overfitting while in my experience too strong
augmentations decrease the performance gains from pretraining. The whole data
augmentation pipeline can be seen as an important hyperparameter of the
algorithm, implementations of other custom image augmentation layers in Keras
can be found in
[this repository](https://github.com/beresandras/image-augmentation-layers-keras).
- **Learning rate schedule**: a constant schedule is used here, but it is
quite common in the literature to use a
[cosine decay schedule](https://www.tensorflow.org/api_docs/python/tf/keras/experimental/CosineDecay),
which can further improve performance.
- **Optimizer**: Adam is used in this example, as it provides good performance
with default parameters. SGD with momentum requires more tuning, however it
could slightly increase performance.

---
## Related works

Other instance-level (image-level) contrastive learning methods:

- [MoCo](https://arxiv.org/abs/1911.05722)
([v2](https://arxiv.org/abs/2003.04297),
[v3](https://arxiv.org/abs/2104.02057)): uses a momentum-encoder as well,
whose weights are an exponential moving average of the target encoder
- [SwAV](https://arxiv.org/abs/2006.09882): uses clustering instead of pairwise
comparison
- [BarlowTwins](https://arxiv.org/abs/2103.03230): uses a cross
correlation-based objective instead of pairwise comparison

Keras implementations of **MoCo** and **BarlowTwins** can be found in
[this repository](https://github.com/beresandras/contrastive-classification-keras),
which includes a Colab notebook.

There is also a new line of works, which optimize a similar objective, but
without the use of any negatives:

- [BYOL](https://arxiv.org/abs/2006.07733): momentum-encoder + no negatives
- [SimSiam](https://arxiv.org/abs/2011.10566)
([Keras example](https://keras.io/examples/vision/simsiam/)):
no momentum-encoder + no negatives

In my experience, these methods are more brittle (they can collapse to a constant
representation, I could not get them to work using this encoder architecture).
Even though they are generally more dependent on the
[model](https://generallyintelligent.ai/understanding-self-supervised-contrastive-learning.html)
[architecture](https://arxiv.org/abs/2010.10241), they can improve
performance at smaller batch sizes.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/semi-supervised-classification-simclr)
and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/semi-supervised-classification).

---
## Acknowledgements

I would like to thank [Martin Gorner](https://twitter.com/martin_gorner) for his thorough review.
Google Cloud credits were provided for this project.

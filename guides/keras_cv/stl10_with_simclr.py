"""
Title: Contrastive Feature Learning with SimCLR in KerasCV
Author: Ian Stenbit, [Luke Wood](https://lukewood.xyz)
Date created: 2022/09/09
Last modified: 2022/09/09
Description: Use KerasCV for contrastive feature learning on STL-10
"""

"""
## Overview

In recent years, contrastive learning has enabled neural
networks to learn from corpuses of unlabelled data.  Techniques such as SimCLR, SimSiam,
and Masked AutoEncoders score within percentage points of state of the art techniques
without leveraging labelled data.

The STL-10 dataset is a benchmark dataset specifically targeted at self supervised
learning techniques.  STL-10 consists of a large unlabelled data corpus and a tiny
labelled corpus.  To solve the dataset, you typically use a contrastive learning
technique to obtain a useful weight set, then finetune your model on the labelled data
to obtain a classifier.

KerasCV offers a contrastive learning API to perform self-supervised feature
learning on unlabelled image data. In this guide, we'll demonstrate the use
of this API to perform self-supervised feature learning and supervised fine-tuning
on the STL-10 dataset.

For a reference on contrastive learning, and SimCLR in particular, check out the
[SimCLR paper](https://arxiv.org/pdf/2002.05709).

To get started, let's sort out all of our imports and define configuration parameters.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from keras_cv import losses
from keras_cv import models
from keras_cv import training


CLASSES = 10
BATCH_SIZE = 64
IMAGE_SIZE = (96, 96)
CONTRASTIVE_EPOCHS = 1
FINE_TUNING_EPOCHS = 1

"""
## Data loading

In this guide, we use the `stl10` dataset from tensorflow-datasets.

This dataset includes 100,000 unlabelled training images, 5,000 labelled
training images, and 8,000 labelled evaluation images.

We start by loading the unlabelled examples. Note that we shuffle our unlabelled
images, and specify `reshuffle_each_iteration` to ensure that training batches
contain unique sets of images each epoch. This is important to maximize the
effect of contrastive learning.
"""

unlabelled_images = tfds.load("stl10", split="unlabelled")
unlabelled_images = unlabelled_images.map(
    lambda entry: entry["image"], num_parallel_calls=tf.data.AUTOTUNE
)

unlabelled_images = unlabelled_images.shuffle(
    buffer_size = 8*BATCH_SIZE, reshuffle_each_iteration=True
)
unlabelled_images = unlabelled_images.batch(BATCH_SIZE)

"""
We then load the labelled examples as separate train and test splits.
"""

train_labelled_images = tfds.load("stl10", split="train")
test_labelled_images = tfds.load("stl10", split="test")

def process_labelled_record(record):
    image = record["image"]
    label = tf.one_hot(record["label"], CLASSES)
    return image, label


train_labelled_images = train_labelled_images.map(
    process_labelled_record, num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE)
test_labelled_images = test_labelled_images.map(
    process_labelled_record, num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE)


"""
## Building our Contrastive Learning Pipieline

### Specifying a Model

With our data prepared, we can create a model to train.

Any Keras model can be trained with the `ContrastiveTrainer`, provided that it:
- Accepts images as input
- Returns a flattened rank-2 tensor (e.g. of shape [BATCH_SIZE, N]), where N is
the number of features in the flattened encoding produced by the model.

Such a model can be built using any KerasCV classification model by specifying
 `include_top=False` and enabling pooling on the output. In this example, we use
a ResNet50V2 from KerasCV with `pooling="avg"`
"""
encoder = models.ResNet50V2(
    include_rescaling=True,
    include_top=False,
    input_shape=IMAGE_SIZE + (3,),
    pooling="avg",
)

"""
### Creating our Trainer

Next, we can create a contrastive trainer. KerasCV offers a flexible `ContrastiveTrainer`
API in addition to a `SimCLRTrainer`, which extends `ContrastiveTrainer` and
provides the default structure of SimCLR. KerasCV also implements a `SimCLRAugmenter`
which performs the augmentation steps which showed the best results in the SimCLR
paper.

We use a `SimCLRTrainer` with a `SimCLRAugmenter` in this example, which gives us
the power of SimCLR with minimal manual configuration. Users who wish to customize
their contrastive augmenters and projectors further can use a `ContrastiveTrainer`
directly.
"""
trainer = training.SimCLRTrainer(
    encoder=encoder,
    augmenter=training.SimCLRAugmenter(
        value_range=(0, 255), target_size=IMAGE_SIZE
    ),
)

"""
### Training Configuration

Next, we set up training configuration including optimizer selection, loss, and
callbacks.

For our loss function we use the loss function used in the SimCLR paper,
"Normalized Temperature-scaled Cross Entropy loss". KerasCV packages this as `SimCLRLoss`.

Note that we perform gradient clipping in our optimizer. In our experiments, we found
gradient clipping reduced gradient explosion when performing training with a probe.

In this guide, as we're training on unlabelled images, no probe is used.
"""

trainer.compile(
    encoder_optimizer=optimizers.SGD(
        learning_rate=0.05, momentum=0.9, global_clipnorm=10
    ),
    encoder_loss=losses.SimCLRLoss(temperature=0.5),
)

training_callbacks = [
    callbacks.EarlyStopping(monitor="loss", patience=5, min_delta=0.1),
    callbacks.BackupAndRestore("backup"),
    callbacks.TensorBoard(log_dir="logs"),
]

"""
### Fitting our Model

Now, we can simply call `trainer.fit` to perform the contrastive learning process
using our unlabelled images.
"""
trainer.fit(
    unlabelled_images.take(100),
    epochs=CONTRASTIVE_EPOCHS,
    callbacks=training_callbacks,
)


"""
## Fine-Tuning

Now that we've performed contrastive feature learning, we can train a classifier
using our pre-trained encoder as input. To do so, we freeze the pre-trained encoder
and add a Dense layer to it to perform classification.
"""

encoder.trainable = False
classification_model = keras.Sequential(
    [encoder, layers.Dense(CLASSES, activation="softmax")]
)

classification_model.compile(
    optimizer="adam", loss=keras.losses.CategoricalCrossentropy()
)

"""
Then, we simply call `fit` on our new classification model using our labelled images.
"""

classification_model.fit(train_labelled_images)

classification_model.evaluate(test_labelled_images)

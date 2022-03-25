"""
Title: Knowledge distillation recipes
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/08/01
Last modified: 2021/08/01
Description: Training better student models via knowledge distillation with function matching.
"""
"""
## Introduction

Knowledge distillation ([Hinton et al.](https://arxiv.org/abs/1503.02531)) is a technique
that enables us to compress larger models into smaller ones. This allows us to reap the
benefits of high performing larger models, while reducing storage and memory costs and
achieving higher inference speed:

* Smaller models -> smaller memory footprint
* Reduced complexity -> fewer floating-point operations (FLOPs)

In [Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237),
Beyer et al. investigate various existing setups for performing knowledge distillation
and show that all of them lead to sub-optimal performance. Due to this,
practitioners often settle for other alternatives (quantization, pruning, weight
clustering, etc.) when developing production systems that are resource-constrained.

Beyer et al. investigate how we can improve the student models that come out
of the knowledge distillation process and always match the performance of
their teacher models. In this example, we will study the recipes introduced by them, using
the [Flowers102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). As a
reference, with these recipes, the authors were able to produce a ResNet50 model that
achieves 82.8% accuracy on the ImageNet-1k dataset.

In case you need a refresher on knowledge distillation and want to study how it is
implemented in Keras, you can refer to
[this example](https://keras.io/examples/vision/knowledge_distillation/).
You can also follow
[this example](https://keras.io/examples/vision/consistency_training/)
that shows an extension of knowledge distillation applied to consistency training.

To follow this example, you will need TensorFlow 2.5 or higher as well as TensorFlow Addons,
which can be installed using the command below:
"""

"""shell
!pip install -q tensorflow-addons
"""

"""
## Imports
"""

from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

"""
## Hyperparameters and contants
"""

AUTO = tf.data.AUTOTUNE  # Used to dynamically adjust parallelism.
BATCH_SIZE = 64

# Comes from Table 4 and "Training setup" section.
TEMPERATURE = 10  # Used to soften the logits before they go to softmax.
INIT_LR = 0.003  # Initial learning rate that will be decayed over the training period.
WEIGHT_DECAY = 0.001  # Used for regularization.
CLIP_THRESHOLD = 1.0  # Used for clipping the gradients by L2-norm.

# We will first resize the training images to a bigger size and then we will take
# random crops of a lower size.
BIGGER = 160
RESIZE = 128

"""
## Load the Flowers102 dataset
"""

train_ds, validation_ds, test_ds = tfds.load(
    "oxford_flowers102", split=["train", "validation", "test"], as_supervised=True
)
print(f"Number of training examples: {train_ds.cardinality()}.")
print(f"Number of validation examples: {validation_ds.cardinality()}.")
print(f"Number of test examples: {test_ds.cardinality()}.")

"""
## Teacher model

As is common with any distillation technique, it's important to first train a
well-performing teacher model which is usually larger than the subsequent student model.
The authors distill a BiT ResNet152x2 model (teacher) into a BiT ResNet50 model
(student).

BiT stands for Big Transfer and was introduced in
[Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370).
BiT variants of ResNets use Group Normalization ([Wu et al.](https://arxiv.org/abs/1803.08494))
and Weight Standardization ([Qiao et al.](https://arxiv.org/abs/1903.10520v2))
in place of Batch Normalization ([Ioffe et al.](https://arxiv.org/abs/1502.03167)).
In order to limit the time it takes to run this example, we will be using a BiT
ResNet101x3 already trained on the Flowers102 dataset. You can refer to
[this notebook](https://github.com/sayakpaul/FunMatch-Distillation/blob/main/train_bit.ipynb)
to learn more about the training process. This model reaches 98.18% accuracy on the
test set of Flowers102.

The model weights are hosted on Kaggle as a dataset.
To download the weights, follow these steps:

1. Create an account on Kaggle [here](https://www.kaggle.com).
2. Go to the "Account" tab of your [user profile](https://www.kaggle.com/account).
3. Select "Create API Token". This will trigger the download of `kaggle.json`, a file
containing your API credentials.
4. From that JSON file, copy your Kaggle username and API key.

Now run the following:

```python
import os

os.environ["KAGGLE_USERNAME"] = "" # TODO: enter your Kaggle user name here
os.environ["KAGGLE_KEY"] = "" # TODO: enter your Kaggle key here
```

Once the environment variables are set, run:

```shell
$ kaggle datasets download -d spsayakpaul/bitresnet101x3flowers102
$ unzip -qq bitresnet101x3flowers102.zip
```

This should generate a folder named `T-r101x3-128` which is essentially a teacher
[`SavedModel`](https://www.tensorflow.org/guide/saved_model).
"""

import os

os.environ["KAGGLE_USERNAME"] = ""  # TODO: enter your Kaggle user name here
os.environ["KAGGLE_KEY"] = ""  # TODO: enter your Kaggle API key here

"""shell
!kaggle datasets download -d spsayakpaul/bitresnet101x3flowers102
"""

"""shell
!unzip -qq bitresnet101x3flowers102.zip
"""

# Since the teacher model is not going to be trained further we make
# it non-trainable.
teacher_model = keras.models.load_model(
    "/home/jupyter/keras-io/examples/keras_recipes/T-r101x3-128"
)
teacher_model.trainable = False
teacher_model.summary()

"""
## The "function matching" recipe

To train a high-quality student model, the authors propose the following changes to the
student training workflow:

* Use an aggressive variant of MixUp ([Zhang et al.](https://arxiv.org/abs/1710.09412)).
This is done by sampling the `alpha` parameter from a uniform distribution instead of a
beta distribution. MixUp is used here in order to help the student model capture the
function underlying the teacher model. MixUp linearly interpolates between different
samples across the data manifold. So the rationale here is if the student is trained to
fit that it should be able to match the teacher model better. To incorporate more
invariance MixUp is coupled with "Inception-style" cropping
([Szegedy et al.](https://arxiv.org/abs/1409.4842)). This is where the
"function matching" term makes its way in the
[original paper](https://arxiv.org/abs/2106.05237).
* Unlike other works ([Noisy Student Training](https://arxiv.org/abs/1911.04252) for
example), both the teacher and student models receive the same copy of an image, which is
mixed up and randomly cropped. By providing the same inputs to both the models, the
authors make the teacher consistent with the student.
* With MixUp, we are essentially introducing a strong form of regularization when
training the student. As such, it should be trained for a
relatively long period of time (1000 epochs at least). Since the student is trained with
strong regularization, the risk of overfitting due to a longer training
schedule are also mitigated.

In summary, one needs to be consistent and patient while training the student model.
"""

"""
## Data input pipeline
"""


def mixup(images, labels):
    alpha = tf.random.uniform([], 0, 1)
    mixedup_images = alpha * images + (1 - alpha) * tf.reverse(images, axis=[0])
    # The labels do not matter here since they are NOT used during
    # training.
    return mixedup_images, labels


def preprocess_image(image, label, train=True):
    image = tf.cast(image, tf.float32) / 255.0

    if train:
        image = tf.image.resize(image, (BIGGER, BIGGER))
        image = tf.image.random_crop(image, (RESIZE, RESIZE, 3))
        image = tf.image.random_flip_left_right(image)
    else:
        # Central fraction amount is from here:
        # https://git.io/J8Kda.
        image = tf.image.central_crop(image, central_fraction=0.875)
        image = tf.image.resize(image, (RESIZE, RESIZE))

    return image, label


def prepare_dataset(dataset, train=True, batch_size=BATCH_SIZE):
    if train:
        dataset = dataset.map(preprocess_image, num_parallel_calls=AUTO)
        dataset = dataset.shuffle(BATCH_SIZE * 10)
    else:
        dataset = dataset.map(
            lambda x, y: (preprocess_image(x, y, train)), num_parallel_calls=AUTO
        )
    dataset = dataset.batch(batch_size)

    if train:
        dataset = dataset.map(mixup, num_parallel_calls=AUTO)

    dataset = dataset.prefetch(AUTO)
    return dataset


"""
Note that for brevity, we used mild crops for the training set but in practice
"Inception-style" preprocessing should be applied. You can refer to
[this script](https://github.com/sayakpaul/FunMatch-Distillation/blob/main/crop_resize.py)
for a closer implementation. Also, _**the ground-truth labels are not used for
training the student.**_
"""

train_ds = prepare_dataset(train_ds, True)
validation_ds = prepare_dataset(validation_ds, False)
test_ds = prepare_dataset(test_ds, False)

"""
## Visualization
"""

sample_images, _ = next(iter(train_ds))
plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(sample_images[n].numpy())
    plt.axis("off")
plt.show()

"""
## Student model

For the purpose of this example, we will use the standard ResNet50V2
([He et al.](https://arxiv.org/abs/1603.05027)).
"""


def get_resnetv2():
    resnet_v2 = keras.applications.ResNet50V2(
        weights=None,
        input_shape=(RESIZE, RESIZE, 3),
        classes=102,
        classifier_activation="linear",
    )
    return resnet_v2


get_resnetv2().count_params()

"""
Compared to the teacher model, this model has 358 Million fewer parameters.
"""

"""
## Distillation utility

We will reuse some code from
[this example](https://keras.io/examples/vision/knowledge_distillation/)
on knowledge distillation.
"""


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.loss_tracker = keras.metrics.Mean(name="distillation_loss")

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_tracker)
        return metrics

    def compile(
        self,
        optimizer,
        metrics,
        distillation_loss_fn,
        temperature=TEMPERATURE,
    ):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, _ = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(distillation_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Report progress
        self.loss_tracker.update_state(distillation_loss)
        return {"distillation_loss": self.loss_tracker.result()}

    def test_step(self, data):
        # Unpack data
        x, y = data

        # Forward passes
        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=False)

        # Calculate the loss
        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
            tf.nn.softmax(student_predictions / self.temperature, axis=1),
        )

        # Report progress
        self.loss_tracker.update_state(distillation_loss)
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        return results


"""
## Learning rate schedule

A warmup cosine learning rate schedule is used in the paper. This schedule is also
typical for many pre-training methods especially for computer vision.
"""

# Some code is taken from:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


"""
We can now plot a a graph of learning rates generated using this schedule.
"""

ARTIFICIAL_EPOCHS = 1000
ARTIFICIAL_BATCH_SIZE = 512
DATASET_NUM_TRAIN_EXAMPLES = 1020
TOTAL_STEPS = int(
    DATASET_NUM_TRAIN_EXAMPLES / ARTIFICIAL_BATCH_SIZE * ARTIFICIAL_EPOCHS
)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=TOTAL_STEPS,
    warmup_learning_rate=0.0,
    warmup_steps=1500,
)

lrs = [scheduled_lrs(step) for step in range(TOTAL_STEPS)]
plt.plot(lrs)
plt.xlabel("Step", fontsize=14)
plt.ylabel("LR", fontsize=14)
plt.show()


"""
The original paper uses at least 1000 epochs and a batch size of 512 to perform
"function matching". The objective of this example is to present a workflow to
implement the recipe and not to demonstrate the results when they are applied at full scale.
However, these recipes will transfer to the original settings from the paper. Please
refer to [this repository](https://github.com/sayakpaul/FunMatch-Distillation) if you are
interested in finding out more.
"""

"""
## Training
"""

optimizer = tfa.optimizers.AdamW(
    weight_decay=WEIGHT_DECAY, learning_rate=scheduled_lrs, clipnorm=CLIP_THRESHOLD
)

student_model = get_resnetv2()

distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer,
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    distillation_loss_fn=keras.losses.KLDivergence(),
    temperature=TEMPERATURE,
)

history = distiller.fit(
    train_ds,
    steps_per_epoch=int(np.ceil(DATASET_NUM_TRAIN_EXAMPLES / BATCH_SIZE)),
    validation_data=validation_ds,
    epochs=30,  # This should be at least 1000.
)

student = distiller.student
student_model.compile(metrics=["accuracy"])
_, top1_accuracy = student.evaluate(test_ds)
print(f"Top-1 accuracy on the test set: {round(top1_accuracy * 100, 2)}%")

"""
## Results

With just 30 epochs of training, the results are nowhere near expected.
This is where the benefits of patience aka a longer training schedule
will come into play. Let's investigate what the model trained for 1000 epochs can do.
"""

"""shell
# Download the pre-trained weights.
!wget https://git.io/JBO3Y -O S-r50x1-128-1000.tar.gz
!tar xf S-r50x1-128-1000.tar.gz
"""

pretrained_student = keras.models.load_model("S-r50x1-128-1000")
pretrained_student.summary()

"""
This model exactly follows what the authors have used in their student models. This is
why the model summary is a bit different.
"""

_, top1_accuracy = pretrained_student.evaluate(test_ds)
print(f"Top-1 accuracy on the test set: {round(top1_accuracy * 100, 2)}%")

"""
With 100000 epochs of training, this same model leads to a top-1 accuracy of 95.54%.

There are a number of important ablations studies presented in the paper that show the
effectiveness of these recipes compared to the prior art. So if you are skeptical about
these recipes, definitely consult the paper.
"""

"""
## Note on training for longer

With TPU-based hardware infrastructure, we can train the model for 1000 epochs faster.
This does not even require adding a lot of changes to this codebase. You
are encouraged to check
[this repository](https://github.com/sayakpaul/FunMatch-Distillation)
as it presents TPU-compatible training workflows for these recipes and can be run on
[Kaggle Kernel](https://www.kaggle.com/kernels) leveraging their free TPU v3-8 hardware.
"""

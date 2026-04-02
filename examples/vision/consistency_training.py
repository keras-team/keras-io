"""
Title: Consistency training with supervision
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/04/13
Last modified: 2026/04/02
Description: Training with consistency regularization for robustness against data distribution shifts.
Accelerator: GPU

Converted to Keras 3 by: [Maitry Sinha](https://github.com/maitry63)
"""

"""
Deep learning models excel in many image recognition tasks when the data is independent
and identically distributed (i.i.d.). However, they can suffer from performance
degradation caused by subtle distribution shifts in the input data  (such as random
noise, contrast change, and blurring). So, naturally, there arises a question of
why. As discussed in [A Fourier Perspective on Model Robustness in Computer Vision](https://arxiv.org/pdf/1906.08988.pdf)),
there's no reason for deep learning models to be robust against such shifts. Standard
model training procedures (such as standard image classification training workflows)
*don't* enable a model to learn beyond what's fed to it in the form of training data.

In this example, we will be training an image classification model enforcing a sense of
*consistency* inside it by doing the following:

* Train a standard image classification model.
* Train an _equal or larger_ model on a noisy version of the dataset (augmented using
[RandAugment](https://arxiv.org/abs/1909.13719)).
* To do this, we will first obtain predictions of the previous model on the clean images
of the dataset.
* We will then use these predictions and train the second model to match these
predictions on the noisy variant of the same images. This is identical to the workflow of
[*Knowledge Distillation*](https://keras.io/examples/vision/knowledge_distillation/) but
since the student model is equal or larger in size this process is also referred to as
***Self-Training***.

This overall training workflow finds its roots in works like
[FixMatch](https://arxiv.org/abs/2001.07685), [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848),
and [Noisy Student Training](https://arxiv.org/abs/1911.04252). Since this training
process encourages a model yield consistent predictions for clean as well as noisy
images, it's often referred to as *consistency training* or *training with consistency
regularization*. Although the example focuses on using consistency training to enhance
the robustness of models to common corruptions this example can also serve a template
for performing _weakly supervised learning_.

This example requires TensorFlow 2.4 or higher, as well as TensorFlow Hub and TensorFlow
Models, which can be installed using the following command:

"""

"""
## Imports and setup
"""

import keras
from keras import layers
from keras import ops
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

"""
## Define hyperparameters
"""

BATCH_SIZE = 128
EPOCHS = 5
NUM_CLASSES = 10

CROP_TO = 72
RESIZE_TO = 96

"""
## Load the CIFAR-10 dataset
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
val_samples = 49500
new_train_x, new_y_train = x_train[: val_samples + 1], y_train[: val_samples + 1]
val_x, val_y = x_train[val_samples:], y_train[val_samples:]

"""
## Create `Dataset` objects
"""

# Initialize `RandAugment` object with 2 layers of
# augmentation transforms and strength of 9.
rand_augment = layers.RandAugment()
rand_augment.num_layers = 2
rand_augment.magnitude = 9

"""
For training the teacher model, we will only be using two geometric augmentation
transforms: random horizontal flip and random crop.
"""


def preprocess_train(image, label, noisy=True):
    image = ops.cast(image, "float32")

    # Random horizontal flip
    if keras.random.uniform(()) > 0.5:
        image = ops.flip(image, axis=1)

    # Resize + random crop
    image = keras.ops.image.resize(image, (RESIZE_TO, RESIZE_TO))
    start_x = keras.random.randint((), 0, RESIZE_TO - CROP_TO)
    start_y = keras.random.randint((), 0, RESIZE_TO - CROP_TO)
    image = image[start_x : start_x + CROP_TO, start_y : start_y + CROP_TO, :]

    # Apply RandAugment if noisy
    if noisy:
        image = rand_augment(image)

    return image, label


def preprocess_test(image, label):
    image = ops.cast(image, "float32")
    image = keras.ops.image.resize(image, (CROP_TO, CROP_TO))
    return image, label


"""
We make sure `train_clean_ds` and `train_noisy_ds` are shuffled using the *same* seed to
ensure their orders are exactly the same. This will be helpful during training the
student model.
"""

# This dataset will be used to train the first model.


def make_dataset(x, y, training=True, noisy=False):
    dataset = []
    for i in range(len(x)):
        if training:
            dataset.append(preprocess_train(x[i], y[i], noisy))
        else:
            dataset.append(preprocess_test(x[i], y[i]))
    return dataset


train_clean_ds = make_dataset(new_train_x, new_y_train, training=True, noisy=False)
train_noisy_ds = make_dataset(new_train_x, new_y_train, training=True, noisy=True)
validation_ds = make_dataset(val_x, val_y, training=False)
test_ds = make_dataset(x_test, y_test, training=False)


# Convert dataset to arrays
def to_arrays(dataset):
    X = np.array([img for img, _ in dataset])
    Y = np.array([lbl for _, lbl in dataset])
    return X, Y


x_train_clean, y_train_clean = to_arrays(train_clean_ds)
x_train_noisy, y_train_noisy = to_arrays(train_noisy_ds)
x_val, y_val = to_arrays(validation_ds)
x_test, y_test = to_arrays(test_ds)

"""
## Visualize the datasets
"""
sample_images, sample_labels = next(iter(train_clean_ds))
plt.figure(figsize=(10, 10))
for i, image in enumerate(sample_images[:9]):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("int"))
    plt.axis("off")

sample_images, sample_labels = next(iter(train_noisy_ds))
plt.figure(figsize=(10, 10))
for i, image in enumerate(sample_images[:9]):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("int"))
    plt.axis("off")


"""
## Define a model building utility function

We now define our model building utility. Our model is based on the [ResNet50V2 architecture](https://arxiv.org/abs/1603.05027).
"""


def get_training_model(num_classes=10):
    base = keras.applications.ResNet50V2(
        weights=None,
        include_top=False,
        input_shape=(CROP_TO, CROP_TO, 3),
    )
    model = keras.Sequential(
        [
            layers.Input((CROP_TO, CROP_TO, 3)),
            layers.Rescaling(scale=1.0 / 127.5, offset=-1),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes),
        ]
    )
    return model


"""
In the interest of reproducibility, we serialize the initial random weights of the
teacher  network.
"""

teacher_model = get_training_model()
teacher_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

"""
## Train the teacher model

As noted in Noisy Student Training, if the teacher model is trained with *geometric
ensembling* and when the student model is forced to mimic that, it leads to better
performance. The original work uses [Stochastic Depth](https://arxiv.org/abs/1603.09382)
and [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) to bring in the ensembling
part but for this example, we will use [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407)
(SWA) which also resembles geometric ensembling.
"""

# Define callbacks
reduce_lr = ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_accuracy")
early_stopping = EarlyStopping(
    patience=10, restore_best_weights=True, monitor="val_accuracy"
)


class SWA(Adam):
    """
    Simple Stochastic Weight Averaging wrapper for Adam optimizer.
    """

    def __init__(self, optimizer, start_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.base_optimizer = optimizer
        self.swa_weights = None
        self.n = 0
        self.start_epoch = start_epoch

    def apply_swa(self, model):
        """
        Update SWA weights after each epoch
        """
        if self.swa_weights is None:
            self.swa_weights = [ops.cast(w, "float32") for w in model.get_weights()]
            self.n = 1
        else:
            for i, w in enumerate(model.get_weights()):
                self.swa_weights[i] = (self.swa_weights[i] * self.n + w) / (self.n + 1)
            self.n += 1


# Compile and train the teacher model.
teacher_model = get_training_model()
teacher_model.set_weights(get_training_model().get_weights())  # load initial weights
teacher_model.compile(
    # Notice that we are wrapping our optimizer within SWA
    optimizer=SWA(Adam()),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
history = teacher_model.fit(
    x_train_clean,
    y_train_clean,
    validation_data=(x_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping],
)
# Evaluate teacher
_, acc = teacher_model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {acc*100:.2f}%")

"""
## Define a self-training utility

For this part, we will borrow the `Distiller` class from [this Keras Example](https://keras.io/examples/vision/knowledge_distillation/).
"""


"""
## Train the student model
"""

# Define the callbacks.
# We are using a larger decay factor to stabilize the training.
reduce_lr = ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_accuracy")
early_stopping = EarlyStopping(
    patience=10, restore_best_weights=True, monitor="val_accuracy"
)

# Build student model
student_model = get_training_model()


# Backend-agnostic SelfTrainingModel
class SelfTrainingModel(keras.Model):
    def __init__(self, student, teacher, temperature=10):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.teacher.trainable = False  # freeze teacher

    def call(self, inputs):
        clean_images, noisy_images = inputs
        teacher_preds = self.teacher(clean_images, training=False)
        student_preds = self.student(noisy_images, training=True)
        return student_preds, teacher_preds


# Custom loss: average of student loss + distillation loss
def consistency_loss(y_true, outputs, temperature=10):
    student_preds, teacher_preds = outputs

    # Supervised student loss
    student_loss = keras.losses.sparse_categorical_crossentropy(
        y_true, student_preds, from_logits=True
    )

    # Soft targets for distillation
    t_soft = ops.softmax(teacher_preds / temperature, axis=-1)
    s_soft = ops.softmax(student_preds / temperature, axis=-1)

    distill_loss = keras.losses.KLDivergence()(t_soft, s_soft)

    # Average the losses (Noisy Student)
    return (student_loss + distill_loss) / 2


# Instantiate the backend-agnostic self-training model
self_training_model = SelfTrainingModel(student_model, teacher_model, temperature=10)

# Compile the model
self_training_model.compile(
    optimizer=Adam(), loss=consistency_loss, metrics=["accuracy"]
)

# Train the student model (consistency training)
self_training_model.fit(
    (x_train_clean, x_train_noisy),  # clean + noisy pairs
    y_train_clean,  # supervised labels
    validation_data=((x_val, x_val), y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping],
)

# Evaluate the student model
_, acc = self_training_model.evaluate((x_test, x_test), y_test, verbose=1)
print(f"Test accuracy from student model: {acc*100:.2f}%")

"""
## Assess the robustness of the models

A standard benchmark of assessing the robustness of vision models is to record their
performance on corrupted datasets like ImageNet-C and CIFAR-10-C both of which were
proposed in [Benchmarking Neural Network Robustness to Common Corruptions and
Perturbations](https://arxiv.org/abs/1903.12261). For this example, we will be using the
CIFAR-10-C dataset which has 19 different corruptions on 5 different severity levels. To
assess the robustness of the models on this dataset, we will do the following:

* Run the pre-trained models on the highest level of severities and obtain the top-1
accuracies.
* Compute the mean top-1 accuracy.

For the purpose of this example, we won't be going through these steps. This is why we
trained the models for only 5 epochs. You can check out [this
repository](https://github.com/sayakpaul/Consistency-Training-with-Supervision) that
demonstrates the full-scale training experiments and also the aforementioned assessment.
The figure below presents an executive summary of that assessment:

![](https://i.ibb.co/HBJkM9R/image.png)

**Mean Top-1** results stand for the CIFAR-10-C dataset and **Test Top-1** results stand
for the CIFAR-10 test set. It's clear that consistency training has an advantage on not
only enhancing the model robustness but also on improving the standard test performance.
"""

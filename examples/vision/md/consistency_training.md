# Consistency training with supervision

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2021/04/13<br>
**Last modified:** 2026/04/30<br>
**Description:** Training with consistency regularization for robustness against data distribution shifts.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/consistency_training.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/consistency_training.py)



Deep learning models excel in many image recognition tasks when the data is independent
and identically distributed (i.i.d.). However, they can suffer from performance
degradation caused by subtle distribution shifts in the input data  (such as random
noise, contrast change, and blurring). So, naturally, there arises a question of
why. As discussed in [A Fourier Perspective on Model Robustness in Computer Vision](https://arxiv.org/abs/1906.08988),
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

---
## Imports and setup


```python
import keras
from keras import layers
from keras import ops
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import PyDataset

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
```

---
## Define hyperparameters


```python
BATCH_SIZE = 128
EPOCHS = 5
NUM_CLASSES = 10

CROP_TO = 72
RESIZE_TO = 96
TEMPERATURE = 10
```

---
## Load the CIFAR-10 dataset


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

val_samples = 49500
train_x, train_y = x_train[:val_samples], y_train[:val_samples]
val_x, val_y = x_train[val_samples:], y_train[val_samples:]

train_y = train_y.reshape(-1)
val_y = val_y.reshape(-1)
y_test = y_test.reshape(-1)
```

---
## Create PyDataset `Dataset` objects


```python
augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.1),
    ]
)
```

For training the teacher model, we will only be using two geometric augmentation
transforms: random horizontal flip and random crop.


```python

def preprocess_train(image, label, noisy=True):
    image = ops.cast(image, "float32")
    # We first resize the original image to a larger dimension
    # and then we take random crops from it.
    image = ops.image.resize(image, (RESIZE_TO, RESIZE_TO))

    x = keras.random.randint((), 0, RESIZE_TO - CROP_TO)
    y = keras.random.randint((), 0, RESIZE_TO - CROP_TO)

    image = image[x : x + CROP_TO, y : y + CROP_TO, :]

    if keras.random.uniform(()) > 0.5:
        image = ops.flip(image, axis=1)

    if noisy:
        image = augment(image)
    return np.array(image), label


def preprocess_test(image, label):
    image = ops.cast(image, "float32")
    image = ops.image.resize(image, (CROP_TO, CROP_TO))
    return image, label

```

We make sure `train_clean_ds` and `train_noisy_ds` are shuffled using the *same* seed to
ensure their orders are exactly the same. This will be helpful during training the
student model.


```python

# This dataset will be used to train the first model.
class TeacherDataset(PyDataset):
    def __init__(self, x, y, batch_size=128, training=True, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.training = training
        self.indices = np.arange(len(x))

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        if self.training:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        ids = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        images, labels = [], []

        for i in ids:
            img, lbl = preprocess_train(self.x[i], self.y[i], noisy=False)
            images.append(img)
            labels.append(lbl)

        return np.array(images), np.array(labels)


class ConsistencyDataset(PyDataset):
    def __init__(self, x, y, batch_size=128, training=True, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.training = training
        self.indices = np.arange(len(x))

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        if self.training:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        ids = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        clean, noisy, labels = [], [], []

        for i in ids:
            img, lbl = self.x[i], self.y[i]
            c, _ = preprocess_train(img, lbl, noisy=False)
            n, _ = preprocess_train(img, lbl, noisy=True)

            clean.append(c)
            noisy.append(n)
            labels.append(lbl)

        clean_batch = np.array(clean, dtype="float32")
        noisy_batch = np.array(noisy, dtype="float32")

        combined_x = np.concatenate([clean_batch, noisy_batch], axis=-1)

        return combined_x, np.array(labels)

```

Eval Dataset


```python

class EvalDataset(PyDataset):
    def __init__(self, x, y, batch_size=128, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        images = [preprocess_test(x, y)[0] for x, y in zip(batch_x, batch_y)]
        return np.array(images), np.array(batch_y)

```

We make sure `train_clean_ds` and `train_noisy_ds` are shuffled using the *same* seed to
ensure their orders are exactly the same. This will be helpful during training the
student model.


```python
# This dataset will be used to train the first model.
train_clean_ds = TeacherDataset(train_x, train_y, BATCH_SIZE, True)
consistency_training_ds = ConsistencyDataset(train_x, train_y, BATCH_SIZE, True)
validation_ds = EvalDataset(val_x, val_y, BATCH_SIZE)
test_ds = EvalDataset(x_test, y_test, BATCH_SIZE)
```

---
## Visualize the datasets


```python
batch_inputs, labels = next(iter(consistency_training_ds))

clean_imgs = batch_inputs[..., :3]
noisy_imgs = batch_inputs[..., 3:]
plt.figure(figsize=(10, 10))

# Clean images
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(clean_imgs[i].astype("uint8"))
    plt.axis("off")

# Noisy images
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(noisy_imgs[i].astype("uint8"))
    plt.axis("off")

plt.tight_layout()
plt.show()
```


    
![png](examples/vision/img/consistency_training/consistency_training_19_0.png)
    



    
![png](examples/vision/img/consistency_training/consistency_training_19_1)
    


---
## Define a model building utility function

We now define our model building utility. Our model is based on the [ResNet50V2 architecture](https://arxiv.org/abs/1603.05027).


```python

def get_training_model():
    base = keras.applications.ResNet50V2(
        weights=None,
        include_top=False,
        input_shape=(CROP_TO, CROP_TO, 3),
    )

    return keras.Sequential(
        [
            layers.Input((CROP_TO, CROP_TO, 3)),
            layers.Rescaling(1 / 127.5, offset=-1),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(NUM_CLASSES),
        ]
    )

```

In the interest of reproducibility, we serialize the initial random weights of the
teacher  network.


```python
initial_model = get_training_model()
initial_model.save("initial_teacher_model.keras")
initial_weights = initial_model.get_weights()
```

---
## Train the teacher model

As noted in Noisy Student Training, if the teacher model is trained with *geometric
ensembling* and when the student model is forced to mimic that, it leads to better
performance. The original work uses [Stochastic Depth](https://arxiv.org/abs/1603.09382)
and [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) to bring in the ensembling
part but for this example, we will use [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407)
(SWA) which also resembles geometric ensembling.


```python
# Define the callbacks.
reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=3)
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Compile and train the teacher model.


teacher_model = get_training_model()
teacher_model.set_weights(initial_weights)

teacher_model.compile(
    optimizer=Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

teacher_model.fit(
    train_clean_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=[
        ReduceLROnPlateau(patience=3),
        EarlyStopping(patience=5, restore_best_weights=True),
    ],
)

teacher_model.save("teacher_model.keras")


# Evaluate the teacher model on the test set.
_, acc = teacher_model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {acc*100}%")
```

<div class="k-default-codeblock">

Epoch 1/5

387/387 ━━━━━━━━━━━━━━━━━━━━ 683s 2s/step - accuracy: 0.4433 - loss: 1.5387 - val_accuracy: 0.4900 - val_loss: 1.4593 - learning_rate: 0.0010

Epoch 2/5

387/387 ━━━━━━━━━━━━━━━━━━━━ 725s 2s/step - accuracy: 0.5755 - loss: 1.1979 - val_accuracy: 0.5660 - val_loss: 1.4343 - learning_rate: 0.0010

Epoch 3/5

387/387 ━━━━━━━━━━━━━━━━━━━━ 714s 2s/step - accuracy: 0.6457 - loss: 1.0012 - val_accuracy: 0.5360 - val_loss: 1.7527 - learning_rate: 0.0010

Epoch 4/5

387/387 ━━━━━━━━━━━━━━━━━━━━ 730s 2s/step - accuracy: 0.6962 - loss: 0.8689 - val_accuracy: 0.6860 - val_loss: 1.0094 - learning_rate: 0.0010

Epoch 5/5

387/387 ━━━━━━━━━━━━━━━━━━━━ 728s 2s/step - accuracy: 0.7354 - loss: 0.7610 - val_accuracy: 0.7360 - val_loss: 0.8522 - learning_rate: 0.0010

Test accuracy: 69.760000705719%
```
</div>

The `DistillationModel` is a custom Keras model that takes in two inputs: the
clean images and the noisy images.


```python

class DistillationModel(keras.Model):
    def __init__(self, student, teacher, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        self.teacher_logits = None

    def call(self, inputs, training=False):
        inputs = ops.cast(inputs, "float32")
        if ops.shape(inputs)[-1] == 6:
            clean = inputs[:, :, :, 0:3]
            noisy = inputs[:, :, :, 3:6]
        else:
            clean = inputs
            noisy = inputs

        self.teacher_logits = self.teacher(clean, training=False)
        return self.student(noisy, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "student": keras.utils.serialize_keras_object(self.student),
                "teacher": keras.utils.serialize_keras_object(self.teacher),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        student_config = config.pop("student")
        teacher_config = config.pop("teacher")
        student = keras.utils.deserialize_keras_object(student_config)
        teacher = keras.utils.deserialize_keras_object(teacher_config)
        return cls(student, teacher, **config)

```

Distillation loss is a custom loss function that takes the true labels and the concatenated logits from the student and teacher models. It computes the standard
sparse categorical cross-entropy loss for the student model and the Kullback-Leibler divergence between the softened outputs of the teacher and student models. The final loss is the average of these two losses, following the No


```python

def distillation_loss(y_true, student_logits):
    teacher_logits = distill_model.teacher_logits
    student_loss = keras.losses.sparse_categorical_crossentropy(
        y_true, student_logits, from_logits=True
    )
    t_soft = ops.softmax(teacher_logits / TEMPERATURE, axis=-1)
    s_soft = ops.softmax(student_logits / TEMPERATURE, axis=-1)

    distill_kl = keras.losses.kl_divergence(t_soft, s_soft)

    return ops.mean(0.5 * student_loss + 0.5 * distill_kl)

```

The only difference in this implementation is the way loss is being calculated. **Instead
of weighted the distillation loss and student loss differently we are taking their
average following Noisy Student Training**.

---
## Train the student model


```python
# Define the callbacks.
# We are using a larger decay factor to stabilize the training.
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    patience=3, factor=0.5, monitor="val_accuracy"
)
early_stopping = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True, monitor="val_accuracy"
)

# Compile and train the student model.
student = get_training_model()
student.set_weights(initial_weights)
distill_model = DistillationModel(student, teacher_model)

distill_model.compile(optimizer=Adam(), loss=distillation_loss, metrics=["accuracy"])

history = distill_model.fit(
    consistency_training_ds,
    epochs=EPOCHS,
    validation_data=validation_ds,
    callbacks=[reduce_lr, early_stopping],
)

student.save("student_model_final.keras")

# Evaluate the student model.
_, acc = distill_model.evaluate(test_ds, verbose=0)
print(f"Test accuracy from student model: {acc*100}%")
```

<div class="k-default-codeblock">

Epoch 1/5

387/387 ━━━━━━━━━━━━━━━━━━━━ 904s 2s/step - accuracy: 0.4076 - loss: 0.8434 - val_accuracy: 0.4040 - val_loss: 0.9461 - learning_rate: 0.0010

Epoch 2/5

387/387 ━━━━━━━━━━━━━━━━━━━━ 901s 2s/step - accuracy: 0.5147 - loss: 0.6926 - val_accuracy: 0.5680 - val_loss: 0.6619 - learning_rate: 0.0010

Epoch 3/5

387/387 ━━━━━━━━━━━━━━━━━━━━ 917s 2s/step - accuracy: 0.5716 - loss: 0.6179 - val_accuracy: 0.5460 - val_loss: 0.7084 - learning_rate: 0.0010

Epoch 4/5

387/387 ━━━━━━━━━━━━━━━━━━━━ 960s 2s/step - accuracy: 0.6182 - loss: 0.5548 - val_accuracy: 0.5600 - val_loss: 0.7931 - learning_rate: 0.0010

Epoch 5/5

387/387 ━━━━━━━━━━━━━━━━━━━━ 1025s 3s/step - accuracy: 0.6485 - loss: 0.5139 - val_accuracy: 0.6040 - val_loss: 0.6095 - learning_rate: 0.0010

Test accuracy from student model: 62.01000213623047%

</div>


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
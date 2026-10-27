"""
Title: Blood cell classification from microscopy images
Authors: [Maitry](https://github.com/maitry63)
Date created: 2026/10/27
Last modified: 2026/10/27
Description: Train a multi-backend blood cell classifier on BloodMNIST with medicai and PyGrain.
Accelerator: GPU
"""

"""
## Introduction

Identifying the different types of cells in a peripheral blood smear is a
routine step in diagnosing infections, anemia, and blood cancers such as
leukemia. Doing it by eye under a microscope is slow and depends on the
experience of the observer, which makes it a natural task for automated image
classification.

In this example, we train a convolutional neural network to classify blood cell
microscopy images into eight cell types, using the BloodMNIST dataset. We rely
on [Mohammed Innat](https://www.linkedin.com/in/innat2k14/) 's
[medicai](https://github.com/innat/medic-ai), a Keras 3 library for 2D and
3D medical image analysis, for the model architecture, the data augmentation,
and the model explanation. We build the input pipeline with
[PyGrain](https://github.com/google/grain). Since neither library is tied to a
specific framework, the same code runs on the JAX, TensorFlow, and PyTorch
backends.

We cover the following steps:

1. Downloading the BloodMNIST dataset and exploring its classes.
2. Building a backend-agnostic input pipeline with PyGrain.
3. Augmenting whole batches with `medicai.transforms`.
4. Training an EfficientNetV2 classifier from `medicai.models` with `fit()`.
5. Evaluating the model on the held-out test set.
6. Explaining predictions with Gradient-weighted Class Activation Mapping (Grad-CAM).

This example is part of a series of medical imaging examples. For a 3D
segmentation workflow, see
[3D Multimodal Brain Tumor Segmentation](https://keras.io/examples/vision/brain_tumor_segmentation/).
"""

"""
## The BloodMNIST dataset

[MedMNIST](https://medmnist.com/) is a collection of standardized biomedical
image datasets, each with predefined training, validation, and test splits.
Its MedMNIST+ extension provides the 2D datasets at resolutions of `28x28`,
`64x64`, `128x128`, and `224x224` pixels.

BloodMNIST is based on a
[dataset of 17,092 microscopic images of normal peripheral blood cells](https://doi.org/10.1016/j.dib.2020.105474),
acquired with a CellaVision DM96 analyzer at the Hospital Clinic of Barcelona.
The images are center-cropped, resized, and split with a 7:1:2 ratio into
11,959 training, 1,712 validation, and 3,421 test images. Each image is
centered on a single cell that belongs to one of eight classes:

- basophils
- eosinophils
- erythroblasts (immature red blood cells)
- immature granulocytes (myelocytes, metamyelocytes, and promyelocytes)
- lymphocytes
- monocytes
- neutrophils
- platelets

BloodMNIST is released under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
"""

"""
## Setup

This example requires `medicai` and PyGrain. We pin `medicai` to a specific
commit to keep the example reproducible. You can install both with:

```shell
pip install grain
pip install git+https://github.com/innat/medic-ai.git@0e58c4ef54fa439507c1a8918848a2610c4a65ef
```

We use the TensorFlow backend, but you can switch to JAX or PyTorch by changing
the `KERAS_BACKEND` environment variable.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # "jax" or "torch".

import grain.python as pygrain
import keras
import math
import medicai
import numpy as np
from keras import ops
from matplotlib import pyplot as plt

keras.utils.set_random_seed(42)
# Mixed precision speeds up training on GPUs. Use "mixed_bfloat16" on TPUs.
keras.mixed_precision.set_global_policy("mixed_float16")

image_size = 128
batch_size = 64
epochs = 10
class_names = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocyte",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]
num_classes = len(class_names)

"""
## Download the data

MedMNIST distributes each dataset and resolution as a single NumPy archive on
[Zenodo](https://zenodo.org/records/10519652). We download the `128x128`
version of BloodMNIST (about 570 MB) and verify its MD5 checksum. You can
switch to another resolution by changing `image_size` above.
"""

# MD5 checksums of the BloodMNIST archives, by image resolution.
file_hashes = {
    64: "2b94928a2ae4916078ca51e05b6b800b",
    128: "adace1e0ed228fccda1f39692059dd4c",
    224: "b718ff6835fcbdb22ba9eacccd7b2601",
}
file_name = f"bloodmnist_{image_size}.npz"
data_path = keras.utils.get_file(
    fname=file_name,
    origin=f"https://zenodo.org/records/10519652/files/{file_name}?download=1",
    file_hash=file_hashes[image_size],
)

with np.load(data_path) as data:
    x_train, y_train = data["train_images"], data["train_labels"].ravel()
    x_val, y_val = data["val_images"], data["val_labels"].ravel()
    x_test, y_test = data["test_images"], data["test_labels"].ravel()

print(f"Training images: {x_train.shape}, {x_train.dtype}")
print(f"Validation images: {x_val.shape}")
print(f"Test images: {x_test.shape}")

"""
The images are stored as `uint8` RGB arrays and the labels as integer class
indices. Let's look at one training image from each class.
"""

plt.figure(figsize=(16, 3))
for class_index, class_name in enumerate(class_names):
    sample_index = np.flatnonzero(y_train == class_index)[0]
    plt.subplot(1, num_classes, class_index + 1)
    plt.imshow(x_train[sample_index])
    plt.title(class_name, fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.show()

"""
The classes are not balanced: neutrophils are almost three times as frequent
as lymphocytes or basophils. We keep this in mind when we look at per-class
results later.
"""

class_counts = np.bincount(y_train, minlength=num_classes)
plt.figure(figsize=(10, 4))
plt.bar(class_names, class_counts)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Number of training images")
plt.tight_layout()
plt.show()

"""
## Build the input pipeline with PyGrain

A PyGrain pipeline starts from a *random-access data source*: any object that
implements `__len__()` and `__getitem__()`. Our source simply indexes into the
in-memory NumPy arrays. Each sample is then scaled to the `[0, 1]` range.
"""


class BloodMNISTSource(pygrain.RandomAccessDataSource):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def preprocess_sample(sample):
    image, label = sample
    return image.astype("float32") / 255.0, np.int32(label)


"""
### Augment batches with medicai

Blood cells have no canonical orientation, so random flips and rotations are
safe augmentations. We combine them with rotations of up to about 30 degrees
(`rotation_factor` is expressed in radians), small random scaling and
translation, and a mild elastic deformation that mimics natural variations in
cell shape.

`medicai.transforms` implements these transforms with `keras.ops`, so they run
on any backend. We apply them to whole batches in the `BHWC` layout (batch,
height, width, channels). Each random transform gets its own
`keras.random.SeedGenerator`, so that every call draws new random parameters.
We create and run the pipeline on the CPU, inside the PyGrain worker threads,
which leaves the accelerator free for the model.
"""

with keras.device("cpu:0"):
    train_augmentation = medicai.transforms.Compose(
        [
            medicai.transforms.RandomFlip(
                keys=["image"],
                spatial_axis=[1, 2],
                prob=0.5,
                input_layout="BHWC",
                seed=keras.random.SeedGenerator(1),
            ),
            medicai.transforms.RandomAffine(
                keys=["image"],
                rotation_factor=0.5,
                scale_factor=0.1,
                translation_factor=0.05,
                interpolation={"image": "bilinear"},
                fill_mode="reflect",
                prob=0.8,
                input_layout="BHWC",
                seed=keras.random.SeedGenerator(2),
            ),
            medicai.transforms.RandomElasticTransform(
                keys=["image"],
                alpha=(1.0, 3.0),
                sigma=4.0,
                control_grid_spacing=(16, 16),
                interpolation={"image": "bilinear"},
                fill_mode="reflect",
                prob=0.3,
                input_layout="BHWC",
                seed=keras.random.SeedGenerator(3),
            ),
        ]
    )


def augment_batch(batch):
    images, labels = batch
    with keras.device("cpu:0"):
        images = train_augmentation({"image": images})["image"]
    return ops.convert_to_numpy(images), labels


"""
### Assemble the datasets

The training pipeline shuffles the samples, preprocesses them, batches them,
and augments each batch. Calling `repeat()` after `shuffle()` turns the
training data into an infinite stream with a different order in every epoch.
The validation and test pipelines only preprocess and batch the samples.
Finally, `to_iter_dataset()` reads and transforms the batches in parallel
background threads.

Keras models consume PyGrain datasets directly in `fit()`, `evaluate()`, and
`predict()`, with every backend. However, Keras does not know in advance how
many batches a PyGrain dataset yields, so we compute these numbers ourselves
and pass them as `steps_per_epoch`, `validation_steps`, and `steps`.
"""


def make_dataset(images, labels, training=False):
    dataset = pygrain.MapDataset.source(BloodMNISTSource(images, labels))
    if training:
        dataset = dataset.shuffle(seed=42).repeat()
    dataset = dataset.map(preprocess_sample)
    dataset = dataset.batch(batch_size, drop_remainder=training)
    if training:
        dataset = dataset.map(augment_batch)
    return dataset.to_iter_dataset(read_options=pygrain.ReadOptions(num_threads=4))


train_dataset = make_dataset(x_train, y_train, training=True)
val_dataset = make_dataset(x_val, y_val)
test_dataset = make_dataset(x_test, y_test)
steps_per_epoch = len(x_train) // batch_size
validation_steps = math.ceil(len(x_val) / batch_size)
test_steps = math.ceil(len(x_test) / batch_size)

"""
Let's check that the augmented training images still look realistic.
"""

images, labels = next(iter(train_dataset))
plt.figure(figsize=(12, 6))
for index in range(8):
    plt.subplot(2, 4, index + 1)
    plt.imshow(np.clip(images[index], 0.0, 1.0))
    plt.title(class_names[labels[index]])
    plt.axis("off")
plt.tight_layout()
plt.show()

"""
## Build the model

We use EfficientNetV2-B1 from `medicai.models`.
[EfficientNetV2](https://arxiv.org/abs/2104.00298) is a family of
convolutional networks designed for fast training and good parameter
efficiency. medicai infers whether to build the 2D or the 3D variant of an
architecture from the rank of `input_shape`. With `include_top=True`, it adds a
global pooling layer and a softmax classifier with `num_classes` outputs. We
train the network from scratch.
"""

model = medicai.models.EfficientNetV2B1(
    input_shape=(image_size, image_size, 3),
    include_top=True,
    num_classes=num_classes,
    classifier_activation="softmax",
)
print(f"Number of parameters: {model.count_params() / 1e6:.2f}M")

"""
Batch normalization layers normalize their inputs with batch statistics during
training, but with moving averages of these statistics during inference. In
this model, the first (`stem_bn`) and the last (`top_bn`) batch normalization
layers update their moving averages with the default momentum of `0.99`, while
all the other ones use `0.9`, like the reference EfficientNetV2 implementation.
With a momentum of `0.99`, the moving averages lag behind the weights, which
change quickly at the beginning of training. As a result, the validation
accuracy stays far below the training accuracy during the first epochs. We
therefore set the momentum of every batch normalization layer to `0.9`.
"""

for layer in model.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.momentum = 0.9

"""
Training from scratch benefits from a relatively high initial learning rate
that decays over time, so we use AdamW with a cosine decay schedule. Since the
labels are integer class indices, we use the sparse categorical cross-entropy
loss.
"""

learning_rate = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=epochs * steps_per_epoch
)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

"""
## Train the model

We monitor the accuracy on the validation set after every epoch and keep the
weights of the best epoch. We train for 10 epochs to keep the run time short;
training for longer, or at a higher resolution, can further improve the
results.
"""

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "bloodmnist.weights.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)
history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[checkpoint_callback],
    shuffle=False,  # The PyGrain pipeline already shuffles the training data.
)

plt.figure(figsize=(12, 4))
for plot_index, metric in enumerate(["loss", "accuracy"]):
    plt.subplot(1, 2, plot_index + 1)
    plt.plot(history.history[metric], label="training")
    plt.plot(history.history[f"val_{metric}"], label="validation")
    plt.xlabel("Epoch")
    plt.title(metric.capitalize())
    plt.legend()
plt.show()

"""
## Evaluate the model

We restore the weights of the best epoch and evaluate them on the test set,
which the model has not seen during training.
"""

model.load_weights("bloodmnist.weights.h5")
test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_steps, verbose=0)
print(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.2%}")

"""
Overall accuracy can hide weaknesses on the less frequent classes, so we also
compute the confusion matrix, along with the precision and recall of each
class.
"""

predictions = model.predict(test_dataset, steps=test_steps, verbose=0).argmax(axis=-1)
confusion = np.zeros((num_classes, num_classes), dtype="int64")
np.add.at(confusion, (y_test, predictions), 1)
precision = confusion.diagonal() / np.maximum(confusion.sum(axis=0), 1)
recall = confusion.diagonal() / confusion.sum(axis=1)
for class_name, class_precision, class_recall in zip(class_names, precision, recall):
    print(
        f"{class_name:>21}: precision={class_precision:.3f}, recall={class_recall:.3f}"
    )

normalized_confusion = confusion / confusion.sum(axis=1, keepdims=True)
plt.figure(figsize=(8, 7))
plt.imshow(normalized_confusion, cmap="Blues", vmin=0.0, vmax=1.0)
for true_index in range(num_classes):
    for predicted_index in range(num_classes):
        value = normalized_confusion[true_index, predicted_index]
        color = "white" if value > 0.5 else "black"
        plt.text(predicted_index, true_index, f"{value:.2f}", ha="center", color=color)
plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
plt.yticks(range(num_classes), class_names)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.tight_layout()
plt.show()

"""
## Explain predictions with Grad-CAM

[Grad-CAM](https://arxiv.org/abs/1610.02391) highlights the image regions that
contribute the most to a class score. It computes the gradients of that score
with respect to the feature maps of a late convolutional layer, uses them to
weight the feature maps, and sums the result into a coarse heatmap. In medical
imaging, Grad-CAM is a useful sanity check: a trustworthy cell classifier should
focus on the cell that it classifies, not on the background or on neighboring
red blood cells.

`medicai.utils.GradCAM` supports 2D and 3D classification and segmentation
models. We target `top_activation`, the last activation layer of EfficientNetV2
before global pooling. When `target_class_index` is not specified, Grad-CAM
explains the class predicted for each image. We apply it to one test image
from each class; the heatmaps are normalized to `[0, 1]` and resized to the
input resolution.
"""

grad_cam = medicai.utils.GradCAM(model, target_layer="top_activation", task_type="auto")

sample_indices = [np.flatnonzero(y_test == index)[0] for index in range(num_classes)]
images = x_test[sample_indices].astype("float32") / 255.0
heatmaps = grad_cam.compute_heatmap(images)
heatmaps = ops.convert_to_numpy(heatmaps).reshape(num_classes, image_size, image_size)

plt.figure(figsize=(16, 5))
for index, sample_index in enumerate(sample_indices):
    predicted_name = class_names[predictions[sample_index]]
    plt.subplot(2, num_classes, index + 1)
    plt.imshow(images[index])
    plt.title(f"True: {class_names[index]}\nPred: {predicted_name}", fontsize=9)
    plt.axis("off")
    plt.subplot(2, num_classes, num_classes + index + 1)
    plt.imshow(images[index])
    plt.imshow(heatmaps[index], cmap="jet", alpha=0.4)
    plt.axis("off")
plt.tight_layout()
plt.show()

"""
## Conclusion

In this example, we built an end-to-end blood cell classification workflow with
Keras 3 and medicai:

- PyGrain provides a fast input pipeline with reproducible shuffling, which
`fit()`, `evaluate()`, and `predict()` consume directly, whatever the backend.
- `medicai.transforms` provides augmentations implemented with `keras.ops`,
which we apply to whole batches on the CPU while the model trains.
- `medicai.models` provides 2D and 3D versions of common architectures, such as
EfficientNetV2, that train with the standard Keras API.
- `medicai.utils.GradCAM` lets us check that the model focuses on the cell that
it classifies.

To go further, you can:

- Train for more epochs, or switch to the `224x224` version of BloodMNIST.
- Try other medicai architectures, which you can list with
`medicai.models.list_models()`.
- Address the class imbalance, for example by weighting the loss or by
oversampling the rare classes in the data source.
- Apply the same pipeline to other 2D MedMNIST datasets, such as PathMNIST or
DermaMNIST, by updating the file name, the checksums, and the class names.

## References

- [MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification](https://www.nature.com/articles/s41597-022-01721-8)
- [Rethinking model prototyping through the MedMNIST+ dataset collection](https://arxiv.org/abs/2404.15786)
- [A dataset of microscopic peripheral blood cell images for development of automatic recognition systems](https://doi.org/10.1016/j.dib.2020.105474)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
"""

"""
Title: Image classification via fine-tuning with EfficientNet
Author: [Yixing Fu](https://github.com/yixingfu)
Date created: 2020/06/30
Last modified: 2026/07/13
Description: Use EfficientNet with weights pre-trained on imagenet for Stanford Dogs classification.
Accelerator: GPU
"""

"""

## Introduction: what is EfficientNet

EfficientNet, first introduced in [Tan and Le, 2019](https://arxiv.org/abs/1905.11946)
is among the most efficient models (i.e. requiring least FLOPS for inference)
that reaches State-of-the-Art accuracy on both
imagenet and common image classification transfer learning tasks.

The smallest base model is similar to [MnasNet](https://arxiv.org/abs/1807.11626), which
reached near-SOTA with a significantly smaller model. By introducing a heuristic way to
scale the model, EfficientNet provides a family of models (B0 to B7) that represents a
good combination of efficiency and accuracy on a variety of scales. Such a scaling
heuristics (compound-scaling, details see
[Tan and Le, 2019](https://arxiv.org/abs/1905.11946)) allows the
efficiency-oriented base model (B0) to surpass models at every scale, while avoiding
extensive grid-search of hyperparameters.

A summary of the latest updates on the model is available at
[here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), where various
augmentation schemes and semi-supervised learning approaches are applied to further
improve the imagenet performance of the models. These extensions of the model can be used
by updating weights without changing model architecture.

## B0 to B7 variants of EfficientNet

*(This section provides some details on "compound scaling", and can be skipped
if you're only interested in using the models)*

Based on the [original paper](https://arxiv.org/abs/1905.11946) people may have the
impression that EfficientNet is a continuous family of models created by arbitrarily
choosing scaling factor in as Eq.(3) of the paper.  However, choice of resolution,
depth and width are also restricted by many factors:

- Resolution: Resolutions not divisible by 8, 16, etc. cause zero-padding near boundaries
of some layers which wastes computational resources. This especially applies to smaller
variants of the model, hence the input resolution for B0 and B1 are chosen as 224 and
240.

- Depth and width: The building blocks of EfficientNet demands channel size to be
multiples of 8.

- Resource limit: Memory limitation may bottleneck resolution when depth
and width can still increase. In such a situation, increasing depth and/or
width but keep resolution can still improve performance.

As a result, the depth, width and resolution of each variant of the EfficientNet models
are hand-picked and proven to produce good results, though they may be significantly
off from the compound scaling formula.
Therefore, the keras implementation (detailed below) only provide these 8 models, B0 to B7,
instead of allowing arbitray choice of width / depth / resolution parameters.

## Keras implementation of EfficientNet

An implementation of EfficientNet B0 to B7 has been shipped with Keras since v2.3. To
use EfficientNetB0 for classifying 1000 classes of images from ImageNet, run:

```python
from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights='imagenet')
```

This model takes input images of shape `(224, 224, 3)`, and the input data should be in the
range `[0, 255]`. Normalization is included as part of the model.

Because training EfficientNet on ImageNet takes a tremendous amount of resources and
several techniques that are not a part of the model architecture itself. Hence the Keras
implementation by default loads pre-trained weights obtained via training with
[AutoAugment](https://arxiv.org/abs/1805.09501).

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

When the model is intended for transfer learning, the Keras implementation
provides a option to remove the top layers:
```
model = EfficientNetB0(include_top=False, weights='imagenet')
```
This option excludes the final `Dense` layer that turns 1280 features on the penultimate
layer into prediction of the 1000 ImageNet classes. Replacing the top layer with custom
layers allows using EfficientNet as a feature extractor in a transfer learning workflow.

Another argument in the model constructor worth noticing is `drop_connect_rate` which controls
the dropout rate responsible for [stochastic depth](https://arxiv.org/abs/1603.09382).
This parameter serves as a toggle for extra regularization in finetuning, but does not
affect loaded weights. For example, when stronger regularization is desired, try:

```python
model = EfficientNetB0(weights='imagenet', drop_connect_rate=0.4)
```
The default value is 0.2.

## Example: EfficientNetB0 for Stanford Dogs.

EfficientNet is capable of a wide range of image classification tasks.
This makes it a good model for transfer learning.
As an end-to-end example, we will show using pre-trained EfficientNetB0 on
[Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) dataset.

"""

"""
## Setup and data loading
"""

import os
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras import layers
from keras.applications import EfficientNetB0

# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224
BATCH_SIZE = 64


"""
### Loading data

We download the Stanford Dogs dataset directly from Stanford's servers using the
built-in `urllib` and `tarfile` modules.
The dataset contains 20,580 images belonging to 120 classes of dog breeds
(12,000 for training and 8,580 for testing).

The dataset is downloaded, extracted, and loaded into lists of NumPy arrays.
Images have variable dimensions and will be resized to a uniform size in the data pipeline.

**Note:** This direct download approach eliminates dependency conflicts that can
occur with `tensorflow_datasets` in some environments (particularly Google Colab
with protobuf version incompatibilities).
"""

# Download and extract Stanford Dogs dataset
import scipy.io

dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
lists_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
data_dir = Path("./stanford_dogs_data")
data_dir.mkdir(exist_ok=True)


def download_and_extract(url, extract_to):
    filename = url.split("/")[-1]
    filepath = data_dir / filename
    if not filepath.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Extracting {filename}...")
        with tarfile.open(filepath, "r") as tar:
            tar.extractall(extract_to)
    return extract_to


# Download dataset
images_dir = download_and_extract(dataset_url, data_dir)
lists_dir = download_and_extract(lists_url, data_dir)


# Parse train/test splits
def load_file_list(filepath):
    mat = scipy.io.loadmat(filepath)
    return [item[0][0] for item in mat["file_list"]]


train_files = load_file_list(data_dir / "train_list.mat")
test_files = load_file_list(data_dir / "test_list.mat")

# Build class name mapping
all_files = train_files + test_files
class_names = sorted(set([f.split("/")[0] for f in all_files]))
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)

print(
    f"Found {NUM_CLASSES} classes, {len(train_files)} training images, {len(test_files)} test images"
)


# Load images and labels
def load_images_and_labels(file_list, base_dir):
    images, labels = [], []
    for file_path in file_list:
        class_name = file_path.split("/")[0]
        img_path = base_dir / "Images" / file_path
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(np.array(img))
            labels.append(class_to_idx[class_name])
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
    return images, np.array(labels)


print("Loading training images...")
train_images, train_labels = load_images_and_labels(train_files, data_dir)
print("Loading test images...")
test_images, test_labels = load_images_and_labels(test_files, data_dir)
print(f"Loaded {len(train_images)} train and {len(test_images)} test images")


"""
Each image can have a different shape, so we resize them to a shared input size
for EfficientNet. In Keras 3, we do this in a backend-agnostic `PyDataset` pipeline
using `keras.ops.image.resize`, which works across TensorFlow, JAX, and PyTorch backends,
rather than the TensorFlow-specific `tf.data` mapping steps.
"""


class ResizeOnlyDataset(keras.utils.PyDataset):
    def __init__(self, images, labels, img_size, batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.labels = labels
        self.img_size = img_size
        self.batch_size = batch_size
        self.indices = np.arange(len(labels))

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_images = np.stack(
            [
                np.array(
                    keras.ops.image.resize(
                        np.array(self.images[i], dtype="float32"),
                        (self.img_size, self.img_size),
                    )
                )
                for i in batch_indices
            ]
        )
        batch_labels = self.labels[batch_indices]
        if self.batch_size == 1:
            return batch_images[0], batch_labels[0]
        return batch_images, batch_labels


# Preview stream with resized images for visualization below
preview_train = ResizeOnlyDataset(train_images, train_labels, IMG_SIZE, batch_size=1)
preview_test = ResizeOnlyDataset(test_images, test_labels, IMG_SIZE, batch_size=1)

"""
### Visualizing the data

The following code shows the first 9 images with their labels.
"""


def format_label(label):
    class_name = class_names[int(label)]
    return class_name.split("-")[1]  # Extract breed name from "n02085620-Chihuahua"


for i, (image, label) in enumerate(preview_train):
    if i >= 9:
        break
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(np.array(image).astype("uint8"))
    plt.title("{}".format(format_label(label)))
    plt.axis("off")


"""
### Data augmentation

We can use Keras preprocessing layers for image augmentation.
These layers are backend-agnostic and can be used during both training and inference.
"""

img_augmentation_layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
]


def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images


"""
The `img_augmentation` function can be used both as a part of the model
we later build, and as a standalone function to preprocess data before feeding
into the model. Using it as a function makes it easy to visualize the augmentation
results. Here we plot 9 examples of augmentation applied to a single image.
"""

first_image, first_label = preview_train[0]

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    aug_img = img_augmentation(np.expand_dims(np.array(first_image), axis=0))
    aug_img = np.array(aug_img)
    plt.imshow(aug_img[0].astype("uint8"))
    plt.title("{}".format(format_label(first_label)))
    plt.axis("off")


"""
### Prepare inputs

Once we verify the input data and augmentation are working correctly,
we prepare backend-agnostic datasets for training.

The input images are resized to uniform `IMG_SIZE`, labels are converted to one-hot
(categorical) encoding, and batches are produced by `keras.utils.PyDataset`.

Compared to the original `tf.data` version, this Keras 3 setup is backend-agnostic
and works seamlessly across TensorFlow, JAX, and PyTorch backends.
"""


class StanfordDogsDataset(keras.utils.PyDataset):
    def __init__(
        self,
        images,
        labels,
        num_classes,
        img_size,
        batch_size,
        augment=False,
        shuffle=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.images = images
        self.labels = labels
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(labels))
        self.on_epoch_end()

    def __len__(self):
        # Match previous drop_remainder=True behavior.
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_images = np.stack(
            [
                np.array(
                    keras.ops.image.resize(
                        np.array(self.images[i], dtype="float32"),
                        (self.img_size, self.img_size),
                    )
                )
                for i in batch_indices
            ]
        )

        if self.augment:
            batch_images = np.array(img_augmentation(batch_images))

        batch_labels = np.array(
            keras.ops.one_hot(self.labels[batch_indices], self.num_classes)
        )

        return batch_images, batch_labels


ds_train = StanfordDogsDataset(
    train_images,
    train_labels,
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    augment=True,
    shuffle=True,
    workers=2,
    use_multiprocessing=False,
)

ds_test = StanfordDogsDataset(
    test_images,
    test_labels,
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    augment=False,
    shuffle=False,
)


"""
## Training a model from scratch

We build an EfficientNetB0 with 120 output classes, initialized from scratch
(no pretrained weights).

**Note:** Training from scratch typically shows slower convergence and may overfit
on smaller datasets like Stanford Dogs.
"""

model = EfficientNetB0(
    include_top=True,
    weights=None,
    classes=NUM_CLASSES,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

epochs = 40  # @param {type: "slider", min:10, max:100}
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test)


"""
Training the model is relatively fast (a few minutes per epoch on modern hardware).
However, training EfficientNet on smaller datasets,
especially those with lower resolution like CIFAR-100, faces the significant challenge of
overfitting.

Training from scratch requires very careful choice of hyperparameters and
suitable regularization. It is also much more demanding in computational resources.
Plotting the training and validation accuracy
makes it clear that validation accuracy stagnates at a low value.
"""

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

"""
## Transfer learning from pretrained weights

Here we initialize the model with pretrained ImageNet weights
and fine-tune it on our own dataset. This is the recommended approach
for most applications.
"""


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


"""
The first step in transfer learning is to freeze all base layers and train only the top
layers. For this step, a relatively large learning rate (1e-2) can be used.

**Note:** Validation accuracy and loss will usually be better than training
accuracy and loss. This is because the regularization is strong, which only
suppresses training-time metrics.

The convergence may take up to 50 epochs depending on the choice of learning rate.
If image augmentation layers were not applied, the validation accuracy may only reach ~60%.
"""

model = build_model(num_classes=NUM_CLASSES)

epochs = 25  # @param {type: "slider", min:8, max:80}
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test)
plot_hist(hist)

"""
The second step is to unfreeze a number of layers and fine-tune the model using a smaller
learning rate. In this example we unfreeze the last 20 layers, but depending on the
specific dataset it may be desirable to only unfreeze a fraction of all layers.

**Advanced usage:** The `unfreeze_model()` function also supports unfreezing by block name
(e.g., `unfreeze_model(model, layers_to_unfreeze="block7")`) to respect EfficientNet's
residual block boundaries. See the "Tips for fine-tuning EfficientNet" section below
for why this matters.

When feature extraction with the pretrained model works well enough, this step provides
only a limited gain in validation accuracy. In our case we only see a small improvement,
as ImageNet pretraining already exposed the model to a good amount of dog images.

On the other hand, when we use pretrained weights on a dataset that is more different
from ImageNet, this fine-tuning step can be crucial as the feature extractor also
needs to be adjusted by a considerable amount. Such a situation can be demonstrated
if choosing CIFAR-100 dataset instead, where fine-tuning boosts validation accuracy
by about 10% to pass 80% on `EfficientNetB0`.

**Note on freezing/unfreezing models:** Setting `trainable` of a `Model` will
simultaneously set all layers belonging to the `Model` to the same `trainable`
attribute. Each layer is trainable only if both the layer itself and the model
containing it are trainable. Hence when we need to partially freeze/unfreeze
a model, we need to make sure the `trainable` attribute of the model is set
to `True`.
"""


def unfreeze_model(
    model,
    layers_to_unfreeze=20,
    learning_rate=1e-5,
    loss="categorical_crossentropy",
    metrics=None,
):
    """Unfreeze part of `model` and recompile it for fine-tuning.

    Args:
        model: A `keras.Model` instance to unfreeze in place.
        layers_to_unfreeze: Either an `int` giving the number of layers,
            counted from the end of `model.layers`, to unfreeze, or a `str`
            substring to match against layer names -- the first matching
            layer and every layer after it (in `model.layers` order) are
            unfrozen. Use a string like `"block7"` to respect EfficientNet's
            residual block boundaries instead of an arbitrary layer count.
            Defaults to `20`.
        learning_rate: Learning rate for the fine-tuning `Adam` optimizer.
            Defaults to `1e-5`.
        loss: Loss function passed to `model.compile()`. Defaults to
            `"categorical_crossentropy"`.
        metrics: List of metrics passed to `model.compile()`. Defaults to
            `["accuracy"]`.

    Returns:
        `model`, with the selected layers unfrozen (except
        `BatchNormalization` layers, which are always kept frozen) and
        recompiled with the new optimizer/loss/metrics.
    """
    if metrics is None:
        metrics = ["accuracy"]

    if isinstance(layers_to_unfreeze, str):
        unfreeze_from = None
        for i, layer in enumerate(model.layers):
            if layers_to_unfreeze in layer.name:
                unfreeze_from = i
                break
        if unfreeze_from is None:
            raise ValueError(f"No layer name contains {layers_to_unfreeze!r}.")
        layers_to_process = model.layers[unfreeze_from:]
    elif isinstance(layers_to_unfreeze, int):
        n_layers_to_unfreeze = min(layers_to_unfreeze, len(model.layers))
        layers_to_process = model.layers[-n_layers_to_unfreeze:]
    else:
        raise TypeError(
            "layers_to_unfreeze must be an int or str, received: "
            f"{type(layers_to_unfreeze)}"
        )

    # We keep BatchNorm layers frozen -- see "Tips for fine-tuning
    # EfficientNet" in the tutorial for why.
    for layer in layers_to_process:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


model = unfreeze_model(model)

epochs = 4  # @param {type: "slider", min:4, max:10}
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test)
plot_hist(hist)

"""
### Tips for fine-tuning EfficientNet

**On unfreezing layers:**

- The `BatchNormalization` layers need to be kept frozen
([more details](https://keras.io/guides/transfer_learning/)).
If they are also turned to trainable, the
first epoch after unfreezing will significantly reduce accuracy.
- In some cases it may be beneficial to unfreeze only a portion of layers instead of
unfreezing all. This will make fine-tuning much faster when going to larger models like
B7.
- Each block needs to be all turned on or off. This is because the architecture includes
a shortcut from the first layer to the last layer for each block. Not respecting blocks
also significantly harms the final performance.

**Some other tips for utilizing EfficientNet:**

- Larger variants of EfficientNet do not guarantee improved performance, especially for
tasks with less data or fewer classes. In such a case, the larger the variant of EfficientNet
chosen, the harder it is to tune hyperparameters.
- EMA (Exponential Moving Average) is very helpful in training EfficientNet from scratch,
but not so much for transfer learning.
- Do not use the RMSprop setup as in the original paper for transfer learning. The
momentum and learning rate are too high for transfer learning. It will easily corrupt the
pretrained weights and blow up the loss. A quick check is to see if loss (as categorical
cross entropy) is getting significantly larger than log(NUM_CLASSES) after the same
epoch. If so, the initial learning rate/momentum is too high.
- Smaller batch sizes benefit validation accuracy, possibly due to effectively providing
regularization.
"""

"""
## Relevant Chapters from Deep Learning with Python
- [Chapter 8: Image classification](https://deeplearningwithpython.io/chapters/chapter08_image-classification)
"""

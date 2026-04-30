"""
Title: Metric learning for image similarity search.
Author: [Owen Vallis](https://twitter.com/owenvallis)
Date created: 2021/09/30
Last modified: 2026/04/29
Description: Example of using similarity metric learning on CIFAR-10 images.
Accelerator: GPU

Converted to Keras 3 by: [Maitry Sinha](https://github.com/maitry63)
"""

"""
## Overview

This example is based on the
["Metric learning for image similarity search" example](https://keras.io/examples/vision/metric_learning/).
We aim to use the same data set but implement the model using Keras.

Metric learning aims to train models that can embed inputs into a
high-dimensional space such that "similar" inputs are pulled closer to each
other and "dissimilar" inputs are pushed farther apart. Once trained, these
models can produce embeddings for downstream systems where such similarity is
useful, for instance as a ranking signal for search or as a form of pretrained
embedding model for another supervised problem.

For a more detailed overview of metric learning, see:

* [What is metric learning?](http://contrib.scikit-learn.org/metric-learn/introduction.html)
* ["Using crossentropy for metric learning" tutorial](https://www.youtube.com/watch?v=Jb4Ewl5RzkI)
"""

"""
## Setup

This tutorial will use Keras's built-in CIFAR-10 dataset and a custom `PyDataset` implementation
to learn and evaluate the similarity embedding.
Keras provides components that:

* Make training contrastive models simple and fast.
* Make it easier to ensure that batches contain pairs of examples.
* Enable the evaluation of the quality of the embedding.
"""

import random
import numpy as np
import keras
from keras import ops
from keras import layers
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1

print("Keras version:", keras.__version__)
print("Backend:", keras.backend.backend())

"""
## Dataset samplers

We will be using the
[CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10)
dataset for this tutorial.

For a similarity model to learn efficiently, each batch must contain at least 2
examples of each class.

To make this easy, we implement a `PyDataset` that enable you to set both
the number of classes and the minimum number of examples of each class per
batch.

The training and validation datasets will be created using the
`SimilarityDataset` object. This creates a sampler that loads datasets
and yields batches containing a target number of classes and a target number of examples
per class. Additionally, we can restrict the sampler to only yield the subset of
classes defined in `class_list`, enabling us to train on a subset of the classes
and then test how the embedding generalizes to the unseen classes. This can be
useful when working on few-shot learning problems.

The following cell creates a train_ds sample that:

* Loads the CIFAR-10 dataset and then takes the `examples_per_class_per_batch`.
* Ensures the sampler restricts the classes to those defined in `class_list`.
* Ensures each batch contains 10 different classes with 8 examples each.

We also create a validation dataset in the same way, but we limit the total number of
examples per class to 100 and the examples per class per batch is set to the
default of 2.
"""


class SimilarityDataset(keras.utils.PyDataset):
    def __init__(
        self,
        x,
        y,
        classes_per_batch,
        examples_per_class,
        steps_per_epoch=4000,
        class_list=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.x = x.astype("float32")
        self.y = y.flatten()
        self.classes_per_batch = classes_per_batch
        self.examples_per_class = examples_per_class
        self.steps_per_epoch = steps_per_epoch

        if class_list is not None:
            mask = np.isin(self.y, class_list)
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.class_list = class_list
        else:
            self.class_list = list(range(10))

        self.class_indices = [np.where(self.y == i)[0] for i in self.class_list]

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = [], []
        selected_classes = np.random.choice(
            len(self.class_indices), self.classes_per_batch, replace=False
        )
        for cls_idx in selected_classes:
            indices = np.random.choice(
                self.class_indices[cls_idx], self.examples_per_class, replace=False
            )
            x_batch.append(self.x[indices])
            y_batch.append(self.y[indices])
        return np.concatenate(x_batch), np.concatenate(y_batch)

    def get_slice(self, begin, size):
        if size == -1:
            return self.x[begin:], self.y[begin:]
        return self.x[begin : begin + size], self.y[begin : begin + size]


# Loading CIFAR-10
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = (
    keras.datasets.cifar10.load_data()
)

# This determines the number of classes used during training.
# Here we are using all the classes.
num_known_classes = 10
class_list = random.sample(population=range(10), k=num_known_classes)

classes_per_batch = 10
# Passing multiple examples per class per batch ensures that each example has
# multiple positive pairs. This can be useful when performing triplet mining or
# when using losses like `MultiSimilarityLoss` or `CircleLoss` as these can
# take a weighted mix of all the positive pairs. In general, more examples per
# class will lead to more information for the positive pairs, while more classes
# per batch will provide more varied information in the negative pairs. However,
# the losses compute the pairwise distance between the examples in a batch so
# the upper limit of the batch size is restricted by the memory.
examples_per_class_per_batch = 8

print(
    "Batch size is: "
    f"{min(classes_per_batch, num_known_classes) * examples_per_class_per_batch}"
)

print(" Create Training Data ".center(34, "#"))
train_ds = SimilarityDataset(
    x_train_raw,
    y_train_raw,
    classes_per_batch=min(classes_per_batch, num_known_classes),
    examples_per_class=examples_per_class_per_batch,
    steps_per_epoch=4000,
    class_list=class_list,
)

print("\n" + " Create Validation Data ".center(34, "#"))
val_ds = SimilarityDataset(
    x_test_raw,
    y_test_raw,
    classes_per_batch=classes_per_batch,
    examples_per_class=2,
    steps_per_epoch=50,
)

"""
## Visualize the dataset

The samplers will shuffle the dataset, so we can get a sense of the dataset by
plotting the first 25 images.

The samplers provide a `get_slice(begin, size)` method that allows us to easily
select a block of samples.

Alternatively, we can use the `generate_batch()` method to yield a batch. This
can allow us to check that a batch contains the expected number of classes and
examples per class.
"""

num_cols = num_rows = 5
# Get the first 25 examples.
x_slice, y_slice = train_ds.get_slice(begin=0, size=num_cols * num_rows)

fig = plt.figure(figsize=(6.0, 6.0))
grid = axes_grid1.ImageGrid(fig, 111, nrows_ncols=(num_cols, num_rows), axes_pad=0.1)

for ax, im, label in zip(grid, x_slice, y_slice):
    ax.imshow(im.astype("uint8"))
    ax.axis("off")

"""
## Embedding model

Next we define a `SimilarityModel` using the Keras Functional API. The model
is a standard convnet with the addition of a `MetricEmbedding` layer that
applies L2 normalization. The metric embedding layer is helpful when using
`Cosine` distance as we only care about the angle between the vectors.

Additionally, the `SimilarityModel` provides a number of helper methods for:

* Indexing embedded examples
* Performing example lookups
* Evaluating the classification
* Evaluating the quality of the embedding space

See the [TensorFlow Similarity documentation](https://github.com/tensorflow/similarity)
for more details.
"""

embedding_size = 256

inputs = keras.layers.Input((32, 32, 3))
x = keras.layers.Rescaling(scale=1.0 / 255)(inputs)
x = keras.layers.Conv2D(64, 3, activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(128, 3, activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPool2D((4, 4))(x)
x = keras.layers.Conv2D(256, 3, activation="relu")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(256, 3, activation="relu")(x)
x = keras.layers.GlobalMaxPool2D()(x)

# MetricEmbedding logic: Dense projection followed by L2 Normalization
x = layers.Dense(embedding_size)(x)
outputs = layers.Lambda(lambda v: ops.normalize(v, axis=1))(x)

# building model
model = keras.Model(inputs, outputs)
model.summary()


# Helper class to replicate Index/Lookup/Calibration logic from SimilarityModel
class SimilarityWrapper:
    def __init__(self, model):
        self.model = model
        self.embeddings = None
        self.labels = None
        self.data = None

    def reset_index(self):
        self.embeddings, self.labels, self.data = None, None, None

    def index(self, x, y, data=None):
        self.embeddings = self.model.predict(x, verbose=0)
        self.labels = np.array(y).flatten()
        self.data = x if data is None else data

    def lookup(self, x_query, k=5):
        query_emb = self.model.predict(x_query, verbose=0)
        sims = ops.convert_to_numpy(
            ops.matmul(query_emb, ops.transpose(self.embeddings))
        )
        indices = np.argsort(sims, axis=1)[:, ::-1][:, :k]
        results = []
        for i in range(len(x_query)):
            res = []
            for idx in indices[i]:
                res.append(
                    {
                        "label": self.labels[idx],
                        "distance": 1.0 - sims[i, idx],
                        "data": self.data[idx],
                    }
                )
            results.append(res)
        return results

    def calibrate(self, x_val, y_val, **kwargs):
        embs = self.model.predict(x_val, verbose=0)
        dists = 1.0 - ops.convert_to_numpy(ops.matmul(embs, ops.transpose(embs)))
        y_val = y_val.flatten()
        adj = (y_val[:, None] == y_val[None, :]).astype(float)
        mask = 1.0 - np.eye(len(y_val))
        thresholds = np.linspace(0, 1, 100)
        p_list, r_list, f1_list = [], [], []
        for t in thresholds:
            preds = (dists <= t) * mask
            tp = np.sum(preds * adj)
            fp = np.sum(preds * (1 - adj))
            fn = np.sum((1 - preds) * adj * mask)
            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)
            p_list.append(p)
            r_list.append(r)
            f1_list.append(2 * p * r / (p + r + 1e-7))
        best_idx = np.argmax(f1_list)
        return {
            "optimal_t": thresholds[best_idx],
            "thresholds": {
                "distance": thresholds,
                "precision": p_list,
                "recall": r_list,
                "f1": f1_list,
            },
        }

    def match(self, x_q, cutpoint_val, no_match_label=10):
        q_emb = self.model.predict(x_q, verbose=0)
        sims = ops.convert_to_numpy(ops.matmul(q_emb, ops.transpose(self.embeddings)))
        best_idx = np.argmax(sims, axis=1)
        best_dist = 1.0 - np.max(sims, axis=1)
        return np.where(
            best_dist <= cutpoint_val, self.labels[best_idx], no_match_label
        )


sim_model = SimilarityWrapper(model)

"""
## Similarity loss

The similarity loss expects batches containing at least 2 examples of each
class, from which it computes the loss over the pairwise positive and negative
distances. Here we are using `MultiSimilarityLoss()`
([paper](ihttps://arxiv.org/abs/1904.06627)), one of several losses in
Keras. This loss
attempts to use all informative pairs in the batch, taking into account the
self-similarity, positive-similarity, and the negative-similarity.
"""


@keras.saving.register_keras_serializable()
class MultiSimilarityLoss(keras.losses.Loss):
    def __init__(self, alpha=2.0, beta=40.0, base=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha, self.beta, self.base = alpha, beta, base

    def call(self, y_true, y_pred):
        sim_mat = ops.matmul(y_pred, ops.transpose(y_pred))
        y_true = ops.reshape(y_true, [-1, 1])
        adj = ops.cast(ops.equal(y_true, ops.transpose(y_true)), "float32")
        mask = 1.0 - ops.eye(ops.shape(y_true)[0])
        pos_m, neg_m = adj * mask, (1.0 - adj) * mask
        pos_l = (1.0 / self.alpha) * ops.log(
            1.0 + ops.sum(ops.exp(-self.alpha * (sim_mat - self.base)) * pos_m, axis=1)
        )
        neg_l = (1.0 / self.beta) * ops.log(
            1.0 + ops.sum(ops.exp(self.beta * (sim_mat - self.base)) * neg_m, axis=1)
        )
        return ops.mean(pos_l + neg_l)


epochs = 3
learning_rate = 0.002
val_steps = 50

# init similarity loss
loss = MultiSimilarityLoss()

# compiling and training
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
)
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

"""
## Indexing

Now that we have trained our model, we can create an index of examples. Here we
batch index the first 200 validation examples by passing the x and y to the index
along with storing the image in the data parameter. The `x_index` values are
embedded and then added to the index to make them searchable. The `y_index` and
data parameters are optional but allow the user to associate metadata with the
embedded example.
"""

x_index, y_index = val_ds.get_slice(begin=0, size=200)
sim_model.reset_index()
sim_model.index(x_index, y_index, data=x_index)

"""
## Calibration

Once the index is built, we can calibrate a distance threshold using a matching
strategy and a calibration metric.

Here we are searching for the optimal F1 score while using K=1 as our
classifier. All matches at or below the calibrated threshold distance will be
labeled as a Positive match between the query example and the label associated
with the match result, while all matches above the threshold distance will be
labeled as a Negative match.

Additionally, we pass in extra metrics to compute as well. All values in the
output are computed at the calibrated threshold.

Finally, `model.calibrate()` returns a `CalibrationResults` object containing:

* `"cutpoints"`: A Python dict mapping the cutpoint name to a dict containing the
`ClassificationMetric` values associated with a particular distance threshold,
e.g., `"optimal" : {"acc": 0.90, "f1": 0.92}`.
* `"thresholds"`: A Python dict mapping `ClassificationMetric` names to a list
containing the metric's value computed at each of the distance thresholds, e.g.,
`{"f1": [0.99, 0.80], "distance": [0.0, 1.0]}`.
"""

x_train, y_train = train_ds.get_slice(begin=0, size=1000)
calibration = sim_model.calibrate(
    x_train,
    y_train,
    calibration_metric="f1",
    matcher="match_nearest",
    extra_metrics=["precision", "recall", "binary_accuracy"],
)

"""
## Visualization

It may be difficult to get a sense of the model quality from the metrics alone.
A complementary approach is to manually inspect a set of query results to get a
feel for the match quality.

Here we take 10 validation examples and plot them with their 5 nearest
neighbors and the distances to the query example. Looking at the results, we see
that while they are imperfect they still represent meaningfully similar images,
and that the model is able to find similar images irrespective of their pose or
image illumination.

We can also see that the model is very confident with certain images, resulting
in very small distances between the query and the neighbors. Conversely, we see
more mistakes in the class labels as the distances become larger. This is one of
the reasons why calibration is critical for matching applications.
"""

num_neighbors = 5
labels = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
    "Unknown",
]
class_mapping = {c_id: c_lbl for c_id, c_lbl in zip(range(11), labels)}

x_display, y_display = val_ds.get_slice(begin=200, size=10)
# lookup nearest neighbors in the index
nns = sim_model.lookup(x_display, k=num_neighbors)

# display
for idx in np.argsort(y_display.flatten()):
    fig, axes = plt.subplots(1, num_neighbors + 1, figsize=(16, 2))
    axes[0].imshow(x_display[idx].astype("uint8"))
    axes[0].set_title(f"Q: {labels[int(y_display[idx])]}")
    axes[0].axis("off")
    for i, nn in enumerate(nns[idx]):
        axes[i + 1].imshow(nn["data"].astype("uint8"))
        axes[i + 1].set_title(f"{labels[int(nn['label'])]}\n{nn['distance']:.3f}")
        axes[i + 1].axis("off")
    plt.show()

"""
## Metrics

We can also plot the extra metrics contained in the `CalibrationResults` to get
a sense of the matching performance as the distance threshold increases.

The following plots show the Precision, Recall, and F1 Score. We can see that
the matching precision degrades as the distance increases, but that the
percentage of the queries that we accept as positive matches (recall) grows
faster up to the calibrated distance threshold.
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
c_res = calibration["thresholds"]
ax1.plot(c_res["distance"], c_res["precision"], label="precision")
ax1.plot(c_res["distance"], c_res["recall"], label="recall")
ax1.plot(c_res["distance"], c_res["f1"], label="f1 score")
ax1.legend()
ax1.set_title("Metric evolution as distance increase")
ax1.set_ylim((-0.05, 1.05))
ax2.plot(c_res["recall"], c_res["precision"])
ax2.set_title("Precision recall curve")
ax2.set_ylim((-0.05, 1.05))
plt.show()

"""
We can also take 100 examples for each class and plot the confusion matrix for
each example and their nearest match. We also add an "extra" 10th class to
represent the matches above the calibrated distance threshold.

We can see that most of the errors are between the animal classes with an
interesting number of confusions between Airplane and Bird. Additionally, we see
that only a few of the 100 examples for each class returned matches outside of
the calibrated distance threshold.
"""

cutpoint = calibration["optimal_t"]

# This yields 100 examples for each class.
x_confusion, y_confusion = val_ds.get_slice(0, -1)

matches = sim_model.match(x_confusion, cutpoint_val=cutpoint, no_match_label=10)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_confusion, matches, labels=list(range(11)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()

"""
## No Match

We can plot the examples outside of the calibrated threshold to see which images
are not matching any indexed examples.

This may provide insight into what other examples may need to be indexed or
surface anomalous examples within the class.
"""

idx_no_match = np.where(np.array(matches) == 10)[0]
if len(idx_no_match):
    plt.imshow(x_confusion[idx_no_match[0]].astype("uint8"))
else:
    print("All queries have a match below the distance threshold.")

"""
## Visualize clusters

One of the best ways to quickly get a sense of the quality of how the model is
doing and understand it's short comings is to project the embedding into a 2D
space.

This allows us to inspect clusters of images and understand which classes are
entangled.
"""

# Each class in val_ds was restricted to 100 examples.
num_examples_to_clusters = 1000
vx, vy = val_ds.get_slice(0, num_examples_to_clusters)

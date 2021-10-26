# Point cloud segmentation with PointNet

**Author:** [Soumik Rakshit](https://github.com/soumik12345), [Sayak Paul](https://github.com/sayakpaul)<br>
**Date created:** 2020/10/23<br>
**Last modified:** 2020/10/24<br>
**Description:** Implementation of a PointNet-based model for segmenting point clouds.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/pointnet_segmentation.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/pointnet_segmentation.py)



---
## Introduction

A "point cloud" is an important type of data structure for storing geometric shape data.
Due to its irregular format, it's often transformed into
regular 3D voxel grids or collections of images before being used in deep learning applications,
a step which makes the data unnecessarily large.
The PointNet family of models solves this problem by directly consuming point clouds, respecting
the permutation-invariance property of the point data. The PointNet family of
models provides a simple, unified architecture
for applications ranging from **object classification**, **part segmentation**, to
**scene semantic parsing**.

In this example, we demonstrate the implementation of the PointNet architecture
for shape segmentation.

### References

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [Point cloud classification with PointNet](https://keras.io/examples/vision/pointnet/)
- [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)

---
## Imports


```python
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
```

---
## Downloading Dataset

The [ShapeNet dataset](https://shapenet.org/) is an ongoing effort to establish a richly-annotated,
large-scale dataset of 3D shapes. **ShapeNetCore** is a subset of the full ShapeNet
dataset with clean single 3D models and manually verified category and alignment
annotations. It covers 55 common object categories, with about 51,300 unique 3D models.

For this example, we use one of the 12 object categories of
[PASCAL 3D+](http://cvgl.stanford.edu/projects/pascal3d.html),
included as part of the ShapenetCore dataset.


```python
dataset_url = "https://git.io/JiY4i"

dataset_path = keras.utils.get_file(
    fname="shapenet.zip",
    origin=dataset_url,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=True,
    archive_format="auto",
    cache_dir="datasets",
)
```

---
## Loading the dataset

We parse the dataset metadata in order to easily map model categories to their
respective directories and segmentation classes to colors for the purpose of
visualization.


```python
with open("/tmp/.keras/datasets/PartAnnotation/metadata.json") as json_file:
    metadata = json.load(json_file)

print(metadata)
```

<div class="k-default-codeblock">
```
{'Airplane': {'directory': '02691156', 'lables': ['wing', 'body', 'tail', 'engine'], 'colors': ['blue', 'green', 'red', 'pink']}, 'Bag': {'directory': '02773838', 'lables': ['handle', 'body'], 'colors': ['blue', 'green']}, 'Cap': {'directory': '02954340', 'lables': ['panels', 'peak'], 'colors': ['blue', 'green']}, 'Car': {'directory': '02958343', 'lables': ['wheel', 'hood', 'roof'], 'colors': ['blue', 'green', 'red']}, 'Chair': {'directory': '03001627', 'lables': ['leg', 'arm', 'back', 'seat'], 'colors': ['blue', 'green', 'red', 'pink']}, 'Earphone': {'directory': '03261776', 'lables': ['earphone', 'headband'], 'colors': ['blue', 'green']}, 'Guitar': {'directory': '03467517', 'lables': ['head', 'body', 'neck'], 'colors': ['blue', 'green', 'red']}, 'Knife': {'directory': '03624134', 'lables': ['handle', 'blade'], 'colors': ['blue', 'green']}, 'Lamp': {'directory': '03636649', 'lables': ['canopy', 'lampshade', 'base'], 'colors': ['blue', 'green', 'red']}, 'Laptop': {'directory': '03642806', 'lables': ['keyboard'], 'colors': ['blue']}, 'Motorbike': {'directory': '03790512', 'lables': ['wheel', 'handle', 'gas_tank', 'light', 'seat'], 'colors': ['blue', 'green', 'red', 'pink', 'yellow']}, 'Mug': {'directory': '03797390', 'lables': ['handle'], 'colors': ['blue']}, 'Pistol': {'directory': '03948459', 'lables': ['trigger_and_guard', 'handle', 'barrel'], 'colors': ['blue', 'green', 'red']}, 'Rocket': {'directory': '04099429', 'lables': ['nose', 'body', 'fin'], 'colors': ['blue', 'green', 'red']}, 'Skateboard': {'directory': '04225987', 'lables': ['wheel', 'deck'], 'colors': ['blue', 'green']}, 'Table': {'directory': '04379243', 'lables': ['leg', 'top'], 'colors': ['blue', 'green']}}

```
</div>
In this example, we train PointNet to segment the parts of an `Airplane` model.


```python
points_dir = "/tmp/.keras/datasets/PartAnnotation/{}/points".format(
    metadata["Airplane"]["directory"]
)
labels_dir = "/tmp/.keras/datasets/PartAnnotation/{}/points_label".format(
    metadata["Airplane"]["directory"]
)
LABELS = metadata["Airplane"]["lables"]
COLORS = metadata["Airplane"]["colors"]

VAL_SPLIT = 0.2
NUM_SAMPLE_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 60
INITIAL_LR = 1e-3
```

---
## Structuring the dataset

We generate the following in-memory data structures from the Airplane point clouds and
their labels:

- `point_clouds` is a list of `np.array` objects that represent the point cloud data in
the form of x, y and z coordinates. Axis 0 represents the number of points in the
point cloud, while axis 1 represents the coordinates. `all_labels` is the list
that represents the label of each coordinate as a string (needed mainly for
visualization purposes).
- `test_point_clouds` is in the same format as `point_clouds`, but doesn't have
corresponding the labels of the point clouds.
- `all_labels` is a list of `np.array` objects that represent the point cloud labels
for each coordinate, corresponding to the `point_clouds` list.
- `point_cloud_labels` is a list of `np.array` objects that represent the point cloud
labels for each coordinate in one-hot encoded form, corresponding to the `point_clouds`
list.


```python
point_clouds, test_point_clouds = [], []
point_cloud_labels, all_labels = [], []

points_files = glob(os.path.join(points_dir, "*.pts"))
for point_file in tqdm(points_files):
    point_cloud = np.loadtxt(point_file)
    if point_cloud.shape[0] < NUM_SAMPLE_POINTS:
        continue

    # Get the file-id of the current point cloud for parsing its
    # labels.
    file_id = point_file.split("/")[-1].split(".")[0]
    label_data, num_labels = {}, 0
    for label in LABELS:
        label_file = os.path.join(labels_dir, label, file_id + ".seg")
        if os.path.exists(label_file):
            label_data[label] = np.loadtxt(label_file).astype("float32")
            num_labels = len(label_data[label])

    # Point clouds having labels will be our training samples.
    try:
        label_map = ["none"] * num_labels
        for label in LABELS:
            for i, data in enumerate(label_data[label]):
                label_map[i] = label if data == 1 else label_map[i]
        label_data = [
            LABELS.index(label) if label != "none" else len(LABELS)
            for label in label_map
        ]
        # Apply one-hot encoding to the dense label representation.
        label_data = keras.utils.to_categorical(label_data, num_classes=len(LABELS) + 1)

        point_clouds.append(point_cloud)
        point_cloud_labels.append(label_data)
        all_labels.append(label_map)
    except KeyError:
        test_point_clouds.append(point_cloud)
```

<div class="k-default-codeblock">
```
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4045/4045 [03:35<00:00, 18.76it/s]

```
</div>
Next, we take a look at some samples from the in-memory arrays we just generated:


```python
for _ in range(5):
    i = random.randint(0, len(point_clouds) - 1)
    print(f"point_clouds[{i}].shape:", point_clouds[0].shape)
    print(f"point_cloud_labels[{i}].shape:", point_cloud_labels[0].shape)
    for j in range(5):
        print(
            f"all_labels[{i}][{j}]:",
            all_labels[i][j],
            f"\tpoint_cloud_labels[{i}][{j}]:",
            point_cloud_labels[i][j],
            "\n",
        )
```

<div class="k-default-codeblock">
```
point_clouds[475].shape: (2602, 3)
point_cloud_labels[475].shape: (2602, 5)
all_labels[475][0]: body 	point_cloud_labels[475][0]: [0. 1. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[475][1]: engine 	point_cloud_labels[475][1]: [0. 0. 0. 1. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[475][2]: body 	point_cloud_labels[475][2]: [0. 1. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[475][3]: body 	point_cloud_labels[475][3]: [0. 1. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[475][4]: wing 	point_cloud_labels[475][4]: [1. 0. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
point_clouds[2712].shape: (2602, 3)
point_cloud_labels[2712].shape: (2602, 5)
all_labels[2712][0]: tail 	point_cloud_labels[2712][0]: [0. 0. 1. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[2712][1]: wing 	point_cloud_labels[2712][1]: [1. 0. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[2712][2]: engine 	point_cloud_labels[2712][2]: [0. 0. 0. 1. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[2712][3]: wing 	point_cloud_labels[2712][3]: [1. 0. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[2712][4]: wing 	point_cloud_labels[2712][4]: [1. 0. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
point_clouds[1413].shape: (2602, 3)
point_cloud_labels[1413].shape: (2602, 5)
all_labels[1413][0]: body 	point_cloud_labels[1413][0]: [0. 1. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[1413][1]: tail 	point_cloud_labels[1413][1]: [0. 0. 1. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[1413][2]: tail 	point_cloud_labels[1413][2]: [0. 0. 1. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[1413][3]: tail 	point_cloud_labels[1413][3]: [0. 0. 1. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[1413][4]: tail 	point_cloud_labels[1413][4]: [0. 0. 1. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
point_clouds[1207].shape: (2602, 3)
point_cloud_labels[1207].shape: (2602, 5)
all_labels[1207][0]: tail 	point_cloud_labels[1207][0]: [0. 0. 1. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[1207][1]: wing 	point_cloud_labels[1207][1]: [1. 0. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[1207][2]: wing 	point_cloud_labels[1207][2]: [1. 0. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[1207][3]: body 	point_cloud_labels[1207][3]: [0. 1. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[1207][4]: body 	point_cloud_labels[1207][4]: [0. 1. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
point_clouds[2492].shape: (2602, 3)
point_cloud_labels[2492].shape: (2602, 5)
all_labels[2492][0]: engine 	point_cloud_labels[2492][0]: [0. 0. 0. 1. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[2492][1]: body 	point_cloud_labels[2492][1]: [0. 1. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[2492][2]: body 	point_cloud_labels[2492][2]: [0. 1. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[2492][3]: body 	point_cloud_labels[2492][3]: [0. 1. 0. 0. 0.] 
```
</div>
    
<div class="k-default-codeblock">
```
all_labels[2492][4]: engine 	point_cloud_labels[2492][4]: [0. 0. 0. 1. 0.] 
```
</div>
    


Now, let's visualize some of the point clouds along with their labels.


```python

def visualize_data(point_cloud, labels):
    df = pd.DataFrame(
        data={
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2],
            "label": labels,
        }
    )
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")
    for index, label in enumerate(LABELS):
        c_df = df[df["label"] == label]
        try:
            ax.scatter(
                c_df["x"], c_df["y"], c_df["z"], label=label, alpha=0.5, c=COLORS[index]
            )
        except IndexError:
            pass
    ax.legend()
    plt.show()


visualize_data(point_clouds[0], all_labels[0])
visualize_data(point_clouds[300], all_labels[300])
```


    
![png](/img/examples/vision/pointnet_segmentation/pointnet_segmentation_15_0.png)
    



    
![png](/img/examples/vision/pointnet_segmentation/pointnet_segmentation_15_1.png)
    


### Preprocessing

Note that all the point clouds that we have loaded consist of a variable number of points,
which makes it difficult for us to batch them together. In order to overcome this problem, we
randomly sample a fixed number of points from each point cloud. We also normalize the
point clouds in order to make the data scale-invariant.


```python
for index in tqdm(range(len(point_clouds))):
    current_point_cloud = point_clouds[index]
    current_label_cloud = point_cloud_labels[index]
    current_labels = all_labels[index]
    num_points = len(current_point_cloud)
    # Randomly sampling respective indices.
    sampled_indices = random.sample(list(range(num_points)), NUM_SAMPLE_POINTS)
    # Sampling points corresponding to sampled indices.
    sampled_point_cloud = np.array([current_point_cloud[i] for i in sampled_indices])
    # Sampling corresponding one-hot encoded labels.
    sampled_label_cloud = np.array([current_label_cloud[i] for i in sampled_indices])
    # Sampling corresponding labels for visualization.
    sampled_labels = np.array([current_labels[i] for i in sampled_indices])
    # Normalizing sampled point cloud.
    norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud, axis=0)
    norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
    point_clouds[index] = norm_point_cloud
    point_cloud_labels[index] = sampled_label_cloud
    all_labels[index] = sampled_labels
```

<div class="k-default-codeblock">
```
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3694/3694 [00:07<00:00, 478.67it/s]

```
</div>
Let's visualize the sampled and normalized point clouds along with their corresponding
labels.


```python
visualize_data(point_clouds[0], all_labels[0])
visualize_data(point_clouds[300], all_labels[300])
```


    
![png](/img/examples/vision/pointnet_segmentation/pointnet_segmentation_19_0.png)
    



    
![png](/img/examples/vision/pointnet_segmentation/pointnet_segmentation_19_1.png)
    


### Creating TensorFlow datasets

We create `tf.data.Dataset` objects for the training and validation data.
We also augment the training point clouds by applying random jitter to them.


```python

def load_data(point_cloud_batch, label_cloud_batch):
    point_cloud_batch.set_shape([NUM_SAMPLE_POINTS, 3])
    label_cloud_batch.set_shape([NUM_SAMPLE_POINTS, len(LABELS) + 1])
    return point_cloud_batch, label_cloud_batch


def augment(point_cloud_batch, label_cloud_batch):
    noise = tf.random.uniform(
        tf.shape(label_cloud_batch), -0.005, 0.005, dtype=tf.float64
    )
    point_cloud_batch += noise[:, :, :3]
    return point_cloud_batch, label_cloud_batch


def generate_dataset(point_clouds, label_clouds, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((point_clouds, label_clouds))
    dataset = dataset.shuffle(BATCH_SIZE * 100) if is_training else dataset
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = (
        dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        if is_training
        else dataset
    )
    return dataset


split_index = int(len(point_clouds) * (1 - VAL_SPLIT))
train_point_clouds = point_clouds[:split_index]
train_label_cloud = point_cloud_labels[:split_index]
total_training_examples = len(train_point_clouds)

val_point_clouds = point_clouds[split_index:]
val_label_cloud = point_cloud_labels[split_index:]

print("Num train point clouds:", len(train_point_clouds))
print("Num train point cloud labels:", len(train_label_cloud))
print("Num val point clouds:", len(val_point_clouds))
print("Num val point cloud labels:", len(val_label_cloud))

train_dataset = generate_dataset(train_point_clouds, train_label_cloud)
val_dataset = generate_dataset(val_point_clouds, val_label_cloud, is_training=False)

print("Train Dataset:", train_dataset)
print("Validation Dataset:", val_dataset)
```

<div class="k-default-codeblock">
```
Num train point clouds: 2955
Num train point cloud labels: 2955
Num val point clouds: 739
Num val point cloud labels: 739

Train Dataset: <ParallelMapDataset shapes: ((None, 1024, 3), (None, 1024, 5)), types: (tf.float64, tf.float32)>
Validation Dataset: <BatchDataset shapes: ((None, 1024, 3), (None, 1024, 5)), types: (tf.float64, tf.float32)>

```
</div>
---
## PointNet model

The figure below depicts the internals of the PointNet model family:

![](https://i.imgur.com/qFLNw5L.png)

Given that PointNet is meant to consume an ***unordered set*** of coordinates as its input data,
its architecture needs to match the following characteristic properties
of point cloud data:

### Permutation invariance

Given the unstructured nature of point cloud data, a scan made up of `n` points has `n!`
permutations. The subsequent data processing must be invariant to the different
representations. In order to make PointNet invariant to input permutations, we use a
symmetric function (such as max-pooling) once the `n` input points are mapped to
higher-dimensional space. The result is a **global feature vector** that aims to capture
an aggregate signature of the `n` input points. The global feature vector is used alongside
local point features for segmentation.

![](https://i.imgur.com/0mrvvjb.png)

### Transformation invariance

Segmentation outputs should be unchanged if the object undergoes certain transformations,
such as translation or scaling. For a given input point cloud, we apply an appropriate
rigid or affine transformation to achieve pose normalization. Because each of the `n` input
points are represented as a vector and are mapped to the embedding spaces independently,
applying a geometric transformation simply amounts to matrix multiplying each point with
a transformation matrix. This is motivated by the concept of
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).

The operations comprising the T-Net are motivated by the higher-level architecture of
PointNet. MLPs (or fully-connected layers) are used to map the input points independently
and identically to a higher-dimensional space; max-pooling is used to encode a global
feature vector whose dimensionality is then reduced with fully-connected layers. The
input-dependent features at the final fully-connected layer are then combined with
globally trainable weights and biases, resulting in a 3-by-3 transformation matrix.

![](https://i.imgur.com/aEj3GYi.png)

### Point interactions

The interaction between neighboring points often carries useful information (i.e., a
single point should not be treated in isolation). Whereas classification need only make
use of global features, segmentation must be able to leverage local point features along
with global point features.


**Note**: The figures presented in this section have been taken from the
[original paper](https://arxiv.org/abs/1612.00593).

Now that we know the pieces that compose the PointNet model, we can implement the model.
We start by implementing the basic blocks i.e., the convolutional block and the multi-layer
perceptron block.


```python

def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

```

We implement a regularizer (taken from
[this example](https://keras.io/examples/vision/pointnet/#build-a-model))
to enforce orthogonality in the feature space. This is needed to ensure
that the magnitudes of the transformed features do not vary too much.


```python

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """Reference: https://keras.io/examples/vision/pointnet/#build-a-model"""

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.identity))

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config

```

The next piece is the transformation network which we explained earlier.


```python

def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=64, name=f"{name}_1")
    x = conv_block(x, filters=128, name=f"{name}_2")
    x = conv_block(x, filters=1024, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)


def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])

```

Finally, we piece the above blocks together and implement the segmentation model.


```python

def get_shape_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(None, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)

```

---
## Instantiate the model


```python
x, y = next(iter(train_dataset))

num_points = x.shape[1]
num_classes = y.shape[-1]

segmentation_model = get_shape_segmentation_model(num_points, num_classes)
segmentation_model.summary()
```

<div class="k-default-codeblock">
```
2021-10-25 01:26:33.563133: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)

Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, None, 3)]    0                                            
__________________________________________________________________________________________________
input_transformation_block_1_co (None, None, 64)     256         input_1[0][0]                    
__________________________________________________________________________________________________
input_transformation_block_1_ba (None, None, 64)     256         input_transformation_block_1_conv
__________________________________________________________________________________________________
input_transformation_block_1_re (None, None, 64)     0           input_transformation_block_1_batc
__________________________________________________________________________________________________
input_transformation_block_2_co (None, None, 128)    8320        input_transformation_block_1_relu
__________________________________________________________________________________________________
input_transformation_block_2_ba (None, None, 128)    512         input_transformation_block_2_conv
__________________________________________________________________________________________________
input_transformation_block_2_re (None, None, 128)    0           input_transformation_block_2_batc
__________________________________________________________________________________________________
input_transformation_block_3_co (None, None, 1024)   132096      input_transformation_block_2_relu
__________________________________________________________________________________________________
input_transformation_block_3_ba (None, None, 1024)   4096        input_transformation_block_3_conv
__________________________________________________________________________________________________
input_transformation_block_3_re (None, None, 1024)   0           input_transformation_block_3_batc
__________________________________________________________________________________________________
global_max_pooling1d (GlobalMax (None, 1024)         0           input_transformation_block_3_relu
__________________________________________________________________________________________________
input_transformation_block_1_1_ (None, 512)          524800      global_max_pooling1d[0][0]       
__________________________________________________________________________________________________
input_transformation_block_1_1_ (None, 512)          2048        input_transformation_block_1_1_de
__________________________________________________________________________________________________
input_transformation_block_1_1_ (None, 512)          0           input_transformation_block_1_1_ba
__________________________________________________________________________________________________
input_transformation_block_2_1_ (None, 256)          131328      input_transformation_block_1_1_re
__________________________________________________________________________________________________
input_transformation_block_2_1_ (None, 256)          1024        input_transformation_block_2_1_de
__________________________________________________________________________________________________
input_transformation_block_2_1_ (None, 256)          0           input_transformation_block_2_1_ba
__________________________________________________________________________________________________
input_transformation_block_fina (None, 9)            2313        input_transformation_block_2_1_re
__________________________________________________________________________________________________
reshape (Reshape)               (None, 3, 3)         0           input_transformation_block_final[
__________________________________________________________________________________________________
input_transformation_block_mm ( (None, None, 3)      0           input_1[0][0]                    
                                                                 reshape[0][0]                    
__________________________________________________________________________________________________
features_64_conv (Conv1D)       (None, None, 64)     256         input_transformation_block_mm[0][
__________________________________________________________________________________________________
features_64_batch_norm (BatchNo (None, None, 64)     256         features_64_conv[0][0]           
__________________________________________________________________________________________________
features_64_relu (Activation)   (None, None, 64)     0           features_64_batch_norm[0][0]     
__________________________________________________________________________________________________
features_128_1_conv (Conv1D)    (None, None, 128)    8320        features_64_relu[0][0]           
__________________________________________________________________________________________________
features_128_1_batch_norm (Batc (None, None, 128)    512         features_128_1_conv[0][0]        
__________________________________________________________________________________________________
features_128_1_relu (Activation (None, None, 128)    0           features_128_1_batch_norm[0][0]  
__________________________________________________________________________________________________
features_128_2_conv (Conv1D)    (None, None, 128)    16512       features_128_1_relu[0][0]        
__________________________________________________________________________________________________
features_128_2_batch_norm (Batc (None, None, 128)    512         features_128_2_conv[0][0]        
__________________________________________________________________________________________________
features_128_2_relu (Activation (None, None, 128)    0           features_128_2_batch_norm[0][0]  
__________________________________________________________________________________________________
transformed_features_1_conv (Co (None, None, 64)     8256        features_128_2_relu[0][0]        
__________________________________________________________________________________________________
transformed_features_1_batch_no (None, None, 64)     256         transformed_features_1_conv[0][0]
__________________________________________________________________________________________________
transformed_features_1_relu (Ac (None, None, 64)     0           transformed_features_1_batch_norm
__________________________________________________________________________________________________
transformed_features_2_conv (Co (None, None, 128)    8320        transformed_features_1_relu[0][0]
__________________________________________________________________________________________________
transformed_features_2_batch_no (None, None, 128)    512         transformed_features_2_conv[0][0]
__________________________________________________________________________________________________
transformed_features_2_relu (Ac (None, None, 128)    0           transformed_features_2_batch_norm
__________________________________________________________________________________________________
transformed_features_3_conv (Co (None, None, 1024)   132096      transformed_features_2_relu[0][0]
__________________________________________________________________________________________________
transformed_features_3_batch_no (None, None, 1024)   4096        transformed_features_3_conv[0][0]
__________________________________________________________________________________________________
transformed_features_3_relu (Ac (None, None, 1024)   0           transformed_features_3_batch_norm
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 1024)         0           transformed_features_3_relu[0][0]
__________________________________________________________________________________________________
transformed_features_1_1_dense  (None, 512)          524800      global_max_pooling1d_1[0][0]     
__________________________________________________________________________________________________
transformed_features_1_1_batch_ (None, 512)          2048        transformed_features_1_1_dense[0]
__________________________________________________________________________________________________
transformed_features_1_1_relu ( (None, 512)          0           transformed_features_1_1_batch_no
__________________________________________________________________________________________________
transformed_features_2_1_dense  (None, 256)          131328      transformed_features_1_1_relu[0][
__________________________________________________________________________________________________
transformed_features_2_1_batch_ (None, 256)          1024        transformed_features_2_1_dense[0]
__________________________________________________________________________________________________
transformed_features_2_1_relu ( (None, 256)          0           transformed_features_2_1_batch_no
__________________________________________________________________________________________________
transformed_features_final (Den (None, 16384)        4210688     transformed_features_2_1_relu[0][
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 128, 128)     0           transformed_features_final[0][0] 
__________________________________________________________________________________________________
transformed_features_mm (Dot)   (None, None, 128)    0           features_128_2_relu[0][0]        
                                                                 reshape_1[0][0]                  
__________________________________________________________________________________________________
features_512_conv (Conv1D)      (None, None, 512)    66048       transformed_features_mm[0][0]    
__________________________________________________________________________________________________
features_512_batch_norm (BatchN (None, None, 512)    2048        features_512_conv[0][0]          
__________________________________________________________________________________________________
features_512_relu (Activation)  (None, None, 512)    0           features_512_batch_norm[0][0]    
__________________________________________________________________________________________________
pre_maxpool_block_conv (Conv1D) (None, None, 2048)   1050624     features_512_relu[0][0]          
__________________________________________________________________________________________________
pre_maxpool_block_batch_norm (B (None, None, 2048)   8192        pre_maxpool_block_conv[0][0]     
__________________________________________________________________________________________________
pre_maxpool_block_relu (Activat (None, None, 2048)   0           pre_maxpool_block_batch_norm[0][0
__________________________________________________________________________________________________
global_features (MaxPooling1D)  (None, None, 2048)   0           pre_maxpool_block_relu[0][0]     
__________________________________________________________________________________________________
tf.tile (TFOpLambda)            (None, None, 2048)   0           global_features[0][0]            
__________________________________________________________________________________________________
segmentation_input (Concatenate (None, None, 3008)   0           features_64_relu[0][0]           
                                                                 features_128_1_relu[0][0]        
                                                                 features_128_2_relu[0][0]        
                                                                 transformed_features_mm[0][0]    
                                                                 features_512_relu[0][0]          
                                                                 tf.tile[0][0]                    
__________________________________________________________________________________________________
segmentation_features_conv (Con (None, None, 128)    385152      segmentation_input[0][0]         
__________________________________________________________________________________________________
segmentation_features_batch_nor (None, None, 128)    512         segmentation_features_conv[0][0] 
__________________________________________________________________________________________________
segmentation_features_relu (Act (None, None, 128)    0           segmentation_features_batch_norm[
__________________________________________________________________________________________________
segmentation_head (Conv1D)      (None, None, 5)      645         segmentation_features_relu[0][0] 
==================================================================================================
Total params: 7,370,062
Trainable params: 7,356,110
Non-trainable params: 13,952
__________________________________________________________________________________________________

```
</div>
---
## Training

For the training the authors recommend using a learning rate schedule that decays the
initial learning rate by half every 20 epochs. In this example, we resort to 15 epochs.


```python
training_step_size = total_training_examples // BATCH_SIZE
total_training_steps = training_step_size * EPOCHS
print(f"Total training steps: {total_training_steps}.")

lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[training_step_size * 15, training_step_size * 15],
    values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25],
)

steps = tf.range(total_training_steps, dtype=tf.int32)
lrs = [lr_schedule(step) for step in steps]

plt.plot(lrs)
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.show()
```

<div class="k-default-codeblock">
```
Total training steps: 5520.

```
</div>
    
![png](/img/examples/vision/pointnet_segmentation/pointnet_segmentation_34_1.png)
    


Finally, we implement a utility for running our experiments and launch model training.


```python

def run_experiment(epochs):

    segmentation_model = get_shape_segmentation_model(num_points, num_classes)
    segmentation_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = segmentation_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    segmentation_model.load_weights(checkpoint_filepath)
    return segmentation_model, history


segmentation_model, history = run_experiment(epochs=EPOCHS)
```

<div class="k-default-codeblock">
```
Epoch 1/60
93/93 [==============================] - 28s 127ms/step - loss: 5.3556 - accuracy: 0.7448 - val_loss: 5.8386 - val_accuracy: 0.7471
Epoch 2/60
93/93 [==============================] - 11s 117ms/step - loss: 4.7077 - accuracy: 0.8181 - val_loss: 5.2614 - val_accuracy: 0.7793
Epoch 3/60
93/93 [==============================] - 11s 118ms/step - loss: 4.6566 - accuracy: 0.8301 - val_loss: 4.7907 - val_accuracy: 0.8269
Epoch 4/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6059 - accuracy: 0.8406 - val_loss: 4.6031 - val_accuracy: 0.8482
Epoch 5/60
93/93 [==============================] - 11s 118ms/step - loss: 4.5828 - accuracy: 0.8444 - val_loss: 4.7692 - val_accuracy: 0.8220
Epoch 6/60
93/93 [==============================] - 11s 118ms/step - loss: 4.6150 - accuracy: 0.8408 - val_loss: 5.4460 - val_accuracy: 0.8192
Epoch 7/60
93/93 [==============================] - 11s 117ms/step - loss: 67.5943 - accuracy: 0.7378 - val_loss: 1617.1846 - val_accuracy: 0.5191
Epoch 8/60
93/93 [==============================] - 11s 117ms/step - loss: 15.2910 - accuracy: 0.6651 - val_loss: 8.1014 - val_accuracy: 0.7046
Epoch 9/60
93/93 [==============================] - 11s 117ms/step - loss: 6.8878 - accuracy: 0.7368 - val_loss: 14.2311 - val_accuracy: 0.6949
Epoch 10/60
93/93 [==============================] - 11s 117ms/step - loss: 5.8362 - accuracy: 0.7549 - val_loss: 14.6942 - val_accuracy: 0.6350
Epoch 11/60
93/93 [==============================] - 11s 117ms/step - loss: 5.4777 - accuracy: 0.7648 - val_loss: 44.1037 - val_accuracy: 0.6422
Epoch 12/60
93/93 [==============================] - 11s 117ms/step - loss: 5.2688 - accuracy: 0.7712 - val_loss: 4.9977 - val_accuracy: 0.7692
Epoch 13/60
93/93 [==============================] - 11s 117ms/step - loss: 5.1041 - accuracy: 0.7837 - val_loss: 6.0642 - val_accuracy: 0.7577
Epoch 14/60
93/93 [==============================] - 11s 117ms/step - loss: 5.0011 - accuracy: 0.7862 - val_loss: 4.9313 - val_accuracy: 0.7840
Epoch 15/60
93/93 [==============================] - 11s 117ms/step - loss: 4.8910 - accuracy: 0.7953 - val_loss: 5.8368 - val_accuracy: 0.7725
Epoch 16/60
93/93 [==============================] - 11s 117ms/step - loss: 4.8698 - accuracy: 0.8074 - val_loss: 73.0260 - val_accuracy: 0.7251
Epoch 17/60
93/93 [==============================] - 11s 117ms/step - loss: 4.8299 - accuracy: 0.8109 - val_loss: 17.1503 - val_accuracy: 0.7415
Epoch 18/60
93/93 [==============================] - 11s 117ms/step - loss: 4.8147 - accuracy: 0.8111 - val_loss: 62.2765 - val_accuracy: 0.7344
Epoch 19/60
93/93 [==============================] - 11s 117ms/step - loss: 4.8316 - accuracy: 0.8141 - val_loss: 5.2200 - val_accuracy: 0.7890
Epoch 20/60
93/93 [==============================] - 11s 117ms/step - loss: 4.7853 - accuracy: 0.8142 - val_loss: 5.7062 - val_accuracy: 0.7719
Epoch 21/60
93/93 [==============================] - 11s 117ms/step - loss: 4.7753 - accuracy: 0.8157 - val_loss: 6.2089 - val_accuracy: 0.7839
Epoch 22/60
93/93 [==============================] - 11s 117ms/step - loss: 4.7681 - accuracy: 0.8161 - val_loss: 5.1077 - val_accuracy: 0.8021
Epoch 23/60
93/93 [==============================] - 11s 117ms/step - loss: 4.7554 - accuracy: 0.8187 - val_loss: 4.7912 - val_accuracy: 0.7912
Epoch 24/60
93/93 [==============================] - 11s 117ms/step - loss: 4.7355 - accuracy: 0.8197 - val_loss: 4.9164 - val_accuracy: 0.7978
Epoch 25/60
93/93 [==============================] - 11s 117ms/step - loss: 4.7483 - accuracy: 0.8197 - val_loss: 13.4724 - val_accuracy: 0.7631
Epoch 26/60
93/93 [==============================] - 11s 117ms/step - loss: 4.7200 - accuracy: 0.8218 - val_loss: 8.3074 - val_accuracy: 0.7596
Epoch 27/60
93/93 [==============================] - 11s 118ms/step - loss: 4.7192 - accuracy: 0.8231 - val_loss: 12.4468 - val_accuracy: 0.7591
Epoch 28/60
93/93 [==============================] - 11s 117ms/step - loss: 4.7151 - accuracy: 0.8241 - val_loss: 23.8681 - val_accuracy: 0.7689
Epoch 29/60
93/93 [==============================] - 11s 117ms/step - loss: 4.7096 - accuracy: 0.8237 - val_loss: 4.9069 - val_accuracy: 0.8104
Epoch 30/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6991 - accuracy: 0.8257 - val_loss: 4.9858 - val_accuracy: 0.7950
Epoch 31/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6852 - accuracy: 0.8260 - val_loss: 5.0130 - val_accuracy: 0.7678
Epoch 32/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6630 - accuracy: 0.8286 - val_loss: 4.8523 - val_accuracy: 0.7676
Epoch 33/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6837 - accuracy: 0.8281 - val_loss: 5.4347 - val_accuracy: 0.8095
Epoch 34/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6571 - accuracy: 0.8296 - val_loss: 10.4595 - val_accuracy: 0.7410
Epoch 35/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6460 - accuracy: 0.8321 - val_loss: 4.9189 - val_accuracy: 0.8083
Epoch 36/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6430 - accuracy: 0.8327 - val_loss: 5.8674 - val_accuracy: 0.7911
Epoch 37/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6530 - accuracy: 0.8309 - val_loss: 4.7946 - val_accuracy: 0.8032
Epoch 38/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6391 - accuracy: 0.8318 - val_loss: 5.0111 - val_accuracy: 0.8024
Epoch 39/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6521 - accuracy: 0.8336 - val_loss: 8.1558 - val_accuracy: 0.7727
Epoch 40/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6443 - accuracy: 0.8329 - val_loss: 42.8513 - val_accuracy: 0.7688
Epoch 41/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6316 - accuracy: 0.8342 - val_loss: 5.0960 - val_accuracy: 0.8066
Epoch 42/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6322 - accuracy: 0.8335 - val_loss: 5.0634 - val_accuracy: 0.8158
Epoch 43/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6175 - accuracy: 0.8370 - val_loss: 6.0642 - val_accuracy: 0.8062
Epoch 44/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6175 - accuracy: 0.8371 - val_loss: 11.1805 - val_accuracy: 0.7790
Epoch 45/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6056 - accuracy: 0.8377 - val_loss: 4.7359 - val_accuracy: 0.8145
Epoch 46/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6108 - accuracy: 0.8383 - val_loss: 5.7125 - val_accuracy: 0.7713
Epoch 47/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6103 - accuracy: 0.8377 - val_loss: 6.3271 - val_accuracy: 0.8105
Epoch 48/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6020 - accuracy: 0.8383 - val_loss: 14.2876 - val_accuracy: 0.7529
Epoch 49/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6035 - accuracy: 0.8382 - val_loss: 4.8244 - val_accuracy: 0.8143
Epoch 50/60
93/93 [==============================] - 11s 117ms/step - loss: 4.6076 - accuracy: 0.8381 - val_loss: 8.2636 - val_accuracy: 0.7528
Epoch 51/60
93/93 [==============================] - 11s 117ms/step - loss: 4.5927 - accuracy: 0.8399 - val_loss: 4.6473 - val_accuracy: 0.8266
Epoch 52/60
93/93 [==============================] - 11s 117ms/step - loss: 4.5927 - accuracy: 0.8408 - val_loss: 4.6443 - val_accuracy: 0.8276
Epoch 53/60
93/93 [==============================] - 11s 117ms/step - loss: 4.5852 - accuracy: 0.8413 - val_loss: 5.1300 - val_accuracy: 0.7768
Epoch 54/60
93/93 [==============================] - 11s 117ms/step - loss: 4.5787 - accuracy: 0.8426 - val_loss: 8.9590 - val_accuracy: 0.7582
Epoch 55/60
93/93 [==============================] - 11s 117ms/step - loss: 4.5837 - accuracy: 0.8410 - val_loss: 5.1501 - val_accuracy: 0.8117
Epoch 56/60
93/93 [==============================] - 11s 117ms/step - loss: 4.5875 - accuracy: 0.8422 - val_loss: 31.3518 - val_accuracy: 0.7590
Epoch 57/60
93/93 [==============================] - 11s 117ms/step - loss: 4.5821 - accuracy: 0.8427 - val_loss: 4.8853 - val_accuracy: 0.8144
Epoch 58/60
93/93 [==============================] - 11s 117ms/step - loss: 4.5751 - accuracy: 0.8446 - val_loss: 4.6653 - val_accuracy: 0.8222
Epoch 59/60
93/93 [==============================] - 11s 117ms/step - loss: 4.5752 - accuracy: 0.8447 - val_loss: 6.0078 - val_accuracy: 0.8014
Epoch 60/60
93/93 [==============================] - 11s 118ms/step - loss: 4.5695 - accuracy: 0.8452 - val_loss: 4.8178 - val_accuracy: 0.8192

```
</div>
---
## Visualize the training landscape


```python

def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("loss")
plot_result("accuracy")
```


    
![png](/img/examples/vision/pointnet_segmentation/pointnet_segmentation_38_0.png)
    



    
![png](/img/examples/vision/pointnet_segmentation/pointnet_segmentation_38_1.png)
    


---
## Inference


```python
validation_batch = next(iter(val_dataset))
val_predictions = segmentation_model.predict(validation_batch[0])
print(f"Validation prediction shape: {val_predictions.shape}")


def visualize_single_point_cloud(point_clouds, label_clouds, idx):
    label_map = LABELS + ["none"]
    point_cloud = point_clouds[idx]
    label_cloud = label_clouds[idx]
    visualize_data(point_cloud, [label_map[np.argmax(label)] for label in label_cloud])


idx = np.random.choice(len(validation_batch[0]))
print(f"Index selected: {idx}")

# Plotting with ground-truth.
visualize_single_point_cloud(validation_batch[0], validation_batch[1], idx)

# Plotting with predicted labels.
visualize_single_point_cloud(validation_batch[0], val_predictions, idx)
```

<div class="k-default-codeblock">
```
Validation prediction shape: (32, 1024, 5)
Index selected: 24

```
</div>
    
![png](/img/examples/vision/pointnet_segmentation/pointnet_segmentation_40_1.png)
    



    
![png](/img/examples/vision/pointnet_segmentation/pointnet_segmentation_40_2.png)
    


---
## Final notes

If you are interested in learning more about this topic, you may find
[this repository](https://github.com/soumik12345/point-cloud-segmentation)
useful.

"""
Title: Point cloud classification with PointNet
Author: [David Griffiths](https://dgriffiths3.github.io)
Date created: 2020/05/25
Last modified: 2020/05/25
Description: Implementation of PointNet for ModelNet10 classification.
"""

"""
## Introduction
"""

"""
Classification, detection and segmentation of unordered 3D point sets i.e. point clouds
is a core problem in computer vision. This example implements the seminal point cloud
deep learning paper [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593). For a
detailed intoduction on PointNet see [this blog
post](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a).
"""

"""
## Setup
"""

"""
If running in google colab first run `!pip install trimesh`.
"""

import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

"""
## Load dataset
"""

"""
We use the ModelNet10 model dataset, the smaller 10 class version of the ModelNet40
dataset. First download the data:
"""

"""shell
!curl -O http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
!unzip ModelNet10.zip
"""

"""
We can use the `trimesh` package to read and visualize the `.off` mesh files.
"""

mesh = trimesh.load("ModelNet10/chair/train/chair_0001.off")
mesh.show()

"""
To convert a mesh file to a point cloud we first need to sample points on the mesh
surface. `.sample()` performs a unifrom random sampling. Here we sample at 2048 locations
and visualize in `matplotlib`.
"""

points = mesh.sample(2048)
trimesh.points.plot_points(points)

"""
To generate a `tf.data.Dataset()` we need to first parse through the ModelNet data
folders. Each mesh is loaded and sampled into a point cloud before being added to a
standard python list and converted to a `numpy` array. We also store the current
enumerate index value as the object label.
"""


def parse_dataset(n_pts=2048):

    train_pts = []
    train_labels = []
    test_pts = []
    test_labels = []

    folders = glob.glob("ModelNet10/[!README]*")

    for i, folder in enumerate(folders):

        print("processing: %s" % folder)
        train_files = glob.glob(folder + "/train/*")
        test_files = glob.glob(folder + "/test/*")

        for f in train_files:
            train_pts.append(trimesh.load(f).sample(n_pts))
            train_labels.append(i)

        for f in test_files:
            test_pts.append(trimesh.load(f).sample(n_pts))
            test_labels.append(i)

    return (
        np.array(train_pts),
        np.array(test_pts),
        np.array(train_labels),
        np.array(test_labels),
    )


"""
Set the number of points to sample and batch size and parse the dataset. This can take
~5minutes to complete.
"""

N_PTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

X_train, X_test, y_train, y_test = parse_dataset(N_PTS)

"""
Our data can now be read into a `tf.data.Dataset()` object. We set the shuffle buffer
size to the entire size of the dataset as prior to this the data is ordered by class.
Data augmentation is important when working with point cloud data. We create a
augmentation function to jitter, shuffle and rotate the train dataset.
"""


def augment(pts, label):

    # jitter points
    pts += tf.random.uniform(pts.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    pts = tf.random.shuffle(pts)
    # random rotation about the z axis
    theta = np.random.uniform(-np.pi, np.pi)
    c = np.cos(theta)
    s = np.sin(theta)
    R = tf.constant([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], tf.float64)
    pts = tf.einsum("...ij,...kj->...ki", R, pts)

    return pts, label


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_dataset.shuffle(len(X_train)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(X_test)).batch(BATCH_SIZE)

"""
### Build a model
"""

"""
Each convolution and fully-connected layer (with exception for end layers) consits of
Convolution / Dense -> Batch Normalization -> ReLU Activation. We can make simple
sub-classed layer to perform all these as a single layer.
"""


class ConvBN(layers.Layer):
    def __init__(self, filters):
        super(ConvBN, self).__init__()

        self.conv = layers.Conv1D(filters, kernel_size=1, padding="valid")
        self.bn = layers.BatchNormalization(momentum=0.0)
        self.activation = tf.nn.relu

    def call(self, inputs):

        x = self.conv(inputs)
        x = self.bn(x)
        return self.activation(x)


class DenseBN(layers.Layer):
    def __init__(self, filters):
        super(DenseBN, self).__init__()

        self.dense = layers.Dense(filters)
        self.bn = layers.BatchNormalization(momentum=0.0)
        self.activation = tf.nn.relu

    def call(self, inputs):

        x = self.dense(inputs)
        x = self.bn(x)
        return self.activation(x)


"""
PointNet consists of two core components. The primary MLP network, and the transformer
net (t-net). The t-net aims to learn an affine transformation matrix by its own mini
network. The mini network strongly resembles the main network. The t-net is used twice.
The first time to transform the input features ($n \times 3$) into a canonical
representation. The second is an affine transformation for alignment in feature space ($n
\times 64$). As per the original paper we constrain the transformation to be close to an
orthogonal matrix (i.e. $||X*X^T - I||$ = 0).
"""


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, k, l2reg=0.001):
        self.k = k
        self.l2reg = l2reg
        self.eye = K.eye(self.k)

    def __call__(self, x):
        x = K.reshape(x, (-1, self.k, self.k))
        xxt = K.batch_dot(x, x, axes=(2, 2))
        return K.sum(self.l2reg * K.square(xxt - self.eye))


"""
 We can define the t-net as a functional model.
"""


def tnet(n_pts, k, name):

    inputs = keras.Input(shape=(n_pts, k))

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(k).flatten())
    reg = OrthogonalRegularizer(k)

    x = ConvBN(64)(inputs)
    x = ConvBN(128)(x)
    x = ConvBN(1024)(x)

    x = layers.GlobalMaxPooling1D()(x)

    x = DenseBN(512)(x)
    x = DenseBN(256)(x)
    x = layers.Dense(
        k * k,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)

    feat_T = layers.Reshape((k, k))(x)

    # Apply affine transformation to input features
    outputs = layers.Dot(axes=(2, 1))([inputs, feat_T])

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


"""
The main network can be then implemented in the same manner where the t-net mini models
can be dropped in a layers in the graph. Here we replicate the network architecture
published in the original paper.
"""

inputs = keras.Input(shape=(N_PTS, 3))

# x = tnet(N_PTS, 3, 'input_tnet')(inputs)

x = ConvBN(64)(inputs)
x = ConvBN(64)(x)

# x = tnet(N_PTS, 64, 'feat_tnet')(x)

x = ConvBN(64)(x)
x = ConvBN(128)(x)
x = ConvBN(1024)(x)

x = layers.GlobalMaxPooling1D()(x)

x = DenseBN(512)(x)
x = layers.Dropout(0.3)(x)

x = DenseBN(256)(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

model.summary()

"""
### Train model
"""

"""
Once the model is defined it can be trained like any other standard classification model
using `.compile()` and `.fit()`.
"""

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=10, validation_data=test_dataset)

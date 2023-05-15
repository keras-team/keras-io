"""
Title: 3D Object Detection with KerasCV
Author: Ian Stenbit, Zhaoqi Leng (Waymo), Guowang Li (Waymo)
Date created: 2023/04/27
Last modified: 2023/04/27
Description: Use KerasCV to train a 3D object detection model for LIDAR data.
Accelerator: GPU
"""

"""
KerasCV offers a set of APIs to train LIDAR-based 3D object detection models,
including dataloading, augmentation, model training, and metric evaluation.
problems. These APIs were designed and implemented in partnership with Waymo.

In this guide, we'll take KerasCV's 3D object detection API for a spin by training
a CenterPillar model for Waymo's Open Dataset, which is a 3D object detection
task for detecting cars and pedestrians for an autonomous vehicle.
"""

"""shell
!pip install --upgrade git+https://github.com/keras-team/keras-cv
!pip install tensorflow==2.11.0
!pip install waymo-open-dataset-tf-2.11.0==1.5.1
"""

import keras_cv
from keras_cv.callbacks import WaymoEvaluationCallback

from keras_cv.datasets.waymo import load
from keras_cv.datasets.waymo import transformer

from keras_cv.layers.object_detection_3d import voxel_utils

from keras_cv.layers import CenterNetLabelEncoder
from keras_cv.layers import DynamicVoxelization

from keras_cv.models.__internal__.unet import UNet

from keras_cv.models.object_detection_3d.center_pillar import MultiClassDetectionHead
from keras_cv.models.object_detection_3d.center_pillar import MultiClassHeatmapDecoder
from keras_cv.models.object_detection_3d.center_pillar import MultiHeadCenterPillar

import tensorflow as tf
from tensorflow import keras

"""
3D object detection is the process of identifying, classifying,
and localizing objects within a 3D space.  Inputs are often in the form of point
clouds, although 2D images are sometimes used as inputs as well for multi-model
3D object detection. KerasCV currently supports unimodal point cloud inputs for
3D object detection.

Point cloud inputs to 3D object detection models typically come from LIDAR
sensors, and are generally structured as large batches of unsorted points
captured at a single point in time.

In KerasCV, we adopt a data format where point clouds are represented as a
dictionary with the following structure:

```python
point_cloud = {
  "point_xyz": FloatTensor[batch_size, 3]
  "point_features": FloatTensor[batch_size, num_features]
  "point_mask": BooleanTensor[batch_size]
}
```

The `point_xyz` field represents the XYZ coordinates of each point in the
point cloud.

The `point_features` field represents the LIDAR features of each point in the
poin cloud. Typical features include range, intensity, and elongation.

In KerasCV, 3D box targets for object detection are represented as vertical
pillars rotated with respect to the Z axis. We encode each box as a list (or
Tensor) of 7 floats: the X, Y, and Z coordinates of the box's center, the width,
length, and height of the box, and the rotation of the box with respect to the
Z axis. (This rotation is referrred to as `phi` and is always in radians).

KerasCV's first 3D object detection model offering is a center-based model like
the one proposed in https://arxiv.org/pdf/2006.11275.pdf.

Let's get to 3D modelling!
We'll start by loading up the Waymo Open Dataset. KerasCV provides a
`waymo_open_dataset.load` function to load the Waymo Open Dataset into our
data format.
"""

from keras_cv.datasets.waymo import load
from keras_cv.datasets.waymo import build_tensors_for_augmentation

# TODO(ianstenbit): Because we can't distribute WOD, we'll need to load wod_one_frame.tfrecord here
data_shard = './training-data'
dataset = load(data_shard)

# By default, WOD point clouds are globally positioned, but for object detection
# we want them with respect to the vehicle, so we transform them to the vehicle
# frame of reference.
dataset = dataset.map(transformer.transform_to_vehicle_frame)

# Because number of points is dynamic, we pad them to make our inputs batchable.
dataset = dataset.map(transformer.pad_or_trim_tensors)

# Then we can easily reformat the tensors into KerasCV's data format!
dataset = dataset.map(build_tensors_for_augmentation)

# We use a small batch size here on CPU. Generally, point clouds can be pretty
# large, so batch sizes are often smaller than in the 2D object detection world.
dataset = dataset.batch(1)

"""
Loading up the Waymo Open Dataset can be a bit tricky, but this makes it pretty
simple!

One important note: Waymo Open Dataset is distributed as TFRecords representing
a Waymo Open Dataset `Frame` proto. This cannot be deserialized in to Tensors
inside of the TensorFlow graph, so this can cause CPU throttling during
training.

Therefore, KerasCV offers a utility for transforming Waymo Open Dataset frames
into tf.Example records which can be more efficiently loaded into a TF graph
for later training. The utility can be found at
https://github.com/keras-team/keras-cv/blob/master/examples/training/object_detection_3d/waymo/serialize_records.py

Next up, let's augment our data! In partnership with Waymo, KerasCV offers a
set of state-of-the-art 3D augmentations for LIDAR data and 3D boxes. They
behave like all Keras preprocessing layers, and they're very easy to set up.
"""

augmentations = keras.Sequential([
  keras_cv.layers.GlobalRandomFlip(),
  keras_cv.layers.GlobalRandomRotation(max_rotation_angle_z=3.14),
])

dataset = dataset.map(augmentations)

"""
In just a few lines of code, we've augmented our input data using a few of the
3D augmentations offered in KerasCV. Check out the whole set of augmentations
at https://github.com/keras-team/keras-cv/tree/master/keras_cv/layers/preprocessing_3d.

Next, we'll create a `MultiHeadCenterPillar` model to train. These models are
very configurable, and the configuration can be a bit overwhelming at first.
So let's start by defining (and explaining!) some of the configuration.

For a more in-depth understanding of how the model works, check out
https://arxiv.org/pdf/2006.11275.pdf.
"""

"""
Our model will group points into voxels in 3D space, and we need to specify
how large these voxels will be. Here, we define the width, length, and height
of each voxel in the units used by the input data (meters, in the case of
Waymo Open Dataset).

Because we're predicting vertical boxes, it's common to use arbitrarily tall
voxels, so in this case we use 1000 for the z dimension.
"""
voxel_size = [0.32, 0.32, 1000]

"""
For voxelization, we also need to specify the global volume of our voxel space,
which represents the overall target area where we will identify boxes. Here
we use a range of -256 * voxel_size to 256 * voxel_size for the x and y
size, and -20 to 20 for the z size. As a result, we will produce voxel features
in an overall grid of 512x512x1 voxels.
"""
# 81.92 = 256 * 0.32
spatial_size = [-81.92, 81.92, -81.92, 81.92, -20, 20]

"""
After voxelizing points, we'll run the results through a point net, which is
a dense network with a configurable feature size. Here we define this feature
size.
"""
voxelization_feature_size = 128

"""
We'll also want to know a prior for the length, width, and height of each of
the classes we're trying to detect. This is somewhat akin to the concept of
anchor sizes in 2D object detection, but is used for numerical regularization
instead of prediction anchoring in this case.
"""
car_anchor_size = [4.5, 2.0, 1.6]
pedestrian_anchor_size = [0.6, 0.8, 1.8]

"""
Now we can build our model!

We'll define a function to create the model so that we can initialize it inside
of a tf.distribute scope later on.
"""


def build_centerpillar_model():
  """
  Our first model component is a voxelization layer. This will be used to
  dynamically map coordinates of a point to a voxel in 3D space.
  """
  voxelization_point_net = tf.keras.Sequential(
      [
          tf.keras.layers.Dense(voxelization_feature_size),
          tf.keras.layers.BatchNormalization(fused=False),
          tf.keras.layers.ReLU(),
      ]
  )
  voxelization_layer = DynamicVoxelization(
      point_net=voxelization_point_net,
      voxel_size=voxel_size,
      spatial_size=spatial_size,
  )

  """
  Our next component is a UNet that will be called on the output of the
  voxelization layer.
  """
  unet_layer = UNet(
      # This is 512,512,128
      # TODO(ianstenbit): Can we remove the need to specify this?
      input_shape=(
          voxel_utils.compute_voxel_spatial_size(
              voxel_size=voxel_size,
              spatial_size=spatial_size,
          )[:-1]
          + [voxelization_feature_size]
      ),
      # These configurations represent the number of filters and number of
      # blocks in each downsampling step of the UNet
      down_block_configs=[(128, 6), (256, 2), (512, 2)],
      # These configurations represent the number of filters in each upsampling
      # step of the UNet
      up_block_configs=[512, 256, 256],
      sync_bn=False,
  )

  """
  Next, we'll need a decoder component to decode predictions into 3D boxes. To
  do this, we'll need to specify how many heading bins we're using for each
  class, the anchor size for each class, and a pooling size for each class.
  """

  # 12 heading bins for cars, 4 for pedestrians.
  num_heading_bins = [12, 4]

  decoder = MultiClassHeatmapDecoder(
      num_classes=2,
      num_head_bin=num_heading_bins,
      anchor_size=[car_anchor_size, pedestrian_anchor_size],
      max_pool_size=[7, 3],
      max_num_box=[800, 400],
      heatmap_threshold=[0.1, 0.1],
      voxel_size=voxel_size,
      spatial_size=spatial_size,
  )

  """
  Finally, we'll create a detection head and then instantiate our full model.
  Now we can compile the model and start training!
  """
  multiclass_head = MultiClassDetectionHead(
      num_classes=2,
      num_head_bin=num_heading_bins,
  )

  model = MultiHeadCenterPillar(
      backbone=unet_layer,
      voxel_net=voxelization_layer,
      multiclass_head=multiclass_head,
      prediction_decoder=decoder,
  )

  return model

"""
Before we start training our model, we'll need to turn our labels into a format
that our model can learn and later predict.

We do this using a label encoder (much like we do in 2D object detection).
"""

# TODO(ianstenbit): Model + augmentation layers should use the same format
# In the interim, the package should do this conversion for us.
def convert_to_model_format(y):
  point_clouds = {
    "point_xyz": y["point_clouds"][..., :3],
    "point_feature": y["point_clouds"][..., 3:-1],
    "point_mask": tf.cast(y["point_clouds"][..., -1], tf.bool)
  }
  boxes = {
    "boxes": y["bounding_boxes"][..., :7],
    "classes": y["bounding_boxes"][..., 7],
    "difficulty": y["bounding_boxes"][..., -1],
    "mask": tf.cast(y["bounding_boxes"][..., 8], tf.bool)
  }
  return {
      "point_clouds": point_clouds,
      "3d_boxes": boxes,
  }

dataset = dataset.map(convert_to_model_format, num_parallel_calls=tf.data.AUTOTUNE)

label_encoder = CenterNetLabelEncoder(
    voxel_size=voxel_size,
    min_radius=[0.8, 0.8, 0], # TODO(ianstenbit): This is unused -- can we delete it?
    max_radius=[8.0, 8.0, 0], # TODO(ianstenbit): Explain this parameter
    spatial_size=spatial_size,
    num_classes=2,
    # The maximum number of target boxes that we should produce per class
    # (in this case 1024 for cars and 512 for pedestrians)
    top_k_heatmap=[1024, 512],
)

dataset = dataset.map(label_encoder, num_parallel_calls=tf.data.AUTOTUNE)

# Up to this point, our data has been in one dictionary per-batch, but
# now we split it up into a standard x, y tuple for training
def separate_points_and_boxes(y):
      x = y["point_clouds"]
      del y["point_clouds"]

      return x, y

dataset = dataset.map(separate_points_and_boxes, num_parallel_calls=tf.data.AUTOTUNE)

"""
Now we can build and compile our model!
We use a one device strategy in this tutorial, but any strategy will work.
"""

strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

with strategy.scope():
  car_box_loss = keras_cv.losses.CenterNetBoxLoss(num_heading_bins=12, anchor_size=car_anchor_size, reduction="sum")
  pedestrian_box_loss = keras_cv.losses.CenterNetBoxLoss(num_heading_bins=4, anchor_size=pedestrian_anchor_size, reduction="sum")

  model = build_centerpillar_model()

  model.compile(
    optimizer='adam',
    heatmap_loss=keras_cv.losses.BinaryPenaltyReducedFocalCrossEntropy(reduction="sum"),
    box_loss=[car_box_loss, pedestrian_box_loss],
  )

"""
Finally, we can train and evaluate our model!
We offer a `WODDetectionEvaluator` callback to easily evaluate Waymo's
detection metrics on an evaluation data set. Note that your evaluation dataset's
labels will be stored in main memory during metric evaluation.
"""

model.fit(dataset, epochs=5, callbacks=[WaymoEvaluationCallback(dataset.take(20).cache())])

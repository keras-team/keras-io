"""
Title: Creating TFRecords
Author: [Dimitre Oliveira](https://www.linkedin.com/in/dimitre-oliveira-7a1a0113a/)
Date created: 2021/02/22
Last modified: 2021/02/22
Description: Converting data to TFRecords files.
"""

"""
## Reference

- [TFRecord and tf.train.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord)

## Introduction

The TFRecord format is a simple format for storing a sequence of binary records.

Converting your data into TFRecords has many advantages, especially if later, you load it
using the [tf.data](https://www.tensorflow.org/guide/data) API. The data will take less
space from disk, it can also be read and processed faster.

Here we will learn how to convert data of different types (image, text, and numeric) into
TFRecords.

## Dependencies
"""

import os
import json
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt

"""
## Downloading the COCO2017 dataset

We will be using the [COCO2017](https://cocodataset.org/) dataset because it has many
different types of features and will serve as a good example of how to encode different
features as TFRecords.
"""

root_dir = "datasets"
tfrecords_dir = "tfrecords"
images_dir = os.path.join(root_dir, "val2017")
annotations_dir = os.path.join(root_dir, "annotations")
annotation_file = os.path.join(annotations_dir, "instances_val2017.json")
images_url = "http://images.cocodataset.org/zips/val2017.zip"
annotations_url = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

# Download image files
if not os.path.exists(images_dir):
    image_zip = tf.keras.utils.get_file(
        "images.zip", cache_dir=os.path.abspath("."), origin=images_url, extract=True,
    )
    os.remove(image_zip)

# Download caption annotation files
if not os.path.exists(annotations_dir):
    annotation_zip = tf.keras.utils.get_file(
        "captions.zip",
        cache_dir=os.path.abspath("."),
        origin=annotations_url,
        extract=True,
    )
    os.remove(annotation_zip)

print("Dataset is downloaded and extracted successfully.")

with open(annotation_file, "r") as f:
    annotations = json.load(f)["annotations"]

print(f"Number of images: {len(annotations)}")

"""
## Looking at the annotations of a single data sample
"""

pprint.pprint(annotations[60])

"""
## Parameters
"""

n_samples = 4096  # number of samples on each TFRecord
n_tfrecods = len(annotations) // n_samples  # total number of TFRecords
if len(annotations) % n_samples:
    n_tfrecods += 1

if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # Creating output folder

"""
## Auxiliary functions
"""


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, example):
    feature = {
        "image": _bytes_feature(image),
        "area": _float_feature(example["area"]),
        "bbox": _float_feature_list(example["bbox"]),
        "category_id": _int64_feature(example["category_id"]),
        "id": _int64_feature(example["id"]),
        "image_id": _int64_feature(example["image_id"]),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.image.decode_jpeg(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])

    return example


"""
## Generating TFRecords

The generated TFRecrods will have the file name format `file_{number}.tfrec`, this is
optional, but writing the number of files at the file name can make counting easier.
"""

for tfrec_num in range(n_tfrecods):
    samples = annotations[(tfrec_num * n_samples) : ((tfrec_num + 1) * n_samples)]

    with tf.io.TFRecordWriter(
        tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
    ) as writer:
        for sample in samples:
            image_path = f"{images_dir}/{sample['image_id']:012d}.jpg"
            image = tf.io.read_file(image_path).numpy()

            example = create_example(image, sample)
            writer.write(example.SerializeToString())

"""
## Looking at a generated TFRecord of a single data sample
"""

raw_dataset = tf.data.TFRecordDataset(f"{tfrecords_dir}/file_00-{n_samples}.tfrec")
parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

for features in parsed_dataset.take(1):
    for key in features.keys():
        if key != "image":
            print(f"{key}: {features[key]}")

    print(f"Image shape: {features['image'].shape}")
    plt.figure(figsize=(7, 7))
    plt.imshow(features["image"].numpy())
    plt.show()

"""
## Conclusion

Now instead of reading images and annotations from different folders, we can have all the
information of the samples reading from the same source, the TFRecords that we just
created. This process makes storing and reading the data much more easy and efficient.
"""

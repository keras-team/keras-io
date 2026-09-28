"""
Title: Semantic segmentation with SegFormer and KerasHub
Author: [Sayak Paul](https://twitter.com/RisingSayak), converted to Keras 3 by [Samudraneel Sarkar](https://github.com/samudraneel05)
Date created: 2023/01/25
Last modified: 2026/09/16
Description: Fine-tuning a SegFormer model variant for semantic segmentation.
Accelerator: GPU
"""

"""
## Introduction

In this example, we show how to fine-tune a SegFormer model variant to do
semantic segmentation on a custom dataset. Semantic segmentation is the task of
assigning a category to each and every pixel of an image. SegFormer was proposed in
[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203).
SegFormer uses a hierarchical Transformer architecture (called "Mix Transformer") as
its encoder and a lightweight decoder for segmentation. As a result, it yields
state-of-the-art performance on semantic segmentation while being more efficient than
existing models. For more details, check out the original paper.

![segformer-arch](https://i.imgur.com/BsrVwYe.png)

We leverage
[KerasHub](https://github.com/keras-team/keras-hub)
to load a pretrained SegFormer checkpoint and fine-tune it on a custom dataset.

**Note:** this example reuses code from the following sources:

* [Official tutorial on segmentation from the TensorFlow team](https://www.tensorflow.org/tutorials/images/segmentation)
* [Hugging Face Task guide on segmentation](https://huggingface.co/docs/transformers/main/en/tasks/semantic_segmentation)

To run this example, we need to install the `keras-hub` library:
"""

"""shell
pip install keras-hub -q
"""

"""
## Load the data

We use the [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset for
this example. We leverage `tensorflow_datasets` to load the dataset.
"""

import tensorflow_datasets as tfds

dataset, info = tfds.load("oxford_iiit_pet:4.*.*", with_info=True)

"""
## Prepare the datasets

For preparing the datasets for training and evaluation, we:

* Normalize the images with the mean and standard deviation used during pre-training
SegFormer.
* Subtract 1 from the segmentation masks so that the pixel values start from 0.
* Resize the images.
"""

import keras
import tensorflow as tf

image_size = 224
mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])


def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, keras.backend.epsilon())
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint["image"], (image_size, image_size))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (image_size, image_size),
        method="nearest",
    )

    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, tf.squeeze(tf.cast(input_mask, tf.int32))


"""
We now use the above utilities to prepare `tf.data.Dataset` objects including
`prefetch()` for performance. Change the `batch_size` to match the size of the GPU memory
on the GPU that you're using for training.
"""

auto = tf.data.AUTOTUNE
batch_size = 4

train_ds = (
    dataset["train"]
    .cache()
    .shuffle(batch_size * 10)
    .map(load_image, num_parallel_calls=auto)
    .batch(batch_size)
    .prefetch(auto)
)
test_ds = (
    dataset["test"]
    .map(load_image, num_parallel_calls=auto)
    .batch(batch_size)
    .prefetch(auto)
)

"""
We can check the shapes of the input images and their segmentation maps:
"""

print(train_ds.element_spec)

"""
## Visualize dataset
"""

import matplotlib.pyplot as plt


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


for images, masks in train_ds.take(2):
    sample_image, sample_mask = images[0], masks[0]
    sample_mask = tf.expand_dims(sample_mask, -1)
    display([sample_image, sample_mask])

"""
## Load a pretrained SegFormer checkpoint

We now load a pretrained SegFormer model variant from KerasHub. The
SegFormer model comes in different variants dubbed as **MiT-B0** to **MiT-B5**.
We load the smallest variant MiT-B0 (pretrained on ADE20K), which produces a
good trade-off between inference efficiency and predictive performance.
"""

import keras_hub

num_classes = 3

image_encoder = keras_hub.models.MiTBackbone.from_preset("mit_b0_ade20k_512")
backbone = keras_hub.models.SegFormerBackbone(
    image_encoder=image_encoder,
    projection_filters=256,
)
model = keras_hub.models.SegFormerImageSegmenter(
    backbone=backbone,
    num_classes=num_classes,
)

"""
Since we're using a custom dataset which has a different set of semantic class
labels than the pre-training dataset (ADE20K), the segmenter initializes a new
segmentation head. This is normal.

We can now initialize an optimizer and compile the model with it.
"""

"""
## Compile the model
"""

lr = 0.00006
optimizer = keras.optimizers.Adam(learning_rate=lr)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

"""
## Prediction callback to monitor training progress

It helps us to visualize some sample predictions when the model is being fine-tuned,
thereby helping us to monitor the progress of the model. This callback is inspired from
[this tutorial](https://www.tensorflow.org/tutorials/images/segmentation).
"""

from IPython.display import clear_output


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, -1)
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for images, masks in dataset.take(num):
            masks = tf.expand_dims(masks, -1)
            pred_masks = model.predict(images)
            display([images[0], masks[0], create_mask(pred_masks)])
    else:
        display(
            [
                sample_image,
                sample_mask,
                create_mask(model.predict(tf.expand_dims(sample_image, 0))),
            ]
        )


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(self.dataset)
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))


"""
## Train model
"""

# Increase the number of epochs if the results are not of expected quality.
epochs = 5

history = model.fit(
    train_ds,
    validation_data=test_ds,
    callbacks=[DisplayCallback(test_ds)],
    epochs=epochs,
)

"""
## Inference

We perform inference on a few samples from the test set.
"""

show_predictions(test_ds, 5)

"""
## Conclusion

In this example, we learned how to fine-tune a SegFormer model variant on a custom
dataset for semantic segmentation. In the interest of brevity, the example
was kept short. However, there are a couple of things, you can further try out:

* Incorporate data augmentation to potentially improve the results.
* Use a larger SegFormer model checkpoint to see how the results are affected —
  e.g. `mit_b2_ade20k_512` or `segformer_b2_ade20k_512` presets.
* Save and share the fine-tuned model with `model.save_to_preset()` or by
  uploading it to Kaggle Models / Hugging Face Hub.
"""

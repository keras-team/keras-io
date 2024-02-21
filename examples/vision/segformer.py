"""
Title: Semantic segmentation with SegFormer and Hugging Face Transformers
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2023/01/25
Last modified: 2023/01/29
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
[Hugging Face Transformers](https://github.com/huggingface/transformers)
to load a pretrained SegFormer checkpoint and fine-tune it on a custom dataset.

**Note:** this example reuses code from the following sources:

* [Official tutorial on segmentation from the TensorFlow team](https://www.tensorflow.org/tutorials/images/segmentation)
* [Hugging Face Task guide on segmentation](https://huggingface.co/docs/transformers/main/en/tasks/semantic_segmentation)

To run this example, we need to install the `transformers` library:
"""

"""shell
pip install transformers -q
"""

"""
## Load the data

We use the [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset for
this example. We leverage `tensorflow_datasets` to load the dataset.
"""

import tensorflow_datasets as tfds

dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)

"""
## Prepare the datasets

For preparing the datasets for training and evaluation, we:

* Normalize the images with the mean and standard deviation used during pre-training
SegFormer.
* Subtract 1 from the segmentation masks so that the pixel values start from 0.
* Resize the images.
* Transpose the images such that they are in `"channels_first"` format. This is to make
them compatible with the SegFormer model from Hugging Face Transformers.
"""

import tensorflow as tf
from tensorflow.keras import backend

image_size = 512
mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])


def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint["image"], (image_size, image_size))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (image_size, image_size),
        method="bilinear",
    )

    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.transpose(input_image, (2, 0, 1))
    return {"pixel_values": input_image, "labels": tf.squeeze(input_mask)}


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
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


for samples in train_ds.take(2):
    sample_image, sample_mask = samples["pixel_values"][0], samples["labels"][0]
    sample_image = tf.transpose(sample_image, (1, 2, 0))
    sample_mask = tf.expand_dims(sample_mask, -1)
    display([sample_image, sample_mask])

"""
## Load a pretrained SegFormer checkpoint

We now load a pretrained SegFormer model variant from Hugging Face Transformers. The
SegFormer model comes in different variants dubbed as **MiT-B0** to **MiT-B5**. You can
find these checkpoints
[here](https://huggingface.co/models?pipeline_tag=image-segmentation&sort=downloads&search=segformer).
We load the smallest variant Mix-B0, which produces a good trade-off
between inference efficiency and predictive performance.
"""

from transformers import TFSegformerForSemanticSegmentation

model_checkpoint = "nvidia/mit-b0"
id2label = {0: "outer", 1: "inner", 2: "border"}
label2id = {label: id for id, label in id2label.items()}
num_labels = len(id2label)
model = TFSegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

"""
The warning is telling us that we're throwing away some weights and newly initializing
some others. Don't panic! This is absolutely normal. Since we're using a custom dataset
which has a different set of semantic class labels than the pre-training dataset,
[`TFSegformerForSemanticSegmentation`](https://huggingface.co/docs/transformers/model_doc/segformer#transformers.TFSegformerForSemanticSegmentation)
is initializing a new decoder head.

We can now initialize an optimizer and compile the model with it.
"""

"""
## Compile the model
"""

lr = 0.00006
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer)

"""
Notice that we are not using any loss function for compiling the model. This is because
the forward pass of the model
[implements](https://github.com/huggingface/transformers/blob/820c46a707ddd033975bc3b0549eea200e64c7da/src/transformers/models/segformer/modeling_tf_segformer.py#L873)
the loss computation part when we provide labels alongside the input images. After
computing the loss, the model returned a structured `dataclass` object which is
then used to guide the training process.

With the compiled model, we can proceed and call `fit()` on it to begin the fine-tuning
process!
"""

"""
## Prediction callback to monitor training progress

It helps us to visualize some sample predictions when the model is being fine-tuned,
thereby helping us to monitor the progress of the model. This callback is inspired from
[this tutorial](https://www.tensorflow.org/tutorials/images/segmentation).
"""

from IPython.display import clear_output


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    pred_mask = tf.expand_dims(pred_mask, -1)
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for sample in dataset.take(num):
            images, masks = sample["pixel_values"], sample["labels"]
            masks = tf.expand_dims(masks, -1)
            pred_masks = model.predict(images).logits
            images = tf.transpose(images, (0, 2, 3, 1))
            display([images[0], masks[0], create_mask(pred_masks)])
    else:
        display(
            [
                sample_image,
                sample_mask,
                create_mask(model.predict(tf.expand_dims(sample_image, 0))),
            ]
        )


class DisplayCallback(tf.keras.callbacks.Callback):
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
* Use a larger SegFormer model checkpoint to see how the results are affected.
* Push the fine-tuned model to the Hugging Face for sharing with the community easily.
You can do so just by doing `model.push_to_hub("your-username/your-awesome-model")`.
And then you can load the model by doing
`TFSegformerForSemanticSegmentation.from_pretrained("your-username/your-awesome-model"`).
[Here](https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation-tf.ipynb)
is an end-to-end example if you're looking for a reference.
* If you'd rather push the model checkpoints to the Hub as the model is being
fine-tuned you can instead use the `PushToHubCallback` Keras callback.
[Here](https://gist.github.com/sayakpaul/f474ffb01f0cdcc8ba239357965c3bca) is an example.
[Here](https://huggingface.co/sayakpaul/mit-b0-finetuned-pets) is an example of a model
repository that was created using this callback.
"""

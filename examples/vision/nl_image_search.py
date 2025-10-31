"""
Title: Natural language image search with CLIP dual encoder
Author: [Vinayak Sharma](https://www.vinayak19th.me/)
Date created: 2025/10/30
Last modified: 2025/10/30
Description: Implementation of the CLIP model for retrieving images that match natural language queries.
Accelerator: GPU
"""

"""
## Introduction
The example demonstrates how to build a dual encoder (also known as two-tower) neural
network model to search for images using natural language. The model is inspired by the
CLIP approach, introduced by Alec Radford et al. The idea is to train a vision encoder
and a text encoder jointly to project the representation of images and their captions
into the same embedding space, such that the caption embeddings are located near the
embeddings of the images they describe.

This example requires Keras 3. In addition, Keras Hub and TensorFlow Text are required
for the BERT model. Finally, we also require the `wget` package to load our dataset.
These libraries can be installed using the following command:

```
pip install -q -U tensorflow-text keras-hub wget
```

While the model training can be done using the Torch or Tensorflow backends, the data pipeline for this demo is written using the
[tf.data](https://www.tensorflow.org/guide/data) API. 

This demo was originally developed by [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/), and has been adapted to Keras 3 and updated by [Vinayak Sharma](https://www.vinayak19th.me/).
"""

import os

# os.environ["KERAS_BACKEND"] = "torch"

import glob
import collections
import json
import numpy as np
import tensorflow as tf
import keras
from keras import ops
import tensorflow_text
from keras import layers
import keras_hub

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

import wget
import zipfile

# Suppressing tf.hub warnings
tf.get_logger().setLevel("ERROR")

"""
## Prepare the MS-COCO Captions Dataset

We will use the MS-COCO dataset to train our dual encoder model. MS-COCO contains over
82,000 images, each of which has at least 5 different caption annotations. The dataset is
usually used for image captioning tasks, but we can repurpose the image-caption pairs to
train our dual encoder model for image search.

Download and extract the data

First, let's download the dataset, which consists of two compressed folders: one with
images, and the other—with associated image captions. Note that the compressed images
folder is 13GB in size.
"""

root_dir = "datasets"
annotations_dir = os.path.join(root_dir, "annotations")
images_dir = os.path.join(root_dir, "train2014", "train2014")
tfrecords_dir = os.path.join(root_dir, "tfrecords")
annotation_file = os.path.join(
    annotations_dir, "annotations", "captions_train2014.json"
)

if not os.path.exists(annotations_dir):
    annotation_zip = wget.download(
        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    )
    with zipfile.ZipFile(annotation_zip, "r") as zip_ref:
        zip_ref.extractall(annotations_dir)
    # os.remove(os.path.join(root_dir,"captions.zip"))

print("\nDownloading the images.")

if not os.path.exists(images_dir):
    image_zip = wget.download("http://images.cocodataset.org/zips/train2014.zip")
    print("Downloaded the images.\nunzipping")
    with zipfile.ZipFile(image_zip, "r") as zip_ref:
        zip_ref.extractall(images_dir)
    # os.remove(os.path.join(root_dir,"train2014.zip"))

print("\nDataset is downloaded and extracted successfully.")

with open(annotation_file, "r") as f:
    annotations = json.load(f)["annotations"]

image_path_to_caption = collections.defaultdict(list)
for element in annotations:
    caption = f"{element['caption'].lower().rstrip('.')}"
    image_path = images_dir + "/COCO_train2014_" + "%012d.jpg" % (element["image_id"])
    image_path_to_caption[image_path].append(caption)

images = glob.glob("datasets/train2014/*.jpg")
image_paths = list(image_path_to_caption.keys())
if len(images) != len(image_paths):
    print(
f"Not all images extracted correctly, expected {len(image_paths)} images, found
{len(images)} images"
    )
print(f"Number of images: {len(image_paths)}")

"""
### Pre-process and save data into TF-Record files
You can change the `sample_size` parameter to control many image-caption pairs will be
used for training the dual encoder model. In this example we set `train_size` to 30,000
images, which is about 35% of the dataset. 

We use 2 captions for each image, thus producing 60,000 image-caption pairs. The size of
the training set affects the quality of the produced encoders, but more examples would
lead to longer training time.
"""

train_size = 30000
valid_size = 5000
captions_per_image = 2
images_per_file = 2000

train_image_paths = image_paths[:train_size]
num_train_files = int(np.ceil(train_size / images_per_file))
train_files_prefix = os.path.join(tfrecords_dir, "train")

valid_image_paths = image_paths[-valid_size:]
num_valid_files = int(np.ceil(valid_size / images_per_file))
valid_files_prefix = os.path.join(tfrecords_dir, "valid")

tf.io.gfile.makedirs(tfrecords_dir)


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example(image_path, caption):
    feature = {
        "caption": bytes_feature(caption.encode()),
        "raw_image": bytes_feature(tf.io.read_file(image_path).numpy()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(file_name, image_paths):
    caption_list = []
    image_path_list = []
    for image_path in image_paths:
        captions = image_path_to_caption[image_path][:captions_per_image]
        caption_list.extend(captions)
        image_path_list.extend([image_path] * len(captions))

    with tf.io.TFRecordWriter(file_name) as writer:
        for example_idx in range(len(image_path_list)):
            example = create_example(
                image_path_list[example_idx], caption_list[example_idx]
            )
            writer.write(example.SerializeToString())
    return example_idx + 1


def write_data(image_paths, num_files, files_prefix):
    example_counter = 0
    for file_idx in tqdm(range(num_files)):
        file_name = files_prefix + "-%02d.tfrecord" % (file_idx)
        start_idx = images_per_file * file_idx
        end_idx = start_idx + images_per_file
        example_counter += write_tfrecords(file_name, image_paths[start_idx:end_idx])
    return example_counter


found_files = glob.glob(os.path.join(root_dir, "tfrecords", "train-*.tfrecord"))
if len(found_files) != num_train_files:
    train_example_count = write_data(
        train_image_paths, num_train_files, train_files_prefix
    )
    print(f"{train_example_count} training examples were written to tfrecord files.")
else:
    print(f"{num_train_files} tfrecord files found.")
    print(f"{num_train_files*images_per_file} training examples in the tfrecord files.")
    train_example_count = 60000

found_files = glob.glob(os.path.join(root_dir, "tfrecords", "valid-*.tfrecord"))
if len(found_files) != num_valid_files:
    valid_example_count = write_data(
        valid_image_paths, num_valid_files, valid_files_prefix
    )
    print(f"{valid_example_count} evaluation examples were written to tfrecord files.")
else:
    print(f"{num_valid_files} tfrecord files found.")
    print(f"{num_valid_files*images_per_file} training examples in the tfrecord files.")
    valid_example_count = 10000

"""
### Create a
[tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) for
training and eval
Converting the Data to a
[tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) allows for
better data loading performance by implementing the following functions:
1. Mapping a `read_example` function which - 
   * Loads images from tfrecord files and decompresses the `jpg` format into tensors.
* Loads the captions and uses the
[TextClassifierPreprocessor](https://keras.io/keras_hub/api/base_classes/text_classifier_p
reprocessor/) to convert the text into tokens for our language model.
* Run all these read and pre-processing operations in parallel via the
`num_parallel_calls` argument.
2. Batch the read examples.
3. Create a `prefetch` pipeline which will load multiple examples into memory for more
efficient training
"""

feature_description = {
    "caption": tf.io.FixedLenFeature([], tf.string),
    "raw_image": tf.io.FixedLenFeature([], tf.string),
}
preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(
    "bert_small_en_uncased"
)


def read_example(example):
    features = tf.io.parse_single_example(example, feature_description)
    raw_image = features.pop("raw_image")
    features["image"] = tf.image.resize(
        tf.image.decode_jpeg(raw_image, channels=3), size=(224, 224)
    )

    features["caption"] = preprocessor(features["caption"])
    return features


def get_dataset(file_pattern, batch_size):

    return (
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
        .map(
            read_example,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


"""
Lets load 2 samples from our dataset with `batch_size` of 4 to see what the input to the
model looks like
"""

train_dataset = get_dataset(os.path.join(root_dir, "tfrecords", "train-*.tfrecord"), 4)
for i in train_dataset.take(2).cache():
    print(f"{i['image'].shape} images\n{i['caption']['token_ids'].shape[0]} captions")

del train_dataset

"""
# Creating Model
The CLIP Model has 3 main components:
1. **Projection Head** : Model to create the unified embedding space
2. **Vision Encoder** : Model to learn an embedding from images
3. **Text Encoder** : Transformer to create text embeddings
"""

"""
---
## Implementing the Projection Head
The projection head is used to transform the image and the text embeddings to the same
embedding space with the same dimensionality. This is done via a set of [Dense
layers](https://keras.io/api/layers/core_layers/dense/) and a final normalized output.
[Dropout layers](https://keras.io/api/layers/regularization_layers/dropout/) are added to
reduce overfitting.
"""


def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = ops.gelu(projected_embeddings)
        x = layers.Dense(projection_dims)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([projected_embeddings, x])
        projected_embeddings = layers.LayerNormalization()(x)
    return projected_embeddings


"""
---
## Vision Encoder
In this example, we use [EfficientNetV2B3 from Keras
Applications](https://keras.io/api/applications/efficientnet_v2/) as the base for the
vision encoder. The pre-trained weights from the model are from the 'Imagenet' dataset. 
"""


def create_vision_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained Xception model to be used as the base encoder.
    efficientNet = keras.applications.EfficientNetV2B3(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    # Set the trainability of the base encoder.
    for layer in efficientNet.layers:
        layer.trainable = trainable
    # Receive the images as inputs.
    inputs = layers.Input(shape=(224, 224, 3), name="image_input")
    # Generate the embeddings for the images using the xception model.
    embeddings = efficientNet(inputs)
    embeddings = keras.layers.Dense(1024, activation="gelu")(embeddings)
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return keras.Model(inputs, outputs, name="vision_encoder")


"""
---
## Language Encoder
In this example, we use [BERT from Keras
Hub](https://keras.io/keras_hub/api/models/bert/bert_backbone/) as the base for the
vision encoder. Specifically, we use the *'small_uncased'* version which has $28.76$M
parameters.
"""


def create_text_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained BERT model to be used as the base encoder.
    bert = keras_hub.models.BertBackbone.from_preset(
        "bert_small_en_uncased", load_weights=True, name="BERT"
    )
    # Set the trainability of the base encoder.
    bert.trainable = trainable
    # Receive the text as inputs.
    inputs = bert.input
    # Preprocess the text.
    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(inputs)["pooled_output"]
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return keras.Model(inputs, outputs, name="text_encoder")


"""
---
## Create Both Encoders
In this experiment, we freeze the base encoders for text and images, and make only the
projection head trainable.

We can now create both models with a `projection_dim` of $256$ and a `dropout_rate` of
0.1.
"""

vision_encoder = create_vision_encoder(
    num_projection_layers=1,
    projection_dims=256,
    dropout_rate=0.1,
    trainable=False,
)
text_encoder = create_text_encoder(
    num_projection_layers=1,
    projection_dims=256,
    dropout_rate=0.1,
    trainable=False,
)

"""
We can see the model summaries to better undertand the architectures
"""

vision_encoder.summary()

text_encoder.summary()

"""
---
## Create Dual Encoder for CLIP Training
To calculate the loss, we compute the pairwise dot-product similarity between each
caption_i and images_j in the batch as the predictions. The target similarity between
caption_i and image_j is computed as the average of the (dot-product similarity between
caption_i and caption_j) and (the dot-product similarity between image_i and image_j).
Then, we use crossentropy to compute the loss between the targets and the predictions.

The `DualEncoder` will also set the `train_step` method which is used by `model.fit()`
based on the Keras Backend so that we can train with PyTorch or Tensorflow.

*NOTE: Since these are relatively large models, it is reccommended to train them using a
single very powerful GPU or multiple GPUs. If both options are not available, use a
smaller batch size.*
"""


class DualEncoder(keras.Model):
    def __init__(self, text_encoder, image_encoder, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="loss")
        ## Select train_step function based on the keras backend
        if keras.config.backend() == "torch":
            self.train_step = self.train_step_torch
        elif keras.config.backend() == "tensorflow":
            self.train_step = self.train_step_tf

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        # Place each encoder on a separate GPU (if available).
        # TF will fallback on available devices if there are fewer than 2 GPUs.
        # Get the embeddings for the captions.
        caption_embeddings = text_encoder(features["caption"], training=training)
        # Get the embeddings for the images.
        image_embeddings = vision_encoder(features["image"], training=training)
        return caption_embeddings, image_embeddings

    def compute_loss(self, caption_embeddings, image_embeddings):
        # logits[i][j] is the dot_similarity(caption_i, image_j).
        logits = ops.divide(
            ops.einsum("ae,be -> ab", caption_embeddings, image_embeddings),
            self.temperature,
        )

        # images_similarity[i][j] is the dot_similarity(image_i, image_j).
        images_similarity = ops.einsum(
            "ae,be -> ab", image_embeddings, image_embeddings
        )
        # captions_similarity[i][j] is the dot_similarity(caption_i, caption_j).
        captions_similarity = ops.einsum(
            "ae,be -> ab", caption_embeddings, caption_embeddings
        )
# targets[i][j] = avarage dot_similarity(caption_i, caption_j) and
dot_similarity(image_i, image_j).
        targets = keras.activations.softmax(
            (captions_similarity + images_similarity) / (2 * self.temperature)
        )
        # Compute the loss for the captions using cross-entropy
        captions_loss = keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )
        # Compute the loss for the images using crossentropy
        images_loss = keras.losses.categorical_crossentropy(
            y_true=ops.transpose(targets),
            y_pred=ops.transpose(logits),
            from_logits=True,
        )
        # Return the mean of the loss over the batch.
        return (captions_loss + images_loss) / 2

    def train_step_tf(self, features):
        with tf.GradientTape() as tape:
            # Forward pass
            caption_embeddings, image_embeddings = self(features, training=True)
            loss = self.compute_loss(caption_embeddings, image_embeddings)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def train_step_torch(self, features):
        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()
        # Forward pass
        caption_embeddings, image_embeddings = self(features, training=True)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        # Backward pass
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        caption_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


"""
We now create the DualEncoder with the [AdamW](https://keras.io/api/optimizers/adamw/)
optimizer. We also set the `run_eagerly` to `False` to improve training performance and
efficiency. 

The `temperature` parameter is used to make the softmax function more seperable. This is
explained in greater detail in [this stack
exachange](https://stats.stackexchange.com/questions/527080/what-is-the-role-of-temperature-in-softmax).
exachange](https://stats.stackexchange.com/questions/527080/what-is-the-role-of-temperature-in-softmax).
"""

dual_encoder = DualEncoder(
    text_encoder, vision_encoder, temperature=0.05, name="DualEncoder"
)
dual_encoder.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.001),
    run_eagerly=False,
)

"""
### Training the Dual Encoder Model
In order to speed up the training we use a `batch_size` of $64$, and train for roughly
$5$-epochs. Normally, we would train for a longer with larger batch-sizes but due to
hardware constraints we are limited here.

This takes ~5 mins of an RTX 4080 SUPER.
"""

num_epochs = 5  # In practice, train for at least 30 epochs
batch_size = 64

"""
We train using the
[ReduceLROnPlateau](https://keras.io/api/callbacks/reduce_lr_on_plateau/) and
[ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/) which reduce the
learning rate if our performance does not improve and saves intermediate models
respectively.
"""

print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
print(f"Number of examples (caption-image pairs): {train_example_count}")
print(f"Batch size: {batch_size}")
print(f"Steps per epoch: {int(np.ceil(train_example_count / batch_size))}")
train_dataset = get_dataset(os.path.join(tfrecords_dir, "train-*.tfrecord"), batch_size)
valid_dataset = get_dataset(os.path.join(tfrecords_dir, "valid-*.tfrecord"), batch_size)

# Create a learning rate scheduler callback.
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3
)

checkpoint_filepath = "./checkpoints/checkpoint.model.keras"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_loss",
    mode="min",
    verbose=1,
    save_best_only=True,
)

"""
You might see some 'ptxas warning :' messages depending on tensorflow verison.
"""

history = dual_encoder.fit(
    train_dataset,
    epochs=5,
    validation_data=valid_dataset,
    callbacks=[reduce_lr, model_checkpoint_callback],
)

"""
The Trained models are saved to be used later.
"""

print("Training completed. Saving vision and text encoders...")
vision_encoder.save("vision_encoder.keras")
text_encoder.save("text_encoder.keras")
print("Models are saved.")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train", "valid"], loc="upper right")
plt.show()

"""
---
# Testing - Search for images using natural language queries

We can then retrieve images corresponding to natural language queries via the following
steps:

1. Generate embeddings for the images by feeding them into the `vision_encoder`.
2. Feed the natural language query to the `text_encoder` to generate a *query embedding*.
3. Compute the similarity between the query embedding and the image embeddings in the
index to retrieve the indices of the top matches.
4. Look up the paths of the top matching images to display them.

Note that, after training the dual encoder, only the fine-tuned `vision_encoder` and
`text_encoder` models will be used, while the `dual_encoder` model will be discarded.
"""

"""
### Generate embeddings for the images
We load the images and feed them into the vision_encoder to generate their embeddings. In
large scale systems, this step is performed using a parallel data processing framework,
such as Apache Spark or Apache Beam. Generating the image embeddings may take several
minutes
"""


def read_image(image_path):
    image_array = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    return tf.image.resize(image_array, (224, 224))


print(f"Generating embeddings for {len(image_paths)} images...")
image_embeddings = vision_encoder.predict(
    tf.data.Dataset.from_tensor_slices(image_paths).map(read_image).batch(batch_size),
    verbose=1,
)
print(f"Image embeddings shape: {image_embeddings.shape}.")

"""
### Retrieve relevant images
In this example, we use exact matching by computing the dot product similarity between
the input query embedding and the image embeddings, and retrieve the top k matches.
However, approximate similarity matching, using frameworks like
[ScaNN](https://github.com/google-research/google-research/tree/master/scann),
[Annoy](https://github.com/spotify/annoy), or
[Faiss](https://github.com/facebookresearch/faiss) is preferred in real-time use cases to
scale with a large number of images.
"""


def find_matches(image_embeddings, queries, k=9, normalize=True):
    # Get the embedding for the query.
    token = preprocessor(queries)
    query_embedding = text_encoder(token)
    # Normalize the query and the image embeddings.
    if normalize:
        image_embeddings = ops.normalize(image_embeddings, axis=1)
        query_embedding = ops.normalize(query_embedding, axis=1)
    # Compute the dot product between the query and the image embeddings.
    dot_similarity = ops.matmul(query_embedding, ops.transpose(image_embeddings))
    # Retrieve top k indices.
    _, results = ops.top_k(dot_similarity, k)
    results = results.numpy()
    # Return matching image paths.
    return [[image_paths[idx] for idx in indices] for indices in results]


"""
Set the query variable to the type of images you want to search for. Try things like: 'a
plate of healthy food', 'a woman wearing a hat is walking down a sidewalk', 'a bird sits
near to the water', or 'wild animals are standing in a field'.
"""

query = "a bird sits near to the water"
matches = find_matches(image_embeddings, [query], normalize=True)[0]

plt.figure(figsize=(20, 20))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(mpimg.imread(matches[i]))
    plt.axis("off")

"""
### Evaluate the retrieval quality

To evaluate the dual encoder model, we use the captions as queries. We use the
out-of-training-sample images and captions to evaluate the retrieval quality, using top k
accuracy. A true prediction is counted if, for a given caption, its associated image is
retrieved within the top k matches.
"""


def compute_top_k_accuracy(image_paths, k=100):
    hits = 0
    num_batches = int(np.ceil(len(image_paths) / batch_size))
    for idx in tqdm(range(num_batches)):
        start_idx = idx * batch_size
        end_idx = start_idx + batch_size
        current_image_paths = image_paths[start_idx:end_idx]
        queries = [
            image_path_to_caption[image_path][0] for image_path in current_image_paths
        ]
        result = find_matches(image_embeddings, queries, k)
        hits += sum(
            [
                image_path in matches
                for (image_path, matches) in list(zip(current_image_paths, result))
            ]
        )

    return hits / len(image_paths)


print("Scoring training data...")
train_accuracy = compute_top_k_accuracy(train_image_paths)
print(f"Train accuracy: {round(train_accuracy * 100, 3)}%")

print("Scoring evaluation data...")
eval_accuracy = compute_top_k_accuracy(image_paths[train_size:])
print(f"Eval accuracy: {round(eval_accuracy * 100, 3)}%")

"""
# Final remarks
You can obtain better results by increasing the size of the training sample, train for
more epochs, explore other base encoders for images and text, set the base encoders to be
trainable, and tune the hyperparameters, especially the temperature for the softmax in
the loss computation.

You can also try to train the model using the PyTorch backend.
"""

"""
Title: Natural language image search with a Dual Encoder
Author: Khalid Salama
Date created: 2021/01/30
Last modified: 2026/02/15
Description: Implementation of a dual encoder model for retrieving images that match natural language queries.
Accelerator: GPU
Converted to Keras 3 by: [Maitry Sinha](https://github.com/maitry63)
"""

"""
## Introduction

The example demonstrates how to build a dual encoder (also known as two-tower) neural network
model to search for images using natural language. The model is inspired by
the [CLIP](https://openai.com/blog/clip/)
approach, introduced by Alec Radford et al. The idea is to train a vision encoder and a text
encoder jointly to project the representation of images and their captions into the same embedding
space, such that the caption embeddings are located near the embeddings of the images they describe.
"""
"""
## Setup
"""

import os
import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

import keras
from keras import layers, ops
from keras.utils import PyDataset, load_img, img_to_array
import keras_nlp

"""
## Prepare the data

We will use the [MS-COCO](https://cocodataset.org/#home) dataset to train our
dual encoder model. MS-COCO contains over 82,000 images, each of which has at least
5 different caption annotations. The dataset is usually used for
[image captioning](https://www.tensorflow.org/tutorials/text/image_captioning)
tasks, but we can repurpose the image-caption pairs to train our dual encoder
model for image search.

###
Download and extract the data

First, let's download the dataset, which consists of two compressed folders:
one with images, and the other—with associated image captions.
Note that the compressed images folder is 13GB in size.
"""

root_dir = "datasets"
annotations_dir = os.path.join(root_dir, "captions_extracted/annotations")
images_dir = os.path.join(root_dir, "train2014_extracted/train2014")
annotation_file = os.path.join(annotations_dir, "captions_train2014.json")

os.makedirs(root_dir, exist_ok=True)

if not os.path.exists(annotations_dir):
    zip_path = keras.utils.get_file(
        "captions.zip",
        origin="https://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        extract=True,
        cache_dir=".",
    )

if not os.path.exists(images_dir):
    zip_path = keras.utils.get_file(
        "train2014.zip",
        origin="https://images.cocodataset.org/zips/train2014.zip",
        extract=True,
        cache_dir=".",
    )

with open(annotation_file, "r") as f:
    annotations = json.load(f)["annotations"]

image_path_to_caption = collections.defaultdict(list)
for ann in annotations:
    caption = ann["caption"].lower().rstrip(".")
    image_path = os.path.join(images_dir, f"COCO_train2014_{ann['image_id']:012d}.jpg")
    image_path_to_caption[image_path].append(caption)

image_paths = list(image_path_to_caption.keys())
print(f"Number of images: {len(image_paths)}")

"""
## Process and save the data

You can change the `sample_size` parameter to control how many image-caption pairs
will be used for training the dual encoder model.
In this example we set `train_size` to 30,000 images,
which is about 35% of the dataset. We use 2 captions for each
image, thus producing 60,000 image-caption pairs. The size of the training set
affects the quality of the produced encoders, but more examples would lead to
longer training time.
"""

train_size = 30000
valid_size = 5000
train_image_paths = image_paths[:train_size]
valid_image_paths = image_paths[train_size : train_size + valid_size]


bert_preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_small_en_uncased"
)


class ImageCaptionDataset(PyDataset):
    def __init__(self, image_paths, batch_size, captions_per_image=2, **kwargs):
        super().__init__(**kwargs)
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.captions_per_image = captions_per_image

        # Total number of image, caption pairs
        self.total_samples = len(self.image_paths) * self.captions_per_image

    @property
    def num_batches(self):
        return int(np.ceil(self.total_samples / self.batch_size))

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.total_samples)

        images, captions = [], []

        for sample_idx in range(start, end):
            image_idx = sample_idx // self.captions_per_image
            caption_idx = sample_idx % self.captions_per_image

            path = self.image_paths[image_idx]

            img = load_img(path, target_size=(299, 299))
            img = (img_to_array(img) / 127.5) - 1.0

            images.append(img)
            captions.append(image_path_to_caption[path][caption_idx])

        tokenized = bert_preprocessor(np.array(captions))

        return {
            "image": np.array(images, dtype="float32"),
            "token_ids": tokenized["token_ids"],
            "padding_mask": tokenized["padding_mask"],
            "segment_ids": tokenized["segment_ids"],
        }, np.zeros((len(images),))


"""
## Implement the projection head

The projection head is used to transform the image and the text embeddings to
the same embedding space with the same dimensionality.
"""


def project_embeddings(x, num_layers, dim, dropout):
    for _ in range(num_layers):
        residual = layers.Dense(dim)(x)
        x = layers.Dense(dim)(residual)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Add()([x, residual])
        x = layers.LayerNormalization()(x)
    return x


"""
## Implement the vision encoder

In this example, we use [Xception](https://keras.io/api/applications/xception/)
from [Keras Applications](https://keras.io/api/applications/) as the base for the
vision encoder.
"""


def create_vision_encoder(num_layers, dim, dropout, trainable=False):

    # Load the pre-trained Xception model to be used as the base encoder.
    base = keras.applications.Xception(
        include_top=False, weights="imagenet", pooling="avg"
    )

    # Set the trainability of the base encoder.
    base.trainable = trainable
    # Receive the images as inputs.
    inputs = layers.Input((299, 299, 3), name="image")
    x = base(inputs)
    # Project the embeddings
    outputs = project_embeddings(x, num_layers, dim, dropout)
    # Create the vision encoder model.
    return keras.Model(inputs, outputs, name="vision_encoder")


"""
## Implement the text encoder
We use [BERT](bert_small_en_uncased)
from [KerasNLP](https://keras.io/api/keras_nlp/models/bert_backbone/) as the text encoder
"""


def create_text_encoder(num_layers, dim, dropout, trainable=False):

    # Load the pre-trained BERT model to be used as the base encoder.
    bert = keras_nlp.models.BertBackbone.from_preset("bert_small_en_uncased")

    # Set the trainability of the base encoder.
    bert.trainable = trainable

    # Receive the text as inputs.
    token_ids = layers.Input(shape=(None,), dtype="int32", name="token_ids")
    padding_mask = layers.Input(shape=(None,), dtype="bool", name="padding_mask")
    segment_ids = layers.Input(shape=(None,), dtype="int32", name="segment_ids")

    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(
        {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "segment_ids": segment_ids,
        }
    )

    # Project the embeddings produced by the model.
    outputs = project_embeddings(embeddings["pooled_output"], num_layers, dim, dropout)

    # Create the text encoder model.
    return keras.Model(
        [token_ids, padding_mask, segment_ids], outputs, name="text_encoder"
    )


"""
## Implement the dual encoder

To calculate the loss, we compute the pairwise dot-product similarity between
each `caption_i` and `images_j` in the batch as the predictions.
The target similarity between `caption_i`  and `image_j` is computed as
the average of the (dot-product similarity between `caption_i` and `caption_j`)
and (the dot-product similarity between `image_i` and `image_j`).
Then, we use crossentropy to compute the loss between the targets and the predictions.
"""


class DualEncoder(keras.Model):
    def __init__(self, text_encoder, vision_encoder, temperature=0.05, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # Forward pass
        text_emb = self.text_encoder(
            [x["token_ids"], x["padding_mask"], x["segment_ids"]]
        )
        vision_emb = self.vision_encoder(x["image"])

        # Normalize embeddings
        text_emb = ops.divide(
            text_emb, ops.norm(text_emb, axis=-1, keepdims=True) + 1e-12
        )
        vision_emb = ops.divide(
            vision_emb, ops.norm(vision_emb, axis=-1, keepdims=True) + 1e-12
        )

        # Contrastive Loss
        logits = ops.matmul(text_emb, ops.transpose(vision_emb)) / self.temperature
        targets = ops.eye(ops.shape(logits)[0])

        # Symmetric Cross-Entropy
        loss_txt = keras.losses.categorical_crossentropy(
            targets, logits, from_logits=True
        )
        loss_img = keras.losses.categorical_crossentropy(
            targets, ops.transpose(logits), from_logits=True
        )
        loss = ops.mean((loss_txt + loss_img) / 2.0)

        # Track the progress
        self.loss_tracker.update_state(loss)
        return loss

    def call(self, x):
        text_emb = self.text_encoder(
            [x["token_ids"], x["padding_mask"], x["segment_ids"]]
        )
        vision_emb = self.vision_encoder(x["image"])
        return ops.matmul(text_emb, ops.transpose(vision_emb))

    @property
    def metrics(self):
        return [self.loss_tracker]


"""
## Train the dual encoder model

In this experiment, we freeze the base encoders for text and images, and make only
the projection head trainable.
"""

num_epochs = 5  # In practice, train for at least 30 epochs
batch_size = 256
vision_encoder = create_vision_encoder(1, 256, 0.1)
text_encoder = create_text_encoder(1, 256, 0.1)
dual_encoder = DualEncoder(text_encoder, vision_encoder)

dual_encoder.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-3)
)

"""
## Note that training the model with 60,000 image-caption pairs, with a batch size of 256,
takes around 12 minutes per epoch using a V100 GPU accelerator. If 2 GPUs are available,
the epoch takes around 8 minutes.
"""

print(f"Batch size: {batch_size}")
# print(f"Steps per epoch: {int(np.ceil(train_example_count / batch_size))}")
train_dataset = ImageCaptionDataset(train_image_paths, batch_size)
valid_dataset = ImageCaptionDataset(valid_image_paths, batch_size)

# Create a learning rate scheduler callback.
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3
)
# Create an early stopping callback.
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = dual_encoder.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=valid_dataset,
    callbacks=[reduce_lr, early_stopping],
)
print("Training completed. Saving vision and text encoders...")
vision_encoder.save("vision_encoder.keras")
text_encoder.save("text_encoder.keras")
print("Models are saved.")

"""
## Plotting the training loss:
"""

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train", "valid"], loc="upper right")
plt.show()

"""
## Search for images using natural language queries

We can then retrieve images corresponding to natural language queries via
the following steps:

1. Generate embeddings for the images by feeding them into the `vision_encoder`.
2. Feed the natural language query to the `text_encoder` to generate a query embedding.
3. Compute the similarity between the query embedding and the image embeddings
in the index to retrieve the indices of the top matches.
4. Look up the paths of the top matching images to display them.

Note that, after training the `dual encoder`, only the fine-tuned `vision_encoder`
and `text_encoder` models will be used, while the `dual_encoder` model will be discarded.

### Generate embeddings for the images

We load the images and feed them into the `vision_encoder` to generate their embeddings.
In large scale systems, this step is performed using a parallel data processing framework,
such as [Apache Spark](https://spark.apache.org) or [Apache Beam](https://beam.apache.org).
Generating the image embeddings may take several minutes.
"""

print("Loading vision and text encoders...")
vision_encoder = keras.models.load_model("vision_encoder.keras")
text_encoder = keras.models.load_model("text_encoder.keras")
print("Models are loaded.")


def read_image(path):
    img = load_img(path, target_size=(299, 299))
    img = img_to_array(img)
    return (img / 127.5) - 1.0


print(f"Generating embeddings for images...")
image_embeddings = vision_encoder.predict(
    np.array([read_image(p) for p in image_paths[:2000]]),
    batch_size=batch_size,
)
print(f"Image embeddings shape: {image_embeddings.shape}.")

"""
### Retrieve relevant images

In this example, we use exact matching by computing the dot product similarity
between the input query embedding and the image embeddings, and retrieve the top k
matches. However, *approximate* similarity matching, using frameworks like
[ScaNN](https://github.com/google-research/google-research/tree/master/scann),
[Annoy](https://github.com/spotify/annoy), or [Faiss](https://github.com/facebookresearch/faiss)
is preferred in real-time use cases to scale with a large number of images.
"""


def find_matches(image_embeddings, queries, k=9):
    tokenized = bert_preprocessor(np.array(queries))
    # Get the embedding for the query.
    query_embedding = text_encoder.predict(
        [tokenized["token_ids"], tokenized["padding_mask"], tokenized["segment_ids"]],
        verbose=0,
    )
    # Normalize the query and the image embeddings.
    query_embedding = query_embedding / (
        np.linalg.norm(query_embedding, axis=-1, keepdims=True) + 1e-12
    )
    # Compute the dot product between the query and the image embeddings.
    dot_similarity = np.dot(query_embedding, image_embeddings.T)

    image_embeddings = image_embeddings / (
        np.linalg.norm(image_embeddings, axis=-1, keepdims=True) + 1e-12
    )
    # Compute the dot product between the query and the image embeddings.
    dot_similarity = np.dot(query_embedding, image_embeddings.T)
    # Retrieve top k indices.
    results = np.argsort(dot_similarity, axis=-1)[:, ::-1][:, :k]
    # Return matching image paths.
    return [[image_paths[i] for i in row] for row in results]


"""
## Set the `query` variable to the type of images you want to search for.
Try things like: 'a plate of healthy food',
'a woman wearing a hat is walking down a sidewalk',
'a bird sits near to the water', or 'wild animals are standing in a field'.
"""

query = "a family standing next to the ocean on a sandy beach with a surf board"
matches = find_matches(image_embeddings, [query])[0]
plt.figure(figsize=(15, 15))
for i, path in enumerate(matches):
    plt.subplot(3, 3, i + 1)
    plt.imshow(mpimg.imread(path))
    plt.axis("off")
plt.show()

"""
## Evaluate the retrieval quality

To evaluate the dual encoder model, we use the captions as queries.
We use the out-of-training-sample images and captions to evaluate the retrieval quality,
using top k accuracy. A true prediction is counted if, for a given caption, its associated image
is retrieved within the top k matches.
"""


def compute_top_k_accuracy(image_paths_to_eval, k=100):
    hits = 0
    num_batches = int(np.ceil(len(image_paths_to_eval) / batch_size))

    for idx in tqdm(range(num_batches)):
        start_idx = idx * batch_size
        end_idx = start_idx + batch_size
        current_image_paths = image_paths_to_eval[start_idx:end_idx]

        # Extract the first caption for each image in the current batch
        queries = [image_path_to_caption[path][0] for path in current_image_paths]

        result_paths = find_matches(image_embeddings, queries, k=k)
        for i, path in enumerate(current_image_paths):
            if path in result_paths[i]:
                hits += 1

    return hits / len(image_paths_to_eval)


print("Scoring training data...")
train_accuracy = compute_top_k_accuracy(train_image_paths[: len(image_embeddings)])
print(f"Train accuracy: {round(train_accuracy * 100, 3)}%")

print("Scoring evaluation data...")
eval_accuracy = compute_top_k_accuracy(valid_image_paths[:1000])
print(f"Eval accuracy: {round(eval_accuracy * 100, 3)}%")

"""
## Final remarks

You can obtain better results by increasing the size of the training sample,
train for more  epochs, explore other base encoders for images and text,
set the base encoders to be trainable, and tune the hyperparameters,
especially the `temperature` for the softmax in the loss computation.

Example available on HuggingFace

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-nl%20image%20search-black.svg)](https://huggingface.co/keras-io/dual-encoder-image-search) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-nl%20image%20search-black.svg)](https://huggingface.co/spaces/keras-io/dual-encoder-image-search) |

"""

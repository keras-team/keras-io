# Natural language image search with a Dual Encoder

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/01/30<br>
**Last modified:** 2026/05/24<br>
**Description:** Implementation of a dual encoder model for retrieving images that match natural language queries.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/nl_image_search.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/nl_image_search.py)



---
## Introduction

The example demonstrates how to build a dual encoder (also known as two-tower) neural network
model to search for images using natural language. The model is inspired by
the [CLIP](https://openai.com/blog/clip/) approach introduced by Alec Radford et al.

The idea is to train a vision encoder and a text encoder jointly to project the
representation of images and their captions into the same embedding space, such that
caption embeddings are located near the embeddings of the images they describe.

---
## Setup


```python
import os
import json

from tqdm import tqdm
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import keras
from keras import layers
from keras import ops
from keras.utils import PyDataset
from keras.utils import load_img
from keras.utils import img_to_array
import keras_hub
```

---
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


```python

def download_and_extract(url, fname, cache_dir):
    return keras.utils.get_file(
        fname,
        origin=url,
        extract=True,
        cache_dir=cache_dir,
    )


def find_dir(root, target_name):
    for dirpath, dirnames, _ in os.walk(root):
        if target_name in dirnames:
            return os.path.join(dirpath, target_name)
    return None


def find_file(root, target_name):
    for dirpath, _, filenames in os.walk(root):
        if target_name in filenames:
            return os.path.join(dirpath, target_name)
    return None


root_dir = os.path.abspath(".")
extract_path = os.path.join(root_dir, "captions_extracted")

annotation_zip_path = keras.utils.get_file(
    fname="annotations_trainval2014.zip",
    origin="http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    extract=True,
    force_download=True,
)

base_dir = os.path.dirname(annotation_zip_path)

image_zip_path = keras.utils.get_file(
    fname="train2014.zip",
    origin="http://images.cocodataset.org/zips/train2014.zip",
    extract=True,
    force_download=False,
)

image_base_dir = os.path.dirname(image_zip_path)

annotation_json = find_file(base_dir, "captions_train2014.json")
actual_image_folder = find_dir(image_base_dir, "train2014")

if annotation_json is None:
    raise FileNotFoundError("captions_train2014.json not found after extraction")

if actual_image_folder is None:
    raise FileNotFoundError("train2014 folder not found after extraction")

print("Processing data...")

with open(annotation_json, "r") as f:
    annotations = json.load(f)["annotations"]

image_path_to_caption = collections.defaultdict(list)

for ann in annotations:
    caption = ann["caption"].lower().rstrip(".")
    image_path = os.path.join(
        actual_image_folder,
        f"COCO_train2014_{ann['image_id']:012d}.jpg",
    )
    image_path_to_caption[image_path].append(caption)

image_paths = list(image_path_to_caption.keys())
print(f"Number of images indexed: {len(image_paths)}")

```

<div class="k-default-codeblock">

Downloading data from http://images.cocodataset.org/annotations/annotations_trainval2014.zip

252872794/252872794 ━━━━━━━━━━━━━━━━━━━━ 53s 0us/step

Processing data...

Number of images indexed: 82783 
</div>

##
Process and save the data

You can change the `sample_size` parameter to control how many image-caption pairs
will be used for training the dual encoder model.
In this example we set `train_size` to 30,000 images,
which is about 35% of the dataset. We use 2 captions for each
image, thus producing 60,000 image-caption pairs. The size of the training set
affects the quality of the produced encoders, but more examples would lead to
longer training time.


```python
train_size = 30000
valid_size = 5000
train_image_paths = image_paths[:train_size]
valid_image_paths = image_paths[train_size : train_size + valid_size]


bert_preprocessor = keras_hub.models.BertPreprocessor.from_preset(
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

```

---
## Implement the projection head

The projection head is used to transform the image and the text embeddings to
the same embedding space with the same dimensionality.


```python

def project_embeddings(x, num_layers, dim, dropout):
    x = layers.Dense(dim)(x)
    for _ in range(num_layers):
        residual = x
        x = layers.Activation("gelu")(x)
        x = layers.Dense(dim)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Add()([residual, x])
        x = layers.LayerNormalization()(x)
    return x

```

---
## Implement the vision encoder

In this example, we use [Xception](https://keras.io/api/applications/xception/)
from [Keras Applications](https://keras.io/api/applications/) as the base for the
vision encoder.


```python

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

```

---
## Implement the text encoder
We use [BERT](bert_small_en_uncased)
from [KerasHUB](https://keras.io/keras_hub/api/models/bert/bert_backbone/) as the text encoder


```python

def create_text_encoder(num_layers, dim, dropout, trainable=False):

    # Load the pre-trained BERT model to be used as the base encoder.
    bert = keras_hub.models.BertBackbone.from_preset("bert_small_en_uncased")

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

```

---
## Implement the dual encoder

To calculate the loss, we compute the pairwise dot-product similarity between
each `caption_i` and `images_j` in the batch as the predictions.
The target similarity between `caption_i`  and `image_j` is computed as
the average of the (dot-product similarity between `caption_i` and `caption_j`)
and (the dot-product similarity between `image_i` and `image_j`).
Then, we use crossentropy to compute the loss between the targets and the predictions.


```python

class DualEncoder(keras.Model):
    def __init__(self, text_encoder, vision_encoder, temperature=0.05, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        if isinstance(y_pred, dict):
            text_emb = y_pred["text_emb"]
            vision_emb = y_pred["vision_emb"]
        else:
            text_emb, vision_emb = y_pred

        # Logic for soft targets
        caption_similarity = ops.matmul(text_emb, ops.transpose(text_emb))
        image_similarity = ops.matmul(vision_emb, ops.transpose(vision_emb))

        targets = (caption_similarity + image_similarity) / 2.0
        targets = ops.softmax(targets, axis=-1)

        # Contrastive Loss
        logits = ops.matmul(text_emb, ops.transpose(vision_emb)) / self.temperature

        # Symmetric Cross-Entropy Loss
        loss_txt = keras.losses.categorical_crossentropy(
            targets, logits, from_logits=True
        )
        loss_img = keras.losses.categorical_crossentropy(
            ops.transpose(targets), ops.transpose(logits), from_logits=True
        )
        loss = ops.mean((loss_txt + loss_img) / 2.0)

        # Track the progress
        self.loss_tracker.update_state(loss)

        return loss

    def call(self, x, training=False):
        text_emb = self.text_encoder(
            [x["token_ids"], x["padding_mask"], x["segment_ids"]], training=training
        )
        vision_emb = self.vision_encoder(x["image"], training=training)

        text_emb = ops.divide(
            text_emb, ops.norm(text_emb, axis=-1, keepdims=True) + 1e-12
        )
        vision_emb = ops.divide(
            vision_emb, ops.norm(vision_emb, axis=-1, keepdims=True) + 1e-12
        )

        return {
            "text_emb": text_emb,
            "vision_emb": vision_emb,
        }

    @property
    def metrics(self):
        return [self.loss_tracker]

```

---
## Train the dual encoder model

In this experiment, we freeze the base encoders for text and images, and make only
the projection head trainable.


```python
num_epochs = 2  # In practice, train for at least 30 epochs
batch_size = 256
vision_encoder = create_vision_encoder(1, 256, 0.1)
text_encoder = create_text_encoder(1, 256, 0.1)
dual_encoder = DualEncoder(text_encoder, vision_encoder)

dual_encoder.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-3)
)
```

---
## Note that training the model with 60,000 image-caption pairs, with a batch size of 256,
takes around 12 minutes per epoch using a V100 GPU accelerator. If 2 GPUs are available,
the epoch takes around 8 minutes.


```python
print(f"Batch size: {batch_size}")

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
```

<div class="k-default-codeblock">
```
Batch size: 256

Epoch 1/2

235/235 ━━━━━━━━━━━━━━━━━━━━ 10975s 47s/step - loss: 5.5575 - val_loss: 5.4761 - learning_rate: 0.0010

Epoch 2/2

235/235 ━━━━━━━━━━━━━━━━━━━━ 19171s 82s/step - loss: 5.5418 - val_loss: 5.4760 - learning_rate: 0.0010

Training completed. Saving vision and text encoders...

Models are saved.
```
</div>

---
## Plotting the training loss:


```python
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train", "valid"], loc="upper right")
plt.show()
```


    
![png](/img/examples/vision/nl_image_search/nl_image_search_23_0.png)
    


---
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


```python
print("Loading vision and text encoders...")
vision_encoder = keras.models.load_model("vision_encoder.keras")
text_encoder = keras.models.load_model("text_encoder.keras")
print("Models are loaded.")


def read_image(path):
    img = load_img(path, target_size=(299, 299))
    img = img_to_array(img)
    return (img / 127.5) - 1.0


print(f"Generating embeddings for images...")


def image_generator(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = np.stack([read_image(p) for p in batch_paths])
        yield (batch_images,)


# Use the first 2000
target_paths = image_paths[:2000]

gen = image_generator(target_paths, batch_size)
image_embeddings = vision_encoder.predict(
    gen,
    # steps= (total_images // batch_size)
    steps=(len(target_paths) + batch_size - 1) // batch_size,
    verbose=1,
)
print(f"Image embeddings shape: {image_embeddings.shape}.")
```

<div class="k-default-codeblock">
```
Loading vision and text encoders...

Models are loaded.
Generating embeddings for images...

8/8 ━━━━━━━━━━━━━━━━━━━━ 132s 16s/step

Image embeddings shape: (2000, 256).
```
</div>

### Retrieve relevant images

In this example, we use exact matching by computing the dot product similarity
between the input query embedding and the image embeddings, and retrieve the top k
matches. However, *approximate* similarity matching, using frameworks like
[ScaNN](https://github.com/google-research/google-research/tree/master/scann),
[Annoy](https://github.com/spotify/annoy), or [Faiss](https://github.com/facebookresearch/faiss)
is preferred in real-time use cases to scale with a large number of images.


```python

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
    image_embeddings = image_embeddings / (
        np.linalg.norm(image_embeddings, axis=-1, keepdims=True) + 1e-12
    )
    # Compute the dot product between the query and the image embeddings.
    dot_similarity = np.dot(query_embedding, image_embeddings.T)
    # Retrieve top k indices.
    results = np.argsort(dot_similarity, axis=-1)[:, ::-1][:, :k]
    # Return matching image paths.
    return [[image_paths[idx] for idx in indices] for indices in results]

```

Set the `query` variable to the type of images you want to search for.
Try things like: 'a plate of healthy food',
'a woman wearing a hat is walking down a sidewalk',
'a bird sits near to the water', or 'wild animals are standing in a field'.


```python
query = "a family standing next to the ocean on a sandy beach with a surf board"
matches = find_matches(image_embeddings, [query])[0]
plt.figure(figsize=(15, 15))
for i, path in enumerate(matches):
    plt.subplot(3, 3, i + 1)
    plt.imshow(mpimg.imread(path))
    plt.axis("off")
plt.show()
```


    
![png](/img/examples/vision/nl_image_search/nl_image_search_30_0.png)
    


---
## Evaluate the retrieval quality

To evaluate the dual encoder model, we use the captions as queries.
We use the out-of-training-sample images and captions to evaluate the retrieval quality,
using top k accuracy. A true prediction is counted if, for a given caption, its associated image
is retrieved within the top k matches.


```python

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
```

<div class="k-default-codeblock">
```
Scoring training data...
```
</div>

  0%|                                                                                                                             | 0/8 [00:00<?, ?it/s]

    
 12%|██████████████▋                                                                                                      | 1/8 [00:19<02:16, 19.44s/it]

    
 25%|█████████████████████████████▎                                                                                       | 2/8 [00:38<01:57, 19.51s/it]

    
 38%|███████████████████████████████████████████▉                                                                         | 3/8 [00:58<01:37, 19.48s/it]

    
 50%|██████████████████████████████████████████████████████████▌                                                          | 4/8 [01:17<01:17, 19.46s/it]

    
 62%|█████████████████████████████████████████████████████████████████████████▏                                           | 5/8 [01:37<00:58, 19.45s/it]

    
 75%|███████████████████████████████████████████████████████████████████████████████████████▊                             | 6/8 [01:56<00:38, 19.45s/it]

    
 88%|██████████████████████████████████████████████████████████████████████████████████████████████████████▍              | 7/8 [02:16<00:19, 19.47s/it]

    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:31<00:00, 18.28s/it]

    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:31<00:00, 19.00s/it]

    


<div class="k-default-codeblock">
```
Train accuracy: 5.1%
Scoring evaluation data...
```
</div>

  0%|                                                                                                                             | 0/4 [00:00<?, ?it/s]

    
 25%|█████████████████████████████▎                                                                                       | 1/4 [00:19<00:58, 19.37s/it]

    
 50%|██████████████████████████████████████████████████████████▌                                                          | 2/4 [00:38<00:38, 19.36s/it]

    
 75%|███████████████████████████████████████████████████████████████████████████████████████▊                             | 3/4 [00:58<00:19, 19.34s/it]

    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [01:15<00:00, 18.59s/it]

    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [01:15<00:00, 18.87s/it]

<div class="k-default-codeblock">
```
Eval accuracy: 0.0%
```
</div>

---
## Final remarks

You can obtain better results by increasing the size of the training sample,
train for more  epochs, explore other base encoders for images and text,
set the base encoders to be trainable, and tune the hyperparameters,
especially the `temperature` for the softmax in the loss computation.

Example available on HuggingFace

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-nl%20image%20search-black.svg)](https://huggingface.co/keras-io/dual-encoder-image-search) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-nl%20image%20search-black.svg)](https://huggingface.co/spaces/keras-io/dual-encoder-image-search) |

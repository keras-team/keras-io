# Context-Aware Music Retrieval with Yambda and JAX Data Parallelism

**Author:** [Rishiraj Acharya](https://github.com/rishiraj)<br>
**Date created:** 2026/04/01<br>
**Last modified:** 2026/04/01<br>
**Description:** Context-aware music retrieval model trained on TPU with JAX data parallelism using Yambda dataset.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_rs/ipynb/contextual_retrieval_yambda.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_rs/contextual_retrieval_yambda.py)



---
## Introduction

Standard retrieval models learn a static embedding for each user. However, user behavior in the real world is highly context-dependent. A user might want completely different music tracks when they are actively browsing and searching (an *organic* context) versus when they are leaning back and listening to an algorithmic radio station (a *recommendation-driven* context).

If we treat all interactions equally, the model will blur these distinct preferences into a single average representation. Instead, we can build a **Context-Aware Retrieval Model** that dynamically shifts the user's representation based on the context of the interaction.

In this tutorial, we will tackle this unique problem using the [**Yambda-5B**](https://huggingface.co/datasets/yandex/yambda) dataset from Hugging Face. Yambda is an industrial-scale music recommendation dataset containing billions of interactions. Crucially, it provides a distinctive `is_organic` flag for every event.

We will:
1. Use Hugging Face `datasets` to stream and filter a subset of the Yambda-50M dataset.
2. Compress the massive sparse ID space to prevent TPU Out-Of-Memory (OOM) errors.
3. Build a Two-Tower Retrieval model where the Query Tower dynamically fuses the User ID embedding and the `is_organic` context embedding.
4. Train the model using **In-Batch Negative Sampling** while avoiding the "Curse of Scale" during validation.
5. Scale our training on a TPU using the JAX backend and `keras.distribution.DataParallel`.

Let's begin by installing the necessary libraries and configuring the JAX backend.


```python
!pip install -q -U jax[tpu]>=0.7.0
!pip install -q tensorflow-cpu
!pip install -q keras-rs datasets
```

```python
import os

# 💡 PRO TIP: Set the Keras backend BEFORE importing keras or any of its submodules.
# We select JAX as our backend to use the robust, out-of-the-box TPU distribution support.
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import keras_rs
import numpy as np
import tensorflow as tf
from datasets import load_dataset
```

---
## TPU Distribution Strategy

To train efficiently on a TPU (such as a v5e-1 node in Google Colab), we will use synchronous data parallelism.
With `keras.distribution.DataParallel`, each TPU core holds a complete replica of the model weights and processes a different mini-batch of the data.

Gradients are calculated locally on each device and then synchronized across all devices before updating the global model weights. This allows us to scale our batch size linearly with the number of available TPU cores.


```python
# Detect available JAX devices (TPU cores)
devices = jax.devices()
print(f"Found {len(devices)} JAX devices.")

# Initialize the DataParallel strategy and set it globally
data_parallel = keras.distribution.DataParallel(devices=devices)
keras.distribution.set_distribution(data_parallel)
```

<div class="k-default-codeblock">
```
Found 8 JAX devices.
```
</div>

---
## Loading and Filtering the Yambda Dataset

The Yambda dataset contains multiple event types: `listen`, `like`, `dislike`, `unlike`, and `undislike`. To train a retrieval model, we only want to learn from **positive** interactions.

According to the Yambda documentation, an implicit `listen` is considered a positive signal if the user played more than 50% of the track (`played_ratio_pct > 50`). An explicit `like` (or reverting a dislike via `undislike`) is also a strong positive signal.

To ensure this example runs quickly on a Colab instance while still providing enough data to learn meaningful embeddings, we will load a 5% slice of the 50M interactions split.


```python
print("Downloading Yambda dataset subset...")
dataset = load_dataset(
    "yandex/yambda",
    data_dir="flat/50m",
    data_files="multi_event.parquet",
    split="train[:5%]",
)


def filter_positive_events(batch):
    """Filters out dislikes, unlikes, and short listens."""
    is_positive = []
    for event, pct in zip(batch["event_type"], batch["played_ratio_pct"]):
        if event in ["like", "undislike"]:
            is_positive.append(True)
        elif event == "listen" and pct is not None and pct >= 50:
            is_positive.append(True)
        else:
            is_positive.append(False)
    return is_positive


print("Filtering positive interactions...")
# Apply the filter in batches for faster processing
positive_dataset = dataset.filter(
    filter_positive_events, batched=True, batch_size=10_000
)
print(f"Remaining positive interactions: {len(positive_dataset)}")
```

<div class="k-default-codeblock">
```
Downloading Yambda dataset subset...

README.md: 0.00B [00:00, ?B/s]

flat/50m/multi_event.parquet:   0%|          | 0.00/384M [00:00<?, ?B/s]

Generating train split: 0 examples [00:00, ? examples/s]

Filtering positive interactions...

Filter:   0%|          | 0/2389522 [00:00<?, ? examples/s]

Remaining positive interactions: 1492025
```
</div>

---
## Vocabulary Compression: Avoiding the XLA HBM OOM Error

In the Yambda dataset, `item_id`s go all the way up to ~9.39 million.
If we initialize an embedding table with 9.4 million rows on a TPU, JAX's XLA compiler pads these tensors for matrix multiplication. The table, the optimizer states, and the XLA padding combined will demand over 15 GB of High-Bandwidth Memory (HBM), instantly causing a `RESOURCE_EXHAUSTED` crash on a standard TPU core.

💡 **PRO TIP: Contiguous Mapping**
In industrial recommender systems, we never use raw, sparse IDs directly in an embedding table. Our 5% subset only contains a fraction of the total catalogue. We extract the unique items and map them to a dense, contiguous range (e.g., `0` to `~150000`). This shrinks the embedding table by 98%, requiring only megabytes of RAM instead of gigabytes!


```python
print("Extracting unique items for vocabulary compression...")
unique_items = np.unique(positive_dataset["item_id"])
item_vocab_size = len(unique_items) + 1  # +1 for Out-Of-Vocabulary (OOV) token

max_uid = max(positive_dataset["uid"])
print(f"Max User ID: {max_uid}")
print(
    f"Unique items in subset: {len(unique_items)}. Compressed vocab size: {item_vocab_size}"
)

# Create a Keras lookup layer to map sparse IDs to contiguous integers (0 to item_vocab_size)
item_lookup = keras.layers.IntegerLookup(vocabulary=unique_items, mask_token=None)
```

<div class="k-default-codeblock">
```
Extracting unique items for vocabulary compression...

Max User ID: 52500
Unique items in subset: 138101. Compressed vocab size: 138102
```
</div>

---
## Preparing a High-Performance tf.data Pipeline

When working with industrial-scale datasets in Hugging Face, loading millions of rows into RAM and passing them to `tf.data.Dataset.from_tensor_slices()` is highly inefficient.

Instead, we use Hugging Face's `to_tf_dataset()`. This method creates a highly efficient bridge between the underlying Apache Arrow files and TensorFlow's execution graph, dynamically batching data directly from disk.

💡 **PRO TIP: XLA Compilation and Static Shapes**
When training on TPUs (which rely on XLA compilation via JAX), tensor shapes must remain strictly static. If the final batch of your dataset is smaller than the rest, XLA will trigger a costly recompilation step. We enforce static shapes by explicitly setting `drop_remainder=True`.


```python
# Define local batch size per TPU core and global batch size
PER_REPLICA_BATCH_SIZE = 1024
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * len(devices)


def build_tf_dataset(hf_dataset):
    # Convert efficiently using Hugging Face's built-in to_tf_dataset
    tf_ds = hf_dataset.to_tf_dataset(
        columns=["uid", "item_id", "is_organic"],
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle=True,
        drop_remainder=True,
    )

    # Format the inputs for the Two-Tower model
    def format_features(batch):
        return (
            {
                "uid": tf.cast(batch["uid"], tf.int32),
                "is_organic": tf.cast(batch["is_organic"], tf.int32),
            },
            {
                # Apply the lookup layer to compress the ID space!
                "item_id": item_lookup(tf.cast(batch["item_id"], tf.int64))
            },
        )

    tf_ds = tf_ds.map(format_features, num_parallel_calls=tf.data.AUTOTUNE)
    tf_ds = tf_ds.cache().prefetch(tf.data.AUTOTUNE)
    return tf_ds


full_tf_ds = build_tf_dataset(positive_dataset)

# Perform an 80-20 Train/Validation Split
total_batches = len(positive_dataset) // GLOBAL_BATCH_SIZE
train_batches = int(total_batches * 0.8)

train_ds = full_tf_ds.take(train_batches)
val_ds = full_tf_ds.skip(train_batches)

print(
    f"Dataset split: {train_batches} Training Batches, {total_batches - train_batches} Validation Batches."
)
```

<div class="k-default-codeblock">
```
Dataset split: 145 Training Batches, 37 Validation Batches.
```
</div>

---
## Building the Context-Aware Model Architecture

We construct a Two-Tower Retrieval model, but we augment the **Query Tower** to handle contextual flags.

1. **User Embedding**: Captures the global, long-term preference of the user.
2. **Context Embedding**: Maps the `is_organic` boolean (0 or 1) to a dense vector.
3. **Query Fusion**: Concatenates the user and context vectors and passes them through a Dense layer to project them back into the shared semantic space.
4. **Candidate Tower**: Standard item embedding.

### The Curse of Scale: Why We Decoupled Retrieval

In earlier examples on tiny datasets (like MovieLens 100K), you might see the `BruteForceRetrieval` layer called directly inside the model's `call()` method during `training=False`.

However, if you attempt to run a full-corpus Top-10 retrieval against hundreds of thousands (or millions) of items during the validation loop, your TPU will attempt to calculate and sort trillions of affinity scores per epoch, causing the training to freeze entirely.

💡 **PRO TIP: Production Architecture**
In production, we *never* compute full-corpus retrieval during the train/val loops. We strictly evaluate the **In-Batch Contrastive Loss**. The massive retrieval layer is decoupled and reserved strictly for the final inference step (or exported to a vector database like ScANN). Notice how our `call()` method below *only* returns embeddings!


```python

class ContextualRetrievalModel(keras.Model):
    def __init__(self, max_uid, item_vocab_size, embed_dim=32, **kwargs):
        super().__init__(**kwargs)

        # Query Tower features
        self.user_embedding = keras.layers.Embedding(max_uid + 1, embed_dim)
        self.context_embedding = keras.layers.Embedding(2, embed_dim)
        self.query_dense = keras.layers.Dense(embed_dim, activation="relu")

        # Candidate Tower features
        self.item_embedding = keras.layers.Embedding(item_vocab_size, embed_dim)

        # Retrieval layer - We will call this MANUALLY during inference only!
        self.retrieval = keras_rs.layers.BruteForceRetrieval(k=10, return_scores=False)

        # Loss tracking
        self.loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def build(self, input_shape):
        # 1. Build the candidate item embedding layer manually.
        # Why? Because it's only used in compute_loss(), not in call(),
        # so super().build() won't automatically build it!
        self.item_embedding.build((None,))

        # 2. Now it is safe to access the embeddings matrix
        self.retrieval.candidate_embeddings = self.item_embedding.embeddings

        # 3. Build the retrieval layer expecting the query embedding shape
        self.retrieval.build((None, self.query_dense.units))

        # 4. Let Keras build the rest of the model (User, Context, Dense) automatically
        super().build(input_shape)

    def call(self, inputs):
        # 1. Extract and embed query inputs
        user_emb = self.user_embedding(inputs["uid"])
        context_emb = self.context_embedding(inputs["is_organic"])

        # 2. Fuse user and context representation
        x = keras.ops.concatenate([user_emb, context_emb], axis=1)
        query_embeddings = self.query_dense(x)

        # NOTE: We ONLY return the embeddings here.
        # We completely bypass BruteForceRetrieval to prevent validation hangs.
        return {"query_embeddings": query_embeddings}

    def compute_loss(self, x, y, y_pred, sample_weight, training=True):
        query_embeddings = y_pred["query_embeddings"]
        candidate_embeddings = self.item_embedding(y["item_id"])

        # Compute in-batch affinity scores (batch_size, batch_size)
        scores = keras.ops.matmul(
            query_embeddings, keras.ops.transpose(candidate_embeddings)
        )

        # True labels are on the diagonal
        batch_size = keras.ops.shape(query_embeddings)[0]
        labels = keras.ops.eye(batch_size)

        loss = self.loss_fn(labels, scores, sample_weight)
        self.loss_tracker.update_state(loss)
        return loss

    @property
    def metrics(self):
        # Only track the in-batch loss during fit/evaluate
        return [self.loss_tracker]

```

---
## Training the Model

We instantiate the model, compile it, and fit it using our prepared JAX DataParallel datasets.

💡 **PRO TIP: Why Adagrad?**
We use the `Adagrad` optimizer instead of Adam here. In recommender systems, categorical embeddings are highly sparse—popular items get updated millions of times, while niche items get updated rarely. Adagrad excels in this scenario because it dynamically scales the learning rate per parameter based on historical gradients, preventing popular items from overwhelming the learning process.


```python
# Instantiate and compile
model = ContextualRetrievalModel(
    max_uid=max_uid, item_vocab_size=item_vocab_size, embed_dim=64
)
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))

print("Starting distributed training...")
history = model.fit(train_ds, validation_data=val_ds, epochs=5)
```

<div class="k-default-codeblock">
```
Starting distributed training...

Epoch 1/5

145/145 ━━━━━━━━━━━━━━━━━━━━ 196s 1s/step - loss: 9.0109 - val_loss: 9.0109

Epoch 2/5

145/145 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 9.0109 - val_loss: 9.0108

Epoch 3/5

145/145 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 9.0108 - val_loss: 9.0108

Epoch 4/5

145/145 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 9.0108 - val_loss: 9.0108

Epoch 5/5

145/145 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 9.0107 - val_loss: 9.0107
```
</div>

---
## Contextual Inference

Now that the model is trained, we want to perform actual retrieval. Because we decoupled the retrieval index from the training forward pass, we execute this in two clean steps. This mirrors exactly how two-tower models are served in production architectures (e.g., Vertex AI Matching Engine):

1. **Query encoding**: Compute the `query_embeddings`.
2. **Approximate Nearest Neighbor (ANN)**: Pass the `query_embeddings` directly to the index (`model.retrieval`) to fetch candidates.

Let's simulate querying the model for a single user in two different settings:
*   `is_organic = 1`: The user is actively searching or exploring their own library.
*   `is_organic = 0`: The user is passively listening to an algorithmic radio.


```python
test_user_id = 1001

print(f"\nComputing recommendations for User {test_user_id}...")

# 1. Organic Context (is_organic = 1)
query_out_organic = model(
    {"uid": keras.ops.array([test_user_id]), "is_organic": keras.ops.array([1])}
)
# Manually invoke the retrieval index
internal_organic_tracks = keras.ops.convert_to_numpy(
    model.retrieval(query_out_organic["query_embeddings"])[0]
)

# 2. Algorithmic Context (is_organic = 0)
query_out_algo = model(
    {"uid": keras.ops.array([test_user_id]), "is_organic": keras.ops.array([0])}
)
internal_algo_tracks = keras.ops.convert_to_numpy(
    model.retrieval(query_out_algo["query_embeddings"])[0]
)
```

    
<div class="k-default-codeblock">
```
Computing recommendations for User 1001...
```
</div>

### Reversing the Lookup Mapping
Because the model was trained on our *internal* contiguous IDs, the `BruteForceRetrieval` outputs internal indices. To get the actual Yambda track IDs back (so we can look them up in a database), we reverse the mapping using our vocabulary array.


```python
# We add a 0 to the beginning of the vocab array to account for the OOV token at index 0
vocab_array = np.concatenate([[0], unique_items])

organic_tracks_yambda_ids = vocab_array[internal_organic_tracks]
algo_tracks_yambda_ids = vocab_array[internal_algo_tracks]

print(f"Organic Context Yambda IDs:     {organic_tracks_yambda_ids}")
print(f"Algorithmic Context Yambda IDs: {algo_tracks_yambda_ids}")
```

<div class="k-default-codeblock">
```
Organic Context Yambda IDs:     [9016778 8060425 6123367 6123431 3729063 5133519 2237560 6882660 6983872
 5390371]
Algorithmic Context Yambda IDs: [3344425 4461258 3323941  658719 8431135 6614565 3928878 6355893 1371572
 5459471]
```
</div>

Notice how the predicted item arrays differ based on the context flag!

By injecting contextual data directly into the query tower, your retrieval pipeline is no longer static. You can easily extend this pattern to include other context parameters, such as the time of day, device type, or even pre-computed multimodal representations (like the audio embeddings provided natively in the Yambda dataset!).

*(Note: Even for a single user, computing the Top-10 against millions of tracks using BruteForce takes a moment. To achieve millisecond latency in production, you would export `model.item_embedding.embeddings` and index them using `ScANN`, as demonstrated in the KerasRS ScANN tutorial!)*

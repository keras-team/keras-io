"""
Title: Text Classification with Gemma 4
Author: [Laxma Reddy Patlolla](https://github.com/laxmareddyp)
Date created: 2026/04/12
Last modified: 2026/04/12
Description: Fine-tune Gemma 4 for text classification using feature extraction, full fine-tuning, LoRA, and quantization.
Accelerator: GPU
"""

"""
## Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across
natural language processing tasks. This guide shows how to adapt
[Gemma 4](https://ai.google.dev/gemma), Google's latest open model family, for
binary text classification on the IMDB sentiment analysis dataset.

Gemma 4 is a significant upgrade over Gemma 3, introducing multimodal
understanding (vision + text, with audio on smaller models), Per-Layer
Embeddings (PLE) for parameter efficiency, Mixture-of-Experts (MoE)
variants, and improved reasoning capabilities — all under the Apache 2.0
license.

We explore a natural progression of techniques:

- **Feature extraction**: Freeze the backbone and train only the classifier
  head.
- **Full fine-tuning**: Unfreeze the entire model for maximum accuracy.
- **LoRA fine-tuning**: The recommended approach — parameter-efficient
  adaptation using [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685).
- **Post-training quantization**: Compress the model for deployment.
- **LoRA + Quantization**: The recommended production pipeline — fine-tune
  first, then quantize.
- **Quantization-Aware Training (QAT)**: Fine-tune a quantized model to
  recover accuracy lost during quantization.

By the end, you will have a complete pipeline from training to
production-optimized deployment.
"""

"""
## Gemma 4 Model Family

Before diving in, here is a quick overview of the Gemma 4 presets
available in KerasHub:

| Preset | Params | Architecture |
|:---|:---|:---|
| `gemma4_26b_a4b` | 26B total, 4B active | 30-layer, MoE |
| `gemma4_instruct_26b_a4b` | 26B total, 4B active | 30-layer, MoE |
| `gemma4_31b` | 31B | 60-layer, dense |
| `gemma4_instruct_31b` | 31B | 60-layer, dense |

The `_instruct` variants are instruction-tuned for conversational
and instruction-following tasks.

The `gemma4_26b_a4b` model uses a **Mixture-of-Experts (MoE)**
architecture: it has 26B total parameters but activates only 4B
per forward pass, running nearly as fast as a dense 4B model while
delivering the quality of a much larger one.

In this guide, we use `gemma4_26b_a4b` — the MoE variant with only
4B active parameters per token, making it the most resource-efficient
choice for fine-tuning demonstrations.
"""

"""
## Setup

Before we start, let's install and import the necessary libraries. We use
KerasHub for the Gemma 4 model and TensorFlow Datasets for the IMDB dataset.

Ensure that `KAGGLE_USERNAME` and `KAGGLE_KEY` are configured to download the
Gemma 4 weights from Kaggle.
"""

"""shell
pip install -q --upgrade keras keras-hub tensorflow-datasets
"""

import os
from google.colab import userdata

os.environ["KERAS_BACKEND"] = "tensorflow"

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")

import keras
from keras import ops
import keras_hub
import numpy as np

# TensorFlow is used only for tf.data pipelines and tfds data loading.
# All model computation uses keras.ops for backend compatibility.
import tensorflow as tf
import tensorflow_datasets as tfds

keras.config.set_dtype_policy("bfloat16")

"""
## Load Dataset

We use the IMDB movie review dataset, a standard binary sentiment
classification benchmark. Each review is labeled as positive (1) or
negative (0).

We take a small subset for fast iteration. For production-quality results,
use the full dataset and train for more epochs.
"""

NUM_TRAIN = 2000
NUM_TEST = 500
BATCH_SIZE = 4
MAX_SEQUENCE_LENGTH = 128
EPOCHS = 10

# Store results as we go for progress tracking.
results = {}

imdb_data = tfds.load("imdb_reviews", as_supervised=True)
train_data = imdb_data["train"].take(NUM_TRAIN)
test_data = imdb_data["test"].take(NUM_TEST)

"""
Let's inspect a few samples.
"""

for text, label in train_data.take(3):
    sentiment = "Positive" if label.numpy() == 1 else "Negative"
    print(f"[{sentiment}] {text.numpy()[:100]}...")
    print()

"""
## Preprocessing

For classification, we tokenize the text using `Gemma4Tokenizer` directly
(rather than the `CausalLMPreprocessor`, which is designed for
prompt/response language modeling). We handle padding, truncation, and
mask creation ourselves.

We set `sequence_length=128` to keep memory usage manageable.
"""

GEMMA4_PRESET = "gemma4_26b_a4b"

tokenizer = keras_hub.models.Gemma4Tokenizer.from_preset(
    GEMMA4_PRESET, sequence_length=MAX_SEQUENCE_LENGTH
)


def tokenize_and_pad(text, label):
    """Tokenize text, truncate/pad to fixed length, and create mask."""
    token_ids = tokenizer(text)
    # Create padding mask (1 for real tokens, 0 for pad).
    padding_mask = tf.cast(tf.not_equal(token_ids, 0), "int32")
    return {"token_ids": token_ids, "padding_mask": padding_mask}, label


train_ds = (
    train_data.map(tokenize_and_pad, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    test_data.map(tokenize_and_pad, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

"""
## Build the Classifier Model

We build a text classifier on top of the Gemma 4 backbone. The architecture
is:

1. **Gemma 4 backbone**: Extracts contextual embeddings from the input tokens.
2. **Global average pooling**: Pools the sequence of embeddings into a single
   vector.
3. **Dense classifier head**: Maps the pooled representation to a binary
   output.

This base classifier is reused across all training strategies.
"""


def print_results_so_far():
    """Print a summary table of results collected so far."""
    if not results:
        return
    print("\n" + "=" * 55)
    print("       Results So Far")
    print("=" * 55)
    print(f"{'Method':<30} {'Accuracy':>10}")
    print("-" * 55)
    for method, accuracy in results.items():
        print(f"{method:<30} {accuracy:>10.4f}")
    print("=" * 55 + "\n")


def build_classifier(backbone, trainable_backbone=True):
    """Build a binary classifier on top of the Gemma 4 backbone."""
    backbone.trainable = trainable_backbone

    # Inputs matching the backbone's expected format.
    token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
    padding_mask = keras.Input(
        shape=(None,), dtype="int32", name="padding_mask"
    )

    # Get sequence embeddings from the backbone.
    sequence_output = backbone(
        {"token_ids": token_ids, "padding_mask": padding_mask}
    )

    # Pool across the sequence dimension.
    # Use the padding mask to ignore padding tokens.
    mask = ops.cast(
        ops.expand_dims(padding_mask, axis=-1), sequence_output.dtype
    )
    masked_output = sequence_output * mask
    pooled_output = ops.sum(masked_output, axis=1) / (
        ops.sum(mask, axis=1) + 1e-8
    )

    # Classification head.
    output = keras.layers.Dense(
        1, activation="sigmoid", dtype="float32", name="classifier"
    )(pooled_output)

    model = keras.Model(
        inputs={"token_ids": token_ids, "padding_mask": padding_mask},
        outputs=output,
    )
    return model


def predict_sentiment(model, text):
    """Predict sentiment for a single text input."""
    token_ids = tokenizer(text)
    padding_mask = tf.cast(tf.not_equal(token_ids, 0), "int32")
    inputs = {
        "token_ids": ops.expand_dims(token_ids, 0),
        "padding_mask": ops.expand_dims(padding_mask, 0),
    }
    prediction = model.predict(inputs, verbose=0)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, float(prediction)


sample_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the "
    "plot kept me on the edge of my seat throughout.",
    "Terrible film. Poor writing, bad acting, and a completely "
    "predictable storyline. Do not waste your time.",
    "A decent movie with some good moments, but overall it felt a bit "
    "too long and the ending was disappointing.",
]


def run_inference(model, method_name):
    """Run inference on sample reviews and print results."""
    print(f"\n--- {method_name}: Sample Predictions ---")
    for review in sample_reviews:
        sentiment, score = predict_sentiment(model, review)
        print(f"  {review[:70]}...")
        print(f"  → {sentiment} (confidence: {score:.4f})")
    print()


"""
## Feature Extraction (Baseline)

In feature extraction mode, we freeze the backbone and train only the
classification head. This is the fastest approach and provides a baseline
accuracy to compare against.

**Why start here?** Feature extraction establishes a lower bound on
performance with minimal training cost. The backbone's pretrained
representations are already powerful for many tasks.
"""

backbone = keras_hub.models.Gemma4Backbone.from_preset(GEMMA4_PRESET)

# Build classifier with the backbone frozen.
feature_extraction_model = build_classifier(backbone, trainable_backbone=False)
feature_extraction_model.summary()

feature_extraction_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n--- Feature Extraction Training ---")
feature_extraction_model.fit(
    train_ds, validation_data=test_ds, epochs=EPOCHS
)

feature_extraction_results = feature_extraction_model.evaluate(test_ds)
results["Feature Extraction"] = feature_extraction_results[1]
print(
    f"Feature Extraction - Loss: {feature_extraction_results[0]:.4f}, "
    f"Accuracy: {feature_extraction_results[1]:.4f}"
)

run_inference(feature_extraction_model, "Feature Extraction")

"""
## Full Fine-Tuning

Full fine-tuning unfreezes the entire backbone, allowing all weights to
adapt to the downstream task. We use a much lower learning rate to avoid
catastrophic forgetting of the pretrained knowledge.

**When to use full fine-tuning?** When you have sufficient memory and
want maximum accuracy. The tradeoff is higher training cost and memory
usage. Note that `gemma4_26b_a4b` has 26B total parameters (MoE), so
full fine-tuning requires substantial resources.
"""

# Unfreeze the backbone for full fine-tuning.
backbone.trainable = True

full_finetune_model = build_classifier(backbone, trainable_backbone=True)
full_finetune_model.summary()

full_finetune_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n--- Full Fine-Tuning ---")
print_results_so_far()
full_finetune_model.fit(
    train_ds, validation_data=test_ds, epochs=EPOCHS
)

full_finetune_results = full_finetune_model.evaluate(test_ds)
results["Full Fine-Tuning"] = full_finetune_results[1]
print(
    f"Full Fine-Tuning - Loss: {full_finetune_results[0]:.4f}, "
    f"Accuracy: {full_finetune_results[1]:.4f}"
)

run_inference(full_finetune_model, "Full Fine-Tuning")

"""
## LoRA Fine-Tuning (⭐ Recommended)

[Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) is the
recommended fine-tuning method for large models. Instead of updating all
model weights, LoRA injects small trainable rank-decomposition matrices
into the attention layers, drastically reducing the number of trainable
parameters.

### Why LoRA?

- **Memory efficient**: Only a fraction of parameters are trainable.
- **Faster training**: Fewer gradients to compute.
- **No inference overhead**: LoRA weights can be merged back into the
  backbone after training.

KerasHub makes enabling LoRA trivially easy with a one-line API.
"""

# Clean up previous models to free memory.
del feature_extraction_model
del full_finetune_model

import gc

gc.collect()

# Load a fresh backbone for LoRA training.
lora_backbone = keras_hub.models.Gemma4Backbone.from_preset(GEMMA4_PRESET)

# Enable LoRA with rank 4. This freezes the backbone and adds trainable
# LoRA layers to the attention projections.
lora_backbone.enable_lora(rank=4)

lora_model = build_classifier(lora_backbone, trainable_backbone=True)

print("\n--- LoRA Model Summary ---")
lora_model.summary()

"""
Notice the dramatic reduction in trainable parameters compared to full
fine-tuning. Only the LoRA adapter weights and the classifier head are
trainable.
"""

lora_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n--- LoRA Fine-Tuning ---")
print_results_so_far()
lora_model.fit(
    train_ds, validation_data=test_ds, epochs=EPOCHS
)

lora_results = lora_model.evaluate(test_ds)
results["LoRA Fine-Tuning"] = lora_results[1]
print(
    f"LoRA Fine-Tuning - Loss: {lora_results[0]:.4f}, "
    f"Accuracy: {lora_results[1]:.4f}"
)

run_inference(lora_model, "LoRA Fine-Tuning")

"""
## Evaluation

Let's compare the results of all three training strategies.
"""

print("\n" + "=" * 60)
print("         Training Strategy Comparison")
print("=" * 60)
print(f"{'Method':<25} {'Accuracy':>10}")
print("-" * 60)
print(f"{'Feature Extraction':<25} {feature_extraction_results[1]:>10.4f}")
print(f"{'Full Fine-Tuning':<25} {full_finetune_results[1]:>10.4f}")
print(f"{'LoRA Fine-Tuning':<25} {lora_results[1]:>10.4f}")
print("=" * 60)

"""
LoRA fine-tuning typically achieves comparable accuracy to full fine-tuning
while using significantly fewer trainable parameters and less memory.

**Note:** These results are from training on a small subset of 2000
samples. With the full IMDB dataset (25,000 samples) and more epochs,
all methods will achieve significantly higher accuracy.
"""

"""
## Post-Training Quantization

Now we move from training to deployment optimization. Post-training
quantization (PTQ) reduces model size by converting weights from
higher-precision formats (like float32/bfloat16) to lower-precision
formats (like int8).

### KerasHub Native Quantization

KerasHub provides a built-in `quantize()` API that works across all Keras
backends. This is the simplest path to a smaller model.
"""

# Clean up LoRA model to free memory.
del lora_model
del lora_backbone
gc.collect()

# Load a fresh backbone and apply quantization.
ptq_backbone = keras_hub.models.Gemma4Backbone.from_preset(GEMMA4_PRESET)
ptq_backbone.quantize("int8")

ptq_model = build_classifier(ptq_backbone, trainable_backbone=False)
ptq_model.summary()

"""
The quantized model has significantly reduced memory footprint. Let's
evaluate it to see the accuracy impact of quantization.
"""

ptq_model.compile(
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n--- Post-Training Quantization Evaluation ---")
print_results_so_far()
ptq_results = ptq_model.evaluate(test_ds)
results["Post-Training Quantization"] = ptq_results[1]
print(
    f"Post-Training Quantization - Loss: {ptq_results[0]:.4f}, "
    f"Accuracy: {ptq_results[1]:.4f}"
)


"""
## LoRA Fine-Tuning + Quantization (Production Workflow)

The recommended production workflow combines LoRA fine-tuning with
post-training quantization. First, fine-tune the model with LoRA to
adapt it to your task, then quantize the trained model for deployment.

Keras 3 provides `quantize("int8")` as a backend-agnostic API for
post-training quantization. This can be applied to any Keras model,
including one that has been fine-tuned with LoRA.
"""

# Clean up PTQ model to free memory.
del ptq_model
del ptq_backbone
gc.collect()

# Load a fresh backbone, enable LoRA, and fine-tune.
lora_q_backbone = keras_hub.models.Gemma4Backbone.from_preset(GEMMA4_PRESET)
lora_q_backbone.enable_lora(rank=4)
lora_q_model = build_classifier(lora_q_backbone, trainable_backbone=True)

lora_q_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n--- LoRA Fine-Tuning + Quantization ---")
print_results_so_far()
lora_q_model.fit(
    train_ds, validation_data=test_ds, epochs=EPOCHS
)

# Now quantize the fine-tuned backbone weights to int8.
lora_q_backbone.quantize("int8")

lora_q_model.compile(
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

lora_q_results = lora_q_model.evaluate(test_ds)
results["LoRA + Quantization"] = lora_q_results[1]
print(
    f"LoRA + Quantization - Loss: {lora_q_results[0]:.4f}, "
    f"Accuracy: {lora_q_results[1]:.4f}"
)

run_inference(lora_q_model, "LoRA + Quantization")

"""
By fine-tuning first and then quantizing, the model retains task-specific
knowledge while benefiting from reduced model size. This is the
recommended approach for production deployments.
"""

"""
## Quantization-Aware Training (QAT)

Quantization-Aware Training (QAT) goes one step further than the
LoRA + Quantization pipeline above. Instead of quantizing *after*
training, we first quantize the model, and then fine-tune *the quantized
model* using LoRA. This allows the model to adapt its weights to the
quantization noise during training, often recovering accuracy lost
during post-training quantization.

### QAT Workflow

1. Load a fresh backbone.
2. Apply int8 quantization.
3. Enable LoRA on the quantized backbone.
4. Fine-tune the quantized + LoRA model.

This approach is especially useful when post-training quantization
causes a noticeable accuracy drop and you want to close the gap.
"""

# Clean up previous models.
del lora_q_model
del lora_q_backbone
gc.collect()

# Step 1: Load a fresh backbone.
qat_backbone = keras_hub.models.Gemma4Backbone.from_preset(GEMMA4_PRESET)

# Step 2: Quantize first — the model learns to compensate for
# quantization noise during training.
qat_backbone.quantize("int8")

# Step 3: Enable LoRA on the quantized backbone.
qat_backbone.enable_lora(rank=4)

qat_model = build_classifier(qat_backbone, trainable_backbone=True)

print("\n--- QAT Model Summary ---")
qat_model.summary()

qat_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n--- Quantization-Aware Training ---")
print_results_so_far()
qat_model.fit(
    train_ds, validation_data=test_ds, epochs=EPOCHS
)

qat_results = qat_model.evaluate(test_ds)
results["QAT (Quantize → LoRA)"] = qat_results[1]
print(
    f"QAT - Loss: {qat_results[0]:.4f}, "
    f"Accuracy: {qat_results[1]:.4f}"
)

run_inference(qat_model, "QAT (Quantize → LoRA)")

"""
### LoRA + Quantization vs QAT

Both pipelines combine LoRA and quantization, but they differ in order:

| Approach | Order | Best For |
|:---|:---|:---|
| LoRA + Quantization | Train → Quantize | When PTQ accuracy is acceptable |
| QAT | Quantize → Train | When PTQ causes noticeable accuracy drop |

In practice, start with LoRA + Quantization. If the post-quantization
accuracy drop is too large, switch to QAT to let the model learn to
compensate for quantization noise.
"""

"""
## Recommended Production Workflow

Here is the recommended end-to-end pipeline for deploying a Gemma 4 text
classifier in production:

1. **Start with pretrained Gemma 4**: Load the `gemma4_26b_a4b` backbone
   from KerasHub.
2. **Fine-tune with LoRA**: Enable LoRA with a small rank (4-8) and train
   on your labeled dataset. This gives the best balance of accuracy and
   efficiency.
3. **Quantize for deployment**: Apply int8 quantization using Keras's
   `quantize("int8")` to reduce the model size.
4. **Save and export**: Use `model.save()` or `save_to_preset()` to save
   the quantized model for deployment.
"""

print("\n" + "=" * 60)
print("         Full Results Summary")
print("=" * 60)
print(f"{'Method':<30} {'Accuracy':>10}")
print("-" * 60)
print(f"{'Feature Extraction':<30} {feature_extraction_results[1]:>10.4f}")
print(f"{'Full Fine-Tuning':<30} {full_finetune_results[1]:>10.4f}")
print(f"{'LoRA Fine-Tuning':<30} {lora_results[1]:>10.4f}")
print(f"{'Post-Training Quantization':<30} {ptq_results[1]:>10.4f}")
print(f"{'LoRA + Quantization':<30} {lora_q_results[1]:>10.4f}")
print(f"{'QAT (Quantize → LoRA)':<30} {qat_results[1]:>10.4f}")
print("=" * 60)

"""
## Conclusion

In this guide, we demonstrated a complete text classification pipeline with
Gemma 4:

- **Feature extraction** provides a quick baseline with minimal training.
- **Full fine-tuning** achieves the highest accuracy but at greater
  computational cost.
- **LoRA fine-tuning** (recommended) achieves near-full-fine-tuning accuracy
  with a fraction of the trainable parameters.
- **Post-training quantization** reduces model size for deployment.
- **LoRA + Quantization** combines fine-tuning and compression for the
  best production workflow.
- **QAT (Quantize → LoRA)** recovers accuracy lost during quantization by
  fine-tuning the already-quantized model.

For production deployments, we recommend: **LoRA fine-tuning → Quantization
→ Save**. This pipeline provides the best balance of accuracy, efficiency,
and model size.

### Next Steps

- Increase the dataset size and training epochs for better results.
- Experiment with different LoRA ranks (2, 4, 8, 16).
- Try the `gemma4_instruct_26b_a4b` preset for instruction-following
  tasks.
- Explore the `gemma4_31b` dense variant for maximum quality on
  complex tasks.
- Use `int4` quantization for even more aggressive compression.

### References

- [Gemma 4 on Kaggle](https://www.kaggle.com/models/keras/gemma4)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [KerasHub documentation](https://keras.io/keras_hub/)
- [Quantization in Keras](https://keras.io/guides/quantization_overview/)
- [INT8 Quantization in Keras](https://keras.io/guides/int8_quantization_in_keras/)
"""

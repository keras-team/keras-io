# INT4 Quantization in Keras

**Author:** [Jyotinder Singh](https://x.com/Jyotinder_Singh)<br>
**Date created:** 2025/10/14<br>
**Last modified:** 2025/10/14<br>
**Description:** Complete guide to using INT4 quantization in Keras and KerasHub.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/int4_quantization_in_keras.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/int4_quantization_in_keras.py)



---
## What is INT4 quantization?

Quantization lowers the numerical precision of weights and activations to reduce memory use
and often speed up inference, at the cost of a small accuracy drop. INT4 post-training
quantization (PTQ) stores model weights in 4-bit signed integers and dynamically quantizes
activations to 8-bit at runtime (a W4A8 scheme). Compared with FP32 this can shrink weight
storage ~8x (2x vs INT8) while retaining acceptable accuracy for many encoder models and
some decoder models. Compute still leverages widely available INT8 Tensor Cores.

4-bit is a more aggressive compression than 8-bit and may
induce larger quality regressions, especially for large autoregressive language models.

---
## How it works

Quantization maps real values to 4-bit integers with a scale:

1. Per-output-channel scale computed for each weight matrix (symmetric abs-max).
2. Weights are quantized to values in `[-8, 7]` (4 bits) and packed two-per-byte.
3. At inference, activations are dynamically scaled and quantized to INT8.
4. Packed INT4 weights are unpacked to an INT8 tensor (still with INT4-range values).
5. INT8 x INT8 matmul accumulates in INT32.
6. Result is dequantized using `(input_scale * per_channel_kernel_scale)`.

This mirrors the INT8 path described in the
[INT8 guide](https://keras.io/guides/int8_quantization_in_keras) with some added unpack
overhead for stronger compression.

---
## Benefits
* Memory / bandwidth bound models: When the implementation spends most of its time on memory I/O,
  reducing the computation time does not reduce its overall runtime. INT4 reduces bytes
  moved by ~8x vs `float32`, improving cache behavior and reducing memory stalls;
  this often helps more than increasing raw FLOPs.
* Accuracy: Many architectures retain acceptable accuracy with INT4; encoder-only models
  often fare better than decoder LLMs. Always validate on your own dataset.
* Compute bound layers on supported hardware: 4-bit kernels are unpacked to INT8 at inference,
  therefore, on NVIDIA GPUs, INT8 [Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)
  speed up matmul/conv, boosting throughput on compute-limited layers.

### What Keras does in INT4 mode

* **Mapping**: Symmetric, linear quantization with INT4 plus a floating-point scale.
* **Weights**: per-output-channel scales to preserve accuracy.
* **Activations**: **dynamic AbsMax** scaling computed at runtime.
* **Graph rewrite**: Quantization is applied after weights are trained and built; the graph
is rewritten so you can run or save immediately.

---
## Overview

This guide shows how to use 4-bit (W4A8) post-training quantization in Keras:

1. [Quantizing a minimal functional model](#quantizing-a-minimal-functional-model)
2. [Saving and reloading a quantized model](#saving-and-reloading-a-quantized-model)
3. [Quantizing a KerasHub model](#quantizing-a-kerashub-model)
4. [When to use INT4 vs INT8](#when-should-i-use-int4-vs-int8)
5. [Performance benchmarks](#performance--benchmarking)
6. [Practical Tips](#practical-tips)
7. [Limitations](#limitations)

---
## Quantizing a Minimal Functional Model

Below we build a small functional model, capture a baseline output, quantize to INT4
in place, and compare outputs with an MSE metric. (For real evaluation use your
validation metric.)


```python
import numpy as np
import keras
from keras import layers

# Create a random number generator.
rng = np.random.default_rng()

# Create a simple functional model.
inputs = keras.Input(shape=(10,))
x = layers.Dense(32, activation="relu")(inputs)
outputs = layers.Dense(1, name="target")(x)
model = keras.Model(inputs, outputs)

# Baseline output with full-precision weights.
x_eval = rng.random((32, 10)).astype("float32")
y_fp32 = model(x_eval)


# Quantize the model in-place to INT4 (W4A8).
model.quantize("int4")

# Compare outputs (MSE).
y_int4 = model(x_eval)
mse = keras.ops.mean(keras.ops.square(y_fp32 - y_int4))
print("Full-Precision vs INT4 MSE:", float(mse))
```

<div class="k-default-codeblock">
```
Full-Precision vs INT4 MSE: 0.00028205406852066517
```
</div>

The INT4 quantized model usually produces outputs close enough for many downstream
tasks. Expect larger deltas than INT8, so always validate on your own data.

---
## Saving and Reloading a Quantized Model

You can use standard Keras saving / loading APIs. Quantization metadata (including
scales and packed weights) is preserved.


```python
# Save the quantized model and reload to verify round-trip.
model.save("int4.keras")
int4_reloaded = keras.saving.load_model("int4.keras")
y_int4_reloaded = int4_reloaded(x_eval)

# Compare outputs (MSE).
roundtrip_mse = keras.ops.mean(keras.ops.square(y_fp32 - y_int4_reloaded))
print("MSE (INT4 vs reloaded INT4):", float(roundtrip_mse))
```

<div class="k-default-codeblock">
```
MSE (INT4 vs reloaded INT4): 0.00028205406852066517
```
</div>

---
## Quantizing a KerasHub Model

All KerasHub models support the `.quantize(...)` API for post-training quantization,
and follow the same workflow as above.

In this example, we will:

1. Load the [gemma3_1b](https://www.kaggle.com/models/keras/gemma3/keras/gemma3_1b)
  preset from KerasHub
2. Generate text using both the full-precision and quantized models, and compare outputs.
3. Save both models to disk and compute storage savings.
4. Reload the INT4 model and verify output consistency with the original quantized model.

```python
import os
from keras_hub.models import Gemma3CausalLM

# Load a Gemma3 preset from KerasHub.
gemma3 = Gemma3CausalLM.from_preset("gemma3_1b")

# Generate with full-precision weights.
fp_output = gemma3.generate("Keras is a", max_length=30)
print("Full-precision output:", fp_output)

# Save the full-precision model to a preset.
gemma3.save_to_preset("gemma3_fp32")

# Quantize to INT4.
gemma3.quantize("int4")

# Generate with INT4 weights.
output = gemma3.generate("Keras is a", max_length=30)
print("Quantized output:", output)

# Save INT4 model to a new preset.
gemma3.save_to_preset("gemma3_int4")

# Reload and compare outputs
gemma3_int4 = Gemma3CausalLM.from_preset("gemma3_int4")

output = gemma3_int4.generate("Keras is a", max_length=30)
print("Quantized reloaded output:", output)


# Compute storage savings
def bytes_to_mib(n):
    return n / (1024**2)


gemma_fp32_size = os.path.getsize("gemma3_fp32/model.weights.h5")
gemma_int4_size = os.path.getsize("gemma3_int4/model.weights.h5")

gemma_reduction = 100.0 * (1.0 - (gemma_int4_size / max(gemma_fp32_size, 1)))
print(f"Gemma3: FP32 file size: {bytes_to_mib(gemma_fp32_size):.2f} MiB")
print(f"Gemma3: INT4 file size: {bytes_to_mib(gemma_int4_size):.2f} MiB")
print(f"Gemma3: Size reduction: {gemma_reduction:.1f}%")
```

<div class="k-default-codeblock">
```
Full-precision output: Keras is a deep learning library for Python. It is a high-level API for neural networks. It is a Python library for deep learning
Quantized output: Keras is a python-based, open-source, and free-to-use, open-source, and a popular, and a
Quantized reloaded output: Keras is a python-based, open-source, and free-to-use, open-source, and a popular, and a
Gemma3: FP32 file size: 3815.32 MiB
Gemma3: INT4 file size: 1488.10 MiB
Gemma3: Size reduction: 61.0%
```
</div>

---
## Performance & Benchmarking

Micro-benchmarks collected on a single NVIDIA L4 (22.5 GB). Baselines are FP32.

### Text Classification (DistilBERT Base on SST-2)

<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/gist/JyotinderSingh/77e874187d6da3f8280c053192f78c06/int4-quantization-micro-benchmark-distilbert.ipynb)

| Metric | FP32 | INT4 | Change |
| ------ | ---- | ---- | ------ |
| Accuracy (↑) | 91.06% | 90.14% | -0.92pp |
| Model Size (MB, ↓) | 255.86 | 159.49 | -37.67% |
| Peak GPU Memory (MiB, ↓) | 1554.00 | 1243.26 | -20.00% |
| Latency (ms/sample, ↓) | 6.43 | 5.73 | -10.83% |
| Throughput (samples/s, ↑) | 155.60 | 174.50 | +12.15% |

**Analysis**: Accuracy drop is modest (<1pp) with notable speed and memory gains;
encoder-only models tend to retain fidelity under heavier weight compression.

### Text Generation (Falcon 1B)

<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/gist/JyotinderSingh/19ab238e0f5b29ae24c0faf4128e7d7e/int4_quantization_micro_benchmark_falcon.ipynb)

| Metric | FP32 | INT4 | Change |
| ------ | ---- | ---- | ------ |
| Perplexity (↓) | 7.44 | 9.98 | +34.15% |
| Model Size (GB, ↓) | 4.8884 | 0.9526 | -80.51% |
| Peak GPU Memory (MiB, ↓) | 8021.12 | 5483.46 | -31.64% |
| First Token Latency (ms, ↓) | 128.87 | 122.50 | -4.95% |
| Sequence Latency (ms, ↓) | 338.29 | 181.93 | -46.22% |
| Token Throughput (tokens/s, ↑) | 174.41 | 256.96 | +47.33% |

**Analysis**: INT4 gives large size (-80%) and memory (-32%) reductions. Perplexity
increases (expected for aggressive compression) yet sequence latency drops and
throughput rises ~50%.

### Text Generation (Gemma3 1B)

<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/gist/JyotinderSingh/9ca7813971868d5d1a16cd7998d0e352/int4_quantization_micro_benchmark_gemma3.ipynb)

| Metric | FP32 | INT4 | Change |
| ------ | ---- | ---- | ------ |
| Perplexity (↓) | 6.17 | 10.46 | +69.61% |
| Model Size (GB, ↓) | 3.7303 | 1.4576 | -60.92% |
| Peak GPU Memory (MiB, ↓) | 6844.67 | 5008.14 | -26.83% |
| First Token Latency (ms, ↓) | 57.42 | 64.21 | +11.83% |
| Sequence Latency (ms, ↓) | 239.78 | 161.18 | -32.78% |
| Token Throughput (tokens/s, ↑) | 246.06 | 366.05 | +48.76% |

**Analysis**: INT4 gives large size (-61%) and memory (-27%) reductions. Perplexity
increases (expected for aggressive compression) yet sequence latency drops and
throughput rises ~50%.

### Text Generation (Llama 3.2 1B)

<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/gist/JyotinderSingh/310f50a0ca0eba3754de41c612b3b8ef/int4_quantization_micro_benchmark_llama3.ipynb)

| Metric | FP32 | INT4 | Change |
| ------ | ---- | ---- | ------ |
| Perplexity (↓) | 6.38 | 14.16 | +121.78% |
| Model Size (GB, ↓) | 5.5890 | 2.4186 | -56.73% |
| Peak GPU Memory (MiB, ↓) | 9509.49 | 6810.26 | -28.38% |
| First Token Latency (ms, ↓) | 209.41 | 219.09 | +4.62% |
| Sequence Latency (ms, ↓) | 322.33 | 262.15 | -18.67% |
| Token Throughput (tokens/s, ↑) | 183.82 | 230.78 | +25.55% |

**Analysis**: INT4 gives large size (-57%) and memory (-28%) reductions. Perplexity
increases (expected for aggressive compression) yet sequence latency drops and
throughput rises ~25%.

### Text Generation (OPT 125M)

<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/gist/JyotinderSingh/918fcdb8a1433dea12800f8ca4a240f5/int4_quantization_micro_benchmark_opt.ipynb)

| Metric | FP32 | INT4 | Change |
| ------ | ---- | ---- | ------ |
| Perplexity (↓) | 13.85 | 21.02 | +51.79% |
| Model Size (MB, ↓) | 468.3 | 284.0 | -39.37% |
| Peak GPU Memory (MiB, ↓) | 1007.23 | 659.28 | -34.54% |
| First Token Latency (ms/sample, ↓) | 95.79 | 97.87 | +2.18% |
| Sequence Latency (ms/sample, ↓) | 60.35 | 54.64 | -9.46% |
| Throughput (samples/s, ↑) | 973.41 | 1075.15 | +10.45% |

**Analysis**: INT4 gives large size (-39%) and memory (-35%) reductions. Perplexity
increases (expected for aggressive compression) yet sequence latency drops and
throughput rises ~10%.

---
## When should I use INT4 vs INT8?

| Goal / Constraint | Prefer INT8 | Prefer INT4 (W4A8) |
| ----------------- | ----------- | ------------------ |
| Minimal accuracy drop critical | ✔︎ |  |
| Maximum compression (disk / RAM) |  | ✔︎ |
| Bandwidth-bound inference | Possible | Often better |
| Decoder LLM | ✔︎ | Try with eval first |
| Encoder / classification models | ✔︎ | ✔︎ |
| Available kernels / tooling maturity | ✔︎ | Emerging |

* Start with INT8; if memory or distribution size is still a bottleneck, evaluate INT4.
* For LLMs, measure task-specific metrics (perplexity, exact match, etc.) after INT4.
* Combine INT4 weights + LoRA adapters for efficient fine-tuning workflows.

---
## Practical Tips

* Post-training quantization (PTQ) is a one-time operation; you cannot train a model
  after quantizing it to INT4.
* Always materialize weights before quantization (e.g., `build()` or a forward pass).
* Evaluate on a representative validation set; track task metrics, not just MSE.
* Use LoRA for further fine-tuning.

---
## Limitations
* Runtime unpack adds overhead (weights are decompressed layer-wise for each forward pass).
* Large compression leads to accuracy drop (especially for decoder-only LLMs).
* LoRA export path is lossy (dequantize -> add delta -> requantize).
* Keras does not yet support native fused INT4 kernels; relies on unpack + INT8 matmul.

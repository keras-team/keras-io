# Quantization in Keras

**Author:** [Jyotinder Singh](https://x.com/Jyotinder_Singh)<br>
**Date created:** 2025/10/09<br>
**Last modified:** 2025/10/09<br>
**Description:** Overview of quantization in Keras (int8, float8, int4, GPTQ).


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/quantization_overview.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/quantization_overview.py)



---
## Introduction

Modern large models are often **memory- and bandwidth-bound**: most inference time is spent moving tensors between memory and compute units rather than doing math. Quantization reduces the number of bits used to represent the model's weights and (optionally) activations, which:

* Shrinks model size and VRAM/RAM footprint.
* Increases effective memory bandwidth (fewer bytes per value).
* Can improve throughput and sometimes latency on supporting hardware with low-precision kernels.

Keras provides first-class **post-training quantization (PTQ)** workflows which support pretrained models and expose a uniform API at both the model and layer level.

At a high level, Keras supports:

* Joint weight + activation PTQ in `int4`, `int8`, and `float8`.
* Weight-only PTQ via **GPTQ** (2/3/4/8-bit) to maximize compression with minimal accuracy impact, especially for large language models (LLMs).

### Terminology

* *Scale / zero-point:* Quantization maps real values `x` to integers `q` using a scale (and optionally a zero-point). Symmetric schemes use only a scale.
* *Per-channel vs per-tensor:* A separate scale per output channel (e.g., per hidden unit) usually preserves accuracy better than a single scale for the whole tensor.
* *Calibration:* A short pass over sample data to estimate activation ranges (e.g., max absolute value).

---
## Quantization Modes

Keras currently focuses on the following numeric formats. Each mode can be applied selectively to layers or to the whole model via the same API.

* **`int8` (8-bit integer)**: **joint weight + activation** PTQ.

  * **How it works:** Values are linearly mapped to 8-bit integers with per-channel scales. Activations are calibrated using dynamic quantization (see note below).
  * **Why use it:** Good accuracy for many architectures; broad hardware support.
  * **What to expect:** ~4x smaller than FP32 parameters (~2x vs FP16) and lower activation bandwidth, with small accuracy loss on many tasks. Throughput gains depend on kernel availability and memory bandwidth.

* **`float8` (FP8: E4M3 / E5M2 variants)**: Low-precision floating-point useful for training and inference on FP8-capable hardware.

  * **How it works:** Values are quantized to FP8 with a dynamic scale. Fused FP8 kernels on supported devices yield speedups.
  * **Why use it:** Mixed-precision training/inference with hardware acceleration while keeping floating-point semantics (since underflow/overflow characteristics differ from int).
  * **What to expect:** Competitive speed and memory reductions where FP8 kernels are available; accuracy varies by model, but is usually acceptable for most tasks.

* **`int4`**: Ultra-low-bit **weights** for aggressive compression; activations remain in higher precision (int8).

  * **How it works:** Two signed 4-bit "nibbles" are packed per int8 byte. Keras uses symmetric per-output-channel scales to dequantize efficiently inside matmul.
  * **Why use it:** Significant VRAM/storage savings for LLMs with acceptable accuracy when combined with robust per-channel scaling.
  * **What to expect:** ~8x smaller than FP32 (~4x vs FP16) for weights; throughput gains depend on kernel availability and memory bandwidth. Competitive accuracy deltas for encoder-only architectures, may show larger regressions on decoder-only models.

* **`GPTQ` (weight-only 2/3/4/8 bits)**: *Second-order, post-training* method minimizing layer output error.

  * **How it works (brief):** For each weight block (group), GPTQ solves a local least-squares problem using a Hessian approximation built from a small calibration set, then quantizes to low bit-width. The result is a packed weight tensor plus per-group parameters (e.g., scales).
  * **Why use it:** Strong accuracy retention at very low bit-widths without retraining; ideal for rapid LLM compression.
  * **What to expect:** Large storage/VRAM savings with small perplexity/accuracy deltas on many decoder-only models when calibrated on task-relevant samples.

### Implementation notes

* **Dynamic activation quantization**: In the `int4`, `int8` PTQ path, activation scales are computed on-the-fly at runtime (per tensor and per batch) using an AbsMax estimator. This avoids maintaining a separate, fixed set of activation scales from a calibration pass and adapts to varying input ranges.
* **4-bit packing**: For `int4`, Keras packs signed 4-bit values (range = [-8, 7]) and stores per-channel scales such as `kernel_scale`. Dequantization happens on the fly, and matmuls use 8-bit (unpacked) kernels.
* **Calibration Strategy**: Activation scaling for `int4` / `int8` / `float8` uses **AbsMax calibration** by default (range set by the maximum absolute value observed). Alternative calibration methods (e.g., percentile) may be added in future releases.
* Per-channel scaling is the default for weights where supported, because it materially improves accuracy at negligible overhead.

---
## Quantizing Keras Models

Quantization is applied explicitly after layers or models are built. The API is designed to be predictable: you call quantize, the graph is rewritten, the weights are replaced, and you can immediately run inference or save the model.

Typical workflow:

1. **Build / load your FP model.** Train if needed. Ensure `build()` or a forward pass has materialized weights.
2. **(GPTQ only)** For GPTQ, Keras runs a short calibration pass to collect activation statistics. You will need to provide a small, representative dataset for this purpose.
3. **Invoke quantization.** Call `model.quantize("<mode>")` or `layer.quantize("<mode>")` with `"int8"`, `"int4"`, `"float8"`, or `"gptq"` (weight-only).
4. **Use or save.** Run inference, or `model.save(...)`. Quantization state (packed weights, scales, metadata) is preserved on save/load.

### Model Quantization


```python
import keras
import numpy as np

# Sample training data.
x_train = keras.ops.array(np.random.rand(100, 10))
y_train = keras.ops.array(np.random.rand(100, 1))

# Build the model.
model = keras.Sequential(
    [
        keras.Input(shape=(10,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ]
)

# Compile and fit the model.
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=1, verbose=0)

# Quantize the model.
model.quantize("int8")
```

**What this does:** Quantizes the weights of the supported layers, and re-wires their forward paths to be compatible with the quantized kernels and quantization scales.

**Note**: Throughput gains depend on backend/hardware kernels; in cases where kernels fall back to dequantized matmul, you still get memory savings but smaller speedups.

### Layer-wise Quantization

The Keras quantization framework allows you to quantize each layer separately, without having to quantize the entire model using the same unified API.


```python
from keras import layers

input_shape = (10,)
layer = layers.Dense(32, activation="relu")
layer.build(input_shape)

layer.quantize("int4")  # Or "int8", "float8", etc.
```

### When to use layer-wise quantization

* To keep numerically sensitive blocks (e.g., small residual paths, logits) at higher precision while quantizing large projection layers.
* To mix modes (e.g., attention projections in int4, feed-forward in int8) and measure trade-offs incrementally.
* Always validate on a small eval set after each step; mixing precisions across residual connections can shift distributions.

---
## Layer & model coverage

Keras supports the following core layers in its quantization framework:

* `Dense`
* `EinsumDense`
* `Embedding`
* `ReversibleEmbedding` (available in KerasHub)

Any composite layers that are built from the above (for example, `MultiHeadAttention`, `GroupedQueryAttention`, feed-forward blocks in Transformers) inherit quantization support by construction. This covers the majority of modern encoder-only and decoder-only Transformer architectures.

Since all KerasHub models subclass `keras.Model`, they automatically support the `model.quantize(...)` API. In practice, this means you can take a popular LLM preset, call a single function to obtain an int8/int4/GPTQ-quantized variant, and then save or serve it—without changing your training code.

---
## Practical guidance

* For GPTQ, use a calibration set that matches your inference domain (a few hundred to a few thousand tokens is often enough to see strong retention).
* Measure both **VRAM** and **throughput/latency**: memory savings are immediate; speedups depend on the availability of fused low-precision kernels on your device.

# 8-bit Integer Quantization in Keras

**Author:** [Jyotinder Singh](https://x.com/Jyotinder_Singh)<br>
**Date created:** 2025/10/14<br>
**Last modified:** 2025/10/14<br>
**Description:** Complete guide to using INT8 quantization in Keras and KerasHub.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/int8_quantization_in_keras.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/int8_quantization_in_keras.py)



---
## What is INT8 quantization?

Quantization lowers the numerical precision of weights and activations to reduce memory use
and often speed up inference, at the cost of a small accuracy drop. Moving from `float32` to
`float16` halves the memory requirements; `float32` to INT8 is ~4x smaller (and ~2x vs
`float16`). On hardware with low-precision kernels (e.g., NVIDIA Tensor Cores), this can also
improve throughput and latency. Actual gains depend on your backend and device.

### How it works

Quantization maps real values to 8-bit integers with a scale:

* Integer domain: `[-128, 127]` (256 levels).
* For a tensor (often per-output-channel for weights) with values `w`:
  * Compute `a_max = max(abs(w))`.
  * Set scale `s = (2 * a_max) / 256`.
  * Quantize: `q = clip(round(w / s), -128, 127)` (stored as INT8) and keep `s`.
* Inference uses `q` and `s` to reconstruct effective weights on the fly
  (`w ≈ s · q`) or folds `s` into the matmul/conv for efficiency.

### Benefits

* Memory / bandwidth bound models: When implementation spends most of its time on memory I/O,
  reducing the computation time does not reduce their overall runtime. INT8 reduces bytes
  moved by ~4x vs `float32`, improving cache behavior and reducing memory stalls;
  this often helps more than increasing raw FLOPs.
* Compute bound layers on supported hardware: On NVIDIA GPUs, INT8
  [Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/) speed up matmul/conv,
  boosting throughput on compute-limited layers.
* Accuracy: Many models retain near-FP accuracy with `float16`; INT8 may introduce a modest
  drop (often ~1-5% depending on task/model/data). Always validate on your own dataset.

### What Keras does in INT8 mode

* **Mapping**: Symmetric, linear quantization with INT8 plus a floating-point scale.
* **Weights**: per-output-channel scales to preserve accuracy.
* **Activations**: **dynamic AbsMax** scaling computed at runtime.
* **Graph rewrite**: Quantization is applied after weights are trained and built; the graph
  is rewritten so you can run or save immediately.

---
## Overview

This guide shows how to use 8-bit integer post-training quantization (PTQ) in Keras:

1. [Quantizing a minimal functional model](#quantizing-a-minimal-functional-model)
2. [Saving and reloading a quantized model](#saving-and-reloading-a-quantized-model)
3. [Quantizing a KerasHub model](#quantizing-a-kerashub-model)

---
## Quantizing a minimal functional model.

We build a small functional model, capture a baseline output, quantize to INT8 in-place,
and then compare outputs with an MSE metric.


```python
import os
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

# Compile and train briefly to materialize meaningful weights.
model.compile(optimizer="adam", loss="mse")
x_train = rng.random((256, 10)).astype("float32")
y_train = rng.random((256, 1)).astype("float32")
model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)

# Sample inputs for evaluation.
x_eval = rng.random((32, 10)).astype("float32")

# Baseline (FP) outputs.
y_fp32 = model(x_eval)

# Quantize the model in-place to INT8.
model.quantize("int8")

# INT8 outputs after quantization.
y_int8 = model(x_eval)

# Compute a simple MSE between FP and INT8 outputs.
mse = keras.ops.mean(keras.ops.square(y_fp32 - y_int8))
print("Full-Precision vs INT8 MSE:", float(mse))

```

<div class="k-default-codeblock">
```
Full-Precision vs INT8 MSE: 4.982496648153756e-06
```
</div>

It is evident that the INT8 quantized model produces outputs close to the original FP32
model, as indicated by the low MSE value.

---
## Saving and reloading a quantized model

You can use the standard Keras saving and loading APIs with quantized models. Quantization
is preserved when saving to `.keras` and loading back.


```python
# Save the quantized model and reload to verify round-trip.
model.save("int8.keras")
int8_reloaded = keras.saving.load_model("int8.keras")
y_int8_reloaded = int8_reloaded(x_eval)
roundtrip_mse = keras.ops.mean(keras.ops.square(y_int8 - y_int8_reloaded))
print("MSE (INT8 vs reloaded-INT8):", float(roundtrip_mse))
```

<div class="k-default-codeblock">
```
MSE (INT8 vs reloaded-INT8): 0.0
```
</div>

---
## Quantizing a KerasHub model

All KerasHub models support the `.quantize(...)` API for post-training quantization,
and follow the same workflow as above.

In this example, we will:

1. Load the [gemma3_1b](https://www.kaggle.com/models/keras/gemma3/keras/gemma3_1b)
  preset from KerasHub
2. Generate text using both the full-precision and quantized models, and compare outputs.
3. Save both models to disk and compute storage savings.
4. Reload the INT8 model and verify output consistency with the original quantized model.


```python
from keras_hub.models import Gemma3CausalLM

# Load from Gemma3 preset
gemma3 = Gemma3CausalLM.from_preset("gemma3_1b")

# Generate text for a single prompt
output = gemma3.generate("Keras is a", max_length=50)
print("Full-precision output:", output)

# Save FP32 Gemma3 model for size comparison.
gemma3.save_to_preset("gemma3_fp32")

# Quantize in-place to INT8 and generate again
gemma3.quantize("int8")

output = gemma3.generate("Keras is a", max_length=50)
print("Quantized output:", output)

# Save INT8 Gemma3 model
gemma3.save_to_preset("gemma3_int8")

# Reload and compare outputs
gemma3_int8 = Gemma3CausalLM.from_preset("gemma3_int8")

output = gemma3_int8.generate("Keras is a", max_length=50)
print("Quantized reloaded output:", output)


# Compute storage savings
def bytes_to_mib(n):
    return n / (1024**2)


gemma_fp32_size = os.path.getsize("gemma3_fp32/model.weights.h5")
gemma_int8_size = os.path.getsize("gemma3_int8/model.weights.h5")

gemma_reduction = 100.0 * (1.0 - (gemma_int8_size / max(gemma_fp32_size, 1)))
print(f"Gemma3: FP32 file size: {bytes_to_mib(gemma_fp32_size):.2f} MiB")
print(f"Gemma3: INT8 file size: {bytes_to_mib(gemma_int8_size):.2f} MiB")
print(f"Gemma3: Size reduction: {gemma_reduction:.1f}%")
```

<div class="k-default-codeblock">
```
Full-precision output: Keras is a deep learning library for Python. It is a high-level API for neural networks. It is a Python library for deep learning. It is a library for deep learning. It is a library for deep learning. It is a
Quantized output: Keras is a Python library for deep learning. It is a high-level API for building deep learning models. It is designed to be easy
Quantized reloaded output: Keras is a Python library for deep learning. It is a high-level API for building deep learning models. It is designed to be easy
Gemma3: FP32 file size: 3815.32 MiB
Gemma3: INT8 file size: 957.81 MiB
Gemma3: Size reduction: 74.9%
```
</div>

---
## Practical tips

* Post-training quantization (PTQ) is a one-time operation; you cannot train a model
  after quantizing it to INT8.
* Always materialize weights before quantization (e.g., `build()` or a forward pass).
* Expect small numerical deltas; quantify with a metric like MSE on a validation batch.
* Storage savings are immediate; speedups depend on backend/device kernels.

---
## References

* [Milvus: How does 8-bit quantization or float16 affect the accuracy and speed of Sentence Transformer embeddings and similarity calculations?](https://milvus.io/ai-quick-reference/how-does-quantization-such-as-int8-quantization-or-using-float16-affect-the-accuracy-and-speed-of-sentence-transformer-embeddings-and-similarity-calculations)
* [NVIDIA Developer Blog: Achieving FP32 accuracy for INT8 inference using quantization-aware training with TensorRT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)

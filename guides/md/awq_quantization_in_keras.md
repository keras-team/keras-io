# AWQ Quantization in Keras

**Author:** [Jyotinder Singh](https://x.com/Jyotinder_Singh)<br>
**Date created:** 2025/01/15<br>
**Last modified:** 2025/01/15<br>
**Description:** How to run weight-only AWQ quantization for Keras & KerasHub models.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/awq_quantization_in_keras.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/awq_quantization_in_keras.py)



---
## What is AWQ?

AWQ (Activation-aware Weight Quantization) is a post-training, weight-only
quantization method that uses activation statistics to identify and protect
salient weights during quantization.

The key insight of AWQ is that not all weights are equally important: a small
fraction of weights (typically <1%) are "salient" because they process
channels with large activation magnitudes. By protecting these weights from
quantization error, AWQ preserves model quality while achieving significant
compression.

Unlike GPTQ which uses second-order (Hessian-based) optimization, AWQ uses a
simpler grid search to find per-channel scales that minimize activation-weighted
quantization error. This makes AWQ generally faster while achieving competitive
accuracy.

### How it works

1. Run a small calibration set through the model to collect per-channel
  activation magnitudes.
2. For each weight matrix, search for optimal per-channel scales that
   minimize activation-weighted quantization error.
3. Multiply weights by the optimal scales before quantization
  (expanding salient weights).
4. Quantize the scaled weights to 4-bit (or other supported bit-width) integers.
5. During inference, dequantize weights and divide by scales to
   restore original magnitude.

The scale formula uses: `scales = activation_max^ratio` where ratio is
searched over a grid from 0 to 1.

Keras supports AWQ quantization for KerasHub models via the
`keras.quantizers.AWQConfig` class.

---
## Load a KerasHub model

This guide uses the `Gemma3CausalLM` model from KerasHub, a small (1B
parameter) causal language model.


```python
import keras
from keras_hub.models import Gemma3CausalLM
from datasets import load_dataset


prompt = "Keras is a"

model = Gemma3CausalLM.from_preset("gemma3_1b")

outputs = model.generate(prompt, max_length=30)
print(outputs)
```

<div class="k-default-codeblock">
```
Keras is a deep learning library for Python. It is a high-level API for neural networks. It is a Python library for deep learning
```
</div>

---
## Configure & run AWQ quantization

You can configure AWQ quantization via the `keras.quantizers.AWQConfig` class.

The AWQ configuration requires a calibration dataset and tokenizer, which it
uses to collect activation statistics and search for optimal scales. Here, we
use a small slice of the WikiText-2 dataset for calibration.

Key parameters:

* `weight_bits`: The bit-width to quantize weights to. AWQ currently only
  supports 4-bit quantization.
* `group_size`: The number of input features to quantize together. Smaller
  groups typically yield better accuracy but may use more memory. Use -1 for
  per-channel (no grouping). A good starting point is 128.
* `num_grid_points`: The number of points to search over when finding optimal
  scales. More points give finer granularity but increase calibration time.
  Default is 20.
* `num_samples`: Number of calibration samples to use for activation
  collection.
* `sequence_length`: Maximum sequence length for calibration samples.

In this example, we first prepare a tiny calibration set, and then run AWQ on
the model using the `.quantize(...)` API.


```python
# Calibration slice (use a larger/representative set in practice)
texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")["text"]

calibration_dataset = [
    s + "." for text in texts for s in map(str.strip, text.split(".")) if s
]

awq_config = keras.quantizers.AWQConfig(
    dataset=calibration_dataset,
    tokenizer=model.preprocessor.tokenizer,
    weight_bits=4,
    group_size=128,
    num_grid_points=20,
    num_samples=128,
    sequence_length=256,
)

model.quantize("awq", config=awq_config)

outputs = model.generate(prompt, max_length=30)
print(outputs)
```

<div class="k-default-codeblock">
```
26/26 ━━━━━━━━━━━━━━━━━━━━ 240s 9s/step

Keras is a Python library for deep learning. It is a high-level interface to the TensorFlow library.

Keras is a great library
```
</div>

---
## Model Export

The AWQ quantized model can be saved to a preset and reloaded elsewhere, just
like any other KerasHub model.


```python
model.save_to_preset("gemma3_awq_w4gs128_preset")
model_from_preset = Gemma3CausalLM.from_preset("gemma3_awq_w4gs128_preset")
output = model_from_preset.generate(prompt, max_length=30)
print(output)
```

<div class="k-default-codeblock">
```
Keras is a Python library for deep learning. It is a high-level interface to the TensorFlow library.

Keras is a great library
```
</div>

---
## Performance & Benchmarking

Micro-benchmarks collected on a single RTX 4070 Ti Super (16 GB).
Baselines are BF16 for Gemma3, and FP32 for Qwen3 and OPT.

Dataset: WikiText-2.


| Model | Pre PPL | Post PPL | PPL Change | Disk Size Change | GPU Mem Change | Throughput Change |
| ----- | ------: | -------: | ---------: | ---------------: | -------------: | ----------------: |
| Qwen3 1.7B | 37.65 | 45.79 | +21.64% | -70.7% | -69.9% | -10.4% |
| Gemma3 1B | 172.45 | 178.03 | +3.23% | -60.2% | -58.3% | -15.5% |
| OPT 125M | 77.06 | 84.75 | +9.97% | -58.3% | -40.9% | -3.3% |


### Analysis

* **Disk size reduction**: 58-71% across models due to 4-bit weight storage.
* **GPU memory reduction**: 41-70% during inference.
* **Perplexity degradation**: +3.2% (Gemma3 1B) to +21.6% (Qwen3 1.7B), model-dependent.
* **Throughput**: -3% to -15% due to dequantization overhead.

AWQ provides substantial memory savings with modest quality degradation,
making it ideal for deploying large models on memory-constrained devices.

---
## AWQ vs GPTQ?

Both AWQ and GPTQ are weight-only quantization methods that require calibration
data. Here's how to choose between them:

| Aspect | AWQ | GPTQ |
| ------ | --- | ---- |
| **Algorithm** | Grid search for activation-aware scales | Hessian-based second-order optimization |
| **Quantization speed** | Faster (no Hessian computation) | Slower (requires Hessian estimation) |
| **Bit-widths supported** | only 4-bit supported for now | 2/3/4/8-bit |
| **Accuracy** | Competitive, especially on encoder models | Often slightly better on decoder LLMs |
| **Memory during quantization** | Lower | Higher (Hessian storage) |
| **Calibration sensitivity** | Less prone to overfitting | May overfit calibration set, affecting out-of-distribution performance |

**Choose AWQ when:**

* You need faster quantization (AWQ is typically 2-3x faster than GPTQ).
* Memory during quantization is constrained.
* 4-bit is sufficient for your use case.
* Your model will be used on diverse/out-of-distribution data (AWQ is less prone to overfitting on calibration data).

**Choose GPTQ when:**

* You need bit-widths other than 4 (e.g., 2-bit or 8-bit).
* Maximum accuracy is critical and you can afford longer quantization time.
* You're working with decoder-only LLMs where GPTQ may have a slight edge.

---
## Practical tips

* AWQ is a post-training technique; training after quantization is not supported.
* Always use the model's own tokenizer for calibration.
* Use a representative calibration set; small slices are only for demos.
* Start with W4 group_size=128; tune per model/task.
* AWQ only supports 4-bit quantization currently.
* For best results, use calibration data that matches your inference domain.

---
## References

* [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
* [MIT HAN Lab AWQ Repository](https://github.com/mit-han-lab/llm-awq)

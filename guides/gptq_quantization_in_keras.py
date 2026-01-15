"""
Title: GPTQ Quantization in Keras
Author: [Jyotinder Singh](https://x.com/Jyotinder_Singh)
Date created: 2025/10/16
Last modified: 2025/10/16
Description: How to run weight-only GPTQ quantization for Keras & KerasHub models.
Accelerator: GPU
"""

"""
## What is GPTQ?

GPTQ ("Generative Pre-Training Quantization") is a post-training, weight-only
quantization method that uses a second-order approximation of the loss (via a
Hessian estimate) to minimize the error introduced when compressing weights to
lower precision, typically 4-bit integers.

Unlike standard post-training techniques, GPTQ keeps activations in
higher-precision and only quantizes the weights. This often preserves model
quality in low bit-width settings while still providing large storage and
memory savings.

Keras supports GPTQ quantization for KerasHub models via the
`keras.quantizers.GPTQConfig` class.
"""

"""
## Load a KerasHub model

This guide uses the `Gemma3CausalLM` model from KerasHub, a small (1B
parameter) causal language model.

"""
import keras
from keras_hub.models import Gemma3CausalLM
from datasets import load_dataset


prompt = "Keras is a"

model = Gemma3CausalLM.from_preset("gemma3_1b")

outputs = model.generate(prompt, max_length=30)
print(outputs)

"""
## Configure & run GPTQ quantization

You can configure GPTQ quantization via the `keras.quantizers.GPTQConfig` class.

The GPTQ configuration requires a calibration dataset and tokenizer, which it
uses to estimate the Hessian and quantization error. Here, we use a small slice
of the WikiText-2 dataset for calibration.

You can tune several parameters to trade off speed, memory, and accuracy. The
most important of these are `weight_bits` (the bit-width to quantize weights to)
and `group_size` (the number of weights to quantize together). The group size
controls the granularity of quantization: smaller groups typically yield better
accuracy but are slower to quantize and may use more memory. A good starting
point is `group_size=128` for 4-bit quantization (`weight_bits=4`).

In this example, we first prepare a tiny calibration set, and then run GPTQ on
the model using the `.quantize(...)` API.
"""

# Calibration slice (use a larger/representative set in practice)
texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")["text"]

calibration_dataset = [
    s + "." for text in texts for s in map(str.strip, text.split(".")) if s
]

gptq_config = keras.quantizers.GPTQConfig(
    dataset=calibration_dataset,
    tokenizer=model.preprocessor.tokenizer,
    weight_bits=4,
    group_size=128,
    num_samples=256,
    sequence_length=256,
    hessian_damping=0.01,
    symmetric=False,
    activation_order=False,
)

model.quantize("gptq", config=gptq_config)

outputs = model.generate(prompt, max_length=30)
print(outputs)

"""
## Model Export

The GPTQ quantized model can be saved to a preset and reloaded elsewhere, just
like any other KerasHub model.
"""

model.save_to_preset("gemma3_gptq_w4gs128_preset")
model_from_preset = Gemma3CausalLM.from_preset("gemma3_gptq_w4gs128_preset")
output = model_from_preset.generate(prompt, max_length=30)
print(output)

"""
## Performance & Benchmarking

Micro-benchmarks collected on a single NVIDIA 4070 Ti Super (16 GB).
Baselines are FP32.

Dataset: WikiText-2.


| Model (preset)                    | Perplexity Increase % (↓ better) | Disk Storage Reduction Δ % (↓ better) | VRAM Reduction Δ % (↓ better) | First-token Latency Δ % (↓ better) | Throughput Δ % (↑ better) |
| --------------------------------- | -------------------------------: | ------------------------------------: | ----------------------------: | ---------------------------------: | ------------------------: |
| GPT2 (gpt2_base_en_cnn_dailymail) |                             1.0% |                              -50.1% ↓ |                      -41.1% ↓ |                            +0.7% ↑ |                  +20.1% ↑ |
| OPT (opt_125m_en)                 |                            10.0% |                              -49.8% ↓ |                      -47.0% ↓ |                            +6.7% ↑ |                  -15.7% ↓ |
| Bloom (bloom_1.1b_multi)          |                             7.0% |                              -47.0% ↓ |                      -54.0% ↓ |                            +1.8% ↑ |                  -15.7% ↓ |
| Gemma3 (gemma3_1b)                |                             3.0% |                              -51.5% ↓ |                      -51.8% ↓ |                           +39.5% ↑ |                   +5.7% ↑ |


Detailed benchmarking numbers and scripts are available
[here](https://github.com/keras-team/keras/pull/21641).

### Analysis

There is notable reduction in disk space and VRAM usage across all models, with
disk space savings around 50% and VRAM savings ranging from 41% to 54%. The
reported disk savings understate the true weight compression because presets
also include non-weight assets.

Perplexity increases only marginally, indicating model quality is largely
preserved after quantization.
"""

"""
## GPTQ vs AWQ?

Both GPTQ and AWQ are weight-only quantization methods that require calibration
data. Here's how to choose between them:

| Aspect | GPTQ | AWQ |
| ------ | ---- | --- |
| **Algorithm** | Hessian-based second-order optimization | Grid search for activation-aware scales |
| **Quantization speed** | Slower (requires Hessian estimation) | Faster (no Hessian computation) |
| **Bit-widths supported** | 2/3/4/8-bit | Only 4-bit supported for now |
| **Accuracy** | Often slightly better on decoder LLMs | Competitive, especially on encoder models |
| **Memory during quantization** | Higher (Hessian storage) | Lower |
| **Calibration sensitivity** | May overfit calibration set, affecting out-of-distribution performance | Less prone to overfitting |

**Choose GPTQ when:**

* You need bit-widths other than 4 (e.g., 2-bit or 8-bit).
* Maximum accuracy is critical and you can afford longer quantization time.
* You're working with decoder-only LLMs where GPTQ may have a slight edge.

**Choose AWQ when:**

* You need faster quantization (AWQ is typically 2-3x faster than GPTQ).
* Memory during quantization is constrained.
* 4-bit is sufficient for your use case.
* Your model will be used on diverse/out-of-distribution data (AWQ is less prone to overfitting on calibration data).
"""

"""
## Practical tips

* GPTQ is a post-training technique; training after quantization is not supported.
* Always use the model's own tokenizer for calibration.
* Use a representative calibration set; small slices are only for demos.
* Start with W4 group_size=128; tune per model/task.
"""

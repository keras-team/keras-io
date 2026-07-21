"""
Title: Exporting Keras models to LiteRT
Author: [Rahul Kumar](https://github.com/pctablet505)
Date created: 2025/12/10
Last modified: 2025/12/10
Description: Learn how to export Keras models to LiteRT for mobile and edge deployment.
Accelerator: None
"""

"""
## Introduction

[LiteRT](https://ai.google.dev/edge/litert) (formerly TensorFlow Lite) lets you run
machine learning models on mobile devices, embedded systems, and browsers with
low latency and small binary size.

This guide shows how to export a Keras model to LiteRT format using the
built-in `model.export()` API, test the exported model, and apply quantization
for smaller file sizes.

We will use **Gemma 3 270M** — a lightweight language model from Google — as our
running example, but the same steps work for any Keras model.

> **Note:** LiteRT export with the Torch backend requires Keras from the
> master branch (not yet in a stable release). Install instructions below.

## Setup

Install Keras from the master branch along with the other required packages:

```shell
pip install -q git+https://github.com/keras-team/keras.git
pip install -q keras-hub ai-edge-litert
```

You can use either the **TensorFlow** or **Torch** backend for export.
Pick one and set it before importing Keras:
"""

import os

# Choose your backend: "tensorflow" or "torch"
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
import keras_hub

print("Keras version:", keras.__version__)
print("KerasHub version:", keras_hub.__version__)

"""
## Export a KerasHub model

Load a pretrained **Gemma 3 270M** causal language model from KerasHub and
export it to LiteRT.
"""

# Load the model and its preprocessor
preset = "gemma3_270m"
preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor.from_preset(
    preset, sequence_length=32
)
model = keras_hub.models.Gemma3CausalLM.from_preset(preset, preprocessor=preprocessor)

print(f"Loaded {preset}")

"""
Run a quick forward pass to make sure weights are built, then export:
"""

# Build weights with a sample input
sample_text = {"prompts": ["Hello"], "responses": ["world"]}
sample_input = preprocessor(sample_text)[0]
_ = model(sample_input)

# Export to LiteRT
model.export("gemma3_270m.tflite", format="litert")
print("Exported to gemma3_270m.tflite")

"""
## Run inference with the LiteRT model

Load the exported `.tflite` file with `ai_edge_litert` and run inference.
"""

from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="gemma3_270m.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape :", input_details[0]["shape"])
print("Output shape:", output_details[0]["shape"])

"""
## Quantization for smaller models

Quantization reduces model size and can speed up inference on edge devices.

### Dynamic range quantization

The simplest approach — quantizes weights to 8-bit integers while keeping
activations in float32. This typically gives **~4× size reduction**.
"""

import tensorflow as tf

model.export(
    "gemma3_270m_dynamic_quant.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
)

print("Exported dynamically quantized model")

"""
### Float16 quantization

Converts weights to 16-bit floats. Gives **~2× size reduction** and is often
faster on GPUs.
"""

model.export(
    "gemma3_270m_float16.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
    target_spec={"supported_types": [tf.float16]},
)

print("Exported Float16 quantized model")

"""
### Advanced quantization with ai-edge-quantizer (recommended for LLMs)

For large language models like Gemma, the built-in TFLite quantization works but
may leave accuracy on the table. The **AI Edge Quantizer** provides
LLM-aware recipes that preserve quality far better for generative models.

Install it:

```shell
pip install -q ai-edge-quantizer
```

Run recipe-based quantization on the exported `.tflite` file:
"""

from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import tfl_interpreter_utils

# Load the exported model
qt = quantizer.Quantizer("gemma3_270m.tflite")

# Use the default recipe optimized for generative (LLM) models
qt.load_full_integer_recipe(
    # Smaller granularity = better accuracy, slightly larger file
    granularity=quantizer.Quantizer.Granularity.CHANNELWISE,
    # Symmetric quantization is usually best for transformers
    symmetric=True,
)


# Calibrate with a few representative samples
# (In production, use real prompts from your validation set)
def calibration_data():
    for _ in range(10):
        yield [np.random.randint(0, 256000, size=(1, 32)).astype(np.int32)]


qt.quantize(calibration_data(), "gemma3_270m_aieq.tflite")

print("Exported ai-edge-quantizer model")

"""
The AI Edge Quantizer typically achieves:
- **~4× size reduction** (similar to dynamic range INT8)
- **Better perplexity / BLEU scores** than plain TFLite quantization on generative tasks
- **Uniform INT8 weights + INT8/INT16 activations** tuned for transformer attention patterns

## Compare file sizes
"""


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


print(f"\nOriginal     : {file_size_mb('gemma3_270m.tflite'):.1f} MB")
print(f"Dynamic INT8 : {file_size_mb('gemma3_270m_dynamic_quant.tflite'):.1f} MB")
print(f"Float16      : {file_size_mb('gemma3_270m_float16.tflite'):.1f} MB")

"""
## Exporting a custom Keras model

The same API works for any Keras model. Here is a quick example with a simple
classifier:
"""

# Build a small custom model
classifier = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

classifier.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train briefly on dummy data for the demo
x = np.random.random((256, 28, 28))
y = np.random.randint(0, 10, 256)
classifier.fit(x, y, epochs=1, verbose=0)

# Export
classifier.export("classifier.tflite", format="litert")
print("\nCustom classifier exported to classifier.tflite")

"""
## Best practices

1. **Always test the exported model** before deploying — run inference and
   compare outputs with the original Keras model.
2. **Start with dynamic range quantization** for the best size/accuracy trade-off.
3. **Use Float16** when targeting GPU-accelerated inference.
4. **Build subclassed models before exporting** — call them on sample data so
   Keras knows the input signature.

## Troubleshooting

| Issue | Solution |
|---|---|
| `ImportError` for `ai_edge_litert` | Run `pip install ai-edge-litert` |
| Shape mismatch at inference | Verify the input shape matches what the model expects |
| Unsupported ops | Add `target_spec={"supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]}` |
| Subclassed model fails to export | Call the model on sample data before `export()` to build weights |
| Out of memory during export | Try quantization or export on a machine with more RAM |

"""

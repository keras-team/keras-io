"""
Title: Exporting Keras models to LiteRT
Author: [Rahul Kumar](https://github.com/pctablet505)
Date created: 2025/12/10
Last modified: 2026/06/02
Description: Learn how to export Keras models to LiteRT for mobile and edge deployment using the PyTorch backend.
Accelerator: None
"""

"""
## Introduction

[LiteRT](https://ai.google.dev/edge/litert) (formerly TensorFlow Lite) lets you run
machine learning models on mobile devices, embedded systems, and browsers with
low latency and small binary size.

Keras provides a built-in `model.export()` API that converts your model to the
LiteRT (`.tflite`) format in a single line of code. This guide covers the
complete workflow:

1. Export a Keras model to LiteRT.
2. Run inference with the LiteRT interpreter.
3. Handle different model types (Sequential, Functional, subclassed, multi-input).
4. Resize inputs at runtime.
5. Use the signature runner for named inputs and outputs.
6. Quantize the model for smaller size and faster inference.

We will use the **PyTorch backend** because it produces models that are fully
compatible with the new LiteRT Android runtime — no flex ops, no deprecated
interpreter APIs. The same `model.export(..., format="litert")` API also works
with the TensorFlow backend, which we summarize in a dedicated section at the
end.

## Setup

Install the required packages. Since LiteRT export is under active
development, install Keras and KerasHub from the master branch to get the
latest fixes:

```shell
pip install -q git+https://github.com/keras-team/keras.git
pip install -q git+https://github.com/keras-team/keras-hub.git
pip install -q ai-edge-litert
```

> **Note:** LiteRT export with the PyTorch backend requires `litert-torch`.
> If it is not already installed, run:
> ```shell
> pip install -q litert-torch
> ```

Set the PyTorch backend before importing Keras:
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras

print("Keras version:", keras.__version__)

"""
## Export a simple model

Build a small classifier and export it to LiteRT.
"""

# Build a simple model
model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu", input_shape=(10,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Build weights with a sample input
sample_input = np.random.random((1, 10)).astype("float32")
_ = model(sample_input)

# Export to LiteRT
model.export("model.tflite", format="litert")
print("Exported to model.tflite")

"""
## Run inference with the LiteRT model

Load the exported `.tflite` file with `ai_edge_litert` and run inference.

> **Important:** `tf.lite.Interpreter` is deprecated and scheduled for deletion.
> Always use `ai_edge_litert.interpreter.Interpreter` for new code.
"""

from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape :", input_details[0]["shape"])
print("Output shape:", output_details[0]["shape"])

interpreter.set_tensor(input_details[0]["index"], sample_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]["index"])
print("Inference output shape:", output.shape)

"""
## Working with different model types

### Subclassed models

For subclassed models, you must call the model on sample data before export so
that Keras can infer the input signature and build the weights.
"""


class TinyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(16, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, x):
        return self.dense2(self.dense1(x))


subclass_model = TinyModel()
subclass_model(np.zeros((1, 10), dtype="float32"))
subclass_model.export("subclass_model.tflite", format="litert")
print("Subclassed model exported")

"""
### Models with multiple inputs
"""

input_a = keras.Input(shape=(10,), name="input_a")
input_b = keras.Input(shape=(10,), name="input_b")
merged = keras.layers.Concatenate()([input_a, input_b])
output = keras.layers.Dense(1)(merged)
multi_input_model = keras.Model(inputs=[input_a, input_b], outputs=output)

a = np.random.random((1, 10)).astype("float32")
b = np.random.random((1, 10)).astype("float32")
_ = multi_input_model([a, b])

multi_input_model.export("multi_input_model.tflite", format="litert")
print("Multi-input model exported")

"""
### Models with dictionary inputs

Dictionary inputs are also supported natively.
"""

input_x = keras.Input(shape=(10,), name="x")
input_y = keras.Input(shape=(10,), name="y")
sum_output = keras.layers.Add()([input_x, input_y])
dict_model = keras.Model(inputs={"x": input_x, "y": input_y}, outputs=sum_output)

x_val = np.random.random((1, 10)).astype("float32")
y_val = np.random.random((1, 10)).astype("float32")
_ = dict_model({"x": x_val, "y": y_val})

dict_model.export("dict_model.tflite", format="litert")
print("Dict-input model exported")

"""
## Dynamic shapes and runtime input resizing

LiteRT models exported from Keras have static shapes in the graph, but the
interpreter supports **runtime input resizing**. This means you can process
different batch sizes with the same exported model without re-exporting.

> **Note:** The PyTorch backend also accepts a `dynamic_shapes` kwarg powered
> by `torch.export.Dim`. However, this path currently has limitations in the
> Keras → LiteRT conversion pipeline for some layer types. The recommended
> approach for variable batch sizes is runtime resizing, which is fully
> supported and tested.
"""

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

# Resize from batch=1 to batch=4
interpreter.resize_tensor_input(input_details[0]["index"], [4, 10])
interpreter.allocate_tensors()

batch_input = np.random.random((4, 10)).astype("float32")
interpreter.set_tensor(input_details[0]["index"], batch_input)
interpreter.invoke()
resized_output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
print("Resized output shape:", resized_output.shape)

"""
## Signature runner

Use the signature runner for cleaner inference code. Input names in the
signature come from the model's export format; you can discover them via
`get_signature_list()`.
"""

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Check available signatures
print("Signatures:", interpreter.get_signature_list())

# Run inference using the signature runner
runner = interpreter.get_signature_runner("serving_default")
sig_output = runner(args_0=sample_input)
print("Signature runner output shape:", list(sig_output.values())[0].shape)

"""
## Custom input signature

You can override the inferred input signature by passing `input_signature` to
`export()`. This is useful when you want to enforce specific shapes or dtypes.

> **Note:** `None` in an `InputSpec` shape (e.g., `shape=(None, 10)`) tells
> Keras that the dimension is unknown, but the exported LiteRT graph still
> uses a concrete batch size (1) for tracing. Use runtime resizing (shown
> above) to handle variable batch sizes at inference time.
"""

model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu", input_shape=(10,)),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile()
model(np.zeros((1, 10), dtype="float32"))

custom_sig = [keras.layers.InputSpec(shape=(None, 10), dtype="float32")]
model.export("custom_sig_model.tflite", format="litert", input_signature=custom_sig)
print("Exported with custom input signature")

"""
## Export a KerasHub model

The same API works for KerasHub presets. Here we export **Gemma 3 270M**,
a lightweight language model that is small enough to run on edge devices.
"""

import keras_hub

preset = "gemma3_270m"
preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor.from_preset(
    preset, sequence_length=32
)
model = keras_hub.models.Gemma3CausalLM.from_preset(preset, preprocessor=preprocessor)

print(f"Loaded {preset}")

# Build weights with a sample input. Use generate_preprocess for inference.
sample_text = {"prompts": ["Hello"]}
sample_input = preprocessor.generate_preprocess(sample_text)
_ = model(sample_input)

model.export("gemma3_270m.tflite", format="litert")
print("Exported to gemma3_270m.tflite")

"""
## Quantization for smaller models

Quantization reduces model size and can speed up inference on edge devices.

### Built-in dynamic range quantization

Pass `optimizations` to `model.export()`. This works on both the TensorFlow and
PyTorch backends and quantizes weights to 8-bit integers while keeping
activations in float32. This typically gives **~4× size reduction**.
"""

import tensorflow as tf

model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu", input_shape=(10,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile()
model(np.zeros((1, 10), dtype="float32"))

model.export(
    "model_dynamic_quant.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
)
print("Exported dynamically quantized model")

"""
### Post-export quantization with ai-edge-quantizer

For finer-grained control (e.g. channel-wise symmetric INT8 weights, mixed
precision, or per-layer recipes), use the **AI Edge Quantizer** on the
already-exported `.tflite` file. This works consistently across both backends.

Install it:

```shell
pip install -q ai-edge-quantizer
```
"""

from ai_edge_quantizer import quantizer, qtyping

qt = quantizer.Quantizer("model.tflite")

# Recipe: channel-wise symmetric INT8 weights, FP32 activations.
recipe = [
    {
        "regex": ".*",
        "operation": qtyping.TFLOperationName.ALL_SUPPORTED,
        "algorithm_key": quantizer.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        "op_config": {
            "weight_tensor_config": {
                "dtype": qtyping.TensorDataType.INT,
                "num_bits": 8,
                "granularity": qtyping.QuantGranularity.CHANNELWISE,
                "symmetric": True,
            },
            "compute_precision": qtyping.ComputePrecision.FLOAT,
            "explicit_dequantize": False,
        },
    },
]

qt.load_quantization_recipe(recipe)

# No calibration needed for this weight-only recipe.
quant_result = qt.quantize()
quant_result.save(".", model_name="model_aieq")

print("Exported ai-edge-quantizer model")

"""
The AI Edge Quantizer typically achieves:
- **Similar file size** to TFLite dynamic-range quantization
- **Better perplexity / BLEU scores** on generative tasks thanks to
  channel-wise symmetric scaling
- **No calibration required** for weight-only recipes

## Compare file sizes
"""


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


print(f"\nOriginal     : {file_size_mb('model.tflite'):.1f} MB")
print(f"Dynamic INT8 : {file_size_mb('model_dynamic_quant.tflite'):.1f} MB")
print(f"AI Edge Quant: {file_size_mb('model_aieq.tflite'):.1f} MB")

"""
## Android dependencies

To use a LiteRT model in an Android app, add one of the following dependencies
to your `build.gradle`:

### New LiteRT runtime (recommended)

Use this for models exported with the PyTorch backend. No flex ops are
required.

```groovy
implementation 'com.google.ai.edge.litert:litert:2.1.5'
```

### Legacy TensorFlow Lite runtime

Use this only if you have older `.tflite` models that still require it.
Note that `tf.lite.Interpreter` is deprecated.

```groovy
implementation 'org.tensorflow:tensorflow-lite:2.16.1'
```

If your model contains flex ops (e.g. from TF backend export), also add:

```groovy
implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1'
```

Models exported with the **PyTorch backend do not contain flex ops**, so the
`select-tf-ops` dependency is not needed.

## TensorFlow backend alternative

The same `model.export(..., format="litert")` API works with the TensorFlow
backend. The differences are:

1. **Converter path:** Uses `tf.lite.TFLiteConverter.from_saved_model()`.
2. **Flex ops:** Enabled by default via `SELECT_TF_OPS`. These ops are not
   supported by the new LiteRT Android runtime.
3. **Interpreter:** The legacy `tf.lite.Interpreter` is deprecated. Use
   `ai_edge_litert.interpreter.Interpreter` for new code.
4. **Extra kwargs:** `target_spec` and `representative_dataset` are supported.

Example:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(10,)),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile()
model(np.zeros((1, 10), dtype="float32"))

# Export with Float16 quantization
model.export(
    "model_f16.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
    target_spec={"supported_types": [tf.float16]},
)
```

## Backend comparison

| Feature | TensorFlow backend | PyTorch backend (recommended) |
|---|---|---|
| Converter path | `tf.lite.TFLiteConverter` | `litert-torch` via `ExportedProgram` |
| Flex ops | Enabled by default | **Not generated** |
| Android runtime | May crash on new LiteRT | Fully compatible |
| Interpreter | `tf.lite.Interpreter` (deprecated) | `ai_edge_litert.interpreter.Interpreter` |
| `optimizations` kwarg | Supported | Supported |
| `target_spec` kwarg | Supported | **Not supported** |
| `representative_dataset` | Supported | **Not supported** |
| Post-export `ai-edge-quantizer` | Supported | **Supported** |
| Runtime input resizing | Supported | **Supported** |

## Best practices

1. **Always test the exported model** before deploying — run inference and
   compare outputs with the original Keras model.
2. **Use the PyTorch backend** for LiteRT export to avoid flex ops and ensure
   compatibility with the new LiteRT Android runtime.
3. **Call subclassed models on sample data** before `export()` to build weights
   and infer the input signature.
4. **Use runtime input resizing** for variable batch sizes rather than
   `dynamic_shapes`, which currently has limitations in the Keras → LiteRT
   pipeline.
5. **Start with `optimizations=[tf.lite.Optimize.DEFAULT]`** for quick
   dynamic-range quantization on either backend.
6. **Use `ai-edge-quantizer`** when you need finer control (channel-wise,
   symmetric, mixed-precision) or when targeting the PyTorch backend with
   Float16.
7. **Keep models under 2 GB per TFLite file** — LiteRT uses a flatbuffer format
   with a 2 GB limit per file. If your model is larger, use the
   `litert-lm` / `litert-lm-builder` pipeline which shards the model into
   multiple sub-models inside a `.litertlm` container.

## Troubleshooting

| Issue | Solution |
|---|---|
| `ImportError` for `ai_edge_litert` | Run `pip install ai-edge-litert` |
| `ImportError` for `litert_torch` | Run `pip install litert-torch` |
| Shape mismatch at inference | Verify the input shape matches what the model expects, or use `resize_tensor_input()` |
| Subclassed model fails to export | Call the model on sample data before `export()` to build weights |
| Unsupported ops on PyTorch | Some ops may not yet be supported by `litert-torch`; try TF backend or simplify the model |
| Out of memory during export | Try quantization or export on a machine with more RAM |
| Flex ops on Android | Re-export with `KERAS_BACKEND=torch` |

"""

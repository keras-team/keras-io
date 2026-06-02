# Exporting Keras models to LiteRT

**Author:** [Rahul Kumar](https://github.com/pctablet505)<br>
**Date created:** 2025/12/10<br>
**Last modified:** 2026/06/02<br>
**Description:** Learn how to export Keras models to LiteRT for mobile and edge deployment using the PyTorch backend.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/litert_export.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/litert_export.py)



---
## Introduction

[LiteRT](https://ai.google.dev/edge/litert) (formerly TensorFlow Lite) lets you run
machine learning models on mobile devices, embedded systems, and browsers with
low latency and small binary size.

This guide shows how to export a Keras model to LiteRT format using the
built-in `model.export()` API, run inference with the new LiteRT interpreter,
and apply quantization for smaller file sizes.

We recommend the **PyTorch backend** for LiteRT export because:

1. **No flex ops** — the TensorFlow backend path uses
   `tf.lite.TFLiteConverter.from_saved_model()`, which enables
   `SELECT_TF_OPS` (flex ops) by default. Flex ops are not supported by the
   new LiteRT Android runtime and the `ai_edge_litert` interpreter.
2. **Future-proof interpreter** — `tf.lite.Interpreter` is deprecated and
   scheduled for removal. The new `ai_edge_litert.interpreter.Interpreter` is
   the supported path, and it requires models without flex ops.
3. **Native PyTorch-to-LiteRT pipeline** — `litert-torch` converts Keras models
   through PyTorch's `ExportedProgram` directly to the LiteRT flatbuffer format.

We will use a tiny transformer as our runnable example, but the same steps work
for any Keras model — including **Gemma 3 270M**, a lightweight language model
that is small enough to export quickly and run on edge devices.

---
## Setup

Install the required packages:

```shell
pip install -q keras keras-hub ai-edge-litert
```

> **Note:** LiteRT export with the PyTorch backend requires `litert-torch`.
> If it is not already installed, run:
> ```shell
> pip install -q litert-torch
> ```

Set the PyTorch backend before importing Keras:


```python
import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras

print("Keras version:", keras.__version__)
```

<div class="k-default-codeblock">
```
Keras version: 3.15.0
```
</div>

---
## Export a Keras model

Build a tiny transformer and export it to LiteRT. The same API works for
KerasHub presets such as `gemma3_270m`.


```python
vocab_size = 256
seq_length = 8
embed_dim = 16
num_heads = 2
dim_feedforward = 32
num_layers = 2

inputs = keras.Input(shape=(seq_length,), dtype="int32")
x = keras.layers.Embedding(vocab_size, embed_dim)(inputs)
for _ in range(num_layers):
    attn = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads
    )(x, x)
    x = keras.layers.LayerNormalization()(x + attn)
    ffn = keras.layers.Dense(dim_feedforward, activation="relu")(x)
    ffn = keras.layers.Dense(embed_dim)(ffn)
    x = keras.layers.LayerNormalization()(x + ffn)
outputs = keras.layers.Dense(vocab_size)(x)
model = keras.Model(inputs, outputs)
model.compile()

# Build weights with a sample input
sample_input = np.zeros((1, seq_length), dtype="int32")
_ = model(sample_input)

# Export to LiteRT
model.export("model.tflite", format="litert")
print("Exported to model.tflite")
```

<div class="k-default-codeblock">
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1780412818.641443 1649431 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
```
</div>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:00)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:00)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Torch Export: serving_default</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:00)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Torch Export: serving_default</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">ExportedProgram Run Decompositions</span>
</pre>



<div class="k-default-codeblock">
```
/usr/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
```
</div>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:00)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Torch Export: serving_default</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">ExportedProgram Run Decompositions</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:00)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Torch Export: serving_default</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:00)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Run FX Passes</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Run FX Passes</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">ExportedProgram Run Decompositions</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Run FX Passes</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">ExportedProgram Run Decompositions</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Run FX Passes</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Lower to MLIR: serving_default</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Lower to MLIR: serving_default</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">ExportedProgram Run Decompositions</span>
</pre>



<div class="k-default-codeblock">
```
/usr/lib/python3.13/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
```
</div>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Lower to MLIR: serving_default</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">ExportedProgram Run Decompositions</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Lower to MLIR: serving_default</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">ExportedProgram Run Decompositions</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Lower to MLIR: serving_default</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">ExportedProgram Run Decompositions</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Lower to MLIR: serving_default</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Create MLIR Module</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Lower to MLIR: serving_default</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Create MLIR Module</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Lower to MLIR: serving_default</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>



<div class="k-default-codeblock">
```
/home/pctablet505/Projects/gemmademo-litert-export/.venv/lib/python3.13/site-packages/litert_torch/_convert/signature.py:52: FutureWarning: `treespec.children_specs` is deprecated. Use `treespec.child(index)` to access a single child, or `treespec.children()` to get all children.
  args_spec, kwargs_spec = spec.children_specs
/home/pctablet505/Projects/gemmademo-litert-export/.venv/lib/python3.13/site-packages/litert_torch/_convert/signature.py:58: FutureWarning: `treespec.children_specs` is deprecated. Use `treespec.child(index)` to access a single child, or `treespec.children()` to get all children.
  kwargs_spec.children_specs, kwargs_spec.context
```
</div>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Merge MLIR Modules</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Merge MLIR Modules</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">LiteRT-Torch Convert</span> &gt; <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Run LiteRT Converter Passes</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> &gt; </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Run LiteRT Converter Passes</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:01)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">LiteRT-Torch Convert</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:01</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:00)</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">[START]</span> <span style="color: #000000; text-decoration-color: #000000; font-weight: bold">Write Model to model.tflite</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(00:00)</span> <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">[ DONE]</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">Write Model to model.tflite</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">(</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">+</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">00:00</span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-weight: bold">)</span>
</pre>



<div class="k-default-codeblock">
```
Saved artifact at 'model.tflite'.

Exported to model.tflite
```
</div>

---
## Run inference with the LiteRT model

Load the exported `.tflite` file with `ai_edge_litert` and run inference.

> **Important:** `tf.lite.Interpreter` is deprecated and scheduled for deletion.
> Always use `ai_edge_litert.interpreter.Interpreter` for new code.


```python
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
```

<div class="k-default-codeblock">
```
Input shape : [1 8]
Output shape: [  1   8 256]
Inference output shape: (1, 8, 256)

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```
</div>

---
## Quantization for smaller models

Quantization reduces model size and can speed up inference on edge devices.

### Post-export quantization with ai-edge-quantizer (recommended)

For the PyTorch backend, we recommend quantizing the model **after** export
using the **AI Edge Quantizer**. It provides finer-grained control (e.g.
channel-wise symmetric INT8 weights) that preserves quality better for
generative models, and does not require passing `optimizations` to the
exporter.

Install it:

```shell
pip install -q ai-edge-quantizer
```

Quantize the already-exported `.tflite` file:


```python
from ai_edge_quantizer import quantizer, qtyping

qt = quantizer.Quantizer("model.tflite")

# Recipe: channel-wise symmetric INT8 weights, FP32 activations.
# This is similar to dynamic-range quantization but uses channel-wise
# scaling which is more accurate for transformer attention weights.
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
```

<div class="k-default-codeblock">
```
Exported ai-edge-quantizer model
```
</div>

The AI Edge Quantizer typically achieves:
- **Similar file size** to TFLite dynamic-range quantization
- **Better perplexity / BLEU scores** on generative tasks thanks to
  channel-wise symmetric scaling
- **No calibration required** for weight-only recipes

### Why not use `optimizations` with the PyTorch backend?

Passing `optimizations=[tf.lite.Optimize.DEFAULT]` to `model.export()` works on
the **TensorFlow** backend but can fail on the **PyTorch** backend due to
differences in how `litert-torch` handles quantized dtypes during the
MLIR pipeline. Until this is fully stabilized, post-export quantization with
`ai-edge-quantizer` is the safer and more accurate path.

---
## Compare file sizes


```python

def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


print(f"\nOriginal     : {file_size_mb('model.tflite'):.1f} MB")
print(f"AI Edge Quant: {file_size_mb('model_aieq.tflite'):.1f} MB")
```

    
<div class="k-default-codeblock">
```
Original     : 0.1 MB
AI Edge Quant: 0.1 MB
```
</div>

---
## Exporting a KerasHub model

The same API works for any Keras model, including KerasHub presets.
Here is how you would export **Gemma 3 270M**:

```python
import keras_hub

preset = "gemma3_270m"
preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor.from_preset(
    preset, sequence_length=32
)
model = keras_hub.models.Gemma3CausalLM.from_preset(
    preset, preprocessor=preprocessor
)

sample_text = {"prompts": ["Hello"], "responses": ["world"]}
sample_input = preprocessor(sample_text)[0]
_ = model(sample_input)

model.export("gemma3_270m.tflite", format="litert")
```

---
## Backend comparison

| Feature | TensorFlow backend | PyTorch backend (recommended) |
|---|---|---|
| Converter path | `tf.lite.TFLiteConverter` | `litert-torch` via `ExportedProgram` |
| Flex ops | Enabled by default | **Not generated** |
| Android runtime | May crash on new LiteRT | Fully compatible |
| Interpreter | `tf.lite.Interpreter` (deprecated) | `ai_edge_litert.interpreter.Interpreter` |
| Quantization via `optimizations` | Supported | Can fail for some models |
| Post-export `ai-edge-quantizer` | Supported | **Supported** |

---
## Best practices

1. **Always test the exported model** before deploying — run inference and
   compare outputs with the original Keras model.
2. **Use the PyTorch backend** for LiteRT export to avoid flex ops and ensure
   compatibility with the new LiteRT Android runtime.
3. **Quantize with `ai-edge-quantizer`** after export for the best size/accuracy
   trade-off, especially for LLMs.
4. **Build subclassed models before exporting** — call them on sample data so
   Keras knows the input signature.
5. **Keep models under 2 GB per TFLite file** — LiteRT uses a flatbuffer format
   with a 2 GB limit per file. If your model is larger, use the
   `litert-lm` / `litert-lm-builder` pipeline which shards the model into
   multiple sub-models inside a `.litertlm` container.

---
## Troubleshooting

| Issue | Solution |
|---|---|
| `ImportError` for `ai_edge_litert` | Run `pip install ai-edge-litert` |
| `ImportError` for `litert_torch` | Run `pip install litert-torch` |
| Shape mismatch at inference | Verify the input shape matches what the model expects |
| Subclassed model fails to export | Call the model on sample data before `export()` to build weights |
| Out of memory during export | Try quantization or export on a machine with more RAM |
| Quantization via `optimizations` fails on PyTorch | Use `ai-edge-quantizer` as a post-export step instead |
| Flex ops on Android | Re-export with `KERAS_BACKEND=torch` |

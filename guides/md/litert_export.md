# Exporting Keras models to LiteRT (TensorFlow Lite)

**Author:** [Rahul Kumar](https://github.com/pctablet505)<br>
**Date created:** 2025/12/10<br>
**Last modified:** 2025/12/10<br>
**Description:** Complete guide to exporting Keras models for mobile and edge deployment.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/litert_export.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/litert_export.py)



---
## Introduction

LiteRT is a solution for running machine learning models
on mobile and edge devices. This guide covers everything you need to know about
exporting Keras models to LiteRT format, including:

- Basic model export
- Different model architectures (Sequential, Functional, Subclassed)
- Quantization for smaller models
- Dynamic shapes support
- Custom input signatures
- Model validation and testing

---
## Setup

First, let's install the required packages and set up the environment.

### Installation

Install the required packages:
```
pip install -q keras tensorflow ai-edge-litert
```

For KerasHub models (optional):
```
pip install -q keras-hub
```



```python
import os

# Set Keras backend to TensorFlow for LiteRT export
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)
```

<div class="k-default-codeblock">
```
Keras version: 3.13.1
TensorFlow version: 2.20.0
```
</div>

---
## Basic Model Export

Let's start with a simple MNIST classifier and export it to LiteRT format.


```python
# Create a simple MNIST classifier
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Generate dummy data for demonstration
x_train = np.random.random((1000, 28, 28))
y_train = np.random.randint(0, 10, 1000)

# Quick training (just for demonstration)
model.fit(x_train, y_train, epochs=1, verbose=0)

print("Model created and trained")
```

<div class="k-default-codeblock">
```
/usr/local/google/home/hellorahul/projects/keras-io/venv/lib/python3.12/site-packages/keras/src/layers/reshaping/flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)

Model created and trained
```
</div>

Now let's export the model to LiteRT format. The `format="litert"` parameter
tells Keras to export in TensorFlow Lite format.


```python
# Export to LiteRT
model.export("mnist_classifier.tflite", format="litert")

print("Model exported to mnist_classifier.tflite")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /tmp/tmpdzrcae2c/assets

INFO:tensorflow:Assets written to: /tmp/tmpdzrcae2c/assets

Saved artifact at '/tmp/tmpdzrcae2c'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name='keras_tensor')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  139725986278352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725986281616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725986280272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725986279120: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'mnist_classifier.tflite'.

Model exported to mnist_classifier.tflite

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1769672030.537569 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672030.537609 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
I0000 00:00:1769672030.545038 2222654 mlir_graph_optimization_pass.cc:437] MLIR V1 optimization pass is not enabled
```
</div>

---
## Testing the Exported Model

Let's verify the exported model works correctly.


```python
# Load and test the exported model
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="mnist_classifier.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nModel Input Details:")
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Type: {input_details[0]['dtype']}")

print("\nModel Output Details:")
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Type: {output_details[0]['dtype']}")

# Test inference
test_input = np.random.random(input_details[0]["shape"]).astype(np.float32)
interpreter.set_tensor(input_details[0]["index"], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]["index"])

print(f"\nInference successful! Output shape: {output.shape}")
```

    
<div class="k-default-codeblock">
```
Model Input Details:
  Shape: [ 1 28 28]
  Type: <class 'numpy.float32'>

Model Output Details:
  Shape: [ 1 10]
  Type: <class 'numpy.float32'>

Inference successful! Output shape: (1, 10)

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```
</div>

---
## Exporting Different Model Types

Keras supports various model architectures. Let's explore how to export them.

### Functional API Models

Functional API models offer more flexibility than Sequential models.


```python
from keras.layers import Input, Dense, concatenate

# Create functional model with multiple inputs
input_a = Input(shape=(32,))
input_b = Input(shape=(32,))

shared_dense = Dense(64, activation="relu")

processed_a = shared_dense(input_a)
processed_b = shared_dense(input_b)

concatenated = concatenate([processed_a, processed_b])
output = Dense(1, activation="sigmoid")(concatenated)

functional_model = keras.Model(inputs=[input_a, input_b], outputs=output)

# Compile and export
functional_model.compile(optimizer="adam", loss="binary_crossentropy")
functional_model.export("functional_model.tflite", format="litert")

print("Functional model exported")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /tmp/tmptibi1cec/assets

INFO:tensorflow:Assets written to: /tmp/tmptibi1cec/assets

Saved artifact at '/tmp/tmptibi1cec'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): List[TensorSpec(shape=(None, 32), dtype=tf.float32, name='keras_tensor_5'), TensorSpec(shape=(None, 32), dtype=tf.float32, name='keras_tensor_6')]
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  139725985121680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725985132432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725985131472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725985131280: TensorSpec(shape=(), dtype=tf.resource, name=None)

W0000 00:00:1769672030.848568 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672030.848599 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.

Saved artifact at 'functional_model.tflite'.

Functional model exported
```
</div>

### Subclassed Models

For complex architectures that require custom forward passes.


```python

class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(32, activation="relu")
        self.output_layer = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


subclassed_model = CustomModel()
subclassed_model.compile(optimizer="adam", loss="binary_crossentropy")

# Call the model to build it
dummy_input = np.random.random((1, 16))
_ = subclassed_model(dummy_input)

subclassed_model.export("subclassed_model.tflite", format="litert")

print("Subclassed model exported")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /tmp/tmpuik9r3ez/assets

INFO:tensorflow:Assets written to: /tmp/tmpuik9r3ez/assets

Saved artifact at '/tmp/tmpuik9r3ez'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 16), dtype=tf.float32, name=None)
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  139725985133968: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725985135504: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725985135888: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725985136080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725292734224: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725292733264: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'subclassed_model.tflite'.

Subclassed model exported

W0000 00:00:1769672031.165668 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672031.165710 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
```
</div>

---
## KerasHub Models

KerasHub provides pretrained models for various tasks. Let's export some.


```python
import keras_hub

# Load a pretrained text model
# Sequence length is configured via the preprocessor
preprocessor = keras_hub.models.BertMaskedLMPreprocessor.from_preset(
    "bert_tiny_en_uncased", sequence_length=128
)

bert_model = keras_hub.models.BertMaskedLM.from_preset(
    "bert_tiny_en_uncased", preprocessor=preprocessor, load_weights=False
)

# Export to LiteRT (sequence length already set)
bert_model.export("bert_tiny_en_uncased.tflite", format="litert")

print("Exported Keras-Hub BERT Tiny model")
```

<div class="k-default-codeblock">
```
Creating adapter for inputs: ['mask_positions', 'padding_mask', 'segment_ids', 'token_ids']

INFO:tensorflow:Assets written to: /tmp/tmp_eujk5s1/assets

INFO:tensorflow:Assets written to: /tmp/tmp_eujk5s1/assets

Saved artifact at '/tmp/tmp_eujk5s1'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): List[TensorSpec(shape=(None, None), dtype=tf.int32, name='mask_positions'), TensorSpec(shape=(None, None), dtype=tf.int32, name='padding_mask'), TensorSpec(shape=(None, None), dtype=tf.int32, name='segment_ids'), TensorSpec(shape=(None, None), dtype=tf.int32, name='token_ids')]
Output Type:
  TensorSpec(shape=(None, None, 30522), dtype=tf.float32, name=None)
Captures:
  139723678962640: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678962448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678964176: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678963024: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678964560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678963216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678964944: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678963600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678965136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678964752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678965712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678962832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678966288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678967248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678966480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678967056: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678968016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678965328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678966672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678963408: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678961104: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678967440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678968208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678967632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678968976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678968592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678968400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678966096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678969168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678970896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678970128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678970704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678971664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678969360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678970320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678969936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678969744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678971088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678971856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678970512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678972624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678973584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678962256: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723678972048: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'bert_tiny_en_uncased.tflite'.

Exported Keras-Hub BERT Tiny model

W0000 00:00:1769672034.783553 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672034.783583 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
```
</div>

For vision models, the image size is determined by the preset:


```python
# Load a vision model
vision_model = keras_hub.models.ImageClassifier.from_preset("resnet_50_imagenet")

# Export (image size already set by preset)
vision_model.export("resnet.tflite", format="litert")

print("Exported Keras-Hub vision model")

# Load an object detection model
# Image size is determined by the preset
object_detector = keras_hub.models.ObjectDetector.from_preset(
    "retinanet_resnet50_fpn_coco"
)

object_detector.export("detector.tflite", format="litert")

print("Exported Keras-Hub object detector")

```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /tmp/tmpfr31_web/assets

INFO:tensorflow:Assets written to: /tmp/tmpfr31_web/assets

Saved artifact at '/tmp/tmpfr31_web'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name='keras_tensor_26')
Output Type:
  TensorSpec(shape=(None, 1000), dtype=tf.float32, name=None)
Captures:
  139723676578832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723676579408: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276788816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723676579984: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723676579792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276785936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276786128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276788432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276786512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276785744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276787472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276785552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276786320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276788624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276784592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276785168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276787856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276786896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276786704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276787088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276789584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276784976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276784784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276787664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276783632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276784208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276784016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276783824: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276788048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276782672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276783248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276783056: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276782864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276785360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276781712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276782288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276782096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276781904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276784400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276794960: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276795728: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276795344: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276795152: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276783440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276781328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276794576: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276794384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276795536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276782480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276793040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276793616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276793424: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276793232: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276794192: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276792272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276779600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276779792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276779984: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276793808: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276781136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276780560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276792080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276781520: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276780752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276780368: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276792848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276794000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276792656: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276792464: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276794768: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276780176: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275961360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275961168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723276780944: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275961936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275960592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275961552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275961744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275960784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275962896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275962320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275962512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275962704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275960400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275963856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275963280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275963472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275963664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275960976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275964816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275964240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275964432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275964624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275962128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275965776: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275965200: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275965392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275965584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275963088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275966736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275966160: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275966352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275966544: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275964048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275967696: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275967120: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275967312: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275967504: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275965008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275968656: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275968080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275968272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275968464: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275965968: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275969616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275969040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275969232: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275969424: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275966928: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275970576: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275970960: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275971152: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275971344: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275968848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275972496: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275971920: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275972112: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275972304: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275969808: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275973456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275972880: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275970000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275970192: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275970384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275967888: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275971536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275973072: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275973264: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275970768: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275974416: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275973840: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275974032: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275974224: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275971728: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275975376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275974800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275974992: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275975184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275972688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275976336: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275975760: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275974608: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275976144: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275975952: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275975568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275973648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274831248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274831440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723275976528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274830288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274829904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274831056: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274830864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274830672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274832400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274831824: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274832016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274832208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274830096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274833360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274832784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274832976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274833168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274830480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274834320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274833744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274833936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274834128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274831632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274835280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274834704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274834896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274835088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274832592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274836240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274835664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274835856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274836048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274833552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274837200: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274836624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274836816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274837008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274834512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274838160: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274837584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274837776: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274837968: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274835472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274839120: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274838544: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274838736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274838928: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274836432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274840080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274839504: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274839696: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274839888: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274837392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274841040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274840464: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274840656: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274840848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274838352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274842000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274842384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274842576: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274842768: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274840272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274843920: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274843344: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274843536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274843728: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274841232: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274844880: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274844304: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274841424: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274841616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274841808: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274839312: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274842960: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274844496: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274844688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274842192: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274845840: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274845264: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274844112: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274845648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274845456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274845072: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274843152: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273733520: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273733712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723274846032: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273732560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273732176: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273733328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273733136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273732944: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273734672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273734096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273734288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273734480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273732368: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273735632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273735056: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273735248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273735440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273732752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273736592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273736016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273736208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273736400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273733904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273737552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723676579600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723676579216: TensorSpec(shape=(), dtype=tf.resource, name=None)

W0000 00:00:1769672042.760344 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672042.760375 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.

Saved artifact at 'resnet.tflite'.

Exported Keras-Hub vision model

INFO:tensorflow:Assets written to: /tmp/tmp8oz0f9gh/assets

INFO:tensorflow:Assets written to: /tmp/tmp8oz0f9gh/assets

Saved artifact at '/tmp/tmp8oz0f9gh'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name='images')
Output Type:
  Dict[['bbox_regression', TensorSpec(shape=(None, None, 4), dtype=tf.float32, name=None)], ['cls_logits', TensorSpec(shape=(None, None, 91), dtype=tf.float32, name=None)]]
Captures:
  139721534904016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534905744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534905936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534904784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534905360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534906320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534906512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534906704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534903632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534907856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534907280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534907472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534907664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534905168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534908816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534908240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534903440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534905552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534904592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534904976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534906896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534908432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534908624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534906128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534909776: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534909200: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534909392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534909584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534907088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534910736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534910160: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534910352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534910544: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534908048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534911696: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534911120: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534911312: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534911504: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534909008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534912656: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534912080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534912272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534912464: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534909968: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534913616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534913040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534913232: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139723273747152: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534910928: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534902480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534914000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534912848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534914192: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534914384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534913424: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530181008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530179664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530179856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530180240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530182160: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530181584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530181776: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530181968: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530180816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530183120: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530182544: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534913808: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530180048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530181200: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534911888: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530180624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530182736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530182928: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530180432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530184080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530183504: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530183696: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530183888: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530181392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530185040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530184464: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530184656: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530184848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530182352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530186000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530185424: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530185616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530185808: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530183312: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530186960: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530186384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530186576: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530186768: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530184272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530187920: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530187344: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530187536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530187728: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530185232: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530188880: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530188304: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530188496: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530188688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530186192: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530189840: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530189264: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530189456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530189648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530187152: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530190800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530190224: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530190416: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530190608: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530188112: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530191760: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530191184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530191376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530191568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530189072: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530192720: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530193104: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530193296: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530193488: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530190992: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530194640: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530194064: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530194256: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530194448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530191952: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530195600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530195024: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530192144: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530192336: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530192528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530190032: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530193680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530193872: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530195408: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530195216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530194832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530192912: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998470672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998470480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530195792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998471248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998469904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998470864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998471056: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998470096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998472208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998471632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998471824: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998472016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998469712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998473168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998472592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998472784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998472976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998470288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998474128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998473552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998473744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998473936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998471440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998475088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998474512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998474704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998474896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998472400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998476048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998475472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998475664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998475856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998473360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998477008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998476432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998476624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998476816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998474320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998477968: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998477392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998477584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998477776: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998475280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998478928: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998478352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998478544: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998478736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998476240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998479888: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998479312: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998479504: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998479696: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998477200: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998480848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998480272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998480464: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998480656: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998478160: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998481808: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998481232: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998481424: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998481616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998479120: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998482768: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998482192: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998482384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998482576: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998480080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998483728: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998483152: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998483344: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998483536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998481040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998484688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998485072: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998483920: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998485456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998485264: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998484880: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998482960: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528591760: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528591952: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998485840: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528590800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528590416: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998484112: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998484304: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998484496: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998482000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139720998485648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528591568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528591376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528591184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528592912: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528592336: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528592528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528592720: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528590608: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528593872: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528593296: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528593488: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528593680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528590992: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528594832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528594256: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528594448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528594640: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528592144: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528595792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528595216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528595408: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528595600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528593104: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528596752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528596176: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528596368: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528596560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528594064: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528597712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528597136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528597328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528597520: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528595024: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528598672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528596944: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528598864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528598288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528597904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528598096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528599248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528599056: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528600592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528598480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528600208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528595984: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528599824: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528599632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528600976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528600016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528601360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528600784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528603664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528601744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528604240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528603088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528603280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528602896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528604624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528604816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528603856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534903824: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721534902864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528601936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528601168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528601552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528602320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528602128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528599440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528602704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721528604048: TensorSpec(shape=(), dtype=tf.resource, name=None)

W0000 00:00:1769672055.422124 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672055.422159 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.

Saved artifact at 'detector.tflite'.

Exported Keras-Hub object detector
```
</div>

---
## Quantization for Smaller Models

Quantization reduces model size and can improve inference speed on edge devices.


```python
# Create a model for quantization
quantization_model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu", input_shape=(784,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

quantization_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Export unquantized model for comparison
quantization_model.export("model_unquantized.tflite", format="litert")
print("Exported unquantized model")
```

<div class="k-default-codeblock">
```
/usr/local/google/home/hellorahul/projects/keras-io/venv/lib/python3.12/site-packages/keras/src/layers/core/dense.py:106: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)

INFO:tensorflow:Assets written to: /tmp/tmphfu2su58/assets

INFO:tensorflow:Assets written to: /tmp/tmphfu2su58/assets

Saved artifact at '/tmp/tmphfu2su58'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 784), dtype=tf.float32, name='keras_tensor_421')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  139721530644560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530643216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645520: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'model_unquantized.tflite'.

Exported unquantized model

W0000 00:00:1769672060.192284 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672060.192328 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
```
</div>

### Dynamic Range Quantization

Dynamic range quantization quantizes weights to 8-bit integers but keeps activations
in float32. This is the default optimization when using `optimizations=[tf.lite.Optimize.DEFAULT]`.
It provides about 4x size reduction with minimal accuracy loss.


```python
quantization_model.export(
    "model_dynamic_range.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
)

print("Exported dynamic range quantized model")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /tmp/tmpxsrbb3bg/assets

INFO:tensorflow:Assets written to: /tmp/tmpxsrbb3bg/assets

Saved artifact at '/tmp/tmpxsrbb3bg'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 784), dtype=tf.float32, name='keras_tensor_421')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  139721530644560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530643216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645520: TensorSpec(shape=(), dtype=tf.resource, name=None)

W0000 00:00:1769672060.478449 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672060.478476 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.

Saved artifact at 'model_dynamic_range.tflite'.

Exported dynamic range quantized model
```
</div>

### Float16 Quantization

Float16 quantization converts weights to 16-bit floating point numbers.
It provides about 2x size reduction and is often GPU-compatible.


```python
quantization_model.export(
    "model_float16.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
    target_spec={"supported_types": [tf.float16]},
)

print("Exported Float16 quantized model")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /tmp/tmp3w9xsjiy/assets

INFO:tensorflow:Assets written to: /tmp/tmp3w9xsjiy/assets

Saved artifact at '/tmp/tmp3w9xsjiy'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 784), dtype=tf.float32, name='keras_tensor_421')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  139721530644560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530643216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645520: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'model_float16.tflite'.

Exported Float16 quantized model

W0000 00:00:1769672060.787827 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672060.787861 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
```
</div>

### Full Integer Quantization (INT8)

Full integer quantization converts both weights and activations to 8-bit integers.
This requires a representative dataset for calibration and is ideal for
edge devices without floating point support (e.g. microcontrollers).


```python

def representative_dataset():
    # In practice, use real data from your validation set
    for _ in range(100):
        data = np.random.random((1, 784)).astype(np.float32)
        yield [data]


quantization_model.export(
    "model_int8.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
    representative_dataset=representative_dataset,
    target_spec={"supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]},
    inference_input_type=tf.int8,
    inference_output_type=tf.int8,
)

print("Exported INT8 quantized model")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /tmp/tmp1mtl7_3s/assets

INFO:tensorflow:Assets written to: /tmp/tmp1mtl7_3s/assets

Saved artifact at '/tmp/tmp1mtl7_3s'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 784), dtype=tf.float32, name='keras_tensor_421')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  139721530644560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530643216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530645520: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'model_int8.tflite'.

Exported INT8 quantized model

/usr/local/google/home/hellorahul/projects/keras-io/venv/lib/python3.12/site-packages/tensorflow/lite/python/convert.py:863: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
  warnings.warn(
W0000 00:00:1769672061.116215 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672061.116242 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
fully_quantize: 0, inference_type: 6, input_inference_type: INT8, output_inference_type: INT8
```
</div>

### Model Size Comparison


```python

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size / 1024  # Convert to KB


print("\nModel Size Comparison:")
print(f"Unquantized: {get_file_size('model_unquantized.tflite'):.2f} KB")
print(f"Dynamic Range: {get_file_size('model_dynamic_range.tflite'):.2f} KB")
print(f"Float16: {get_file_size('model_float16.tflite'):.2f} KB")
print(f"Int8: {get_file_size('model_int8.tflite'):.2f} KB")
```

    
<div class="k-default-codeblock">
```
Model Size Comparison:
Unquantized: 206.98 KB
Dynamic Range: 55.20 KB
Float16: 104.81 KB
Int8: 54.53 KB
```
</div>

---
## Dynamic Shapes

Dynamic shapes allow models to handle variable input sizes at runtime.


```python
# Create model with dynamic batch size
dynamic_model = keras.Sequential(
    [
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

# Use None for dynamic dimensions
dynamic_model.build((None, 784))  # None = dynamic batch size

# Export with dynamic shapes
dynamic_model.export("dynamic_model.tflite", format="litert")

print("Exported model with dynamic shapes")

# Verify dynamic shapes in the exported model
interpreter = Interpreter(model_path="dynamic_model.tflite")
input_details = interpreter.get_input_details()

print(f"\nInput shape: {input_details[0]['shape']}")
print("Note: -1 indicates a dynamic dimension")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /tmp/tmpg0z4f12n/assets

INFO:tensorflow:Assets written to: /tmp/tmpg0z4f12n/assets

Saved artifact at '/tmp/tmpg0z4f12n'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 784), dtype=tf.float32, name='keras_tensor_425')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  139721530646480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530648016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530651472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139721530644752: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'dynamic_model.tflite'.

Exported model with dynamic shapes

Input shape: [  1 784]
Note: -1 indicates a dynamic dimension

W0000 00:00:1769672061.416110 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672061.416147 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
```
</div>

---
## Custom Input Signatures

For models with complex input requirements or multiple inputs/outputs.


```python
# Model with custom signature using functional API
sig_input_a = Input(shape=(10,), name="input_a")
sig_input_b = Input(shape=(10,), name="input_b")

# Create outputs with custom names
sig_output1 = sig_input_a + sig_input_b  # Addition
sig_output2 = sig_input_a * sig_input_b  # Multiplication

# Create model with named inputs and outputs
signature_model = keras.Model(
    inputs={"input_a": sig_input_a, "input_b": sig_input_b},
    outputs={"output1": sig_output1, "output2": sig_output2},
)
```

---
## Model Validation

Always verify your exported model before deploying to production.


```python

def validate_tflite_model(model_path, keras_model):
    """Compare TFLite model output with Keras model."""

    # Load TFLite model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Generate test input
    test_input = np.random.random((1, 28, 28)).astype(np.float32)

    # Keras prediction
    keras_output = keras_model(test_input, training=False)

    # TFLite prediction
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])

    # Compare outputs
    np.testing.assert_allclose(keras_output.numpy(), tflite_output, atol=1e-5)
    print("✓ Model validation passed!")


# Validate our basic model
validate_tflite_model("mnist_classifier.tflite", model)
```

<div class="k-default-codeblock">
```
✓ Model validation passed!
```
</div>

---
## Advanced Export Options

Keras export supports various advanced options for LiteRT conversion.


```python
# Example with advanced options - supporting both TFLite builtins and TF ops
model.export(
    "model_advanced.tflite",
    format="litert",
    target_spec={
        "supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    },
)

print("Exported model with advanced options")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /tmp/tmp3k3_32g_/assets

INFO:tensorflow:Assets written to: /tmp/tmp3k3_32g_/assets

Saved artifact at '/tmp/tmp3k3_32g_'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name='keras_tensor')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  139725986278352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725986281616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725986280272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  139725986279120: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'model_advanced.tflite'.

Exported model with advanced options

W0000 00:00:1769672061.698513 2222654 tf_tfl_flatbuffer_helpers.cc:364] Ignored output_format.
W0000 00:00:1769672061.698541 2222654 tf_tfl_flatbuffer_helpers.cc:367] Ignored drop_control_dependency.
```
</div>

---
## Best Practices

1. **Test thoroughly**: Always validate exported models before deployment
2. **Choose appropriate quantization**: Balance size vs accuracy based on your use case
3. **Handle dynamic shapes**: Use when input sizes vary at runtime
4. **Optimize for target hardware**: Consider GPU/CPU/NPU capabilities
5. **Version control**: Keep track of model versions and export parameters

---
## Troubleshooting

Common issues and solutions:

- **Import errors**: Ensure `tensorflow` and `ai_edge_litert` are installed.
- **Shape mismatches**: Verify input shapes match model expectations.
- **Unsupported ops**: Use `SELECT_TF_OPS` for TensorFlow operations:
    ```python
    model.export(
        "model.tflite",
        format="litert",
        target_spec={
            "supported_ops": [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        }
    )
    ```
- **Unable to infer input signature**: For Subclassed models, call the model with sample data before exporting to build it.
- **Out of memory**: Large models may require significant RAM. Try exporting with quantization or using a machine with more RAM.
- **Accuracy drops**: Start with float16 quantization instead of full int8 if accuracy drops significantly.


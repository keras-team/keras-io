# Exporting Keras models to LiteRT (TensorFlow Lite)

**Author:** [Rahul Kumar](https://github.com/pctablet505)<br>
**Date created:** 2025/12/10<br>
**Last modified:** 2025/12/10<br>
**Description:** Complete guide to exporting Keras models for mobile and edge deployment.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/litert_export.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/litert_export.py)



---
## Introduction

TensorFlow Lite (LiteRT) is TensorFlow's solution for running machine learning models
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
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/keras/src/export/tf2onnx_lib.py:8: FutureWarning: In the future `np.object` will be defined as the corresponding NumPy scalar.
  if not hasattr(np, "object"):

Keras version: 3.13.0
TensorFlow version: 2.19.1
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
X_train = np.random.random((1000, 28, 28))
y_train = np.random.randint(0, 10, 1000)

# Quick training (just for demonstration)
model.fit(X_train, y_train, epochs=1, verbose=0)

print("Model created and trained")
```

<div class="k-default-codeblock">
```
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/keras/src/layers/reshaping/flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
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
INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpshk7ah7v/assets

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpshk7ah7v/assets

Saved artifact at '/var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpshk7ah7v'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name='keras_tensor')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  13352648976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13352652048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13352651472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13352650320: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'mnist_classifier.tflite'.

Model exported to mnist_classifier.tflite

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1769064919.442013 3729126 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1769064919.442031 3729126 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
I0000 00:00:1769064919.443906 3729126 mlir_graph_optimization_pass.cc:425] MLIR V1 optimization pass is not enabled
```
</div>

---
## Testing the Exported Model

Let's verify the exported model works correctly.


```python
# Load and test the exported model
litert_available = False
try:
    from ai_edge_litert.interpreter import Interpreter

    print("Using ai_edge_litert for inference")
    litert_available = True
except ImportError:
    try:
        from tensorflow.lite import Interpreter

        print("Using tensorflow.lite for inference")
        litert_available = True
    except ImportError:
        try:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter

            print("Using tf.lite.Interpreter for inference")
            litert_available = True
        except (ImportError, AttributeError):
            print("LiteRT interpreter not available. Skipping inference test.")
            print(
                "To test inference, install ai_edge_litert: pip install ai-edge-litert"
            )

if litert_available:
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
else:
    print("Skipping inference test due to missing LiteRT interpreter.")
```

<div class="k-default-codeblock">
```
Using tf.lite.Interpreter for inference

Model Input Details:
  Shape: [ 1 28 28]
  Type: <class 'numpy.float32'>

Model Output Details:
  Shape: [ 1 10]
  Type: <class 'numpy.float32'>

Inference successful! Output shape: (1, 10)

/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
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
INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpbzc9s861/assets

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpbzc9s861/assets

Saved artifact at '/var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpbzc9s861'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): List[TensorSpec(shape=(None, 32), dtype=tf.float32, name='keras_tensor_5'), TensorSpec(shape=(None, 32), dtype=tf.float32, name='keras_tensor_6')]
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  13352657040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13352650512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13352664720: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13352664912: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'functional_model.tflite'.

Functional model exported

W0000 00:00:1769064919.567768 3729126 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1769064919.567777 3729126 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
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
INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmp5byz7ya5/assets

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmp5byz7ya5/assets

Saved artifact at '/var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmp5byz7ya5'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 16), dtype=tf.float32, name=None)
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  13429197904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13429195792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13429196752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13429196560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13429198672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13429197136: TensorSpec(shape=(), dtype=tf.resource, name=None)

W0000 00:00:1769064919.692938 3729126 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1769064919.692947 3729126 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.

Saved artifact at 'subclassed_model.tflite'.

Subclassed model exported
```
</div>

---
## KerasHub Models

KerasHub provides pretrained models for various tasks. Let's export some.


```python
keras_hub_available = False
try:
    import keras_hub

    keras_hub_available = True
except ImportError:
    print("keras-hub not available. Skipping Keras-Hub example.")
    print("To run this example, install keras-hub: pip install keras-hub")

if keras_hub_available:
    try:
        # Load a pretrained text model
        # Sequence length is configured via the preprocessor
        preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor.from_preset(
            "gemma3_1b", sequence_length=128
        )

        gemma_model = keras_hub.models.Gemma3CausalLM.from_preset(
            "gemma3_1b", preprocessor=preprocessor, load_weights=False
        )

        # Export to LiteRT (sequence length already set)
        gemma_model.export("gemma3_1b.tflite", format="litert")

        print("Exported Keras-Hub Gemma3 1B model")
    except Exception as e:
        print(f"Failed to load Gemma3 model: {e}")
        print("Skipping Gemma3 model export due to memory/resource constraints.")

    """
    For vision models, the image size is determined by the preset:
    """

    try:
        # Load a vision model
        vision_model = keras_hub.models.ImageClassifier.from_preset(
            "resnet_50_imagenet"
        )

        # Export (image size already set by preset)
        vision_model.export("resnet.tflite", format="litert")

        print("Exported Keras-Hub vision model")
    except Exception as e:
        print(f"Failed to load vision model: {e}")
        print("Skipping vision model export.")
else:
    print("Skipping Keras-Hub model export due to missing keras-hub.")
```

<div class="k-default-codeblock">
```
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.

Creating adapter for inputs: ['padding_mask', 'token_ids']

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpslixvywa/assets

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpslixvywa/assets

Saved artifact at '/var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpslixvywa'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): List[TensorSpec(shape=(None, None), dtype=tf.int32, name='padding_mask'), TensorSpec(shape=(None, None), dtype=tf.int32, name='token_ids')]
Output Type:
  TensorSpec(shape=(None, None, 262144), dtype=tf.float32, name=None)
Captures:
  13499785168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499785744: TensorSpec(shape=(), dtype=tf.float32, name=None)
  13499785360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499786320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499782864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499785552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499787472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499787088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499784208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499784400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499788048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499784784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499786704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499788624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499786512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499786128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499789008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499786896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499787664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499789968: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499789584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499787856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499788432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499790544: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499787280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499789200: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499791120: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499788240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499785936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499791504: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499789392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499790160: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499792464: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499792080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499790352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499790928: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499793040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499789776: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499791696: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499791888: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499792848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499793232: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499792656: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499791312: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499788816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499792272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13499790736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192412240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192410704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192412816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192411856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192411664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192413392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192411280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192410896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192413776: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192411472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192412432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192414736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192414352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192412624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192413200: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192415312: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192412048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192413968: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192415888: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192413008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192411088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192416272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192414160: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192414928: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192417232: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192416848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192415120: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192415696: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192417808: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192414544: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192416464: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192418384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192415504: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192413584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192418768: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192416656: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192417424: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192419728: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192419344: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192417616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192418192: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192420304: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192417040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192418960: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192420880: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192418000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192416080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192421264: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192419152: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192419920: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192422224: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192421840: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192420112: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192420688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192422800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192419536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192421456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192423376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192420496: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192418576: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192423760: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192421648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192422416: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192424720: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192424336: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192422608: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192423184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192425296: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192422032: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192423952: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192425872: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192422992: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192421072: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192426256: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192426640: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192424912: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192424144: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192426832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192423568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192425488: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192425680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192425104: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192424528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192426448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195933840: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14192426064: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195934992: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195934608: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195933456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195935952: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195935568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195934032: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195933648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195936528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195934800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195935184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195937104: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195934224: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195934416: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195937488: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195935376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195936144: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195938448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195938064: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195936336: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195936912: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195939024: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195935760: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195937680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195939600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195936720: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195933264: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195939984: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195937872: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195938640: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195940944: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195940560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195938832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195939408: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195941520: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195938256: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195940176: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195942096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195939216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195937296: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195942480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195940368: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195941136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195943440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195943056: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195941328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195941904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195944016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195940752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195942672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195944592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195941712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195939792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195944976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195942864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195943632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195945936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195945552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195943824: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195944400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195946512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195943248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195945168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195947088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195944208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195942288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195947472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195945360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195946128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195948432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195948048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195946320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195946896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195949008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195945744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195947664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195947856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195949200: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195946704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195948624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195949392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195944784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195947280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195948816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14195948240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196851920: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196852688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196851152: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196851536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196853264: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196852112: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196853648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196850960: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196851728: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196853840: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196854800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196854416: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196851344: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196852496: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196855376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196853456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196854032: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196855952: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196853072: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196852880: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196856336: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196854224: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196854992: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196857296: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196856912: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196855184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196855760: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196857872: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196854608: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196856528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196858448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196855568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196852304: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196858832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196856720: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196857488: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196859792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196859408: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196857680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196858256: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196860368: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196857104: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196859024: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196860944: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196858064: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196856144: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196861328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196859216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196859984: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196862288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196861904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196860176: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196860752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196862864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196859600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196861520: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196863440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196860560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196858640: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196863824: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196861712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196862480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196864784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196864400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196862672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196863248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196865360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196862096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196864016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196865936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196863056: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196861136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196866320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196866704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196864976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196864208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196866896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196863632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196865552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196865744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196865168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196864592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196866512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197932688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196866128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197933840: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197933456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197932304: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197934800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197934416: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197932880: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197932496: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197935376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197933648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197934032: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197935952: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197933072: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197933264: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197936336: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197934224: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197934992: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197937296: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197936912: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197935184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197935760: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197937872: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197934608: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197936528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197938448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197935568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14196850768: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197932112: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197936720: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197938640: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197939600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197939216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197937680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197938256: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197940176: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197938832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197937488: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197940752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197936144: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14197938064: TensorSpec(shape=(), dtype=tf.resource, name=None)

W0000 00:00:1769064932.594390 3729126 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1769064932.594402 3729126 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.

Saved artifact at 'gemma3_1b.tflite'.

Exported Keras-Hub Gemma3 1B model

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpchd97qm1/assets

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpchd97qm1/assets

Saved artifact at '/var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpchd97qm1'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name='keras_tensor_44')
Output Type:
  TensorSpec(shape=(None, 1000), dtype=tf.float32, name=None)
Captures:
  18367706896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367708624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367707280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18192219024: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367708240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367707088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198718544: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198718736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198719312: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198720272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198718928: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198719888: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198720080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198719504: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198721232: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198720656: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367708432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367708816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367707664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367709008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367708048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198722384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198722576: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198721808: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198719120: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198721424: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198721040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198720848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198722000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198723536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198722960: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198723152: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198723344: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198722192: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198724496: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198723920: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198724112: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198724304: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198719696: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198725456: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198724880: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198725072: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198725264: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198722768: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198726416: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198725840: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198726032: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198726224: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198723728: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198727376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198726800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198726992: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198727184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198724688: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198728336: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198728720: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198728912: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198729104: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198726608: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198730256: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198729680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198729872: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198730064: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198727568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198731216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198730640: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198727760: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198727952: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198728144: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198725648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198729296: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198730832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198731024: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198728528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198732176: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198731600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198731792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198731984: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198729488: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198733136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198732560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198732752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198732944: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198730448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198734096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198733520: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198733712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198734288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198734672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198733328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198732368: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198733904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385175760: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198734480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  14198731408: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385175376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385174800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385174608: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385174992: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385176720: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385176144: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385176336: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385176528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385175568: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385177680: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385177104: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385177296: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385177488: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385175184: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385178640: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385178064: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385178256: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385178448: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385175952: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385179600: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385179024: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385179216: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385179408: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385176912: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385180560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385180944: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385181136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385181328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385178832: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385182480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385181904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385182096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385182288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385179792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385183440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385182864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385179984: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385180176: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385180368: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385177872: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385181520: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385183056: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385183248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385180752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385184400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385183824: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385184016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385184208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385181712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385185360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385184784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385184976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385185168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385182672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385186320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385185744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385185936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385186128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385183632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385187280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385186704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385186896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385187088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385184592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385188240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385187664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385187856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385188048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385185552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385189200: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385188624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385188816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385189008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385186512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385190160: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385189584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385189776: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385190352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385190736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385189392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385188432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385189968: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382324176: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385190544: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18385187472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382323984: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382324752: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382324560: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382324368: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382325904: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382325328: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382325520: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382325712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382324944: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382326864: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382326288: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382326480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382326672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382323792: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382327824: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382327248: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382327440: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382327632: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382325136: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382328784: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382328208: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382328400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382328592: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382326096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382329744: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382329168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382329360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382329552: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382327056: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382330704: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382330128: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382330320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382330512: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382328016: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382331664: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382332048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382332240: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382332432: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382329936: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382333584: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382333008: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382333200: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382333392: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382330896: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382334544: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382333968: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382331088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382331280: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382331472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382328976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382332624: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382334160: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382334352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382331856: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382335504: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382334928: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382335120: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382335312: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382332816: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382336464: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382335888: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382336080: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382336272: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382333776: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382337424: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382336848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367706320: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382337616: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382334736: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382338384: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382337232: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382338000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382338192: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382337040: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382339152: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382336656: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382338768: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382338576: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382339920: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382339344: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382335696: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382339536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382339728: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382338960: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18382337808: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367707472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367706704: TensorSpec(shape=(), dtype=tf.resource, name=None)

W0000 00:00:1769064983.667405 3729126 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1769064983.667414 3729126 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.

Saved artifact at 'resnet.tflite'.

Exported Keras-Hub vision model
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

# Basic quantization (reduces precision from float32 to int8)
quantization_model.export(
    "model_quantized.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
)

print("Exported quantized model")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpk4utq2_o/assets

/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/keras/src/layers/core/dense.py:106: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpk4utq2_o/assets

Saved artifact at '/var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpk4utq2_o'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 784), dtype=tf.float32, name='keras_tensor_226')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  18367704400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18615286480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367705168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609390800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609391376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609390416: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'model_quantized.tflite'.

Exported quantized model

W0000 00:00:1769064985.572898 3729126 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1769064985.572911 3729126 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
```
</div>

### Float16 Quantization

Float16 quantization offers a good balance between model size and accuracy,
especially for GPU inference.


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
INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpoa2rxgyh/assets

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpoa2rxgyh/assets

Saved artifact at '/var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpoa2rxgyh'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 784), dtype=tf.float32, name='keras_tensor_226')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  18367704400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18615286480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367705168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609390800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609391376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609390416: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'model_float16.tflite'.

Exported Float16 quantized model

W0000 00:00:1769064985.690148 3729126 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1769064985.690160 3729126 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
```
</div>

### Dynamic Range Quantization

Dynamic range quantization quantizes weights but keeps activations in float32.


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
INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpahkarg_q/assets

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpahkarg_q/assets

Saved artifact at '/var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpahkarg_q'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 784), dtype=tf.float32, name='keras_tensor_226')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  18367704400: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18615286480: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18367705168: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609390800: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609391376: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609390416: TensorSpec(shape=(), dtype=tf.resource, name=None)

W0000 00:00:1769064985.811013 3729126 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1769064985.811020 3729126 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.

Saved artifact at 'model_dynamic_range.tflite'.

Exported dynamic range quantized model
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
if litert_available:
    interpreter = Interpreter(model_path="dynamic_model.tflite")
    input_details = interpreter.get_input_details()

    print(f"\nInput shape: {input_details[0]['shape']}")
    print("Note: -1 indicates a dynamic dimension")
else:
    print("Skipping dynamic shapes verification due to missing LiteRT interpreter.")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpz2ulxpxp/assets

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpz2ulxpxp/assets

Saved artifact at '/var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpz2ulxpxp'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 784), dtype=tf.float32, name='keras_tensor_230')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  18609397712: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609398672: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609401360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  18609395408: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'dynamic_model.tflite'.

Exported model with dynamic shapes

Input shape: [  1 784]
Note: -1 indicates a dynamic dimension

W0000 00:00:1769064985.928684 3729126 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1769064985.928693 3729126 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
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
    if not litert_available:
        print("Skipping validation: LiteRT interpreter not available")
        return None

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
    diff = np.abs(keras_output.numpy() - tflite_output).max()
    print(f"Maximum difference: {diff}")

    if diff < 1e-5:
        print("✓ Model validation passed!")
        return True
    else:
        print("✗ Model validation failed!")
        return False


# Validate our basic model
if litert_available:
    validate_tflite_model("mnist_classifier.tflite", model)
```

<div class="k-default-codeblock">
```
Maximum difference: 4.470348358154297e-08
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
INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpi0wq4emp/assets

INFO:tensorflow:Assets written to: /var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpi0wq4emp/assets

Saved artifact at '/var/folders/kk/6bvt2y611ns5qk0zdmww21x801b8p6/T/tmpi0wq4emp'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name='keras_tensor')
Output Type:
  TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
Captures:
  13352648976: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13352652048: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13352651472: TensorSpec(shape=(), dtype=tf.resource, name=None)
  13352650320: TensorSpec(shape=(), dtype=tf.resource, name=None)

Saved artifact at 'model_advanced.tflite'.

Exported model with advanced options

W0000 00:00:1769064986.054394 3729126 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1769064986.054410 3729126 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
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

- **Import errors**: Ensure TensorFlow and ai_edge_litert are installed
- **Shape mismatches**: Verify input shapes match model expectations
- **Unsupported ops**: Use SELECT_TF_OPS for TensorFlow operations
- **Memory issues**: Reduce model size with quantization
- **Accuracy drops**: Start with float16 instead of full int8 quantization

---
## Next Steps

- Deploy to mobile apps using TensorFlow Lite Android/iOS SDKs
- Optimize for specific hardware with TensorFlow Lite delegates
- Explore model compression techniques beyond quantization
- Consider using TensorFlow Model Optimization Toolkit for advanced optimization

"""
Title: Exporting Keras models to LiteRT (TensorFlow Lite)
Author: [Rahul Kumar](https://github.com/pctablet505)
Date created: 2025/12/10
Last modified: 2025/12/10
Description: Complete guide to exporting Keras models for mobile and edge deployment.
Accelerator: None
"""

"""
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

## Setup

First, let's install the required packages and set up the environment.
"""

"""
### Installation

Install the required packages:
```
pip install -q keras tensorflow ai-edge-litert
```

For KerasHub models (optional):
```
pip install -q keras-hub
```


"""

import os

# Set Keras backend to TensorFlow for LiteRT export
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)

"""
## Basic Model Export

Let's start with a simple MNIST classifier and export it to LiteRT format.
"""

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

"""
Now let's export the model to LiteRT format. The `format="litert"` parameter
tells Keras to export in TensorFlow Lite format.
"""

# Export to LiteRT
model.export("mnist_classifier.tflite", format="litert")

print("Model exported to mnist_classifier.tflite")

"""
## Testing the Exported Model

Let's verify the exported model works correctly.
"""

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

"""
## Exporting Different Model Types

Keras supports various model architectures. Let's explore how to export them.
"""

"""
### Functional API Models

Functional API models offer more flexibility than Sequential models.
"""

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

"""
### Subclassed Models

For complex architectures that require custom forward passes.
"""


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

"""
## KerasHub Models

KerasHub provides pretrained models for various tasks. Let's export some.
"""

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

"""
For vision models, the image size is determined by the preset:
"""

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


"""
## Quantization for Smaller Models

Quantization reduces model size and can improve inference speed on edge devices.
"""

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

"""
### Dynamic Range Quantization

Dynamic range quantization quantizes weights to 8-bit integers but keeps activations
in float32. This is the default optimization when using `optimizations=[tf.lite.Optimize.DEFAULT]`.
It provides about 4x size reduction with minimal accuracy loss.
"""

quantization_model.export(
    "model_dynamic_range.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
)

print("Exported dynamic range quantized model")

"""
### Float16 Quantization

Float16 quantization converts weights to 16-bit floating point numbers.
It provides about 2x size reduction and is often GPU-compatible.
"""

quantization_model.export(
    "model_float16.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
    target_spec={"supported_types": [tf.float16]},
)

print("Exported Float16 quantized model")

"""
### Full Integer Quantization (INT8)

Full integer quantization converts both weights and activations to 8-bit integers.
This requires a representative dataset for calibration and is ideal for
edge devices without floating point support (e.g. microcontrollers).
"""


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

"""
### Model Size Comparison
"""


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size / 1024  # Convert to KB


print("\nModel Size Comparison:")
print(f"Unquantized: {get_file_size('model_unquantized.tflite'):.2f} KB")
print(f"Dynamic Range: {get_file_size('model_dynamic_range.tflite'):.2f} KB")
print(f"Float16: {get_file_size('model_float16.tflite'):.2f} KB")
print(f"Int8: {get_file_size('model_int8.tflite'):.2f} KB")

"""
## Dynamic Shapes

Dynamic shapes allow models to handle variable input sizes at runtime.
"""

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

"""
## Custom Input Signatures

For models with complex input requirements or multiple inputs/outputs.
"""

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

"""
## Model Validation

Always verify your exported model before deploying to production.

"""


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

"""
## Advanced Export Options

Keras export supports various advanced options for LiteRT conversion.
"""

# Example with advanced options - supporting both TFLite builtins and TF ops
model.export(
    "model_advanced.tflite",
    format="litert",
    target_spec={
        "supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    },
)

print("Exported model with advanced options")

"""
## Best Practices

1. **Test thoroughly**: Always validate exported models before deployment
2. **Choose appropriate quantization**: Balance size vs accuracy based on your use case
3. **Handle dynamic shapes**: Use when input sizes vary at runtime
4. **Optimize for target hardware**: Consider GPU/CPU/NPU capabilities
5. **Version control**: Keep track of model versions and export parameters

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


"""

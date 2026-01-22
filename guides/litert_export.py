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

TensorFlow Lite (LiteRT) is TensorFlow's solution for running machine learning models
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
X_train = np.random.random((1000, 28, 28))
y_train = np.random.randint(0, 10, 1000)

# Quick training (just for demonstration)
model.fit(X_train, y_train, epochs=1, verbose=0)

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

# Basic quantization (reduces precision from float32 to int8)
quantization_model.export(
    "model_quantized.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
)

print("Exported quantized model")

"""
### Float16 Quantization

Float16 quantization offers a good balance between model size and accuracy,
especially for GPU inference.
"""

quantization_model.export(
    "model_float16.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
    target_spec={"supported_types": [tf.float16]},
)

print("Exported Float16 quantized model")

"""
### Dynamic Range Quantization

Dynamic range quantization quantizes weights but keeps activations in float32.
"""

quantization_model.export(
    "model_dynamic_range.tflite",
    format="litert",
    optimizations=[tf.lite.Optimize.DEFAULT],
)

print("Exported dynamic range quantized model")

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
if litert_available:
    interpreter = Interpreter(model_path="dynamic_model.tflite")
    input_details = interpreter.get_input_details()

    print(f"\nInput shape: {input_details[0]['shape']}")
    print("Note: -1 indicates a dynamic dimension")
else:
    print("Skipping dynamic shapes verification due to missing LiteRT interpreter.")

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

- **Import errors**: Ensure TensorFlow and ai_edge_litert are installed
- **Shape mismatches**: Verify input shapes match model expectations
- **Unsupported ops**: Use SELECT_TF_OPS for TensorFlow operations
- **Memory issues**: Reduce model size with quantization
- **Accuracy drops**: Start with float16 instead of full int8 quantization

## Next Steps

- Deploy to mobile apps using TensorFlow Lite Android/iOS SDKs
- Optimize for specific hardware with TensorFlow Lite delegates
- Explore model compression techniques beyond quantization
- Consider using TensorFlow Model Optimization Toolkit for advanced optimization
"""

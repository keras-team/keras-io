"""
Title: Customizing Quantization with QuantizationConfig
Author: [Jyotinder Singh](https://x.com/Jyotinder_Singh)
Date created: 2025/12/18
Last modified: 2025/12/18
Description: Guide on using QuantizationConfig for weight-only quantization and custom quantizers.
Accelerator: GPU
"""

"""
## Introduction

This guide explores the flexible `QuantizationConfig` API in Keras, introduced to give you granular control over how your models are quantized.
While `model.quantize("int8")` provides a great default, you often need more control. For example, to perform **weight-only quantization** (common in LLMs) or to use **custom quantization schemes** (like percentile-based clipping).

We will cover:

1.  **Customizing INT8 Quantization**: Modifying the default parameters (e.g., custom value range).
2.  **Weight-Only Quantization (INT4)**: Quantizing weights to 4-bit while keeping activations in float, using `Int4QuantizationConfig`.
3.  **Custom Quantizers**: Implementing a completely custom quantizer (e.g., `PercentileQuantizer`) and using it with `QuantizationConfig`.
"""

"""
## Setup
"""

import keras
import numpy as np
from keras import ops

rng = np.random.default_rng()


def get_model():
    """Builds a simple Sequential model for demonstration."""
    return keras.Sequential(
        [
            keras.Input(shape=(10,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )


"""
## 1. Customizing INT8 Quantization

By default, `model.quantize("int8")` uses `AbsMaxQuantizer` for both weights and activations which uses the default value range of [-127, 127].
You might want to specify different parameters, such as a restricted value range (if you expect your activations to be within a certain range).
You can do this by creating an `Int8QuantizationConfig`.
"""

from keras.quantizers import Int8QuantizationConfig, AbsMaxQuantizer

model = get_model()

# Create a custom config
# Here we restrict the weight range to [-100, 100] instead of the default [-127, 127]
custom_int8_config = Int8QuantizationConfig(
    weight_quantizer=AbsMaxQuantizer(value_range=(-100, 100), axis=0),
    activation_quantizer=AbsMaxQuantizer(value_range=(-100, 100), axis=-1),
)

# Apply quantization with the custom config
model.quantize(config=custom_int8_config)

print("Layer 0 kernel dtype:", model.layers[0].kernel.dtype)
# Ensure all kernel values are within the specified range
assert ops.all(
    ops.less_equal(model.layers[0].kernel, 100)
), "Kernel values are not <= 100"
assert ops.all(
    ops.greater_equal(model.layers[0].kernel, -100)
), "Kernel values are not >= -100"

"""
## 2. Weight-Only Quantization (INT4)

By default, `model.quantize("int4")` quantizes activations to INT8 while keeping weights in INT4.
For large language models and memory-constrained environments, **weight-only quantization** is a popular technique.
It reduces the model size significantly (keeping weights in 4-bit) while maintaining higher precision for activations.

To achieve this, we set `activation_quantizer=None` in the `Int4QuantizationConfig`.
"""

from keras.quantizers import Int4QuantizationConfig

model = get_model()

# Define Int4 weight-only config
# We enable Int4 for weights, but disable activation quantization by setting it to None.
# Note that we use `"int8"` as the output dtype since TensorFlow and PyTorch don't support
# `int4`. However, we still benefit from the lower memory usage of int4 weights because of
# bitpacking implemented by Keras.
custom_int4_config = Int4QuantizationConfig(
    weight_quantizer=AbsMaxQuantizer(value_range=(-8, 7), output_dtype="int8", axis=0),
    activation_quantizer=None,
)

model.quantize(config=custom_int4_config)

# Verify that weights are quantized (int8 backing int4) but no activation quantization logic is added
print("Layer 0 kernel dtype:", model.layers[0].kernel.dtype)
print("Layer 0 has inputs_quantizer:", model.layers[0].inputs_quantizer is not None)

"""
## 3. Custom Quantizers: Implementing a Percentile Quantizer

Sometimes, standard absolute-max quantization isn't enough. You might want to be robust to outliers by using **percentile-based quantization**.
Keras allows you to define your own quantizer by subclassing `keras.quantizers.Quantizer`.

Below is an implementation of a `PercentileQuantizer` that sets the scale based on a specified percentile of the absolute values.
"""

from keras.quantizers import Quantizer
from keras import backend


class PercentileQuantizer(Quantizer):
    """Quantizes x using the percentile-based scale."""

    def __init__(
        self,
        percentile=99.9,
        value_range=(-127, 127),  # Default range for int8
        epsilon=backend.epsilon(),
        output_dtype="int8",  # Default dtype for int8
    ):
        super().__init__(output_dtype=output_dtype)
        self.percentile = percentile
        self.value_range = value_range
        self.epsilon = epsilon

    def __call__(self, x, axis, to_numpy=False):
        """Quantizes x using the percentile-based scale.

        `to_numpy` can be set to True to perform the computation on the host CPU,
        which saves device memory.
        """
        # 1. Compute the percentile value of absolute inputs
        x_abs = ops.abs(x)

        if to_numpy:
            x_np = ops.convert_to_numpy(x_abs)
            max_val = np.percentile(x_np, self.percentile, axis=axis, keepdims=True)
        else:
            max_val = ops.quantile(
                x_abs, self.percentile / 100, axis=axis, keepdims=True
            )

        # 2. Compute scale
        # scale = range_max / max_val
        # We ensure max_val is at least epsilon
        scale = ops.divide(self.value_range[1], ops.add(max_val, self.epsilon))
        if not to_numpy:
            scale = ops.cast(scale, backend.standardize_dtype(x.dtype))

        # 3. Quantize
        # q = x * scale
        outputs = ops.multiply(x, scale)
        outputs = ops.clip(ops.round(outputs), self.value_range[0], self.value_range[1])
        outputs = ops.cast(outputs, self.output_dtype)

        return outputs, scale

    def get_config(self):
        """Returns the config of the quantizer for serialization support."""
        return {
            "percentile": self.percentile,
            "value_range": self.value_range,
            "epsilon": self.epsilon,
            "output_dtype": self.output_dtype,
        }


"""
Now we can use this `PercentileQuantizer` in our configuration.
"""

model = get_model()

# Use the custom quantizer for activations
custom_int8_config = Int8QuantizationConfig(
    weight_quantizer=AbsMaxQuantizer(axis=0),
    activation_quantizer=PercentileQuantizer(percentile=99.9),
)

model.quantize(config=custom_int8_config)

# Verify the integration
print(
    "Layer 0 uses custom activation quantizer:",
    isinstance(model.layers[0].inputs_quantizer, PercentileQuantizer),
)

"""
## Conclusion

With `QuantizationConfig`, you are no longer limited to stock quantization options.
Whether you need weight-only quantization or custom quantizers for specialized hardware or research,
Keras provides the modularity to build exactly what you need.
"""

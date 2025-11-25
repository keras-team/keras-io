# Writing Quantization-Compatible Layers in Keras

**Author:** [Jyotinder Singh](https://x.com/Jyotinder_Singh)<br>
**Date created:** 2025/10/16<br>
**Last modified:** 2025/10/16<br>
**Description:** Complete guide for writing quantization-compatible Keras layers.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/writing_quantization_compatible_layers.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/writing_quantization_compatible_layers.py)



---
## What are quantization-compatible layers?

Keras lets you optimize models via post-training quantization (PTQ) by calling
the `layer.quantize(...)` or `model.quantize(...)` APIs. Keras exposes an
extensible framework for defining quantization-compatible layers. This lets you
author custom layers that plug into the quantization framework, can be quantized
to INT8 or INT4, and saved/loaded with quantization metadata.

A quantization-compatible layer needs to implement a few hooks, so that it can:

- Switch its variables to quantized representations.
- Use a quantization-aware forward path at inference.
- Save and load quantization metadata with the model.

In this guide, we'll implement a simple layer that supports INT8 PTQ. The same
patterns generalize to INT4 quantization and FP8 mixed-precision training.

---
## The hooks you'll implement

At minimum, your layer should define:

- `quantize(mode, **kwargs)`: Converts existing variables to quantized form and
    switches the dtype policy
- `_int8_build(...)`: Allocates INT8 variables needed by your layer
- `_int8_call(inputs, training=None)`: Minimal INT8 forward path

We'll implement these for a very small layer called `SimpleScale`, which
multiplies the inputs by a trainable per-feature vector (elementwise scaling on
the last dimension). The same patterns generalize to more sophisticated layers.

---
## Writing a Simple Quantization-Compatible Layer

We start with a tiny layer that learns a per-feature multiplier. The
full-precision path just computes `y = x * w`. We'll add the quantization hooks
step by step.


```python
import numpy as np
import keras
from keras import ops, quantizers, dtype_policies
from keras.layers import Layer, Input


class SimpleScale(Layer):
    """A layer that learns a per-feature scaling factor."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self._kernel = self.add_weight(
            name="kernel",
            shape=(input_dim,),
            initializer="random_uniform",
        )

    def call(self, inputs, training=None):
        return ops.multiply(inputs, self._kernel)

```

### The `quantize()` method

PTQ is a one-time rewrite. After you train or load your FP32 layer, you call
`layer.quantize("int8")`. The layer should:

1. Read its existing full-precision variables (e.g., `self._kernel`).
2. Quantize them to INT8 values plus a quantization scale.
3. Replace full-precision variables with INT8 storage and assign the quantized
  data.
4. Switch the `dtype_policy` to a quantized variant (e.g., `int8_from_float32`).


```python

def quantize(self, mode, **kwargs):
    if mode != "int8":
        raise NotImplementedError(f"Unsupported quantization mode: {mode}")

    quantized_kernel, scale = quantizers.abs_max_quantize(
        self._kernel, axis=0, dtype="int8", to_numpy=True
    )
    scale = ops.squeeze(scale, axis=0)

    kernel_shape = self._kernel.shape

    del self._kernel

    # Allocate INT8 variables. Discussed in the next section.
    self._int8_build(kernel_shape)

    self._kernel.assign(quantized_kernel)
    self.scale.assign(scale)

    # `_is_quantized` should be set before changing dtype policy to inform
    # the setter that quantized variables are initialized.
    self._is_quantized = True

    if self.dtype_policy.quantization_mode is None:
        policy = dtype_policies.get(f"{mode}_from_{self.dtype_policy.name}")
        self.dtype_policy = policy

```

#### Note

1. The `quantize(...)` method should validate `mode` and raise a
  `NotImplementedError` if the mode is not supported.
2. Ensure your `quantize(...)` sets a quantized dtype policy based on the
  prior policy, e.g., `int8_from_float32` or `int8_from_bfloat16`. This ensures
  that the layer's `quantization_mode` is correctly set.

3. The `_is_quantized` flag should be set before changing the dtype policy to
  inform the setter that quantized variables are initialized.

### The `_int8_build(...)` method

This `int8_build(...)` method is called from `quantize(...)` to initialize the
INT8 variables. It should allocate:

- `self._kernel` as an INT8 vector of shape `(input_dim,)` (the same shape as
  the original full-precision kernel).
- `self.scale` as the scalar quantization scale in the layer's variable dtype,
  which is FP32 in this case.


```python

def _int8_build(self, kernel_shape):
    (input_dim,) = kernel_shape
    self._kernel = self.add_weight(
        name="kernel",
        shape=(input_dim,),
        initializer="zeros",
        dtype="int8",
        trainable=False,
    )
    self.scale = self.add_weight(
        name="scale",
        initializer="ones",
        trainable=False,
    )

```

#### Note

1. INT8 variables should be created with `trainable=False`, as quantized parameters
  are not meant to be updated during training. Subsequent fine-tuning should not
  alter these quantized variables.
2. If you support INT4 quantization, implement a similar `_int4_build(...)`
  method that allocates packed INT4 storage (often packed into INT8) plus
  per-feature scales. The original unpacked dimensions and packing axis should
  be recorded as instance variables for use in the call path. A reference
  implementation is available in the Keras
  [Dense](https://github.com/keras-team/keras/blob/3c3d6adc08db627d89b5ad5e7f9b0ba3e88f2641/keras/src/layers/core/dense.py#L481-L512)
  layer.

### The `_int8_call(...)` method

The `_int8_call(...)` method implements a minimal INT8 forward path. It uses the
quantized variables allocated in `_int8_build(...)` and de-scales the output
back to floating-point.

The base `keras.Layer` class automatically dispatches to this method when the
layer is quantized, without requiring you to wire it up manually.

The INT8 path mirrors the float computation `y = x * w` but performs:

1. Elementwise multiply using the quantized weight.
2. De-scale back to float by dividing with the `scale`.


```python

def _int8_call(self, inputs, training=None):
    x = ops.multiply(inputs, self._kernel)
    x = ops.divide(x, self.scale)
    return x

```

---
## Complete `SimpleScale` class with hooks

Below is the full class definition that incorporates the all the hooks shown above (`quantize`, `_int8_build`,
`_int8_call`).


```python

class SimpleScale(Layer):
    """A layer that learns a per-feature scaling factor."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self._kernel = self.add_weight(
            name="kernel",
            shape=(input_dim,),
            initializer="random_uniform",
        )

    def call(self, inputs, training=None):
        return ops.multiply(inputs, self._kernel)

    def quantize(self, mode, **kwargs):
        if mode != "int8":
            raise NotImplementedError(f"Unsupported quantization mode: {mode}")

        quantized_kernel, scale = quantizers.abs_max_quantize(
            self._kernel, axis=0, dtype="int8", to_numpy=True
        )
        scale = ops.squeeze(scale, axis=0)

        kernel_shape = self._kernel.shape

        del self._kernel

        self._int8_build(kernel_shape)

        self._kernel.assign(quantized_kernel)
        self.scale.assign(scale)

        self._is_quantized = True

        if self.dtype_policy.quantization_mode is None:
            policy = dtype_policies.get(f"{mode}_from_{self.dtype_policy.name}")
            self.dtype_policy = policy

    def _int8_build(self, kernel_shape):
        (input_dim,) = kernel_shape
        self._kernel = self.add_weight(
            name="kernel",
            shape=(input_dim,),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        self.scale = self.add_weight(
            name="scale",
            initializer="ones",
            trainable=False,
        )

    def _int8_call(self, inputs, training=None):
        x = ops.multiply(inputs, self._kernel)
        x = ops.divide(x, self.scale)
        return x

```

---
## Try it: quantize and run a forward pass

Below we build the layer, then quantize to INT8 and call it again.


```python
# Sample inputs
rng = np.random.default_rng()
x = rng.random((2, 4)).astype("float32")

layer = SimpleScale()

# Forward pass in float
y_fp = layer(x)

# Quantize to INT8 and run again
layer.quantize("int8")
y_int8 = layer(x)

print("SimpleScale FP32 sample:", y_fp[0].numpy())
print("SimpleScale INT8 sample:", y_int8[0].numpy())

```

<div class="k-default-codeblock">
```
SimpleScale FP32 sample: [-0.01259411  0.00385596  0.0053392  -0.00877095]
SimpleScale INT8 sample: [-0.01256325  0.0038252   0.00535317 -0.00877098]
```
</div>

---
## Extending to INT4

If you want to support INT4 quantization, add:

- `_int4_build(...)`: allocate a packed 4-bit storage (often packed into int8) plus per-feature scales
- `_int4_call(...)`: unpack at runtime and follow the same de-scale pattern
- `quantize("int4")`: quantize weights with `value_range=(-8, 7)`, then pack to int4 storage

See the
[Dense](https://github.com/keras-team/keras/blob/3c3d6adc08db627d89b5ad5e7f9b0ba3e88f2641/keras/src/layers/core/dense.py#L602-L653)
reference for a complete packed int4 example, including how to track and use the
original (unpacked) dimension in the call path.

---
## Adding Serialization Support

Keras depends on a fixed serialization contract for saving and loading models.
This contract is complicated by quantization, since the variables you need to
save and load depend on the quantization mode.

The framework provides two hooks for layers to customize variable serialization:

- `save_own_variables(self, store)`: Write variables to `store` in a fixed
  order.
- `load_own_variables(self, store)`: Read variables from `store` in the same
  order.

Additionally, the `build(...)` method should also be modified to allocate the
correct variables based on presence (or absence) of a `self.quantization_mode`.

For this layer we only aim to support two modes (Non-quantized and INT8), so the
serialization contract is:

- None (no quantization): `["kernel"]`
- INT8: `["kernel", "scale"]`

The following code implements the required hooks; Keras will call them during
`model.save(...)` and `keras.saving.load_model(...)`.


```python

def save_own_variables(self, store):
    # Write variables to `store` in a fixed order based on quantization mode.
    # `store` is a key-value mapping provided by Keras during model.save().
    # Values are tensors.
    if not self.built:
        return
    mode = self.quantization_mode
    idx = 0
    if mode is None:
        # Order: _kernel
        store[str(idx)] = self._kernel
    elif mode == "int8":
        # Order: _kernel, scale
        store[str(idx)] = self._kernel
        idx += 1
        store[str(idx)] = self.scale
    else:
        raise ValueError(f"Unsupported quantization mode for save: {mode}")


def load_own_variables(self, store):
    # Read variables from `store` in the same order used by
    # `save_own_variables`. Keras calls this during
    # `keras.saving.load_model(...)`.
    if not self.built:
        return
    mode = self.quantization_mode
    idx = 0
    if mode is None:
        self._kernel.assign(store[str(idx)])
    elif mode == "int8":
        self._kernel.assign(store[str(idx)])
        idx += 1
        self.scale.assign(store[str(idx)])
    else:
        raise ValueError(f"Unsupported quantization mode for load: {mode}")

```

### Modify the `build(...)` method

The build method itself also needs to be aware of quantization mode. If a saved
quantized layer is being loaded/deserialized, `self.quantization_mode` will be
set before `build(...)` is called. In that case, we need to allocate quantized
variables directly instead of full-precision ones.


```python

def build(self, input_shape):
    input_dim = input_shape[-1]

    # Quantized build path.
    if self.quantization_mode:
        if self.quantization_mode == "int8":
            self._int8_build((input_dim,))
    else:
        # Regular FP32 build path.
        self._kernel = self.add_weight(
            name="kernel",
            shape=(input_dim,),
            initializer="random_uniform",
        )

```

---
## Complete implementation with serialization

The full class with serialization support looks like this:


```python

@keras.saving.register_keras_serializable()
class SimpleScale(Layer):
    """A layer that learns a per-feature scaling factor."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if self.quantization_mode:
            if self.quantization_mode == "int8":
                self._int8_build((input_dim,))
        else:
            self._kernel = self.add_weight(
                name="kernel",
                shape=(input_dim,),
                initializer="random_uniform",
            )

    def call(self, inputs, training=None):
        return ops.multiply(inputs, self._kernel)

    def quantize(self, mode, **kwargs):
        if mode != "int8":
            raise NotImplementedError(f"Unsupported quantization mode: {mode}")

        quantized_kernel, scale = quantizers.abs_max_quantize(
            self._kernel, axis=0, dtype="int8", to_numpy=True
        )
        scale = ops.squeeze(scale, axis=0)

        kernel_shape = self._kernel.shape

        del self._kernel

        self._int8_build(kernel_shape)

        self._kernel.assign(quantized_kernel)
        self.scale.assign(scale)

        self._is_quantized = True

        if self.dtype_policy.quantization_mode is None:
            policy = dtype_policies.get(f"{mode}_from_{self.dtype_policy.name}")
            self.dtype_policy = policy

    def _int8_build(self, kernel_shape):
        (input_dim,) = kernel_shape
        self._kernel = self.add_weight(
            name="kernel",
            shape=(input_dim,),
            initializer="zeros",
            dtype="int8",
            trainable=False,
        )
        self.scale = self.add_weight(
            name="scale",
            initializer="ones",
            trainable=False,
        )

    def _int8_call(self, inputs, training=None):
        x = ops.multiply(inputs, self._kernel)
        x = ops.divide(x, self.scale)
        return x

    def save_own_variables(self, store):
        # Write variables to `store` in a fixed order based on quantization mode.
        # `store` is a key-value mapping provided by Keras during model.save(); values are tensors.
        if not self.built:
            return
        mode = self.quantization_mode
        idx = 0
        if mode is None:
            # Order: _kernel
            store[str(idx)] = self._kernel
        elif mode == "int8":
            # Order: _kernel, scale
            store[str(idx)] = self._kernel
            idx += 1
            store[str(idx)] = self.scale
        else:
            raise ValueError(f"Unsupported quantization mode for save: {mode}")

    def load_own_variables(self, store):
        # Read variables from `store` in the same order used by `save_own_variables`.
        # Keras calls this during `keras.saving.load_model(...)`.
        if not self.built:
            return
        mode = self.quantization_mode
        idx = 0
        if mode is None:
            self._kernel.assign(store[str(idx)])
        elif mode == "int8":
            self._kernel.assign(store[str(idx)])
            idx += 1
            self.scale.assign(store[str(idx)])
        else:
            raise ValueError(f"Unsupported quantization mode for load: {mode}")

```

#### Note

The `@keras.saving.register_keras_serializable()` decorator is needed to
register the class for serialization.

---
## Try it: quantize, save, and load


```python
model = keras.Sequential([Input(shape=(4,)), SimpleScale()])
model.build((None, 4))

# Quantize to INT8.
model.quantize("int8")
y_int8 = model(x)
print("SimpleScale INT8 sample:", y_int8[0].numpy())

# Save and load the quantized model.
model.save("simplescale_int8.keras")
loaded = keras.saving.load_model("simplescale_int8.keras")

y_loaded = loaded(x)
print("Loaded INT8 sample:", y_loaded[0].numpy())
```

<div class="k-default-codeblock">
```
SimpleScale INT8 sample: [ 0.01568618 -0.00546078  0.00163636  0.00331613]
Loaded INT8 sample: [ 0.01568618 -0.00546078  0.00163636  0.00331613]
```
</div>

---
## Practical tips

Here are concrete patterns you can reuse when making your own layers PTQ-friendly.

- Build-time vs call-time responsibilities
  - In `build(...)`, if `self.quantization_mode` is set: allocate the quantized
    variables and skip allocating the float kernel to avoid duplicates.
- Record any metadata you need for the call path, e.g., for INT4:
  - The axis you packed along (e.g., `_int4_pack_axis`).
  - The original (unpacked) length on that axis (e.g., `_original_input_dim` or
    `_original_length_along_pack_axis`).
- In quantized call hooks, compute with the quantized buffers and de-scale back
  to float at the end, wherever possible. This allows you to leverage optimized
  low-precision kernels (e.g., cuBLAS INT8 GEMM).

- INT4 specifics (packed nibbles)
  - Quantize to INT4 values in range [-8, 7] (still dtype int8), then pack two
    4-bit integers per byte with `quantizers.pack_int4(..., axis=pack_axis)`.
  - Store the packed kernel with `dtype="int8"`. Unpack on the fly in the call
    path with `quantizers.unpack_int4(packed, orig_len, axis=pack_axis)`.
  - Keep the original length and pack axis so you can unpack for LoRA,
    gradients, and serialization.

- Inputs quantization and broadcasting
  - In the forward path de-scale outputs using
  `outputs /= (inputs_scale * kernel_scale)`; make sure both scales broadcast to
  the output shape.

- Dtype policy lifecycle
  - During `quantize(mode)`: delete FP32 variables, allocate quantized ones,
    assign values, then set `self._is_quantized = True` before changing the
    dtype policy.
  - Only change policy if the current policy has `quantization_mode is None` to
    avoid an infinite loop.

- Serialization contract
  - Provide a fixed-order logic for variable serialization so save/load is
    deterministic.
  - Write variables in a fixed order per mode (e.g., None: [kernel, bias],
    `"int8"`: [kernel, bias, kernel_scale], `"int4"`: [kernel, bias, kernel_scale]).

- Validation and error handling
  - Validate `mode` early and raise a `NotImplementedError` for unsupported
    modes.
  - After quantization, run a tiny smoke test and assert the output matches the
    FP32 path and values are within a reasonable tolerance after de-scale.

- Performance hygiene
  - Avoid repeated transformations hot paths; precompute as much information
    as possible and keep the forward-pass hooks lightweight.
  - Keep quantized buffers `trainable=False` and prefer vectorized operations.

For more advanced patterns, refer to the
[Dense](https://github.com/keras-team/keras/blob/3c3d6adc08db627d89b5ad5e7f9b0ba3e88f2641/keras/src/layers/core/dense.py) and
[EinsumDense](https://github.com/keras-team/keras/blob/3c3d6adc08db627d89b5ad5e7f9b0ba3e88f2641/keras/src/layers/core/einsum_dense.py)
reference implementations.

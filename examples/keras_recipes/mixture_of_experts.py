"""
Title: Standalone Mixture-of-Experts (MoE) Layers
Author: [Abhinav Kumar Singh](https://github.com/Abhinavexists)
Date created: 2025/12/10
Last modified: 2025/12/10
Description: Implementing standalone MoE layers as drop-in replacements for Dense and Conv2D.
Accelerator: GPU
"""

"""
## Introduction

This example demonstrates the implementation of Mixture-of-Experts (MoE) layers
as drop-in replacements for standard Keras `Dense` and `Conv2D` layers.

MoE is a technique for scaling neural networks that increases model capacity while
keeping computational costs manageable. Instead of processing all inputs through
the same parameters, MoE layers use a gating mechanism to route each input to a
weighted combination of specialized "expert" networks. This allows the model to
learn specialized sub-networks for different types of inputs.

The implementation is based on the approach described in
[Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)
by Shazeer et al., and draws inspiration from the reference implementation by
[Emin Orhan](https://github.com/eminorhan/mixture-of-experts).

In this example, we implement two MoE layer types:

- **DenseMoE**: A Mixture-of-Experts version of the Dense layer
- **Conv2DMoE**: A Mixture-of-Experts version of the Conv2D layer

We demonstrate their usage on the CIFAR-10 image classification task and
compare performance against a baseline model using standard layers.
"""

"""
## Setup
"""

import keras
from keras import layers, ops, activations, optimizers
from keras.datasets import cifar10
import matplotlib.pyplot as plt

"""
## Implement the DenseMoE layer

The DenseMoE layer computes: `output = sum_i gate_i(x) * expert_i(x)`

where `gate_i(x)` is the learned gating weight for expert `i` and `expert_i(x)`
is the output of expert `i`. Each input is processed by all experts, and their
outputs are combined using learned gating weights. This approach uses soft routing,
which is more efficient than top-k routing for smaller numbers of experts.
"""


class DenseMoE(layers.Layer):
    """
    Mixture-of-Experts Dense layer.

    A drop-in replacement for keras.layers.Dense that uses multiple expert
    networks combined via a learned gating mechanism.

    Args:
        units: Positive Integer, dimensionality of the output space
        n_experts: Positive Integer, number of expert Dense layers
        expert_activation: Activation function for expert model (for example, "relu" or "tanh")
        gating_activation: Activation function for gating network (for example, "softmax")
        use_expert_bias: Boolean, whether to use bias in expert layers
        use_gating_bias: Boolean, whether to use bias in gating layer
    """

    def __init__(
        self,
        units,
        n_experts,
        expert_activation=None,
        gating_activation=None,
        use_expert_bias=True,
        use_gating_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.n_experts = n_experts
        self.expert_activation = activations.get(expert_activation)
        self.gating_activation = activations.get(gating_activation)
        self.use_expert_bias = use_expert_bias
        self.use_gating_bias = use_gating_bias

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.expert_kernel = self.add_weight(
            name="expert_kernel",
            shape=(input_dim, self.units, self.n_experts),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_expert_bias:
            self.expert_bias = self.add_weight(
                name="expert_bias",
                shape=(self.units, self.n_experts),
                initializer="zeros",
                trainable=True,
            )

        self.gating_kernel = self.add_weight(
            name="gating_kernel",
            shape=(input_dim, self.n_experts),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_gating_bias:
            self.gating_bias = self.add_weight(
                name="gating_bias",
                shape=(self.n_experts,),
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def call(self, inputs):
        # Compute expert outputs
        expert_outputs = ops.tensordot(inputs, self.expert_kernel, axes=1)
        if self.use_expert_bias:
            expert_outputs = expert_outputs + self.expert_bias
        expert_outputs = self.expert_activation(expert_outputs)

        # Compute gating weights
        gating_outputs = ops.tensordot(inputs, self.gating_kernel, axes=1)
        if self.use_gating_bias:
            gating_outputs = gating_outputs + self.gating_bias
        gating_outputs = self.gating_activation(gating_outputs)

        # Load balancing loss: encourages uniform expert utilization
        if self.trainable:
            importance = ops.mean(gating_outputs, axis=0)
            load_loss = self.n_experts * ops.sum(ops.square(importance))
            self.add_loss(1e-2 * load_loss)

        # Weighted combination of expert outputs
        gating_outputs = ops.expand_dims(gating_outputs, axis=1)
        output = ops.sum(expert_outputs * gating_outputs, axis=-1)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "n_experts": self.n_experts,
                "expert_activation": activations.serialize(self.expert_activation),
                "gating_activation": activations.serialize(self.gating_activation),
                "use_expert_bias": self.use_expert_bias,
                "use_gating_bias": self.use_gating_bias,
            }
        )
        return config


"""
## Implement the Conv2DMoE layer

The Conv2DMoE layer applies the MoE concept to convolutional operations.
Similar to DenseMoE, it uses multiple expert Conv2D layers combined via learned
gating weights. The gating network uses 1x1 convolutions for efficient
per-position gating across spatial dimensions.
"""


class Conv2DMoE(layers.Layer):
    """
    Mixture-of-Experts Conv2D layer.

    A drop-in replacement for keras.layers.Conv2D that uses multiple expert
    convolutional models combined via a learned gating mechanism.

    Args:
        filters: Positive Integer, number of output filters
        kernel_size: Size of the convolutional kernel
        n_experts: Positive Integer, number of expert Conv2D layers
        strides: Positive Integer or Tuple of 2 Integers, stride of the convolution
        padding: String, padding mode ('valid' or 'same')
        expert_activation: Activation function for expert model (for example, "relu" or "tanh")
        gating_activation: Activation function for gating network (for example, "softmax")
        use_expert_bias: Boolean, whether to use bias in expert layers
        use_gating_bias: Boolean, whether to use bias in gating layer
    """

    def __init__(
        self,
        filters,
        kernel_size,
        n_experts,
        strides=1,
        padding="valid",
        expert_activation=None,
        gating_activation=None,
        use_expert_bias=True,
        use_gating_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.n_experts = n_experts
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
        self.expert_activation = activations.get(expert_activation)
        self.gating_activation = activations.get(gating_activation)
        self.use_expert_bias = use_expert_bias
        self.use_gating_bias = use_gating_bias

    def build(self, input_shape):
        input_channels = input_shape[-1]

        self.expert_kernel = self.add_weight(
            name="expert_kernel",
            shape=(
                self.kernel_size[0],
                self.kernel_size[1],
                input_channels,
                self.filters,
                self.n_experts,
            ),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_expert_bias:
            self.expert_bias = self.add_weight(
                name="expert_bias",
                shape=(self.filters, self.n_experts),
                initializer="zeros",
                trainable=True,
            )

        # Gating uses 1x1 convolution for efficient per-position gating
        self.gating_kernel = self.add_weight(
            name="gating_kernel",
            shape=(1, 1, input_channels, self.n_experts),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_gating_bias:
            self.gating_bias = self.add_weight(
                name="gating_bias",
                shape=(self.n_experts,),
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def call(self, inputs):
        # Compute expert outputs
        expert_kernel = ops.reshape(
            self.expert_kernel,
            (
                self.kernel_size[0],
                self.kernel_size[1],
                ops.shape(inputs)[-1],
                self.filters * self.n_experts,
            ),
        )
        expert_outputs = ops.conv(
            inputs,
            expert_kernel,
            strides=self.strides,
            padding=self.padding,
        )
        output_shape = ops.shape(expert_outputs)
        expert_outputs = ops.reshape(
            expert_outputs,
            (
                output_shape[0],
                output_shape[1],
                output_shape[2],
                self.filters,
                self.n_experts,
            ),
        )
        if self.use_expert_bias:
            expert_outputs = expert_outputs + self.expert_bias
        expert_outputs = self.expert_activation(expert_outputs)

        # Compute gating weights using 1x1 convolution
        gating_outputs = ops.conv(
            inputs,
            self.gating_kernel,
            strides=self.strides,
            padding=self.padding,
        )
        if self.use_gating_bias:
            gating_outputs = gating_outputs + self.gating_bias
        gating_outputs = self.gating_activation(gating_outputs)

        # Load balancing loss: encourages uniform expert utilization
        if self.trainable:
            importance = ops.mean(gating_outputs, axis=[0, 1, 2])
            load_loss = self.n_experts * ops.sum(ops.square(importance))
            self.add_loss(1e-2 * load_loss)

        # Weighted combination of expert outputs
        gating_outputs = ops.expand_dims(gating_outputs, axis=-2)
        output = ops.sum(expert_outputs * gating_outputs, axis=-1)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "n_experts": self.n_experts,
                "strides": self.strides,
                "padding": self.padding,
                "expert_activation": activations.serialize(self.expert_activation),
                "gating_activation": activations.serialize(self.gating_activation),
                "use_expert_bias": self.use_expert_bias,
                "use_gating_bias": self.use_gating_bias,
            }
        )
        return config


"""
## Prepare the data

For this example, we use the CIFAR-10 dataset. To keep training time
reasonable for demonstration purposes, we use only a subset of the training data.
"""

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

"""
## Build a baseline model

First, we create a baseline CNN model using standard Keras layers.
"""


def create_baseline_model():
    inputs = keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="baseline_model")
    return model


"""
## Build a model with MoE layers

Now we create an equivalent model using MoE layers. We replace the Conv2D
and Dense layers with their MoE counterparts: Conv2DMoE and DenseMoE.
"""


def create_moe_model():
    inputs = keras.Input(shape=(32, 32, 3))

    x = Conv2DMoE(
        32,
        3,
        n_experts=4,
        padding="same",
        expert_activation="relu",
        gating_activation="softmax",
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = Conv2DMoE(
        64,
        3,
        n_experts=4,
        padding="same",
        expert_activation="relu",
        gating_activation="softmax",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = DenseMoE(
        128,
        n_experts=8,
        expert_activation="relu",
        gating_activation="softmax",
    )(x)
    x = layers.Dropout(0.5)(x)
    x = DenseMoE(
        64,
        n_experts=4,
        expert_activation="relu",
        gating_activation="softmax",
    )(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="moe_model")
    return model


"""
## Train and compare models

We train both models and compare their performance. The MoE model has more
total parameters due to the multiple expert networks, but can potentially
learn more diverse representations through expert specialization.
"""

epochs = 10
batch_size = 128

print("BASELINE MODEL")
baseline_model = create_baseline_model()
baseline_model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
baseline_model.summary()

print("\nTraining baseline model...")
baseline_history = baseline_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1,
)

baseline_test_loss, baseline_test_acc = baseline_model.evaluate(
    x_test, y_test, verbose=0
)
print(f"\nBaseline Test Accuracy: {baseline_test_acc:.4f}")

print("\nMIXTURE-OF-EXPERTS MODEL")
moe_model = create_moe_model()
moe_model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
moe_model.summary()

print("\nTraining MoE model...")
moe_history = moe_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1,
)

moe_test_loss, moe_test_acc = moe_model.evaluate(x_test, y_test, verbose=0)
print(f"\nMoE Test Accuracy: {moe_test_acc:.4f}")

print("\nMODEL COMPARISON")
print(
    f"Baseline - Params: {baseline_model.count_params():,} | "
    f"Test Accuracy: {baseline_test_acc:.4f}"
)
print(
    f"MoE Model - Params: {moe_model.count_params():,} | "
    f"Test Accuracy: {moe_test_acc:.4f}"
)

"""
## Visualize training history

Let's compare the training curves of both models to understand their learning dynamics.
"""

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(baseline_history.history["accuracy"], label="Baseline Train", linestyle="--")
plt.plot(baseline_history.history["val_accuracy"], label="Baseline Val", linestyle="--")
plt.plot(moe_history.history["accuracy"], label="MoE Train")
plt.plot(moe_history.history["val_accuracy"], label="MoE Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(baseline_history.history["loss"], label="Baseline Train", linestyle="--")
plt.plot(baseline_history.history["val_loss"], label="Baseline Val", linestyle="--")
plt.plot(moe_history.history["loss"], label="MoE Train")
plt.plot(moe_history.history["val_loss"], label="MoE Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss Comparison")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

"""
## Analyzing expert utilization

One key aspect of MoE models is how the gating network learns to distribute
inputs across experts. Let's examine the gating weights to understand expert
utilization patterns.
"""

dense_moe_layer = None
for layer in moe_model.layers:
    if isinstance(layer, DenseMoE):
        dense_moe_layer = layer
        break

if dense_moe_layer:
    feature_extractor = keras.Model(
        inputs=moe_model.input,
        outputs=dense_moe_layer.input,
    )

    test_features = feature_extractor.predict(x_test[:100], verbose=0)

    gating_logits = ops.tensordot(test_features, dense_moe_layer.gating_kernel, axes=1)
    if dense_moe_layer.use_gating_bias:
        gating_logits = gating_logits + dense_moe_layer.gating_bias
    gating_weights = dense_moe_layer.gating_activation(gating_logits)

    gating_weights_np = ops.convert_to_numpy(gating_weights)

    print("\nGating weights distribution across experts:")
    print(f"Mean gating weights per expert: {gating_weights_np.mean(axis=0)}")
    print(f"Std of gating weights per expert: {gating_weights_np.std(axis=0)}")
    print("\nA more uniform distribution indicates all experts are being utilized.")
    print("Highly skewed distributions suggest some experts dominate.")

    mean_weights = gating_weights_np.mean(axis=0)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(mean_weights)), mean_weights)
    plt.xlabel("Expert Index")
    plt.ylabel("Average Gating Weight")
    plt.title("Average Expert Utilization")
    plt.grid(True, alpha=0.3, axis="y")

    plt.subplot(1, 2, 2)
    plt.imshow(gating_weights_np[:50].T, aspect="auto", cmap="viridis")
    plt.xlabel("Sample Index")
    plt.ylabel("Expert Index")
    plt.title("Gating Weights Heatmap (50 samples)")
    plt.colorbar(label="Gating Weight")

    plt.tight_layout()
    plt.show()

"""
## Key takeaways

This example demonstrated how to implement Mixture-of-Experts layers as drop-in
replacements for standard Keras layers. Here are the main points:

**MoE characteristics:**

- **Increased capacity**: MoE models have more parameters due to multiple expert
  networks, which can capture more diverse patterns in the data.
- **Learned routing**: The gating network learns to dynamically route inputs to
  appropriate experts based on the input features.
- **Soft routing**: This implementation uses soft routing (weighted combination
  of all experts) rather than hard top-k selection, providing smooth gradients
  during training.

**When to use MoE layers:**

- You need high model capacity but want conditional computation
- Your dataset has diverse patterns that could benefit from specialized sub-networks
- You have sufficient training data to properly train multiple experts
- You're working on large-scale tasks where model capacity is important

**Practical tips:**

- Start with 4-8 experts and increase if needed
- Monitor gating weights to ensure all experts are being utilized
- MoE layers work best in deeper parts of the network where features are more abstract
- For very large numbers of experts (100+), consider top-k routing for efficiency
  (see the [Switch Transformer example](https://keras.io/examples/nlp/text_classification_with_switch_transformer/))

**Implementation notes:**

This implementation uses soft routing, which computes a weighted combination of
all expert outputs. This approach provides smooth gradients and is well-suited
for moderate numbers of experts (4-8).

We also include a **load balancing loss** that encourages uniform expert utilization.
Without this loss, the model may learn to rely heavily on just one or two dominant
experts, wasting the capacity of the others. The load balancing loss penalizes
imbalanced gating distributions, helping ensure all experts contribute to the model.

For very large expert counts (100+), consider top-k routing for efficiency
(see the [Switch Transformer example](https://keras.io/examples/nlp/text_classification_with_switch_transformer/)),
where load balancing becomes even more critical.
"""

"""
## Conclusion

This example showed how to implement Mixture-of-Experts layers as drop-in
replacements for standard Keras Dense and Conv2D layers. MoE enables scaling
model capacity through expert specialization while maintaining computational
efficiency through learned routing.

You can use `DenseMoE` and `Conv2DMoE` in your own models wherever you would
normally use `Dense` or `Conv2D` layers. Try experimenting with different numbers
of experts and observe how the gating network learns to route inputs for your
specific task and dataset.
"""

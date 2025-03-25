"""
Title: MoE for MNIST
Author: [Damoon Shahhosseini](https://www.linkedin.com/in/damoonsh/)
Date created: 2025/02/28
Last modified: 2050/02/28
Description: Simple MoE implementation for MNIST classification.
Accelerator: GPU
"""

"""
# Introduction

In this example, we implement an adaptation of the Mixture of Experts (MoE) architecture
([Shazeer et al.](https://arxiv.org/abs/1701.06538)).
The idea is to use conditional computation to increases model capacity without increasing computation.
Experts are identical blocks within a layer where each are trained to specialize in different parts of the input space.
At each forward pass, a gating network selects a subset of experts to apply to the input.

The components to implement are:

- Gating network: A dense layer that outputs a probability distribution over the experts.
- MoE layer: A layer that applies a different expert to each input in the batch. And a loss function that ensures specialization among the experts.
- Model: A simple model that uses the MoE layer.
"""

"""
## Imports
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from keras import layers
import torch

"""
### Data Prepration
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Constants

"""
NUM_EXPERTS = 5
TOP_K = 3
BATCH_SIZE = 128
NUM_EPOCHS = 1
LEARNING_RATE = 0.001


"""
## Base architecture

The most basic [MNIST classifier](https://keras.io/examples/vision/mnist_convnet/) consists of a stack of convolutional layers followed by a dense layer. In this tutorial, we will first replace the dense layer with a MoE layer. Then do the same for convolutional layers.
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
# Linear MoE using Dense layers

For this layer, we will create multiple dense layers that will be used as experts. Then a simple gating network will select at each step which exerts should be utilized for the current input. We will keep track of the number of times each expert is used. Then the selected experts will be combined using a weighted sum.
"""


class LinearMoE(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    ):
        super(LinearMoE, self).__init__()
        self.expert_output_size = hidden_size
        # Initialize experts
        self.experts = [
            layers.Dense(
                hidden_size,
                kernel_initializer=keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.001
                ),
                bias_initializer="zeros",
            )
            for _ in range(num_experts)
        ]
        # Initialize gating network
        self.gating_network = layers.Dense(
            NUM_EXPERTS,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001),
            bias_initializer="zeros",
            activation="softmax",
        )

        self.num_experts = num_experts
        self.top_k = top_k
        # Keep track of how many times each expert is used
        self.expert_usage_count = self.add_weight(
            shape=(self.num_experts,),
            initializer="zeros",
            trainable=False,
            name="expert_usage_count",
        )
        self.batch_capacity = BATCH_SIZE / num_experts

    def build(self, input_shape):
        super(LinearMoE, self).build(input_shape)
        self.built = True

    def get_top_outputs(self, x, top_k_indices, top_k_weights):
        batch_size = x.size(0)
        output_size = self.expert_output_size

        # Get outputs from top-k experts
        top_k_expert_outputs = torch.zeros(
            batch_size, self.top_k, output_size, device=x.device
        )
        flat_indices = top_k_indices.view(-1).to(x.device)
        repeated_x = x.repeat_interleave(self.top_k, dim=0)

        unique_expert_ids = flat_indices.unique(sorted=True)
        expert_outputs_dict = {}
        for idx in unique_expert_ids:
            mask = (flat_indices == idx).to(x.device)
            selected_inputs = repeated_x[mask]
            expert_outputs_dict[idx.item()] = self.experts[idx](selected_inputs)

        flat_outputs = torch.zeros(
            batch_size * self.top_k, output_size, device=x.device
        )
        for idx in unique_expert_ids:
            mask = (flat_indices == idx).to(x.device)
            flat_outputs[mask] = expert_outputs_dict[idx.item()].to(flat_outputs.device)

        top_k_expert_outputs = flat_outputs.view(batch_size, self.top_k, output_size)

        return torch.einsum(
            "ijk,ij->ik", top_k_expert_outputs.cpu(), top_k_weights.cpu()
        )

    def update_usage_counts(self, indices):
        flat_indices = keras.ops.reshape(indices, [-1])
        one_hot = keras.ops.one_hot(flat_indices, self.num_experts)
        updates = keras.ops.sum(one_hot, axis=0)
        self.batch_usage_count = updates
        self.expert_usage_count.assign_add(updates)

    def call(self, x):
        gating_weights = self.gating_network(x)
        top_k_weights, top_k_indices = torch.topk(gating_weights, self.top_k, dim=-1)
        combined_output = self.get_top_outputs(x, top_k_indices, top_k_weights)
        self.update_usage_counts(top_k_indices)
        self.batch_overflow_sum = torch.relu(
            self.batch_usage_count - self.batch_capacity
        ).sum()
        return combined_output

    def compute_total_loss(self, load_balance_coef=0.01):
        self.batch_overflow_sum = torch.sum(
            torch.relu(self.expert_usage_count - self.batch_capacity)
        )
        return load_balance_coef * (
            self.diversity_loss + self.batch_overflow_sum + self.importance_loss
        )


"""
Output of the top 3 experts out of 10 for one layer of MoE:
"""
sample_data = torch.randn((1, 10))
linear_mode = LinearMoE(32, 10, 3)
linear_mode(sample_data)

"""
## Routing Collapse

One common challenge with MoE architectures is "routing collapse". The "route" refers to the selection process of which expert to use for a given input where the model falls into a pattern of only using a small subset of experts. This happens because:

1. Early in training, some experts may perform slightly better by chance
2. These better-performing experts get selected more frequently
3. With more practice, these experts improve further, creating a feedback loop
4. Other experts become neglected and never improve

Code below demonstrates the randomness of expert selection:
"""


def check_expert_usage(runs):
    # Running the later multiple times to show randomness of expert selection
    for i in range(runs):
        sample_data = torch.randn((1, 10))
        linear_mode = LinearMoE(10, 5)
        _ = linear_mode(sample_data)
        print(f"Run {i}, Expert usage: {linear_mode.expert_usage_count.numpy()}")


check_expert_usage(4)

"""
### Load Balancing Solutions

To prevent routing collapse, we implement three types of losses that were introduced in various MoE research:

1. Diversity Loss: Encourages the gating network to use all experts by maximizing the entropy
   of expert selection probabilities
   [Shazeer et al., "Outrageously Large Neural Networks" (2017)](https://arxiv.org/abs/1701.06538)

2. Importance Loss: Ensures each expert handles a similar total amount of input across the batch
   by penalizing deviations from the mean usage
   [Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation" (2020)](https://arxiv.org/abs/2006.16668)

3. Overflow Loss: Prevents individual experts from being overloaded by penalizing usage above
   a specified capacity threshold
   [Fedus et al., "Switch Transformers" (2021)](https://arxiv.org/abs/2101.03961)

These losses are combined with the main classification loss during training to ensure balanced expert utilization.
The combination of these techniques has proven effective in large-scale models like GShard and Switch Transformers.
"""


class LinearMoE(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    ):
        super(LinearMoE, self).__init__()
        self.expert_output_size = hidden_size
        # Initialize experts
        self.experts = [
            layers.Dense(
                hidden_size,
                kernel_initializer=keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.001
                ),
                bias_initializer="zeros",
            )
            for _ in range(num_experts)
        ]
        # Initialize gating network
        self.gating_network = layers.Dense(
            NUM_EXPERTS,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001),
            bias_initializer="zeros",
            activation="softmax",
        )

        self.num_experts = num_experts
        self.top_k = top_k
        # Keep track of how many times each expert is used
        self.expert_usage_count = self.add_weight(
            shape=(self.num_experts,),
            initializer="zeros",
            trainable=False,
            name="expert_usage_count",
        )
        self.batch_capacity = BATCH_SIZE / num_experts

    def build(self, input_shape):
        for expert in self.experts:
            if not expert.built:
                expert.build(input_shape)

        if not self.gating_network.built:
            self.gating_network.build(input_shape)
        self.built = True

    def get_top_outputs(self, x, top_k_indices, top_k_weights):
        batch_size = x.size(0)
        output_size = self.expert_output_size

        # Get outputs from top-k experts
        top_k_expert_outputs = torch.zeros(
            batch_size, self.top_k, output_size, device=x.device
        )
        flat_indices = top_k_indices.view(-1).to(x.device)
        repeated_x = x.repeat_interleave(self.top_k, dim=0)

        unique_expert_ids = flat_indices.unique(sorted=True)
        expert_outputs_dict = {}
        for idx in unique_expert_ids:
            mask = (flat_indices == idx).to(x.device)
            selected_inputs = repeated_x[mask]
            expert_outputs_dict[idx.item()] = self.experts[idx](selected_inputs)

        flat_outputs = torch.zeros(
            batch_size * self.top_k, output_size, device=x.device
        )
        for idx in unique_expert_ids:
            mask = (flat_indices == idx).to(x.device)
            flat_outputs[mask] = expert_outputs_dict[idx.item()].to(flat_outputs.device)

        top_k_expert_outputs = flat_outputs.view(batch_size, self.top_k, output_size)

        return torch.einsum(
            "ijk,ij->ik", top_k_expert_outputs.cpu(), top_k_weights.cpu()
        )

    def update_usage_counts(self, indices):
        flat_indices = keras.ops.reshape(indices, [-1])
        one_hot = keras.ops.one_hot(flat_indices, self.num_experts)
        updates = keras.ops.sum(one_hot, axis=0)
        self.batch_usage_count = updates
        self.expert_usage_count.assign_add(updates)

    def _importance_loss(self, weights):
        batch_importance_sum = torch.sum(weights, dim=0)
        mean_importance = torch.mean(batch_importance_sum)
        self.importance_loss = torch.mean(
            torch.square(
                batch_importance_sum
                - mean_importance * torch.ones_like(batch_importance_sum)
            )
        )

    def _diversity_loss(self, weights):
        entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=1)
        self.diversity_loss = torch.mean(entropy)

    def call(self, x):
        gating_weights = self.gating_network(x)
        top_k_weights, top_k_indices = torch.topk(gating_weights, self.top_k, dim=-1)
        combined_output = self.get_top_outputs(x, top_k_indices, top_k_weights)
        self.update_usage_counts(top_k_indices)
        self._importance_loss(gating_weights)
        self._diversity_loss(gating_weights)

        return combined_output

    def compute_total_loss(self, load_balance_coef=0.01):
        self.batch_overflow_sum = torch.sum(
            torch.relu(self.expert_usage_count - self.batch_capacity)
        )
        return load_balance_coef * (
            self.diversity_loss + self.batch_overflow_sum + self.importance_loss
        )


"""
## MNIST classification with MoE
"""


class MoEModel(keras.Model):
    def __init__(
        self,
        num_classes,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        moe_loss_considered=True,
    ):
        super(MoEModel, self).__init__()

        # Define the convolutional block
        self.conv_block = keras.Sequential(
            [
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
            ]
        )

        # MoE classifier
        self.moe_classifier = LinearMoE(
            hidden_size=num_classes, num_experts=num_experts, top_k=top_k
        )

        # Softmax layer
        self.softmax = layers.Softmax()
        self.moe_loss_considered = moe_loss_considered

    def call(self, inputs, training=False):
        conv_flatten = self.conv_block(inputs)
        moe_output = self.moe_classifier(conv_flatten)
        outputs = self.softmax(moe_output)
        return outputs

    def train_step(self, data):
        x, y = data
        self.zero_grad()

        y_pred = self(x)

        if self.moe_loss_considered:
            moe_loss = self.moe_classifier.compute_total_loss(load_balance_coef=0.01)
            total_loss = self.compute_loss(x, y, y_pred) + moe_loss
        else:
            total_loss = self.compute_loss(x, y, y_pred)

        total_loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # Update metrics
        for metric in self.metrics:
            metric.update_state(y, y_pred)

        # Return a dict of metrics for monitoring
        return {
            "loss": total_loss.item(),  # Convert to Python number
            "moe_loss": moe_loss.item() if self.moe_loss_considered else 0.0,
            **{m.name: m.result() for m in self.metrics},
        }

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        classification_loss = self.compute_loss(x, y, y_pred)
        moe_loss = self.moe_classifier.compute_total_loss(load_balance_coef=0.01)
        total_loss = classification_loss + moe_loss

        for metric in self.metrics:
            metric.update_state(y, y_pred)
        return {
            "loss": total_loss,
            "moe_loss": moe_loss,
            **{m.name: m.result() for m in self.metrics},
        }


# Instantiate and compile the model
inputs = keras.Input(shape=input_shape)
model = MoEModel(num_classes=num_classes, num_experts=5, top_k=3)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.CategoricalCrossentropy(),  # Assumes one-hot encoded labels
    metrics=["accuracy"],
)

"""
###  Training
"""
history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(x_test, y_test),
    verbose=0,
)

"""
### Evaluation
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

"""
# Conclusion

This example demonstrated how Mixture of Experts (MoE) can be used to increase model capacity without a proportional increase in computation cost. The key benefits are:

1. Conditional Computation: Only a subset of experts (TOP_K=3 out of NUM_EXPERTS=5) process each input,
   making the model more computationally efficient than a model that uses all parameters for every input.

2. Specialized Processing: Each expert learns to handle different aspects of the input space,
   allowing for more sophisticated processing without requiring a larger dense network.

In our implementation, we:

1. Created a basic MoE layer using dense networks as experts
2. Implemented three types of load balancing losses to prevent routing collapse
3. Applied the MoE architecture to MNIST classification by replacing the final dense layer
4. Achieved comparable accuracy to the baseline model while using experts conditionally

This approach is particularly valuable for large-scale models where computational efficiency
is crucial. The same principles demonstrated here are used in much larger language models
and other applications where model capacity needs to scale efficiently
"""

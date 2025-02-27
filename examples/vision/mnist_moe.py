"""
Title: MoE for MNIST
Author: [Damoon Shahhosseini](https://www.linkedin.com/in/damoonsh/)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: Showcasing concepts relates to Mixture of Experts (MoE).
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

In this example, we will first implement a linear MoE layer and then a CNN-based MoE layer. Lastly we will combine the two using an abstract implementation to showcase its capacties.
"""

"""
## Imports
"""

import numpy as np
import keras
from keras import layers, models
import tensorflow as tf
from tensorflow.keras import backend as K

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
NUM_EPOCHS = 20
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

        # Initialize experts
        self.experts = [
            layers.Dense(
                hidden_size,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.001
                ),
                bias_initializer="zeros",
            )
            for _ in range(num_experts)
        ]
        # Initialize gating network
        self.gating_network = layers.Dense(
            NUM_EXPERTS,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.001
            ),
            bias_initializer="zeros",
        )

        self.num_experts = num_experts
        self.top_k = top_k
        # Keep track of how many times each expert is used
        self.expert_usage_count = tf.Variable(
            tf.zeros((num_experts,), dtype=tf.float32)
        )

    def call(self, x):
        # Get gating weights
        gating_weights = self.gating_network(x)

        # Get the top k experts based on the gating weights
        top_k_weights, top_k_indices = tf.math.top_k(gating_weights, k=self.top_k)

        # Count usage of each expert symbolically
        updates = tf.ones_like(tf.reshape(top_k_indices, [-1]), dtype=tf.float32)
        # Use tf.tensor_scatter_nd_add to increment the usage count
        self.expert_usage_count.assign(
            tf.tensor_scatter_nd_add(
                self.expert_usage_count, tf.reshape(top_k_indices, [-1, 1]), updates
            )
        )

        # Get outputs from only the top-k experts
        top_k_expert_outputs = tf.stack(
            [
                self.experts[expert_index](x)
                for expert_index in top_k_indices.numpy()[0]
            ],
            axis=1,
        )  # Stack outputs along axis 1

        # Combine outputs using top-k weights
        combined_output = tf.einsum("ijk,ij->ik", top_k_expert_outputs, top_k_weights)

        return combined_output


"""
Output of the top 3 experts out of 10 for one layer of MoE:
"""
sample_data = tf.random.uniform((1, 10))
linear_mode = LinearMoE(32, 10, 3)
linear_mode(sample_data)

"""
## Routing Collapse

Routing collapse is a problem that occurs with MoE layers. The route terminology refers to the selection process of which expert to use for a given input.

Route collapse happens when a routing model, early in training, starts favoring just a few experts because they perform slightly better due to random starting conditions. This leads to most examples being sent to these experts, leaving others unused and reducing the model’s overall capacity.

Code below demonstrates the randomness of expert selection:
"""


def check_expert_usage(runs):
    # Running the later multiple times to show randomness of expert selection
    for i in range(runs):
        sample_data = tf.random.uniform((1, 10))
        linear_mode = LinearMoE(10, 5)
        _ = linear_mode(sample_data)
        print(f"Run {i}, Expert usage: {linear_mode.expert_usage_count.numpy()}")


check_expert_usage(4)

"""
### Adding loss functions to prevent route collapse
To fix this, the authors use extra rules (importance and load losses), ideas borrowed from [Shazeer et al.](https://arxiv.org/abs/1701.06538), to ensure all experts get used evenly.

The importance_loss calculates how much the usage of each expert (tracked in batch_importance_sum) deviates from the average usage (mean_importance) by using mean squared error, aiming to balance expert utilization. This helps prevent route collapse by discouraging the model from overloading a few experts, instead promoting an even distribution of examples across all experts to maintain diverse and effective routing.

#### Load losses
    - Diversity loss: Diversity loss helps prevent route collapse by encouraging the routing model to evenly distribute examples across all experts, rather than favoring just a few due to their initial performance. It does this by maximizing the entropy of the gating weights, ensuring balanced expert utilization and improving the model's overall capacity.
    - Overflow loss: The batch_overflow_sum measures how much the usage of experts exceeds a set capacity by applying ReLU to the difference between usage_counts (how many examples each expert handles) and batch_capacity (the allowed limit), then summing the excesses. This helps prevent route collapse by penalizing situations where certain experts are overused, encouraging a more even spread of examples across all experts to keep the model's capacity balanced.
"""


class LinearMoE(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    ):
        super(LinearMoE, self).__init__()

        # Initialize experts
        self.experts = [
            layers.Dense(
                hidden_size,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.001
                ),
                bias_initializer="zeros",
            )
            for _ in range(num_experts)
        ]
        # Initialize gating network
        self.gating_network = layers.Dense(
            num_experts,  # Match output to num_experts
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.001
            ),
            bias_initializer="zeros",
        )

        self.num_experts = num_experts
        self.top_k = top_k
        # Keep track of how many times each expert is used as a layer weight
        self.expert_usage_count = tf.Variable(
            tf.zeros((num_experts,), dtype=tf.float32)
        )

        self.batch_capacity = BATCH_SIZE // num_experts

    def _diversity_loss(self, weights):
        entropy = -K.sum(weights * K.log(weights + 1e-10), axis=1)
        self.diversity_loss = -K.mean(entropy)

    def _importance_loss(self, gating_weights):
        batch_importance_sum = K.sum(gating_weights, axis=0)
        mean_importance = K.mean(batch_importance_sum)
        self.importance_loss = K.mean(
            K.square(
                batch_importance_sum
                - mean_importance * tf.ones_like(batch_importance_sum)
            )
        )

    def call(self, x):
        # Get gating weights and normalize
        gating_weights = self.gating_network(x)
        gating_weights = K.softmax(gating_weights)  # Ensure weights are probabilities
        self._diversity_loss(gating_weights)
        self._importance_loss(gating_weights)

        # Get the top k experts based on the gating weights
        top_k_weights, top_k_indices = tf.math.top_k(gating_weights, k=self.top_k)

        # Count usage of each expert symbolically
        updates = tf.ones_like(tf.reshape(top_k_indices, [-1]), dtype=tf.float32)
        # Use tf.tensor_scatter_nd_add to increment the usage count
        self.expert_usage_count.assign(
            tf.tensor_scatter_nd_add(
                self.expert_usage_count, tf.reshape(top_k_indices, [-1, 1]), updates
            )
        )

        # Calculate overflow using updated usage count
        self.batch_overflow_sum = K.sum(
            K.relu(tf.convert_to_tensor(self.expert_usage_count) - self.batch_capacity)
        )

        # Compute all expert outputs
        expert_outputs = tf.stack(
            [expert(x) for expert in self.experts], axis=1
        )  # Shape: (batch_size, num_experts, hidden_size)

        # Gather the top-k expert outputs using top_k_indices
        batch_size = tf.shape(x)[0]
        batch_indices = tf.expand_dims(
            tf.range(batch_size), 1
        )  # Shape: (batch_size, 1)
        batch_indices = tf.tile(
            batch_indices, [1, self.top_k]
        )  # Shape: (batch_size, top_k)

        # Create indices for gathering
        indices = tf.stack(
            [batch_indices, top_k_indices], axis=2
        )  # Shape: (batch_size, top_k, 2)
        top_k_expert_outputs = tf.gather_nd(
            expert_outputs, indices
        )  # Shape: (batch_size, top_k, hidden_size)

        # Combine outputs using top-k weights
        combined_output = tf.reduce_sum(
            top_k_expert_outputs * tf.expand_dims(top_k_weights, axis=-1), axis=1
        )

        return combined_output

    def compute_total_loss(self, load_balance_coef=0.01):
        return load_balance_coef * (
            self.diversity_loss + self.batch_overflow_sum + self.importance_loss
        )


"""
## MNIST classification with MoE
"""


class MoEModel(keras.Model):
    def __init__(self, input_shape, num_classes, num_experts=NUM_EXPERTS, top_k=TOP_K):
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

    def call(self, inputs, training=False):
        conv_flatten = self.conv_block(inputs)
        moe_output = self.moe_classifier(conv_flatten)
        outputs = self.softmax(moe_output)
        return outputs

    def train_step(self, data):
        x, y = data  # Unpack input data and labels

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            classification_loss = self.compute_loss(x, y, y_pred)
            moe_loss = self.moe_classifier.compute_total_loss(load_balance_coef=0.01)
            total_loss = classification_loss + moe_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )  # Update metrics (e.g., accuracy)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict of metrics for monitoring
        return {
            "loss": total_loss,
            "moe_loss": moe_loss,
            **{m.name: m.result() for m in self.metrics},
        }

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        classification_loss = self.compute_loss(x, y, y_pred)
        moe_loss = self.moe_classifier.compute_total_loss(load_balance_coef=0.01)
        total_loss = classification_loss + moe_loss

        self.compiled_metrics.update_state(y, y_pred)
        return {
            "loss": total_loss,
            "moe_loss": moe_loss,
            **{m.name: m.result() for m in self.metrics},
        }


# Instantiate and compile the model
inputs = keras.Input(shape=input_shape)
model = MoEModel(
    input_shape=input_shape, num_classes=num_classes, num_experts=6, top_k=4
)

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
)

"""
### Evaluation
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

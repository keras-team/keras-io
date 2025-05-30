# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
[RLHF with KerasNLP](https://keras.io/guides/keras_nlp/rlhf_with_kerasnlp/)

## Introduction

Reinforcement Learning from Human Feedback (RLHF) is a technique used to align
large language models (LLMs) with human preferences. It involves training a policy
model (the LLM) using feedback from a reward model, which itself is trained to
predict human preferences.

This guide provides a simplified implementation of the RLHF loop using Keras
for a toy problem: generating sequences of numbers that sum to a target value.

## Setup

First, let's import the necessary libraries. We'll need Keras for building our
models, NumPy for numerical operations, and TensorFlow for backend operations.
"""

import keras
from keras import layers
import numpy as np
import tensorflow as tf

"""
## Core RLHF Concepts

1.  **Policy Model**: This is the model we want to train. In the context of LLMs,
    it's the language model itself. In our toy example, it will be a simple
    neural network that generates sequences of numbers.

2.  **Reward Model**: This model is trained to predict human preferences. It takes
    a generated output (e.g., a sequence of text or, in our case, numbers) and
    assigns a scalar reward score indicating how good that output is according
    to human judgment. In a real RLHF setup, this model is trained on a dataset
    of human-ranked responses. For simplicity, we'll simulate this with a
    hardcoded reward function.

3.  **Proximal Policy Optimization (PPO)**: PPO is a reinforcement learning
    algorithm commonly used in RLHF. It's designed to make stable updates to the
    policy model by penalizing large deviations from the previous policy. This
    helps in preventing the policy from changing too drastically and destabilizing
    training. We will implement a simplified version of the PPO update.

## Implementing the Policy Model

Let's define a simple Keras model that will act as our policy model. This model
will take a dummy input (representing a prompt or context) and output a sequence
of numbers.
"""

# Define the policy model
# For this toy example, our "action space" is generating numbers.
# Let's say we want to generate sequences of 3 numbers.
sequence_length = 3
vocab_size = 10  # Numbers from 0 to 9

policy_model = keras.Sequential(
    [
        keras.Input(shape=(1,)), # Dummy input
        layers.Dense(64, activation="relu"),
        layers.Dense(sequence_length * vocab_size),
        layers.Reshape((sequence_length, vocab_size)),
        layers.Softmax(axis=-1),  # Output probabilities for each number in the sequence
    ],
    name="policy_model",
)

policy_model.summary()

"""
## Simulating Human Feedback (Reward Function)

In a real RLHF pipeline, you'd have a separate reward model trained on human
preference data. Here, we'll create a simple Python function that simulates this.
Let's say we want our model to generate sequences of numbers that sum up to a
target value (e.g., 15).
"""

TARGET_SUM = 15

def get_reward_for_actions(action_sequences):
    """
    Calculates the reward for a batch of action sequences (integers).
    The reward is higher if the sum of numbers in a sequence is closer to TARGET_SUM.
    """
    rewards = []
    for seq in action_sequences: # seq is now a list/array of integers
        current_sum = np.sum(seq)
        # Reward is inversely proportional to the absolute difference from the target sum
        reward = 1.0 / (1.0 + abs(current_sum - TARGET_SUM))
        rewards.append(reward)
    return np.array(rewards)

"""
## Implementing the RLHF Training Loop with PPO

The RLHF training loop generally involves the following steps:

1.  **Generate Data**: Use the current policy model to generate a batch of outputs
    (sequences of numbers in our case).
2.  **Collect Feedback**: Use the reward model (or our simulated reward function)
    to get rewards for the generated outputs.
3.  **Calculate Advantages**: Advantages represent how much better an action is
    compared to the average action at a given state. This is often calculated
    using techniques like Generalized Advantage Estimation (GAE), but we'll use
    a simplified version: `Advantage = Reward - Value`.
    For this, we need a value function, often part of a critic model.
    For simplicity, we'll use the rewards directly as a proxy for advantages,
    or a very simple baseline.
4.  **Update Policy**: Update the policy model's weights using the PPO algorithm.
    The PPO objective function tries to maximize the expected advantage while
    penalizing large changes to the policy.

### Simplified PPO Update

The core idea of PPO is to clip the objective function:

`L_CLIP(θ) = E_t [ min( r_t(θ) * A_t, clip(r_t(θ), 1 - ε, 1 + ε) * A_t ) ]`

Where:
- `r_t(θ)` is the probability ratio: `π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`
- `A_t` is the advantage
- `ε` is a small hyperparameter (e.g., 0.2)

We'll implement a simplified version of this. We also need an "old" policy
to calculate the probability ratio.

### Value Model (Critic)

To calculate advantages properly, PPO typically uses a critic model that estimates
the value function `V(s)`, which predicts the expected future reward from a given
state. Let's define a simple value model.
"""

value_model = keras.Sequential(
    [
        keras.Input(shape=(1,)), # Dummy input, same as policy
        layers.Dense(64, activation="relu"),
        layers.Dense(1),  # Outputs a single value
    ],
    name="value_model",
)
value_model.summary()

# Optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=1e-3)
value_optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# PPO Hyperparameters
epsilon = 0.2  # Clipping parameter
epochs_ppo = 10  # Number of PPO updates per RLHF iteration
batch_size_rlhf = 32 # Number of sequences generated per RLHF iteration

"""
Now, let's put it all together in a training loop.
"""

def rlhf_training_step(policy_model, value_model, dummy_input_data):
    """Performs one step of RLHF training."""

    # Store the "old" policy state (probabilities) for PPO
    # Detach from graph to ensure they are treated as constants in PPO loss
    old_policy_probs_tensor = policy_model(dummy_input_data)
    old_policy_probs_numpy = tf.stop_gradient(old_policy_probs_tensor).numpy() # (batch_size, seq_len, vocab_size)


    # 1. Generate Data with the current policy (by sampling or argmax)
    # For this example, we'll use argmax to define "chosen actions"
    # In a more robust PPO, you would sample from the policy's distribution.
    # chosen_actions = np.argmax(old_policy_probs_numpy, axis=-1) # (batch_size, seq_len)

    # Sample actions from the policy distribution to encourage exploration
    chosen_actions_list = []
    for t_step in range(sequence_length):
        timestep_probs = old_policy_probs_numpy[:, t_step, :]
        # tf.random.categorical expects log-probabilities. Add small epsilon for numerical stability.
        sampled_actions_at_t = tf.random.categorical(tf.math.log(timestep_probs + 1e-9), 1)
        chosen_actions_list.append(sampled_actions_at_t.numpy())
    chosen_actions = np.concatenate(chosen_actions_list, axis=-1) # Shape: (batch_size, sequence_length)


    # 2. Collect Simulated Human Feedback (Rewards)
    # The reward function needs to be aware of chosen_actions if its logic depends on them directly,
    # or it can re-derive them from old_policy_probs_numpy if it uses argmax internally.
    # Our current get_reward uses argmax on the passed probs, so we should pass chosen_actions to it
    # OR modify get_reward to accept chosen numbers directly.
    # For now, let's modify get_reward slightly or ensure it's consistent.
    # The current get_reward uses argmax(seq_probs), so it will ignore our sampling if we pass old_policy_probs_numpy.
    # Let's make get_reward use the actual chosen_actions.
    # For that, we need to pass chosen_actions to get_reward.
    # Let's adjust get_reward function signature and logic.
    #
    # Quick fix for now: The reward function will re-calculate argmax if given probs.
    # To make the reward based on SAMPLED actions, the reward function needs the sampled actions.
    # For now, the `get_reward` function is defined as:
    # chosen_numbers = np.argmax(seq_probs, axis=-1)
    # This means our sampling above is NOT YET USED by the reward function.
    # Let's adjust the call to get_reward or the get_reward function itself.
    # Easiest: modify get_reward to take chosen_numbers.

    # Pass the probabilities, but be aware reward is based on argmax unless get_reward is changed.
    # To make training effective, reward MUST be calculated based on the actions that were actually taken (sampled_actions).
    # So, we need to adjust `get_reward` or how we call it.
    # We will modify `get_reward` later. For now, the impact of sampling will be on the `chosen_actions` used in PPO loss,
    # but not directly on reward calculation. This is a discrepancy.

    # Let's assume for a moment the reward is calculated based on the sampled `chosen_actions`.
    # This means `get_reward` should be: `get_reward(chosen_actions_as_sequences)`
    # The current `get_reward` expects probabilities.
    # For a quick test, let's make a temporary version of get_reward or pass what it expects.
    # The current `get_reward` will compute its own argmax.
    # This means `rewards` will be based on deterministic actions from old_policy,
    # while PPO loss will be based on sampled actions. This is inconsistent.

    # Correct approach: The reward must be for the actions taken.
    # So, `get_reward` must operate on `chosen_actions`.
    # We will modify `get_reward` to accept integer sequences.
    rewards = get_reward_for_actions(chosen_actions) # (batch_size,)


    # 3. Calculate Advantages
    values_tensor = value_model(dummy_input_data) # (batch_size, 1)
    values_numpy = tf.stop_gradient(tf.squeeze(values_tensor, axis=-1)).numpy() # (batch_size,)

    # Advantages A_t = R_t - V(s_t)
    advantages = rewards - values_numpy
    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # Convert to TensorFlow tensors
    rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
    advantages_tf = tf.convert_to_tensor(advantages, dtype=tf.float32)
    chosen_actions_tf = tf.convert_to_tensor(chosen_actions, dtype=tf.int32)

    # 4. Update Policy and Value Model using PPO-like updates
    for _ in range(epochs_ppo):
        with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
            # Current policy probabilities and value estimates
            current_policy_probs_tensor = policy_model(dummy_input_data) # (batch_size, seq_len, vocab_size)
            current_values_tensor = value_model(dummy_input_data) # (batch_size, 1)
            current_values_squeezed = tf.squeeze(current_values_tensor, axis=-1) # (batch_size,)

            # Calculate log probabilities of chosen actions under current and old policies
            # This is crucial for PPO.
            # tf.gather_nd is used to pick the probabilities of the specific actions that were chosen.
            # Create indices for gather_nd
            batch_indices = tf.range(tf.shape(chosen_actions_tf)[0])[:, tf.newaxis, tf.newaxis]
            batch_indices = tf.tile(batch_indices, [1, sequence_length, 1])

            sequence_indices = tf.range(sequence_length)[tf.newaxis, :, tf.newaxis]
            sequence_indices = tf.tile(sequence_indices, [tf.shape(chosen_actions_tf)[0], 1, 1])

            action_indices = chosen_actions_tf[:, :, tf.newaxis]

            full_indices = tf.concat([batch_indices, sequence_indices, action_indices], axis=-1)

            # Log probs for current policy
            current_log_probs_of_actions = tf.math.log(
                tf.gather_nd(current_policy_probs_tensor, full_indices) + 1e-10
            ) # (batch_size, seq_len)
            # Sum log probs across the sequence: log(P(sequence)) = sum(log(P(action_t)))
            current_sum_log_probs = tf.reduce_sum(current_log_probs_of_actions, axis=1) # (batch_size,)

            # Log probs for old policy
            old_log_probs_of_actions = tf.math.log(
                tf.gather_nd(old_policy_probs_tensor, full_indices) + 1e-10
            ) # (batch_size, seq_len)
            old_sum_log_probs = tf.reduce_sum(old_log_probs_of_actions, axis=1) # (batch_size,)
            old_sum_log_probs = tf.stop_gradient(old_sum_log_probs) # Treat old policy log_probs as constants

            # Probability ratio: r_t(θ) = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))
            ratios = tf.exp(current_sum_log_probs - old_sum_log_probs) # (batch_size,)

            # Diagnostic prints (commented out as training is now working)
            # if _ == 0 and current_rlhf_iteration < 2: # Print only for first PPO epoch and first few RLHF iterations
            #     tf.print("PPO Epoch:", _, "RLHF Iter:", current_rlhf_iteration)
            #     tf.print("chosen_actions_tf[0]:", chosen_actions_tf[0])
            #     tf.print("current_sum_log_probs[0]:", current_sum_log_probs[0])
            #     tf.print("old_sum_log_probs[0]:", old_sum_log_probs[0])
            #     tf.print("ratios[0]:", ratios[0])
            #     tf.print("advantages_tf[0]:", advantages_tf[0])


            # PPO Clipped Surrogate Objective for Policy
            surr1 = ratios * advantages_tf
            surr2 = tf.clip_by_value(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages_tf
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Value Function Loss (Mean Squared Error)
            # Target for value function is the observed rewards
            value_loss = keras.losses.mean_squared_error(rewards_tf, current_values_squeezed)
            value_loss = tf.reduce_mean(value_loss)

        # Calculate gradients and update policy model
        policy_grads = policy_tape.gradient(policy_loss, policy_model.trainable_variables)
        policy_optimizer.apply_gradients(zip(policy_grads, policy_model.trainable_variables))

        # Calculate gradients and update value model
        value_grads = value_tape.gradient(value_loss, value_model.trainable_variables)
        value_optimizer.apply_gradients(zip(value_grads, value_model.trainable_variables))

    return policy_loss, value_loss, rewards

"""
## Running a Demo

Let's run a small demo of the RLHF training loop.
"""

if __name__ == "__main__":
    num_rlhf_iterations = 100
    print_every_n = 10

    # Dummy input data (e.g., representing prompts)
    # For this simple example, it's just a single feature.
    # Add global variable i to be used in rlhf_training_step for printing
    global current_rlhf_iteration
    current_rlhf_iteration = 0

    dummy_prompts = np.ones((batch_size_rlhf, 1))

    print(f"Starting RLHF training for {num_rlhf_iterations} iterations...")
    print(f"Target sum for sequences: {TARGET_SUM}")
    print(f"Policy Model: {policy_model.name}, Value Model: {value_model.name}")
    print(f"PPO Epochs per iteration: {epochs_ppo}, RLHF Batch Size: {batch_size_rlhf}\n")

    for i_iter in range(num_rlhf_iterations):
        current_rlhf_iteration = i_iter
        policy_loss, value_loss, rewards = rlhf_training_step(
            policy_model,
            value_model,
            dummy_prompts
        )

        if (i_iter + 1) % print_every_n == 0:
            print(f"Iteration {i_iter+1}/{num_rlhf_iterations}:")
            print(f"  Policy Loss: {policy_loss.numpy():.4f}")
            print(f"  Value Loss: {value_loss.numpy():.4f}")
            print(f"  Mean Reward: {np.mean(rewards):.4f}")
            print(f"  Std Reward: {np.std(rewards):.4f}")

            # Print some diagnostic values from the last PPO step of this iteration
            # This requires passing 'i' to rlhf_training_step or accessing it globally
            # For now, let's just show generated sequences
            sample_generated_probs = policy_model(dummy_prompts[:5]) # Take 5 samples
            sample_generated_numbers = np.argmax(sample_generated_probs.numpy(), axis=-1)
            sample_sums = [np.sum(s) for s in sample_generated_numbers]
            print(f"  Sample generated sequences (first 5): {sample_generated_numbers.tolist()}")
            print(f"  Sums of these sequences: {sample_sums}")
            print("-" * 30)

    print("\nRLHF training finished.")

    # Example of generating a sequence after training
    print("\nExample generation after training:")
    test_prompt = np.ones((1, 1)) # Single prompt
    final_generated_probs = policy_model(test_prompt)
    final_generated_sequence = np.argmax(final_generated_probs.numpy(), axis=-1)
    final_sum = np.sum(final_generated_sequence)
    print(f"Generated sequence: {final_generated_sequence.tolist()}")
    print(f"Sum of sequence: {final_sum} (Target: {TARGET_SUM})")

"""
## Conclusion

This guide provided a very simplified walkthrough of RLHF using Keras.
Key takeaways:
- RLHF involves a policy model, a reward model (simulated here), and an RL
  algorithm like PPO.
- The policy generates outputs, the reward model scores them, and PPO updates
  the policy based on these scores (advantages).
- This example uses a toy problem. Real-world RLHF for LLMs is significantly
  more complex, involving large-scale models, sophisticated reward modeling,
  and distributed training.

Further improvements and complexities in real RLHF systems include:
- **More sophisticated reward models**: Trained on large datasets of human
  comparisons.
- **Full PPO implementation**: Including Generalized Advantage Estimation (GAE),
  entropy bonuses for exploration, and more careful handling of probability ratios.
- **Exploration strategies**: To ensure the policy model explores a diverse range
  of outputs.
- **KL divergence penalty**: An additional term in the PPO objective to limit
  how much the policy changes from a reference (often the initial SFT) model.
- **Handling large action spaces**: For text generation, the action space (vocab
  size) is very large.
"""

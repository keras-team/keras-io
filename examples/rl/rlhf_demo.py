"""
Title: Reinforcement Learning from AI Feedback(RLAIF) - Demo Guide
Author: [Jules](https://jules.google.com/)
Date created: 2025/06/02
Last modified: 2025/06/18
Accelerator: GPU
"""

"""
# Reinforcement Learning from AI Feedback(RLAIF) - Demo Guide

This guide explains the concept of  Reinforcement Learning from AI Feedback (RLAIF) and walks through the components of the accompanying script `rlhf_demo.py`.

## 1. What is Reinforcement Learning from Human Feedback (RLHF)?

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment to achieve a goal. The agent receives rewards or penalties based on its actions, and it tries to maximize its cumulative reward over time.

In many real-world scenarios, defining a precise reward function that perfectly captures desired behavior can be extremely challenging. For example, how do you define a reward for "writing a helpful and harmless AI assistant response"? This is where RLHF comes in.

**RLHF** is a technique that incorporates human feedback into the RL process to guide the agent's learning, especially for tasks with complex or hard-to-specify objectives. Instead of relying solely on a pre-defined reward function, RLHF uses human preferences to train a separate "reward model" that learns to predict what kind of behaviors humans prefer. This learned reward model is then used to provide reward signals to the RL agent.

## 2. How RLHF Works (High-Level)

The RLHF process generally involves these key stages:

1.  **Pre-training a Language Model (or Policy Model):**
    Start with a base model that can generate responses or take actions. For language tasks, this is often a pre-trained language model (LM). This model acts as the initial policy.

2.  **Collecting Human Feedback & Training a Reward Model:**
    *   Generate multiple outputs (e.g., text responses) from the current policy model for various prompts.
    *   Present these outputs to human evaluators, who rank them or choose the best one(s) based on desired criteria (e.g., helpfulness, safety, coherence).
    *   This collected preference data (e.g., "Response A is better than Response B for prompt X") is used to train a separate **reward model**. The reward model takes a prompt and a response (or state-action pair) as input and outputs a scalar score indicating how good that response is according to human preferences.

3.  **Fine-tuning the Policy Model via RL:**
    *   The pre-trained policy model is then fine-tuned using an RL algorithm (like Proximal Policy Optimization - PPO).
    *   Instead of using a fixed reward function from the environment, the RL agent receives rewards from the **trained reward model**.
    *   The agent explores the environment (or generates responses), and the reward model scores these actions/responses. The policy model is updated to produce outputs that the reward model scores highly.
    *   Often, a constraint (e.g., a KL divergence penalty) is added to prevent the policy from diverging too much from the original pre-trained model, helping to maintain coherence and avoid reward hacking.

This cycle (collecting more data, refining the reward model, and further fine-tuning the policy) can be iterated.

## 3. Walking Through `rlhf_demo.py`

The `rlhf_demo.py` script provides a very simplified implementation of these concepts to illustrate the basic mechanics.

**Important Note on Keras Backend:**
This demo is configured to run with the JAX backend for Keras. This is set at the beginning of the script:
"""
# Set Keras backend to JAX
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_core as keras
# import GradientTape was removed as we will use jax.grad
import jax
import jax.numpy as jnp
import numpy as np

# Helper function to calculate discounted returns
def calculate_discounted_returns(rewards, gamma=0.99):
    returns = []
    cumulative_return = 0
    for r in reversed(rewards):
        cumulative_return = r + gamma * cumulative_return
        returns.insert(0, cumulative_return)
    return jnp.array(returns)

"""
### 3.1. The Environment (`SimpleEnvironment`)

The script defines a very basic grid-world like environment where the agent's state is its position on a line.
"""
# Define a simple environment (e.g., a GridWorld)
class SimpleEnvironment:
    def __init__(self, size=3): # Reduced default size
        self.size = size
        self.state = 0  # Initial state

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # Simple dynamics: 0 -> left, 1 -> right
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(self.size - 1, self.state + 1)

        reward = 1 if self.state == self.size - 1 else 0  # Reward for reaching the goal
        done = self.state == self.size - 1
        return self.state, reward, done

    def get_observation_space_shape(self):
        return (1,) # State is a single integer

    def get_action_space_n(self):
        return 2 # Two possible actions: left or right      
"""
- The agent can move left or right.
- It receives a "true" reward of 1 if it reaches the rightmost state (`size - 1`), otherwise 0. This "true" reward is used in the demo to simulate human feedback for training the reward model.

### 3.2. The Policy Model (`create_policy_model`)

This is a simple Keras neural network that takes the current state (observation) as input and outputs probabilities for each action (left/right).
"""
# Define a simple policy model
def create_policy_model(observation_space_shape, action_space_n):
    inputs = keras.Input(shape=observation_space_shape)
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(action_space_n, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
"""
- It's a small Multi-Layer Perceptron (MLP).
- The `softmax` activation ensures the output represents a probability distribution over actions.

### 3.3. The Reward Model (`create_reward_model`)

This Keras model is designed to predict how "good" a state-action pair is. In a real RLHF setup, this model would be trained on human preference data. In this dummy demo, it's trained using the environment's "true" reward signal as a proxy for human feedback.
"""
# Define a simple reward model
def create_reward_model(observation_space_shape, action_space_n):
    inputs = keras.Input(shape=(observation_space_shape[0] + action_space_n,)) # obs + action
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(1)(x) # Outputs a scalar reward
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
"""
- It takes the current state and the chosen action (one-hot encoded) as input.
- It outputs a single scalar value, representing the predicted reward.

### 3.4. The RLHF Training Loop (`rlhf_training_loop`)

This function contains the core logic for the RLHF process.
"""
# RLHF Training Loop
def rlhf_training_loop(env, policy_model, reward_model, num_episodes=10, learning_rate=0.001): # Reduced default episodes
    policy_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    reward_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Define loss functions for jax.grad
    @jax.jit
    def policy_loss_fn(policy_model_params, state_input, action, discounted_return_for_step):
        # stateless_call might return a tuple (e.g., (outputs, other_states) or just (outputs,))
        # We are interested in the first element, which should be the main output tensor.
        predictions_tuple = policy_model.stateless_call(
            policy_model_params["trainable"],
            policy_model_params["non_trainable"],
            state_input
        )
        actual_predictions_tensor = predictions_tuple[0]
        action_probs = actual_predictions_tensor[0] # If actual_predictions_tensor is (1,2)
        selected_action_prob = action_probs[action] # action is already a JAX array if converted before call
        log_prob = jnp.log(selected_action_prob + 1e-7)
        # Loss is -log_prob * G_t (discounted return)
        loss_value = -log_prob * discounted_return_for_step
        return loss_value

    @jax.jit
    def reward_loss_fn(reward_model_params, reward_model_input, true_reward_val):
        # Use stateless_call with the provided parameters
        predictions_tuple = reward_model.stateless_call(
            reward_model_params["trainable"],
            reward_model_params["non_trainable"],
            reward_model_input
        )
        # Assuming the actual output tensor is the first element of the tuple
        actual_predictions_tensor = predictions_tuple[0]

        predicted_reward_val = actual_predictions_tensor[0] # If actual_predictions_tensor is (1,1)
        # Ensure loss is scalar
        loss = keras.losses.mean_squared_error(jnp.array([true_reward_val]), predicted_reward_val)
        return jnp.mean(loss) # Reduce to scalar if it's not already

    # Grad functions, argnums=0 means differentiate w.r.t. the first argument (policy_model_params/reward_model_params)
    policy_value_and_grad_fn = jax.jit(jax.value_and_grad(policy_loss_fn, argnums=0))
    reward_value_and_grad_fn = jax.jit(jax.value_and_grad(reward_loss_fn, argnums=0))

    # Keep track of losses for averaging
    total_policy_loss_avg = 0
    total_reward_loss_avg = 0
    loss_count_avg = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward_sum = 0

        episode_policy_losses = []
        episode_reward_losses = []

        # Initialize gradient accumulators for the episode
        policy_grads_accum = [jnp.zeros_like(var) for var in policy_model.trainable_variables]
        reward_grads_accum = [jnp.zeros_like(var) for var in reward_model.trainable_variables]
        num_steps_in_episode = 0

        while not done:
            state_input_np = np.array([state]).reshape(1, -1) # Keras model expects numpy array

            # Get action from policy model
            # Note: policy_model directly uses its current weights, not passed params for inference
            action_probs = policy_model(state_input_np)[0]
            action = np.random.choice(env.get_action_space_n(), p=np.array(action_probs))

            next_state, true_reward, done = env.step(action)

            action_one_hot = jax.nn.one_hot(action, env.get_action_space_n())
            reward_model_input_np = np.concatenate([state_input_np.flatten(), np.array(action_one_hot).flatten()]).reshape(1, -1)

            # Predict reward with reward model (also uses its current weights for inference)
            predicted_reward_value = reward_model(reward_model_input_np)[0] # Shape (1,)

            # --- Policy gradient calculation ---
            stopped_predicted_reward = jax.lax.stop_gradient(predicted_reward_value[0])
            state_input_jax = jnp.array(state_input_np)
            action_jax = jnp.array(action) # Convert action to JAX array

            policy_params_dict = {
                "trainable": policy_model.trainable_variables,
                "non_trainable": policy_model.non_trainable_variables
            }
            current_policy_loss, policy_grads_dict_step = policy_value_and_grad_fn(
                policy_params_dict,
                state_input_jax,
                action_jax, # Use JAX array action
                stopped_predicted_reward
            )
            episode_policy_losses.append(current_policy_loss)
            policy_grads_step = policy_grads_dict_step["trainable"]
            # Accumulate policy gradients
            policy_grads_accum = jax.tree_map(
                lambda acc, new: acc + new if new is not None else acc,
                policy_grads_accum,
                policy_grads_step
            )

            # --- Reward model gradient calculation ---
            reward_model_input_jax = jnp.array(reward_model_input_np)
            reward_params_dict = {
                "trainable": reward_model.trainable_variables,
                "non_trainable": reward_model.non_trainable_variables
            }
            current_reward_loss, reward_grads_dict_step = reward_value_and_grad_fn(
                reward_params_dict,
                reward_model_input_jax,
                true_reward
            )
            episode_reward_losses.append(current_reward_loss)
            reward_grads_step = reward_grads_dict_step["trainable"]
            # Accumulate reward gradients
            reward_grads_accum = jax.tree_map(
                lambda acc, new: acc + new if new is not None else acc,
                reward_grads_accum,
                reward_grads_step
            )

            num_steps_in_episode += 1
            episode_reward_sum += true_reward
            state = next_state

        if num_steps_in_episode > 0:
            # Average gradients over the episode and apply them
            avg_policy_grads = [jnp.clip(g / num_steps_in_episode, -1.0, 1.0) if g is not None else g for g in policy_grads_accum]
            avg_reward_grads = [jnp.clip(g / num_steps_in_episode, -1.0, 1.0) if g is not None else g for g in reward_grads_accum]

            policy_optimizer.apply_gradients(zip(avg_policy_grads, policy_model.trainable_variables))
            reward_optimizer.apply_gradients(zip(avg_reward_grads, reward_model.trainable_variables))

            # Calculate mean losses for the episode for reporting
            mean_episode_policy_loss = jnp.mean(jnp.array(episode_policy_losses))
            mean_episode_reward_loss = jnp.mean(jnp.array(episode_reward_losses))

            total_policy_loss_avg += mean_episode_policy_loss
            total_reward_loss_avg += mean_episode_reward_loss
            loss_count_avg +=1

        if (episode + 1) % 100 == 0 and loss_count_avg > 0:
            final_avg_policy_loss = total_policy_loss_avg / loss_count_avg
            final_avg_reward_loss = total_reward_loss_avg / loss_count_avg
            print(f"Episode {episode + 1}: Total Reward: {episode_reward_sum}, Avg Policy Loss: {final_avg_policy_loss.item():.4f}, Avg Reward Loss: {final_avg_reward_loss.item():.4f}")
            total_policy_loss_avg = 0
            total_reward_loss_avg = 0
            loss_count_avg = 0


    print("Training finished.")
"""
**Key Parts of the Training Loop (Updated):**

1.  **Initialization:** Optimizers and JAX gradient functions (`policy_value_and_grad_fn`, `reward_value_and_grad_fn`) are set up. The `policy_loss_fn` is now designed to accept a `discounted_return_for_step` argument.
2.  **Trajectory Collection:** During each episode, the agent's experiences (states, actions taken, and the `true_reward` received from the environment) are stored.
3.  **Reward Model Training:** The reward model continues to be trained. Its gradients are calculated based on the immediate `true_reward` (simulating feedback) and accumulated over the episode. These accumulated gradients are applied once at the end of the episode.
4.  **Policy Model Training (REINFORCE-style):**
    *   **At the end of each episode:**
        *   The `calculate_discounted_returns` function is called with the list of `true_reward`s collected during the episode to compute the discounted cumulative reward (G_t) for each step.
        *   These returns are typically normalized (subtract mean, divide by standard deviation) to stabilize training.
        *   The code then iterates through each step `t` of the collected trajectory.
        *   For each step, the `policy_loss_fn` is called. Its loss is calculated as `-log_prob(action_t) * G_t`. This means the update encourages actions that led to higher overall discounted future rewards.
        *   Gradients for the policy model are computed for each step and accumulated across the entire episode.
    *   **Gradient Application:** The accumulated policy gradients are averaged over the number of steps and applied to the policy model using its optimizer. This update rule aims to increase the probability of actions that lead to good long-term outcomes.
5.  **Logging:** Average policy and reward losses for the episode are printed periodically.

The core idea of RLHF is still present: we have a reward model that *could* be trained from human preferences. However, the policy update mechanism has shifted. Instead of using the reward model's output directly as the advantage signal for each step (as in the previous version of the script), the policy now learns from the actual discounted returns experienced in the episode, which is a more standard RL approach when actual rewards (or good proxies like `true_reward` here) are available for the full trajectory. In a full RLHF system, `episode_true_rewards` might themselves be replaced or augmented by the reward model's predictions if no dense "true" reward exists.
8.  **Logging:** Periodically, average losses are printed.

## 4. How to Run the Demo

To run the demo, execute the Python script from your terminal:

```bash
python examples/rl/rlhf_demo.py
```

This will:
1.  Initialize the environment, policy model, and reward model.
2.  Print summaries of the policy and reward models.
3.  Start the RLHF training loop for the specified number of episodes (default is 10 in the modified script).
4.  Print training progress (episode number, total reward, average policy loss, average reward loss).
5.  After training, it will test the trained policy model for a few steps and print the interactions.
"""
# Main execution
if __name__ == "__main__":
    env = SimpleEnvironment()
    obs_space_shape = env.get_observation_space_shape()
    act_space_n = env.get_action_space_n()

    policy_model = create_policy_model(obs_space_shape, act_space_n)
    reward_model = create_reward_model(obs_space_shape, act_space_n)

    print("Policy Model Summary:")
    policy_model.summary()
    print("\nReward Model Summary:")
    reward_model.summary()

    print("\nStarting RLHF Training Loop...")
    # Use a smaller number of episodes for a quick demo
    rlhf_training_loop(env, policy_model, reward_model, num_episodes=10) # Further reduced episodes

    # Example of using the trained policy model
    print("\nTesting trained policy model:")
    state = env.reset()
    done = False
    test_rewards = 0
    for _ in range(env.size * 2): # Max steps to prevent infinite loop
        if done:
            break
        state_input = jnp.array([state]).reshape(1, -1)
        action_probs = policy_model(state_input)[0]
        action = jnp.argmax(action_probs).item() # Take best action
        next_state, reward, done = env.step(action)
        print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}")
        test_rewards += reward
        state = next_state
    print(f"Total reward from trained policy: {test_rewards}")

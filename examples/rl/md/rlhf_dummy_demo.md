# Reinforcement Learning from Human Feedback (RLHF) - Dummy Demo Guide

This guide explains the concept of Reinforcement Learning from Human Feedback (RLHF) and walks through the components of the accompanying dummy demo script `rlhf_dummy_demo.py`.

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

## 3. Walking Through `rlhf_dummy_demo.py`

The `rlhf_dummy_demo.py` script provides a very simplified, "dummy" implementation of these concepts to illustrate the basic mechanics.

**Important Note on Keras Backend:**
This demo is configured to run with the JAX backend for Keras. This is set at the beginning of the script:
```python
import os
os.environ["KERAS_BACKEND"] = "jax"
```

### 3.1. The Environment (`SimpleEnvironment`)

The script defines a very basic grid-world like environment where the agent's state is its position on a line.
```python
class SimpleEnvironment:
    def __init__(self, size=3): # Default size is small
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
        
        # Reward for reaching the goal (rightmost state)
        reward = 1 if self.state == self.size - 1 else 0 
        done = self.state == self.size - 1
        return self.state, reward, done

    def get_observation_space_shape(self):
        return (1,) 

    def get_action_space_n(self):
        return 2 # Two possible actions: left or right
```
- The agent can move left or right.
- It receives a "true" reward of 1 if it reaches the rightmost state (`size - 1`), otherwise 0. This "true" reward is used in the demo to simulate human feedback for training the reward model.

### 3.2. The Policy Model (`create_policy_model`)

This is a simple Keras neural network that takes the current state (observation) as input and outputs probabilities for each action (left/right).
```python
import keras_core as keras
import jax.numpy as jnp

def create_policy_model(observation_space_shape, action_space_n):
    inputs = keras.Input(shape=observation_space_shape)
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(action_space_n, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
```
- It's a small Multi-Layer Perceptron (MLP).
- The `softmax` activation ensures the output represents a probability distribution over actions.

### 3.3. The Reward Model (`create_reward_model`)

This Keras model is designed to predict how "good" a state-action pair is. In a real RLHF setup, this model would be trained on human preference data. In this dummy demo, it's trained using the environment's "true" reward signal as a proxy for human feedback.
```python
def create_reward_model(observation_space_shape, action_space_n):
    # Input is observation + one-hot encoded action
    inputs = keras.Input(shape=(observation_space_shape[0] + action_space_n,)) 
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(1)(x) # Outputs a scalar reward prediction
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
```
- It takes the current state and the chosen action (one-hot encoded) as input.
- It outputs a single scalar value, representing the predicted reward.

### 3.4. The RLHF Training Loop (`rlhf_training_loop`)

This function contains the core logic for the RLHF process.

```python
def rlhf_training_loop(env, policy_model, reward_model, num_episodes=10, learning_rate=0.001):
    policy_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    reward_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # JAX gradient functions using model.stateless_call
    @jax.jit
    def policy_loss_fn(policy_model_params, state_input, action, predicted_reward_value_stopped):
        # ... (calculates policy loss based on predicted_reward_value_stopped)
        predictions_tuple = policy_model.stateless_call(...)
        actual_predictions_tensor = predictions_tuple[0]
        action_probs = actual_predictions_tensor[0]
        log_prob = jnp.log(action_probs[action] + 1e-7)
        return -log_prob * predicted_reward_value_stopped

    @jax.jit
    def reward_loss_fn(reward_model_params, reward_model_input, true_reward_val):
        # ... (calculates MSE loss between predicted reward and true_reward_val)
        predictions_tuple = reward_model.stateless_call(...)
        actual_predictions_tensor = predictions_tuple[0]
        predicted_reward_val = actual_predictions_tensor[0]
        loss = keras.losses.mean_squared_error(jnp.array([true_reward_val]), predicted_reward_val)
        return jnp.mean(loss)

    policy_value_and_grad_fn = jax.jit(jax.value_and_grad(policy_loss_fn, argnums=0))
    reward_value_and_grad_fn = jax.jit(jax.value_and_grad(reward_loss_fn, argnums=0))

    for episode in range(num_episodes):
        state = env.reset()
        # Initialize gradient accumulators
        policy_grads_accum = [...] 
        reward_grads_accum = [...]
        num_steps_in_episode = 0

        while not done:
            # 1. Get action from policy model
            action_probs = policy_model(np.array([state]).reshape(1, -1))[0]
            action = np.random.choice(env.get_action_space_n(), p=np.array(action_probs))
            
            next_state, true_reward, done = env.step(action) # Environment step

            # 2. Predict reward with reward model
            #    (Input to reward model: current state + action taken)
            action_one_hot = jax.nn.one_hot(action, env.get_action_space_n())
            reward_model_input_np = np.concatenate([...]).reshape(1, -1)
            predicted_reward_value = reward_model(reward_model_input_np)[0]

            # 3. Reward Model Update (Simulating Human Feedback)
            #    In this demo, the reward model learns from the "true_reward" from the environment.
            #    In real RLHF, true_reward would be derived from human preferences.
            reward_params_dict = {"trainable": reward_model.trainable_variables, ...}
            current_reward_loss, reward_grads_step = reward_value_and_grad_fn(
                reward_params_dict, jnp.array(reward_model_input_np), true_reward
            )
            # Accumulate reward_grads_step

            # 4. Policy Model Update
            #    The policy model is updated using the reward predicted by our *reward_model*.
            stopped_predicted_reward = jax.lax.stop_gradient(predicted_reward_value[0])
            policy_params_dict = {"trainable": policy_model.trainable_variables, ...}
            current_policy_loss, policy_grads_step = policy_value_and_grad_fn(
                policy_params_dict, jnp.array([state]).reshape(1, -1), jnp.array(action), stopped_predicted_reward
            )
            # Accumulate policy_grads_step
            
            state = next_state
            num_steps_in_episode += 1
        
        # Apply accumulated gradients at the end of the episode
        if num_steps_in_episode > 0:
            avg_policy_grads = [jnp.clip(g / num_steps_in_episode, -1.0, 1.0) ...]
            policy_optimizer.apply_gradients(zip(avg_policy_grads, policy_model.trainable_variables))
            # Similarly for reward model gradients...
            
        if (episode + 1) % 10 == 0: # Reduced print frequency
            # Print average losses
            ...
```

**Key Parts of the Training Loop:**

1.  **Initialization:** Optimizers for both policy and reward models are created. JAX gradient functions (`policy_value_and_grad_fn`, `reward_value_and_grad_fn`) are defined using `jax.value_and_grad` and `model.stateless_call` for compatibility with JAX's functional programming paradigm.
2.  **Episode Iteration:** The agent interacts with the environment for a set number of episodes.
3.  **Action Selection:** The policy model determines the action to take in the current state.
4.  **Environment Interaction:** The agent performs the action and receives the next state, a "true" reward (for training the reward model in this demo), and a done signal.
5.  **Reward Model Training (Simulated Human Feedback):**
    *   The reward model predicts a reward for the (state, action) pair.
    *   Its loss is calculated against the `true_reward` from the environment (this step simulates learning from human preferences where `true_reward` would be derived from human rankings/choices).
    *   Gradients are computed and accumulated.
6.  **Policy Model Training:**
    *   The policy model's loss is calculated. Crucially, this loss is based on the reward predicted by our **current reward model** (not the environment's true reward). This is the core of making the policy learn what the *reward model* thinks is good.
    *   `jax.lax.stop_gradient` is used on the reward model's prediction when calculating the policy loss. This ensures that the policy update doesn't try to backpropagate gradients *through* the reward model (i.e., the policy takes the current reward model's scores as ground truth for its update).
    *   Gradients are computed and accumulated.
7.  **Gradient Application:** At the end of each episode, the accumulated gradients for both the policy and reward models are averaged, clipped (to prevent excessively large updates), and applied using their respective optimizers.
8.  **Logging:** Periodically, average losses are printed.

## 4. How to Run the Demo

To run the demo, execute the Python script from your terminal:

```bash
python examples/rl/rlhf_dummy_demo.py
```

This will:
1.  Initialize the environment, policy model, and reward model.
2.  Print summaries of the policy and reward models.
3.  Start the RLHF training loop for the specified number of episodes (default is 10 in the modified script).
4.  Print training progress (episode number, total reward, average policy loss, average reward loss).
5.  After training, it will test the trained policy model for a few steps and print the interactions.

## 5. Note on Current Timeout Issues (Development Context)

During the development and testing of this `rlhf_dummy_demo.py` script in a specific sandboxed environment, persistent timeout issues were encountered. Even with a significantly reduced environment size (`size=3`), a small number of episodes (`num_episodes=10`), and JIT compilation enabled for JAX functions, the script would often exceed the execution time limit (approx. 6-7 minutes).

The root cause of this extreme slowdown in that particular context was not definitively pinpointed but could be due to:
*   Specific interactions or inefficiencies within the Keras/JAX stack (`model.stateless_call`, `jax.grad`, optimizer updates) for this setup.
*   Severe performance limitations of the testing sandbox.
*   Subtle JAX JIT recompilation issues triggered by type or shape inconsistencies that were not fully resolved.

The script, as provided, represents the logical structure of a dummy RLHF loop. If you encounter similar performance issues in your environment, further profiling and investigation specific to your JAX/Keras versions and hardware would be necessary. For typical local machine execution, 10 episodes of this simple demo should complete very quickly.

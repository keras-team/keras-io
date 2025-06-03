# Set Keras backend to JAX
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_core as keras
# import GradientTape was removed as we will use jax.grad
import jax
import jax.numpy as jnp
import numpy as np

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

# Define a simple policy model
def create_policy_model(observation_space_shape, action_space_n):
    inputs = keras.Input(shape=observation_space_shape)
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(action_space_n, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Define a simple reward model
def create_reward_model(observation_space_shape, action_space_n):
    inputs = keras.Input(shape=(observation_space_shape[0] + action_space_n,)) # obs + action
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(1)(x) # Outputs a scalar reward
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# RLHF Training Loop
def rlhf_training_loop(env, policy_model, reward_model, num_episodes=10, learning_rate=0.001): # Reduced default episodes
    policy_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    reward_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Define loss functions for jax.grad
    @jax.jit
    def policy_loss_fn(policy_model_params, state_input, action, predicted_reward_value_stopped):
        # stateless_call might return a tuple (e.g., (outputs, other_states) or just (outputs,))
        # We are interested in the first element, which should be the main output tensor.
        predictions_tuple = policy_model.stateless_call(
            policy_model_params["trainable"], 
            policy_model_params["non_trainable"], 
            state_input
        )
        actual_predictions_tensor = predictions_tuple[0] 
        action_probs = actual_predictions_tensor[0] # If actual_predictions_tensor is (1,2)
        selected_action_prob = action_probs[action]
        log_prob = jnp.log(selected_action_prob + 1e-7)
        loss_value = -log_prob * predicted_reward_value_stopped
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

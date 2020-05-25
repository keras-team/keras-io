"""
Title: Deep Q Network Method
Author: [Jacob Chapman](https://twitter.com/jacoblchapman)
Date created: 2020/05/23
Last modified: 2020/05/23
Description: Implement Deep Q Network in Breakout Atari environment.
"""
"""
## Introduction

This script shows an implementation of Deep Q Network method on BreakoutNoFrameskip-v4 environment.

### Deep Q Network Method

As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to an action. An agent will choose an action 
in a given state based on a Q quantity, which is a weighted reward based on the 
expected highest long term reward. This method is considered an Off-Policy method, 
meaning its Q values are updated assuming that the best action was chosen, even 
if the best action was not chosen. Thus A Q-Learning Agent learns to perform its 
tasks, such that the recommended action maximizes the potential rewards.


### BreakoutNoFrameskip-v4

A board moves along the bottom of the screen returning a ball that will destroy blocks
at the top of the screen. The aim of the game is to remove all blocks and breakout of the
level. The agent must learn to control the board by moving left and right, returning the 
ball and removing all the blocks without the ball passing the board.   

### References

- [Q-Learning](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)
- [Deep Q-Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning)
"""
"""
## Setup
"""

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Minimum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("BreakoutNoFrameskip-v4")
# warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

"""
## Implement Deep Q Network

This network learns an approximation of the Q table, which is a mapping between
the states and actions that an agent will take. For every state we'll have two
actions, that can be taken. The environment provides the state and the action
is choose by selecting the larger of the two q-values predicted in the output layer.

"""

num_actions = 4

# Network defined by the Deepmind paper
inputs = layers.Input(shape=(84, 84, 4,))

# Convolutions on the frames on the screen
layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

layer4 = layers.Flatten()(layer3)

layer5 = layers.Dense(512, activation="relu")(layer4)
layer6 = layers.Dense(512, activation="relu")(layer5)
action = layers.Dense(num_actions, activation="linear")(layer6)

model = keras.Model(inputs=inputs, outputs=action)

"""
## Train
"""

optimizer = keras.optimizers.RMSprop(learning_rate=0.00025, momentum=0.95)
mse_loss = keras.losses.MeanSquaredError()
model.compile(loss=mse_loss, optimizer=optimizer)


action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
running_reward = 0
episode_count = 0
frame_count = 0
epsilon_random_frames = 50000  # Number of frames to take random action and observe output
epsilon_greedy_frames = 1000000.0  # Number of frames for exploration
max_memory_length = 1000000  # Maximum replay length
update_after_actions = 4

while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor)
            # Take best action
            action = np.argmax(action_probs[0])

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward if not done else -1.0)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            done_sample = [done_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]

            future_rewards = model(state_next_sample)
            q_values = np.array(model(state_sample))

            history = zip(
                rewards_sample, done_sample, action_sample, future_rewards, q_values
            )

            # Build the new updated Q values for the sampled states and actions
            updated_q_values = []
            for (
                sampled_reward,
                sampled_done,
                sampled_action,
                future_reward,
                q_value,
            ) in history:
                # If final frame take the last reward as the new Q value
                q_update = sampled_reward
                if not sampled_done:
                    # Calculate the new Q value
                    # Q value = reward + discount factor * expected future reward
                    q_update = sampled_reward + gamma * np.amax(future_reward)

                # Replace old Q value with the updated value
                q_value[sampled_action] = q_update
                updated_q_values.append(q_value)

            # Train the model on the states and updated Q values
            model.fit(state_sample, np.array(updated_q_values), verbose=0)

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break
    # Update running reward to check condition for solving
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    # Log details
    episode_count += 1
    if episode_count % 500 == 0:
        template = "running reward: {:.2f} at episode {}, frame count {}, epsilon {}"
        print(template.format(running_reward, episode_count, frame_count, epsilon))

    if running_reward > 2000:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

"""
Title: Deep Q Network Method
Author: [Jacob Chapman](https://twitter.com/jacoblchapman)
Date created: 2020/05/23
Last modified: 2020/05/23
Description: Implement Deep Q Network in CartPole environment.
"""
"""
## Introduction

This script shows an implementation of Deep Q Network method on CartPole-V0 environment.

### Deep Q Network Method

As an agent takes actions and moves through an environment, it learns to map the observed
state of the environment to an action. A Q-Learning Agent learns to perform its tasks,
such that the recommended action maximizes the potential rewards.

### CartPole-V0

A pole is attached to a cart placed on a frictionless track. The agent has to apply force
to move the cart. It is rewarded for for every time step the pole remains upright. The
agent, therefore, must learn to keep the pole from falling over.

### References

- [CartPole](http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf)
- [Q-Learning](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)

"""

"""
## Setup

"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # epsilon greedy parameter
epsilon_min = 0.01  # minimum epsilon greedy parameter
epsilon_decay = 0.999  # rate at which to reduce chance of random action being taken
batch_size = 128  # size of batch taken from replay buffer
max_steps_per_episode = 10000
env = gym.make("CartPole-v0")  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


"""
## Implement Deep Q Network

This network learns an approximation of the q table, which is a mapping between the
states and actions that an agent will take. For every state we'll have two actions, that
can be taken. The environment provides the state and the action is choose by selecting
the larger of the two q-values predicted in the output layer.

"""

num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])


"""
## Train

"""


optimizer = keras.optimizers.Adam(learning_rate=0.01)
mse_loss = keras.losses.MeanSquaredError()
model.compile(loss=mse_loss, optimizer=optimizer)

action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
running_reward = 0
episode_count = 0

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.

        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)

        # Predict action q-values
        # from environment state
        action_probs = model(state_tensor)

        # take best action
        action = np.argmax(action_probs[0])

        # use epsilon-greedy for exploration
        if epsilon > np.random.rand(1)[0]:
            # take random action
            action = np.random.choice(num_actions)

        # decay probability of taking random action
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)

        episode_reward += reward

        # save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward if not done else -1)
        state = state_next

        if len(done_history) > batch_size:
            # sample from history for replay
            indices = np.random.choice(range(len(state_history)), size=batch_size)

            state_sample = np.take(state_history, indices, axis=0)
            state_next_sample = np.take(state_next_history, indices, axis=0)
            rewards_sample = np.take(rewards_history, indices, axis=0)
            done_sample = np.take(done_history, indices, axis=0)
            action_sample = np.take(action_history, indices, axis=0)

            history = zip(
                state_sample,
                state_next_sample,
                rewards_sample,
                done_sample,
                action_sample,
            )

            updated_q_values = []
            for (
                sampled_state,
                sampled_state_next,
                sampled_reward,
                sampled_done,
                sampled_action,
            ) in history:

                q_update = sampled_reward
                if not sampled_done:
                    # Calculate the new q value
                    sampled_state_next = tf.convert_to_tensor(sampled_state_next)
                    sampled_state_next = tf.expand_dims(sampled_state_next, 0)
                    # q value = reward + discount factor * expected future reward
                    q_update = sampled_reward + gamma * np.amax(
                        model(sampled_state_next)[0]
                    )

                sampled_state = tf.convert_to_tensor(sampled_state)
                sampled_state = tf.expand_dims(sampled_state, 0)

                # replace old q value with the updated value
                q_value = np.array(model(sampled_state))[0]
                q_value[sampled_action] = q_update

                updated_q_values.append(q_value)

            # train the model on the states and updated q values
            model.fit(
                tf.convert_to_tensor(state_sample),
                tf.convert_to_tensor(updated_q_values),
                verbose=0,
            )

            # limit the state and reward history
            mem = 100000
            if len(rewards_history) > mem:
                rewards_history = rewards_history[-mem:]
                state_history = state_history[-mem:]
                state_next_history = state_next_history[-mem:]
                action_history = action_history[-mem:]
                done_history = done_history[-mem:]

        if done:
            break
    # Update running reward to check condition for solving
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}, {}"
        print(template.format(running_reward, episode_count, epsilon))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

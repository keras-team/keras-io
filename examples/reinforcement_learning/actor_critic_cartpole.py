"""
Title: Actor Critic Method
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/13
Last modified: 2020/05/13
Description: Implement Actor Critic Method in CartPole environment
"""
"""
## Setup
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import math
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras import optimizers

seed = 543
gamma = 0.99
env = gym.make("CartPole-v0")
env.seed(seed)
eps = np.finfo(np.float32).eps.item()

"""
## Implement Actor Critic network
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

optimizer = optimizers.Adam(learning_rate=0.01)
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:  # run till solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:  # keep track of gradients
        for timestep in range(1, 10000):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action prob distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(math.log(action_probs[0, action]))

            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Perform backprop
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalise
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)
            critic_losses.append(
                losses.Huber()(tf.expand_dims(value, 0), tf.expand_dims(diff, 0))
            )

        loss_value = sum(actor_losses) + sum(critic_losses)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved! at episode {}".format(episode_count))
        break

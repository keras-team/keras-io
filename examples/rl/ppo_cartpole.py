"""
Title: Proximal Policy Optimization
Author: [Ilias Chrysovergis](https://twitter.com/iliachry)
Date created: 2021/06/24
Last modified: 2021/06/24
Description: Implementation of a Proximal Policy Optimization agent for the CartPole-v0 environment.
Accelerator: NONE
"""

"""
## Introduction

This code example solves the CartPole-v0 environment using a Proximal Policy Optimization (PPO) agent.

### CartPole-v0

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The system is controlled by applying a force of +1 or -1 to the cart.
The pendulum starts upright, and the goal is to prevent it from falling over.
A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
After 200 steps the episode ends. Thus, the highest return we can get is equal to 200.

[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/)

### Proximal Policy Optimization

PPO is a policy gradient method and can be used for environments with either discrete or continuous action spaces.
It trains a stochastic policy in an on-policy way. Also, it utilizes the actor critic method. The actor maps the
observation to an action and the critic gives an expectation of the rewards of the agent for the observation given.
Firstly, it collects a set of trajectories for each epoch by sampling from the latest version of the stochastic policy.
Then, the rewards-to-go and the advantage estimates are computed in order to update the policy and fit the value function.
The policy is updated via a stochastic gradient ascent optimizer, while the value function is fitted via some gradient descent algorithm.
This procedure is applied for many epochs until the environment is solved.

![Algorithm](https://i.imgur.com/rd5tda1.png)

- [PPO Original Paper](https://arxiv.org/pdf/1707.06347.pdf)
- [OpenAI Spinning Up docs - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

### Note

This code example uses Keras and Tensorflow v2. It is based on the PPO Original Paper,
the OpenAI's Spinning Up docs for PPO, and the OpenAI's Spinning Up implementation of PPO using Tensorflow v1.

[OpenAI Spinning Up Github - PPO](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/ppo.py)
"""

"""
## Libraries

For this example the following libraries are used:

1. `numpy` for n-dimensional arrays
2. `tensorflow` and `keras` for building the deep RL PPO agent
3. `gym` for getting everything we need about the environment
4. `scipy.signal` for calculating the discounted cumulative sums of vectors
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time

"""
## Functions and class
"""


class PPO:
    def __init__(
        self,
        observation_dimensions,
        num_actions,
        steps_per_epoch,
        policy_learning_rate=3e-4,
        value_function_learning_rate=1e-3,
        clip_ratio=0.2,
        hidden_sizes=(64, 64),
        gamma=0.99,
        lam=0.95
    ):
        self.observation_dimensions = observation_dimensions
        self.steps_per_epoch = steps_per_epoch
        self.hidden_sizes = hidden_sizes
        self.num_actions = num_actions
        self.policy_learning_rate = policy_learning_rate
        self.value_function_learning_rate = value_function_learning_rate
        self.clip_ratio = clip_ratio

        # Initialize the buffer
        self.buffer = self.Buffer(
            self.observation_dimensions,
            self.steps_per_epoch
        )

        # Initialize the actor and the critic as keras models
        observation_input = keras.Input(
            shape=(self.observation_dimensions,),
            dtype=tf.float32
        )

        logits = self.mlp(
            observation_input,
            list(self.hidden_sizes) + [self.num_actions],
            tf.tanh,
            None
        )

        self.actor = keras.Model(inputs=observation_input, outputs=logits)

        value = tf.squeeze(
            self.mlp(
                observation_input,
                list(self.hidden_sizes) + [1],
                tf.tanh,
                None
            ),
            axis=1
        )

        self.critic = keras.Model(inputs=observation_input, outputs=value)

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(
            learning_rate=self.policy_learning_rate
        )

        self.value_optimizer = keras.optimizers.Adam(
            learning_rate=self.value_function_learning_rate
        )

    class Buffer:
        # Buffer for storing trajectories
        def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
            # Buffer initialization
            self.gamma = gamma
            self.lam = lam
            self.pointer = 0
            self.trajectory_start_index = 0

            shape = (size, observation_dimensions)
            self.observation_buffer = np.zeros(shape, dtype=np.float32)
            self.action_buffer = np.zeros(size, dtype=np.int32)
            self.advantage_buffer = np.zeros(size, dtype=np.float32)
            self.reward_buffer = np.zeros(size, dtype=np.float32)
            self.return_buffer = np.zeros(size, dtype=np.float32)
            self.value_buffer = np.zeros(size, dtype=np.float32)
            self.logprobability_buffer = np.zeros(size, dtype=np.float32)

        def store(self, observation, action, reward, value, logprobability):
            # Append one step of agent-environment interaction
            self.observation_buffer[self.pointer] = observation
            self.action_buffer[self.pointer] = action
            self.reward_buffer[self.pointer] = reward
            self.value_buffer[self.pointer] = value
            self.logprobability_buffer[self.pointer] = logprobability
            self.pointer += 1

        def finish_trajectory(self, last_value=0):
            # Finish the trajectory by computing advantage estimates and rewards-to-go
            path_slice = slice(self.trajectory_start_index, self.pointer)
            rewards = np.append(self.reward_buffer[path_slice], last_value)
            values = np.append(self.value_buffer[path_slice], last_value)

            # Compute the GAE-Lambda advantage estimates
            deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
            self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
                deltas, self.gamma * self.lam
            )

            # Compute the rewards-to-go
            self.return_buffer[path_slice] = self.discounted_cumulative_sums(
                rewards, self.gamma
            )[:-1]

            self.trajectory_start_index = self.pointer

        def get(self):
            # Get all data of the buffer and normalize the advantages
            self.pointer, self.trajectory_start_index = 0, 0
            advantage_mean, advantage_std = (
                np.mean(self.advantage_buffer),
                np.std(self.advantage_buffer),
            )
            self.advantage_buffer = (
                self.advantage_buffer - advantage_mean) / advantage_std
            return (
                self.observation_buffer,
                self.action_buffer,
                self.advantage_buffer,
                self.return_buffer,
                self.logprobability_buffer,
            )

        def discounted_cumulative_sums(self, x, discount):
            # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def mlp(self, x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)

    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    @tf.function
    def sample_action(self, observation):
        # Sample action from actor
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    @tf.function
    def train_policy(self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        # Train the policy by maxizing the PPO-Clip objective
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(
                    observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(
            policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self.actor.trainable_variables))

        self.kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        self.kl = tf.reduce_sum(self.kl)
        return self.kl

    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        # Train the value function by regression on mean-squared error
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean(
                (return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(
            value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.critic.trainable_variables))

    def save(self, path):
        """Salva o modelo treinado em um arquivo."""
        self.actor.save(f"{path}_actor.h5")
        self.critic.save(f"{path}_critic.h5")

    def load(self, path):
        """Carrega um modelo previamente treinado de um arquivo."""
        self.actor = keras.models.load_model(f"{path}_actor.h5")
        self.critic = keras.models.load_model(f"{path}_critic.h5")


"""
## Hyperparameters
"""

# Hyperparameters of the PPO algorithm
steps_per_epoch = 4000
epochs = 30
train_policy_iterations = 80
train_value_iterations = 80
target_kl = 0.01
hidden_sizes = (64, 64)

# True if you want to render the environment
render = False

"""
## Initializations
"""

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = gym.make("CartPole-v0")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize PPO
ppo = PPO(
    observation_dimensions=observation_dimensions,
    num_actions=num_actions,
    steps_per_epoch=steps_per_epoch,
)

# Initialize the observation, episode return and episode length
observation, episode_return, episode_length = env.reset(), 0, 0

"""
## Train
"""
# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        if render:
            env.render()

        # Get the logits, action, and take one step in the environment
        observation = observation.reshape(1, -1)
        logits, action = ppo.sample_action(observation)
        observation_new, reward, done, _ = env.step(action[0].numpy())
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = ppo.critic(observation)
        logprobability_t = ppo.logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        ppo.buffer.store(observation, action, reward,
                         value_t, logprobability_t)

        # Update the observation
        observation = observation_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else ppo.critic(observation.reshape(1, -1))
            ppo.buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, episode_return, episode_length = env.reset(), 0, 0

    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = ppo.buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = ppo.train_policy(
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        ppo.train_value_function(observation_buffer, return_buffer)

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )


"""
## Visualizations

Before training:

![Imgur](https://i.imgur.com/rKXDoMC.gif)

After 8 epochs of training:

![Imgur](https://i.imgur.com/M0FbhF0.gif)

After 20 epochs of training:

![Imgur](https://i.imgur.com/tKhTEaF.gif)
"""

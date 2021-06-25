"""
Title: Proximal Policy Optimization
Author: [Ilias Chrysovergis](https://twitter.com/iliachry)
Date created: 2021/06/24
Last modified: 2021/06/24
Description: Implementation of a Proximal Policy Optimization agent for the CartPole-v0 environment.
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

1. Numpy for n-dimensional arrays
2. Tensorflow and Keras for building the deep RL PPO agent
3. Gym for getting everything we need about the environment
4. Scipy.signal for calculating the discounted cumulative sums of vectors
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal

"""
## Functions and class
"""


def discount_cumsum(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, obs_dim, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.tr_start_idx = 0, 0

    def store(self, obs, act, rew, val, logp):
        # Append one step of agent-environment interaction
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_tr(self, last_val=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.tr_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.tr_start_idx = self.ptr

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.ptr, self.tr_start_idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobs(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logp_all = tf.nn.log_softmax(logits)
    logp = tf.reduce_sum(tf.one_hot(a, n_acts) * logp_all, axis=1)
    return logp


"""
## Hyperparameters
"""

# Hyperparameters of the PPO algorithm
steps_per_epoch = 4000
epochs = 30
gamma = 0.99
clip_ratio = 0.2
pi_lr = 3e-4
vf_lr = 1e-3
train_pi_iters = 80
train_v_iters = 80
lam = 0.97
taget_kl = 0.01
hidden_sizes = (64, 64)

# True if you want to render the environment
render = False

"""
## Initializations
"""

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = gym.make("CartPole-v0")
obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n

# Initialize the buffer
buf = Buffer(obs_dim, steps_per_epoch)

# Initialize the actor and the critic as keras models
obs_in = keras.Input(shape=(obs_dim,), dtype=tf.float32)
logits = mlp(obs_in, list(hidden_sizes) + [n_acts], tf.tanh, None)
actor = keras.Model(inputs=obs_in, outputs=logits)
v = tf.squeeze(mlp(obs_in, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
critic = keras.Model(inputs=obs_in, outputs=v)

# Initialize the policy and the value function optimizers
pi_optimizer = keras.optimizers.Adam(learning_rate=pi_lr)
v_optimizer = keras.optimizers.Adam(learning_rate=vf_lr)

# Initialize the observation, episode return and episode length
obs, ep_ret, ep_len = env.reset(), 0, 0

"""
## Train
"""
# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_ret = 0
    num_ep = 0

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        if render:
            env.render()

        # Get the logits, action, and take one step in the environment
        obs = obs.reshape(1, -1)
        logits = actor(obs)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        obs_n, rew, done, _ = env.step(action[0].numpy())
        ep_ret += rew
        ep_len += 1

        # Get the value and log-probability of the action
        v_t = critic(obs)
        logp_pi_t = logprobs(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buf.store(obs, action, rew, v_t, logp_pi_t)

        # Update the observation
        obs = obs_n

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_val = 0 if done else critic(obs.reshape(1, -1))
            buf.finish_tr(last_val)
            sum_ret += ep_ret
            num_ep += 1
            obs, ep_ret, ep_len = env.reset(), 0, 0

    # Get values from the buffer
    obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buf.get()

    # Update the policy by maxizing the PPO-Clip objective
    for _ in range(train_pi_iters):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(logprobs(actor(obs_buf), act_buf) - logp_buf)
            min_adv = tf.where(
                adv_buf > 0, (1 + clip_ratio) * adv_buf, (1 - clip_ratio) * adv_buf
            )

            pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_buf, min_adv))
        pi_grads = tape.gradient(pi_loss, actor.trainable_variables)
        pi_optimizer.apply_gradients(zip(pi_grads, actor.trainable_variables))

        kl = tf.reduce_mean(logp_buf - logprobs(actor(obs_buf), act_buf))
        kl = tf.reduce_sum(kl)
        if kl > 1.5 * taget_kl:
            # Early Stopping
            break

    # Update the value function by regression on mean-squared error
    for _ in range(train_v_iters):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            v_loss = tf.reduce_mean((ret_buf - critic(obs_buf)) ** 2)
        v_grads = tape.gradient(v_loss, critic.trainable_variables)
        v_optimizer.apply_gradients(zip(v_grads, critic.trainable_variables))

    # Print mean return and length for each epoch
    print(" \n Epoch: " + str(epoch + 1) + ". Mean Return: " + str(sum_ret / num_ep))


"""
## Visualizations

Before training:

![Imgur](https://i.imgur.com/rKXDoMC.gif)

After 8 epochs of training:

![Imgur](https://i.imgur.com/M0FbhF0.gif)

After 20 epochs of training:

![Imgur](https://i.imgur.com/tKhTEaF.gif)
"""

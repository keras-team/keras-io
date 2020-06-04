"""
Title: Deep Deterministic Policy Gradient (DDPG)
Author: [amifunny](https://github.com/amifunny)
Date created: 2020/06/04
Last modified: 2020/06/04
Description: Implementing DDPG algorithm with Experience Relay on the Inverted Pendulum Problem.
"""
"""
# Introduction
**Deep Deterministic Policy Gradient (DDPG)** is a popular algorithm for learning **good
actions** corresponding to agent's **State**.

This tutorial closely follow this paper -  
[Continuous control with deep reinforcement
learning](https://arxiv.org/pdf/1509.02971.pdf)

# Problem
We are trying to solve classic control problem of **Inverted Pendulum**. In this we can
take only two actions - Swing LEFT or Swing RIGHT. 

Now what make this **problem challenging for Q-learning Algorithms** is that **actions
are Continuous** instead of being Discrete. That is instead of using two discrete actions
like [ -1 or +1] , we have to select from infinite actions ranging from -2 to +2.

# Quick Theory

Just like A2C Method , we have two Networks -

1. Actor - It just takes the action.
2. Critic - It tell if action is good( gives +ve value) or   bad(-ve value)

But DDPG uses two more tricks -

**First, Uses two Target Networks.**

**Why?** Because it add stability to training. In short , We are learning from estimated
targets and Target Network are updated slowly hence keeping our estimated targets stable.

Conceptually it's like saying, "I have an idea of how to play this well, I'm going to try
it out for a bit until I find something better" as opposed to saying "I'm going to
retrain myself how to play this entire game after every move". See this answer -
[stackoverflow](https://stackoverflow.com/a/54238556/13475679)

**Second , Uses Experience Relay.**

It is basically list of tuples of (state,action,reward,next_state). So instead of
learning from recent experiences , **you learn from sample with fair amount of
successful,  failed, early and recent experiences.**


Now lets see how is it implemented.
"""

# We use openai gym for Pendulum Env.
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

"""
Standard Way of creating [GYM Environment](http://gym.openai.com/docs). We will use
**upper_bound** to scale our actions later.
"""

problem = "Pendulum-v0"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print("Size of State Space = >  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space = >  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action = >  {}".format(upper_bound))
print("Min Value of Action = >  {}".format(lower_bound))

"""
Now for Exploration by our Actor , we use noisy perturbation, specifically
**Ornstein-Uhlenbeck process** as described in paper. Its basically sampling noise from a
"correlated" normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # Its standard code for this process.
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )

        # This makes this noise more correlated
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mean)


"""
**Buffer** class implements the Experience Relay Concept.

---
![Imgur](https://i.imgur.com/mS6iGyJ.jpg)


---


**Critic Loss** - Mean Squared Error of **( y - Q(s,a) )**
where **y** is expected return determined by target critic network and Q(s,a) is value of
state-action given by critic network. We train our critic network using the computed
loss. 

**y** is a moving target that critic model tries to achieve, but we make this target
stable by updating out target model slowly.

**Actor Loss** - This is computed using mean of value given by critic model for the
actions taken by Actor network. We like to maximize this. So use negative sign before the
computed mean and use this to do gradient descent.

Hence we update Actor Network such that it produces actions that gets maximum value from
critic , for a given state.



"""


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_ctr = 0

        # Instead of list of tuples as the exp.relay concept go
        # We use different np.arrays for each tuple element
        # But is its more easy to convert and keeps thing clean.
        self.state_buff = np.zeros((self.buffer_capacity, num_states))
        self.action_buff = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buff = np.zeros((self.buffer_capacity, 1))
        self.next_state_buff = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):

        # To make index zero if buffer_capacity excedded
        # Hence replacing old records
        index = self.buffer_ctr % self.buffer_capacity

        self.state_buff[index] = obs_tuple[0]
        self.action_buff[index] = obs_tuple[1]
        self.reward_buff[index] = obs_tuple[2]
        self.next_state_buff[index] = obs_tuple[3]

        self.buffer_ctr += 1

    # We compute loss and update parameters
    def learn(self):

        # Get range upto which to sample
        record_range = min(self.buffer_ctr, self.buffer_capacity)
        # Randomly sample indexes
        batch_idx = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buff[batch_idx])
        action_batch = tf.convert_to_tensor(self.action_buff[batch_idx])
        reward_batch = tf.convert_to_tensor(self.reward_buff[batch_idx])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buff[batch_idx])

        # Training and updating Actor - Critic Networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:

            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])

            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:

            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            # Used -ve as we want to max the value given by critic on our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )


# This update target Parameters slowly
#  On basis of tau that is much less than one.
def update_target(tau):

    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic_model.weights):

        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables = target_actor.weights
    for i, variable in enumerate(actor_model.weights):

        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_actor.set_weights(new_weights)


"""
Here we declare Actor and Critic Networks. These are basic Multiple Dense layer Networks
with 'ReLU' Activation.

NOTICE : We use initialization for last layer of actor to be between -0.003 to 0.003 as
this prevents from reaching 1 or -1 value in initial stages which will cut off our
Gradient to Zero, as 'tanh' is used
"""


def get_actor():

    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(512, activation="relu")(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(512, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    # This scale out our Actions
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():

    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(32, activation="relu")(state_out)
    state_out = layers.BatchNormalization()(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)
    action_out = layers.BatchNormalization()(action_out)

    # Both are passed through seperate layer before concatenating.
    merged = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(512, activation="relu")(merged)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(512, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give State-Action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


"""
Policy( ) returns Action given by our Actor Network plus some Noise for exploration. 
"""


def policy(state, noise_object):

    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


"""
HYPER PARAMETERS and OBJECT Declaration
"""

stddev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(stddev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the Weights same at start
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

"""
Now we implement our Main Loop , and iterate through episodes. We take action using
policy() and learn() at each time step, along with updating target networks using 'tau'.



"""

ep_reward_list = []
avg_reward_list = []

with tf.device("/device:GPU:0"):

    # Takes about 20 min to train
    for ep in range(total_episodes):

        prev_state = env.reset()
        episodic_r = 0

        while True:

            # Uncomment this to see the action
            # But not in notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            buffer.record((prev_state, action, reward, state))
            episodic_r += reward

            buffer.learn()
            update_target(tau)

            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_r)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

# Plot a Graph
# Episodes vs Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

"""
![Graph](https://i.imgur.com/sqEtM6M.png)
"""

"""
If Networks Learn properly , Average Episodic Reward wil increase with time.

Feel Free to Try Different learning rates , tau and architectures for Actor - Critic
Networks.

The Inverted Pendulum problem has low complexity but DDPG work great on any problem.

Another Great Environment to try this on is 'LunarLandingContinuous' but will take more
episodes than this but gives good results.

"""

# Save the Weights
actor_model.save_weights("pendulum_actor.h5")
critic_model.save_weights("pendulum_critic.h5")

target_actor.save_weights("pendulum_t_actor.h5")
target_critic.save_weights("pendulum_t_critic.h5")

"""
Before Training :-

![before_img](https://i.imgur.com/ox6b9rC.gif)
"""

"""
After 100 episodes :-

![after_img](https://i.imgur.com/eEH8Cz6.gif)

"""

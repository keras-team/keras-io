"""
Title: Deep Deterministic Policy Gradient (DDPG)
Author: [amifunny](https://github.com/amifunny)
Date created: 2020/06/3
Last modified: 2020/06/3
Description: Implementing DDPG algorithm with Experience Relay on Pendulum Problem.
"""
"""
# Introduction
**Deep Deterministic Policy Gradient (DDPG)** is a popular algorithm for learning **good
actions** corresponding to agent's **State**.

This tutorial closely follow this paper - [Continuous control with deep reinforcement
learning](https://arxiv.org/pdf/1509.02971.pdf)

# Problem
We are trying to solve classic control problem of **Pendulum**. In this we can take only
two actions - Swing LEFT or Swing RIGHT.

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


Enough with theory lets get to interesting part.**CODE!!**


"""

# We use gym for Pendulum Env.
# and tf 2.0 bcz ... we love it!
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

nS = env.observation_space.shape[0]
print("Size of State Space = >  {}".format(nS))
nA = env.action_space.shape[0]
print("Size of Action Space = >  {}".format(nA))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action = >  {}".format(upper_bound))
print("Min Value of Action = >  {}".format(lower_bound))

"""
Now instead of using good old e-greedy , we use noisy perturbation, specifically
**Ornstein-Uhlenbeck process** as described in paper. Don't get scared from the big name
, its basically sampling noise from a "special" normal distribution.
"""


class OUActionNoise:
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # Its standard code for this process , formula taken from wiki.
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        #  this makes this noise more correlated
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


"""
**Buffer** class implements the Experience Relay Concept.

  ![Imgur](https://i.imgur.com/mS6iGyJ.jpg)
"""


class Buffer:
    def __init__(self, buff_cap=100000, batch_size=64):

        # buff_cap is number of "experience" to store at max
        self.buff_cap = buff_cap
        # batch_size - num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buff_ctr = 0

        # I know this is not list of tuples as the exp.relay concept go.
        # But is its more easy to convert and keeps thing clean.
        self.state_buff = np.zeros((self.buff_cap, nS))
        self.action_buff = np.zeros((self.buff_cap, nA))
        self.reward_buff = np.zeros((self.buff_cap, 1))
        self.next_state_buff = np.zeros((self.buff_cap, nS))

    def record(self, obs_tuple):

        # To make index zero if buff_cap excedded
        # hence replaing oldese records
        index = self.buff_ctr % self.buff_cap

        self.state_buff[index] = obs_tuple[0]
        self.action_buff[index] = obs_tuple[1]
        self.reward_buff[index] = obs_tuple[2]
        self.next_state_buff[index] = obs_tuple[3]

        self.buff_ctr += 1

    # This is the main juice.
    def learn(self):

        # get range upto which to sample
        record_range = min(self.buff_ctr, self.buff_cap)
        batch_idx = np.random.choice(record_range, self.batch_size)

        # convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buff[batch_idx])
        action_batch = tf.convert_to_tensor(self.action_buff[batch_idx])
        reward_batch = tf.convert_to_tensor(self.reward_buff[batch_idx])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buff[batch_idx])

        # Training and updating Actor - Critic Networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:

            target_actions = target_actor(next_state_batch)
            y = reward_batch + GAMMA * target_critic([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])

            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:

            actor_part = actor_model(state_batch)
            critic_part = critic_model([state_batch, actor_part])
            # used -ve as we want to max the value given by critic on our actions
            actor_loss = -tf.math.reduce_mean(critic_part)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )


# This update target Parameters slowly
#  on basis of tau that is much less than one.
def update_target(tau):

    for i, target_layer in enumerate(target_critic.layers):

        layer = critic_model.get_layer(index=i)

        if target_layer.weights != []:

            layer_var = layer.get_weights()
            target_var = target_layer.get_weights()

            new_W = layer_var[0] * tau + target_var[0] * (1 - tau)
            new_b = layer_var[1] * tau + target_var[1] * (1 - tau)

            target_layer.set_weights([new_W, new_b])

    for i, target_layer in enumerate(target_actor.layers):

        layer = actor_model.get_layer(index=i)

        if target_layer.weights != []:

            layer_var = layer.get_weights()
            target_var = target_layer.get_weights()

            new_W = layer_var[0] * tau + target_var[0] * (1 - tau)
            new_b = layer_var[1] * tau + target_var[1] * (1 - tau)

            target_layer.set_weights([new_W, new_b])


"""
Here we declare Actor and Critic Networks. Its simple Deep Learning 101. Basic Dense
layers with 'ReLU' Activation.

NOTICE : We use initialization for last layer of actor to be between -0.003 to 0.003 as
this prevents from reaching 1 or -1 value in initial stages which will make our make our
Gradient Zero Because its "tanh"!!
"""


def get_actor():

    init = tf.keras.initializers.GlorotNormal(seed=1)
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(nS))
    out = layers.Dense(512, activation="relu", kernel_initializer=init)(inputs)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(512, activation="relu", kernel_initializer=init)(out)
    out = layers.LayerNormalization()(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # our upper bound is 2.0 for Pendulum.
    # This scale out our Actions
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():

    init = tf.keras.initializers.GlorotNormal(seed=1)

    # state as input
    input_s = layers.Input(shape=(nS))
    state_h1 = layers.Dense(24, activation="relu", kernel_initializer=init)(input_s)
    state_h1 = layers.LayerNormalization()(state_h1)
    state_h2 = layers.Dense(48, activation="relu", kernel_initializer=init)(state_h1)
    state_h2 = layers.LayerNormalization()(state_h2)

    # Action as input
    input_a = layers.Input(shape=(1))
    action_h1 = layers.Dense(48, activation="relu", kernel_initializer=init)(input_a)
    action_h1 = layers.LayerNormalization()(action_h1)

    # Both are passed through seperate layer before concatenating.
    merged = layers.Concatenate()([input_s, input_a])

    out = layers.Dense(512, activation="relu", kernel_initializer=init)(merged)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(512, activation="relu", kernel_initializer=init)(out)
    out = layers.LayerNormalization()(out)
    outputs = layers.Dense(1, kernel_initializer=init)(out)

    # Outputs single value for give State-Action
    model = tf.keras.Model([input_s, input_a], outputs)

    return model


"""
Policy give Action given by out Actor Network plus some Exploratory Noise for
exploration.
"""


def policy(state, noise_obj):

    sampled_nums = tf.squeeze(actor_model(state))
    noise = noise_obj()
    sampled_nums = sampled_nums.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_nums, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


"""
**HYPER PARAMETERS and OBJECT Declaration**
"""

stddev = 0.2
ou_noise = OUActionNoise(mu=np.zeros(1), sigma=float(stddev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the Weights same at start
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

c_lr = 0.002
a_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(c_lr)
actor_optimizer = tf.keras.optimizers.Adam(a_lr)

total_eps = 100
GAMMA = 0.99
tau = 0.01

buffer = Buffer(50000, 64)

"""
**Its Play time!!** See Creating Objects paid off making our Main Loop Concise.
"""

ep_reward_list = []
avg_reward_list = []

with tf.device("/device:CPU:0"):

    # takes about 20 min to train
    for ep in range(total_eps):

        prev_state = env.reset()
        episodic_r = 0

        while True:

            # Uncomment this to see the action
            # But not in notebook ;)
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise)
            state, reward, done, info = env.step(action)

            buffer.record((prev_state, action, reward, state))
            episodic_r += reward

            # buffer.learn()
            # update_target(tau)

            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_r)

        # mean of last 20 episodes
        avg_reward = np.mean(ep_reward_list[-20:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, episodic_r, avg_reward))
        avg_reward_list.append(avg_reward)


# Plot a Graph
# Episodes vs Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")

plt.show()


"""
![img](https://i.imgur.com/yJg2t3e.png)
"""

"""
Now that what i call a beautiful graph.

I hope this tutorial helped you see the beauty and possibilities of Reinforcement
Learning and ease of implementation using Tensorflow 2.0 and Keras.

Key Takeaways are Target Networks and Buffer that have become standards in RL field.

Feel Free to Try Different learning rates , tau and architectures for Actor - Critic
Networks.

Pendulum problem has low complexity but DDPG work great on any problem.

Another Great Environment to try this is 'LunarContinuousLanding' but will take more
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

![after_img](https://i.imgur.com/oVUMC7H.gif)

"""

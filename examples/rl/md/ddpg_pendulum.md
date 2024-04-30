# Deep Deterministic Policy Gradient (DDPG)

**Author:** [amifunny](https://github.com/amifunny)<br>
**Date created:** 2020/06/04<br>
**Last modified:** 2024/03/23<br>
**Description:** Implementing DDPG algorithm on the Inverted Pendulum Problem.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py)



---
## Introduction

**Deep Deterministic Policy Gradient (DDPG)** is a model-free off-policy algorithm for
learning continuous actions.

It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network).
It uses Experience Replay and slow-learning target networks from DQN, and it is based on
DPG, which can operate over continuous action spaces.

This tutorial closely follow this paper -
[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

---
## Problem

We are trying to solve the classic **Inverted Pendulum** control problem.
In this setting, we can take only two actions: swing left or swing right.

What make this problem challenging for Q-Learning Algorithms is that actions
are **continuous** instead of being **discrete**. That is, instead of using two
discrete actions like `-1` or `+1`, we have to select from infinite actions
ranging from `-2` to `+2`.

---
## Quick theory

Just like the Actor-Critic method, we have two networks:

1. Actor - It proposes an action given a state.
2. Critic - It predicts if the action is good (positive value) or bad (negative value)
given a state and an action.

DDPG uses two more techniques not present in the original DQN:

**First, it uses two Target networks.**

**Why?** Because it add stability to training. In short, we are learning from estimated
targets and Target networks are updated slowly, hence keeping our estimated targets
stable.

Conceptually, this is like saying, "I have an idea of how to play this well,
I'm going to try it out for a bit until I find something better",
as opposed to saying "I'm going to re-learn how to play this entire game after every
move".
See this [StackOverflow answer](https://stackoverflow.com/a/54238556/13475679).

**Second, it uses Experience Replay.**

We store list of tuples `(state, action, reward, next_state)`, and instead of
learning only from recent experience, we learn from sampling all of our experience
accumulated so far.

Now, let's see how is it implemented.


```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

import tensorflow as tf
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
```

We use [Gymnasium](https://gymnasium.farama.org/) to create the environment.
We will use the `upper_bound` parameter to scale our actions later.


```python
# Specify the `render_mode` parameter to show the attempts of the agent in a pop up window.
env = gym.make("Pendulum-v1", render_mode="human")

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))
```

<div class="k-default-codeblock">
```
Size of State Space ->  3
Size of Action Space ->  1
Max Value of Action ->  2.0
Min Value of Action ->  -2.0

```
</div>
To implement better exploration by the Actor network, we use noisy perturbations,
specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.


```python

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

```

The `Buffer` class implements Experience Replay.

---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---


**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.

**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.

Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.


```python

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = keras.ops.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(
            self.next_state_buffer[batch_indices]
        )

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)

```

Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.

Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.


```python

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = keras.Model([state_input, action_input], outputs)

    return model

```

`policy()` returns an action sampled from our Actor network plus some noise for
exploration.


```python

def policy(state, noise_object):
    sampled_actions = keras.ops.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

```

---
## Training hyperparameters


```python
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = keras.optimizers.Adam(critic_lr)
actor_optimizer = keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)
```

Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.


```python
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0

    while True:
        tf_prev_state = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(prev_state), 0
        )

        action = policy(tf_prev_state, ou_noise)
        # Receive state and reward from environment.
        state, reward, done, truncated, _ = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()

        update_target(target_actor, actor_model, tau)
        update_target(target_critic, critic_model, tau)

        # End this episode when `done` or `truncated` is True
        if done or truncated:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
```

<div class="k-default-codeblock">
```
Episode * 0 * Avg Reward is ==> -1020.8244931732263

Episode * 1 * Avg Reward is ==> -1338.2811167733332

Episode * 2 * Avg Reward is ==> -1450.0427316158366

Episode * 3 * Avg Reward is ==> -1529.0751774957375

Episode * 4 * Avg Reward is ==> -1560.3468658090717

Episode * 5 * Avg Reward is ==> -1525.6201906715812

Episode * 6 * Avg Reward is ==> -1522.0047531836371

Episode * 7 * Avg Reward is ==> -1507.4391205141226

Episode * 8 * Avg Reward is ==> -1443.4147334537984

Episode * 9 * Avg Reward is ==> -1452.0432974943765

Episode * 10 * Avg Reward is ==> -1344.1960761302823

Episode * 11 * Avg Reward is ==> -1327.0472948059835

Episode * 12 * Avg Reward is ==> -1332.4638031402194

Episode * 13 * Avg Reward is ==> -1287.4884456842617

Episode * 14 * Avg Reward is ==> -1257.3643575644046

Episode * 15 * Avg Reward is ==> -1210.9679762262906

Episode * 16 * Avg Reward is ==> -1165.8684037899104

Episode * 17 * Avg Reward is ==> -1107.6228192573426

Episode * 18 * Avg Reward is ==> -1049.4192654959388

Episode * 19 * Avg Reward is ==> -1003.3255480245641

Episode * 20 * Avg Reward is ==> -961.6386918013155

Episode * 21 * Avg Reward is ==> -929.1847739440876

Episode * 22 * Avg Reward is ==> -894.356849609832

Episode * 23 * Avg Reward is ==> -872.3450419603026

Episode * 24 * Avg Reward is ==> -842.5992147531034

Episode * 25 * Avg Reward is ==> -818.8730806655396

Episode * 26 * Avg Reward is ==> -793.3147256249664

Episode * 27 * Avg Reward is ==> -769.6124209263007

Episode * 28 * Avg Reward is ==> -747.5122117563488

Episode * 29 * Avg Reward is ==> -726.8111953151997

Episode * 30 * Avg Reward is ==> -707.3781885286952

Episode * 31 * Avg Reward is ==> -688.9993520703357

Episode * 32 * Avg Reward is ==> -672.0164054875188

Episode * 33 * Avg Reward is ==> -652.3297236089893

Episode * 34 * Avg Reward is ==> -633.7305579653394

Episode * 35 * Avg Reward is ==> -622.6444438529929

Episode * 36 * Avg Reward is ==> -612.2391199605028

Episode * 37 * Avg Reward is ==> -599.2441039477458

Episode * 38 * Avg Reward is ==> -593.713500114108

Episode * 39 * Avg Reward is ==> -582.062487157142

Episode * 40 * Avg Reward is ==> -556.559275313473

Episode * 41 * Avg Reward is ==> -518.053376711216

Episode * 42 * Avg Reward is ==> -482.2191305356082

Episode * 43 * Avg Reward is ==> -441.1561293090619

Episode * 44 * Avg Reward is ==> -402.0403515001418

Episode * 45 * Avg Reward is ==> -371.3376110030464

Episode * 46 * Avg Reward is ==> -336.8145387714556

Episode * 47 * Avg Reward is ==> -301.7732070717081

Episode * 48 * Avg Reward is ==> -281.4823965447058

Episode * 49 * Avg Reward is ==> -243.2750024568545

Episode * 50 * Avg Reward is ==> -236.6512197943394

Episode * 51 * Avg Reward is ==> -211.20860968588096

Episode * 52 * Avg Reward is ==> -176.31339260650844

Episode * 53 * Avg Reward is ==> -158.77021134671222

Episode * 54 * Avg Reward is ==> -146.76749516161257

Episode * 55 * Avg Reward is ==> -133.93793525539664

Episode * 56 * Avg Reward is ==> -129.24881351771964

Episode * 57 * Avg Reward is ==> -129.49219614666802

Episode * 58 * Avg Reward is ==> -132.53205721511375

Episode * 59 * Avg Reward is ==> -132.60389802731262

Episode * 60 * Avg Reward is ==> -132.62344822194035

Episode * 61 * Avg Reward is ==> -133.2372468795715

Episode * 62 * Avg Reward is ==> -133.1046546040286

Episode * 63 * Avg Reward is ==> -127.17488349564069

Episode * 64 * Avg Reward is ==> -130.02349725294775

Episode * 65 * Avg Reward is ==> -127.32475296620544

Episode * 66 * Avg Reward is ==> -126.99528350924034

Episode * 67 * Avg Reward is ==> -126.65903554713267

Episode * 68 * Avg Reward is ==> -126.63950221408372

Episode * 69 * Avg Reward is ==> -129.4066259498526

Episode * 70 * Avg Reward is ==> -129.34372109952105

Episode * 71 * Avg Reward is ==> -132.29705860930432

Episode * 72 * Avg Reward is ==> -132.00732697620566

Episode * 73 * Avg Reward is ==> -138.01483877165032

Episode * 74 * Avg Reward is ==> -145.33430273020608

Episode * 75 * Avg Reward is ==> -145.32777005464345

Episode * 76 * Avg Reward is ==> -142.4835146046417

Episode * 77 * Avg Reward is ==> -139.59338840338395

Episode * 78 * Avg Reward is ==> -133.04552232142163

Episode * 79 * Avg Reward is ==> -132.93288588036899

Episode * 80 * Avg Reward is ==> -136.16012471382237

Episode * 81 * Avg Reward is ==> -139.21305348031393

Episode * 82 * Avg Reward is ==> -133.23691621529298

Episode * 83 * Avg Reward is ==> -135.92990594024982

Episode * 84 * Avg Reward is ==> -136.03027429930435

Episode * 85 * Avg Reward is ==> -135.97360824863455

Episode * 86 * Avg Reward is ==> -136.10527880830494

Episode * 87 * Avg Reward is ==> -139.05391439010512

Episode * 88 * Avg Reward is ==> -142.56133171606365

Episode * 89 * Avg Reward is ==> -161.33989090345662

Episode * 90 * Avg Reward is ==> -170.82788477632195

Episode * 91 * Avg Reward is ==> -170.8558841498521

Episode * 92 * Avg Reward is ==> -173.9910213401168

Episode * 93 * Avg Reward is ==> -176.87631595893498

Episode * 94 * Avg Reward is ==> -170.97863292694336

Episode * 95 * Avg Reward is ==> -173.88549953443538

Episode * 96 * Avg Reward is ==> -170.7028462286189

Episode * 97 * Avg Reward is ==> -173.47564018610032

Episode * 98 * Avg Reward is ==> -173.42104867150212

Episode * 99 * Avg Reward is ==> -173.2394285933109

```
</div>
    
![png](/img/examples/rl/ddpg_pendulum/ddpg_pendulum_16_100.png)
    


If training proceeds correctly, the average episodic reward will increase with time.

Feel free to try different learning rates, `tau` values, and architectures for the
Actor and Critic networks.

The Inverted Pendulum problem has low complexity, but DDPG work great on many other
problems.

Another great environment to try this on is `LunarLander-v2` continuous, but it will take
more episodes to obtain good results.


```python
# Save the weights
actor_model.save_weights("pendulum_actor.weights.h5")
critic_model.save_weights("pendulum_critic.weights.h5")

target_actor.save_weights("pendulum_target_actor.weights.h5")
target_critic.save_weights("pendulum_target_critic.weights.h5")
```

Before Training:

![before_img](https://i.imgur.com/ox6b9rC.gif)

After 100 episodes:

![after_img](https://i.imgur.com/eEH8Cz6.gif)

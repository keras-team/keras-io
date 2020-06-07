
# Deep Deterministic Policy Gradient (DDPG)

**Author:** [amifunny](https://github.com/amifunny)<br>
**Date created:** 2020/06/04<br>
**Last modified:** 2020/06/06<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/DDPG_Pendulum.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/rl/DDPG_Pendulum.py)


**Description:** Implementing DDPG algorithm on the Inverted Pendulum Problem.

---
## Introduction

**Deep Deterministic Policy Gradient (DDPG)** is a model-free off-policy algorithm for
learning continous actions.

It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network).
It uses Experience Replay and slow-learning target networks from DQN, and it is based on DPG,
which can operate over continuous action spaces.

This tutorial closely follow this paper -
[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)

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
as opposed to saying "I'm going to re-learn how to play this entire game after every move".
See this [StackOverflow answer](https://stackoverflow.com/a/54238556/13475679).

**Second, it uses Experience Replay.**

We store list of tuples `(state, action, reward, next_state)`, and instead of
learning only from recent experience, we learn from sampling all of our experience
accumulated so far.

Now, let's see how is it implemented.



```python
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

```

We use [OpenAIGym](http://gym.openai.com/docs) to create the environment.
We will use the `upper_bound` parameter to scale our actions later.



```python
problem = "Pendulum-v0"
env = gym.make(problem)

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
To implement better exploration by the Actor network, we use noisy perturbations, specifically
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
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
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

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
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
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
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


```

Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation. `BatchNormalization` is used to normalize dimensions across
samples in a mini-batch, as activations can vary a lot due to fluctuating values of input
state and action.

Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.



```python

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

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(512, activation="relu")(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(512, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


```

`policy()` returns an action sampled from our Actor network plus some noise for
exploration.



```python

def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
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

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

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

# Takes about 20 min to train
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(tau)

        # End this episode when `done` is True
        if done:
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
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

```

<div class="k-default-codeblock">
```
Episode * 0 * Avg Reward is ==> -1489.3890805419135
Episode * 1 * Avg Reward is ==> -1417.7353714901142
Episode * 2 * Avg Reward is ==> -1326.232044085693
Episode * 3 * Avg Reward is ==> -1384.9411195441705
Episode * 4 * Avg Reward is ==> -1383.8934864226756
Episode * 5 * Avg Reward is ==> -1400.1074693384555
Episode * 6 * Avg Reward is ==> -1394.6643095913626
Episode * 7 * Avg Reward is ==> -1407.3381411470502
Episode * 8 * Avg Reward is ==> -1363.6902698989882
Episode * 9 * Avg Reward is ==> -1319.4564710758827
Episode * 10 * Avg Reward is ==> -1295.43896943519
Episode * 11 * Avg Reward is ==> -1241.183380010504
Episode * 12 * Avg Reward is ==> -1165.8945722966746
Episode * 13 * Avg Reward is ==> -1110.3749473375963
Episode * 14 * Avg Reward is ==> -1062.13202147442
Episode * 15 * Avg Reward is ==> -1089.16425453295
Episode * 16 * Avg Reward is ==> -1041.028719048436
Episode * 17 * Avg Reward is ==> -990.1791232904058
Episode * 18 * Avg Reward is ==> -938.1971800288959
Episode * 19 * Avg Reward is ==> -897.4241796403319
Episode * 20 * Avg Reward is ==> -860.7338498861728
Episode * 21 * Avg Reward is ==> -821.743968467624
Episode * 22 * Avg Reward is ==> -797.0804306472012
Episode * 23 * Avg Reward is ==> -773.413449933685
Episode * 24 * Avg Reward is ==> -747.0553014914324
Episode * 25 * Avg Reward is ==> -735.8975663566166
Episode * 26 * Avg Reward is ==> -712.9330240620812
Episode * 27 * Avg Reward is ==> -696.4750684497031
Episode * 28 * Avg Reward is ==> -676.930087032212
Episode * 29 * Avg Reward is ==> -666.5135447779842
Episode * 30 * Avg Reward is ==> -649.0862358859673
Episode * 31 * Avg Reward is ==> -632.8041524119487
Episode * 32 * Avg Reward is ==> -617.3236460039853
Episode * 33 * Avg Reward is ==> -606.5602064058115
Episode * 34 * Avg Reward is ==> -595.7110275186086
Episode * 35 * Avg Reward is ==> -582.5483234107044
Episode * 36 * Avg Reward is ==> -570.1126630159052
Episode * 37 * Avg Reward is ==> -558.1729328102862
Episode * 38 * Avg Reward is ==> -550.1078542277539
Episode * 39 * Avg Reward is ==> -542.6100894367607
Episode * 40 * Avg Reward is ==> -511.2971618526708
Episode * 41 * Avg Reward is ==> -480.8394990945736
Episode * 42 * Avg Reward is ==> -461.10637824268025
Episode * 43 * Avg Reward is ==> -425.1001200191812
Episode * 44 * Avg Reward is ==> -393.7177551866972
Episode * 45 * Avg Reward is ==> -359.81608694332783
Episode * 46 * Avg Reward is ==> -333.1376942262487
Episode * 47 * Avg Reward is ==> -298.9184906265776
Episode * 48 * Avg Reward is ==> -276.84274428651634
Episode * 49 * Avg Reward is ==> -259.8818019404149
Episode * 50 * Avg Reward is ==> -242.35059855811218
Episode * 51 * Avg Reward is ==> -229.49790356353964
Episode * 52 * Avg Reward is ==> -229.10175658330922
Episode * 53 * Avg Reward is ==> -219.59934262185433
Episode * 54 * Avg Reward is ==> -215.96203603861682
Episode * 55 * Avg Reward is ==> -178.7932605346407
Episode * 56 * Avg Reward is ==> -175.27864481339498
Episode * 57 * Avg Reward is ==> -172.2955474461013
Episode * 58 * Avg Reward is ==> -175.50410988912722
Episode * 59 * Avg Reward is ==> -181.75766787368448
Episode * 60 * Avg Reward is ==> -187.91363761574002
Episode * 61 * Avg Reward is ==> -191.08368645636907
Episode * 62 * Avg Reward is ==> -193.79766082744766
Episode * 63 * Avg Reward is ==> -191.24435658811157
Episode * 64 * Avg Reward is ==> -191.5774382205419
Episode * 65 * Avg Reward is ==> -183.47534931288618
Episode * 66 * Avg Reward is ==> -183.81712139312694
Episode * 67 * Avg Reward is ==> -180.7524703112453
Episode * 68 * Avg Reward is ==> -180.8500651651906
Episode * 69 * Avg Reward is ==> -171.82990930347563
Episode * 70 * Avg Reward is ==> -171.69721630092738
Episode * 71 * Avg Reward is ==> -174.88614542976734
Episode * 72 * Avg Reward is ==> -174.75749187961858
Episode * 73 * Avg Reward is ==> -177.17381725276474
Episode * 74 * Avg Reward is ==> -174.5311685490747
Episode * 75 * Avg Reward is ==> -174.7197747295589
Episode * 76 * Avg Reward is ==> -174.72520621618378
Episode * 77 * Avg Reward is ==> -171.90803101307432
Episode * 78 * Avg Reward is ==> -172.03606636256777
Episode * 79 * Avg Reward is ==> -174.29270674144328
Episode * 80 * Avg Reward is ==> -174.2570256166555
Episode * 81 * Avg Reward is ==> -174.0338296337281
Episode * 82 * Avg Reward is ==> -168.19068909791156
Episode * 83 * Avg Reward is ==> -171.21378744883273
Episode * 84 * Avg Reward is ==> -168.16867412482776
Episode * 85 * Avg Reward is ==> -168.2053585328696
Episode * 86 * Avg Reward is ==> -163.8153408273806
Episode * 87 * Avg Reward is ==> -163.87735389565233
Episode * 88 * Avg Reward is ==> -163.8430382545905
Episode * 89 * Avg Reward is ==> -163.75861189697486
Episode * 90 * Avg Reward is ==> -160.9971396666046
Episode * 91 * Avg Reward is ==> -161.10154608360122
Episode * 92 * Avg Reward is ==> -158.10297051288825
Episode * 93 * Avg Reward is ==> -158.06640770665825
Episode * 94 * Avg Reward is ==> -155.02585690745173
Episode * 95 * Avg Reward is ==> -163.97503824209235
Episode * 96 * Avg Reward is ==> -166.80937070106452
Episode * 97 * Avg Reward is ==> -175.6895355138373
Episode * 98 * Avg Reward is ==> -175.37352764579805
Episode * 99 * Avg Reward is ==> -169.19861107775358

```
</div>
![png](/img/examples/rl/DDPG_Pendulum/DDPG_Pendulum_16_1.png)


![Graph](https://i.imgur.com/sqEtM6M.png)


If training proceeds correctly, the average episodic reward will increase with time.

Feel free to try different learning rates, `tau` values, and architectures for the
Actor and Critic networks.

The Inverted Pendulum problem has low complexity, but DDPG work great on many other
problems.

Another great environment to try this on is `LunarLandingContinuous-v2`, but it will take
more episodes to obtain good results.



```python
# Save the weights
actor_model.save_weights("pendulum_actor.h5")
critic_model.save_weights("pendulum_critic.h5")

target_actor.save_weights("pendulum_target_actor.h5")
target_critic.save_weights("pendulum_target_critic.h5")

```

Before Training:

![before_img](https://i.imgur.com/ox6b9rC.gif)


After 100 episodes:

![after_img](https://i.imgur.com/eEH8Cz6.gif)


# Deep Deterministic Policy Gradient (DDPG)

**Author:** [amifunny](https://github.com/amifunny)<br>
**Date created:** 2020/06/04<br>
**Last modified:** 2020/06/06<br>
**Description:** Implementing DDPG algorithm on the Inverted Pendulum Problem.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py)



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

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
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

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

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

# Takes about 4 min to train
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
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

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
Episode * 0 * Avg Reward is ==> -1371.4181638420068
Episode * 1 * Avg Reward is ==> -1314.0612898183517
Episode * 2 * Avg Reward is ==> -1464.2192688746993
Episode * 3 * Avg Reward is ==> -1516.8312248770326
Episode * 4 * Avg Reward is ==> -1517.864549040632
Episode * 5 * Avg Reward is ==> -1497.7723609189197
Episode * 6 * Avg Reward is ==> -1492.8937590729608
Episode * 7 * Avg Reward is ==> -1454.9366428708543
Episode * 8 * Avg Reward is ==> -1399.5522160364344
Episode * 9 * Avg Reward is ==> -1354.202457744552
Episode * 10 * Avg Reward is ==> -1367.8548297710145
Episode * 11 * Avg Reward is ==> -1287.3515898493968
Episode * 12 * Avg Reward is ==> -1198.1809904485956
Episode * 13 * Avg Reward is ==> -1220.4155521104058
Episode * 14 * Avg Reward is ==> -1173.9341997718216
Episode * 15 * Avg Reward is ==> -1100.9277411795692
Episode * 16 * Avg Reward is ==> -1124.981105745213
Episode * 17 * Avg Reward is ==> -1077.1206408913347
Episode * 18 * Avg Reward is ==> -1027.0408700398395
Episode * 19 * Avg Reward is ==> -975.8426997684504
Episode * 20 * Avg Reward is ==> -959.1902798752656
Episode * 21 * Avg Reward is ==> -927.1840263346446
Episode * 22 * Avg Reward is ==> -892.612266942912
Episode * 23 * Avg Reward is ==> -855.508060041338
Episode * 24 * Avg Reward is ==> -821.3499130754724
Episode * 25 * Avg Reward is ==> -794.7915032657822
Episode * 26 * Avg Reward is ==> -770.0461336160848
Episode * 27 * Avg Reward is ==> -746.5915976220238
Episode * 28 * Avg Reward is ==> -724.882774481543
Episode * 29 * Avg Reward is ==> -708.8097210473583
Episode * 30 * Avg Reward is ==> -693.6540802546614
Episode * 31 * Avg Reward is ==> -680.1082065155553
Episode * 32 * Avg Reward is ==> -663.1525119451389
Episode * 33 * Avg Reward is ==> -647.3577482477368
Episode * 34 * Avg Reward is ==> -632.2227763645933
Episode * 35 * Avg Reward is ==> -621.1470618152719
Episode * 36 * Avg Reward is ==> -610.8763131480948
Episode * 37 * Avg Reward is ==> -601.1080786098471
Episode * 38 * Avg Reward is ==> -588.8937956573642
Episode * 39 * Avg Reward is ==> -577.1370759573672
Episode * 40 * Avg Reward is ==> -546.1016330297531
Episode * 41 * Avg Reward is ==> -517.8129711884756
Episode * 42 * Avg Reward is ==> -476.7372191003733
Episode * 43 * Avg Reward is ==> -434.90297062442903
Episode * 44 * Avg Reward is ==> -399.9924055046493
Episode * 45 * Avg Reward is ==> -368.34116621789343
Episode * 46 * Avg Reward is ==> -334.7988665631502
Episode * 47 * Avg Reward is ==> -308.35234111467645
Episode * 48 * Avg Reward is ==> -290.42965322776564
Episode * 49 * Avg Reward is ==> -269.91445568123055
Episode * 50 * Avg Reward is ==> -235.48119514323594
Episode * 51 * Avg Reward is ==> -234.9780305270961
Episode * 52 * Avg Reward is ==> -234.68082375928253
Episode * 53 * Avg Reward is ==> -206.0439960595881
Episode * 54 * Avg Reward is ==> -198.84998610796242
Episode * 55 * Avg Reward is ==> -204.5921590378192
Episode * 56 * Avg Reward is ==> -169.77658707644076
Episode * 57 * Avg Reward is ==> -166.3741301266993
Episode * 58 * Avg Reward is ==> -163.27524402874297
Episode * 59 * Avg Reward is ==> -163.26578255995543
Episode * 60 * Avg Reward is ==> -150.83917288056216
Episode * 61 * Avg Reward is ==> -150.33692281389943
Episode * 62 * Avg Reward is ==> -155.98114100460194
Episode * 63 * Avg Reward is ==> -164.28751452953804
Episode * 64 * Avg Reward is ==> -183.49980092017807
Episode * 65 * Avg Reward is ==> -186.08335109594302
Episode * 66 * Avg Reward is ==> -188.90766269760917
Episode * 67 * Avg Reward is ==> -186.135142789066
Episode * 68 * Avg Reward is ==> -186.33785759561263
Episode * 69 * Avg Reward is ==> -183.43136225582504
Episode * 70 * Avg Reward is ==> -183.41186926659506
Episode * 71 * Avg Reward is ==> -182.60613280355884
Episode * 72 * Avg Reward is ==> -182.7037383640803
Episode * 73 * Avg Reward is ==> -185.4013115481893
Episode * 74 * Avg Reward is ==> -185.42531596154873
Episode * 75 * Avg Reward is ==> -185.48309246275477
Episode * 76 * Avg Reward is ==> -182.4696934013826
Episode * 77 * Avg Reward is ==> -179.469902168214
Episode * 78 * Avg Reward is ==> -179.4397569222765
Episode * 79 * Avg Reward is ==> -179.7321282886106
Episode * 80 * Avg Reward is ==> -182.59359329236494
Episode * 81 * Avg Reward is ==> -182.51806873633717
Episode * 82 * Avg Reward is ==> -185.26610147945834
Episode * 83 * Avg Reward is ==> -188.2264040879636
Episode * 84 * Avg Reward is ==> -188.3609930226932
Episode * 85 * Avg Reward is ==> -188.13718541441744
Episode * 86 * Avg Reward is ==> -188.28316131455077
Episode * 87 * Avg Reward is ==> -185.04436608010914
Episode * 88 * Avg Reward is ==> -181.98169990248874
Episode * 89 * Avg Reward is ==> -181.83018056026938
Episode * 90 * Avg Reward is ==> -178.69695581997735
Episode * 91 * Avg Reward is ==> -172.17918175775264
Episode * 92 * Avg Reward is ==> -172.23729180238666
Episode * 93 * Avg Reward is ==> -169.35093435401376
Episode * 94 * Avg Reward is ==> -166.613037321441
Episode * 95 * Avg Reward is ==> -166.65996513691647
Episode * 96 * Avg Reward is ==> -166.9107514751376
Episode * 97 * Avg Reward is ==> -163.78820050048745
Episode * 98 * Avg Reward is ==> -163.78511089269722
Episode * 99 * Avg Reward is ==> -169.71227626035832

```
</div>
    
![png](/img/examples/rl/ddpg_pendulum/ddpg_pendulum_16_1.png)
    


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

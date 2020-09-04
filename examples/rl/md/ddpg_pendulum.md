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
Episode * 0 * Avg Reward is ==> -1649.7749136222078
Episode * 1 * Avg Reward is ==> -1517.4486184149769
Episode * 2 * Avg Reward is ==> -1531.5183006979476
Episode * 3 * Avg Reward is ==> -1506.074787052287
Episode * 4 * Avg Reward is ==> -1504.6124802159331
Episode * 5 * Avg Reward is ==> -1503.1880302718955
Episode * 6 * Avg Reward is ==> -1483.5610245534192
Episode * 7 * Avg Reward is ==> -1452.5994030901663
Episode * 8 * Avg Reward is ==> -1413.587287561734
Episode * 9 * Avg Reward is ==> -1393.8817512718601
Episode * 10 * Avg Reward is ==> -1410.349519023605
Episode * 11 * Avg Reward is ==> -1358.854013961542
Episode * 12 * Avg Reward is ==> -1380.754239815423
Episode * 13 * Avg Reward is ==> -1321.0171154888974
Episode * 14 * Avg Reward is ==> -1335.969721110595
Episode * 15 * Avg Reward is ==> -1284.3939065233806
Episode * 16 * Avg Reward is ==> -1232.349914217674
Episode * 17 * Avg Reward is ==> -1239.743740482351
Episode * 18 * Avg Reward is ==> -1253.9743119057596
Episode * 19 * Avg Reward is ==> -1230.9121098540822
Episode * 20 * Avg Reward is ==> -1190.9498301655674
Episode * 21 * Avg Reward is ==> -1142.757838899654
Episode * 22 * Avg Reward is ==> -1093.1678664448343
Episode * 23 * Avg Reward is ==> -1052.9458864153005
Episode * 24 * Avg Reward is ==> -1021.1097011419374
Episode * 25 * Avg Reward is ==> -986.6825261932893
Episode * 26 * Avg Reward is ==> -954.5402236501969
Episode * 27 * Avg Reward is ==> -925.0348865931173
Episode * 28 * Avg Reward is ==> -897.4755330588606
Episode * 29 * Avg Reward is ==> -871.7011232982851
Episode * 30 * Avg Reward is ==> -851.5790517288391
Episode * 31 * Avg Reward is ==> -828.946333400579
Episode * 32 * Avg Reward is ==> -807.3604217875239
Episode * 33 * Avg Reward is ==> -790.4487656493948
Episode * 34 * Avg Reward is ==> -771.4249213043466
Episode * 35 * Avg Reward is ==> -750.1303874575156
Episode * 36 * Avg Reward is ==> -738.2276258453605
Episode * 37 * Avg Reward is ==> -722.0546722693595
Episode * 38 * Avg Reward is ==> -709.4768424005381
Episode * 39 * Avg Reward is ==> -691.8656751237129
Episode * 40 * Avg Reward is ==> -653.6442340446984
Episode * 41 * Avg Reward is ==> -625.1913416053745
Episode * 42 * Avg Reward is ==> -589.3207093465375
Episode * 43 * Avg Reward is ==> -556.4814070749678
Episode * 44 * Avg Reward is ==> -528.6240783303273
Episode * 45 * Avg Reward is ==> -500.64554392182106
Episode * 46 * Avg Reward is ==> -472.3086153334718
Episode * 47 * Avg Reward is ==> -441.48035712393704
Episode * 48 * Avg Reward is ==> -417.0529339923727
Episode * 49 * Avg Reward is ==> -389.6036227296889
Episode * 50 * Avg Reward is ==> -355.8528469187061
Episode * 51 * Avg Reward is ==> -336.08246667891524
Episode * 52 * Avg Reward is ==> -298.19811993533585
Episode * 53 * Avg Reward is ==> -290.69553241501654
Episode * 54 * Avg Reward is ==> -255.2182523956195
Episode * 55 * Avg Reward is ==> -248.5643067912606
Episode * 56 * Avg Reward is ==> -241.7973891355776
Episode * 57 * Avg Reward is ==> -210.82587994100126
Episode * 58 * Avg Reward is ==> -182.3478247995969
Episode * 59 * Avg Reward is ==> -168.89420457723676
Episode * 60 * Avg Reward is ==> -165.11645307026401
Episode * 61 * Avg Reward is ==> -164.98917005538596
Episode * 62 * Avg Reward is ==> -174.07929741173712
Episode * 63 * Avg Reward is ==> -170.9146870099085
Episode * 64 * Avg Reward is ==> -164.53580505104134
Episode * 65 * Avg Reward is ==> -164.51637155969624
Episode * 66 * Avg Reward is ==> -167.30951143288934
Episode * 67 * Avg Reward is ==> -167.37242062147388
Episode * 68 * Avg Reward is ==> -173.1032030183493
Episode * 69 * Avg Reward is ==> -175.9730148726038
Episode * 70 * Avg Reward is ==> -175.8748322010726
Episode * 71 * Avg Reward is ==> -178.90670223775666
Episode * 72 * Avg Reward is ==> -181.86622241941095
Episode * 73 * Avg Reward is ==> -182.1962236992579
Episode * 74 * Avg Reward is ==> -185.2912374892455
Episode * 75 * Avg Reward is ==> -194.4832442136879
Episode * 76 * Avg Reward is ==> -186.779488251999
Episode * 77 * Avg Reward is ==> -183.73585590806175
Episode * 78 * Avg Reward is ==> -181.2475970989047
Episode * 79 * Avg Reward is ==> -181.17939386667186
Episode * 80 * Avg Reward is ==> -181.42752243523327
Episode * 81 * Avg Reward is ==> -178.39111106351203
Episode * 82 * Avg Reward is ==> -178.18967199129696
Episode * 83 * Avg Reward is ==> -178.19995434343437
Episode * 84 * Avg Reward is ==> -171.58622544456907
Episode * 85 * Avg Reward is ==> -165.34379188265922
Episode * 86 * Avg Reward is ==> -165.818479166733
Episode * 87 * Avg Reward is ==> -168.77470830644293
Episode * 88 * Avg Reward is ==> -171.58145877011208
Episode * 89 * Avg Reward is ==> -168.7177824041847
Episode * 90 * Avg Reward is ==> -166.24676261409203
Episode * 91 * Avg Reward is ==> -169.45257585773433
Episode * 92 * Avg Reward is ==> -169.4489017594307
Episode * 93 * Avg Reward is ==> -163.40315682462477
Episode * 94 * Avg Reward is ==> -163.3605953522997
Episode * 95 * Avg Reward is ==> -163.2997607563818
Episode * 96 * Avg Reward is ==> -166.2461479056317
Episode * 97 * Avg Reward is ==> -163.1364838325657
Episode * 98 * Avg Reward is ==> -156.82625141684883
Episode * 99 * Avg Reward is ==> -150.5283724694878

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


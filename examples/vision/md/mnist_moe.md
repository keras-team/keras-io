# MoE for MNIST

**Author:** [Damoon Shahhosseini](https://www.linkedin.com/in/damoonsh/)<br>
**Date created:** 2015/06/19<br>
**Last modified:** 2020/04/21<br>
**Description:** Showcasing concepts relates to Mixture of Experts (MoE).


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/mnist_moe.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_moe.py)



# Introduction

In this example, we implement an adaptation of the Mixture of Experts (MoE) architecture
([Shazeer et al.](https://arxiv.org/abs/1701.06538)).
The idea is to use conditional computation to increases model capacity without increasing computation.
Experts are identical blocks within a layer where each are trained to specialize in different parts of the input space.
At each forward pass, a gating network selects a subset of experts to apply to the input.

The components to implement are:
- Gating network: A dense layer that outputs a probability distribution over the experts.
- MoE layer: A layer that applies a different expert to each input in the batch. And a loss function that ensures specialization among the experts.
- Model: A simple model that uses the MoE layer.

In this example, we will first implement a linear MoE layer and then a CNN-based MoE layer. Lastly we will combine the two using an abstract implementation to showcase its capacties.

---
## Imports


```python
import numpy as np
import keras
from keras import layers, models
import tensorflow as tf
from tensorflow.keras import backend as K
```

### Data Prepration


```python
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

<div class="k-default-codeblock">
```
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples

```
</div>
---
## Constants


```python
NUM_EXPERTS = 5
TOP_K = 3
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

```

---
## Base architecture

The most basic [MNIST classifier](https://keras.io/examples/vision/mnist_convnet/) consists of a stack of convolutional layers followed by a dense layer. In this tutorial, we will first replace the dense layer with a MoE layer. Then do the same for convolutional layers.


```python
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">26</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │           <span style="color: #00af00; text-decoration-color: #00af00">320</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">11</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │        <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)       │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1600</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1600</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">16,010</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,826</span> (136.04 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,826</span> (136.04 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



# Linear MoE using Dense layers

For this layer, we will create multiple dense layers that will be used as experts. Then a simple gating network will select at each step which exerts should be utilized for the current input. We will keep track of the number of times each expert is used. Then the selected experts will be combined using a weighted sum.


```python

class LinearMoE(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    ):
        super(LinearMoE, self).__init__()

        # Initialize experts
        self.experts = [
            layers.Dense(
                hidden_size,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.001
                ),
                bias_initializer="zeros",
            )
            for _ in range(num_experts)
        ]
        # Initialize gating network
        self.gating_network = layers.Dense(
            NUM_EXPERTS,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.001
            ),
            bias_initializer="zeros",
        )

        self.num_experts = num_experts
        self.top_k = top_k
        # Keep track of how many times each expert is used
        self.expert_usage_count = tf.Variable(
            tf.zeros((num_experts,), dtype=tf.float32)
        )

    def call(self, x):
        # Get gating weights
        gating_weights = self.gating_network(x)

        # Get the top k experts based on the gating weights
        top_k_weights, top_k_indices = tf.math.top_k(gating_weights, k=self.top_k)

        # Count usage of each expert symbolically
        updates = tf.ones_like(tf.reshape(top_k_indices, [-1]), dtype=tf.float32)
        # Use tf.tensor_scatter_nd_add to increment the usage count
        self.expert_usage_count.assign(
            tf.tensor_scatter_nd_add(
                self.expert_usage_count, tf.reshape(top_k_indices, [-1, 1]), updates
            )
        )

        # Get outputs from only the top-k experts
        top_k_expert_outputs = tf.stack(
            [
                self.experts[expert_index](x)
                for expert_index in top_k_indices.numpy()[0]
            ],
            axis=1,
        )  # Stack outputs along axis 1

        # Combine outputs using top-k weights
        combined_output = tf.einsum("ijk,ij->ik", top_k_expert_outputs, top_k_weights)

        return combined_output

```

Output of the top 3 experts out of 10 for one layer of MoE:


```python
sample_data = tf.random.uniform((1, 10))
linear_mode = LinearMoE(32, 10, 3)
linear_mode(sample_data)
```




<div class="k-default-codeblock">
```
<tf.Tensor: shape=(1, 32), dtype=float32, numpy=
array([[ 7.8205157e-06, -9.6895346e-06, -1.3567289e-06, -1.1132732e-05,
        -2.8532945e-06,  2.5079655e-06, -1.9245979e-06, -1.1963858e-06,
         7.0713571e-07, -6.1005521e-06, -1.4404541e-06,  8.7282970e-06,
        -1.2715462e-05, -6.8048112e-06, -1.4842963e-06, -1.1498453e-06,
        -1.2819509e-06,  2.1932642e-06,  9.3587869e-06, -2.8901491e-07,
         4.9251821e-06,  6.3183566e-06,  4.5951542e-06,  2.1709191e-06,
        -9.7254415e-06, -9.1735465e-06, -1.9290899e-06,  1.2122581e-07,
         4.6307746e-06,  1.4301847e-06, -8.8591560e-06, -1.0252905e-06]],
      dtype=float32)>

```
</div>
---
## Routing Collapse

Routing collapse is a problem that occurs with MoE layers. The route terminology refers to the selection process of which expert to use for a given input.

Route collapse happens when a routing model, early in training, starts favoring just a few experts because they perform slightly better due to random starting conditions. This leads to most examples being sent to these experts, leaving others unused and reducing the model’s overall capacity.

Code below demonstrates the randomness of expert selection:


```python

def check_expert_usage(runs):
    # Running the later multiple times to show randomness of expert selection
    for i in range(runs):
        sample_data = tf.random.uniform((1, 10))
        linear_mode = LinearMoE(10, 5)
        _ = linear_mode(sample_data)
        print(f"Run {i}, Expert usage: {linear_mode.expert_usage_count.numpy()}")


check_expert_usage(4)
```

<div class="k-default-codeblock">
```
Run 0, Expert usage: [1. 0. 1. 1. 0.]
Run 1, Expert usage: [0. 1. 1. 0. 1.]
Run 2, Expert usage: [1. 1. 0. 1. 0.]
Run 3, Expert usage: [1. 0. 1. 1. 0.]

```
</div>
### Adding loss functions to prevent route collapse
To fix this, the authors use extra rules (importance and load losses), ideas borrowed from [Shazeer et al.](https://arxiv.org/abs/1701.06538), to ensure all experts get used evenly.

The importance_loss calculates how much the usage of each expert (tracked in batch_importance_sum) deviates from the average usage (mean_importance) by using mean squared error, aiming to balance expert utilization. This helps prevent route collapse by discouraging the model from overloading a few experts, instead promoting an even distribution of examples across all experts to maintain diverse and effective routing.

#### Load losses:
    - Diversity loss: Diversity loss helps prevent route collapse by encouraging the routing model to evenly distribute examples across all experts, rather than favoring just a few due to their initial performance. It does this by maximizing the entropy of the gating weights, ensuring balanced expert utilization and improving the model's overall capacity.
    - Overflow loss: The batch_overflow_sum measures how much the usage of experts exceeds a set capacity by applying ReLU to the difference between usage_counts (how many examples each expert handles) and batch_capacity (the allowed limit), then summing the excesses. This helps prevent route collapse by penalizing situations where certain experts are overused, encouraging a more even spread of examples across all experts to keep the model's capacity balanced.


```python

class LinearMoE(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    ):
        super(LinearMoE, self).__init__()

        # Initialize experts
        self.experts = [
            layers.Dense(
                hidden_size,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=0.001
                ),
                bias_initializer="zeros",
            )
            for _ in range(num_experts)
        ]
        # Initialize gating network
        self.gating_network = layers.Dense(
            num_experts,  # Match output to num_experts
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.001
            ),
            bias_initializer="zeros",
        )

        self.num_experts = num_experts
        self.top_k = top_k
        # Keep track of how many times each expert is used as a layer weight
        self.expert_usage_count = tf.Variable(
            tf.zeros((num_experts,), dtype=tf.float32)
        )

        self.batch_capacity = BATCH_SIZE // num_experts

    def _diversity_loss(self, weights):
        entropy = -K.sum(weights * K.log(weights + 1e-10), axis=1)
        self.diversity_loss = -K.mean(entropy)

    def _importance_loss(self, gating_weights):
        batch_importance_sum = K.sum(gating_weights, axis=0)
        mean_importance = K.mean(batch_importance_sum)
        self.importance_loss = K.mean(
            K.square(
                batch_importance_sum
                - mean_importance * tf.ones_like(batch_importance_sum)
            )
        )

    def call(self, x):
        # Get gating weights and normalize
        gating_weights = self.gating_network(x)
        gating_weights = K.softmax(gating_weights)  # Ensure weights are probabilities
        self._diversity_loss(gating_weights)
        self._importance_loss(gating_weights)

        # Get the top k experts based on the gating weights
        top_k_weights, top_k_indices = tf.math.top_k(gating_weights, k=self.top_k)

        # Count usage of each expert symbolically
        updates = tf.ones_like(tf.reshape(top_k_indices, [-1]), dtype=tf.float32)
        # Use tf.tensor_scatter_nd_add to increment the usage count
        self.expert_usage_count.assign(
            tf.tensor_scatter_nd_add(
                self.expert_usage_count, tf.reshape(top_k_indices, [-1, 1]), updates
            )
        )

        # Calculate overflow using updated usage count
        self.batch_overflow_sum = K.sum(
            K.relu(tf.convert_to_tensor(self.expert_usage_count) - self.batch_capacity)
        )

        # Compute all expert outputs
        expert_outputs = tf.stack(
            [expert(x) for expert in self.experts], axis=1
        )  # Shape: (batch_size, num_experts, hidden_size)

        # Gather the top-k expert outputs using top_k_indices
        batch_size = tf.shape(x)[0]
        batch_indices = tf.expand_dims(
            tf.range(batch_size), 1
        )  # Shape: (batch_size, 1)
        batch_indices = tf.tile(
            batch_indices, [1, self.top_k]
        )  # Shape: (batch_size, top_k)

        # Create indices for gathering
        indices = tf.stack(
            [batch_indices, top_k_indices], axis=2
        )  # Shape: (batch_size, top_k, 2)
        top_k_expert_outputs = tf.gather_nd(
            expert_outputs, indices
        )  # Shape: (batch_size, top_k, hidden_size)

        # Combine outputs using top-k weights
        combined_output = tf.reduce_sum(
            top_k_expert_outputs * tf.expand_dims(top_k_weights, axis=-1), axis=1
        )

        return combined_output

    def compute_total_loss(self, load_balance_coef=0.01):
        return load_balance_coef * (
            self.diversity_loss + self.batch_overflow_sum + self.importance_loss
        )

```

---
## MNIST classification with MoE


```python

class MoEModel(keras.Model):
    def __init__(self, input_shape, num_classes, num_experts=NUM_EXPERTS, top_k=TOP_K):
        super(MoEModel, self).__init__()

        # Define the convolutional block
        self.conv_block = keras.Sequential(
            [
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
            ]
        )

        # MoE classifier
        self.moe_classifier = LinearMoE(
            hidden_size=num_classes, num_experts=num_experts, top_k=top_k
        )

        # Softmax layer
        self.softmax = layers.Softmax()

    def call(self, inputs, training=False):
        conv_flatten = self.conv_block(inputs)
        moe_output = self.moe_classifier(conv_flatten)
        outputs = self.softmax(moe_output)
        return outputs

    def train_step(self, data):
        x, y = data  # Unpack input data and labels

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            classification_loss = self.compute_loss(x, y, y_pred)
            moe_loss = self.moe_classifier.compute_total_loss(load_balance_coef=0.01)
            total_loss = classification_loss + moe_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )  # Update metrics (e.g., accuracy)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict of metrics for monitoring
        return {
            "loss": total_loss,
            "moe_loss": moe_loss,
            **{m.name: m.result() for m in self.metrics},
        }

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        classification_loss = self.compute_loss(x, y, y_pred)
        moe_loss = self.moe_classifier.compute_total_loss(load_balance_coef=0.01)
        total_loss = classification_loss + moe_loss

        self.compiled_metrics.update_state(y, y_pred)
        return {
            "loss": total_loss,
            "moe_loss": moe_loss,
            **{m.name: m.result() for m in self.metrics},
        }


# Instantiate and compile the model
inputs = keras.Input(shape=input_shape)
model = MoEModel(
    input_shape=input_shape, num_classes=num_classes, num_experts=6, top_k=4
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.CategoricalCrossentropy(),  # Assumes one-hot encoded labels
    metrics=["accuracy"],
)
```

###  Training


```python
history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(x_test, y_test),
)
```

<div class="k-default-codeblock">
```
Epoch 1/20

/opt/homebrew/Caskroom/miniforge/base/envs/keras-io/lib/python3.11/site-packages/keras/src/backend/tensorflow/trainer.py:642: UserWarning: `model.compiled_metrics()` is deprecated. Instead, use e.g.:
```
for metric in self.metrics:
    metric.update_state(y, y_pred)
```
```
</div>
    
<div class="k-default-codeblock">
```
  return self._compiled_metrics_update_state(

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  8:21 1s/step - accuracy: 0.1406 - loss: 0.1000 - moe_loss: 3.8421

<div class="k-default-codeblock">
```

```
</div>
   4/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 17ms/step - accuracy: 0.1637 - loss: 0.1000 - moe_loss: 11.5298

<div class="k-default-codeblock">
```

```
</div>
   8/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.2071 - loss: 0.1000 - moe_loss: 21.7700

<div class="k-default-codeblock">
```

```
</div>
  12/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.2438 - loss: 0.1000 - moe_loss: 32.0082

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.2744 - loss: 0.1000 - moe_loss: 42.2476

<div class="k-default-codeblock">
```

```
</div>
  20/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.2988 - loss: 0.1000 - moe_loss: 52.4873

<div class="k-default-codeblock">
```

```
</div>
  24/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.3206 - loss: 0.1000 - moe_loss: 62.7278

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.3409 - loss: 0.1000 - moe_loss: 72.9716

<div class="k-default-codeblock">
```

```
</div>
  32/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.3598 - loss: 0.1000 - moe_loss: 83.2221

<div class="k-default-codeblock">
```

```
</div>
  36/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.3774 - loss: 0.1000 - moe_loss: 93.4818

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.3938 - loss: 0.1000 - moe_loss: 103.7338

<div class="k-default-codeblock">
```

```
</div>
  44/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.4092 - loss: 0.1000 - moe_loss: 113.9789

<div class="k-default-codeblock">
```

```
</div>
  48/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.4237 - loss: 0.1000 - moe_loss: 124.2205

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.4373 - loss: 0.1000 - moe_loss: 134.4638

<div class="k-default-codeblock">
```

```
</div>
  56/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.4503 - loss: 0.1000 - moe_loss: 144.7069

<div class="k-default-codeblock">
```

```
</div>
  60/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.4626 - loss: 0.1000 - moe_loss: 154.9452

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.4744 - loss: 0.1000 - moe_loss: 165.1867

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.4828 - loss: 0.1000 - moe_loss: 172.8672

<div class="k-default-codeblock">
```

```
</div>
  71/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.4937 - loss: 0.1000 - moe_loss: 183.1078

<div class="k-default-codeblock">
```

```
</div>
  75/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.5039 - loss: 0.1000 - moe_loss: 193.3479

<div class="k-default-codeblock">
```

```
</div>
  79/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.5137 - loss: 0.1000 - moe_loss: 203.5896

<div class="k-default-codeblock">
```

```
</div>
  82/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.5207 - loss: 0.1000 - moe_loss: 211.2682

<div class="k-default-codeblock">
```

```
</div>
  86/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.5297 - loss: 0.1000 - moe_loss: 221.5096

<div class="k-default-codeblock">
```

```
</div>
  89/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.5362 - loss: 0.1000 - moe_loss: 229.1885

<div class="k-default-codeblock">
```

```
</div>
  93/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.5445 - loss: 0.1000 - moe_loss: 239.4279

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.5524 - loss: 0.1000 - moe_loss: 249.6689

<div class="k-default-codeblock">
```

```
</div>
 100/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.5581 - loss: 0.1000 - moe_loss: 257.3486

<div class="k-default-codeblock">
```

```
</div>
 104/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.5654 - loss: 0.1000 - moe_loss: 267.5893

<div class="k-default-codeblock">
```

```
</div>
 107/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.5707 - loss: 0.1000 - moe_loss: 275.2696

<div class="k-default-codeblock">
```

```
</div>
 110/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.5758 - loss: 0.1000 - moe_loss: 282.9488

<div class="k-default-codeblock">
```

```
</div>
 113/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.5808 - loss: 0.1000 - moe_loss: 290.6284

<div class="k-default-codeblock">
```

```
</div>
 116/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.5856 - loss: 0.1000 - moe_loss: 298.3105

<div class="k-default-codeblock">
```

```
</div>
 119/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.5903 - loss: 0.1000 - moe_loss: 305.9899

<div class="k-default-codeblock">
```

```
</div>
 123/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.5963 - loss: 0.1000 - moe_loss: 316.2303

<div class="k-default-codeblock">
```

```
</div>
 126/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.6007 - loss: 0.1000 - moe_loss: 323.9091

<div class="k-default-codeblock">
```

```
</div>
 129/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.6049 - loss: 0.1000 - moe_loss: 331.5898

<div class="k-default-codeblock">
```

```
</div>
 133/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.6104 - loss: 0.1000 - moe_loss: 341.8302

<div class="k-default-codeblock">
```

```
</div>
 136/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.6143 - loss: 0.1000 - moe_loss: 349.5105

<div class="k-default-codeblock">
```

```
</div>
 140/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.6195 - loss: 0.1000 - moe_loss: 359.7512

<div class="k-default-codeblock">
```

```
</div>
 143/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.6232 - loss: 0.1000 - moe_loss: 367.4311

<div class="k-default-codeblock">
```

```
</div>
 147/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.6281 - loss: 0.1000 - moe_loss: 377.6704

<div class="k-default-codeblock">
```

```
</div>
 150/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.6316 - loss: 0.1000 - moe_loss: 385.3510

<div class="k-default-codeblock">
```

```
</div>
 154/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.6361 - loss: 0.1000 - moe_loss: 395.5921

<div class="k-default-codeblock">
```

```
</div>
 157/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.6394 - loss: 0.1000 - moe_loss: 403.2722

<div class="k-default-codeblock">
```

```
</div>
 160/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.6427 - loss: 0.1000 - moe_loss: 410.9522

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.6458 - loss: 0.1000 - moe_loss: 418.6319

<div class="k-default-codeblock">
```

```
</div>
 167/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6499 - loss: 0.1000 - moe_loss: 428.8718

<div class="k-default-codeblock">
```

```
</div>
 171/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6539 - loss: 0.1000 - moe_loss: 439.1111

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6578 - loss: 0.1000 - moe_loss: 449.3512

<div class="k-default-codeblock">
```

```
</div>
 179/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6615 - loss: 0.1000 - moe_loss: 459.5908

<div class="k-default-codeblock">
```

```
</div>
 182/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6643 - loss: 0.1000 - moe_loss: 467.2707

<div class="k-default-codeblock">
```

```
</div>
 186/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6678 - loss: 0.1000 - moe_loss: 477.5105

<div class="k-default-codeblock">
```

```
</div>
 190/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6713 - loss: 0.1000 - moe_loss: 487.7509

<div class="k-default-codeblock">
```

```
</div>
 194/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6747 - loss: 0.1000 - moe_loss: 497.9902

<div class="k-default-codeblock">
```

```
</div>
 197/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6771 - loss: 0.1000 - moe_loss: 505.6700

<div class="k-default-codeblock">
```

```
</div>
 201/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6804 - loss: 0.1000 - moe_loss: 515.9094

<div class="k-default-codeblock">
```

```
</div>
 204/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6827 - loss: 0.1000 - moe_loss: 523.5893

<div class="k-default-codeblock">
```

```
</div>
 207/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6850 - loss: 0.1000 - moe_loss: 531.2690

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6880 - loss: 0.1000 - moe_loss: 541.5093

<div class="k-default-codeblock">
```

```
</div>
 215/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6910 - loss: 0.1000 - moe_loss: 551.7495

<div class="k-default-codeblock">
```

```
</div>
 219/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6938 - loss: 0.1000 - moe_loss: 561.9893

<div class="k-default-codeblock">
```

```
</div>
 222/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6959 - loss: 0.1000 - moe_loss: 569.6691

<div class="k-default-codeblock">
```

```
</div>
 225/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.6980 - loss: 0.1000 - moe_loss: 577.3488

<div class="k-default-codeblock">
```

```
</div>
 229/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 17ms/step - accuracy: 0.7007 - loss: 0.1000 - moe_loss: 587.5882

<div class="k-default-codeblock">
```

```
</div>
 233/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 17ms/step - accuracy: 0.7033 - loss: 0.1000 - moe_loss: 597.8292

<div class="k-default-codeblock">
```

```
</div>
 237/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.7059 - loss: 0.1000 - moe_loss: 608.0698

<div class="k-default-codeblock">
```

```
</div>
 241/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.7084 - loss: 0.1000 - moe_loss: 618.3093

<div class="k-default-codeblock">
```

```
</div>
 244/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.7102 - loss: 0.1000 - moe_loss: 625.9894

<div class="k-default-codeblock">
```

```
</div>
 247/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.7121 - loss: 0.1000 - moe_loss: 633.6691

<div class="k-default-codeblock">
```

```
</div>
 251/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.7144 - loss: 0.1000 - moe_loss: 643.9095

<div class="k-default-codeblock">
```

```
</div>
 255/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.7167 - loss: 0.1000 - moe_loss: 654.1490

<div class="k-default-codeblock">
```

```
</div>
 258/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.7184 - loss: 0.1000 - moe_loss: 661.8292

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.7207 - loss: 0.1000 - moe_loss: 672.0692

<div class="k-default-codeblock">
```

```
</div>
 265/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.7223 - loss: 0.1000 - moe_loss: 679.7494

<div class="k-default-codeblock">
```

```
</div>
 269/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.7245 - loss: 0.1000 - moe_loss: 689.9895

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.7266 - loss: 0.1000 - moe_loss: 700.2294

<div class="k-default-codeblock">
```

```
</div>
 277/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.7286 - loss: 0.1000 - moe_loss: 710.4691

<div class="k-default-codeblock">
```

```
</div>
 281/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.7306 - loss: 0.1000 - moe_loss: 720.7094

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.7326 - loss: 0.1000 - moe_loss: 730.9502

<div class="k-default-codeblock">
```

```
</div>
 289/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.7346 - loss: 0.1000 - moe_loss: 741.1905

<div class="k-default-codeblock">
```

```
</div>
 293/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.7365 - loss: 0.1000 - moe_loss: 751.4304

<div class="k-default-codeblock">
```

```
</div>
 295/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.7374 - loss: 0.1000 - moe_loss: 756.5504

<div class="k-default-codeblock">
```

```
</div>
 298/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.7388 - loss: 0.1000 - moe_loss: 764.2300

<div class="k-default-codeblock">
```

```
</div>
 302/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.7406 - loss: 0.1000 - moe_loss: 774.4699

<div class="k-default-codeblock">
```

```
</div>
 306/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.7424 - loss: 0.1000 - moe_loss: 784.7094

<div class="k-default-codeblock">
```

```
</div>
 310/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.7441 - loss: 0.1000 - moe_loss: 794.9492

<div class="k-default-codeblock">
```

```
</div>
 314/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.7458 - loss: 0.1000 - moe_loss: 805.1893

<div class="k-default-codeblock">
```

```
</div>
 318/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.7475 - loss: 0.1000 - moe_loss: 815.4291

<div class="k-default-codeblock">
```

```
</div>
 321/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.7488 - loss: 0.1000 - moe_loss: 823.1090

<div class="k-default-codeblock">
```

```
</div>
 325/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.7504 - loss: 0.1000 - moe_loss: 833.3490

<div class="k-default-codeblock">
```

```
</div>
 329/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.7520 - loss: 0.1000 - moe_loss: 843.5892

<div class="k-default-codeblock">
```

```
</div>
 332/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.7532 - loss: 0.1000 - moe_loss: 851.2693

<div class="k-default-codeblock">
```

```
</div>
 336/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.7548 - loss: 0.1000 - moe_loss: 861.5092

<div class="k-default-codeblock">
```

```
</div>
 340/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.7563 - loss: 0.1000 - moe_loss: 871.7490

<div class="k-default-codeblock">
```

```
</div>
 343/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.7575 - loss: 0.1000 - moe_loss: 879.4288

<div class="k-default-codeblock">
```

```
</div>
 347/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.7590 - loss: 0.1000 - moe_loss: 889.6685

<div class="k-default-codeblock">
```

```
</div>
 350/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.7601 - loss: 0.1000 - moe_loss: 897.3481

<div class="k-default-codeblock">
```

```
</div>
 353/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.7612 - loss: 0.1000 - moe_loss: 905.0280

<div class="k-default-codeblock">
```

```
</div>
 357/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.7626 - loss: 0.1000 - moe_loss: 915.2678

<div class="k-default-codeblock">
```

```
</div>
 361/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.7640 - loss: 0.1000 - moe_loss: 925.5076

<div class="k-default-codeblock">
```

```
</div>
 365/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.7654 - loss: 0.1000 - moe_loss: 935.7476

<div class="k-default-codeblock">
```

```
</div>
 368/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.7664 - loss: 0.1000 - moe_loss: 943.4277

<div class="k-default-codeblock">
```

```
</div>
 372/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.7678 - loss: 0.1000 - moe_loss: 953.6683

<div class="k-default-codeblock">
```

```
</div>
 375/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.7688 - loss: 0.1000 - moe_loss: 961.3480

<div class="k-default-codeblock">
```

```
</div>
 378/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.7698 - loss: 0.1000 - moe_loss: 969.0279

<div class="k-default-codeblock">
```

```
</div>
 382/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.7711 - loss: 0.1000 - moe_loss: 979.2679

<div class="k-default-codeblock">
```

```
</div>
 386/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.7724 - loss: 0.1000 - moe_loss: 989.5076

<div class="k-default-codeblock">
```

```
</div>
 390/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.7736 - loss: 0.1000 - moe_loss: 999.7477

<div class="k-default-codeblock">
```

```
</div>
 394/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.7749 - loss: 0.1000 - moe_loss: 1009.9877

<div class="k-default-codeblock">
```

```
</div>
 398/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.7761 - loss: 0.1000 - moe_loss: 1020.2275

<div class="k-default-codeblock">
```

```
</div>
 402/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.7773 - loss: 0.1000 - moe_loss: 1030.4677

<div class="k-default-codeblock">
```

```
</div>
 406/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.7785 - loss: 0.1000 - moe_loss: 1040.7075

<div class="k-default-codeblock">
```

```
</div>
 410/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.7797 - loss: 0.1000 - moe_loss: 1050.9473

<div class="k-default-codeblock">
```

```
</div>
 414/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.7808 - loss: 0.1000 - moe_loss: 1061.1871

<div class="k-default-codeblock">
```

```
</div>
 418/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.7820 - loss: 0.1000 - moe_loss: 1071.4269

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.7828 - loss: 0.1000 - moe_loss: 1079.1069

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.7839 - loss: 0.1000 - moe_loss: 1089.3467

<div class="k-default-codeblock">
```

```
</div>
 429/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.7850 - loss: 0.1000 - moe_loss: 1099.5865

<div class="k-default-codeblock">
```

```
</div>
 433/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.7861 - loss: 0.1000 - moe_loss: 1109.8267

<div class="k-default-codeblock">
```

```
</div>
 436/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.7869 - loss: 0.1000 - moe_loss: 1117.5068

<div class="k-default-codeblock">
```

```
</div>
 439/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.7877 - loss: 0.1000 - moe_loss: 1125.1870

<div class="k-default-codeblock">
```

```
</div>
 443/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.7887 - loss: 0.1000 - moe_loss: 1135.4268

<div class="k-default-codeblock">
```

```
</div>
 446/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.7895 - loss: 0.1000 - moe_loss: 1143.1067

<div class="k-default-codeblock">
```

```
</div>
 450/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.7905 - loss: 0.1000 - moe_loss: 1153.3466

<div class="k-default-codeblock">
```

```
</div>
 454/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.7916 - loss: 0.1000 - moe_loss: 1163.5869

<div class="k-default-codeblock">
```

```
</div>
 458/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.7926 - loss: 0.1000 - moe_loss: 1173.8270

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.7933 - loss: 0.1000 - moe_loss: 1181.5070

<div class="k-default-codeblock">
```

```
</div>
 464/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.7940 - loss: 0.1000 - moe_loss: 1189.1869

<div class="k-default-codeblock">
```

```
</div>
 468/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.7950 - loss: 0.1000 - moe_loss: 1199.4266

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 9s 18ms/step - accuracy: 0.7956 - loss: 0.1000 - moe_loss: 1204.5302 - val_loss: 0.1000 - val_moe_loss: 2798.7275


<div class="k-default-codeblock">
```
Epoch 2/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  14s 30ms/step - accuracy: 0.9688 - loss: 0.1000 - moe_loss: 2803.8604

<div class="k-default-codeblock">
```

```
</div>
   5/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9682 - loss: 0.1000 - moe_loss: 2814.1450 

<div class="k-default-codeblock">
```

```
</div>
   9/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9677 - loss: 0.1000 - moe_loss: 2824.3696

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9686 - loss: 0.1000 - moe_loss: 2834.6130

<div class="k-default-codeblock">
```

```
</div>
  17/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9693 - loss: 0.1000 - moe_loss: 2844.8762

<div class="k-default-codeblock">
```

```
</div>
  20/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9693 - loss: 0.1000 - moe_loss: 2852.5579

<div class="k-default-codeblock">
```

```
</div>
  23/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9692 - loss: 0.1000 - moe_loss: 2860.2383

<div class="k-default-codeblock">
```

```
</div>
  26/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9691 - loss: 0.1000 - moe_loss: 2867.9192

<div class="k-default-codeblock">
```

```
</div>
  29/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9688 - loss: 0.1000 - moe_loss: 2875.5964

<div class="k-default-codeblock">
```

```
</div>
  33/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9683 - loss: 0.1000 - moe_loss: 2885.8335

<div class="k-default-codeblock">
```

```
</div>
  36/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9679 - loss: 0.1000 - moe_loss: 2893.5164

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9676 - loss: 0.1000 - moe_loss: 2903.7554

<div class="k-default-codeblock">
```

```
</div>
  44/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9672 - loss: 0.1000 - moe_loss: 2913.9944

<div class="k-default-codeblock">
```

```
</div>
  48/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9668 - loss: 0.1000 - moe_loss: 2924.2329

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9665 - loss: 0.1000 - moe_loss: 2934.4727

<div class="k-default-codeblock">
```

```
</div>
  56/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9663 - loss: 0.1000 - moe_loss: 2944.7114

<div class="k-default-codeblock">
```

```
</div>
  60/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9661 - loss: 0.1000 - moe_loss: 2954.9500

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9660 - loss: 0.1000 - moe_loss: 2965.1897

<div class="k-default-codeblock">
```

```
</div>
  68/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9659 - loss: 0.1000 - moe_loss: 2975.4287

<div class="k-default-codeblock">
```

```
</div>
  72/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9659 - loss: 0.1000 - moe_loss: 2985.6675

<div class="k-default-codeblock">
```

```
</div>
  75/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9659 - loss: 0.1000 - moe_loss: 2993.3472

<div class="k-default-codeblock">
```

```
</div>
  79/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9659 - loss: 0.1000 - moe_loss: 3003.5850

<div class="k-default-codeblock">
```

```
</div>
  83/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9659 - loss: 0.1000 - moe_loss: 3013.8240

<div class="k-default-codeblock">
```

```
</div>
  87/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9659 - loss: 0.1000 - moe_loss: 3024.0654

<div class="k-default-codeblock">
```

```
</div>
  91/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9659 - loss: 0.1000 - moe_loss: 3034.3062

<div class="k-default-codeblock">
```

```
</div>
  95/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9659 - loss: 0.1000 - moe_loss: 3044.5454

<div class="k-default-codeblock">
```

```
</div>
  99/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9659 - loss: 0.1000 - moe_loss: 3054.7854

<div class="k-default-codeblock">
```

```
</div>
 103/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9660 - loss: 0.1000 - moe_loss: 3065.0247

<div class="k-default-codeblock">
```

```
</div>
 107/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9660 - loss: 0.1000 - moe_loss: 3075.2642

<div class="k-default-codeblock">
```

```
</div>
 110/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9660 - loss: 0.1000 - moe_loss: 3082.9436

<div class="k-default-codeblock">
```

```
</div>
 114/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9661 - loss: 0.1000 - moe_loss: 3093.1829

<div class="k-default-codeblock">
```

```
</div>
 117/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9661 - loss: 0.1000 - moe_loss: 3100.8628

<div class="k-default-codeblock">
```

```
</div>
 120/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9661 - loss: 0.1000 - moe_loss: 3108.5425

<div class="k-default-codeblock">
```

```
</div>
 123/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9662 - loss: 0.1000 - moe_loss: 3116.2224

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9662 - loss: 0.1000 - moe_loss: 3126.4617

<div class="k-default-codeblock">
```

```
</div>
 131/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9662 - loss: 0.1000 - moe_loss: 3136.7017

<div class="k-default-codeblock">
```

```
</div>
 134/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9662 - loss: 0.1000 - moe_loss: 3144.3816

<div class="k-default-codeblock">
```

```
</div>
 138/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9662 - loss: 0.1000 - moe_loss: 3154.6211

<div class="k-default-codeblock">
```

```
</div>
 142/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9663 - loss: 0.1000 - moe_loss: 3164.8611

<div class="k-default-codeblock">
```

```
</div>
 145/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9663 - loss: 0.1000 - moe_loss: 3172.5408

<div class="k-default-codeblock">
```

```
</div>
 148/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9663 - loss: 0.1000 - moe_loss: 3180.2202

<div class="k-default-codeblock">
```

```
</div>
 151/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9663 - loss: 0.1000 - moe_loss: 3187.8999

<div class="k-default-codeblock">
```

```
</div>
 154/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9663 - loss: 0.1000 - moe_loss: 3195.5798

<div class="k-default-codeblock">
```

```
</div>
 158/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9663 - loss: 0.1000 - moe_loss: 3205.8191

<div class="k-default-codeblock">
```

```
</div>
 162/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9663 - loss: 0.1000 - moe_loss: 3216.0586

<div class="k-default-codeblock">
```

```
</div>
 166/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9663 - loss: 0.1000 - moe_loss: 3226.2993

<div class="k-default-codeblock">
```

```
</div>
 170/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9663 - loss: 0.1000 - moe_loss: 3236.5393

<div class="k-default-codeblock">
```

```
</div>
 174/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9664 - loss: 0.1000 - moe_loss: 3246.7805

<div class="k-default-codeblock">
```

```
</div>
 178/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9664 - loss: 0.1000 - moe_loss: 3257.0203

<div class="k-default-codeblock">
```

```
</div>
 182/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9664 - loss: 0.1000 - moe_loss: 3267.2603

<div class="k-default-codeblock">
```

```
</div>
 185/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9664 - loss: 0.1000 - moe_loss: 3274.9407

<div class="k-default-codeblock">
```

```
</div>
 188/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9664 - loss: 0.1000 - moe_loss: 3282.6201

<div class="k-default-codeblock">
```

```
</div>
 192/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9665 - loss: 0.1000 - moe_loss: 3292.8596

<div class="k-default-codeblock">
```

```
</div>
 195/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9665 - loss: 0.1000 - moe_loss: 3300.5391

<div class="k-default-codeblock">
```

```
</div>
 199/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9665 - loss: 0.1000 - moe_loss: 3310.7786

<div class="k-default-codeblock">
```

```
</div>
 202/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9665 - loss: 0.1000 - moe_loss: 3318.4583

<div class="k-default-codeblock">
```

```
</div>
 206/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9666 - loss: 0.1000 - moe_loss: 3328.6982

<div class="k-default-codeblock">
```

```
</div>
 210/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9666 - loss: 0.1000 - moe_loss: 3338.9380

<div class="k-default-codeblock">
```

```
</div>
 213/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9666 - loss: 0.1000 - moe_loss: 3346.6179

<div class="k-default-codeblock">
```

```
</div>
 217/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9667 - loss: 0.1000 - moe_loss: 3356.8574

<div class="k-default-codeblock">
```

```
</div>
 221/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9667 - loss: 0.1000 - moe_loss: 3367.0972

<div class="k-default-codeblock">
```

```
</div>
 225/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9667 - loss: 0.1000 - moe_loss: 3377.3372

<div class="k-default-codeblock">
```

```
</div>
 229/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 17ms/step - accuracy: 0.9667 - loss: 0.1000 - moe_loss: 3387.5769

<div class="k-default-codeblock">
```

```
</div>
 233/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 17ms/step - accuracy: 0.9668 - loss: 0.1000 - moe_loss: 3397.8169

<div class="k-default-codeblock">
```

```
</div>
 237/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9668 - loss: 0.1000 - moe_loss: 3408.0564

<div class="k-default-codeblock">
```

```
</div>
 241/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9669 - loss: 0.1000 - moe_loss: 3418.2964

<div class="k-default-codeblock">
```

```
</div>
 244/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9669 - loss: 0.1000 - moe_loss: 3425.9768

<div class="k-default-codeblock">
```

```
</div>
 247/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9669 - loss: 0.1000 - moe_loss: 3433.6567

<div class="k-default-codeblock">
```

```
</div>
 251/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9669 - loss: 0.1000 - moe_loss: 3443.8967

<div class="k-default-codeblock">
```

```
</div>
 255/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 16ms/step - accuracy: 0.9670 - loss: 0.1000 - moe_loss: 3454.1365

<div class="k-default-codeblock">
```

```
</div>
 259/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9670 - loss: 0.1000 - moe_loss: 3464.3762

<div class="k-default-codeblock">
```

```
</div>
 263/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9670 - loss: 0.1000 - moe_loss: 3474.6157

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9671 - loss: 0.1000 - moe_loss: 3484.8552

<div class="k-default-codeblock">
```

```
</div>
 271/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9671 - loss: 0.1000 - moe_loss: 3495.0950

<div class="k-default-codeblock">
```

```
</div>
 274/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9671 - loss: 0.1000 - moe_loss: 3502.7751

<div class="k-default-codeblock">
```

```
</div>
 278/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9672 - loss: 0.1000 - moe_loss: 3513.0149

<div class="k-default-codeblock">
```

```
</div>
 282/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 16ms/step - accuracy: 0.9672 - loss: 0.1000 - moe_loss: 3523.2546

<div class="k-default-codeblock">
```

```
</div>
 286/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 16ms/step - accuracy: 0.9672 - loss: 0.1000 - moe_loss: 3533.4944

<div class="k-default-codeblock">
```

```
</div>
 290/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 16ms/step - accuracy: 0.9673 - loss: 0.1000 - moe_loss: 3543.7341

<div class="k-default-codeblock">
```

```
</div>
 294/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 16ms/step - accuracy: 0.9673 - loss: 0.1000 - moe_loss: 3553.9744

<div class="k-default-codeblock">
```

```
</div>
 298/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 16ms/step - accuracy: 0.9674 - loss: 0.1000 - moe_loss: 3564.2141

<div class="k-default-codeblock">
```

```
</div>
 302/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 16ms/step - accuracy: 0.9674 - loss: 0.1000 - moe_loss: 3574.4539

<div class="k-default-codeblock">
```

```
</div>
 306/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9674 - loss: 0.1000 - moe_loss: 3584.6936

<div class="k-default-codeblock">
```

```
</div>
 310/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9675 - loss: 0.1000 - moe_loss: 3594.9331

<div class="k-default-codeblock">
```

```
</div>
 314/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9675 - loss: 0.1000 - moe_loss: 3605.1729

<div class="k-default-codeblock">
```

```
</div>
 318/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9675 - loss: 0.1000 - moe_loss: 3615.4126

<div class="k-default-codeblock">
```

```
</div>
 322/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9676 - loss: 0.1000 - moe_loss: 3625.6523

<div class="k-default-codeblock">
```

```
</div>
 325/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9676 - loss: 0.1000 - moe_loss: 3633.3323

<div class="k-default-codeblock">
```

```
</div>
 328/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9676 - loss: 0.1000 - moe_loss: 3641.0125

<div class="k-default-codeblock">
```

```
</div>
 332/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 16ms/step - accuracy: 0.9676 - loss: 0.1000 - moe_loss: 3651.2524

<div class="k-default-codeblock">
```

```
</div>
 336/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 16ms/step - accuracy: 0.9677 - loss: 0.1000 - moe_loss: 3661.4922

<div class="k-default-codeblock">
```

```
</div>
 340/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 16ms/step - accuracy: 0.9677 - loss: 0.1000 - moe_loss: 3671.7319

<div class="k-default-codeblock">
```

```
</div>
 344/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 16ms/step - accuracy: 0.9677 - loss: 0.1000 - moe_loss: 3681.9717

<div class="k-default-codeblock">
```

```
</div>
 348/469 ━━━━━━━━━━━━━━[37m━━━━━━  1s 16ms/step - accuracy: 0.9678 - loss: 0.1000 - moe_loss: 3692.2117

<div class="k-default-codeblock">
```

```
</div>
 352/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9678 - loss: 0.1000 - moe_loss: 3702.4514

<div class="k-default-codeblock">
```

```
</div>
 356/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9678 - loss: 0.1000 - moe_loss: 3712.6914

<div class="k-default-codeblock">
```

```
</div>
 360/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9679 - loss: 0.1000 - moe_loss: 3722.9312

<div class="k-default-codeblock">
```

```
</div>
 364/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9679 - loss: 0.1000 - moe_loss: 3733.1711

<div class="k-default-codeblock">
```

```
</div>
 367/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9679 - loss: 0.1000 - moe_loss: 3740.8511

<div class="k-default-codeblock">
```

```
</div>
 370/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9679 - loss: 0.1000 - moe_loss: 3748.5310

<div class="k-default-codeblock">
```

```
</div>
 374/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9680 - loss: 0.1000 - moe_loss: 3758.7710

<div class="k-default-codeblock">
```

```
</div>
 378/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 16ms/step - accuracy: 0.9680 - loss: 0.1000 - moe_loss: 3769.0112

<div class="k-default-codeblock">
```

```
</div>
 381/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9680 - loss: 0.1000 - moe_loss: 3776.6914

<div class="k-default-codeblock">
```

```
</div>
 384/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9680 - loss: 0.1000 - moe_loss: 3784.3713

<div class="k-default-codeblock">
```

```
</div>
 388/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9680 - loss: 0.1000 - moe_loss: 3794.6113

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9681 - loss: 0.1000 - moe_loss: 3802.2913

<div class="k-default-codeblock">
```

```
</div>
 392/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9681 - loss: 0.1000 - moe_loss: 3804.8511

<div class="k-default-codeblock">
```

```
</div>
 395/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9681 - loss: 0.1000 - moe_loss: 3812.5310

<div class="k-default-codeblock">
```

```
</div>
 398/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9681 - loss: 0.1000 - moe_loss: 3820.2109

<div class="k-default-codeblock">
```

```
</div>
 401/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9681 - loss: 0.1000 - moe_loss: 3827.8906

<div class="k-default-codeblock">
```

```
</div>
 404/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9682 - loss: 0.1000 - moe_loss: 3835.5706

<div class="k-default-codeblock">
```

```
</div>
 407/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9682 - loss: 0.1000 - moe_loss: 3843.2505

<div class="k-default-codeblock">
```

```
</div>
 410/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9682 - loss: 0.1000 - moe_loss: 3850.9304

<div class="k-default-codeblock">
```

```
</div>
 413/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9682 - loss: 0.1000 - moe_loss: 3858.6106

<div class="k-default-codeblock">
```

```
</div>
 417/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9682 - loss: 0.1000 - moe_loss: 3868.8503

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9683 - loss: 0.1000 - moe_loss: 3879.0901

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9683 - loss: 0.1000 - moe_loss: 3889.3303

<div class="k-default-codeblock">
```

```
</div>
 429/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9683 - loss: 0.1000 - moe_loss: 3899.5706

<div class="k-default-codeblock">
```

```
</div>
 432/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9683 - loss: 0.1000 - moe_loss: 3907.2507

<div class="k-default-codeblock">
```

```
</div>
 435/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9684 - loss: 0.1000 - moe_loss: 3914.9309

<div class="k-default-codeblock">
```

```
</div>
 438/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9684 - loss: 0.1000 - moe_loss: 3922.6106

<div class="k-default-codeblock">
```

```
</div>
 441/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9684 - loss: 0.1000 - moe_loss: 3930.2908

<div class="k-default-codeblock">
```

```
</div>
 445/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9684 - loss: 0.1000 - moe_loss: 3940.5305

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9685 - loss: 0.1000 - moe_loss: 3950.7703

<div class="k-default-codeblock">
```

```
</div>
 452/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9685 - loss: 0.1000 - moe_loss: 3958.4500

<div class="k-default-codeblock">
```

```
</div>
 456/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9685 - loss: 0.1000 - moe_loss: 3968.6899

<div class="k-default-codeblock">
```

```
</div>
 459/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9685 - loss: 0.1000 - moe_loss: 3976.3699

<div class="k-default-codeblock">
```

```
</div>
 462/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9686 - loss: 0.1000 - moe_loss: 3984.0498

<div class="k-default-codeblock">
```

```
</div>
 466/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9686 - loss: 0.1000 - moe_loss: 3994.2898

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.9686 - loss: 0.1000 - moe_loss: 4004.5132 - val_loss: 0.1000 - val_moe_loss: 5598.7266


<div class="k-default-codeblock">
```
Epoch 3/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  13s 28ms/step - accuracy: 0.9766 - loss: 0.1000 - moe_loss: 5603.8740

<div class="k-default-codeblock">
```

```
</div>
   5/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9725 - loss: 0.1000 - moe_loss: 5614.1147 

<div class="k-default-codeblock">
```

```
</div>
   9/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9711 - loss: 0.1000 - moe_loss: 5624.3594

<div class="k-default-codeblock">
```

```
</div>
  12/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9713 - loss: 0.1000 - moe_loss: 5632.0366

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9720 - loss: 0.1000 - moe_loss: 5642.2812

<div class="k-default-codeblock">
```

```
</div>
  20/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9727 - loss: 0.1000 - moe_loss: 5652.5317

<div class="k-default-codeblock">
```

```
</div>
  24/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9731 - loss: 0.1000 - moe_loss: 5662.7671

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9734 - loss: 0.1000 - moe_loss: 5673.0073

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9736 - loss: 0.1000 - moe_loss: 5680.6851

<div class="k-default-codeblock">
```

```
</div>
  35/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9738 - loss: 0.1000 - moe_loss: 5690.9282

<div class="k-default-codeblock">
```

```
</div>
  39/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9740 - loss: 0.1000 - moe_loss: 5701.1680

<div class="k-default-codeblock">
```

```
</div>
  43/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9742 - loss: 0.1000 - moe_loss: 5711.4087

<div class="k-default-codeblock">
```

```
</div>
  47/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9742 - loss: 0.1000 - moe_loss: 5721.6470

<div class="k-default-codeblock">
```

```
</div>
  51/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9743 - loss: 0.1000 - moe_loss: 5731.8843

<div class="k-default-codeblock">
```

```
</div>
  54/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9743 - loss: 0.1000 - moe_loss: 5739.5645

<div class="k-default-codeblock">
```

```
</div>
  58/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9744 - loss: 0.1000 - moe_loss: 5749.8052

<div class="k-default-codeblock">
```

```
</div>
  61/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9745 - loss: 0.1000 - moe_loss: 5757.4844

<div class="k-default-codeblock">
```

```
</div>
  65/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9746 - loss: 0.1000 - moe_loss: 5767.7251

<div class="k-default-codeblock">
```

```
</div>
  69/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9747 - loss: 0.1000 - moe_loss: 5777.9648

<div class="k-default-codeblock">
```

```
</div>
  73/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9748 - loss: 0.1000 - moe_loss: 5788.2041

<div class="k-default-codeblock">
```

```
</div>
  77/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9749 - loss: 0.1000 - moe_loss: 5798.4434

<div class="k-default-codeblock">
```

```
</div>
  81/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9750 - loss: 0.1000 - moe_loss: 5808.6831

<div class="k-default-codeblock">
```

```
</div>
  84/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9751 - loss: 0.1000 - moe_loss: 5816.3623

<div class="k-default-codeblock">
```

```
</div>
  88/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9752 - loss: 0.1000 - moe_loss: 5826.6025

<div class="k-default-codeblock">
```

```
</div>
  92/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9752 - loss: 0.1000 - moe_loss: 5836.8413

<div class="k-default-codeblock">
```

```
</div>
  96/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9753 - loss: 0.1000 - moe_loss: 5847.0811

<div class="k-default-codeblock">
```

```
</div>
 100/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9753 - loss: 0.1000 - moe_loss: 5857.3213

<div class="k-default-codeblock">
```

```
</div>
 104/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9754 - loss: 0.1000 - moe_loss: 5867.5610

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9755 - loss: 0.1000 - moe_loss: 5877.8013

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9755 - loss: 0.1000 - moe_loss: 5885.4810

<div class="k-default-codeblock">
```

```
</div>
 115/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9756 - loss: 0.1000 - moe_loss: 5895.7212

<div class="k-default-codeblock">
```

```
</div>
 119/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9756 - loss: 0.1000 - moe_loss: 5905.9614

<div class="k-default-codeblock">
```

```
</div>
 122/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9757 - loss: 0.1000 - moe_loss: 5913.6421

<div class="k-default-codeblock">
```

```
</div>
 126/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9758 - loss: 0.1000 - moe_loss: 5923.8813

<div class="k-default-codeblock">
```

```
</div>
 129/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9758 - loss: 0.1000 - moe_loss: 5931.5615

<div class="k-default-codeblock">
```

```
</div>
 132/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9758 - loss: 0.1000 - moe_loss: 5939.2412

<div class="k-default-codeblock">
```

```
</div>
 136/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9759 - loss: 0.1000 - moe_loss: 5949.4810

<div class="k-default-codeblock">
```

```
</div>
 140/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9759 - loss: 0.1000 - moe_loss: 5959.7207

<div class="k-default-codeblock">
```

```
</div>
 144/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9760 - loss: 0.1000 - moe_loss: 5969.9600

<div class="k-default-codeblock">
```

```
</div>
 148/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9760 - loss: 0.1000 - moe_loss: 5980.2007

<div class="k-default-codeblock">
```

```
</div>
 152/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9761 - loss: 0.1000 - moe_loss: 5990.4404

<div class="k-default-codeblock">
```

```
</div>
 156/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9761 - loss: 0.1000 - moe_loss: 6000.6802

<div class="k-default-codeblock">
```

```
</div>
 160/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9762 - loss: 0.1000 - moe_loss: 6010.9199

<div class="k-default-codeblock">
```

```
</div>
 164/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9762 - loss: 0.1000 - moe_loss: 6021.1602

<div class="k-default-codeblock">
```

```
</div>
 168/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9763 - loss: 0.1000 - moe_loss: 6031.3994

<div class="k-default-codeblock">
```

```
</div>
 172/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9763 - loss: 0.1000 - moe_loss: 6041.6392

<div class="k-default-codeblock">
```

```
</div>
 173/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9763 - loss: 0.1000 - moe_loss: 6044.1992

<div class="k-default-codeblock">
```

```
</div>
 176/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9764 - loss: 0.1000 - moe_loss: 6051.8794

<div class="k-default-codeblock">
```

```
</div>
 179/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9764 - loss: 0.1000 - moe_loss: 6059.5596

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9765 - loss: 0.1000 - moe_loss: 6069.7998

<div class="k-default-codeblock">
```

```
</div>
 187/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9765 - loss: 0.1000 - moe_loss: 6080.0405

<div class="k-default-codeblock">
```

```
</div>
 191/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9765 - loss: 0.1000 - moe_loss: 6090.2808

<div class="k-default-codeblock">
```

```
</div>
 195/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9766 - loss: 0.1000 - moe_loss: 6100.5200

<div class="k-default-codeblock">
```

```
</div>
 199/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9766 - loss: 0.1000 - moe_loss: 6110.7603

<div class="k-default-codeblock">
```

```
</div>
 203/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9767 - loss: 0.1000 - moe_loss: 6120.9995

<div class="k-default-codeblock">
```

```
</div>
 207/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9767 - loss: 0.1000 - moe_loss: 6131.2402

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9768 - loss: 0.1000 - moe_loss: 6141.4800

<div class="k-default-codeblock">
```

```
</div>
 215/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9768 - loss: 0.1000 - moe_loss: 6151.7197

<div class="k-default-codeblock">
```

```
</div>
 219/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9769 - loss: 0.1000 - moe_loss: 6161.9600

<div class="k-default-codeblock">
```

```
</div>
 223/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9769 - loss: 0.1000 - moe_loss: 6172.1992

<div class="k-default-codeblock">
```

```
</div>
 227/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9769 - loss: 0.1000 - moe_loss: 6182.4390

<div class="k-default-codeblock">
```

```
</div>
 231/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 17ms/step - accuracy: 0.9770 - loss: 0.1000 - moe_loss: 6192.6792

<div class="k-default-codeblock">
```

```
</div>
 235/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9770 - loss: 0.1000 - moe_loss: 6202.9194

<div class="k-default-codeblock">
```

```
</div>
 239/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9771 - loss: 0.1000 - moe_loss: 6213.1592

<div class="k-default-codeblock">
```

```
</div>
 243/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9771 - loss: 0.1000 - moe_loss: 6223.3989

<div class="k-default-codeblock">
```

```
</div>
 246/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9771 - loss: 0.1000 - moe_loss: 6231.0786

<div class="k-default-codeblock">
```

```
</div>
 250/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9772 - loss: 0.1000 - moe_loss: 6241.3188

<div class="k-default-codeblock">
```

```
</div>
 253/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9772 - loss: 0.1000 - moe_loss: 6248.9990

<div class="k-default-codeblock">
```

```
</div>
 256/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9772 - loss: 0.1000 - moe_loss: 6256.6792

<div class="k-default-codeblock">
```

```
</div>
 260/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9773 - loss: 0.1000 - moe_loss: 6266.9189

<div class="k-default-codeblock">
```

```
</div>
 264/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9773 - loss: 0.1000 - moe_loss: 6277.1587

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9773 - loss: 0.1000 - moe_loss: 6284.8384

<div class="k-default-codeblock">
```

```
</div>
 270/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9773 - loss: 0.1000 - moe_loss: 6292.5186

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9773 - loss: 0.1000 - moe_loss: 6300.1987

<div class="k-default-codeblock">
```

```
</div>
 276/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9774 - loss: 0.1000 - moe_loss: 6307.8789

<div class="k-default-codeblock">
```

```
</div>
 279/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9774 - loss: 0.1000 - moe_loss: 6315.5586

<div class="k-default-codeblock">
```

```
</div>
 282/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9774 - loss: 0.1000 - moe_loss: 6323.2388

<div class="k-default-codeblock">
```

```
</div>
 286/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9774 - loss: 0.1000 - moe_loss: 6333.4790

<div class="k-default-codeblock">
```

```
</div>
 290/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9775 - loss: 0.1000 - moe_loss: 6343.7188

<div class="k-default-codeblock">
```

```
</div>
 294/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9775 - loss: 0.1000 - moe_loss: 6353.9590

<div class="k-default-codeblock">
```

```
</div>
 298/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9775 - loss: 0.1000 - moe_loss: 6364.1992

<div class="k-default-codeblock">
```

```
</div>
 302/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9776 - loss: 0.1000 - moe_loss: 6374.4390

<div class="k-default-codeblock">
```

```
</div>
 305/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9776 - loss: 0.1000 - moe_loss: 6382.1191

<div class="k-default-codeblock">
```

```
</div>
 309/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9776 - loss: 0.1000 - moe_loss: 6392.3589

<div class="k-default-codeblock">
```

```
</div>
 313/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9776 - loss: 0.1000 - moe_loss: 6402.5991

<div class="k-default-codeblock">
```

```
</div>
 317/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9777 - loss: 0.1000 - moe_loss: 6412.8389

<div class="k-default-codeblock">
```

```
</div>
 321/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9777 - loss: 0.1000 - moe_loss: 6423.0786

<div class="k-default-codeblock">
```

```
</div>
 325/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9777 - loss: 0.1000 - moe_loss: 6433.3184

<div class="k-default-codeblock">
```

```
</div>
 329/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9778 - loss: 0.1000 - moe_loss: 6443.5581

<div class="k-default-codeblock">
```

```
</div>
 333/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9778 - loss: 0.1000 - moe_loss: 6453.7983

<div class="k-default-codeblock">
```

```
</div>
 336/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9778 - loss: 0.1000 - moe_loss: 6461.4780

<div class="k-default-codeblock">
```

```
</div>
 340/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9778 - loss: 0.1000 - moe_loss: 6471.7178

<div class="k-default-codeblock">
```

```
</div>
 344/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9779 - loss: 0.1000 - moe_loss: 6481.9580

<div class="k-default-codeblock">
```

```
</div>
 348/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9779 - loss: 0.1000 - moe_loss: 6492.1978

<div class="k-default-codeblock">
```

```
</div>
 352/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9779 - loss: 0.1000 - moe_loss: 6502.4375

<div class="k-default-codeblock">
```

```
</div>
 356/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9779 - loss: 0.1000 - moe_loss: 6512.6777

<div class="k-default-codeblock">
```

```
</div>
 360/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9780 - loss: 0.1000 - moe_loss: 6522.9180

<div class="k-default-codeblock">
```

```
</div>
 364/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9780 - loss: 0.1000 - moe_loss: 6533.1577

<div class="k-default-codeblock">
```

```
</div>
 367/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9780 - loss: 0.1000 - moe_loss: 6540.8379

<div class="k-default-codeblock">
```

```
</div>
 371/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9780 - loss: 0.1000 - moe_loss: 6551.0776

<div class="k-default-codeblock">
```

```
</div>
 375/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9780 - loss: 0.1000 - moe_loss: 6561.3174

<div class="k-default-codeblock">
```

```
</div>
 379/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9781 - loss: 0.1000 - moe_loss: 6571.5576

<div class="k-default-codeblock">
```

```
</div>
 383/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9781 - loss: 0.1000 - moe_loss: 6581.7974

<div class="k-default-codeblock">
```

```
</div>
 387/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9781 - loss: 0.1000 - moe_loss: 6592.0371

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9781 - loss: 0.1000 - moe_loss: 6602.2773

<div class="k-default-codeblock">
```

```
</div>
 395/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9781 - loss: 0.1000 - moe_loss: 6612.5176

<div class="k-default-codeblock">
```

```
</div>
 398/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9782 - loss: 0.1000 - moe_loss: 6620.1973

<div class="k-default-codeblock">
```

```
</div>
 402/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9782 - loss: 0.1000 - moe_loss: 6630.4375

<div class="k-default-codeblock">
```

```
</div>
 405/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9782 - loss: 0.1000 - moe_loss: 6638.1172

<div class="k-default-codeblock">
```

```
</div>
 409/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9782 - loss: 0.1000 - moe_loss: 6648.3569

<div class="k-default-codeblock">
```

```
</div>
 413/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9782 - loss: 0.1000 - moe_loss: 6658.5972

<div class="k-default-codeblock">
```

```
</div>
 416/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9782 - loss: 0.1000 - moe_loss: 6666.2769

<div class="k-default-codeblock">
```

```
</div>
 419/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9783 - loss: 0.1000 - moe_loss: 6673.9570

<div class="k-default-codeblock">
```

```
</div>
 423/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9783 - loss: 0.1000 - moe_loss: 6684.1973

<div class="k-default-codeblock">
```

```
</div>
 426/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9783 - loss: 0.1000 - moe_loss: 6691.8770

<div class="k-default-codeblock">
```

```
</div>
 429/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9783 - loss: 0.1000 - moe_loss: 6699.5571

<div class="k-default-codeblock">
```

```
</div>
 433/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9783 - loss: 0.1000 - moe_loss: 6709.7969

<div class="k-default-codeblock">
```

```
</div>
 437/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9783 - loss: 0.1000 - moe_loss: 6720.0366

<div class="k-default-codeblock">
```

```
</div>
 441/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9783 - loss: 0.1000 - moe_loss: 6730.2764

<div class="k-default-codeblock">
```

```
</div>
 445/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9784 - loss: 0.1000 - moe_loss: 6740.5166

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9784 - loss: 0.1000 - moe_loss: 6750.7563

<div class="k-default-codeblock">
```

```
</div>
 453/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9784 - loss: 0.1000 - moe_loss: 6760.9961

<div class="k-default-codeblock">
```

```
</div>
 457/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9784 - loss: 0.1000 - moe_loss: 6771.2363

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9784 - loss: 0.1000 - moe_loss: 6781.4766

<div class="k-default-codeblock">
```

```
</div>
 465/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9784 - loss: 0.1000 - moe_loss: 6791.7163

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.9784 - loss: 0.1000 - moe_loss: 6801.9536

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.9785 - loss: 0.1000 - moe_loss: 6804.5000 - val_loss: 0.1000 - val_moe_loss: 8398.7275


<div class="k-default-codeblock">
```
Epoch 4/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 26ms/step - accuracy: 0.9766 - loss: 0.1000 - moe_loss: 8403.8486

<div class="k-default-codeblock">
```

```
</div>
   5/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9783 - loss: 0.1000 - moe_loss: 8414.1064 

<div class="k-default-codeblock">
```

```
</div>
   9/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9777 - loss: 0.1000 - moe_loss: 8424.3496

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9775 - loss: 0.1000 - moe_loss: 8434.5850

<div class="k-default-codeblock">
```

```
</div>
  17/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9771 - loss: 0.1000 - moe_loss: 8444.8232

<div class="k-default-codeblock">
```

```
</div>
  21/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9770 - loss: 0.1000 - moe_loss: 8455.0625

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9772 - loss: 0.1000 - moe_loss: 8465.3047

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9773 - loss: 0.1000 - moe_loss: 8472.9844

<div class="k-default-codeblock">
```

```
</div>
  32/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9774 - loss: 0.1000 - moe_loss: 8483.2256

<div class="k-default-codeblock">
```

```
</div>
  36/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9776 - loss: 0.1000 - moe_loss: 8493.4678

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9779 - loss: 0.1000 - moe_loss: 8503.7090

<div class="k-default-codeblock">
```

```
</div>
  44/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9781 - loss: 0.1000 - moe_loss: 8513.9502

<div class="k-default-codeblock">
```

```
</div>
  48/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9782 - loss: 0.1000 - moe_loss: 8524.1924

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9783 - loss: 0.1000 - moe_loss: 8534.4336

<div class="k-default-codeblock">
```

```
</div>
  56/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9784 - loss: 0.1000 - moe_loss: 8544.6738

<div class="k-default-codeblock">
```

```
</div>
  60/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9786 - loss: 0.1000 - moe_loss: 8554.9131

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9787 - loss: 0.1000 - moe_loss: 8565.1514

<div class="k-default-codeblock">
```

```
</div>
  68/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9789 - loss: 0.1000 - moe_loss: 8575.3916

<div class="k-default-codeblock">
```

```
</div>
  72/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9790 - loss: 0.1000 - moe_loss: 8585.6318

<div class="k-default-codeblock">
```

```
</div>
  75/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9791 - loss: 0.1000 - moe_loss: 8593.3125

<div class="k-default-codeblock">
```

```
</div>
  79/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9792 - loss: 0.1000 - moe_loss: 8603.5527

<div class="k-default-codeblock">
```

```
</div>
  83/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9793 - loss: 0.1000 - moe_loss: 8613.7930

<div class="k-default-codeblock">
```

```
</div>
  87/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9793 - loss: 0.1000 - moe_loss: 8624.0332

<div class="k-default-codeblock">
```

```
</div>
  90/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9794 - loss: 0.1000 - moe_loss: 8631.7139

<div class="k-default-codeblock">
```

```
</div>
  94/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9794 - loss: 0.1000 - moe_loss: 8641.9541

<div class="k-default-codeblock">
```

```
</div>
  98/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9795 - loss: 0.1000 - moe_loss: 8652.1943

<div class="k-default-codeblock">
```

```
</div>
 101/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9795 - loss: 0.1000 - moe_loss: 8659.8740

<div class="k-default-codeblock">
```

```
</div>
 104/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9796 - loss: 0.1000 - moe_loss: 8667.5547

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9797 - loss: 0.1000 - moe_loss: 8677.7939

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9797 - loss: 0.1000 - moe_loss: 8685.4736

<div class="k-default-codeblock">
```

```
</div>
 115/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9798 - loss: 0.1000 - moe_loss: 8695.7139

<div class="k-default-codeblock">
```

```
</div>
 119/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9799 - loss: 0.1000 - moe_loss: 8705.9541

<div class="k-default-codeblock">
```

```
</div>
 122/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9799 - loss: 0.1000 - moe_loss: 8713.6348

<div class="k-default-codeblock">
```

```
</div>
 126/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9800 - loss: 0.1000 - moe_loss: 8723.8750

<div class="k-default-codeblock">
```

```
</div>
 130/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9800 - loss: 0.1000 - moe_loss: 8734.1143

<div class="k-default-codeblock">
```

```
</div>
 134/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9801 - loss: 0.1000 - moe_loss: 8744.3545

<div class="k-default-codeblock">
```

```
</div>
 138/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9802 - loss: 0.1000 - moe_loss: 8754.5947

<div class="k-default-codeblock">
```

```
</div>
 142/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9802 - loss: 0.1000 - moe_loss: 8764.8350

<div class="k-default-codeblock">
```

```
</div>
 146/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9803 - loss: 0.1000 - moe_loss: 8775.0742

<div class="k-default-codeblock">
```

```
</div>
 149/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9803 - loss: 0.1000 - moe_loss: 8782.7549

<div class="k-default-codeblock">
```

```
</div>
 152/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9804 - loss: 0.1000 - moe_loss: 8790.4346

<div class="k-default-codeblock">
```

```
</div>
 156/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9805 - loss: 0.1000 - moe_loss: 8800.6738

<div class="k-default-codeblock">
```

```
</div>
 160/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9805 - loss: 0.1000 - moe_loss: 8810.9141

<div class="k-default-codeblock">
```

```
</div>
 164/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9806 - loss: 0.1000 - moe_loss: 8821.1533

<div class="k-default-codeblock">
```

```
</div>
 168/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9806 - loss: 0.1000 - moe_loss: 8831.3936

<div class="k-default-codeblock">
```

```
</div>
 172/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9806 - loss: 0.1000 - moe_loss: 8841.6328

<div class="k-default-codeblock">
```

```
</div>
 176/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9807 - loss: 0.1000 - moe_loss: 8851.8730

<div class="k-default-codeblock">
```

```
</div>
 180/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9807 - loss: 0.1000 - moe_loss: 8862.1123

<div class="k-default-codeblock">
```

```
</div>
 184/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9808 - loss: 0.1000 - moe_loss: 8872.3525

<div class="k-default-codeblock">
```

```
</div>
 188/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9808 - loss: 0.1000 - moe_loss: 8882.5928

<div class="k-default-codeblock">
```

```
</div>
 192/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9809 - loss: 0.1000 - moe_loss: 8892.8330

<div class="k-default-codeblock">
```

```
</div>
 196/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9809 - loss: 0.1000 - moe_loss: 8903.0732

<div class="k-default-codeblock">
```

```
</div>
 200/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9809 - loss: 0.1000 - moe_loss: 8913.3135

<div class="k-default-codeblock">
```

```
</div>
 204/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9810 - loss: 0.1000 - moe_loss: 8923.5537

<div class="k-default-codeblock">
```

```
</div>
 207/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9810 - loss: 0.1000 - moe_loss: 8931.2334

<div class="k-default-codeblock">
```

```
</div>
 210/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9810 - loss: 0.1000 - moe_loss: 8938.9131

<div class="k-default-codeblock">
```

```
</div>
 214/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9810 - loss: 0.1000 - moe_loss: 8949.1523

<div class="k-default-codeblock">
```

```
</div>
 218/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9811 - loss: 0.1000 - moe_loss: 8959.3926

<div class="k-default-codeblock">
```

```
</div>
 222/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 16ms/step - accuracy: 0.9811 - loss: 0.1000 - moe_loss: 8969.6318

<div class="k-default-codeblock">
```

```
</div>
 226/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 16ms/step - accuracy: 0.9811 - loss: 0.1000 - moe_loss: 8979.8721

<div class="k-default-codeblock">
```

```
</div>
 229/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 16ms/step - accuracy: 0.9811 - loss: 0.1000 - moe_loss: 8987.5518

<div class="k-default-codeblock">
```

```
</div>
 233/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 16ms/step - accuracy: 0.9812 - loss: 0.1000 - moe_loss: 8997.7920

<div class="k-default-codeblock">
```

```
</div>
 236/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 16ms/step - accuracy: 0.9812 - loss: 0.1000 - moe_loss: 9005.4717

<div class="k-default-codeblock">
```

```
</div>
 240/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 16ms/step - accuracy: 0.9812 - loss: 0.1000 - moe_loss: 9015.7119

<div class="k-default-codeblock">
```

```
</div>
 244/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 16ms/step - accuracy: 0.9812 - loss: 0.1000 - moe_loss: 9025.9521

<div class="k-default-codeblock">
```

```
</div>
 248/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 16ms/step - accuracy: 0.9813 - loss: 0.1000 - moe_loss: 9036.1914

<div class="k-default-codeblock">
```

```
</div>
 252/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 16ms/step - accuracy: 0.9813 - loss: 0.1000 - moe_loss: 9046.4316

<div class="k-default-codeblock">
```

```
</div>
 255/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 16ms/step - accuracy: 0.9813 - loss: 0.1000 - moe_loss: 9054.1113

<div class="k-default-codeblock">
```

```
</div>
 258/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9813 - loss: 0.1000 - moe_loss: 9061.7910

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9813 - loss: 0.1000 - moe_loss: 9072.0312

<div class="k-default-codeblock">
```

```
</div>
 266/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9814 - loss: 0.1000 - moe_loss: 9082.2715

<div class="k-default-codeblock">
```

```
</div>
 269/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9814 - loss: 0.1000 - moe_loss: 9089.9512

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9814 - loss: 0.1000 - moe_loss: 9100.1914

<div class="k-default-codeblock">
```

```
</div>
 277/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9814 - loss: 0.1000 - moe_loss: 9110.4307

<div class="k-default-codeblock">
```

```
</div>
 280/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 16ms/step - accuracy: 0.9814 - loss: 0.1000 - moe_loss: 9118.1113

<div class="k-default-codeblock">
```

```
</div>
 284/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 16ms/step - accuracy: 0.9814 - loss: 0.1000 - moe_loss: 9128.3516

<div class="k-default-codeblock">
```

```
</div>
 288/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 16ms/step - accuracy: 0.9815 - loss: 0.1000 - moe_loss: 9138.5908

<div class="k-default-codeblock">
```

```
</div>
 292/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 16ms/step - accuracy: 0.9815 - loss: 0.1000 - moe_loss: 9148.8311

<div class="k-default-codeblock">
```

```
</div>
 296/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 16ms/step - accuracy: 0.9815 - loss: 0.1000 - moe_loss: 9159.0713

<div class="k-default-codeblock">
```

```
</div>
 300/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 16ms/step - accuracy: 0.9815 - loss: 0.1000 - moe_loss: 9169.3105

<div class="k-default-codeblock">
```

```
</div>
 304/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 16ms/step - accuracy: 0.9816 - loss: 0.1000 - moe_loss: 9179.5508

<div class="k-default-codeblock">
```

```
</div>
 307/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9816 - loss: 0.1000 - moe_loss: 9187.2305

<div class="k-default-codeblock">
```

```
</div>
 311/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9816 - loss: 0.1000 - moe_loss: 9197.4707

<div class="k-default-codeblock">
```

```
</div>
 314/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9816 - loss: 0.1000 - moe_loss: 9205.1504

<div class="k-default-codeblock">
```

```
</div>
 318/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9816 - loss: 0.1000 - moe_loss: 9215.3906

<div class="k-default-codeblock">
```

```
</div>
 322/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9816 - loss: 0.1000 - moe_loss: 9225.6309

<div class="k-default-codeblock">
```

```
</div>
 326/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 16ms/step - accuracy: 0.9817 - loss: 0.1000 - moe_loss: 9235.8711

<div class="k-default-codeblock">
```

```
</div>
 329/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 16ms/step - accuracy: 0.9817 - loss: 0.1000 - moe_loss: 9243.5508

<div class="k-default-codeblock">
```

```
</div>
 332/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 16ms/step - accuracy: 0.9817 - loss: 0.1000 - moe_loss: 9251.2314

<div class="k-default-codeblock">
```

```
</div>
 336/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 16ms/step - accuracy: 0.9817 - loss: 0.1000 - moe_loss: 9261.4707

<div class="k-default-codeblock">
```

```
</div>
 340/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 16ms/step - accuracy: 0.9817 - loss: 0.1000 - moe_loss: 9271.7109

<div class="k-default-codeblock">
```

```
</div>
 344/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 16ms/step - accuracy: 0.9817 - loss: 0.1000 - moe_loss: 9281.9512

<div class="k-default-codeblock">
```

```
</div>
 348/469 ━━━━━━━━━━━━━━[37m━━━━━━  1s 16ms/step - accuracy: 0.9818 - loss: 0.1000 - moe_loss: 9292.1914

<div class="k-default-codeblock">
```

```
</div>
 351/469 ━━━━━━━━━━━━━━[37m━━━━━━  1s 16ms/step - accuracy: 0.9818 - loss: 0.1000 - moe_loss: 9299.8711

<div class="k-default-codeblock">
```

```
</div>
 355/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9818 - loss: 0.1000 - moe_loss: 9310.1113

<div class="k-default-codeblock">
```

```
</div>
 359/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9818 - loss: 0.1000 - moe_loss: 9320.3516

<div class="k-default-codeblock">
```

```
</div>
 363/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9818 - loss: 0.1000 - moe_loss: 9330.5908

<div class="k-default-codeblock">
```

```
</div>
 367/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9818 - loss: 0.1000 - moe_loss: 9340.8311

<div class="k-default-codeblock">
```

```
</div>
 370/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9819 - loss: 0.1000 - moe_loss: 9348.5107

<div class="k-default-codeblock">
```

```
</div>
 374/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 16ms/step - accuracy: 0.9819 - loss: 0.1000 - moe_loss: 9358.7510

<div class="k-default-codeblock">
```

```
</div>
 378/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 16ms/step - accuracy: 0.9819 - loss: 0.1000 - moe_loss: 9368.9912

<div class="k-default-codeblock">
```

```
</div>
 381/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 16ms/step - accuracy: 0.9819 - loss: 0.1000 - moe_loss: 9376.6709

<div class="k-default-codeblock">
```

```
</div>
 385/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 16ms/step - accuracy: 0.9819 - loss: 0.1000 - moe_loss: 9386.9111

<div class="k-default-codeblock">
```

```
</div>
 389/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 16ms/step - accuracy: 0.9819 - loss: 0.1000 - moe_loss: 9397.1514

<div class="k-default-codeblock">
```

```
</div>
 392/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 16ms/step - accuracy: 0.9819 - loss: 0.1000 - moe_loss: 9404.8311

<div class="k-default-codeblock">
```

```
</div>
 396/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 16ms/step - accuracy: 0.9819 - loss: 0.1000 - moe_loss: 9415.0713

<div class="k-default-codeblock">
```

```
</div>
 399/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 16ms/step - accuracy: 0.9819 - loss: 0.1000 - moe_loss: 9422.7510

<div class="k-default-codeblock">
```

```
</div>
 403/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9820 - loss: 0.1000 - moe_loss: 9432.9912

<div class="k-default-codeblock">
```

```
</div>
 406/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9820 - loss: 0.1000 - moe_loss: 9440.6709

<div class="k-default-codeblock">
```

```
</div>
 409/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9820 - loss: 0.1000 - moe_loss: 9448.3506

<div class="k-default-codeblock">
```

```
</div>
 413/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9820 - loss: 0.1000 - moe_loss: 9458.5908

<div class="k-default-codeblock">
```

```
</div>
 417/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9820 - loss: 0.1000 - moe_loss: 9468.8311

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9820 - loss: 0.1000 - moe_loss: 9479.0703

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9820 - loss: 0.1000 - moe_loss: 9489.3105

<div class="k-default-codeblock">
```

```
</div>
 429/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9820 - loss: 0.1000 - moe_loss: 9499.5498

<div class="k-default-codeblock">
```

```
</div>
 433/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9820 - loss: 0.1000 - moe_loss: 9509.7900

<div class="k-default-codeblock">
```

```
</div>
 437/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 9520.0303

<div class="k-default-codeblock">
```

```
</div>
 440/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 9527.7100

<div class="k-default-codeblock">
```

```
</div>
 443/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 9535.3896

<div class="k-default-codeblock">
```

```
</div>
 447/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 9545.6299

<div class="k-default-codeblock">
```

```
</div>
 450/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 9553.3096

<div class="k-default-codeblock">
```

```
</div>
 453/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 9560.9902

<div class="k-default-codeblock">
```

```
</div>
 456/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 9568.6699

<div class="k-default-codeblock">
```

```
</div>
 460/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 9578.9102

<div class="k-default-codeblock">
```

```
</div>
 463/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 9586.5898

<div class="k-default-codeblock">
```

```
</div>
 466/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 9594.2695

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 17ms/step - accuracy: 0.9822 - loss: 0.1000 - moe_loss: 9604.4932 - val_loss: 0.1000 - val_moe_loss: 11198.7256


<div class="k-default-codeblock">
```
Epoch 5/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 26ms/step - accuracy: 0.9688 - loss: 0.1000 - moe_loss: 11203.8506

<div class="k-default-codeblock">
```

```
</div>
   5/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9796 - loss: 0.1000 - moe_loss: 11214.1025 

<div class="k-default-codeblock">
```

```
</div>
   8/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9802 - loss: 0.1000 - moe_loss: 11221.7900

<div class="k-default-codeblock">
```

```
</div>
  11/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9807 - loss: 0.1000 - moe_loss: 11229.4717

<div class="k-default-codeblock">
```

```
</div>
  15/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9807 - loss: 0.1000 - moe_loss: 11239.7148

<div class="k-default-codeblock">
```

```
</div>
  18/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9809 - loss: 0.1000 - moe_loss: 11247.3945

<div class="k-default-codeblock">
```

```
</div>
  21/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9810 - loss: 0.1000 - moe_loss: 11255.0732

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9813 - loss: 0.1000 - moe_loss: 11265.3115

<div class="k-default-codeblock">
```

```
</div>
  29/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9813 - loss: 0.1000 - moe_loss: 11275.5498

<div class="k-default-codeblock">
```

```
</div>
  33/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9814 - loss: 0.1000 - moe_loss: 11285.7881

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9814 - loss: 0.1000 - moe_loss: 11296.0273

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9814 - loss: 0.1000 - moe_loss: 11303.7070

<div class="k-default-codeblock">
```

```
</div>
  42/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9815 - loss: 0.1000 - moe_loss: 11308.8281

<div class="k-default-codeblock">
```

```
</div>
  43/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 22ms/step - accuracy: 0.9815 - loss: 0.1000 - moe_loss: 11311.3887

<div class="k-default-codeblock">
```

```
</div>
  46/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9816 - loss: 0.1000 - moe_loss: 11319.0693

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9817 - loss: 0.1000 - moe_loss: 11326.7500

<div class="k-default-codeblock">
```

```
</div>
  53/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9818 - loss: 0.1000 - moe_loss: 11336.9912

<div class="k-default-codeblock">
```

```
</div>
  56/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9818 - loss: 0.1000 - moe_loss: 11344.6709

<div class="k-default-codeblock">
```

```
</div>
  60/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9820 - loss: 0.1000 - moe_loss: 11354.9111

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 11365.1504

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 11372.8301

<div class="k-default-codeblock">
```

```
</div>
  70/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 11380.5098

<div class="k-default-codeblock">
```

```
</div>
  74/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9822 - loss: 0.1000 - moe_loss: 11390.7490

<div class="k-default-codeblock">
```

```
</div>
  77/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9822 - loss: 0.1000 - moe_loss: 11398.4287

<div class="k-default-codeblock">
```

```
</div>
  81/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 19ms/step - accuracy: 0.9822 - loss: 0.1000 - moe_loss: 11408.6680

<div class="k-default-codeblock">
```

```
</div>
  84/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 19ms/step - accuracy: 0.9822 - loss: 0.1000 - moe_loss: 11416.3486

<div class="k-default-codeblock">
```

```
</div>
  87/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 19ms/step - accuracy: 0.9823 - loss: 0.1000 - moe_loss: 11424.0283

<div class="k-default-codeblock">
```

```
</div>
  90/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 19ms/step - accuracy: 0.9823 - loss: 0.1000 - moe_loss: 11431.7080

<div class="k-default-codeblock">
```

```
</div>
  94/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 19ms/step - accuracy: 0.9823 - loss: 0.1000 - moe_loss: 11441.9473

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 19ms/step - accuracy: 0.9823 - loss: 0.1000 - moe_loss: 11449.6270

<div class="k-default-codeblock">
```

```
</div>
 100/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9824 - loss: 0.1000 - moe_loss: 11457.3076

<div class="k-default-codeblock">
```

```
</div>
 104/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9824 - loss: 0.1000 - moe_loss: 11467.5479

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9824 - loss: 0.1000 - moe_loss: 11477.7871

<div class="k-default-codeblock">
```

```
</div>
 112/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9825 - loss: 0.1000 - moe_loss: 11488.0273

<div class="k-default-codeblock">
```

```
</div>
 116/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9825 - loss: 0.1000 - moe_loss: 11498.2666

<div class="k-default-codeblock">
```

```
</div>
 120/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9825 - loss: 0.1000 - moe_loss: 11508.5078

<div class="k-default-codeblock">
```

```
</div>
 124/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9826 - loss: 0.1000 - moe_loss: 11518.7480

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9826 - loss: 0.1000 - moe_loss: 11526.4277

<div class="k-default-codeblock">
```

```
</div>
 131/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9827 - loss: 0.1000 - moe_loss: 11536.6680

<div class="k-default-codeblock">
```

```
</div>
 134/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9827 - loss: 0.1000 - moe_loss: 11544.3477

<div class="k-default-codeblock">
```

```
</div>
 138/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9827 - loss: 0.1000 - moe_loss: 11554.5879

<div class="k-default-codeblock">
```

```
</div>
 142/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9828 - loss: 0.1000 - moe_loss: 11564.8271

<div class="k-default-codeblock">
```

```
</div>
 146/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9828 - loss: 0.1000 - moe_loss: 11575.0674

<div class="k-default-codeblock">
```

```
</div>
 150/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9829 - loss: 0.1000 - moe_loss: 11585.3066

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9829 - loss: 0.1000 - moe_loss: 11592.9873

<div class="k-default-codeblock">
```

```
</div>
 157/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9830 - loss: 0.1000 - moe_loss: 11603.2266

<div class="k-default-codeblock">
```

```
</div>
 161/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9830 - loss: 0.1000 - moe_loss: 11613.4668

<div class="k-default-codeblock">
```

```
</div>
 165/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9830 - loss: 0.1000 - moe_loss: 11623.7061

<div class="k-default-codeblock">
```

```
</div>
 169/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9831 - loss: 0.1000 - moe_loss: 11633.9463

<div class="k-default-codeblock">
```

```
</div>
 172/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9831 - loss: 0.1000 - moe_loss: 11641.6270

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9831 - loss: 0.1000 - moe_loss: 11649.3066

<div class="k-default-codeblock">
```

```
</div>
 179/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9832 - loss: 0.1000 - moe_loss: 11659.5459

<div class="k-default-codeblock">
```

```
</div>
 182/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9832 - loss: 0.1000 - moe_loss: 11667.2256

<div class="k-default-codeblock">
```

```
</div>
 185/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9832 - loss: 0.1000 - moe_loss: 11674.9062

<div class="k-default-codeblock">
```

```
</div>
 189/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9833 - loss: 0.1000 - moe_loss: 11685.1455

<div class="k-default-codeblock">
```

```
</div>
 193/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9833 - loss: 0.1000 - moe_loss: 11695.3857

<div class="k-default-codeblock">
```

```
</div>
 197/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9833 - loss: 0.1000 - moe_loss: 11705.6260

<div class="k-default-codeblock">
```

```
</div>
 201/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9834 - loss: 0.1000 - moe_loss: 11715.8662

<div class="k-default-codeblock">
```

```
</div>
 204/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9834 - loss: 0.1000 - moe_loss: 11723.5459

<div class="k-default-codeblock">
```

```
</div>
 208/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9834 - loss: 0.1000 - moe_loss: 11733.7861

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9834 - loss: 0.1000 - moe_loss: 11741.4658

<div class="k-default-codeblock">
```

```
</div>
 215/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9835 - loss: 0.1000 - moe_loss: 11751.7061

<div class="k-default-codeblock">
```

```
</div>
 218/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9835 - loss: 0.1000 - moe_loss: 11759.3857

<div class="k-default-codeblock">
```

```
</div>
 221/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9835 - loss: 0.1000 - moe_loss: 11767.0654

<div class="k-default-codeblock">
```

```
</div>
 225/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9835 - loss: 0.1000 - moe_loss: 11777.3057

<div class="k-default-codeblock">
```

```
</div>
 229/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9836 - loss: 0.1000 - moe_loss: 11787.5459

<div class="k-default-codeblock">
```

```
</div>
 233/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9836 - loss: 0.1000 - moe_loss: 11797.7861

<div class="k-default-codeblock">
```

```
</div>
 236/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9836 - loss: 0.1000 - moe_loss: 11805.4658

<div class="k-default-codeblock">
```

```
</div>
 240/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9836 - loss: 0.1000 - moe_loss: 11815.7051

<div class="k-default-codeblock">
```

```
</div>
 243/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9837 - loss: 0.1000 - moe_loss: 11823.3857

<div class="k-default-codeblock">
```

```
</div>
 246/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9837 - loss: 0.1000 - moe_loss: 11831.0654

<div class="k-default-codeblock">
```

```
</div>
 249/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9837 - loss: 0.1000 - moe_loss: 11838.7461

<div class="k-default-codeblock">
```

```
</div>
 253/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9837 - loss: 0.1000 - moe_loss: 11848.9854

<div class="k-default-codeblock">
```

```
</div>
 257/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9837 - loss: 0.1000 - moe_loss: 11859.2256

<div class="k-default-codeblock">
```

```
</div>
 260/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9838 - loss: 0.1000 - moe_loss: 11866.9053

<div class="k-default-codeblock">
```

```
</div>
 263/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9838 - loss: 0.1000 - moe_loss: 11874.5859

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9838 - loss: 0.1000 - moe_loss: 11884.8252

<div class="k-default-codeblock">
```

```
</div>
 270/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9838 - loss: 0.1000 - moe_loss: 11892.5059

<div class="k-default-codeblock">
```

```
</div>
 274/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9838 - loss: 0.1000 - moe_loss: 11902.7451

<div class="k-default-codeblock">
```

```
</div>
 278/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9839 - loss: 0.1000 - moe_loss: 11912.9854

<div class="k-default-codeblock">
```

```
</div>
 281/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9839 - loss: 0.1000 - moe_loss: 11920.6650

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9839 - loss: 0.1000 - moe_loss: 11930.9053

<div class="k-default-codeblock">
```

```
</div>
 288/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9839 - loss: 0.1000 - moe_loss: 11938.5850

<div class="k-default-codeblock">
```

```
</div>
 291/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9839 - loss: 0.1000 - moe_loss: 11946.2656

<div class="k-default-codeblock">
```

```
</div>
 294/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9839 - loss: 0.1000 - moe_loss: 11953.9453

<div class="k-default-codeblock">
```

```
</div>
 298/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9839 - loss: 0.1000 - moe_loss: 11964.1855

<div class="k-default-codeblock">
```

```
</div>
 302/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9839 - loss: 0.1000 - moe_loss: 11974.4248

<div class="k-default-codeblock">
```

```
</div>
 305/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 11982.1055

<div class="k-default-codeblock">
```

```
</div>
 308/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 11989.7852

<div class="k-default-codeblock">
```

```
</div>
 312/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 12000.0254

<div class="k-default-codeblock">
```

```
</div>
 315/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 12007.7051

<div class="k-default-codeblock">
```

```
</div>
 318/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 12015.3848

<div class="k-default-codeblock">
```

```
</div>
 321/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 12023.0654

<div class="k-default-codeblock">
```

```
</div>
 324/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 12030.7451

<div class="k-default-codeblock">
```

```
</div>
 327/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 12038.4248

<div class="k-default-codeblock">
```

```
</div>
 330/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 12046.1055

<div class="k-default-codeblock">
```

```
</div>
 333/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 12053.7852

<div class="k-default-codeblock">
```

```
</div>
 336/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 12061.4648

<div class="k-default-codeblock">
```

```
</div>
 340/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9840 - loss: 0.1000 - moe_loss: 12071.7051

<div class="k-default-codeblock">
```

```
</div>
 344/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12081.9453

<div class="k-default-codeblock">
```

```
</div>
 348/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12092.1846

<div class="k-default-codeblock">
```

```
</div>
 352/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12102.4248

<div class="k-default-codeblock">
```

```
</div>
 355/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12110.1055

<div class="k-default-codeblock">
```

```
</div>
 359/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12120.3447

<div class="k-default-codeblock">
```

```
</div>
 362/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12128.0254

<div class="k-default-codeblock">
```

```
</div>
 365/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12135.7051

<div class="k-default-codeblock">
```

```
</div>
 368/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12143.3848

<div class="k-default-codeblock">
```

```
</div>
 372/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12153.6250

<div class="k-default-codeblock">
```

```
</div>
 376/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12163.8652

<div class="k-default-codeblock">
```

```
</div>
 380/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12174.1055

<div class="k-default-codeblock">
```

```
</div>
 384/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12184.3447

<div class="k-default-codeblock">
```

```
</div>
 387/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12192.0254

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12202.2646

<div class="k-default-codeblock">
```

```
</div>
 395/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12212.5049

<div class="k-default-codeblock">
```

```
</div>
 399/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9841 - loss: 0.1000 - moe_loss: 12222.7451

<div class="k-default-codeblock">
```

```
</div>
 403/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12232.9854

<div class="k-default-codeblock">
```

```
</div>
 407/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12243.2256

<div class="k-default-codeblock">
```

```
</div>
 411/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12253.4658

<div class="k-default-codeblock">
```

```
</div>
 414/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12261.1455

<div class="k-default-codeblock">
```

```
</div>
 418/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12271.3857

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12279.0654

<div class="k-default-codeblock">
```

```
</div>
 424/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12286.7451

<div class="k-default-codeblock">
```

```
</div>
 427/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12294.4248

<div class="k-default-codeblock">
```

```
</div>
 431/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12304.6650

<div class="k-default-codeblock">
```

```
</div>
 434/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12312.3447

<div class="k-default-codeblock">
```

```
</div>
 438/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12322.5850

<div class="k-default-codeblock">
```

```
</div>
 441/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12330.2646

<div class="k-default-codeblock">
```

```
</div>
 445/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12340.5049

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12350.7451

<div class="k-default-codeblock">
```

```
</div>
 452/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12358.4248

<div class="k-default-codeblock">
```

```
</div>
 456/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12368.6650

<div class="k-default-codeblock">
```

```
</div>
 460/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12378.9053

<div class="k-default-codeblock">
```

```
</div>
 463/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9842 - loss: 0.1000 - moe_loss: 12386.5850

<div class="k-default-codeblock">
```

```
</div>
 467/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9843 - loss: 0.1000 - moe_loss: 12396.8252

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.9843 - loss: 0.1000 - moe_loss: 12404.4883 - val_loss: 0.1000 - val_moe_loss: 13998.7246


<div class="k-default-codeblock">
```
Epoch 6/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  13s 29ms/step - accuracy: 0.9609 - loss: 0.1000 - moe_loss: 14003.8555

<div class="k-default-codeblock">
```

```
</div>
   5/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9740 - loss: 0.1000 - moe_loss: 14014.0947 

<div class="k-default-codeblock">
```

```
</div>
   8/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9780 - loss: 0.1000 - moe_loss: 14021.7832

<div class="k-default-codeblock">
```

```
</div>
  12/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9807 - loss: 0.1000 - moe_loss: 14032.0244

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9821 - loss: 0.1000 - moe_loss: 14042.2637

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9827 - loss: 0.1000 - moe_loss: 14049.9434

<div class="k-default-codeblock">
```

```
</div>
  22/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9834 - loss: 0.1000 - moe_loss: 14057.6221

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9838 - loss: 0.1000 - moe_loss: 14065.3027

<div class="k-default-codeblock">
```

```
</div>
  29/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9843 - loss: 0.1000 - moe_loss: 14075.5439

<div class="k-default-codeblock">
```

```
</div>
  33/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9848 - loss: 0.1000 - moe_loss: 14085.7842

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9852 - loss: 0.1000 - moe_loss: 14096.0244

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9854 - loss: 0.1000 - moe_loss: 14103.7041

<div class="k-default-codeblock">
```

```
</div>
  43/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9856 - loss: 0.1000 - moe_loss: 14111.3838

<div class="k-default-codeblock">
```

```
</div>
  46/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9858 - loss: 0.1000 - moe_loss: 14119.0635

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9859 - loss: 0.1000 - moe_loss: 14126.7432

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9860 - loss: 0.1000 - moe_loss: 14134.4229

<div class="k-default-codeblock">
```

```
</div>
  55/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9861 - loss: 0.1000 - moe_loss: 14142.1035

<div class="k-default-codeblock">
```

```
</div>
  59/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9862 - loss: 0.1000 - moe_loss: 14152.3428

<div class="k-default-codeblock">
```

```
</div>
  62/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9863 - loss: 0.1000 - moe_loss: 14160.0225

<div class="k-default-codeblock">
```

```
</div>
  65/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9864 - loss: 0.1000 - moe_loss: 14167.7021

<div class="k-default-codeblock">
```

```
</div>
  68/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9865 - loss: 0.1000 - moe_loss: 14175.3818

<div class="k-default-codeblock">
```

```
</div>
  71/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9865 - loss: 0.1000 - moe_loss: 14183.0625

<div class="k-default-codeblock">
```

```
</div>
  74/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9866 - loss: 0.1000 - moe_loss: 14190.7422

<div class="k-default-codeblock">
```

```
</div>
  77/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9866 - loss: 0.1000 - moe_loss: 14198.4219

<div class="k-default-codeblock">
```

```
</div>
  80/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9867 - loss: 0.1000 - moe_loss: 14206.1016

<div class="k-default-codeblock">
```

```
</div>
  83/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9867 - loss: 0.1000 - moe_loss: 14213.7812

<div class="k-default-codeblock">
```

```
</div>
  86/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9868 - loss: 0.1000 - moe_loss: 14221.4619

<div class="k-default-codeblock">
```

```
</div>
  89/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9868 - loss: 0.1000 - moe_loss: 14229.1416

<div class="k-default-codeblock">
```

```
</div>
  92/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9869 - loss: 0.1000 - moe_loss: 14236.8223

<div class="k-default-codeblock">
```

```
</div>
  95/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9869 - loss: 0.1000 - moe_loss: 14244.5029

<div class="k-default-codeblock">
```

```
</div>
  98/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9869 - loss: 0.1000 - moe_loss: 14252.1826

<div class="k-default-codeblock">
```

```
</div>
 102/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9869 - loss: 0.1000 - moe_loss: 14262.4229

<div class="k-default-codeblock">
```

```
</div>
 106/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9869 - loss: 0.1000 - moe_loss: 14272.6621

<div class="k-default-codeblock">
```

```
</div>
 109/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9870 - loss: 0.1000 - moe_loss: 14280.3428

<div class="k-default-codeblock">
```

```
</div>
 112/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9870 - loss: 0.1000 - moe_loss: 14288.0225

<div class="k-default-codeblock">
```

```
</div>
 115/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9870 - loss: 0.1000 - moe_loss: 14295.7021

<div class="k-default-codeblock">
```

```
</div>
 118/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9870 - loss: 0.1000 - moe_loss: 14303.3818

<div class="k-default-codeblock">
```

```
</div>
 121/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9871 - loss: 0.1000 - moe_loss: 14311.0625

<div class="k-default-codeblock">
```

```
</div>
 124/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9871 - loss: 0.1000 - moe_loss: 14318.7422

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9871 - loss: 0.1000 - moe_loss: 14326.4229

<div class="k-default-codeblock">
```

```
</div>
 130/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9871 - loss: 0.1000 - moe_loss: 14334.1035

<div class="k-default-codeblock">
```

```
</div>
 133/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9871 - loss: 0.1000 - moe_loss: 14341.7832

<div class="k-default-codeblock">
```

```
</div>
 136/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9871 - loss: 0.1000 - moe_loss: 14349.4629

<div class="k-default-codeblock">
```

```
</div>
 140/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9871 - loss: 0.1000 - moe_loss: 14359.7031

<div class="k-default-codeblock">
```

```
</div>
 143/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 14367.3838

<div class="k-default-codeblock">
```

```
</div>
 146/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 14375.0635

<div class="k-default-codeblock">
```

```
</div>
 150/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 14385.3037

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 14392.9844

<div class="k-default-codeblock">
```

```
</div>
 156/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 14400.6641

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 14408.3447

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 14418.5840

<div class="k-default-codeblock">
```

```
</div>
 166/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 14426.2637

<div class="k-default-codeblock">
```

```
</div>
 169/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 14433.9443

<div class="k-default-codeblock">
```

```
</div>
 172/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 14441.6240

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14449.3037

<div class="k-default-codeblock">
```

```
</div>
 179/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14459.5430

<div class="k-default-codeblock">
```

```
</div>
 182/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14467.2236

<div class="k-default-codeblock">
```

```
</div>
 185/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14474.9033

<div class="k-default-codeblock">
```

```
</div>
 188/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14482.5830

<div class="k-default-codeblock">
```

```
</div>
 191/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14490.2627

<div class="k-default-codeblock">
```

```
</div>
 194/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14497.9434

<div class="k-default-codeblock">
```

```
</div>
 197/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14505.6230

<div class="k-default-codeblock">
```

```
</div>
 200/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14513.3037

<div class="k-default-codeblock">
```

```
</div>
 204/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14523.5430

<div class="k-default-codeblock">
```

```
</div>
 207/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14531.2236

<div class="k-default-codeblock">
```

```
</div>
 210/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14538.9033

<div class="k-default-codeblock">
```

```
</div>
 213/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14546.5840

<div class="k-default-codeblock">
```

```
</div>
 216/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14554.2637

<div class="k-default-codeblock">
```

```
</div>
 219/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14561.9434

<div class="k-default-codeblock">
```

```
</div>
 222/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14569.6230

<div class="k-default-codeblock">
```

```
</div>
 225/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14577.3037

<div class="k-default-codeblock">
```

```
</div>
 228/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14584.9834

<div class="k-default-codeblock">
```

```
</div>
 231/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14592.6631

<div class="k-default-codeblock">
```

```
</div>
 234/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14600.3428

<div class="k-default-codeblock">
```

```
</div>
 237/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14608.0234

<div class="k-default-codeblock">
```

```
</div>
 240/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14615.7031

<div class="k-default-codeblock">
```

```
</div>
 243/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14623.3828

<div class="k-default-codeblock">
```

```
</div>
 246/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14631.0625

<div class="k-default-codeblock">
```

```
</div>
 249/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14638.7432

<div class="k-default-codeblock">
```

```
</div>
 252/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14646.4229

<div class="k-default-codeblock">
```

```
</div>
 255/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14654.1025

<div class="k-default-codeblock">
```

```
</div>
 258/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14661.7832

<div class="k-default-codeblock">
```

```
</div>
 261/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14669.4629

<div class="k-default-codeblock">
```

```
</div>
 264/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14677.1426

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14684.8223

<div class="k-default-codeblock">
```

```
</div>
 270/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14692.5029

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14700.1826

<div class="k-default-codeblock">
```

```
</div>
 276/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14707.8623

<div class="k-default-codeblock">
```

```
</div>
 279/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14715.5420

<div class="k-default-codeblock">
```

```
</div>
 282/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14723.2227

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14730.9023

<div class="k-default-codeblock">
```

```
</div>
 288/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14738.5820

<div class="k-default-codeblock">
```

```
</div>
 291/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14746.2627

<div class="k-default-codeblock">
```

```
</div>
 294/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14753.9424

<div class="k-default-codeblock">
```

```
</div>
 297/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14761.6221

<div class="k-default-codeblock">
```

```
</div>
 300/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14769.3027

<div class="k-default-codeblock">
```

```
</div>
 303/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14776.9824

<div class="k-default-codeblock">
```

```
</div>
 306/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14784.6631

<div class="k-default-codeblock">
```

```
</div>
 309/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14792.3428

<div class="k-default-codeblock">
```

```
</div>
 312/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14800.0234

<div class="k-default-codeblock">
```

```
</div>
 315/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14807.7031

<div class="k-default-codeblock">
```

```
</div>
 318/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14815.3828

<div class="k-default-codeblock">
```

```
</div>
 321/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14823.0625

<div class="k-default-codeblock">
```

```
</div>
 324/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14830.7432

<div class="k-default-codeblock">
```

```
</div>
 327/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14838.4229

<div class="k-default-codeblock">
```

```
</div>
 330/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14846.1025

<div class="k-default-codeblock">
```

```
</div>
 333/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14853.7832

<div class="k-default-codeblock">
```

```
</div>
 336/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14861.4629

<div class="k-default-codeblock">
```

```
</div>
 339/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14869.1426

<div class="k-default-codeblock">
```

```
</div>
 342/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14876.8232

<div class="k-default-codeblock">
```

```
</div>
 345/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14884.5029

<div class="k-default-codeblock">
```

```
</div>
 348/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14892.1826

<div class="k-default-codeblock">
```

```
</div>
 351/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14899.8623

<div class="k-default-codeblock">
```

```
</div>
 354/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14907.5430

<div class="k-default-codeblock">
```

```
</div>
 357/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14915.2227

<div class="k-default-codeblock">
```

```
</div>
 360/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14922.9023

<div class="k-default-codeblock">
```

```
</div>
 363/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14930.5820

<div class="k-default-codeblock">
```

```
</div>
 367/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14940.8223

<div class="k-default-codeblock">
```

```
</div>
 370/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14948.5029

<div class="k-default-codeblock">
```

```
</div>
 373/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14956.1826

<div class="k-default-codeblock">
```

```
</div>
 376/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14963.8623

<div class="k-default-codeblock">
```

```
</div>
 379/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14971.5430

<div class="k-default-codeblock">
```

```
</div>
 382/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14979.2227

<div class="k-default-codeblock">
```

```
</div>
 385/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14986.9023

<div class="k-default-codeblock">
```

```
</div>
 388/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 14994.5830

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 15002.2627

<div class="k-default-codeblock">
```

```
</div>
 394/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 15009.9424

<div class="k-default-codeblock">
```

```
</div>
 398/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 15020.1826

<div class="k-default-codeblock">
```

```
</div>
 401/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 15027.8623

<div class="k-default-codeblock">
```

```
</div>
 404/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 15035.5420

<div class="k-default-codeblock">
```

```
</div>
 407/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 15043.2227

<div class="k-default-codeblock">
```

```
</div>
 410/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 15050.9023

<div class="k-default-codeblock">
```

```
</div>
 413/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 15058.5820

<div class="k-default-codeblock">
```

```
</div>
 416/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9873 - loss: 0.1000 - moe_loss: 15066.2627

<div class="k-default-codeblock">
```

```
</div>
 419/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15073.9424

<div class="k-default-codeblock">
```

```
</div>
 422/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15081.6221

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15089.3027

<div class="k-default-codeblock">
```

```
</div>
 427/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15094.4229

<div class="k-default-codeblock">
```

```
</div>
 431/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15104.6621

<div class="k-default-codeblock">
```

```
</div>
 434/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15112.3428

<div class="k-default-codeblock">
```

```
</div>
 437/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15120.0225

<div class="k-default-codeblock">
```

```
</div>
 440/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15127.7021

<div class="k-default-codeblock">
```

```
</div>
 443/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15135.3828

<div class="k-default-codeblock">
```

```
</div>
 446/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15143.0625

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15150.7422

<div class="k-default-codeblock">
```

```
</div>
 452/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15158.4219

<div class="k-default-codeblock">
```

```
</div>
 455/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15166.1025

<div class="k-default-codeblock">
```

```
</div>
 458/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15173.7822

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15181.4619

<div class="k-default-codeblock">
```

```
</div>
 464/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15189.1426

<div class="k-default-codeblock">
```

```
</div>
 467/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15196.8223

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 15204.4854 - val_loss: 0.1000 - val_moe_loss: 16798.7246


<div class="k-default-codeblock">
```
Epoch 7/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  13s 29ms/step - accuracy: 0.9844 - loss: 0.1000 - moe_loss: 16803.8555

<div class="k-default-codeblock">
```

```
</div>
   4/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9907 - loss: 0.1000 - moe_loss: 16811.5371 

<div class="k-default-codeblock">
```

```
</div>
   7/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 16819.2188

<div class="k-default-codeblock">
```

```
</div>
  10/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 16826.9043

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 16834.5898

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 16842.2676

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 16849.9531

<div class="k-default-codeblock">
```

```
</div>
  22/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 16857.6309

<div class="k-default-codeblock">
```

```
</div>
  23/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9907 - loss: 0.1000 - moe_loss: 16860.1934

<div class="k-default-codeblock">
```

```
</div>
  24/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9906 - loss: 0.1000 - moe_loss: 16862.7539

<div class="k-default-codeblock">
```

```
</div>
  26/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9904 - loss: 0.1000 - moe_loss: 16867.8711

<div class="k-default-codeblock">
```

```
</div>
  29/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9902 - loss: 0.1000 - moe_loss: 16875.5527

<div class="k-default-codeblock">
```

```
</div>
  33/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 24ms/step - accuracy: 0.9901 - loss: 0.1000 - moe_loss: 16885.7910

<div class="k-default-codeblock">
```

```
</div>
  36/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 23ms/step - accuracy: 0.9899 - loss: 0.1000 - moe_loss: 16893.4688

<div class="k-default-codeblock">
```

```
</div>
  39/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9898 - loss: 0.1000 - moe_loss: 16901.1484 

<div class="k-default-codeblock">
```

```
</div>
  42/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9897 - loss: 0.1000 - moe_loss: 16908.8281

<div class="k-default-codeblock">
```

```
</div>
  45/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 22ms/step - accuracy: 0.9896 - loss: 0.1000 - moe_loss: 16916.5078

<div class="k-default-codeblock">
```

```
</div>
  48/469 ━━[37m━━━━━━━━━━━━━━━━━━  9s 22ms/step - accuracy: 0.9895 - loss: 0.1000 - moe_loss: 16924.1875

<div class="k-default-codeblock">
```

```
</div>
  51/469 ━━[37m━━━━━━━━━━━━━━━━━━  9s 22ms/step - accuracy: 0.9895 - loss: 0.1000 - moe_loss: 16931.8691

<div class="k-default-codeblock">
```

```
</div>
  54/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 16939.5488

<div class="k-default-codeblock">
```

```
</div>
  57/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 16947.2285

<div class="k-default-codeblock">
```

```
</div>
  60/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 16954.9082

<div class="k-default-codeblock">
```

```
</div>
  63/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 16962.5898

<div class="k-default-codeblock">
```

```
</div>
  66/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9892 - loss: 0.1000 - moe_loss: 16970.2676

<div class="k-default-codeblock">
```

```
</div>
  69/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9892 - loss: 0.1000 - moe_loss: 16977.9473

<div class="k-default-codeblock">
```

```
</div>
  72/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9891 - loss: 0.1000 - moe_loss: 16985.6270

<div class="k-default-codeblock">
```

```
</div>
  75/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9891 - loss: 0.1000 - moe_loss: 16993.3086

<div class="k-default-codeblock">
```

```
</div>
  78/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9890 - loss: 0.1000 - moe_loss: 17000.9863

<div class="k-default-codeblock">
```

```
</div>
  81/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9890 - loss: 0.1000 - moe_loss: 17008.6660

<div class="k-default-codeblock">
```

```
</div>
  84/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9889 - loss: 0.1000 - moe_loss: 17016.3457

<div class="k-default-codeblock">
```

```
</div>
  87/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9889 - loss: 0.1000 - moe_loss: 17024.0273

<div class="k-default-codeblock">
```

```
</div>
  90/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9888 - loss: 0.1000 - moe_loss: 17031.7070

<div class="k-default-codeblock">
```

```
</div>
  93/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9888 - loss: 0.1000 - moe_loss: 17039.3867

<div class="k-default-codeblock">
```

```
</div>
  96/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9888 - loss: 0.1000 - moe_loss: 17047.0664

<div class="k-default-codeblock">
```

```
</div>
  99/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9887 - loss: 0.1000 - moe_loss: 17054.7461

<div class="k-default-codeblock">
```

```
</div>
 102/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 19ms/step - accuracy: 0.9887 - loss: 0.1000 - moe_loss: 17062.4277

<div class="k-default-codeblock">
```

```
</div>
 105/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 19ms/step - accuracy: 0.9887 - loss: 0.1000 - moe_loss: 17070.1074

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9887 - loss: 0.1000 - moe_loss: 17077.7871

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9886 - loss: 0.1000 - moe_loss: 17085.4668

<div class="k-default-codeblock">
```

```
</div>
 114/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9886 - loss: 0.1000 - moe_loss: 17093.1465

<div class="k-default-codeblock">
```

```
</div>
 117/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9886 - loss: 0.1000 - moe_loss: 17100.8262

<div class="k-default-codeblock">
```

```
</div>
 120/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9886 - loss: 0.1000 - moe_loss: 17108.5059

<div class="k-default-codeblock">
```

```
</div>
 123/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9886 - loss: 0.1000 - moe_loss: 17116.1855

<div class="k-default-codeblock">
```

```
</div>
 126/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9885 - loss: 0.1000 - moe_loss: 17123.8652

<div class="k-default-codeblock">
```

```
</div>
 129/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9885 - loss: 0.1000 - moe_loss: 17131.5449

<div class="k-default-codeblock">
```

```
</div>
 132/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9885 - loss: 0.1000 - moe_loss: 17139.2246

<div class="k-default-codeblock">
```

```
</div>
 135/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9885 - loss: 0.1000 - moe_loss: 17146.9043

<div class="k-default-codeblock">
```

```
</div>
 138/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9885 - loss: 0.1000 - moe_loss: 17154.5840

<div class="k-default-codeblock">
```

```
</div>
 141/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17162.2637

<div class="k-default-codeblock">
```

```
</div>
 144/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17169.9434

<div class="k-default-codeblock">
```

```
</div>
 147/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17177.6230

<div class="k-default-codeblock">
```

```
</div>
 150/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17185.3047

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17192.9844

<div class="k-default-codeblock">
```

```
</div>
 156/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17200.6641

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17208.3438

<div class="k-default-codeblock">
```

```
</div>
 162/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17216.0234

<div class="k-default-codeblock">
```

```
</div>
 165/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17223.7031

<div class="k-default-codeblock">
```

```
</div>
 168/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17231.3848

<div class="k-default-codeblock">
```

```
</div>
 171/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17239.0645

<div class="k-default-codeblock">
```

```
</div>
 174/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17246.7441

<div class="k-default-codeblock">
```

```
</div>
 177/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17254.4238

<div class="k-default-codeblock">
```

```
</div>
 180/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17262.1035

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17269.7832

<div class="k-default-codeblock">
```

```
</div>
 186/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17277.4629

<div class="k-default-codeblock">
```

```
</div>
 189/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17285.1445

<div class="k-default-codeblock">
```

```
</div>
 192/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17292.8242

<div class="k-default-codeblock">
```

```
</div>
 195/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17300.5039

<div class="k-default-codeblock">
```

```
</div>
 198/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17308.1836

<div class="k-default-codeblock">
```

```
</div>
 201/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 19ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17315.8633

<div class="k-default-codeblock">
```

```
</div>
 204/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 19ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17323.5430

<div class="k-default-codeblock">
```

```
</div>
 207/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 19ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17331.2227

<div class="k-default-codeblock">
```

```
</div>
 210/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 19ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17338.9023

<div class="k-default-codeblock">
```

```
</div>
 213/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 19ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17346.5820

<div class="k-default-codeblock">
```

```
</div>
 216/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 19ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17354.2617

<div class="k-default-codeblock">
```

```
</div>
 219/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 19ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17361.9434

<div class="k-default-codeblock">
```

```
</div>
 222/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17369.6230

<div class="k-default-codeblock">
```

```
</div>
 225/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17377.3027

<div class="k-default-codeblock">
```

```
</div>
 228/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17384.9824

<div class="k-default-codeblock">
```

```
</div>
 231/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17392.6621

<div class="k-default-codeblock">
```

```
</div>
 234/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17400.3438

<div class="k-default-codeblock">
```

```
</div>
 237/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17408.0234

<div class="k-default-codeblock">
```

```
</div>
 240/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17415.7031

<div class="k-default-codeblock">
```

```
</div>
 243/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17423.3828

<div class="k-default-codeblock">
```

```
</div>
 246/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17431.0625

<div class="k-default-codeblock">
```

```
</div>
 249/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17438.7422

<div class="k-default-codeblock">
```

```
</div>
 252/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17446.4219

<div class="k-default-codeblock">
```

```
</div>
 255/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17454.1035

<div class="k-default-codeblock">
```

```
</div>
 258/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17461.7832

<div class="k-default-codeblock">
```

```
</div>
 261/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17469.4629

<div class="k-default-codeblock">
```

```
</div>
 264/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17477.1426

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17484.8223

<div class="k-default-codeblock">
```

```
</div>
 270/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17492.5020

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17500.1816

<div class="k-default-codeblock">
```

```
</div>
 276/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17507.8633

<div class="k-default-codeblock">
```

```
</div>
 279/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17515.5430

<div class="k-default-codeblock">
```

```
</div>
 282/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17523.2227

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17530.9023

<div class="k-default-codeblock">
```

```
</div>
 288/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17538.5820

<div class="k-default-codeblock">
```

```
</div>
 291/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17546.2617

<div class="k-default-codeblock">
```

```
</div>
 294/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17553.9414

<div class="k-default-codeblock">
```

```
</div>
 297/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17561.6211

<div class="k-default-codeblock">
```

```
</div>
 300/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17569.3027

<div class="k-default-codeblock">
```

```
</div>
 303/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17576.9824

<div class="k-default-codeblock">
```

```
</div>
 306/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17584.6621

<div class="k-default-codeblock">
```

```
</div>
 309/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17592.3418

<div class="k-default-codeblock">
```

```
</div>
 312/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17600.0234

<div class="k-default-codeblock">
```

```
</div>
 315/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17607.7031

<div class="k-default-codeblock">
```

```
</div>
 318/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17615.3828

<div class="k-default-codeblock">
```

```
</div>
 321/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17623.0625

<div class="k-default-codeblock">
```

```
</div>
 324/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17630.7422

<div class="k-default-codeblock">
```

```
</div>
 327/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17638.4219

<div class="k-default-codeblock">
```

```
</div>
 330/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17646.1016

<div class="k-default-codeblock">
```

```
</div>
 333/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17653.7832

<div class="k-default-codeblock">
```

```
</div>
 336/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17661.4629

<div class="k-default-codeblock">
```

```
</div>
 339/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17669.1426

<div class="k-default-codeblock">
```

```
</div>
 342/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17676.8223

<div class="k-default-codeblock">
```

```
</div>
 345/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17684.5020

<div class="k-default-codeblock">
```

```
</div>
 348/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17692.1816

<div class="k-default-codeblock">
```

```
</div>
 351/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17699.8613

<div class="k-default-codeblock">
```

```
</div>
 354/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17707.5430

<div class="k-default-codeblock">
```

```
</div>
 357/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17715.2227

<div class="k-default-codeblock">
```

```
</div>
 360/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17722.9023

<div class="k-default-codeblock">
```

```
</div>
 363/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17730.5820

<div class="k-default-codeblock">
```

```
</div>
 366/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17738.2617

<div class="k-default-codeblock">
```

```
</div>
 369/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17745.9414

<div class="k-default-codeblock">
```

```
</div>
 372/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 17753.6230

<div class="k-default-codeblock">
```

```
</div>
 375/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17761.3027

<div class="k-default-codeblock">
```

```
</div>
 378/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17768.9824

<div class="k-default-codeblock">
```

```
</div>
 381/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17776.6621

<div class="k-default-codeblock">
```

```
</div>
 384/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17784.3418

<div class="k-default-codeblock">
```

```
</div>
 387/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17792.0215

<div class="k-default-codeblock">
```

```
</div>
 390/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17799.7012

<div class="k-default-codeblock">
```

```
</div>
 393/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17807.3828

<div class="k-default-codeblock">
```

```
</div>
 396/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17815.0625

<div class="k-default-codeblock">
```

```
</div>
 399/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17822.7422

<div class="k-default-codeblock">
```

```
</div>
 402/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17830.4219

<div class="k-default-codeblock">
```

```
</div>
 405/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17838.1016

<div class="k-default-codeblock">
```

```
</div>
 408/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17845.7832

<div class="k-default-codeblock">
```

```
</div>
 411/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17853.4629

<div class="k-default-codeblock">
```

```
</div>
 414/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17861.1426

<div class="k-default-codeblock">
```

```
</div>
 417/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17868.8223

<div class="k-default-codeblock">
```

```
</div>
 420/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17876.5020

<div class="k-default-codeblock">
```

```
</div>
 423/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17884.1816

<div class="k-default-codeblock">
```

```
</div>
 426/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17891.8633

<div class="k-default-codeblock">
```

```
</div>
 429/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17899.5430

<div class="k-default-codeblock">
```

```
</div>
 432/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 17907.2227

<div class="k-default-codeblock">
```

```
</div>
 435/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17914.9023

<div class="k-default-codeblock">
```

```
</div>
 438/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17922.5820

<div class="k-default-codeblock">
```

```
</div>
 441/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17930.2617

<div class="k-default-codeblock">
```

```
</div>
 444/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17937.9414

<div class="k-default-codeblock">
```

```
</div>
 447/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17945.6230

<div class="k-default-codeblock">
```

```
</div>
 450/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17953.3027

<div class="k-default-codeblock">
```

```
</div>
 453/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17960.9824

<div class="k-default-codeblock">
```

```
</div>
 456/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17968.6621

<div class="k-default-codeblock">
```

```
</div>
 459/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17976.3418

<div class="k-default-codeblock">
```

```
</div>
 462/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17984.0215

<div class="k-default-codeblock">
```

```
</div>
 465/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17991.7031

<div class="k-default-codeblock">
```

```
</div>
 468/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 17999.3828

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 9s 19ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 18004.4863 - val_loss: 0.1000 - val_moe_loss: 19598.7227


<div class="k-default-codeblock">
```
Epoch 8/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  13s 28ms/step - accuracy: 0.9844 - loss: 0.1000 - moe_loss: 19603.8477

<div class="k-default-codeblock">
```

```
</div>
   4/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 18ms/step - accuracy: 0.9844 - loss: 0.1000 - moe_loss: 19611.5293 

<div class="k-default-codeblock">
```

```
</div>
   7/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 18ms/step - accuracy: 0.9864 - loss: 0.1000 - moe_loss: 19619.2109

<div class="k-default-codeblock">
```

```
</div>
  10/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 18ms/step - accuracy: 0.9872 - loss: 0.1000 - moe_loss: 19626.8906

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 18ms/step - accuracy: 0.9877 - loss: 0.1000 - moe_loss: 19634.5723

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 18ms/step - accuracy: 0.9880 - loss: 0.1000 - moe_loss: 19642.2539

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 18ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 19649.9355

<div class="k-default-codeblock">
```

```
</div>
  22/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 18ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 19657.6152

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9884 - loss: 0.1000 - moe_loss: 19665.2969

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9886 - loss: 0.1000 - moe_loss: 19672.9785

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9887 - loss: 0.1000 - moe_loss: 19680.6582

<div class="k-default-codeblock">
```

```
</div>
  34/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9889 - loss: 0.1000 - moe_loss: 19688.3379

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9890 - loss: 0.1000 - moe_loss: 19696.0195

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9891 - loss: 0.1000 - moe_loss: 19703.6992

<div class="k-default-codeblock">
```

```
</div>
  43/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9892 - loss: 0.1000 - moe_loss: 19711.3809

<div class="k-default-codeblock">
```

```
</div>
  46/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 19719.0605

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 19726.7422

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 19734.4219

<div class="k-default-codeblock">
```

```
</div>
  55/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 19742.1016

<div class="k-default-codeblock">
```

```
</div>
  58/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 19749.7812

<div class="k-default-codeblock">
```

```
</div>
  61/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19757.4609

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19765.1406

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19772.8203

<div class="k-default-codeblock">
```

```
</div>
  70/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19780.5000

<div class="k-default-codeblock">
```

```
</div>
  73/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19788.1797

<div class="k-default-codeblock">
```

```
</div>
  76/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9895 - loss: 0.1000 - moe_loss: 19795.8594

<div class="k-default-codeblock">
```

```
</div>
  79/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 18ms/step - accuracy: 0.9895 - loss: 0.1000 - moe_loss: 19803.5391

<div class="k-default-codeblock">
```

```
</div>
  82/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9895 - loss: 0.1000 - moe_loss: 19811.2188

<div class="k-default-codeblock">
```

```
</div>
  85/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19818.9004

<div class="k-default-codeblock">
```

```
</div>
  88/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19826.5801

<div class="k-default-codeblock">
```

```
</div>
  91/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19834.2598

<div class="k-default-codeblock">
```

```
</div>
  94/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19841.9395

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19849.6191

<div class="k-default-codeblock">
```

```
</div>
 100/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19857.3008

<div class="k-default-codeblock">
```

```
</div>
 103/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19864.9805

<div class="k-default-codeblock">
```

```
</div>
 106/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19872.6602

<div class="k-default-codeblock">
```

```
</div>
 109/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19880.3398

<div class="k-default-codeblock">
```

```
</div>
 112/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19888.0195

<div class="k-default-codeblock">
```

```
</div>
 115/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19895.6992

<div class="k-default-codeblock">
```

```
</div>
 118/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19903.3809

<div class="k-default-codeblock">
```

```
</div>
 121/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19911.0605

<div class="k-default-codeblock">
```

```
</div>
 124/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19918.7402

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19926.4199

<div class="k-default-codeblock">
```

```
</div>
 130/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19934.0996

<div class="k-default-codeblock">
```

```
</div>
 133/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19941.7812

<div class="k-default-codeblock">
```

```
</div>
 136/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19949.4609

<div class="k-default-codeblock">
```

```
</div>
 139/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19957.1406

<div class="k-default-codeblock">
```

```
</div>
 142/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19964.8203

<div class="k-default-codeblock">
```

```
</div>
 145/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19972.5000

<div class="k-default-codeblock">
```

```
</div>
 148/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19980.1797

<div class="k-default-codeblock">
```

```
</div>
 151/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19987.8594

<div class="k-default-codeblock">
```

```
</div>
 154/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 19995.5391

<div class="k-default-codeblock">
```

```
</div>
 157/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20003.2207

<div class="k-default-codeblock">
```

```
</div>
 160/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20010.9004

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20018.5801

<div class="k-default-codeblock">
```

```
</div>
 166/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20026.2598

<div class="k-default-codeblock">
```

```
</div>
 169/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20033.9395

<div class="k-default-codeblock">
```

```
</div>
 172/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20041.6191

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20049.3008

<div class="k-default-codeblock">
```

```
</div>
 178/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20056.9805

<div class="k-default-codeblock">
```

```
</div>
 181/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20064.6602

<div class="k-default-codeblock">
```

```
</div>
 184/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20072.3398

<div class="k-default-codeblock">
```

```
</div>
 187/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20080.0195

<div class="k-default-codeblock">
```

```
</div>
 190/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20087.6992

<div class="k-default-codeblock">
```

```
</div>
 193/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20095.3789

<div class="k-default-codeblock">
```

```
</div>
 196/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20103.0586

<div class="k-default-codeblock">
```

```
</div>
 199/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20110.7402

<div class="k-default-codeblock">
```

```
</div>
 202/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20118.4199

<div class="k-default-codeblock">
```

```
</div>
 205/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20126.0996

<div class="k-default-codeblock">
```

```
</div>
 208/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20133.7793

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20141.4590

<div class="k-default-codeblock">
```

```
</div>
 214/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20149.1387

<div class="k-default-codeblock">
```

```
</div>
 217/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20156.8203

<div class="k-default-codeblock">
```

```
</div>
 220/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20164.5000

<div class="k-default-codeblock">
```

```
</div>
 223/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20172.1797

<div class="k-default-codeblock">
```

```
</div>
 226/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20179.8594

<div class="k-default-codeblock">
```

```
</div>
 229/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20187.5391

<div class="k-default-codeblock">
```

```
</div>
 232/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20195.2188

<div class="k-default-codeblock">
```

```
</div>
 235/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20202.8984

<div class="k-default-codeblock">
```

```
</div>
 238/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20210.5801

<div class="k-default-codeblock">
```

```
</div>
 241/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20218.2598

<div class="k-default-codeblock">
```

```
</div>
 244/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20225.9395

<div class="k-default-codeblock">
```

```
</div>
 247/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20233.6191

<div class="k-default-codeblock">
```

```
</div>
 250/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20241.2988

<div class="k-default-codeblock">
```

```
</div>
 253/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20248.9805

<div class="k-default-codeblock">
```

```
</div>
 256/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20256.6602

<div class="k-default-codeblock">
```

```
</div>
 259/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20264.3398

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20272.0195

<div class="k-default-codeblock">
```

```
</div>
 265/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20279.6992

<div class="k-default-codeblock">
```

```
</div>
 268/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20287.3789

<div class="k-default-codeblock">
```

```
</div>
 271/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20295.0586

<div class="k-default-codeblock">
```

```
</div>
 274/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20302.7402

<div class="k-default-codeblock">
```

```
</div>
 277/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20310.4199

<div class="k-default-codeblock">
```

```
</div>
 280/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20318.0996

<div class="k-default-codeblock">
```

```
</div>
 283/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20325.7793

<div class="k-default-codeblock">
```

```
</div>
 286/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20333.4590

<div class="k-default-codeblock">
```

```
</div>
 289/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20341.1387

<div class="k-default-codeblock">
```

```
</div>
 292/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20348.8203

<div class="k-default-codeblock">
```

```
</div>
 295/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20356.5000

<div class="k-default-codeblock">
```

```
</div>
 298/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20364.1797

<div class="k-default-codeblock">
```

```
</div>
 301/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20371.8594

<div class="k-default-codeblock">
```

```
</div>
 304/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20379.5391

<div class="k-default-codeblock">
```

```
</div>
 307/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20387.2188

<div class="k-default-codeblock">
```

```
</div>
 310/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20394.9004

<div class="k-default-codeblock">
```

```
</div>
 313/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20402.5801

<div class="k-default-codeblock">
```

```
</div>
 316/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20410.2598

<div class="k-default-codeblock">
```

```
</div>
 319/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20417.9395

<div class="k-default-codeblock">
```

```
</div>
 320/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20420.5000

<div class="k-default-codeblock">
```

```
</div>
 323/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20428.1797

<div class="k-default-codeblock">
```

```
</div>
 326/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20435.8594

<div class="k-default-codeblock">
```

```
</div>
 329/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20443.5391

<div class="k-default-codeblock">
```

```
</div>
 332/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20451.2188

<div class="k-default-codeblock">
```

```
</div>
 335/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20458.8984

<div class="k-default-codeblock">
```

```
</div>
 338/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20466.5781

<div class="k-default-codeblock">
```

```
</div>
 341/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20474.2598

<div class="k-default-codeblock">
```

```
</div>
 344/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20481.9395

<div class="k-default-codeblock">
```

```
</div>
 347/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20489.6191

<div class="k-default-codeblock">
```

```
</div>
 350/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20497.2988

<div class="k-default-codeblock">
```

```
</div>
 353/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20504.9785

<div class="k-default-codeblock">
```

```
</div>
 356/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20512.6582

<div class="k-default-codeblock">
```

```
</div>
 359/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20520.3398

<div class="k-default-codeblock">
```

```
</div>
 362/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 19ms/step - accuracy: 0.9894 - loss: 0.1000 - moe_loss: 20528.0195

<div class="k-default-codeblock">
```

```
</div>
 365/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20535.6992

<div class="k-default-codeblock">
```

```
</div>
 368/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20543.3789

<div class="k-default-codeblock">
```

```
</div>
 371/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20551.0586

<div class="k-default-codeblock">
```

```
</div>
 374/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20558.7383

<div class="k-default-codeblock">
```

```
</div>
 377/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20566.4180

<div class="k-default-codeblock">
```

```
</div>
 380/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20574.0996

<div class="k-default-codeblock">
```

```
</div>
 383/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20581.7793

<div class="k-default-codeblock">
```

```
</div>
 386/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20589.4590

<div class="k-default-codeblock">
```

```
</div>
 389/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20597.1387

<div class="k-default-codeblock">
```

```
</div>
 392/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20604.8184

<div class="k-default-codeblock">
```

```
</div>
 393/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20607.3789

<div class="k-default-codeblock">
```

```
</div>
 394/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20609.9395

<div class="k-default-codeblock">
```

```
</div>
 395/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20612.4980

<div class="k-default-codeblock">
```

```
</div>
 398/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20620.1797

<div class="k-default-codeblock">
```

```
</div>
 401/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20627.8594

<div class="k-default-codeblock">
```

```
</div>
 404/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20635.5391

<div class="k-default-codeblock">
```

```
</div>
 407/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20643.2188

<div class="k-default-codeblock">
```

```
</div>
 410/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20650.8984

<div class="k-default-codeblock">
```

```
</div>
 413/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20658.5781

<div class="k-default-codeblock">
```

```
</div>
 416/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20666.2578

<div class="k-default-codeblock">
```

```
</div>
 419/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 20ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20673.9395

<div class="k-default-codeblock">
```

```
</div>
 422/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 20ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20681.6191

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20689.2988

<div class="k-default-codeblock">
```

```
</div>
 428/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20696.9785

<div class="k-default-codeblock">
```

```
</div>
 431/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20704.6582

<div class="k-default-codeblock">
```

```
</div>
 434/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20712.3379

<div class="k-default-codeblock">
```

```
</div>
 437/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20720.0195

<div class="k-default-codeblock">
```

```
</div>
 440/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20727.6992

<div class="k-default-codeblock">
```

```
</div>
 443/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20735.3789

<div class="k-default-codeblock">
```

```
</div>
 446/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20743.0586

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20750.7383

<div class="k-default-codeblock">
```

```
</div>
 452/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20758.4180

<div class="k-default-codeblock">
```

```
</div>
 455/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20766.0996

<div class="k-default-codeblock">
```

```
</div>
 458/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20773.7793

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20781.4590

<div class="k-default-codeblock">
```

```
</div>
 464/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20789.1387

<div class="k-default-codeblock">
```

```
</div>
 467/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 19ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20796.8184

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 10s 20ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 20804.4824 - val_loss: 0.1000 - val_moe_loss: 22398.7227


<div class="k-default-codeblock">
```
Epoch 9/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  15s 33ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 22403.8516

<div class="k-default-codeblock">
```

```
</div>
   4/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9959 - loss: 0.1000 - moe_loss: 22411.5312 

<div class="k-default-codeblock">
```

```
</div>
   7/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9964 - loss: 0.1000 - moe_loss: 22419.2129

<div class="k-default-codeblock">
```

```
</div>
  10/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9967 - loss: 0.1000 - moe_loss: 22426.8926

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9965 - loss: 0.1000 - moe_loss: 22434.5742

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9960 - loss: 0.1000 - moe_loss: 22442.2539

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9954 - loss: 0.1000 - moe_loss: 22449.9355

<div class="k-default-codeblock">
```

```
</div>
  22/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9948 - loss: 0.1000 - moe_loss: 22457.6172

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 22465.2969

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 22472.9785

<div class="k-default-codeblock">
```

```
</div>
  30/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 22478.0977

<div class="k-default-codeblock">
```

```
</div>
  32/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 22483.2168

<div class="k-default-codeblock">
```

```
</div>
  35/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 22490.8965

<div class="k-default-codeblock">
```

```
</div>
  38/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 22498.5762

<div class="k-default-codeblock">
```

```
</div>
  41/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9927 - loss: 0.1000 - moe_loss: 22506.2559

<div class="k-default-codeblock">
```

```
</div>
  44/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 22513.9375

<div class="k-default-codeblock">
```

```
</div>
  47/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 22521.6172

<div class="k-default-codeblock">
```

```
</div>
  50/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 22529.2969

<div class="k-default-codeblock">
```

```
</div>
  53/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 22536.9766

<div class="k-default-codeblock">
```

```
</div>
  56/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 22544.6562

<div class="k-default-codeblock">
```

```
</div>
  59/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 22552.3359

<div class="k-default-codeblock">
```

```
</div>
  62/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 22560.0176

<div class="k-default-codeblock">
```

```
</div>
  65/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 22567.6973

<div class="k-default-codeblock">
```

```
</div>
  68/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 22575.3770

<div class="k-default-codeblock">
```

```
</div>
  71/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 22583.0566

<div class="k-default-codeblock">
```

```
</div>
  74/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 22590.7363

<div class="k-default-codeblock">
```

```
</div>
  77/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 22598.4180

<div class="k-default-codeblock">
```

```
</div>
  80/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 22606.0977

<div class="k-default-codeblock">
```

```
</div>
  83/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 22613.7793

<div class="k-default-codeblock">
```

```
</div>
  86/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 22621.4590

<div class="k-default-codeblock">
```

```
</div>
  89/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 22629.1387

<div class="k-default-codeblock">
```

```
</div>
  92/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 22636.8184

<div class="k-default-codeblock">
```

```
</div>
  95/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 22644.5000

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 22649.6191

<div class="k-default-codeblock">
```

```
</div>
  99/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 22654.7383

<div class="k-default-codeblock">
```

```
</div>
 102/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 22662.4199

<div class="k-default-codeblock">
```

```
</div>
 105/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 22670.0996

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 22677.7793

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 22685.4590

<div class="k-default-codeblock">
```

```
</div>
 114/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 22693.1387

<div class="k-default-codeblock">
```

```
</div>
 117/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 22700.8184

<div class="k-default-codeblock">
```

```
</div>
 120/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 22708.5000

<div class="k-default-codeblock">
```

```
</div>
 123/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 22716.1797

<div class="k-default-codeblock">
```

```
</div>
 126/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 22723.8594

<div class="k-default-codeblock">
```

```
</div>
 129/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 22731.5391

<div class="k-default-codeblock">
```

```
</div>
 132/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 22739.2188

<div class="k-default-codeblock">
```

```
</div>
 135/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 22746.9004

<div class="k-default-codeblock">
```

```
</div>
 138/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 22754.5801

<div class="k-default-codeblock">
```

```
</div>
 141/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 22762.2598

<div class="k-default-codeblock">
```

```
</div>
 144/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 22769.9395

<div class="k-default-codeblock">
```

```
</div>
 147/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 22777.6211

<div class="k-default-codeblock">
```

```
</div>
 150/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 22785.3008

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 22792.9805

<div class="k-default-codeblock">
```

```
</div>
 156/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 22800.6602

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 22808.3398

<div class="k-default-codeblock">
```

```
</div>
 162/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 22816.0195

<div class="k-default-codeblock">
```

```
</div>
 165/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 22823.7012

<div class="k-default-codeblock">
```

```
</div>
 168/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 22831.3809

<div class="k-default-codeblock">
```

```
</div>
 171/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 22839.0605

<div class="k-default-codeblock">
```

```
</div>
 174/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22846.7402

<div class="k-default-codeblock">
```

```
</div>
 177/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22854.4199

<div class="k-default-codeblock">
```

```
</div>
 180/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22862.1016

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22869.7812

<div class="k-default-codeblock">
```

```
</div>
 186/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22877.4609

<div class="k-default-codeblock">
```

```
</div>
 189/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22885.1406

<div class="k-default-codeblock">
```

```
</div>
 192/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22892.8203

<div class="k-default-codeblock">
```

```
</div>
 195/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22900.5000

<div class="k-default-codeblock">
```

```
</div>
 198/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22908.1797

<div class="k-default-codeblock">
```

```
</div>
 201/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22915.8594

<div class="k-default-codeblock">
```

```
</div>
 204/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22923.5410

<div class="k-default-codeblock">
```

```
</div>
 207/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 22931.2207

<div class="k-default-codeblock">
```

```
</div>
 210/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 22938.9004

<div class="k-default-codeblock">
```

```
</div>
 213/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 22946.5801

<div class="k-default-codeblock">
```

```
</div>
 216/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 22954.2598

<div class="k-default-codeblock">
```

```
</div>
 219/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 22961.9395

<div class="k-default-codeblock">
```

```
</div>
 222/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 22969.6191

<div class="k-default-codeblock">
```

```
</div>
 225/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 22977.3008

<div class="k-default-codeblock">
```

```
</div>
 228/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 22984.9805

<div class="k-default-codeblock">
```

```
</div>
 231/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 22992.6602

<div class="k-default-codeblock">
```

```
</div>
 234/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 23000.3398

<div class="k-default-codeblock">
```

```
</div>
 237/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 23008.0195

<div class="k-default-codeblock">
```

```
</div>
 240/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 23015.6992

<div class="k-default-codeblock">
```

```
</div>
 243/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 23023.3789

<div class="k-default-codeblock">
```

```
</div>
 246/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 23031.0605

<div class="k-default-codeblock">
```

```
</div>
 249/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 23038.7402

<div class="k-default-codeblock">
```

```
</div>
 252/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 23046.4199

<div class="k-default-codeblock">
```

```
</div>
 255/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 23054.0996

<div class="k-default-codeblock">
```

```
</div>
 258/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23061.7793

<div class="k-default-codeblock">
```

```
</div>
 261/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23069.4590

<div class="k-default-codeblock">
```

```
</div>
 264/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23077.1387

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23084.8203

<div class="k-default-codeblock">
```

```
</div>
 270/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23092.5000

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23100.1797

<div class="k-default-codeblock">
```

```
</div>
 276/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23107.8594

<div class="k-default-codeblock">
```

```
</div>
 279/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23115.5391

<div class="k-default-codeblock">
```

```
</div>
 282/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23123.2188

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23130.9004

<div class="k-default-codeblock">
```

```
</div>
 288/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23138.5801

<div class="k-default-codeblock">
```

```
</div>
 291/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23146.2598

<div class="k-default-codeblock">
```

```
</div>
 294/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23153.9395

<div class="k-default-codeblock">
```

```
</div>
 297/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23161.6191

<div class="k-default-codeblock">
```

```
</div>
 300/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23169.2988

<div class="k-default-codeblock">
```

```
</div>
 303/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23176.9785

<div class="k-default-codeblock">
```

```
</div>
 306/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23184.6602

<div class="k-default-codeblock">
```

```
</div>
 309/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 20ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 23192.3398

<div class="k-default-codeblock">
```

```
</div>
 312/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23200.0195

<div class="k-default-codeblock">
```

```
</div>
 315/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23207.6992

<div class="k-default-codeblock">
```

```
</div>
 318/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23215.3789

<div class="k-default-codeblock">
```

```
</div>
 321/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23223.0586

<div class="k-default-codeblock">
```

```
</div>
 324/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23230.7383

<div class="k-default-codeblock">
```

```
</div>
 327/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23238.4180

<div class="k-default-codeblock">
```

```
</div>
 330/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23246.0996

<div class="k-default-codeblock">
```

```
</div>
 333/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23253.7793

<div class="k-default-codeblock">
```

```
</div>
 336/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23261.4590

<div class="k-default-codeblock">
```

```
</div>
 339/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23269.1387

<div class="k-default-codeblock">
```

```
</div>
 342/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23276.8184

<div class="k-default-codeblock">
```

```
</div>
 345/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23284.4980

<div class="k-default-codeblock">
```

```
</div>
 348/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23292.1797

<div class="k-default-codeblock">
```

```
</div>
 351/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23299.8594

<div class="k-default-codeblock">
```

```
</div>
 354/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23307.5391

<div class="k-default-codeblock">
```

```
</div>
 357/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23315.2188

<div class="k-default-codeblock">
```

```
</div>
 360/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23322.8984

<div class="k-default-codeblock">
```

```
</div>
 363/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23330.5781

<div class="k-default-codeblock">
```

```
</div>
 365/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 23335.6992

<div class="k-default-codeblock">
```

```
</div>
 368/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23343.3789

<div class="k-default-codeblock">
```

```
</div>
 370/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23348.5000

<div class="k-default-codeblock">
```

```
</div>
 373/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23356.1797

<div class="k-default-codeblock">
```

```
</div>
 376/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23363.8594

<div class="k-default-codeblock">
```

```
</div>
 379/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23371.5391

<div class="k-default-codeblock">
```

```
</div>
 382/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23379.2188

<div class="k-default-codeblock">
```

```
</div>
 385/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23386.8984

<div class="k-default-codeblock">
```

```
</div>
 388/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23394.5801

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23402.2598

<div class="k-default-codeblock">
```

```
</div>
 394/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23409.9395

<div class="k-default-codeblock">
```

```
</div>
 397/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23417.6191

<div class="k-default-codeblock">
```

```
</div>
 400/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23425.2988

<div class="k-default-codeblock">
```

```
</div>
 403/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23432.9785

<div class="k-default-codeblock">
```

```
</div>
 406/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23440.6582

<div class="k-default-codeblock">
```

```
</div>
 409/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23448.3379

<div class="k-default-codeblock">
```

```
</div>
 412/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23456.0195

<div class="k-default-codeblock">
```

```
</div>
 415/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23463.6992

<div class="k-default-codeblock">
```

```
</div>
 418/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23471.3789

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23479.0586

<div class="k-default-codeblock">
```

```
</div>
 424/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 23486.7383

<div class="k-default-codeblock">
```

```
</div>
 427/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23494.4180

<div class="k-default-codeblock">
```

```
</div>
 430/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23502.0996

<div class="k-default-codeblock">
```

```
</div>
 433/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23509.7793

<div class="k-default-codeblock">
```

```
</div>
 436/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23517.4590

<div class="k-default-codeblock">
```

```
</div>
 439/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23525.1387

<div class="k-default-codeblock">
```

```
</div>
 442/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23532.8184

<div class="k-default-codeblock">
```

```
</div>
 445/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23540.4980

<div class="k-default-codeblock">
```

```
</div>
 448/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23548.1797

<div class="k-default-codeblock">
```

```
</div>
 451/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23555.8594

<div class="k-default-codeblock">
```

```
</div>
 454/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23563.5391

<div class="k-default-codeblock">
```

```
</div>
 457/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23571.2188

<div class="k-default-codeblock">
```

```
</div>
 460/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23578.8984

<div class="k-default-codeblock">
```

```
</div>
 463/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23586.5781

<div class="k-default-codeblock">
```

```
</div>
 466/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23594.2578

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23601.9355

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 10s 21ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 23604.4824 - val_loss: 0.1000 - val_moe_loss: 25198.7246


<div class="k-default-codeblock">
```
Epoch 10/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  15s 33ms/step - accuracy: 1.0000 - loss: 0.1000 - moe_loss: 25203.8496

<div class="k-default-codeblock">
```

```
</div>
   4/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 20ms/step - accuracy: 0.9963 - loss: 0.1000 - moe_loss: 25211.5312 

<div class="k-default-codeblock">
```

```
</div>
   7/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 20ms/step - accuracy: 0.9955 - loss: 0.1000 - moe_loss: 25219.2148

<div class="k-default-codeblock">
```

```
</div>
  10/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 20ms/step - accuracy: 0.9951 - loss: 0.1000 - moe_loss: 25226.8945

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 20ms/step - accuracy: 0.9948 - loss: 0.1000 - moe_loss: 25234.5781

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 20ms/step - accuracy: 0.9946 - loss: 0.1000 - moe_loss: 25242.2578

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 25249.9375

<div class="k-default-codeblock">
```

```
</div>
  22/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 25257.6191

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 25265.2988

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 25272.9785

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 25280.6582

<div class="k-default-codeblock">
```

```
</div>
  34/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 25288.3379

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 25296.0176

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 25303.6992

<div class="k-default-codeblock">
```

```
</div>
  43/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 25311.3789

<div class="k-default-codeblock">
```

```
</div>
  46/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 25319.0586

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 25326.7383

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 25334.4160

<div class="k-default-codeblock">
```

```
</div>
  55/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 25342.0977

<div class="k-default-codeblock">
```

```
</div>
  58/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 25349.7773

<div class="k-default-codeblock">
```

```
</div>
  61/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9927 - loss: 0.1000 - moe_loss: 25357.4570

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9926 - loss: 0.1000 - moe_loss: 25365.1367

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 25372.8164

<div class="k-default-codeblock">
```

```
</div>
  70/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 25380.4961

<div class="k-default-codeblock">
```

```
</div>
  73/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 25388.1758

<div class="k-default-codeblock">
```

```
</div>
  76/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 25395.8555

<div class="k-default-codeblock">
```

```
</div>
  79/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 25403.5352

<div class="k-default-codeblock">
```

```
</div>
  82/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 25411.2148

<div class="k-default-codeblock">
```

```
</div>
  85/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 25418.8965

<div class="k-default-codeblock">
```

```
</div>
  88/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 25426.5762

<div class="k-default-codeblock">
```

```
</div>
  91/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 25434.2559

<div class="k-default-codeblock">
```

```
</div>
  94/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 25441.9355

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 25449.6152

<div class="k-default-codeblock">
```

```
</div>
 100/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 25457.2949

<div class="k-default-codeblock">
```

```
</div>
 103/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 25464.9766

<div class="k-default-codeblock">
```

```
</div>
 106/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 25472.6562

<div class="k-default-codeblock">
```

```
</div>
 109/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 25480.3359

<div class="k-default-codeblock">
```

```
</div>
 112/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 25488.0156

<div class="k-default-codeblock">
```

```
</div>
 115/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 25495.6953

<div class="k-default-codeblock">
```

```
</div>
 118/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 25503.3770

<div class="k-default-codeblock">
```

```
</div>
 121/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 25511.0566

<div class="k-default-codeblock">
```

```
</div>
 124/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 25518.7363

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 25526.4160

<div class="k-default-codeblock">
```

```
</div>
 130/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 25534.0977

<div class="k-default-codeblock">
```

```
</div>
 133/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 25541.7773

<div class="k-default-codeblock">
```

```
</div>
 136/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 25549.4570

<div class="k-default-codeblock">
```

```
</div>
 139/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 25557.1367

<div class="k-default-codeblock">
```

```
</div>
 142/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 25564.8164

<div class="k-default-codeblock">
```

```
</div>
 145/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 25572.4961

<div class="k-default-codeblock">
```

```
</div>
 148/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25580.1758

<div class="k-default-codeblock">
```

```
</div>
 151/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25587.8574

<div class="k-default-codeblock">
```

```
</div>
 154/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25595.5371

<div class="k-default-codeblock">
```

```
</div>
 157/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25603.2168

<div class="k-default-codeblock">
```

```
</div>
 160/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25610.8965

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25618.5762

<div class="k-default-codeblock">
```

```
</div>
 166/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25626.2559

<div class="k-default-codeblock">
```

```
</div>
 169/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25633.9375

<div class="k-default-codeblock">
```

```
</div>
 172/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25641.6172

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25649.2969

<div class="k-default-codeblock">
```

```
</div>
 178/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25656.9766

<div class="k-default-codeblock">
```

```
</div>
 181/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25664.6562

<div class="k-default-codeblock">
```

```
</div>
 184/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25672.3359

<div class="k-default-codeblock">
```

```
</div>
 187/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 25680.0156

<div class="k-default-codeblock">
```

```
</div>
 190/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25687.6973

<div class="k-default-codeblock">
```

```
</div>
 193/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25695.3770

<div class="k-default-codeblock">
```

```
</div>
 196/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25703.0566

<div class="k-default-codeblock">
```

```
</div>
 199/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25710.7363

<div class="k-default-codeblock">
```

```
</div>
 202/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25718.4160

<div class="k-default-codeblock">
```

```
</div>
 205/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25726.0957

<div class="k-default-codeblock">
```

```
</div>
 208/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25733.7773

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25741.4570

<div class="k-default-codeblock">
```

```
</div>
 214/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25749.1367

<div class="k-default-codeblock">
```

```
</div>
 217/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25756.8164

<div class="k-default-codeblock">
```

```
</div>
 220/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25764.4961

<div class="k-default-codeblock">
```

```
</div>
 223/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25772.1758

<div class="k-default-codeblock">
```

```
</div>
 226/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25779.8574

<div class="k-default-codeblock">
```

```
</div>
 229/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25787.5371

<div class="k-default-codeblock">
```

```
</div>
 232/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25795.2168

<div class="k-default-codeblock">
```

```
</div>
 235/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25802.8965

<div class="k-default-codeblock">
```

```
</div>
 238/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25810.5762

<div class="k-default-codeblock">
```

```
</div>
 241/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25818.2559

<div class="k-default-codeblock">
```

```
</div>
 244/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25825.9355

<div class="k-default-codeblock">
```

```
</div>
 247/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25833.6172

<div class="k-default-codeblock">
```

```
</div>
 250/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25841.2969

<div class="k-default-codeblock">
```

```
</div>
 253/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25848.9766

<div class="k-default-codeblock">
```

```
</div>
 256/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25856.6562

<div class="k-default-codeblock">
```

```
</div>
 259/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25864.3359

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25872.0176

<div class="k-default-codeblock">
```

```
</div>
 265/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25879.6973

<div class="k-default-codeblock">
```

```
</div>
 268/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25887.3770

<div class="k-default-codeblock">
```

```
</div>
 271/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25895.0566

<div class="k-default-codeblock">
```

```
</div>
 274/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25902.7363

<div class="k-default-codeblock">
```

```
</div>
 277/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25910.4160

<div class="k-default-codeblock">
```

```
</div>
 280/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25918.0977

<div class="k-default-codeblock">
```

```
</div>
 283/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25925.7773

<div class="k-default-codeblock">
```

```
</div>
 286/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25933.4570

<div class="k-default-codeblock">
```

```
</div>
 288/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 21ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25938.5762

<div class="k-default-codeblock">
```

```
</div>
 289/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25941.1367

<div class="k-default-codeblock">
```

```
</div>
 290/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25943.6973

<div class="k-default-codeblock">
```

```
</div>
 293/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25951.3770

<div class="k-default-codeblock">
```

```
</div>
 296/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25959.0566

<div class="k-default-codeblock">
```

```
</div>
 299/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25966.7363

<div class="k-default-codeblock">
```

```
</div>
 302/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25974.4160

<div class="k-default-codeblock">
```

```
</div>
 305/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25982.0977

<div class="k-default-codeblock">
```

```
</div>
 308/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 25989.7773

<div class="k-default-codeblock">
```

```
</div>
 311/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 25997.4570

<div class="k-default-codeblock">
```

```
</div>
 314/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26005.1367

<div class="k-default-codeblock">
```

```
</div>
 317/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26012.8164

<div class="k-default-codeblock">
```

```
</div>
 320/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26020.4961

<div class="k-default-codeblock">
```

```
</div>
 323/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26028.1777

<div class="k-default-codeblock">
```

```
</div>
 326/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26035.8574

<div class="k-default-codeblock">
```

```
</div>
 329/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26043.5371

<div class="k-default-codeblock">
```

```
</div>
 332/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26051.2168

<div class="k-default-codeblock">
```

```
</div>
 335/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26058.8965

<div class="k-default-codeblock">
```

```
</div>
 338/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26066.5781

<div class="k-default-codeblock">
```

```
</div>
 341/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26074.2578

<div class="k-default-codeblock">
```

```
</div>
 344/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26081.9375

<div class="k-default-codeblock">
```

```
</div>
 347/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26089.6172

<div class="k-default-codeblock">
```

```
</div>
 350/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26097.2969

<div class="k-default-codeblock">
```

```
</div>
 353/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26104.9766

<div class="k-default-codeblock">
```

```
</div>
 356/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26112.6562

<div class="k-default-codeblock">
```

```
</div>
 359/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26120.3379

<div class="k-default-codeblock">
```

```
</div>
 362/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26128.0176

<div class="k-default-codeblock">
```

```
</div>
 365/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26135.6973

<div class="k-default-codeblock">
```

```
</div>
 368/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26143.3770

<div class="k-default-codeblock">
```

```
</div>
 371/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26151.0566

<div class="k-default-codeblock">
```

```
</div>
 374/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26158.7363

<div class="k-default-codeblock">
```

```
</div>
 377/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26166.4180

<div class="k-default-codeblock">
```

```
</div>
 380/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26174.0977

<div class="k-default-codeblock">
```

```
</div>
 383/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26181.7773

<div class="k-default-codeblock">
```

```
</div>
 385/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26186.8965

<div class="k-default-codeblock">
```

```
</div>
 388/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26194.5762

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26202.2578

<div class="k-default-codeblock">
```

```
</div>
 394/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26209.9375

<div class="k-default-codeblock">
```

```
</div>
 397/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26217.6172

<div class="k-default-codeblock">
```

```
</div>
 400/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26225.2969

<div class="k-default-codeblock">
```

```
</div>
 403/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26232.9766

<div class="k-default-codeblock">
```

```
</div>
 406/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26240.6562

<div class="k-default-codeblock">
```

```
</div>
 409/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26248.3379

<div class="k-default-codeblock">
```

```
</div>
 412/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26256.0176

<div class="k-default-codeblock">
```

```
</div>
 415/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26263.6973

<div class="k-default-codeblock">
```

```
</div>
 418/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26271.3770

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26279.0566

<div class="k-default-codeblock">
```

```
</div>
 424/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26286.7363

<div class="k-default-codeblock">
```

```
</div>
 427/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26294.4180

<div class="k-default-codeblock">
```

```
</div>
 430/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26302.0977

<div class="k-default-codeblock">
```

```
</div>
 433/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26309.7773

<div class="k-default-codeblock">
```

```
</div>
 436/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26317.4570

<div class="k-default-codeblock">
```

```
</div>
 439/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26325.1367

<div class="k-default-codeblock">
```

```
</div>
 442/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26332.8164

<div class="k-default-codeblock">
```

```
</div>
 445/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26340.4980

<div class="k-default-codeblock">
```

```
</div>
 448/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26348.1777

<div class="k-default-codeblock">
```

```
</div>
 451/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26355.8574

<div class="k-default-codeblock">
```

```
</div>
 453/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26360.9766

<div class="k-default-codeblock">
```

```
</div>
 456/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26368.6562

<div class="k-default-codeblock">
```

```
</div>
 459/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26376.3379

<div class="k-default-codeblock">
```

```
</div>
 462/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26384.0176

<div class="k-default-codeblock">
```

```
</div>
 465/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26391.6973

<div class="k-default-codeblock">
```

```
</div>
 467/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 22ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26396.8164

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 11s 23ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 26404.4805 - val_loss: 0.1000 - val_moe_loss: 27998.7227


<div class="k-default-codeblock">
```
Epoch 11/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  18s 39ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28003.8535

<div class="k-default-codeblock">
```

```
</div>
   4/469 [37m━━━━━━━━━━━━━━━━━━━━  10s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 28011.5312

<div class="k-default-codeblock">
```

```
</div>
   7/469 [37m━━━━━━━━━━━━━━━━━━━━  10s 22ms/step - accuracy: 0.9948 - loss: 0.1000 - moe_loss: 28019.2109

<div class="k-default-codeblock">
```

```
</div>
  10/469 [37m━━━━━━━━━━━━━━━━━━━━  10s 23ms/step - accuracy: 0.9955 - loss: 0.1000 - moe_loss: 28026.8926

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  10s 23ms/step - accuracy: 0.9954 - loss: 0.1000 - moe_loss: 28034.5723

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  10s 23ms/step - accuracy: 0.9952 - loss: 0.1000 - moe_loss: 28042.2539

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  10s 23ms/step - accuracy: 0.9950 - loss: 0.1000 - moe_loss: 28049.9336

<div class="k-default-codeblock">
```

```
</div>
  22/469 [37m━━━━━━━━━━━━━━━━━━━━  10s 23ms/step - accuracy: 0.9947 - loss: 0.1000 - moe_loss: 28057.6152

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 23ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 28065.2949

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 28072.9766 

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 28080.6562

<div class="k-default-codeblock">
```

```
</div>
  34/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 28088.3359

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 28096.0156

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 28103.6953

<div class="k-default-codeblock">
```

```
</div>
  43/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 28111.3750

<div class="k-default-codeblock">
```

```
</div>
  46/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 28119.0566

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 28126.7363

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  9s 22ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 28134.4160

<div class="k-default-codeblock">
```

```
</div>
  55/469 ━━[37m━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 28142.0957

<div class="k-default-codeblock">
```

```
</div>
  58/469 ━━[37m━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 28149.7754

<div class="k-default-codeblock">
```

```
</div>
  61/469 ━━[37m━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 28157.4551

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 28165.1348

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 28172.8145

<div class="k-default-codeblock">
```

```
</div>
  70/469 ━━[37m━━━━━━━━━━━━━━━━━━  9s 23ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 28180.4941

<div class="k-default-codeblock">
```

```
</div>
  73/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 28188.1738

<div class="k-default-codeblock">
```

```
</div>
  76/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 28195.8555

<div class="k-default-codeblock">
```

```
</div>
  79/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 28203.5352

<div class="k-default-codeblock">
```

```
</div>
  82/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 28211.2148

<div class="k-default-codeblock">
```

```
</div>
  85/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 28218.8945

<div class="k-default-codeblock">
```

```
</div>
  88/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 28226.5762

<div class="k-default-codeblock">
```

```
</div>
  91/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 28234.2559

<div class="k-default-codeblock">
```

```
</div>
  94/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 28241.9355

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 28249.6172

<div class="k-default-codeblock">
```

```
</div>
 100/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 28257.2969

<div class="k-default-codeblock">
```

```
</div>
 102/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 28262.4160

<div class="k-default-codeblock">
```

```
</div>
 105/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 28270.0957

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 28277.7773

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 28285.4570

<div class="k-default-codeblock">
```

```
</div>
 114/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 23ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 28293.1367

<div class="k-default-codeblock">
```

```
</div>
 117/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 28300.8164

<div class="k-default-codeblock">
```

```
</div>
 120/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 28308.4961

<div class="k-default-codeblock">
```

```
</div>
 123/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 28316.1758

<div class="k-default-codeblock">
```

```
</div>
 126/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 28323.8574

<div class="k-default-codeblock">
```

```
</div>
 129/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9927 - loss: 0.1000 - moe_loss: 28331.5371

<div class="k-default-codeblock">
```

```
</div>
 132/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9927 - loss: 0.1000 - moe_loss: 28339.2168

<div class="k-default-codeblock">
```

```
</div>
 135/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9927 - loss: 0.1000 - moe_loss: 28346.8965

<div class="k-default-codeblock">
```

```
</div>
 138/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9927 - loss: 0.1000 - moe_loss: 28354.5762

<div class="k-default-codeblock">
```

```
</div>
 141/469 ━━━━━━[37m━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9926 - loss: 0.1000 - moe_loss: 28362.2559

<div class="k-default-codeblock">
```

```
</div>
 144/469 ━━━━━━[37m━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9926 - loss: 0.1000 - moe_loss: 28369.9355

<div class="k-default-codeblock">
```

```
</div>
 147/469 ━━━━━━[37m━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9926 - loss: 0.1000 - moe_loss: 28377.6172

<div class="k-default-codeblock">
```

```
</div>
 150/469 ━━━━━━[37m━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9926 - loss: 0.1000 - moe_loss: 28385.2969

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9926 - loss: 0.1000 - moe_loss: 28392.9766

<div class="k-default-codeblock">
```

```
</div>
 156/469 ━━━━━━[37m━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9926 - loss: 0.1000 - moe_loss: 28400.6562

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  7s 23ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 28408.3359

<div class="k-default-codeblock">
```

```
</div>
 162/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 28416.0156

<div class="k-default-codeblock">
```

```
</div>
 165/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 28423.6953

<div class="k-default-codeblock">
```

```
</div>
 168/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 28431.3770

<div class="k-default-codeblock">
```

```
</div>
 171/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 28439.0566

<div class="k-default-codeblock">
```

```
</div>
 174/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 28446.7363

<div class="k-default-codeblock">
```

```
</div>
 177/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 28454.4160

<div class="k-default-codeblock">
```

```
</div>
 179/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 28459.5371

<div class="k-default-codeblock">
```

```
</div>
 181/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 28464.6562

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 28469.7773

<div class="k-default-codeblock">
```

```
</div>
 185/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 28474.8965

<div class="k-default-codeblock">
```

```
</div>
 188/469 ━━━━━━━━[37m━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 28482.5762

<div class="k-default-codeblock">
```

```
</div>
 191/469 ━━━━━━━━[37m━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 28490.2578

<div class="k-default-codeblock">
```

```
</div>
 194/469 ━━━━━━━━[37m━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 28497.9375

<div class="k-default-codeblock">
```

```
</div>
 197/469 ━━━━━━━━[37m━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 28505.6172

<div class="k-default-codeblock">
```

```
</div>
 200/469 ━━━━━━━━[37m━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 28513.2969

<div class="k-default-codeblock">
```

```
</div>
 203/469 ━━━━━━━━[37m━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28520.9766

<div class="k-default-codeblock">
```

```
</div>
 206/469 ━━━━━━━━[37m━━━━━━━━━━━━  6s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28528.6562

<div class="k-default-codeblock">
```

```
</div>
 209/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28536.3379

<div class="k-default-codeblock">
```

```
</div>
 212/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28544.0176

<div class="k-default-codeblock">
```

```
</div>
 215/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28551.6973

<div class="k-default-codeblock">
```

```
</div>
 218/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28559.3770

<div class="k-default-codeblock">
```

```
</div>
 221/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28567.0566

<div class="k-default-codeblock">
```

```
</div>
 224/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28574.7363

<div class="k-default-codeblock">
```

```
</div>
 227/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28582.4160

<div class="k-default-codeblock">
```

```
</div>
 230/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28590.0977

<div class="k-default-codeblock">
```

```
</div>
 233/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28597.7773

<div class="k-default-codeblock">
```

```
</div>
 236/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28605.4570

<div class="k-default-codeblock">
```

```
</div>
 239/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28613.1367

<div class="k-default-codeblock">
```

```
</div>
 242/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28620.8164

<div class="k-default-codeblock">
```

```
</div>
 245/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28628.4961

<div class="k-default-codeblock">
```

```
</div>
 247/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28633.6172

<div class="k-default-codeblock">
```

```
</div>
 250/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28641.2969

<div class="k-default-codeblock">
```

```
</div>
 253/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28648.9766

<div class="k-default-codeblock">
```

```
</div>
 256/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28656.6562

<div class="k-default-codeblock">
```

```
</div>
 259/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28664.3359

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28672.0176

<div class="k-default-codeblock">
```

```
</div>
 265/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28679.6973

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28684.8164

<div class="k-default-codeblock">
```

```
</div>
 269/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28689.9375

<div class="k-default-codeblock">
```

```
</div>
 272/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28697.6172

<div class="k-default-codeblock">
```

```
</div>
 275/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 23ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 28705.2969

<div class="k-default-codeblock">
```

```
</div>
 277/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28710.4160

<div class="k-default-codeblock">
```

```
</div>
 280/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28718.0957

<div class="k-default-codeblock">
```

```
</div>
 283/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28725.7773

<div class="k-default-codeblock">
```

```
</div>
 286/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28733.4570

<div class="k-default-codeblock">
```

```
</div>
 289/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28741.1367

<div class="k-default-codeblock">
```

```
</div>
 292/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28748.8164

<div class="k-default-codeblock">
```

```
</div>
 295/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28756.4961

<div class="k-default-codeblock">
```

```
</div>
 297/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28761.6172

<div class="k-default-codeblock">
```

```
</div>
 299/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28766.7363

<div class="k-default-codeblock">
```

```
</div>
 302/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28774.4160

<div class="k-default-codeblock">
```

```
</div>
 305/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28782.0957

<div class="k-default-codeblock">
```

```
</div>
 308/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28789.7773

<div class="k-default-codeblock">
```

```
</div>
 311/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28797.4570

<div class="k-default-codeblock">
```

```
</div>
 314/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28805.1367

<div class="k-default-codeblock">
```

```
</div>
 317/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28812.8164

<div class="k-default-codeblock">
```

```
</div>
 320/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28820.4961

<div class="k-default-codeblock">
```

```
</div>
 323/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28828.1777

<div class="k-default-codeblock">
```

```
</div>
 325/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28833.2969

<div class="k-default-codeblock">
```

```
</div>
 328/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28840.9766

<div class="k-default-codeblock">
```

```
</div>
 331/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28848.6562

<div class="k-default-codeblock">
```

```
</div>
 333/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28853.7773

<div class="k-default-codeblock">
```

```
</div>
 336/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28861.4570

<div class="k-default-codeblock">
```

```
</div>
 338/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28866.5762

<div class="k-default-codeblock">
```

```
</div>
 341/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28874.2578

<div class="k-default-codeblock">
```

```
</div>
 343/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28879.3770

<div class="k-default-codeblock">
```

```
</div>
 346/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28887.0566

<div class="k-default-codeblock">
```

```
</div>
 349/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 23ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 28894.7363

<div class="k-default-codeblock">
```

```
</div>
 352/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28902.4160

<div class="k-default-codeblock">
```

```
</div>
 355/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28910.0977

<div class="k-default-codeblock">
```

```
</div>
 358/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28917.7773

<div class="k-default-codeblock">
```

```
</div>
 361/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28925.4570

<div class="k-default-codeblock">
```

```
</div>
 364/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28933.1367

<div class="k-default-codeblock">
```

```
</div>
 367/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28940.8164

<div class="k-default-codeblock">
```

```
</div>
 370/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28948.4961

<div class="k-default-codeblock">
```

```
</div>
 373/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28956.1777

<div class="k-default-codeblock">
```

```
</div>
 376/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28963.8574

<div class="k-default-codeblock">
```

```
</div>
 379/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28971.5371

<div class="k-default-codeblock">
```

```
</div>
 381/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28976.6562

<div class="k-default-codeblock">
```

```
</div>
 384/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28984.3359

<div class="k-default-codeblock">
```

```
</div>
 387/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28992.0176

<div class="k-default-codeblock">
```

```
</div>
 390/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 28999.6973

<div class="k-default-codeblock">
```

```
</div>
 392/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 29004.8164

<div class="k-default-codeblock">
```

```
</div>
 394/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 29009.9375

<div class="k-default-codeblock">
```

```
</div>
 396/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 29015.0566

<div class="k-default-codeblock">
```

```
</div>
 398/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 29020.1777

<div class="k-default-codeblock">
```

```
</div>
 400/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 29025.2969

<div class="k-default-codeblock">
```

```
</div>
 402/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29030.4180

<div class="k-default-codeblock">
```

```
</div>
 404/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29035.5371

<div class="k-default-codeblock">
```

```
</div>
 406/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29040.6562

<div class="k-default-codeblock">
```

```
</div>
 408/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29045.7773

<div class="k-default-codeblock">
```

```
</div>
 410/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29050.8965

<div class="k-default-codeblock">
```

```
</div>
 412/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29056.0176

<div class="k-default-codeblock">
```

```
</div>
 414/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29061.1367

<div class="k-default-codeblock">
```

```
</div>
 416/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29066.2578

<div class="k-default-codeblock">
```

```
</div>
 418/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29071.3770

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29079.0566

<div class="k-default-codeblock">
```

```
</div>
 423/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29084.1777

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29089.2969

<div class="k-default-codeblock">
```

```
</div>
 427/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29094.4160

<div class="k-default-codeblock">
```

```
</div>
 429/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29099.5371

<div class="k-default-codeblock">
```

```
</div>
 431/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29104.6562

<div class="k-default-codeblock">
```

```
</div>
 434/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29112.3359

<div class="k-default-codeblock">
```

```
</div>
 436/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29117.4570

<div class="k-default-codeblock">
```

```
</div>
 438/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29122.5762

<div class="k-default-codeblock">
```

```
</div>
 440/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29127.6973

<div class="k-default-codeblock">
```

```
</div>
 442/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29132.8164

<div class="k-default-codeblock">
```

```
</div>
 444/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29137.9375

<div class="k-default-codeblock">
```

```
</div>
 446/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29143.0566

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29150.7363

<div class="k-default-codeblock">
```

```
</div>
 451/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29155.8574

<div class="k-default-codeblock">
```

```
</div>
 453/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29160.9766

<div class="k-default-codeblock">
```

```
</div>
 455/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 29166.0977

<div class="k-default-codeblock">
```

```
</div>
 457/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 29171.2168

<div class="k-default-codeblock">
```

```
</div>
 459/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 29176.3359

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 29181.4570

<div class="k-default-codeblock">
```

```
</div>
 463/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 29186.5762

<div class="k-default-codeblock">
```

```
</div>
 465/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 29191.6973

<div class="k-default-codeblock">
```

```
</div>
 467/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 24ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 29196.8164

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 12s 25ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 29204.4805 - val_loss: 0.1000 - val_moe_loss: 30798.7227


<div class="k-default-codeblock">
```
Epoch 12/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  20s 45ms/step - accuracy: 0.9844 - loss: 0.1000 - moe_loss: 30803.8516

<div class="k-default-codeblock">
```

```
</div>
   3/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9883 - loss: 0.1000 - moe_loss: 30808.9746

<div class="k-default-codeblock">
```

```
</div>
   5/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9880 - loss: 0.1000 - moe_loss: 30814.0918

<div class="k-default-codeblock">
```

```
</div>
   7/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9889 - loss: 0.1000 - moe_loss: 30819.2129

<div class="k-default-codeblock">
```

```
</div>
  10/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9896 - loss: 0.1000 - moe_loss: 30826.8965

<div class="k-default-codeblock">
```

```
</div>
  12/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9898 - loss: 0.1000 - moe_loss: 30832.0176

<div class="k-default-codeblock">
```

```
</div>
  15/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9900 - loss: 0.1000 - moe_loss: 30839.6973

<div class="k-default-codeblock">
```

```
</div>
  18/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9903 - loss: 0.1000 - moe_loss: 30847.3789

<div class="k-default-codeblock">
```

```
</div>
  21/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9904 - loss: 0.1000 - moe_loss: 30855.0605

<div class="k-default-codeblock">
```

```
</div>
  24/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9906 - loss: 0.1000 - moe_loss: 30862.7422

<div class="k-default-codeblock">
```

```
</div>
  26/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9907 - loss: 0.1000 - moe_loss: 30867.8633

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 30872.9824

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 30880.6621

<div class="k-default-codeblock">
```

```
</div>
  33/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 30885.7812

<div class="k-default-codeblock">
```

```
</div>
  36/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 30893.4609

<div class="k-default-codeblock">
```

```
</div>
  38/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 30898.5801

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 30903.6992

<div class="k-default-codeblock">
```

```
</div>
  42/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 30908.8203

<div class="k-default-codeblock">
```

```
</div>
  44/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 30913.9395

<div class="k-default-codeblock">
```

```
</div>
  46/469 ━[37m━━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 30919.0586

<div class="k-default-codeblock">
```

```
</div>
  48/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 30924.1797

<div class="k-default-codeblock">
```

```
</div>
  50/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 30929.2988

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 30934.4199

<div class="k-default-codeblock">
```

```
</div>
  54/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 30939.5391

<div class="k-default-codeblock">
```

```
</div>
  56/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 30944.6602

<div class="k-default-codeblock">
```

```
</div>
  58/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 30949.7793

<div class="k-default-codeblock">
```

```
</div>
  60/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 30954.8984

<div class="k-default-codeblock">
```

```
</div>
  62/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 25ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 30960.0195

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 29ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 30965.1387

<div class="k-default-codeblock">
```

```
</div>
  65/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 29ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 30967.6992

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 29ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 30972.8184

<div class="k-default-codeblock">
```

```
</div>
  69/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 29ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 30977.9395

<div class="k-default-codeblock">
```

```
</div>
  71/469 ━━━[37m━━━━━━━━━━━━━━━━━  11s 29ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 30983.0586

<div class="k-default-codeblock">
```

```
</div>
  74/469 ━━━[37m━━━━━━━━━━━━━━━━━  11s 29ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 30990.7383

<div class="k-default-codeblock">
```

```
</div>
  76/469 ━━━[37m━━━━━━━━━━━━━━━━━  11s 29ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 30995.8594

<div class="k-default-codeblock">
```

```
</div>
  78/469 ━━━[37m━━━━━━━━━━━━━━━━━  11s 29ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 31000.9785

<div class="k-default-codeblock">
```

```
</div>
  80/469 ━━━[37m━━━━━━━━━━━━━━━━━  11s 29ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 31006.0996

<div class="k-default-codeblock">
```

```
</div>
  82/469 ━━━[37m━━━━━━━━━━━━━━━━━  11s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 31011.2188

<div class="k-default-codeblock">
```

```
</div>
  85/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 31018.9004

<div class="k-default-codeblock">
```

```
</div>
  87/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 31024.0195

<div class="k-default-codeblock">
```

```
</div>
  89/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 31029.1406

<div class="k-default-codeblock">
```

```
</div>
  91/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 31034.2598

<div class="k-default-codeblock">
```

```
</div>
  93/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 31039.3789

<div class="k-default-codeblock">
```

```
</div>
  95/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 31044.5000

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31049.6191

<div class="k-default-codeblock">
```

```
</div>
  99/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31054.7402

<div class="k-default-codeblock">
```

```
</div>
 101/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31059.8594

<div class="k-default-codeblock">
```

```
</div>
 103/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31064.9805

<div class="k-default-codeblock">
```

```
</div>
 105/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31070.0996

<div class="k-default-codeblock">
```

```
</div>
 107/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31075.2188

<div class="k-default-codeblock">
```

```
</div>
 109/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31080.3398

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31085.4590

<div class="k-default-codeblock">
```

```
</div>
 113/469 ━━━━[37m━━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31090.5801 

<div class="k-default-codeblock">
```

```
</div>
 115/469 ━━━━[37m━━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31095.6992

<div class="k-default-codeblock">
```

```
</div>
 117/469 ━━━━[37m━━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31100.8203

<div class="k-default-codeblock">
```

```
</div>
 119/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31105.9395

<div class="k-default-codeblock">
```

```
</div>
 121/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31111.0586

<div class="k-default-codeblock">
```

```
</div>
 123/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31116.1797

<div class="k-default-codeblock">
```

```
</div>
 125/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31121.2988

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31126.4199

<div class="k-default-codeblock">
```

```
</div>
 129/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31131.5391

<div class="k-default-codeblock">
```

```
</div>
 131/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31136.6602

<div class="k-default-codeblock">
```

```
</div>
 133/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31141.7793

<div class="k-default-codeblock">
```

```
</div>
 135/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31146.8984

<div class="k-default-codeblock">
```

```
</div>
 137/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31152.0195

<div class="k-default-codeblock">
```

```
</div>
 139/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31157.1387

<div class="k-default-codeblock">
```

```
</div>
 141/469 ━━━━━━[37m━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31162.2598

<div class="k-default-codeblock">
```

```
</div>
 143/469 ━━━━━━[37m━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31167.3789

<div class="k-default-codeblock">
```

```
</div>
 145/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31172.5000

<div class="k-default-codeblock">
```

```
</div>
 147/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31177.6191

<div class="k-default-codeblock">
```

```
</div>
 149/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31182.7402

<div class="k-default-codeblock">
```

```
</div>
 151/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31187.8594

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31192.9805

<div class="k-default-codeblock">
```

```
</div>
 155/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 27ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31198.0996

<div class="k-default-codeblock">
```

```
</div>
 157/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 27ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31203.2207

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 27ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31208.3398

<div class="k-default-codeblock">
```

```
</div>
 161/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 27ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31213.4609

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 27ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31218.5801

<div class="k-default-codeblock">
```

```
</div>
 165/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 27ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31223.6992

<div class="k-default-codeblock">
```

```
</div>
 166/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31226.2598

<div class="k-default-codeblock">
```

```
</div>
 167/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31228.8203

<div class="k-default-codeblock">
```

```
</div>
 169/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31233.9395

<div class="k-default-codeblock">
```

```
</div>
 171/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31239.0605

<div class="k-default-codeblock">
```

```
</div>
 173/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31244.1797

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31249.2988

<div class="k-default-codeblock">
```

```
</div>
 177/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31254.4199

<div class="k-default-codeblock">
```

```
</div>
 179/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31259.5391

<div class="k-default-codeblock">
```

```
</div>
 181/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31264.6602

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31269.7793

<div class="k-default-codeblock">
```

```
</div>
 185/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31274.8984

<div class="k-default-codeblock">
```

```
</div>
 187/469 ━━━━━━━[37m━━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31280.0195

<div class="k-default-codeblock">
```

```
</div>
 189/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31285.1387

<div class="k-default-codeblock">
```

```
</div>
 191/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31290.2598

<div class="k-default-codeblock">
```

```
</div>
 193/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31295.3789

<div class="k-default-codeblock">
```

```
</div>
 195/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31300.5000

<div class="k-default-codeblock">
```

```
</div>
 197/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31305.6191

<div class="k-default-codeblock">
```

```
</div>
 199/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31310.7383

<div class="k-default-codeblock">
```

```
</div>
 201/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31315.8594

<div class="k-default-codeblock">
```

```
</div>
 203/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31320.9785

<div class="k-default-codeblock">
```

```
</div>
 205/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31326.0996

<div class="k-default-codeblock">
```

```
</div>
 207/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31331.2188

<div class="k-default-codeblock">
```

```
</div>
 209/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31336.3398

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31341.4590

<div class="k-default-codeblock">
```

```
</div>
 213/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31346.5801

<div class="k-default-codeblock">
```

```
</div>
 215/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31351.6992

<div class="k-default-codeblock">
```

```
</div>
 217/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31356.8184

<div class="k-default-codeblock">
```

```
</div>
 219/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31361.9395

<div class="k-default-codeblock">
```

```
</div>
 221/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31367.0586

<div class="k-default-codeblock">
```

```
</div>
 223/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31372.1797

<div class="k-default-codeblock">
```

```
</div>
 225/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31377.2988

<div class="k-default-codeblock">
```

```
</div>
 227/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31382.4199

<div class="k-default-codeblock">
```

```
</div>
 229/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31387.5391

<div class="k-default-codeblock">
```

```
</div>
 231/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31392.6582

<div class="k-default-codeblock">
```

```
</div>
 233/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31397.7793

<div class="k-default-codeblock">
```

```
</div>
 235/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31402.8984

<div class="k-default-codeblock">
```

```
</div>
 237/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31408.0195

<div class="k-default-codeblock">
```

```
</div>
 239/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31413.1387

<div class="k-default-codeblock">
```

```
</div>
 241/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31418.2578

<div class="k-default-codeblock">
```

```
</div>
 243/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31423.3789

<div class="k-default-codeblock">
```

```
</div>
 245/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31428.4980

<div class="k-default-codeblock">
```

```
</div>
 247/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31433.6191

<div class="k-default-codeblock">
```

```
</div>
 249/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31438.7383

<div class="k-default-codeblock">
```

```
</div>
 251/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31443.8574

<div class="k-default-codeblock">
```

```
</div>
 253/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31448.9785

<div class="k-default-codeblock">
```

```
</div>
 255/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31454.0977

<div class="k-default-codeblock">
```

```
</div>
 257/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31459.2188

<div class="k-default-codeblock">
```

```
</div>
 259/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31464.3379

<div class="k-default-codeblock">
```

```
</div>
 261/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 31469.4590

<div class="k-default-codeblock">
```

```
</div>
 263/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31474.5781

<div class="k-default-codeblock">
```

```
</div>
 265/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31479.6992

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31484.8184

<div class="k-default-codeblock">
```

```
</div>
 269/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31489.9375

<div class="k-default-codeblock">
```

```
</div>
 271/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31495.0586

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31500.1777

<div class="k-default-codeblock">
```

```
</div>
 275/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31505.2988

<div class="k-default-codeblock">
```

```
</div>
 277/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31510.4180

<div class="k-default-codeblock">
```

```
</div>
 279/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31515.5391

<div class="k-default-codeblock">
```

```
</div>
 281/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31520.6582

<div class="k-default-codeblock">
```

```
</div>
 283/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31525.7773

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31530.8984

<div class="k-default-codeblock">
```

```
</div>
 287/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31536.0176

<div class="k-default-codeblock">
```

```
</div>
 289/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31541.1387

<div class="k-default-codeblock">
```

```
</div>
 291/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31546.2578

<div class="k-default-codeblock">
```

```
</div>
 293/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31551.3770

<div class="k-default-codeblock">
```

```
</div>
 295/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31556.4980

<div class="k-default-codeblock">
```

```
</div>
 297/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31561.6172

<div class="k-default-codeblock">
```

```
</div>
 299/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31566.7383

<div class="k-default-codeblock">
```

```
</div>
 301/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31571.8574

<div class="k-default-codeblock">
```

```
</div>
 303/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31576.9785

<div class="k-default-codeblock">
```

```
</div>
 305/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31582.0977

<div class="k-default-codeblock">
```

```
</div>
 307/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31587.2168

<div class="k-default-codeblock">
```

```
</div>
 309/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31592.3379

<div class="k-default-codeblock">
```

```
</div>
 311/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31597.4570

<div class="k-default-codeblock">
```

```
</div>
 313/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31602.5781

<div class="k-default-codeblock">
```

```
</div>
 315/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31607.6973

<div class="k-default-codeblock">
```

```
</div>
 317/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31612.8184

<div class="k-default-codeblock">
```

```
</div>
 319/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31617.9375

<div class="k-default-codeblock">
```

```
</div>
 321/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31623.0586

<div class="k-default-codeblock">
```

```
</div>
 323/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31628.1777

<div class="k-default-codeblock">
```

```
</div>
 325/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31633.2969

<div class="k-default-codeblock">
```

```
</div>
 327/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31638.4180

<div class="k-default-codeblock">
```

```
</div>
 329/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31643.5371

<div class="k-default-codeblock">
```

```
</div>
 331/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31648.6582

<div class="k-default-codeblock">
```

```
</div>
 333/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31653.7773

<div class="k-default-codeblock">
```

```
</div>
 335/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31658.8984

<div class="k-default-codeblock">
```

```
</div>
 337/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31664.0176

<div class="k-default-codeblock">
```

```
</div>
 339/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31669.1367

<div class="k-default-codeblock">
```

```
</div>
 341/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31674.2578

<div class="k-default-codeblock">
```

```
</div>
 343/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31679.3770

<div class="k-default-codeblock">
```

```
</div>
 345/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31684.4980

<div class="k-default-codeblock">
```

```
</div>
 347/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31689.6172

<div class="k-default-codeblock">
```

```
</div>
 349/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31694.7383

<div class="k-default-codeblock">
```

```
</div>
 351/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31699.8574

<div class="k-default-codeblock">
```

```
</div>
 353/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31704.9785

<div class="k-default-codeblock">
```

```
</div>
 355/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31710.0977

<div class="k-default-codeblock">
```

```
</div>
 357/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31715.2168

<div class="k-default-codeblock">
```

```
</div>
 359/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31720.3379

<div class="k-default-codeblock">
```

```
</div>
 361/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31725.4570

<div class="k-default-codeblock">
```

```
</div>
 363/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31730.5781

<div class="k-default-codeblock">
```

```
</div>
 365/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31735.6973

<div class="k-default-codeblock">
```

```
</div>
 367/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31740.8184

<div class="k-default-codeblock">
```

```
</div>
 369/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31745.9375

<div class="k-default-codeblock">
```

```
</div>
 371/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31751.0566

<div class="k-default-codeblock">
```

```
</div>
 373/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31756.1777

<div class="k-default-codeblock">
```

```
</div>
 375/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31761.2969

<div class="k-default-codeblock">
```

```
</div>
 377/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31766.4180

<div class="k-default-codeblock">
```

```
</div>
 379/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31771.5371

<div class="k-default-codeblock">
```

```
</div>
 381/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 28ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 31776.6582

<div class="k-default-codeblock">
```

```
</div>
 383/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31781.7773

<div class="k-default-codeblock">
```

```
</div>
 385/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31786.8965

<div class="k-default-codeblock">
```

```
</div>
 387/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31792.0176

<div class="k-default-codeblock">
```

```
</div>
 389/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31797.1367

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31802.2578

<div class="k-default-codeblock">
```

```
</div>
 393/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31807.3770

<div class="k-default-codeblock">
```

```
</div>
 395/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31812.4980

<div class="k-default-codeblock">
```

```
</div>
 397/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31817.6172

<div class="k-default-codeblock">
```

```
</div>
 399/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31822.7383

<div class="k-default-codeblock">
```

```
</div>
 401/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31827.8574

<div class="k-default-codeblock">
```

```
</div>
 403/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31832.9766

<div class="k-default-codeblock">
```

```
</div>
 405/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31838.0977

<div class="k-default-codeblock">
```

```
</div>
 407/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31843.2168

<div class="k-default-codeblock">
```

```
</div>
 409/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31848.3379

<div class="k-default-codeblock">
```

```
</div>
 411/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31853.4570

<div class="k-default-codeblock">
```

```
</div>
 413/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31858.5781

<div class="k-default-codeblock">
```

```
</div>
 415/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31863.6973

<div class="k-default-codeblock">
```

```
</div>
 417/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31868.8164

<div class="k-default-codeblock">
```

```
</div>
 419/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31873.9375

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31879.0566

<div class="k-default-codeblock">
```

```
</div>
 423/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31884.1777

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31889.2969

<div class="k-default-codeblock">
```

```
</div>
 427/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31894.4180

<div class="k-default-codeblock">
```

```
</div>
 429/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31899.5371

<div class="k-default-codeblock">
```

```
</div>
 431/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31904.6562

<div class="k-default-codeblock">
```

```
</div>
 433/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31909.7773

<div class="k-default-codeblock">
```

```
</div>
 435/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31914.8965

<div class="k-default-codeblock">
```

```
</div>
 437/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31920.0176

<div class="k-default-codeblock">
```

```
</div>
 439/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31925.1367

<div class="k-default-codeblock">
```

```
</div>
 441/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31930.2578

<div class="k-default-codeblock">
```

```
</div>
 443/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31935.3770

<div class="k-default-codeblock">
```

```
</div>
 445/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31940.4961

<div class="k-default-codeblock">
```

```
</div>
 447/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31945.6172

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31950.7363

<div class="k-default-codeblock">
```

```
</div>
 451/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31955.8574

<div class="k-default-codeblock">
```

```
</div>
 453/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31960.9766

<div class="k-default-codeblock">
```

```
</div>
 455/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31966.0977

<div class="k-default-codeblock">
```

```
</div>
 457/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31971.2168

<div class="k-default-codeblock">
```

```
</div>
 459/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31976.3379

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31981.4570

<div class="k-default-codeblock">
```

```
</div>
 463/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31986.5762

<div class="k-default-codeblock">
```

```
</div>
 465/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31991.6973

<div class="k-default-codeblock">
```

```
</div>
 467/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 28ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 31996.8164

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 14s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 32004.4805 - val_loss: 0.1000 - val_moe_loss: 33598.7227


<div class="k-default-codeblock">
```
Epoch 13/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  21s 45ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 33603.8633

<div class="k-default-codeblock">
```

```
</div>
   4/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 24ms/step - accuracy: 0.9954 - loss: 0.1000 - moe_loss: 33611.5352

<div class="k-default-codeblock">
```

```
</div>
   6/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9953 - loss: 0.1000 - moe_loss: 33616.6523

<div class="k-default-codeblock">
```

```
</div>
   9/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9948 - loss: 0.1000 - moe_loss: 33624.3320

<div class="k-default-codeblock">
```

```
</div>
  11/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 33629.4531

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 25ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 33634.5742

<div class="k-default-codeblock">
```

```
</div>
  15/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 33639.6953

<div class="k-default-codeblock">
```

```
</div>
  17/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 33644.8125

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 33649.9336

<div class="k-default-codeblock">
```

```
</div>
  21/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9926 - loss: 0.1000 - moe_loss: 33655.0547

<div class="k-default-codeblock">
```

```
</div>
  23/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 33660.1758

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 33665.2930

<div class="k-default-codeblock">
```

```
</div>
  27/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 33670.4141

<div class="k-default-codeblock">
```

```
</div>
  29/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 33675.5352

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 33680.6562

<div class="k-default-codeblock">
```

```
</div>
  33/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 33685.7734

<div class="k-default-codeblock">
```

```
</div>
  35/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 33690.8945

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 33696.0156

<div class="k-default-codeblock">
```

```
</div>
  39/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 33701.1328

<div class="k-default-codeblock">
```

```
</div>
  41/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 33706.2539

<div class="k-default-codeblock">
```

```
</div>
  43/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 33711.3750

<div class="k-default-codeblock">
```

```
</div>
  45/469 ━[37m━━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 33716.4961

<div class="k-default-codeblock">
```

```
</div>
  47/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 33721.6133

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 33726.7344

<div class="k-default-codeblock">
```

```
</div>
  51/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 33731.8555

<div class="k-default-codeblock">
```

```
</div>
  53/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 33736.9766

<div class="k-default-codeblock">
```

```
</div>
  55/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 33742.0938

<div class="k-default-codeblock">
```

```
</div>
  57/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 33747.2148

<div class="k-default-codeblock">
```

```
</div>
  59/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 33752.3359

<div class="k-default-codeblock">
```

```
</div>
  61/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 33757.4531

<div class="k-default-codeblock">
```

```
</div>
  63/469 ━━[37m━━━━━━━━━━━━━━━━━━  11s 27ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 33762.5742

<div class="k-default-codeblock">
```

```
</div>
  65/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 33767.6953

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33772.8164

<div class="k-default-codeblock">
```

```
</div>
  69/469 ━━[37m━━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33777.9336

<div class="k-default-codeblock">
```

```
</div>
  71/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33783.0547

<div class="k-default-codeblock">
```

```
</div>
  73/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33788.1758

<div class="k-default-codeblock">
```

```
</div>
  75/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33793.2930

<div class="k-default-codeblock">
```

```
</div>
  77/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33798.4141

<div class="k-default-codeblock">
```

```
</div>
  79/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33803.5352

<div class="k-default-codeblock">
```

```
</div>
  81/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33808.6562

<div class="k-default-codeblock">
```

```
</div>
  83/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33813.7734

<div class="k-default-codeblock">
```

```
</div>
  85/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33818.8945

<div class="k-default-codeblock">
```

```
</div>
  87/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33824.0156

<div class="k-default-codeblock">
```

```
</div>
  89/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33829.1367

<div class="k-default-codeblock">
```

```
</div>
  91/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33834.2539

<div class="k-default-codeblock">
```

```
</div>
  93/469 ━━━[37m━━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33839.3750

<div class="k-default-codeblock">
```

```
</div>
  95/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33844.4961

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33849.6133

<div class="k-default-codeblock">
```

```
</div>
  99/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33854.7344

<div class="k-default-codeblock">
```

```
</div>
 101/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33859.8555

<div class="k-default-codeblock">
```

```
</div>
 103/469 ━━━━[37m━━━━━━━━━━━━━━━━  10s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33864.9766

<div class="k-default-codeblock">
```

```
</div>
 105/469 ━━━━[37m━━━━━━━━━━━━━━━━  9s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33870.0938 

<div class="k-default-codeblock">
```

```
</div>
 107/469 ━━━━[37m━━━━━━━━━━━━━━━━  9s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33875.2148

<div class="k-default-codeblock">
```

```
</div>
 109/469 ━━━━[37m━━━━━━━━━━━━━━━━  9s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33880.3359

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  9s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33885.4531

<div class="k-default-codeblock">
```

```
</div>
 113/469 ━━━━[37m━━━━━━━━━━━━━━━━  9s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33890.5742

<div class="k-default-codeblock">
```

```
</div>
 115/469 ━━━━[37m━━━━━━━━━━━━━━━━  9s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33895.6953

<div class="k-default-codeblock">
```

```
</div>
 117/469 ━━━━[37m━━━━━━━━━━━━━━━━  9s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33900.8164

<div class="k-default-codeblock">
```

```
</div>
 119/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 27ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 33905.9336

<div class="k-default-codeblock">
```

```
</div>
 121/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33911.0547

<div class="k-default-codeblock">
```

```
</div>
 123/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33916.1758

<div class="k-default-codeblock">
```

```
</div>
 125/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33921.2930

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33926.4141

<div class="k-default-codeblock">
```

```
</div>
 129/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33931.5352

<div class="k-default-codeblock">
```

```
</div>
 131/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33936.6562

<div class="k-default-codeblock">
```

```
</div>
 133/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33941.7734

<div class="k-default-codeblock">
```

```
</div>
 135/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33946.8945

<div class="k-default-codeblock">
```

```
</div>
 137/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33952.0156

<div class="k-default-codeblock">
```

```
</div>
 139/469 ━━━━━[37m━━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33957.1367

<div class="k-default-codeblock">
```

```
</div>
 141/469 ━━━━━━[37m━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33962.2539

<div class="k-default-codeblock">
```

```
</div>
 143/469 ━━━━━━[37m━━━━━━━━━━━━━━  9s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33967.3750

<div class="k-default-codeblock">
```

```
</div>
 145/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33972.4961

<div class="k-default-codeblock">
```

```
</div>
 147/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 33977.6172

<div class="k-default-codeblock">
```

```
</div>
 149/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 33982.7344

<div class="k-default-codeblock">
```

```
</div>
 151/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 33987.8555

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 33992.9766

<div class="k-default-codeblock">
```

```
</div>
 155/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 33998.0938

<div class="k-default-codeblock">
```

```
</div>
 157/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 34003.2148

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 34008.3359

<div class="k-default-codeblock">
```

```
</div>
 161/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 34013.4570

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 34018.5742

<div class="k-default-codeblock">
```

```
</div>
 165/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 34023.6953

<div class="k-default-codeblock">
```

```
</div>
 167/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 34028.8164

<div class="k-default-codeblock">
```

```
</div>
 169/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 34033.9336

<div class="k-default-codeblock">
```

```
</div>
 171/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 34039.0547

<div class="k-default-codeblock">
```

```
</div>
 173/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 34044.1758

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34049.2969

<div class="k-default-codeblock">
```

```
</div>
 177/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34054.4141

<div class="k-default-codeblock">
```

```
</div>
 179/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34059.5352

<div class="k-default-codeblock">
```

```
</div>
 181/469 ━━━━━━━[37m━━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34064.6562

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34069.7734

<div class="k-default-codeblock">
```

```
</div>
 185/469 ━━━━━━━[37m━━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34074.8945

<div class="k-default-codeblock">
```

```
</div>
 186/469 ━━━━━━━[37m━━━━━━━━━━━━━  7s 28ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34077.4531

<div class="k-default-codeblock">
```

```
</div>
 187/469 ━━━━━━━[37m━━━━━━━━━━━━━  8s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34080.0156

<div class="k-default-codeblock">
```

```
</div>
 188/469 ━━━━━━━━[37m━━━━━━━━━━━━  8s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34082.5742

<div class="k-default-codeblock">
```

```
</div>
 190/469 ━━━━━━━━[37m━━━━━━━━━━━━  8s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34087.6953

<div class="k-default-codeblock">
```

```
</div>
 192/469 ━━━━━━━━[37m━━━━━━━━━━━━  8s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34092.8164

<div class="k-default-codeblock">
```

```
</div>
 194/469 ━━━━━━━━[37m━━━━━━━━━━━━  8s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34097.9336

<div class="k-default-codeblock">
```

```
</div>
 196/469 ━━━━━━━━[37m━━━━━━━━━━━━  8s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34103.0547

<div class="k-default-codeblock">
```

```
</div>
 198/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34108.1758

<div class="k-default-codeblock">
```

```
</div>
 200/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34113.2930

<div class="k-default-codeblock">
```

```
</div>
 202/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34118.4141

<div class="k-default-codeblock">
```

```
</div>
 204/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34123.5352

<div class="k-default-codeblock">
```

```
</div>
 206/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34128.6562

<div class="k-default-codeblock">
```

```
</div>
 208/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34133.7734

<div class="k-default-codeblock">
```

```
</div>
 210/469 ━━━━━━━━[37m━━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34138.8945

<div class="k-default-codeblock">
```

```
</div>
 212/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34144.0156

<div class="k-default-codeblock">
```

```
</div>
 214/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34149.1367

<div class="k-default-codeblock">
```

```
</div>
 216/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34154.2539

<div class="k-default-codeblock">
```

```
</div>
 218/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34159.3750

<div class="k-default-codeblock">
```

```
</div>
 220/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34164.4961

<div class="k-default-codeblock">
```

```
</div>
 222/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34169.6133

<div class="k-default-codeblock">
```

```
</div>
 224/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34174.7344

<div class="k-default-codeblock">
```

```
</div>
 226/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34179.8555

<div class="k-default-codeblock">
```

```
</div>
 228/469 ━━━━━━━━━[37m━━━━━━━━━━━  7s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34184.9766

<div class="k-default-codeblock">
```

```
</div>
 230/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34190.0938

<div class="k-default-codeblock">
```

```
</div>
 232/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34195.2148

<div class="k-default-codeblock">
```

```
</div>
 234/469 ━━━━━━━━━[37m━━━━━━━━━━━  6s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34200.3359

<div class="k-default-codeblock">
```

```
</div>
 236/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34205.4531

<div class="k-default-codeblock">
```

```
</div>
 238/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34210.5742

<div class="k-default-codeblock">
```

```
</div>
 240/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34215.6953

<div class="k-default-codeblock">
```

```
</div>
 242/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34220.8164

<div class="k-default-codeblock">
```

```
</div>
 244/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34225.9336

<div class="k-default-codeblock">
```

```
</div>
 246/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34231.0547

<div class="k-default-codeblock">
```

```
</div>
 248/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34236.1758

<div class="k-default-codeblock">
```

```
</div>
 250/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34241.2969

<div class="k-default-codeblock">
```

```
</div>
 252/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34246.4141

<div class="k-default-codeblock">
```

```
</div>
 254/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34251.5352

<div class="k-default-codeblock">
```

```
</div>
 256/469 ━━━━━━━━━━[37m━━━━━━━━━━  6s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34256.6562

<div class="k-default-codeblock">
```

```
</div>
 258/469 ━━━━━━━━━━━[37m━━━━━━━━━  6s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34261.7734

<div class="k-default-codeblock">
```

```
</div>
 260/469 ━━━━━━━━━━━[37m━━━━━━━━━  6s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34266.8945

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  6s 29ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34272.0156

<div class="k-default-codeblock">
```

```
</div>
 263/469 ━━━━━━━━━━━[37m━━━━━━━━━  6s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34274.5742

<div class="k-default-codeblock">
```

```
</div>
 265/469 ━━━━━━━━━━━[37m━━━━━━━━━  6s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34279.6953

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34284.8164

<div class="k-default-codeblock">
```

```
</div>
 269/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34289.9336

<div class="k-default-codeblock">
```

```
</div>
 271/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34295.0547

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34300.1758

<div class="k-default-codeblock">
```

```
</div>
 275/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34305.2930

<div class="k-default-codeblock">
```

```
</div>
 277/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34310.4141

<div class="k-default-codeblock">
```

```
</div>
 279/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34315.5352

<div class="k-default-codeblock">
```

```
</div>
 281/469 ━━━━━━━━━━━[37m━━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34320.6562

<div class="k-default-codeblock">
```

```
</div>
 283/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34325.7734

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34330.8945

<div class="k-default-codeblock">
```

```
</div>
 287/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34336.0156

<div class="k-default-codeblock">
```

```
</div>
 289/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34341.1328

<div class="k-default-codeblock">
```

```
</div>
 291/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34346.2539

<div class="k-default-codeblock">
```

```
</div>
 293/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34351.3750

<div class="k-default-codeblock">
```

```
</div>
 295/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34356.4961

<div class="k-default-codeblock">
```

```
</div>
 297/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34361.6133

<div class="k-default-codeblock">
```

```
</div>
 299/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34366.7344

<div class="k-default-codeblock">
```

```
</div>
 301/469 ━━━━━━━━━━━━[37m━━━━━━━━  5s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34371.8555

<div class="k-default-codeblock">
```

```
</div>
 303/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34376.9727

<div class="k-default-codeblock">
```

```
</div>
 305/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34382.0938

<div class="k-default-codeblock">
```

```
</div>
 307/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34387.2148

<div class="k-default-codeblock">
```

```
</div>
 309/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34392.3359

<div class="k-default-codeblock">
```

```
</div>
 311/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34397.4531

<div class="k-default-codeblock">
```

```
</div>
 313/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34402.5742

<div class="k-default-codeblock">
```

```
</div>
 315/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34407.6953

<div class="k-default-codeblock">
```

```
</div>
 317/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34412.8125

<div class="k-default-codeblock">
```

```
</div>
 319/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34417.9336

<div class="k-default-codeblock">
```

```
</div>
 321/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34423.0547

<div class="k-default-codeblock">
```

```
</div>
 323/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34428.1758

<div class="k-default-codeblock">
```

```
</div>
 325/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34433.2930

<div class="k-default-codeblock">
```

```
</div>
 327/469 ━━━━━━━━━━━━━[37m━━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34438.4141

<div class="k-default-codeblock">
```

```
</div>
 329/469 ━━━━━━━━━━━━━━[37m━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34443.5352

<div class="k-default-codeblock">
```

```
</div>
 331/469 ━━━━━━━━━━━━━━[37m━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34448.6562

<div class="k-default-codeblock">
```

```
</div>
 333/469 ━━━━━━━━━━━━━━[37m━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34453.7734

<div class="k-default-codeblock">
```

```
</div>
 335/469 ━━━━━━━━━━━━━━[37m━━━━━━  4s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34458.8945

<div class="k-default-codeblock">
```

```
</div>
 337/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34464.0156

<div class="k-default-codeblock">
```

```
</div>
 339/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34469.1328

<div class="k-default-codeblock">
```

```
</div>
 341/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34474.2539

<div class="k-default-codeblock">
```

```
</div>
 343/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34479.3750

<div class="k-default-codeblock">
```

```
</div>
 345/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34484.4961

<div class="k-default-codeblock">
```

```
</div>
 347/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34489.6133

<div class="k-default-codeblock">
```

```
</div>
 349/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34494.7344

<div class="k-default-codeblock">
```

```
</div>
 351/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34499.8555

<div class="k-default-codeblock">
```

```
</div>
 353/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34504.9727

<div class="k-default-codeblock">
```

```
</div>
 355/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34510.0938

<div class="k-default-codeblock">
```

```
</div>
 357/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34515.2148

<div class="k-default-codeblock">
```

```
</div>
 359/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34520.3359

<div class="k-default-codeblock">
```

```
</div>
 361/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34525.4531

<div class="k-default-codeblock">
```

```
</div>
 363/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34530.5742

<div class="k-default-codeblock">
```

```
</div>
 365/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34535.6953

<div class="k-default-codeblock">
```

```
</div>
 367/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34540.8164

<div class="k-default-codeblock">
```

```
</div>
 369/469 ━━━━━━━━━━━━━━━[37m━━━━━  3s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34545.9336

<div class="k-default-codeblock">
```

```
</div>
 371/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34551.0547

<div class="k-default-codeblock">
```

```
</div>
 373/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34556.1758

<div class="k-default-codeblock">
```

```
</div>
 375/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34561.2930

<div class="k-default-codeblock">
```

```
</div>
 377/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34566.4141

<div class="k-default-codeblock">
```

```
</div>
 379/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34571.5352

<div class="k-default-codeblock">
```

```
</div>
 381/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34576.6562

<div class="k-default-codeblock">
```

```
</div>
 383/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34581.7734

<div class="k-default-codeblock">
```

```
</div>
 385/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34586.8945

<div class="k-default-codeblock">
```

```
</div>
 387/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34592.0156

<div class="k-default-codeblock">
```

```
</div>
 389/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34597.1328

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34602.2539

<div class="k-default-codeblock">
```

```
</div>
 393/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34607.3750

<div class="k-default-codeblock">
```

```
</div>
 395/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34612.4961

<div class="k-default-codeblock">
```

```
</div>
 397/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34617.6133

<div class="k-default-codeblock">
```

```
</div>
 399/469 ━━━━━━━━━━━━━━━━━[37m━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34622.7344

<div class="k-default-codeblock">
```

```
</div>
 401/469 ━━━━━━━━━━━━━━━━━[37m━━━  2s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34627.8555

<div class="k-default-codeblock">
```

```
</div>
 403/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34632.9727

<div class="k-default-codeblock">
```

```
</div>
 405/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 30ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 34638.0938

<div class="k-default-codeblock">
```

```
</div>
 407/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34643.2148

<div class="k-default-codeblock">
```

```
</div>
 409/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34648.3359

<div class="k-default-codeblock">
```

```
</div>
 411/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34653.4531

<div class="k-default-codeblock">
```

```
</div>
 413/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34658.5742

<div class="k-default-codeblock">
```

```
</div>
 415/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34663.6953

<div class="k-default-codeblock">
```

```
</div>
 417/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34668.8164

<div class="k-default-codeblock">
```

```
</div>
 419/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34673.9336

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34679.0547

<div class="k-default-codeblock">
```

```
</div>
 423/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34684.1758

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34689.2930

<div class="k-default-codeblock">
```

```
</div>
 427/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34694.4141

<div class="k-default-codeblock">
```

```
</div>
 429/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34699.5352

<div class="k-default-codeblock">
```

```
</div>
 431/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34704.6562

<div class="k-default-codeblock">
```

```
</div>
 433/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34709.7734

<div class="k-default-codeblock">
```

```
</div>
 435/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34714.8945

<div class="k-default-codeblock">
```

```
</div>
 437/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34720.0156

<div class="k-default-codeblock">
```

```
</div>
 439/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34725.1328

<div class="k-default-codeblock">
```

```
</div>
 441/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34730.2539

<div class="k-default-codeblock">
```

```
</div>
 443/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34735.3750

<div class="k-default-codeblock">
```

```
</div>
 445/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34740.4961

<div class="k-default-codeblock">
```

```
</div>
 447/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34745.6133

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34750.7344

<div class="k-default-codeblock">
```

```
</div>
 451/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34755.8555

<div class="k-default-codeblock">
```

```
</div>
 453/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34760.9727

<div class="k-default-codeblock">
```

```
</div>
 455/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34766.0938

<div class="k-default-codeblock">
```

```
</div>
 457/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34771.2148

<div class="k-default-codeblock">
```

```
</div>
 459/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34776.3359

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34781.4531

<div class="k-default-codeblock">
```

```
</div>
 463/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34786.5742

<div class="k-default-codeblock">
```

```
</div>
 465/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34791.6953

<div class="k-default-codeblock">
```

```
</div>
 467/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34796.8164

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34801.9336

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 15s 32ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 34804.4766 - val_loss: 0.1000 - val_moe_loss: 36398.7227


<div class="k-default-codeblock">
```
Epoch 14/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  23s 50ms/step - accuracy: 1.0000 - loss: 0.1000 - moe_loss: 36403.8555

<div class="k-default-codeblock">
```

```
</div>
   3/469 [37m━━━━━━━━━━━━━━━━━━━━  13s 28ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 36408.9766

<div class="k-default-codeblock">
```

```
</div>
   5/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9905 - loss: 0.1000 - moe_loss: 36414.0938

<div class="k-default-codeblock">
```

```
</div>
   7/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9898 - loss: 0.1000 - moe_loss: 36419.2148

<div class="k-default-codeblock">
```

```
</div>
   9/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9898 - loss: 0.1000 - moe_loss: 36424.3320

<div class="k-default-codeblock">
```

```
</div>
  11/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9899 - loss: 0.1000 - moe_loss: 36429.4531

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9899 - loss: 0.1000 - moe_loss: 36434.5742

<div class="k-default-codeblock">
```

```
</div>
  15/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9900 - loss: 0.1000 - moe_loss: 36439.6914

<div class="k-default-codeblock">
```

```
</div>
  17/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9901 - loss: 0.1000 - moe_loss: 36444.8125

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9901 - loss: 0.1000 - moe_loss: 36449.9336

<div class="k-default-codeblock">
```

```
</div>
  21/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9902 - loss: 0.1000 - moe_loss: 36455.0547

<div class="k-default-codeblock">
```

```
</div>
  23/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9903 - loss: 0.1000 - moe_loss: 36460.1719

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9903 - loss: 0.1000 - moe_loss: 36465.2930

<div class="k-default-codeblock">
```

```
</div>
  27/469 ━[37m━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9904 - loss: 0.1000 - moe_loss: 36470.4141

<div class="k-default-codeblock">
```

```
</div>
  29/469 ━[37m━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9905 - loss: 0.1000 - moe_loss: 36475.5312

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9906 - loss: 0.1000 - moe_loss: 36480.6523

<div class="k-default-codeblock">
```

```
</div>
  33/469 ━[37m━━━━━━━━━━━━━━━━━━━  12s 28ms/step - accuracy: 0.9906 - loss: 0.1000 - moe_loss: 36485.7734

<div class="k-default-codeblock">
```

```
</div>
  35/469 ━[37m━━━━━━━━━━━━━━━━━━━  12s 29ms/step - accuracy: 0.9907 - loss: 0.1000 - moe_loss: 36490.8945

<div class="k-default-codeblock">
```

```
</div>
  36/469 ━[37m━━━━━━━━━━━━━━━━━━━  13s 32ms/step - accuracy: 0.9907 - loss: 0.1000 - moe_loss: 36493.4531

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  14s 33ms/step - accuracy: 0.9907 - loss: 0.1000 - moe_loss: 36496.0117

<div class="k-default-codeblock">
```

```
</div>
  39/469 ━[37m━━━━━━━━━━━━━━━━━━━  14s 34ms/step - accuracy: 0.9908 - loss: 0.1000 - moe_loss: 36501.1328

<div class="k-default-codeblock">
```

```
</div>
  41/469 ━[37m━━━━━━━━━━━━━━━━━━━  14s 34ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 36506.2539

<div class="k-default-codeblock">
```

```
</div>
  42/469 ━[37m━━━━━━━━━━━━━━━━━━━  14s 35ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 36508.8125

<div class="k-default-codeblock">
```

```
</div>
  44/469 ━[37m━━━━━━━━━━━━━━━━━━━  15s 37ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 36513.9336

<div class="k-default-codeblock">
```

```
</div>
  45/469 ━[37m━━━━━━━━━━━━━━━━━━━  16s 39ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 36516.4922

<div class="k-default-codeblock">
```

```
</div>
  46/469 ━[37m━━━━━━━━━━━━━━━━━━━  18s 44ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 36519.0508

<div class="k-default-codeblock">
```

```
</div>
  48/469 ━━[37m━━━━━━━━━━━━━━━━━━  18s 44ms/step - accuracy: 0.9910 - loss: 0.1000 - moe_loss: 36524.1719

<div class="k-default-codeblock">
```

```
</div>
  50/469 ━━[37m━━━━━━━━━━━━━━━━━━  18s 43ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 36529.2930

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  18s 43ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 36534.4102

<div class="k-default-codeblock">
```

```
</div>
  54/469 ━━[37m━━━━━━━━━━━━━━━━━━  17s 43ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 36539.5312

<div class="k-default-codeblock">
```

```
</div>
  55/469 ━━[37m━━━━━━━━━━━━━━━━━━  18s 44ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 36542.0898

<div class="k-default-codeblock">
```

```
</div>
  56/469 ━━[37m━━━━━━━━━━━━━━━━━━  18s 44ms/step - accuracy: 0.9912 - loss: 0.1000 - moe_loss: 36544.6523

<div class="k-default-codeblock">
```

```
</div>
  58/469 ━━[37m━━━━━━━━━━━━━━━━━━  17s 44ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 36549.7695

<div class="k-default-codeblock">
```

```
</div>
  60/469 ━━[37m━━━━━━━━━━━━━━━━━━  17s 43ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 36554.8906

<div class="k-default-codeblock">
```

```
</div>
  62/469 ━━[37m━━━━━━━━━━━━━━━━━━  17s 43ms/step - accuracy: 0.9913 - loss: 0.1000 - moe_loss: 36560.0117

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  17s 43ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 36565.1328

<div class="k-default-codeblock">
```

```
</div>
  66/469 ━━[37m━━━━━━━━━━━━━━━━━━  17s 43ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 36570.2500

<div class="k-default-codeblock">
```

```
</div>
  68/469 ━━[37m━━━━━━━━━━━━━━━━━━  17s 43ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 36575.3711

<div class="k-default-codeblock">
```

```
</div>
  70/469 ━━[37m━━━━━━━━━━━━━━━━━━  16s 42ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 36580.4922

<div class="k-default-codeblock">
```

```
</div>
  72/469 ━━━[37m━━━━━━━━━━━━━━━━━  16s 42ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 36585.6133

<div class="k-default-codeblock">
```

```
</div>
  74/469 ━━━[37m━━━━━━━━━━━━━━━━━  16s 42ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 36590.7305

<div class="k-default-codeblock">
```

```
</div>
  76/469 ━━━[37m━━━━━━━━━━━━━━━━━  16s 42ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 36595.8516

<div class="k-default-codeblock">
```

```
</div>
  78/469 ━━━[37m━━━━━━━━━━━━━━━━━  16s 41ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 36600.9727

<div class="k-default-codeblock">
```

```
</div>
  80/469 ━━━[37m━━━━━━━━━━━━━━━━━  16s 41ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 36606.0898

<div class="k-default-codeblock">
```

```
</div>
  82/469 ━━━[37m━━━━━━━━━━━━━━━━━  15s 41ms/step - accuracy: 0.9915 - loss: 0.1000 - moe_loss: 36611.2109

<div class="k-default-codeblock">
```

```
</div>
  84/469 ━━━[37m━━━━━━━━━━━━━━━━━  15s 41ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 36616.3320

<div class="k-default-codeblock">
```

```
</div>
  86/469 ━━━[37m━━━━━━━━━━━━━━━━━  15s 41ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 36621.4531

<div class="k-default-codeblock">
```

```
</div>
  88/469 ━━━[37m━━━━━━━━━━━━━━━━━  15s 41ms/step - accuracy: 0.9916 - loss: 0.1000 - moe_loss: 36626.5703

<div class="k-default-codeblock">
```

```
</div>
  90/469 ━━━[37m━━━━━━━━━━━━━━━━━  15s 41ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 36631.6914

<div class="k-default-codeblock">
```

```
</div>
  91/469 ━━━[37m━━━━━━━━━━━━━━━━━  15s 41ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 36634.2500

<div class="k-default-codeblock">
```

```
</div>
  93/469 ━━━[37m━━━━━━━━━━━━━━━━━  15s 41ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 36639.3711

<div class="k-default-codeblock">
```

```
</div>
  95/469 ━━━━[37m━━━━━━━━━━━━━━━━  15s 40ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 36644.4922

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  15s 40ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 36649.6133

<div class="k-default-codeblock">
```

```
</div>
  99/469 ━━━━[37m━━━━━━━━━━━━━━━━  14s 40ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 36654.7344

<div class="k-default-codeblock">
```

```
</div>
 101/469 ━━━━[37m━━━━━━━━━━━━━━━━  14s 40ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 36659.8516

<div class="k-default-codeblock">
```

```
</div>
 103/469 ━━━━[37m━━━━━━━━━━━━━━━━  14s 40ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 36664.9727

<div class="k-default-codeblock">
```

```
</div>
 105/469 ━━━━[37m━━━━━━━━━━━━━━━━  14s 40ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 36670.0938

<div class="k-default-codeblock">
```

```
</div>
 107/469 ━━━━[37m━━━━━━━━━━━━━━━━  14s 40ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 36675.2148

<div class="k-default-codeblock">
```

```
</div>
 109/469 ━━━━[37m━━━━━━━━━━━━━━━━  14s 40ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 36680.3320

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  14s 40ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 36685.4531

<div class="k-default-codeblock">
```

```
</div>
 113/469 ━━━━[37m━━━━━━━━━━━━━━━━  14s 40ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 36690.5742

<div class="k-default-codeblock">
```

```
</div>
 115/469 ━━━━[37m━━━━━━━━━━━━━━━━  14s 40ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 36695.6953

<div class="k-default-codeblock">
```

```
</div>
 117/469 ━━━━[37m━━━━━━━━━━━━━━━━  13s 40ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36700.8125

<div class="k-default-codeblock">
```

```
</div>
 119/469 ━━━━━[37m━━━━━━━━━━━━━━━  13s 40ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36705.9336

<div class="k-default-codeblock">
```

```
</div>
 121/469 ━━━━━[37m━━━━━━━━━━━━━━━  13s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36711.0547

<div class="k-default-codeblock">
```

```
</div>
 123/469 ━━━━━[37m━━━━━━━━━━━━━━━  13s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36716.1758

<div class="k-default-codeblock">
```

```
</div>
 125/469 ━━━━━[37m━━━━━━━━━━━━━━━  13s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36721.2930

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  13s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36726.4141

<div class="k-default-codeblock">
```

```
</div>
 129/469 ━━━━━[37m━━━━━━━━━━━━━━━  13s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36731.5352

<div class="k-default-codeblock">
```

```
</div>
 131/469 ━━━━━[37m━━━━━━━━━━━━━━━  13s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36736.6562

<div class="k-default-codeblock">
```

```
</div>
 133/469 ━━━━━[37m━━━━━━━━━━━━━━━  13s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36741.7734

<div class="k-default-codeblock">
```

```
</div>
 135/469 ━━━━━[37m━━━━━━━━━━━━━━━  12s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36746.8945

<div class="k-default-codeblock">
```

```
</div>
 137/469 ━━━━━[37m━━━━━━━━━━━━━━━  12s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36752.0156

<div class="k-default-codeblock">
```

```
</div>
 139/469 ━━━━━[37m━━━━━━━━━━━━━━━  12s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36757.1328

<div class="k-default-codeblock">
```

```
</div>
 141/469 ━━━━━━[37m━━━━━━━━━━━━━━  12s 39ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36762.2539

<div class="k-default-codeblock">
```

```
</div>
 143/469 ━━━━━━[37m━━━━━━━━━━━━━━  12s 38ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36767.3750

<div class="k-default-codeblock">
```

```
</div>
 145/469 ━━━━━━[37m━━━━━━━━━━━━━━  12s 38ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36772.4961

<div class="k-default-codeblock">
```

```
</div>
 147/469 ━━━━━━[37m━━━━━━━━━━━━━━  12s 38ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36777.6133

<div class="k-default-codeblock">
```

```
</div>
 149/469 ━━━━━━[37m━━━━━━━━━━━━━━  12s 38ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 36782.7344

<div class="k-default-codeblock">
```

```
</div>
 151/469 ━━━━━━[37m━━━━━━━━━━━━━━  12s 38ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36787.8555

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  12s 38ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36792.9727

<div class="k-default-codeblock">
```

```
</div>
 155/469 ━━━━━━[37m━━━━━━━━━━━━━━  11s 38ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36798.0938

<div class="k-default-codeblock">
```

```
</div>
 157/469 ━━━━━━[37m━━━━━━━━━━━━━━  11s 38ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36803.2148

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  11s 38ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36808.3359

<div class="k-default-codeblock">
```

```
</div>
 161/469 ━━━━━━[37m━━━━━━━━━━━━━━  11s 38ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36813.4531

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  11s 38ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36818.5742

<div class="k-default-codeblock">
```

```
</div>
 165/469 ━━━━━━━[37m━━━━━━━━━━━━━  11s 38ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36823.6953

<div class="k-default-codeblock">
```

```
</div>
 167/469 ━━━━━━━[37m━━━━━━━━━━━━━  11s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36828.8125

<div class="k-default-codeblock">
```

```
</div>
 169/469 ━━━━━━━[37m━━━━━━━━━━━━━  11s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36833.9336

<div class="k-default-codeblock">
```

```
</div>
 171/469 ━━━━━━━[37m━━━━━━━━━━━━━  11s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36839.0547

<div class="k-default-codeblock">
```

```
</div>
 173/469 ━━━━━━━[37m━━━━━━━━━━━━━  11s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36844.1758

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36849.2930

<div class="k-default-codeblock">
```

```
</div>
 177/469 ━━━━━━━[37m━━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36854.4141

<div class="k-default-codeblock">
```

```
</div>
 179/469 ━━━━━━━[37m━━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36859.5352

<div class="k-default-codeblock">
```

```
</div>
 181/469 ━━━━━━━[37m━━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36864.6523

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36869.7734

<div class="k-default-codeblock">
```

```
</div>
 185/469 ━━━━━━━[37m━━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36874.8945

<div class="k-default-codeblock">
```

```
</div>
 187/469 ━━━━━━━[37m━━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36880.0156

<div class="k-default-codeblock">
```

```
</div>
 189/469 ━━━━━━━━[37m━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36885.1328

<div class="k-default-codeblock">
```

```
</div>
 191/469 ━━━━━━━━[37m━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36890.2539

<div class="k-default-codeblock">
```

```
</div>
 193/469 ━━━━━━━━[37m━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36895.3750

<div class="k-default-codeblock">
```

```
</div>
 195/469 ━━━━━━━━[37m━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36900.4922

<div class="k-default-codeblock">
```

```
</div>
 197/469 ━━━━━━━━[37m━━━━━━━━━━━━  9s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36905.6133 

<div class="k-default-codeblock">
```

```
</div>
 198/469 ━━━━━━━━[37m━━━━━━━━━━━━  10s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36908.1758

<div class="k-default-codeblock">
```

```
</div>
 200/469 ━━━━━━━━[37m━━━━━━━━━━━━  9s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36913.2930 

<div class="k-default-codeblock">
```

```
</div>
 202/469 ━━━━━━━━[37m━━━━━━━━━━━━  9s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36918.4141

<div class="k-default-codeblock">
```

```
</div>
 204/469 ━━━━━━━━[37m━━━━━━━━━━━━  9s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36923.5352

<div class="k-default-codeblock">
```

```
</div>
 206/469 ━━━━━━━━[37m━━━━━━━━━━━━  9s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36928.6523

<div class="k-default-codeblock">
```

```
</div>
 208/469 ━━━━━━━━[37m━━━━━━━━━━━━  9s 37ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 36933.7734

<div class="k-default-codeblock">
```

```
</div>
 209/469 ━━━━━━━━[37m━━━━━━━━━━━━  9s 38ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36936.3359

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  9s 38ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36941.4531

<div class="k-default-codeblock">
```

```
</div>
 213/469 ━━━━━━━━━[37m━━━━━━━━━━━  9s 38ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36946.5742

<div class="k-default-codeblock">
```

```
</div>
 215/469 ━━━━━━━━━[37m━━━━━━━━━━━  9s 38ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36951.6953

<div class="k-default-codeblock">
```

```
</div>
 217/469 ━━━━━━━━━[37m━━━━━━━━━━━  9s 38ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36956.8125

<div class="k-default-codeblock">
```

```
</div>
 219/469 ━━━━━━━━━[37m━━━━━━━━━━━  9s 38ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36961.9336

<div class="k-default-codeblock">
```

```
</div>
 221/469 ━━━━━━━━━[37m━━━━━━━━━━━  9s 38ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36967.0547

<div class="k-default-codeblock">
```

```
</div>
 222/469 ━━━━━━━━━[37m━━━━━━━━━━━  9s 38ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36969.6133

<div class="k-default-codeblock">
```

```
</div>
 224/469 ━━━━━━━━━[37m━━━━━━━━━━━  9s 38ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36974.7344

<div class="k-default-codeblock">
```

```
</div>
 225/469 ━━━━━━━━━[37m━━━━━━━━━━━  26s 107ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36977.2930

<div class="k-default-codeblock">
```

```
</div>
 226/469 ━━━━━━━━━[37m━━━━━━━━━━━  26s 107ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36979.8555

<div class="k-default-codeblock">
```

```
</div>
 227/469 ━━━━━━━━━[37m━━━━━━━━━━━  26s 109ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36982.4141

<div class="k-default-codeblock">
```

```
</div>
 228/469 ━━━━━━━━━[37m━━━━━━━━━━━  26s 109ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36984.9766

<div class="k-default-codeblock">
```

```
</div>
 229/469 ━━━━━━━━━[37m━━━━━━━━━━━  27s 113ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36987.5352

<div class="k-default-codeblock">
```

```
</div>
 230/469 ━━━━━━━━━[37m━━━━━━━━━━━  26s 112ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36990.0938

<div class="k-default-codeblock">
```

```
</div>
 231/469 ━━━━━━━━━[37m━━━━━━━━━━━  26s 112ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36992.6562

<div class="k-default-codeblock">
```

```
</div>
 232/469 ━━━━━━━━━[37m━━━━━━━━━━━  26s 112ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36995.2148

<div class="k-default-codeblock">
```

```
</div>
 233/469 ━━━━━━━━━[37m━━━━━━━━━━━  26s 112ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 36997.7734

<div class="k-default-codeblock">
```

```
</div>
 235/469 ━━━━━━━━━━[37m━━━━━━━━━━  26s 111ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37002.8945

<div class="k-default-codeblock">
```

```
</div>
 236/469 ━━━━━━━━━━[37m━━━━━━━━━━  25s 111ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37005.4531

<div class="k-default-codeblock">
```

```
</div>
 238/469 ━━━━━━━━━━[37m━━━━━━━━━━  25s 111ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37010.5742

<div class="k-default-codeblock">
```

```
</div>
 240/469 ━━━━━━━━━━[37m━━━━━━━━━━  25s 110ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37015.6953

<div class="k-default-codeblock">
```

```
</div>
 242/469 ━━━━━━━━━━[37m━━━━━━━━━━  24s 110ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37020.8164

<div class="k-default-codeblock">
```

```
</div>
 244/469 ━━━━━━━━━━[37m━━━━━━━━━━  24s 109ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37025.9336

<div class="k-default-codeblock">
```

```
</div>
 246/469 ━━━━━━━━━━[37m━━━━━━━━━━  24s 108ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37031.0547

<div class="k-default-codeblock">
```

```
</div>
 248/469 ━━━━━━━━━━[37m━━━━━━━━━━  23s 108ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37036.1758

<div class="k-default-codeblock">
```

```
</div>
 250/469 ━━━━━━━━━━[37m━━━━━━━━━━  23s 107ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37041.2930

<div class="k-default-codeblock">
```

```
</div>
 252/469 ━━━━━━━━━━[37m━━━━━━━━━━  23s 107ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37046.4141

<div class="k-default-codeblock">
```

```
</div>
 254/469 ━━━━━━━━━━[37m━━━━━━━━━━  22s 106ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37051.5352

<div class="k-default-codeblock">
```

```
</div>
 256/469 ━━━━━━━━━━[37m━━━━━━━━━━  22s 106ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37056.6562

<div class="k-default-codeblock">
```

```
</div>
 258/469 ━━━━━━━━━━━[37m━━━━━━━━━  22s 105ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37061.7734

<div class="k-default-codeblock">
```

```
</div>
 260/469 ━━━━━━━━━━━[37m━━━━━━━━━  21s 105ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37066.8945

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  21s 104ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37072.0156

<div class="k-default-codeblock">
```

```
</div>
 264/469 ━━━━━━━━━━━[37m━━━━━━━━━  21s 104ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37077.1328

<div class="k-default-codeblock">
```

```
</div>
 266/469 ━━━━━━━━━━━[37m━━━━━━━━━  20s 103ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37082.2539

<div class="k-default-codeblock">
```

```
</div>
 268/469 ━━━━━━━━━━━[37m━━━━━━━━━  20s 103ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37087.3750

<div class="k-default-codeblock">
```

```
</div>
 270/469 ━━━━━━━━━━━[37m━━━━━━━━━  20s 102ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37092.4961

<div class="k-default-codeblock">
```

```
</div>
 272/469 ━━━━━━━━━━━[37m━━━━━━━━━  19s 101ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37097.6133

<div class="k-default-codeblock">
```

```
</div>
 274/469 ━━━━━━━━━━━[37m━━━━━━━━━  19s 101ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37102.7344

<div class="k-default-codeblock">
```

```
</div>
 276/469 ━━━━━━━━━━━[37m━━━━━━━━━  19s 100ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37107.8555

<div class="k-default-codeblock">
```

```
</div>
 279/469 ━━━━━━━━━━━[37m━━━━━━━━━  18s 100ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37115.5352

<div class="k-default-codeblock">
```

```
</div>
 282/469 ━━━━━━━━━━━━[37m━━━━━━━━  18s 99ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37123.2148 

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  18s 98ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37130.8945

<div class="k-default-codeblock">
```

```
</div>
 288/469 ━━━━━━━━━━━━[37m━━━━━━━━  17s 97ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 37138.5742

<div class="k-default-codeblock">
```

```
</div>
 291/469 ━━━━━━━━━━━━[37m━━━━━━━━  17s 96ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37146.2539

<div class="k-default-codeblock">
```

```
</div>
 294/469 ━━━━━━━━━━━━[37m━━━━━━━━  16s 95ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37153.9336

<div class="k-default-codeblock">
```

```
</div>
 297/469 ━━━━━━━━━━━━[37m━━━━━━━━  16s 95ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37161.6133

<div class="k-default-codeblock">
```

```
</div>
 300/469 ━━━━━━━━━━━━[37m━━━━━━━━  15s 94ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37169.2930

<div class="k-default-codeblock">
```

```
</div>
 304/469 ━━━━━━━━━━━━[37m━━━━━━━━  15s 93ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37179.5352

<div class="k-default-codeblock">
```

```
</div>
 308/469 ━━━━━━━━━━━━━[37m━━━━━━━  14s 92ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37189.7734

<div class="k-default-codeblock">
```

```
</div>
 312/469 ━━━━━━━━━━━━━[37m━━━━━━━  14s 91ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37200.0156

<div class="k-default-codeblock">
```

```
</div>
 316/469 ━━━━━━━━━━━━━[37m━━━━━━━  13s 90ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37210.2539

<div class="k-default-codeblock">
```

```
</div>
 320/469 ━━━━━━━━━━━━━[37m━━━━━━━  13s 89ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37220.4961

<div class="k-default-codeblock">
```

```
</div>
 323/469 ━━━━━━━━━━━━━[37m━━━━━━━  12s 88ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37228.1758

<div class="k-default-codeblock">
```

```
</div>
 326/469 ━━━━━━━━━━━━━[37m━━━━━━━  12s 88ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37235.8555

<div class="k-default-codeblock">
```

```
</div>
 329/469 ━━━━━━━━━━━━━━[37m━━━━━━  12s 87ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37243.5352

<div class="k-default-codeblock">
```

```
</div>
 332/469 ━━━━━━━━━━━━━━[37m━━━━━━  11s 86ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37251.2148

<div class="k-default-codeblock">
```

```
</div>
 335/469 ━━━━━━━━━━━━━━[37m━━━━━━  11s 86ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37258.8945

<div class="k-default-codeblock">
```

```
</div>
 338/469 ━━━━━━━━━━━━━━[37m━━━━━━  11s 85ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37266.5742

<div class="k-default-codeblock">
```

```
</div>
 342/469 ━━━━━━━━━━━━━━[37m━━━━━━  10s 84ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37276.8164

<div class="k-default-codeblock">
```

```
</div>
 346/469 ━━━━━━━━━━━━━━[37m━━━━━━  10s 84ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37287.0547

<div class="k-default-codeblock">
```

```
</div>
 350/469 ━━━━━━━━━━━━━━[37m━━━━━━  9s 83ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37297.2969 

<div class="k-default-codeblock">
```

```
</div>
 353/469 ━━━━━━━━━━━━━━━[37m━━━━━  9s 82ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37304.9766

<div class="k-default-codeblock">
```

```
</div>
 357/469 ━━━━━━━━━━━━━━━[37m━━━━━  9s 81ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37315.2148

<div class="k-default-codeblock">
```

```
</div>
 360/469 ━━━━━━━━━━━━━━━[37m━━━━━  8s 81ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37322.8945

<div class="k-default-codeblock">
```

```
</div>
 363/469 ━━━━━━━━━━━━━━━[37m━━━━━  8s 80ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37330.5742

<div class="k-default-codeblock">
```

```
</div>
 367/469 ━━━━━━━━━━━━━━━[37m━━━━━  8s 80ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37340.8164

<div class="k-default-codeblock">
```

```
</div>
 370/469 ━━━━━━━━━━━━━━━[37m━━━━━  7s 79ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37348.4961

<div class="k-default-codeblock">
```

```
</div>
 373/469 ━━━━━━━━━━━━━━━[37m━━━━━  7s 79ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37356.1758

<div class="k-default-codeblock">
```

```
</div>
 377/469 ━━━━━━━━━━━━━━━━[37m━━━━  7s 78ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37366.4141

<div class="k-default-codeblock">
```

```
</div>
 381/469 ━━━━━━━━━━━━━━━━[37m━━━━  6s 77ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37376.6562

<div class="k-default-codeblock">
```

```
</div>
 384/469 ━━━━━━━━━━━━━━━━[37m━━━━  6s 77ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37384.3359

<div class="k-default-codeblock">
```

```
</div>
 387/469 ━━━━━━━━━━━━━━━━[37m━━━━  6s 76ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37392.0156

<div class="k-default-codeblock">
```

```
</div>
 390/469 ━━━━━━━━━━━━━━━━[37m━━━━  6s 76ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37399.6953

<div class="k-default-codeblock">
```

```
</div>
 392/469 ━━━━━━━━━━━━━━━━[37m━━━━  5s 76ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37404.8164

<div class="k-default-codeblock">
```

```
</div>
 395/469 ━━━━━━━━━━━━━━━━[37m━━━━  5s 76ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37412.4961

<div class="k-default-codeblock">
```

```
</div>
 398/469 ━━━━━━━━━━━━━━━━[37m━━━━  5s 75ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37420.1758

<div class="k-default-codeblock">
```

```
</div>
 401/469 ━━━━━━━━━━━━━━━━━[37m━━━  5s 75ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37427.8555

<div class="k-default-codeblock">
```

```
</div>
 404/469 ━━━━━━━━━━━━━━━━━[37m━━━  4s 74ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37435.5352

<div class="k-default-codeblock">
```

```
</div>
 407/469 ━━━━━━━━━━━━━━━━━[37m━━━  4s 74ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37443.2148

<div class="k-default-codeblock">
```

```
</div>
 411/469 ━━━━━━━━━━━━━━━━━[37m━━━  4s 73ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37453.4531

<div class="k-default-codeblock">
```

```
</div>
 414/469 ━━━━━━━━━━━━━━━━━[37m━━━  4s 73ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37461.1367

<div class="k-default-codeblock">
```

```
</div>
 417/469 ━━━━━━━━━━━━━━━━━[37m━━━  3s 73ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37468.8164

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  3s 72ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37479.0547

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  3s 72ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37489.2930

<div class="k-default-codeblock">
```

```
</div>
 429/469 ━━━━━━━━━━━━━━━━━━[37m━━  2s 71ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37499.5352

<div class="k-default-codeblock">
```

```
</div>
 433/469 ━━━━━━━━━━━━━━━━━━[37m━━  2s 70ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37509.7734

<div class="k-default-codeblock">
```

```
</div>
 437/469 ━━━━━━━━━━━━━━━━━━[37m━━  2s 70ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37520.0156

<div class="k-default-codeblock">
```

```
</div>
 440/469 ━━━━━━━━━━━━━━━━━━[37m━━  2s 70ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37527.6953

<div class="k-default-codeblock">
```

```
</div>
 444/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 69ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37537.9336

<div class="k-default-codeblock">
```

```
</div>
 448/469 ━━━━━━━━━━━━━━━━━━━[37m━  1s 69ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37548.1758

<div class="k-default-codeblock">
```

```
</div>
 452/469 ━━━━━━━━━━━━━━━━━━━[37m━  1s 68ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37558.4141

<div class="k-default-codeblock">
```

```
</div>
 454/469 ━━━━━━━━━━━━━━━━━━━[37m━  1s 68ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37563.5352

<div class="k-default-codeblock">
```

```
</div>
 457/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 68ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37571.2148

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 67ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37581.4531

<div class="k-default-codeblock">
```

```
</div>
 465/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 67ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37591.6953

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37601.9336

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 32s 67ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 37604.4766 - val_loss: 0.1000 - val_moe_loss: 39198.7227


<div class="k-default-codeblock">
```
Epoch 15/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 39203.8633

<div class="k-default-codeblock">
```

```
</div>
   5/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9954 - loss: 0.1000 - moe_loss: 39214.0938 

<div class="k-default-codeblock">
```

```
</div>
   9/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9945 - loss: 0.1000 - moe_loss: 39224.3320

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 39234.5742

<div class="k-default-codeblock">
```

```
</div>
  17/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39244.8125

<div class="k-default-codeblock">
```

```
</div>
  21/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39255.0547

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39265.2930

<div class="k-default-codeblock">
```

```
</div>
  29/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 39275.5352

<div class="k-default-codeblock">
```

```
</div>
  33/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 39285.7773

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 39296.0156

<div class="k-default-codeblock">
```

```
</div>
  41/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39306.2539

<div class="k-default-codeblock">
```

```
</div>
  45/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39316.4961

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39326.7344

<div class="k-default-codeblock">
```

```
</div>
  53/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39336.9766

<div class="k-default-codeblock">
```

```
</div>
  57/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39347.2148

<div class="k-default-codeblock">
```

```
</div>
  61/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39357.4531

<div class="k-default-codeblock">
```

```
</div>
  65/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39367.6953

<div class="k-default-codeblock">
```

```
</div>
  69/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39377.9336

<div class="k-default-codeblock">
```

```
</div>
  72/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39385.6133

<div class="k-default-codeblock">
```

```
</div>
  76/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39395.8555

<div class="k-default-codeblock">
```

```
</div>
  80/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39406.0938

<div class="k-default-codeblock">
```

```
</div>
  84/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39416.3320

<div class="k-default-codeblock">
```

```
</div>
  88/469 ━━━[37m━━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39426.5742

<div class="k-default-codeblock">
```

```
</div>
  92/469 ━━━[37m━━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39436.8125

<div class="k-default-codeblock">
```

```
</div>
  95/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39444.4922

<div class="k-default-codeblock">
```

```
</div>
  98/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39452.1758

<div class="k-default-codeblock">
```

```
</div>
 101/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39459.8555

<div class="k-default-codeblock">
```

```
</div>
 102/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39462.4141

<div class="k-default-codeblock">
```

```
</div>
 103/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39464.9727

<div class="k-default-codeblock">
```

```
</div>
 105/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39470.0938

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39477.7734

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39485.4531

<div class="k-default-codeblock">
```

```
</div>
 114/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39493.1328

<div class="k-default-codeblock">
```

```
</div>
 117/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 39500.8125

<div class="k-default-codeblock">
```

```
</div>
 120/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39508.4922

<div class="k-default-codeblock">
```

```
</div>
 123/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39516.1719

<div class="k-default-codeblock">
```

```
</div>
 126/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39523.8555

<div class="k-default-codeblock">
```

```
</div>
 129/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39531.5352

<div class="k-default-codeblock">
```

```
</div>
 132/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39539.2148

<div class="k-default-codeblock">
```

```
</div>
 135/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39546.8945

<div class="k-default-codeblock">
```

```
</div>
 138/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39554.5742

<div class="k-default-codeblock">
```

```
</div>
 141/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39562.2539

<div class="k-default-codeblock">
```

```
</div>
 144/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39569.9336

<div class="k-default-codeblock">
```

```
</div>
 147/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39577.6133

<div class="k-default-codeblock">
```

```
</div>
 150/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39585.2930

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39592.9727

<div class="k-default-codeblock">
```

```
</div>
 156/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39600.6523

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39608.3320

<div class="k-default-codeblock">
```

```
</div>
 162/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39616.0117

<div class="k-default-codeblock">
```

```
</div>
 165/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39623.6914

<div class="k-default-codeblock">
```

```
</div>
 168/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39631.3750

<div class="k-default-codeblock">
```

```
</div>
 172/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39641.6133

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39649.2930

<div class="k-default-codeblock">
```

```
</div>
 179/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39659.5352

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39669.7734

<div class="k-default-codeblock">
```

```
</div>
 187/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39680.0117

<div class="k-default-codeblock">
```

```
</div>
 190/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39687.6953

<div class="k-default-codeblock">
```

```
</div>
 194/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39697.9336

<div class="k-default-codeblock">
```

```
</div>
 198/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39708.1719

<div class="k-default-codeblock">
```

```
</div>
 201/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39715.8555

<div class="k-default-codeblock">
```

```
</div>
 204/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39723.5352

<div class="k-default-codeblock">
```

```
</div>
 208/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39733.7734

<div class="k-default-codeblock">
```

```
</div>
 212/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39744.0156

<div class="k-default-codeblock">
```

```
</div>
 216/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39754.2539

<div class="k-default-codeblock">
```

```
</div>
 219/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39761.9336

<div class="k-default-codeblock">
```

```
</div>
 223/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39772.1758

<div class="k-default-codeblock">
```

```
</div>
 226/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39779.8555

<div class="k-default-codeblock">
```

```
</div>
 230/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39790.0938

<div class="k-default-codeblock">
```

```
</div>
 234/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39800.3359

<div class="k-default-codeblock">
```

```
</div>
 238/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39810.5742

<div class="k-default-codeblock">
```

```
</div>
 242/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39820.8125

<div class="k-default-codeblock">
```

```
</div>
 245/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39828.4922

<div class="k-default-codeblock">
```

```
</div>
 248/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39836.1758

<div class="k-default-codeblock">
```

```
</div>
 252/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39846.4141

<div class="k-default-codeblock">
```

```
</div>
 256/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39856.6523

<div class="k-default-codeblock">
```

```
</div>
 259/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39864.3320

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39872.0156

<div class="k-default-codeblock">
```

```
</div>
 265/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39879.6953

<div class="k-default-codeblock">
```

```
</div>
 269/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39889.9336

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39900.1758

<div class="k-default-codeblock">
```

```
</div>
 277/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39910.4141

<div class="k-default-codeblock">
```

```
</div>
 281/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 39920.6523

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39930.8945

<div class="k-default-codeblock">
```

```
</div>
 289/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39941.1328

<div class="k-default-codeblock">
```

```
</div>
 292/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39948.8125

<div class="k-default-codeblock">
```

```
</div>
 296/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39959.0547

<div class="k-default-codeblock">
```

```
</div>
 299/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39966.7344

<div class="k-default-codeblock">
```

```
</div>
 303/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39976.9727

<div class="k-default-codeblock">
```

```
</div>
 307/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39987.2148

<div class="k-default-codeblock">
```

```
</div>
 311/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 39997.4531

<div class="k-default-codeblock">
```

```
</div>
 315/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40007.6953

<div class="k-default-codeblock">
```

```
</div>
 319/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40017.9336

<div class="k-default-codeblock">
```

```
</div>
 323/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40028.1758

<div class="k-default-codeblock">
```

```
</div>
 327/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40038.4141

<div class="k-default-codeblock">
```

```
</div>
 331/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40048.6523

<div class="k-default-codeblock">
```

```
</div>
 335/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40058.8945

<div class="k-default-codeblock">
```

```
</div>
 339/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40069.1328

<div class="k-default-codeblock">
```

```
</div>
 343/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40079.3750

<div class="k-default-codeblock">
```

```
</div>
 346/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40087.0547

<div class="k-default-codeblock">
```

```
</div>
 350/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40097.2930

<div class="k-default-codeblock">
```

```
</div>
 354/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40107.5352

<div class="k-default-codeblock">
```

```
</div>
 358/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40117.7734

<div class="k-default-codeblock">
```

```
</div>
 362/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40128.0156

<div class="k-default-codeblock">
```

```
</div>
 366/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40138.2539

<div class="k-default-codeblock">
```

```
</div>
 369/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40145.9336

<div class="k-default-codeblock">
```

```
</div>
 373/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40156.1758

<div class="k-default-codeblock">
```

```
</div>
 377/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40166.4141

<div class="k-default-codeblock">
```

```
</div>
 381/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40176.6523

<div class="k-default-codeblock">
```

```
</div>
 385/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40186.8945

<div class="k-default-codeblock">
```

```
</div>
 389/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40197.1328

<div class="k-default-codeblock">
```

```
</div>
 393/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40207.3750

<div class="k-default-codeblock">
```

```
</div>
 396/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40215.0547

<div class="k-default-codeblock">
```

```
</div>
 400/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40225.2930

<div class="k-default-codeblock">
```

```
</div>
 404/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40235.5352

<div class="k-default-codeblock">
```

```
</div>
 408/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40245.7734

<div class="k-default-codeblock">
```

```
</div>
 412/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40256.0156

<div class="k-default-codeblock">
```

```
</div>
 416/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40266.2539

<div class="k-default-codeblock">
```

```
</div>
 420/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40276.4961

<div class="k-default-codeblock">
```

```
</div>
 423/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40284.1758

<div class="k-default-codeblock">
```

```
</div>
 427/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40294.4141

<div class="k-default-codeblock">
```

```
</div>
 431/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40304.6523

<div class="k-default-codeblock">
```

```
</div>
 435/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40314.8945

<div class="k-default-codeblock">
```

```
</div>
 439/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40325.1328

<div class="k-default-codeblock">
```

```
</div>
 443/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40335.3750

<div class="k-default-codeblock">
```

```
</div>
 447/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40345.6133

<div class="k-default-codeblock">
```

```
</div>
 451/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40355.8555

<div class="k-default-codeblock">
```

```
</div>
 454/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40363.5352

<div class="k-default-codeblock">
```

```
</div>
 457/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40371.2148

<div class="k-default-codeblock">
```

```
</div>
 460/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40378.8945

<div class="k-default-codeblock">
```

```
</div>
 463/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40386.5742

<div class="k-default-codeblock">
```

```
</div>
 466/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40394.2539

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 40404.4766 - val_loss: 0.1000 - val_moe_loss: 41998.7227


<div class="k-default-codeblock">
```
Epoch 16/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  11s 26ms/step - accuracy: 0.9844 - loss: 0.1000 - moe_loss: 42003.8633

<div class="k-default-codeblock">
```

```
</div>
   5/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9882 - loss: 0.1000 - moe_loss: 42014.1016 

<div class="k-default-codeblock">
```

```
</div>
   9/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9901 - loss: 0.1000 - moe_loss: 42024.3359

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9909 - loss: 0.1000 - moe_loss: 42034.5742

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9914 - loss: 0.1000 - moe_loss: 42042.2539

<div class="k-default-codeblock">
```

```
</div>
  20/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 42052.4922

<div class="k-default-codeblock">
```

```
</div>
  24/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 42062.7344

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 42072.9727

<div class="k-default-codeblock">
```

```
</div>
  32/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 42083.2109

<div class="k-default-codeblock">
```

```
</div>
  36/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 42093.4531

<div class="k-default-codeblock">
```

```
</div>
  39/469 ━[37m━━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 42101.1328

<div class="k-default-codeblock">
```

```
</div>
  42/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 42108.8125

<div class="k-default-codeblock">
```

```
</div>
  45/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 42116.4922

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 42126.7344

<div class="k-default-codeblock">
```

```
</div>
  53/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 42136.9727

<div class="k-default-codeblock">
```

```
</div>
  57/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 42147.2148

<div class="k-default-codeblock">
```

```
</div>
  61/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 42157.4531

<div class="k-default-codeblock">
```

```
</div>
  65/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42167.6953

<div class="k-default-codeblock">
```

```
</div>
  69/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42177.9336

<div class="k-default-codeblock">
```

```
</div>
  73/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42188.1758

<div class="k-default-codeblock">
```

```
</div>
  77/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42198.4141

<div class="k-default-codeblock">
```

```
</div>
  81/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42208.6562

<div class="k-default-codeblock">
```

```
</div>
  85/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42218.8945

<div class="k-default-codeblock">
```

```
</div>
  89/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42229.1328

<div class="k-default-codeblock">
```

```
</div>
  93/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42239.3750

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 16ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42249.6133

<div class="k-default-codeblock">
```

```
</div>
 100/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42257.2930

<div class="k-default-codeblock">
```

```
</div>
 104/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42267.5352

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42277.7734

<div class="k-default-codeblock">
```

```
</div>
 112/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42288.0156

<div class="k-default-codeblock">
```

```
</div>
 116/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42298.2539

<div class="k-default-codeblock">
```

```
</div>
 120/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42308.4961

<div class="k-default-codeblock">
```

```
</div>
 124/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42318.7344

<div class="k-default-codeblock">
```

```
</div>
 128/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42328.9766

<div class="k-default-codeblock">
```

```
</div>
 132/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42339.2148

<div class="k-default-codeblock">
```

```
</div>
 136/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42349.4531

<div class="k-default-codeblock">
```

```
</div>
 140/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42359.6953

<div class="k-default-codeblock">
```

```
</div>
 144/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42369.9336

<div class="k-default-codeblock">
```

```
</div>
 148/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 16ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42380.1758

<div class="k-default-codeblock">
```

```
</div>
 150/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42385.2930

<div class="k-default-codeblock">
```

```
</div>
 152/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42390.4141

<div class="k-default-codeblock">
```

```
</div>
 155/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42398.0938

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42408.3359

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42418.5742

<div class="k-default-codeblock">
```

```
</div>
 167/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42428.8125

<div class="k-default-codeblock">
```

```
</div>
 171/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42439.0547

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42449.2930

<div class="k-default-codeblock">
```

```
</div>
 179/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42459.5352

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 42469.7734

<div class="k-default-codeblock">
```

```
</div>
 187/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42480.0156

<div class="k-default-codeblock">
```

```
</div>
 191/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42490.2539

<div class="k-default-codeblock">
```

```
</div>
 195/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42500.4922

<div class="k-default-codeblock">
```

```
</div>
 199/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42510.7344

<div class="k-default-codeblock">
```

```
</div>
 203/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42520.9727

<div class="k-default-codeblock">
```

```
</div>
 207/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42531.2148

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42541.4531

<div class="k-default-codeblock">
```

```
</div>
 214/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42549.1328

<div class="k-default-codeblock">
```

```
</div>
 218/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42559.3750

<div class="k-default-codeblock">
```

```
</div>
 222/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42569.6133

<div class="k-default-codeblock">
```

```
</div>
 226/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42579.8555

<div class="k-default-codeblock">
```

```
</div>
 230/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42590.0938

<div class="k-default-codeblock">
```

```
</div>
 234/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42600.3320

<div class="k-default-codeblock">
```

```
</div>
 238/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42610.5742

<div class="k-default-codeblock">
```

```
</div>
 242/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 42620.8125

<div class="k-default-codeblock">
```

```
</div>
 246/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42631.0547

<div class="k-default-codeblock">
```

```
</div>
 250/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42641.2930

<div class="k-default-codeblock">
```

```
</div>
 254/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42651.5352

<div class="k-default-codeblock">
```

```
</div>
 258/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42661.7734

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42672.0117

<div class="k-default-codeblock">
```

```
</div>
 266/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42682.2539

<div class="k-default-codeblock">
```

```
</div>
 269/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42689.9336

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42700.1719

<div class="k-default-codeblock">
```

```
</div>
 276/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42707.8555

<div class="k-default-codeblock">
```

```
</div>
 279/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42715.5352

<div class="k-default-codeblock">
```

```
</div>
 281/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42720.6523

<div class="k-default-codeblock">
```

```
</div>
 282/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42723.2148

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42730.8945

<div class="k-default-codeblock">
```

```
</div>
 288/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42738.5742

<div class="k-default-codeblock">
```

```
</div>
 291/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 42746.2539

<div class="k-default-codeblock">
```

```
</div>
 294/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42753.9336

<div class="k-default-codeblock">
```

```
</div>
 298/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42764.1719

<div class="k-default-codeblock">
```

```
</div>
 302/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42774.4141

<div class="k-default-codeblock">
```

```
</div>
 305/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42782.0938

<div class="k-default-codeblock">
```

```
</div>
 309/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42792.3320

<div class="k-default-codeblock">
```

```
</div>
 312/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42800.0117

<div class="k-default-codeblock">
```

```
</div>
 315/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42807.6914

<div class="k-default-codeblock">
```

```
</div>
 318/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42815.3750

<div class="k-default-codeblock">
```

```
</div>
 321/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42823.0547

<div class="k-default-codeblock">
```

```
</div>
 324/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42830.7344

<div class="k-default-codeblock">
```

```
</div>
 327/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42838.4141

<div class="k-default-codeblock">
```

```
</div>
 331/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42848.6523

<div class="k-default-codeblock">
```

```
</div>
 335/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42858.8945

<div class="k-default-codeblock">
```

```
</div>
 338/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42866.5742

<div class="k-default-codeblock">
```

```
</div>
 342/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42876.8125

<div class="k-default-codeblock">
```

```
</div>
 346/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42887.0547

<div class="k-default-codeblock">
```

```
</div>
 349/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42894.7344

<div class="k-default-codeblock">
```

```
</div>
 353/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42904.9727

<div class="k-default-codeblock">
```

```
</div>
 357/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 42915.2148

<div class="k-default-codeblock">
```

```
</div>
 361/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42925.4531

<div class="k-default-codeblock">
```

```
</div>
 364/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42933.1328

<div class="k-default-codeblock">
```

```
</div>
 368/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42943.3750

<div class="k-default-codeblock">
```

```
</div>
 371/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42951.0547

<div class="k-default-codeblock">
```

```
</div>
 374/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42958.7344

<div class="k-default-codeblock">
```

```
</div>
 378/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42968.9727

<div class="k-default-codeblock">
```

```
</div>
 382/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42979.2148

<div class="k-default-codeblock">
```

```
</div>
 386/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42989.4531

<div class="k-default-codeblock">
```

```
</div>
 390/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 42999.6914

<div class="k-default-codeblock">
```

```
</div>
 393/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43007.3750

<div class="k-default-codeblock">
```

```
</div>
 397/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43017.6133

<div class="k-default-codeblock">
```

```
</div>
 401/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43027.8516

<div class="k-default-codeblock">
```

```
</div>
 405/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43038.0938

<div class="k-default-codeblock">
```

```
</div>
 409/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43048.3320

<div class="k-default-codeblock">
```

```
</div>
 413/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43058.5742

<div class="k-default-codeblock">
```

```
</div>
 416/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43066.2539

<div class="k-default-codeblock">
```

```
</div>
 420/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43076.4922

<div class="k-default-codeblock">
```

```
</div>
 423/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43084.1719

<div class="k-default-codeblock">
```

```
</div>
 426/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43091.8516

<div class="k-default-codeblock">
```

```
</div>
 430/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43102.0938

<div class="k-default-codeblock">
```

```
</div>
 434/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43112.3320

<div class="k-default-codeblock">
```

```
</div>
 438/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43122.5742

<div class="k-default-codeblock">
```

```
</div>
 442/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43132.8125

<div class="k-default-codeblock">
```

```
</div>
 446/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43143.0547

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43150.7344

<div class="k-default-codeblock">
```

```
</div>
 453/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43160.9727

<div class="k-default-codeblock">
```

```
</div>
 457/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43171.2148

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43181.4531

<div class="k-default-codeblock">
```

```
</div>
 465/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43191.6953

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43201.9297

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 43204.4766 - val_loss: 0.1000 - val_moe_loss: 44798.7227


<div class="k-default-codeblock">
```
Epoch 17/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 26ms/step - accuracy: 1.0000 - loss: 0.1000 - moe_loss: 44803.8477

<div class="k-default-codeblock">
```

```
</div>
   5/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9947 - loss: 0.1000 - moe_loss: 44814.0938 

<div class="k-default-codeblock">
```

```
</div>
   9/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 44824.3359

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 16ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 44834.5742

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 44842.2539

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 44849.9336

<div class="k-default-codeblock">
```

```
</div>
  23/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 44860.1719

<div class="k-default-codeblock">
```

```
</div>
  27/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 44870.4141

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 44880.6523

<div class="k-default-codeblock">
```

```
</div>
  33/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 44885.7734

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 44896.0117

<div class="k-default-codeblock">
```

```
</div>
  41/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 44906.2539

<div class="k-default-codeblock">
```

```
</div>
  45/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 44916.4922

<div class="k-default-codeblock">
```

```
</div>
  48/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 44924.1719

<div class="k-default-codeblock">
```

```
</div>
  51/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 44931.8516

<div class="k-default-codeblock">
```

```
</div>
  55/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 44942.0938

<div class="k-default-codeblock">
```

```
</div>
  59/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 44952.3320

<div class="k-default-codeblock">
```

```
</div>
  63/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 44962.5742

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 44972.8125

<div class="k-default-codeblock">
```

```
</div>
  71/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 44983.0547

<div class="k-default-codeblock">
```

```
</div>
  74/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 44990.7344

<div class="k-default-codeblock">
```

```
</div>
  78/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 45000.9727

<div class="k-default-codeblock">
```

```
</div>
  81/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 45008.6523

<div class="k-default-codeblock">
```

```
</div>
  84/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 45016.3320

<div class="k-default-codeblock">
```

```
</div>
  87/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 45024.0117

<div class="k-default-codeblock">
```

```
</div>
  90/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 45031.6914

<div class="k-default-codeblock">
```

```
</div>
  94/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 45041.9336

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 45049.6133

<div class="k-default-codeblock">
```

```
</div>
 101/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 45059.8516

<div class="k-default-codeblock">
```

```
</div>
 105/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 45070.0938

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 45077.7734

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 45085.4531

<div class="k-default-codeblock">
```

```
</div>
 114/469 ━━━━[37m━━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 45093.1328

<div class="k-default-codeblock">
```

```
</div>
 118/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 45103.3711

<div class="k-default-codeblock">
```

```
</div>
 121/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 45111.0508

<div class="k-default-codeblock">
```

```
</div>
 124/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 45118.7305

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 45126.4102

<div class="k-default-codeblock">
```

```
</div>
 130/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 45134.0938

<div class="k-default-codeblock">
```

```
</div>
 133/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 45141.7734

<div class="k-default-codeblock">
```

```
</div>
 136/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 45149.4531

<div class="k-default-codeblock">
```

```
</div>
 139/469 ━━━━━[37m━━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 45157.1328

<div class="k-default-codeblock">
```

```
</div>
 142/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 45164.8125

<div class="k-default-codeblock">
```

```
</div>
 145/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 45172.4922

<div class="k-default-codeblock">
```

```
</div>
 149/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 45182.7344

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 45192.9727

<div class="k-default-codeblock">
```

```
</div>
 156/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45200.6523

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45208.3320

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45218.5742

<div class="k-default-codeblock">
```

```
</div>
 167/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45228.8125

<div class="k-default-codeblock">
```

```
</div>
 170/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45236.4922

<div class="k-default-codeblock">
```

```
</div>
 174/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45246.7344

<div class="k-default-codeblock">
```

```
</div>
 177/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45254.4141

<div class="k-default-codeblock">
```

```
</div>
 180/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45262.0938

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45269.7734

<div class="k-default-codeblock">
```

```
</div>
 186/469 ━━━━━━━[37m━━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45277.4531

<div class="k-default-codeblock">
```

```
</div>
 189/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45285.1328

<div class="k-default-codeblock">
```

```
</div>
 193/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45295.3711

<div class="k-default-codeblock">
```

```
</div>
 196/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45303.0547

<div class="k-default-codeblock">
```

```
</div>
 199/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45310.7344

<div class="k-default-codeblock">
```

```
</div>
 202/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45318.4141

<div class="k-default-codeblock">
```

```
</div>
 205/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45326.0938

<div class="k-default-codeblock">
```

```
</div>
 208/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45333.7734

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45341.4531

<div class="k-default-codeblock">
```

```
</div>
 214/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45349.1328

<div class="k-default-codeblock">
```

```
</div>
 217/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45356.8125

<div class="k-default-codeblock">
```

```
</div>
 220/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45364.4922

<div class="k-default-codeblock">
```

```
</div>
 223/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45372.1719

<div class="k-default-codeblock">
```

```
</div>
 227/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45382.4141

<div class="k-default-codeblock">
```

```
</div>
 231/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45392.6523

<div class="k-default-codeblock">
```

```
</div>
 234/469 ━━━━━━━━━[37m━━━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45400.3320

<div class="k-default-codeblock">
```

```
</div>
 237/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45408.0117

<div class="k-default-codeblock">
```

```
</div>
 241/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45418.2539

<div class="k-default-codeblock">
```

```
</div>
 244/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45425.9336

<div class="k-default-codeblock">
```

```
</div>
 247/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45433.6133

<div class="k-default-codeblock">
```

```
</div>
 250/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45441.2930

<div class="k-default-codeblock">
```

```
</div>
 253/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45448.9727

<div class="k-default-codeblock">
```

```
</div>
 257/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45459.2109

<div class="k-default-codeblock">
```

```
</div>
 261/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45469.4531

<div class="k-default-codeblock">
```

```
</div>
 264/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45477.1328

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45484.8125

<div class="k-default-codeblock">
```

```
</div>
 270/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45492.4922

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45500.1719

<div class="k-default-codeblock">
```

```
</div>
 276/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45507.8516

<div class="k-default-codeblock">
```

```
</div>
 279/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45515.5312

<div class="k-default-codeblock">
```

```
</div>
 282/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45523.2109

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45530.8945

<div class="k-default-codeblock">
```

```
</div>
 288/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45538.5742

<div class="k-default-codeblock">
```

```
</div>
 291/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45546.2539

<div class="k-default-codeblock">
```

```
</div>
 294/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45553.9336

<div class="k-default-codeblock">
```

```
</div>
 297/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45561.6133

<div class="k-default-codeblock">
```

```
</div>
 300/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45569.2930

<div class="k-default-codeblock">
```

```
</div>
 303/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45576.9727

<div class="k-default-codeblock">
```

```
</div>
 306/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45584.6523

<div class="k-default-codeblock">
```

```
</div>
 307/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45587.2109

<div class="k-default-codeblock">
```

```
</div>
 310/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45594.8906

<div class="k-default-codeblock">
```

```
</div>
 313/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45602.5703

<div class="k-default-codeblock">
```

```
</div>
 316/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45610.2539

<div class="k-default-codeblock">
```

```
</div>
 319/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45617.9336

<div class="k-default-codeblock">
```

```
</div>
 322/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45625.6133

<div class="k-default-codeblock">
```

```
</div>
 326/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45635.8516

<div class="k-default-codeblock">
```

```
</div>
 329/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45643.5312

<div class="k-default-codeblock">
```

```
</div>
 332/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45651.2109

<div class="k-default-codeblock">
```

```
</div>
 335/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45658.8906

<div class="k-default-codeblock">
```

```
</div>
 338/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45666.5703

<div class="k-default-codeblock">
```

```
</div>
 341/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45674.2539

<div class="k-default-codeblock">
```

```
</div>
 344/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45681.9336

<div class="k-default-codeblock">
```

```
</div>
 347/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45689.6133

<div class="k-default-codeblock">
```

```
</div>
 350/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45697.2930

<div class="k-default-codeblock">
```

```
</div>
 353/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45704.9727

<div class="k-default-codeblock">
```

```
</div>
 356/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45712.6523

<div class="k-default-codeblock">
```

```
</div>
 359/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45720.3320

<div class="k-default-codeblock">
```

```
</div>
 362/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45728.0117

<div class="k-default-codeblock">
```

```
</div>
 365/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45735.6914

<div class="k-default-codeblock">
```

```
</div>
 368/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45743.3711

<div class="k-default-codeblock">
```

```
</div>
 371/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45751.0508

<div class="k-default-codeblock">
```

```
</div>
 374/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45758.7305

<div class="k-default-codeblock">
```

```
</div>
 377/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45766.4141

<div class="k-default-codeblock">
```

```
</div>
 380/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45774.0938

<div class="k-default-codeblock">
```

```
</div>
 383/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45781.7734

<div class="k-default-codeblock">
```

```
</div>
 386/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45789.4531

<div class="k-default-codeblock">
```

```
</div>
 389/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 45797.1328

<div class="k-default-codeblock">
```

```
</div>
 392/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45804.8125

<div class="k-default-codeblock">
```

```
</div>
 395/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45812.4922

<div class="k-default-codeblock">
```

```
</div>
 398/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45820.1719

<div class="k-default-codeblock">
```

```
</div>
 401/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45827.8516

<div class="k-default-codeblock">
```

```
</div>
 404/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45835.5312

<div class="k-default-codeblock">
```

```
</div>
 407/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45843.2109

<div class="k-default-codeblock">
```

```
</div>
 410/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45850.8906

<div class="k-default-codeblock">
```

```
</div>
 413/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45858.5742

<div class="k-default-codeblock">
```

```
</div>
 416/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45866.2539

<div class="k-default-codeblock">
```

```
</div>
 419/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45873.9336

<div class="k-default-codeblock">
```

```
</div>
 422/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45881.6133

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45889.2930

<div class="k-default-codeblock">
```

```
</div>
 428/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45896.9727

<div class="k-default-codeblock">
```

```
</div>
 431/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45904.6523

<div class="k-default-codeblock">
```

```
</div>
 434/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45912.3320

<div class="k-default-codeblock">
```

```
</div>
 437/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45920.0117

<div class="k-default-codeblock">
```

```
</div>
 440/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45927.6914

<div class="k-default-codeblock">
```

```
</div>
 443/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45935.3711

<div class="k-default-codeblock">
```

```
</div>
 446/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45943.0547

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45950.7344

<div class="k-default-codeblock">
```

```
</div>
 452/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45958.4141

<div class="k-default-codeblock">
```

```
</div>
 455/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45966.0938

<div class="k-default-codeblock">
```

```
</div>
 458/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45973.7734

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45981.4531

<div class="k-default-codeblock">
```

```
</div>
 464/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45989.1328

<div class="k-default-codeblock">
```

```
</div>
 467/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 45996.8125

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 46004.4766 - val_loss: 0.1000 - val_moe_loss: 47598.7227


<div class="k-default-codeblock">
```
Epoch 18/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  12s 27ms/step - accuracy: 1.0000 - loss: 0.1000 - moe_loss: 47603.8477

<div class="k-default-codeblock">
```

```
</div>
   4/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9967 - loss: 0.1000 - moe_loss: 47611.5312 

<div class="k-default-codeblock">
```

```
</div>
   7/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9965 - loss: 0.1000 - moe_loss: 47619.2109

<div class="k-default-codeblock">
```

```
</div>
  10/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9965 - loss: 0.1000 - moe_loss: 47626.8906

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9966 - loss: 0.1000 - moe_loss: 47634.5703

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9964 - loss: 0.1000 - moe_loss: 47642.2500

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9961 - loss: 0.1000 - moe_loss: 47649.9336

<div class="k-default-codeblock">
```

```
</div>
  22/469 [37m━━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9957 - loss: 0.1000 - moe_loss: 47657.6133

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9954 - loss: 0.1000 - moe_loss: 47665.2930

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9952 - loss: 0.1000 - moe_loss: 47672.9727

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9950 - loss: 0.1000 - moe_loss: 47680.6523

<div class="k-default-codeblock">
```

```
</div>
  34/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9950 - loss: 0.1000 - moe_loss: 47688.3359

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9949 - loss: 0.1000 - moe_loss: 47696.0156

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9948 - loss: 0.1000 - moe_loss: 47703.6953

<div class="k-default-codeblock">
```

```
</div>
  43/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9947 - loss: 0.1000 - moe_loss: 47711.3750

<div class="k-default-codeblock">
```

```
</div>
  46/469 ━[37m━━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9946 - loss: 0.1000 - moe_loss: 47719.0547

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9946 - loss: 0.1000 - moe_loss: 47726.7344

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9945 - loss: 0.1000 - moe_loss: 47734.4141

<div class="k-default-codeblock">
```

```
</div>
  55/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9945 - loss: 0.1000 - moe_loss: 47742.0938

<div class="k-default-codeblock">
```

```
</div>
  58/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9945 - loss: 0.1000 - moe_loss: 47749.7734

<div class="k-default-codeblock">
```

```
</div>
  61/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9945 - loss: 0.1000 - moe_loss: 47757.4570

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9945 - loss: 0.1000 - moe_loss: 47765.1367

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 17ms/step - accuracy: 0.9945 - loss: 0.1000 - moe_loss: 47772.8164

<div class="k-default-codeblock">
```

```
</div>
  70/469 ━━[37m━━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9945 - loss: 0.1000 - moe_loss: 47780.4961

<div class="k-default-codeblock">
```

```
</div>
  73/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 47788.1758

<div class="k-default-codeblock">
```

```
</div>
  76/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 47795.8555

<div class="k-default-codeblock">
```

```
</div>
  79/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 47803.5352

<div class="k-default-codeblock">
```

```
</div>
  82/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 47811.2148

<div class="k-default-codeblock">
```

```
</div>
  85/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 47818.8945

<div class="k-default-codeblock">
```

```
</div>
  88/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 47826.5742

<div class="k-default-codeblock">
```

```
</div>
  91/469 ━━━[37m━━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9944 - loss: 0.1000 - moe_loss: 47834.2539

<div class="k-default-codeblock">
```

```
</div>
  94/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47841.9336

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47849.6133

<div class="k-default-codeblock">
```

```
</div>
 100/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47857.2930

<div class="k-default-codeblock">
```

```
</div>
 103/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47864.9727

<div class="k-default-codeblock">
```

```
</div>
 106/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 17ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47872.6562

<div class="k-default-codeblock">
```

```
</div>
 107/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 18ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47875.2148

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47877.7734

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47885.4531

<div class="k-default-codeblock">
```

```
</div>
 114/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47893.1328

<div class="k-default-codeblock">
```

```
</div>
 117/469 ━━━━[37m━━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47900.8164

<div class="k-default-codeblock">
```

```
</div>
 121/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 47911.0547

<div class="k-default-codeblock">
```

```
</div>
 124/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47918.7344

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47926.4141

<div class="k-default-codeblock">
```

```
</div>
 130/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47934.0938

<div class="k-default-codeblock">
```

```
</div>
 133/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47941.7734

<div class="k-default-codeblock">
```

```
</div>
 136/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47949.4531

<div class="k-default-codeblock">
```

```
</div>
 139/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47957.1328

<div class="k-default-codeblock">
```

```
</div>
 142/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47964.8125

<div class="k-default-codeblock">
```

```
</div>
 145/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47972.4922

<div class="k-default-codeblock">
```

```
</div>
 148/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47980.1758

<div class="k-default-codeblock">
```

```
</div>
 151/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47987.8555

<div class="k-default-codeblock">
```

```
</div>
 154/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 47995.5352

<div class="k-default-codeblock">
```

```
</div>
 157/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48003.2148

<div class="k-default-codeblock">
```

```
</div>
 160/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48010.8945

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  5s 19ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48018.5742

<div class="k-default-codeblock">
```

```
</div>
 166/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48026.2539

<div class="k-default-codeblock">
```

```
</div>
 169/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48033.9336

<div class="k-default-codeblock">
```

```
</div>
 172/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48041.6133

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48049.2930

<div class="k-default-codeblock">
```

```
</div>
 178/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48056.9727

<div class="k-default-codeblock">
```

```
</div>
 181/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48064.6562

<div class="k-default-codeblock">
```

```
</div>
 184/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48072.3359

<div class="k-default-codeblock">
```

```
</div>
 187/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48080.0156

<div class="k-default-codeblock">
```

```
</div>
 190/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48087.6953

<div class="k-default-codeblock">
```

```
</div>
 193/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48095.3750

<div class="k-default-codeblock">
```

```
</div>
 196/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48103.0547

<div class="k-default-codeblock">
```

```
</div>
 199/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48110.7344

<div class="k-default-codeblock">
```

```
</div>
 202/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48118.4141

<div class="k-default-codeblock">
```

```
</div>
 205/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48126.0938

<div class="k-default-codeblock">
```

```
</div>
 208/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48133.7734

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48141.4531

<div class="k-default-codeblock">
```

```
</div>
 214/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48149.1328

<div class="k-default-codeblock">
```

```
</div>
 217/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48156.8164

<div class="k-default-codeblock">
```

```
</div>
 220/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 48164.4961

<div class="k-default-codeblock">
```

```
</div>
 223/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48172.1758

<div class="k-default-codeblock">
```

```
</div>
 226/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48179.8555

<div class="k-default-codeblock">
```

```
</div>
 229/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48187.5352

<div class="k-default-codeblock">
```

```
</div>
 232/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48195.2148

<div class="k-default-codeblock">
```

```
</div>
 235/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48202.8945

<div class="k-default-codeblock">
```

```
</div>
 238/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48210.5742

<div class="k-default-codeblock">
```

```
</div>
 241/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48218.2539

<div class="k-default-codeblock">
```

```
</div>
 244/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48225.9336

<div class="k-default-codeblock">
```

```
</div>
 247/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48233.6133

<div class="k-default-codeblock">
```

```
</div>
 250/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48241.2930

<div class="k-default-codeblock">
```

```
</div>
 253/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48248.9727

<div class="k-default-codeblock">
```

```
</div>
 256/469 ━━━━━━━━━━[37m━━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48256.6562

<div class="k-default-codeblock">
```

```
</div>
 259/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48264.3359

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48272.0156

<div class="k-default-codeblock">
```

```
</div>
 265/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48279.6953

<div class="k-default-codeblock">
```

```
</div>
 268/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48287.3750

<div class="k-default-codeblock">
```

```
</div>
 271/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48295.0547

<div class="k-default-codeblock">
```

```
</div>
 274/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48302.7344

<div class="k-default-codeblock">
```

```
</div>
 277/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48310.4141

<div class="k-default-codeblock">
```

```
</div>
 280/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48318.0938

<div class="k-default-codeblock">
```

```
</div>
 283/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48325.7734

<div class="k-default-codeblock">
```

```
</div>
 286/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48333.4531

<div class="k-default-codeblock">
```

```
</div>
 289/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48341.1328

<div class="k-default-codeblock">
```

```
</div>
 292/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48348.8125

<div class="k-default-codeblock">
```

```
</div>
 295/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48356.4922

<div class="k-default-codeblock">
```

```
</div>
 298/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48364.1758

<div class="k-default-codeblock">
```

```
</div>
 301/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48371.8555

<div class="k-default-codeblock">
```

```
</div>
 304/469 ━━━━━━━━━━━━[37m━━━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48379.5352

<div class="k-default-codeblock">
```

```
</div>
 307/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48387.2148

<div class="k-default-codeblock">
```

```
</div>
 310/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48394.8945

<div class="k-default-codeblock">
```

```
</div>
 313/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48402.5742

<div class="k-default-codeblock">
```

```
</div>
 316/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48410.2539

<div class="k-default-codeblock">
```

```
</div>
 319/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48417.9336

<div class="k-default-codeblock">
```

```
</div>
 322/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48425.6133

<div class="k-default-codeblock">
```

```
</div>
 325/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48433.2930

<div class="k-default-codeblock">
```

```
</div>
 328/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48440.9727

<div class="k-default-codeblock">
```

```
</div>
 331/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48448.6523

<div class="k-default-codeblock">
```

```
</div>
 334/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48456.3320

<div class="k-default-codeblock">
```

```
</div>
 337/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48464.0156

<div class="k-default-codeblock">
```

```
</div>
 340/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48471.6953

<div class="k-default-codeblock">
```

```
</div>
 343/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48479.3750

<div class="k-default-codeblock">
```

```
</div>
 346/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48487.0547

<div class="k-default-codeblock">
```

```
</div>
 349/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48494.7344

<div class="k-default-codeblock">
```

```
</div>
 352/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48502.4141

<div class="k-default-codeblock">
```

```
</div>
 355/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48510.0938

<div class="k-default-codeblock">
```

```
</div>
 358/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48517.7734

<div class="k-default-codeblock">
```

```
</div>
 361/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48525.4531

<div class="k-default-codeblock">
```

```
</div>
 364/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48533.1328

<div class="k-default-codeblock">
```

```
</div>
 367/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48540.8125

<div class="k-default-codeblock">
```

```
</div>
 370/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48548.4922

<div class="k-default-codeblock">
```

```
</div>
 373/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48556.1719

<div class="k-default-codeblock">
```

```
</div>
 376/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48563.8555

<div class="k-default-codeblock">
```

```
</div>
 379/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48571.5352

<div class="k-default-codeblock">
```

```
</div>
 382/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48579.2148

<div class="k-default-codeblock">
```

```
</div>
 385/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48586.8945

<div class="k-default-codeblock">
```

```
</div>
 388/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48594.5742

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48602.2539

<div class="k-default-codeblock">
```

```
</div>
 394/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48609.9336

<div class="k-default-codeblock">
```

```
</div>
 397/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48617.6133

<div class="k-default-codeblock">
```

```
</div>
 400/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48625.2930

<div class="k-default-codeblock">
```

```
</div>
 403/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48632.9727

<div class="k-default-codeblock">
```

```
</div>
 406/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48640.6523

<div class="k-default-codeblock">
```

```
</div>
 409/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48648.3320

<div class="k-default-codeblock">
```

```
</div>
 412/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48656.0117

<div class="k-default-codeblock">
```

```
</div>
 415/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48663.6914

<div class="k-default-codeblock">
```

```
</div>
 418/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48671.3750

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48679.0547

<div class="k-default-codeblock">
```

```
</div>
 424/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48686.7344

<div class="k-default-codeblock">
```

```
</div>
 427/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48694.4141

<div class="k-default-codeblock">
```

```
</div>
 430/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48702.0938

<div class="k-default-codeblock">
```

```
</div>
 433/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48709.7734

<div class="k-default-codeblock">
```

```
</div>
 436/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 48717.4531

<div class="k-default-codeblock">
```

```
</div>
 439/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48725.1328

<div class="k-default-codeblock">
```

```
</div>
 442/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48732.8125

<div class="k-default-codeblock">
```

```
</div>
 445/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48740.4922

<div class="k-default-codeblock">
```

```
</div>
 448/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48748.1719

<div class="k-default-codeblock">
```

```
</div>
 451/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48755.8516

<div class="k-default-codeblock">
```

```
</div>
 454/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48763.5352

<div class="k-default-codeblock">
```

```
</div>
 457/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48771.2148

<div class="k-default-codeblock">
```

```
</div>
 460/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48778.8945

<div class="k-default-codeblock">
```

```
</div>
 463/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48786.5742

<div class="k-default-codeblock">
```

```
</div>
 466/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48794.2539

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48801.9297

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 9s 19ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 48804.4766 - val_loss: 0.1000 - val_moe_loss: 50398.7227


<div class="k-default-codeblock">
```
Epoch 19/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  14s 31ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 50403.8594

<div class="k-default-codeblock">
```

```
</div>
   4/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50411.5312 

<div class="k-default-codeblock">
```

```
</div>
   7/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50419.2070

<div class="k-default-codeblock">
```

```
</div>
  10/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50426.8867

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50434.5703

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50442.2500

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50449.9297

<div class="k-default-codeblock">
```

```
</div>
  22/469 [37m━━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50457.6094

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50465.2891

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50472.9688

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 50480.6523

<div class="k-default-codeblock">
```

```
</div>
  34/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 50488.3320

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 50496.0117

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 50503.6914

<div class="k-default-codeblock">
```

```
</div>
  43/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 19ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 50511.3711

<div class="k-default-codeblock">
```

```
</div>
  46/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 50519.0508

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 50526.7305

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 50534.4102

<div class="k-default-codeblock">
```

```
</div>
  55/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 50542.0938

<div class="k-default-codeblock">
```

```
</div>
  58/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 20ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 50549.7734

<div class="k-default-codeblock">
```

```
</div>
  61/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 19ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 50557.4531

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 19ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 50565.1328

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 50572.8125

<div class="k-default-codeblock">
```

```
</div>
  70/469 ━━[37m━━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 50580.4922

<div class="k-default-codeblock">
```

```
</div>
  73/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 50588.1719

<div class="k-default-codeblock">
```

```
</div>
  76/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 50595.8516

<div class="k-default-codeblock">
```

```
</div>
  79/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 50603.5312

<div class="k-default-codeblock">
```

```
</div>
  82/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 50611.2148

<div class="k-default-codeblock">
```

```
</div>
  85/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 50618.8945

<div class="k-default-codeblock">
```

```
</div>
  88/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50626.5742

<div class="k-default-codeblock">
```

```
</div>
  91/469 ━━━[37m━━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50634.2539

<div class="k-default-codeblock">
```

```
</div>
  94/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50641.9336

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50649.6133

<div class="k-default-codeblock">
```

```
</div>
 100/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50657.2930

<div class="k-default-codeblock">
```

```
</div>
 103/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50664.9727

<div class="k-default-codeblock">
```

```
</div>
 106/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50672.6562

<div class="k-default-codeblock">
```

```
</div>
 109/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50680.3359

<div class="k-default-codeblock">
```

```
</div>
 112/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50688.0156

<div class="k-default-codeblock">
```

```
</div>
 115/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50695.6953

<div class="k-default-codeblock">
```

```
</div>
 118/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50703.3750

<div class="k-default-codeblock">
```

```
</div>
 121/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 50711.0547

<div class="k-default-codeblock">
```

```
</div>
 124/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50718.7344

<div class="k-default-codeblock">
```

```
</div>
 127/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50726.4141

<div class="k-default-codeblock">
```

```
</div>
 130/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50734.0938

<div class="k-default-codeblock">
```

```
</div>
 133/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50741.7734

<div class="k-default-codeblock">
```

```
</div>
 136/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50749.4531

<div class="k-default-codeblock">
```

```
</div>
 139/469 ━━━━━[37m━━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50757.1328

<div class="k-default-codeblock">
```

```
</div>
 142/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50764.8125

<div class="k-default-codeblock">
```

```
</div>
 145/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50772.4922

<div class="k-default-codeblock">
```

```
</div>
 148/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 50780.1719

<div class="k-default-codeblock">
```

```
</div>
 151/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 50787.8555

<div class="k-default-codeblock">
```

```
</div>
 154/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 50795.5352

<div class="k-default-codeblock">
```

```
</div>
 157/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 50803.2148

<div class="k-default-codeblock">
```

```
</div>
 160/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 50810.8945

<div class="k-default-codeblock">
```

```
</div>
 163/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 50818.5742

<div class="k-default-codeblock">
```

```
</div>
 166/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 20ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 50826.2539

<div class="k-default-codeblock">
```

```
</div>
 169/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 50833.9336

<div class="k-default-codeblock">
```

```
</div>
 172/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 50841.6133

<div class="k-default-codeblock">
```

```
</div>
 175/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50849.2930

<div class="k-default-codeblock">
```

```
</div>
 178/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50856.9727

<div class="k-default-codeblock">
```

```
</div>
 181/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50864.6523

<div class="k-default-codeblock">
```

```
</div>
 184/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50872.3320

<div class="k-default-codeblock">
```

```
</div>
 187/469 ━━━━━━━[37m━━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50880.0117

<div class="k-default-codeblock">
```

```
</div>
 190/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50887.6914

<div class="k-default-codeblock">
```

```
</div>
 193/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50895.3750

<div class="k-default-codeblock">
```

```
</div>
 196/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50903.0547

<div class="k-default-codeblock">
```

```
</div>
 199/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 50910.7344

<div class="k-default-codeblock">
```

```
</div>
 202/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50918.4141

<div class="k-default-codeblock">
```

```
</div>
 205/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50926.0938

<div class="k-default-codeblock">
```

```
</div>
 208/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50933.7734

<div class="k-default-codeblock">
```

```
</div>
 211/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50941.4531

<div class="k-default-codeblock">
```

```
</div>
 214/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 20ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50949.1328

<div class="k-default-codeblock">
```

```
</div>
 217/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 20ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50956.8125

<div class="k-default-codeblock">
```

```
</div>
 220/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 20ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50964.4922

<div class="k-default-codeblock">
```

```
</div>
 223/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 20ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50972.1758

<div class="k-default-codeblock">
```

```
</div>
 226/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 20ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50979.8555

<div class="k-default-codeblock">
```

```
</div>
 229/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 20ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 50987.5352

<div class="k-default-codeblock">
```

```
</div>
 232/469 ━━━━━━━━━[37m━━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 50995.2148

<div class="k-default-codeblock">
```

```
</div>
 235/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51002.8945

<div class="k-default-codeblock">
```

```
</div>
 238/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51010.5742

<div class="k-default-codeblock">
```

```
</div>
 241/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51018.2539

<div class="k-default-codeblock">
```

```
</div>
 242/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51020.8125

<div class="k-default-codeblock">
```

```
</div>
 243/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51023.3750

<div class="k-default-codeblock">
```

```
</div>
 245/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51028.4922

<div class="k-default-codeblock">
```

```
</div>
 247/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51033.6133

<div class="k-default-codeblock">
```

```
</div>
 250/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51041.2930

<div class="k-default-codeblock">
```

```
</div>
 253/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51048.9727

<div class="k-default-codeblock">
```

```
</div>
 256/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51056.6523

<div class="k-default-codeblock">
```

```
</div>
 259/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51064.3320

<div class="k-default-codeblock">
```

```
</div>
 262/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51072.0117

<div class="k-default-codeblock">
```

```
</div>
 265/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51079.6953

<div class="k-default-codeblock">
```

```
</div>
 268/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 20ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 51087.3750

<div class="k-default-codeblock">
```

```
</div>
 271/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51095.0547

<div class="k-default-codeblock">
```

```
</div>
 274/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51102.7344

<div class="k-default-codeblock">
```

```
</div>
 277/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51110.4141

<div class="k-default-codeblock">
```

```
</div>
 280/469 ━━━━━━━━━━━[37m━━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51118.0938

<div class="k-default-codeblock">
```

```
</div>
 283/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51125.7734

<div class="k-default-codeblock">
```

```
</div>
 286/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51133.4531

<div class="k-default-codeblock">
```

```
</div>
 289/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51141.1328

<div class="k-default-codeblock">
```

```
</div>
 292/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51148.8125

<div class="k-default-codeblock">
```

```
</div>
 295/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51156.4922

<div class="k-default-codeblock">
```

```
</div>
 298/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51164.1719

<div class="k-default-codeblock">
```

```
</div>
 301/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51171.8555

<div class="k-default-codeblock">
```

```
</div>
 304/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51179.5352

<div class="k-default-codeblock">
```

```
</div>
 307/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51187.2148

<div class="k-default-codeblock">
```

```
</div>
 310/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51194.8945

<div class="k-default-codeblock">
```

```
</div>
 313/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51202.5742

<div class="k-default-codeblock">
```

```
</div>
 316/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51210.2539

<div class="k-default-codeblock">
```

```
</div>
 319/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51217.9336

<div class="k-default-codeblock">
```

```
</div>
 322/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51225.6133

<div class="k-default-codeblock">
```

```
</div>
 325/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51233.2930

<div class="k-default-codeblock">
```

```
</div>
 328/469 ━━━━━━━━━━━━━[37m━━━━━━━  2s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51240.9727

<div class="k-default-codeblock">
```

```
</div>
 331/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51248.6523

<div class="k-default-codeblock">
```

```
</div>
 334/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 51256.3320

<div class="k-default-codeblock">
```

```
</div>
 337/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51264.0117

<div class="k-default-codeblock">
```

```
</div>
 340/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51271.6953

<div class="k-default-codeblock">
```

```
</div>
 343/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51279.3750

<div class="k-default-codeblock">
```

```
</div>
 346/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51287.0547

<div class="k-default-codeblock">
```

```
</div>
 349/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51294.7344

<div class="k-default-codeblock">
```

```
</div>
 352/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51302.4141

<div class="k-default-codeblock">
```

```
</div>
 355/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51310.0938

<div class="k-default-codeblock">
```

```
</div>
 358/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51317.7734

<div class="k-default-codeblock">
```

```
</div>
 361/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51325.4531

<div class="k-default-codeblock">
```

```
</div>
 364/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51333.1328

<div class="k-default-codeblock">
```

```
</div>
 367/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51340.8125

<div class="k-default-codeblock">
```

```
</div>
 370/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51348.4922

<div class="k-default-codeblock">
```

```
</div>
 373/469 ━━━━━━━━━━━━━━━[37m━━━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51356.1719

<div class="k-default-codeblock">
```

```
</div>
 376/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51363.8516

<div class="k-default-codeblock">
```

```
</div>
 379/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51371.5352

<div class="k-default-codeblock">
```

```
</div>
 382/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51379.2148

<div class="k-default-codeblock">
```

```
</div>
 385/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51386.8945

<div class="k-default-codeblock">
```

```
</div>
 388/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51394.5742

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51402.2539

<div class="k-default-codeblock">
```

```
</div>
 394/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51409.9336

<div class="k-default-codeblock">
```

```
</div>
 397/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51417.6133

<div class="k-default-codeblock">
```

```
</div>
 400/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51425.2930

<div class="k-default-codeblock">
```

```
</div>
 403/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51432.9727

<div class="k-default-codeblock">
```

```
</div>
 406/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51440.6523

<div class="k-default-codeblock">
```

```
</div>
 409/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51448.3320

<div class="k-default-codeblock">
```

```
</div>
 412/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51456.0117

<div class="k-default-codeblock">
```

```
</div>
 415/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51463.6914

<div class="k-default-codeblock">
```

```
</div>
 418/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51471.3711

<div class="k-default-codeblock">
```

```
</div>
 421/469 ━━━━━━━━━━━━━━━━━[37m━━━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51479.0547

<div class="k-default-codeblock">
```

```
</div>
 424/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51486.7344

<div class="k-default-codeblock">
```

```
</div>
 427/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51494.4141

<div class="k-default-codeblock">
```

```
</div>
 430/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51502.0938

<div class="k-default-codeblock">
```

```
</div>
 433/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51509.7734

<div class="k-default-codeblock">
```

```
</div>
 436/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51517.4531

<div class="k-default-codeblock">
```

```
</div>
 439/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51525.1328

<div class="k-default-codeblock">
```

```
</div>
 442/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51532.8125

<div class="k-default-codeblock">
```

```
</div>
 445/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51540.4922

<div class="k-default-codeblock">
```

```
</div>
 448/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51548.1719

<div class="k-default-codeblock">
```

```
</div>
 451/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51555.8516

<div class="k-default-codeblock">
```

```
</div>
 454/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51563.5312

<div class="k-default-codeblock">
```

```
</div>
 457/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 20ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51571.2109

<div class="k-default-codeblock">
```

```
</div>
 459/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 21ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51576.3320

<div class="k-default-codeblock">
```

```
</div>
 460/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 21ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51578.8945

<div class="k-default-codeblock">
```

```
</div>
 463/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 21ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51586.5742

<div class="k-default-codeblock">
```

```
</div>
 466/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 21ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51594.2539

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51601.9297

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 10s 22ms/step - accuracy: 0.9943 - loss: 0.1000 - moe_loss: 51604.4766 - val_loss: 0.1000 - val_moe_loss: 53198.7227


<div class="k-default-codeblock">
```
Epoch 20/20

```
</div>
    
   1/469 [37m━━━━━━━━━━━━━━━━━━━━  18s 40ms/step - accuracy: 0.9844 - loss: 0.1000 - moe_loss: 53203.8633

<div class="k-default-codeblock">
```

```
</div>
   4/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9893 - loss: 0.1000 - moe_loss: 53211.5391 

<div class="k-default-codeblock">
```

```
</div>
   7/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9911 - loss: 0.1000 - moe_loss: 53219.2188

<div class="k-default-codeblock">
```

```
</div>
  10/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 53226.8945

<div class="k-default-codeblock">
```

```
</div>
  13/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 53234.5742

<div class="k-default-codeblock">
```

```
</div>
  16/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 53242.2578

<div class="k-default-codeblock">
```

```
</div>
  19/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9917 - loss: 0.1000 - moe_loss: 53249.9375

<div class="k-default-codeblock">
```

```
</div>
  22/469 [37m━━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9918 - loss: 0.1000 - moe_loss: 53257.6172

<div class="k-default-codeblock">
```

```
</div>
  25/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9919 - loss: 0.1000 - moe_loss: 53265.2969

<div class="k-default-codeblock">
```

```
</div>
  28/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 53272.9766

<div class="k-default-codeblock">
```

```
</div>
  31/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 53280.6562

<div class="k-default-codeblock">
```

```
</div>
  34/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 53288.3359

<div class="k-default-codeblock">
```

```
</div>
  37/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9920 - loss: 0.1000 - moe_loss: 53296.0156

<div class="k-default-codeblock">
```

```
</div>
  40/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 53303.6953

<div class="k-default-codeblock">
```

```
</div>
  43/469 ━[37m━━━━━━━━━━━━━━━━━━━  9s 21ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 53311.3750

<div class="k-default-codeblock">
```

```
</div>
  46/469 ━[37m━━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9921 - loss: 0.1000 - moe_loss: 53319.0547

<div class="k-default-codeblock">
```

```
</div>
  49/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 53326.7383

<div class="k-default-codeblock">
```

```
</div>
  52/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 53334.4180

<div class="k-default-codeblock">
```

```
</div>
  55/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9922 - loss: 0.1000 - moe_loss: 53342.0977

<div class="k-default-codeblock">
```

```
</div>
  58/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9923 - loss: 0.1000 - moe_loss: 53349.7773

<div class="k-default-codeblock">
```

```
</div>
  61/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 53357.4570

<div class="k-default-codeblock">
```

```
</div>
  64/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9924 - loss: 0.1000 - moe_loss: 53365.1367

<div class="k-default-codeblock">
```

```
</div>
  67/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9925 - loss: 0.1000 - moe_loss: 53372.8164

<div class="k-default-codeblock">
```

```
</div>
  70/469 ━━[37m━━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9926 - loss: 0.1000 - moe_loss: 53380.4961

<div class="k-default-codeblock">
```

```
</div>
  73/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9926 - loss: 0.1000 - moe_loss: 53388.1758

<div class="k-default-codeblock">
```

```
</div>
  76/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9927 - loss: 0.1000 - moe_loss: 53395.8555

<div class="k-default-codeblock">
```

```
</div>
  79/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9927 - loss: 0.1000 - moe_loss: 53403.5352

<div class="k-default-codeblock">
```

```
</div>
  82/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 53411.2148

<div class="k-default-codeblock">
```

```
</div>
  85/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9928 - loss: 0.1000 - moe_loss: 53418.8945

<div class="k-default-codeblock">
```

```
</div>
  88/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 53426.5742

<div class="k-default-codeblock">
```

```
</div>
  91/469 ━━━[37m━━━━━━━━━━━━━━━━━  8s 21ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 53434.2539

<div class="k-default-codeblock">
```

```
</div>
  94/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 22ms/step - accuracy: 0.9929 - loss: 0.1000 - moe_loss: 53441.9336

<div class="k-default-codeblock">
```

```
</div>
  97/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 22ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 53449.6133

<div class="k-default-codeblock">
```

```
</div>
 100/469 ━━━━[37m━━━━━━━━━━━━━━━━  8s 22ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 53457.2930

<div class="k-default-codeblock">
```

```
</div>
 102/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 53462.4141

<div class="k-default-codeblock">
```

```
</div>
 105/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9930 - loss: 0.1000 - moe_loss: 53470.0938

<div class="k-default-codeblock">
```

```
</div>
 108/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 53477.7734

<div class="k-default-codeblock">
```

```
</div>
 111/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 53485.4531

<div class="k-default-codeblock">
```

```
</div>
 114/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 53493.1328

<div class="k-default-codeblock">
```

```
</div>
 117/469 ━━━━[37m━━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 53500.8125

<div class="k-default-codeblock">
```

```
</div>
 120/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9931 - loss: 0.1000 - moe_loss: 53508.4922

<div class="k-default-codeblock">
```

```
</div>
 123/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 53516.1758

<div class="k-default-codeblock">
```

```
</div>
 126/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 53523.8555

<div class="k-default-codeblock">
```

```
</div>
 129/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 53531.5352

<div class="k-default-codeblock">
```

```
</div>
 132/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 53539.2148

<div class="k-default-codeblock">
```

```
</div>
 135/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9932 - loss: 0.1000 - moe_loss: 53546.8945

<div class="k-default-codeblock">
```

```
</div>
 138/469 ━━━━━[37m━━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 53554.5742

<div class="k-default-codeblock">
```

```
</div>
 141/469 ━━━━━━[37m━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 53562.2539

<div class="k-default-codeblock">
```

```
</div>
 144/469 ━━━━━━[37m━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 53569.9336

<div class="k-default-codeblock">
```

```
</div>
 147/469 ━━━━━━[37m━━━━━━━━━━━━━━  7s 22ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 53577.6133

<div class="k-default-codeblock">
```

```
</div>
 150/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9933 - loss: 0.1000 - moe_loss: 53585.2930

<div class="k-default-codeblock">
```

```
</div>
 153/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 53592.9727

<div class="k-default-codeblock">
```

```
</div>
 156/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 53600.6523

<div class="k-default-codeblock">
```

```
</div>
 159/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 53608.3320

<div class="k-default-codeblock">
```

```
</div>
 162/469 ━━━━━━[37m━━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 53616.0156

<div class="k-default-codeblock">
```

```
</div>
 165/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9934 - loss: 0.1000 - moe_loss: 53623.6953

<div class="k-default-codeblock">
```

```
</div>
 168/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 53631.3750

<div class="k-default-codeblock">
```

```
</div>
 171/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 53639.0547

<div class="k-default-codeblock">
```

```
</div>
 174/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 53646.7344

<div class="k-default-codeblock">
```

```
</div>
 177/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 53654.4141

<div class="k-default-codeblock">
```

```
</div>
 180/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 53662.0938

<div class="k-default-codeblock">
```

```
</div>
 183/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9935 - loss: 0.1000 - moe_loss: 53669.7734

<div class="k-default-codeblock">
```

```
</div>
 186/469 ━━━━━━━[37m━━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 53677.4531

<div class="k-default-codeblock">
```

```
</div>
 189/469 ━━━━━━━━[37m━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 53685.1328

<div class="k-default-codeblock">
```

```
</div>
 192/469 ━━━━━━━━[37m━━━━━━━━━━━━  6s 22ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 53692.8125

<div class="k-default-codeblock">
```

```
</div>
 195/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 53700.4922

<div class="k-default-codeblock">
```

```
</div>
 198/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 53708.1719

<div class="k-default-codeblock">
```

```
</div>
 201/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 53715.8516

<div class="k-default-codeblock">
```

```
</div>
 204/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 53723.5352

<div class="k-default-codeblock">
```

```
</div>
 207/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9936 - loss: 0.1000 - moe_loss: 53731.2148

<div class="k-default-codeblock">
```

```
</div>
 210/469 ━━━━━━━━[37m━━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 53738.8945

<div class="k-default-codeblock">
```

```
</div>
 213/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 53746.5742

<div class="k-default-codeblock">
```

```
</div>
 214/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 53749.1328

<div class="k-default-codeblock">
```

```
</div>
 216/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 53754.2539

<div class="k-default-codeblock">
```

```
</div>
 219/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 53761.9336

<div class="k-default-codeblock">
```

```
</div>
 222/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 53769.6133

<div class="k-default-codeblock">
```

```
</div>
 225/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 53777.2930

<div class="k-default-codeblock">
```

```
</div>
 228/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 53784.9727

<div class="k-default-codeblock">
```

```
</div>
 231/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9937 - loss: 0.1000 - moe_loss: 53792.6523

<div class="k-default-codeblock">
```

```
</div>
 234/469 ━━━━━━━━━[37m━━━━━━━━━━━  5s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 53800.3320

<div class="k-default-codeblock">
```

```
</div>
 237/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 53808.0117

<div class="k-default-codeblock">
```

```
</div>
 240/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 53815.6914

<div class="k-default-codeblock">
```

```
</div>
 243/469 ━━━━━━━━━━[37m━━━━━━━━━━  5s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 53823.3711

<div class="k-default-codeblock">
```

```
</div>
 246/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 53831.0547

<div class="k-default-codeblock">
```

```
</div>
 249/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 53838.7344

<div class="k-default-codeblock">
```

```
</div>
 252/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 53846.4141

<div class="k-default-codeblock">
```

```
</div>
 255/469 ━━━━━━━━━━[37m━━━━━━━━━━  4s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 53854.0938

<div class="k-default-codeblock">
```

```
</div>
 258/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 53861.7734

<div class="k-default-codeblock">
```

```
</div>
 261/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 22ms/step - accuracy: 0.9938 - loss: 0.1000 - moe_loss: 53869.4531

<div class="k-default-codeblock">
```

```
</div>
 264/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53877.1328

<div class="k-default-codeblock">
```

```
</div>
 267/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53884.8125

<div class="k-default-codeblock">
```

```
</div>
 270/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53892.4922

<div class="k-default-codeblock">
```

```
</div>
 273/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53900.1719

<div class="k-default-codeblock">
```

```
</div>
 276/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53907.8516

<div class="k-default-codeblock">
```

```
</div>
 279/469 ━━━━━━━━━━━[37m━━━━━━━━━  4s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53915.5312

<div class="k-default-codeblock">
```

```
</div>
 282/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53923.2109

<div class="k-default-codeblock">
```

```
</div>
 285/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53930.8945

<div class="k-default-codeblock">
```

```
</div>
 288/469 ━━━━━━━━━━━━[37m━━━━━━━━  4s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53938.5742

<div class="k-default-codeblock">
```

```
</div>
 291/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53946.2539

<div class="k-default-codeblock">
```

```
</div>
 294/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53953.9336

<div class="k-default-codeblock">
```

```
</div>
 297/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9939 - loss: 0.1000 - moe_loss: 53961.6133

<div class="k-default-codeblock">
```

```
</div>
 300/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 53969.2930

<div class="k-default-codeblock">
```

```
</div>
 303/469 ━━━━━━━━━━━━[37m━━━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 53976.9727

<div class="k-default-codeblock">
```

```
</div>
 306/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 53984.6523

<div class="k-default-codeblock">
```

```
</div>
 309/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 53992.3320

<div class="k-default-codeblock">
```

```
</div>
 312/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54000.0117

<div class="k-default-codeblock">
```

```
</div>
 315/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54007.6914

<div class="k-default-codeblock">
```

```
</div>
 318/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54015.3711

<div class="k-default-codeblock">
```

```
</div>
 321/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54023.0508

<div class="k-default-codeblock">
```

```
</div>
 324/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54030.7305

<div class="k-default-codeblock">
```

```
</div>
 327/469 ━━━━━━━━━━━━━[37m━━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54038.4141

<div class="k-default-codeblock">
```

```
</div>
 330/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54046.0938

<div class="k-default-codeblock">
```

```
</div>
 333/469 ━━━━━━━━━━━━━━[37m━━━━━━  3s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54053.7734

<div class="k-default-codeblock">
```

```
</div>
 336/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54061.4531

<div class="k-default-codeblock">
```

```
</div>
 339/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54069.1328

<div class="k-default-codeblock">
```

```
</div>
 342/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54076.8125

<div class="k-default-codeblock">
```

```
</div>
 344/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54081.9336

<div class="k-default-codeblock">
```

```
</div>
 346/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9940 - loss: 0.1000 - moe_loss: 54087.0508

<div class="k-default-codeblock">
```

```
</div>
 349/469 ━━━━━━━━━━━━━━[37m━━━━━━  2s 22ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54094.7305

<div class="k-default-codeblock">
```

```
</div>
 352/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54102.4141

<div class="k-default-codeblock">
```

```
</div>
 354/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54107.5312

<div class="k-default-codeblock">
```

```
</div>
 357/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54115.2109

<div class="k-default-codeblock">
```

```
</div>
 360/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54122.8906

<div class="k-default-codeblock">
```

```
</div>
 363/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54130.5703

<div class="k-default-codeblock">
```

```
</div>
 366/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 22ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54138.2539

<div class="k-default-codeblock">
```

```
</div>
 369/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54145.9336

<div class="k-default-codeblock">
```

```
</div>
 372/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54153.6133

<div class="k-default-codeblock">
```

```
</div>
 375/469 ━━━━━━━━━━━━━━━[37m━━━━━  2s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54161.2930

<div class="k-default-codeblock">
```

```
</div>
 378/469 ━━━━━━━━━━━━━━━━[37m━━━━  2s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54168.9727

<div class="k-default-codeblock">
```

```
</div>
 381/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54176.6523

<div class="k-default-codeblock">
```

```
</div>
 384/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54184.3320

<div class="k-default-codeblock">
```

```
</div>
 386/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54189.4531

<div class="k-default-codeblock">
```

```
</div>
 389/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54197.1328

<div class="k-default-codeblock">
```

```
</div>
 391/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54202.2539

<div class="k-default-codeblock">
```

```
</div>
 393/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54207.3711

<div class="k-default-codeblock">
```

```
</div>
 395/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54212.4922

<div class="k-default-codeblock">
```

```
</div>
 397/469 ━━━━━━━━━━━━━━━━[37m━━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54217.6133

<div class="k-default-codeblock">
```

```
</div>
 399/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54222.7305

<div class="k-default-codeblock">
```

```
</div>
 402/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54230.4141

<div class="k-default-codeblock">
```

```
</div>
 405/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54238.0938

<div class="k-default-codeblock">
```

```
</div>
 407/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54243.2109

<div class="k-default-codeblock">
```

```
</div>
 410/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54250.8906

<div class="k-default-codeblock">
```

```
</div>
 413/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54258.5742

<div class="k-default-codeblock">
```

```
</div>
 416/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54266.2539

<div class="k-default-codeblock">
```

```
</div>
 419/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 23ms/step - accuracy: 0.9941 - loss: 0.1000 - moe_loss: 54273.9336

<div class="k-default-codeblock">
```

```
</div>
 422/469 ━━━━━━━━━━━━━━━━━[37m━━━  1s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54281.6133

<div class="k-default-codeblock">
```

```
</div>
 425/469 ━━━━━━━━━━━━━━━━━━[37m━━  1s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54289.2930

<div class="k-default-codeblock">
```

```
</div>
 428/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54296.9727

<div class="k-default-codeblock">
```

```
</div>
 431/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54304.6523

<div class="k-default-codeblock">
```

```
</div>
 434/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54312.3320

<div class="k-default-codeblock">
```

```
</div>
 437/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54320.0117

<div class="k-default-codeblock">
```

```
</div>
 440/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54327.6914

<div class="k-default-codeblock">
```

```
</div>
 443/469 ━━━━━━━━━━━━━━━━━━[37m━━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54335.3711

<div class="k-default-codeblock">
```

```
</div>
 446/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54343.0508

<div class="k-default-codeblock">
```

```
</div>
 449/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54350.7305

<div class="k-default-codeblock">
```

```
</div>
 452/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54358.4141

<div class="k-default-codeblock">
```

```
</div>
 455/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54366.0938

<div class="k-default-codeblock">
```

```
</div>
 458/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54373.7734

<div class="k-default-codeblock">
```

```
</div>
 461/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54381.4531

<div class="k-default-codeblock">
```

```
</div>
 464/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54389.1328

<div class="k-default-codeblock">
```

```
</div>
 467/469 ━━━━━━━━━━━━━━━━━━━[37m━  0s 23ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54396.8125

<div class="k-default-codeblock">
```

```
</div>
 469/469 ━━━━━━━━━━━━━━━━━━━━ 11s 24ms/step - accuracy: 0.9942 - loss: 0.1000 - moe_loss: 54404.4766 - val_loss: 0.1000 - val_moe_loss: 55998.7227


### Evaluation


```python
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
```

<div class="k-default-codeblock">
```
Test loss: tf.Tensor(0.10000026, shape=(), dtype=float32)
Test accuracy: {'accuracy': <tf.Tensor: shape=(), dtype=float32, numpy=0.9932000041007996>}

```
</div>
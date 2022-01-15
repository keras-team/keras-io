# Graph attention network (GAT) for node classification

**Author:** [akensert](https://github.com/akensert)<br>
**Date created:** 2021/09/13<br>
**Last modified:** 2021/12/26<br>
**Description:** An implementation of a Graph Attention Network (GAT) for node classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/graph/ipynb/gat_node_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/graph/gat_node_classification.py)



---
## Introduction

[Graph neural networks](https://en.wikipedia.org/wiki/Graph_neural_network)
is the prefered neural network architecture for processing data structured as
graphs (for example, social networks or molecule structures), yielding
better results than fully-connected networks or convolutional networks.

In this tutorial, we will implement a specific graph neural network known as a
[Graph Attention Network](https://arxiv.org/abs/1710.10903) (GAT) to predict labels of
scientific papers based on what type of papers cite them (using the
[Cora](https://linqs.soe.ucsc.edu/data) dataset).

### References

For more information on GAT, see the original paper
[Graph Attention Networks](https://arxiv.org/abs/1710.10903) as well as
[DGL's Graph Attention Networks](https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html)
documentation.

### Import packages


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)
np.random.seed(2)
```

---
## Obtain the dataset

The preparation of the [Cora dataset](https://linqs.soe.ucsc.edu/data) follows that of the
[Node classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations/)
tutorial. Refer to this tutorial for more details on the dataset and exploratory data analysis.
In brief, the Cora dataset consists of two files: `cora.cites` which contains *directed links* (citations) between
papers; and `cora.content` which contains *features* of the corresponding papers and one
of seven labels (the *subject* of the paper).


```python
zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)

data_dir = os.path.join(os.path.dirname(zip_file), "cora")

citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)

papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"),
    sep="\t",
    header=None,
    names=["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"],
)

class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

print(citations)

print(papers)
```

<div class="k-default-codeblock">
```
      target  source
0          0      21
1          0     905
2          0     906
...      ...     ...
5426    1874    2586
5427    1876    1874
5428    1897    2707
```
</div>
    
<div class="k-default-codeblock">
```
[5429 rows x 2 columns]
      paper_id  term_0  term_1  ...  term_1431  term_1432  subject
0          462       0       0  ...          0          0        2
1         1911       0       0  ...          0          0        5
2         2002       0       0  ...          0          0        4
...        ...     ...     ...  ...        ...        ...      ...
2705      2372       0       0  ...          0          0        1
2706       955       0       0  ...          0          0        0
2707       376       0       0  ...          0          0        2
```
</div>
    
<div class="k-default-codeblock">
```
[2708 rows x 1435 columns]

```
</div>
### Split the dataset


```python
# Obtain random indices
random_indices = np.random.permutation(range(papers.shape[0]))

# 50/50 split
train_data = papers.iloc[random_indices[: len(random_indices) // 2]]
test_data = papers.iloc[random_indices[len(random_indices) // 2 :]]
```

### Prepare the graph data


```python
# Obtain paper indices which will be used to gather node states
# from the graph later on when training the model
train_indices = train_data["paper_id"].to_numpy()
test_indices = test_data["paper_id"].to_numpy()

# Obtain ground truth labels corresponding to each paper_id
train_labels = train_data["subject"].to_numpy()
test_labels = test_data["subject"].to_numpy()

# Define graph, namely an edge tensor and a node feature tensor
edges = tf.convert_to_tensor(citations[["target", "source"]])
node_states = tf.convert_to_tensor(papers.sort_values("paper_id").iloc[:, 1:-1])

# Print shapes of the graph
print("Edges shape:\t\t", edges.shape)
print("Node features shape:", node_states.shape)
```

<div class="k-default-codeblock">
```
Edges shape:		 (5429, 2)
Node features shape: (2708, 1433)

```
</div>
---
## Build the model

GAT takes as input a graph (namely an edge tensor and a node feature tensor) and
outputs \[updated\] node states. The node states are, for each target node, neighborhood
aggregated information of *N*-hops (where *N* is decided by the number of layers of the
GAT). Importantly, in contrast to the
[graph convolutional network](https://arxiv.org/abs/1609.02907) (GCN)
the GAT makes use of attention machanisms
to aggregate information from neighboring nodes (or *source nodes*). In other words, instead of simply
averaging/summing node states from source nodes (*source papers*) to the target node (*target papers*),
GAT first applies normalized attention scores to each source node state and then sums.

### (Multi-head) graph attention layer

The GAT model implements multi-head graph attention layers. The `MultiHeadGraphAttention`
layer is simply a concatenation (or averaging) of multiple graph attention layers
(`GraphAttention`), each with separate learnable weights `W`. The `GraphAttention` layer
does the following:

Consider inputs node states `h^{l}` which are linearly transformed by `W^{l}`, resulting in `z^{l}`.

For each target node:

1. Computes pair-wise attention scores `a^{l}^{T}(z^{l}_{i}||z^{l}_{j})` for all `j`,
resulting in `e_{ij}` (for all `j`).
`||` denotes a concatenation, `_{i}` corresponds to the target node, and `_{j}`
corresponds to a given 1-hop neighbor/source node.
2. Normalizes `e_{ij}` via softmax, so as the sum of incoming edges' attention scores
to the target node (`sum_{k}{e_{norm}_{ik}}`) will add up to 1.
3. Applies attention scores `e_{norm}_{ij}` to `z_{j}`
and adds it to the new target node state `h^{l+1}_{i}`, for all `j`.


```python

class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)

```

### Implement training logic with custom `train_step`, `test_step`, and `predict_step` methods

Notice, the GAT model operates on the entire graph (namely, `node_states` and
`edges`) in all phases (training, validation and testing). Hence, `node_states` and
`edges` are passed to the constructor of the `keras.Model` and used as attributes.
The difference between the phases are the indices (and labels), which gathers
certain outputs (`tf.gather(outputs, indices)`).


```python

class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        node_states,
        edges,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

    def train_step(self, data):
        indices, labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self([self.node_states, self.edges])
            # Compute loss
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Compute gradients
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients (update weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute probabilities
        return tf.nn.softmax(tf.gather(outputs, indices))

    def test_step(self, data):
        indices, labels = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute loss
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

```

### Train and evaluate


```python
# Define hyper-parameters
HIDDEN_UNITS = 100
NUM_HEADS = 8
NUM_LAYERS = 3
OUTPUT_DIM = len(class_values)

NUM_EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 3e-1
MOMENTUM = 0.9

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)
accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_acc", min_delta=1e-5, patience=5, restore_best_weights=True
)

# Build model
gat_model = GraphAttentionNetwork(
    node_states, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
)

# Compile model
gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

gat_model.fit(
    x=train_indices,
    y=train_labels,
    validation_split=VALIDATION_SPLIT,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping],
    verbose=2,
)

_, test_accuracy = gat_model.evaluate(x=test_indices, y=test_labels, verbose=0)

print("--" * 38 + f"\nTest Accuracy {test_accuracy*100:.1f}%")
```

<div class="k-default-codeblock">
```
Epoch 1/100
5/5 - 26s - loss: 1.8418 - acc: 0.2980 - val_loss: 1.5117 - val_acc: 0.4044 - 26s/epoch - 5s/step
Epoch 2/100
5/5 - 6s - loss: 1.2422 - acc: 0.5640 - val_loss: 1.0407 - val_acc: 0.6471 - 6s/epoch - 1s/step
Epoch 3/100
5/5 - 5s - loss: 0.7092 - acc: 0.7906 - val_loss: 0.8201 - val_acc: 0.7868 - 5s/epoch - 996ms/step
Epoch 4/100
5/5 - 5s - loss: 0.4768 - acc: 0.8604 - val_loss: 0.7451 - val_acc: 0.8088 - 5s/epoch - 934ms/step
Epoch 5/100
5/5 - 5s - loss: 0.2641 - acc: 0.9294 - val_loss: 0.7499 - val_acc: 0.8088 - 5s/epoch - 945ms/step
Epoch 6/100
5/5 - 5s - loss: 0.1487 - acc: 0.9663 - val_loss: 0.6803 - val_acc: 0.8382 - 5s/epoch - 967ms/step
Epoch 7/100
5/5 - 5s - loss: 0.0970 - acc: 0.9811 - val_loss: 0.6688 - val_acc: 0.8088 - 5s/epoch - 960ms/step
Epoch 8/100
5/5 - 5s - loss: 0.0597 - acc: 0.9934 - val_loss: 0.7295 - val_acc: 0.8162 - 5s/epoch - 981ms/step
Epoch 9/100
5/5 - 5s - loss: 0.0398 - acc: 0.9967 - val_loss: 0.7551 - val_acc: 0.8309 - 5s/epoch - 991ms/step
Epoch 10/100
5/5 - 5s - loss: 0.0312 - acc: 0.9984 - val_loss: 0.7666 - val_acc: 0.8309 - 5s/epoch - 987ms/step
Epoch 11/100
5/5 - 5s - loss: 0.0219 - acc: 0.9992 - val_loss: 0.7726 - val_acc: 0.8309 - 5s/epoch - 1s/step
----------------------------------------------------------------------------
Test Accuracy 76.5%

```
</div>
### Predict (probabilities)


```python
test_probs = gat_model.predict(x=test_indices)

mapping = {v: k for (k, v) in class_idx.items()}

for i, (probs, label) in enumerate(zip(test_probs[:10], test_labels[:10])):
    print(f"Example {i+1}: {mapping[label]}")
    for j, c in zip(probs, class_idx.keys()):
        print(f"\tProbability of {c: <24} = {j*100:7.3f}%")
    print("---" * 20)
```

<div class="k-default-codeblock">
```
Example 1: Probabilistic_Methods
	Probability of Case_Based               =   0.919%
	Probability of Genetic_Algorithms       =   0.180%
	Probability of Neural_Networks          =  37.896%
	Probability of Probabilistic_Methods    =  59.801%
	Probability of Reinforcement_Learning   =   0.705%
	Probability of Rule_Learning            =   0.044%
	Probability of Theory                   =   0.454%
------------------------------------------------------------
Example 2: Genetic_Algorithms
	Probability of Case_Based               =   0.005%
	Probability of Genetic_Algorithms       =  99.993%
	Probability of Neural_Networks          =   0.001%
	Probability of Probabilistic_Methods    =   0.000%
	Probability of Reinforcement_Learning   =   0.000%
	Probability of Rule_Learning            =   0.000%
	Probability of Theory                   =   0.000%
------------------------------------------------------------
Example 3: Theory
	Probability of Case_Based               =   8.151%
	Probability of Genetic_Algorithms       =   1.021%
	Probability of Neural_Networks          =   0.569%
	Probability of Probabilistic_Methods    =  40.220%
	Probability of Reinforcement_Learning   =   0.792%
	Probability of Rule_Learning            =   6.910%
	Probability of Theory                   =  42.337%
------------------------------------------------------------
Example 4: Neural_Networks
	Probability of Case_Based               =   0.097%
	Probability of Genetic_Algorithms       =   0.026%
	Probability of Neural_Networks          =  93.539%
	Probability of Probabilistic_Methods    =   6.206%
	Probability of Reinforcement_Learning   =   0.028%
	Probability of Rule_Learning            =   0.010%
	Probability of Theory                   =   0.094%
------------------------------------------------------------
Example 5: Theory
	Probability of Case_Based               =  25.259%
	Probability of Genetic_Algorithms       =   4.381%
	Probability of Neural_Networks          =  11.776%
	Probability of Probabilistic_Methods    =  15.053%
	Probability of Reinforcement_Learning   =   1.571%
	Probability of Rule_Learning            =  23.589%
	Probability of Theory                   =  18.370%
------------------------------------------------------------
Example 6: Genetic_Algorithms
	Probability of Case_Based               =   0.000%
	Probability of Genetic_Algorithms       = 100.000%
	Probability of Neural_Networks          =   0.000%
	Probability of Probabilistic_Methods    =   0.000%
	Probability of Reinforcement_Learning   =   0.000%
	Probability of Rule_Learning            =   0.000%
	Probability of Theory                   =   0.000%
------------------------------------------------------------
Example 7: Neural_Networks
	Probability of Case_Based               =   0.296%
	Probability of Genetic_Algorithms       =   0.291%
	Probability of Neural_Networks          =  93.419%
	Probability of Probabilistic_Methods    =   5.696%
	Probability of Reinforcement_Learning   =   0.050%
	Probability of Rule_Learning            =   0.072%
	Probability of Theory                   =   0.177%
------------------------------------------------------------
Example 8: Genetic_Algorithms
	Probability of Case_Based               =   0.000%
	Probability of Genetic_Algorithms       = 100.000%
	Probability of Neural_Networks          =   0.000%
	Probability of Probabilistic_Methods    =   0.000%
	Probability of Reinforcement_Learning   =   0.000%
	Probability of Rule_Learning            =   0.000%
	Probability of Theory                   =   0.000%
------------------------------------------------------------
Example 9: Theory
	Probability of Case_Based               =   4.103%
	Probability of Genetic_Algorithms       =   5.217%
	Probability of Neural_Networks          =  14.532%
	Probability of Probabilistic_Methods    =  66.747%
	Probability of Reinforcement_Learning   =   3.008%
	Probability of Rule_Learning            =   1.782%
	Probability of Theory                   =   4.611%
------------------------------------------------------------
Example 10: Case_Based
	Probability of Case_Based               =  99.566%
	Probability of Genetic_Algorithms       =   0.017%
	Probability of Neural_Networks          =   0.016%
	Probability of Probabilistic_Methods    =   0.155%
	Probability of Reinforcement_Learning   =   0.026%
	Probability of Rule_Learning            =   0.192%
	Probability of Theory                   =   0.028%
------------------------------------------------------------

```
</div>
---
## Conclusions

The results look OK! The GAT model seems to correctly predict the subjects of the papers,
based on what they cite, about 80% of the time. Further improvements could be
made by fine-tuning the hyper-parameters of the GAT. For instance, try changing the number of layers,
the number of hidden units, or the optimizer/learning rate; add regularization (e.g., dropout);
or modify the preprocessing step. We could also try to implement *self-loops*
(i.e., paper X cites paper X) and/or make the graph *undirected*.

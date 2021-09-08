"""
Title: Graph Attention Networks for node classification
Author: [akensert](https://github.com/akensert)
Date created: 2021/09/08
Last modified: 2021/09/08
Description: Graph Attention Networks (GAT) for node classification.
"""
"""
## Introduction

Neural networks are commonly used to learn something useful from data (e.g., predict a
future outcome based on some input data). These input data are sometimes structured as
graphs (for example, social networks and molecules), for which
[graph neural networks](https://en.wikipedia.org/wiki/Graph_neural_network)
would be preferred over other neural network architectures (such as fully-connected networks or convolutional
networks).

In this tutorial, we will implement a specific graph neural network known as a
[graph attention network](https://arxiv.org/abs/1710.10903) (GAT) to predict labels of
scientific papers based on the papers they cite (see
[Cora](https://linqs.soe.ucsc.edu/data)).


### References

For more information on GAT see the original paper
[Graph Attention Networks](https://arxiv.org/abs/1710.10903) and
[dgl's Graph Attention Networks](https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html).
"""

"""
### Install dependencies

(For easier data preparation)
```
pip -q install pandas
```

(For visualization of models)
```
pip -q install pydot
sudo apt-get -qq install graphviz
```
"""

"""
### Import packages
"""

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

"""
## Dataset

The preparation of the [Cora dataset](https://linqs.soe.ucsc.edu/data) follow that of the
[Node Classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations/)
tutorial. For more details on the dataset and exploratory data analysis see
[Node Classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations/).
In brief, the Cora dataset consist of two files: `cora.cites` which contains *directed links* (citations) between
papers; and `cora.content` which contains *features* of the corresponding papers and one
of seven labels (the *subject* of the paper).
"""

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

"""
"""

citations

"""
"""

papers

"""
### Split dataset

Notice, splits differ somewhat from that of
[Node Classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations/).
"""

# Obtain random indices
random_indices = np.random.permutation(range(papers.shape[0]))

# train data: 50%
train_data = papers.iloc[random_indices[:1354]]
# valid data: 10%
valid_data = papers.iloc[random_indices[1354:1625]]
# test data:  40%
test_data = papers.iloc[random_indices[1625:]]

"""
### Prepare graph
"""

# Obtain paper indices which will be used to gather node states
# from the graph later on when training the model
train_indices = train_data["paper_id"].to_numpy()
valid_indices = valid_data["paper_id"].to_numpy()
test_indices = test_data["paper_id"].to_numpy()

# Obtain ground truth labels corresponding to each paper_id
train_labels = train_data["subject"].to_numpy()
valid_labels = valid_data["subject"].to_numpy()
test_labels = test_data["subject"].to_numpy()

# Define graph, namely an adjacency tensor and a node feature tensor
adjacency = tf.convert_to_tensor(citations[["source", "target"]])
node_features = tf.convert_to_tensor(papers.sort_values("paper_id").iloc[:, 1:-1])

# Define global constants, which will be used for building the model
NODE_DIM = node_features.shape[1]
OUTPUT_DIM = len(class_values)

# Print shapes of the graph
print("Adjacency shape:", adjacency.shape)
print("Nodes shape:\t", node_features.shape)

"""
## Model

GAT takes as input a graph (namely an adjacecy tensor and a node feature tensor) and
outputs \[updated\] node states. The node states are, for each source node, neighborhood
aggregated information of *N*-hops (where *N* is decided by the number of layers of the
GAT). Importantly, in contrast to the [graph convolutional
network](https://arxiv.org/abs/1609.02907) (GCN) the GAT make use of attention machanisms
to aggregate information from neighboring nodes. In other words, instead of simply
averaging/summing node states from neighbors to the source node, GAT first applies
normalized attention scores to each neighbor node state and then sums.
"""

"""
### Hyper-parameters
"""

# GAT
HIDDEN_UNITS = 100
NUM_HEADS = 8
NUM_LAYERS = 3

NUM_ITERATIONS = 141

# SGD
LEARNING_RATE = 3e-1
MOMENTUM = 0.9

"""
### (Multi-head) graph attention layer

The GAT model implements multi-head graph attention layers. The `MultiHeadGraphAttention`
layer is simply a concatenation (or averaging) of multiple graph attention layers
(`GraphAttention`), each with separate learnable weights `W`. The `GraphAttention` layer
does the following:

1. Inputs node states `h^{l}` which are linearly transformed by `W^{l}`, resulting in
`z^{l}`.

*(for each source node)*

2. Computes pair-wise attention scores `a^{l}^{T}(z^{l}_{i}||z^{l}_{j})` for all `j`,
resulting in `e_{ij}` (for all `j`). `||` denotes a concatenation, `_{i}` corresponds to
the source node, and `_{j}` corresponds to a given 1-hop neighbor node.
3. Normalizes `e_{ij}` via softmax, so as the sum of incoming edges' attention scores
(`sum_{k}{e_{norm}_{ik}}`) will add up to 1. (Notice, in this tutorial, *incoming edges*
are defined as edges pointing from the *source paper* (the paper *citing*) to the *target
paper* (the paper *cited*), which is counter inuitive. However, this seems to work better
in practice. In other words, we want to learn the label of the source paper based on what
it cites (target papers).)
4. Applies attention scores `e_{norm}_{ij}` to `z_{j}` and adds it to the new source node
state `h^{l+1}_{i}`, for all `j`.

"""


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
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
        )
        self.built = True

    def call(self, inputs):
        node_features, edges = inputs

        # (1) Linearly transform node features (node states)
        node_features_transformed = tf.matmul(node_features, self.kernel)

        # (2) Compute pair-wise attention scores
        node_features_expanded = tf.gather(node_features_transformed, edges)
        node_features_expanded = tf.reshape(
            node_features_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_features_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (3) Normalize attention scores
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

        # (4) Gather node states of neighbors, apply attention scores and aggregate
        node_features_neighbors = tf.gather(node_features_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_features_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_features)[0],
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


def GraphAttentionNetwork(hidden_units, num_heads, num_layers):

    node_features = layers.Input((NODE_DIM), dtype="float32")
    adjacency = layers.Input((2), dtype="int32")

    x = layers.Dense(hidden_units * num_heads, activation="relu")(node_features)

    for _ in range(num_layers):
        x_updated = MultiHeadGraphAttention(hidden_units, num_heads)([x, adjacency])
        x = layers.Add()([x_updated, x])

    out = layers.Dense(OUTPUT_DIM)(x)

    model = keras.Model(inputs=[node_features, adjacency], outputs=[out])

    return model


# Build graph attention network model
gat_model = GraphAttentionNetwork(HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS)

# Visualize built model
keras.utils.plot_model(gat_model, show_shapes=True, show_layer_names=False)

"""
### Define custom `train_step`
"""


def get_train_step_fn(model, loss_fn, optimizer):
    @tf.function
    def train_step(inputs, labels, indices):
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = loss_fn(labels, tf.gather(outputs, indices))
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss, outputs

    return train_step


"""
### Train and predict
"""

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)
accuracy_fn = lambda y_true, y_pred: tf.reduce_mean(
    keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
)

train_step = get_train_step_fn(gat_model, loss_fn, optimizer)

best_accuracy = 0.0
for i in range(NUM_ITERATIONS):

    loss, preds = train_step(
        inputs=(node_features, adjacency), labels=train_labels, indices=train_indices
    )

    # Obtain valid accuracy, and if best so far, obtain test accuracy
    # and test probabilities (for later use)
    valid_preds = tf.gather(preds, valid_indices)
    valid_accuracy = accuracy_fn(valid_labels, valid_preds)
    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        test_preds = tf.gather(preds, test_indices)
        test_accuracy = accuracy_fn(test_labels, test_preds)
        test_probs = tf.nn.softmax(test_preds).numpy()

    if i % 10 == 0:
        print(
            f"Iteration {i:03d} : "
            + f"Loss {loss:.3f} : "
            + f"Valid Accuracy {valid_accuracy*100:.1f}%"
        )


print("--" * 26)
print(f"\t\t\t  Test Accuracy {test_accuracy*100:.1f}%")

"""
### Inspect predictions
"""

mapping = {v: k for (k, v) in class_idx.items()}

for i, (probs, label) in enumerate(zip(test_probs[:10], test_labels[:10])):
    print(f"Example {i+1}: {mapping[label]}")
    for j, c in zip(probs, class_idx.keys()):
        print(f"\tProbability of {c: <24} = {j*100:7.3f}%")
    print("---" * 20)

"""
**Not too bad!**
"""

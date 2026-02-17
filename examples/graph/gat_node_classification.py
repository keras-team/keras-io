"""
Title: Graph attention network (GAT) for node classification
Author: [akensert](https://github.com/akensert)
Date created: 2021/09/13
Last modified: 2021/12/26
Description: An implementation of a Graph Attention Network (GAT) for node classification.
Accelerator: GPU
Converted to Keras 3 by: [LakshmiKalaKadali](https://github.com/LakshmiKalaKadali)
"""

"""
## Introduction

[Graph neural networks](https://en.wikipedia.org/wiki/Graph_neural_network)
is the preferred neural network architecture for processing data structured as
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
"""

"""
### Import packages
"""

import os


os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
from keras import ops
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)

keras.utils.set_random_seed(2)

"""
## Obtain the dataset

The preparation of the [Cora dataset](https://linqs.soe.ucsc.edu/data) follows that of the
[Node classification with Graph Neural Networks](https://keras.io/examples/graph/gnn_citations/)
tutorial. Refer to this tutorial for more details on the dataset and exploratory data analysis.
In brief, the Cora dataset consists of two files: `cora.cites` which contains *directed links* (citations) between
papers; and `cora.content` which contains *features* of the corresponding papers and one
of seven labels (the *subject* of the paper).
"""


zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)
data_dir = os.path.join(zip_file, "cora")

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

"""
### Split the dataset
"""

# Obtain random indices
random_indices = np.random.permutation(range(papers.shape[0]))

# 50/50 split
train_data = papers.iloc[random_indices[: len(random_indices) // 2]]
test_data = papers.iloc[random_indices[len(random_indices) // 2 :]]

"""
### Prepare the graph data
"""

# Obtain paper indices which will be used to gather node states
# from the graph later on when training the model
train_indices = train_data["paper_id"].to_numpy()
test_indices = test_data["paper_id"].to_numpy()

# Obtain ground truth labels corresponding to each paper_id
train_labels = train_data["subject"].to_numpy()
test_labels = test_data["subject"].to_numpy()

# Define graph, namely an edge tensor and a node feature tensor
edges = ops.convert_to_tensor(citations[["target", "source"]].to_numpy(), dtype="int32")
node_states = ops.convert_to_tensor(
    papers.sort_values("paper_id").iloc[:, 1:-1].to_numpy(), dtype="float32"
)

print("Edges shape:\t\t", edges.shape)
print("Node features shape:", node_states.shape)

"""
## Build the model

GAT takes as input a graph (namely an edge tensor and a node feature tensor) and
outputs \[updated\] node states. The node states are, for each target node, neighborhood
aggregated information of *N*-hops (where *N* is decided by the number of layers of the
GAT). Importantly, in contrast to the
[graph convolutional network](https://arxiv.org/abs/1609.02907) (GCN)
the GAT makes use of attention mechanisms
to aggregate information from neighboring nodes (or *source nodes*). In other words, instead of simply
averaging/summing node states from source nodes (*source papers*) to the target node (*target papers*),
GAT first applies normalized attention scores to each source node state and then sums.
"""

"""
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

        z = ops.matmul(node_states, self.kernel)

        source_indices = edges[:, 1]
        target_indices = edges[:, 0]

        z_target = ops.take(z, target_indices, axis=0)
        z_source = ops.take(z, source_indices, axis=0)

        z_concat = ops.concatenate([z_target, z_source], axis=-1)
        attention_scores = ops.leaky_relu(ops.matmul(z_concat, self.kernel_attention))
        attention_scores = ops.squeeze(attention_scores, -1)

        attention_scores = ops.exp(ops.clip(attention_scores, -2, 2))

        num_nodes = ops.shape(node_states)[0]
        attention_sum = ops.segment_sum(
            attention_scores, target_indices, num_segments=num_nodes
        )

        # Broadcast sum back to edges to normalize
        attention_sum_per_edge = ops.take(attention_sum, target_indices, axis=0)
        attention_norm = attention_scores / (attention_sum_per_edge + 1e-8)

        node_states_neighbors = ops.take(z, source_indices, axis=0)
        weighted_neighbors = node_states_neighbors * ops.expand_dims(
            attention_norm, axis=-1
        )

        return ops.segment_sum(
            weighted_neighbors, target_indices, num_segments=num_nodes
        )


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        node_states, edges = inputs
        outputs = [layer([node_states, edges]) for layer in self.attention_layers]

        if self.merge_type == "concat":
            outputs = ops.concatenate(outputs, axis=-1)
        else:
            outputs = ops.mean(ops.stack(outputs, axis=0), axis=0)

        return ops.relu(outputs)


"""
### Implement the Graph Attention Network

The GAT model operates on the entire graph (both node_states and edges) during all phases.
To maintain backend agnosticism and leverage Keras 3's built-in training optimizations, 
we store the graph data as internal tensors and design the call method to accept 
the target node indices as its primary input.
"""


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

    def call(self, inputs, training=False):
        # inputs here are the indices of nodes we want predictions for
        indices = inputs

        x = self.preprocess(self.node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, self.edges]) + x

        # Return only the requested node states
        outputs = self.output_layer(x)
        return ops.take(outputs, indices, axis=0)


"""
### Train and evaluate
"""

HIDDEN_UNITS = 100
NUM_HEADS = 8
NUM_LAYERS = 3
OUTPUT_DIM = len(class_values)

# Build and compile model
gat_model = GraphAttentionNetwork(
    node_states, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
)

gat_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.SGD(learning_rate=0.003, momentum=0.9),
    metrics=["accuracy"],
)

gat_model.fit(
    x=train_indices,
    y=train_labels,
    validation_split=0.1,
    batch_size=256,
    epochs=100,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        )
    ],
    verbose=2,
)
_, test_accuracy = gat_model.evaluate(x=test_indices, y=test_labels, verbose=0)

print("--" * 38 + f"\nTest Accuracy {test_accuracy*100:.1f}%")


"""
### Predict (probabilities)
"""
test_logits = gat_model.predict(x=test_indices)

test_probs = ops.softmax(test_logits)

test_probs_np = ops.convert_to_numpy(test_probs)

mapping = {v: k for (k, v) in class_idx.items()}

for i, (probs, label) in enumerate(zip(test_probs_np[:10], test_labels[:10])):
    print(f"Example {i+1}: {mapping[label]}")
    for j, c in zip(probs, class_idx.keys()):
        print(f"\tProbability of {c: <24} = {j*100:7.3f}%")
    print("---" * 20)

"""
## Conclusions

The results look OK! The GAT model seems to correctly predict the subjects of the papers,
based on what they cite, about 80% of the time. Further improvements could be
made by fine-tuning the hyper-parameters of the GAT. For instance, try changing the number of layers,
the number of hidden units, or the optimizer/learning rate; add regularization (e.g., dropout);
or modify the preprocessing step. We could also try to implement *self-loops*
(i.e., paper X cites paper X) and/or make the graph *undirected*.

## Relevant Chapters from Deep Learning with Python
- [Chapter 7: A deep dive on Keras](https://deeplearningwithpython.io/chapters/chapter07_deep-dive-keras)
- [Chapter 15: Language models and the Transformer](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer)
"""

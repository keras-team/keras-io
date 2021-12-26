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
scientific papers based on the papers they cite (using the
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
edges = tf.convert_to_tensor(citations[["source", "target"]])
node_features = tf.convert_to_tensor(papers.sort_values("paper_id").iloc[:, 1:-1])

# Print shapes of the graph
print("Edges shape:\t\t", edges.shape)
print("Node features shape:", node_features.shape)
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
outputs \[updated\] node states. The node states are, for each source node, neighborhood
aggregated information of *N*-hops (where *N* is decided by the number of layers of the
GAT). Importantly, in contrast to the
[graph convolutional network](https://arxiv.org/abs/1609.02907) (GCN)
the GAT make use of attention machanisms
to aggregate information from neighboring nodes. In other words, instead of simply
averaging/summing node states from neighbors to the source node, GAT first applies
normalized attention scores to each neighbor node state and then sums.

### (Multi-head) graph attention layer

The GAT model implements multi-head graph attention layers. The `MultiHeadGraphAttention`
layer is simply a concatenation (or averaging) of multiple graph attention layers
(`GraphAttention`), each with separate learnable weights `W`. The `GraphAttention` layer
does the following:

Consider inputs node states `h^{l}` which are linearly transformed by `W^{l}`, resulting in `z^{l}`.

For each source node:

1. Computes pair-wise attention scores `a^{l}^{T}(z^{l}_{i}||z^{l}_{j})` for all `j`,
resulting in `e_{ij}` (for all `j`).
`||` denotes a concatenation, `_{i}` corresponds to the source node, and `_{j}`
corresponds to a given 1-hop neighbor node.
2. Normalizes `e_{ij}` via softmax, so as the sum of incoming edges' attention scores
to the target node (`sum_{k}{e_{norm}_{ik}}`) will add up to 1.
3. Applies attention scores `e_{norm}_{ij}` to `z_{j}`
and adds it to the new source node state `h^{l+1}_{i}`, for all `j`.

In other words, we want to learn the label of the target paper based on what cites it (source papers).


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

        # Linearly transform node features (node states)
        node_features_transformed = tf.matmul(node_features, self.kernel)

        # (1) Compute pair-wise attention scores
        node_features_expanded = tf.gather(node_features_transformed, edges)
        node_features_expanded = tf.reshape(
            node_features_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_features_expanded, self.kernel_attention)
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

```

### Implement training logic with custom `train_step`, `test_step`, and `predict_step` methods

Notice, the GAT model operates on the entire graph (namely, `node_features` and
`edges`) in all phases (training, validation and testing). Hence, `node_features` and
`edges` are passed to the constructor of the `keras.Model` and used as attributes.
The difference between the phases are the indices (and labels), which gathers
certain output units (`tf.gather(outputs, indices)`).


```python

class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        node_features,
        edges,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_features = node_features
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_features, edges = inputs
        x = self.preprocess(node_features)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

    def train_step(self, data):
        indices, labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self([self.node_features, self.edges])
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
        outputs = self([self.node_features, self.edges])
        # Compute probabilities
        return tf.nn.softmax(tf.gather(outputs, indices))

    def test_step(self, data):
        indices, labels = data
        # Forward pass
        outputs = self([self.node_features, self.edges])
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
    node_features, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
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
5/5 - 23s - loss: 1.8716 - acc: 0.3128 - val_loss: 1.6772 - val_acc: 0.5074 - 23s/epoch - 5s/step
Epoch 2/100
5/5 - 4s - loss: 1.2072 - acc: 0.5911 - val_loss: 0.9890 - val_acc: 0.6471 - 4s/epoch - 798ms/step
Epoch 3/100
5/5 - 4s - loss: 0.7453 - acc: 0.7841 - val_loss: 0.7037 - val_acc: 0.8015 - 4s/epoch - 835ms/step
Epoch 4/100
5/5 - 4s - loss: 0.4258 - acc: 0.8900 - val_loss: 0.5779 - val_acc: 0.8456 - 4s/epoch - 855ms/step
Epoch 5/100
5/5 - 4s - loss: 0.2543 - acc: 0.9327 - val_loss: 0.4693 - val_acc: 0.8382 - 4s/epoch - 864ms/step
Epoch 6/100
5/5 - 4s - loss: 0.1695 - acc: 0.9516 - val_loss: 0.5757 - val_acc: 0.8456 - 4s/epoch - 868ms/step
Epoch 7/100
5/5 - 5s - loss: 0.0845 - acc: 0.9803 - val_loss: 0.5505 - val_acc: 0.8456 - 5s/epoch - 960ms/step
Epoch 8/100
5/5 - 9s - loss: 0.0714 - acc: 0.9836 - val_loss: 0.5345 - val_acc: 0.8750 - 9s/epoch - 2s/step
Epoch 9/100
5/5 - 9s - loss: 0.0544 - acc: 0.9885 - val_loss: 0.6439 - val_acc: 0.8750 - 9s/epoch - 2s/step
Epoch 10/100
5/5 - 10s - loss: 0.0379 - acc: 0.9918 - val_loss: 0.5563 - val_acc: 0.8603 - 10s/epoch - 2s/step
Epoch 11/100
5/5 - 10s - loss: 0.0192 - acc: 0.9975 - val_loss: 0.5827 - val_acc: 0.8603 - 10s/epoch - 2s/step
Epoch 12/100
5/5 - 8s - loss: 0.0187 - acc: 0.9967 - val_loss: 0.5792 - val_acc: 0.8603 - 8s/epoch - 2s/step
Epoch 13/100
5/5 - 7s - loss: 0.0131 - acc: 0.9975 - val_loss: 0.5494 - val_acc: 0.8529 - 7s/epoch - 1s/step
----------------------------------------------------------------------------
Test Accuracy 83.2%

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
	Probability of Case_Based               =   0.000%
	Probability of Genetic_Algorithms       =   0.000%
	Probability of Neural_Networks          =   0.029%
	Probability of Probabilistic_Methods    =  99.970%
	Probability of Reinforcement_Learning   =   0.000%
	Probability of Rule_Learning            =   0.000%
	Probability of Theory                   =   0.001%
------------------------------------------------------------
Example 2: Genetic_Algorithms
	Probability of Case_Based               =   0.000%
	Probability of Genetic_Algorithms       = 100.000%
	Probability of Neural_Networks          =   0.000%
	Probability of Probabilistic_Methods    =   0.000%
	Probability of Reinforcement_Learning   =   0.000%
	Probability of Rule_Learning            =   0.000%
	Probability of Theory                   =   0.000%
------------------------------------------------------------
Example 3: Theory
	Probability of Case_Based               =   2.270%
	Probability of Genetic_Algorithms       =   0.243%
	Probability of Neural_Networks          =   0.222%
	Probability of Probabilistic_Methods    =  12.699%
	Probability of Reinforcement_Learning   =   0.374%
	Probability of Rule_Learning            =  10.696%
	Probability of Theory                   =  73.496%
------------------------------------------------------------
Example 4: Neural_Networks
	Probability of Case_Based               =   0.001%
	Probability of Genetic_Algorithms       =   0.001%
	Probability of Neural_Networks          =  99.996%
	Probability of Probabilistic_Methods    =   0.000%
	Probability of Reinforcement_Learning   =   0.001%
	Probability of Rule_Learning            =   0.000%
	Probability of Theory                   =   0.001%
------------------------------------------------------------
Example 5: Theory
	Probability of Case_Based               =  71.277%
	Probability of Genetic_Algorithms       =   0.039%
	Probability of Neural_Networks          =   3.736%
	Probability of Probabilistic_Methods    =   1.131%
	Probability of Reinforcement_Learning   =   0.042%
	Probability of Rule_Learning            =   4.115%
	Probability of Theory                   =  19.661%
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
	Probability of Case_Based               =   0.925%
	Probability of Genetic_Algorithms       =   1.133%
	Probability of Neural_Networks          =  96.297%
	Probability of Probabilistic_Methods    =   0.499%
	Probability of Reinforcement_Learning   =   0.612%
	Probability of Rule_Learning            =   0.107%
	Probability of Theory                   =   0.427%
------------------------------------------------------------
Example 8: Genetic_Algorithms
	Probability of Case_Based               =   0.000%
	Probability of Genetic_Algorithms       =  99.999%
	Probability of Neural_Networks          =   0.000%
	Probability of Probabilistic_Methods    =   0.000%
	Probability of Reinforcement_Learning   =   0.001%
	Probability of Rule_Learning            =   0.000%
	Probability of Theory                   =   0.000%
------------------------------------------------------------
Example 9: Theory
	Probability of Case_Based               =   0.037%
	Probability of Genetic_Algorithms       =   0.011%
	Probability of Neural_Networks          =   0.035%
	Probability of Probabilistic_Methods    =  92.343%
	Probability of Reinforcement_Learning   =   0.020%
	Probability of Rule_Learning            =   0.054%
	Probability of Theory                   =   7.501%
------------------------------------------------------------
Example 10: Case_Based
	Probability of Case_Based               = 100.000%
	Probability of Genetic_Algorithms       =   0.000%
	Probability of Neural_Networks          =   0.000%
	Probability of Probabilistic_Methods    =   0.000%
	Probability of Reinforcement_Learning   =   0.000%
	Probability of Rule_Learning            =   0.000%
	Probability of Theory                   =   0.000%
------------------------------------------------------------

```
</div>
---
## Conclusions

The results look OK! The GAT model seems to correctly predict the subjects of the papers,
based on what they cite, about 80-85% of the time. Further improvements could be
made by fine-tuning the hyper-parameters of the GAT. For instance, try changing the number of layers,
the number of hidden units, or the optimizer/learning rate; add regularization (e.g., dropout);
or modify the preprocessing step. We could also try to implement *self-loops*
(i.e., paper X cites paper X) and/or make the graph *undirected*.

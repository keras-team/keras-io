# Graph attention network (GAT) for node classification

**Author:** [akensert](https://github.com/akensert)<br>
**Date created:** 2021/09/13<br>
**Last modified:** 2026/02/17<br>
**Description:** An implementation of a Graph Attention Network (GAT) for node classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/graph/ipynb/gat_node_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/graph/gat_node_classification.py)



---
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

### Import packages


```python
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

[5429 rows x 2 columns]
      paper_id  term_0  term_1  ...  term_1431  term_1432  subject
0          462       0       0  ...          0          0        2
1         1911       0       0  ...          0          0        5
2         2002       0       0  ...          0          0        4
...        ...     ...     ...  ...        ...        ...      ...
2705      2372       0       0  ...          0          0        1
2706       955       0       0  ...          0          0        0
2707       376       0       0  ...          0          0        2

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
edges = ops.convert_to_tensor(citations[["target", "source"]].to_numpy(), dtype="int32")
node_states = ops.convert_to_tensor(
    papers.sort_values("paper_id").iloc[:, 1:-1].to_numpy(), dtype="float32"
)

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
the GAT makes use of attention mechanisms
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
        attention_norm = attention_scores / ops.maximum(attention_sum_per_edge, 1e-8)

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

```

### Implement the Graph Attention Network

The GAT model operates on the entire graph (both node_states and edges) during all phases.
To maintain backend agnosticism and leverage Keras 3's built-in training optimizations,
we store the graph data as internal tensors and design the call method to accept
the target node indices as its primary input.


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

    def call(self, inputs, training=False):
        # inputs here are the indices of nodes we want predictions for
        indices = inputs

        x = self.preprocess(self.node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, self.edges]) + x

        # Return only the requested node states
        outputs = self.output_layer(x)
        return ops.take(outputs, indices, axis=0)

```

### Train and evaluate


```python
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

```

<div class="k-default-codeblock">
```
Epoch 1/100

5/5 - 6s - 1s/step - accuracy: 0.1429 - loss: 1.9723 - val_accuracy: 0.1324 - val_loss: 1.9576

Epoch 2/100

5/5 - 1s - 151ms/step - accuracy: 0.1814 - loss: 1.9191 - val_accuracy: 0.2721 - val_loss: 1.9039

Epoch 3/100

5/5 - 1s - 149ms/step - accuracy: 0.2553 - loss: 1.8739 - val_accuracy: 0.3088 - val_loss: 1.8803

Epoch 4/100

5/5 - 1s - 145ms/step - accuracy: 0.2800 - loss: 1.8530 - val_accuracy: 0.2868 - val_loss: 1.8698

Epoch 5/100

5/5 - 1s - 140ms/step - accuracy: 0.2857 - loss: 1.8346 - val_accuracy: 0.3088 - val_loss: 1.8545

Epoch 6/100

5/5 - 1s - 145ms/step - accuracy: 0.2956 - loss: 1.8116 - val_accuracy: 0.3162 - val_loss: 1.8375

Epoch 7/100

5/5 - 1s - 148ms/step - accuracy: 0.3136 - loss: 1.7893 - val_accuracy: 0.3162 - val_loss: 1.8211

Epoch 8/100

5/5 - 1s - 151ms/step - accuracy: 0.3415 - loss: 1.7683 - val_accuracy: 0.3235 - val_loss: 1.8041

Epoch 9/100

5/5 - 1s - 161ms/step - accuracy: 0.3539 - loss: 1.7471 - val_accuracy: 0.3309 - val_loss: 1.7867

Epoch 10/100

5/5 - 1s - 144ms/step - accuracy: 0.3539 - loss: 1.7258 - val_accuracy: 0.3309 - val_loss: 1.7701

Epoch 11/100

5/5 - 1s - 141ms/step - accuracy: 0.3547 - loss: 1.7049 - val_accuracy: 0.3456 - val_loss: 1.7545

Epoch 12/100

5/5 - 1s - 140ms/step - accuracy: 0.3612 - loss: 1.6843 - val_accuracy: 0.3456 - val_loss: 1.7393

Epoch 13/100

5/5 - 1s - 140ms/step - accuracy: 0.3768 - loss: 1.6639 - val_accuracy: 0.3676 - val_loss: 1.7240

Epoch 14/100

5/5 - 1s - 139ms/step - accuracy: 0.3957 - loss: 1.6436 - val_accuracy: 0.3824 - val_loss: 1.7084

Epoch 15/100

5/5 - 1s - 138ms/step - accuracy: 0.4171 - loss: 1.6234 - val_accuracy: 0.3897 - val_loss: 1.6923

Epoch 16/100

5/5 - 1s - 142ms/step - accuracy: 0.4368 - loss: 1.6032 - val_accuracy: 0.4118 - val_loss: 1.6759

Epoch 17/100

5/5 - 1s - 139ms/step - accuracy: 0.4475 - loss: 1.5829 - val_accuracy: 0.4191 - val_loss: 1.6593

Epoch 18/100

5/5 - 1s - 139ms/step - accuracy: 0.4598 - loss: 1.5626 - val_accuracy: 0.4265 - val_loss: 1.6427

Epoch 19/100

5/5 - 1s - 136ms/step - accuracy: 0.4778 - loss: 1.5422 - val_accuracy: 0.4338 - val_loss: 1.6263

Epoch 20/100

5/5 - 1s - 152ms/step - accuracy: 0.4885 - loss: 1.5217 - val_accuracy: 0.4338 - val_loss: 1.6098

Epoch 21/100

5/5 - 1s - 153ms/step - accuracy: 0.5082 - loss: 1.5012 - val_accuracy: 0.4412 - val_loss: 1.5933

Epoch 22/100

5/5 - 1s - 160ms/step - accuracy: 0.5213 - loss: 1.4807 - val_accuracy: 0.4485 - val_loss: 1.5767

Epoch 23/100

5/5 - 1s - 153ms/step - accuracy: 0.5279 - loss: 1.4601 - val_accuracy: 0.4632 - val_loss: 1.5599

Epoch 24/100

5/5 - 1s - 149ms/step - accuracy: 0.5411 - loss: 1.4395 - val_accuracy: 0.4779 - val_loss: 1.5430

Epoch 25/100

5/5 - 1s - 153ms/step - accuracy: 0.5534 - loss: 1.4189 - val_accuracy: 0.4779 - val_loss: 1.5260

Epoch 26/100

5/5 - 1s - 153ms/step - accuracy: 0.5608 - loss: 1.3983 - val_accuracy: 0.4779 - val_loss: 1.5088

Epoch 27/100

5/5 - 1s - 143ms/step - accuracy: 0.5772 - loss: 1.3777 - val_accuracy: 0.5000 - val_loss: 1.4917

Epoch 28/100

5/5 - 1s - 200ms/step - accuracy: 0.5903 - loss: 1.3572 - val_accuracy: 0.5147 - val_loss: 1.4746

Epoch 29/100

5/5 - 1s - 149ms/step - accuracy: 0.6002 - loss: 1.3367 - val_accuracy: 0.5221 - val_loss: 1.4575

Epoch 30/100

5/5 - 1s - 149ms/step - accuracy: 0.6092 - loss: 1.3163 - val_accuracy: 0.5368 - val_loss: 1.4405

Epoch 31/100

5/5 - 1s - 148ms/step - accuracy: 0.6190 - loss: 1.2960 - val_accuracy: 0.5588 - val_loss: 1.4235

Epoch 32/100

5/5 - 1s - 142ms/step - accuracy: 0.6330 - loss: 1.2759 - val_accuracy: 0.5588 - val_loss: 1.4065

Epoch 33/100

5/5 - 1s - 151ms/step - accuracy: 0.6445 - loss: 1.2560 - val_accuracy: 0.5735 - val_loss: 1.3898

Epoch 34/100

5/5 - 1s - 149ms/step - accuracy: 0.6502 - loss: 1.2362 - val_accuracy: 0.5735 - val_loss: 1.3732

Epoch 35/100

5/5 - 1s - 157ms/step - accuracy: 0.6593 - loss: 1.2167 - val_accuracy: 0.5882 - val_loss: 1.3568

Epoch 36/100

5/5 - 1s - 149ms/step - accuracy: 0.6667 - loss: 1.1975 - val_accuracy: 0.5882 - val_loss: 1.3405

Epoch 37/100

5/5 - 1s - 154ms/step - accuracy: 0.6749 - loss: 1.1785 - val_accuracy: 0.5956 - val_loss: 1.3245

Epoch 38/100

5/5 - 1s - 150ms/step - accuracy: 0.6814 - loss: 1.1598 - val_accuracy: 0.5882 - val_loss: 1.3087

Epoch 39/100

5/5 - 1s - 141ms/step - accuracy: 0.6897 - loss: 1.1414 - val_accuracy: 0.5956 - val_loss: 1.2932

Epoch 40/100

5/5 - 1s - 146ms/step - accuracy: 0.6979 - loss: 1.1233 - val_accuracy: 0.6029 - val_loss: 1.2779

Epoch 41/100

5/5 - 1s - 150ms/step - accuracy: 0.7053 - loss: 1.1056 - val_accuracy: 0.6103 - val_loss: 1.2628

Epoch 42/100

5/5 - 1s - 147ms/step - accuracy: 0.7110 - loss: 1.0882 - val_accuracy: 0.6103 - val_loss: 1.2481

Epoch 43/100

5/5 - 1s - 151ms/step - accuracy: 0.7184 - loss: 1.0712 - val_accuracy: 0.6103 - val_loss: 1.2336

Epoch 44/100

5/5 - 1s - 149ms/step - accuracy: 0.7241 - loss: 1.0545 - val_accuracy: 0.6250 - val_loss: 1.2194

Epoch 45/100

5/5 - 1s - 147ms/step - accuracy: 0.7274 - loss: 1.0382 - val_accuracy: 0.6250 - val_loss: 1.2055

Epoch 46/100

5/5 - 1s - 144ms/step - accuracy: 0.7323 - loss: 1.0223 - val_accuracy: 0.6397 - val_loss: 1.1919

Epoch 47/100

5/5 - 1s - 141ms/step - accuracy: 0.7406 - loss: 1.0067 - val_accuracy: 0.6397 - val_loss: 1.1786

Epoch 48/100

5/5 - 1s - 143ms/step - accuracy: 0.7463 - loss: 0.9915 - val_accuracy: 0.6471 - val_loss: 1.1656

Epoch 49/100

5/5 - 1s - 154ms/step - accuracy: 0.7504 - loss: 0.9766 - val_accuracy: 0.6544 - val_loss: 1.1530

Epoch 50/100

5/5 - 1s - 152ms/step - accuracy: 0.7553 - loss: 0.9621 - val_accuracy: 0.6618 - val_loss: 1.1406

Epoch 51/100

5/5 - 1s - 147ms/step - accuracy: 0.7562 - loss: 0.9480 - val_accuracy: 0.6544 - val_loss: 1.1286

Epoch 52/100

5/5 - 1s - 148ms/step - accuracy: 0.7594 - loss: 0.9342 - val_accuracy: 0.6618 - val_loss: 1.1169

Epoch 53/100

5/5 - 1s - 144ms/step - accuracy: 0.7652 - loss: 0.9208 - val_accuracy: 0.6691 - val_loss: 1.1054

Epoch 54/100

5/5 - 1s - 151ms/step - accuracy: 0.7685 - loss: 0.9077 - val_accuracy: 0.6691 - val_loss: 1.0943

Epoch 55/100

5/5 - 1s - 148ms/step - accuracy: 0.7734 - loss: 0.8949 - val_accuracy: 0.6691 - val_loss: 1.0834

Epoch 56/100

5/5 - 1s - 149ms/step - accuracy: 0.7775 - loss: 0.8825 - val_accuracy: 0.6765 - val_loss: 1.0728

Epoch 57/100

5/5 - 1s - 145ms/step - accuracy: 0.7783 - loss: 0.8703 - val_accuracy: 0.6912 - val_loss: 1.0626

Epoch 58/100

5/5 - 1s - 149ms/step - accuracy: 0.7833 - loss: 0.8585 - val_accuracy: 0.6912 - val_loss: 1.0526

Epoch 59/100

5/5 - 1s - 141ms/step - accuracy: 0.7849 - loss: 0.8470 - val_accuracy: 0.6912 - val_loss: 1.0429

Epoch 60/100

5/5 - 1s - 145ms/step - accuracy: 0.7874 - loss: 0.8358 - val_accuracy: 0.6912 - val_loss: 1.0335

Epoch 61/100

5/5 - 1s - 149ms/step - accuracy: 0.7923 - loss: 0.8248 - val_accuracy: 0.6838 - val_loss: 1.0243

Epoch 62/100

5/5 - 1s - 147ms/step - accuracy: 0.7956 - loss: 0.8141 - val_accuracy: 0.7059 - val_loss: 1.0154

Epoch 63/100

5/5 - 1s - 149ms/step - accuracy: 0.7989 - loss: 0.8037 - val_accuracy: 0.7132 - val_loss: 1.0067

Epoch 64/100

5/5 - 1s - 147ms/step - accuracy: 0.8021 - loss: 0.7935 - val_accuracy: 0.7132 - val_loss: 0.9983

Epoch 65/100

5/5 - 1s - 150ms/step - accuracy: 0.8030 - loss: 0.7836 - val_accuracy: 0.7206 - val_loss: 0.9901

Epoch 66/100

5/5 - 1s - 149ms/step - accuracy: 0.8079 - loss: 0.7739 - val_accuracy: 0.7206 - val_loss: 0.9821

Epoch 67/100

5/5 - 1s - 145ms/step - accuracy: 0.8103 - loss: 0.7644 - val_accuracy: 0.7206 - val_loss: 0.9744

Epoch 68/100

5/5 - 1s - 145ms/step - accuracy: 0.8112 - loss: 0.7551 - val_accuracy: 0.7279 - val_loss: 0.9668

Epoch 69/100

5/5 - 1s - 141ms/step - accuracy: 0.8161 - loss: 0.7461 - val_accuracy: 0.7279 - val_loss: 0.9595

Epoch 70/100

5/5 - 1s - 148ms/step - accuracy: 0.8194 - loss: 0.7373 - val_accuracy: 0.7279 - val_loss: 0.9524

Epoch 71/100

5/5 - 1s - 147ms/step - accuracy: 0.8227 - loss: 0.7286 - val_accuracy: 0.7353 - val_loss: 0.9455

Epoch 72/100

5/5 - 1s - 146ms/step - accuracy: 0.8251 - loss: 0.7202 - val_accuracy: 0.7353 - val_loss: 0.9388

Epoch 73/100

5/5 - 1s - 155ms/step - accuracy: 0.8259 - loss: 0.7120 - val_accuracy: 0.7353 - val_loss: 0.9323

Epoch 74/100

5/5 - 1s - 151ms/step - accuracy: 0.8292 - loss: 0.7039 - val_accuracy: 0.7353 - val_loss: 0.9260

Epoch 75/100

5/5 - 1s - 149ms/step - accuracy: 0.8325 - loss: 0.6960 - val_accuracy: 0.7353 - val_loss: 0.9198

Epoch 76/100

5/5 - 1s - 151ms/step - accuracy: 0.8358 - loss: 0.6883 - val_accuracy: 0.7426 - val_loss: 0.9138

Epoch 77/100

5/5 - 1s - 147ms/step - accuracy: 0.8374 - loss: 0.6807 - val_accuracy: 0.7426 - val_loss: 0.9081

Epoch 78/100

5/5 - 1s - 149ms/step - accuracy: 0.8391 - loss: 0.6733 - val_accuracy: 0.7426 - val_loss: 0.9024

Epoch 79/100

5/5 - 1s - 146ms/step - accuracy: 0.8432 - loss: 0.6661 - val_accuracy: 0.7426 - val_loss: 0.8970

Epoch 80/100

5/5 - 1s - 149ms/step - accuracy: 0.8424 - loss: 0.6590 - val_accuracy: 0.7500 - val_loss: 0.8917

Epoch 81/100

5/5 - 1s - 152ms/step - accuracy: 0.8448 - loss: 0.6520 - val_accuracy: 0.7500 - val_loss: 0.8865

Epoch 82/100

5/5 - 1s - 153ms/step - accuracy: 0.8465 - loss: 0.6452 - val_accuracy: 0.7574 - val_loss: 0.8815

Epoch 83/100

5/5 - 1s - 150ms/step - accuracy: 0.8481 - loss: 0.6385 - val_accuracy: 0.7574 - val_loss: 0.8767

Epoch 84/100

5/5 - 1s - 151ms/step - accuracy: 0.8514 - loss: 0.6320 - val_accuracy: 0.7574 - val_loss: 0.8719

Epoch 85/100

5/5 - 1s - 150ms/step - accuracy: 0.8514 - loss: 0.6256 - val_accuracy: 0.7574 - val_loss: 0.8673

Epoch 86/100

5/5 - 1s - 149ms/step - accuracy: 0.8530 - loss: 0.6193 - val_accuracy: 0.7574 - val_loss: 0.8628

Epoch 87/100

5/5 - 1s - 149ms/step - accuracy: 0.8580 - loss: 0.6131 - val_accuracy: 0.7574 - val_loss: 0.8585

----------------------------------------------------------------------------
Test Accuracy 72.6%
```
</div>

### Predict (probabilities)


```python
test_logits = gat_model.predict(x=test_indices)

test_probs = ops.softmax(test_logits)

test_probs_np = ops.convert_to_numpy(test_probs)

mapping = {v: k for (k, v) in class_idx.items()}

for i, (probs, label) in enumerate(zip(test_probs_np[:10], test_labels[:10])):
    print(f"Example {i+1}: {mapping[label]}")
    for j, c in zip(probs, class_idx.keys()):
        print(f"\tProbability of {c: <24} = {j*100:7.3f}%")
    print("---" * 20)
```

    
<div class="k-default-codeblock">
```
43/43 ━━━━━━━━━━━━━━━━━━━━ 3s 66ms/step

Example 1: Probabilistic_Methods
	Probability of Case_Based               =   6.931%
	Probability of Genetic_Algorithms       =   6.779%
	Probability of Neural_Networks          =  51.883%
	Probability of Probabilistic_Methods    =  17.229%
	Probability of Reinforcement_Learning   =   5.418%
	Probability of Rule_Learning            =   3.978%
	Probability of Theory                   =   7.783%
------------------------------------------------------------
Example 2: Genetic_Algorithms
	Probability of Case_Based               =   7.132%
	Probability of Genetic_Algorithms       =  71.367%
	Probability of Neural_Networks          =   2.382%
	Probability of Probabilistic_Methods    =   1.951%
	Probability of Reinforcement_Learning   =   7.571%
	Probability of Rule_Learning            =   5.162%
	Probability of Theory                   =   4.436%
------------------------------------------------------------
Example 3: Theory
	Probability of Case_Based               =   9.217%
	Probability of Genetic_Algorithms       =  15.571%
	Probability of Neural_Networks          =  15.906%
	Probability of Probabilistic_Methods    =  18.614%
	Probability of Reinforcement_Learning   =   8.412%
	Probability of Rule_Learning            =  10.117%
	Probability of Theory                   =  22.164%
------------------------------------------------------------
Example 4: Neural_Networks
	Probability of Case_Based               =   4.347%
	Probability of Genetic_Algorithms       =   0.897%
	Probability of Neural_Networks          =  65.504%
	Probability of Probabilistic_Methods    =  18.453%
	Probability of Reinforcement_Learning   =   3.058%
	Probability of Rule_Learning            =   3.204%
	Probability of Theory                   =   4.537%
------------------------------------------------------------
Example 5: Theory
	Probability of Case_Based               =  10.485%
	Probability of Genetic_Algorithms       =  15.121%
	Probability of Neural_Networks          =  23.244%
	Probability of Probabilistic_Methods    =  18.306%
	Probability of Reinforcement_Learning   =   6.920%
	Probability of Rule_Learning            =   9.746%
	Probability of Theory                   =  16.179%
------------------------------------------------------------
Example 6: Genetic_Algorithms
	Probability of Case_Based               =   0.118%
	Probability of Genetic_Algorithms       =  98.859%
	Probability of Neural_Networks          =   0.288%
	Probability of Probabilistic_Methods    =   0.097%
	Probability of Reinforcement_Learning   =   0.343%
	Probability of Rule_Learning            =   0.160%
	Probability of Theory                   =   0.136%
------------------------------------------------------------
Example 7: Neural_Networks
	Probability of Case_Based               =   3.101%
	Probability of Genetic_Algorithms       =   1.111%
	Probability of Neural_Networks          =  52.974%
	Probability of Probabilistic_Methods    =  31.954%
	Probability of Reinforcement_Learning   =   2.311%
	Probability of Rule_Learning            =   2.409%
	Probability of Theory                   =   6.140%
------------------------------------------------------------
Example 8: Genetic_Algorithms
	Probability of Case_Based               =   1.059%
	Probability of Genetic_Algorithms       =  94.610%
	Probability of Neural_Networks          =   0.490%
	Probability of Probabilistic_Methods    =   0.525%
	Probability of Reinforcement_Learning   =   0.849%
	Probability of Rule_Learning            =   1.468%
	Probability of Theory                   =   0.998%
------------------------------------------------------------
Example 9: Theory
	Probability of Case_Based               =  11.802%
	Probability of Genetic_Algorithms       =  10.381%
	Probability of Neural_Networks          =  31.400%
	Probability of Probabilistic_Methods    =  21.771%
	Probability of Reinforcement_Learning   =   8.059%
	Probability of Rule_Learning            =   6.866%
	Probability of Theory                   =   9.721%
------------------------------------------------------------
Example 10: Case_Based
	Probability of Case_Based               =  39.797%
	Probability of Genetic_Algorithms       =   6.685%
	Probability of Neural_Networks          =  14.621%
	Probability of Probabilistic_Methods    =  15.383%
	Probability of Reinforcement_Learning   =   6.294%
	Probability of Rule_Learning            =   9.628%
	Probability of Theory                   =   7.594%
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

---
## Relevant Chapters from Deep Learning with Python
- [Chapter 7: A deep dive on Keras](https://deeplearningwithpython.io/chapters/chapter07_deep-dive-keras)
- [Chapter 15: Language models and the Transformer](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer)

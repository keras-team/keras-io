"""
Title: Message-passing neural network for molecular property prediction
Author: [akensert](http://github.com/akensert)
Date created: 2021/08/16
Last modified: 2021/08/16
Description: Implementation of an MPNN to predict blood-brain barrier permeability.
"""
"""
## Introduction

In this tutorial, we will implement a type of graph neural network (GNN) known as
_ message passing neural network_ (MPNN) to predict graph properties. Specifically, we will
implement an MPNN to predict a molecular property known as
_blood-brain barrier permeability_ (BBBP).

Motivation: as molecules are naturally represented as an undirected graph `G = (V, E)`,
where `V` is a set or vertices (nodes; atoms) and `E` a set of edges (bonds), GNNs (such
as MPNN) are proving to be a useful method for predicting molecular properties.

Until now, more traditional methods, such as random forests, support vector machines, etc.,
have been commonly used to predict molecular properties. In contrast to GNNs, these
traditional approaches often operate on precomputed molecular features such as
molecular weight, polarity, charge, number of carbon atoms, etc. Although these
molecular features prove to be good predictors for various molecular properties, it is
hypothesized that operating on these more "raw", "low-level", features could prove even
better.

### References

In recent years, a lot of effort has been put into developing neural networks for
graph data, including molecular graphs. For a summary of graph neural networks, see e.g.,
[A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596) and
[Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434);
and for further reading on the specific
graph neural network implemented in this tutorial see
[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) and
[DeepChem's MPNNModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#mpnnmodel).
"""

"""
## Setup

### Install RDKit and other dependencies

(Text below taken from
[this tutorial](https://keras.io/examples/generative/wgan-graphs/)).

[RDKit](https://www.rdkit.org/) is a collection of cheminformatics and machine-learning
software written in C++ and Python. In this tutorial, RDKit is used to conveniently and
efficiently transform
[SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) to
molecule objects, and then from those obtain sets of atoms and bonds.

SMILES expresses the structure of a given molecule in the form of an ASCII string.
The SMILES string is a compact encoding which, for smaller molecules, is relatively
human-readable. Encoding molecules as a string both alleviates and facilitates database
and/or web searching of a given molecule. RDKit uses algorithms to
accurately transform a given SMILES to a molecule object, which can then
be used to compute a great number of molecular properties/features.

Notice, RDKit is commonly installed via [Conda](https://www.rdkit.org/docs/Install.html).
However, thanks to
[rdkit_platform_wheels](https://github.com/kuelumbus/rdkit_platform_wheels), rdkit
can now (for the sake of this tutorial) be installed easily via pip, as follows:

```
pip -q install rdkit-pypi
```

And for easy and efficient reading of csv files and visualization, the below needs to be
installed:

```
pip -q install pandas
pip -q install Pillow
pip -q install matplotlib
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
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
import logging

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

np.random.seed(42)
tf.random.set_seed(42)

"""
## Dataset

Information about the dataset can be found in
[A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling](https://pubs.acs.org/doi/10.1021/ci300124c)
and [MoleculeNet: A Benchmark for Molecular Machine Learning](https://arxiv.org/abs/1703.00564).
The dataset will be downloaded from [MoleculeNet.ai](http://moleculenet.ai/datasets-1).

### About

The dataset contains **2,050** molecules. Each molecule come with a **name**, **label**
and **SMILES** string.

The blood-brain barrier (BBB) is a membrane separating the blood from the brain
extracellular fluid, hence blocking out most drugs (molecules) from reaching
the brain. Because of this, the BBBP has been important to study for the development of
new drugs that aim to target the central nervous system. The labels for this
data set are binary (1 or 0) and indicate the permeability of the molecules.
"""

csv_path = keras.utils.get_file(
    "BBBP.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
)

df = pd.read_csv(csv_path, usecols=[1, 2, 3])
df.iloc[96:104]

"""
### Define features

To encode features for atoms and bonds (which we will need later),
we'll define two classes: `AtomFeaturizer` and `BondFeatuzier` respectively.

To reduce the lines of code, i.e., to keep this tutorial short and concise,
only about a handful of (atom and bond) features will be considered: \[atom features\]
[symbol (element)](https://en.wikipedia.org/wiki/Chemical_element),
[number of valence electrons](https://en.wikipedia.org/wiki/Valence_electron),
[number of hydrogen bonds](https://en.wikipedia.org/wiki/Hydrogen),
[orbital hybridization](https://en.wikipedia.org/wiki/Orbital_hybridisation),
\[bond features\]
[(covalent) bond type](https://en.wikipedia.org/wiki/Covalent_bond), and
[conjugation](https://en.wikipedia.org/wiki/Conjugated_system).
"""


class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)


"""
### Generate graphs

Before we can generate complete graphs from SMILES, we need to implement the following functions:

1. `molecule_from_smiles`, which takes as input a SMILES and returns a molecule object.
This is all handled by RDKit.

2. `graph_from_molecule`, which takes as input a molecule object and returns a graph,
represented as a three-tuple (atom_features, bond_features, pair_indices). For this we
will make use of the classes defined previously.

Finally, we can now implement the function `graphs_from_smiles`, which applies function (1)
and subsequently (2) on all SMILES of the training, validation and test datasets.

Notice: although scaffold splitting is recommended for this data set (see
[here](https://arxiv.org/abs/1703.00564)), for simplicity, simple random splittings were
performed.
"""


def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loop. Notice, this also helps against some edge cases where the
        # last node has no edges. Alternatively, if no self-loops are used, for these
        # edge cases, zero-padding on the output of the edge network is needed.
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        atom_neighbors = atom.GetNeighbors()

        for neighbor in atom_neighbors:
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


def graphs_from_smiles(smiles_list):
    # Initialize graphs
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    # Convert lists to ragged tensors for tf.data.Dataset later on
    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )


# Shuffle array of indices ranging from 0 to 2049
permuted_indices = np.random.permutation(np.arange(df.shape[0]))

# Train set: 80 % of data
train_index = permuted_indices[: int(df.shape[0] * 0.8)]
x_train = graphs_from_smiles(df.iloc[train_index].smiles)
y_train = df.iloc[train_index].p_np

# Valid set: 19 % of data
valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.99)]
x_valid = graphs_from_smiles(df.iloc[valid_index].smiles)
y_valid = df.iloc[valid_index].p_np

# Test set: 1 % of data
test_index = permuted_indices[int(df.shape[0] * 0.99) :]
x_test = graphs_from_smiles(df.iloc[test_index].smiles)
y_test = df.iloc[test_index].p_np

"""
### Test the functions
"""

print(f"Name:\t{df.name[100]}\nSMILES:\t{df.smiles[100]}\nBBBP:\t{df.p_np[100]}")
molecule = molecule_from_smiles(df.iloc[100].smiles)
print("Molecule:")
molecule

"""
"""

graph = graph_from_molecule(molecule)
print("Graph (including self-loops):")
print("\tatom features\t", graph[0].shape)
print("\tbond features\t", graph[1].shape)
print("\tpair indices\t", graph[2].shape)


"""
### Create a `tf.data.Dataset`

In this tutorial, the MPNN implementation will take as input (per iteration) a single graph.
Therefore, given a batch of (sub)graphs (molecules), we need to merge them into a
single graph (we'll refer to this graph as *global graph*).
This global graph is a disconnected graph where each subgraph is
completely separated from the other subgraphs.
"""


def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    atom_features, bond_features, pair_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices. atom_partition_indices will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = tf.range(len(num_atoms))
    atom_partition_indices = tf.repeat(molecule_indices, num_atoms)
    bond_partition_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(
        tf.gather(increment, bond_partition_indices), [(num_bonds[0], 0)]
    )
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, atom_partition_indices), y_batch


def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1)


"""
## Model

The MPNN model can take on various shapes and forms. In this tutorial, we will implement an
MPNN based on the original paper
[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) and
[DeepChem's MPNNModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#mpnnmodel).
The MPNN of this tutorial consists of three stages: message passing, readout and
classification.


### Message passing

The message passing step itself consists of two parts:

1. The *edge network*, which passes messages from 1-hop neighbors `w^{t}_{i}` of `v^{t}`
to `v^{t}`, based on the edge features between them (`e_{v^{t}w^{t}_{i}}`, where `t =
0`), resulting in an updated node state `v^{t+1}`. `_{i}` denotes the `i:th` neighbor of
`v^{t}` and `^{t}` the `t:th` state of `v` or `w`. An important feature of the edge
network (in contrast to e.g. the relational graph convolutional network) is that it
allows for non-discrete edge features. However, in this tutorial, only discrete edge
features will be used.


2. The *gated recurrent unit* (GRU), which takes as input the most recent node state
(e.g., `v^{t+1}`) and updates it based on previous node state(s) (e.g., `v^{t}`). In
other words, the most recent node states serves as the input to the GRU, while the previous
node state(s) are incorporated within the memory state of the GRU.

Importantly, step (1) and (2) are repeated for `k steps`, and where at each step `1...k`,
the radius (or # hops) of aggregated information from the source node `v` increases by 1.
"""


class EdgeNetwork(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            trainable=True,
            initializer="glorot_uniform",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim), trainable=True, initializer="zeros",
        )
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Apply linear transformation to bond features
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

        # Obtain atom features of neighbors
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

        # Apply neighborhood aggregation
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.segment_sum(
            transformed_features, pair_indices[:, 0]
        )
        return aggregated_features


class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Pad atom features if number of desired units exceeds atom_features dim
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate atom_features from neighbors
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )

            # Update aggregated atom_features via a step of GRU
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated


"""
### Readout

When the message passing procedure ends, the k-step-aggregated node states are to be partitioned
into subgraphs (correspoding to each molecule in the batch) and subsequently
reduced to graph-level embeddings. In the
[original paper](https://arxiv.org/abs/1704.01212), a
[set-to-set layer](https://arxiv.org/abs/1511.06391) was used for this purpose.
In this tutorial however, a transformer encoder will be used. Specifically:

* the k-step-aggregated node states will be partitioned into the subgraphs
(corresponding to each molecule in the batch);
* each subgraph will then be padded to match the subgraph with the greatest number of nodes, followed
by a `tf.stack(...)`;
* the (stacked) padded tensor, encoding subgraphs (each subgraph containing sets of node states), are
masked to make sure the paddings don't interfere with training;
* finally, the padded tensor is passed to the transformer followed by an average pooling.
"""


class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, atom_partition_indices = inputs

        # Obtain subgraphs
        atom_features = tf.dynamic_partition(
            atom_features, atom_partition_indices, self.batch_size
        )

        # Pad and stack subgraphs
        num_atoms = [tf.shape(f)[0] for f in atom_features]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_padded = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features, num_atoms)
            ],
            axis=0,
        )

        # Remove empty subgraphs (usually for last batch)
        nonempty_examples = tf.where(tf.reduce_sum(atom_features_padded, (1, 2)) != 0)
        nonempty_examples = tf.squeeze(nonempty_examples, axis=-1)

        return tf.gather(atom_features_padded, nonempty_examples, axis=0)


class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads=8, embed_dim=64, dense_dim=512, **kwargs):
        super().__init__(**kwargs)

        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        attention_mask = mask[:, tf.newaxis, :] if mask is not None else None
        attention_output = self.attention(inputs, inputs, attention_mask=attention_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        return self.layernorm_2(proj_input + self.dense_proj(proj_input))


"""
### Message Passing Neural Network (MPNN)

It is now time to complete the MPNN model. In addition to the message passing
and readout, a two-layer classification network will be implemented to make
predictions of BBBP.
"""


def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):
    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    atom_partition_indices = layers.Input(
        (), dtype="int32", name="atom_partition_indices"
    )

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = PartitionPadding(batch_size)([x, atom_partition_indices])

    x = layers.Masking()(x)

    x = TransformerEncoder(num_attention_heads, message_units, dense_units)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, atom_partition_indices],
        outputs=[x],
    )
    return model


mpnn = MPNNModel(
    atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
)

mpnn.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    metrics=[keras.metrics.AUC(name="AUC")],
)

keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)

"""
### Training
"""

train_dataset = MPNNDataset(x_train, y_train)
valid_dataset = MPNNDataset(x_valid, y_valid)
test_dataset = MPNNDataset(x_test, y_test)

history = mpnn.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=40,
    verbose=2,
    class_weight={0: 2.0, 1: 0.5},
)

plt.figure(figsize=(10, 6))
plt.plot(history.history["AUC"], label="train AUC")
plt.plot(history.history["val_AUC"], label="valid AUC")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("AUC", fontsize=16)
plt.legend(fontsize=16)

"""
### Predicting
"""

molecules = [molecule_from_smiles(df.smiles.values[index]) for index in test_index]
y_true = [df.p_np.values[index] for index in test_index]
y_pred = tf.squeeze(mpnn.predict(test_dataset), axis=1)

legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f}" for i in range(len(y_true))]
MolsToGridImage(molecules, molsPerRow=4, legends=legends)

"""
## Conclusions

In this tutorial, we demonstarted a message passing neural network (MPNN) to
predict blood-brain barrier permeability (BBBP) for a number of different molecules. We
first had to construct graphs from SMILES, and then build a Keras model that could
operate on these graphs.
"""

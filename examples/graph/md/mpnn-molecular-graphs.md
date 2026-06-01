# Message-passing neural network (MPNN) for molecular property prediction

**Author:** [akensert](http://github.com/akensert)<br>
**Date created:** 2021/08/16<br>
**Last modified:** 2026/06/01<br>
**Description:** Implementation of an MPNN to predict blood-brain barrier permeability.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/graph/ipynb/mpnn-molecular-graphs.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/graph/mpnn-molecular-graphs.py)



---
## Introduction

In this tutorial, we will implement a type of graph neural network (GNN)
known as _message passing neural network_ (MPNN) to predict graph
properties. Specifically, we will
implement an MPNN to predict a molecular property known as
_blood-brain barrier permeability_ (BBBP).

Motivation: as molecules are naturally represented as an undirected
graph `G = (V, E)`, where `V` is a set or vertices (nodes; atoms) and
`E` a set of edges (bonds), GNNs (such
as MPNN) are proving to be a useful method for predicting molecular properties.

Until now, more traditional methods, such as random forests, support
vector machines, etc.,
have been commonly used to predict molecular properties. In contrast to GNNs, these
traditional approaches often operate on precomputed molecular features such as
molecular weight, polarity, charge, number of carbon atoms, etc. Although these
molecular features prove to be good predictors for various molecular properties, it is
hypothesized that operating on these more "raw", "low-level", features could prove even
better.

### References

In recent years, a lot of effort has been put into developing neural
networks for graph data, including molecular graphs. For a summary of
graph neural networks, see e.g.,
[A Comprehensive Survey on Graph Neural
Networks](https://arxiv.org/abs/1901.00596) and [Graph Neural Networks:
A Review of Methods and Applications](https://arxiv.org/abs/1812.08434);
and for further reading on the specific
graph neural network implemented in this tutorial see
[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) and
[DeepChem's MPNNModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#mpnnmodel).

---
## Setup

### Install RDKit and other dependencies

(Text below taken from
[this tutorial](https://keras.io/examples/generative/wgan-graphs/)).

[RDKit](https://www.rdkit.org/) is a collection of cheminformatics and
machine-learning software written in C++ and Python. In this tutorial,
RDKit is used to conveniently and
efficiently transform
[SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)
to
molecule objects, and then from those obtain sets of atoms and bonds.

SMILES expresses the structure of a given molecule in the form of an ASCII string.
The SMILES string is a compact encoding which, for smaller molecules, is
relatively
human-readable. Encoding molecules as a string both alleviates and facilitates database
and/or web searching of a given molecule. RDKit uses algorithms to
accurately transform a given SMILES to a molecule object, which can then
be used to compute a great number of molecular properties/features.

Notice, RDKit is commonly installed via
[Conda](https://www.rdkit.org/docs/Install.html).
However, thanks to
[rdkit_platform_wheels](https://github.com/kuelumbus/rdkit_platform_wheels), rdkit
can now (for the sake of this tutorial) be installed easily via pip, as follows:

```
pip -q install rdkit
```

And for easy and efficient reading of csv files and visualization, the
below needs to be installed:

```
pip -q install pandas
pip -q install Pillow
pip -q install matplotlib
pip -q install pydot
pip -q install graphviz
```

### Import packages


```python
import os

# Set backend before importing keras (Options: 'jax', 'torch', 'tensorflow')
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers, ops, regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import MolsToGridImage
from tqdm import tqdm

# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# Set random seeds using Keras 3 utility
keras.utils.set_random_seed(42)

# --- GLOBAL CONFIGURATION ---
MAX_ATOMS = 70  # Maximum atoms per molecule
MAX_BONDS = 150  # Maximum bonds per molecule
BATCH_SIZE = 64  # Increased for faster GPU utilization
EPOCHS = 40
LEARNING_RATE = 5e-4
```

---
## Dataset

Information about the dataset can be found in
[A Bayesian Approach to in Silico Blood-Brain Barrier Penetration
Modeling](https://pubs.acs.org/doi/10.1021/ci300124c) and [MoleculeNet:
A Benchmark for Molecular Machine Learning](https://arxiv.org/abs/1703.00564).
The dataset will be downloaded from
[MoleculeNet.org](https://moleculenet.org/datasets-1).

### About

The dataset contains **2,050** molecules. Each molecule come with a
**name**, **label**
and **SMILES** string.

The blood-brain barrier (BBB) is a membrane separating the blood from the brain
extracellular fluid, hence blocking out most drugs (molecules) from reaching
the brain. Because of this, the BBBP has been important to study for the
development of
new drugs that aim to target the central nervous system. The labels for this
data set are binary (1 or 0) and indicate the permeability of the molecules.


```python
csv_path = keras.utils.get_file(
    "BBBP.csv",
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
)
df = pd.read_csv(csv_path, usecols=["name", "p_np", "smiles"])
df.iloc[96:104]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

<div class="k-default-codeblock">
```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```
</div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>p_np</th>
      <th>smiles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>96</th>
      <td>cefoxitin</td>
      <td>1</td>
      <td>CO[C@]1(NC(=O)Cc2sccc2)[C@H]3SCC(=C(N3C1=O)C(O...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Org34167</td>
      <td>1</td>
      <td>NC(CC=C)c1ccccc1c2noc3c2cccc3</td>
    </tr>
    <tr>
      <th>98</th>
      <td>9-OH Risperidone</td>
      <td>1</td>
      <td>OC1C(N2CCC1)=NC(C)=C(CCN3CCC(CC3)c4c5ccc(F)cc5...</td>
    </tr>
    <tr>
      <th>99</th>
      <td>acetaminophen</td>
      <td>1</td>
      <td>CC(=O)Nc1ccc(O)cc1</td>
    </tr>
    <tr>
      <th>100</th>
      <td>acetylsalicylate</td>
      <td>0</td>
      <td>CC(=O)Oc1ccccc1C(O)=O</td>
    </tr>
    <tr>
      <th>101</th>
      <td>allopurinol</td>
      <td>0</td>
      <td>O=C1N=CN=C2NNC=C12</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Alprostadil</td>
      <td>0</td>
      <td>CCCCC[C@H](O)/C=C/[C@H]1[C@H](O)CC(=O)[C@@H]1C...</td>
    </tr>
    <tr>
      <th>103</th>
      <td>aminophylline</td>
      <td>0</td>
      <td>CN1C(=O)N(C)c2nc[nH]c2C1=O.CN3C(=O)N(C)c4nc[nH...</td>
    </tr>
  </tbody>
</table>
</div>



### Define features

To encode features for atoms and bonds (which we will need later),
we'll define two classes: `AtomFeaturizer` and `BondFeaturizer` respectively.

To reduce the lines of code, i.e., to keep this tutorial short and concise,
only about a handful of (atom and bond) features will be considered:
\[atom features\]
[symbol (element)](https://en.wikipedia.org/wiki/Chemical_element),
[number of valence electrons](https://en.wikipedia.org/wiki/Valence_electron),
[number of hydrogen bonds](https://en.wikipedia.org/wiki/Hydrogen),
[orbital hybridization](https://en.wikipedia.org/wiki/Orbital_hybridisation),
\[bond features\]
[(covalent) bond type](https://en.wikipedia.org/wiki/Covalent_bond), and
[conjugation](https://en.wikipedia.org/wiki/Conjugated_system).


```python

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for feature_name, allowable_values in allowable_sets.items():
            allowable_values = sorted(list(allowable_values))
            self.features_mapping[feature_name] = dict(
                zip(
                    allowable_values,
                    range(self.dim, len(allowable_values) + self.dim),
                )
            )
            self.dim += len(allowable_values)

    def encode(self, inputs):
        output = np.zeros((self.dim,), dtype="float32")
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature in feature_mapping:
                output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
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
        output = np.zeros((self.dim,), dtype="float32")
        if bond is None:
            output[-1] = 1.0
            return output
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(bond)
            if feature in feature_mapping:
                output[feature_mapping[feature]] = 1.0
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    {
        "symbol": {
            "B",
            "Br",
            "C",
            "Ca",
            "Cl",
            "F",
            "H",
            "I",
            "N",
            "Na",
            "O",
            "P",
            "S",
        },
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)
bond_featurizer = BondFeaturizer(
    {
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)

```

### Generate graphs

Before generating complete graphs from SMILES, we need to implement the
following functions:

1. `molecule_from_smiles`: This takes a SMILES string as input and
returns an RDKit molecule object. This process remains handled by RDKit
on the CPU.
2. `smiles_to_graph`: This takes a SMILES string and returns a graph
represented as a four-tuple:
`(atom_features, bond_features, pair_indices, mask)`.
The original implementation utilized tf.RaggedTensor, which is exclusive
to TensorFlow. To remain backend-agnostic and support JAX and PyTorch,
we now use fixed-size buffers (MAX_ATOMS and MAX_BONDS). We also
introduce a mask - a boolean array that allows the model to distinguish
between valid chemical data and zero-padding.
Finally, implemented a pre-featurization step. Instead of featurizing
during the training loop (which creates a CPU bottleneck), we process
all SMILES once and store them in a list of NumPy arrays. This allows
the GPU backends to run at 100% efficiency.


```python

def molecule_from_smiles(smiles):
    # Standard RDKit sanitization and stereochemistry assignment
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def smiles_to_graph(smiles):
    """
    Converts SMILES to a graph with fixed-size buffers for
    Keras 3 compatibility.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # Pre-allocate fixed buffers for static shapes (required for JAX/Torch)
    atom_features = np.zeros((MAX_ATOMS, atom_featurizer.dim), dtype="float32")
    bond_features = np.zeros((MAX_BONDS, bond_featurizer.dim), dtype="float32")
    pair_indices = np.zeros((MAX_BONDS, 2), dtype="int32")
    mask = np.zeros((MAX_ATOMS,), dtype="float32")

    atoms = mol.GetAtoms()
    for atom_index, atom in enumerate(atoms):
        if atom_index >= MAX_ATOMS:
            break
        atom_features[atom_index] = atom_featurizer.encode(atom)
        mask[atom_index] = 1.0

    bond_count = 0
    for atom_index, atom in enumerate(atoms):
        if atom_index >= MAX_ATOMS or bond_count >= MAX_BONDS:
            break
        # Add self-loop (standard in MPNN)
        pair_indices[bond_count] = [atom_index, atom_index]
        bond_features[bond_count] = bond_featurizer.encode(None)
        bond_count += 1

        for neighbor_atom in atom.GetNeighbors():
            neighbor_index = neighbor_atom.GetIdx()
            if neighbor_index >= MAX_ATOMS or bond_count >= MAX_BONDS:
                continue
            pair_indices[bond_count] = [atom_index, neighbor_index]
            bond_features[bond_count] = bond_featurizer.encode(
                mol.GetBondBetweenAtoms(atom_index, neighbor_index)
            )
            bond_count += 1

    return atom_features, bond_features, pair_indices, mask


csv_path = keras.utils.get_file(
    "BBBP.csv",
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
)
df = pd.read_csv(csv_path, usecols=["name", "p_np", "smiles"])

# Pre-featurize once to remove the RDKit bottleneck during training.
print("Pre-featurizing Dataset...")
processed_data = []
for smiles_string in tqdm(df.smiles.values):
    graph = smiles_to_graph(smiles_string)
    if graph is None:
        # Placeholder for failed molecules to maintain index alignment
        processed_data.append(
            (
                np.zeros((MAX_ATOMS, atom_featurizer.dim)),
                np.zeros((MAX_BONDS, bond_featurizer.dim)),
                np.zeros((MAX_BONDS, 2), dtype="int32"),
                np.zeros((MAX_ATOMS,)),
            )
        )
    else:
        processed_data.append(graph)

```

<div class="k-default-codeblock">
```
Pre-featurizing Dataset...
```
</div>

  0%|                                                                                                                          | 0/2050 [00:00<?, ?it/s]

    
 20%|█████████████████████▉                                                                                        | 408/2050 [00:00<00:00, 4075.89it/s]

    
 40%|████████████████████████████████████████████▎                                                                 | 826/2050 [00:00<00:00, 4135.50it/s]

    
 60%|█████████████████████████████████████████████████████████████████▉                                           | 1240/2050 [00:00<00:00, 4021.16it/s]

    
 83%|██████████████████████████████████████████████████████████████████████████████████████████                   | 1695/2050 [00:00<00:00, 4225.12it/s]

    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2050/2050 [00:00<00:00, 4248.14it/s]

    


### Test the functions

We can now inspect a sample molecule and its corresponding graph
representation. Note that the output shapes are now constant
(e.g., 70 atoms and 150 bonds), ensuring compatibility across all
Keras 3 backends.


```python
sample_idx = 100
print(
    f"Name:\t{df.name[sample_idx]}\n"
    f"SMILES:\t{df.smiles[sample_idx]}\n"
    f"BBBP:\t{df.p_np[sample_idx]}"
)

molecule = molecule_from_smiles(df.smiles.values[sample_idx])
print("Molecule object created successfully.")
molecule

# Convert to graph and check constant shapes
(
    sample_atom_features,
    sample_bond_features,
    sample_pair_indices,
    sample_mask,
) = smiles_to_graph(df.smiles.values[sample_idx])
print("Graph (including self-loops and padding):")
print(f"\tatom features\t {sample_atom_features.shape}")
print(f"\tbond features\t {sample_bond_features.shape}")
print(f"\tpair indices \t {sample_pair_indices.shape}")
```

<div class="k-default-codeblock">
```
Name:	acetylsalicylate
SMILES:	CC(=O)Oc1ccccc1C(O)=O
BBBP:	0
Molecule object created successfully.
Graph (including self-loops and padding):
	atom features	 (70, 29)
	bond features	 (150, 7)
	pair indices 	 (150, 2)
```
</div>

### Data Loading with PyDataset

In this tutorial, the MPNN implementation takes a single graph as input
per iteration.
To process a batch of molecules, we merge them into a single global
graph (also known as a disjoint graph).
This global graph is a disconnected structure where each molecule
(subgraph) is separated from the others.


```python

class MPNNDataset(keras.utils.PyDataset):
    def __init__(self, data, labels, batch_size=64, shuffle=False, **kwargs):
        super().__init__(**kwargs)
        self.data, self.labels, self.batch_size, self.shuffle = (
            data,
            labels,
            batch_size,
            shuffle,
        )
        self.indices = np.arange(len(data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        start, end = idx * self.batch_size, min(
            (idx + 1) * self.batch_size, len(self.indices)
        )
        batch_idx = self.indices[start:end]
        batch_atom_features = np.zeros(
            (self.batch_size, MAX_ATOMS, atom_featurizer.dim), dtype="float32"
        )
        batch_bond_features = np.zeros(
            (self.batch_size, MAX_BONDS, bond_featurizer.dim), dtype="float32"
        )
        batch_pair_indices = np.zeros((self.batch_size, MAX_BONDS, 2), dtype="int32")
        batch_mask = np.zeros((self.batch_size, MAX_ATOMS), dtype="float32")
        batch_labels = np.zeros((self.batch_size, 1), dtype="float32")

        for i, real_idx in enumerate(batch_idx):
            (
                batch_atom_features[i],
                batch_bond_features[i],
                batch_pair_indices[i],
                batch_mask[i],
            ) = self.data[real_idx]
            batch_labels[i] = self.labels[real_idx]
            batch_pair_indices[i] += i * MAX_ATOMS

        return {
            "atom_features": ops.convert_to_tensor(
                batch_atom_features.reshape(-1, atom_featurizer.dim)
            ),
            "bond_features": ops.convert_to_tensor(
                batch_bond_features.reshape(-1, bond_featurizer.dim)
            ),
            "pair_indices": ops.convert_to_tensor(batch_pair_indices.reshape(-1, 2)),
            "molecule_indicator": ops.convert_to_tensor(
                np.repeat(np.arange(self.batch_size), MAX_ATOMS), dtype="int32"
            ),
            "mask": ops.convert_to_tensor(batch_mask.reshape(-1)),
        }, ops.convert_to_tensor(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# Shuffle and split the data indices
perm = np.random.permutation(len(processed_data))

# Train: 80% | Valid: 19% | Test: 1%
train_idx = perm[: int(len(df) * 0.8)]
val_idx = perm[int(len(df) * 0.8) : int(len(df) * 0.99)]
test_idx = perm[int(len(df) * 0.99) :]

# Create the PyDatasets
train_dataset = MPNNDataset(
    [processed_data[data_index] for data_index in train_idx],
    df.p_np.values[train_idx],
    batch_size=BATCH_SIZE,
    shuffle=True,
)

valid_dataset = MPNNDataset(
    [processed_data[data_index] for data_index in val_idx],
    df.p_np.values[val_idx],
    batch_size=BATCH_SIZE,
)

# Instantiate the test dataset
test_dataset = MPNNDataset(
    [processed_data[data_index] for data_index in test_idx],
    df.p_np.values[test_idx],
    batch_size=BATCH_SIZE,
)

print(
    f"Dataset Split: Train={len(train_idx)}, "
    f"Valid={len(val_idx)}, Test={len(test_idx)}"
)

```

<div class="k-default-codeblock">
```
Dataset Split: Train=1640, Valid=389, Test=21
```
</div>

---
## Model

The MPNN model can take on various shapes and forms. In this tutorial,
we will implement an
MPNN based on the original paper
[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) and
[DeepChem's MPNNModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#mpnnmodel).
The MPNN of this tutorial consists of three stages: message passing, readout and
classification.


### Message passing

The Message Passing Neural Network (MPNN) architecture implemented in
this tutorial consists of three stages: message passing, readout, and
classification. The message passing step is the core of the model,
enabling information to flow through the molecular graph. It consists of
two main components:

1. The *edge network*, which passes messages from 1-hop neighbors `w_{i}` of `v`
to `v`, based on the edge features between them (`e_{vw_{i}}`),
resulting in an updated node (state) `v'`. `w_{i}` denotes the `i:th` neighbor of
`v`.

2. The *gated recurrent unit* (GRU), which takes as input the most
recent node state
and updates it based on previous node states. In
other words, the most recent node state serves as the input to the GRU,
while the previous
node states are incorporated within the memory state of the GRU. This allows information
to travel from one node state (e.g., `v`) to another (e.g., `v''`).

Importantly, step (1) and (2) are repeated for `k steps`, and where at
each step `1...k`,
the radius (or number of hops) of aggregated information from `v` increases by 1.


```python

class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim, self.bond_dim = input_shape[0][-1], input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim**2),
            initializer="glorot_uniform",
        )
        self.bias = self.add_weight(shape=(self.atom_dim**2,), initializer="zeros")

    def call(self, inputs):
        atom_feat, bond_feat, pair_idx = inputs
        bond_transformed = ops.matmul(bond_feat, self.kernel) + self.bias
        bond_transformed = ops.reshape(
            bond_transformed, (-1, self.atom_dim, self.atom_dim)
        )
        neighbor_feat = ops.take(atom_feat, ops.cast(pair_idx[:, 1], "int32"), axis=0)
        messages = ops.squeeze(
            ops.matmul(bond_transformed, ops.expand_dims(neighbor_feat, -1)), -1
        )
        return ops.segment_sum(
            messages,
            ops.cast(pair_idx[:, 0], "int32"),
            num_segments=ops.shape(atom_feat)[0],
        )


class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units, self.steps = units, steps
        self.edge_net = EdgeNetwork()
        self.gru = layers.GRUCell(units)
        self.norm = layers.LayerNormalization()

    def call(self, inputs):
        atom_feat, bond_feat, pair_idx = inputs
        atom_feat = ops.pad(
            atom_feat,
            [(0, 0), (0, max(0, self.units - ops.shape(atom_feat)[-1]))],
        )
        for _ in range(self.steps):
            messages = self.edge_net([atom_feat, bond_feat, pair_idx])
            atom_feat, _ = self.gru(messages, atom_feat)
            atom_feat = self.norm(atom_feat)  # Normalize every step
        return atom_feat

```

### Readout

When the message passing procedure ends, the k-step-aggregated node
states are to be partitioned
into subgraphs (corresponding to each molecule in the batch) and subsequently
reduced to graph-level embeddings. In the
[original paper](https://arxiv.org/abs/1704.01212), a
[set-to-set layer](https://arxiv.org/abs/1511.06391) was used for this purpose.
In this tutorial, we utilize a Gated Readout combined with Hybrid
Pooling (Mean and Max).
This approach is highly stable and fully compatible with JAX, PyTorch,
and TensorFlow. The process works as follows:
Gating Mechanism: Each node state passes through a learned gating
function (using sigmoid and tanh activations). This allows the model to
"decide" which atoms are most important for the molecular property being
predicted.
Masking: We use the mask generated in our data pipeline to ensure that
padded (zero) atoms do not contribute to the final graph embedding.
Hybrid Segment Pooling: Instead of physically partitioning the tensors,
we use the molecule_indicator (batch index) to logically group atoms. We
calculate both the Mean and the Max of the node states for each
molecule.
Concatenation: The mean and max features are concatenated to form a
robust, fixed-size graph-level representation.


```python

class GatedReadout(layers.Layer):
    """A more stable readout using both Mean and Max pooling."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.gate = layers.Dense(embed_dim, activation="sigmoid")
        self.feat = layers.Dense(embed_dim, activation="tanh")

    def call(self, inputs):
        nodes, indicator, mask = inputs
        mask = ops.expand_dims(mask, -1)
        num_molecules = ops.max(ops.cast(indicator, "int32")) + 1

        # Gated logic: atoms "decide" how much they contribute
        gated_x = self.gate(nodes) * self.feat(nodes)
        gated_x = gated_x * mask

        # Combined Mean and Max pooling for robustness
        x_mean = ops.segment_sum(
            gated_x,
            ops.cast(indicator, "int32"),
            num_segments=num_molecules,
        ) / ops.maximum(
            ops.segment_sum(
                mask,
                ops.cast(indicator, "int32"),
                num_segments=num_molecules,
            ),
            1e-6,
        )
        x_max = ops.segment_max(
            gated_x,
            ops.cast(indicator, "int32"),
            num_segments=num_molecules,
        )

        return ops.concatenate([x_mean, x_max], axis=-1)

```

### Message Passing Neural Network (MPNN)

It is now time to complete the MPNN model. In addition to the message passing
and readout, a two-layer classification network will be implemented to make
predictions of BBBP.


```python

def MPNNModel(atom_dim, bond_dim):
    atom_input = layers.Input(shape=(atom_dim,), name="atom_features")
    bond_input = layers.Input(shape=(bond_dim,), name="bond_features")
    pair_indices_input = layers.Input(shape=(2,), dtype="int32", name="pair_indices")
    molecule_indicator_input = layers.Input(
        shape=(), dtype="int32", name="molecule_indicator"
    )
    mask_input = layers.Input(shape=(), name="mask")

    hidden_features = MessagePassing(64, steps=4)(
        [atom_input, bond_input, pair_indices_input]
    )
    hidden_features = GatedReadout(64)(
        [hidden_features, molecule_indicator_input, mask_input]
    )

    hidden_features = layers.Dense(
        256, activation="relu", kernel_regularizer=regularizers.l2(1e-3)
    )(hidden_features)
    hidden_features = layers.Dropout(0.5)(
        hidden_features
    )  # High dropout for smoothness
    return keras.Model(
        inputs=[
            atom_input,
            bond_input,
            pair_indices_input,
            molecule_indicator_input,
            mask_input,
        ],
        outputs=layers.Dense(1, activation="sigmoid")(hidden_features),
    )


# Learning Rate: Slower warmup, lower peak
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0,
    decay_steps=BATCH_SIZE * EPOCHS,
    warmup_target=LEARNING_RATE,
    warmup_steps=BATCH_SIZE * 5,
)

mpnn = MPNNModel(atom_featurizer.dim, bond_featurizer.dim)
mpnn.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=1e-3, global_clipnorm=0.5
    ),
    metrics=[keras.metrics.AUC(name="AUC")],
)
# keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)
```

### Training


```python
history = mpnn.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=1,
    class_weight={0: 2.0, 1: 0.5},
)

# Final Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history["AUC"], label="train AUC", linewidth=2)
plt.plot(history.history["val_AUC"], label="valid AUC", linewidth=2)
plt.grid(True, alpha=0.3)
plt.title("Optimized Stable MPNN Training")
plt.xlabel("Epochs")
plt.ylabel("AUC")
plt.legend()

```

<div class="k-default-codeblock">
```
Epoch 1/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 246ms/step - AUC: 0.5641 - loss: 0.7656 - val_AUC: 0.4880 - val_loss: 0.9153

Epoch 2/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 236ms/step - AUC: 0.6871 - loss: 0.7284 - val_AUC: 0.7429 - val_loss: 0.8283

Epoch 3/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 237ms/step - AUC: 0.8076 - loss: 0.6689 - val_AUC: 0.8287 - val_loss: 0.7392

Epoch 4/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 236ms/step - AUC: 0.8719 - loss: 0.5904 - val_AUC: 0.9333 - val_loss: 0.6054

Epoch 5/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 237ms/step - AUC: 0.8926 - loss: 0.5386 - val_AUC: 0.8830 - val_loss: 0.6064

Epoch 6/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 239ms/step - AUC: 0.9056 - loss: 0.5033 - val_AUC: 0.9173 - val_loss: 0.5284

Epoch 7/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 240ms/step - AUC: 0.9163 - loss: 0.4654 - val_AUC: 0.8994 - val_loss: 0.5299

Epoch 8/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 243ms/step - AUC: 0.9198 - loss: 0.4475 - val_AUC: 0.9263 - val_loss: 0.4681

Epoch 9/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 241ms/step - AUC: 0.9267 - loss: 0.4245 - val_AUC: 0.9186 - val_loss: 0.4927

Epoch 10/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 238ms/step - AUC: 0.9296 - loss: 0.4152 - val_AUC: 0.9443 - val_loss: 0.4165

Epoch 11/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 243ms/step - AUC: 0.9361 - loss: 0.3933 - val_AUC: 0.9201 - val_loss: 0.4750

Epoch 12/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 242ms/step - AUC: 0.9311 - loss: 0.3967 - val_AUC: 0.9368 - val_loss: 0.4119

Epoch 13/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 245ms/step - AUC: 0.9127 - loss: 0.4392 - val_AUC: 0.9419 - val_loss: 0.3968

Epoch 14/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 247ms/step - AUC: 0.9151 - loss: 0.4155 - val_AUC: 0.9111 - val_loss: 0.4468

Epoch 15/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 250ms/step - AUC: 0.9341 - loss: 0.3654 - val_AUC: 0.9274 - val_loss: 0.4094

Epoch 16/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 6s 250ms/step - AUC: 0.9417 - loss: 0.3484 - val_AUC: 0.9383 - val_loss: 0.3747

Epoch 17/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 7s 255ms/step - AUC: 0.9463 - loss: 0.3399 - val_AUC: 0.9195 - val_loss: 0.4401

Epoch 18/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 7s 259ms/step - AUC: 0.9507 - loss: 0.3162 - val_AUC: 0.8753 - val_loss: 0.5406

Epoch 19/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 7s 268ms/step - AUC: 0.9453 - loss: 0.3283 - val_AUC: 0.9455 - val_loss: 0.3459

Epoch 20/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 7s 274ms/step - AUC: 0.9525 - loss: 0.3079 - val_AUC: 0.9239 - val_loss: 0.4015

Epoch 21/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 7s 286ms/step - AUC: 0.9593 - loss: 0.2843 - val_AUC: 0.9296 - val_loss: 0.3761

Epoch 22/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 295ms/step - AUC: 0.9607 - loss: 0.2795 - val_AUC: 0.9674 - val_loss: 0.2956

Epoch 23/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 304ms/step - AUC: 0.9605 - loss: 0.2784 - val_AUC: 0.9175 - val_loss: 0.4183

Epoch 24/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 311ms/step - AUC: 0.9624 - loss: 0.2725 - val_AUC: 0.9511 - val_loss: 0.3246

Epoch 25/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 316ms/step - AUC: 0.9679 - loss: 0.2516 - val_AUC: 0.9609 - val_loss: 0.3052

Epoch 26/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 315ms/step - AUC: 0.9726 - loss: 0.2392 - val_AUC: 0.9547 - val_loss: 0.3139

Epoch 27/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 327ms/step - AUC: 0.9705 - loss: 0.2384 - val_AUC: 0.9446 - val_loss: 0.3331

Epoch 28/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 9s 329ms/step - AUC: 0.9734 - loss: 0.2262 - val_AUC: 0.9561 - val_loss: 0.2988

Epoch 29/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 318ms/step - AUC: 0.9820 - loss: 0.1992 - val_AUC: 0.9451 - val_loss: 0.3309

Epoch 30/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 315ms/step - AUC: 0.9802 - loss: 0.1964 - val_AUC: 0.9634 - val_loss: 0.2908

Epoch 31/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 314ms/step - AUC: 0.9812 - loss: 0.1918 - val_AUC: 0.9439 - val_loss: 0.3579

Epoch 32/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 310ms/step - AUC: 0.9813 - loss: 0.1948 - val_AUC: 0.9590 - val_loss: 0.2953

Epoch 33/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 309ms/step - AUC: 0.9856 - loss: 0.1737 - val_AUC: 0.9599 - val_loss: 0.2968

Epoch 34/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 310ms/step - AUC: 0.9867 - loss: 0.1637 - val_AUC: 0.9576 - val_loss: 0.2894

Epoch 35/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 308ms/step - AUC: 0.9888 - loss: 0.1587 - val_AUC: 0.9631 - val_loss: 0.2650

Epoch 36/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 306ms/step - AUC: 0.9863 - loss: 0.1641 - val_AUC: 0.9563 - val_loss: 0.2904

Epoch 37/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 306ms/step - AUC: 0.9899 - loss: 0.1467 - val_AUC: 0.9584 - val_loss: 0.2911

Epoch 38/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 309ms/step - AUC: 0.9904 - loss: 0.1431 - val_AUC: 0.9537 - val_loss: 0.3060

Epoch 39/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 305ms/step - AUC: 0.9905 - loss: 0.1420 - val_AUC: 0.9585 - val_loss: 0.2752

Epoch 40/40

26/26 ━━━━━━━━━━━━━━━━━━━━ 8s 304ms/step - AUC: 0.9923 - loss: 0.1260 - val_AUC: 0.9634 - val_loss: 0.2735

<matplotlib.legend.Legend at 0x173dd57f0>
```
</div>

![png](/examples/graph/img/mpnn-molecular-graphs/mpnn-molecular-graphs_22_1121.png)
    


### Predicting


```python
molecules = [Chem.MolFromSmiles(df.smiles.values[index]) for index in test_idx]
y_true = [df.p_np.values[index] for index in test_idx]

predictions = mpnn.predict(test_dataset)
y_pred = ops.convert_to_numpy(predictions)[: len(test_idx), 0]

legends = [
    f"y_true/y_pred = {y_true[sample_index]}/{y_pred[sample_index]:.2f}"
    for sample_index in range(len(y_true))
]

MolsToGridImage(molecules, molsPerRow=4, legends=legends)
```

    
<div class="k-default-codeblock">
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 227ms/step
```
</div>

![png](/examples/graph/img/mpnn-molecular-graphs/mpnn-molecular-graphs_24_2.png)
    



---
## Conclusions

In this tutorial, we demonstrated a message passing neural network (MPNN) to
predict blood-brain barrier permeability (BBBP) for a number of different molecules. We
predict blood-brain barrier permeability (BBBP) for a number of
different molecules. We
first had to construct graphs from SMILES, then build a Keras model that could
operate on these graphs, and finally train the model to make the predictions.

Example available on HuggingFace

| Trained Model | Demo |
| :--: | :--: |
| https://huggingface.co/keras-io/MPNN-for-molecular-property-prediction |
| https://huggingface.co/spaces/keras-io/molecular-property-prediction |

---
## Relevant Chapters from Deep Learning with Python
- Chapter 7: A deep dive on Keras
- https://deeplearningwithpython.io/chapters/chapter07_deep-dive-keras
- Chapter 15: Language models and the Transformer
- https://deeplearningwithpython.io/chapters/chapter15_language-models-and-
    the-transformer

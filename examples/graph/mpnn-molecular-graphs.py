"""
Title: Message-passing neural network (MPNN) for molecular property prediction
Author: [akensert](http://github.com/akensert)
Date created: 2021/08/16
Last modified: 2021/12/27
Description: Implementation of an MPNN to predict blood-brain barrier permeability.
Accelerator: GPU
Converted to Keras 3 by: [LakshmiKalaKadali](https://github.com/LakshmiKalaKadali)
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
pip -q install rdkit
```

And for easy and efficient reading of csv files and visualization, the below needs to be
installed:

```
pip -q install pandas
pip -q install Pillow
pip -q install matplotlib
pip -q install pydot
pip -q install graphviz
```
"""

"""
### Import packages
"""

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

"""
## Dataset

Information about the dataset can be found in
[A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling](https://pubs.acs.org/doi/10.1021/ci300124c)
and [MoleculeNet: A Benchmark for Molecular Machine Learning](https://arxiv.org/abs/1703.00564).
The dataset will be downloaded from [MoleculeNet.org](https://moleculenet.org/datasets-1).

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
we'll define two classes: `AtomFeaturizer` and `BondFeaturizer` respectively.

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
        "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
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


"""
### Generate graphs

Before generating complete graphs from SMILES, we need to implement the following functions:

1. `molecule_from_smiles`: This takes a SMILES string as input and returns an RDKit molecule object. This process remains handled by RDKit on the CPU.
2. `smiles_to_graph`: This takes a SMILES string and returns a graph represented as a four-tuple: (atom_features, bond_features, pair_indices, mask).
The original implementation utilized tf.RaggedTensor, which is exclusive to TensorFlow. To remain backend-agnostic and support JAX and PyTorch, we now use fixed-size buffers (MAX_ATOMS and MAX_BONDS). We also introduce a mask—a boolean array that allows the model to distinguish between valid chemical data and zero-padding.
Finally, implemented a pre-featurization step. Instead of featurizing during the training loop (which creates a CPU bottleneck), we process all SMILES once and store them in a list of NumPy arrays. This allows the GPU backends to run at 100% efficiency.
"""


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
    Converts SMILES to a graph with fixed-size buffers for Keras 3 compatibility.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # Pre-allocate fixed buffers for static shapes (required for JAX/Torch)
    a_feat = np.zeros((MAX_ATOMS, atom_featurizer.dim), dtype="float32")
    b_feat = np.zeros((MAX_BONDS, bond_featurizer.dim), dtype="float32")
    p_idx = np.zeros((MAX_BONDS, 2), dtype="int32")
    mask = np.zeros((MAX_ATOMS,), dtype="float32")

    atoms = mol.GetAtoms()
    for i, atom in enumerate(atoms):
        if i >= MAX_ATOMS:
            break
        a_feat[i] = atom_featurizer.encode(atom)
        mask[i] = 1.0

    b_count = 0
    for i, atom in enumerate(atoms):
        if i >= MAX_ATOMS or b_count >= MAX_BONDS:
            break
        # Add self-loop (standard in MPNN)
        p_idx[b_count] = [i, i]
        b_feat[b_count] = bond_featurizer.encode(None)
        b_count += 1

        for nb in atom.GetNeighbors():
            j = nb.GetIdx()
            if j >= MAX_ATOMS or b_count >= MAX_BONDS:
                continue
            p_idx[b_count] = [i, j]
            b_feat[b_count] = bond_featurizer.encode(mol.GetBondBetweenAtoms(i, j))
            b_count += 1

    return a_feat, b_feat, p_idx, mask


csv_path = keras.utils.get_file(
    "BBBP.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
)
df = pd.read_csv(csv_path, usecols=[1, 2, 3])

# Pre-featurize the entire dataset once to remove the RDKit bottleneck during training
print("Pre-featurizing Dataset...")
processed_data = []
for s in tqdm(df.smiles.values):
    graph = smiles_to_graph(s)
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

print("Pre-featurizing Dataset...")
processed_data = [
    smiles_to_graph(s)
    or (
        np.zeros((MAX_ATOMS, atom_featurizer.dim)),
        np.zeros((MAX_BONDS, bond_featurizer.dim)),
        np.zeros((MAX_BONDS, 2), dtype="int32"),
        np.zeros((MAX_ATOMS,)),
    )
    for s in tqdm(df.smiles.values)
]


"""
### Test the functions

We can now inspect a sample molecule and its corresponding graph representation. Note that the output shapes are now constant (e.g., 70 atoms and 150 bonds), ensuring compatibility across all Keras 3 backends.
"""

sample_idx = 100
print(
    f"Name:\t{df.name[sample_idx]}\nSMILES:\t{df.smiles[sample_idx]}\nBBBP:\t{df.p_np[sample_idx]}"
)

molecule = molecule_from_smiles(df.smiles.values[sample_idx])
print("Molecule object created successfully.")
molecule
"""
"""

# Convert to graph and check constant shapes
a, b, p, m = smiles_to_graph(df.smiles.values[sample_idx])
print("Graph (including self-loops and padding):")
print(f"\tatom features\t {a.shape}")
print(f"\tbond features\t {b.shape}")
print(f"\tpair indices \t {p.shape}")
"""
### Data Loading with PyDataset

In this tutorial, the MPNN implementation takes a single graph as input per iteration. 
To process a batch of molecules, we merge them into a single global graph (also known as a disjoint graph). 
This global graph is a disconnected structure where each molecule (subgraph) is separated from the others.
"""


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
        a = np.zeros((self.batch_size, MAX_ATOMS, atom_featurizer.dim), dtype="float32")
        b = np.zeros((self.batch_size, MAX_BONDS, bond_featurizer.dim), dtype="float32")
        p = np.zeros((self.batch_size, MAX_BONDS, 2), dtype="int32")
        m = np.zeros((self.batch_size, MAX_ATOMS), dtype="float32")
        y = np.zeros((self.batch_size, 1), dtype="float32")

        for i, real_idx in enumerate(batch_idx):
            a[i], b[i], p[i], m[i] = self.data[real_idx]
            y[i] = self.labels[real_idx]
            p[i] += i * MAX_ATOMS

        return {
            "atom_features": ops.convert_to_tensor(a.reshape(-1, atom_featurizer.dim)),
            "bond_features": ops.convert_to_tensor(b.reshape(-1, bond_featurizer.dim)),
            "pair_indices": ops.convert_to_tensor(p.reshape(-1, 2)),
            "molecule_indicator": ops.convert_to_tensor(
                np.repeat(np.arange(self.batch_size), MAX_ATOMS), dtype="int32"
            ),
            "mask": ops.convert_to_tensor(m.reshape(-1)),
        }, ops.convert_to_tensor(y)

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
    [processed_data[i] for i in train_idx],
    df.p_np.values[train_idx],
    batch_size=BATCH_SIZE,
    shuffle=True,
)

valid_dataset = MPNNDataset(
    [processed_data[i] for i in val_idx], df.p_np.values[val_idx], batch_size=BATCH_SIZE
)

# Instantiate the test dataset
test_dataset = MPNNDataset(
    [processed_data[i] for i in test_idx],
    df.p_np.values[test_idx],
    batch_size=BATCH_SIZE,
)

print(
    f"Dataset Split: Train={len(train_idx)}, Valid={len(val_idx)}, Test={len(test_idx)}"
)


"""
## Model

The MPNN model can take on various shapes and forms. In this tutorial, we will implement an
MPNN based on the original paper
[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) and
[DeepChem's MPNNModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#mpnnmodel).
The MPNN of this tutorial consists of three stages: message passing, readout and
classification.


### Message passing

The Message Passing Neural Network (MPNN) architecture implemented in this tutorial consists of three stages: message passing, readout, and classification. The message passing step is the core of the model, enabling information to flow through the molecular graph. It consists of two main components:

1. The *edge network*, which passes messages from 1-hop neighbors `w_{i}` of `v`
to `v`, based on the edge features between them (`e_{vw_{i}}`),
resulting in an updated node (state) `v'`. `w_{i}` denotes the `i:th` neighbor of
`v`.

2. The *gated recurrent unit* (GRU), which takes as input the most recent node state
and updates it based on previous node states. In
other words, the most recent node state serves as the input to the GRU, while the previous
node states are incorporated within the memory state of the GRU. This allows information
to travel from one node state (e.g., `v`) to another (e.g., `v''`).

Importantly, step (1) and (2) are repeated for `k steps`, and where at each step `1...k`,
the radius (or number of hops) of aggregated information from `v` increases by 1.
"""


class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim, self.bond_dim = input_shape[0][-1], input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim**2), initializer="glorot_uniform"
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
            num_segments=BATCH_SIZE * MAX_ATOMS,
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
            atom_feat, [(0, 0), (0, max(0, self.units - ops.shape(atom_feat)[-1]))]
        )
        for _ in range(self.steps):
            m = self.edge_net([atom_feat, bond_feat, pair_idx])
            atom_feat, _ = self.gru(m, atom_feat)
            atom_feat = self.norm(atom_feat)  # Normalize every step
        return atom_feat


"""
### Readout

When the message passing procedure ends, the k-step-aggregated node states are to be partitioned
into subgraphs (corresponding to each molecule in the batch) and subsequently
reduced to graph-level embeddings. In the
[original paper](https://arxiv.org/abs/1704.01212), a
[set-to-set layer](https://arxiv.org/abs/1511.06391) was used for this purpose.
In this tutorial, we utilize a Gated Readout combined with Hybrid Pooling (Mean and Max).
This approach is highly stable and fully compatible with JAX, PyTorch, and TensorFlow. The process works as follows:
Gating Mechanism: Each node state passes through a learned gating function (using sigmoid and tanh activations). This allows the model to "decide" which atoms are most important for the molecular property being predicted.
Masking: We use the mask generated in our data pipeline to ensure that padded (zero) atoms do not contribute to the final graph embedding.
Hybrid Segment Pooling: Instead of physically partitioning the tensors, we use the molecule_indicator (batch index) to logically group atoms. We calculate both the Mean and the Max of the node states for each molecule.
Concatenation: The mean and max features are concatenated to form a robust, fixed-size graph-level representation.
"""


class GatedReadout(layers.Layer):
    """A more stable readout using both Mean and Max pooling."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.gate = layers.Dense(embed_dim, activation="sigmoid")
        self.feat = layers.Dense(embed_dim, activation="tanh")

    def call(self, inputs):
        nodes, indicator, mask = inputs
        mask = ops.expand_dims(mask, -1)

        # Gated logic: atoms "decide" how much they contribute
        gated_x = self.gate(nodes) * self.feat(nodes)
        gated_x = gated_x * mask

        # Combined Mean and Max pooling for robustness
        x_mean = ops.segment_sum(
            gated_x, ops.cast(indicator, "int32"), num_segments=BATCH_SIZE
        ) / ops.maximum(
            ops.segment_sum(
                mask, ops.cast(indicator, "int32"), num_segments=BATCH_SIZE
            ),
            1e-6,
        )
        x_max = ops.segment_max(
            gated_x, ops.cast(indicator, "int32"), num_segments=BATCH_SIZE
        )

        return ops.concatenate([x_mean, x_max], axis=-1)


"""
### Message Passing Neural Network (MPNN)

It is now time to complete the MPNN model. In addition to the message passing
and readout, a two-layer classification network will be implemented to make
predictions of BBBP.
"""


def MPNNModel(atom_dim, bond_dim):
    a_in = layers.Input(shape=(atom_dim,), name="atom_features")
    b_in = layers.Input(shape=(bond_dim,), name="bond_features")
    p_in = layers.Input(shape=(2,), dtype="int32", name="pair_indices")
    i_in = layers.Input(shape=(), dtype="int32", name="molecule_indicator")
    m_in = layers.Input(shape=(), name="mask")

    x = MessagePassing(64, steps=4)([a_in, b_in, p_in])
    x = GatedReadout(64)([x, i_in, m_in])

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(
        x
    )
    x = layers.Dropout(0.5)(x)  # High dropout for smoothness
    return keras.Model(
        inputs=[a_in, b_in, p_in, i_in, m_in],
        outputs=layers.Dense(1, activation="sigmoid")(x),
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

keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)

"""
### Training
"""

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


"""
### Predicting
"""

molecules = [Chem.MolFromSmiles(df.smiles.values[index]) for index in test_idx]
y_true = [df.p_np.values[index] for index in test_idx]

predictions = mpnn.predict(test_dataset)
y_pred = ops.convert_to_numpy(predictions)[: len(test_idx), 0]

legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f}" for i in range(len(y_true))]

MolsToGridImage(molecules, molsPerRow=4, legends=legends)

"""
## Conclusions

In this tutorial, we demonstrated a message passing neural network (MPNN) to
predict blood-brain barrier permeability (BBBP) for a number of different molecules. We
first had to construct graphs from SMILES, then build a Keras model that could
operate on these graphs, and finally train the model to make the predictions.

Example available on HuggingFace

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-mpnn%20molecular%20graphs-black.svg)](https://huggingface.co/keras-io/MPNN-for-molecular-property-prediction) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-mpnn%20molecular%20graphs-black.svg)](https://huggingface.co/spaces/keras-io/molecular-property-prediction) |

## Relevant Chapters from Deep Learning with Python
- [Chapter 7: A deep dive on Keras](https://deeplearningwithpython.io/chapters/chapter07_deep-dive-keras)
- [Chapter 15: Language models and the Transformer](https://deeplearningwithpython.io/chapters/chapter15_language-models-and-the-transformer)
"""

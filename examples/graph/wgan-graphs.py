"""
Title: WGAN-GP with R-GCN for the generation fo small molecular graphs
Author: [akensert](https://github.com/akensert)
Date created: 2021/06/27
Last modified: 2021/06/27
Description: Complete implementation of WGAN-GP and R-GCN to generate novel molecules.
"""

"""
## References

The implementation in this tutorial is based on/inspired by the [MolGAN
paper](https://arxiv.org/abs/1805.11973) and DeepChem's [Basic
MolGAN](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#basicmolganmod
el).

## Further reading
Recent implementations of generative models for molecular graphs also include
[Mol-CycleGAN](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0404-1),
[GraphVAE](https://arxiv.org/abs/1802.03480) and
[JT-VAE](https://arxiv.org/abs/1802.04364). For more information on generative
adverserial networks, see [GAN](https://arxiv.org/abs/1406.2661),
[WGAN](https://arxiv.org/abs/1701.07875) and [WGAN-GP](https://arxiv.org/abs/1704.00028).

"""

"""
## Setup

### Install RDKit via conda in Google Colab

[RDKit](https://www.rdkit.org/) is a collection of cheminformatics and machine-learning
software written in C++ and Python. In this tutorial, RDKit is used to conviently and
efficiently transform
[SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) to
molecule objects, and then from those obtain sets of atoms and bonds. Installation
normally takes about 1-2 minutes.

RDKit is commonly installed via [Conda](https://www.rdkit.org/docs/Install.html). However,
thanks to [kuelumbus'](https://github.com/kuelumbus/)
[rdkit_platform_wheels](https://github.com/kuelumbus/rdkit_platform_wheels), rdkit
can now (for the sake of this tutorial) be installed easily via pip.
"""

"""shell
pip -q install Pillow
pip -q install rdkit-pypi

"""

"""
### Import packages
"""

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

RDLogger.DisableLog("rdApp.*")

"""
## Dataset

The dataset used in this tutorial is a [quantum mechanics
dataset](http://quantum-machine.org/datasets/) (QM9), obtained from
[MoleculeNet](http://moleculenet.ai/datasets-1). Although many feature and label columns
come with the dataset, we'll only focus on the
[SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)
column ("smiles"). The QM9 dataset is a good first dataset to work with for generating
graphs, as the maximum number of \[heavy\] atoms found in a molecule is only nine.
"""

csv_path = tf.keras.utils.get_file(
    "qm9.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
)

data = []
with open(csv_path, "r") as f:
    for line in f.readlines()[1:]:
        data.append(line.split(",")[1])

# Let's look at a molecule of the dataset
smiles = data[1000]
print("SMILES:", smiles)
mol = Chem.MolFromSmiles(smiles)
print("Num heavy atoms:", mol.GetNumHeavyAtoms())
mol

"""
### Define helper functions
These helper functions will help convert SMILES to graphs and graphs to molecule objects.

**Representing a molecular graph**. Molecules can naturally be expressed as undirected
graphs `G = (V, E)`, where `V` is a set of vertices (atoms), and `E` a set of edges
(bonds). As for this implementation, each graph (molecule) will be represented as an
adjacency tensor `A`, which encodes existence/non-existence of atom-pairs with their
one-hot encoded bond types stretching an extra dimension, and a feature tensor `H`, which
for each atom, one-hot encodes its atom type.

"""

atom_mapping = {
    "C": 0,
    0: "C",
    "N": 1,
    1: "N",
    "O": 2,
    2: "O",
    "F": 3,
    3: "F",
}

bond_mapping = {
    "SINGLE": 0,
    0: Chem.BondType.SINGLE,
    "DOUBLE": 1,
    1: Chem.BondType.DOUBLE,
    "TRIPLE": 2,
    2: Chem.BondType.TRIPLE,
    "AROMATIC": 3,
    3: Chem.BondType.AROMATIC,
}

NUM_ATOMS = 9  # Maximum number of atoms
ATOM_DIM = 4 + 1  # Number of atom types
BOND_DIM = 4 + 1  # Number of bond types
LATENT_DIM = 64  # Size of the latent space


def smiles_to_graph(smiles):
    # Converts SMILES to molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Initialize adjacency and feature tensor
    A = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    H = np.zeros((NUM_ATOMS, ATOM_DIM), "float32")

    # loop over each atom in mol
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        H[i] = np.eye(ATOM_DIM)[atom_type]
        # loop over one-hop neighbors
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            A[bond_type_idx, [i, j], [j, i]] = 1

    # Where no bond, add 1 to last channel (indicating "non-bond")
    # Notice: channels-first
    A[-1, np.sum(A, axis=0) == 0] = 1

    # Where no atom, add 1 to last column (indicating "non-atom")
    H[np.where(np.sum(H, axis=1) == 0)[0], -1] = 1

    return A, H


def graph_to_mol(graph):
    # Unpack graph
    A, H = graph

    # RWMol is a molecule object intended to be edited
    mol = Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where(
        (np.argmax(H, axis=1) != ATOM_DIM - 1) & (np.sum(A[:-1], axis=(0, 1)) != 0)
    )[0]
    H = H[keep_idx]
    A = A[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to mol
    for atom_type_idx in np.argmax(H, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        _ = mol.AddAtom(atom)

    # Add bonds between atoms in mol; based on the upper triangles
    # of the [symmetric] adjacency tensor
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(A) == 1)
    for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        mol.AddBond(int(atom_i), int(atom_j), bond_type)

    # Sanitize the molecule; for more information on sanitization, see
    # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = Chem.SanitizeMol(mol, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return mol


# Test helper functions
graph_to_mol(smiles_to_graph(smiles))

"""
### Generate training set

To save training time, we'll only use a tenth of the QM9 dataset.
"""

adjacency_tensor, feature_tensor = [], []
for smiles in data[::10]:
    A, X = smiles_to_graph(smiles)
    adjacency_tensor.append(A)
    feature_tensor.append(X)

adjacency_tensor = np.array(adjacency_tensor)
feature_tensor = np.array(feature_tensor)

print("A.shape =", adjacency_tensor.shape)
print("H.shape =", feature_tensor.shape)

"""
## Model


### Generator
"""


def GraphGenerator(
    dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape,
):
    z = keras.layers.Input(shape=(LATENT_DIM,))
    # Propagate through one or more densely connected layers
    x = z
    for units in dense_units:
        x = keras.layers.Dense(units, activation="tanh")(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    # Map outputs of previous layers (x) to [continuous] adjacency tensors (x_A)
    x_A = keras.layers.Dense(tf.math.reduce_prod(adjacency_shape))(x)
    x_A = keras.layers.Reshape(adjacency_shape)(x_A)
    # Symmetrify tensors in the last two dimensions
    x_A = keras.layers.Lambda(lambda x: (x + tf.transpose(x, (0, 1, 3, 2))) / 2)(x_A)
    x_A = keras.layers.Softmax(axis=1)(x_A)

    # Map outputs of previous layers (x) to [continuous] feature tensors (x_H)
    x_H = keras.layers.Dense(tf.math.reduce_prod(feature_shape))(x)
    x_H = keras.layers.Reshape(feature_shape)(x_H)
    x_H = keras.layers.Softmax(axis=2)(x_H)

    return keras.Model(inputs=z, outputs=[x_A, x_H], name="Generator")


generator = GraphGenerator(
    dense_units=[128, 256, 512],
    dropout_rate=0.2,
    latent_dim=LATENT_DIM,
    adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
    feature_shape=(NUM_ATOMS, ATOM_DIM),
)
generator.summary()

"""
## Discriminator


**Graph convolutional layer**. The [relational graph convolutional
layers](https://arxiv.org/abs/1703.06103) implements non-linearly transformed
neighborhood aggregations. We can define these layers as follows:

`H^{l+1} = σ(D^{-1} @ A @ H^{l+1} @ W^{l})`


Where `σ` denotes the non-linear transformation (commonly a ReLU activation), `A` the
adjacency tensor, `H^{l}` the feature tensor at the `l:th` layer, `D^{-1}` the inverse
diagonal degree tensor of `A`, and `W^{l}` the trainable weight tensor at the `l:th`
layer. Specifically, for each bond type (relation), the degree tensor expresses, in the
diagonal, the number of bonds attached to each atom. Notice, in this tutorial `D^{-1}` is
omitted, for two reasons: (1) it's not obvious how to apply this normalization on the
continuous adjacency tensors (generated by the generator), and (2) the performance of the
WGAN without normalization seems to work just fine. Furthermore, in contrast to the
[original paper](https://arxiv.org/abs/1703.06103), no self-loop is defined, as we don't
want to train the generator to predict "self-bonding".



"""


class RelationalGraphConvLayer(keras.layers.Layer):
    def __init__(
        self,
        units=128,
        activation="relu",
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        super(RelationalGraphConvLayer, self).__init__(**kwargs)

        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]

        self.W = self.add_weight(
            shape=(bond_dim, atom_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="W",
            dtype=tf.float32,
        )

        if self.use_bias:
            self.b = self.add_weight(
                shape=(bond_dim, 1, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name="b",
                dtype=tf.float32,
            )

        self.built = True

    def call(self, inputs, training=False):
        A, H = inputs
        # Aggregate information from neighbors
        X = tf.matmul(A, H[:, None, :, :])
        # Apply linear transformation
        X = tf.matmul(X, self.W)
        if self.use_bias:
            X += self.b
        # Reduce bond types dim
        X = tf.reduce_sum(X, axis=1)
        # Apply non-linear transformation
        return self.activation(X)


def GraphDiscriminator(
    gconv_units, dense_units, dropout_rate, adjacency_shape, feature_shape
):

    A = keras.layers.Input(shape=adjacency_shape)
    H0 = keras.layers.Input(shape=feature_shape)

    # Propagate through one or more graph convolutional layers
    H = H0
    for units in gconv_units:
        H = RelationalGraphConvLayer(units)([A, H])

    # Reduce 2-D representation of molecule to 1-D
    x = keras.layers.GlobalAveragePooling1D()(H)

    # Propagate through one or more densely connected layers
    for units in dense_units:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    # For each molecule, output a single scalar value expressing the
    # "realness" of the molecule
    out = keras.layers.Dense(1, dtype="float32")(x)

    return keras.Model(inputs=[A, H0], outputs=out)


discriminator = GraphDiscriminator(
    gconv_units=[128, 128, 128, 128],
    dense_units=[512, 512],
    dropout_rate=0.2,
    adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
    feature_shape=(NUM_ATOMS, ATOM_DIM),
)
discriminator.summary()

"""
### WGAN-GP
"""


class GraphWGAN(keras.Model):
    def __init__(
        self,
        generator,
        discriminator,
        discriminator_steps=1,
        generator_steps=1,
        gp_weight=10,
        **kwargs
    ):
        super(GraphWGAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.gp_weight = gp_weight
        self.latent_dim = self.generator.input_shape[-1]

    def compile(self, optimizer_generator, optimizer_discriminator, **kwargs):
        super(GraphWGAN, self).compile(**kwargs)
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.metric_generator = keras.metrics.Mean(name="loss_gen")
        self.metric_discriminator = keras.metrics.Mean(name="loss_dis")

    def train_step(self, inputs):

        if isinstance(inputs[0], tuple):
            inputs = inputs[0]

        G_real = inputs

        self.batch_size = tf.shape(inputs[0])[0]

        # Train the discriminator for one or more steps
        for _ in range(self.discriminator_steps):
            z = tf.random.normal((self.batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                G_gen = self.generator(z, training=True)
                loss = self._loss_discriminator(G_real, G_gen)

            grads = tape.gradient(loss, self.discriminator.trainable_weights)
            self.optimizer_discriminator.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )
            self.metric_discriminator.update_state(loss)

        # Train the generator for one or more steps
        for _ in range(self.generator_steps):
            z = tf.random.normal((self.batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            G_gen = self.generator(z, training=True)
            loss = self._loss_generator(G_gen)

            grads = tape.gradient(loss, self.generator.trainable_weights)
            self.optimizer_generator.apply_gradients(
                zip(grads, self.generator.trainable_weights)
            )
            self.metric_generator.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def _loss_discriminator(self, G_real, G_gen):
        logits_real = self.discriminator(G_real, training=True)
        logits_gen = self.discriminator(G_gen, training=True)
        loss = tf.reduce_mean(logits_gen) - tf.reduce_mean(logits_real)
        return loss + self._gradient_penalty(G_real, G_gen) * self.gp_weight

    def _loss_generator(self, G_gen):
        logits_gen = self.discriminator(G_gen, training=True)
        return -tf.reduce_mean(logits_gen)

    def _gradient_penalty(self, G_real, G_gen):
        # Unpack graphs
        A_real, H_real = G_real
        A_gen, H_gen = G_gen

        # Generate interpolated graphs (A_interp and H_interp)
        alpha = tf.random.uniform([self.batch_size])
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
        A_interp = (A_real * alpha) + (1 - alpha) * A_gen
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1))
        H_interp = (H_real * alpha) + (1 - alpha) * H_gen

        # Compute the logits of interpolated graphs
        with tf.GradientTape() as tape:
            tape.watch(A_interp)
            tape.watch(H_interp)
            logits = self.discriminator([A_interp, H_interp], training=True)

        # Compute the gradients with respect to the interpolated graphs
        grads = tape.gradient(logits, [A_interp, H_interp])
        # Compute the gradient penalty
        grads_A_penalty = (1 - tf.norm(grads[0], axis=1)) ** 2
        grads_H_penalty = (1 - tf.norm(grads[1], axis=2)) ** 2
        return tf.reduce_mean(
            tf.reduce_mean(grads_A_penalty, axis=(-2, -1)) +
            tf.reduce_mean(grads_H_penalty, axis=(-1))
        )


"""
## Train the model

To save time (if run on a CPU), we'll only train the model for 10 epochs.
"""

wgan = GraphWGAN(generator, discriminator, discriminator_steps=1)

wgan.compile(
    optimizer_generator=keras.optimizers.Adam(5e-4),
    optimizer_discriminator=keras.optimizers.Adam(5e-4),
)

wgan.fit([adjacency_tensor, feature_tensor], epochs=10, batch_size=16)

"""
## Sample novel molecules with the generator
"""


def sample(generator, batch_size):
    z = tf.random.normal((batch_size, LATENT_DIM))
    graph = generator(z)
    A = tf.argmax(graph[0], axis=1)
    A = tf.one_hot(A, depth=BOND_DIM, axis=1)
    A = tf.linalg.set_diag(A, tf.zeros(tf.shape(A)[:-1]))  # Remove potential self-loops
    X = tf.argmax(graph[1], axis=2)
    X = tf.one_hot(X, depth=ATOM_DIM, axis=2)
    return [graph_to_mol([A[i].numpy(), X[i].numpy()]) for i in range(batch_size)]


mols = sample(wgan.generator, batch_size=32)

MolsToGridImage(
    [m for m in mols if m is not None][:25], molsPerRow=5, subImgSize=(150, 150)
)

"""
## Concluding thoughts

**Inspecting the results**. Ten epochs of training seemed enough to generate some decent
looking molecules! Notice, in contrast to the [MolGAN paper](https://arxiv.org/abs/1805.11973),
the uniqueness of the generated molecules in this tutorial seems really high, which is great!

**What we've learned, and prospects**. In this tutorial, a generative model for molecular
graphs was succesfully implemented, which allowed us to generate novel molecules. In the
future, it would be interesting to implement generative models that can modify existing
molecules (for instance, to optimize solubility or protein-binding of an existing
molecule). For that however, a reconstruction loss would likely be needed, which is
tricky to implement as there's no easy and obvious way to compute similarity between two
molecular graphs.
"""

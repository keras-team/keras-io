# Drug Molecule Generation with VAE

**Author:** [Victor Basu](https://www.linkedin.com/in/victor-basu-520958147)<br>
**Date created:** 2022/03/10<br>
**Last modified:** 2024/12/05<br>
**Description:** Implementing a Convolutional Variational AutoEncoder (VAE) for Drug Discovery.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/molecule_generation.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/molecule_generation.py)



---
## Introduction

In this example, we use a Variational Autoencoder to generate molecules for drug discovery.
We use the research papers
[Automatic chemical design using a data-driven continuous representation of molecules](https://arxiv.org/abs/1610.02415)
and [MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973)
as a reference.

The model described in the paper **Automatic chemical design using a data-driven
continuous representation of molecules** generates new molecules via efficient exploration
of open-ended spaces of chemical compounds. The model consists of
three components: Encoder, Decoder and Predictor. The Encoder converts the discrete
representation of a molecule into a real-valued continuous vector, and the Decoder
converts these continuous vectors back to discrete molecule representations. The
Predictor estimates chemical properties from the latent continuous vector representation
of the molecule. Continuous representations allow the use of gradient-based
optimization to efficiently guide the search for optimized functional compounds.

![intro](https://bit.ly/3CtPMzM)

**Figure (a)** - A diagram of the autoencoder used for molecule design, including the
joint property prediction model. Starting from a discrete molecule representation, such
as a SMILES string, the encoder network converts each molecule into a vector in the
latent space, which is effectively a continuous molecule representation. Given a point
in the latent space, the decoder network produces a corresponding SMILES string. A
multilayer perceptron network estimates the value of target properties associated with
each molecule.

**Figure (b)** - Gradient-based optimization in continuous latent space. After training a
surrogate model `f(z)` to predict the properties of molecules based on their latent
representation `z`, we can optimize `f(z)` with respect to `z` to find new latent
representations expected to match specific desired properties. These new latent
representations can then be decoded into SMILES strings, at which point their properties
can be tested empirically.

For an explanation and implementation of MolGAN, please refer to the Keras Example
[**WGAN-GP with R-GCN for the generation of small molecular graphs**](https://bit.ly/3pU6zXK) by
Alexander Kensert. Many of the functions used in the present example are from the above Keras example.

---
## Setup

RDKit is an open source toolkit for cheminformatics and machine learning. This toolkit come in handy
if one is into drug discovery domain. In this example, RDKit is used to conveniently
and efficiently transform SMILES to molecule objects, and then from those obtain sets of atoms
and bonds.

Quoting from
[WGAN-GP with R-GCN for the generation of small molecular graphs](https://keras.io/examples/generative/wgan-graphs/)):

**"SMILES expresses the structure of a given molecule in the form of an ASCII string.
The SMILES string is a compact encoding which, for smaller molecules, is relatively human-readable.
Encoding molecules as a string both alleviates and facilitates database and/or web searching
of a given molecule. RDKit uses algorithms to accurately transform a given SMILES to
a molecule object, which can then be used to compute a great number of molecular properties/features."**


```python
!pip -q install rdkit-pypi==2021.9.4
```

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import ast

import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras import layers
from keras import ops

import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType
from rdkit.Chem.Draw import MolsToGridImage

RDLogger.DisableLog("rdApp.*")
```
<div class="k-default-codeblock">
```
[33mWARNING: You are using pip version 21.2.4; however, version 24.3.1 is available.
You should consider upgrading via the '/Users/chunduriv/generative_mg/env/bin/python3 -m pip install --upgrade pip' command.

```
</div>
---
## Dataset

We use the [**ZINC – A Free Database of Commercially Available Compounds for
Virtual Screening**](https://bit.ly/3IVBI4x) dataset. The dataset comes with molecule
formula in SMILE representation along with their respective molecular properties such as
**logP** (water–octanal partition coefficient), **SAS** (synthetic
accessibility score) and **QED** (Qualitative Estimate of Drug-likeness).


```python
csv_path = keras.utils.get_file(
    "250k_rndm_zinc_drugs_clean_3.csv",
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
)

df = pd.read_csv(csv_path)
df["smiles"] = df["smiles"].apply(lambda s: s.replace("\n", ""))
df.head()
```

<div class="k-default-codeblock">
```
Downloading data from https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv

```
</div>
    
        0/22606589 [37m━━━━━━━━━━━━━━━━━━━━  0s 0s/step

<div class="k-default-codeblock">
```

```
</div>
  1499136/22606589 ━[37m━━━━━━━━━━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
  2875392/22606589 ━━[37m━━━━━━━━━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
  5619712/22606589 ━━━━[37m━━━━━━━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
  7774208/22606589 ━━━━━━[37m━━━━━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
  9904128/22606589 ━━━━━━━━[37m━━━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 11804672/22606589 ━━━━━━━━━━[37m━━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 13180928/22606589 ━━━━━━━━━━━[37m━━━━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 16015360/22606589 ━━━━━━━━━━━━━━[37m━━━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 18374656/22606589 ━━━━━━━━━━━━━━━━[37m━━━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 20602880/22606589 ━━━━━━━━━━━━━━━━━━[37m━━  0s 0us/step

<div class="k-default-codeblock">
```

```
</div>
 22606589/22606589 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step





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
      <th>smiles</th>
      <th>logP</th>
      <th>qed</th>
      <th>SAS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1</td>
      <td>5.05060</td>
      <td>0.702012</td>
      <td>2.084095</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1</td>
      <td>3.11370</td>
      <td>0.928975</td>
      <td>3.432004</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)...</td>
      <td>4.96778</td>
      <td>0.599682</td>
      <td>2.470633</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c...</td>
      <td>4.00022</td>
      <td>0.690944</td>
      <td>2.822753</td>
    </tr>
    <tr>
      <th>4</th>
      <td>N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])[C@H](C#...</td>
      <td>3.60956</td>
      <td>0.789027</td>
      <td>4.035182</td>
    </tr>
  </tbody>
</table>
</div>



---
## Hyperparameters


```python
SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br"]'

bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
bond_mapping.update(
    {0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC}
)
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)

MAX_MOLSIZE = max(df["smiles"].str.len())
SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
atom_mapping = dict(SMILE_to_index)
atom_mapping.update(index_to_SMILE)

BATCH_SIZE = 100
EPOCHS = 10

VAE_LR = 5e-4
NUM_ATOMS = 120  # Maximum number of atoms

ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
BOND_DIM = 4 + 1  # Number of bond types
LATENT_DIM = 435  # Size of the latent space


def smiles_to_graph(smiles):
    # Converts SMILES to molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Initialize adjacency and feature tensor
    adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    features = np.zeros((NUM_ATOMS, ATOM_DIM), "float32")

    # loop over each atom in molecule
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        features[i] = np.eye(ATOM_DIM)[atom_type]
        # loop over one-hop neighbors
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, [i, j], [j, i]] = 1

    # Where no bond, add 1 to last channel (indicating "non-bond")
    # Notice: channels-first
    adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1

    # Where no atom, add 1 to last column (indicating "non-atom")
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

    return adjacency, features


def graph_to_molecule(graph):
    # Unpack graph
    adjacency, features = graph

    # RWMol is a molecule object intended to be edited
    molecule = Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where(
        (np.argmax(features, axis=1) != ATOM_DIM - 1)
        & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
    )[0]
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule
    for atom_type_idx in np.argmax(features, axis=1):
        atom = Chem.Atom(atom_mapping[atom_type_idx])
        _ = molecule.AddAtom(atom)

    # Add bonds between atoms in molecule; based on the upper triangles
    # of the [symmetric] adjacency tensor
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
    for bond_ij, atom_i, atom_j in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atom_i), int(atom_j), bond_type)

    # Sanitize the molecule; for more information on sanitization, see
    # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule

```

---
##  Generate training set


```python
train_df = df.sample(frac=0.75, random_state=42)  # random state is a seed value
train_df.reset_index(drop=True, inplace=True)

adjacency_tensor, feature_tensor, qed_tensor = [], [], []
for idx in range(8000):
    adjacency, features = smiles_to_graph(train_df.loc[idx]["smiles"])
    qed = train_df.loc[idx]["qed"]
    adjacency_tensor.append(adjacency)
    feature_tensor.append(features)
    qed_tensor.append(qed)

adjacency_tensor = np.array(adjacency_tensor)
feature_tensor = np.array(feature_tensor)
qed_tensor = np.array(qed_tensor)


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
        super().__init__(**kwargs)

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

        self.kernel = self.add_weight(
            shape=(bond_dim, atom_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="W",
            dtype="float32",
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(bond_dim, 1, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name="b",
                dtype="float32",
            )

        self.built = True

    def call(self, inputs, training=False):
        adjacency, features = inputs
        # Aggregate information from neighbors
        x = ops.matmul(adjacency, features[:, None])
        # Apply linear transformation
        x = ops.matmul(x, self.kernel)
        if self.use_bias:
            x += self.bias
        # Reduce bond types dim
        x_reduced = ops.sum(x, axis=1)
        # Apply non-linear transformation
        return self.activation(x_reduced)

```

---
## Build the Encoder and Decoder

The Encoder takes as input a molecule's graph adjacency matrix and feature matrix.
These features are processed via a Graph Convolution layer, then are flattened and
processed by several Dense layers to derive `z_mean` and `log_var`, the
latent-space representation of the molecule.

**Graph Convolution layer**: The relational graph convolution layer implements
non-linearly transformed neighbourhood aggregations. We can define these layers as
follows:

`H_hat**(l+1) = σ(D_hat**(-1) * A_hat * H_hat**(l+1) * W**(l))`

Where `σ` denotes the non-linear transformation (commonly a ReLU activation), `A` the
adjacency tensor, `H_hat**(l)` the feature tensor at the `l-th` layer, `D_hat**(-1)` the
inverse diagonal degree tensor of `A_hat`, and `W_hat**(l)` the trainable weight tensor
at the `l-th` layer. Specifically, for each bond type (relation), the degree tensor
expresses, in the diagonal, the number of bonds attached to each atom.

Source:
[WGAN-GP with R-GCN for the generation of small molecular graphs](https://keras.io/examples/generative/wgan-graphs/))

The Decoder takes as input the latent-space representation and predicts
the graph adjacency matrix and feature matrix of the corresponding molecules.


```python

def get_encoder(
    gconv_units, latent_dim, adjacency_shape, feature_shape, dense_units, dropout_rate
):
    adjacency = layers.Input(shape=adjacency_shape)
    features = layers.Input(shape=feature_shape)

    # Propagate through one or more graph convolutional layers
    features_transformed = features
    for units in gconv_units:
        features_transformed = RelationalGraphConvLayer(units)(
            [adjacency, features_transformed]
        )
    # Reduce 2-D representation of molecule to 1-D
    x = layers.GlobalAveragePooling1D()(features_transformed)

    # Propagate through one or more densely connected layers
    for units in dense_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    z_mean = layers.Dense(latent_dim, dtype="float32", name="z_mean")(x)
    log_var = layers.Dense(latent_dim, dtype="float32", name="log_var")(x)

    encoder = keras.Model([adjacency, features], [z_mean, log_var], name="encoder")

    return encoder


def get_decoder(dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape):
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = latent_inputs
    for units in dense_units:
        x = layers.Dense(units, activation="tanh")(x)
        x = layers.Dropout(dropout_rate)(x)

    # Map outputs of previous layer (x) to [continuous] adjacency tensors (x_adjacency)
    x_adjacency = layers.Dense(np.prod(adjacency_shape))(x)
    x_adjacency = layers.Reshape(adjacency_shape)(x_adjacency)
    # Symmetrify tensors in the last two dimensions
    x_adjacency = (x_adjacency + ops.transpose(x_adjacency, (0, 1, 3, 2))) / 2
    x_adjacency = layers.Softmax(axis=1)(x_adjacency)

    # Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
    x_features = layers.Dense(np.prod(feature_shape))(x)
    x_features = layers.Reshape(feature_shape)(x_features)
    x_features = layers.Softmax(axis=2)(x_features)

    decoder = keras.Model(
        latent_inputs, outputs=[x_adjacency, x_features], name="decoder"
    )

    return decoder

```

---
## Build the Sampling layer


```python

class Sampling(layers.Layer):
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_log_var)[0]
        dim = ops.shape(z_log_var)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

```

---
## Build the VAE

This model is trained to optimize four losses:

* Categorical crossentropy
* KL divergence loss
* Property prediction loss
* Graph loss (gradient penalty)

The categorical crossentropy loss function measures the model's
reconstruction accuracy. The Property prediction loss estimates the mean squared
error between predicted and actual properties after running the latent representation
through a property prediction model. The property
prediction of the model is optimized via binary crossentropy. The gradient
penalty is further guided by the model's property (QED) prediction.

A gradient penalty is an alternative soft constraint on the
1-Lipschitz continuity as an improvement upon the gradient clipping scheme from the
original neural network
("1-Lipschitz continuity" means that the norm of the gradient is at most 1 at every single
point of the function).
It adds a regularization term to the loss function.


```python

class MoleculeGenerator(keras.Model):
    def __init__(self, encoder, decoder, max_len, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.property_prediction_layer = layers.Dense(1)
        self.max_len = max_len
        self.seed_generator = keras.random.SeedGenerator(seed)
        self.sampling_layer = Sampling(seed=seed)

        self.train_total_loss_tracker = keras.metrics.Mean(name="train_total_loss")
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")

    def train_step(self, data):
        adjacency_tensor, feature_tensor, qed_tensor = data[0]
        graph_real = [adjacency_tensor, feature_tensor]
        self.batch_size = ops.shape(qed_tensor)[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, qed_pred, gen_adjacency, gen_features = self(
                graph_real, training=True
            )
            graph_generated = [gen_adjacency, gen_features]
            total_loss = self._compute_loss(
                z_log_var, z_mean, qed_tensor, qed_pred, graph_real, graph_generated
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.train_total_loss_tracker.update_state(total_loss)
        return {"loss": self.train_total_loss_tracker.result()}

    def _compute_loss(
        self, z_log_var, z_mean, qed_true, qed_pred, graph_real, graph_generated
    ):
        adjacency_real, features_real = graph_real
        adjacency_gen, features_gen = graph_generated

        adjacency_loss = ops.mean(
            ops.sum(
                keras.losses.categorical_crossentropy(
                    adjacency_real, adjacency_gen, axis=1
                ),
                axis=(1, 2),
            )
        )
        features_loss = ops.mean(
            ops.sum(
                keras.losses.categorical_crossentropy(features_real, features_gen),
                axis=(1),
            )
        )
        kl_loss = -0.5 * ops.sum(
            1 + z_log_var - z_mean**2 - ops.minimum(ops.exp(z_log_var), 1e6), 1
        )
        kl_loss = ops.mean(kl_loss)

        property_loss = ops.mean(
            keras.losses.binary_crossentropy(qed_true, ops.squeeze(qed_pred, axis=1))
        )

        graph_loss = self._gradient_penalty(graph_real, graph_generated)

        return kl_loss + property_loss + graph_loss + adjacency_loss + features_loss

    def _gradient_penalty(self, graph_real, graph_generated):
        # Unpack graphs
        adjacency_real, features_real = graph_real
        adjacency_generated, features_generated = graph_generated

        # Generate interpolated graphs (adjacency_interp and features_interp)
        alpha = keras.random.uniform(shape=(self.batch_size,), seed=self.seed_generator)
        alpha = ops.reshape(alpha, (self.batch_size, 1, 1, 1))
        adjacency_interp = (adjacency_real * alpha) + (
            1.0 - alpha
        ) * adjacency_generated
        alpha = ops.reshape(alpha, (self.batch_size, 1, 1))
        features_interp = (features_real * alpha) + (1.0 - alpha) * features_generated

        # Compute the logits of interpolated graphs
        with tf.GradientTape() as tape:
            tape.watch(adjacency_interp)
            tape.watch(features_interp)
            _, _, logits, _, _ = self(
                [adjacency_interp, features_interp], training=True
            )

        # Compute the gradients with respect to the interpolated graphs
        grads = tape.gradient(logits, [adjacency_interp, features_interp])
        # Compute the gradient penalty
        grads_adjacency_penalty = (1 - ops.norm(grads[0], axis=1)) ** 2
        grads_features_penalty = (1 - ops.norm(grads[1], axis=2)) ** 2
        return ops.mean(
            ops.mean(grads_adjacency_penalty, axis=(-2, -1))
            + ops.mean(grads_features_penalty, axis=(-1))
        )

    def inference(self, batch_size):
        z = keras.random.normal(
            shape=(batch_size, LATENT_DIM), seed=self.seed_generator
        )
        reconstruction_adjacency, reconstruction_features = model.decoder.predict(z)
        # obtain one-hot encoded adjacency tensor
        adjacency = ops.argmax(reconstruction_adjacency, axis=1)
        adjacency = ops.one_hot(adjacency, num_classes=BOND_DIM, axis=1)
        # Remove potential self-loops from adjacency
        adjacency = adjacency * (1.0 - ops.eye(NUM_ATOMS, dtype="float32")[None, None])
        # obtain one-hot encoded feature tensor
        features = ops.argmax(reconstruction_features, axis=2)
        features = ops.one_hot(features, num_classes=ATOM_DIM, axis=2)
        return [
            graph_to_molecule([adjacency[i].numpy(), features[i].numpy()])
            for i in range(batch_size)
        ]

    def call(self, inputs):
        z_mean, log_var = self.encoder(inputs)
        z = self.sampling_layer([z_mean, log_var])

        gen_adjacency, gen_features = self.decoder(z)

        property_pred = self.property_prediction_layer(z_mean)

        return z_mean, log_var, property_pred, gen_adjacency, gen_features

```

---
## Train the model


```python
vae_optimizer = keras.optimizers.Adam(learning_rate=VAE_LR)

encoder = get_encoder(
    gconv_units=[9],
    adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
    feature_shape=(NUM_ATOMS, ATOM_DIM),
    latent_dim=LATENT_DIM,
    dense_units=[512],
    dropout_rate=0.0,
)
decoder = get_decoder(
    dense_units=[128, 256, 512],
    dropout_rate=0.2,
    latent_dim=LATENT_DIM,
    adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
    feature_shape=(NUM_ATOMS, ATOM_DIM),
)

model = MoleculeGenerator(encoder, decoder, MAX_MOLSIZE)

model.compile(vae_optimizer)
history = model.fit([adjacency_tensor, feature_tensor, qed_tensor], epochs=EPOCHS)
```

<div class="k-default-codeblock">
```
Epoch 1/10

```
</div>
    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  24:18 6s/step - loss: 24216.9707

<div class="k-default-codeblock">
```

```
</div>
   2/250 [37m━━━━━━━━━━━━━━━━━━━━  1:46 431ms/step - loss: 23922.6836

<div class="k-default-codeblock">
```

```
</div>
   3/250 [37m━━━━━━━━━━━━━━━━━━━━  1:45 426ms/step - loss: 23553.0898

<div class="k-default-codeblock">
```

```
</div>
   4/250 [37m━━━━━━━━━━━━━━━━━━━━  1:42 418ms/step - loss: 23099.4629

<div class="k-default-codeblock">
```

```
</div>
   5/250 [37m━━━━━━━━━━━━━━━━━━━━  1:42 419ms/step - loss: 22571.8809

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  1:52 461ms/step - loss: 21988.0430

<div class="k-default-codeblock">
```

```
</div>
   7/250 [37m━━━━━━━━━━━━━━━━━━━━  1:52 463ms/step - loss: 21366.1270

<div class="k-default-codeblock">
```

```
</div>
   8/250 [37m━━━━━━━━━━━━━━━━━━━━  1:56 482ms/step - loss: 20725.1309

<div class="k-default-codeblock">
```

```
</div>
   9/250 [37m━━━━━━━━━━━━━━━━━━━━  1:56 485ms/step - loss: 20080.8965

<div class="k-default-codeblock">
```

```
</div>
  10/250 [37m━━━━━━━━━━━━━━━━━━━━  1:54 477ms/step - loss: 19446.5879

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  1:52 472ms/step - loss: 18831.7070

<div class="k-default-codeblock">
```

```
</div>
  12/250 [37m━━━━━━━━━━━━━━━━━━━━  1:54 479ms/step - loss: 18242.2402

<div class="k-default-codeblock">
```

```
</div>
  13/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:53 477ms/step - loss: 17681.2422

<div class="k-default-codeblock">
```

```
</div>
  14/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:51 472ms/step - loss: 17149.8457

<div class="k-default-codeblock">
```

```
</div>
  15/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:49 466ms/step - loss: 16647.8164

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:49 468ms/step - loss: 16174.1885

<div class="k-default-codeblock">
```

```
</div>
  17/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:48 465ms/step - loss: 15727.4893

<div class="k-default-codeblock">
```

```
</div>
  18/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:47 461ms/step - loss: 15306.0811

<div class="k-default-codeblock">
```

```
</div>
  19/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:46 460ms/step - loss: 14908.2500

<div class="k-default-codeblock">
```

```
</div>
  20/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:45 459ms/step - loss: 14532.3252

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:45 461ms/step - loss: 14176.7090

<div class="k-default-codeblock">
```

```
</div>
  22/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:45 461ms/step - loss: 13839.8926

<div class="k-default-codeblock">
```

```
</div>
  23/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:45 463ms/step - loss: 13520.4697

<div class="k-default-codeblock">
```

```
</div>
  24/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:45 465ms/step - loss: 13217.1934

<div class="k-default-codeblock">
```

```
</div>
  25/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:44 464ms/step - loss: 12928.8936

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:44 464ms/step - loss: 12654.4717

<div class="k-default-codeblock">
```

```
</div>
  27/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:44 469ms/step - loss: 12392.9443

<div class="k-default-codeblock">
```

```
</div>
  28/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:44 471ms/step - loss: 12143.4199

<div class="k-default-codeblock">
```

```
</div>
  29/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:44 473ms/step - loss: 11905.0713

<div class="k-default-codeblock">
```

```
</div>
  30/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:44 473ms/step - loss: 11677.1650

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:43 472ms/step - loss: 11458.9990

<div class="k-default-codeblock">
```

```
</div>
  32/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:42 471ms/step - loss: 11249.9590

<div class="k-default-codeblock">
```

```
</div>
  33/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:41 470ms/step - loss: 11049.4512

<div class="k-default-codeblock">
```

```
</div>
  34/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:41 468ms/step - loss: 10856.9639

<div class="k-default-codeblock">
```

```
</div>
  35/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:40 467ms/step - loss: 10672.0029

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:39 465ms/step - loss: 10494.1250

<div class="k-default-codeblock">
```

```
</div>
  37/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:38 464ms/step - loss: 10322.9199

<div class="k-default-codeblock">
```

```
</div>
  38/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:38 463ms/step - loss: 10158.0088

<div class="k-default-codeblock">
```

```
</div>
  39/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:37 462ms/step - loss: 9999.0420 

<div class="k-default-codeblock">
```

```
</div>
  40/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:36 461ms/step - loss: 9845.6836

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:36 460ms/step - loss: 9697.6484

<div class="k-default-codeblock">
```

```
</div>
  42/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:35 459ms/step - loss: 9554.6533

<div class="k-default-codeblock">
```

```
</div>
  43/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 458ms/step - loss: 9416.4375

<div class="k-default-codeblock">
```

```
</div>
  44/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 457ms/step - loss: 9282.7588

<div class="k-default-codeblock">
```

```
</div>
  45/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:33 456ms/step - loss: 9153.3799

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:32 455ms/step - loss: 9028.0830

<div class="k-default-codeblock">
```

```
</div>
  47/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:32 454ms/step - loss: 8906.6738

<div class="k-default-codeblock">
```

```
</div>
  48/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:31 453ms/step - loss: 8788.9668

<div class="k-default-codeblock">
```

```
</div>
  49/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:31 455ms/step - loss: 8674.8018

<div class="k-default-codeblock">
```

```
</div>
  50/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:30 454ms/step - loss: 8564.0078

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:30 454ms/step - loss: 8456.4336

<div class="k-default-codeblock">
```

```
</div>
  52/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:29 454ms/step - loss: 8351.9385

<div class="k-default-codeblock">
```

```
</div>
  53/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:29 454ms/step - loss: 8250.3828

<div class="k-default-codeblock">
```

```
</div>
  54/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:28 453ms/step - loss: 8151.6396

<div class="k-default-codeblock">
```

```
</div>
  55/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:28 453ms/step - loss: 8055.5903

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:27 453ms/step - loss: 7962.1230

<div class="k-default-codeblock">
```

```
</div>
  57/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:27 452ms/step - loss: 7871.1304

<div class="k-default-codeblock">
```

```
</div>
  58/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:26 451ms/step - loss: 7782.5098

<div class="k-default-codeblock">
```

```
</div>
  59/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:26 451ms/step - loss: 7696.1650

<div class="k-default-codeblock">
```

```
</div>
  60/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:25 450ms/step - loss: 7612.0059

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:25 450ms/step - loss: 7529.9478

<div class="k-default-codeblock">
```

```
</div>
  62/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:24 450ms/step - loss: 7449.9106

<div class="k-default-codeblock">
```

```
</div>
  63/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:24 449ms/step - loss: 7371.8184

<div class="k-default-codeblock">
```

```
</div>
  64/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:23 450ms/step - loss: 7295.5957

<div class="k-default-codeblock">
```

```
</div>
  65/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:23 449ms/step - loss: 7221.1738

<div class="k-default-codeblock">
```

```
</div>
  66/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:22 448ms/step - loss: 7148.4858

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:21 448ms/step - loss: 7077.4717

<div class="k-default-codeblock">
```

```
</div>
  68/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:21 447ms/step - loss: 7008.0703

<div class="k-default-codeblock">
```

```
</div>
  69/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:20 447ms/step - loss: 6940.2212

<div class="k-default-codeblock">
```

```
</div>
  70/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:20 447ms/step - loss: 6873.8730

<div class="k-default-codeblock">
```

```
</div>
  71/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:20 448ms/step - loss: 6808.9795

<div class="k-default-codeblock">
```

```
</div>
  72/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:19 449ms/step - loss: 6745.4873

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:19 451ms/step - loss: 6683.3521

<div class="k-default-codeblock">
```

```
</div>
  74/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:19 451ms/step - loss: 6622.5269

<div class="k-default-codeblock">
```

```
</div>
  75/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:19 452ms/step - loss: 6562.9702

<div class="k-default-codeblock">
```

```
</div>
  76/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:18 451ms/step - loss: 6504.6411

<div class="k-default-codeblock">
```

```
</div>
  77/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:18 452ms/step - loss: 6447.4995

<div class="k-default-codeblock">
```

```
</div>
  78/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:17 452ms/step - loss: 6391.5078

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:17 452ms/step - loss: 6336.6318

<div class="k-default-codeblock">
```

```
</div>
  80/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:16 451ms/step - loss: 6282.8364

<div class="k-default-codeblock">
```

```
</div>
  81/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:16 450ms/step - loss: 6230.0874

<div class="k-default-codeblock">
```

```
</div>
  82/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:15 450ms/step - loss: 6178.3535

<div class="k-default-codeblock">
```

```
</div>
  83/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:15 450ms/step - loss: 6127.6055

<div class="k-default-codeblock">
```

```
</div>
  84/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:14 449ms/step - loss: 6077.8125

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:14 449ms/step - loss: 6028.9473

<div class="k-default-codeblock">
```

```
</div>
  86/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:13 449ms/step - loss: 5980.9834

<div class="k-default-codeblock">
```

```
</div>
  87/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:13 449ms/step - loss: 5933.8940

<div class="k-default-codeblock">
```

```
</div>
  88/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:12 449ms/step - loss: 5887.6538

<div class="k-default-codeblock">
```

```
</div>
  89/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:12 449ms/step - loss: 5842.2402

<div class="k-default-codeblock">
```

```
</div>
  90/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:11 449ms/step - loss: 5797.6313

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:11 448ms/step - loss: 5753.8037

<div class="k-default-codeblock">
```

```
</div>
  92/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:10 448ms/step - loss: 5710.7383

<div class="k-default-codeblock">
```

```
</div>
  93/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:10 447ms/step - loss: 5668.4141

<div class="k-default-codeblock">
```

```
</div>
  94/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:09 447ms/step - loss: 5626.8115

<div class="k-default-codeblock">
```

```
</div>
  95/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:09 448ms/step - loss: 5585.9097

<div class="k-default-codeblock">
```

```
</div>
  96/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:08 447ms/step - loss: 5545.6914

<div class="k-default-codeblock">
```

```
</div>
  97/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:08 447ms/step - loss: 5506.1382

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 447ms/step - loss: 5467.2324

<div class="k-default-codeblock">
```

```
</div>
  99/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 447ms/step - loss: 5428.9585

<div class="k-default-codeblock">
```

```
</div>
 100/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:07 448ms/step - loss: 5391.3008

<div class="k-default-codeblock">
```

```
</div>
 101/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:06 447ms/step - loss: 5354.2427

<div class="k-default-codeblock">
```

```
</div>
 102/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:06 447ms/step - loss: 5317.7700

<div class="k-default-codeblock">
```

```
</div>
 103/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:05 447ms/step - loss: 5281.8696

<div class="k-default-codeblock">
```

```
</div>
 104/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:05 447ms/step - loss: 5246.5264

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:04 446ms/step - loss: 5211.7271

<div class="k-default-codeblock">
```

```
</div>
 106/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:04 446ms/step - loss: 5177.4585

<div class="k-default-codeblock">
```

```
</div>
 107/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:03 446ms/step - loss: 5143.7090

<div class="k-default-codeblock">
```

```
</div>
 108/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:03 446ms/step - loss: 5110.4658

<div class="k-default-codeblock">
```

```
</div>
 109/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:02 446ms/step - loss: 5077.7178

<div class="k-default-codeblock">
```

```
</div>
 110/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:02 445ms/step - loss: 5045.4531

<div class="k-default-codeblock">
```

```
</div>
 111/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 445ms/step - loss: 5013.6606

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 445ms/step - loss: 4982.3296

<div class="k-default-codeblock">
```

```
</div>
 113/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:00 444ms/step - loss: 4951.4507

<div class="k-default-codeblock">
```

```
</div>
 114/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:00 444ms/step - loss: 4921.0117

<div class="k-default-codeblock">
```

```
</div>
 115/250 ━━━━━━━━━[37m━━━━━━━━━━━  59s 444ms/step - loss: 4891.0034 

<div class="k-default-codeblock">
```

```
</div>
 116/250 ━━━━━━━━━[37m━━━━━━━━━━━  59s 444ms/step - loss: 4861.4165

<div class="k-default-codeblock">
```

```
</div>
 117/250 ━━━━━━━━━[37m━━━━━━━━━━━  58s 444ms/step - loss: 4832.2422

<div class="k-default-codeblock">
```

```
</div>
 118/250 ━━━━━━━━━[37m━━━━━━━━━━━  58s 444ms/step - loss: 4803.4712

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  58s 443ms/step - loss: 4775.0957

<div class="k-default-codeblock">
```

```
</div>
 120/250 ━━━━━━━━━[37m━━━━━━━━━━━  57s 443ms/step - loss: 4747.1074

<div class="k-default-codeblock">
```

```
</div>
 121/250 ━━━━━━━━━[37m━━━━━━━━━━━  57s 444ms/step - loss: 4719.4976

<div class="k-default-codeblock">
```

```
</div>
 122/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 444ms/step - loss: 4692.2568

<div class="k-default-codeblock">
```

```
</div>
 123/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 444ms/step - loss: 4665.3789

<div class="k-default-codeblock">
```

```
</div>
 124/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 443ms/step - loss: 4638.8560

<div class="k-default-codeblock">
```

```
</div>
 125/250 ━━━━━━━━━━[37m━━━━━━━━━━  55s 443ms/step - loss: 4612.6812

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  54s 443ms/step - loss: 4586.8467

<div class="k-default-codeblock">
```

```
</div>
 127/250 ━━━━━━━━━━[37m━━━━━━━━━━  54s 443ms/step - loss: 4561.3452

<div class="k-default-codeblock">
```

```
</div>
 128/250 ━━━━━━━━━━[37m━━━━━━━━━━  53s 443ms/step - loss: 4536.1714

<div class="k-default-codeblock">
```

```
</div>
 129/250 ━━━━━━━━━━[37m━━━━━━━━━━  53s 443ms/step - loss: 4511.3193

<div class="k-default-codeblock">
```

```
</div>
 130/250 ━━━━━━━━━━[37m━━━━━━━━━━  53s 442ms/step - loss: 4486.7812

<div class="k-default-codeblock">
```

```
</div>
 131/250 ━━━━━━━━━━[37m━━━━━━━━━━  52s 442ms/step - loss: 4462.5513

<div class="k-default-codeblock">
```

```
</div>
 132/250 ━━━━━━━━━━[37m━━━━━━━━━━  52s 442ms/step - loss: 4438.6235

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 442ms/step - loss: 4414.9922

<div class="k-default-codeblock">
```

```
</div>
 134/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 443ms/step - loss: 4391.6519

<div class="k-default-codeblock">
```

```
</div>
 135/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 443ms/step - loss: 4368.5972

<div class="k-default-codeblock">
```

```
</div>
 136/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 443ms/step - loss: 4345.8218

<div class="k-default-codeblock">
```

```
</div>
 137/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 443ms/step - loss: 4323.3213

<div class="k-default-codeblock">
```

```
</div>
 138/250 ━━━━━━━━━━━[37m━━━━━━━━━  49s 443ms/step - loss: 4301.0908

<div class="k-default-codeblock">
```

```
</div>
 139/250 ━━━━━━━━━━━[37m━━━━━━━━━  49s 443ms/step - loss: 4279.1250

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  48s 442ms/step - loss: 4257.4180

<div class="k-default-codeblock">
```

```
</div>
 141/250 ━━━━━━━━━━━[37m━━━━━━━━━  48s 442ms/step - loss: 4235.9658

<div class="k-default-codeblock">
```

```
</div>
 142/250 ━━━━━━━━━━━[37m━━━━━━━━━  47s 442ms/step - loss: 4214.7637

<div class="k-default-codeblock">
```

```
</div>
 143/250 ━━━━━━━━━━━[37m━━━━━━━━━  47s 442ms/step - loss: 4193.8062

<div class="k-default-codeblock">
```

```
</div>
 144/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 442ms/step - loss: 4173.0898

<div class="k-default-codeblock">
```

```
</div>
 145/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 442ms/step - loss: 4152.6108

<div class="k-default-codeblock">
```

```
</div>
 146/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 442ms/step - loss: 4132.3638

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 442ms/step - loss: 4112.3452

<div class="k-default-codeblock">
```

```
</div>
 148/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 442ms/step - loss: 4092.5505

<div class="k-default-codeblock">
```

```
</div>
 149/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 442ms/step - loss: 4072.9775

<div class="k-default-codeblock">
```

```
</div>
 150/250 ━━━━━━━━━━━━[37m━━━━━━━━  44s 441ms/step - loss: 4053.6206

<div class="k-default-codeblock">
```

```
</div>
 151/250 ━━━━━━━━━━━━[37m━━━━━━━━  43s 441ms/step - loss: 4034.4766

<div class="k-default-codeblock">
```

```
</div>
 152/250 ━━━━━━━━━━━━[37m━━━━━━━━  43s 441ms/step - loss: 4015.5425

<div class="k-default-codeblock">
```

```
</div>
 153/250 ━━━━━━━━━━━━[37m━━━━━━━━  42s 441ms/step - loss: 3996.8145

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  42s 441ms/step - loss: 3978.2888

<div class="k-default-codeblock">
```

```
</div>
 155/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 441ms/step - loss: 3959.9624

<div class="k-default-codeblock">
```

```
</div>
 156/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 441ms/step - loss: 3941.8311

<div class="k-default-codeblock">
```

```
</div>
 157/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 441ms/step - loss: 3923.8921

<div class="k-default-codeblock">
```

```
</div>
 158/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 441ms/step - loss: 3906.1421

<div class="k-default-codeblock">
```

```
</div>
 159/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 441ms/step - loss: 3888.5786

<div class="k-default-codeblock">
```

```
</div>
 160/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 440ms/step - loss: 3871.1982

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 440ms/step - loss: 3853.9980

<div class="k-default-codeblock">
```

```
</div>
 162/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 440ms/step - loss: 3836.9753

<div class="k-default-codeblock">
```

```
</div>
 163/250 ━━━━━━━━━━━━━[37m━━━━━━━  38s 440ms/step - loss: 3820.1270

<div class="k-default-codeblock">
```

```
</div>
 164/250 ━━━━━━━━━━━━━[37m━━━━━━━  37s 440ms/step - loss: 3803.4495

<div class="k-default-codeblock">
```

```
</div>
 165/250 ━━━━━━━━━━━━━[37m━━━━━━━  37s 440ms/step - loss: 3786.9407

<div class="k-default-codeblock">
```

```
</div>
 166/250 ━━━━━━━━━━━━━[37m━━━━━━━  36s 440ms/step - loss: 3770.5984

<div class="k-default-codeblock">
```

```
</div>
 167/250 ━━━━━━━━━━━━━[37m━━━━━━━  36s 440ms/step - loss: 3754.4199

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  36s 440ms/step - loss: 3738.4026

<div class="k-default-codeblock">
```

```
</div>
 169/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 440ms/step - loss: 3722.5442

<div class="k-default-codeblock">
```

```
</div>
 170/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 440ms/step - loss: 3706.8423

<div class="k-default-codeblock">
```

```
</div>
 171/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 440ms/step - loss: 3691.2939

<div class="k-default-codeblock">
```

```
</div>
 172/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 440ms/step - loss: 3675.8970

<div class="k-default-codeblock">
```

```
</div>
 173/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 440ms/step - loss: 3660.6497

<div class="k-default-codeblock">
```

```
</div>
 174/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 439ms/step - loss: 3645.5496

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  32s 439ms/step - loss: 3630.5942

<div class="k-default-codeblock">
```

```
</div>
 176/250 ━━━━━━━━━━━━━━[37m━━━━━━  32s 439ms/step - loss: 3615.7812

<div class="k-default-codeblock">
```

```
</div>
 177/250 ━━━━━━━━━━━━━━[37m━━━━━━  32s 439ms/step - loss: 3601.1091

<div class="k-default-codeblock">
```

```
</div>
 178/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 439ms/step - loss: 3586.5754

<div class="k-default-codeblock">
```

```
</div>
 179/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 439ms/step - loss: 3572.1777

<div class="k-default-codeblock">
```

```
</div>
 180/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 439ms/step - loss: 3557.9141

<div class="k-default-codeblock">
```

```
</div>
 181/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 439ms/step - loss: 3543.7830

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 439ms/step - loss: 3529.7820

<div class="k-default-codeblock">
```

```
</div>
 183/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 439ms/step - loss: 3515.9099

<div class="k-default-codeblock">
```

```
</div>
 184/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 439ms/step - loss: 3502.1646

<div class="k-default-codeblock">
```

```
</div>
 185/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 439ms/step - loss: 3488.5439

<div class="k-default-codeblock">
```

```
</div>
 186/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 439ms/step - loss: 3475.0469

<div class="k-default-codeblock">
```

```
</div>
 187/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 439ms/step - loss: 3461.6714

<div class="k-default-codeblock">
```

```
</div>
 188/250 ━━━━━━━━━━━━━━━[37m━━━━━  27s 440ms/step - loss: 3448.4160

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  26s 440ms/step - loss: 3435.2781

<div class="k-default-codeblock">
```

```
</div>
 190/250 ━━━━━━━━━━━━━━━[37m━━━━━  26s 439ms/step - loss: 3422.2568

<div class="k-default-codeblock">
```

```
</div>
 191/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 439ms/step - loss: 3409.3506

<div class="k-default-codeblock">
```

```
</div>
 192/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 439ms/step - loss: 3396.5579

<div class="k-default-codeblock">
```

```
</div>
 193/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 439ms/step - loss: 3383.8770

<div class="k-default-codeblock">
```

```
</div>
 194/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 439ms/step - loss: 3371.3062

<div class="k-default-codeblock">
```

```
</div>
 195/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 439ms/step - loss: 3358.8442

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 439ms/step - loss: 3346.4895

<div class="k-default-codeblock">
```

```
</div>
 197/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 439ms/step - loss: 3334.2402

<div class="k-default-codeblock">
```

```
</div>
 198/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 439ms/step - loss: 3322.0955

<div class="k-default-codeblock">
```

```
</div>
 199/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 439ms/step - loss: 3310.0532

<div class="k-default-codeblock">
```

```
</div>
 200/250 ━━━━━━━━━━━━━━━━[37m━━━━  21s 439ms/step - loss: 3298.1128

<div class="k-default-codeblock">
```

```
</div>
 201/250 ━━━━━━━━━━━━━━━━[37m━━━━  21s 439ms/step - loss: 3286.2725

<div class="k-default-codeblock">
```

```
</div>
 202/250 ━━━━━━━━━━━━━━━━[37m━━━━  21s 439ms/step - loss: 3274.5308

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 439ms/step - loss: 3262.8870

<div class="k-default-codeblock">
```

```
</div>
 204/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 439ms/step - loss: 3251.3398

<div class="k-default-codeblock">
```

```
</div>
 205/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 439ms/step - loss: 3239.8879

<div class="k-default-codeblock">
```

```
</div>
 206/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 438ms/step - loss: 3228.5300

<div class="k-default-codeblock">
```

```
</div>
 207/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 438ms/step - loss: 3217.2651

<div class="k-default-codeblock">
```

```
</div>
 208/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 438ms/step - loss: 3206.0918

<div class="k-default-codeblock">
```

```
</div>
 209/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 438ms/step - loss: 3195.0090

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 438ms/step - loss: 3184.0156

<div class="k-default-codeblock">
```

```
</div>
 211/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 438ms/step - loss: 3173.1106

<div class="k-default-codeblock">
```

```
</div>
 212/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 438ms/step - loss: 3162.2930

<div class="k-default-codeblock">
```

```
</div>
 213/250 ━━━━━━━━━━━━━━━━━[37m━━━  16s 438ms/step - loss: 3151.5615

<div class="k-default-codeblock">
```

```
</div>
 214/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 438ms/step - loss: 3140.9150

<div class="k-default-codeblock">
```

```
</div>
 215/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 438ms/step - loss: 3130.3523

<div class="k-default-codeblock">
```

```
</div>
 216/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 437ms/step - loss: 3119.8723

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 437ms/step - loss: 3109.4746

<div class="k-default-codeblock">
```

```
</div>
 218/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 437ms/step - loss: 3099.1575

<div class="k-default-codeblock">
```

```
</div>
 219/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 437ms/step - loss: 3088.9204

<div class="k-default-codeblock">
```

```
</div>
 220/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 437ms/step - loss: 3078.7622

<div class="k-default-codeblock">
```

```
</div>
 221/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 437ms/step - loss: 3068.6821

<div class="k-default-codeblock">
```

```
</div>
 222/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 437ms/step - loss: 3058.6792

<div class="k-default-codeblock">
```

```
</div>
 223/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 437ms/step - loss: 3048.7524

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 437ms/step - loss: 3038.9011

<div class="k-default-codeblock">
```

```
</div>
 225/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 437ms/step - loss: 3029.1243

<div class="k-default-codeblock">
```

```
</div>
 226/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 437ms/step - loss: 3019.4211

<div class="k-default-codeblock">
```

```
</div>
 227/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 437ms/step - loss: 3009.7908

<div class="k-default-codeblock">
```

```
</div>
 228/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 437ms/step - loss: 3000.2324 

<div class="k-default-codeblock">
```

```
</div>
 229/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 437ms/step - loss: 2990.7449

<div class="k-default-codeblock">
```

```
</div>
 230/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 437ms/step - loss: 2981.3279

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 437ms/step - loss: 2971.9802

<div class="k-default-codeblock">
```

```
</div>
 232/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 437ms/step - loss: 2962.7012

<div class="k-default-codeblock">
```

```
</div>
 233/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 437ms/step - loss: 2953.4900

<div class="k-default-codeblock">
```

```
</div>
 234/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 437ms/step - loss: 2944.3457

<div class="k-default-codeblock">
```

```
</div>
 235/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 436ms/step - loss: 2935.2681

<div class="k-default-codeblock">
```

```
</div>
 236/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 436ms/step - loss: 2926.2561

<div class="k-default-codeblock">
```

```
</div>
 237/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 436ms/step - loss: 2917.3091

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  5s 436ms/step - loss: 2908.4263

<div class="k-default-codeblock">
```

```
</div>
 239/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 436ms/step - loss: 2899.6067

<div class="k-default-codeblock">
```

```
</div>
 240/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 436ms/step - loss: 2890.8499

<div class="k-default-codeblock">
```

```
</div>
 241/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 436ms/step - loss: 2882.1548

<div class="k-default-codeblock">
```

```
</div>
 242/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 436ms/step - loss: 2873.5212

<div class="k-default-codeblock">
```

```
</div>
 243/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 436ms/step - loss: 2864.9485

<div class="k-default-codeblock">
```

```
</div>
 244/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 436ms/step - loss: 2856.4351

<div class="k-default-codeblock">
```

```
</div>
 245/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 436ms/step - loss: 2847.9814

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 436ms/step - loss: 2839.5872

<div class="k-default-codeblock">
```

```
</div>
 247/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 435ms/step - loss: 2831.2505

<div class="k-default-codeblock">
```

```
</div>
 248/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 435ms/step - loss: 2822.9712

<div class="k-default-codeblock">
```

```
</div>
 249/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 435ms/step - loss: 2814.7488

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 435ms/step - loss: 2806.5825

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 114s 436ms/step - loss: 2798.4812


<div class="k-default-codeblock">
```
Epoch 2/10

```
</div>
    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  1:54 459ms/step - loss: 179.4747

<div class="k-default-codeblock">
```

```
</div>
   2/250 [37m━━━━━━━━━━━━━━━━━━━━  1:48 438ms/step - loss: 182.6920

<div class="k-default-codeblock">
```

```
</div>
   3/250 [37m━━━━━━━━━━━━━━━━━━━━  1:45 426ms/step - loss: 184.3708

<div class="k-default-codeblock">
```

```
</div>
   4/250 [37m━━━━━━━━━━━━━━━━━━━━  1:43 422ms/step - loss: 185.5217

<div class="k-default-codeblock">
```

```
</div>
   5/250 [37m━━━━━━━━━━━━━━━━━━━━  1:42 418ms/step - loss: 185.5805

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 416ms/step - loss: 185.2376

<div class="k-default-codeblock">
```

```
</div>
   7/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 415ms/step - loss: 185.0059

<div class="k-default-codeblock">
```

```
</div>
   8/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 417ms/step - loss: 184.8391

<div class="k-default-codeblock">
```

```
</div>
   9/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 416ms/step - loss: 184.6048

<div class="k-default-codeblock">
```

```
</div>
  10/250 [37m━━━━━━━━━━━━━━━━━━━━  1:39 413ms/step - loss: 184.3389

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  1:38 414ms/step - loss: 184.0912

<div class="k-default-codeblock">
```

```
</div>
  12/250 [37m━━━━━━━━━━━━━━━━━━━━  1:38 413ms/step - loss: 183.8590

<div class="k-default-codeblock">
```

```
</div>
  13/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:38 414ms/step - loss: 183.6169

<div class="k-default-codeblock">
```

```
</div>
  14/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 412ms/step - loss: 183.3595

<div class="k-default-codeblock">
```

```
</div>
  15/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 411ms/step - loss: 183.1529

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 410ms/step - loss: 182.9640

<div class="k-default-codeblock">
```

```
</div>
  17/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 411ms/step - loss: 182.7744

<div class="k-default-codeblock">
```

```
</div>
  18/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 412ms/step - loss: 182.6077

<div class="k-default-codeblock">
```

```
</div>
  19/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 412ms/step - loss: 182.4455

<div class="k-default-codeblock">
```

```
</div>
  20/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 413ms/step - loss: 182.2913

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 412ms/step - loss: 182.1252

<div class="k-default-codeblock">
```

```
</div>
  22/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 412ms/step - loss: 181.9659

<div class="k-default-codeblock">
```

```
</div>
  23/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:33 412ms/step - loss: 181.8105

<div class="k-default-codeblock">
```

```
</div>
  24/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:33 412ms/step - loss: 181.6765

<div class="k-default-codeblock">
```

```
</div>
  25/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 412ms/step - loss: 181.5663

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 412ms/step - loss: 181.4488

<div class="k-default-codeblock">
```

```
</div>
  27/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 412ms/step - loss: 181.3460

<div class="k-default-codeblock">
```

```
</div>
  28/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 412ms/step - loss: 181.2930

<div class="k-default-codeblock">
```

```
</div>
  29/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 411ms/step - loss: 181.2451

<div class="k-default-codeblock">
```

```
</div>
  30/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 411ms/step - loss: 181.2255

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 411ms/step - loss: 181.2283

<div class="k-default-codeblock">
```

```
</div>
  32/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 412ms/step - loss: 181.2764

<div class="k-default-codeblock">
```

```
</div>
  33/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 412ms/step - loss: 181.3701

<div class="k-default-codeblock">
```

```
</div>
  34/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 413ms/step - loss: 181.5173

<div class="k-default-codeblock">
```

```
</div>
  35/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 413ms/step - loss: 181.7070

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 414ms/step - loss: 181.9332

<div class="k-default-codeblock">
```

```
</div>
  37/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 414ms/step - loss: 182.1902

<div class="k-default-codeblock">
```

```
</div>
  38/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 414ms/step - loss: 182.4795

<div class="k-default-codeblock">
```

```
</div>
  39/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 415ms/step - loss: 182.7845

<div class="k-default-codeblock">
```

```
</div>
  40/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 415ms/step - loss: 183.0999

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 414ms/step - loss: 183.4286

<div class="k-default-codeblock">
```

```
</div>
  42/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 414ms/step - loss: 183.7604

<div class="k-default-codeblock">
```

```
</div>
  43/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 414ms/step - loss: 184.0931

<div class="k-default-codeblock">
```

```
</div>
  44/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 414ms/step - loss: 184.4234

<div class="k-default-codeblock">
```

```
</div>
  45/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 414ms/step - loss: 184.7474

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 414ms/step - loss: 185.0590

<div class="k-default-codeblock">
```

```
</div>
  47/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 415ms/step - loss: 185.3617

<div class="k-default-codeblock">
```

```
</div>
  48/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:23 415ms/step - loss: 185.6534

<div class="k-default-codeblock">
```

```
</div>
  49/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:23 415ms/step - loss: 185.9341

<div class="k-default-codeblock">
```

```
</div>
  50/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:23 415ms/step - loss: 186.1977

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 416ms/step - loss: 186.4462

<div class="k-default-codeblock">
```

```
</div>
  52/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 416ms/step - loss: 186.6853

<div class="k-default-codeblock">
```

```
</div>
  53/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 416ms/step - loss: 186.9152

<div class="k-default-codeblock">
```

```
</div>
  54/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 416ms/step - loss: 187.1340

<div class="k-default-codeblock">
```

```
</div>
  55/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 415ms/step - loss: 187.3395

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 415ms/step - loss: 187.5313

<div class="k-default-codeblock">
```

```
</div>
  57/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 415ms/step - loss: 187.7116

<div class="k-default-codeblock">
```

```
</div>
  58/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 416ms/step - loss: 187.8851

<div class="k-default-codeblock">
```

```
</div>
  59/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 416ms/step - loss: 188.0505

<div class="k-default-codeblock">
```

```
</div>
  60/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 416ms/step - loss: 188.2051

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:18 416ms/step - loss: 188.3540

<div class="k-default-codeblock">
```

```
</div>
  62/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:18 416ms/step - loss: 188.4969

<div class="k-default-codeblock">
```

```
</div>
  63/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:17 416ms/step - loss: 188.6323

<div class="k-default-codeblock">
```

```
</div>
  64/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:17 416ms/step - loss: 188.7607

<div class="k-default-codeblock">
```

```
</div>
  65/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 416ms/step - loss: 188.8837

<div class="k-default-codeblock">
```

```
</div>
  66/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 416ms/step - loss: 189.0040

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 416ms/step - loss: 189.1186

<div class="k-default-codeblock">
```

```
</div>
  68/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 416ms/step - loss: 189.2269

<div class="k-default-codeblock">
```

```
</div>
  69/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 417ms/step - loss: 189.3287

<div class="k-default-codeblock">
```

```
</div>
  70/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 417ms/step - loss: 189.4250

<div class="k-default-codeblock">
```

```
</div>
  71/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 417ms/step - loss: 189.5169

<div class="k-default-codeblock">
```

```
</div>
  72/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 417ms/step - loss: 189.6035

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 417ms/step - loss: 189.6846

<div class="k-default-codeblock">
```

```
</div>
  74/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 417ms/step - loss: 189.7601

<div class="k-default-codeblock">
```

```
</div>
  75/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 417ms/step - loss: 189.8284

<div class="k-default-codeblock">
```

```
</div>
  76/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 417ms/step - loss: 189.8901

<div class="k-default-codeblock">
```

```
</div>
  77/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 417ms/step - loss: 189.9476

<div class="k-default-codeblock">
```

```
</div>
  78/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 417ms/step - loss: 190.0023

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 417ms/step - loss: 190.0521

<div class="k-default-codeblock">
```

```
</div>
  80/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 416ms/step - loss: 190.0990

<div class="k-default-codeblock">
```

```
</div>
  81/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 416ms/step - loss: 190.1415

<div class="k-default-codeblock">
```

```
</div>
  82/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 417ms/step - loss: 190.1799

<div class="k-default-codeblock">
```

```
</div>
  83/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 416ms/step - loss: 190.2143

<div class="k-default-codeblock">
```

```
</div>
  84/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 417ms/step - loss: 190.2457

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 417ms/step - loss: 190.2757

<div class="k-default-codeblock">
```

```
</div>
  86/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 416ms/step - loss: 190.3033

<div class="k-default-codeblock">
```

```
</div>
  87/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:07 416ms/step - loss: 190.3278

<div class="k-default-codeblock">
```

```
</div>
  88/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 416ms/step - loss: 190.3485

<div class="k-default-codeblock">
```

```
</div>
  89/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 416ms/step - loss: 190.3663

<div class="k-default-codeblock">
```

```
</div>
  90/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 416ms/step - loss: 190.3837

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 416ms/step - loss: 190.3993

<div class="k-default-codeblock">
```

```
</div>
  92/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 416ms/step - loss: 190.4126

<div class="k-default-codeblock">
```

```
</div>
  93/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 416ms/step - loss: 190.4245

<div class="k-default-codeblock">
```

```
</div>
  94/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 416ms/step - loss: 190.4333

<div class="k-default-codeblock">
```

```
</div>
  95/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 416ms/step - loss: 190.4381

<div class="k-default-codeblock">
```

```
</div>
  96/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 416ms/step - loss: 190.4431

<div class="k-default-codeblock">
```

```
</div>
  97/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 416ms/step - loss: 190.4465

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 416ms/step - loss: 190.4480

<div class="k-default-codeblock">
```

```
</div>
  99/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:02 416ms/step - loss: 190.4500

<div class="k-default-codeblock">
```

```
</div>
 100/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:02 416ms/step - loss: 190.4518

<div class="k-default-codeblock">
```

```
</div>
 101/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 416ms/step - loss: 190.4517

<div class="k-default-codeblock">
```

```
</div>
 102/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 416ms/step - loss: 190.4504

<div class="k-default-codeblock">
```

```
</div>
 103/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 416ms/step - loss: 190.4491

<div class="k-default-codeblock">
```

```
</div>
 104/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 416ms/step - loss: 190.4475

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 416ms/step - loss: 190.4469

<div class="k-default-codeblock">
```

```
</div>
 106/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 416ms/step - loss: 190.4470 

<div class="k-default-codeblock">
```

```
</div>
 107/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 416ms/step - loss: 190.4470

<div class="k-default-codeblock">
```

```
</div>
 108/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 416ms/step - loss: 190.4469

<div class="k-default-codeblock">
```

```
</div>
 109/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 416ms/step - loss: 190.4473

<div class="k-default-codeblock">
```

```
</div>
 110/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 416ms/step - loss: 190.4474

<div class="k-default-codeblock">
```

```
</div>
 111/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 416ms/step - loss: 190.4470

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 416ms/step - loss: 190.4472

<div class="k-default-codeblock">
```

```
</div>
 113/250 ━━━━━━━━━[37m━━━━━━━━━━━  57s 416ms/step - loss: 190.4474

<div class="k-default-codeblock">
```

```
</div>
 114/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 416ms/step - loss: 190.4481

<div class="k-default-codeblock">
```

```
</div>
 115/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 416ms/step - loss: 190.4486

<div class="k-default-codeblock">
```

```
</div>
 116/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 416ms/step - loss: 190.4478

<div class="k-default-codeblock">
```

```
</div>
 117/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 416ms/step - loss: 190.4457

<div class="k-default-codeblock">
```

```
</div>
 118/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 416ms/step - loss: 190.4439

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 416ms/step - loss: 190.4540

<div class="k-default-codeblock">
```

```
</div>
 120/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 416ms/step - loss: 190.4636

<div class="k-default-codeblock">
```

```
</div>
 121/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 416ms/step - loss: 190.4767

<div class="k-default-codeblock">
```

```
</div>
 122/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 416ms/step - loss: 190.4951

<div class="k-default-codeblock">
```

```
</div>
 123/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 416ms/step - loss: 190.5217

<div class="k-default-codeblock">
```

```
</div>
 124/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 416ms/step - loss: 190.5585

<div class="k-default-codeblock">
```

```
</div>
 125/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 416ms/step - loss: 190.6081

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 416ms/step - loss: 190.6728

<div class="k-default-codeblock">
```

```
</div>
 127/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 416ms/step - loss: 190.7545

<div class="k-default-codeblock">
```

```
</div>
 128/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 416ms/step - loss: 190.8531

<div class="k-default-codeblock">
```

```
</div>
 129/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 416ms/step - loss: 190.9697

<div class="k-default-codeblock">
```

```
</div>
 130/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 416ms/step - loss: 191.1034

<div class="k-default-codeblock">
```

```
</div>
 131/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 416ms/step - loss: 191.2543

<div class="k-default-codeblock">
```

```
</div>
 132/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 416ms/step - loss: 191.4209

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 416ms/step - loss: 191.6022

<div class="k-default-codeblock">
```

```
</div>
 134/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 416ms/step - loss: 191.7976

<div class="k-default-codeblock">
```

```
</div>
 135/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 416ms/step - loss: 192.0056

<div class="k-default-codeblock">
```

```
</div>
 136/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 416ms/step - loss: 192.2238

<div class="k-default-codeblock">
```

```
</div>
 137/250 ━━━━━━━━━━[37m━━━━━━━━━━  46s 416ms/step - loss: 192.4513

<div class="k-default-codeblock">
```

```
</div>
 138/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 416ms/step - loss: 192.6864

<div class="k-default-codeblock">
```

```
</div>
 139/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 415ms/step - loss: 192.9280

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 416ms/step - loss: 193.1747

<div class="k-default-codeblock">
```

```
</div>
 141/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 416ms/step - loss: 193.4255

<div class="k-default-codeblock">
```

```
</div>
 142/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 415ms/step - loss: 193.6793

<div class="k-default-codeblock">
```

```
</div>
 143/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 415ms/step - loss: 193.9350

<div class="k-default-codeblock">
```

```
</div>
 144/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 415ms/step - loss: 194.1921

<div class="k-default-codeblock">
```

```
</div>
 145/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 415ms/step - loss: 194.4497

<div class="k-default-codeblock">
```

```
</div>
 146/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 415ms/step - loss: 194.7072

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 415ms/step - loss: 194.9638

<div class="k-default-codeblock">
```

```
</div>
 148/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 415ms/step - loss: 195.2188

<div class="k-default-codeblock">
```

```
</div>
 149/250 ━━━━━━━━━━━[37m━━━━━━━━━  41s 415ms/step - loss: 195.4721

<div class="k-default-codeblock">
```

```
</div>
 150/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 415ms/step - loss: 195.7231

<div class="k-default-codeblock">
```

```
</div>
 151/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 415ms/step - loss: 195.9713

<div class="k-default-codeblock">
```

```
</div>
 152/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 415ms/step - loss: 196.2161

<div class="k-default-codeblock">
```

```
</div>
 153/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 415ms/step - loss: 196.4575

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 415ms/step - loss: 196.6951

<div class="k-default-codeblock">
```

```
</div>
 155/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 415ms/step - loss: 196.9283

<div class="k-default-codeblock">
```

```
</div>
 156/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 415ms/step - loss: 197.1576

<div class="k-default-codeblock">
```

```
</div>
 157/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 415ms/step - loss: 197.3831

<div class="k-default-codeblock">
```

```
</div>
 158/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 415ms/step - loss: 197.6042

<div class="k-default-codeblock">
```

```
</div>
 159/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 415ms/step - loss: 197.8206

<div class="k-default-codeblock">
```

```
</div>
 160/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 416ms/step - loss: 198.0327

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 416ms/step - loss: 198.2943

<div class="k-default-codeblock">
```

```
</div>
 162/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 416ms/step - loss: 198.5505

<div class="k-default-codeblock">
```

```
</div>
 163/250 ━━━━━━━━━━━━━[37m━━━━━━━  36s 416ms/step - loss: 198.8024

<div class="k-default-codeblock">
```

```
</div>
 164/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 416ms/step - loss: 199.0510

<div class="k-default-codeblock">
```

```
</div>
 165/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 416ms/step - loss: 199.2975

<div class="k-default-codeblock">
```

```
</div>
 166/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 416ms/step - loss: 199.5427

<div class="k-default-codeblock">
```

```
</div>
 167/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 416ms/step - loss: 199.7875

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 416ms/step - loss: 200.0332

<div class="k-default-codeblock">
```

```
</div>
 169/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 416ms/step - loss: 200.2812

<div class="k-default-codeblock">
```

```
</div>
 170/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 416ms/step - loss: 200.5319

<div class="k-default-codeblock">
```

```
</div>
 171/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 416ms/step - loss: 200.7862

<div class="k-default-codeblock">
```

```
</div>
 172/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 416ms/step - loss: 201.0448

<div class="k-default-codeblock">
```

```
</div>
 173/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 416ms/step - loss: 201.3081

<div class="k-default-codeblock">
```

```
</div>
 174/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 416ms/step - loss: 201.5766

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 416ms/step - loss: 201.8505

<div class="k-default-codeblock">
```

```
</div>
 176/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 416ms/step - loss: 202.1304

<div class="k-default-codeblock">
```

```
</div>
 177/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 416ms/step - loss: 202.4161

<div class="k-default-codeblock">
```

```
</div>
 178/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 416ms/step - loss: 202.7081

<div class="k-default-codeblock">
```

```
</div>
 179/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 416ms/step - loss: 203.0058

<div class="k-default-codeblock">
```

```
</div>
 180/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 416ms/step - loss: 203.3092

<div class="k-default-codeblock">
```

```
</div>
 181/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 416ms/step - loss: 203.6179

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 416ms/step - loss: 203.9318

<div class="k-default-codeblock">
```

```
</div>
 183/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 416ms/step - loss: 204.2508

<div class="k-default-codeblock">
```

```
</div>
 184/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 416ms/step - loss: 204.5744

<div class="k-default-codeblock">
```

```
</div>
 185/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 416ms/step - loss: 204.9020

<div class="k-default-codeblock">
```

```
</div>
 186/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 416ms/step - loss: 205.2338

<div class="k-default-codeblock">
```

```
</div>
 187/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 415ms/step - loss: 205.5692

<div class="k-default-codeblock">
```

```
</div>
 188/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 415ms/step - loss: 205.9078

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 415ms/step - loss: 206.2492

<div class="k-default-codeblock">
```

```
</div>
 190/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 415ms/step - loss: 206.5931

<div class="k-default-codeblock">
```

```
</div>
 191/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 415ms/step - loss: 206.9393

<div class="k-default-codeblock">
```

```
</div>
 192/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 415ms/step - loss: 207.2872

<div class="k-default-codeblock">
```

```
</div>
 193/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 415ms/step - loss: 207.6366

<div class="k-default-codeblock">
```

```
</div>
 194/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 415ms/step - loss: 207.9871

<div class="k-default-codeblock">
```

```
</div>
 195/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 415ms/step - loss: 208.3384

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 415ms/step - loss: 208.6901

<div class="k-default-codeblock">
```

```
</div>
 197/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 415ms/step - loss: 209.0420

<div class="k-default-codeblock">
```

```
</div>
 198/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 415ms/step - loss: 209.3939

<div class="k-default-codeblock">
```

```
</div>
 199/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 415ms/step - loss: 209.7454

<div class="k-default-codeblock">
```

```
</div>
 200/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 415ms/step - loss: 210.0963

<div class="k-default-codeblock">
```

```
</div>
 201/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 415ms/step - loss: 210.4461

<div class="k-default-codeblock">
```

```
</div>
 202/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 415ms/step - loss: 210.7948

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 415ms/step - loss: 211.1421

<div class="k-default-codeblock">
```

```
</div>
 204/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 415ms/step - loss: 211.4877

<div class="k-default-codeblock">
```

```
</div>
 205/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 415ms/step - loss: 211.8313

<div class="k-default-codeblock">
```

```
</div>
 206/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 415ms/step - loss: 212.1726

<div class="k-default-codeblock">
```

```
</div>
 207/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 415ms/step - loss: 212.5119

<div class="k-default-codeblock">
```

```
</div>
 208/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 415ms/step - loss: 212.8489

<div class="k-default-codeblock">
```

```
</div>
 209/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 415ms/step - loss: 213.1834

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 415ms/step - loss: 213.5148

<div class="k-default-codeblock">
```

```
</div>
 211/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 415ms/step - loss: 213.8434

<div class="k-default-codeblock">
```

```
</div>
 212/250 ━━━━━━━━━━━━━━━━[37m━━━━  15s 415ms/step - loss: 214.1689

<div class="k-default-codeblock">
```

```
</div>
 213/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 415ms/step - loss: 214.4911

<div class="k-default-codeblock">
```

```
</div>
 214/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 415ms/step - loss: 214.8099

<div class="k-default-codeblock">
```

```
</div>
 215/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 415ms/step - loss: 215.1255

<div class="k-default-codeblock">
```

```
</div>
 216/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 415ms/step - loss: 215.4374

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 415ms/step - loss: 215.7454

<div class="k-default-codeblock">
```

```
</div>
 218/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 415ms/step - loss: 216.0494

<div class="k-default-codeblock">
```

```
</div>
 219/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 415ms/step - loss: 216.3493

<div class="k-default-codeblock">
```

```
</div>
 220/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 415ms/step - loss: 216.6452

<div class="k-default-codeblock">
```

```
</div>
 221/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 415ms/step - loss: 216.9370

<div class="k-default-codeblock">
```

```
</div>
 222/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 415ms/step - loss: 217.2246

<div class="k-default-codeblock">
```

```
</div>
 223/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 415ms/step - loss: 217.5080

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  10s 415ms/step - loss: 217.7871

<div class="k-default-codeblock">
```

```
</div>
 225/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 415ms/step - loss: 218.0619

<div class="k-default-codeblock">
```

```
</div>
 226/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 414ms/step - loss: 218.3815 

<div class="k-default-codeblock">
```

```
</div>
 227/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 414ms/step - loss: 218.6964

<div class="k-default-codeblock">
```

```
</div>
 228/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 415ms/step - loss: 219.0069

<div class="k-default-codeblock">
```

```
</div>
 229/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 415ms/step - loss: 219.3135

<div class="k-default-codeblock">
```

```
</div>
 230/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 415ms/step - loss: 219.6167

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 415ms/step - loss: 219.9170

<div class="k-default-codeblock">
```

```
</div>
 232/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 415ms/step - loss: 220.2149

<div class="k-default-codeblock">
```

```
</div>
 233/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 414ms/step - loss: 220.5108

<div class="k-default-codeblock">
```

```
</div>
 234/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 414ms/step - loss: 220.8053

<div class="k-default-codeblock">
```

```
</div>
 235/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 414ms/step - loss: 221.0987

<div class="k-default-codeblock">
```

```
</div>
 236/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 414ms/step - loss: 221.3913

<div class="k-default-codeblock">
```

```
</div>
 237/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 414ms/step - loss: 221.6837

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 415ms/step - loss: 221.9760

<div class="k-default-codeblock">
```

```
</div>
 239/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 415ms/step - loss: 222.2686

<div class="k-default-codeblock">
```

```
</div>
 240/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 415ms/step - loss: 222.5616

<div class="k-default-codeblock">
```

```
</div>
 241/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 415ms/step - loss: 222.8555

<div class="k-default-codeblock">
```

```
</div>
 242/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 414ms/step - loss: 223.1503

<div class="k-default-codeblock">
```

```
</div>
 243/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 414ms/step - loss: 223.4460

<div class="k-default-codeblock">
```

```
</div>
 244/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 414ms/step - loss: 223.7427

<div class="k-default-codeblock">
```

```
</div>
 245/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 414ms/step - loss: 224.0404

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 414ms/step - loss: 224.3391

<div class="k-default-codeblock">
```

```
</div>
 247/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 414ms/step - loss: 224.6389

<div class="k-default-codeblock">
```

```
</div>
 248/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 414ms/step - loss: 224.9396

<div class="k-default-codeblock">
```

```
</div>
 249/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 414ms/step - loss: 225.2412

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 414ms/step - loss: 225.5435

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 104s 414ms/step - loss: 225.8434


<div class="k-default-codeblock">
```
Epoch 3/10

```
</div>
    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  1:52 454ms/step - loss: 495.7987

<div class="k-default-codeblock">
```

```
</div>
   2/250 [37m━━━━━━━━━━━━━━━━━━━━  1:47 435ms/step - loss: 494.8696

<div class="k-default-codeblock">
```

```
</div>
   3/250 [37m━━━━━━━━━━━━━━━━━━━━  1:46 432ms/step - loss: 493.7273

<div class="k-default-codeblock">
```

```
</div>
   4/250 [37m━━━━━━━━━━━━━━━━━━━━  1:46 431ms/step - loss: 492.8675

<div class="k-default-codeblock">
```

```
</div>
   5/250 [37m━━━━━━━━━━━━━━━━━━━━  1:46 434ms/step - loss: 491.4964

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  1:44 430ms/step - loss: 489.9811

<div class="k-default-codeblock">
```

```
</div>
   7/250 [37m━━━━━━━━━━━━━━━━━━━━  1:43 427ms/step - loss: 488.1873

<div class="k-default-codeblock">
```

```
</div>
   8/250 [37m━━━━━━━━━━━━━━━━━━━━  1:42 424ms/step - loss: 486.3602

<div class="k-default-codeblock">
```

```
</div>
   9/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 421ms/step - loss: 484.4749

<div class="k-default-codeblock">
```

```
</div>
  10/250 [37m━━━━━━━━━━━━━━━━━━━━  1:39 416ms/step - loss: 482.5312

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  1:39 415ms/step - loss: 480.5388

<div class="k-default-codeblock">
```

```
</div>
  12/250 [37m━━━━━━━━━━━━━━━━━━━━  1:38 415ms/step - loss: 478.5017

<div class="k-default-codeblock">
```

```
</div>
  13/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:38 415ms/step - loss: 476.3973

<div class="k-default-codeblock">
```

```
</div>
  14/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 415ms/step - loss: 474.2900

<div class="k-default-codeblock">
```

```
</div>
  15/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 416ms/step - loss: 472.2240

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 415ms/step - loss: 470.0991

<div class="k-default-codeblock">
```

```
</div>
  17/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 414ms/step - loss: 467.9718

<div class="k-default-codeblock">
```

```
</div>
  18/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 413ms/step - loss: 465.8323

<div class="k-default-codeblock">
```

```
</div>
  19/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 414ms/step - loss: 463.6825

<div class="k-default-codeblock">
```

```
</div>
  20/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 414ms/step - loss: 461.5219

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 415ms/step - loss: 459.3542

<div class="k-default-codeblock">
```

```
</div>
  22/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 415ms/step - loss: 457.1796

<div class="k-default-codeblock">
```

```
</div>
  23/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 416ms/step - loss: 455.0059

<div class="k-default-codeblock">
```

```
</div>
  24/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:33 415ms/step - loss: 452.8408

<div class="k-default-codeblock">
```

```
</div>
  25/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:33 415ms/step - loss: 450.6788

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 414ms/step - loss: 448.5187

<div class="k-default-codeblock">
```

```
</div>
  27/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 414ms/step - loss: 446.3770

<div class="k-default-codeblock">
```

```
</div>
  28/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 415ms/step - loss: 444.2436

<div class="k-default-codeblock">
```

```
</div>
  29/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 415ms/step - loss: 442.1278

<div class="k-default-codeblock">
```

```
</div>
  30/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 415ms/step - loss: 440.0292

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 414ms/step - loss: 437.9384

<div class="k-default-codeblock">
```

```
</div>
  32/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 414ms/step - loss: 435.8581

<div class="k-default-codeblock">
```

```
</div>
  33/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 414ms/step - loss: 433.7884

<div class="k-default-codeblock">
```

```
</div>
  34/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 414ms/step - loss: 431.7317

<div class="k-default-codeblock">
```

```
</div>
  35/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 414ms/step - loss: 429.6883

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 414ms/step - loss: 427.6593

<div class="k-default-codeblock">
```

```
</div>
  37/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 414ms/step - loss: 425.6501

<div class="k-default-codeblock">
```

```
</div>
  38/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 413ms/step - loss: 423.6524

<div class="k-default-codeblock">
```

```
</div>
  39/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 414ms/step - loss: 421.6729

<div class="k-default-codeblock">
```

```
</div>
  40/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 413ms/step - loss: 419.7121

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 413ms/step - loss: 417.7712

<div class="k-default-codeblock">
```

```
</div>
  42/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 413ms/step - loss: 415.8497

<div class="k-default-codeblock">
```

```
</div>
  43/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 413ms/step - loss: 413.9522

<div class="k-default-codeblock">
```

```
</div>
  44/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 413ms/step - loss: 412.0774

<div class="k-default-codeblock">
```

```
</div>
  45/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 412ms/step - loss: 410.2212

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 412ms/step - loss: 408.3870

<div class="k-default-codeblock">
```

```
</div>
  47/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:23 412ms/step - loss: 406.5710

<div class="k-default-codeblock">
```

```
</div>
  48/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:23 412ms/step - loss: 404.7713

<div class="k-default-codeblock">
```

```
</div>
  49/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:22 412ms/step - loss: 402.9886

<div class="k-default-codeblock">
```

```
</div>
  50/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 412ms/step - loss: 401.2256

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 412ms/step - loss: 399.4842

<div class="k-default-codeblock">
```

```
</div>
  52/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 412ms/step - loss: 397.7682

<div class="k-default-codeblock">
```

```
</div>
  53/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 412ms/step - loss: 396.0712

<div class="k-default-codeblock">
```

```
</div>
  54/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 412ms/step - loss: 394.4041

<div class="k-default-codeblock">
```

```
</div>
  55/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 412ms/step - loss: 392.7572

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 412ms/step - loss: 391.1288

<div class="k-default-codeblock">
```

```
</div>
  57/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 411ms/step - loss: 389.5233

<div class="k-default-codeblock">
```

```
</div>
  58/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 412ms/step - loss: 387.9408

<div class="k-default-codeblock">
```

```
</div>
  59/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:18 412ms/step - loss: 386.3769

<div class="k-default-codeblock">
```

```
</div>
  60/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:18 412ms/step - loss: 384.8352

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:17 411ms/step - loss: 383.3161

<div class="k-default-codeblock">
```

```
</div>
  62/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:17 411ms/step - loss: 381.8172

<div class="k-default-codeblock">
```

```
</div>
  63/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 411ms/step - loss: 380.3367

<div class="k-default-codeblock">
```

```
</div>
  64/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 411ms/step - loss: 378.8751

<div class="k-default-codeblock">
```

```
</div>
  65/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 411ms/step - loss: 377.4348

<div class="k-default-codeblock">
```

```
</div>
  66/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 411ms/step - loss: 376.0145

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 411ms/step - loss: 374.6158

<div class="k-default-codeblock">
```

```
</div>
  68/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 412ms/step - loss: 373.2339

<div class="k-default-codeblock">
```

```
</div>
  69/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 412ms/step - loss: 371.8703

<div class="k-default-codeblock">
```

```
</div>
  70/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 412ms/step - loss: 370.5236

<div class="k-default-codeblock">
```

```
</div>
  71/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 412ms/step - loss: 369.1952

<div class="k-default-codeblock">
```

```
</div>
  72/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 412ms/step - loss: 367.8866

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:12 412ms/step - loss: 366.5981

<div class="k-default-codeblock">
```

```
</div>
  74/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:12 412ms/step - loss: 365.3262

<div class="k-default-codeblock">
```

```
</div>
  75/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 412ms/step - loss: 364.0735

<div class="k-default-codeblock">
```

```
</div>
  76/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 413ms/step - loss: 362.8384

<div class="k-default-codeblock">
```

```
</div>
  77/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 412ms/step - loss: 361.6190

<div class="k-default-codeblock">
```

```
</div>
  78/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 412ms/step - loss: 360.4145

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 412ms/step - loss: 359.2265

<div class="k-default-codeblock">
```

```
</div>
  80/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 413ms/step - loss: 358.0531

<div class="k-default-codeblock">
```

```
</div>
  81/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 413ms/step - loss: 356.8968

<div class="k-default-codeblock">
```

```
</div>
  82/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 412ms/step - loss: 355.7571

<div class="k-default-codeblock">
```

```
</div>
  83/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 412ms/step - loss: 354.6343

<div class="k-default-codeblock">
```

```
</div>
  84/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 412ms/step - loss: 353.5269

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 412ms/step - loss: 352.4331

<div class="k-default-codeblock">
```

```
</div>
  86/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:07 412ms/step - loss: 351.3546

<div class="k-default-codeblock">
```

```
</div>
  87/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:07 412ms/step - loss: 350.2896

<div class="k-default-codeblock">
```

```
</div>
  88/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 412ms/step - loss: 349.2373

<div class="k-default-codeblock">
```

```
</div>
  89/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 412ms/step - loss: 348.1985

<div class="k-default-codeblock">
```

```
</div>
  90/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 412ms/step - loss: 347.3675

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 413ms/step - loss: 346.5445

<div class="k-default-codeblock">
```

```
</div>
  92/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 413ms/step - loss: 345.7323

<div class="k-default-codeblock">
```

```
</div>
  93/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 413ms/step - loss: 344.9333

<div class="k-default-codeblock">
```

```
</div>
  94/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 413ms/step - loss: 344.1494

<div class="k-default-codeblock">
```

```
</div>
  95/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 413ms/step - loss: 343.3821

<div class="k-default-codeblock">
```

```
</div>
  96/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 413ms/step - loss: 342.6351

<div class="k-default-codeblock">
```

```
</div>
  97/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 413ms/step - loss: 341.9112

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:02 413ms/step - loss: 341.2122

<div class="k-default-codeblock">
```

```
</div>
  99/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:02 413ms/step - loss: 340.5393

<div class="k-default-codeblock">
```

```
</div>
 100/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 413ms/step - loss: 339.8941

<div class="k-default-codeblock">
```

```
</div>
 101/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 414ms/step - loss: 339.2773

<div class="k-default-codeblock">
```

```
</div>
 102/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 415ms/step - loss: 338.6903

<div class="k-default-codeblock">
```

```
</div>
 103/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 415ms/step - loss: 338.1334

<div class="k-default-codeblock">
```

```
</div>
 104/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 414ms/step - loss: 337.6076

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 414ms/step - loss: 337.1126

<div class="k-default-codeblock">
```

```
</div>
 106/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 414ms/step - loss: 336.6477 

<div class="k-default-codeblock">
```

```
</div>
 107/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 414ms/step - loss: 336.2123

<div class="k-default-codeblock">
```

```
</div>
 108/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 414ms/step - loss: 335.8059

<div class="k-default-codeblock">
```

```
</div>
 109/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 414ms/step - loss: 335.4285

<div class="k-default-codeblock">
```

```
</div>
 110/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 413ms/step - loss: 335.0782

<div class="k-default-codeblock">
```

```
</div>
 111/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 413ms/step - loss: 334.7551

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 413ms/step - loss: 334.4582

<div class="k-default-codeblock">
```

```
</div>
 113/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 413ms/step - loss: 334.1861

<div class="k-default-codeblock">
```

```
</div>
 114/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 413ms/step - loss: 333.9383

<div class="k-default-codeblock">
```

```
</div>
 115/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 413ms/step - loss: 333.7141

<div class="k-default-codeblock">
```

```
</div>
 116/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 413ms/step - loss: 333.5119

<div class="k-default-codeblock">
```

```
</div>
 117/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 412ms/step - loss: 333.3299

<div class="k-default-codeblock">
```

```
</div>
 118/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 412ms/step - loss: 333.1669

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 412ms/step - loss: 333.0220

<div class="k-default-codeblock">
```

```
</div>
 120/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 412ms/step - loss: 332.8948

<div class="k-default-codeblock">
```

```
</div>
 121/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 412ms/step - loss: 332.7838

<div class="k-default-codeblock">
```

```
</div>
 122/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 412ms/step - loss: 332.6875

<div class="k-default-codeblock">
```

```
</div>
 123/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 412ms/step - loss: 332.6048

<div class="k-default-codeblock">
```

```
</div>
 124/250 ━━━━━━━━━[37m━━━━━━━━━━━  51s 412ms/step - loss: 332.5355

<div class="k-default-codeblock">
```

```
</div>
 125/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 412ms/step - loss: 332.4789

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 412ms/step - loss: 332.4327

<div class="k-default-codeblock">
```

```
</div>
 127/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 412ms/step - loss: 332.3969

<div class="k-default-codeblock">
```

```
</div>
 128/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 412ms/step - loss: 332.3708

<div class="k-default-codeblock">
```

```
</div>
 129/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 412ms/step - loss: 332.3533

<div class="k-default-codeblock">
```

```
</div>
 130/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 412ms/step - loss: 332.3432

<div class="k-default-codeblock">
```

```
</div>
 131/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 412ms/step - loss: 332.3401

<div class="k-default-codeblock">
```

```
</div>
 132/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 412ms/step - loss: 332.3431

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 411ms/step - loss: 332.3511

<div class="k-default-codeblock">
```

```
</div>
 134/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 412ms/step - loss: 332.3632

<div class="k-default-codeblock">
```

```
</div>
 135/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 412ms/step - loss: 332.3791

<div class="k-default-codeblock">
```

```
</div>
 136/250 ━━━━━━━━━━[37m━━━━━━━━━━  46s 412ms/step - loss: 332.3976

<div class="k-default-codeblock">
```

```
</div>
 137/250 ━━━━━━━━━━[37m━━━━━━━━━━  46s 412ms/step - loss: 332.4176

<div class="k-default-codeblock">
```

```
</div>
 138/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 411ms/step - loss: 332.4392

<div class="k-default-codeblock">
```

```
</div>
 139/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 411ms/step - loss: 332.4614

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 411ms/step - loss: 332.4839

<div class="k-default-codeblock">
```

```
</div>
 141/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 411ms/step - loss: 332.5063

<div class="k-default-codeblock">
```

```
</div>
 142/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 411ms/step - loss: 332.5280

<div class="k-default-codeblock">
```

```
</div>
 143/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 411ms/step - loss: 332.5485

<div class="k-default-codeblock">
```

```
</div>
 144/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 411ms/step - loss: 332.5671

<div class="k-default-codeblock">
```

```
</div>
 145/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 411ms/step - loss: 332.5833

<div class="k-default-codeblock">
```

```
</div>
 146/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 410ms/step - loss: 332.5967

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 410ms/step - loss: 332.6078

<div class="k-default-codeblock">
```

```
</div>
 148/250 ━━━━━━━━━━━[37m━━━━━━━━━  41s 411ms/step - loss: 332.6156

<div class="k-default-codeblock">
```

```
</div>
 149/250 ━━━━━━━━━━━[37m━━━━━━━━━  41s 411ms/step - loss: 332.6199

<div class="k-default-codeblock">
```

```
</div>
 150/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 411ms/step - loss: 332.6202

<div class="k-default-codeblock">
```

```
</div>
 151/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 411ms/step - loss: 332.6164

<div class="k-default-codeblock">
```

```
</div>
 152/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 411ms/step - loss: 332.6082

<div class="k-default-codeblock">
```

```
</div>
 153/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 410ms/step - loss: 332.5954

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 410ms/step - loss: 332.5781

<div class="k-default-codeblock">
```

```
</div>
 155/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 410ms/step - loss: 332.5559

<div class="k-default-codeblock">
```

```
</div>
 156/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 410ms/step - loss: 332.5287

<div class="k-default-codeblock">
```

```
</div>
 157/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 410ms/step - loss: 332.4969

<div class="k-default-codeblock">
```

```
</div>
 158/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 410ms/step - loss: 332.4605

<div class="k-default-codeblock">
```

```
</div>
 159/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 410ms/step - loss: 332.4196

<div class="k-default-codeblock">
```

```
</div>
 160/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 410ms/step - loss: 332.3740

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 410ms/step - loss: 332.3274

<div class="k-default-codeblock">
```

```
</div>
 162/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 410ms/step - loss: 332.2766

<div class="k-default-codeblock">
```

```
</div>
 163/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 410ms/step - loss: 332.2218

<div class="k-default-codeblock">
```

```
</div>
 164/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 410ms/step - loss: 332.1630

<div class="k-default-codeblock">
```

```
</div>
 165/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 410ms/step - loss: 332.1009

<div class="k-default-codeblock">
```

```
</div>
 166/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 409ms/step - loss: 332.0354

<div class="k-default-codeblock">
```

```
</div>
 167/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 409ms/step - loss: 331.9666

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 409ms/step - loss: 331.8947

<div class="k-default-codeblock">
```

```
</div>
 169/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 409ms/step - loss: 331.8199

<div class="k-default-codeblock">
```

```
</div>
 170/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 409ms/step - loss: 331.7424

<div class="k-default-codeblock">
```

```
</div>
 171/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 409ms/step - loss: 331.6625

<div class="k-default-codeblock">
```

```
</div>
 172/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 409ms/step - loss: 331.5803

<div class="k-default-codeblock">
```

```
</div>
 173/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 408ms/step - loss: 331.4958

<div class="k-default-codeblock">
```

```
</div>
 174/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 408ms/step - loss: 331.4089

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 408ms/step - loss: 331.3201

<div class="k-default-codeblock">
```

```
</div>
 176/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 408ms/step - loss: 331.2295

<div class="k-default-codeblock">
```

```
</div>
 177/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 408ms/step - loss: 331.1369

<div class="k-default-codeblock">
```

```
</div>
 178/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 408ms/step - loss: 331.0426

<div class="k-default-codeblock">
```

```
</div>
 179/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 407ms/step - loss: 330.9467

<div class="k-default-codeblock">
```

```
</div>
 180/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 407ms/step - loss: 330.8492

<div class="k-default-codeblock">
```

```
</div>
 181/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 407ms/step - loss: 330.7500

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 407ms/step - loss: 330.6496

<div class="k-default-codeblock">
```

```
</div>
 183/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 407ms/step - loss: 330.5479

<div class="k-default-codeblock">
```

```
</div>
 184/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 407ms/step - loss: 330.4449

<div class="k-default-codeblock">
```

```
</div>
 185/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 407ms/step - loss: 330.3405

<div class="k-default-codeblock">
```

```
</div>
 186/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 407ms/step - loss: 330.2346

<div class="k-default-codeblock">
```

```
</div>
 187/250 ━━━━━━━━━━━━━━[37m━━━━━━  25s 407ms/step - loss: 330.1272

<div class="k-default-codeblock">
```

```
</div>
 188/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 407ms/step - loss: 330.0183

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 407ms/step - loss: 329.9078

<div class="k-default-codeblock">
```

```
</div>
 190/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 407ms/step - loss: 329.7960

<div class="k-default-codeblock">
```

```
</div>
 191/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 407ms/step - loss: 329.6832

<div class="k-default-codeblock">
```

```
</div>
 192/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 406ms/step - loss: 329.5692

<div class="k-default-codeblock">
```

```
</div>
 193/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 406ms/step - loss: 329.4538

<div class="k-default-codeblock">
```

```
</div>
 194/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 406ms/step - loss: 329.3373

<div class="k-default-codeblock">
```

```
</div>
 195/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 406ms/step - loss: 329.2197

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 406ms/step - loss: 329.1009

<div class="k-default-codeblock">
```

```
</div>
 197/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 406ms/step - loss: 328.9808

<div class="k-default-codeblock">
```

```
</div>
 198/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 406ms/step - loss: 328.8595

<div class="k-default-codeblock">
```

```
</div>
 199/250 ━━━━━━━━━━━━━━━[37m━━━━━  20s 406ms/step - loss: 328.7371

<div class="k-default-codeblock">
```

```
</div>
 200/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 406ms/step - loss: 328.6134

<div class="k-default-codeblock">
```

```
</div>
 201/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 405ms/step - loss: 328.4886

<div class="k-default-codeblock">
```

```
</div>
 202/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 405ms/step - loss: 328.3626

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 405ms/step - loss: 328.2355

<div class="k-default-codeblock">
```

```
</div>
 204/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 405ms/step - loss: 328.1075

<div class="k-default-codeblock">
```

```
</div>
 205/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 405ms/step - loss: 327.9783

<div class="k-default-codeblock">
```

```
</div>
 206/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 405ms/step - loss: 327.8479

<div class="k-default-codeblock">
```

```
</div>
 207/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 405ms/step - loss: 327.7163

<div class="k-default-codeblock">
```

```
</div>
 208/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 405ms/step - loss: 327.5840

<div class="k-default-codeblock">
```

```
</div>
 209/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 405ms/step - loss: 327.4511

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 405ms/step - loss: 327.3169

<div class="k-default-codeblock">
```

```
</div>
 211/250 ━━━━━━━━━━━━━━━━[37m━━━━  15s 405ms/step - loss: 327.1818

<div class="k-default-codeblock">
```

```
</div>
 212/250 ━━━━━━━━━━━━━━━━[37m━━━━  15s 405ms/step - loss: 327.0457

<div class="k-default-codeblock">
```

```
</div>
 213/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 405ms/step - loss: 326.9087

<div class="k-default-codeblock">
```

```
</div>
 214/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 405ms/step - loss: 326.7707

<div class="k-default-codeblock">
```

```
</div>
 215/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 405ms/step - loss: 326.6318

<div class="k-default-codeblock">
```

```
</div>
 216/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 405ms/step - loss: 326.4921

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 405ms/step - loss: 326.3514

<div class="k-default-codeblock">
```

```
</div>
 218/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 405ms/step - loss: 326.2099

<div class="k-default-codeblock">
```

```
</div>
 219/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 405ms/step - loss: 326.0676

<div class="k-default-codeblock">
```

```
</div>
 220/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 405ms/step - loss: 325.9243

<div class="k-default-codeblock">
```

```
</div>
 221/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 405ms/step - loss: 325.7802

<div class="k-default-codeblock">
```

```
</div>
 222/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 405ms/step - loss: 325.6354

<div class="k-default-codeblock">
```

```
</div>
 223/250 ━━━━━━━━━━━━━━━━━[37m━━━  10s 404ms/step - loss: 325.4898

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  10s 404ms/step - loss: 325.3434

<div class="k-default-codeblock">
```

```
</div>
 225/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 404ms/step - loss: 325.1963

<div class="k-default-codeblock">
```

```
</div>
 226/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 404ms/step - loss: 325.0484 

<div class="k-default-codeblock">
```

```
</div>
 227/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 404ms/step - loss: 324.9001

<div class="k-default-codeblock">
```

```
</div>
 228/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 404ms/step - loss: 324.7509

<div class="k-default-codeblock">
```

```
</div>
 229/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 404ms/step - loss: 324.6011

<div class="k-default-codeblock">
```

```
</div>
 230/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 404ms/step - loss: 324.4504

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 404ms/step - loss: 324.2990

<div class="k-default-codeblock">
```

```
</div>
 232/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 404ms/step - loss: 324.1469

<div class="k-default-codeblock">
```

```
</div>
 233/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 404ms/step - loss: 323.9941

<div class="k-default-codeblock">
```

```
</div>
 234/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 404ms/step - loss: 323.8406

<div class="k-default-codeblock">
```

```
</div>
 235/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 404ms/step - loss: 323.6865

<div class="k-default-codeblock">
```

```
</div>
 236/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 404ms/step - loss: 323.5320

<div class="k-default-codeblock">
```

```
</div>
 237/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 403ms/step - loss: 323.3767

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 403ms/step - loss: 323.2209

<div class="k-default-codeblock">
```

```
</div>
 239/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 403ms/step - loss: 323.0648

<div class="k-default-codeblock">
```

```
</div>
 240/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 403ms/step - loss: 322.9081

<div class="k-default-codeblock">
```

```
</div>
 241/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 403ms/step - loss: 322.7509

<div class="k-default-codeblock">
```

```
</div>
 242/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 403ms/step - loss: 322.5933

<div class="k-default-codeblock">
```

```
</div>
 243/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 403ms/step - loss: 322.4350

<div class="k-default-codeblock">
```

```
</div>
 244/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 403ms/step - loss: 322.2763

<div class="k-default-codeblock">
```

```
</div>
 245/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 403ms/step - loss: 322.1170

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 403ms/step - loss: 321.9573

<div class="k-default-codeblock">
```

```
</div>
 247/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 403ms/step - loss: 321.7974

<div class="k-default-codeblock">
```

```
</div>
 248/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 403ms/step - loss: 321.6370

<div class="k-default-codeblock">
```

```
</div>
 249/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 403ms/step - loss: 321.4762

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 402ms/step - loss: 321.3149

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 101s 403ms/step - loss: 321.1549


<div class="k-default-codeblock">
```
Epoch 4/10

```
</div>
    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  1:48 436ms/step - loss: 178.5828

<div class="k-default-codeblock">
```

```
</div>
   2/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 405ms/step - loss: 179.6100

<div class="k-default-codeblock">
```

```
</div>
   3/250 [37m━━━━━━━━━━━━━━━━━━━━  1:37 393ms/step - loss: 178.9901

<div class="k-default-codeblock">
```

```
</div>
   4/250 [37m━━━━━━━━━━━━━━━━━━━━  1:36 391ms/step - loss: 179.4316

<div class="k-default-codeblock">
```

```
</div>
   5/250 [37m━━━━━━━━━━━━━━━━━━━━  1:35 391ms/step - loss: 179.3869

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  1:35 390ms/step - loss: 179.3118

<div class="k-default-codeblock">
```

```
</div>
   7/250 [37m━━━━━━━━━━━━━━━━━━━━  1:33 384ms/step - loss: 179.1437

<div class="k-default-codeblock">
```

```
</div>
   8/250 [37m━━━━━━━━━━━━━━━━━━━━  1:32 384ms/step - loss: 178.9843

<div class="k-default-codeblock">
```

```
</div>
   9/250 [37m━━━━━━━━━━━━━━━━━━━━  1:31 380ms/step - loss: 178.7291

<div class="k-default-codeblock">
```

```
</div>
  10/250 [37m━━━━━━━━━━━━━━━━━━━━  1:31 380ms/step - loss: 178.5738

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  1:31 384ms/step - loss: 178.4117

<div class="k-default-codeblock">
```

```
</div>
  12/250 [37m━━━━━━━━━━━━━━━━━━━━  1:31 382ms/step - loss: 178.1950

<div class="k-default-codeblock">
```

```
</div>
  13/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:30 383ms/step - loss: 177.9536

<div class="k-default-codeblock">
```

```
</div>
  14/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:30 383ms/step - loss: 177.7314

<div class="k-default-codeblock">
```

```
</div>
  15/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:30 386ms/step - loss: 177.5361

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:30 387ms/step - loss: 177.3954

<div class="k-default-codeblock">
```

```
</div>
  17/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:30 388ms/step - loss: 177.2744

<div class="k-default-codeblock">
```

```
</div>
  18/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:29 387ms/step - loss: 177.8696

<div class="k-default-codeblock">
```

```
</div>
  19/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:29 388ms/step - loss: 178.3914

<div class="k-default-codeblock">
```

```
</div>
  20/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:29 389ms/step - loss: 178.8593

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:29 389ms/step - loss: 179.2904

<div class="k-default-codeblock">
```

```
</div>
  22/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:28 389ms/step - loss: 179.7068

<div class="k-default-codeblock">
```

```
</div>
  23/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:28 388ms/step - loss: 180.1333

<div class="k-default-codeblock">
```

```
</div>
  24/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:27 388ms/step - loss: 180.5986

<div class="k-default-codeblock">
```

```
</div>
  25/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:27 387ms/step - loss: 181.0998

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:26 388ms/step - loss: 181.6656

<div class="k-default-codeblock">
```

```
</div>
  27/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:26 388ms/step - loss: 182.2796

<div class="k-default-codeblock">
```

```
</div>
  28/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:26 389ms/step - loss: 182.9467

<div class="k-default-codeblock">
```

```
</div>
  29/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:25 389ms/step - loss: 183.6573

<div class="k-default-codeblock">
```

```
</div>
  30/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:25 388ms/step - loss: 184.4168

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:25 388ms/step - loss: 185.2211

<div class="k-default-codeblock">
```

```
</div>
  32/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:24 389ms/step - loss: 186.0683

<div class="k-default-codeblock">
```

```
</div>
  33/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:24 390ms/step - loss: 186.9464

<div class="k-default-codeblock">
```

```
</div>
  34/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:24 390ms/step - loss: 187.8541

<div class="k-default-codeblock">
```

```
</div>
  35/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:23 390ms/step - loss: 188.7983

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:23 391ms/step - loss: 189.7666

<div class="k-default-codeblock">
```

```
</div>
  37/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:23 391ms/step - loss: 190.7489

<div class="k-default-codeblock">
```

```
</div>
  38/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:22 391ms/step - loss: 191.7415

<div class="k-default-codeblock">
```

```
</div>
  39/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:22 391ms/step - loss: 192.7368

<div class="k-default-codeblock">
```

```
</div>
  40/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:22 392ms/step - loss: 193.7366

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:21 392ms/step - loss: 194.7372

<div class="k-default-codeblock">
```

```
</div>
  42/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:21 392ms/step - loss: 195.7315

<div class="k-default-codeblock">
```

```
</div>
  43/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:20 391ms/step - loss: 196.7168

<div class="k-default-codeblock">
```

```
</div>
  44/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:20 391ms/step - loss: 197.6899

<div class="k-default-codeblock">
```

```
</div>
  45/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:20 391ms/step - loss: 198.6506

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:19 391ms/step - loss: 199.5946

<div class="k-default-codeblock">
```

```
</div>
  47/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:19 390ms/step - loss: 200.5223

<div class="k-default-codeblock">
```

```
</div>
  48/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:18 390ms/step - loss: 201.4281

<div class="k-default-codeblock">
```

```
</div>
  49/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:18 390ms/step - loss: 202.3140

<div class="k-default-codeblock">
```

```
</div>
  50/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:18 391ms/step - loss: 203.1731

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:17 391ms/step - loss: 204.0074

<div class="k-default-codeblock">
```

```
</div>
  52/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:17 391ms/step - loss: 204.8244

<div class="k-default-codeblock">
```

```
</div>
  53/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:17 391ms/step - loss: 205.6193

<div class="k-default-codeblock">
```

```
</div>
  54/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:16 391ms/step - loss: 206.3965

<div class="k-default-codeblock">
```

```
</div>
  55/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:16 391ms/step - loss: 207.1523

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:15 391ms/step - loss: 207.8861

<div class="k-default-codeblock">
```

```
</div>
  57/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:15 391ms/step - loss: 208.5963

<div class="k-default-codeblock">
```

```
</div>
  58/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:15 391ms/step - loss: 209.2856

<div class="k-default-codeblock">
```

```
</div>
  59/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:14 392ms/step - loss: 209.9551

<div class="k-default-codeblock">
```

```
</div>
  60/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:14 392ms/step - loss: 210.6025

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:14 393ms/step - loss: 211.2273

<div class="k-default-codeblock">
```

```
</div>
  62/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:14 394ms/step - loss: 211.8279

<div class="k-default-codeblock">
```

```
</div>
  63/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 395ms/step - loss: 212.4052

<div class="k-default-codeblock">
```

```
</div>
  64/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 397ms/step - loss: 212.9616

<div class="k-default-codeblock">
```

```
</div>
  65/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 397ms/step - loss: 213.4984

<div class="k-default-codeblock">
```

```
</div>
  66/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 397ms/step - loss: 214.0156

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:12 397ms/step - loss: 214.5090

<div class="k-default-codeblock">
```

```
</div>
  68/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:12 397ms/step - loss: 214.9814

<div class="k-default-codeblock">
```

```
</div>
  69/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:11 398ms/step - loss: 215.4339

<div class="k-default-codeblock">
```

```
</div>
  70/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:11 397ms/step - loss: 215.8659

<div class="k-default-codeblock">
```

```
</div>
  71/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:11 397ms/step - loss: 216.2795

<div class="k-default-codeblock">
```

```
</div>
  72/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:10 397ms/step - loss: 216.6750

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:10 397ms/step - loss: 217.0520

<div class="k-default-codeblock">
```

```
</div>
  74/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:09 397ms/step - loss: 217.4095

<div class="k-default-codeblock">
```

```
</div>
  75/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 397ms/step - loss: 217.7484

<div class="k-default-codeblock">
```

```
</div>
  76/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 397ms/step - loss: 218.0720

<div class="k-default-codeblock">
```

```
</div>
  77/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 397ms/step - loss: 218.3791

<div class="k-default-codeblock">
```

```
</div>
  78/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 397ms/step - loss: 218.6699

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:07 397ms/step - loss: 218.9672

<div class="k-default-codeblock">
```

```
</div>
  80/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:07 396ms/step - loss: 219.2498

<div class="k-default-codeblock">
```

```
</div>
  81/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:06 396ms/step - loss: 219.5194

<div class="k-default-codeblock">
```

```
</div>
  82/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:06 396ms/step - loss: 219.7801

<div class="k-default-codeblock">
```

```
</div>
  83/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:06 396ms/step - loss: 220.0320

<div class="k-default-codeblock">
```

```
</div>
  84/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:05 397ms/step - loss: 220.2764

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:05 397ms/step - loss: 220.5133

<div class="k-default-codeblock">
```

```
</div>
  86/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:05 396ms/step - loss: 220.7444

<div class="k-default-codeblock">
```

```
</div>
  87/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:04 396ms/step - loss: 220.9696

<div class="k-default-codeblock">
```

```
</div>
  88/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 396ms/step - loss: 221.1908

<div class="k-default-codeblock">
```

```
</div>
  89/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 396ms/step - loss: 221.4100

<div class="k-default-codeblock">
```

```
</div>
  90/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 396ms/step - loss: 221.6263

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 397ms/step - loss: 221.8412

<div class="k-default-codeblock">
```

```
</div>
  92/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:02 397ms/step - loss: 222.0556

<div class="k-default-codeblock">
```

```
</div>
  93/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:02 397ms/step - loss: 222.2681

<div class="k-default-codeblock">
```

```
</div>
  94/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:01 397ms/step - loss: 222.4789

<div class="k-default-codeblock">
```

```
</div>
  95/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:01 397ms/step - loss: 222.6892

<div class="k-default-codeblock">
```

```
</div>
  96/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:01 398ms/step - loss: 222.8991

<div class="k-default-codeblock">
```

```
</div>
  97/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:00 398ms/step - loss: 223.1096

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:00 398ms/step - loss: 223.3206

<div class="k-default-codeblock">
```

```
</div>
  99/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:00 398ms/step - loss: 223.5312

<div class="k-default-codeblock">
```

```
</div>
 100/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 398ms/step - loss: 223.7407 

<div class="k-default-codeblock">
```

```
</div>
 101/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 398ms/step - loss: 223.9497

<div class="k-default-codeblock">
```

```
</div>
 102/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 398ms/step - loss: 224.1570

<div class="k-default-codeblock">
```

```
</div>
 103/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 399ms/step - loss: 224.3632

<div class="k-default-codeblock">
```

```
</div>
 104/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 398ms/step - loss: 224.5684

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 398ms/step - loss: 224.7726

<div class="k-default-codeblock">
```

```
</div>
 106/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 398ms/step - loss: 224.9745

<div class="k-default-codeblock">
```

```
</div>
 107/250 ━━━━━━━━[37m━━━━━━━━━━━━  56s 399ms/step - loss: 225.1745

<div class="k-default-codeblock">
```

```
</div>
 108/250 ━━━━━━━━[37m━━━━━━━━━━━━  56s 398ms/step - loss: 225.3721

<div class="k-default-codeblock">
```

```
</div>
 109/250 ━━━━━━━━[37m━━━━━━━━━━━━  56s 398ms/step - loss: 225.5677

<div class="k-default-codeblock">
```

```
</div>
 110/250 ━━━━━━━━[37m━━━━━━━━━━━━  55s 398ms/step - loss: 225.7601

<div class="k-default-codeblock">
```

```
</div>
 111/250 ━━━━━━━━[37m━━━━━━━━━━━━  55s 399ms/step - loss: 225.9502

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  55s 399ms/step - loss: 226.1383

<div class="k-default-codeblock">
```

```
</div>
 113/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 399ms/step - loss: 226.3233

<div class="k-default-codeblock">
```

```
</div>
 114/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 399ms/step - loss: 226.5056

<div class="k-default-codeblock">
```

```
</div>
 115/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 399ms/step - loss: 226.6861

<div class="k-default-codeblock">
```

```
</div>
 116/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 399ms/step - loss: 226.8636

<div class="k-default-codeblock">
```

```
</div>
 117/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 399ms/step - loss: 227.0379

<div class="k-default-codeblock">
```

```
</div>
 118/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 399ms/step - loss: 227.2088

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 399ms/step - loss: 227.3763

<div class="k-default-codeblock">
```

```
</div>
 120/250 ━━━━━━━━━[37m━━━━━━━━━━━  51s 399ms/step - loss: 227.5403

<div class="k-default-codeblock">
```

```
</div>
 121/250 ━━━━━━━━━[37m━━━━━━━━━━━  51s 400ms/step - loss: 227.7012

<div class="k-default-codeblock">
```

```
</div>
 122/250 ━━━━━━━━━[37m━━━━━━━━━━━  51s 399ms/step - loss: 227.8584

<div class="k-default-codeblock">
```

```
</div>
 123/250 ━━━━━━━━━[37m━━━━━━━━━━━  50s 400ms/step - loss: 228.0118

<div class="k-default-codeblock">
```

```
</div>
 124/250 ━━━━━━━━━[37m━━━━━━━━━━━  50s 399ms/step - loss: 228.1609

<div class="k-default-codeblock">
```

```
</div>
 125/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 400ms/step - loss: 228.3063

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 399ms/step - loss: 228.4473

<div class="k-default-codeblock">
```

```
</div>
 127/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 399ms/step - loss: 228.5846

<div class="k-default-codeblock">
```

```
</div>
 128/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 399ms/step - loss: 228.7184

<div class="k-default-codeblock">
```

```
</div>
 129/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 399ms/step - loss: 228.8484

<div class="k-default-codeblock">
```

```
</div>
 130/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 399ms/step - loss: 228.9741

<div class="k-default-codeblock">
```

```
</div>
 131/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 399ms/step - loss: 229.0955

<div class="k-default-codeblock">
```

```
</div>
 132/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 399ms/step - loss: 229.2126

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  46s 399ms/step - loss: 229.3257

<div class="k-default-codeblock">
```

```
</div>
 134/250 ━━━━━━━━━━[37m━━━━━━━━━━  46s 399ms/step - loss: 229.4352

<div class="k-default-codeblock">
```

```
</div>
 135/250 ━━━━━━━━━━[37m━━━━━━━━━━  45s 399ms/step - loss: 229.5408

<div class="k-default-codeblock">
```

```
</div>
 136/250 ━━━━━━━━━━[37m━━━━━━━━━━  45s 400ms/step - loss: 229.6421

<div class="k-default-codeblock">
```

```
</div>
 137/250 ━━━━━━━━━━[37m━━━━━━━━━━  45s 400ms/step - loss: 229.7392

<div class="k-default-codeblock">
```

```
</div>
 138/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 400ms/step - loss: 229.8323

<div class="k-default-codeblock">
```

```
</div>
 139/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 400ms/step - loss: 229.9210

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 400ms/step - loss: 230.0056

<div class="k-default-codeblock">
```

```
</div>
 141/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 400ms/step - loss: 230.0856

<div class="k-default-codeblock">
```

```
</div>
 142/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 401ms/step - loss: 230.1613

<div class="k-default-codeblock">
```

```
</div>
 143/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 401ms/step - loss: 230.2324

<div class="k-default-codeblock">
```

```
</div>
 144/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 401ms/step - loss: 230.3000

<div class="k-default-codeblock">
```

```
</div>
 145/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 401ms/step - loss: 230.3635

<div class="k-default-codeblock">
```

```
</div>
 146/250 ━━━━━━━━━━━[37m━━━━━━━━━  41s 401ms/step - loss: 230.6203

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  41s 401ms/step - loss: 230.8694

<div class="k-default-codeblock">
```

```
</div>
 148/250 ━━━━━━━━━━━[37m━━━━━━━━━  40s 401ms/step - loss: 231.1123

<div class="k-default-codeblock">
```

```
</div>
 149/250 ━━━━━━━━━━━[37m━━━━━━━━━  40s 401ms/step - loss: 231.3506

<div class="k-default-codeblock">
```

```
</div>
 150/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 401ms/step - loss: 231.5850

<div class="k-default-codeblock">
```

```
</div>
 151/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 401ms/step - loss: 231.8168

<div class="k-default-codeblock">
```

```
</div>
 152/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 401ms/step - loss: 232.0476

<div class="k-default-codeblock">
```

```
</div>
 153/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 401ms/step - loss: 232.2789

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 401ms/step - loss: 232.5118

<div class="k-default-codeblock">
```

```
</div>
 155/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 401ms/step - loss: 232.7468

<div class="k-default-codeblock">
```

```
</div>
 156/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 401ms/step - loss: 232.9846

<div class="k-default-codeblock">
```

```
</div>
 157/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 401ms/step - loss: 233.2263

<div class="k-default-codeblock">
```

```
</div>
 158/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 401ms/step - loss: 233.4728

<div class="k-default-codeblock">
```

```
</div>
 159/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 401ms/step - loss: 233.7243

<div class="k-default-codeblock">
```

```
</div>
 160/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 401ms/step - loss: 233.9812

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  35s 401ms/step - loss: 234.2435

<div class="k-default-codeblock">
```

```
</div>
 162/250 ━━━━━━━━━━━━[37m━━━━━━━━  35s 401ms/step - loss: 234.5116

<div class="k-default-codeblock">
```

```
</div>
 163/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 400ms/step - loss: 234.7855

<div class="k-default-codeblock">
```

```
</div>
 164/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 400ms/step - loss: 235.0654

<div class="k-default-codeblock">
```

```
</div>
 165/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 400ms/step - loss: 235.3510

<div class="k-default-codeblock">
```

```
</div>
 166/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 400ms/step - loss: 235.6424

<div class="k-default-codeblock">
```

```
</div>
 167/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 401ms/step - loss: 235.9394

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 401ms/step - loss: 236.2414

<div class="k-default-codeblock">
```

```
</div>
 169/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 401ms/step - loss: 236.5484

<div class="k-default-codeblock">
```

```
</div>
 170/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 401ms/step - loss: 236.8599

<div class="k-default-codeblock">
```

```
</div>
 171/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 401ms/step - loss: 237.1758

<div class="k-default-codeblock">
```

```
</div>
 172/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 401ms/step - loss: 237.4952

<div class="k-default-codeblock">
```

```
</div>
 173/250 ━━━━━━━━━━━━━[37m━━━━━━━  30s 401ms/step - loss: 237.8177

<div class="k-default-codeblock">
```

```
</div>
 174/250 ━━━━━━━━━━━━━[37m━━━━━━━  30s 401ms/step - loss: 238.1427

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 401ms/step - loss: 238.4700

<div class="k-default-codeblock">
```

```
</div>
 176/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 401ms/step - loss: 238.7992

<div class="k-default-codeblock">
```

```
</div>
 177/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 401ms/step - loss: 239.1299

<div class="k-default-codeblock">
```

```
</div>
 178/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 401ms/step - loss: 239.4617

<div class="k-default-codeblock">
```

```
</div>
 179/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 401ms/step - loss: 239.7945

<div class="k-default-codeblock">
```

```
</div>
 180/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 401ms/step - loss: 240.1274

<div class="k-default-codeblock">
```

```
</div>
 181/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 401ms/step - loss: 240.4603

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 402ms/step - loss: 240.7926

<div class="k-default-codeblock">
```

```
</div>
 183/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 402ms/step - loss: 241.1242

<div class="k-default-codeblock">
```

```
</div>
 184/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 402ms/step - loss: 241.4550

<div class="k-default-codeblock">
```

```
</div>
 185/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 402ms/step - loss: 241.7845

<div class="k-default-codeblock">
```

```
</div>
 186/250 ━━━━━━━━━━━━━━[37m━━━━━━  25s 402ms/step - loss: 242.1123

<div class="k-default-codeblock">
```

```
</div>
 187/250 ━━━━━━━━━━━━━━[37m━━━━━━  25s 402ms/step - loss: 242.4380

<div class="k-default-codeblock">
```

```
</div>
 188/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 402ms/step - loss: 242.7613

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 403ms/step - loss: 243.0821

<div class="k-default-codeblock">
```

```
</div>
 190/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 403ms/step - loss: 243.3999

<div class="k-default-codeblock">
```

```
</div>
 191/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 403ms/step - loss: 243.7144

<div class="k-default-codeblock">
```

```
</div>
 192/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 403ms/step - loss: 244.0252

<div class="k-default-codeblock">
```

```
</div>
 193/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 403ms/step - loss: 244.3325

<div class="k-default-codeblock">
```

```
</div>
 194/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 403ms/step - loss: 244.6360

<div class="k-default-codeblock">
```

```
</div>
 195/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 403ms/step - loss: 244.9358

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 403ms/step - loss: 245.2313

<div class="k-default-codeblock">
```

```
</div>
 197/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 403ms/step - loss: 245.5223

<div class="k-default-codeblock">
```

```
</div>
 198/250 ━━━━━━━━━━━━━━━[37m━━━━━  20s 403ms/step - loss: 245.8087

<div class="k-default-codeblock">
```

```
</div>
 199/250 ━━━━━━━━━━━━━━━[37m━━━━━  20s 403ms/step - loss: 246.0907

<div class="k-default-codeblock">
```

```
</div>
 200/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 404ms/step - loss: 246.3678

<div class="k-default-codeblock">
```

```
</div>
 201/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 404ms/step - loss: 246.6401

<div class="k-default-codeblock">
```

```
</div>
 202/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 404ms/step - loss: 246.9076

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 404ms/step - loss: 247.1701

<div class="k-default-codeblock">
```

```
</div>
 204/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 404ms/step - loss: 247.4275

<div class="k-default-codeblock">
```

```
</div>
 205/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 404ms/step - loss: 247.6802

<div class="k-default-codeblock">
```

```
</div>
 206/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 404ms/step - loss: 247.9284

<div class="k-default-codeblock">
```

```
</div>
 207/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 404ms/step - loss: 248.1717

<div class="k-default-codeblock">
```

```
</div>
 208/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 404ms/step - loss: 248.4103

<div class="k-default-codeblock">
```

```
</div>
 209/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 404ms/step - loss: 248.6443

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 404ms/step - loss: 248.8761

<div class="k-default-codeblock">
```

```
</div>
 211/250 ━━━━━━━━━━━━━━━━[37m━━━━  15s 404ms/step - loss: 249.1029

<div class="k-default-codeblock">
```

```
</div>
 212/250 ━━━━━━━━━━━━━━━━[37m━━━━  15s 404ms/step - loss: 249.3254

<div class="k-default-codeblock">
```

```
</div>
 213/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 404ms/step - loss: 249.5436

<div class="k-default-codeblock">
```

```
</div>
 214/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 404ms/step - loss: 249.7575

<div class="k-default-codeblock">
```

```
</div>
 215/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 404ms/step - loss: 249.9677

<div class="k-default-codeblock">
```

```
</div>
 216/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 404ms/step - loss: 250.1742

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 404ms/step - loss: 250.3772

<div class="k-default-codeblock">
```

```
</div>
 218/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 404ms/step - loss: 250.5769

<div class="k-default-codeblock">
```

```
</div>
 219/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 404ms/step - loss: 250.7733

<div class="k-default-codeblock">
```

```
</div>
 220/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 404ms/step - loss: 250.9668

<div class="k-default-codeblock">
```

```
</div>
 221/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 404ms/step - loss: 251.1576

<div class="k-default-codeblock">
```

```
</div>
 222/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 404ms/step - loss: 251.3454

<div class="k-default-codeblock">
```

```
</div>
 223/250 ━━━━━━━━━━━━━━━━━[37m━━━  10s 404ms/step - loss: 251.5306

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  10s 404ms/step - loss: 251.7132

<div class="k-default-codeblock">
```

```
</div>
 225/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 404ms/step - loss: 251.8934

<div class="k-default-codeblock">
```

```
</div>
 226/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 404ms/step - loss: 252.0711 

<div class="k-default-codeblock">
```

```
</div>
 227/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 404ms/step - loss: 252.2465

<div class="k-default-codeblock">
```

```
</div>
 228/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 404ms/step - loss: 252.4198

<div class="k-default-codeblock">
```

```
</div>
 229/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 404ms/step - loss: 252.5910

<div class="k-default-codeblock">
```

```
</div>
 230/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 404ms/step - loss: 252.7596

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 405ms/step - loss: 252.9261

<div class="k-default-codeblock">
```

```
</div>
 232/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 405ms/step - loss: 253.0904

<div class="k-default-codeblock">
```

```
</div>
 233/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 405ms/step - loss: 253.2526

<div class="k-default-codeblock">
```

```
</div>
 234/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 405ms/step - loss: 253.4126

<div class="k-default-codeblock">
```

```
</div>
 235/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 405ms/step - loss: 253.5704

<div class="k-default-codeblock">
```

```
</div>
 236/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 405ms/step - loss: 253.7260

<div class="k-default-codeblock">
```

```
</div>
 237/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 405ms/step - loss: 253.8793

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 405ms/step - loss: 254.0305

<div class="k-default-codeblock">
```

```
</div>
 239/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 405ms/step - loss: 254.1795

<div class="k-default-codeblock">
```

```
</div>
 240/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 405ms/step - loss: 254.3264

<div class="k-default-codeblock">
```

```
</div>
 241/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 405ms/step - loss: 254.4712

<div class="k-default-codeblock">
```

```
</div>
 242/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 405ms/step - loss: 254.6140

<div class="k-default-codeblock">
```

```
</div>
 243/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 405ms/step - loss: 254.7546

<div class="k-default-codeblock">
```

```
</div>
 244/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 405ms/step - loss: 254.8930

<div class="k-default-codeblock">
```

```
</div>
 245/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 405ms/step - loss: 255.0293

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 405ms/step - loss: 255.1634

<div class="k-default-codeblock">
```

```
</div>
 247/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 405ms/step - loss: 255.2954

<div class="k-default-codeblock">
```

```
</div>
 248/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 405ms/step - loss: 255.4251

<div class="k-default-codeblock">
```

```
</div>
 249/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 405ms/step - loss: 255.5527

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 405ms/step - loss: 255.6782

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 101s 405ms/step - loss: 255.8026


<div class="k-default-codeblock">
```
Epoch 5/10

```
</div>
    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  1:46 427ms/step - loss: 223.2762

<div class="k-default-codeblock">
```

```
</div>
   2/250 [37m━━━━━━━━━━━━━━━━━━━━  1:35 386ms/step - loss: 219.8572

<div class="k-default-codeblock">
```

```
</div>
   3/250 [37m━━━━━━━━━━━━━━━━━━━━  1:38 400ms/step - loss: 217.2949

<div class="k-default-codeblock">
```

```
</div>
   4/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 407ms/step - loss: 215.5167

<div class="k-default-codeblock">
```

```
</div>
   5/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 410ms/step - loss: 214.1932

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 412ms/step - loss: 213.4324

<div class="k-default-codeblock">
```

```
</div>
   7/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 412ms/step - loss: 212.6033

<div class="k-default-codeblock">
```

```
</div>
   8/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 415ms/step - loss: 211.8697

<div class="k-default-codeblock">
```

```
</div>
   9/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 416ms/step - loss: 211.1901

<div class="k-default-codeblock">
```

```
</div>
  10/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 417ms/step - loss: 210.5859

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 420ms/step - loss: 210.0340

<div class="k-default-codeblock">
```

```
</div>
  12/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 421ms/step - loss: 209.6114

<div class="k-default-codeblock">
```

```
</div>
  13/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:40 423ms/step - loss: 209.2315

<div class="k-default-codeblock">
```

```
</div>
  14/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:39 422ms/step - loss: 208.8248

<div class="k-default-codeblock">
```

```
</div>
  15/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:38 419ms/step - loss: 208.4697

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:38 420ms/step - loss: 208.1076

<div class="k-default-codeblock">
```

```
</div>
  17/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 420ms/step - loss: 207.7465

<div class="k-default-codeblock">
```

```
</div>
  18/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 420ms/step - loss: 207.3898

<div class="k-default-codeblock">
```

```
</div>
  19/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 419ms/step - loss: 207.0360

<div class="k-default-codeblock">
```

```
</div>
  20/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 418ms/step - loss: 206.7211

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 418ms/step - loss: 206.4058

<div class="k-default-codeblock">
```

```
</div>
  22/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 418ms/step - loss: 206.0855

<div class="k-default-codeblock">
```

```
</div>
  23/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 419ms/step - loss: 205.7691

<div class="k-default-codeblock">
```

```
</div>
  24/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 419ms/step - loss: 205.4677

<div class="k-default-codeblock">
```

```
</div>
  25/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:34 420ms/step - loss: 205.1721

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:34 421ms/step - loss: 204.8931

<div class="k-default-codeblock">
```

```
</div>
  27/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:33 420ms/step - loss: 204.6247

<div class="k-default-codeblock">
```

```
</div>
  28/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:33 419ms/step - loss: 204.3688

<div class="k-default-codeblock">
```

```
</div>
  29/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 419ms/step - loss: 204.1195

<div class="k-default-codeblock">
```

```
</div>
  30/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 419ms/step - loss: 203.8810

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 419ms/step - loss: 203.6406

<div class="k-default-codeblock">
```

```
</div>
  32/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 419ms/step - loss: 203.4079

<div class="k-default-codeblock">
```

```
</div>
  33/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 419ms/step - loss: 203.1730

<div class="k-default-codeblock">
```

```
</div>
  34/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 419ms/step - loss: 202.9393

<div class="k-default-codeblock">
```

```
</div>
  35/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 419ms/step - loss: 202.7087

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 418ms/step - loss: 202.4873

<div class="k-default-codeblock">
```

```
</div>
  37/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 419ms/step - loss: 202.2683

<div class="k-default-codeblock">
```

```
</div>
  38/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:28 419ms/step - loss: 202.0474

<div class="k-default-codeblock">
```

```
</div>
  39/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:28 419ms/step - loss: 201.8329

<div class="k-default-codeblock">
```

```
</div>
  40/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 418ms/step - loss: 201.6190

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 419ms/step - loss: 201.4044

<div class="k-default-codeblock">
```

```
</div>
  42/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 418ms/step - loss: 201.1937

<div class="k-default-codeblock">
```

```
</div>
  43/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 418ms/step - loss: 200.9805

<div class="k-default-codeblock">
```

```
</div>
  44/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 418ms/step - loss: 200.7667

<div class="k-default-codeblock">
```

```
</div>
  45/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 417ms/step - loss: 200.5561

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 418ms/step - loss: 200.3427

<div class="k-default-codeblock">
```

```
</div>
  47/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 418ms/step - loss: 200.1360

<div class="k-default-codeblock">
```

```
</div>
  48/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 418ms/step - loss: 200.0112

<div class="k-default-codeblock">
```

```
</div>
  49/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:23 418ms/step - loss: 199.8875

<div class="k-default-codeblock">
```

```
</div>
  50/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:23 418ms/step - loss: 199.7655

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:23 418ms/step - loss: 199.6555

<div class="k-default-codeblock">
```

```
</div>
  52/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:23 421ms/step - loss: 199.5594

<div class="k-default-codeblock">
```

```
</div>
  53/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 420ms/step - loss: 199.4797

<div class="k-default-codeblock">
```

```
</div>
  54/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 421ms/step - loss: 199.4212

<div class="k-default-codeblock">
```

```
</div>
  55/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 421ms/step - loss: 199.3858

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 421ms/step - loss: 199.3778

<div class="k-default-codeblock">
```

```
</div>
  57/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 421ms/step - loss: 199.3963

<div class="k-default-codeblock">
```

```
</div>
  58/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 421ms/step - loss: 199.4407

<div class="k-default-codeblock">
```

```
</div>
  59/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 421ms/step - loss: 199.5139

<div class="k-default-codeblock">
```

```
</div>
  60/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 421ms/step - loss: 199.6200

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 421ms/step - loss: 199.7569

<div class="k-default-codeblock">
```

```
</div>
  62/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 421ms/step - loss: 199.9223

<div class="k-default-codeblock">
```

```
</div>
  63/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:18 421ms/step - loss: 200.1137

<div class="k-default-codeblock">
```

```
</div>
  64/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:18 421ms/step - loss: 200.3300

<div class="k-default-codeblock">
```

```
</div>
  65/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:17 421ms/step - loss: 200.5680

<div class="k-default-codeblock">
```

```
</div>
  66/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:17 421ms/step - loss: 200.8261

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:17 421ms/step - loss: 201.1059

<div class="k-default-codeblock">
```

```
</div>
  68/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 421ms/step - loss: 201.4054

<div class="k-default-codeblock">
```

```
</div>
  69/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 421ms/step - loss: 201.7219

<div class="k-default-codeblock">
```

```
</div>
  70/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 421ms/step - loss: 202.0553

<div class="k-default-codeblock">
```

```
</div>
  71/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 421ms/step - loss: 202.4035

<div class="k-default-codeblock">
```

```
</div>
  72/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 420ms/step - loss: 202.7644

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 420ms/step - loss: 203.1363

<div class="k-default-codeblock">
```

```
</div>
  74/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 420ms/step - loss: 203.5168

<div class="k-default-codeblock">
```

```
</div>
  75/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:13 420ms/step - loss: 203.9049

<div class="k-default-codeblock">
```

```
</div>
  76/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:13 420ms/step - loss: 204.2990

<div class="k-default-codeblock">
```

```
</div>
  77/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 420ms/step - loss: 204.6979

<div class="k-default-codeblock">
```

```
</div>
  78/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 420ms/step - loss: 205.1009

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 420ms/step - loss: 205.5064

<div class="k-default-codeblock">
```

```
</div>
  80/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 420ms/step - loss: 205.9141

<div class="k-default-codeblock">
```

```
</div>
  81/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 420ms/step - loss: 206.3213

<div class="k-default-codeblock">
```

```
</div>
  82/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 421ms/step - loss: 206.7273

<div class="k-default-codeblock">
```

```
</div>
  83/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 420ms/step - loss: 207.1327

<div class="k-default-codeblock">
```

```
</div>
  84/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 420ms/step - loss: 207.5359

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 420ms/step - loss: 207.9355

<div class="k-default-codeblock">
```

```
</div>
  86/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 420ms/step - loss: 208.3313

<div class="k-default-codeblock">
```

```
</div>
  87/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 419ms/step - loss: 208.7242

<div class="k-default-codeblock">
```

```
</div>
  88/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 419ms/step - loss: 209.1112

<div class="k-default-codeblock">
```

```
</div>
  89/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 419ms/step - loss: 209.4934

<div class="k-default-codeblock">
```

```
</div>
  90/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 419ms/step - loss: 209.8702

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 419ms/step - loss: 210.2416

<div class="k-default-codeblock">
```

```
</div>
  92/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 419ms/step - loss: 210.6073

<div class="k-default-codeblock">
```

```
</div>
  93/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 419ms/step - loss: 210.9674

<div class="k-default-codeblock">
```

```
</div>
  94/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 419ms/step - loss: 211.3213

<div class="k-default-codeblock">
```

```
</div>
  95/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 419ms/step - loss: 211.6687

<div class="k-default-codeblock">
```

```
</div>
  96/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 419ms/step - loss: 212.0088

<div class="k-default-codeblock">
```

```
</div>
  97/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 418ms/step - loss: 212.3419

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 418ms/step - loss: 212.6684

<div class="k-default-codeblock">
```

```
</div>
  99/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 418ms/step - loss: 212.9881

<div class="k-default-codeblock">
```

```
</div>
 100/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:02 418ms/step - loss: 213.3002

<div class="k-default-codeblock">
```

```
</div>
 101/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:02 418ms/step - loss: 213.6050

<div class="k-default-codeblock">
```

```
</div>
 102/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 418ms/step - loss: 213.9029

<div class="k-default-codeblock">
```

```
</div>
 103/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 418ms/step - loss: 214.1935

<div class="k-default-codeblock">
```

```
</div>
 104/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 418ms/step - loss: 214.4776

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 417ms/step - loss: 214.7554

<div class="k-default-codeblock">
```

```
</div>
 106/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 417ms/step - loss: 215.0260

<div class="k-default-codeblock">
```

```
</div>
 107/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 417ms/step - loss: 215.2894 

<div class="k-default-codeblock">
```

```
</div>
 108/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 417ms/step - loss: 215.5463

<div class="k-default-codeblock">
```

```
</div>
 109/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 417ms/step - loss: 215.7967

<div class="k-default-codeblock">
```

```
</div>
 110/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 417ms/step - loss: 216.0399

<div class="k-default-codeblock">
```

```
</div>
 111/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 417ms/step - loss: 216.2762

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 417ms/step - loss: 216.5059

<div class="k-default-codeblock">
```

```
</div>
 113/250 ━━━━━━━━━[37m━━━━━━━━━━━  57s 417ms/step - loss: 216.7290

<div class="k-default-codeblock">
```

```
</div>
 114/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 417ms/step - loss: 216.9459

<div class="k-default-codeblock">
```

```
</div>
 115/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 417ms/step - loss: 217.1564

<div class="k-default-codeblock">
```

```
</div>
 116/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 417ms/step - loss: 217.3602

<div class="k-default-codeblock">
```

```
</div>
 117/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 417ms/step - loss: 217.5578

<div class="k-default-codeblock">
```

```
</div>
 118/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 417ms/step - loss: 217.7501

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 417ms/step - loss: 217.9362

<div class="k-default-codeblock">
```

```
</div>
 120/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 417ms/step - loss: 218.1171

<div class="k-default-codeblock">
```

```
</div>
 121/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 417ms/step - loss: 218.2922

<div class="k-default-codeblock">
```

```
</div>
 122/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 417ms/step - loss: 218.4623

<div class="k-default-codeblock">
```

```
</div>
 123/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 417ms/step - loss: 218.6268

<div class="k-default-codeblock">
```

```
</div>
 124/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 417ms/step - loss: 218.7859

<div class="k-default-codeblock">
```

```
</div>
 125/250 ━━━━━━━━━━[37m━━━━━━━━━━  52s 417ms/step - loss: 218.9401

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 417ms/step - loss: 219.0889

<div class="k-default-codeblock">
```

```
</div>
 127/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 416ms/step - loss: 219.2326

<div class="k-default-codeblock">
```

```
</div>
 128/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 416ms/step - loss: 219.3713

<div class="k-default-codeblock">
```

```
</div>
 129/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 416ms/step - loss: 219.5053

<div class="k-default-codeblock">
```

```
</div>
 130/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 416ms/step - loss: 219.6342

<div class="k-default-codeblock">
```

```
</div>
 131/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 416ms/step - loss: 219.7586

<div class="k-default-codeblock">
```

```
</div>
 132/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 416ms/step - loss: 219.8786

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 416ms/step - loss: 219.9951

<div class="k-default-codeblock">
```

```
</div>
 134/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 416ms/step - loss: 220.1071

<div class="k-default-codeblock">
```

```
</div>
 135/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 416ms/step - loss: 220.2145

<div class="k-default-codeblock">
```

```
</div>
 136/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 416ms/step - loss: 220.3173

<div class="k-default-codeblock">
```

```
</div>
 137/250 ━━━━━━━━━━[37m━━━━━━━━━━  46s 416ms/step - loss: 220.4165

<div class="k-default-codeblock">
```

```
</div>
 138/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 416ms/step - loss: 220.5110

<div class="k-default-codeblock">
```

```
</div>
 139/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 415ms/step - loss: 220.6019

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 415ms/step - loss: 220.6894

<div class="k-default-codeblock">
```

```
</div>
 141/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 415ms/step - loss: 220.7736

<div class="k-default-codeblock">
```

```
</div>
 142/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 415ms/step - loss: 220.8542

<div class="k-default-codeblock">
```

```
</div>
 143/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 415ms/step - loss: 220.9317

<div class="k-default-codeblock">
```

```
</div>
 144/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 416ms/step - loss: 221.0059

<div class="k-default-codeblock">
```

```
</div>
 145/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 416ms/step - loss: 221.0775

<div class="k-default-codeblock">
```

```
</div>
 146/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 416ms/step - loss: 221.1463

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 416ms/step - loss: 221.2124

<div class="k-default-codeblock">
```

```
</div>
 148/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 415ms/step - loss: 221.2755

<div class="k-default-codeblock">
```

```
</div>
 149/250 ━━━━━━━━━━━[37m━━━━━━━━━  41s 415ms/step - loss: 221.3352

<div class="k-default-codeblock">
```

```
</div>
 150/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 415ms/step - loss: 221.3920

<div class="k-default-codeblock">
```

```
</div>
 151/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 415ms/step - loss: 221.4455

<div class="k-default-codeblock">
```

```
</div>
 152/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 415ms/step - loss: 221.4958

<div class="k-default-codeblock">
```

```
</div>
 153/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 415ms/step - loss: 221.5435

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 414ms/step - loss: 221.5889

<div class="k-default-codeblock">
```

```
</div>
 155/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 414ms/step - loss: 221.6320

<div class="k-default-codeblock">
```

```
</div>
 156/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 414ms/step - loss: 221.6723

<div class="k-default-codeblock">
```

```
</div>
 157/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 414ms/step - loss: 221.7102

<div class="k-default-codeblock">
```

```
</div>
 158/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 414ms/step - loss: 221.7458

<div class="k-default-codeblock">
```

```
</div>
 159/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 414ms/step - loss: 221.7794

<div class="k-default-codeblock">
```

```
</div>
 160/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 414ms/step - loss: 221.8111

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 414ms/step - loss: 221.8405

<div class="k-default-codeblock">
```

```
</div>
 162/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 414ms/step - loss: 221.8683

<div class="k-default-codeblock">
```

```
</div>
 163/250 ━━━━━━━━━━━━━[37m━━━━━━━  36s 414ms/step - loss: 221.8943

<div class="k-default-codeblock">
```

```
</div>
 164/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 414ms/step - loss: 221.9182

<div class="k-default-codeblock">
```

```
</div>
 165/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 414ms/step - loss: 221.9402

<div class="k-default-codeblock">
```

```
</div>
 166/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 414ms/step - loss: 221.9605

<div class="k-default-codeblock">
```

```
</div>
 167/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 414ms/step - loss: 221.9790

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 414ms/step - loss: 221.9954

<div class="k-default-codeblock">
```

```
</div>
 169/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 414ms/step - loss: 222.0103

<div class="k-default-codeblock">
```

```
</div>
 170/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 414ms/step - loss: 222.0231

<div class="k-default-codeblock">
```

```
</div>
 171/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 415ms/step - loss: 222.0337

<div class="k-default-codeblock">
```

```
</div>
 172/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 414ms/step - loss: 222.0426

<div class="k-default-codeblock">
```

```
</div>
 173/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 414ms/step - loss: 222.0499

<div class="k-default-codeblock">
```

```
</div>
 174/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 414ms/step - loss: 222.0557

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 414ms/step - loss: 222.0600

<div class="k-default-codeblock">
```

```
</div>
 176/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 414ms/step - loss: 222.0627

<div class="k-default-codeblock">
```

```
</div>
 177/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 414ms/step - loss: 222.0636

<div class="k-default-codeblock">
```

```
</div>
 178/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 414ms/step - loss: 222.0630

<div class="k-default-codeblock">
```

```
</div>
 179/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 414ms/step - loss: 222.0609

<div class="k-default-codeblock">
```

```
</div>
 180/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 414ms/step - loss: 222.0574

<div class="k-default-codeblock">
```

```
</div>
 181/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 414ms/step - loss: 222.0525

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 414ms/step - loss: 222.0462

<div class="k-default-codeblock">
```

```
</div>
 183/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 414ms/step - loss: 222.0388

<div class="k-default-codeblock">
```

```
</div>
 184/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 414ms/step - loss: 222.0303

<div class="k-default-codeblock">
```

```
</div>
 185/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 414ms/step - loss: 222.0205

<div class="k-default-codeblock">
```

```
</div>
 186/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 414ms/step - loss: 222.0096

<div class="k-default-codeblock">
```

```
</div>
 187/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 415ms/step - loss: 221.9975

<div class="k-default-codeblock">
```

```
</div>
 188/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 415ms/step - loss: 221.9844

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 415ms/step - loss: 221.9698

<div class="k-default-codeblock">
```

```
</div>
 190/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 415ms/step - loss: 221.9537

<div class="k-default-codeblock">
```

```
</div>
 191/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 416ms/step - loss: 221.9366

<div class="k-default-codeblock">
```

```
</div>
 192/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 416ms/step - loss: 221.9182

<div class="k-default-codeblock">
```

```
</div>
 193/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 416ms/step - loss: 221.8988

<div class="k-default-codeblock">
```

```
</div>
 194/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 416ms/step - loss: 221.8782

<div class="k-default-codeblock">
```

```
</div>
 195/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 416ms/step - loss: 221.8565

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 416ms/step - loss: 221.8340

<div class="k-default-codeblock">
```

```
</div>
 197/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 416ms/step - loss: 221.8108

<div class="k-default-codeblock">
```

```
</div>
 198/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 416ms/step - loss: 221.7933

<div class="k-default-codeblock">
```

```
</div>
 199/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 416ms/step - loss: 221.7753

<div class="k-default-codeblock">
```

```
</div>
 200/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 416ms/step - loss: 221.7568

<div class="k-default-codeblock">
```

```
</div>
 201/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 416ms/step - loss: 221.7389

<div class="k-default-codeblock">
```

```
</div>
 202/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 416ms/step - loss: 221.7216

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 416ms/step - loss: 221.7053

<div class="k-default-codeblock">
```

```
</div>
 204/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 416ms/step - loss: 221.6905

<div class="k-default-codeblock">
```

```
</div>
 205/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 416ms/step - loss: 221.6779

<div class="k-default-codeblock">
```

```
</div>
 206/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 416ms/step - loss: 221.6677

<div class="k-default-codeblock">
```

```
</div>
 207/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 416ms/step - loss: 221.6606

<div class="k-default-codeblock">
```

```
</div>
 208/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 416ms/step - loss: 221.6569

<div class="k-default-codeblock">
```

```
</div>
 209/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 416ms/step - loss: 221.6568

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 417ms/step - loss: 221.6603

<div class="k-default-codeblock">
```

```
</div>
 211/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 417ms/step - loss: 221.6678

<div class="k-default-codeblock">
```

```
</div>
 212/250 ━━━━━━━━━━━━━━━━[37m━━━━  15s 417ms/step - loss: 221.6795

<div class="k-default-codeblock">
```

```
</div>
 213/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 417ms/step - loss: 221.6954

<div class="k-default-codeblock">
```

```
</div>
 214/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 418ms/step - loss: 221.7153

<div class="k-default-codeblock">
```

```
</div>
 215/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 418ms/step - loss: 221.7396

<div class="k-default-codeblock">
```

```
</div>
 216/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 419ms/step - loss: 221.7677

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 419ms/step - loss: 221.7998

<div class="k-default-codeblock">
```

```
</div>
 218/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 420ms/step - loss: 221.8357

<div class="k-default-codeblock">
```

```
</div>
 219/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 420ms/step - loss: 221.8751

<div class="k-default-codeblock">
```

```
</div>
 220/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 420ms/step - loss: 221.9181

<div class="k-default-codeblock">
```

```
</div>
 221/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 420ms/step - loss: 221.9640

<div class="k-default-codeblock">
```

```
</div>
 222/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 421ms/step - loss: 222.0128

<div class="k-default-codeblock">
```

```
</div>
 223/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 421ms/step - loss: 222.0645

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  10s 421ms/step - loss: 222.1187

<div class="k-default-codeblock">
```

```
</div>
 225/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 422ms/step - loss: 222.1752

<div class="k-default-codeblock">
```

```
</div>
 226/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 422ms/step - loss: 222.2337

<div class="k-default-codeblock">
```

```
</div>
 227/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 422ms/step - loss: 222.2938 

<div class="k-default-codeblock">
```

```
</div>
 228/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 422ms/step - loss: 222.3555

<div class="k-default-codeblock">
```

```
</div>
 229/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 422ms/step - loss: 222.4183

<div class="k-default-codeblock">
```

```
</div>
 230/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 423ms/step - loss: 222.4821

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 423ms/step - loss: 222.5469

<div class="k-default-codeblock">
```

```
</div>
 232/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 423ms/step - loss: 222.6126

<div class="k-default-codeblock">
```

```
</div>
 233/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 423ms/step - loss: 222.6787

<div class="k-default-codeblock">
```

```
</div>
 234/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 424ms/step - loss: 222.7450

<div class="k-default-codeblock">
```

```
</div>
 235/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 425ms/step - loss: 222.8112

<div class="k-default-codeblock">
```

```
</div>
 236/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 426ms/step - loss: 222.8772

<div class="k-default-codeblock">
```

```
</div>
 237/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 426ms/step - loss: 222.9430

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  5s 426ms/step - loss: 223.0084

<div class="k-default-codeblock">
```

```
</div>
 239/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 427ms/step - loss: 223.0731

<div class="k-default-codeblock">
```

```
</div>
 240/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 427ms/step - loss: 223.1374

<div class="k-default-codeblock">
```

```
</div>
 241/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 427ms/step - loss: 223.2013

<div class="k-default-codeblock">
```

```
</div>
 242/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 427ms/step - loss: 223.2644

<div class="k-default-codeblock">
```

```
</div>
 243/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 427ms/step - loss: 223.3267

<div class="k-default-codeblock">
```

```
</div>
 244/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 427ms/step - loss: 223.3882

<div class="k-default-codeblock">
```

```
</div>
 245/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 427ms/step - loss: 223.4486

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 427ms/step - loss: 223.5080

<div class="k-default-codeblock">
```

```
</div>
 247/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 427ms/step - loss: 223.5661

<div class="k-default-codeblock">
```

```
</div>
 248/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 428ms/step - loss: 223.6232

<div class="k-default-codeblock">
```

```
</div>
 249/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 428ms/step - loss: 223.6790

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 428ms/step - loss: 223.7335

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 107s 428ms/step - loss: 223.7877


<div class="k-default-codeblock">
```
Epoch 6/10

```
</div>
    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  1:58 477ms/step - loss: 196.7945

<div class="k-default-codeblock">
```

```
</div>
   2/250 [37m━━━━━━━━━━━━━━━━━━━━  1:53 457ms/step - loss: 195.6742

<div class="k-default-codeblock">
```

```
</div>
   3/250 [37m━━━━━━━━━━━━━━━━━━━━  2:00 488ms/step - loss: 194.0928

<div class="k-default-codeblock">
```

```
</div>
   4/250 [37m━━━━━━━━━━━━━━━━━━━━  1:58 481ms/step - loss: 195.5873

<div class="k-default-codeblock">
```

```
</div>
   5/250 [37m━━━━━━━━━━━━━━━━━━━━  1:55 472ms/step - loss: 196.3146

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  1:53 466ms/step - loss: 197.1003

<div class="k-default-codeblock">
```

```
</div>
   7/250 [37m━━━━━━━━━━━━━━━━━━━━  1:51 458ms/step - loss: 197.5786

<div class="k-default-codeblock">
```

```
</div>
   8/250 [37m━━━━━━━━━━━━━━━━━━━━  1:49 452ms/step - loss: 197.9180

<div class="k-default-codeblock">
```

```
</div>
   9/250 [37m━━━━━━━━━━━━━━━━━━━━  1:48 450ms/step - loss: 198.1149

<div class="k-default-codeblock">
```

```
</div>
  10/250 [37m━━━━━━━━━━━━━━━━━━━━  1:46 444ms/step - loss: 198.2755

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  1:45 442ms/step - loss: 198.4848

<div class="k-default-codeblock">
```

```
</div>
  12/250 [37m━━━━━━━━━━━━━━━━━━━━  1:45 441ms/step - loss: 198.6258

<div class="k-default-codeblock">
```

```
</div>
  13/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:43 438ms/step - loss: 198.8443

<div class="k-default-codeblock">
```

```
</div>
  14/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:42 436ms/step - loss: 199.0643

<div class="k-default-codeblock">
```

```
</div>
  15/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:41 432ms/step - loss: 199.3095

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:41 433ms/step - loss: 199.5619

<div class="k-default-codeblock">
```

```
</div>
  17/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:41 437ms/step - loss: 199.8108

<div class="k-default-codeblock">
```

```
</div>
  18/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:42 442ms/step - loss: 200.0598

<div class="k-default-codeblock">
```

```
</div>
  19/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:42 444ms/step - loss: 200.3138

<div class="k-default-codeblock">
```

```
</div>
  20/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:41 443ms/step - loss: 200.5696

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:41 442ms/step - loss: 200.8367

<div class="k-default-codeblock">
```

```
</div>
  22/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:41 445ms/step - loss: 201.1085

<div class="k-default-codeblock">
```

```
</div>
  23/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:41 447ms/step - loss: 201.3658

<div class="k-default-codeblock">
```

```
</div>
  24/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:41 449ms/step - loss: 201.6172

<div class="k-default-codeblock">
```

```
</div>
  25/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:41 453ms/step - loss: 201.8492

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:41 454ms/step - loss: 202.0800

<div class="k-default-codeblock">
```

```
</div>
  27/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:41 457ms/step - loss: 202.2897

<div class="k-default-codeblock">
```

```
</div>
  28/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:41 459ms/step - loss: 202.4819

<div class="k-default-codeblock">
```

```
</div>
  29/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:41 461ms/step - loss: 202.6684

<div class="k-default-codeblock">
```

```
</div>
  30/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:41 462ms/step - loss: 202.8389

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:41 463ms/step - loss: 202.9858

<div class="k-default-codeblock">
```

```
</div>
  32/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:41 464ms/step - loss: 203.1252

<div class="k-default-codeblock">
```

```
</div>
  33/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:40 462ms/step - loss: 203.2624

<div class="k-default-codeblock">
```

```
</div>
  34/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:39 460ms/step - loss: 203.3810

<div class="k-default-codeblock">
```

```
</div>
  35/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:38 459ms/step - loss: 203.4821

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:38 458ms/step - loss: 203.5687

<div class="k-default-codeblock">
```

```
</div>
  37/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:37 458ms/step - loss: 203.6480

<div class="k-default-codeblock">
```

```
</div>
  38/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:36 457ms/step - loss: 203.7122

<div class="k-default-codeblock">
```

```
</div>
  39/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:36 457ms/step - loss: 203.7597

<div class="k-default-codeblock">
```

```
</div>
  40/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:36 457ms/step - loss: 203.7988

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:35 458ms/step - loss: 203.8224

<div class="k-default-codeblock">
```

```
</div>
  42/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:35 458ms/step - loss: 203.8344

<div class="k-default-codeblock">
```

```
</div>
  43/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 457ms/step - loss: 203.8381

<div class="k-default-codeblock">
```

```
</div>
  44/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 458ms/step - loss: 203.8291

<div class="k-default-codeblock">
```

```
</div>
  45/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:33 458ms/step - loss: 203.8116

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 462ms/step - loss: 203.7846

<div class="k-default-codeblock">
```

```
</div>
  47/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:35 469ms/step - loss: 203.7532

<div class="k-default-codeblock">
```

```
</div>
  48/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 469ms/step - loss: 203.7148

<div class="k-default-codeblock">
```

```
</div>
  49/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 469ms/step - loss: 203.6660

<div class="k-default-codeblock">
```

```
</div>
  50/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:33 469ms/step - loss: 203.6116

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:33 469ms/step - loss: 203.5533

<div class="k-default-codeblock">
```

```
</div>
  52/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:32 468ms/step - loss: 203.4890

<div class="k-default-codeblock">
```

```
</div>
  53/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:32 468ms/step - loss: 203.4465

<div class="k-default-codeblock">
```

```
</div>
  54/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:31 468ms/step - loss: 203.3990

<div class="k-default-codeblock">
```

```
</div>
  55/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:31 469ms/step - loss: 203.3499

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:31 469ms/step - loss: 203.3042

<div class="k-default-codeblock">
```

```
</div>
  57/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:30 468ms/step - loss: 203.2592

<div class="k-default-codeblock">
```

```
</div>
  58/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:29 467ms/step - loss: 203.2167

<div class="k-default-codeblock">
```

```
</div>
  59/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:29 468ms/step - loss: 203.1732

<div class="k-default-codeblock">
```

```
</div>
  60/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:28 468ms/step - loss: 203.1372

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:28 467ms/step - loss: 203.1050

<div class="k-default-codeblock">
```

```
</div>
  62/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:27 467ms/step - loss: 203.0792

<div class="k-default-codeblock">
```

```
</div>
  63/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:27 469ms/step - loss: 203.0604

<div class="k-default-codeblock">
```

```
</div>
  64/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:27 472ms/step - loss: 203.0465

<div class="k-default-codeblock">
```

```
</div>
  65/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:27 473ms/step - loss: 203.0426

<div class="k-default-codeblock">
```

```
</div>
  66/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:27 476ms/step - loss: 203.0453

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:27 477ms/step - loss: 203.0546

<div class="k-default-codeblock">
```

```
</div>
  68/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:26 478ms/step - loss: 203.0689

<div class="k-default-codeblock">
```

```
</div>
  69/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:26 479ms/step - loss: 203.0934

<div class="k-default-codeblock">
```

```
</div>
  70/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:26 481ms/step - loss: 203.1225

<div class="k-default-codeblock">
```

```
</div>
  71/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:26 482ms/step - loss: 203.1566

<div class="k-default-codeblock">
```

```
</div>
  72/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:25 482ms/step - loss: 203.1955

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:25 482ms/step - loss: 203.2385

<div class="k-default-codeblock">
```

```
</div>
  74/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:24 482ms/step - loss: 203.2838

<div class="k-default-codeblock">
```

```
</div>
  75/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:24 481ms/step - loss: 203.3304

<div class="k-default-codeblock">
```

```
</div>
  76/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:23 481ms/step - loss: 203.3784

<div class="k-default-codeblock">
```

```
</div>
  77/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:23 481ms/step - loss: 203.4280

<div class="k-default-codeblock">
```

```
</div>
  78/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:22 482ms/step - loss: 203.4774

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:22 482ms/step - loss: 203.5272

<div class="k-default-codeblock">
```

```
</div>
  80/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:22 483ms/step - loss: 203.5786

<div class="k-default-codeblock">
```

```
</div>
  81/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:21 483ms/step - loss: 203.6280

<div class="k-default-codeblock">
```

```
</div>
  82/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:21 485ms/step - loss: 203.6752

<div class="k-default-codeblock">
```

```
</div>
  83/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:21 486ms/step - loss: 203.7207

<div class="k-default-codeblock">
```

```
</div>
  84/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:20 486ms/step - loss: 203.7647

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:20 485ms/step - loss: 203.8082

<div class="k-default-codeblock">
```

```
</div>
  86/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:19 485ms/step - loss: 203.8492

<div class="k-default-codeblock">
```

```
</div>
  87/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:19 485ms/step - loss: 203.8886

<div class="k-default-codeblock">
```

```
</div>
  88/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:18 485ms/step - loss: 203.9259

<div class="k-default-codeblock">
```

```
</div>
  89/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:18 485ms/step - loss: 203.9623

<div class="k-default-codeblock">
```

```
</div>
  90/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:17 485ms/step - loss: 203.9966

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:17 484ms/step - loss: 204.0294

<div class="k-default-codeblock">
```

```
</div>
  92/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:16 485ms/step - loss: 204.0588

<div class="k-default-codeblock">
```

```
</div>
  93/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:16 485ms/step - loss: 204.0854

<div class="k-default-codeblock">
```

```
</div>
  94/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:15 484ms/step - loss: 204.1090

<div class="k-default-codeblock">
```

```
</div>
  95/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:15 485ms/step - loss: 204.1308

<div class="k-default-codeblock">
```

```
</div>
  96/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:14 485ms/step - loss: 204.1495

<div class="k-default-codeblock">
```

```
</div>
  97/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:14 485ms/step - loss: 204.1671

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:13 485ms/step - loss: 204.1818

<div class="k-default-codeblock">
```

```
</div>
  99/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:13 486ms/step - loss: 204.1941

<div class="k-default-codeblock">
```

```
</div>
 100/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:12 486ms/step - loss: 204.2032

<div class="k-default-codeblock">
```

```
</div>
 101/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:12 486ms/step - loss: 204.2100

<div class="k-default-codeblock">
```

```
</div>
 102/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:11 485ms/step - loss: 204.2150

<div class="k-default-codeblock">
```

```
</div>
 103/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:11 485ms/step - loss: 204.2182

<div class="k-default-codeblock">
```

```
</div>
 104/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:10 484ms/step - loss: 204.2199

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:10 484ms/step - loss: 204.2194

<div class="k-default-codeblock">
```

```
</div>
 106/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:09 484ms/step - loss: 204.2165

<div class="k-default-codeblock">
```

```
</div>
 107/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:09 484ms/step - loss: 204.2270

<div class="k-default-codeblock">
```

```
</div>
 108/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:08 484ms/step - loss: 204.2361

<div class="k-default-codeblock">
```

```
</div>
 109/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:08 484ms/step - loss: 204.2451

<div class="k-default-codeblock">
```

```
</div>
 110/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:07 483ms/step - loss: 204.2569

<div class="k-default-codeblock">
```

```
</div>
 111/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:07 484ms/step - loss: 204.2731

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:06 484ms/step - loss: 204.2959

<div class="k-default-codeblock">
```

```
</div>
 113/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:06 484ms/step - loss: 204.3277

<div class="k-default-codeblock">
```

```
</div>
 114/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:05 484ms/step - loss: 204.3700

<div class="k-default-codeblock">
```

```
</div>
 115/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:05 484ms/step - loss: 204.4241

<div class="k-default-codeblock">
```

```
</div>
 116/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:04 484ms/step - loss: 204.4934

<div class="k-default-codeblock">
```

```
</div>
 117/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:04 485ms/step - loss: 204.5780

<div class="k-default-codeblock">
```

```
</div>
 118/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:04 485ms/step - loss: 204.6788

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:03 486ms/step - loss: 204.7955

<div class="k-default-codeblock">
```

```
</div>
 120/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:03 485ms/step - loss: 204.9281

<div class="k-default-codeblock">
```

```
</div>
 121/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:02 486ms/step - loss: 205.0774

<div class="k-default-codeblock">
```

```
</div>
 122/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:02 486ms/step - loss: 205.2423

<div class="k-default-codeblock">
```

```
</div>
 123/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:01 485ms/step - loss: 205.4230

<div class="k-default-codeblock">
```

```
</div>
 124/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:01 485ms/step - loss: 205.6195

<div class="k-default-codeblock">
```

```
</div>
 125/250 ━━━━━━━━━━[37m━━━━━━━━━━  1:00 485ms/step - loss: 205.8307

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  1:00 485ms/step - loss: 206.0557

<div class="k-default-codeblock">
```

```
</div>
 127/250 ━━━━━━━━━━[37m━━━━━━━━━━  59s 485ms/step - loss: 206.2938 

<div class="k-default-codeblock">
```

```
</div>
 128/250 ━━━━━━━━━━[37m━━━━━━━━━━  59s 485ms/step - loss: 206.5438

<div class="k-default-codeblock">
```

```
</div>
 129/250 ━━━━━━━━━━[37m━━━━━━━━━━  58s 484ms/step - loss: 206.8050

<div class="k-default-codeblock">
```

```
</div>
 130/250 ━━━━━━━━━━[37m━━━━━━━━━━  58s 484ms/step - loss: 207.0758

<div class="k-default-codeblock">
```

```
</div>
 131/250 ━━━━━━━━━━[37m━━━━━━━━━━  57s 484ms/step - loss: 207.3553

<div class="k-default-codeblock">
```

```
</div>
 132/250 ━━━━━━━━━━[37m━━━━━━━━━━  57s 484ms/step - loss: 207.6427

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  56s 484ms/step - loss: 207.9369

<div class="k-default-codeblock">
```

```
</div>
 134/250 ━━━━━━━━━━[37m━━━━━━━━━━  56s 484ms/step - loss: 208.2367

<div class="k-default-codeblock">
```

```
</div>
 135/250 ━━━━━━━━━━[37m━━━━━━━━━━  55s 484ms/step - loss: 208.5411

<div class="k-default-codeblock">
```

```
</div>
 136/250 ━━━━━━━━━━[37m━━━━━━━━━━  55s 484ms/step - loss: 208.8493

<div class="k-default-codeblock">
```

```
</div>
 137/250 ━━━━━━━━━━[37m━━━━━━━━━━  54s 484ms/step - loss: 209.1599

<div class="k-default-codeblock">
```

```
</div>
 138/250 ━━━━━━━━━━━[37m━━━━━━━━━  54s 484ms/step - loss: 209.4722

<div class="k-default-codeblock">
```

```
</div>
 139/250 ━━━━━━━━━━━[37m━━━━━━━━━  53s 484ms/step - loss: 209.7853

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  53s 485ms/step - loss: 210.0986

<div class="k-default-codeblock">
```

```
</div>
 141/250 ━━━━━━━━━━━[37m━━━━━━━━━  52s 485ms/step - loss: 210.4112

<div class="k-default-codeblock">
```

```
</div>
 142/250 ━━━━━━━━━━━[37m━━━━━━━━━  52s 486ms/step - loss: 210.7227

<div class="k-default-codeblock">
```

```
</div>
 143/250 ━━━━━━━━━━━[37m━━━━━━━━━  52s 487ms/step - loss: 211.0325

<div class="k-default-codeblock">
```

```
</div>
 144/250 ━━━━━━━━━━━[37m━━━━━━━━━  51s 488ms/step - loss: 211.3398

<div class="k-default-codeblock">
```

```
</div>
 145/250 ━━━━━━━━━━━[37m━━━━━━━━━  51s 489ms/step - loss: 211.6442

<div class="k-default-codeblock">
```

```
</div>
 146/250 ━━━━━━━━━━━[37m━━━━━━━━━  50s 489ms/step - loss: 211.9451

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  50s 490ms/step - loss: 212.2424

<div class="k-default-codeblock">
```

```
</div>
 148/250 ━━━━━━━━━━━[37m━━━━━━━━━  49s 490ms/step - loss: 212.5355

<div class="k-default-codeblock">
```

```
</div>
 149/250 ━━━━━━━━━━━[37m━━━━━━━━━  49s 490ms/step - loss: 212.8237

<div class="k-default-codeblock">
```

```
</div>
 150/250 ━━━━━━━━━━━━[37m━━━━━━━━  49s 490ms/step - loss: 213.1068

<div class="k-default-codeblock">
```

```
</div>
 151/250 ━━━━━━━━━━━━[37m━━━━━━━━  48s 489ms/step - loss: 213.3846

<div class="k-default-codeblock">
```

```
</div>
 152/250 ━━━━━━━━━━━━[37m━━━━━━━━  47s 489ms/step - loss: 213.6575

<div class="k-default-codeblock">
```

```
</div>
 153/250 ━━━━━━━━━━━━[37m━━━━━━━━  47s 488ms/step - loss: 213.9249

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  46s 488ms/step - loss: 214.1867

<div class="k-default-codeblock">
```

```
</div>
 155/250 ━━━━━━━━━━━━[37m━━━━━━━━  46s 488ms/step - loss: 214.4433

<div class="k-default-codeblock">
```

```
</div>
 156/250 ━━━━━━━━━━━━[37m━━━━━━━━  45s 487ms/step - loss: 214.6947

<div class="k-default-codeblock">
```

```
</div>
 157/250 ━━━━━━━━━━━━[37m━━━━━━━━  45s 486ms/step - loss: 214.9407

<div class="k-default-codeblock">
```

```
</div>
 158/250 ━━━━━━━━━━━━[37m━━━━━━━━  44s 486ms/step - loss: 215.1816

<div class="k-default-codeblock">
```

```
</div>
 159/250 ━━━━━━━━━━━━[37m━━━━━━━━  44s 486ms/step - loss: 215.4185

<div class="k-default-codeblock">
```

```
</div>
 160/250 ━━━━━━━━━━━━[37m━━━━━━━━  43s 486ms/step - loss: 215.6513

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  43s 486ms/step - loss: 215.8804

<div class="k-default-codeblock">
```

```
</div>
 162/250 ━━━━━━━━━━━━[37m━━━━━━━━  42s 485ms/step - loss: 216.1067

<div class="k-default-codeblock">
```

```
</div>
 163/250 ━━━━━━━━━━━━━[37m━━━━━━━  42s 485ms/step - loss: 216.3304

<div class="k-default-codeblock">
```

```
</div>
 164/250 ━━━━━━━━━━━━━[37m━━━━━━━  41s 485ms/step - loss: 216.5515

<div class="k-default-codeblock">
```

```
</div>
 165/250 ━━━━━━━━━━━━━[37m━━━━━━━  41s 486ms/step - loss: 216.7708

<div class="k-default-codeblock">
```

```
</div>
 166/250 ━━━━━━━━━━━━━[37m━━━━━━━  40s 486ms/step - loss: 216.9883

<div class="k-default-codeblock">
```

```
</div>
 167/250 ━━━━━━━━━━━━━[37m━━━━━━━  40s 486ms/step - loss: 217.2041

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  39s 486ms/step - loss: 217.4186

<div class="k-default-codeblock">
```

```
</div>
 169/250 ━━━━━━━━━━━━━[37m━━━━━━━  39s 486ms/step - loss: 217.6318

<div class="k-default-codeblock">
```

```
</div>
 170/250 ━━━━━━━━━━━━━[37m━━━━━━━  38s 486ms/step - loss: 217.8441

<div class="k-default-codeblock">
```

```
</div>
 171/250 ━━━━━━━━━━━━━[37m━━━━━━━  38s 486ms/step - loss: 218.0556

<div class="k-default-codeblock">
```

```
</div>
 172/250 ━━━━━━━━━━━━━[37m━━━━━━━  37s 485ms/step - loss: 218.2661

<div class="k-default-codeblock">
```

```
</div>
 173/250 ━━━━━━━━━━━━━[37m━━━━━━━  37s 485ms/step - loss: 218.4760

<div class="k-default-codeblock">
```

```
</div>
 174/250 ━━━━━━━━━━━━━[37m━━━━━━━  36s 485ms/step - loss: 218.6852

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  36s 484ms/step - loss: 218.8935

<div class="k-default-codeblock">
```

```
</div>
 176/250 ━━━━━━━━━━━━━━[37m━━━━━━  35s 484ms/step - loss: 219.1008

<div class="k-default-codeblock">
```

```
</div>
 177/250 ━━━━━━━━━━━━━━[37m━━━━━━  35s 483ms/step - loss: 219.3073

<div class="k-default-codeblock">
```

```
</div>
 178/250 ━━━━━━━━━━━━━━[37m━━━━━━  34s 483ms/step - loss: 219.5134

<div class="k-default-codeblock">
```

```
</div>
 179/250 ━━━━━━━━━━━━━━[37m━━━━━━  34s 482ms/step - loss: 219.7187

<div class="k-default-codeblock">
```

```
</div>
 180/250 ━━━━━━━━━━━━━━[37m━━━━━━  33s 482ms/step - loss: 219.9232

<div class="k-default-codeblock">
```

```
</div>
 181/250 ━━━━━━━━━━━━━━[37m━━━━━━  33s 481ms/step - loss: 220.1268

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  32s 481ms/step - loss: 220.3292

<div class="k-default-codeblock">
```

```
</div>
 183/250 ━━━━━━━━━━━━━━[37m━━━━━━  32s 480ms/step - loss: 220.5310

<div class="k-default-codeblock">
```

```
</div>
 184/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 480ms/step - loss: 220.7315

<div class="k-default-codeblock">
```

```
</div>
 185/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 480ms/step - loss: 220.9307

<div class="k-default-codeblock">
```

```
</div>
 186/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 479ms/step - loss: 221.1285

<div class="k-default-codeblock">
```

```
</div>
 187/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 479ms/step - loss: 221.3250

<div class="k-default-codeblock">
```

```
</div>
 188/250 ━━━━━━━━━━━━━━━[37m━━━━━  29s 479ms/step - loss: 221.5197

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  29s 479ms/step - loss: 221.7130

<div class="k-default-codeblock">
```

```
</div>
 190/250 ━━━━━━━━━━━━━━━[37m━━━━━  28s 478ms/step - loss: 221.9046

<div class="k-default-codeblock">
```

```
</div>
 191/250 ━━━━━━━━━━━━━━━[37m━━━━━  28s 478ms/step - loss: 222.0945

<div class="k-default-codeblock">
```

```
</div>
 192/250 ━━━━━━━━━━━━━━━[37m━━━━━  27s 478ms/step - loss: 222.2825

<div class="k-default-codeblock">
```

```
</div>
 193/250 ━━━━━━━━━━━━━━━[37m━━━━━  27s 477ms/step - loss: 222.4684

<div class="k-default-codeblock">
```

```
</div>
 194/250 ━━━━━━━━━━━━━━━[37m━━━━━  26s 477ms/step - loss: 222.6523

<div class="k-default-codeblock">
```

```
</div>
 195/250 ━━━━━━━━━━━━━━━[37m━━━━━  26s 477ms/step - loss: 222.8341

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 476ms/step - loss: 223.0136

<div class="k-default-codeblock">
```

```
</div>
 197/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 476ms/step - loss: 223.1910

<div class="k-default-codeblock">
```

```
</div>
 198/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 475ms/step - loss: 223.3660

<div class="k-default-codeblock">
```

```
</div>
 199/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 475ms/step - loss: 223.5390

<div class="k-default-codeblock">
```

```
</div>
 200/250 ━━━━━━━━━━━━━━━━[37m━━━━  23s 475ms/step - loss: 223.7097

<div class="k-default-codeblock">
```

```
</div>
 201/250 ━━━━━━━━━━━━━━━━[37m━━━━  23s 474ms/step - loss: 223.8778

<div class="k-default-codeblock">
```

```
</div>
 202/250 ━━━━━━━━━━━━━━━━[37m━━━━  22s 474ms/step - loss: 224.0437

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  22s 474ms/step - loss: 224.2074

<div class="k-default-codeblock">
```

```
</div>
 204/250 ━━━━━━━━━━━━━━━━[37m━━━━  21s 473ms/step - loss: 224.4810

<div class="k-default-codeblock">
```

```
</div>
 205/250 ━━━━━━━━━━━━━━━━[37m━━━━  21s 473ms/step - loss: 224.7513

<div class="k-default-codeblock">
```

```
</div>
 206/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 473ms/step - loss: 225.0183

<div class="k-default-codeblock">
```

```
</div>
 207/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 473ms/step - loss: 225.2828

<div class="k-default-codeblock">
```

```
</div>
 208/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 472ms/step - loss: 225.5454

<div class="k-default-codeblock">
```

```
</div>
 209/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 472ms/step - loss: 225.8064

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 472ms/step - loss: 226.0668

<div class="k-default-codeblock">
```

```
</div>
 211/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 472ms/step - loss: 226.3269

<div class="k-default-codeblock">
```

```
</div>
 212/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 471ms/step - loss: 226.5871

<div class="k-default-codeblock">
```

```
</div>
 213/250 ━━━━━━━━━━━━━━━━━[37m━━━  17s 471ms/step - loss: 226.8479

<div class="k-default-codeblock">
```

```
</div>
 214/250 ━━━━━━━━━━━━━━━━━[37m━━━  16s 471ms/step - loss: 227.1099

<div class="k-default-codeblock">
```

```
</div>
 215/250 ━━━━━━━━━━━━━━━━━[37m━━━  16s 471ms/step - loss: 227.3733

<div class="k-default-codeblock">
```

```
</div>
 216/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 470ms/step - loss: 227.6385

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 470ms/step - loss: 227.9056

<div class="k-default-codeblock">
```

```
</div>
 218/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 470ms/step - loss: 228.1748

<div class="k-default-codeblock">
```

```
</div>
 219/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 469ms/step - loss: 228.4462

<div class="k-default-codeblock">
```

```
</div>
 220/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 469ms/step - loss: 228.7199

<div class="k-default-codeblock">
```

```
</div>
 221/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 468ms/step - loss: 228.9958

<div class="k-default-codeblock">
```

```
</div>
 222/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 468ms/step - loss: 229.2738

<div class="k-default-codeblock">
```

```
</div>
 223/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 468ms/step - loss: 229.5540

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 468ms/step - loss: 229.8360

<div class="k-default-codeblock">
```

```
</div>
 225/250 ━━━━━━━━━━━━━━━━━━[37m━━  11s 468ms/step - loss: 230.1200

<div class="k-default-codeblock">
```

```
</div>
 226/250 ━━━━━━━━━━━━━━━━━━[37m━━  11s 467ms/step - loss: 230.4056

<div class="k-default-codeblock">
```

```
</div>
 227/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 467ms/step - loss: 230.6926

<div class="k-default-codeblock">
```

```
</div>
 228/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 467ms/step - loss: 230.9808

<div class="k-default-codeblock">
```

```
</div>
 229/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 467ms/step - loss: 231.2699 

<div class="k-default-codeblock">
```

```
</div>
 230/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 467ms/step - loss: 231.5598

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 467ms/step - loss: 231.8501

<div class="k-default-codeblock">
```

```
</div>
 232/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 467ms/step - loss: 232.1406

<div class="k-default-codeblock">
```

```
</div>
 233/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 467ms/step - loss: 232.4309

<div class="k-default-codeblock">
```

```
</div>
 234/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 467ms/step - loss: 232.7209

<div class="k-default-codeblock">
```

```
</div>
 235/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 467ms/step - loss: 233.0103

<div class="k-default-codeblock">
```

```
</div>
 236/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 466ms/step - loss: 233.2989

<div class="k-default-codeblock">
```

```
</div>
 237/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 466ms/step - loss: 233.5865

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  5s 466ms/step - loss: 233.8728

<div class="k-default-codeblock">
```

```
</div>
 239/250 ━━━━━━━━━━━━━━━━━━━[37m━  5s 466ms/step - loss: 234.1576

<div class="k-default-codeblock">
```

```
</div>
 240/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 466ms/step - loss: 234.4406

<div class="k-default-codeblock">
```

```
</div>
 241/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 467ms/step - loss: 234.7215

<div class="k-default-codeblock">
```

```
</div>
 242/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 467ms/step - loss: 235.0006

<div class="k-default-codeblock">
```

```
</div>
 243/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 467ms/step - loss: 235.2773

<div class="k-default-codeblock">
```

```
</div>
 244/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 467ms/step - loss: 235.5515

<div class="k-default-codeblock">
```

```
</div>
 245/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 467ms/step - loss: 235.8231

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 467ms/step - loss: 236.0920

<div class="k-default-codeblock">
```

```
</div>
 247/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 467ms/step - loss: 236.3580

<div class="k-default-codeblock">
```

```
</div>
 248/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 466ms/step - loss: 236.6210

<div class="k-default-codeblock">
```

```
</div>
 249/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 466ms/step - loss: 236.8812

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 466ms/step - loss: 237.1382

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 117s 466ms/step - loss: 237.3931


<div class="k-default-codeblock">
```
Epoch 7/10

```
</div>
    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  1:52 451ms/step - loss: 230.7783

<div class="k-default-codeblock">
```

```
</div>
   2/250 [37m━━━━━━━━━━━━━━━━━━━━  1:45 424ms/step - loss: 228.7282

<div class="k-default-codeblock">
```

```
</div>
   3/250 [37m━━━━━━━━━━━━━━━━━━━━  1:45 425ms/step - loss: 227.8020

<div class="k-default-codeblock">
```

```
</div>
   4/250 [37m━━━━━━━━━━━━━━━━━━━━  1:44 424ms/step - loss: 226.0346

<div class="k-default-codeblock">
```

```
</div>
   5/250 [37m━━━━━━━━━━━━━━━━━━━━  1:43 423ms/step - loss: 224.6127

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  1:42 421ms/step - loss: 223.5043

<div class="k-default-codeblock">
```

```
</div>
   7/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 419ms/step - loss: 222.2601

<div class="k-default-codeblock">
```

```
</div>
   8/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 418ms/step - loss: 221.2196

<div class="k-default-codeblock">
```

```
</div>
   9/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 417ms/step - loss: 220.2722

<div class="k-default-codeblock">
```

```
</div>
  10/250 [37m━━━━━━━━━━━━━━━━━━━━  1:39 416ms/step - loss: 219.4177

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  1:39 417ms/step - loss: 218.4819

<div class="k-default-codeblock">
```

```
</div>
  12/250 [37m━━━━━━━━━━━━━━━━━━━━  1:39 418ms/step - loss: 233.3924

<div class="k-default-codeblock">
```

```
</div>
  13/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:38 417ms/step - loss: 244.7812

<div class="k-default-codeblock">
```

```
</div>
  14/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:38 418ms/step - loss: 253.7622

<div class="k-default-codeblock">
```

```
</div>
  15/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:38 419ms/step - loss: 261.1548

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 418ms/step - loss: 267.5500

<div class="k-default-codeblock">
```

```
</div>
  17/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 419ms/step - loss: 273.3145

<div class="k-default-codeblock">
```

```
</div>
  18/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 418ms/step - loss: 278.7340

<div class="k-default-codeblock">
```

```
</div>
  19/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 418ms/step - loss: 284.0503

<div class="k-default-codeblock">
```

```
</div>
  20/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 416ms/step - loss: 289.3721

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 416ms/step - loss: 294.7890

<div class="k-default-codeblock">
```

```
</div>
  22/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 415ms/step - loss: 300.3404

<div class="k-default-codeblock">
```

```
</div>
  23/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 415ms/step - loss: 306.0573

<div class="k-default-codeblock">
```

```
</div>
  24/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:33 414ms/step - loss: 311.9369

<div class="k-default-codeblock">
```

```
</div>
  25/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:33 414ms/step - loss: 317.9684

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 414ms/step - loss: 324.1443

<div class="k-default-codeblock">
```

```
</div>
  27/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 415ms/step - loss: 330.4308

<div class="k-default-codeblock">
```

```
</div>
  28/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 414ms/step - loss: 336.8038

<div class="k-default-codeblock">
```

```
</div>
  29/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 418ms/step - loss: 343.2542

<div class="k-default-codeblock">
```

```
</div>
  30/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 422ms/step - loss: 349.7287

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 425ms/step - loss: 356.2101

<div class="k-default-codeblock">
```

```
</div>
  32/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:33 427ms/step - loss: 362.6566

<div class="k-default-codeblock">
```

```
</div>
  33/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 427ms/step - loss: 369.0403

<div class="k-default-codeblock">
```

```
</div>
  34/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 428ms/step - loss: 375.3504

<div class="k-default-codeblock">
```

```
</div>
  35/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 430ms/step - loss: 381.5593

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 431ms/step - loss: 387.6447

<div class="k-default-codeblock">
```

```
</div>
  37/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 432ms/step - loss: 393.5865

<div class="k-default-codeblock">
```

```
</div>
  38/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:31 432ms/step - loss: 399.3713

<div class="k-default-codeblock">
```

```
</div>
  39/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:31 433ms/step - loss: 404.9859

<div class="k-default-codeblock">
```

```
</div>
  40/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:30 432ms/step - loss: 410.4187

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:30 432ms/step - loss: 415.6632

<div class="k-default-codeblock">
```

```
</div>
  42/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:29 432ms/step - loss: 420.7108

<div class="k-default-codeblock">
```

```
</div>
  43/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:29 432ms/step - loss: 425.5550

<div class="k-default-codeblock">
```

```
</div>
  44/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:28 432ms/step - loss: 430.1914

<div class="k-default-codeblock">
```

```
</div>
  45/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:28 431ms/step - loss: 434.6220

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:28 432ms/step - loss: 438.8544

<div class="k-default-codeblock">
```

```
</div>
  47/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 431ms/step - loss: 442.8886

<div class="k-default-codeblock">
```

```
</div>
  48/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 431ms/step - loss: 446.7245

<div class="k-default-codeblock">
```

```
</div>
  49/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 432ms/step - loss: 450.3654

<div class="k-default-codeblock">
```

```
</div>
  50/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:26 431ms/step - loss: 453.8139

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:25 430ms/step - loss: 457.0742

<div class="k-default-codeblock">
```

```
</div>
  52/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:25 430ms/step - loss: 460.1498

<div class="k-default-codeblock">
```

```
</div>
  53/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:24 429ms/step - loss: 463.0476

<div class="k-default-codeblock">
```

```
</div>
  54/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:24 429ms/step - loss: 465.7706

<div class="k-default-codeblock">
```

```
</div>
  55/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:23 428ms/step - loss: 468.3248

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 428ms/step - loss: 470.7177

<div class="k-default-codeblock">
```

```
</div>
  57/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 428ms/step - loss: 472.9520

<div class="k-default-codeblock">
```

```
</div>
  58/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 427ms/step - loss: 475.0370

<div class="k-default-codeblock">
```

```
</div>
  59/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 427ms/step - loss: 476.9779

<div class="k-default-codeblock">
```

```
</div>
  60/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 427ms/step - loss: 478.7801

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 427ms/step - loss: 480.4507

<div class="k-default-codeblock">
```

```
</div>
  62/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 427ms/step - loss: 481.9948

<div class="k-default-codeblock">
```

```
</div>
  63/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:19 426ms/step - loss: 483.4156

<div class="k-default-codeblock">
```

```
</div>
  64/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:19 426ms/step - loss: 484.7220

<div class="k-default-codeblock">
```

```
</div>
  65/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:18 425ms/step - loss: 485.9168

<div class="k-default-codeblock">
```

```
</div>
  66/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:18 425ms/step - loss: 487.0049

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:17 425ms/step - loss: 487.9926

<div class="k-default-codeblock">
```

```
</div>
  68/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:17 425ms/step - loss: 488.8860

<div class="k-default-codeblock">
```

```
</div>
  69/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 425ms/step - loss: 489.6893

<div class="k-default-codeblock">
```

```
</div>
  70/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 425ms/step - loss: 490.4074

<div class="k-default-codeblock">
```

```
</div>
  71/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 425ms/step - loss: 491.0452

<div class="k-default-codeblock">
```

```
</div>
  72/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 424ms/step - loss: 491.6037

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 425ms/step - loss: 492.0884

<div class="k-default-codeblock">
```

```
</div>
  74/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 424ms/step - loss: 492.5034

<div class="k-default-codeblock">
```

```
</div>
  75/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:14 424ms/step - loss: 492.8525

<div class="k-default-codeblock">
```

```
</div>
  76/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:13 423ms/step - loss: 493.1375

<div class="k-default-codeblock">
```

```
</div>
  77/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:13 423ms/step - loss: 493.3639

<div class="k-default-codeblock">
```

```
</div>
  78/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 423ms/step - loss: 493.5314

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 423ms/step - loss: 493.6447

<div class="k-default-codeblock">
```

```
</div>
  80/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 422ms/step - loss: 493.7096

<div class="k-default-codeblock">
```

```
</div>
  81/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 422ms/step - loss: 493.7257

<div class="k-default-codeblock">
```

```
</div>
  82/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 422ms/step - loss: 493.6966

<div class="k-default-codeblock">
```

```
</div>
  83/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 422ms/step - loss: 493.6244

<div class="k-default-codeblock">
```

```
</div>
  84/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 421ms/step - loss: 493.5103

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 422ms/step - loss: 493.3579

<div class="k-default-codeblock">
```

```
</div>
  86/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 421ms/step - loss: 493.1701

<div class="k-default-codeblock">
```

```
</div>
  87/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 421ms/step - loss: 492.9467

<div class="k-default-codeblock">
```

```
</div>
  88/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:08 421ms/step - loss: 492.6921

<div class="k-default-codeblock">
```

```
</div>
  89/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 421ms/step - loss: 492.4088

<div class="k-default-codeblock">
```

```
</div>
  90/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 421ms/step - loss: 492.0965

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 421ms/step - loss: 491.7561

<div class="k-default-codeblock">
```

```
</div>
  92/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 420ms/step - loss: 491.3908

<div class="k-default-codeblock">
```

```
</div>
  93/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 420ms/step - loss: 491.0007

<div class="k-default-codeblock">
```

```
</div>
  94/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 420ms/step - loss: 490.5865

<div class="k-default-codeblock">
```

```
</div>
  95/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 420ms/step - loss: 490.1513

<div class="k-default-codeblock">
```

```
</div>
  96/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 419ms/step - loss: 489.6958

<div class="k-default-codeblock">
```

```
</div>
  97/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 419ms/step - loss: 489.2214

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 419ms/step - loss: 488.7286

<div class="k-default-codeblock">
```

```
</div>
  99/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 419ms/step - loss: 488.2183

<div class="k-default-codeblock">
```

```
</div>
 100/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:02 419ms/step - loss: 487.6928

<div class="k-default-codeblock">
```

```
</div>
 101/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:02 419ms/step - loss: 487.1534

<div class="k-default-codeblock">
```

```
</div>
 102/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:02 419ms/step - loss: 486.6000

<div class="k-default-codeblock">
```

```
</div>
 103/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 420ms/step - loss: 486.0341

<div class="k-default-codeblock">
```

```
</div>
 104/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 420ms/step - loss: 485.4558

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 420ms/step - loss: 484.8666

<div class="k-default-codeblock">
```

```
</div>
 106/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 420ms/step - loss: 484.2667

<div class="k-default-codeblock">
```

```
</div>
 107/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 425ms/step - loss: 483.6568

<div class="k-default-codeblock">
```

```
</div>
 108/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 426ms/step - loss: 483.0373

<div class="k-default-codeblock">
```

```
</div>
 109/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 426ms/step - loss: 482.4080 

<div class="k-default-codeblock">
```

```
</div>
 110/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 426ms/step - loss: 481.7707

<div class="k-default-codeblock">
```

```
</div>
 111/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 427ms/step - loss: 481.1264

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 427ms/step - loss: 480.4753

<div class="k-default-codeblock">
```

```
</div>
 113/250 ━━━━━━━━━[37m━━━━━━━━━━━  58s 428ms/step - loss: 479.8244

<div class="k-default-codeblock">
```

```
</div>
 114/250 ━━━━━━━━━[37m━━━━━━━━━━━  58s 428ms/step - loss: 479.1668

<div class="k-default-codeblock">
```

```
</div>
 115/250 ━━━━━━━━━[37m━━━━━━━━━━━  57s 429ms/step - loss: 478.5037

<div class="k-default-codeblock">
```

```
</div>
 116/250 ━━━━━━━━━[37m━━━━━━━━━━━  57s 429ms/step - loss: 477.8365

<div class="k-default-codeblock">
```

```
</div>
 117/250 ━━━━━━━━━[37m━━━━━━━━━━━  57s 429ms/step - loss: 477.1657

<div class="k-default-codeblock">
```

```
</div>
 118/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 429ms/step - loss: 476.4924

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 430ms/step - loss: 475.8179

<div class="k-default-codeblock">
```

```
</div>
 120/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 430ms/step - loss: 475.1429

<div class="k-default-codeblock">
```

```
</div>
 121/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 431ms/step - loss: 474.4682

<div class="k-default-codeblock">
```

```
</div>
 122/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 431ms/step - loss: 473.7940

<div class="k-default-codeblock">
```

```
</div>
 123/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 431ms/step - loss: 473.1212

<div class="k-default-codeblock">
```

```
</div>
 124/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 432ms/step - loss: 472.4501

<div class="k-default-codeblock">
```

```
</div>
 125/250 ━━━━━━━━━━[37m━━━━━━━━━━  53s 432ms/step - loss: 471.7816

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  53s 432ms/step - loss: 471.1150

<div class="k-default-codeblock">
```

```
</div>
 127/250 ━━━━━━━━━━[37m━━━━━━━━━━  53s 432ms/step - loss: 470.4509

<div class="k-default-codeblock">
```

```
</div>
 128/250 ━━━━━━━━━━[37m━━━━━━━━━━  52s 433ms/step - loss: 469.7891

<div class="k-default-codeblock">
```

```
</div>
 129/250 ━━━━━━━━━━[37m━━━━━━━━━━  52s 433ms/step - loss: 469.1298

<div class="k-default-codeblock">
```

```
</div>
 130/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 433ms/step - loss: 468.4732

<div class="k-default-codeblock">
```

```
</div>
 131/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 434ms/step - loss: 467.8195

<div class="k-default-codeblock">
```

```
</div>
 132/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 434ms/step - loss: 467.1685

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 434ms/step - loss: 466.5202

<div class="k-default-codeblock">
```

```
</div>
 134/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 434ms/step - loss: 465.8748

<div class="k-default-codeblock">
```

```
</div>
 135/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 435ms/step - loss: 465.2325

<div class="k-default-codeblock">
```

```
</div>
 136/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 435ms/step - loss: 464.5927

<div class="k-default-codeblock">
```

```
</div>
 137/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 436ms/step - loss: 463.9560

<div class="k-default-codeblock">
```

```
</div>
 138/250 ━━━━━━━━━━━[37m━━━━━━━━━  48s 436ms/step - loss: 463.3223

<div class="k-default-codeblock">
```

```
</div>
 139/250 ━━━━━━━━━━━[37m━━━━━━━━━  48s 437ms/step - loss: 462.6910

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  48s 437ms/step - loss: 462.0624

<div class="k-default-codeblock">
```

```
</div>
 141/250 ━━━━━━━━━━━[37m━━━━━━━━━  47s 439ms/step - loss: 461.4359

<div class="k-default-codeblock">
```

```
</div>
 142/250 ━━━━━━━━━━━[37m━━━━━━━━━  47s 439ms/step - loss: 460.8118

<div class="k-default-codeblock">
```

```
</div>
 143/250 ━━━━━━━━━━━[37m━━━━━━━━━  47s 440ms/step - loss: 460.1898

<div class="k-default-codeblock">
```

```
</div>
 144/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 440ms/step - loss: 459.5699

<div class="k-default-codeblock">
```

```
</div>
 145/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 440ms/step - loss: 458.9531

<div class="k-default-codeblock">
```

```
</div>
 146/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 440ms/step - loss: 458.3379

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 441ms/step - loss: 457.7248

<div class="k-default-codeblock">
```

```
</div>
 148/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 441ms/step - loss: 457.1140

<div class="k-default-codeblock">
```

```
</div>
 149/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 441ms/step - loss: 456.5059

<div class="k-default-codeblock">
```

```
</div>
 150/250 ━━━━━━━━━━━━[37m━━━━━━━━  44s 441ms/step - loss: 455.9000

<div class="k-default-codeblock">
```

```
</div>
 151/250 ━━━━━━━━━━━━[37m━━━━━━━━  43s 441ms/step - loss: 455.2958

<div class="k-default-codeblock">
```

```
</div>
 152/250 ━━━━━━━━━━━━[37m━━━━━━━━  43s 440ms/step - loss: 454.6937

<div class="k-default-codeblock">
```

```
</div>
 153/250 ━━━━━━━━━━━━[37m━━━━━━━━  42s 441ms/step - loss: 454.0937

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  42s 442ms/step - loss: 453.4959

<div class="k-default-codeblock">
```

```
</div>
 155/250 ━━━━━━━━━━━━[37m━━━━━━━━  42s 444ms/step - loss: 452.9001

<div class="k-default-codeblock">
```

```
</div>
 156/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 444ms/step - loss: 452.3061

<div class="k-default-codeblock">
```

```
</div>
 157/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 447ms/step - loss: 451.7140

<div class="k-default-codeblock">
```

```
</div>
 158/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 447ms/step - loss: 451.1235

<div class="k-default-codeblock">
```

```
</div>
 159/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 447ms/step - loss: 450.5345

<div class="k-default-codeblock">
```

```
</div>
 160/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 446ms/step - loss: 449.9469

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 446ms/step - loss: 449.3609

<div class="k-default-codeblock">
```

```
</div>
 162/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 446ms/step - loss: 448.7766

<div class="k-default-codeblock">
```

```
</div>
 163/250 ━━━━━━━━━━━━━[37m━━━━━━━  38s 446ms/step - loss: 448.1938

<div class="k-default-codeblock">
```

```
</div>
 164/250 ━━━━━━━━━━━━━[37m━━━━━━━  38s 445ms/step - loss: 447.6124

<div class="k-default-codeblock">
```

```
</div>
 165/250 ━━━━━━━━━━━━━[37m━━━━━━━  37s 445ms/step - loss: 447.0326

<div class="k-default-codeblock">
```

```
</div>
 166/250 ━━━━━━━━━━━━━[37m━━━━━━━  37s 445ms/step - loss: 446.4539

<div class="k-default-codeblock">
```

```
</div>
 167/250 ━━━━━━━━━━━━━[37m━━━━━━━  36s 445ms/step - loss: 445.8768

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  36s 444ms/step - loss: 445.3011

<div class="k-default-codeblock">
```

```
</div>
 169/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 444ms/step - loss: 444.7269

<div class="k-default-codeblock">
```

```
</div>
 170/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 444ms/step - loss: 444.1541

<div class="k-default-codeblock">
```

```
</div>
 171/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 444ms/step - loss: 443.5829

<div class="k-default-codeblock">
```

```
</div>
 172/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 444ms/step - loss: 443.0133

<div class="k-default-codeblock">
```

```
</div>
 173/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 443ms/step - loss: 442.4450

<div class="k-default-codeblock">
```

```
</div>
 174/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 443ms/step - loss: 441.8784

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  33s 443ms/step - loss: 441.3132

<div class="k-default-codeblock">
```

```
</div>
 176/250 ━━━━━━━━━━━━━━[37m━━━━━━  32s 443ms/step - loss: 440.7495

<div class="k-default-codeblock">
```

```
</div>
 177/250 ━━━━━━━━━━━━━━[37m━━━━━━  32s 443ms/step - loss: 440.1870

<div class="k-default-codeblock">
```

```
</div>
 178/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 443ms/step - loss: 439.6260

<div class="k-default-codeblock">
```

```
</div>
 179/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 442ms/step - loss: 439.0662

<div class="k-default-codeblock">
```

```
</div>
 180/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 442ms/step - loss: 438.5078

<div class="k-default-codeblock">
```

```
</div>
 181/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 442ms/step - loss: 437.9511

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 442ms/step - loss: 437.3962

<div class="k-default-codeblock">
```

```
</div>
 183/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 442ms/step - loss: 436.8425

<div class="k-default-codeblock">
```

```
</div>
 184/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 442ms/step - loss: 436.2903

<div class="k-default-codeblock">
```

```
</div>
 185/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 442ms/step - loss: 435.7395

<div class="k-default-codeblock">
```

```
</div>
 186/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 441ms/step - loss: 435.1902

<div class="k-default-codeblock">
```

```
</div>
 187/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 441ms/step - loss: 434.6425

<div class="k-default-codeblock">
```

```
</div>
 188/250 ━━━━━━━━━━━━━━━[37m━━━━━  27s 441ms/step - loss: 434.0967

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  26s 441ms/step - loss: 433.5528

<div class="k-default-codeblock">
```

```
</div>
 190/250 ━━━━━━━━━━━━━━━[37m━━━━━  26s 441ms/step - loss: 433.0105

<div class="k-default-codeblock">
```

```
</div>
 191/250 ━━━━━━━━━━━━━━━[37m━━━━━  26s 441ms/step - loss: 432.4698

<div class="k-default-codeblock">
```

```
</div>
 192/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 441ms/step - loss: 431.9305

<div class="k-default-codeblock">
```

```
</div>
 193/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 441ms/step - loss: 431.3928

<div class="k-default-codeblock">
```

```
</div>
 194/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 440ms/step - loss: 430.8564

<div class="k-default-codeblock">
```

```
</div>
 195/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 440ms/step - loss: 430.3218

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 440ms/step - loss: 429.7888

<div class="k-default-codeblock">
```

```
</div>
 197/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 440ms/step - loss: 429.2577

<div class="k-default-codeblock">
```

```
</div>
 198/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 440ms/step - loss: 428.7283

<div class="k-default-codeblock">
```

```
</div>
 199/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 440ms/step - loss: 428.2005

<div class="k-default-codeblock">
```

```
</div>
 200/250 ━━━━━━━━━━━━━━━━[37m━━━━  21s 439ms/step - loss: 427.6748

<div class="k-default-codeblock">
```

```
</div>
 201/250 ━━━━━━━━━━━━━━━━[37m━━━━  21s 439ms/step - loss: 427.1506

<div class="k-default-codeblock">
```

```
</div>
 202/250 ━━━━━━━━━━━━━━━━[37m━━━━  21s 439ms/step - loss: 426.6282

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 439ms/step - loss: 426.1073

<div class="k-default-codeblock">
```

```
</div>
 204/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 439ms/step - loss: 425.5881

<div class="k-default-codeblock">
```

```
</div>
 205/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 439ms/step - loss: 425.0705

<div class="k-default-codeblock">
```

```
</div>
 206/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 438ms/step - loss: 424.5545

<div class="k-default-codeblock">
```

```
</div>
 207/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 438ms/step - loss: 424.0403

<div class="k-default-codeblock">
```

```
</div>
 208/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 438ms/step - loss: 423.5276

<div class="k-default-codeblock">
```

```
</div>
 209/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 438ms/step - loss: 423.0164

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 438ms/step - loss: 422.5070

<div class="k-default-codeblock">
```

```
</div>
 211/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 438ms/step - loss: 421.9994

<div class="k-default-codeblock">
```

```
</div>
 212/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 437ms/step - loss: 421.4934

<div class="k-default-codeblock">
```

```
</div>
 213/250 ━━━━━━━━━━━━━━━━━[37m━━━  16s 437ms/step - loss: 420.9890

<div class="k-default-codeblock">
```

```
</div>
 214/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 437ms/step - loss: 420.4868

<div class="k-default-codeblock">
```

```
</div>
 215/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 437ms/step - loss: 419.9861

<div class="k-default-codeblock">
```

```
</div>
 216/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 437ms/step - loss: 419.4871

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 437ms/step - loss: 418.9898

<div class="k-default-codeblock">
```

```
</div>
 218/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 437ms/step - loss: 418.4940

<div class="k-default-codeblock">
```

```
</div>
 219/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 437ms/step - loss: 418.0000

<div class="k-default-codeblock">
```

```
</div>
 220/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 437ms/step - loss: 417.5076

<div class="k-default-codeblock">
```

```
</div>
 221/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 437ms/step - loss: 417.0168

<div class="k-default-codeblock">
```

```
</div>
 222/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 437ms/step - loss: 416.5278

<div class="k-default-codeblock">
```

```
</div>
 223/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 436ms/step - loss: 416.0404

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 436ms/step - loss: 415.5549

<div class="k-default-codeblock">
```

```
</div>
 225/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 436ms/step - loss: 415.0713

<div class="k-default-codeblock">
```

```
</div>
 226/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 436ms/step - loss: 414.5893

<div class="k-default-codeblock">
```

```
</div>
 227/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 437ms/step - loss: 414.1091

<div class="k-default-codeblock">
```

```
</div>
 228/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 436ms/step - loss: 413.6305 

<div class="k-default-codeblock">
```

```
</div>
 229/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 436ms/step - loss: 413.1535

<div class="k-default-codeblock">
```

```
</div>
 230/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 436ms/step - loss: 412.6786

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 436ms/step - loss: 412.2053

<div class="k-default-codeblock">
```

```
</div>
 232/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 436ms/step - loss: 411.7339

<div class="k-default-codeblock">
```

```
</div>
 233/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 436ms/step - loss: 411.2644

<div class="k-default-codeblock">
```

```
</div>
 234/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 436ms/step - loss: 410.7965

<div class="k-default-codeblock">
```

```
</div>
 235/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 436ms/step - loss: 410.3302

<div class="k-default-codeblock">
```

```
</div>
 236/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 436ms/step - loss: 409.8654

<div class="k-default-codeblock">
```

```
</div>
 237/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 436ms/step - loss: 409.4025

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  5s 436ms/step - loss: 408.9413

<div class="k-default-codeblock">
```

```
</div>
 239/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 436ms/step - loss: 408.4817

<div class="k-default-codeblock">
```

```
</div>
 240/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 436ms/step - loss: 408.0238

<div class="k-default-codeblock">
```

```
</div>
 241/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 436ms/step - loss: 407.5676

<div class="k-default-codeblock">
```

```
</div>
 242/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 436ms/step - loss: 407.1132

<div class="k-default-codeblock">
```

```
</div>
 243/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 436ms/step - loss: 406.6604

<div class="k-default-codeblock">
```

```
</div>
 244/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 436ms/step - loss: 406.2091

<div class="k-default-codeblock">
```

```
</div>
 245/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 436ms/step - loss: 405.7596

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 436ms/step - loss: 405.3119

<div class="k-default-codeblock">
```

```
</div>
 247/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 436ms/step - loss: 404.8656

<div class="k-default-codeblock">
```

```
</div>
 248/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 436ms/step - loss: 404.4208

<div class="k-default-codeblock">
```

```
</div>
 249/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 436ms/step - loss: 403.9777

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 436ms/step - loss: 403.5362

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 109s 436ms/step - loss: 403.0982


<div class="k-default-codeblock">
```
Epoch 8/10

```
</div>
    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  1:59 481ms/step - loss: 167.5494

<div class="k-default-codeblock">
```

```
</div>
   2/250 [37m━━━━━━━━━━━━━━━━━━━━  1:43 419ms/step - loss: 166.5712

<div class="k-default-codeblock">
```

```
</div>
   3/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 412ms/step - loss: 165.9277

<div class="k-default-codeblock">
```

```
</div>
   4/250 [37m━━━━━━━━━━━━━━━━━━━━  1:45 427ms/step - loss: 165.7298

<div class="k-default-codeblock">
```

```
</div>
   5/250 [37m━━━━━━━━━━━━━━━━━━━━  1:47 439ms/step - loss: 165.9603

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  1:46 438ms/step - loss: 166.5857

<div class="k-default-codeblock">
```

```
</div>
   7/250 [37m━━━━━━━━━━━━━━━━━━━━  1:47 440ms/step - loss: 167.1435

<div class="k-default-codeblock">
```

```
</div>
   8/250 [37m━━━━━━━━━━━━━━━━━━━━  1:46 442ms/step - loss: 167.9836

<div class="k-default-codeblock">
```

```
</div>
   9/250 [37m━━━━━━━━━━━━━━━━━━━━  1:45 438ms/step - loss: 168.5315

<div class="k-default-codeblock">
```

```
</div>
  10/250 [37m━━━━━━━━━━━━━━━━━━━━  1:44 437ms/step - loss: 168.8648

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  1:44 436ms/step - loss: 169.1477

<div class="k-default-codeblock">
```

```
</div>
  12/250 [37m━━━━━━━━━━━━━━━━━━━━  1:43 436ms/step - loss: 169.3245

<div class="k-default-codeblock">
```

```
</div>
  13/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:43 438ms/step - loss: 169.4775

<div class="k-default-codeblock">
```

```
</div>
  14/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:43 437ms/step - loss: 169.6300

<div class="k-default-codeblock">
```

```
</div>
  15/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:41 434ms/step - loss: 169.7367

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:41 432ms/step - loss: 169.8109

<div class="k-default-codeblock">
```

```
</div>
  17/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:40 431ms/step - loss: 169.8900

<div class="k-default-codeblock">
```

```
</div>
  18/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:39 431ms/step - loss: 169.9803

<div class="k-default-codeblock">
```

```
</div>
  19/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:39 431ms/step - loss: 170.0796

<div class="k-default-codeblock">
```

```
</div>
  20/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:39 431ms/step - loss: 170.1600

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:38 432ms/step - loss: 170.2360

<div class="k-default-codeblock">
```

```
</div>
  22/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:38 431ms/step - loss: 170.2980

<div class="k-default-codeblock">
```

```
</div>
  23/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 431ms/step - loss: 170.3607

<div class="k-default-codeblock">
```

```
</div>
  24/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 432ms/step - loss: 170.4314

<div class="k-default-codeblock">
```

```
</div>
  25/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:36 431ms/step - loss: 170.5068

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:36 431ms/step - loss: 170.5768

<div class="k-default-codeblock">
```

```
</div>
  27/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:36 431ms/step - loss: 170.6309

<div class="k-default-codeblock">
```

```
</div>
  28/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:35 432ms/step - loss: 170.6772

<div class="k-default-codeblock">
```

```
</div>
  29/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:35 433ms/step - loss: 170.7230

<div class="k-default-codeblock">
```

```
</div>
  30/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:35 434ms/step - loss: 170.7707

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:35 436ms/step - loss: 170.8111

<div class="k-default-codeblock">
```

```
</div>
  32/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:35 438ms/step - loss: 170.8510

<div class="k-default-codeblock">
```

```
</div>
  33/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:35 438ms/step - loss: 170.8908

<div class="k-default-codeblock">
```

```
</div>
  34/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:35 440ms/step - loss: 170.9207

<div class="k-default-codeblock">
```

```
</div>
  35/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:34 441ms/step - loss: 170.9514

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:35 449ms/step - loss: 170.9819

<div class="k-default-codeblock">
```

```
</div>
  37/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:35 449ms/step - loss: 171.0122

<div class="k-default-codeblock">
```

```
</div>
  38/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:35 449ms/step - loss: 171.0372

<div class="k-default-codeblock">
```

```
</div>
  39/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:35 451ms/step - loss: 171.0627

<div class="k-default-codeblock">
```

```
</div>
  40/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 452ms/step - loss: 171.0977

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 452ms/step - loss: 171.1311

<div class="k-default-codeblock">
```

```
</div>
  42/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 453ms/step - loss: 171.1653

<div class="k-default-codeblock">
```

```
</div>
  43/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:33 453ms/step - loss: 171.2005

<div class="k-default-codeblock">
```

```
</div>
  44/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:33 454ms/step - loss: 171.2353

<div class="k-default-codeblock">
```

```
</div>
  45/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:33 457ms/step - loss: 171.2660

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:33 459ms/step - loss: 171.3053

<div class="k-default-codeblock">
```

```
</div>
  47/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 463ms/step - loss: 171.3390

<div class="k-default-codeblock">
```

```
</div>
  48/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 468ms/step - loss: 171.3695

<div class="k-default-codeblock">
```

```
</div>
  49/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:34 468ms/step - loss: 171.4000

<div class="k-default-codeblock">
```

```
</div>
  50/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:33 468ms/step - loss: 171.4255

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:33 470ms/step - loss: 171.4530

<div class="k-default-codeblock">
```

```
</div>
  52/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:33 471ms/step - loss: 171.4832

<div class="k-default-codeblock">
```

```
</div>
  53/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:33 473ms/step - loss: 171.5135

<div class="k-default-codeblock">
```

```
</div>
  54/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:33 476ms/step - loss: 171.5460

<div class="k-default-codeblock">
```

```
</div>
  55/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:33 478ms/step - loss: 171.5745

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:32 479ms/step - loss: 171.5997

<div class="k-default-codeblock">
```

```
</div>
  57/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:32 480ms/step - loss: 171.6283

<div class="k-default-codeblock">
```

```
</div>
  58/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:32 480ms/step - loss: 171.6547

<div class="k-default-codeblock">
```

```
</div>
  59/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:31 479ms/step - loss: 171.6827

<div class="k-default-codeblock">
```

```
</div>
  60/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:31 481ms/step - loss: 171.7096

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:31 482ms/step - loss: 171.7334

<div class="k-default-codeblock">
```

```
</div>
  62/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:30 483ms/step - loss: 171.7561

<div class="k-default-codeblock">
```

```
</div>
  63/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:30 484ms/step - loss: 171.8493

<div class="k-default-codeblock">
```

```
</div>
  64/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:29 484ms/step - loss: 171.9426

<div class="k-default-codeblock">
```

```
</div>
  65/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:29 485ms/step - loss: 172.0470

<div class="k-default-codeblock">
```

```
</div>
  66/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:29 487ms/step - loss: 172.1695

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:29 487ms/step - loss: 172.3196

<div class="k-default-codeblock">
```

```
</div>
  68/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:28 489ms/step - loss: 172.5052

<div class="k-default-codeblock">
```

```
</div>
  69/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:28 489ms/step - loss: 172.7334

<div class="k-default-codeblock">
```

```
</div>
  70/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:28 490ms/step - loss: 173.0098

<div class="k-default-codeblock">
```

```
</div>
  71/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:27 491ms/step - loss: 173.3424

<div class="k-default-codeblock">
```

```
</div>
  72/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:27 491ms/step - loss: 173.7328

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:27 494ms/step - loss: 174.1856

<div class="k-default-codeblock">
```

```
</div>
  74/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:26 493ms/step - loss: 174.7045

<div class="k-default-codeblock">
```

```
</div>
  75/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:26 492ms/step - loss: 175.2906

<div class="k-default-codeblock">
```

```
</div>
  76/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:25 492ms/step - loss: 175.9435

<div class="k-default-codeblock">
```

```
</div>
  77/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:24 491ms/step - loss: 176.6629

<div class="k-default-codeblock">
```

```
</div>
  78/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:24 490ms/step - loss: 177.4480

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:23 489ms/step - loss: 178.2950

<div class="k-default-codeblock">
```

```
</div>
  80/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:23 489ms/step - loss: 179.1989

<div class="k-default-codeblock">
```

```
</div>
  81/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:22 488ms/step - loss: 180.1562

<div class="k-default-codeblock">
```

```
</div>
  82/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:21 487ms/step - loss: 181.1624

<div class="k-default-codeblock">
```

```
</div>
  83/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:21 487ms/step - loss: 182.2160

<div class="k-default-codeblock">
```

```
</div>
  84/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:20 487ms/step - loss: 183.3137

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:20 486ms/step - loss: 184.4529

<div class="k-default-codeblock">
```

```
</div>
  86/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:19 486ms/step - loss: 185.6261

<div class="k-default-codeblock">
```

```
</div>
  87/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:19 486ms/step - loss: 186.8312

<div class="k-default-codeblock">
```

```
</div>
  88/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:18 485ms/step - loss: 188.0626

<div class="k-default-codeblock">
```

```
</div>
  89/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:17 484ms/step - loss: 189.3195

<div class="k-default-codeblock">
```

```
</div>
  90/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:17 484ms/step - loss: 190.5970

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:16 484ms/step - loss: 191.8893

<div class="k-default-codeblock">
```

```
</div>
  92/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:16 483ms/step - loss: 193.1962

<div class="k-default-codeblock">
```

```
</div>
  93/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:15 482ms/step - loss: 194.5132

<div class="k-default-codeblock">
```

```
</div>
  94/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:15 481ms/step - loss: 195.8352

<div class="k-default-codeblock">
```

```
</div>
  95/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:14 481ms/step - loss: 197.1619

<div class="k-default-codeblock">
```

```
</div>
  96/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:14 481ms/step - loss: 198.4901

<div class="k-default-codeblock">
```

```
</div>
  97/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:13 481ms/step - loss: 199.8166

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:13 482ms/step - loss: 201.1390

<div class="k-default-codeblock">
```

```
</div>
  99/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:12 483ms/step - loss: 202.4565

<div class="k-default-codeblock">
```

```
</div>
 100/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:12 483ms/step - loss: 203.7667

<div class="k-default-codeblock">
```

```
</div>
 101/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:11 483ms/step - loss: 205.0674

<div class="k-default-codeblock">
```

```
</div>
 102/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:11 482ms/step - loss: 206.3581

<div class="k-default-codeblock">
```

```
</div>
 103/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:10 482ms/step - loss: 207.6366

<div class="k-default-codeblock">
```

```
</div>
 104/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:10 481ms/step - loss: 208.9015

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:09 481ms/step - loss: 210.1523

<div class="k-default-codeblock">
```

```
</div>
 106/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:09 481ms/step - loss: 211.3879

<div class="k-default-codeblock">
```

```
</div>
 107/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:08 481ms/step - loss: 212.6073

<div class="k-default-codeblock">
```

```
</div>
 108/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:08 481ms/step - loss: 213.8100

<div class="k-default-codeblock">
```

```
</div>
 109/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:07 481ms/step - loss: 214.9946

<div class="k-default-codeblock">
```

```
</div>
 110/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:07 481ms/step - loss: 216.1614

<div class="k-default-codeblock">
```

```
</div>
 111/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:06 480ms/step - loss: 217.3096

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:06 480ms/step - loss: 218.4381

<div class="k-default-codeblock">
```

```
</div>
 113/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:05 479ms/step - loss: 219.5470

<div class="k-default-codeblock">
```

```
</div>
 114/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:05 479ms/step - loss: 220.6354

<div class="k-default-codeblock">
```

```
</div>
 115/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:04 479ms/step - loss: 221.7028

<div class="k-default-codeblock">
```

```
</div>
 116/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:04 478ms/step - loss: 222.7497

<div class="k-default-codeblock">
```

```
</div>
 117/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:03 478ms/step - loss: 223.7750

<div class="k-default-codeblock">
```

```
</div>
 118/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:02 477ms/step - loss: 224.7790

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:02 477ms/step - loss: 225.7630

<div class="k-default-codeblock">
```

```
</div>
 120/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:01 476ms/step - loss: 226.7258

<div class="k-default-codeblock">
```

```
</div>
 121/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:01 476ms/step - loss: 227.6678

<div class="k-default-codeblock">
```

```
</div>
 122/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:00 476ms/step - loss: 228.5887

<div class="k-default-codeblock">
```

```
</div>
 123/250 ━━━━━━━━━[37m━━━━━━━━━━━  1:00 475ms/step - loss: 229.4889

<div class="k-default-codeblock">
```

```
</div>
 124/250 ━━━━━━━━━[37m━━━━━━━━━━━  59s 475ms/step - loss: 230.3685 

<div class="k-default-codeblock">
```

```
</div>
 125/250 ━━━━━━━━━━[37m━━━━━━━━━━  59s 475ms/step - loss: 231.2277

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  58s 475ms/step - loss: 232.0662

<div class="k-default-codeblock">
```

```
</div>
 127/250 ━━━━━━━━━━[37m━━━━━━━━━━  58s 474ms/step - loss: 232.8841

<div class="k-default-codeblock">
```

```
</div>
 128/250 ━━━━━━━━━━[37m━━━━━━━━━━  57s 474ms/step - loss: 233.6818

<div class="k-default-codeblock">
```

```
</div>
 129/250 ━━━━━━━━━━[37m━━━━━━━━━━  57s 474ms/step - loss: 234.4601

<div class="k-default-codeblock">
```

```
</div>
 130/250 ━━━━━━━━━━[37m━━━━━━━━━━  56s 474ms/step - loss: 235.2189

<div class="k-default-codeblock">
```

```
</div>
 131/250 ━━━━━━━━━━[37m━━━━━━━━━━  56s 474ms/step - loss: 235.9593

<div class="k-default-codeblock">
```

```
</div>
 132/250 ━━━━━━━━━━[37m━━━━━━━━━━  55s 473ms/step - loss: 236.6811

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  55s 473ms/step - loss: 237.3843

<div class="k-default-codeblock">
```

```
</div>
 134/250 ━━━━━━━━━━[37m━━━━━━━━━━  54s 472ms/step - loss: 238.0698

<div class="k-default-codeblock">
```

```
</div>
 135/250 ━━━━━━━━━━[37m━━━━━━━━━━  54s 472ms/step - loss: 238.7377

<div class="k-default-codeblock">
```

```
</div>
 136/250 ━━━━━━━━━━[37m━━━━━━━━━━  53s 472ms/step - loss: 239.3880

<div class="k-default-codeblock">
```

```
</div>
 137/250 ━━━━━━━━━━[37m━━━━━━━━━━  53s 472ms/step - loss: 240.0218

<div class="k-default-codeblock">
```

```
</div>
 138/250 ━━━━━━━━━━━[37m━━━━━━━━━  52s 471ms/step - loss: 240.6387

<div class="k-default-codeblock">
```

```
</div>
 139/250 ━━━━━━━━━━━[37m━━━━━━━━━  52s 471ms/step - loss: 241.2392

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  51s 470ms/step - loss: 241.8240

<div class="k-default-codeblock">
```

```
</div>
 141/250 ━━━━━━━━━━━[37m━━━━━━━━━  51s 470ms/step - loss: 242.3929

<div class="k-default-codeblock">
```

```
</div>
 142/250 ━━━━━━━━━━━[37m━━━━━━━━━  50s 470ms/step - loss: 242.9487

<div class="k-default-codeblock">
```

```
</div>
 143/250 ━━━━━━━━━━━[37m━━━━━━━━━  50s 470ms/step - loss: 243.4895

<div class="k-default-codeblock">
```

```
</div>
 144/250 ━━━━━━━━━━━[37m━━━━━━━━━  49s 469ms/step - loss: 244.0163

<div class="k-default-codeblock">
```

```
</div>
 145/250 ━━━━━━━━━━━[37m━━━━━━━━━  49s 469ms/step - loss: 244.5294

<div class="k-default-codeblock">
```

```
</div>
 146/250 ━━━━━━━━━━━[37m━━━━━━━━━  48s 468ms/step - loss: 245.0295

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  48s 468ms/step - loss: 245.5170

<div class="k-default-codeblock">
```

```
</div>
 148/250 ━━━━━━━━━━━[37m━━━━━━━━━  47s 467ms/step - loss: 245.9920

<div class="k-default-codeblock">
```

```
</div>
 149/250 ━━━━━━━━━━━[37m━━━━━━━━━  47s 467ms/step - loss: 246.4551

<div class="k-default-codeblock">
```

```
</div>
 150/250 ━━━━━━━━━━━━[37m━━━━━━━━  46s 467ms/step - loss: 246.9066

<div class="k-default-codeblock">
```

```
</div>
 151/250 ━━━━━━━━━━━━[37m━━━━━━━━  46s 467ms/step - loss: 247.3466

<div class="k-default-codeblock">
```

```
</div>
 152/250 ━━━━━━━━━━━━[37m━━━━━━━━  45s 467ms/step - loss: 247.7758

<div class="k-default-codeblock">
```

```
</div>
 153/250 ━━━━━━━━━━━━[37m━━━━━━━━  45s 467ms/step - loss: 248.1946

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  44s 467ms/step - loss: 248.6032

<div class="k-default-codeblock">
```

```
</div>
 155/250 ━━━━━━━━━━━━[37m━━━━━━━━  44s 467ms/step - loss: 249.0015

<div class="k-default-codeblock">
```

```
</div>
 156/250 ━━━━━━━━━━━━[37m━━━━━━━━  43s 467ms/step - loss: 249.3898

<div class="k-default-codeblock">
```

```
</div>
 157/250 ━━━━━━━━━━━━[37m━━━━━━━━  43s 466ms/step - loss: 249.7688

<div class="k-default-codeblock">
```

```
</div>
 158/250 ━━━━━━━━━━━━[37m━━━━━━━━  42s 466ms/step - loss: 250.1387

<div class="k-default-codeblock">
```

```
</div>
 159/250 ━━━━━━━━━━━━[37m━━━━━━━━  42s 465ms/step - loss: 250.4993

<div class="k-default-codeblock">
```

```
</div>
 160/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 467ms/step - loss: 250.8510

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 470ms/step - loss: 251.1944

<div class="k-default-codeblock">
```

```
</div>
 162/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 471ms/step - loss: 251.5295

<div class="k-default-codeblock">
```

```
</div>
 163/250 ━━━━━━━━━━━━━[37m━━━━━━━  41s 474ms/step - loss: 251.8566

<div class="k-default-codeblock">
```

```
</div>
 164/250 ━━━━━━━━━━━━━[37m━━━━━━━  40s 475ms/step - loss: 252.1759

<div class="k-default-codeblock">
```

```
</div>
 165/250 ━━━━━━━━━━━━━[37m━━━━━━━  40s 475ms/step - loss: 252.4875

<div class="k-default-codeblock">
```

```
</div>
 166/250 ━━━━━━━━━━━━━[37m━━━━━━━  40s 476ms/step - loss: 252.7914

<div class="k-default-codeblock">
```

```
</div>
 167/250 ━━━━━━━━━━━━━[37m━━━━━━━  39s 477ms/step - loss: 253.0878

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  39s 478ms/step - loss: 253.3768

<div class="k-default-codeblock">
```

```
</div>
 169/250 ━━━━━━━━━━━━━[37m━━━━━━━  38s 479ms/step - loss: 253.6587

<div class="k-default-codeblock">
```

```
</div>
 170/250 ━━━━━━━━━━━━━[37m━━━━━━━  38s 479ms/step - loss: 253.9335

<div class="k-default-codeblock">
```

```
</div>
 171/250 ━━━━━━━━━━━━━[37m━━━━━━━  37s 480ms/step - loss: 254.2010

<div class="k-default-codeblock">
```

```
</div>
 172/250 ━━━━━━━━━━━━━[37m━━━━━━━  37s 481ms/step - loss: 254.4613

<div class="k-default-codeblock">
```

```
</div>
 173/250 ━━━━━━━━━━━━━[37m━━━━━━━  37s 482ms/step - loss: 254.7150

<div class="k-default-codeblock">
```

```
</div>
 174/250 ━━━━━━━━━━━━━[37m━━━━━━━  36s 482ms/step - loss: 254.9618

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  36s 483ms/step - loss: 255.2018

<div class="k-default-codeblock">
```

```
</div>
 176/250 ━━━━━━━━━━━━━━[37m━━━━━━  35s 483ms/step - loss: 255.4353

<div class="k-default-codeblock">
```

```
</div>
 177/250 ━━━━━━━━━━━━━━[37m━━━━━━  35s 484ms/step - loss: 255.6625

<div class="k-default-codeblock">
```

```
</div>
 178/250 ━━━━━━━━━━━━━━[37m━━━━━━  34s 486ms/step - loss: 255.8835

<div class="k-default-codeblock">
```

```
</div>
 179/250 ━━━━━━━━━━━━━━[37m━━━━━━  34s 487ms/step - loss: 256.0988

<div class="k-default-codeblock">
```

```
</div>
 180/250 ━━━━━━━━━━━━━━[37m━━━━━━  34s 488ms/step - loss: 256.3084

<div class="k-default-codeblock">
```

```
</div>
 181/250 ━━━━━━━━━━━━━━[37m━━━━━━  33s 488ms/step - loss: 256.5120

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  33s 489ms/step - loss: 256.7101

<div class="k-default-codeblock">
```

```
</div>
 183/250 ━━━━━━━━━━━━━━[37m━━━━━━  32s 490ms/step - loss: 256.9029

<div class="k-default-codeblock">
```

```
</div>
 184/250 ━━━━━━━━━━━━━━[37m━━━━━━  32s 490ms/step - loss: 257.0904

<div class="k-default-codeblock">
```

```
</div>
 185/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 490ms/step - loss: 257.2723

<div class="k-default-codeblock">
```

```
</div>
 186/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 491ms/step - loss: 257.4491

<div class="k-default-codeblock">
```

```
</div>
 187/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 491ms/step - loss: 257.6210

<div class="k-default-codeblock">
```

```
</div>
 188/250 ━━━━━━━━━━━━━━━[37m━━━━━  30s 492ms/step - loss: 257.7879

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  30s 492ms/step - loss: 257.9500

<div class="k-default-codeblock">
```

```
</div>
 190/250 ━━━━━━━━━━━━━━━[37m━━━━━  29s 492ms/step - loss: 258.1073

<div class="k-default-codeblock">
```

```
</div>
 191/250 ━━━━━━━━━━━━━━━[37m━━━━━  29s 492ms/step - loss: 258.2599

<div class="k-default-codeblock">
```

```
</div>
 192/250 ━━━━━━━━━━━━━━━[37m━━━━━  28s 493ms/step - loss: 258.4078

<div class="k-default-codeblock">
```

```
</div>
 193/250 ━━━━━━━━━━━━━━━[37m━━━━━  28s 493ms/step - loss: 258.5514

<div class="k-default-codeblock">
```

```
</div>
 194/250 ━━━━━━━━━━━━━━━[37m━━━━━  27s 494ms/step - loss: 258.6907

<div class="k-default-codeblock">
```

```
</div>
 195/250 ━━━━━━━━━━━━━━━[37m━━━━━  27s 494ms/step - loss: 258.8260

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  26s 494ms/step - loss: 258.9570

<div class="k-default-codeblock">
```

```
</div>
 197/250 ━━━━━━━━━━━━━━━[37m━━━━━  26s 495ms/step - loss: 259.0840

<div class="k-default-codeblock">
```

```
</div>
 198/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 495ms/step - loss: 259.2071

<div class="k-default-codeblock">
```

```
</div>
 199/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 496ms/step - loss: 259.3262

<div class="k-default-codeblock">
```

```
</div>
 200/250 ━━━━━━━━━━━━━━━━[37m━━━━  24s 496ms/step - loss: 259.4411

<div class="k-default-codeblock">
```

```
</div>
 201/250 ━━━━━━━━━━━━━━━━[37m━━━━  24s 497ms/step - loss: 259.5522

<div class="k-default-codeblock">
```

```
</div>
 202/250 ━━━━━━━━━━━━━━━━[37m━━━━  23s 497ms/step - loss: 259.6594

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  23s 497ms/step - loss: 259.7626

<div class="k-default-codeblock">
```

```
</div>
 204/250 ━━━━━━━━━━━━━━━━[37m━━━━  22s 498ms/step - loss: 259.8620

<div class="k-default-codeblock">
```

```
</div>
 205/250 ━━━━━━━━━━━━━━━━[37m━━━━  22s 498ms/step - loss: 259.9583

<div class="k-default-codeblock">
```

```
</div>
 206/250 ━━━━━━━━━━━━━━━━[37m━━━━  21s 499ms/step - loss: 260.0512

<div class="k-default-codeblock">
```

```
</div>
 207/250 ━━━━━━━━━━━━━━━━[37m━━━━  21s 499ms/step - loss: 260.1408

<div class="k-default-codeblock">
```

```
</div>
 208/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 499ms/step - loss: 260.2269

<div class="k-default-codeblock">
```

```
</div>
 209/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 500ms/step - loss: 260.3098

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 500ms/step - loss: 260.3897

<div class="k-default-codeblock">
```

```
</div>
 211/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 500ms/step - loss: 260.4664

<div class="k-default-codeblock">
```

```
</div>
 212/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 501ms/step - loss: 260.5400

<div class="k-default-codeblock">
```

```
</div>
 213/250 ━━━━━━━━━━━━━━━━━[37m━━━  18s 501ms/step - loss: 260.6106

<div class="k-default-codeblock">
```

```
</div>
 214/250 ━━━━━━━━━━━━━━━━━[37m━━━  18s 501ms/step - loss: 260.6784

<div class="k-default-codeblock">
```

```
</div>
 215/250 ━━━━━━━━━━━━━━━━━[37m━━━  17s 501ms/step - loss: 260.7433

<div class="k-default-codeblock">
```

```
</div>
 216/250 ━━━━━━━━━━━━━━━━━[37m━━━  17s 502ms/step - loss: 260.8054

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  16s 502ms/step - loss: 260.8649

<div class="k-default-codeblock">
```

```
</div>
 218/250 ━━━━━━━━━━━━━━━━━[37m━━━  16s 502ms/step - loss: 260.9216

<div class="k-default-codeblock">
```

```
</div>
 219/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 503ms/step - loss: 260.9846

<div class="k-default-codeblock">
```

```
</div>
 220/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 503ms/step - loss: 261.0452

<div class="k-default-codeblock">
```

```
</div>
 221/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 503ms/step - loss: 261.1040

<div class="k-default-codeblock">
```

```
</div>
 222/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 503ms/step - loss: 261.1618

<div class="k-default-codeblock">
```

```
</div>
 223/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 504ms/step - loss: 261.2195

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 504ms/step - loss: 261.2781

<div class="k-default-codeblock">
```

```
</div>
 225/250 ━━━━━━━━━━━━━━━━━━[37m━━  12s 504ms/step - loss: 261.3386

<div class="k-default-codeblock">
```

```
</div>
 226/250 ━━━━━━━━━━━━━━━━━━[37m━━  12s 504ms/step - loss: 261.4021

<div class="k-default-codeblock">
```

```
</div>
 227/250 ━━━━━━━━━━━━━━━━━━[37m━━  11s 505ms/step - loss: 261.4691

<div class="k-default-codeblock">
```

```
</div>
 228/250 ━━━━━━━━━━━━━━━━━━[37m━━  11s 505ms/step - loss: 261.5407

<div class="k-default-codeblock">
```

```
</div>
 229/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 505ms/step - loss: 261.6174

<div class="k-default-codeblock">
```

```
</div>
 230/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 505ms/step - loss: 261.6999

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 505ms/step - loss: 261.7886 

<div class="k-default-codeblock">
```

```
</div>
 232/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 506ms/step - loss: 261.8840

<div class="k-default-codeblock">
```

```
</div>
 233/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 506ms/step - loss: 261.9863

<div class="k-default-codeblock">
```

```
</div>
 234/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 506ms/step - loss: 262.0963

<div class="k-default-codeblock">
```

```
</div>
 235/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 506ms/step - loss: 262.2137

<div class="k-default-codeblock">
```

```
</div>
 236/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 507ms/step - loss: 262.3383

<div class="k-default-codeblock">
```

```
</div>
 237/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 507ms/step - loss: 262.4702

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  6s 507ms/step - loss: 262.6093

<div class="k-default-codeblock">
```

```
</div>
 239/250 ━━━━━━━━━━━━━━━━━━━[37m━  5s 508ms/step - loss: 262.7552

<div class="k-default-codeblock">
```

```
</div>
 240/250 ━━━━━━━━━━━━━━━━━━━[37m━  5s 508ms/step - loss: 262.9080

<div class="k-default-codeblock">
```

```
</div>
 241/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 508ms/step - loss: 263.0675

<div class="k-default-codeblock">
```

```
</div>
 242/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 508ms/step - loss: 263.2331

<div class="k-default-codeblock">
```

```
</div>
 243/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 507ms/step - loss: 263.4047

<div class="k-default-codeblock">
```

```
</div>
 244/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 507ms/step - loss: 263.5821

<div class="k-default-codeblock">
```

```
</div>
 245/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 506ms/step - loss: 263.7649

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 506ms/step - loss: 263.9527

<div class="k-default-codeblock">
```

```
</div>
 247/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 505ms/step - loss: 264.1455

<div class="k-default-codeblock">
```

```
</div>
 248/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 505ms/step - loss: 264.3427

<div class="k-default-codeblock">
```

```
</div>
 249/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 505ms/step - loss: 264.5440

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 504ms/step - loss: 264.7491

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 126s 504ms/step - loss: 264.9526


<div class="k-default-codeblock">
```
Epoch 9/10

```
</div>
    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  1:53 456ms/step - loss: 646.4985

<div class="k-default-codeblock">
```

```
</div>
   2/250 [37m━━━━━━━━━━━━━━━━━━━━  1:46 431ms/step - loss: 642.0029

<div class="k-default-codeblock">
```

```
</div>
   3/250 [37m━━━━━━━━━━━━━━━━━━━━  1:45 429ms/step - loss: 638.9329

<div class="k-default-codeblock">
```

```
</div>
   4/250 [37m━━━━━━━━━━━━━━━━━━━━  1:42 416ms/step - loss: 635.5615

<div class="k-default-codeblock">
```

```
</div>
   5/250 [37m━━━━━━━━━━━━━━━━━━━━  1:42 417ms/step - loss: 632.1136

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 418ms/step - loss: 628.6561

<div class="k-default-codeblock">
```

```
</div>
   7/250 [37m━━━━━━━━━━━━━━━━━━━━  1:42 422ms/step - loss: 624.9097

<div class="k-default-codeblock">
```

```
</div>
   8/250 [37m━━━━━━━━━━━━━━━━━━━━  1:42 422ms/step - loss: 621.3226

<div class="k-default-codeblock">
```

```
</div>
   9/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 421ms/step - loss: 617.7669

<div class="k-default-codeblock">
```

```
</div>
  10/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 420ms/step - loss: 614.1563

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 419ms/step - loss: 610.5945

<div class="k-default-codeblock">
```

```
</div>
  12/250 [37m━━━━━━━━━━━━━━━━━━━━  1:39 416ms/step - loss: 607.0633

<div class="k-default-codeblock">
```

```
</div>
  13/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:38 415ms/step - loss: 603.5840

<div class="k-default-codeblock">
```

```
</div>
  14/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 415ms/step - loss: 600.1556

<div class="k-default-codeblock">
```

```
</div>
  15/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 416ms/step - loss: 596.7750

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 416ms/step - loss: 593.3998

<div class="k-default-codeblock">
```

```
</div>
  17/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 415ms/step - loss: 590.0424

<div class="k-default-codeblock">
```

```
</div>
  18/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 415ms/step - loss: 586.7529

<div class="k-default-codeblock">
```

```
</div>
  19/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 415ms/step - loss: 583.5004

<div class="k-default-codeblock">
```

```
</div>
  20/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 415ms/step - loss: 580.3078

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 413ms/step - loss: 577.1579

<div class="k-default-codeblock">
```

```
</div>
  22/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:33 412ms/step - loss: 574.0593

<div class="k-default-codeblock">
```

```
</div>
  23/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:33 412ms/step - loss: 571.0004

<div class="k-default-codeblock">
```

```
</div>
  24/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:33 412ms/step - loss: 567.9940

<div class="k-default-codeblock">
```

```
</div>
  25/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 412ms/step - loss: 565.0201

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 412ms/step - loss: 562.0765

<div class="k-default-codeblock">
```

```
</div>
  27/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 413ms/step - loss: 559.1669

<div class="k-default-codeblock">
```

```
</div>
  28/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 413ms/step - loss: 556.2775

<div class="k-default-codeblock">
```

```
</div>
  29/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 412ms/step - loss: 553.4158

<div class="k-default-codeblock">
```

```
</div>
  30/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 412ms/step - loss: 550.5992

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 412ms/step - loss: 547.8106

<div class="k-default-codeblock">
```

```
</div>
  32/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 412ms/step - loss: 545.0480

<div class="k-default-codeblock">
```

```
</div>
  33/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 412ms/step - loss: 542.3104

<div class="k-default-codeblock">
```

```
</div>
  34/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 412ms/step - loss: 539.5994

<div class="k-default-codeblock">
```

```
</div>
  35/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 413ms/step - loss: 536.9167

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 413ms/step - loss: 534.2676

<div class="k-default-codeblock">
```

```
</div>
  37/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 413ms/step - loss: 531.6412

<div class="k-default-codeblock">
```

```
</div>
  38/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 414ms/step - loss: 529.0388

<div class="k-default-codeblock">
```

```
</div>
  39/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 415ms/step - loss: 526.4630

<div class="k-default-codeblock">
```

```
</div>
  40/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 416ms/step - loss: 523.9141

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 416ms/step - loss: 521.3948

<div class="k-default-codeblock">
```

```
</div>
  42/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 416ms/step - loss: 518.9051

<div class="k-default-codeblock">
```

```
</div>
  43/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 415ms/step - loss: 516.4391

<div class="k-default-codeblock">
```

```
</div>
  44/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 415ms/step - loss: 513.9993

<div class="k-default-codeblock">
```

```
</div>
  45/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 415ms/step - loss: 511.5807

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 415ms/step - loss: 509.1854

<div class="k-default-codeblock">
```

```
</div>
  47/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 416ms/step - loss: 506.8095

<div class="k-default-codeblock">
```

```
</div>
  48/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:23 415ms/step - loss: 504.4574

<div class="k-default-codeblock">
```

```
</div>
  49/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:23 415ms/step - loss: 502.1298

<div class="k-default-codeblock">
```

```
</div>
  50/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 415ms/step - loss: 499.8315

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 415ms/step - loss: 497.5585

<div class="k-default-codeblock">
```

```
</div>
  52/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 415ms/step - loss: 495.3099

<div class="k-default-codeblock">
```

```
</div>
  53/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 415ms/step - loss: 493.0859

<div class="k-default-codeblock">
```

```
</div>
  54/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 415ms/step - loss: 490.8855

<div class="k-default-codeblock">
```

```
</div>
  55/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 414ms/step - loss: 488.7129

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 415ms/step - loss: 486.5654

<div class="k-default-codeblock">
```

```
</div>
  57/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 415ms/step - loss: 484.4443

<div class="k-default-codeblock">
```

```
</div>
  58/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 415ms/step - loss: 482.3470

<div class="k-default-codeblock">
```

```
</div>
  59/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 415ms/step - loss: 480.2719

<div class="k-default-codeblock">
```

```
</div>
  60/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:18 414ms/step - loss: 478.2212

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:18 414ms/step - loss: 476.1954

<div class="k-default-codeblock">
```

```
</div>
  62/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:17 414ms/step - loss: 474.1927

<div class="k-default-codeblock">
```

```
</div>
  63/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:17 415ms/step - loss: 472.2122

<div class="k-default-codeblock">
```

```
</div>
  64/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:17 415ms/step - loss: 470.2585

<div class="k-default-codeblock">
```

```
</div>
  65/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 415ms/step - loss: 468.3266

<div class="k-default-codeblock">
```

```
</div>
  66/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 415ms/step - loss: 466.4192

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 414ms/step - loss: 464.5316

<div class="k-default-codeblock">
```

```
</div>
  68/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 415ms/step - loss: 462.6649

<div class="k-default-codeblock">
```

```
</div>
  69/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 415ms/step - loss: 460.8205

<div class="k-default-codeblock">
```

```
</div>
  70/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 414ms/step - loss: 458.9972

<div class="k-default-codeblock">
```

```
</div>
  71/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 415ms/step - loss: 457.1936

<div class="k-default-codeblock">
```

```
</div>
  72/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 415ms/step - loss: 455.4173

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 414ms/step - loss: 453.6611

<div class="k-default-codeblock">
```

```
</div>
  74/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:12 414ms/step - loss: 451.9244

<div class="k-default-codeblock">
```

```
</div>
  75/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 414ms/step - loss: 450.2066

<div class="k-default-codeblock">
```

```
</div>
  76/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 414ms/step - loss: 448.5110

<div class="k-default-codeblock">
```

```
</div>
  77/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 414ms/step - loss: 446.8387

<div class="k-default-codeblock">
```

```
</div>
  78/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 414ms/step - loss: 445.1873

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 415ms/step - loss: 443.5575

<div class="k-default-codeblock">
```

```
</div>
  80/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 414ms/step - loss: 441.9479

<div class="k-default-codeblock">
```

```
</div>
  81/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 414ms/step - loss: 440.3604

<div class="k-default-codeblock">
```

```
</div>
  82/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 414ms/step - loss: 438.7961

<div class="k-default-codeblock">
```

```
</div>
  83/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 414ms/step - loss: 437.2520

<div class="k-default-codeblock">
```

```
</div>
  84/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 414ms/step - loss: 435.7281

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 414ms/step - loss: 434.2253

<div class="k-default-codeblock">
```

```
</div>
  86/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:07 414ms/step - loss: 432.7428

<div class="k-default-codeblock">
```

```
</div>
  87/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:07 414ms/step - loss: 431.2796

<div class="k-default-codeblock">
```

```
</div>
  88/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:07 414ms/step - loss: 429.8348

<div class="k-default-codeblock">
```

```
</div>
  89/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 414ms/step - loss: 428.4097

<div class="k-default-codeblock">
```

```
</div>
  90/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 414ms/step - loss: 427.0048

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 414ms/step - loss: 425.6187

<div class="k-default-codeblock">
```

```
</div>
  92/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 413ms/step - loss: 424.2509

<div class="k-default-codeblock">
```

```
</div>
  93/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 414ms/step - loss: 422.9009

<div class="k-default-codeblock">
```

```
</div>
  94/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 414ms/step - loss: 421.5679

<div class="k-default-codeblock">
```

```
</div>
  95/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 414ms/step - loss: 420.2514

<div class="k-default-codeblock">
```

```
</div>
  96/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 414ms/step - loss: 418.9509

<div class="k-default-codeblock">
```

```
</div>
  97/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 414ms/step - loss: 417.6669

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:02 414ms/step - loss: 416.3988

<div class="k-default-codeblock">
```

```
</div>
  99/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:02 415ms/step - loss: 415.1455

<div class="k-default-codeblock">
```

```
</div>
 100/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:02 414ms/step - loss: 413.9077

<div class="k-default-codeblock">
```

```
</div>
 101/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 414ms/step - loss: 412.6855

<div class="k-default-codeblock">
```

```
</div>
 102/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 414ms/step - loss: 411.4779

<div class="k-default-codeblock">
```

```
</div>
 103/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 414ms/step - loss: 410.2847

<div class="k-default-codeblock">
```

```
</div>
 104/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 415ms/step - loss: 409.1050

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 414ms/step - loss: 407.9411

<div class="k-default-codeblock">
```

```
</div>
 106/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 414ms/step - loss: 406.7898 

<div class="k-default-codeblock">
```

```
</div>
 107/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 414ms/step - loss: 405.6519

<div class="k-default-codeblock">
```

```
</div>
 108/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 414ms/step - loss: 404.5273

<div class="k-default-codeblock">
```

```
</div>
 109/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 414ms/step - loss: 403.4159

<div class="k-default-codeblock">
```

```
</div>
 110/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 414ms/step - loss: 402.3159

<div class="k-default-codeblock">
```

```
</div>
 111/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 414ms/step - loss: 401.2286

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 414ms/step - loss: 400.1529

<div class="k-default-codeblock">
```

```
</div>
 113/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 414ms/step - loss: 399.0889

<div class="k-default-codeblock">
```

```
</div>
 114/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 414ms/step - loss: 398.0365

<div class="k-default-codeblock">
```

```
</div>
 115/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 414ms/step - loss: 396.9951

<div class="k-default-codeblock">
```

```
</div>
 116/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 414ms/step - loss: 395.9649

<div class="k-default-codeblock">
```

```
</div>
 117/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 415ms/step - loss: 394.9451

<div class="k-default-codeblock">
```

```
</div>
 118/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 415ms/step - loss: 393.9358

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 415ms/step - loss: 392.9367

<div class="k-default-codeblock">
```

```
</div>
 120/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 415ms/step - loss: 391.9474

<div class="k-default-codeblock">
```

```
</div>
 121/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 416ms/step - loss: 390.9677

<div class="k-default-codeblock">
```

```
</div>
 122/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 417ms/step - loss: 389.9976

<div class="k-default-codeblock">
```

```
</div>
 123/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 417ms/step - loss: 389.0376

<div class="k-default-codeblock">
```

```
</div>
 124/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 417ms/step - loss: 388.0872

<div class="k-default-codeblock">
```

```
</div>
 125/250 ━━━━━━━━━━[37m━━━━━━━━━━  52s 417ms/step - loss: 387.1459

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 417ms/step - loss: 386.2149

<div class="k-default-codeblock">
```

```
</div>
 127/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 417ms/step - loss: 385.2925

<div class="k-default-codeblock">
```

```
</div>
 128/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 417ms/step - loss: 384.3787

<div class="k-default-codeblock">
```

```
</div>
 129/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 417ms/step - loss: 383.4734

<div class="k-default-codeblock">
```

```
</div>
 130/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 417ms/step - loss: 382.5767

<div class="k-default-codeblock">
```

```
</div>
 131/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 417ms/step - loss: 381.6892

<div class="k-default-codeblock">
```

```
</div>
 132/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 417ms/step - loss: 380.8104

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 417ms/step - loss: 379.9398

<div class="k-default-codeblock">
```

```
</div>
 134/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 418ms/step - loss: 379.0777

<div class="k-default-codeblock">
```

```
</div>
 135/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 418ms/step - loss: 378.2242

<div class="k-default-codeblock">
```

```
</div>
 136/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 418ms/step - loss: 377.3794

<div class="k-default-codeblock">
```

```
</div>
 137/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 418ms/step - loss: 376.5418

<div class="k-default-codeblock">
```

```
</div>
 138/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 418ms/step - loss: 375.7126

<div class="k-default-codeblock">
```

```
</div>
 139/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 418ms/step - loss: 374.8906

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 418ms/step - loss: 374.0759

<div class="k-default-codeblock">
```

```
</div>
 141/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 418ms/step - loss: 373.2688

<div class="k-default-codeblock">
```

```
</div>
 142/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 418ms/step - loss: 372.4691

<div class="k-default-codeblock">
```

```
</div>
 143/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 418ms/step - loss: 371.6768

<div class="k-default-codeblock">
```

```
</div>
 144/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 418ms/step - loss: 370.8911

<div class="k-default-codeblock">
```

```
</div>
 145/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 418ms/step - loss: 370.1124

<div class="k-default-codeblock">
```

```
</div>
 146/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 418ms/step - loss: 369.3405

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 418ms/step - loss: 368.5755

<div class="k-default-codeblock">
```

```
</div>
 148/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 418ms/step - loss: 367.8175

<div class="k-default-codeblock">
```

```
</div>
 149/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 418ms/step - loss: 367.0659

<div class="k-default-codeblock">
```

```
</div>
 150/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 418ms/step - loss: 366.3209

<div class="k-default-codeblock">
```

```
</div>
 151/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 418ms/step - loss: 365.5820

<div class="k-default-codeblock">
```

```
</div>
 152/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 418ms/step - loss: 364.8508

<div class="k-default-codeblock">
```

```
</div>
 153/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 418ms/step - loss: 364.1254

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 418ms/step - loss: 363.4059

<div class="k-default-codeblock">
```

```
</div>
 155/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 418ms/step - loss: 362.6928

<div class="k-default-codeblock">
```

```
</div>
 156/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 418ms/step - loss: 361.9863

<div class="k-default-codeblock">
```

```
</div>
 157/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 418ms/step - loss: 361.2857

<div class="k-default-codeblock">
```

```
</div>
 158/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 418ms/step - loss: 360.5913

<div class="k-default-codeblock">
```

```
</div>
 159/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 418ms/step - loss: 359.9026

<div class="k-default-codeblock">
```

```
</div>
 160/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 418ms/step - loss: 359.2199

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 418ms/step - loss: 358.5428

<div class="k-default-codeblock">
```

```
</div>
 162/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 418ms/step - loss: 357.8713

<div class="k-default-codeblock">
```

```
</div>
 163/250 ━━━━━━━━━━━━━[37m━━━━━━━  36s 418ms/step - loss: 357.2055

<div class="k-default-codeblock">
```

```
</div>
 164/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 418ms/step - loss: 356.5456

<div class="k-default-codeblock">
```

```
</div>
 165/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 418ms/step - loss: 355.8911

<div class="k-default-codeblock">
```

```
</div>
 166/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 418ms/step - loss: 355.2422

<div class="k-default-codeblock">
```

```
</div>
 167/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 417ms/step - loss: 354.5988

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 419ms/step - loss: 353.9608

<div class="k-default-codeblock">
```

```
</div>
 169/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 419ms/step - loss: 353.3281

<div class="k-default-codeblock">
```

```
</div>
 170/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 419ms/step - loss: 352.7009

<div class="k-default-codeblock">
```

```
</div>
 171/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 419ms/step - loss: 352.0796

<div class="k-default-codeblock">
```

```
</div>
 172/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 419ms/step - loss: 351.4639

<div class="k-default-codeblock">
```

```
</div>
 173/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 419ms/step - loss: 350.8537

<div class="k-default-codeblock">
```

```
</div>
 174/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 420ms/step - loss: 350.2487

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 420ms/step - loss: 349.6489

<div class="k-default-codeblock">
```

```
</div>
 176/250 ━━━━━━━━━━━━━━[37m━━━━━━  31s 420ms/step - loss: 349.0540

<div class="k-default-codeblock">
```

```
</div>
 177/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 420ms/step - loss: 348.4641

<div class="k-default-codeblock">
```

```
</div>
 178/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 419ms/step - loss: 347.8792

<div class="k-default-codeblock">
```

```
</div>
 179/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 419ms/step - loss: 347.2990

<div class="k-default-codeblock">
```

```
</div>
 180/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 420ms/step - loss: 346.7231

<div class="k-default-codeblock">
```

```
</div>
 181/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 419ms/step - loss: 346.1514

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 419ms/step - loss: 345.5844

<div class="k-default-codeblock">
```

```
</div>
 183/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 419ms/step - loss: 345.0217

<div class="k-default-codeblock">
```

```
</div>
 184/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 419ms/step - loss: 344.4632

<div class="k-default-codeblock">
```

```
</div>
 185/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 419ms/step - loss: 343.9087

<div class="k-default-codeblock">
```

```
</div>
 186/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 419ms/step - loss: 343.3586

<div class="k-default-codeblock">
```

```
</div>
 187/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 419ms/step - loss: 342.8125

<div class="k-default-codeblock">
```

```
</div>
 188/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 419ms/step - loss: 342.2704

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 419ms/step - loss: 341.7319

<div class="k-default-codeblock">
```

```
</div>
 190/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 419ms/step - loss: 341.1971

<div class="k-default-codeblock">
```

```
</div>
 191/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 419ms/step - loss: 340.6661

<div class="k-default-codeblock">
```

```
</div>
 192/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 419ms/step - loss: 340.1386

<div class="k-default-codeblock">
```

```
</div>
 193/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 419ms/step - loss: 339.6147

<div class="k-default-codeblock">
```

```
</div>
 194/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 419ms/step - loss: 339.0948

<div class="k-default-codeblock">
```

```
</div>
 195/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 419ms/step - loss: 338.5784

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 419ms/step - loss: 338.0656

<div class="k-default-codeblock">
```

```
</div>
 197/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 419ms/step - loss: 337.5563

<div class="k-default-codeblock">
```

```
</div>
 198/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 419ms/step - loss: 337.0506

<div class="k-default-codeblock">
```

```
</div>
 199/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 419ms/step - loss: 336.5486

<div class="k-default-codeblock">
```

```
</div>
 200/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 419ms/step - loss: 336.0503

<div class="k-default-codeblock">
```

```
</div>
 201/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 419ms/step - loss: 335.5555

<div class="k-default-codeblock">
```

```
</div>
 202/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 419ms/step - loss: 335.0640

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 419ms/step - loss: 334.5754

<div class="k-default-codeblock">
```

```
</div>
 204/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 419ms/step - loss: 334.0901

<div class="k-default-codeblock">
```

```
</div>
 205/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 419ms/step - loss: 333.6084

<div class="k-default-codeblock">
```

```
</div>
 206/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 419ms/step - loss: 333.1297

<div class="k-default-codeblock">
```

```
</div>
 207/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 419ms/step - loss: 332.6539

<div class="k-default-codeblock">
```

```
</div>
 208/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 419ms/step - loss: 332.1812

<div class="k-default-codeblock">
```

```
</div>
 209/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 419ms/step - loss: 331.7129

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 419ms/step - loss: 331.2482

<div class="k-default-codeblock">
```

```
</div>
 211/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 419ms/step - loss: 330.7868

<div class="k-default-codeblock">
```

```
</div>
 212/250 ━━━━━━━━━━━━━━━━[37m━━━━  15s 419ms/step - loss: 330.3284

<div class="k-default-codeblock">
```

```
</div>
 213/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 419ms/step - loss: 329.8735

<div class="k-default-codeblock">
```

```
</div>
 214/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 419ms/step - loss: 329.4221

<div class="k-default-codeblock">
```

```
</div>
 215/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 419ms/step - loss: 328.9743

<div class="k-default-codeblock">
```

```
</div>
 216/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 419ms/step - loss: 328.5297

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 419ms/step - loss: 328.0885

<div class="k-default-codeblock">
```

```
</div>
 218/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 419ms/step - loss: 327.6510

<div class="k-default-codeblock">
```

```
</div>
 219/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 419ms/step - loss: 327.2173

<div class="k-default-codeblock">
```

```
</div>
 220/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 419ms/step - loss: 326.7874

<div class="k-default-codeblock">
```

```
</div>
 221/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 419ms/step - loss: 326.3609

<div class="k-default-codeblock">
```

```
</div>
 222/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 419ms/step - loss: 325.9380

<div class="k-default-codeblock">
```

```
</div>
 223/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 418ms/step - loss: 325.5188

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  10s 418ms/step - loss: 325.1032

<div class="k-default-codeblock">
```

```
</div>
 225/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 418ms/step - loss: 324.6912

<div class="k-default-codeblock">
```

```
</div>
 226/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 418ms/step - loss: 324.2823

<div class="k-default-codeblock">
```

```
</div>
 227/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 418ms/step - loss: 323.8769 

<div class="k-default-codeblock">
```

```
</div>
 228/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 418ms/step - loss: 323.4752

<div class="k-default-codeblock">
```

```
</div>
 229/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 418ms/step - loss: 323.0768

<div class="k-default-codeblock">
```

```
</div>
 230/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 418ms/step - loss: 322.6819

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 418ms/step - loss: 322.2902

<div class="k-default-codeblock">
```

```
</div>
 232/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 418ms/step - loss: 321.9017

<div class="k-default-codeblock">
```

```
</div>
 233/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 418ms/step - loss: 321.5165

<div class="k-default-codeblock">
```

```
</div>
 234/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 418ms/step - loss: 321.1344

<div class="k-default-codeblock">
```

```
</div>
 235/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 418ms/step - loss: 320.7556

<div class="k-default-codeblock">
```

```
</div>
 236/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 418ms/step - loss: 320.3799

<div class="k-default-codeblock">
```

```
</div>
 237/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 418ms/step - loss: 320.0069

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  5s 418ms/step - loss: 319.6371

<div class="k-default-codeblock">
```

```
</div>
 239/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 418ms/step - loss: 319.2701

<div class="k-default-codeblock">
```

```
</div>
 240/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 418ms/step - loss: 318.9056

<div class="k-default-codeblock">
```

```
</div>
 241/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 418ms/step - loss: 318.5438

<div class="k-default-codeblock">
```

```
</div>
 242/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 418ms/step - loss: 318.1846

<div class="k-default-codeblock">
```

```
</div>
 243/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 418ms/step - loss: 317.8280

<div class="k-default-codeblock">
```

```
</div>
 244/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 418ms/step - loss: 317.4738

<div class="k-default-codeblock">
```

```
</div>
 245/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 418ms/step - loss: 317.1221

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 418ms/step - loss: 316.7728

<div class="k-default-codeblock">
```

```
</div>
 247/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 418ms/step - loss: 316.4260

<div class="k-default-codeblock">
```

```
</div>
 248/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 418ms/step - loss: 316.0815

<div class="k-default-codeblock">
```

```
</div>
 249/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 418ms/step - loss: 315.7393

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 418ms/step - loss: 315.3996

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 104s 418ms/step - loss: 315.0625


<div class="k-default-codeblock">
```
Epoch 10/10

```
</div>
    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  1:50 445ms/step - loss: 211.7476

<div class="k-default-codeblock">
```

```
</div>
   2/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 409ms/step - loss: 208.9853

<div class="k-default-codeblock">
```

```
</div>
   3/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 411ms/step - loss: 207.8150

<div class="k-default-codeblock">
```

```
</div>
   4/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 412ms/step - loss: 207.2226

<div class="k-default-codeblock">
```

```
</div>
   5/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 415ms/step - loss: 206.7438

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  1:41 415ms/step - loss: 206.2510

<div class="k-default-codeblock">
```

```
</div>
   7/250 [37m━━━━━━━━━━━━━━━━━━━━  1:40 412ms/step - loss: 205.8702

<div class="k-default-codeblock">
```

```
</div>
   8/250 [37m━━━━━━━━━━━━━━━━━━━━  1:39 411ms/step - loss: 205.6369

<div class="k-default-codeblock">
```

```
</div>
   9/250 [37m━━━━━━━━━━━━━━━━━━━━  1:38 411ms/step - loss: 205.3155

<div class="k-default-codeblock">
```

```
</div>
  10/250 [37m━━━━━━━━━━━━━━━━━━━━  1:38 409ms/step - loss: 204.9164

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  1:37 410ms/step - loss: 204.5660

<div class="k-default-codeblock">
```

```
</div>
  12/250 [37m━━━━━━━━━━━━━━━━━━━━  1:37 412ms/step - loss: 204.2301

<div class="k-default-codeblock">
```

```
</div>
  13/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 411ms/step - loss: 203.9077

<div class="k-default-codeblock">
```

```
</div>
  14/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:37 413ms/step - loss: 203.5736

<div class="k-default-codeblock">
```

```
</div>
  15/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 412ms/step - loss: 203.2336

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 413ms/step - loss: 202.9046

<div class="k-default-codeblock">
```

```
</div>
  17/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:36 412ms/step - loss: 202.5730

<div class="k-default-codeblock">
```

```
</div>
  18/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 413ms/step - loss: 202.2344

<div class="k-default-codeblock">
```

```
</div>
  19/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:35 412ms/step - loss: 201.9463

<div class="k-default-codeblock">
```

```
</div>
  20/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 413ms/step - loss: 201.6942

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:34 412ms/step - loss: 201.4394

<div class="k-default-codeblock">
```

```
</div>
  22/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:33 412ms/step - loss: 201.1689

<div class="k-default-codeblock">
```

```
</div>
  23/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:33 412ms/step - loss: 200.8983

<div class="k-default-codeblock">
```

```
</div>
  24/250 ━[37m━━━━━━━━━━━━━━━━━━━  1:33 414ms/step - loss: 200.6035

<div class="k-default-codeblock">
```

```
</div>
  25/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 413ms/step - loss: 200.3140

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 413ms/step - loss: 200.0246

<div class="k-default-codeblock">
```

```
</div>
  27/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:32 413ms/step - loss: 199.7557

<div class="k-default-codeblock">
```

```
</div>
  28/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 413ms/step - loss: 199.5043

<div class="k-default-codeblock">
```

```
</div>
  29/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 415ms/step - loss: 199.2651

<div class="k-default-codeblock">
```

```
</div>
  30/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:31 414ms/step - loss: 199.0303

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 414ms/step - loss: 198.8001

<div class="k-default-codeblock">
```

```
</div>
  32/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:30 413ms/step - loss: 198.5739

<div class="k-default-codeblock">
```

```
</div>
  33/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 413ms/step - loss: 198.3474

<div class="k-default-codeblock">
```

```
</div>
  34/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:29 412ms/step - loss: 198.1231

<div class="k-default-codeblock">
```

```
</div>
  35/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 412ms/step - loss: 197.9032

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:28 412ms/step - loss: 197.6875

<div class="k-default-codeblock">
```

```
</div>
  37/250 ━━[37m━━━━━━━━━━━━━━━━━━  1:27 413ms/step - loss: 197.4724

<div class="k-default-codeblock">
```

```
</div>
  38/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 413ms/step - loss: 197.2578

<div class="k-default-codeblock">
```

```
</div>
  39/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:27 412ms/step - loss: 197.0414

<div class="k-default-codeblock">
```

```
</div>
  40/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 413ms/step - loss: 196.8281

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:26 413ms/step - loss: 196.6165

<div class="k-default-codeblock">
```

```
</div>
  42/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 413ms/step - loss: 196.4067

<div class="k-default-codeblock">
```

```
</div>
  43/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 413ms/step - loss: 196.2001

<div class="k-default-codeblock">
```

```
</div>
  44/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:25 413ms/step - loss: 195.9948

<div class="k-default-codeblock">
```

```
</div>
  45/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 413ms/step - loss: 195.7882

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:24 413ms/step - loss: 195.5862

<div class="k-default-codeblock">
```

```
</div>
  47/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:23 413ms/step - loss: 195.3852

<div class="k-default-codeblock">
```

```
</div>
  48/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:23 413ms/step - loss: 195.1844

<div class="k-default-codeblock">
```

```
</div>
  49/250 ━━━[37m━━━━━━━━━━━━━━━━━  1:22 412ms/step - loss: 194.9880

<div class="k-default-codeblock">
```

```
</div>
  50/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 413ms/step - loss: 194.7908

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:22 413ms/step - loss: 194.5988

<div class="k-default-codeblock">
```

```
</div>
  52/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 413ms/step - loss: 194.4096

<div class="k-default-codeblock">
```

```
</div>
  53/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 414ms/step - loss: 194.2222

<div class="k-default-codeblock">
```

```
</div>
  54/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:21 414ms/step - loss: 194.0372

<div class="k-default-codeblock">
```

```
</div>
  55/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 414ms/step - loss: 193.8545

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:20 414ms/step - loss: 193.6765

<div class="k-default-codeblock">
```

```
</div>
  57/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 414ms/step - loss: 193.4996

<div class="k-default-codeblock">
```

```
</div>
  58/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 414ms/step - loss: 193.3228

<div class="k-default-codeblock">
```

```
</div>
  59/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:19 414ms/step - loss: 193.1480

<div class="k-default-codeblock">
```

```
</div>
  60/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:18 414ms/step - loss: 192.9738

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:18 414ms/step - loss: 192.8013

<div class="k-default-codeblock">
```

```
</div>
  62/250 ━━━━[37m━━━━━━━━━━━━━━━━  1:17 414ms/step - loss: 192.6320

<div class="k-default-codeblock">
```

```
</div>
  63/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:17 413ms/step - loss: 192.4647

<div class="k-default-codeblock">
```

```
</div>
  64/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 414ms/step - loss: 192.3008

<div class="k-default-codeblock">
```

```
</div>
  65/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 414ms/step - loss: 192.1390

<div class="k-default-codeblock">
```

```
</div>
  66/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:16 413ms/step - loss: 191.9801

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 414ms/step - loss: 191.8251

<div class="k-default-codeblock">
```

```
</div>
  68/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:15 414ms/step - loss: 191.6736

<div class="k-default-codeblock">
```

```
</div>
  69/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 414ms/step - loss: 191.5252

<div class="k-default-codeblock">
```

```
</div>
  70/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 414ms/step - loss: 191.3795

<div class="k-default-codeblock">
```

```
</div>
  71/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:14 414ms/step - loss: 191.2366

<div class="k-default-codeblock">
```

```
</div>
  72/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 414ms/step - loss: 191.0975

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:13 414ms/step - loss: 190.9597

<div class="k-default-codeblock">
```

```
</div>
  74/250 ━━━━━[37m━━━━━━━━━━━━━━━  1:12 414ms/step - loss: 190.8241

<div class="k-default-codeblock">
```

```
</div>
  75/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:12 414ms/step - loss: 190.6929

<div class="k-default-codeblock">
```

```
</div>
  76/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 414ms/step - loss: 190.5656

<div class="k-default-codeblock">
```

```
</div>
  77/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 413ms/step - loss: 190.4417

<div class="k-default-codeblock">
```

```
</div>
  78/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:11 413ms/step - loss: 190.3218

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 413ms/step - loss: 190.2043

<div class="k-default-codeblock">
```

```
</div>
  80/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:10 413ms/step - loss: 190.0883

<div class="k-default-codeblock">
```

```
</div>
  81/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 414ms/step - loss: 189.9727

<div class="k-default-codeblock">
```

```
</div>
  82/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 414ms/step - loss: 189.8598

<div class="k-default-codeblock">
```

```
</div>
  83/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:09 414ms/step - loss: 189.7464

<div class="k-default-codeblock">
```

```
</div>
  84/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 413ms/step - loss: 189.6351

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:08 413ms/step - loss: 189.5255

<div class="k-default-codeblock">
```

```
</div>
  86/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:07 413ms/step - loss: 189.4160

<div class="k-default-codeblock">
```

```
</div>
  87/250 ━━━━━━[37m━━━━━━━━━━━━━━  1:07 413ms/step - loss: 189.3061

<div class="k-default-codeblock">
```

```
</div>
  88/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 413ms/step - loss: 189.1984

<div class="k-default-codeblock">
```

```
</div>
  89/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 413ms/step - loss: 189.0919

<div class="k-default-codeblock">
```

```
</div>
  90/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:06 413ms/step - loss: 188.9883

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 413ms/step - loss: 188.8874

<div class="k-default-codeblock">
```

```
</div>
  92/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:05 413ms/step - loss: 188.7877

<div class="k-default-codeblock">
```

```
</div>
  93/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 413ms/step - loss: 188.6889

<div class="k-default-codeblock">
```

```
</div>
  94/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:04 413ms/step - loss: 188.5921

<div class="k-default-codeblock">
```

```
</div>
  95/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 413ms/step - loss: 188.4958

<div class="k-default-codeblock">
```

```
</div>
  96/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 413ms/step - loss: 188.4012

<div class="k-default-codeblock">
```

```
</div>
  97/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:03 412ms/step - loss: 188.3073

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:02 413ms/step - loss: 188.2146

<div class="k-default-codeblock">
```

```
</div>
  99/250 ━━━━━━━[37m━━━━━━━━━━━━━  1:02 413ms/step - loss: 188.1229

<div class="k-default-codeblock">
```

```
</div>
 100/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 413ms/step - loss: 188.0322

<div class="k-default-codeblock">
```

```
</div>
 101/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 413ms/step - loss: 187.9417

<div class="k-default-codeblock">
```

```
</div>
 102/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:01 413ms/step - loss: 187.8521

<div class="k-default-codeblock">
```

```
</div>
 103/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 413ms/step - loss: 187.7627

<div class="k-default-codeblock">
```

```
</div>
 104/250 ━━━━━━━━[37m━━━━━━━━━━━━  1:00 413ms/step - loss: 187.6738

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 412ms/step - loss: 187.5862 

<div class="k-default-codeblock">
```

```
</div>
 106/250 ━━━━━━━━[37m━━━━━━━━━━━━  59s 412ms/step - loss: 187.4989

<div class="k-default-codeblock">
```

```
</div>
 107/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 412ms/step - loss: 187.4125

<div class="k-default-codeblock">
```

```
</div>
 108/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 412ms/step - loss: 187.3278

<div class="k-default-codeblock">
```

```
</div>
 109/250 ━━━━━━━━[37m━━━━━━━━━━━━  58s 413ms/step - loss: 187.2453

<div class="k-default-codeblock">
```

```
</div>
 110/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 412ms/step - loss: 187.1631

<div class="k-default-codeblock">
```

```
</div>
 111/250 ━━━━━━━━[37m━━━━━━━━━━━━  57s 413ms/step - loss: 187.0843

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  56s 412ms/step - loss: 187.0070

<div class="k-default-codeblock">
```

```
</div>
 113/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 413ms/step - loss: 186.9298

<div class="k-default-codeblock">
```

```
</div>
 114/250 ━━━━━━━━━[37m━━━━━━━━━━━  56s 413ms/step - loss: 186.8534

<div class="k-default-codeblock">
```

```
</div>
 115/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 412ms/step - loss: 186.7773

<div class="k-default-codeblock">
```

```
</div>
 116/250 ━━━━━━━━━[37m━━━━━━━━━━━  55s 412ms/step - loss: 186.7022

<div class="k-default-codeblock">
```

```
</div>
 117/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 412ms/step - loss: 186.6274

<div class="k-default-codeblock">
```

```
</div>
 118/250 ━━━━━━━━━[37m━━━━━━━━━━━  54s 412ms/step - loss: 186.5530

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 412ms/step - loss: 186.4787

<div class="k-default-codeblock">
```

```
</div>
 120/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 412ms/step - loss: 186.4051

<div class="k-default-codeblock">
```

```
</div>
 121/250 ━━━━━━━━━[37m━━━━━━━━━━━  53s 412ms/step - loss: 186.3325

<div class="k-default-codeblock">
```

```
</div>
 122/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 412ms/step - loss: 186.2598

<div class="k-default-codeblock">
```

```
</div>
 123/250 ━━━━━━━━━[37m━━━━━━━━━━━  52s 412ms/step - loss: 186.1888

<div class="k-default-codeblock">
```

```
</div>
 124/250 ━━━━━━━━━[37m━━━━━━━━━━━  51s 412ms/step - loss: 186.1183

<div class="k-default-codeblock">
```

```
</div>
 125/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 412ms/step - loss: 186.0488

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  51s 412ms/step - loss: 185.9801

<div class="k-default-codeblock">
```

```
</div>
 127/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 412ms/step - loss: 185.9121

<div class="k-default-codeblock">
```

```
</div>
 128/250 ━━━━━━━━━━[37m━━━━━━━━━━  50s 411ms/step - loss: 185.8446

<div class="k-default-codeblock">
```

```
</div>
 129/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 411ms/step - loss: 185.7775

<div class="k-default-codeblock">
```

```
</div>
 130/250 ━━━━━━━━━━[37m━━━━━━━━━━  49s 411ms/step - loss: 185.7108

<div class="k-default-codeblock">
```

```
</div>
 131/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 411ms/step - loss: 185.6454

<div class="k-default-codeblock">
```

```
</div>
 132/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 411ms/step - loss: 185.5812

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  48s 411ms/step - loss: 185.5181

<div class="k-default-codeblock">
```

```
</div>
 134/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 411ms/step - loss: 185.4556

<div class="k-default-codeblock">
```

```
</div>
 135/250 ━━━━━━━━━━[37m━━━━━━━━━━  47s 411ms/step - loss: 185.3941

<div class="k-default-codeblock">
```

```
</div>
 136/250 ━━━━━━━━━━[37m━━━━━━━━━━  46s 411ms/step - loss: 185.3333

<div class="k-default-codeblock">
```

```
</div>
 137/250 ━━━━━━━━━━[37m━━━━━━━━━━  46s 411ms/step - loss: 185.2732

<div class="k-default-codeblock">
```

```
</div>
 138/250 ━━━━━━━━━━━[37m━━━━━━━━━  46s 411ms/step - loss: 185.2138

<div class="k-default-codeblock">
```

```
</div>
 139/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 411ms/step - loss: 185.1546

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  45s 411ms/step - loss: 185.0956

<div class="k-default-codeblock">
```

```
</div>
 141/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 410ms/step - loss: 185.0369

<div class="k-default-codeblock">
```

```
</div>
 142/250 ━━━━━━━━━━━[37m━━━━━━━━━  44s 410ms/step - loss: 184.9791

<div class="k-default-codeblock">
```

```
</div>
 143/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 410ms/step - loss: 184.9216

<div class="k-default-codeblock">
```

```
</div>
 144/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 410ms/step - loss: 184.8649

<div class="k-default-codeblock">
```

```
</div>
 145/250 ━━━━━━━━━━━[37m━━━━━━━━━  43s 410ms/step - loss: 184.8089

<div class="k-default-codeblock">
```

```
</div>
 146/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 410ms/step - loss: 184.7534

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  42s 410ms/step - loss: 184.6985

<div class="k-default-codeblock">
```

```
</div>
 148/250 ━━━━━━━━━━━[37m━━━━━━━━━  41s 410ms/step - loss: 184.6445

<div class="k-default-codeblock">
```

```
</div>
 149/250 ━━━━━━━━━━━[37m━━━━━━━━━  41s 410ms/step - loss: 184.5908

<div class="k-default-codeblock">
```

```
</div>
 150/250 ━━━━━━━━━━━━[37m━━━━━━━━  41s 410ms/step - loss: 184.5374

<div class="k-default-codeblock">
```

```
</div>
 151/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 410ms/step - loss: 184.4844

<div class="k-default-codeblock">
```

```
</div>
 152/250 ━━━━━━━━━━━━[37m━━━━━━━━  40s 410ms/step - loss: 184.4322

<div class="k-default-codeblock">
```

```
</div>
 153/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 410ms/step - loss: 184.3806

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  39s 410ms/step - loss: 184.3291

<div class="k-default-codeblock">
```

```
</div>
 155/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 410ms/step - loss: 184.2784

<div class="k-default-codeblock">
```

```
</div>
 156/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 410ms/step - loss: 184.2283

<div class="k-default-codeblock">
```

```
</div>
 157/250 ━━━━━━━━━━━━[37m━━━━━━━━  38s 410ms/step - loss: 184.1783

<div class="k-default-codeblock">
```

```
</div>
 158/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 410ms/step - loss: 184.1288

<div class="k-default-codeblock">
```

```
</div>
 159/250 ━━━━━━━━━━━━[37m━━━━━━━━  37s 410ms/step - loss: 184.0793

<div class="k-default-codeblock">
```

```
</div>
 160/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 410ms/step - loss: 184.0304

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 410ms/step - loss: 183.9818

<div class="k-default-codeblock">
```

```
</div>
 162/250 ━━━━━━━━━━━━[37m━━━━━━━━  36s 410ms/step - loss: 183.9334

<div class="k-default-codeblock">
```

```
</div>
 163/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 410ms/step - loss: 183.8856

<div class="k-default-codeblock">
```

```
</div>
 164/250 ━━━━━━━━━━━━━[37m━━━━━━━  35s 410ms/step - loss: 183.8383

<div class="k-default-codeblock">
```

```
</div>
 165/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 410ms/step - loss: 183.7912

<div class="k-default-codeblock">
```

```
</div>
 166/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 410ms/step - loss: 183.7440

<div class="k-default-codeblock">
```

```
</div>
 167/250 ━━━━━━━━━━━━━[37m━━━━━━━  34s 410ms/step - loss: 183.6975

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 410ms/step - loss: 183.6511

<div class="k-default-codeblock">
```

```
</div>
 169/250 ━━━━━━━━━━━━━[37m━━━━━━━  33s 410ms/step - loss: 183.6050

<div class="k-default-codeblock">
```

```
</div>
 170/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 410ms/step - loss: 183.6425

<div class="k-default-codeblock">
```

```
</div>
 171/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 410ms/step - loss: 183.6795

<div class="k-default-codeblock">
```

```
</div>
 172/250 ━━━━━━━━━━━━━[37m━━━━━━━  32s 410ms/step - loss: 183.7170

<div class="k-default-codeblock">
```

```
</div>
 173/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 410ms/step - loss: 183.7562

<div class="k-default-codeblock">
```

```
</div>
 174/250 ━━━━━━━━━━━━━[37m━━━━━━━  31s 410ms/step - loss: 183.7987

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 411ms/step - loss: 183.8455

<div class="k-default-codeblock">
```

```
</div>
 176/250 ━━━━━━━━━━━━━━[37m━━━━━━  30s 411ms/step - loss: 183.8981

<div class="k-default-codeblock">
```

```
</div>
 177/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 411ms/step - loss: 183.9577

<div class="k-default-codeblock">
```

```
</div>
 178/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 410ms/step - loss: 184.0258

<div class="k-default-codeblock">
```

```
</div>
 179/250 ━━━━━━━━━━━━━━[37m━━━━━━  29s 410ms/step - loss: 184.1031

<div class="k-default-codeblock">
```

```
</div>
 180/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 410ms/step - loss: 184.1906

<div class="k-default-codeblock">
```

```
</div>
 181/250 ━━━━━━━━━━━━━━[37m━━━━━━  28s 410ms/step - loss: 184.2890

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 410ms/step - loss: 184.3992

<div class="k-default-codeblock">
```

```
</div>
 183/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 410ms/step - loss: 184.5218

<div class="k-default-codeblock">
```

```
</div>
 184/250 ━━━━━━━━━━━━━━[37m━━━━━━  27s 410ms/step - loss: 184.6576

<div class="k-default-codeblock">
```

```
</div>
 185/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 410ms/step - loss: 184.8064

<div class="k-default-codeblock">
```

```
</div>
 186/250 ━━━━━━━━━━━━━━[37m━━━━━━  26s 410ms/step - loss: 184.9685

<div class="k-default-codeblock">
```

```
</div>
 187/250 ━━━━━━━━━━━━━━[37m━━━━━━  25s 410ms/step - loss: 185.1438

<div class="k-default-codeblock">
```

```
</div>
 188/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 410ms/step - loss: 185.3325

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  25s 410ms/step - loss: 185.5339

<div class="k-default-codeblock">
```

```
</div>
 190/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 410ms/step - loss: 185.7474

<div class="k-default-codeblock">
```

```
</div>
 191/250 ━━━━━━━━━━━━━━━[37m━━━━━  24s 410ms/step - loss: 185.9724

<div class="k-default-codeblock">
```

```
</div>
 192/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 410ms/step - loss: 186.2086

<div class="k-default-codeblock">
```

```
</div>
 193/250 ━━━━━━━━━━━━━━━[37m━━━━━  23s 410ms/step - loss: 186.4555

<div class="k-default-codeblock">
```

```
</div>
 194/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 410ms/step - loss: 186.7127

<div class="k-default-codeblock">
```

```
</div>
 195/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 410ms/step - loss: 186.9798

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  22s 410ms/step - loss: 187.2562

<div class="k-default-codeblock">
```

```
</div>
 197/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 410ms/step - loss: 187.5410

<div class="k-default-codeblock">
```

```
</div>
 198/250 ━━━━━━━━━━━━━━━[37m━━━━━  21s 410ms/step - loss: 187.8340

<div class="k-default-codeblock">
```

```
</div>
 199/250 ━━━━━━━━━━━━━━━[37m━━━━━  20s 410ms/step - loss: 188.1339

<div class="k-default-codeblock">
```

```
</div>
 200/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 410ms/step - loss: 188.4406

<div class="k-default-codeblock">
```

```
</div>
 201/250 ━━━━━━━━━━━━━━━━[37m━━━━  20s 410ms/step - loss: 188.7534

<div class="k-default-codeblock">
```

```
</div>
 202/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 410ms/step - loss: 189.0715

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  19s 410ms/step - loss: 189.3942

<div class="k-default-codeblock">
```

```
</div>
 204/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 410ms/step - loss: 189.7214

<div class="k-default-codeblock">
```

```
</div>
 205/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 410ms/step - loss: 190.0522

<div class="k-default-codeblock">
```

```
</div>
 206/250 ━━━━━━━━━━━━━━━━[37m━━━━  18s 410ms/step - loss: 190.3864

<div class="k-default-codeblock">
```

```
</div>
 207/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 410ms/step - loss: 190.7234

<div class="k-default-codeblock">
```

```
</div>
 208/250 ━━━━━━━━━━━━━━━━[37m━━━━  17s 410ms/step - loss: 191.0625

<div class="k-default-codeblock">
```

```
</div>
 209/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 410ms/step - loss: 191.4030

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  16s 410ms/step - loss: 191.7450

<div class="k-default-codeblock">
```

```
</div>
 211/250 ━━━━━━━━━━━━━━━━[37m━━━━  15s 410ms/step - loss: 192.0880

<div class="k-default-codeblock">
```

```
</div>
 212/250 ━━━━━━━━━━━━━━━━[37m━━━━  15s 410ms/step - loss: 192.4316

<div class="k-default-codeblock">
```

```
</div>
 213/250 ━━━━━━━━━━━━━━━━━[37m━━━  15s 410ms/step - loss: 192.7754

<div class="k-default-codeblock">
```

```
</div>
 214/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 410ms/step - loss: 193.1192

<div class="k-default-codeblock">
```

```
</div>
 215/250 ━━━━━━━━━━━━━━━━━[37m━━━  14s 410ms/step - loss: 193.4624

<div class="k-default-codeblock">
```

```
</div>
 216/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 410ms/step - loss: 193.8050

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 409ms/step - loss: 194.1469

<div class="k-default-codeblock">
```

```
</div>
 218/250 ━━━━━━━━━━━━━━━━━[37m━━━  13s 409ms/step - loss: 194.4876

<div class="k-default-codeblock">
```

```
</div>
 219/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 409ms/step - loss: 194.8271

<div class="k-default-codeblock">
```

```
</div>
 220/250 ━━━━━━━━━━━━━━━━━[37m━━━  12s 410ms/step - loss: 195.1647

<div class="k-default-codeblock">
```

```
</div>
 221/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 410ms/step - loss: 195.5008

<div class="k-default-codeblock">
```

```
</div>
 222/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 410ms/step - loss: 195.8349

<div class="k-default-codeblock">
```

```
</div>
 223/250 ━━━━━━━━━━━━━━━━━[37m━━━  11s 410ms/step - loss: 196.1669

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  10s 410ms/step - loss: 196.4967

<div class="k-default-codeblock">
```

```
</div>
 225/250 ━━━━━━━━━━━━━━━━━━[37m━━  10s 410ms/step - loss: 196.8240

<div class="k-default-codeblock">
```

```
</div>
 226/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 410ms/step - loss: 197.1486 

<div class="k-default-codeblock">
```

```
</div>
 227/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 410ms/step - loss: 197.4706

<div class="k-default-codeblock">
```

```
</div>
 228/250 ━━━━━━━━━━━━━━━━━━[37m━━  9s 409ms/step - loss: 197.7899

<div class="k-default-codeblock">
```

```
</div>
 229/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 409ms/step - loss: 198.1065

<div class="k-default-codeblock">
```

```
</div>
 230/250 ━━━━━━━━━━━━━━━━━━[37m━━  8s 409ms/step - loss: 198.4202

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 409ms/step - loss: 198.7307

<div class="k-default-codeblock">
```

```
</div>
 232/250 ━━━━━━━━━━━━━━━━━━[37m━━  7s 409ms/step - loss: 199.0381

<div class="k-default-codeblock">
```

```
</div>
 233/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 409ms/step - loss: 199.3426

<div class="k-default-codeblock">
```

```
</div>
 234/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 409ms/step - loss: 199.6438

<div class="k-default-codeblock">
```

```
</div>
 235/250 ━━━━━━━━━━━━━━━━━━[37m━━  6s 409ms/step - loss: 199.9416

<div class="k-default-codeblock">
```

```
</div>
 236/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 409ms/step - loss: 200.2361

<div class="k-default-codeblock">
```

```
</div>
 237/250 ━━━━━━━━━━━━━━━━━━[37m━━  5s 409ms/step - loss: 200.5270

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 409ms/step - loss: 200.8147

<div class="k-default-codeblock">
```

```
</div>
 239/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 409ms/step - loss: 201.0989

<div class="k-default-codeblock">
```

```
</div>
 240/250 ━━━━━━━━━━━━━━━━━━━[37m━  4s 409ms/step - loss: 201.3797

<div class="k-default-codeblock">
```

```
</div>
 241/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 409ms/step - loss: 201.6572

<div class="k-default-codeblock">
```

```
</div>
 242/250 ━━━━━━━━━━━━━━━━━━━[37m━  3s 409ms/step - loss: 201.9312

<div class="k-default-codeblock">
```

```
</div>
 243/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 409ms/step - loss: 202.2020

<div class="k-default-codeblock">
```

```
</div>
 244/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 409ms/step - loss: 202.4696

<div class="k-default-codeblock">
```

```
</div>
 245/250 ━━━━━━━━━━━━━━━━━━━[37m━  2s 409ms/step - loss: 202.7338

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 409ms/step - loss: 202.9948

<div class="k-default-codeblock">
```

```
</div>
 247/250 ━━━━━━━━━━━━━━━━━━━[37m━  1s 409ms/step - loss: 203.2525

<div class="k-default-codeblock">
```

```
</div>
 248/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 409ms/step - loss: 203.5071

<div class="k-default-codeblock">
```

```
</div>
 249/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 409ms/step - loss: 203.7583

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 409ms/step - loss: 204.0063

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 102s 409ms/step - loss: 204.2523


---
## Inference

We use our model to generate new valid molecules from different points of the latent space.

### Generate unique Molecules with the model


```python
molecules = model.inference(1000)

MolsToGridImage(
    [m for m in molecules if m is not None][:1000], molsPerRow=5, subImgSize=(260, 160)
)
```

    
  1/32 [37m━━━━━━━━━━━━━━━━━━━━  3s 120ms/step

<div class="k-default-codeblock">
```

```
</div>
  3/32 ━[37m━━━━━━━━━━━━━━━━━━━  0s 26ms/step 

<div class="k-default-codeblock">
```

```
</div>
  5/32 ━━━[37m━━━━━━━━━━━━━━━━━  0s 26ms/step

<div class="k-default-codeblock">
```

```
</div>
  7/32 ━━━━[37m━━━━━━━━━━━━━━━━  0s 26ms/step

<div class="k-default-codeblock">
```

```
</div>
 10/32 ━━━━━━[37m━━━━━━━━━━━━━━  0s 26ms/step

<div class="k-default-codeblock">
```

```
</div>
 13/32 ━━━━━━━━[37m━━━━━━━━━━━━  0s 25ms/step

<div class="k-default-codeblock">
```

```
</div>
 15/32 ━━━━━━━━━[37m━━━━━━━━━━━  0s 25ms/step

<div class="k-default-codeblock">
```

```
</div>
 17/32 ━━━━━━━━━━[37m━━━━━━━━━━  0s 25ms/step

<div class="k-default-codeblock">
```

```
</div>
 20/32 ━━━━━━━━━━━━[37m━━━━━━━━  0s 25ms/step

<div class="k-default-codeblock">
```

```
</div>
 22/32 ━━━━━━━━━━━━━[37m━━━━━━━  0s 25ms/step

<div class="k-default-codeblock">
```

```
</div>
 24/32 ━━━━━━━━━━━━━━━[37m━━━━━  0s 25ms/step

<div class="k-default-codeblock">
```

```
</div>
 26/32 ━━━━━━━━━━━━━━━━[37m━━━━  0s 25ms/step

<div class="k-default-codeblock">
```

```
</div>
 28/32 ━━━━━━━━━━━━━━━━━[37m━━━  0s 25ms/step

<div class="k-default-codeblock">
```

```
</div>
 30/32 ━━━━━━━━━━━━━━━━━━[37m━━  0s 25ms/step

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step

<div class="k-default-codeblock">
```

```
</div>
 32/32 ━━━━━━━━━━━━━━━━━━━━ 1s 28ms/step





    
![png](/img/examples/generative/molecule_generation/molecule_generation_21_16.png)
    



### Display latent space clusters with respect to molecular properties (QAE)


```python

def plot_latent(vae, data, labels):
    # display a 2D plot of the property in the latent space
    z_mean, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


plot_latent(model, [adjacency_tensor[:8000], feature_tensor[:8000]], qed_tensor[:8000])
```

    
   1/250 [37m━━━━━━━━━━━━━━━━━━━━  26s 107ms/step

<div class="k-default-codeblock">
```

```
</div>
   6/250 [37m━━━━━━━━━━━━━━━━━━━━  2s 10ms/step  

<div class="k-default-codeblock">
```

```
</div>
  11/250 [37m━━━━━━━━━━━━━━━━━━━━  2s 10ms/step

<div class="k-default-codeblock">
```

```
</div>
  16/250 ━[37m━━━━━━━━━━━━━━━━━━━  2s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  21/250 ━[37m━━━━━━━━━━━━━━━━━━━  2s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  26/250 ━━[37m━━━━━━━━━━━━━━━━━━  2s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  31/250 ━━[37m━━━━━━━━━━━━━━━━━━  2s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  36/250 ━━[37m━━━━━━━━━━━━━━━━━━  2s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  41/250 ━━━[37m━━━━━━━━━━━━━━━━━  2s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  46/250 ━━━[37m━━━━━━━━━━━━━━━━━  2s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  51/250 ━━━━[37m━━━━━━━━━━━━━━━━  2s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  56/250 ━━━━[37m━━━━━━━━━━━━━━━━  2s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  61/250 ━━━━[37m━━━━━━━━━━━━━━━━  2s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  67/250 ━━━━━[37m━━━━━━━━━━━━━━━  1s 11ms/step

<div class="k-default-codeblock">
```

```
</div>
  73/250 ━━━━━[37m━━━━━━━━━━━━━━━  1s 10ms/step

<div class="k-default-codeblock">
```

```
</div>
  79/250 ━━━━━━[37m━━━━━━━━━━━━━━  1s 10ms/step

<div class="k-default-codeblock">
```

```
</div>
  85/250 ━━━━━━[37m━━━━━━━━━━━━━━  1s 10ms/step

<div class="k-default-codeblock">
```

```
</div>
  91/250 ━━━━━━━[37m━━━━━━━━━━━━━  1s 10ms/step

<div class="k-default-codeblock">
```

```
</div>
  98/250 ━━━━━━━[37m━━━━━━━━━━━━━  1s 10ms/step

<div class="k-default-codeblock">
```

```
</div>
 105/250 ━━━━━━━━[37m━━━━━━━━━━━━  1s 10ms/step

<div class="k-default-codeblock">
```

```
</div>
 112/250 ━━━━━━━━[37m━━━━━━━━━━━━  1s 10ms/step

<div class="k-default-codeblock">
```

```
</div>
 119/250 ━━━━━━━━━[37m━━━━━━━━━━━  1s 9ms/step 

<div class="k-default-codeblock">
```

```
</div>
 126/250 ━━━━━━━━━━[37m━━━━━━━━━━  1s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 133/250 ━━━━━━━━━━[37m━━━━━━━━━━  1s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 140/250 ━━━━━━━━━━━[37m━━━━━━━━━  0s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 147/250 ━━━━━━━━━━━[37m━━━━━━━━━  0s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 154/250 ━━━━━━━━━━━━[37m━━━━━━━━  0s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 161/250 ━━━━━━━━━━━━[37m━━━━━━━━  0s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 168/250 ━━━━━━━━━━━━━[37m━━━━━━━  0s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 175/250 ━━━━━━━━━━━━━━[37m━━━━━━  0s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 182/250 ━━━━━━━━━━━━━━[37m━━━━━━  0s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 189/250 ━━━━━━━━━━━━━━━[37m━━━━━  0s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 196/250 ━━━━━━━━━━━━━━━[37m━━━━━  0s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 203/250 ━━━━━━━━━━━━━━━━[37m━━━━  0s 9ms/step

<div class="k-default-codeblock">
```

```
</div>
 210/250 ━━━━━━━━━━━━━━━━[37m━━━━  0s 8ms/step

<div class="k-default-codeblock">
```

```
</div>
 217/250 ━━━━━━━━━━━━━━━━━[37m━━━  0s 8ms/step

<div class="k-default-codeblock">
```

```
</div>
 224/250 ━━━━━━━━━━━━━━━━━[37m━━━  0s 8ms/step

<div class="k-default-codeblock">
```

```
</div>
 231/250 ━━━━━━━━━━━━━━━━━━[37m━━  0s 8ms/step

<div class="k-default-codeblock">
```

```
</div>
 238/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 8ms/step

<div class="k-default-codeblock">
```

```
</div>
 246/250 ━━━━━━━━━━━━━━━━━━━[37m━  0s 8ms/step

<div class="k-default-codeblock">
```

```
</div>
 250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step



    
![png](/img/examples/generative/molecule_generation/molecule_generation_23_41.png)
    


---
## Conclusion

In this example, we combined model architectures from two papers,
"Automatic chemical design using a data-driven continuous representation of
molecules" from 2016 and the "MolGAN" paper from 2018. The former paper
treats SMILES inputs as strings and seeks to generate molecule strings in SMILES format,
while the later paper considers SMILES inputs as graphs (a combination of adjacency
matrices and feature matrices) and seeks to generate molecules as graphs.

This hybrid approach enables a new type of directed gradient-based search through chemical space.

Example available on HuggingFace

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-molecule%20generation%20with%20VAE-black.svg)](https://huggingface.co/keras-io/drug-molecule-generation-with-VAE) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-molecule%20generation%20with%20VAE-black.svg)](https://huggingface.co/spaces/keras-io/generating-drug-molecule-with-VAE) |

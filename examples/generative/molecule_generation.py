"""
Title: Drug Molecule Generation
Author: [Victor Basu](https://www.linkedin.com/in/victor-basu-520958147)
Date created: 2022/02/27
Last modified: 2022/02/27
Description: Implementing a Convolutional Variational AutoEncoder (VAE) for Drug Discovery.
"""
"""
## Introduction

In this example, we would be trying to solve one of the Drug Discovery tasks which is to
generate molecules with the help of a Variational Autoencoder.
We would be considering the research paper [**Automatic chemical design using a
data-driven continuous representation of molecules**](https://arxiv.org/abs/1610.02415).
The model explained in this paper generate new molecules for efficient exploration and
optimization through open-ended spaces of chemical compounds. The Model consists of three
components: Encoder, Decoder and Predictor.  The encoder converts the discrete
representation of a molecule into a real-valued continuous vector, and the decoder
converts these continuous vectors back to discrete molecular representations. The
predictor estimates chemical properties from the latent continuous vector representation
of the molecule. Continuous representations allow the use of powerful gradient-based
optimization to efficiently guide the search for optimized functional compounds.

![intro](https://pubs.acs.org/na101/home/literatum/publisher/achs/journals/content/acscii/2018/acscii.2018.4.issue-2/acscentsci.7b00572/20180223/images/medium/oc-2017-00572f_0001.gif)

**Figure (a)** - A diagram of the autoencoder used for molecular design, including the
joint property prediction model. Starting from a discrete molecular representation, such
as a SMILES string, the encoder network converts each molecule into a vector in the
latent space, which is effectively a continuous molecular representation. Given a point
in the latent space, the decoder network produces a corresponding SMILES string. A
mutilayer perceptron network estimates the value of target properties associated with
each molecule.

**Figure (b)** - Gradient-based optimization in continuous latent space. After training a
surrogate model f(z) to predict the properties of molecules based on their latent
representation z, we can optimize f(z) with respect to z to find new latent
representations expected to have high values of desired properties. These new latent
representations can then be decoded into SMILES strings, at which point their properties
can be tested empirically.
"""

"""
## Dataset

We would be using [**ZINC – A Free Database of Commercially Available Compounds for
Virtual Screening**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1360656/) dataset. The
dataset comes with molecular formula in SMILE representation along with their respective
molecular properties such as **logP**( Shows the water–octanal partition coefficient ),
**SAS**( Shows the synthetic accessibility score ) and **QED**( Shows the Qualitative
Estimate of Drug-likeness ).

"""

"""shell
!wget
https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv
"""

"""
## Setup
"""

"""shell
!pip -q install rdkit-pypi
"""

import ast

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage

import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt

tf.random.set_seed(2022)

df = pd.read_csv("/content/250k_rndm_zinc_drugs_clean_3.csv")
df["smiles"] = df["smiles"].apply(lambda s: s.replace("\n", ""))
df.head()


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


print(f"SMILES:\t{df.smiles[100]}\nlogP:\t{df.logP[100]}\nqed:\t{df.qed[100]}")
molecule = molecule_from_smiles(df.iloc[100].smiles)
print("Molecule:")
molecule

SMILE_CHARSET = '[" ", "#", ")", "(", "+", "-", "/", "1", "3", "2", "5", "4", "7", \
                 "6", "=", "@", "C", "B", "F", "I", "H", "O", "N", "S", "[", "]", \
                 "c", "l", "o", "n", "P", "s", "r"]'

SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)
SMILE_CHARSET.append("\\")

MAX_MOLSIZE = max(df["smiles"].str.len())
SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
print("Max molecule size: {}".format(MAX_MOLSIZE))
print("Character set Length: {}".format(len(SMILE_CHARSET)))

"""
## Hyperparameters
"""

latent_dim = 292
BATCH_SIZE = 256
EPOCHS = 50
MAX_LEN = 120
VAE_LR = 1e-4

"""
## Build Dataloader
"""


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, mapping, max_len, batch_size=6, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.indices = self.data.index.tolist()
        self.mapping = mapping
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # Generate one batch of data
        # Generate indices of the batch
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        smile, qed = self.data_generation(batch)

        return smile, qed

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load(self, idx):
        """
        Load molecules in SMILE representation
        and their respective QED value.
        """
        item = self.data.loc[idx]["smiles"]
        item = list(item)
        item = [*map(self.mapping.get, item)]

        # padding
        item = item + [0] * abs((len(item) - self.max_len))

        # One hot Encoding
        item = tf.one_hot(item, depth=34)

        qed = self.data.loc[idx]["qed"]

        return item, qed

    def data_generation(self, batch):

        x0 = np.empty((self.batch_size, self.max_len, len(self.mapping)))
        x1 = np.empty((self.batch_size,))

        for i, batch_id in enumerate(batch):
            (
                x0[
                    i,
                ],
                x1[
                    i,
                ],
            ) = self.load(batch_id)

        return x0, x1


train_df = df.sample(frac=0.75, random_state=42)  # random state is a seed value
test_df = df.drop(train_df.index)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

"""
# Build the Encoder and Decoder
"""


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.conv_1 = layers.Conv1D(filters=9, kernel_size=9, activation="gelu")
        self.conv_2 = layers.Conv1D(filters=9, kernel_size=9, activation="gelu")
        self.conv_3 = layers.Conv1D(filters=10, kernel_size=11, activation="gelu")
        self.flatten = layers.Flatten()
        self.linear_0 = layers.Dense(435, activation="selu")
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.log_var = layers.Dense(latent_dim, name="log_var")

    def call(self, input_tensor):
        x = self.conv_1(input_tensor)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        x = self.linear_0(x)
        z_mean = self.z_mean(x)
        log_var = self.log_var(x)

        return z_mean, log_var


class Decoder(tf.keras.Model):
    def __init__(self, latent_dim, max_length, charset_length, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.linear_1 = layers.Dense(latent_dim, activation="selu")
        self.repeat_vector = layers.RepeatVector(max_length)
        self.gru_1 = layers.GRU(701, dropout=0.0, return_sequences=True)
        self.gru_2 = layers.GRU(701, dropout=0.4, return_sequences=True)
        self.gru_3 = layers.GRU(901, dropout=0.0, return_sequences=True)
        self.linear_2 = layers.TimeDistributed(layers.Dense(charset_length))

    def call(self, x):
        x = self.linear_1(x)
        x = self.repeat_vector(x)
        x = self.gru_1(x)
        x = self.gru_2(x)
        x = self.gru_3(x)
        x = self.linear_2(x)
        x = tf.keras.activations.softmax(x, axis=2)

        return x


"""
## Build Sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_log_var)[0]
        dim = tf.shape(z_log_var)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
## Building the VAE
With this model we try to optimize three losses:


* Categorical Crossentropy Loss
* KL Loss
* Property prediction Loss

The Categorical Crossentropy loss is to give a measure of the model's reconstruction
capacity. The Property prediction loss is to estimate the mean squared error of the
predicted and actual property after passing the ```z_mean(generated from the encoder)```
through a property prediction model or property prediction layer.
"""


class MoleculeGenerator(tf.keras.Model):
    def __init__(self, encoder, decoder, max_len, **kwargs):
        super(MoleculeGenerator, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.property_prediction_layer = layers.Dense(1)
        self.max_len = max_len

        self.train_total_loss_tracker = tf.keras.metrics.Mean(name="train_total_loss")
        self.train_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="train_reconstruction_loss"
        )
        self.train_kl_loss_tracker = tf.keras.metrics.Mean(name="train_kl_loss")
        self.train_pp_loss_tracker = tf.keras.metrics.Mean(
            name="train_property_prediction_loss"
        )

        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="val_reconstruction_loss"
        )
        self.val_kl_loss_tracker = tf.keras.metrics.Mean(name="val_kl_loss")
        self.val_pp_loss_tracker = tf.keras.metrics.Mean(
            name="val_property_prediction_loss"
        )

    def compile(self, vae_optimizer, **kwargs):
        super().compile(**kwargs)
        self.vae_optimizer = vae_optimizer

    def train_step(self, data):
        smile, sas = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, reconstruction, property_prediction = self(
                smile, training=True
            )

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.categorical_crossentropy(smile, reconstruction),
                    axis=(1),
                )
            )
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1
            )
            kl_loss = tf.reduce_mean(kl_loss)

            pp_loss = tf.reduce_sum(
                tf.keras.losses.mean_squared_error(sas, property_prediction)
            )
            total_loss = kl_loss + pp_loss + reconstruction_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.vae_optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.train_total_loss_tracker.update_state(total_loss)
        self.train_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.train_kl_loss_tracker.update_state(kl_loss)
        self.train_pp_loss_tracker.update_state(pp_loss)
        return {
            "loss": self.train_total_loss_tracker.result(),
            "reconstruction_loss": self.train_reconstruction_loss_tracker.result(),
            "kl_loss": self.train_kl_loss_tracker.result(),
            "property_prediction_loss": self.train_pp_loss_tracker.result(),
        }

    def test_step(self, data):
        smile, sas = data

        z_mean, z_log_var, z, reconstruction, property_prediction = self(
            smile, training=False
        )

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.categorical_crossentropy(smile, reconstruction),
                axis=(1),
            )
        )

        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1
        )
        kl_loss = tf.reduce_mean(kl_loss)

        pp_loss = tf.keras.losses.mean_squared_error(sas, property_prediction)
        pp_loss = tf.reduce_sum(
            tf.keras.losses.mean_squared_error(sas, property_prediction)
        )

        total_loss = kl_loss + pp_loss + reconstruction_loss

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        self.val_pp_loss_tracker.update_state(pp_loss)

        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
            "property_prediction_loss": self.val_pp_loss_tracker.result(),
        }

    def call(self, x):
        z_mean, log_var = self.encoder(x)
        z = Sampling()([z_mean, log_var])

        reconstruction = self.decoder(z)

        property_prediction = self.property_prediction_layer(z_mean)

        return z_mean, log_var, z, reconstruction, property_prediction


"""
## Model Training
"""

train_loader = DataGenerator(
    data=train_df[:5000], mapping=SMILE_to_index, max_len=MAX_LEN, batch_size=BATCH_SIZE
)

validation_loader = DataGenerator(
    data=test_df[:5000], mapping=SMILE_to_index, max_len=MAX_LEN, batch_size=BATCH_SIZE
)

vae_optimizer = tf.keras.optimizers.Adam(learning_rate=VAE_LR)

encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim, max_length=MAX_LEN, charset_length=len(SMILE_CHARSET))

model = MoleculeGenerator(encoder, decoder, MAX_MOLSIZE)

model.compile(vae_optimizer)
history = model.fit(train_loader, epochs=EPOCHS, validation_data=validation_loader)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("VAE Loss")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

"""
## Model Inferencing
"""

mol = model.predict(next(iter(validation_loader))[0])[3].argmax(axis=2)[0]
mol = [*map(index_to_SMILE.get, mol)]
mol = "".join(mol)
mol = molecule_from_smiles("".join(mol))
print("Molecule:")

mol

"""
### Trying to generate 100 unique Molecules with the model.
"""

store_gen_mol = []
for x in range(100):
    gen_mol = tf.argmax(model.decoder(np.random.randn(1, latent_dim)), axis=2).numpy()[
        0
    ]
    gen_mol = [*map(index_to_SMILE.get, gen_mol)]
    store_gen_mol.append("".join(gen_mol))

store_gen_mol = np.unique(np.array(store_gen_mol))
print(store_gen_mol)

"""
## Display the latent space clusters with respect to Molecular properties( QAE ).
"""


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the property in the latent space
    z_mean, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


train_data = DataGenerator(
    data=train_df, mapping=SMILE_to_index, max_len=MAX_LEN, batch_size=40000
)
x_train, y_train = next(iter(train_data))
plot_label_clusters(model, x_train, y_train)

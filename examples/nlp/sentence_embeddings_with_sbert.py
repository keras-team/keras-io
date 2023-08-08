"""
Title: Sentence embeddings using Siamese RoBERTa-networks with kerasNLP
Author: [Mohammed Abu El-Nasr](https://github.com/abuelnasr0)
Date created: 2023/07/14
Last modified: 2023/08/08
Description: Use kerasNLP to Fine-tune a RoBERTa model to generate sentence embeddings for
semantic similarity and clustering tasks.
Accelerator: GPU
"""

"""
## Introduction

BERT or RoBERTa can be used for semantic textual similarity tasks, where two sentences
are passed to the model, and the network predicts whether they are similar or not. But
what if we have a large collection of sentences and want to find the most similar pairs
in that collection? That will take n*(n-1)/2 inference computations, where n is the
number of sentences in the collection. For example, if n = 10000, the required time will
be 65 hours on a V100 GPU.

A common method to overcome the time overhead issue is to pass one sentence to the model,
then average the output of the model, or take the first token (the [CLS] token) and use
them as a [sentence embedding](https://en.wikipedia.org/wiki/Sentence_embedding), then
use a vector similarity measure like cosine similarity or Manhatten / Euclidean distance
to find close sentences (semantically similar sentences). That will reduce the time to
find the most similar pairs in a collection of 10,000 sentences from 65 hours to 5
seconds!

If we use RoBERTa directly, that will yield rather bad sentence embeddings. But if we
fine-tune RoBERTa using a Siamese network, that will generate semantically meaningful
sentence embeddings. This will enable RoBERTa to be used for new tasks. These tasks
include:

- Large-scale semantic similarity comparison.
- Clustering.
- Information retrieval via semantic search.

In this example, we will show how to fine-tune a RoBERTa model using Siamese networks
such that it will be able to produce semantically meaningful sentence embeddings and use
them in a semantic similarity and clustering example. This method of fine-tuning was 
introduced in [Sentence-BERT](https://arxiv.org/abs/1908.10084)
"""

"""
## Setup

Let's install and import the libraries we need. We'll be using the
[KerasNLP](https://keras.io/keras_nlp/) and [Keras Core](https://keras.io/keras_core/)
libraris in this example. Keras core enables to work with any of Tensorflow, JAX, or
Torch. Keras Core is supported by KerasNLP, simply change the `KERAS_BACKEND` environment
variable to select a backend of your choice.

We will also enable [mixed perceciosn](https://keras.io/keras_core/api/mixed_precision/policy/)
training. This will help us reduce the training time.
"""

"""shell
pip install keras-core --upgrade -q
pip install keras-nlp --upgrade -q
"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # or tensorflow or torch

import keras_core as keras
import keras_nlp
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn.cluster as cluster

policy = keras.mixed_precision.Policy("mixed_float16")
keras.mixed_precision.set_global_policy(policy)

"""
## Fine-tune the model using siamese networks

[Siamese network](https://en.wikipedia.org/wiki/Siamese_neural_network) is a neural
network architecture that contains two or more subnetworks. The subnetworks share the
same weights. It is used to generate feature vectors for each input and then compare them
for similarity.

For our example, the subnetwork will be a RoBERTa model that has a pooling layer on top
of it to produce the embeddings of the input sentences. These embeddings will then be
compared with each other to learn to produce semantically meaningful embeddings.

The pooling strategies used are mean, max, and CLS pooling. Mean pooling produces the
best results. We will use it in our examples.

We will use two methods for fine-tuning:

- Fine-tune using the regression objective function
- Fine-tune using the triplet Objective function
"""

"""
### Build the encoder model.

Now, we'll build the encoder model class that will produce the sentence embeddings. This
model will be the subnetwork for the Siamese newtork in both fine-tuning methods

The encoder consists of:

- A backbone model that will generate the contextual representation of each token in the
sentence.
- A mean pooling layer to produce the embeddings. We will use `keras.layers.GlobalAveragePooling1D`
to apply the mean pooling to the backbone outputs. We will pass the padding mask to the
layer to exclude padded tokens from being averaged.
- A normalization layer to normalize the embeddings. `normalize` argument will decide
whether to apply the normalization.
"""


class Encoder(keras.Model):
    def __init__(self, backbone, normalize, **kwargs):
        token_ids = keras.Input(shape=(512,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(shape=(512,), dtype="bool", name="padding_mask")

        inputs = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        h = backbone(inputs)
        embeddings = keras.layers.GlobalAveragePooling1D(name="pooling_layer")(
            h, mask=inputs["padding_mask"]
        )

        if normalize:
            embeddings = keras.layers.UnitNormalization(axis=-1)(embeddings)

        super().__init__(
            inputs=inputs,
            outputs=embeddings,
            **kwargs,
        )

        self.backbone = backbone
        self.normalize = normalize


"""
### Fine-tune using the regression objective function

For building the siamese network with the regression objective function, the siamese
network is asked to predict the cosine similarity between the embeddings of the two input
sentences.

Cosine similarity indicates the angle between the sentence embeddings. If the cosine
similarity is high, that means there is a small angle between the embeddings; hence, they
are semantically similar.
"""

"""
#### Load the dataset

We will use the STSB dataset to fine-tune the model for the regression objective. STSB
consists of a collection of sentence pairs that are labelled in the range [0, 5]. 0
indicates the least semantic similarity between the two sentences, and 5 indicates the
most semantic similarity between the two sentences.

We will use `tensorflow_datasets` to load the data and `tf.data` API to preprocess the
data.

We will preprocess the sentences using `keras_nlp.models.RobertaPreprocessor` that will
generate `token_ids`, and `padding_mask` which are the input for the RoBERTa model.

The range of the cosine similarity scores is [-1, 1] and it's the output of the siamese
network, but the range of the labels in the dataset is [0, 5]. We need to unify the range
between the cosine similarity and the dataset labels, so while preparing the dataset, we 
will divide the labels by 2.5 and subtract 1.

We will use 2400 pairs for training data and 400 pairs for testing data.
"""

TRAIN_BATCH_SIZE = 8
TRAIN_NUM_BATCHS = 300

VALIDATION_BATCH_SIZE = 8
VALIDATION_NUM_BATCHS = 50

# Used to tune the value dynamically at runtime.
AUTOTUNE = tf.data.AUTOTUNE

preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset("roberta_base_en")


def one_dictionary(x):
    """
    Returns the data as one dictionary to be appropriate input for
    `keras_core.Model`
    """
    return {
        "token_ids_1": x[0]["token_ids"],
        "padding_mask_1": x[0]["padding_mask"],
        "token_ids_2": x[1]["token_ids"],
        "padding_mask_2": x[1]["padding_mask"],
    }


def change_range(x):
    return (x / 2.5) - 1


def prepare_dataset(dataset, num_batchs, batch_size):
    dataset = dataset.batch(batch_size)
    dataset = dataset.take(num_batchs)
    dataset = dataset.map(
        lambda x: (
            one_dictionary(
                (preprocessor(x["sentence1"]), preprocessor(x["sentence2"]))
            ),
            change_range(x["label"]),
        ),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


stsb_ds = tfds.load(
    "glue/stsb",
)
stsb_train, stsb_valid = stsb_ds["train"], stsb_ds["validation"]

stsb_train = prepare_dataset(stsb_train, TRAIN_NUM_BATCHS, TRAIN_BATCH_SIZE)
stsb_valid = prepare_dataset(stsb_valid, VALIDATION_NUM_BATCHS, VALIDATION_BATCH_SIZE)

"""
#### Build the Siamese network with the regression objective function.

It's described above that the Siamese network has two or more subnetworks, and for this
Siamese network, we need two encoders. But we don't have two encoders; we have only one
encoder, but we will pass the two sentences through it. That way, we can have two paths
to get the embeddings and also shared weights between the two paths.

The Siamese network will receive two preprocessed sentences. They will be passed to the
encoder with `normalize = True` that will produce normalized embeddings (because we are
calculating the cosine simalarity scores). We will apply the row wise dot product between
the normalized embeddings to get the cosine simalarity scores between each pair of
sentences.
"""


class RowWiseDotProduct(keras.layers.Layer):
    def call(self, x, y):
        return keras.ops.einsum("ij,ij->i", x, y)


class RegressionSiamese(keras.Model):
    def __init__(self, encoder, **kwargs):
        token_ids_1 = keras.Input(shape=(512,), dtype="int32", name="token_ids_1")
        padding_mask_1 = keras.Input(shape=(512,), dtype="bool", name="padding_mask_1")
        token_ids_2 = keras.Input(shape=(512,), dtype="int32", name="token_ids_2")
        padding_mask_2 = keras.Input(shape=(512,), dtype="bool", name="padding_mask_2")

        # first sentence preprocessed input
        sentence_1 = {
            "token_ids": token_ids_1,
            "padding_mask": padding_mask_1,
        }
        # second sentence preprocessed input
        sentence_2 = {
            "token_ids": token_ids_2,
            "padding_mask": padding_mask_2,
        }

        u = encoder(sentence_1)
        v = encoder(sentence_2)
        cosine_similarity_scores = RowWiseDotProduct()(u, v)

        super().__init__(
            inputs={
                "token_ids_1": token_ids_1,
                "padding_mask_1": padding_mask_1,
                "token_ids_2": token_ids_2,
                "padding_mask_2": padding_mask_2,
            },
            outputs=cosine_similarity_scores,
            **kwargs,
        )

        self.encoder = encoder

    def get_encoder(self):
        return self.encoder


"""
#### Fit the model

Let's build and compile the model. We will first create the encoder by passing the
RoBERTa backbone and specifying `normalize = True`. Then, we will pass the encoder to the
Siamese network and compile the Siamese network model. We will use `MeanSquaredError()`
as loss function, and `Adam()` optimizer with learning rate = 2e-5. We will enable [jit
compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation) for faster training.
"""

backbone = keras_nlp.models.RobertaBackbone.from_preset("roberta_base_en")
roberta_encoder = Encoder(backbone, True)

siamese_regression_roberta = RegressionSiamese(roberta_encoder)

siamese_regression_roberta.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(2e-5),
    jit_compile=True,
)

siamese_regression_roberta.summary()

"""
Let's try this example before training and compare it to the output after training.
"""

sentences = [
    "Today is a very sunny day.",
    "I am hungry, I will get my meal.",
    "The dog is eating his food.",
]
query = ["The dog is enjoying his meal."]

encoder = siamese_regression_roberta.get_encoder()

sentence_embeddings = keras.ops.transpose(encoder(preprocessor(sentences)))
query_embedding = encoder(preprocessor(query))

if keras.backend.backend() == "torch":
    sentence_embeddings = sentence_embeddings.cpu().detach().numpy()
    query_embedding = query_embedding.cpu().detach().numpy()

cosine_similarity_scores = keras.ops.matmul(query_embedding, sentence_embeddings)
for i, sim in enumerate(cosine_similarity_scores[0]):
    print(f"cosine similarity score between sentence {i+1} and the query = {sim} ")

"""
Let's fit the model!
"""

siamese_regression_roberta.fit(
    stsb_train,
    epochs=1,
)

"""
Let's try the example again after the training and see the difference.
"""

sentences = [
    "Today is a very sunny day.",
    "I am hungry, I will get my meal.",
    "The dog is eating his food.",
]
query = ["The dog is enjoying his meal."]

encoder = siamese_regression_roberta.get_encoder()

sentence_embeddings = keras.ops.transpose(encoder(preprocessor(sentences)))
query_embedding = encoder(preprocessor(query))

if os.environ["KERAS_BACKEND"] == "torch":
    sentence_embeddings = sentence_embeddings.cpu().detach().numpy()
    query_embedding = query_embedding.cpu().detach().numpy()

cosine_similarity_scores = keras.ops.matmul(query_embedding, sentence_embeddings)
for i, sim in enumerate(cosine_similarity_scores[0]):
    print(f"cosine similarity score between sentence {i+1} and the query = {sim} ")

"""
Let's evaluate the model.
"""

siamese_regression_roberta.evaluate(
    stsb_valid,
)

"""
### Fine-tune Using the triplet Objective Function

For the Siamese network with the triplet objective function, three sentences are passed
to the Siamese network *anchor*, *positive*, and *negative* sentences. *anchor* and
*positive* sentences are semantically similar, and *anchor* and *negative* sentences are
semantically dissimilar. The objective is to minimize the distance between the *anchor*
sentence and the *positive* sentence, and to maximize the distance between the *anchor*
sentence and the *negative* sentence.
"""

"""
#### Load the dataset

We will use the Wikipedia-sections-triplets dataset for fine-tuning. This dataset consists 
of sentences derived from the Wikipedia website. It has a collection of 3 sentences 
*anchor*, *positive*, *negative*. *anchor* and *positive* are derived from the same 
section. *anchor* and *negative* are derived from different sections.

This dataset has 1.8 million training triplets and 220,000 test triplets. In this example,
we will only use 1800 triplets for training and 300 for testing.
"""

"""shell
wget https://sbert.net/datasets/wikipedia-sections-triplets.zip -q
unzip wikipedia-sections-triplets.zip  -d  wikipedia-sections-triplets
"""

NUM_TRAIN_BATCHS = 300
NUM_TEST_BATCHS = 50
AUTOTUNE = tf.data.experimental.AUTOTUNE

preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset("roberta_base_en")


def one_dictionary(x):
    return {
        "token_ids_a": x[0]["token_ids"],
        "padding_mask_a": x[0]["padding_mask"],
        "token_ids_p": x[1]["token_ids"],
        "padding_mask_p": x[1]["padding_mask"],
        "token_ids_n": x[2]["token_ids"],
        "padding_mask_n": x[2]["padding_mask"],
    }


def prepare_wiki_data(dataset, num_batchs):
    dataset = dataset.map(
        lambda z: (
            one_dictionary(
                (
                    preprocessor(z["Sentence1"]),
                    preprocessor(z["Sentence2"]),
                    preprocessor(z["Sentence3"]),
                )
            ),
            0,
        )
    )
    dataset = dataset.take(num_batchs)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


wiki_train = tf.data.experimental.make_csv_dataset(
    "wikipedia-sections-triplets/train.csv",
    batch_size=6,
    num_epochs=1,
)
wiki_test = tf.data.experimental.make_csv_dataset(
    "wikipedia-sections-triplets/test.csv",
    batch_size=6,
    num_epochs=1,
)

wiki_train = prepare_wiki_data(wiki_train, NUM_TRAIN_BATCHS)
wiki_test = prepare_wiki_data(wiki_test, NUM_TEST_BATCHS)

"""
#### Build the Siamese network with the triplet objective function

For the Siamese network with the triplet objective function, we will build the model with
an encoder, and we will pass the three sentences through that encoder. We will get an
embedding for each sentence, and we will calculate the distance between the *anchor* and
the *positive* embeddings `positive_distance`,
and the distance between the *anchor* and the *negative* embeddings `negative_distance`
that will be passed to the loss function described below.
"""


class EuclideanDistance(keras.layers.Layer):
    """
    Calculates the euclidean distance between two embeddings.
    """

    def call(self, x, y):
        squared_sum = keras.ops.sum(keras.ops.power(x - y, 2), axis=1)
        return keras.ops.sqrt(squared_sum)


class StackLayer(keras.layers.Layer):
    """
    Stacks `positive_distance` and `negative_distance` to be passed to the loss
    function.
    """

    def call(self, x, y):
        return keras.ops.stack([x, y], axis=1)


class TripletSiamese(keras.Model):
    def __init__(self, encoder, **kwargs):
        token_ids_a = keras.Input(shape=(512,), dtype="int32", name="token_ids_a")
        padding_mask_a = keras.Input(shape=(512,), dtype="bool", name="padding_mask_a")

        token_ids_p = keras.Input(shape=(512,), dtype="int32", name="token_ids_p")
        padding_mask_p = keras.Input(shape=(512,), dtype="bool", name="padding_mask_p")

        token_ids_n = keras.Input(shape=(512,), dtype="int32", name="token_ids_n")
        padding_mask_n = keras.Input(shape=(512,), dtype="bool", name="padding_mask_n")

        anchor = {
            "token_ids": token_ids_a,
            "padding_mask": padding_mask_a,
        }
        positive = {
            "token_ids": token_ids_p,
            "padding_mask": padding_mask_p,
        }
        negative = {
            "token_ids": token_ids_n,
            "padding_mask": padding_mask_n,
        }

        a = encoder(anchor)
        p = encoder(positive)
        n = encoder(negative)

        positive_distance = EuclideanDistance()(a, p)
        negative_distance = EuclideanDistance()(a, n)

        output = StackLayer()(positive_distance, negative_distance)

        super().__init__(
            inputs={
                "token_ids_a": token_ids_a,
                "padding_mask_a": padding_mask_a,
                "token_ids_p": token_ids_p,
                "padding_mask_p": padding_mask_p,
                "token_ids_n": token_ids_n,
                "padding_mask_n": padding_mask_n,
            },
            outputs=output,
            **kwargs,
        )

        self.encoder = encoder

    def get_encoder(self):
        return self.encoder


"""
We will use a custom loss function for the triplet objective. The loss function will
receive the `positive_distance`, and the `negative_distance`, where they are stacked
together in `y_pred`.

We will use `positive_distance` and `negative_distance` to compute the loss such that
`negative_distance` is larger than `positive_distance` at least by a specific margin.
Mathematically, we will minimize this loss function: `max( positive_distance -
negative_distance + margin, 0)`.

There is no `y_true` used in this loss function. Note that we set the labels in the
dataset to zero, but they will not be used.
"""


class TripletLoss(keras.losses.Loss):
    def __init__(self, margin=1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        positive_dist, negative_dist = keras.ops.unstack(y_pred, axis=1)

        losses = keras.ops.relu(positive_dist - negative_dist + self.margin)
        return keras.ops.mean(losses, axis=0)


"""
#### Fit the model

Let's build and compile the model. We will first create the encoder by passing the RoBERTa
backbone and specifying `normalize = False`. Then, we will pass the encoder to the
siamese network and compile the Siamese network model. We will use `TripletLoss()` as
loss function, and `Adam()` optimizer with learning rate = 2e-5. We will enable [jit
compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation) for faster training.
"""

backbone = backbone = keras_nlp.models.RobertaBackbone.from_preset("roberta_base_en")
roberta_encoder = Encoder(backbone, False)

siamese_triplet_roberta = TripletSiamese(roberta_encoder)

siamese_triplet_roberta.compile(
    loss=TripletLoss(), optimizer=keras.optimizers.Adam(2e-5), jit_compile=True
)

siamese_triplet_roberta.summary()

"""
Let's fit the model!
"""

siamese_triplet_roberta.fit(
    wiki_train,
    epochs=1,
)

"""
Let's evaluate the model.
"""

siamese_triplet_roberta.evaluate(
    wiki_test,
)

"""
Let's try this model in a clustering example. Here are 6 questions. first 3 questions are
about learning English, and the last 3 questions are about working online. Let's see if
the embeddings produced by our encoder will cluster the questions correctly.
"""

questions = [
    "What should I do to improve my English writting?",
    "How to be good at speaking English?",
    "How can I improve my spoken English?",
    "How to earn money online?",
    "How do I earn money online?",
    "How to work and earn money through internet?",
]

encoder = siamese_triplet_roberta.get_encoder()

embeddings = encoder(preprocessor(questions))
if keras.backend.backend() == "torch":
    embeddings = embeddings.cpu().detach().numpy()

kmeans = cluster.KMeans(n_clusters=2, random_state=0, n_init="auto").fit(embeddings)

for i, label in enumerate(kmeans.labels_):
    print(f"sentence ({questions[i]}) belongs to cluster {label}")

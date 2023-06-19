"""
# Sentence embeddings using siamese BERT-networks

**Author:** [Mohammed Abu El-Nasr](https://github.com/abuelnasr0)<br>
**Date created:** 2023/05/27<br>
**Last modified:** 2023/05/27<br>
**Description:** fine-tune a BERT or RoBERTa model to generate sentence embeddings using
kerasNLP
"""

"""
## Introduction
"""

"""
BERT, and RoBERTa can be used for semantic textual similarity tasks, where two sentences
are passed to model and the network predicts whether they are similiar or not. but what
if we have a collection of large number of sentences, and we want to find the most
similiar pairs in that collection, that will take n*(n-1)/2 inference computations, where
n is the number of sentences in the collection. for example if n=10000, the required time
will be 65 hours on a V100 GPU.<br>
A common used method to overcome the time overhead issue, is to pass one sentence to the
model, then average the output of the model, or take the first token (the [CLS] token)
and use them as [sentence embedding](https://en.wikipedia.org/wiki/Sentence_embedding),
then use a vector similarity measure like cosine similarity or Manhatten / Euclidean
distance to find close sentences (semantically similair sentences). that will reduce the
time to find the most similiar pairs in a collection of 10,000 sentences from 65 hours to
5 seconds!<br>
If we use BERT or RoBERa directly that will yield rather bad sentence embeddings. but if
we fine-tune BERT using siamese network, that will generate simantically meaningful
sentence embeddings. this will enable BERT to be used for new tasks. these tasks include:
- Large-scale semantic similarity comparison
- Clustring
- Information retrival via semantic search
<br>
In this example, we will show how to fine-tune a BERT or RoBERTa model using siamese
network so they will be able to produce semantically meaningful sentence embeddings and
use them in semantic search and clustring example.<br>
This method of fine-tuning was introduced is sentence-bert
([SBERT](https://arxiv.org/abs/1908.10084))
"""

"""
## Setup
"""

"""shell
!pip install keras-nlp -q
"""

import keras_nlp
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import sklearn.cluster as cluster

policy = keras.mixed_precision.Policy("mixed_float16")
keras.mixed_precision.set_global_policy(policy)

"""
## Fine-tune the model using siamese networks
"""

"""
[Siamese network](https://en.wikipedia.org/wiki/Siamese_neural_network) is a neural
network architecture that contains two or more subnetworks. and the subnetworks share the
same weights. it is used to generate feature vectors for each input and then compare them
for simialrity.<br>
For our example, the subnetworks will be RoBERTa model that have a pooling layer on top
of it to produce the embeddings of the input sentences. this embeddings will then be
compared to each other to learn to produce semantically meaningful embeddings. <br>
the pooling stratigies used are mean, max, and cls pooling. mean pooling produces the
best results. we will use it in our examples.
"""

"""
### Fine-tune using Regression Objective Function
For building the siamese network with the regression objective function, the siamese
network is asked to predict the cosine similarity between the embedddings of the two
input sentences.<br>
cosine similarity indicates the angle between the sentence embeddings. if the cosine
similarity is high, that means there is a small angle between the embeddings, hence they
are semantically similiar. and vice verce.<br>
"""

"""
#### Load the dataset
We will use STSB dataset to fine-tune the model for the regression objective.<br>
STSB consists of a collection of sentence pairs that are labeled in the range [0, 5]. 0
indicates the least semantic similarity between two sentences, 5 indicates the most
semantic similarity between two sentences.<br>
the range of the cosine similarity is [-1,1] and it's the output of the siamese network,
but the labels in the dataset range is [0, 5]. we need to unify the range between the
cosine similarity and the dataset labels, so while preparing the dataset we will divide
the labels by 2.5 ans subtract 1.<br>
"""

TRAIN_BATCH_SIZE = 6
VALIDATION_BATCH_SIZE = 8

TRAIN_NUM_BATCHS = 300
VALIDATION_NUM_BATCHS = 40

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 4


def change_range(x):
    return (x / 2.5) - 1


def prepare_dataset(dataset, num_batchs, batch_size):
    dataset = dataset.map(
        lambda z: (
            [z["sentence1"], z["sentence2"]],
            [tf.cast(change_range(z["label"]), tf.float32)],
        ),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.take(num_batchs)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


stsb_ds = tfds.load(
    "glue/stsb",
)
stsb_train, stsb_valid = stsb_ds["train"], stsb_ds["validation"]

stsb_train = prepare_dataset(stsb_train, TRAIN_NUM_BATCHS, TRAIN_BATCH_SIZE)
stsb_valid = prepare_dataset(stsb_valid, VALIDATION_NUM_BATCHS, VALIDATION_BATCH_SIZE)

"""
Let's see examples from the dataset of two sentenses and their similarity.
"""

for x, y in stsb_train:
    for i, example in enumerate(x):
        print(f"sentence 1 : {example[0]} ")
        print(f"sentence 2 : {example[1]} ")
        print(f"similarity : {y[i]} \n")
    break

"""
####Build the encoder model
Now, we will build the encoder model that will produce the sentence embeddings and it
consists of:

- a preprocessor layer to tokenize and generate padding mask for the sentences.
- a backbone model that will generate the contextual representation of each token is the
sentence.
- a mean pooling layer to produce the embeddings.
- a normalization layer to normalize the embeddings as we are using the cosine
similarity.<br>
"""

# Mean pooling layer.
class mean_pooling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, last_state, padding_mask):
        # Only average the unpadded output.
        mask = tf.expand_dims(tf.cast(padding_mask, "float16"), axis=-1)
        count = tf.math.reduce_sum(mask, axis=1)
        sum = tf.math.reduce_sum(last_state * mask, axis=1)
        pooled = sum / count
        return pooled


preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset("roberta_base_en")
backbone = keras_nlp.models.RobertaBackbone.from_preset("roberta_base_en")
inputs = keras.Input(shape=(1), dtype="string", name="sentence")
x = preprocessor(inputs)
h = backbone(x)
embedding = mean_pooling(name="pooling_layer")(h, x["padding_mask"])
n_embedding = tf.linalg.normalize(embedding, axis=1)[0]
roberta_encoder = keras.Model(inputs=inputs, outputs=n_embedding)

roberta_encoder.summary()

"""
#### Build the siamese network with regrssion objective function
It's descriped above that the the siamese network has two or more subnetworks, and for
this siamese model we need two encoders. But we actually don't have two encoders, we have
only one encoder, but we will pass the two sentences through it.That way we can have two
paths to get the embeddings and also shared weights between the two paths.<br>
After passing the the two sentences to the model and getting the normalized embeddings we
will multiply the two normalized embeddings to get the cosine similarity between the two
sentences.<br>
"""


class regression_siamese(keras.Model):
    def __init__(self, encoder, **kwargs):
        inputs = keras.Input(shape=(2), dtype="string", name="sentences")
        sen1, sen2 = tf.split(inputs, num_or_size_splits=2, axis=1, name="split")
        u = encoder(sen1)
        v = encoder(sen2)
        cosine_similarity = tf.matmul(u, tf.transpose(v))

        super().__init__(
            inputs=inputs,
            outputs=cosine_similarity,
            **kwargs,
        )

        self.encoder = encoder

    def get_encoder(self):
        return self.encoder


"""
#### Fit the model to the data
"""

"""
Let's first try this example before training and compare it to the output after
training.<br>
"""

sentences = [
    "today is a very sunny day",
    "I am hungry, I will get my meal.",
    "the dog is eating his food.",
]
query = ["the dog is enjoying his meal."]

encoder = roberta_encoder

sentence_embeddings = encoder(tf.constant(sentences))
query_embedding = encoder(tf.constant(query))

cosine_simalarities = tf.matmul(query_embedding, tf.transpose(sentence_embeddings))
for i, sim in enumerate(cosine_simalarities[0]):
    print(f"cosine similarity between sentence {i+1} and the query = {sim} ")

"""
for the training we will use `MeanSquaredError()` as loss function, and `Adam()`
optimizer with learning rate = 2e-5.
"""

roberta_siamese = regression_siamese(roberta_encoder)
roberta_siamese.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(2e-5),
    metrics=[keras.metrics.MeanSquaredError()],
)

roberta_siamese.fit(stsb_train, validation_data=stsb_valid, epochs=1)

"""
Let's try the model after training, we will notice a huge differnce in the output. that
means that the model after fine-tuning is capabale of producing semantically meaningfull
embeddings. where the sematically similar sentences have small angle between them. and
semantically dissimilar sentences have large angle between them.
"""

sentences = [
    "today is a very sunny day",
    "I am hungry, I will get my meal.",
    "the dog is eating his food.",
]
query = ["the dog is enjoying his food."]

encoder = roberta_siamese.get_encoder()

sentence_embeddings = encoder(tf.constant(sentences))
query_embedding = encoder(tf.constant(query))

cosine_simalarities = tf.matmul(query_embedding, tf.transpose(sentence_embeddings))
for i, sim in enumerate(cosine_simalarities[0]):
    print(f"cosine similarity between sentence {i+1} and the query = {sim} ")

"""
### Fine-tune Using Triplet Objective Function
For the siamese network with the triplet objective function, three sentences are passed
to the siamese network *anchor*, *positive*, and *negative* sentences. *anchor* and
*positive* sentences are semantically similiar, and *anchor* and *negative* sentences are
semantically dissimilar.<br>the objective is to minimize the distance between the
*anchor* sentence and the *positive* sentence. and to maximize the distance between the
*anchor* sentence and the *negative* sentence.
"""

"""
#### Load the dataset
We will use wikipedia-sections-triplets dataset for fine-tuning. this data set consists
of senteces derived from wikipedia website. it has a collection of 3 sentences *anchor*,
*positive*, *negative*. *anchor* and *positive* are derived from the same section.
*anchor* and *negative* are derived from differnet sections.<br>
this dataset has 1.8m trainig triplets. and 220,000 test triplets. We will only use 1200
triplets for the training. and 300 for testing in this example.
"""

"""shell
!wget https://sbert.net/datasets/wikipedia-sections-triplets.zip -q
!unzip wikipedia-sections-triplets.zip  -d  wikipedia-sections-triplets
"""

NUM_TRAIN_BATCHS = 300
NUM_TEST_BATCHS = 75
AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 4


def prepare_wiki_data(dataset, num_batchs):
    dataset = dataset.map(
        lambda z: ((z["Sentence1"], z["Sentence2"], z["Sentence3"]), 0)
    )
    dataset = dataset.batch(6)
    dataset = dataset.take(num_batchs)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


wiki_train = tf.data.experimental.make_csv_dataset(
    "wikipedia-sections-triplets/train.csv",
    batch_size=1,
    num_epochs=1,
)
wiki_test = tf.data.experimental.make_csv_dataset(
    "wikipedia-sections-triplets/test.csv",
    batch_size=1,
    num_epochs=1,
)

wiki_train = prepare_wiki_data(wiki_train, NUM_TRAIN_BATCHS)
wiki_test = prepare_wiki_data(wiki_test, NUM_TEST_BATCHS)

"""
#### Build the encoder model
For this encoder model we will use RoBERTa alongside with mean pooling and we will not
normalize the output embeddings.
the encoder model that consists of:

- a preprocessor layer to tokenize and generate padding mask for the sentences.
- a backbone model that will generate the contextual representation of each token is the
sentence.
- a mean pooling layer to produce the embeddings.
<br>
"""

preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset("roberta_base_en")
backbone = keras_nlp.models.RobertaBackbone.from_preset("roberta_base_en")
input = keras.Input(shape=(1), dtype="string", name="sentence")

x = preprocessor(input)
h = backbone(x)
embedding = mean_pooling(name="pooling_layer")(h, x["padding_mask"])

roberta_triplet_encoder = keras.Model(inputs=input, outputs=embedding)

roberta_triplet_encoder.summary()

"""
#### Build the siamese network with triplet objective
for siamese triplet network, we will build the model with an encoder and we will pass the
three ssentences thtough that encoder. we will get an embedding for each sentence and we
will calculate the `positive_dist` and `negative_dist` that will be passed to the loss
function described below.
"""


class triplet_siamese(keras.Model):
    def __init__(self, encoder, **kwargs):

        anchor = keras.Input(shape=(1), dtype="string")
        positive = keras.Input(shape=(1), dtype="string")
        negative = keras.Input(shape=(1), dtype="string")

        ea = encoder(anchor)
        ep = encoder(positive)
        en = encoder(negative)

        positive_dist = tf.math.reduce_sum(tf.math.pow(ea - ep, 2), axis=1)
        negative_dist = tf.math.reduce_sum(tf.math.pow(ea - en, 2), axis=1)

        positive_dist = tf.math.sqrt(positive_dist)
        negative_dist = tf.math.sqrt(negative_dist)

        output = tf.stack([positive_dist, negative_dist], axis=0)

        super().__init__(inputs=[anchor, positive, negative], outputs=output, **kwargs)

        self.encoder = encoder

    def get_encoder(self):
        return self.encoder


"""
We will use a custom loss function for the triplet objective. the loss function will
recieve the distance between the *anchor* and the *positive* embeddings `positive_dist`,
and the distance between the *anchor* and the *negative* embeddings `negative_dist`,
where they are stacked together in `y_pred`.<br>
we will use `positive_dist` and `negative_dist` to compute the loss such that
`negative_dist` is larger than `positive_dist` at least by a specific margin.
mathimatically we will minimize this loss function: max( `positive_dist` -
`negative_dist` + margin, 0).<br>
and there is no `y_true` used in this loss function. note that we set the labels in the
data set as zero but it will not be used.
"""


class Triplet_loss(keras.losses.Loss):
    def __init__(self, margin=1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        positive_dist, negative_dist = tf.unstack(y_pred, axis=0)

        losses = tf.nn.relu(positive_dist - negative_dist + self.margin)
        return tf.math.reduce_mean(losses, axis=0)


# Metric for evaluation that have same logic as `Triplet_loss()` loss.
def metric(y_true, y_pred):
    margin = 1
    positive_dist, negative_dist = tf.unstack(y_pred, axis=0)
    losses = tf.nn.relu(positive_dist - negative_dist + margin)
    return tf.math.reduce_mean(losses, axis=0)


"""
#### Fit the model to the data
for the training we will use the custom `Triplet_loss()` as loss function, and `Adam()`
optimizer with learning rate = 2e-5.
"""

triplet_roberta = triplet_siamese(roberta_triplet_encoder)

triplet_roberta.compile(
    loss=Triplet_loss(),
    optimizer=keras.optimizers.Adam(2e-5),
)

triplet_roberta.fit(wiki_train, validation_data=wiki_test, epochs=1)

"""
Let's try this model in clustring example. here we have 6 questions. first 3 questions
about learning english, and last 3 questions about working online. let's see if the
embeddings produced by our encoder will cluster them right.
"""

questions = [
    "What should I do to improve my English writting?",
    "How to be good at speaking English?",
    "how can I improve my English?",
    "How to earn money online?",
    "How do I earn money online?",
    "how to work and ean money through internet?",
]

encoder = triplet_roberta.get_encoder()
embeddings = encoder(tf.constant(questions))
kmeans = cluster.KMeans(n_clusters=2, random_state=0, n_init="auto").fit(embeddings)

for i, label in enumerate(kmeans.labels_):
    print(f"sentence ({questions[i]}) belongs to cluster {label}")

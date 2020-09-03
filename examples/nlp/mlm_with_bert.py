"""
Title: Masked Language Modeling with BERT
Author: [Ankur Singh](https://twitter.com/ankur310794)
Date created: 2020/09/03
Last modified: 2020/09/03
Description: Implement a Masked Language Modeling with BERT and train on TPU
"""
"""
## Introduction

Masked language modeling is a fill-in-the-blank task, where a model uses the context words surrounding a [MASK] token to try to predict what the [MASK] word should be.
"""

"""
## Setup

Install HuggingFace transformers via pip install transformers (version >= 3.1.0).
"""


from dataclasses import dataclass
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import TFAutoModelWithLMHead, AutoTokenizer
from transformers import pipeline
from pprint import pprint


"""
## Set-up Configuration
"""


@dataclass
class Config:
    MAX_LEN = 128
    BATCH_SIZE = 16  # per TPU core
    TOTAL_STEPS = 2000  # thats approx 4 epochs
    EVALUATE_EVERY = 200
    LR = 1e-5
    PRETRAINED_MODEL = "bert-base-uncased"


flags = Config()
AUTO = tf.data.experimental.AUTOTUNE


"""
## Set-up TPU Runtime
"""


def connect_to_TPU():
    """Detect hardware, return appropriate distribution strategy"""
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU ", tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    global_batch_size = flags.BATCH_SIZE * strategy.num_replicas_in_sync

    return tpu, strategy, global_batch_size


tpu, strategy, global_batch_size = connect_to_TPU()
print("REPLICAS: ", strategy.num_replicas_in_sync)


"""
## Prepare Masked Language Dataset
"""


class Dataset:
    def __init__(self, tokenizer, strategy):

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.strategy = strategy

    def get_processed_dataset(self, texts, maxlen):

        # encode
        X_data = self.encode(texts, maxlen=maxlen)

        X_train_mlm, y_train_mlm = self.prepare_mlm_input_and_labels(X_data)

        train_dist_dataset = self.create_dist_dataset(X_train_mlm, y_train_mlm, True)

        return train_dist_dataset

    def encode(self, texts, maxlen=512):
        enc_di = self.tokenizer.batch_encode_plus(
            texts,
            return_attention_mask=False,
            return_token_type_ids=False,
            pad_to_max_length=True,
            max_length=maxlen,
            truncation=True,
        )

        return np.array(enc_di["input_ids"])

    def prepare_mlm_input_and_labels(self, X):

        # 15% BERT masking
        inp_mask = np.random.rand(*X.shape) < 0.15
        # do not mask special tokens
        inp_mask[X <= 2] = False
        # set targets to -1 by default, it means ignore
        labels = -1 * np.ones(X.shape, dtype=int)
        # set labels for masked tokens
        labels[inp_mask] = X[inp_mask]

        # prepare input
        X_mlm = np.copy(X)
        # set input to [MASK] which is the last token for the 90% of tokens
        # this means leaving 10% unchanged
        inp_mask_2mask = inp_mask & (np.random.rand(*X.shape) < 0.90)
        X_mlm[
            inp_mask_2mask
        ] = self.tokenizer.mask_token_id  # mask token is the last in the dict

        # set 10% to a random token
        inp_mask_2random = inp_mask_2mask & (np.random.rand(*X.shape) < 1 / 9)
        X_mlm[inp_mask_2random] = np.random.randint(
            3, self.tokenizer.mask_token_id, inp_mask_2random.sum()
        )

        return X_mlm, labels

    def create_dist_dataset(self, X, y=None, training=False):

        dataset = tf.data.Dataset.from_tensor_slices(X)

        ### Add y if present ###
        if y is not None:
            dataset_y = tf.data.Dataset.from_tensor_slices(y)
            dataset = tf.data.Dataset.zip((dataset, dataset_y))

        ### Repeat if training ###
        if training:
            dataset = dataset.shuffle(len(X)).repeat()

        dataset = dataset.batch(global_batch_size).prefetch(AUTO)

        ### make it distributed  ###
        dist_dataset = self.strategy.experimental_distribute_dataset(dataset)

        return dist_dataset


"""shell
wget https://raw.githubusercontent.com/SrinidhiRaghavan/AI-Sentiment-Analysis-on-IMDB-Dataset/master/imdb_tr.csv
"""

data = pd.read_csv("imdb_tr.csv", encoding="ISO-8859-1")
tokenizer = AutoTokenizer.from_pretrained(flags.PRETRAINED_MODEL)
dataset = Dataset(tokenizer, strategy)
texts = data.text.values
train_dist_dataset = dataset.get_processed_dataset(texts, flags.MAX_LEN)


"""
## Create MaskedLanguageModel using huggingface transformers
"""


class MaskedLanguageModel:
    def __init__(self, strategy, model_name):

        self.strategy = strategy
        with self.strategy.scope():
            self.model = TFAutoModelWithLMHead.from_pretrained(model_name)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=flags.LR)

        (
            self.compute_mlm_loss,
            self.train_mlm_loss_metric,
        ) = self.get_mlm_loss_and_metrics()

    def get_mlm_loss_and_metrics(self):

        with self.strategy.scope():
            mlm_loss_object = self.masked_sparse_categorical_crossentropy

            def compute_mlm_loss(labels, predictions):
                per_example_loss = mlm_loss_object(labels, predictions)
                loss = tf.nn.compute_average_loss(
                    per_example_loss, global_batch_size=global_batch_size
                )
                return loss

            train_mlm_loss_metric = tf.keras.metrics.Mean()

        return compute_mlm_loss, train_mlm_loss_metric

    def masked_sparse_categorical_crossentropy(self, y_true, y_pred):
        y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
        y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_masked, y_pred_masked, from_logits=True
        )
        return loss

    def train_mlm(self, train_dist_dataset, total_steps=2000, evaluate_every=200):
        step = 0
        ### Training lopp ###
        for batch in train_dist_dataset:
            self.distributed_mlm_train_step(batch)
            step += 1

            if step % evaluate_every == 0:
                ### Print train metrics ###
                train_metric = self.train_mlm_loss_metric.result().numpy()
                print("Step %d, train loss: %.2f" % (step, train_metric))

                ### Reset  metrics ###
                self.train_mlm_loss_metric.reset_states()

            if step == total_steps:
                break

    @tf.function
    def distributed_mlm_train_step(self, data):
        strategy.experimental_run_v2(self.mlm_train_step, args=(data,))

    @tf.function
    def mlm_train_step(self, inputs):
        features, labels = inputs

        with tf.GradientTape() as tape:
            predictions = self.model(features, training=True)[0]
            loss = self.compute_mlm_loss(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_mlm_loss_metric.update_state(loss)


"""
## Train and Save
"""

mlm_model = MaskedLanguageModel(strategy, flags.PRETRAINED_MODEL)
mlm_model.train_mlm(train_dist_dataset, flags.TOTAL_STEPS, flags.EVALUATE_EVERY)
mlm_model.model.save_pretrained("imdb_bert_uncased")

"""
## Load and Test
"""

imdb_bert_model = TFAutoModelWithLMHead.from_pretrained("imdb_bert_uncased")
nlp = pipeline("fill-mask", model=imdb_bert_model, tokenizer=tokenizer, framework="tf")
pprint(nlp(f"I watched {nlp.tokenizer.mask_token} and that was awesome"))

"""
Title: Semantic Similarity with BERT
Author: [Mohamad Merchant](https://twitter.com/mohmadmerchant1)
Date created: 2020/08/15
Last modified: 2020/08/21
Description: Natural Language Inference by Fine-tuning BERT model on SNLI Corpus.
"""
"""
## Introduction

Semantic Similarity is the task of determining how similar
two sentences are, in terms of what they mean.
This example demonstrates the use of SNLI (Standford Natural Language Inference) Corpus
to predict sentence semantic similarity with Transformers.
We will fine-tune a BERT model that takes two sentences as inputs
and that outputs a similarity score for these two sentences.

### References

* [BERT](https://arxiv.org/pdf/1810.04805.pdf)
* [SNLI](https://nlp.stanford.edu/projects/snli/)
"""

"""
## Setup

Note: install HuggingFace `transformers` via `!pip install transformers==2.11.0`.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

"""
## Configurations
"""

max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 2
learning_rate = 3e-5
# Labels in our dataset.
labels = ["contradiction", "entailment", "neutral"]

"""
## Load the Data
"""

"""shell
curl -LO https://raw.githubusercontent.com/MohamadMerchant/SNLI/master/data.tar.gz
tar -xvzf data.tar.gz
"""
# There are more than 550k samples in total; we will use 100k for this example.
train_df = pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv")
test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv")

# Shape of the data
print(f"Total train samples : {train_df.shape[0]}")
print(f"Total validation samples: {valid_df.shape[0]}")
print(f"Total test samples: {valid_df.shape[0]}")

"""
Dataset Overview:

- sentence1: The premise caption that was supplied to the author of the pair.
- sentence2: The hypothesis caption that was written by the author of the pair.
- similarity: This is the label chosen by the majority of annotators.
Where no majority exists, the label "-" is used (we will skip such samples here).

Here are the "similarity" label values in our dataset:

- Contradiction: The sentences share no similarity.
- Entailment: The sentences have similar meaning.
- Neutral: The sentences are neutral.
"""

"""
Let's look at one sample from the dataset:
"""
print(f"Sentence1: {train_df.loc[1, 'sentence1']}")
print(f"Sentence2: {train_df.loc[1, 'sentence2']}")
print(f"Similarity: {train_df.loc[1, 'similarity']}")

"""
## Preprocessing
"""

# We have some NaN entries in our train data, we will simply drop them.
print("Number of Missing Values")
print(train_df.isnull().sum())
train_df.dropna(axis=0, inplace=True)

"""
Distribution of our training targets.
"""
print("Train Target Distribution")
print(train_df.similarity.value_counts())

"""
Distribution of our validation targets.
"""
print("Validation Target Distribution")
print(valid_df.similarity.value_counts())

"""
The value "-" appears as part of our training and validation targets.
We will skip these samples.
"""
train_df = (
    train_df[train_df.similarity != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)
valid_df = (
    valid_df[valid_df.similarity != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

"""
One-hot encode training, validation, and test labels.
"""
train_df["label"] = train_df["similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)

valid_df["label"] = valid_df["similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=3)

test_df["label"] = test_df["similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=3)

"""
## Keras Custom Data Generator
"""


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence1: Array of premise input sentences.
        sentence2: Array of the hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence1,
        sentence2,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence1) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence1 = self.sentence1[indexes]
        sentence2 = self.sentence2[indexes]
        batch_input_ids = []
        batch_attention_masks = []
        batch_token_type_ids = []
        # With BERT tokenizer's encode plus both the sentences are
        # encoded together and separated by [SEP] token.
        # Here we are encoding batch of sentences together.
        for s1, s2 in zip(sentence1, sentence2):
            encoded = self.tokenizer.encode_plus(
                s1,
                s2,
                add_special_tokens=True,
                max_length=max_length,
                return_attention_mask=True,
                return_token_type_ids=True,
                padding=True,
                pad_to_max_length=True,
                return_tensors="tf",
            )
            batch_input_ids.extend(encoded["input_ids"])
            batch_attention_masks.extend(encoded["attention_mask"])
            batch_token_type_ids.extend(encoded["token_type_ids"])

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(batch_input_ids, dtype="int32")
        masks = np.array(batch_attention_masks, dtype="int32")
        token_type_ids = np.array(batch_token_type_ids, dtype="int32")

        # set to true if data generator is used for training/validation
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, masks, token_type_ids], labels
        else:
            return [input_ids, masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        self.indexes = np.arange(len(self.sentence1))
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


"""
## Build The Model
"""


def build_model():
    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="att_mask"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    # Loading pretrained BERT model.
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    sequence_output, pooled_output = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    # Applying hybrid pooling approach to bert sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(sequence_output)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(3, activation="softmax")(dropout)

    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )
    return model


"""
Create the model under a distribution strategy scope.
"""
# Build model with distributed strategy.
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()

print(f"Strategy: {strategy}")
model.summary()

"""
Create train and validation data generators
"""
train_data = BertSemanticDataGenerator(
    train_df.sentence1.astype("str"),
    train_df.sentence2.astype("str"),
    y_train,
    batch_size=batch_size,
    shuffle=True,
)
valid_data = BertSemanticDataGenerator(
    valid_df.sentence1.astype("str"),
    valid_df.sentence2.astype("str"),
    y_val,
    batch_size=batch_size,
    shuffle=False,
)

"""
## Train the Model
"""
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)
"""
## Evaluate model on the test set
"""
test_data = BertSemanticDataGenerator(
    test_df.sentence1.astype("str"),
    test_df.sentence2.astype("str"),
    y_test,
    batch_size=batch_size,
    shuffle=False,
)
model.evaluate(test_data, verbose=1)

"""
## Inference on custom sentences
"""


def check_similarity(sentence1, sentence2):
    sentence1 = np.array([str(sentence1)])
    sentence2 = np.array([str(sentence2)])
    test_data = BertSemanticDataGenerator(
        sentence1,
        sentence2,
        labels=None,
        batch_size=1,
        shuffle=False,
        include_targets=False,
    )

    proba = model.predict(test_data)[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return pred, proba


"""
Check results on some example sentence pairs.
"""
sentence1 = "The man is sleeping"
sentence2 = "A man inspects the uniform"
print(check_similarity(sentence1, sentence2))

"""
Check results on some example sentence pairs.
"""
sentence1 = "A smiling costumed woman is holding an umbrella"
sentence2 = "A happy woman in a fairy costume holds an umbrella"
print(check_similarity(sentence1, sentence2))

"""
Check results on some example sentence pairs
"""
sentence1 = "A soccer game with multiple males playing"
sentence2 = "Some men are playing a sport"
print(check_similarity(sentence1, sentence2))

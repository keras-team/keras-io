# End-to-end Masked Language Modeling with BERT

**Author:** [Ankur Singh](https://twitter.com/ankur310794)<br>
**Date created:** 2020/09/18<br>
**Last modified:** 2020/09/18<br>


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/masked_language_modeling.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/masked_language_modeling.py)


**Description:** Implement a Masked Language Model (MLM) with BERT and fine-tune it on the IMDB Reviews dataset.

---
## Introduction

Masked Language Modeling is a fill-in-the-blank task,
where a model uses the context words surrounding a mask token to try to predict what the
masked word should be.

For an input that contains one or more mask tokens,
the model will generate the most likely substitution for each.

Example:

- Input: "I have watched this [MASK] and it was awesome."
- Output: "I have watched this movie and it was awesome."

Masked language modeling is a great way to train a language
model in a self-supervised setting (without human-annotated labels).
Such a model can then be fine-tuned to accomplish various supervised
NLP tasks.

This example teaches you how to build a BERT model from scratch,
train it with the masked language modeling task,
and then fine-tune this model on a sentiment classification task.

We will use the Keras `TextVectorization` and `MultiHeadAttention` layers
to create a BERT Transformer-Encoder network architecture.

Note: This example should be run with `tf-nightly`.

---
## Setup

Install `tf-nightly` via `pip install tf-nightly`.


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from dataclasses import dataclass
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint
```

---
## Set-up Configuration


```python

@dataclass
class Config:
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1


config = Config()
```

---
## Load the data

We will first download the IMDB data and load into a Pandas dataframe.


```python
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```

```python

def get_text_list_from_files(files):
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list


def get_data_from_text_files(folder_name):

    pos_files = glob.glob("aclImdb/" + folder_name + "/pos/*.txt")
    pos_texts = get_text_list_from_files(pos_files)
    neg_files = glob.glob("aclImdb/" + folder_name + "/neg/*.txt")
    neg_texts = get_text_list_from_files(neg_files)
    df = pd.DataFrame(
        {
            "review": pos_texts + neg_texts,
            "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
        }
    )
    df = df.sample(len(df)).reset_index(drop=True)
    return df


train_df = get_data_from_text_files("train")
test_df = get_data_from_text_files("test")

all_data = train_df.append(test_df)
```
<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 80.2M  100 80.2M    0     0  45.3M      0  0:00:01  0:00:01 --:--:-- 45.3M

```
</div>
---
## Dataset preparation

We will use the `TextVectorization` layer to vectorize the text into integer token ids.
It transforms a batch of strings into either
a sequence of token indices (one sample = 1D array of integer token indices, in order)
or a dense representation (one sample = 1D array of float values encoding an unordered set of tokens).

Below, we define 3 preprocessing functions.

1.  The `get_vectorize_layer` function builds the `TextVectorization` layer.
2.  The `encode` function encodes raw text into integer token ids.
3.  The `get_masked_input_and_labels` function will mask input token ids.
It masks 15% of all input tokens in each sequence at random.


```python

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


def get_vectorize_layer(texts, vocab_size, max_seq, special_tokens=["[MASK]"]):
    """Build Text vectorization layer

    Args:
      texts (list): List of string i.e input texts
      vocab_size (int): vocab size
      max_seq (int): Maximum sequence lenght.
      special_tokens (list, optional): List of special tokens. Defaults to ['[MASK]'].

    Returns:
        layers.Layer: Return TextVectorization Keras Layer
    """
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        standardize=custom_standardization,
        output_sequence_length=max_seq,
    )
    vectorize_layer.adapt(texts)

    # Insert mask token in vocabulary
    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[2 : vocab_size - len(special_tokens)] + ["[mask]"]
    vectorize_layer.set_vocabulary(vocab)
    return vectorize_layer


vectorize_layer = get_vectorize_layer(
    all_data.review.values.tolist(),
    config.VOCAB_SIZE,
    config.MAX_LEN,
    special_tokens=["[mask]"],
)

# Get mask token id for masked language model
mask_token_id = vectorize_layer(["[mask]"]).numpy()[0][0]


def encode(texts):
    encoded_texts = vectorize_layer(texts)
    return encoded_texts.numpy()


def get_masked_input_and_labels(encoded_texts):
    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 2] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[
        inp_mask_2mask
    ] = mask_token_id  # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights


# We have 25000 examples for training
x_train = encode(train_df.review.values)  # encode reviews with vectorizer
y_train = train_df.sentiment.values
train_classifier_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1000)
    .batch(config.BATCH_SIZE)
)

# We have 25000 examples for testing
x_test = encode(test_df.review.values)
y_test = test_df.sentiment.values
test_classifier_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
    config.BATCH_SIZE
)

# Build dataset for end to end model input (will be used at the end)
test_raw_classifier_ds = tf.data.Dataset.from_tensor_slices(
    (test_df.review.values, y_test)
).batch(config.BATCH_SIZE)

# Prepare data for masked language model
x_all_review = encode(all_data.review.values)
x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(
    x_all_review
)

mlm_ds = tf.data.Dataset.from_tensor_slices(
    (x_masked_train, y_masked_labels, sample_weights)
)
mlm_ds = mlm_ds.shuffle(1000).batch(config.BATCH_SIZE)
```

---
## Create BERT model (Pretraining Model) for masked language modeling

We will create a BERT-like pretraining model architecture
using the `MultiHeadAttention` layer.
It will take token ids as inputs (including masked tokens)
and it will predict the correct ids for the masked input tokens.


```python

def bert_module(query, key, value, i):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


loss_fn = keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)
loss_tracker = tf.keras.metrics.Mean(name="loss")


class MaskedLanguageModel(tf.keras.Model):
    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]


def create_masked_language_bert_model():
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)

    word_embeddings = layers.Embedding(
        config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
    )(inputs)
    position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(
        encoder_output
    )
    mlm_model = MaskedLanguageModel(inputs, mlm_output, name="masked_bert_model")

    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    mlm_model.compile(optimizer=optimizer)
    return mlm_model


id2token = dict(enumerate(vectorize_layer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}


class MaskedTextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5):
        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens):
        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(sample_tokens[0].numpy()),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)


sample_tokens = vectorize_layer(["I have watched this [mask] and it was awesome"])
generator_callback = MaskedTextGenerator(sample_tokens.numpy())

bert_masked_model = create_masked_language_bert_model()
bert_masked_model.summary()
```

<div class="k-default-codeblock">
```
Model: "masked_bert_model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 256)]        0                                            
__________________________________________________________________________________________________
word_embedding (Embedding)      (None, 256, 128)     3840000     input_1[0][0]                    
__________________________________________________________________________________________________
tf.__operators__.add (TFOpLambd (None, 256, 128)     0           word_embedding[0][0]             
__________________________________________________________________________________________________
encoder_0/multiheadattention (M (None, 256, 128)     66048       tf.__operators__.add[0][0]       
                                                                 tf.__operators__.add[0][0]       
                                                                 tf.__operators__.add[0][0]       
__________________________________________________________________________________________________
encoder_0/att_dropout (Dropout) (None, 256, 128)     0           encoder_0/multiheadattention[0][0
__________________________________________________________________________________________________
tf.__operators__.add_1 (TFOpLam (None, 256, 128)     0           tf.__operators__.add[0][0]       
                                                                 encoder_0/att_dropout[0][0]      
__________________________________________________________________________________________________
encoder_0/att_layernormalizatio (None, 256, 128)     256         tf.__operators__.add_1[0][0]     
__________________________________________________________________________________________________
encoder_0/ffn (Sequential)      (None, 256, 128)     33024       encoder_0/att_layernormalization[
__________________________________________________________________________________________________
encoder_0/ffn_dropout (Dropout) (None, 256, 128)     0           encoder_0/ffn[0][0]              
__________________________________________________________________________________________________
tf.__operators__.add_2 (TFOpLam (None, 256, 128)     0           encoder_0/att_layernormalization[
                                                                 encoder_0/ffn_dropout[0][0]      
__________________________________________________________________________________________________
encoder_0/ffn_layernormalizatio (None, 256, 128)     256         tf.__operators__.add_2[0][0]     
__________________________________________________________________________________________________
mlm_cls (Dense)                 (None, 256, 30000)   3870000     encoder_0/ffn_layernormalization[
==================================================================================================
Total params: 7,809,584
Trainable params: 7,809,584
Non-trainable params: 0
__________________________________________________________________________________________________

```
</div>
---
## Train and Save


```python
bert_masked_model.fit(mlm_ds, epochs=5, callbacks=[generator_callback])
bert_masked_model.save("bert_mlm_imdb.h5")
```

<div class="k-default-codeblock">
```
Epoch 1/5
1563/1563 [==============================] - ETA: 0s - loss: 7.0111{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'this',
 'prediction': 'i have watched this this and it was awesome',
 'probability': 0.086307295}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'i',
 'prediction': 'i have watched this i and it was awesome',
 'probability': 0.066265985}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'movie',
 'prediction': 'i have watched this movie and it was awesome',
 'probability': 0.044195656}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'a',
 'prediction': 'i have watched this a and it was awesome',
 'probability': 0.04020928}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'was',
 'prediction': 'i have watched this was and it was awesome',
 'probability': 0.027878676}
1563/1563 [==============================] - 661s 423ms/step - loss: 7.0111
Epoch 2/5
1563/1563 [==============================] - ETA: 0s - loss: 6.4498{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'movie',
 'prediction': 'i have watched this movie and it was awesome',
 'probability': 0.44448906}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'film',
 'prediction': 'i have watched this film and it was awesome',
 'probability': 0.1507494}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'is',
 'prediction': 'i have watched this is and it was awesome',
 'probability': 0.06385628}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'one',
 'prediction': 'i have watched this one and it was awesome',
 'probability': 0.023549262}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'was',
 'prediction': 'i have watched this was and it was awesome',
 'probability': 0.022277055}
1563/1563 [==============================] - 660s 422ms/step - loss: 6.4498
Epoch 3/5
1563/1563 [==============================] - ETA: 0s - loss: 5.8709{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'movie',
 'prediction': 'i have watched this movie and it was awesome',
 'probability': 0.4759983}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'film',
 'prediction': 'i have watched this film and it was awesome',
 'probability': 0.18642229}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'one',
 'prediction': 'i have watched this one and it was awesome',
 'probability': 0.045611132}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'is',
 'prediction': 'i have watched this is and it was awesome',
 'probability': 0.028308254}
{'input_text': 'i have watched this [mask] and it was awesome',
 'predicted mask token': 'series',
 'prediction': 'i have watched this series and it was awesome',
 'probability': 0.027862877}
1563/1563 [==============================] - 661s 423ms/step - loss: 5.8709
Epoch 4/5
 771/1563 [=============>................] - ETA: 5:35 - loss: 5.3782

```
</div>
---
## Fine-tune a sentiment classification model

We will fine-tune our self-supervised model on a downstream task of sentiment classification.
To do this, let's create a classifier by adding a pooling layer and a `Dense` layer on top of the
pretrained BERT features.


```python
# Load pretrained bert model
mlm_model = keras.models.load_model(
    "bert_mlm_imdb.h5", custom_objects={"MaskedLanguageModel": MaskedLanguageModel}
)
pretrained_bert_model = tf.keras.Model(
    mlm_model.input, mlm_model.get_layer("encoder_0/ffn_layernormalization").output
)

# Freeze it
pretrained_bert_model.trainable = False


def create_classifier_bert_model():
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)
    sequence_output = pretrained_bert_model(inputs)
    pooled_output = layers.GlobalMaxPooling1D()(sequence_output)
    hidden_layer = layers.Dense(64, activation="relu")(pooled_output)
    outputs = layers.Dense(1, activation="sigmoid")(hidden_layer)
    classifer_model = keras.Model(inputs, outputs, name="classification")
    optimizer = keras.optimizers.Adam()
    classifer_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return classifer_model


classifer_model = create_classifier_bert_model()
classifer_model.summary()

# Train the classifier with frozen BERT stage
classifer_model.fit(
    train_classifier_ds,
    epochs=5,
    validation_data=test_classifier_ds,
)

# Unfreeze the BERT model for fine-tuning
pretrained_bert_model.trainable = True
optimizer = keras.optimizers.Adam()
classifer_model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)
classifer_model.fit(
    train_classifier_ds,
    epochs=5,
    validation_data=test_classifier_ds,
)
```

<div class="k-default-codeblock">
```
Model: "classification"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 256)]             0         
_________________________________________________________________
model (Functional)           (None, 256, 128)          3939584   
_________________________________________________________________
global_max_pooling1d (Global (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 65        
=================================================================
Total params: 3,947,905
Trainable params: 8,321
Non-trainable params: 3,939,584
_________________________________________________________________
Epoch 1/5
782/782 [==============================] - 15s 19ms/step - loss: 0.8096 - accuracy: 0.5498 - val_loss: 0.6406 - val_accuracy: 0.6329
Epoch 2/5
782/782 [==============================] - 14s 18ms/step - loss: 0.6551 - accuracy: 0.6220 - val_loss: 0.6423 - val_accuracy: 0.6338
Epoch 3/5
782/782 [==============================] - 14s 18ms/step - loss: 0.6473 - accuracy: 0.6310 - val_loss: 0.6380 - val_accuracy: 0.6350
Epoch 4/5
782/782 [==============================] - 14s 18ms/step - loss: 0.6307 - accuracy: 0.6471 - val_loss: 0.6432 - val_accuracy: 0.6312
Epoch 5/5
782/782 [==============================] - 14s 18ms/step - loss: 0.6278 - accuracy: 0.6465 - val_loss: 0.6107 - val_accuracy: 0.6678
Epoch 1/5
782/782 [==============================] - 46s 59ms/step - loss: 0.5234 - accuracy: 0.7373 - val_loss: 0.3533 - val_accuracy: 0.8427
Epoch 2/5
782/782 [==============================] - 45s 57ms/step - loss: 0.2808 - accuracy: 0.8814 - val_loss: 0.3252 - val_accuracy: 0.8633
Epoch 3/5
782/782 [==============================] - 43s 55ms/step - loss: 0.1493 - accuracy: 0.9413 - val_loss: 0.4374 - val_accuracy: 0.8486
Epoch 4/5
782/782 [==============================] - 43s 55ms/step - loss: 0.0600 - accuracy: 0.9803 - val_loss: 0.6422 - val_accuracy: 0.8380
Epoch 5/5
782/782 [==============================] - 43s 55ms/step - loss: 0.0305 - accuracy: 0.9893 - val_loss: 0.6064 - val_accuracy: 0.8440

<tensorflow.python.keras.callbacks.History at 0x7f35af4367f0>

```
</div>
---
## Create an end-to-end model and evaluate it

When you want to deploy a model, it's best if it already includes its preprocessing
pipeline, so that you don't have to reimplement the preprocessing logic in your
production environment. Let's create an end-to-end model that incorporates
the `TextVectorization` layer, and let's evaluate. Our model will accept raw strings
as input.


```python

def get_end_to_end(model):
    inputs_string = keras.Input(shape=(1,), dtype="string")
    indices = vectorize_layer(inputs_string)
    outputs = model(indices)
    end_to_end_model = keras.Model(inputs_string, outputs, name="end_to_end_model")
    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    end_to_end_model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return end_to_end_model


end_to_end_classification_model = get_end_to_end(classifer_model)
end_to_end_classification_model.evaluate(test_raw_classifier_ds)
```

<div class="k-default-codeblock">
```
782/782 [==============================] - 8s 11ms/step - loss: 0.5967 - accuracy: 0.8446

[0.6064175963401794, 0.8439599871635437]

```
</div>
# Semantic Similarity with BERT

**Author:** [Mohamad Merchant](https://twitter.com/mohmadmerchant1)<br>
**Date created:** 2020/08/15<br>
**Last modified:** 2020/08/29<br>
**Description:** Natural Language Inference by fine-tuning BERT model on SNLI Corpus.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/semantic_similarity_with_bert.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/semantic_similarity_with_bert.py)



---
## Introduction

Semantic Similarity is the task of determining how similar
two sentences are, in terms of what they mean.
This example demonstrates the use of SNLI (Stanford Natural Language Inference) Corpus
to predict sentence semantic similarity with Transformers.
We will fine-tune a BERT model that takes two sentences as inputs
and that outputs a similarity score for these two sentences.

### References

* [BERT](https://arxiv.org/pdf/1810.04805.pdf)
* [SNLI](https://nlp.stanford.edu/projects/snli/)

---
## Setup

Note: install HuggingFace `transformers` via `pip install transformers` (version >= 2.11.0).


```python
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
```

---
## Configuration


```python
max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 2

# Labels in our dataset.
labels = ["contradiction", "entailment", "neutral"]
```

---
## Load the Data


```python
!curl -LO https://raw.githubusercontent.com/MohamadMerchant/SNLI/master/data.tar.gz
!tar -xvzf data.tar.gz
```

```python
# There are more than 550k samples in total; we will use 100k for this example.
train_df = pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", nrows=100000)
valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv")
test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv")

# Shape of the data
print(f"Total train samples : {train_df.shape[0]}")
print(f"Total validation samples: {valid_df.shape[0]}")
print(f"Total test samples: {valid_df.shape[0]}")
```
<div class="k-default-codeblock">
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 11.1M  100 11.1M    0     0  5231k      0  0:00:02  0:00:02 --:--:-- 5231k
SNLI_Corpus/
SNLI_Corpus/snli_1.0_dev.csv
SNLI_Corpus/snli_1.0_train.csv
SNLI_Corpus/snli_1.0_test.csv

Total train samples : 100000
Total validation samples: 10000
Total test samples: 10000

```
</div>
Dataset Overview:

- sentence1: The premise caption that was supplied to the author of the pair.
- sentence2: The hypothesis caption that was written by the author of the pair.
- similarity: This is the label chosen by the majority of annotators.
Where no majority exists, the label "-" is used (we will skip such samples here).

Here are the "similarity" label values in our dataset:

- Contradiction: The sentences share no similarity.
- Entailment: The sentences have similar meaning.
- Neutral: The sentences are neutral.

Let's look at one sample from the dataset:


```python
print(f"Sentence1: {train_df.loc[1, 'sentence1']}")
print(f"Sentence2: {train_df.loc[1, 'sentence2']}")
print(f"Similarity: {train_df.loc[1, 'similarity']}")
```

<div class="k-default-codeblock">
```
Sentence1: A person on a horse jumps over a broken down airplane.
Sentence2: A person is at a diner, ordering an omelette.
Similarity: contradiction

```
</div>
---
## Preprocessing


```python
# We have some NaN entries in our train data, we will simply drop them.
print("Number of missing values")
print(train_df.isnull().sum())
train_df.dropna(axis=0, inplace=True)
```

<div class="k-default-codeblock">
```
Number of missing values
similarity    0
sentence1     0
sentence2     3
dtype: int64

```
</div>
Distribution of our training targets.


```python
print("Train Target Distribution")
print(train_df.similarity.value_counts())
```

<div class="k-default-codeblock">
```
Train Target Distribution
entailment       33384
contradiction    33310
neutral          33193
-                  110
Name: similarity, dtype: int64

```
</div>
Distribution of our validation targets.


```python
print("Validation Target Distribution")
print(valid_df.similarity.value_counts())
```

<div class="k-default-codeblock">
```
Validation Target Distribution
entailment       3329
contradiction    3278
neutral          3235
-                 158
Name: similarity, dtype: int64

```
</div>
The value "-" appears as part of our training and validation targets.
We will skip these samples.


```python
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
```

One-hot encode training, validation, and test labels.


```python
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
```

---
## Create a custom data generator


```python

class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
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
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)

```

---
## Build the model


```python
# Create the model under a distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    # Loading pretrained BERT model.
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    bert_output = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    sequence_output = bert_output.last_hidden_state
    pooled_output = bert_output.pooler_output
    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(sequence_output)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )


print(f"Strategy: {strategy}")
model.summary()
```


<div class="k-default-codeblock">
```
HBox(children=(FloatProgress(value=0.0, description='Downloading', max=433.0, style=ProgressStyle(description_…

```
</div>
    



<div class="k-default-codeblock">
```
HBox(children=(FloatProgress(value=0.0, description='Downloading', max=536063208.0, style=ProgressStyle(descri…

```
</div>
    
<div class="k-default-codeblock">
```
Strategy: <tensorflow.python.distribute.mirrored_strategy.MirroredStrategy object at 0x7faf9dc63a90>
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, 128)]        0                                            
__________________________________________________________________________________________________
attention_masks (InputLayer)    [(None, 128)]        0                                            
__________________________________________________________________________________________________
token_type_ids (InputLayer)     [(None, 128)]        0                                            
__________________________________________________________________________________________________
tf_bert_model (TFBertModel)     ((None, 128, 768), ( 109482240   input_ids[0][0]                  
                                                                 attention_masks[0][0]            
                                                                 token_type_ids[0][0]             
__________________________________________________________________________________________________
bidirectional (Bidirectional)   (None, 128, 128)     426496      tf_bert_model[0][0]              
__________________________________________________________________________________________________
global_average_pooling1d (Globa (None, 128)          0           bidirectional[0][0]              
__________________________________________________________________________________________________
global_max_pooling1d (GlobalMax (None, 128)          0           bidirectional[0][0]              
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 256)          0           global_average_pooling1d[0][0]   
                                                                 global_max_pooling1d[0][0]       
__________________________________________________________________________________________________
dropout_37 (Dropout)            (None, 256)          0           concatenate[0][0]                
__________________________________________________________________________________________________
dense (Dense)                   (None, 3)            771         dropout_37[0][0]                 
==================================================================================================
Total params: 109,909,507
Trainable params: 427,267
Non-trainable params: 109,482,240
__________________________________________________________________________________________________

```
</div>
Create train and validation data generators


```python
train_data = BertSemanticDataGenerator(
    train_df[["sentence1", "sentence2"]].values.astype("str"),
    y_train,
    batch_size=batch_size,
    shuffle=True,
)
valid_data = BertSemanticDataGenerator(
    valid_df[["sentence1", "sentence2"]].values.astype("str"),
    y_val,
    batch_size=batch_size,
    shuffle=False,
)
```


<div class="k-default-codeblock">
```
HBox(children=(FloatProgress(value=0.0, description='Downloading', max=231508.0, style=ProgressStyle(descripti…

```
</div>
    


---
## Train the Model

Training is done only for the top layers to perform "feature extraction",
which will allow the model to use the representations of the pretrained model.


```python
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)
```

<div class="k-default-codeblock">
```
Epoch 1/2
3121/3121 [==============================] - 666s 213ms/step - loss: 0.6925 - acc: 0.7049 - val_loss: 0.5294 - val_acc: 0.7899
Epoch 2/2
3121/3121 [==============================] - 661s 212ms/step - loss: 0.5917 - acc: 0.7587 - val_loss: 0.4955 - val_acc: 0.8052

```
</div>
---
## Fine-tuning

This step must only be performed after the feature extraction model has
been trained to convergence on the new data.

This is an optional last step where `bert_model` is unfreezed and retrained
with a very low learning rate. This can deliver meaningful improvement by
incrementally adapting the pretrained features to the new data.


```python
# Unfreeze the bert_model.
bert_model.trainable = True
# Recompile the model to make the change effective.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
```

<div class="k-default-codeblock">
```
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, 128)]        0                                            
__________________________________________________________________________________________________
attention_masks (InputLayer)    [(None, 128)]        0                                            
__________________________________________________________________________________________________
token_type_ids (InputLayer)     [(None, 128)]        0                                            
__________________________________________________________________________________________________
tf_bert_model (TFBertModel)     ((None, 128, 768), ( 109482240   input_ids[0][0]                  
                                                                 attention_masks[0][0]            
                                                                 token_type_ids[0][0]             
__________________________________________________________________________________________________
bidirectional (Bidirectional)   (None, 128, 128)     426496      tf_bert_model[0][0]              
__________________________________________________________________________________________________
global_average_pooling1d (Globa (None, 128)          0           bidirectional[0][0]              
__________________________________________________________________________________________________
global_max_pooling1d (GlobalMax (None, 128)          0           bidirectional[0][0]              
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 256)          0           global_average_pooling1d[0][0]   
                                                                 global_max_pooling1d[0][0]       
__________________________________________________________________________________________________
dropout_37 (Dropout)            (None, 256)          0           concatenate[0][0]                
__________________________________________________________________________________________________
dense (Dense)                   (None, 3)            771         dropout_37[0][0]                 
==================================================================================================
Total params: 109,909,507
Trainable params: 109,909,507
Non-trainable params: 0
__________________________________________________________________________________________________

```
</div>
## Train the entire model end-to-end


```python
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)
```

<div class="k-default-codeblock">
```
Epoch 1/2
3121/3121 [==============================] - 1574s 504ms/step - loss: 0.4698 - accuracy: 0.8181 - val_loss: 0.3787 - val_accuracy: 0.8598
Epoch 2/2
3121/3121 [==============================] - 1569s 503ms/step - loss: 0.3516 - accuracy: 0.8702 - val_loss: 0.3416 - val_accuracy: 0.8757

```
</div>
---
## Evaluate model on the test set


```python
test_data = BertSemanticDataGenerator(
    test_df[["sentence1", "sentence2"]].values.astype("str"),
    y_test,
    batch_size=batch_size,
    shuffle=False,
)
model.evaluate(test_data, verbose=1)
```

<div class="k-default-codeblock">
```
312/312 [==============================] - 55s 177ms/step - loss: 0.3697 - accuracy: 0.8629

[0.3696725070476532, 0.8628805875778198]

```
</div>
---
## Inference on custom sentences


```python

def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return pred, proba

```

Check results on some example sentence pairs.


```python
sentence1 = "Two women are observing something together."
sentence2 = "Two women are standing with their eyes closed."
check_similarity(sentence1, sentence2)
```




<div class="k-default-codeblock">
```
('contradiction', ' 0.91%')

```
</div>
Check results on some example sentence pairs.


```python
sentence1 = "A smiling costumed woman is holding an umbrella"
sentence2 = "A happy woman in a fairy costume holds an umbrella"
check_similarity(sentence1, sentence2)
```




<div class="k-default-codeblock">
```
('neutral', ' 0.88%')

```
</div>
Check results on some example sentence pairs


```python
sentence1 = "A soccer game with multiple males playing"
sentence2 = "Some men are playing a sport"
check_similarity(sentence1, sentence2)
```




<div class="k-default-codeblock">
```
('entailment', ' 0.94%')

```
</div>
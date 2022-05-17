# Text classification using Decision Forests and pretrained embeddings

**Author:** Gitesh Chawda<br>
**Date created:** 09/05/2022<br>
**Last modified:** 09/05/2022<br>
**Description:** Using Tensorflow Decision Forests for text classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/tweet-classification-using-tfdf.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/tweet-classification-using-tfdf.py)



---
## Introduction

[TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF)
is a collection of state-of-the-art algorithms for Decision Forest models that are
compatible with Keras APIs. The module includes Random Forests, Gradient Boosted Trees,
and CART, and can be used for regression, classification, and ranking tasks.

In this example we will use Gradient Boosted Trees with pretrained embeddings to
classify disaster-related tweets.

### See also:

- [TF-DF beginner tutorial](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab)
- [TF-DF intermediate tutorial](https://www.tensorflow.org/decision_forests/tutorials/intermediate_colab).

Install Tensorflow Decision Forest using following command :
`pip install tensorflow_decision_forests`

---
## Imports


```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt
```

---
## Get the data

The Dataset is avalaible on [Kaggle](https://www.kaggle.com/c/nlp-getting-started)

Dataset description:

**Files:**

- train.csv: the training set

**Columns:**

- id: a unique identifier for each tweet
- text: the text of the tweet
- location: the location the tweet was sent from (may be blank)
- keyword: a particular keyword from the tweet (may be blank)
- target: in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)


```python
# Turn .csv files into pandas DataFrame's
df = pd.read_csv(
    "https://raw.githubusercontent.com/IMvision12/Tweets-Classification-NLP/main/train.csv"
)
print(df.head())
```

<div class="k-default-codeblock">
```
   id keyword location                                               text  \
0   1     NaN      NaN  Our Deeds are the Reason of this #earthquake M...   
1   4     NaN      NaN             Forest fire near La Ronge Sask. Canada   
2   5     NaN      NaN  All residents asked to 'shelter in place' are ...   
3   6     NaN      NaN  13,000 people receive #wildfires evacuation or...   
4   7     NaN      NaN  Just got sent this photo from Ruby #Alaska as ...   
```
</div>
    
<div class="k-default-codeblock">
```
   target  
0       1  
1       1  
2       1  
3       1  
4       1  

```
</div>
The dataset includes 7613 samples with 5 columns:


```python
print(f"Training dataset shape: {df.shape}")
```

<div class="k-default-codeblock">
```
Training dataset shape: (7613, 5)

```
</div>
Shuffling and dropping unnecessary columns:


```python
df_shuffled = df.sample(frac=1, random_state=42)
# Dropping id, keyword and location columns as these columns consists of mostly nan values
# we will be using only text and target columns
df_shuffled.drop(["id", "keyword", "location"], axis=1, inplace=True)
df_shuffled.reset_index(inplace=True, drop=True)
print(df_shuffled.head())
```

<div class="k-default-codeblock">
```
                                                text  target
0  So you have a new weapon that can cause un-ima...       1
1  The f$&amp;@ing things I do for #GISHWHES Just...       0
2  DT @georgegalloway: RT @Galloway4Mayor: ÛÏThe...       1
3  Aftershock back to school kick off was great. ...       0
4  in response to trauma Children of Addicts deve...       0

```
</div>
Printing information about the shuffled dataframe:


```python
print(df_shuffled.info())
```

<div class="k-default-codeblock">
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7613 entries, 0 to 7612
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   text    7613 non-null   object
 1   target  7613 non-null   int64 
dtypes: int64(1), object(1)
memory usage: 119.1+ KB
None

```
</div>
Total number of "disaster" and "non-disaster" tweets:


```python
print(
    "Total Number of disaster and non-disaster tweets: "
    f"{df_shuffled.target.value_counts()}"
)
```

<div class="k-default-codeblock">
```
Total Number of disaster and non-disaster tweets: 0    4342
1    3271
Name: target, dtype: int64

```
</div>
Let's preview a few samples:


```python
for index, example in df_shuffled[:5].iterrows():
    print(f"Example #{index}")
    print(f"\tTarget : {example['target']}")
    print(f"\tText : {example['text']}")
```

<div class="k-default-codeblock">
```
Example #0
	Target : 1
	Text : So you have a new weapon that can cause un-imaginable destruction.
Example #1
	Target : 0
	Text : The f$&amp;@ing things I do for #GISHWHES Just got soaked in a deluge going for pads and tampons. Thx @mishacollins @/@
Example #2
	Target : 1
	Text : DT @georgegalloway: RT @Galloway4Mayor: ÛÏThe CoL police can catch a pickpocket in Liverpool Stree... http://t.co/vXIn1gOq4Q
Example #3
	Target : 0
	Text : Aftershock back to school kick off was great. I want to thank everyone for making it possible. What a great night.
Example #4
	Target : 0
	Text : in response to trauma Children of Addicts develop a defensive self - one that decreases vulnerability. (3

```
</div>
Splitting dataset into training and test sets:


```python
test_df = df_shuffled.sample(frac=0.1, random_state=42)
train_df = df_shuffled.drop(test_df.index)
print(f"Using {len(train_df)} samples for training and {len(test_df)} for validation")
```

<div class="k-default-codeblock">
```
Using 6852 samples for training and 761 for validation

```
</div>
Total number of "disaster" and "non-disaster" tweets in the training data:


```python
print(train_df["target"].value_counts())
```

<div class="k-default-codeblock">
```
0    3929
1    2923
Name: target, dtype: int64

```
</div>
Total number of "disaster" and "non-disaster" tweets in the test data:


```python
print(test_df["target"].value_counts())
```

<div class="k-default-codeblock">
```
0    413
1    348
Name: target, dtype: int64

```
</div>
---
## Convert data to a `tf.data.Dataset`


```python

def create_dataset(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["text"].to_numpy(), df["target"].to_numpy())
    )
    dataset = dataset.batch(100)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_ds = create_dataset(train_df)
test_ds = create_dataset(test_df)
```

---
## Downloading pretrained embeddings

The Universal Sentence Encoder embeddings encode text into high-dimensional vectors that can be
used for text classification, semantic similarity, clustering and other natural language
tasks. They're trained on a variety of data sources and a variety of tasks. Their input is
variable-length English text and their output is a 512 dimensional vector.

To learn more about these pretrained embeddings, see
[Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4).


```python
sentence_encoder_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4"
)
```

---
## Creating our models

We create two models. In the first model (model_1) raw text will be first encoded via
pretrained embeddings and then passed to a Gradient Boosted Tree model for
classification. In the second model (model_2) raw text will be directly passed to
the Gradient Boosted Trees model.

Building model_1


```python
inputs = layers.Input(shape=(), dtype=tf.string)
outputs = sentence_encoder_layer(inputs)
preprocessor = keras.Model(inputs=inputs, outputs=outputs)
model_1 = tfdf.keras.GradientBoostedTreesModel(preprocessing=preprocessor)
```

<div class="k-default-codeblock">
```
Use /tmp/tmpkpl10aj9 as temporary training directory

```
</div>
Building model_2


```python
model_2 = tfdf.keras.GradientBoostedTreesModel()
```

<div class="k-default-codeblock">
```
Use /tmp/tmpysfsq6o0 as temporary training directory

```
</div>
---
## Train the models

We compile our model by passing the metrics `Accuracy`, `Recall`, `Precision` and
`AUC`. When it comes to the loss, TF-DF automatically detects the best loss for the task
(Classification or regression). It is printed in the model summary.

Also, because they're batch-training models rather than mini-batch gradient descent models,
TF-DF models do not need a validation dataset to monitor overfitting, or to stop
training early. Some algorithms do not use a validation dataset (e.g. Random Forest)
while some others do (e.g. Gradient Boosted Trees). If a validation dataset is
needed, it will be extracted automatically from the training dataset.


```python
# Compiling model_1
model_1.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
model_1.fit(train_ds)

# Compiling model_2
model_2.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
model_2.fit(train_ds)
```

<div class="k-default-codeblock">
```
Starting reading the dataset
77/77 [==============================] - ETA: 0s
Dataset read in 0:00:15.844516
Training model
Model trained in 0:02:30.922245
Compiling model
77/77 [==============================] - 167s 2s/step
Starting reading the dataset
55/77 [====================>.........] - ETA: 0s
Dataset read in 0:00:00.219258
Training model
Model trained in 0:00:00.289591
Compiling model
77/77 [==============================] - 1s 6ms/step

<keras.callbacks.History at 0x7f453f9349d0>

```
</div>

## Plotting training metrics


```python

def plot_curve(logs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Loss")

    plt.show()


plot_curve(logs_1)
plot_curve(logs_2)
```


![png](/img/examples/nlp/tweet-classification-using-tfdf/tweet-classification-using-tfdf_41_0.png)



![png](/img/examples/nlp/tweet-classification-using-tfdf/tweet-classification-using-tfdf_41_1.png)


---
## Evaluating on test data


```python
results = model_1.evaluate(test_ds, return_dict=True, verbose=0)
print("model_1 Evaluation: \n")
for name, value in results.items():
    print(f"{name}: {value:.4f}")

results = model_2.evaluate(test_ds, return_dict=True, verbose=0)
print("model_2 Evaluation: \n")
for name, value in results.items():
    print(f"{name}: {value:.4f}")
```

<div class="k-default-codeblock">
```
model_1 Evaluation: 
```
</div>
    
<div class="k-default-codeblock">
```
loss: 0.0000
Accuracy: 0.9631
recall: 0.9425
precision: 0.9707
auc: 0.9890
model_2 Evaluation: 
```
</div>
    
<div class="k-default-codeblock">
```
loss: 0.0000
Accuracy: 0.5731
recall: 0.0064
precision: 1.0000
auc: 0.5035

```
</div>
## Predicting on validation data


```python
test_df.reset_index(inplace=True, drop=True)
for index, row in test_df.iterrows():
    text = tf.expand_dims(row["text"], axis=0)
    preds = model_1.predict_step(text)
    preds = tf.squeeze(tf.round(preds))
    print(f"Text: {row['text']}")
    print(f"Prediction: {int(preds)}")
    print(f"Ground Truth : {row['target']}")
    if index == 10:
        break
```

<div class="k-default-codeblock">
```
Text: DFR EP016 Monthly Meltdown - On Dnbheaven 2015.08.06 http://t.co/EjKRf8N8A8 #Drum and Bass #heavy #nasty http://t.co/SPHWE6wFI5
Prediction: 0
Ground Truth : 0
Text: FedEx no longer to transport bioterror germs in wake of anthrax lab mishaps http://t.co/qZQc8WWwcN via @usatoday
Prediction: 1
Ground Truth : 0
Text: Gunmen kill four in El Salvador bus attack: Suspected Salvadoran gang members killed four people and wounded s... http://t.co/CNtwB6ScZj
Prediction: 1
Ground Truth : 1
Text: @camilacabello97 Internally and externally screaming
Prediction: 1
Ground Truth : 1
Text: Radiation emergency #preparedness starts with knowing to: get inside stay inside and stay tuned http://t.co/RFFPqBAz2F via @CDCgov
Prediction: 1
Ground Truth : 1
Text: Investigators rule catastrophic structural failure resulted in 2014 Virg.. Related Articles: http://t.co/Cy1LFeNyV8
Prediction: 1
Ground Truth : 1
Text: How the West was burned: Thousands of wildfires ablaze in #California alone http://t.co/iCSjGZ9tE1 #climate #energy http://t.co/9FxmN0l0Bd
Prediction: 1
Ground Truth : 1
Text: Map: Typhoon Soudelor's predicted path as it approaches Taiwan; expected to make landfall over southern China by SÛ_ http://t.co/JDVSGVhlIs
Prediction: 1
Ground Truth : 1
Text: Ûª93 blasts accused Yeda Yakub dies in Karachi of heart attack http://t.co/mfKqyxd8XG #Mumbai
Prediction: 1
Ground Truth : 1
Text: My ears are bleeding  https://t.co/k5KnNwugwT
Prediction: 0
Ground Truth : 0
Text: @RedCoatJackpot *As it was typical for them their bullets collided and none managed to reach their targets; such was the ''curse'' of a --
Prediction: 0
Ground Truth : 0

```
</div>
---
## Concluding remarks

The TensorFlow Decision Forests package provides powerful models
that work especially well with structured data. In our experiments,
the Gradient Boosted Tree model with pretrained embeddings achieved 96.31%
test accuracy while the plain Gradient Boosted Tree model had 57.31% accuracy.

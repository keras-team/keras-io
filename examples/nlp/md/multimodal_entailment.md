# Multimodal entailment

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2021/08/08<br>
**Last modified:** 2025/01/03<br>
**Description:** Training a multimodal model for predicting entailment.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/multimodal_entailment.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/multimodal_entailment.py)



---
## Introduction

In this example, we will build and train a model for predicting multimodal entailment. We will be
using the
[multimodal entailment dataset](https://github.com/google-research-datasets/recognizing-multimodal-entailment)
recently introduced by Google Research.

### What is multimodal entailment?

On social media platforms, to audit and moderate content
we may want to find answers to the
following questions in near real-time:

* Does a given piece of information contradict the other?
* Does a given piece of information imply the other?

In NLP, this task is called analyzing _textual entailment_. However, that's only
when the information comes from text content.
In practice, it's often the case the information available comes not just
from text content, but from a multimodal combination of text, images, audio, video, etc.
_Multimodal entailment_ is simply the extension of textual entailment to a variety
of new input modalities.

### Requirements

This example requires TensorFlow 2.5 or higher. In addition, TensorFlow Hub and
TensorFlow Text are required for the BERT model
([Devlin et al.](https://arxiv.org/abs/1810.04805)). These libraries can be installed
using the following command:


```python
!pip install -q tensorflow_text
```

    
<div class="k-default-codeblock">
```
 [[34;49mnotice[1;39;49m][39;49m A new release of pip is available: [31;49m24.0[39;49m -> [32;49m24.3.1
 [[34;49mnotice[1;39;49m][39;49m To update, run: [32;49mpip install --upgrade pip

```
</div>
---
## Imports


```python
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

import tensorflow as tf
import keras
import keras_hub
```

<div class="k-default-codeblock">
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1735907683.393230   12828 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1735907683.399130   12828 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

```
</div>
---
## Define a label map


```python
label_map = {"Contradictory": 0, "Implies": 1, "NoEntailment": 2}
```

---
## Collect the dataset

The original dataset is available
[here](https://github.com/google-research-datasets/recognizing-multimodal-entailment).
It comes with URLs of images which are hosted on Twitter's photo storage system called
the
[Photo Blob Storage (PBS for short)](https://blog.twitter.com/engineering/en_us/a/2012/blobstore-twitter-s-in-house-photo-storage-system).
We will be working with the downloaded images along with additional data that comes with
the original dataset. Thanks to
[Nilabhra Roy Chowdhury](https://de.linkedin.com/in/nilabhraroychowdhury) who worked on
preparing the image data.


```python
image_base_path = keras.utils.get_file(
    "tweet_images",
    "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/releases/download/v1.0.0/tweet_images.tar.gz",
    untar=True,
)
```

---
## Read the dataset and apply basic preprocessing


```python
df = pd.read_csv(
    "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/raw/main/csvs/tweets.csv"
)
df.sample(10)
```




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
      <th>id_1</th>
      <th>text_1</th>
      <th>image_1</th>
      <th>id_2</th>
      <th>text_2</th>
      <th>image_2</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>1372954867178729473</td>
      <td>JK reports152 new positive cases, 125118 recov...</td>
      <td>http://pbs.twimg.com/media/Ew23LJ6U8AA1Een.jpg</td>
      <td>1376577693056069634</td>
      <td>JK reports 235 new positive cases, 126129 reco...</td>
      <td>http://pbs.twimg.com/media/ExqWHX2UUAUVIqD.jpg</td>
      <td>Contradictory</td>
    </tr>
    <tr>
      <th>104</th>
      <td>1341296172032548864</td>
      <td>US president donald trump issues executive ord...</td>
      <td>http://pbs.twimg.com/media/Ep09wrSXEAArQdp.jpg</td>
      <td>1365036686628048897</td>
      <td>president joe biden revokes trump's executive ...</td>
      <td>http://pbs.twimg.com/media/EvGVofdXMAAkK80.jpg</td>
      <td>Implies</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1357584913449558016</td>
      <td>3 Day Temperature Forecast for Glasgow from 20...</td>
      <td>http://pbs.twimg.com/media/EtccSKvXAAICUee.png</td>
      <td>1358309729735499777</td>
      <td>3 Day Temperature Forecast for Glasgow from 20...</td>
      <td>http://pbs.twimg.com/media/EtmvgBLXEAELqon.png</td>
      <td>Contradictory</td>
    </tr>
    <tr>
      <th>670</th>
      <td>1356330489527853056</td>
      <td>#SPX500 SSI is at -1.76\n\nRisk Warning: Losse...</td>
      <td>http://pbs.twimg.com/media/EtKnZJoXUAARsQw.png</td>
      <td>1377161273830305800</td>
      <td>#USOil SSI is at -1.25\n\nRisk Warning: Losses...</td>
      <td>http://pbs.twimg.com/media/Exyo4X4VcAgVjQa.png</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>1083</th>
      <td>1359121217173610504</td>
      <td>"ನಂದಾದೀಪ"\nLiving up to the legacy .\n\nWatch ...</td>
      <td>http://pbs.twimg.com/media/EtyRixhU0Ac7VD4.jpg</td>
      <td>1361662126675292160</td>
      <td>Sons of villains..\nStar directors..\nBrothers...</td>
      <td>http://pbs.twimg.com/media/EuWYfMaUYAk2ii0.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>135</th>
      <td>1344231606035345408</td>
      <td>Becoming breezy mid-morning with winds out of ...</td>
      <td>http://pbs.twimg.com/media/EqergoGXMAMOC-m.jpg</td>
      <td>1370009652516241416</td>
      <td>Today will make 3 in a row 70° or higher and w...</td>
      <td>http://pbs.twimg.com/media/EwNAgsjWEAIbA3k.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>1378336964731146247</td>
      <td>$HOT has bounced off the first support line. L...</td>
      <td>http://pbs.twimg.com/media/EyDVXuiWgAAyiV9.png</td>
      <td>1378695758455656452</td>
      <td>$HOT has just broken through resistance. 👀\n#H...</td>
      <td>http://pbs.twimg.com/media/EyIcXmSW8AIYr8c.png</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>543</th>
      <td>1335155454985400320</td>
      <td>[NOMINATION WEEK DEADLINE COUNTDOWN, D-3]\n\nD...</td>
      <td>http://pbs.twimg.com/media/EodszoQUUAI42yW.jpg</td>
      <td>1335879723801407488</td>
      <td>[NOMINATION WEEK DEADLINE COUNTDOWN, D-1] \nDe...</td>
      <td>http://pbs.twimg.com/media/Eon-YvCVQAc4LHS.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1358551424334893056</td>
      <td>Sun protection recommended from 8:20 am to 4:3...</td>
      <td>http://pbs.twimg.com/media/EtqLMAiVgAAF0Tw.png</td>
      <td>1363987146743238658</td>
      <td>Sun protection recommended from 8:30 am to 4:2...</td>
      <td>http://pbs.twimg.com/media/Eu3bENsUcAAVtML.png</td>
      <td>Contradictory</td>
    </tr>
    <tr>
      <th>1310</th>
      <td>1378514622542540800</td>
      <td>Friends, interested all go to have a look!\n@F...</td>
      <td>http://pbs.twimg.com/media/EyF3vjiXIAE1MH2.jpg</td>
      <td>1380603591589724168</td>
      <td>Friends! Anyone interested? Go and have a look...</td>
      <td>http://pbs.twimg.com/media/EyjjpodUYAIanwc.jpg</td>
      <td>Contradictory</td>
    </tr>
  </tbody>
</table>
</div>



The columns we are interested in are the following:

* `text_1`
* `image_1`
* `text_2`
* `image_2`
* `label`

The entailment task is formulated as the following:

***Given the pairs of (`text_1`, `image_1`) and (`text_2`, `image_2`) do they entail (or
not entail or contradict) each other?***

We have the images already downloaded. `image_1` is downloaded as `id1` as its filename
and `image2` is downloaded as `id2` as its filename. In the next step, we will add two
more columns to `df` - filepaths of `image_1`s and `image_2`s.


```python
images_one_paths = []
images_two_paths = []

for idx in range(len(df)):
    current_row = df.iloc[idx]
    id_1 = current_row["id_1"]
    id_2 = current_row["id_2"]
    extentsion_one = current_row["image_1"].split(".")[-1]
    extentsion_two = current_row["image_2"].split(".")[-1]

    image_one_path = os.path.join(image_base_path, str(id_1) + f".{extentsion_one}")
    image_two_path = os.path.join(image_base_path, str(id_2) + f".{extentsion_two}")

    images_one_paths.append(image_one_path)
    images_two_paths.append(image_two_path)

df["image_1_path"] = images_one_paths
df["image_2_path"] = images_two_paths

# Create another column containing the integer ids of
# the string labels.
df["label_idx"] = df["label"].apply(lambda x: label_map[x])
```

---
## Dataset visualization


```python

def visualize(idx):
    current_row = df.iloc[idx]
    image_1 = plt.imread(current_row["image_1_path"])
    image_2 = plt.imread(current_row["image_2_path"])
    text_1 = current_row["text_1"]
    text_2 = current_row["text_2"]
    label = current_row["label"]

    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.axis("off")
    plt.title("Image One")
    plt.subplot(1, 2, 2)
    plt.imshow(image_1)
    plt.axis("off")
    plt.title("Image Two")
    plt.show()

    print(f"Text one: {text_1}")
    print(f"Text two: {text_2}")
    print(f"Label: {label}")


random_idx = random.choice(range(len(df)))
visualize(random_idx)

random_idx = random.choice(range(len(df)))
visualize(random_idx)
```


    
![png](/img/examples/nlp/multimodal_entailment/multimodal_entailment_14_0.png)
    


<div class="k-default-codeblock">
```
Text one: Learn to play Piano this lockdown
#music #musiclessons #lockdown #lockdownlearning #lockdown2021 #onlinelearning #onlinelessons #musiconline #keeplearning #motivation #pianolessons #learnpiano #piano https://t.co/PMuMBWFlzn
Text two: Learn to play Piano this lockdown
#music #musiclessons #lockdown #lockdownlearning #lockdown2021 #onlinelearning #onlinelessons #musiconline #keeplearning #motivation #pianolessons #learnpiano #piano https://t.co/9tfaWn8Uzc
Label: Implies

```
</div>
    
![png](/img/examples/nlp/multimodal_entailment/multimodal_entailment_14_2.png)
    


<div class="k-default-codeblock">
```
Text one: The delicious new Shake Me! Vegan Peanut Butter* is now available for pre-order. The world’s est Complete Meal Replacement. https://t.co/TsgFDgIRP2 #vegan #kosher #globalhealth #penutbutter #plantbasednutrition https://t.co/yG3pOj0CI0
Text two: The delicious new Shake Me! Vegan Peanut Butter is now available! https://t.co/TsgFDgrgXu #nutrition #vegan #kosher #globalhealth #penutbutter #vivri https://t.co/T9cRZZJMXy
Label: NoEntailment

```
</div>
---
## Train/test split

The dataset suffers from
[class imbalance problem](https://developers.google.com/machine-learning/glossary#class-imbalanced-dataset).
We can confirm that in the following cell.


```python
df["label"].value_counts()
```




<div class="k-default-codeblock">
```
label
NoEntailment     1182
Implies           109
Contradictory     109
Name: count, dtype: int64

```
</div>
To account for that we will go for a stratified split.


```python
# 10% for test
train_df, test_df = train_test_split(
    df, test_size=0.1, stratify=df["label"].values, random_state=42
)
# 5% for validation
train_df, val_df = train_test_split(
    train_df, test_size=0.05, stratify=train_df["label"].values, random_state=42
)

print(f"Total training examples: {len(train_df)}")
print(f"Total validation examples: {len(val_df)}")
print(f"Total test examples: {len(test_df)}")
```

<div class="k-default-codeblock">
```
Total training examples: 1197
Total validation examples: 63
Total test examples: 140

```
</div>
---
## Data input pipeline

Keras Hub provides
[variety of BERT family of models](https://keras.io/keras_hub/presets/).
Each of those models comes with a
corresponding preprocessing layer. You can learn more about these models and their
preprocessing layers from
[this resource](https://www.kaggle.com/models/keras/bert/keras/bert_base_en_uncased/2).

To keep the runtime of this example relatively short, we will use a base_unacased variant of
the original BERT model.

text preprocessing using KerasHub


```python
text_preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(
    "bert_base_en_uncased",
    sequence_length=128,
)
```

<div class="k-default-codeblock">
```
W0000 00:00:1735907697.293486   12828 gpu_device.cc:2344] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...

```
</div>
### Run the preprocessor on a sample input


```python
idx = random.choice(range(len(train_df)))
row = train_df.iloc[idx]
sample_text_1, sample_text_2 = row["text_1"], row["text_2"]
print(f"Text 1: {sample_text_1}")
print(f"Text 2: {sample_text_2}")

test_text = [sample_text_1, sample_text_2]
text_preprocessed = text_preprocessor(test_text)

print("Keys           : ", list(text_preprocessed.keys()))
print("Shape Token Ids : ", text_preprocessed["token_ids"].shape)
print("Token Ids       : ", text_preprocessed["token_ids"][0, :16])
print(" Shape Padding Mask     : ", text_preprocessed["padding_mask"].shape)
print("Padding Mask     : ", text_preprocessed["padding_mask"][0, :16])
print("Shape Segment Ids : ", text_preprocessed["segment_ids"].shape)
print("Segment Ids       : ", text_preprocessed["segment_ids"][0, :16])

```

<div class="k-default-codeblock">
```
Text 1: Sign up for one of our 5 or 6 day courses, like #FOR500, #FOR572 or #FOR585, and choose from an iPad mini, a Galaxy Tab S5e, or take $300 off.
```
</div>
    
<div class="k-default-codeblock">
```
https://t.co/knpN3xIwoI
Note: Valid in US only https://t.co/5kIt74v5FB
Text 2: Hurry - offer ends tomorrow! Sign up for one of our 5 or 6 day courses, like #FOR500, #FOR572 or #FOR585, and choose from an iPad mini, a Galaxy Tab S5e, or take $300 off.
```
</div>
    
<div class="k-default-codeblock">
```
https://t.co/knpN3xIwoI
Note: Valid in US only https://t.co/9IyveqSd68
Keys           :  ['token_ids', 'padding_mask', 'segment_ids']
Shape Token Ids :  (2, 128)
Token Ids       :  tf.Tensor(
[ 101 3696 2039 2005 2028 1997 2256 1019 2030 1020 2154 5352 1010 2066
 1001 2005], shape=(16,), dtype=int32)
 Shape Padding Mask     :  (2, 128)
Padding Mask     :  tf.Tensor(
[ True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True], shape=(16,), dtype=bool)
Shape Segment Ids :  (2, 128)
Segment Ids       :  tf.Tensor([0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], shape=(16,), dtype=int32)

```
</div>
We will now create `tf.data.Dataset` objects from the dataframes.

Note that the text inputs will be preprocessed as a part of the data input pipeline. But
the preprocessing modules can also be a part of their corresponding BERT models. This
helps reduce the training/serving skew and lets our models operate with raw text inputs.
Follow [this tutorial](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)
to learn more about how to incorporate the preprocessing modules directly inside the
models.


```python

def dataframe_to_dataset(dataframe):
    columns = ["image_1_path", "image_2_path", "text_1", "text_2", "label_idx"]
    dataframe = dataframe[columns].copy()
    labels = dataframe.pop("label_idx")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

```

### Preprocessing utilities


```python
resize = (128, 128)
bert_input_features = ["padding_mask", "segment_ids", "token_ids"]


def preprocess_image(image_path):
    extension = tf.strings.split(image_path)[-1]

    image = tf.io.read_file(image_path)
    if extension == b"jpg":
        image = tf.image.decode_jpeg(image, 3)
    else:
        image = tf.image.decode_png(image, 3)
    image = keras.ops.image.resize(image, resize)
    return image


def preprocess_text(text_1, text_2):
    text_1 = keras.ops.convert_to_tensor([text_1])
    text_2 = keras.ops.convert_to_tensor([text_2])
    output = text_preprocessor((text_1, text_2))
    output = {
        feature: keras.ops.reshape(output[feature], [-1])
        for feature in bert_input_features
    }
    return output


def preprocess_text_and_image(sample):
    image_1 = preprocess_image(sample["image_1_path"])
    image_2 = preprocess_image(sample["image_2_path"])
    text = preprocess_text(sample["text_1"], sample["text_2"])
    return {
        "image_1": image_1,
        "image_2": image_2,
        "padding_mask": text["padding_mask"],
        "segment_ids": text["segment_ids"],
        "token_ids": text["token_ids"],
    }

```

### Create the final datasets


```python
batch_size = 32
auto = tf.data.AUTOTUNE


def prepare_dataset(dataframe, training=True):
    ds = dataframe_to_dataset(dataframe)
    if training:
        ds = ds.shuffle(len(train_df))
    ds = ds.map(lambda x, y: (preprocess_text_and_image(x), y)).cache()
    ds = ds.batch(batch_size).prefetch(auto)
    return ds


train_ds = prepare_dataset(train_df)
validation_ds = prepare_dataset(val_df, False)
test_ds = prepare_dataset(test_df, False)

```

---
## Model building utilities

Our final model will accept two images along with their text counterparts. While the
images will be directly fed to the model the text inputs will first be preprocessed and
then will make it into the model. Below is a visual illustration of this approach:

![](https://github.com/sayakpaul/Multimodal-Entailment-Baseline/raw/main/figures/brief_architecture.png)

The model consists of the following elements:

* A standalone encoder for the images. We will use a
[ResNet50V2](https://arxiv.org/abs/1603.05027) pre-trained on the ImageNet-1k dataset for
this.
* A standalone encoder for the images. A pre-trained BERT will be used for this.

After extracting the individual embeddings, they will be projected in an identical space.
Finally, their projections will be concatenated and be fed to the final classification
layer.

This is a multi-class classification problem involving the following classes:

* NoEntailment
* Implies
* Contradictory

`project_embeddings()`, `create_vision_encoder()`, and `create_text_encoder()` utilities
are referred from [this example](https://keras.io/examples/nlp/nl_image_search/).

Projection utilities


```python

def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = keras.layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = keras.ops.nn.gelu(projected_embeddings)
        x = keras.layers.Dense(projection_dims)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = keras.layers.LayerNormalization()(x)
    return projected_embeddings

```

Vision encoder utilities


```python

def create_vision_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained ResNet50V2 model to be used as the base encoder.
    resnet_v2 = keras.applications.ResNet50V2(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # Set the trainability of the base encoder.
    for layer in resnet_v2.layers:
        layer.trainable = trainable

    # Receive the images as inputs.
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
    image_2 = keras.Input(shape=(128, 128, 3), name="image_2")

    # Preprocess the input image.
    preprocessed_1 = keras.applications.resnet_v2.preprocess_input(image_1)
    preprocessed_2 = keras.applications.resnet_v2.preprocess_input(image_2)

    # Generate the embeddings for the images using the resnet_v2 model
    # concatenate them.
    embeddings_1 = resnet_v2(preprocessed_1)
    embeddings_2 = resnet_v2(preprocessed_2)
    embeddings = keras.layers.Concatenate()([embeddings_1, embeddings_2])

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return keras.Model([image_1, image_2], outputs, name="vision_encoder")

```

Text encoder utilities


```python

def create_text_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained BERT BackBone using KerasHub.
    bert = keras_hub.models.BertBackbone.from_preset(
        "bert_base_en_uncased", num_classes=3
    )

    # Set the trainability of the base encoder.
    bert.trainable = trainable

    # Receive the text as inputs.
    bert_input_features = ["padding_mask", "segment_ids", "token_ids"]
    inputs = {
        feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features
    }

    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(inputs)["pooled_output"]

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return keras.Model(inputs, outputs, name="text_encoder")

```

Multimodal model utilities


```python

def create_multimodal_model(
    num_projection_layers=1,
    projection_dims=256,
    dropout_rate=0.1,
    vision_trainable=False,
    text_trainable=False,
):
    # Receive the images as inputs.
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
    image_2 = keras.Input(shape=(128, 128, 3), name="image_2")

    # Receive the text as inputs.
    bert_input_features = ["padding_mask", "segment_ids", "token_ids"]
    text_inputs = {
        feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features
    }
    text_inputs = list(text_inputs.values())
    # Create the encoders.
    vision_encoder = create_vision_encoder(
        num_projection_layers, projection_dims, dropout_rate, vision_trainable
    )
    text_encoder = create_text_encoder(
        num_projection_layers, projection_dims, dropout_rate, text_trainable
    )

    # Fetch the embedding projections.
    vision_projections = vision_encoder([image_1, image_2])
    text_projections = text_encoder(text_inputs)

    # Concatenate the projections and pass through the classification layer.
    concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
    outputs = keras.layers.Dense(3, activation="softmax")(concatenated)
    return keras.Model([image_1, image_2, *text_inputs], outputs)


multimodal_model = create_multimodal_model()
keras.utils.plot_model(multimodal_model, show_shapes=True)
```




    
![png](/img/examples/nlp/multimodal_entailment/multimodal_entailment_38_0.png)
    



You can inspect the structure of the individual encoders as well by setting the
`expand_nested` argument of `plot_model()` to `True`. You are encouraged
to play with the different hyperparameters involved in building this model and
observe how the final performance is affected.

---
## Compile and train the model


```python
multimodal_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = multimodal_model.fit(train_ds, validation_data=validation_ds, epochs=1)
```

<div class="k-default-codeblock">
```
/home/humbulani/jax/env/lib/python3.11/site-packages/keras/src/models/functional.py:238: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: {'padding_mask': 'padding_mask', 'segment_ids': 'segment_ids', 'token_ids': 'token_ids'}
Received: inputs=['Tensor(shape=(None, None))', 'Tensor(shape=(None, None))', 'Tensor(shape=(None, None))']
  warnings.warn(msg)

```
</div>
    
  1/38 [37m━━━━━━━━━━━━━━━━━━━━  42:56 70s/step - accuracy: 0.2500 - loss: 1.6472

<div class="k-default-codeblock">
```

```
</div>
  2/38 ━[37m━━━━━━━━━━━━━━━━━━━  13:24 22s/step - accuracy: 0.3906 - loss: 2.0702

<div class="k-default-codeblock">
```

```
</div>
  3/38 ━[37m━━━━━━━━━━━━━━━━━━━  13:20 23s/step - accuracy: 0.4757 - loss: 2.2410

<div class="k-default-codeblock">
```

```
</div>
  4/38 ━━[37m━━━━━━━━━━━━━━━━━━  12:59 23s/step - accuracy: 0.5326 - loss: 2.2835

<div class="k-default-codeblock">
```

```
</div>
  5/38 ━━[37m━━━━━━━━━━━━━━━━━━  12:27 23s/step - accuracy: 0.5710 - loss: 2.2807

<div class="k-default-codeblock">
```

```
</div>
  6/38 ━━━[37m━━━━━━━━━━━━━━━━━  12:06 23s/step - accuracy: 0.6017 - loss: 2.2483

<div class="k-default-codeblock">
```

```
</div>
  7/38 ━━━[37m━━━━━━━━━━━━━━━━━  11:42 23s/step - accuracy: 0.6216 - loss: 2.2529

<div class="k-default-codeblock">
```

```
</div>
  8/38 ━━━━[37m━━━━━━━━━━━━━━━━  11:16 23s/step - accuracy: 0.6313 - loss: 2.2501

<div class="k-default-codeblock">
```

```
</div>
  9/38 ━━━━[37m━━━━━━━━━━━━━━━━  10:53 23s/step - accuracy: 0.6407 - loss: 2.2301

<div class="k-default-codeblock">
```

```
</div>
 10/38 ━━━━━[37m━━━━━━━━━━━━━━━  10:31 23s/step - accuracy: 0.6491 - loss: 2.2041

<div class="k-default-codeblock">
```

```
</div>
 11/38 ━━━━━[37m━━━━━━━━━━━━━━━  10:08 23s/step - accuracy: 0.6572 - loss: 2.1754

<div class="k-default-codeblock">
```

```
</div>
 12/38 ━━━━━━[37m━━━━━━━━━━━━━━  9:46 23s/step - accuracy: 0.6650 - loss: 2.1464 

<div class="k-default-codeblock">
```

```
</div>
 13/38 ━━━━━━[37m━━━━━━━━━━━━━━  9:23 23s/step - accuracy: 0.6724 - loss: 2.1149

<div class="k-default-codeblock">
```

```
</div>
 14/38 ━━━━━━━[37m━━━━━━━━━━━━━  8:59 22s/step - accuracy: 0.6791 - loss: 2.0861

<div class="k-default-codeblock">
```

```
</div>
 15/38 ━━━━━━━[37m━━━━━━━━━━━━━  8:37 23s/step - accuracy: 0.6853 - loss: 2.0569

<div class="k-default-codeblock">
```

```
</div>
 16/38 ━━━━━━━━[37m━━━━━━━━━━━━  8:14 22s/step - accuracy: 0.6910 - loss: 2.0276

<div class="k-default-codeblock">
```

```
</div>
 17/38 ━━━━━━━━[37m━━━━━━━━━━━━  7:52 23s/step - accuracy: 0.6966 - loss: 1.9968

<div class="k-default-codeblock">
```

```
</div>
 18/38 ━━━━━━━━━[37m━━━━━━━━━━━  7:29 22s/step - accuracy: 0.7019 - loss: 1.9657

<div class="k-default-codeblock">
```

```
</div>
 19/38 ━━━━━━━━━━[37m━━━━━━━━━━  7:08 23s/step - accuracy: 0.7068 - loss: 1.9354

<div class="k-default-codeblock">
```

```
</div>
 20/38 ━━━━━━━━━━[37m━━━━━━━━━━  6:45 23s/step - accuracy: 0.7113 - loss: 1.9068

<div class="k-default-codeblock">
```

```
</div>
 21/38 ━━━━━━━━━━━[37m━━━━━━━━━  6:22 23s/step - accuracy: 0.7149 - loss: 1.8800

<div class="k-default-codeblock">
```

```
</div>
 22/38 ━━━━━━━━━━━[37m━━━━━━━━━  6:00 23s/step - accuracy: 0.7182 - loss: 1.8538

<div class="k-default-codeblock">
```

```
</div>
 23/38 ━━━━━━━━━━━━[37m━━━━━━━━  5:37 22s/step - accuracy: 0.7213 - loss: 1.8284

<div class="k-default-codeblock">
```

```
</div>
 24/38 ━━━━━━━━━━━━[37m━━━━━━━━  5:14 22s/step - accuracy: 0.7239 - loss: 1.8047

<div class="k-default-codeblock">
```

```
</div>
 25/38 ━━━━━━━━━━━━━[37m━━━━━━━  4:52 22s/step - accuracy: 0.7266 - loss: 1.7814

<div class="k-default-codeblock">
```

```
</div>
 26/38 ━━━━━━━━━━━━━[37m━━━━━━━  4:29 22s/step - accuracy: 0.7292 - loss: 1.7587

<div class="k-default-codeblock">
```

```
</div>
 27/38 ━━━━━━━━━━━━━━[37m━━━━━━  4:07 22s/step - accuracy: 0.7318 - loss: 1.7363

<div class="k-default-codeblock">
```

```
</div>
 28/38 ━━━━━━━━━━━━━━[37m━━━━━━  3:45 23s/step - accuracy: 0.7343 - loss: 1.7149

<div class="k-default-codeblock">
```

```
</div>
 29/38 ━━━━━━━━━━━━━━━[37m━━━━━  3:22 23s/step - accuracy: 0.7367 - loss: 1.6941

<div class="k-default-codeblock">
```

```
</div>
 30/38 ━━━━━━━━━━━━━━━[37m━━━━━  3:00 23s/step - accuracy: 0.7391 - loss: 1.6741

<div class="k-default-codeblock">
```

```
</div>
 31/38 ━━━━━━━━━━━━━━━━[37m━━━━  2:37 23s/step - accuracy: 0.7413 - loss: 1.6548

<div class="k-default-codeblock">
```

```
</div>
 32/38 ━━━━━━━━━━━━━━━━[37m━━━━  2:15 23s/step - accuracy: 0.7434 - loss: 1.6366

<div class="k-default-codeblock">
```

```
</div>
 33/38 ━━━━━━━━━━━━━━━━━[37m━━━  1:52 23s/step - accuracy: 0.7454 - loss: 1.6191

<div class="k-default-codeblock">
```

```
</div>
 34/38 ━━━━━━━━━━━━━━━━━[37m━━━  1:29 22s/step - accuracy: 0.7474 - loss: 1.6019

<div class="k-default-codeblock">
```

```
</div>
 35/38 ━━━━━━━━━━━━━━━━━━[37m━━  1:07 23s/step - accuracy: 0.7495 - loss: 1.5849

<div class="k-default-codeblock">
```

```
</div>
 36/38 ━━━━━━━━━━━━━━━━━━[37m━━  44s 22s/step - accuracy: 0.7514 - loss: 1.5685 

<div class="k-default-codeblock">
```

```
</div>
 37/38 ━━━━━━━━━━━━━━━━━━━[37m━  22s 22s/step - accuracy: 0.7532 - loss: 1.5531

<div class="k-default-codeblock">
```

```
</div>
 38/38 ━━━━━━━━━━━━━━━━━━━━ 0s 22s/step - accuracy: 0.7550 - loss: 1.5384 

<div class="k-default-codeblock">
```

```
</div>
 38/38 ━━━━━━━━━━━━━━━━━━━━ 943s 24s/step - accuracy: 0.7566 - loss: 1.5244 - val_accuracy: 0.8413 - val_loss: 0.6571


---
## Evaluate the model


```python
_, acc = multimodal_model.evaluate(test_ds)
print(f"Accuracy on the test set: {round(acc * 100, 2)}%.")
```

    
 1/5 ━━━━[37m━━━━━━━━━━━━━━━━  1:28 22s/step - accuracy: 0.8438 - loss: 0.5892

<div class="k-default-codeblock">
```

```
</div>
 2/5 ━━━━━━━━[37m━━━━━━━━━━━━  1:05 22s/step - accuracy: 0.8672 - loss: 0.4839

<div class="k-default-codeblock">
```

```
</div>
 3/5 ━━━━━━━━━━━━[37m━━━━━━━━  43s 22s/step - accuracy: 0.8767 - loss: 0.4635 

<div class="k-default-codeblock">
```

```
</div>
 4/5 ━━━━━━━━━━━━━━━━[37m━━━━  21s 22s/step - accuracy: 0.8802 - loss: 0.4634

<div class="k-default-codeblock">
```

```
</div>
 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 18s/step - accuracy: 0.8799 - loss: 0.4729 

<div class="k-default-codeblock">
```

```
</div>
 5/5 ━━━━━━━━━━━━━━━━━━━━ 94s 18s/step - accuracy: 0.8797 - loss: 0.4792


<div class="k-default-codeblock">
```
Accuracy on the test set: 87.86%.

```
</div>
---
## Additional notes regarding training

**Incorporating regularization**:

The training logs suggest that the model is starting to overfit and may have benefitted
from regularization. Dropout ([Srivastava et al.](https://jmlr.org/papers/v15/srivastava14a.html))
is a simple yet powerful regularization technique that we can use in our model.
But how should we apply it here?

We could always introduce Dropout (`keras.layers.Dropout`) in between different layers of the model.
But here is another recipe. Our model expects inputs from two different data modalities.
What if either of the modalities is not present during inference? To account for this,
we can introduce Dropout to the individual projections just before they get concatenated:

```python
vision_projections = keras.layers.Dropout(rate)(vision_projections)
text_projections = keras.layers.Dropout(rate)(text_projections)
concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
```

**Attending to what matters**:

Do all parts of the images correspond equally to their textual counterparts? It's likely
not the case. To make our model only focus on the most important bits of the images that relate
well to their corresponding textual parts we can use "cross-attention":

```python
# Embeddings.
vision_projections = vision_encoder([image_1, image_2])
text_projections = text_encoder(text_inputs)

# Cross-attention (Luong-style).
query_value_attention_seq = keras.layers.Attention(use_scale=True, dropout=0.2)(
    [vision_projections, text_projections]
)
# Concatenate.
concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
contextual = keras.layers.Concatenate()([concatenated, query_value_attention_seq])
```

To see this in action, refer to
[this notebook](https://github.com/sayakpaul/Multimodal-Entailment-Baseline/blob/main/multimodal_entailment_attn.ipynb).

**Handling class imbalance**:

The dataset suffers from class imbalance. Investigating the confusion matrix of the
above model reveals that it performs poorly on the minority classes. If we had used a
weighted loss then the training would have been more guided. You can check out
[this notebook](https://github.com/sayakpaul/Multimodal-Entailment-Baseline/blob/main/multimodal_entailment.ipynb)
that takes class-imbalance into account during model training.

**Using only text inputs**:

Also, what if we had only incorporated text inputs for the entailment task? Because of
the nature of the text inputs encountered on social media platforms, text inputs alone
would have hurt the final performance. Under a similar training setup, by only using
text inputs we get to 67.14% top-1 accuracy on the same test set. Refer to
[this notebook](https://github.com/sayakpaul/Multimodal-Entailment-Baseline/blob/main/text_entailment.ipynb)
for details.

Finally, here is a table comparing different approaches taken for the entailment task:

| Type  | Standard<br>Cross-entropy     | Loss-weighted<br>Cross-entropy    | Focal Loss    |
|:---:  |:---:  |:---:    |:---:    |
| Multimodal    | 77.86%    | 67.86%    | 86.43%    |
| Only text     | 67.14%    | 11.43%    | 37.86%    |

You can check out [this repository](https://git.io/JR0HU) to learn more about how the
experiments were conducted to obtain these numbers.

---
## Final remarks

* The architecture we used in this example is too large for the number of data points
available for training. It's going to benefit from more data.
* We used a smaller variant of the original BERT model. Chances are high that with a
larger variant, this performance will be improved. TensorFlow Hub
[provides](https://www.tensorflow.org/text/tutorials/bert_glue#loading_models_from_tensorflow_hub)
a number of different BERT models that you can experiment with.
* We kept the pre-trained models frozen. Fine-tuning them on the multimodal entailment
task would could resulted in better performance.
* We built a simple baseline model for the multimodal entailment task. There are various
approaches that have been proposed to tackle the entailment problem.
[This presentation deck](https://docs.google.com/presentation/d/1mAB31BCmqzfedreNZYn4hsKPFmgHA9Kxz219DzyRY3c/edit?usp=sharing)
from the
[Recognizing Multimodal Entailment](https://multimodal-entailment.github.io/)
tutorial provides a comprehensive overview.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/multimodal-entailment)
and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/multimodal_entailment)

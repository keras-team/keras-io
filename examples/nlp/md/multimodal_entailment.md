# Learning Multimodal Entailment

**Author:** [Sayak Paul](https://twitter.com/RisingSayak)<br>
**Date created:** 2021/08/08<br>
**Last modified:** 2021/08/08<br>
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

---
## Imports


```python
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras
```

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
[Nilabhra Roy Chowdhury](https://de.linkedin.com/in/nilabhraroychowdhury) who worked preparing
the data.


```python
image_base_path = keras.utils.get_file(
    "tweet_images",
    "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/releases/download/v1.0.0/tweet_images.tar.gz",
    untar=True,
)
```

<div class="k-default-codeblock">
```
Downloading data from https://github.com/sayakpaul/Multimodal-Entailment-Baseline/releases/download/v1.0.0/tweet_images.tar.gz
344276992/344273442 [==============================] - 5s 0us/step

```
</div>
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
      <th>990</th>
      <td>1382932004736536577</td>
      <td>buy $IOTX on #binance @ 0.048767 https://t.co/...</td>
      <td>http://pbs.twimg.com/media/EzEpU4eWgAAn7Vp.jpg</td>
      <td>1383066642800902144</td>
      <td>sell $FUN on #binance @ 0.042262 https://t.co/...</td>
      <td>http://pbs.twimg.com/media/EzGjx5OXIAAJjk_.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>854</th>
      <td>1382134026144788481</td>
      <td>I am looking at the Unidentified COSMOS-F160W-...</td>
      <td>http://pbs.twimg.com/media/Ey5TkrTXIAMPns7.jpg</td>
      <td>1383045031125250048</td>
      <td>I am looking at the Unidentified COSMOS-F160W-...</td>
      <td>http://pbs.twimg.com/media/EzGQIJBWUAcGLhu.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1370198753240489985</td>
      <td>Top tweeps for #TBZ:\n1 @_tnr1\n2 @01_Z_B\n3 @...</td>
      <td>http://pbs.twimg.com/media/EwPsgTbUYAQ3mBE.jpg</td>
      <td>1373463499867758593</td>
      <td>Top tweeps for #TBZ:\n1 @JUTT__B\n2 @01_Z_B\n3...</td>
      <td>http://pbs.twimg.com/media/Ew-FxfyUUAEwe4L.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>1351</th>
      <td>1381256604926967813</td>
      <td>Finally completed the skin rendering. Will sta...</td>
      <td>http://pbs.twimg.com/media/Eys1j7NVIAgF-YF.jpg</td>
      <td>1381630932092784641</td>
      <td>Hair rendering. Will finish the hair by tomorr...</td>
      <td>http://pbs.twimg.com/media/EyyKAoaUUAElm-e.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>331</th>
      <td>1335817457018286080</td>
      <td>Nokia 5.4 full specifications leak online ahea...</td>
      <td>http://pbs.twimg.com/media/EonG2ybWEAEOjfo.png</td>
      <td>1335819879350321153</td>
      <td>Nokia 5.4 full specifications leak online ahea...</td>
      <td>http://pbs.twimg.com/media/EonJGCNUYAAMdRC.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>533</th>
      <td>1369797307256954882</td>
      <td>#NowWatching ANOTHER ROUND https://t.co/bU09LO...</td>
      <td>http://pbs.twimg.com/media/EwJ_Y0uXEAE51iq.jpg</td>
      <td>1373038097483624453</td>
      <td>#NowWatching Another Round https://t.co/ExMBaA...</td>
      <td>http://pbs.twimg.com/media/Ew4C30KWQBkZ-dA.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>1266</th>
      <td>1333010336971108353</td>
      <td>Daily almanac for November 29: https://t.co/NL...</td>
      <td>http://pbs.twimg.com/media/En_N1ieXMAEanWD.png</td>
      <td>1333010424627867653</td>
      <td>Daily almanac for November 29: https://t.co/XO...</td>
      <td>http://pbs.twimg.com/media/En_N6pnW4AYMDhW.png</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>804</th>
      <td>1354138253591502848</td>
      <td>Kuwait lists 505 new COVID-19 cases as infecti...</td>
      <td>http://pbs.twimg.com/media/EsrdiKBXYAAFZkK.jpg</td>
      <td>1362443439720325122</td>
      <td>#kuwait_moh Announce #وزارة_الصحة 979 new case...</td>
      <td>http://pbs.twimg.com/media/EuhfEfoXUAAjCBP.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>1019</th>
      <td>1381034028355227653</td>
      <td>My #RTRNaps are:\n\nO'Reilly @ 13:40\nHitman F...</td>
      <td>http://pbs.twimg.com/media/EyprIQFWgAANzsp.jpg</td>
      <td>1382154268338184195</td>
      <td>My #RTRNaps are:\n\nThe Big Bite @ 14:05\nCapt...</td>
      <td>http://pbs.twimg.com/media/Ey5l-1zWQAEFCwh.jpg</td>
      <td>NoEntailment</td>
    </tr>
    <tr>
      <th>1280</th>
      <td>1380242549336502274</td>
      <td>$NUVB held up pretty well today too https://t....</td>
      <td>http://pbs.twimg.com/media/EyebSFbU8AAhbid.jpg</td>
      <td>1380606940644995072</td>
      <td>$NUVB high of day near close. Nice. https://t....</td>
      <td>http://pbs.twimg.com/media/EyjmsfkU4AIdaIQ.jpg</td>
      <td>NoEntailment</td>
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


random_idx = np.random.choice(len(df))
visualize(random_idx)

random_idx = np.random.choice(len(df))
visualize(random_idx)
```


    
![png](/img/examples/nlp/multimodal_entailment/multimodal_entailment_14_0.png)
    


<div class="k-default-codeblock">
```
Text one: Join us for the Dell Power Cup!
```
</div>
    
<div class="k-default-codeblock">
```
https://t.co/kHsmpNJXjM
```
</div>
    
<div class="k-default-codeblock">
```
#dellpowercup https://t.co/e2qjwDbzad
Text two: Have you registered yet? Show your knowledge about Dell Technologies! You still have time to earn some extra points playing the individual games before the National Qualifiers start. CLICK TO REGISTER NOW! #ThePowerCup #DellPartner
 https://t.co/WdJ3xamoCP #Iwork4Dell https://t.co/FN4NMSzXE2
Label: NoEntailment

```
</div>
    
![png](/img/examples/nlp/multimodal_entailment/multimodal_entailment_14_2.png)
    


<div class="k-default-codeblock">
```
Text one: Do you know which hand wins?😆 Leave a comment and claim👇
```
</div>
    
<div class="k-default-codeblock">
```
Free chips🎁https://t.co/nfdHMPVkaA
```
</div>
    
<div class="k-default-codeblock">
```
#pokeronline #poker #pokerstar #pokerface #LosAngeles #PokerIndonesia #pokerplayer #LasVegas #casino #onlinecasino #macao #melbourne #포커 #parx #game #fun #home #WSOP #augupoker https://t.co/5ACLtIklSi
Text two: Do you know which hand wins?😆
```
</div>
    
<div class="k-default-codeblock">
```
Get it started👉 https://t.co/nfdHMPVkaA
```
</div>
    
<div class="k-default-codeblock">
```
#augupoker #bonus #rewards #pokerhands #PokerIndonesia https://t.co/g0mBOYHfpp
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
NoEntailment     1182
Implies           109
Contradictory     109
Name: label, dtype: int64

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

TensorFlow Hub provides
[variety of BERT family of models](https://www.tensorflow.org/text/tutorials/bert_glue#loading_models_from_tensorflow_hub).
Each of those models comes with a
corresponding preprocessing layer. You can learn more about these models and their
preprocessing layers from
[this resource](https://www.tensorflow.org/text/tutorials/bert_glue#loading_models_from_tensorflow_hub).

To keep the runtime of this example relatively short, we will use a smaller variant of
the original BERT model.


```python
# Define TF Hub paths to the BERT encoder and its preprocessor
bert_model_path = (
    "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1"
)
bert_preprocess_path = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
```

Our text preprocessing code mostly comes from
[this tutorial](https://www.tensorflow.org/text/tutorials/bert_glue).
You are highly encouraged to check out the tutorial to learn more about the input
preprocessing.


```python

def make_bert_preprocessing_model(sentence_features, seq_length=128):
    """Returns Model mapping string features to BERT inputs.

  Args:
    sentence_features: A list with the names of string-valued features.
    seq_length: An integer that defines the sequence length of BERT inputs.

  Returns:
    A Keras Model that can be called on a list or dict of string Tensors
    (with the order or names, resp., given by sentence_features) and
    returns a dict of tensors for input to BERT.
  """

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features
    ]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(bert_preprocess_path)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name="tokenizer")
    segments = [tokenizer(s) for s in input_segments]

    # Optional: Trim segments in a smart way to fit seq_length.
    # Simple cases (like this example) can skip this step and let
    # the next step apply a default truncation to approximately equal lengths.
    truncated_segments = segments

    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(
        bert_preprocess.bert_pack_inputs,
        arguments=dict(seq_length=seq_length),
        name="packer",
    )
    model_inputs = packer(truncated_segments)
    return keras.Model(input_segments, model_inputs)


bert_preprocess_model = make_bert_preprocessing_model(["text_1", "text_2"])
keras.utils.plot_model(bert_preprocess_model, show_shapes=True, show_dtype=True)
```




    
![png](/img/examples/nlp/multimodal_entailment/multimodal_entailment_22_0.png)
    



### Run the preprocessor on a sample input


```python
idx = np.random.choice(len(train_df))
row = train_df.iloc[idx]
sample_text_1, sample_text_2 = row["text_1"], row["text_2"]
print(f"Text 1: {sample_text_1}")
print(f"Text 2: {sample_text_2}")

test_text = [np.array([sample_text_1]), np.array([sample_text_2])]
text_preprocessed = bert_preprocess_model(test_text)

print("Keys           : ", list(text_preprocessed.keys()))
print("Shape Word Ids : ", text_preprocessed["input_word_ids"].shape)
print("Word Ids       : ", text_preprocessed["input_word_ids"][0, :16])
print("Shape Mask     : ", text_preprocessed["input_mask"].shape)
print("Input Mask     : ", text_preprocessed["input_mask"][0, :16])
print("Shape Type Ids : ", text_preprocessed["input_type_ids"].shape)
print("Type Ids       : ", text_preprocessed["input_type_ids"][0, :16])

```

<div class="k-default-codeblock">
```
Text 1: #pahrumpweather Friday
```
</div>
    
<div class="k-default-codeblock">
```
Sunny, with a high near 66. Northwest wind 11 to 14 mph, with gusts as high as 20 mph.
```
</div>
    
<div class="k-default-codeblock">
```
Tonight - Increasing clouds, with a low around 42. Northeast wind around 5 mph becoming calm. @visitpahrump @NWSVegas https://t.co/X2hNXwRxmf
Text 2: #pahrumpweather Wednesday 
```
</div>
    
<div class="k-default-codeblock">
```
Mostly sunny, with a high near 66. East southeast wind 5 to 7 mph becoming south in the afternoon.
```
</div>
    
<div class="k-default-codeblock">
```
Tonight - Mostly cloudy, with a low around 45. East southeast wind around 7 mph. @visitpahrump @NWSVegas https://t.co/2oQnaIZRvX
Keys           :  ['input_word_ids', 'input_type_ids', 'input_mask']
Shape Word Ids :  (1, 128)
Word Ids       :  tf.Tensor(
[  101  1001  6643  8093 24237 28949  5958 11559  1010  2007  1037  2152
  2379  5764  1012  4514], shape=(16,), dtype=int32)
Shape Mask     :  (1, 128)
Input Mask     :  tf.Tensor([1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1], shape=(16,), dtype=int32)
Shape Type Ids :  (1, 128)
Type Ids       :  tf.Tensor([0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], shape=(16,), dtype=int32)

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
bert_input_features = ["input_word_ids", "input_type_ids", "input_mask"]


def preprocess_image(image_path):
    extension = tf.strings.split(image_path)[-1]

    image = tf.io.read_file(image_path)
    if extension == b"jpg":
        image = tf.image.decode_jpeg(image, 3)
    else:
        image = tf.image.decode_png(image, 3)
    image = tf.image.resize(image, resize)
    return image


def preprocess_text(text_1, text_2):
    text_1 = tf.convert_to_tensor([text_1])
    text_2 = tf.convert_to_tensor([text_2])
    output = bert_preprocess_model([text_1, text_2])
    output = {feature: tf.squeeze(output[feature]) for feature in bert_input_features}
    return output


def preprocess_text_and_image(sample):
    image_1 = preprocess_image(sample["image_1_path"])
    image_2 = preprocess_image(sample["image_2_path"])
    text = preprocess_text(sample["text_1"], sample["text_2"])
    return {"image_1": image_1, "image_2": image_2, "text": text}

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


```python

def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = keras.layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = keras.layers.Dense(projection_dims)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = keras.layers.LayerNormalization()(x)
    return projected_embeddings


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


vision_encoder = create_vision_encoder(
    num_projection_layers=1, projection_dims=256, dropout_rate=0.1
)
keras.utils.plot_model(vision_encoder, show_shapes=True)


def create_text_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained BERT model to be used as the base encoder.
    bert = hub.KerasLayer(bert_model_path, name="bert",)
    # Set the trainability of the base encoder.
    bert.trainable = trainable

    # Receive the text as inputs.
    bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
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


text_encoder = create_text_encoder(
    num_projection_layers=1, projection_dims=256, dropout_rate=0.1
)
keras.utils.plot_model(text_encoder, show_shapes=True)


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
    bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
    text_inputs = {
        feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features
    }

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
    return keras.Model([image_1, image_2, text_inputs], outputs)


multimodal_model = create_multimodal_model()
keras.utils.plot_model(multimodal_model, show_shapes=True)
```

<div class="k-default-codeblock">
```
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
94674944/94668760 [==============================] - 1s 0us/step

```
</div>
    
![png](/img/examples/nlp/multimodal_entailment/multimodal_entailment_32_1.png)
    



You are encouraged to play with the different hyperparameters involved in building this
model and observe how the final performance is affected.

---
## Compile and train the model


```python
multimodal_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)

history = multimodal_model.fit(train_ds, validation_data=validation_ds, epochs=10)
```

<div class="k-default-codeblock">
```
Epoch 1/10
38/38 [==============================] - 40s 469ms/step - loss: 0.9932 - accuracy: 0.8279 - val_loss: 0.6666 - val_accuracy: 0.8571
Epoch 2/10
38/38 [==============================] - 3s 89ms/step - loss: 0.4068 - accuracy: 0.8772 - val_loss: 0.5720 - val_accuracy: 0.8571
Epoch 3/10
38/38 [==============================] - 3s 90ms/step - loss: 0.3627 - accuracy: 0.8864 - val_loss: 0.6029 - val_accuracy: 0.8571
Epoch 4/10
38/38 [==============================] - 3s 90ms/step - loss: 0.2852 - accuracy: 0.9006 - val_loss: 0.6908 - val_accuracy: 0.8571
Epoch 5/10
38/38 [==============================] - 3s 91ms/step - loss: 0.1701 - accuracy: 0.9474 - val_loss: 0.9603 - val_accuracy: 0.8413
Epoch 6/10
38/38 [==============================] - 3s 90ms/step - loss: 0.1029 - accuracy: 0.9666 - val_loss: 1.2907 - val_accuracy: 0.8730
Epoch 7/10
38/38 [==============================] - 3s 91ms/step - loss: 0.0712 - accuracy: 0.9791 - val_loss: 1.3902 - val_accuracy: 0.8571
Epoch 8/10
38/38 [==============================] - 3s 91ms/step - loss: 0.1101 - accuracy: 0.9607 - val_loss: 1.5143 - val_accuracy: 0.8413
Epoch 9/10
38/38 [==============================] - 3s 91ms/step - loss: 0.0916 - accuracy: 0.9691 - val_loss: 1.2580 - val_accuracy: 0.8095
Epoch 10/10
38/38 [==============================] - 3s 91ms/step - loss: 0.0595 - accuracy: 0.9799 - val_loss: 1.6426 - val_accuracy: 0.8413

```
</div>
---
## Evaluate the model


```python
_, acc = multimodal_model.evaluate(test_ds)
print(f"Accuracy on the test set: {round(acc * 100, 2)}%.")
```

<div class="k-default-codeblock">
```
5/5 [==============================] - 3s 538ms/step - loss: 1.2403 - accuracy: 0.8500
Accuracy on the test set: 85.0%.

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
|:---:  |:---:  |---    |---    |
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

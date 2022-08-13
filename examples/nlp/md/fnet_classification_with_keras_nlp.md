# Text Classification using FNet

**Author:** [Abheesht Sharma](https://github.com/abheesht17/)<br>
**Date created:** 2022/06/01<br>
**Last modified:** 2022/06/01<br>
**Description:** Text Classification on the IMDb Dataset using `keras_nlp.layers.FNetEncoder` layer.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/fnet_classification_with_keras_nlp.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/fnet_classification_with_keras_nlp.py)



---
## Introduction

In this example, we will demonstrate the ability of FNet to achieve comparable
results with a vanilla Transformer model on the text classification task.
We will be using the IMDb dataset, which is a
collection of movie reviews labelled either positive or negative (sentiment
analysis).

To build the tokenizer, model, etc., we will use components from
[KerasNLP](https://github.com/keras-team/keras-nlp). KerasNLP makes life easier
for people who want to build NLP pipelines! :)

### Model

Transformer-based language models (LMs) such as BERT, RoBERTa, XLNet, etc. have
demonstrated the effectiveness of the self-attention mechanism for computing
rich embeddings for input text. However, the self-attention mechanism is an
expensive operation, with a time complexity of `O(n^2)`, where `n` is the number
of tokens in the input. Hence, there has been an effort to reduce the time
complexity of the self-attention mechanism and improve performance without
sacrificing the quality of results.

In 2020, a paper titled
[FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)
replaced the self-attention layer in BERT with a simple Fourier Transform layer
for "token mixing". This resulted in comparable accuracy and a speed-up during
training. In particular, a couple of points from the paper stand out:

* The authors claim that FNet is 80% faster than BERT on GPUs and 70% faster on
TPUs. The reason for this speed-up is two-fold: a) the Fourier Transform layer
is unparametrized, it does not have any parameters, and b) the authors use Fast
Fourier Transform (FFT); this reduces the time complexity from `O(n^2)`
(in the case of self-attention) to `O(n log n)`.
* FNet manages to achieve 92-97% of the accuracy of BERT on the GLUE benchmark.

---
## Setup

Before we start with the implementation, let's install and import all the
necessary packages.


```python
!pip uninstall -y keras-nlp
!pip install -q git+https://github.com/keras-team/keras-nlp.git@fc01b18125bb707a0de7e442a6a8e48c8fe913a3
```

```python
import keras_nlp
import random
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

keras.utils.set_random_seed(42)
```
<div class="k-default-codeblock">
```
Found existing installation: keras-nlp 0.3.0
Uninstalling keras-nlp-0.3.0:
  Successfully uninstalled keras-nlp-0.3.0
  Building wheel for keras-nlp (setup.py) ... [?25l[?25hdone

```
</div>
Let's also define our hyperparameters.


```python
BATCH_SIZE = 64
EPOCHS = 3
MAX_SEQUENCE_LENGTH = 512
VOCAB_SIZE = 15000

EMBED_DIM = 128
INTERMEDIATE_DIM = 512
```

---
## Loading the dataset

First, let's download the IMDB dataset and extract it.


```python
!wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xzf aclImdb_v1.tar.gz
```

<div class="k-default-codeblock">
```
--2022-08-13 08:02:45--  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
Resolving ai.stanford.edu (ai.stanford.edu)... 171.64.68.10
Connecting to ai.stanford.edu (ai.stanford.edu)|171.64.68.10|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 84125825 (80M) [application/x-gzip]
Saving to: ‘aclImdb_v1.tar.gz’
```
</div>
    
<div class="k-default-codeblock">
```
aclImdb_v1.tar.gz   100%[===================>]  80.23M  15.8MB/s    in 8.3s    
```
</div>
    
<div class="k-default-codeblock">
```
2022-08-13 08:02:53 (9.68 MB/s) - ‘aclImdb_v1.tar.gz’ saved [84125825/84125825]
```
</div>
    


Samples are present in the form of text files. Let's inspect the structure of
the directory.


```python
print(os.listdir("./aclImdb"))
print(os.listdir("./aclImdb/train"))
print(os.listdir("./aclImdb/test"))
```

<div class="k-default-codeblock">
```
['test', 'imdbEr.txt', 'train', 'README', 'imdb.vocab']
['unsupBow.feat', 'unsup', 'labeledBow.feat', 'urls_neg.txt', 'neg', 'urls_unsup.txt', 'pos', 'urls_pos.txt']
['labeledBow.feat', 'urls_neg.txt', 'neg', 'pos', 'urls_pos.txt']

```
</div>
The directory contains two sub-directories: `train` and `test`. Each subdirectory
in turn contains two folders: `pos` and `neg` for positive and negative reviews,
respectively. Before we load the dataset, let's delete the `./aclImdb/train/unsup`
folder since it has unlabelled samples.


```python
!rm -rf aclImdb/train/unsup
```

We'll use the `keras.utils.text_dataset_from_directory` utility to generate
our labelled `tf.data.Dataset` dataset from text files.


```python
train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
)
val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
)
test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=BATCH_SIZE)
```

<div class="k-default-codeblock">
```
Found 25000 files belonging to 2 classes.
Using 20000 files for training.
Found 25000 files belonging to 2 classes.
Using 5000 files for validation.
Found 25000 files belonging to 2 classes.

```
</div>
We will now convert the text to lowercase.


```python
train_ds = train_ds.map(lambda x, y: (tf.strings.lower(x), y))
val_ds = val_ds.map(lambda x, y: (tf.strings.lower(x), y))
test_ds = test_ds.map(lambda x, y: (tf.strings.lower(x), y))
```

Let's print a few samples.


```python
for text_batch, label_batch in train_ds.take(1):
    for i in range(3):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])

```

<div class="k-default-codeblock">
```
b'an illegal immigrant resists the social support system causing dire consequences for many. well filmed and acted even though the story is a bit forced, yet the slow pacing really sets off the conclusion. the feeling of being lost in the big city is effectively conveyed. the little person lost in the big society is something to which we can all relate, but i cannot endorse going out of your way to see this movie.'
0
b"to get in touch with the beauty of this film pay close attention to the sound track, not only the music, but the way all sounds help to weave the imagery. how beautifully the opening scene leading to the expulsion of gino establishes the theme of moral ambiguity! note the way music introduces the characters as we are led inside giovanna's marriage. don't expect to find much here of the political life of italy in 1943. that's not what this is about. on the other hand, if you are susceptible to the music of images and sounds, you will be led into a word that reaches beyond neo-realism. by the end of the film we there are moments antonioni-like landscape that has more to do with the inner life of the characters than with real places. this is one of my favorite visconti films."
1
b'"hollywood hotel" has relationships to many films like "ella cinders" and "merton of the movies" about someone winning a contest including a contract to make films in hollywood, only to find the road to stardom either paved with pitfalls or non-existent. in fact, as i was watching it tonight, on turner classic movies, i was considering whether or not the authors of the later musical classic "singing in the rain" may have taken some of their ideas from "hollywood hotel", most notably a temperamental leading lady star in a movie studio and a conclusion concerning one person singing a film score while another person got the credit by mouthing along on screen.<br /><br />"hollywood hotel" is a fascinating example of movie making in the 1930s. among the supporting players is louella parsons, playing herself (and, despite some negative comments i\'ve seen, she has a very ingratiating personality on screen and a natural command of her lines). she is not the only real person in the script. make-up specialist perc westmore briefly appears as himself to try to make one character resemble another.<br /><br />this film also was one of the first in the career of young mr. ronald reagan, playing a radio interviewer at a movie premiere. reagan actually does quite nicely in his brief scenes - particularly when he realizes that nobody dick powell is about to take over the microphone when it should be used with more important people.<br /><br />dick powell has won a hollywood contract in a contest, and is leaving his job as a saxophonist in benny goodman\'s band. the beginning of this film, by the way, is quite impressive, as the band drives in a parade of trucks to give a proper goodbye to powell. they end up singing "hooray for hollywood". the interesting thing about this wonderful number is that a lyric has been left out on purpose. throughout the johnny mercer lyrics are references to such hollywood as max factor the make-up king, rin tin tin, and even a hint of tarzan. but the original song lyric referred to looking like tyrone power. obviously jack warner and his brothers were not going to advertise the leading man of 20th century fox, and the name donald duck was substituted. in any event the number showed the singers and instrumentalists of goodman\'s orchestra at their best. so did a later five minute section of the film, where the band is rehearsing.<br /><br />powell leaves the band and his girl friend (frances langford) and goes to hollywood, only to find he is a contract player (most likely for musicals involving saxophonists). he is met by allen joslyn, the publicist of the studio (the owner is grant mitchell). joslyn is not a bad fellow, but he is busy and he tends to slough off people unless it is necessary to speak to them. he parks powell at a room at the hollywood hotel, which is also where the studio\'s temperamental star (lola lane) lives with her father (hugh herbert), her sister (mabel todd), and her sensible if cynical assistant (glenda farrell). lane is like jean hagen in "singing in the rain", except her speaking voice is good. her version of "dan lockwood" is one "alexander dupre" (alan mowbray, scene stealing with ease several times). the only difference is that mowbray is not a nice guy like gene kelly was, and lane (when not wrapped up in her ego) is fully aware of it. having a fit on being by-passed for an out-of-the ordinary role she wanted, she refuses to attend the premiere of her latest film. joslyn finds a double for her (lola\'s real life sister rosemary lane), and rosemary is made up to play the star at the premiere and the follow-up party. but she attends with powell (joslyn wanting someone who doesn\'t know the real lola). this leads to powell knocking down mowbray when the latter makes a pest of himself. but otherwise the evening is a success, and when the two are together they start finding each other attractive.<br /><br />the complications deal with lola coming back and slapping powell in the face, after mowbray complains he was attacked by powell ("and his gang of hoodlums"). powell\'s contract is bought out. working with photographer turned agent ted healey (actually not too bad in this film - he even tries to do a jolson imitation at one point), the two try to find work, ending up as employees at a hamburger stand run by bad tempered edgar kennedy (the number of broken dishes and singing customers in the restaurant give edgar plenty of time to do his slow burns with gusto). eventually powell gets a "break" by being hired to be dupre\'s singing voice in a rip-off of "gone with the wind". this leads to the final section of the film, when rosemary lane, herbert, and healey help give powell his chance to show it\'s his voice, not mowbrays.<br /><br />it\'s quite a cute and appealing film even now. the worst aspects are due to it\'s time. several jokes concerning african-americans are no longer tolerable (while trying to photograph powell as he arrives in hollywood, healey accidentally photographs a porter, and mentions to joslyn to watch out, powell photographs too darkly - get the point?). also a bit with curt bois as a fashion designer for lola lane, who is (shall we say) too high strung is not very tolerable either. herbert\'s "hoo-hoo"ing is a bit much (too much of the time) but it was really popular in 1937. and an incident where healey nearly gets into a brawl at the premiere (this was one of his last films) reminds people of the tragic, still mysterious end of the comedian in december 1937. but most of the film is quite good, and won\'t disappoint the viewer in 2008.'
1

```
</div>
### Tokenizing the data

We'll be using the `keras_nlp.tokenizers.WordPieceTokenizer` layer to tokenize
the text. `keras_nlp.tokenizers.WordPieceTokenizer` takes a WordPiece vocabulary
and has functions for tokenizing the text, and detokenizing sequences of tokens.

Before we define the tokenizer, we first need to train it on the dataset
we have. The WordPiece tokenization algorithm is a subword tokenization algorithm;
training it on a corpus gives us a vocabulary of subwords. A subword tokenizer
is a compromise between word tokenizers (word tokenizers need very large
vocabularies for good coverage of input words), and character tokenizers
(characters don't really encode meaning like words do). The
`keras_nlp.tokenizers.compute_word_piece_vocabulary` utility function makes it
very simple to train WordPiece on a corpus.

Every vocabulary has a few special, reserved tokens. We have two such tokens:

- `"[PAD]"` - Padding token. Padding tokens are appended to the input sequence length
when the input sequence length is shorter than the maximum sequence length.
- `"[UNK]"` - Unknown token.

Note: The official implementation of FNet uses the SentencePiece Tokenizer.


```python
reserved_tokens = ["[PAD]", "[UNK]"]
train_sentences = train_ds.unbatch().map(lambda x, y: x)
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    data=train_sentences,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=reserved_tokens,
)
```

Let's see some tokens!


```python
print("Tokens: ", vocab[100:110])
```

<div class="k-default-codeblock">
```
Tokens:  ['ò', 'ó', 'ô', 'õ', 'ö', 'ø', 'ù', 'ú', 'û', 'ü']

```
</div>
Now, let's define the tokenizer. We will configure the tokenizer with the
the vocabularies trained above. We will define a maximum sequence length so that
all sequences are padded to the same length, if the length of the sequence is
less than the specified sequence length. Otherwise, the sequence is truncated.


```python
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=False,
    sequence_length=MAX_SEQUENCE_LENGTH,
)
```

Let's try and tokenize a sample from our dataset! To verify whether the text has
been tokenized correctly, we can also detokenize the list of tokens back to the
original text.


```python
input_sentence_ex = train_ds.take(1).get_single_element()[0][0]
input_tokens_ex = tokenizer(input_sentence_ex)

print("Sentence: ", input_sentence_ex)
print("Tokens: ", input_tokens_ex)
print("Recovered text after detokenizing: ", tokenizer.detokenize(input_tokens_ex))

```

<div class="k-default-codeblock">
```
Sentence:  tf.Tensor(b"the 60\xc2\xb4s is a well balanced mini series between historical facts and a good plot. in four deliveries, we follow a north american family, with 3 members. but we don't only see them. we also follow the story of several characters as a black reverend, an extremist student leader, and a soldier in vietnam. the filmography is just extraordinary. in the first chapters, we see some shots of the vietnam war, in between the scenes. the next chapter, doesn't start where the last one finished, it starts some time after, giving us a little mystery on what happened. in general, the 60\xc2\xb4s mini series, is a must see, not only for hippies fanatics, but for everyone with little curiosity about the topic.", shape=(), dtype=string)
Tokens:  tf.Tensor(
[  122     1   126    40   190  7681  2570   325   316  1607  2529   123
    40   168   229    15   129   813  1759   933    13   187   917    40
  2540   420   334    13   136    20  1213    15   137   187   207     8
    59   182   185   214    15   187   199   917   122   181   124   585
   221   133    40   450 12742    13   153  1365  9257  6956  1281  1669
  2304    13   123    40  1725   129  2729    15   122  7972   126   161
  3171    15   129   122   201 10238    13   187   185   167   795   124
   122  2729   451    13   129   316   122   261    15   122   494  5091
    13   270     8    59   507   237   122   359   147  1989    13   128
   639   167   175   219    13   872   302    40   235   861   139   166
   694    15   129   924    13   122     1  2570   325    13   126    40
   330   185    13   141   182   134  6889 10843    13   137   134   426
   136   235  3910   162   122  3282    15     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0], shape=(512,), dtype=int32)
Recovered text after detokenizing:  tf.Tensor(b"the [UNK] is a well balanced mini series between historical facts and a good plot . in four deliveries , we follow a north american family , with 3 members . but we don ' t only see them . we also follow the story of several characters as a black reverend , an extremist student leader , and a soldier in vietnam . the filmography is just extraordinary . in the first chapters , we see some shots of the vietnam war , in between the scenes . the next chapter , doesn ' t start where the last one finished , it starts some time after , giving us a little mystery on what happened . in general , the [UNK] mini series , is a must see , not only for hippies fanatics , but for everyone with little curiosity about the topic . [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]", shape=(), dtype=string)

```
</div>
---
## Formatting the dataset

Next, we'll format our datasets in the form that will be fed to the models. We
need to tokenize the text.


```python

def format_dataset(sentence, label):
    sentence = tokenizer(sentence)
    return ({"input_ids": sentence}, label)


def make_dataset(dataset):
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(512).prefetch(16).cache()


train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)
test_ds = make_dataset(test_ds)
```

---
## Building the model

Now, let's move on to the exciting part - defining our model!
We first need an embedding layer, i.e., a layer that maps every token in the input
sequence to a vector. This embedding layer can be initialised randomly. We also
need a positional embedding layer which encodes the word order in the sequence.
The convention is to add, i.e., sum, these two embeddings. KerasNLP has a
`keras_nlp.layers.TokenAndPositionEmbedding ` layer which does all of the above
steps for us.

Our FNet classification model consists of three `keras_nlp.layers.FNetEncoder`
layers with a `keras.layers.Dense` layer on top.

Note: For FNet, masking the padding tokens has a minimal effect on results. In the
official implementation, the padding tokens are not masked.


```python
input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)

x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)


x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

fnet_classifier = keras.Model(input_ids, outputs, name="fnet_classifier")
```

---
## Training our model

We'll use accuracy to monitor training progress on the validation data. Let's
train our model for 3 epochs.


```python
fnet_classifier.summary()
fnet_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
fnet_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
```

<div class="k-default-codeblock">
```
Model: "fnet_classifier"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_ids (InputLayer)      [(None, None)]            0         
                                                                 
 token_and_position_embeddin  (None, None, 128)        1985536   
 g (TokenAndPositionEmbeddin                                     
 g)                                                              
                                                                 
 f_net_encoder (FNetEncoder)  (None, None, 128)        132224    
                                                                 
 f_net_encoder_1 (FNetEncode  (None, None, 128)        132224    
 r)                                                              
                                                                 
 f_net_encoder_2 (FNetEncode  (None, None, 128)        132224    
 r)                                                              
                                                                 
 global_average_pooling1d (G  (None, 128)              0         
 lobalAveragePooling1D)                                          
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense (Dense)               (None, 1)                 129       
                                                                 
=================================================================
Total params: 2,382,337
Trainable params: 2,382,337
Non-trainable params: 0
_________________________________________________________________
Epoch 1/3
313/313 [==============================] - 34s 80ms/step - loss: 0.5810 - accuracy: 0.6431 - val_loss: 0.3597 - val_accuracy: 0.8434
Epoch 2/3
313/313 [==============================] - 23s 75ms/step - loss: 0.3081 - accuracy: 0.8674 - val_loss: 0.3300 - val_accuracy: 0.8584
Epoch 3/3
313/313 [==============================] - 24s 76ms/step - loss: 0.2055 - accuracy: 0.9208 - val_loss: 0.3944 - val_accuracy: 0.8456

<keras.callbacks.History at 0x7fe9f00804d0>

```
</div>
We obtain a train accuracy of around 92% and a validation accuracy of around
85%. Moreover, for 3 epochs, it takes around 86 seconds to train the model
(on Colab with a 16 GB Tesla T4 GPU).

Let's calculate the test accuracy.


```python
fnet_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)

```

<div class="k-default-codeblock">
```
391/391 [==============================] - 16s 26ms/step - loss: 0.4171 - accuracy: 0.8316

[0.4171285033226013, 0.83160001039505]

```
</div>
---
## Comparison with Transformer model

Let's compare our FNet Classifier model with a Transformer Classifier model. We
keep all the parameters/hyperparameters the same. For example, we use three
`TransformerEncoder` layers.

We set the number of heads to 2.


```python
NUM_HEADS = 2
input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")


x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(input_ids)

x = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
x = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
x = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)


x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

transformer_classifier = keras.Model(input_ids, outputs, name="transformer_classifier")


transformer_classifier.summary()
transformer_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
transformer_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
```

<div class="k-default-codeblock">
```
Model: "transformer_classifier"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_ids (InputLayer)      [(None, None)]            0         
                                                                 
 token_and_position_embeddin  (None, None, 128)        1985536   
 g_1 (TokenAndPositionEmbedd                                     
 ing)                                                            
                                                                 
 transformer_encoder (Transf  (None, None, 128)        198272    
 ormerEncoder)                                                   
                                                                 
 transformer_encoder_1 (Tran  (None, None, 128)        198272    
 sformerEncoder)                                                 
                                                                 
 transformer_encoder_2 (Tran  (None, None, 128)        198272    
 sformerEncoder)                                                 
                                                                 
 global_average_pooling1d_1   (None, 128)              0         
 (GlobalAveragePooling1D)                                        
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 2,580,481
Trainable params: 2,580,481
Non-trainable params: 0
_________________________________________________________________
Epoch 1/3
313/313 [==============================] - 50s 151ms/step - loss: 0.4522 - accuracy: 0.7678 - val_loss: 0.3437 - val_accuracy: 0.8570
Epoch 2/3
313/313 [==============================] - 46s 149ms/step - loss: 0.2061 - accuracy: 0.9218 - val_loss: 0.4238 - val_accuracy: 0.8472
Epoch 3/3
313/313 [==============================] - 47s 149ms/step - loss: 0.1633 - accuracy: 0.9403 - val_loss: 0.3789 - val_accuracy: 0.8746

<keras.callbacks.History at 0x7fe9fb4ec610>

```
</div>
We obtain a train accuracy of around 94% and a validation accuracy of around
86.5%. It takes around 146 seconds to train the model (on Colab with a 16 GB Tesla
T4 GPU).

Let's calculate the test accuracy.


```python
transformer_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)
```

<div class="k-default-codeblock">
```
391/391 [==============================] - 22s 56ms/step - loss: 0.4404 - accuracy: 0.8556

[0.4403669834136963, 0.8555999994277954]

```
</div>
Let's make a table and compare the two models. We can see that FNet
significantly speeds up our run time (1.7x), with only a small sacrifice in
overall accuracy (drop of 0.75%).

|                         | **FNet Classifier** | **Transformer Classifier** |
|:-----------------------:|:-------------------:|:--------------------------:|
|    **Training Time**    |      86 seconds     |         146 seconds        |
|    **Train Accuracy**   |        92.34%       |           93.85%           |
| **Validation Accuracy** |        85.21%       |           86.42%           |
|    **Test Accuracy**    |        83.94%       |           84.69%           |
|       **#Params**       |      2,321,921      |          2,520,065         |

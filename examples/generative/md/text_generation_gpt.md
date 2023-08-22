# GPT text generation from scratch with KerasNLP

**Author:** [Jesse Chan](https://github.com/jessechancy/)<br>
**Converted to Keras Core by:** [Anshuman Mishra](https://github.com/shivance)<br>
**Date created:** 2022/07/25<br>
**Last modified:** 2023/08/17<br>
**Description:** Using KerasNLP to train a mini-GPT model for text generation.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/text_generation_gpt.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/text_generation_gpt.py)



---
## Introduction

In this example, we will use KerasNLP to build a scaled down Generative
Pre-Trained (GPT) model. GPT is a Transformer-based model that allows you to generate
sophisticated text from a prompt.

We will train the model on the [simplebooks-92](https://arxiv.org/abs/1911.12391) corpus,
which is a dataset made from several novels. It is a good dataset for this example since
it has a small vocabulary and high word frequency, which is beneficial when training a
model with few parameters.

This example combines concepts from
[Text generation with a miniature GPT](https://keras.io/examples/generative/text_generation_with_miniature_gpt/)
with KerasNLP abstractions. We will demonstrate how KerasNLP tokenization, layers and
metrics simplify the training
process, and then show how to generate output text using the KerasNLP sampling utilities.


This examples uses [Keras Core](https://keras.io/keras_core/) to work in any of
`"tensorflow"`, `"jax"` or `"torch"`. Support for Keras Core is baked into
KerasNLP, simply change the `"KERAS_BACKEND"` environment variable to select
the backend of your choice. We select the tensorflow backend below, but feel free
to mix it up!


Note: If you are running this example on a Colab,
make sure to enable GPU runtime for faster training.

This example requires KerasNLP. You can install it via the following command:
`pip install keras-nlp`

---
## Setup


```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax" or "torch"

import keras_nlp
import keras_core as keras
import tensorflow as tf

```

<div class="k-default-codeblock">
```
Using TensorFlow backend

```
</div>
---
## Settings & hyperparameters


```python
# Data
BATCH_SIZE = 64
SEQ_LEN = 128
MIN_TRAINING_SEQ_LEN = 450

# Model
EMBED_DIM = 256
FEED_FORWARD_DIM = 256
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000  # Limits parameters in model.

# Training
EPOCHS = 6

# Inference
NUM_TOKENS_TO_GENERATE = 80
```

---
## Load the data

Now, let's download the dataset! The SimpleBooks dataset consists of 1,573 Gutenberg books, and has
one of the smallest vocabulary size to word-level tokens ratio. It has a vocabulary size of ~98k,
a third of WikiText-103's, with around the same number of tokens (~100M). This makes it easy to fit a small model.


```python
keras.utils.get_file(
    origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
    extract=True,
)
dir = os.path.expanduser("~/.keras/datasets/simplebooks/")

# Load simplebooks-92 train set and filter out short lines.
raw_train_ds = (
    tf.data.TextLineDataset(dir + "simplebooks-92-raw/train.txt")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

# Load simplebooks-92 validation set and filter out short lines.
raw_val_ds = (
    tf.data.TextLineDataset(dir + "simplebooks-92-raw/valid.txt")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
)
```

<div class="k-default-codeblock">
```
Downloading data from https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip
 282386239/282386239 ━━━━━━━━━━━━━━━━━━━━ 23s 0us/step

```
</div>
---
## Train the tokenizer

We train the tokenizer from the training dataset for a vocabulary size of `VOCAB_SIZE`,
which is a tuned hyperparameter. We want to limit the vocabulary as much as possible, as
we will see later on
that it has a large effect on the number of model parameters. We also don't want to include
*too few* vocabulary terms, or there would be too many out-of-vocabulary (OOV) sub-words. In
addition, three tokens are reserved in the vocabulary:

- `"[PAD]"` for padding sequences to `SEQ_LEN`. This token has index 0 in both
`reserved_tokens` and `vocab`, since `WordPieceTokenizer` (and other layers) consider
`0`/`vocab[0]` as the default padding.
- `"[UNK]"` for OOV sub-words, which should match the default `oov_token="[UNK]"` in
`WordPieceTokenizer`.
- `"[BOS]"` stands for beginning of sentence, but here technically it is a token
representing the beginning of each line of training data.


```python
# Train tokenizer vocabulary
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)
```

---
## Load tokenizer

We use the vocabulary data to initialize
`keras_nlp.tokenizers.WordPieceTokenizer`. WordPieceTokenizer is an efficient
implementation of the WordPiece algorithm used by BERT and other models. It will strip,
lower-case and do other irreversible preprocessing operations.


```python
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)
```

---
## Tokenize data

We preprocess the dataset by tokenizing and splitting it into `features` and `labels`.


```python
# packer adds a start token
start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)


def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels


# Tokenize and split into train and label sequences.
train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
```

---
## Build the model

We create our scaled down GPT model with the following layers:

- One `keras_nlp.layers.TokenAndPositionEmbedding` layer, which combines the embedding
for the token and its position.
- Multiple `keras_nlp.layers.TransformerDecoder` layers, with the default causal masking.
The layer has no cross-attention when run with decoder sequence only.
- One final dense linear layer


```python
inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)
# Embedding.
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)
x = embedding_layer(inputs)
# Transformer decoders.
for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM,
    )
    x = decoder_layer(x)  # Giving one argument only skips cross-attention.
# Output.
outputs = keras.layers.Dense(VOCAB_SIZE)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])
```

---
## Training

Now that we have our model, let's train it with the `fit()` method.


```python
model.fit(train_ds, validation_data=val_ds, verbose=2, epochs=EPOCHS)
```

<div class="k-default-codeblock">
```
Epoch 1/6

/usr/local/lib/python3.10/dist-packages/keras_core/src/layers/layer.py:759: UserWarning: Layer 'position_embedding' (of type PositionEmbedding) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/keras_core/src/layers/layer.py:759: UserWarning: Layer 'query' (of type EinsumDense) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/keras_core/src/layers/layer.py:759: UserWarning: Layer 'key' (of type EinsumDense) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/keras_core/src/layers/layer.py:759: UserWarning: Layer 'value' (of type EinsumDense) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
  warnings.warn(
/usr/lib/python3.10/contextlib.py:153: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self.gen.throw(typ, value, traceback)

3169/3169 - 181s - 57ms/step - loss: 0.0362 - perplexity: 103.1788 - val_loss: 0.0323 - val_perplexity: 63.1814
Epoch 2/6
3169/3169 - 155s - 49ms/step - loss: 0.0319 - perplexity: 59.9559 - val_loss: 0.0314 - val_perplexity: 56.4214
Epoch 3/6
3169/3169 - 154s - 49ms/step - loss: 0.0310 - perplexity: 53.0826 - val_loss: 0.0308 - val_perplexity: 51.8652
Epoch 4/6
3169/3169 - 202s - 64ms/step - loss: 0.0305 - perplexity: 49.5741 - val_loss: 0.0305 - val_perplexity: 49.8602
Epoch 5/6
3169/3169 - 202s - 64ms/step - loss: 0.0301 - perplexity: 47.2358 - val_loss: 0.0303 - val_perplexity: 48.8951
Epoch 6/6
3169/3169 - 203s - 64ms/step - loss: 0.0298 - perplexity: 45.6656 - val_loss: 0.0300 - val_perplexity: 47.2041

<keras_core.src.callbacks.history.History at 0x7819a7f9ae90>

```
</div>
---
## Inference

With our trained model, we can test it out to gauge its performance. To do this
we can seed our model with an input sequence starting with the `"[BOS]"` token,
and progressively sample the model by making predictions for each subsequent
token in a loop.

To start lets build a prompt with the same shape as our model inputs, containing
only the `"[BOS]"` token.


```python
# The "packer" layers adds the [BOS] token for us.
prompt_tokens = start_packer(tokenizer([""]))
prompt_tokens
```




<div class="k-default-codeblock">
```
<tf.Tensor: shape=(1, 128), dtype=int32, numpy=
array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>

```
</div>
We will use the `keras_nlp.samplers` module for inference, which requires a
callback function wrapping the model we just trained. This wrapper calls
the model and returns the logit predictions for the current token we are
generating.

Note: There are two pieces of more advanced functionality available when
defining your callback. The first is the ability to take in a `cache` of states
computed in previous generation steps, which can be used to speed up generation.
The second is the ability to output the final dense "hidden state" of each
generated token. This is used by `keras_nlp.samplers.ContrastiveSampler`, which
avoids repetition by penalizing repeated hidden states. Both are optional, and
we will ignore them for now.


```python

def next(prompt, cache, index):
    logits = model(prompt)[:, index - 1, :]
    # Ignore hidden states for now; only needed for contrastive search.
    hidden_states = None
    return logits, hidden_states, cache

```

Creating the wrapper function is the most complex part of using these functions. Now that
it's done, let's test out the different utilities, starting with greedy search.

### Greedy search

We greedily pick the most probable token at each timestep. In other words, we get the
argmax of the model output.


```python
sampler = keras_nlp.samplers.GreedySampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,  # Start sampling immediately after the [BOS] token.
)
txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")
```

<div class="k-default-codeblock">
```
Greedy search generated text: 
[b'[BOS] " i have been thinking of the matter , " he said , " but i have been thinking of that , and i have been thinking of it , and i have been thinking of it . i have been thinking of it , and i have been thinking of it , and i have been thinking of it , and i have been thinking of it . i have been thinking of it , and i have been thinking of it , and i have been thinking of it , and that it was a very good thing to be done . i have been thinking of it , and i have been thinking of it , and i have been thinking of it ,']
```
</div>
    


As you can see, greedy search starts out making some sense, but quickly starts repeating
itself. This is a common problem with text generation that can be fixed by some of the
probabilistic text generation utilities shown later on!

### Beam search

At a high-level, beam search keeps track of the `num_beams` most probable sequences at
each timestep, and predicts the best next token from all sequences. It is an improvement
over greedy search since it stores more possibilities. However, it is less efficient than
greedy search since it has to compute and store multiple potential sequences.

**Note:** beam search with `num_beams=1` is identical to greedy search.


```python
sampler = keras_nlp.samplers.BeamSampler(num_beams=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Beam search generated text: \n{txt}\n")
```

<div class="k-default-codeblock">
```
Beam search generated text: 
[b'[BOS] " i \' ll tell you , " he said . " i \' ll tell you what i \' ll do . i \' ll tell you about it , and i \' ll tell you about it . i \' ll tell you about it , and i \' ll tell you about it . i \' ll tell you about it , and i \' ll tell you about it . i \' ll tell you about it , and i \' ll tell you about it . i \' ll tell you about it , and i \' ll tell you about it . i \' ll tell you about it . i \' ll tell you about it . i \' ll']
```
</div>
    


Similar to greedy search, beam search quickly starts repeating itself, since it is still
a deterministic method.

### Random search

Random search is our first probabilistic method. At each time step, it samples the next
token using the softmax probabilities provided by the model.


```python
sampler = keras_nlp.samplers.RandomSampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Random search generated text: \n{txt}\n")
```

<div class="k-default-codeblock">
```
Random search generated text: 
[b"[BOS] at the end of this he drew the girl to her prayer . he began to read and write out the words out of the read . at rosalind it till she made everything something of it and said it soon felt signor and needed her . it was the same . ' she was not going to keep off and guarded it . nor ' have she wanted to know that she would want him to go ahead , for she would not find herself in point of speed , but so chicago . the sentry was trotting up , also a gnomet thought everybody would break out of her mind when the puzzled"]
```
</div>
    


Voilà, no repetitions! However, with random search, we may see some nonsensical words
appearing since any word in the vocabulary has a chance of appearing with this sampling
method. This is fixed by our next search utility, top-k search.

### Top-K search

Similar to random search, we sample the next token from the probability distribution
provided by the model. The only difference is that here, we select out the top `k` most
probable tokens, and distribute the probability mass over them before sampling. This way,
we won't be sampling from low probability tokens, and hence we would have less
nonsensical words!


```python
sampler = keras_nlp.samplers.TopKSampler(k=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-K search generated text: \n{txt}\n")
```

<div class="k-default-codeblock">
```
Top-K search generated text: 
[b"[BOS] the shif ' less of the spaniards was not so easily pleased , but he took up a large prophet with the other . the scotrocluctance was so long that his mother would not have a good chance to go back to appeal , and to make a very happy time , when he saw the old woman ' s face , and she was so much younger than she was . he was not quite able to make it a good deal more for her to be affective . he was too much afraid of the girl and the rest"]
```
</div>
    


### Top-P search

Even with the top-k search, there is something to improve upon. With top-k search, the
number `k` is fixed, which means it selects the same number of tokens for any probability
distribution. Consider two scenarios, one where the probability mass is concentrated over
2 words and another where the probability mass is evenly concentrated across 10. Should
we choose `k=2` or `k=10`? There is no one size that fits all `k` here.

This is where top-p search comes in! Instead of choosing a `k`, we choose a probability
`p` that we want the probabilities of the top tokens to sum up to. This way, we can
dynamically adjust the `k` based on the probability distribution. By setting `p=0.9`, if
90% of the probability mass is concentrated on the top 2 tokens, we can filter out the
top 2 tokens to sample from. If instead the 90% is distributed over 10 tokens, it will
similarly filter out the top 10 tokens to sample from.


```python
sampler = keras_nlp.samplers.TopPSampler(p=0.5)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-P search generated text: \n{txt}\n")
```

<div class="k-default-codeblock">
```
Top-P search generated text: 
[b'[BOS] the two scottish knights were already able to meet with a number of french knights , and were , of course , as they were going to the south , they would have to march across to la . the english knights were to be in readiness for the attack of the british . they had had the interview with the french . the three had been killed by the french . the others had been killed , and they had been killed by the party of the portuguese , who had been wounded , and had been taken prisoner in the battle . they had , however , been killed by the spaniards , and they had killed many of the']
```
</div>
    


### Using callbacks for text generation

We can also wrap the utilities in a callback, which allows you to print out a prediction
sequence for every epoch of the model! Here is an example of a callback for top-k search:


```python

class TopKTextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model using top-k."""

    def __init__(self, k):
        self.sampler = keras_nlp.samplers.TopKSampler(k)

    def on_epoch_end(self, epoch, logs=None):
        output_tokens = self.sampler(
            next=next,
            prompt=prompt_tokens,
            index=1,
        )
        txt = tokenizer.detokenize(output_tokens)
        print(f"Top-K search generated text: \n{txt}\n")


text_generation_callback = TopKTextGenerator(k=10)
# Dummy training loop to demonstrate callback.
model.fit(train_ds.take(1), verbose=2, epochs=2, callbacks=[text_generation_callback])
```

<div class="k-default-codeblock">
```
Epoch 1/2
Top-K search generated text: 
[b'[BOS] when he came to a little brook of the river , he said that he would never have done anything to him , and he would not have had the same time when he came home . they did not know , but that he was a great one , and the boys did not think he had any more . the boys were afraid of the fish . the birds sang and their song , and they heard it singing as if it were not long in the distance . they were singing and singing in silence . they heard the song and song , and all their music songs were singing , but they were glad of it . [PAD] was it so sweetly']
```
</div>
    
<div class="k-default-codeblock">
```
1/1 - 13s - 13s/step - loss: 0.0295 - perplexity: 43.5042
Epoch 2/2
Top-K search generated text: 
[b"[BOS] the young prince , who was a man of very tall youth , was very proud . it was an excellent fellow , but his father was very strong , and he was very fond of him , and had to do so , and a little boy , and so the two boys , that they had a little time to do that . he was very fond of him , and he was always in the habit of keeping his mother ' s mind and he was so kind to him . and when mr . treascoming went to the village of the village called his sister , and he had to do this house ; and he was in a"]
```
</div>
    
<div class="k-default-codeblock">
```
1/1 - 14s - 14s/step - loss: 0.0281 - perplexity: 36.7171

<keras_core.src.callbacks.history.History at 0x7819a21b8610>

```
</div>
---
## Conclusion

To recap, in this example, we use KerasNLP layers to train a sub-word vocabulary,
tokenize training data, create a miniature GPT model, and perform inference with the
text generation library.

If you would like to understand how Transformers work, or learn more about training the
full GPT model, here are some further readings:

- Attention Is All You Need [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- GPT-3 Paper [Brown et al., 2020](https://arxiv.org/abs/2005.14165)

# GPT text generation from scratch with KerasNLP

**Author:** [Jesse Chan](https://github.com/jessechancy)<br>
**Date created:** 2022/07/25<br>
**Last modified:** 2022/07/25<br>
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

Note: If you are running this example on a Colab,
make sure to enable GPU runtime for faster training.

This example requires KerasNLP. You can install it via the following command:
`pip install keras-nlp`

---
## Setup


```python
import os
import keras_nlp
import tensorflow as tf
from tensorflow import keras
```

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

---
## Train the tokenizer

We train the tokenizer from the training dataset for a vocabulary size of `VOCAB_SIZE`,
which is a tuned hyperparameter. We want to limit the vocabulary as much as possible, as
we will see later on
that it has a large affect on the number of model parameters. We also don't want to include
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
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])
```

Let's take a look at our model summary - a large majority of the
parameters are in the `token_and_position_embedding` and the output `dense` layer!
This means that the vocabulary size (`VOCAB_SIZE`) has a large affect on the size of the model,
while the number of Transformer decoder layers (`NUM_LAYERS`) doesn't affect it as much.


```python
model.summary()
```

<div class="k-default-codeblock">
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, None)]            0         
                                                                 
 token_and_position_embeddin  (None, None, 256)        1312768   
 g (TokenAndPositionEmbeddin                                     
 g)                                                              
                                                                 
 transformer_decoder (Transf  (None, None, 256)        394749    
 ormerDecoder)                                                   
                                                                 
 transformer_decoder_1 (Tran  (None, None, 256)        394749    
 sformerDecoder)                                                 
                                                                 
 dense (Dense)               (None, None, 5000)        1285000   
                                                                 
=================================================================
Total params: 3,387,266
Trainable params: 3,387,266
Non-trainable params: 0
_________________________________________________________________

```
</div>
---
## Training

Now that we have our model, let's train it with the `fit()` method.


```python
model.fit(train_ds, validation_data=val_ds, verbose=2, epochs=EPOCHS)
```

<div class="k-default-codeblock">
```
Epoch 1/6
3169/3169 - 132s - loss: 4.5592 - perplexity: 95.8829 - val_loss: 4.1382 - val_perplexity: 63.2792 - 132s/epoch - 42ms/step
Epoch 2/6
3169/3169 - 63s - loss: 4.0597 - perplexity: 58.1860 - val_loss: 4.0272 - val_perplexity: 56.6228 - 63s/epoch - 20ms/step
Epoch 3/6
3169/3169 - 64s - loss: 3.9437 - perplexity: 51.8076 - val_loss: 3.9825 - val_perplexity: 54.1286 - 64s/epoch - 20ms/step
Epoch 4/6
3169/3169 - 64s - loss: 3.8803 - perplexity: 48.6225 - val_loss: 3.9078 - val_perplexity: 50.1429 - 64s/epoch - 20ms/step
Epoch 5/6
3169/3169 - 64s - loss: 3.8357 - perplexity: 46.5021 - val_loss: 3.8531 - val_perplexity: 47.4559 - 64s/epoch - 20ms/step
Epoch 6/6
3169/3169 - 64s - loss: 3.8020 - perplexity: 44.9577 - val_loss: 3.8446 - val_perplexity: 47.1300 - 64s/epoch - 20ms/step

<keras.callbacks.History at 0x7f414008a970>

```
</div>
---
## Inference

With our trained model, we can test it out to gauge it's performance. To do this
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
it's done, let's test out the different utilties, starting with greedy search.

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
[b'[BOS] " i have been a good deal of trouble , " the captain said , " but i have been a good deal more than i have been when i have been a good deal worse than i have been . i have been a good deal worse than i have been when i have been a boy , and have been a boy , and i have been a boy , and have been a boy , and have been a boy , and i have been a boy , and i have been a boy , and i have been a boy , and i have been a boy , and i have been a boy , and i have been a boy , and']
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
[b'[BOS] " well , i don \' t know what to do , " he said . " i don \' t know what to do . i don \' t know what to do , but i don \' t know what to do . i don \' t know what to do . i don \' t know what to do , but i don \' t know what to do . i don \' t know what to do , but i don \' t know what to do . i don \' t know what to do , but i don \' t know what to do . i don \' t know what to do , but i don \' t want to']
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
[b"[BOS] the boat was convoying in the hand and mercer , there has been room at home . ages before miss m ' r smiling , miss penfold , edith , and miss penfold had endeavored to stare icy . the principal had mind to set his own house to fetch two children besides , and helen was as hot as savage as you allen . we were only as silly at large an honest site - looking man back with thirts the gate , and this was a neat tower , and it was full of calm - faced than once upon the g"]
```
</div>
    


Voila, no repetitions! However, with random search, we may see some nonsensical words
appearing since any word in the vocabulary has a chance of appearing with this sampling
method. This is fixed by our next search utility, top-k search.

### Top-K search

Similar to random search, we sample the next token from the probability distribution
provided by the model. The only difference is that here, we select out the top `k` most
probable tokens, and distribute the probabiltiy mass over them before sampling. This way,
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
[b'[BOS] " that is , indeed , " she said ; " i think the saxon is to be sublican , as a matter to the french . the french will be able to give up as many of us as you can carry off the french . if you can see a french force at a time they will be in the first place . but now , as they are , i think it is , you know , we must not think of doing it , and , in the course of time the fight , it is as likely , and i will not do so , as you would have done , but as you see that']
```
</div>
    


### Top-P search

Even with the top-k search, there is something to improve upon. With top-k search, the
number `k` is fixed, which means it selects the same number of tokens for any probability
distribution. Consider two scenarios, one where the probability mass is concentrated over
2 words and another where the probability mass is evenly concentrated across 10. Should
we choose `k=2` or `k=10`? There is not a one size fits all `k` here.

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
[b'[BOS] " i don \' t know what to do , " said the gracious discussion . " it was the general of the conde , that i was , at least , and i was going to have arrested , and he had just got a little idea of the situation . it was a man , but i had not told him , but he was as good as his head , and i could see the way of a chatterer , and had the first thing i thought of was the old man who had no fear of any prosperity . " [PAD] , i know , that']
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
[b'[BOS] " but now i am glad to find your mother in this state of the situation , sir . i am sure of the chum of your own . i have come to ask you to take your own place , and that is why i must go to the king . i think that it will be very good and kind of positive and inquisitive as i have done . i am glad , however , that i may be glad to see that you are not very likely to have some money with me ; and i have never seen before , as i am sure that you are not very sure i should go to school']
```
</div>
    
<div class="k-default-codeblock">
```
1/1 - 4s - loss: 3.8255 - perplexity: 45.9749 - 4s/epoch - 4s/step
Epoch 2/2
Top-K search generated text: 
[b'[BOS] " well , sir , the admiral was a man of the king , and his men would , if a man would say that the whole army of his men would be in charge , but it would be , in all haste to take his place , and be on arriving at brussels . it would be a hard thing to be done , if it were to be a great , and the country would be in the best of alleville , and the army would be in no hurry , for that country was in no way a little way to make the country road . [PAD] , in all respects the french , the country']
```
</div>
    
<div class="k-default-codeblock">
```
1/1 - 4s - loss: 3.5802 - perplexity: 35.9931 - 4s/epoch - 4s/step

<keras.callbacks.History at 0x7f412bfff8e0>

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

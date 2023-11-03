# Sequence to sequence learning for performing number addition

**Author:** [Smerity](https://twitter.com/Smerity) and others<br>
**Date created:** 2015/08/17<br>
**Last modified:** 2020/04/17<br>
**Description:** A model that learns to add strings of numbers, e.g. "535+61" -> "596".


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/addition_rnn.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/nlp/addition_rnn.py)



---
## Introduction

In this example, we train a model to learn to add two numbers, provided as strings.

**Example:**

- Input: "535+61"
- Output: "596"

Input may optionally be reversed, which was shown to increase performance in many tasks
 in: [Learning to Execute](http://arxiv.org/abs/1410.4615) and
[Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf).

Theoretically, sequence order inversion introduces shorter term dependencies between
 source and target for this problem.

**Results:**

For two digits (reversed):

+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs

Three digits (reversed):

+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

Four digits (reversed):

+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs

Five digits (reversed):

+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs

---
## Setup


```python
import keras
from keras import layers
import numpy as np

# Parameters for the model and dataset.
TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True

# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS
```

---
## Generate the data


```python

class CharacterTable:
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.
        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)


# All the numbers, plus sign and space for padding.
chars = "0123456789+ "
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()
print("Generating data...")
while len(questions) < TRAINING_SIZE:
    f = lambda: int(
        "".join(
            np.random.choice(list("0123456789"))
            for i in range(np.random.randint(1, DIGITS + 1))
        )
    )
    a, b = f(), f()
    # Skip any addition questions we've already seen
    # Also skip any such that x+Y == Y+x (hence the sorting).
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN.
    q = "{}+{}".format(a, b)
    query = q + " " * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1.
    ans += " " * (DIGITS + 1 - len(ans))
    if REVERSE:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
        # space used for padding.)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print("Total questions:", len(questions))
```

<div class="k-default-codeblock">
```
Generating data...
Total questions: 50000

```
</div>
---
## Vectorize the data


```python
print("Vectorization...")
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print("Training Data:")
print(x_train.shape)
print(y_train.shape)

print("Validation Data:")
print(x_val.shape)
print(y_val.shape)
```

<div class="k-default-codeblock">
```
Vectorization...
Training Data:
(45000, 7, 12)
(45000, 4, 12)
Validation Data:
(5000, 7, 12)
(5000, 4, 12)

```
</div>
---
## Build the model


```python
print("Build model...")
num_layers = 1  # Try to add more LSTM layers!

model = keras.Sequential()
# "Encode" the input sequence using a LSTM, producing an output of size 128.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(layers.Input((MAXLEN, len(chars))))
model.add(layers.LSTM(128))
# As the decoder RNN's input, repeatedly provide with the last output of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(num_layers):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(layers.LSTM(128, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.Dense(len(chars), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
```

<div class="k-default-codeblock">
```
Build model...

```
</div>
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">    Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)               │     <span style="color: #00af00; text-decoration-color: #00af00">72,192</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ repeat_vector (<span style="color: #0087ff; text-decoration-color: #0087ff">RepeatVector</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ lstm_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │    <span style="color: #00af00; text-decoration-color: #00af00">131,584</span> │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>)             │      <span style="color: #00af00; text-decoration-color: #00af00">1,548</span> │
└─────────────────────────────────┴───────────────────────────┴────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">205,324</span> (802.05 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">205,324</span> (802.05 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



---
## Train the model


```python
epochs = 30
batch_size = 32


# Train the model each generation and show predictions against the validation
# dataset.
for epoch in range(1, epochs):
    print()
    print("Iteration", epoch)
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=1,
        validation_data=(x_val, y_val),
    )
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = np.argmax(model.predict(rowx), axis=-1)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print("Q", q[::-1] if REVERSE else q, end=" ")
        print("T", correct, end=" ")
        if correct == guess:
            print("☑ " + guess)
        else:
            print("☒ " + guess)
```

    
<div class="k-default-codeblock">
```
Iteration 1
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 4ms/step - accuracy: 0.3224 - loss: 1.8852 - val_accuracy: 0.4116 - val_loss: 1.5662
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 386ms/step
Q 986+63  T 1049 ☒ 903 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 905+57  T 962  ☒ 901 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 942us/step
Q 649+29  T 678  ☒ 606 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 924us/step
Q 53+870  T 923  ☒ 881 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 444+283 T 727  ☒ 513 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 29+601  T 630  ☒ 201 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 44+618  T 662  ☒ 571 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 941us/step
Q 73+989  T 1062 ☒ 906 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 858us/step
Q 108+928 T 1036 ☒ 103 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 206+61  T 267  ☒ 276 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 2
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.4672 - loss: 1.4301 - val_accuracy: 0.5708 - val_loss: 1.1566
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 34+611  T 645  ☒ 651 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 41+657  T 698  ☒ 619 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 28+461  T 489  ☒ 591 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 6+114   T 120  ☒ 121 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 80+42   T 122  ☒ 131 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 123+47  T 170  ☒ 175 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 563+540 T 1103 ☒ 1019
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 8+937   T 945  ☒ 960 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 3+568   T 571  ☒ 570 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 32+771  T 803  ☒ 819 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 3
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.5986 - loss: 1.0737 - val_accuracy: 0.6534 - val_loss: 0.9349
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 976us/step
Q 521+5   T 526  ☒ 524 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 4+250   T 254  ☒ 256 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 927us/step
Q 467+74  T 541  ☒ 542 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 584+5   T 589  ☒ 580 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 434+99  T 533  ☒ 526 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 549+8   T 557  ☒ 552 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 610+870 T 1480 ☒ 1376
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 828+969 T 1797 ☒ 1710
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 968+0   T 968  ☒ 969 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 629+654 T 1283 ☒ 1275
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 4
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.6779 - loss: 0.8780 - val_accuracy: 0.7011 - val_loss: 0.7973
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 12+647  T 659  ☒ 657 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 889us/step
Q 82+769  T 851  ☒ 857 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 975us/step
Q 79+412  T 491  ☑ 491 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 973us/step
Q 712+31  T 743  ☒ 745 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 37+37   T 74   ☒ 73  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 576+28  T 604  ☒ 607 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 916us/step
Q 39+102  T 141  ☒ 140 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 922us/step
Q 649+472 T 1121 ☒ 1111
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 563+540 T 1103 ☒ 1100
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 86+391  T 477  ☑ 477 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 5
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.7195 - loss: 0.7628 - val_accuracy: 0.7436 - val_loss: 0.6989
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 816+39  T 855  ☒ 859 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 27+99   T 126  ☒ 123 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 975us/step
Q 871+98  T 969  ☒ 965 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 966us/step
Q 394+10  T 404  ☒ 409 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 969us/step
Q 63+63   T 126  ☒ 129 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 78+334  T 412  ☒ 419 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 957us/step
Q 112+4   T 116  ☑ 116 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 911us/step
Q 990+37  T 1027 ☒ 1029
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 882us/step
Q 75+63   T 138  ☒ 139 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 871us/step
Q 38+481  T 519  ☑ 519 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 6
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.7527 - loss: 0.6754 - val_accuracy: 0.7705 - val_loss: 0.6260
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 475+21  T 496  ☒ 497 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 324+76  T 400  ☑ 400 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 479+385 T 864  ☒ 867 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 892us/step
Q 36+213  T 249  ☒ 247 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 671+259 T 930  ☒ 934 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 3+10    T 13   ☒ 20  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 86+319  T 405  ☑ 405 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 63+63   T 126  ☒ 127 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 459+833 T 1292 ☒ 1390
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 991us/step
Q 17+465  T 482  ☒ 489 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 7
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.7927 - loss: 0.5712 - val_accuracy: 0.8573 - val_loss: 0.3966
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 965us/step
Q 43+945  T 988  ☑ 988 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 821us/step
Q 371+96  T 467  ☒ 468 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step  
Q 873+894 T 1767 ☒ 1768
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 912us/step
Q 117+82  T 199  ☑ 199 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 978us/step
Q 25+95   T 120  ☒ 110 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 958us/step
Q 26+99   T 125  ☑ 125 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 829us/step
Q 29+6    T 35   ☒ 34  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 977us/step
Q 857+81  T 938  ☒ 939 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 972us/step
Q 668+97  T 765  ☑ 765 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 916us/step
Q 85+903  T 988  ☑ 988 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 8
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8902 - loss: 0.3306 - val_accuracy: 0.9228 - val_loss: 0.2472
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 21+478  T 499  ☑ 499 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 978us/step
Q 795+16  T 811  ☑ 811 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 97+616  T 713  ☑ 713 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 889+342 T 1231 ☒ 1221
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 266+274 T 540  ☒ 530 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 975us/step
Q 751+830 T 1581 ☑ 1581
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 929us/step
Q 674+3   T 677  ☑ 677 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 902+167 T 1069 ☒ 1068
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 505+1   T 506  ☑ 506 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 955us/step
Q 944+775 T 1719 ☑ 1719
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 9
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9536 - loss: 0.1820 - val_accuracy: 0.9665 - val_loss: 0.1333
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 679+61  T 740  ☑ 740 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 921+49  T 970  ☑ 970 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 863+16  T 879  ☑ 879 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 27+560  T 587  ☑ 587 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 34+941  T 975  ☑ 975 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 7+278   T 285  ☑ 285 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step  
Q 165+43  T 208  ☑ 208 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step
Q 695+44  T 739  ☑ 739 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 25+165  T 190  ☑ 190 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step  
Q 34+184  T 218  ☑ 218 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 10
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9739 - loss: 0.1127 - val_accuracy: 0.9774 - val_loss: 0.0889
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 79+85   T 164  ☑ 164 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 816+353 T 1169 ☑ 1169
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 950us/step
Q 405+16  T 421  ☑ 421 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 1+615   T 616  ☑ 616 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 954+996 T 1950 ☑ 1950
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 62+254  T 316  ☑ 316 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 31+196  T 227  ☑ 227 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 885+88  T 973  ☑ 973 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 975us/step
Q 586+74  T 660  ☑ 660 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 299+94  T 393  ☑ 393 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 11
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9863 - loss: 0.0675 - val_accuracy: 0.9807 - val_loss: 0.0721
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 844+638 T 1482 ☑ 1482
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 833+98  T 931  ☑ 931 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 85+68   T 153  ☑ 153 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 201+18  T 219  ☑ 219 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 7+386   T 393  ☑ 393 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 854+66  T 920  ☑ 920 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 80+624  T 704  ☒ 705 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 524+721 T 1245 ☑ 1245
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 311+86  T 397  ☑ 397 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 746+67  T 813  ☑ 813 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 12
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9832 - loss: 0.0671 - val_accuracy: 0.9842 - val_loss: 0.0557
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 10+577  T 587  ☑ 587 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 960us/step
Q 257+3   T 260  ☑ 260 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 938us/step
Q 83+53   T 136  ☑ 136 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 977us/step
Q 17+898  T 915  ☑ 915 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 12+6    T 18   ☑ 18  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 761+54  T 815  ☑ 815 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 935us/step
Q 813+742 T 1555 ☑ 1555
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 3+10    T 13   ☑ 13  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 22+49   T 71   ☑ 71  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 81+618  T 699  ☑ 699 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 13
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9888 - loss: 0.0459 - val_accuracy: 0.9810 - val_loss: 0.0623
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 991+45  T 1036 ☑ 1036
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 952us/step
Q 683+1   T 684  ☑ 684 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 994us/step
Q 49+70   T 119  ☑ 119 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 1+500   T 501  ☑ 501 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 969us/step
Q 51+444  T 495  ☑ 495 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 60+47   T 107  ☑ 107 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 976us/step
Q 76+921  T 997  ☑ 997 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 83+732  T 815  ☑ 815 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 936+22  T 958  ☑ 958 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 975us/step
Q 790+770 T 1560 ☑ 1560
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 14
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9827 - loss: 0.0592 - val_accuracy: 0.9970 - val_loss: 0.0188
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 2+715   T 717  ☑ 717 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 928us/step
Q 767+7   T 774  ☑ 774 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 967+27  T 994  ☑ 994 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 976+23  T 999  ☑ 999 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 695+77  T 772  ☑ 772 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 989us/step
Q 7+963   T 970  ☑ 970 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 949us/step
Q 91+461  T 552  ☑ 552 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 901us/step
Q 41+657  T 698  ☑ 698 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 948us/step
Q 796+14  T 810  ☑ 810 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 956us/step
Q 321+11  T 332  ☑ 332 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 15
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9970 - loss: 0.0177 - val_accuracy: 0.9902 - val_loss: 0.0339
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 50+859  T 909  ☑ 909 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 733+351 T 1084 ☑ 1084
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 373+25  T 398  ☑ 398 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 771+1   T 772  ☑ 772 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 144+799 T 943  ☑ 943 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 994us/step
Q 7+897   T 904  ☑ 904 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 58+50   T 108  ☑ 108 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 986us/step
Q 731+12  T 743  ☑ 743 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 641+58  T 699  ☑ 699 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 978us/step
Q 577+97  T 674  ☑ 674 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 16
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9945 - loss: 0.0238 - val_accuracy: 0.9921 - val_loss: 0.0332
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 987us/step
Q 37+501  T 538  ☑ 538 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 989us/step
Q 188+44  T 232  ☑ 232 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 935us/step
Q 2+292   T 294  ☑ 294 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 938us/step
Q 620+206 T 826  ☑ 826 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 882us/step
Q 417+20  T 437  ☑ 437 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 914us/step
Q 59+590  T 649  ☑ 649 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 885us/step
Q 486+38  T 524  ☑ 524 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 934us/step
Q 521+307 T 828  ☑ 828 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 963us/step
Q 777+825 T 1602 ☒ 1502
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 922us/step
Q 9+285   T 294  ☑ 294 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 17
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9965 - loss: 0.0171 - val_accuracy: 0.9711 - val_loss: 0.0850
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 899+99  T 998  ☑ 998 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 4+516   T 520  ☑ 520 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 568+45  T 613  ☑ 613 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 742+339 T 1081 ☑ 1081
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 124+655 T 779  ☑ 779 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 958us/step
Q 7+640   T 647  ☑ 647 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 958us/step
Q 77+922  T 999  ☑ 999 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 888us/step
Q 478+54  T 532  ☑ 532 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 836us/step
Q 62+260  T 322  ☑ 322 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 997us/step
Q 344+426 T 770  ☑ 770 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 18
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9867 - loss: 0.0433 - val_accuracy: 0.9565 - val_loss: 0.1465
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 918+4   T 922  ☑ 922 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 5+657   T 662  ☒ 672 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 988us/step
Q 76+40   T 116  ☒ 117 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 999us/step
Q 704+807 T 1511 ☒ 1512
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 21+326  T 347  ☒ 348 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 6+859   T 865  ☑ 865 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 980us/step
Q 533+804 T 1337 ☒ 1327
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 998us/step
Q 70+495  T 565  ☒ 566 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 919us/step
Q 50+477  T 527  ☑ 527 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 463+33  T 496  ☑ 496 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 19
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9879 - loss: 0.0406 - val_accuracy: 0.9965 - val_loss: 0.0162
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 717+39  T 756  ☑ 756 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 45+518  T 563  ☑ 563 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 461+5   T 466  ☑ 466 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 12+6    T 18   ☑ 18  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 584+5   T 589  ☑ 589 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 154+133 T 287  ☑ 287 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 242+25  T 267  ☑ 267 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 937us/step
Q 36+824  T 860  ☑ 860 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 901us/step
Q 894+339 T 1233 ☑ 1233
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 983us/step
Q 820+625 T 1445 ☑ 1445
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 20
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9973 - loss: 0.0128 - val_accuracy: 0.9791 - val_loss: 0.0587
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 587+606 T 1193 ☑ 1193
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 742+67  T 809  ☑ 809 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 92+84   T 176  ☑ 176 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 681+695 T 1376 ☑ 1376
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 936+0   T 936  ☑ 936 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 983+0   T 983  ☑ 983 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 544+95  T 639  ☑ 639 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 659+411 T 1070 ☑ 1070
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 876+63  T 939  ☑ 939 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 856us/step
Q 206+82  T 288  ☑ 288 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 21
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9909 - loss: 0.0293 - val_accuracy: 0.9982 - val_loss: 0.0087
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 459+4   T 463  ☑ 463 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 85+903  T 988  ☑ 988 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 484+17  T 501  ☑ 501 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 136+412 T 548  ☑ 548 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 0+761   T 761  ☑ 761 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 41+945  T 986  ☑ 986 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 952us/step
Q 450+517 T 967  ☑ 967 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 162+15  T 177  ☑ 177 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 365+807 T 1172 ☑ 1172
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 34+499  T 533  ☑ 533 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 22
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9959 - loss: 0.0158 - val_accuracy: 0.9953 - val_loss: 0.0197
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 820+51  T 871  ☑ 871 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 904us/step
Q 3+228   T 231  ☑ 231 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 87+634  T 721  ☑ 721 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 2+715   T 717  ☑ 717 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 76+885  T 961  ☑ 961 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 917us/step
Q 92+896  T 988  ☑ 988 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 434+417 T 851  ☑ 851 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 91+346  T 437  ☑ 437 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 174+697 T 871  ☑ 871 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 44+506  T 550  ☑ 550 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 23
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9968 - loss: 0.0136 - val_accuracy: 0.9984 - val_loss: 0.0085
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 4+966   T 970  ☑ 970 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 10+53   T 63   ☑ 63  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 701+841 T 1542 ☑ 1542
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 10+45   T 55   ☑ 55  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 294+600 T 894  ☑ 894 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 9+182   T 191  ☑ 191 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 722+522 T 1244 ☑ 1244
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 811+38  T 849  ☑ 849 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 531+69  T 600  ☑ 600 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 978us/step
Q 17+59   T 76   ☑ 76  
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 24
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9981 - loss: 0.0084 - val_accuracy: 0.9952 - val_loss: 0.0175
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 6+668   T 674  ☑ 674 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 994us/step
Q 198+295 T 493  ☑ 493 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 988us/step
Q 89+828  T 917  ☑ 917 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 915us/step
Q 286+907 T 1193 ☑ 1193
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 146+16  T 162  ☑ 162 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 81+267  T 348  ☑ 348 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 907us/step
Q 95+921  T 1016 ☑ 1016
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 914us/step
Q 6+475   T 481  ☑ 481 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 85+521  T 606  ☑ 606 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 976us/step
Q 597+819 T 1416 ☑ 1416
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 25
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9977 - loss: 0.0101 - val_accuracy: 0.9752 - val_loss: 0.0939
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 998us/step
Q 84+194  T 278  ☑ 278 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 6+543   T 549  ☑ 549 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 962us/step
Q 455+99  T 554  ☑ 554 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 74+232  T 306  ☑ 306 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 958us/step
Q 27+48   T 75   ☑ 75  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 184+435 T 619  ☑ 619 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 959us/step
Q 257+674 T 931  ☒ 1031
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 978us/step
Q 887+7   T 894  ☑ 894 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 957us/step
Q 43+0    T 43   ☑ 43  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 829us/step
Q 629+542 T 1171 ☑ 1171
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 26
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9949 - loss: 0.0188 - val_accuracy: 0.9983 - val_loss: 0.0081
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 964+524 T 1488 ☑ 1488
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 991us/step
Q 556+47  T 603  ☑ 603 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 433+56  T 489  ☑ 489 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 982+11  T 993  ☑ 993 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 887+39  T 926  ☑ 926 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 656+57  T 713  ☑ 713 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 843+186 T 1029 ☑ 1029
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 367+596 T 963  ☑ 963 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 921us/step
Q 40+133  T 173  ☑ 173 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 407+80  T 487  ☑ 487 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 27
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9966 - loss: 0.0126 - val_accuracy: 0.9985 - val_loss: 0.0076
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 985us/step
Q 462+0   T 462  ☑ 462 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 13+861  T 874  ☑ 874 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 919us/step
Q 122+439 T 561  ☑ 561 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 85+420  T 505  ☑ 505 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 938us/step
Q 371+69  T 440  ☑ 440 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 933us/step
Q 11+150  T 161  ☑ 161 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 930us/step
Q 694+26  T 720  ☑ 720 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 932us/step
Q 422+485 T 907  ☑ 907 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 954us/step
Q 146+130 T 276  ☑ 276 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 921us/step
Q 103+19  T 122  ☑ 122 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 28
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9963 - loss: 0.0134 - val_accuracy: 0.9754 - val_loss: 0.0840
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 923+68  T 991  ☑ 991 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 398+74  T 472  ☑ 472 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 319+805 T 1124 ☑ 1124
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 936+10  T 946  ☑ 946 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 933+721 T 1654 ☑ 1654
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 89+965  T 1054 ☑ 1054
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 37+469  T 506  ☒ 516 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 84+194  T 278  ☑ 278 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 973us/step
Q 965+5   T 970  ☑ 970 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 120+480 T 600  ☑ 600 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 29
 1407/1407 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9942 - loss: 0.0236 - val_accuracy: 0.9883 - val_loss: 0.0342
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 626+584 T 1210 ☑ 1210
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 952us/step
Q 501+615 T 1116 ☑ 1116
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 1+827   T 828  ☑ 828 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 527+651 T 1178 ☑ 1178
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 977us/step
Q 53+44   T 97   ☑ 97  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 899us/step
Q 79+474  T 553  ☑ 553 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 941us/step
Q 34+949  T 983  ☑ 983 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 954us/step
Q 66+807  T 873  ☑ 873 
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 827us/step
Q 49+28   T 77   ☑ 77  
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
Q 628+93  T 721  ☑ 721 

```
</div>
You'll get to 99+% validation accuracy after ~30 epochs.

Example available on HuggingFace.

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/🤗%20Model-Addition%20LSTM-black.svg)](https://huggingface.co/keras-io/addition-lstm) | [![Generic badge](https://img.shields.io/badge/🤗%20Spaces-Addition%20LSTM-black.svg)](https://huggingface.co/spaces/keras-io/addition-lstm) |
